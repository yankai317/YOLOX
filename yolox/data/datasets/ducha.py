#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import glob
import cv2
import numpy as np
import torch

from ..dataloading import get_yolox_datadir
from .datasets_wrapper import Dataset
from .ducha_classes import DUCHA_CLASSES
from yolox.evaluators.ducha_eval import det_eval
import pickle

class DUCHADataset(Dataset):
    """
    Darknet dataset class.
    """

    def __init__(
        self,
        data_dir=None,
        name="train",
        img_size=(416, 416),
        preproc=None,
    ):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
        """
        super().__init__(img_size)
        if data_dir is None:
            data_dir = os.path.join(get_yolox_datadir(), "ducha_det/images/{}".format(name))
            self.data_dir = data_dir
        else:
            self.data_dir = os.path.join(data_dir, "images/{}".format(name))
        self.images_list = glob.glob(self.data_dir+ '/*.jpg')
        self.labels_list = []
        for img in self.images_list:
            txt_path = img.replace("images","labels").replace(".jpg",".txt")
            assert os.path.exists(txt_path), "{} lack of label".format(img)
            self.labels_list.append(txt_path)
        
        self.ids = list(range(len(self.images_list)))
        self.class_ids = [0]
        self.name = name
        self.img_size = img_size
        self.preproc = preproc
        self.image_size_info = {}

    def __len__(self):
        return len(self.labels_list)

    def load_anno(self, index, img_info):
        id_ = self.ids[index]
        label_path = self.labels_list[id_]

        width = img_info[1]
        height = img_info[0]
        self.image_size_info[index] = img_info
        # load labels
        objs = []
        with open(label_path, 'r') as txt_read:
            for box_line in txt_read.readlines():
                cls, x_c, y_c, w, h = map(float, box_line.split(" ")[:5])
                x1 = np.max((0, (x_c - w/2)*width))
                y1 = np.max((0, (y_c - h/2)*height))
                x2 = np.min((width - 1, np.max((0, (x_c + w/2)*width))))
                y2 = np.min((height - 1, np.max((0, (y_c + h/2)*height))))
                obj = [x1,y1,x2,y2,cls]
                objs.append(obj)
        
        res = np.asarray(objs) if len(objs) else np.zeros([0, 5])

        return res

    def pull_item(self, index):
        id_ = self.ids[index]
        img_file = self.images_list[id_]

        img = cv2.imread(img_file)
        assert img is not None
        img_info = (img.shape[0], img.shape[1])
        # load anno
        res = self.load_anno(index, img_info)

        return img, res, img_info, id_

    @Dataset.resize_getitem
    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up and pre-processed.

        Args:
            index (int): data index

        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data.
                The shape is :math:`[max_labels, 5]`.
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w, nh, nw, dx, dy.
                h, w (int): original shape of the image
                nh, nw (int): shape of the resized image without padding
                dx, dy (int): pad size
            img_id (int): same as the input index. Used for evaluation.
        """
        img, res, img_info, img_id = self.pull_item(index)
    
        if self.preproc is not None:
            img, target = self.preproc(img, res, self.input_dim)
        else:
            target = res
        return img, target, img_info, img_id

    def evaluate_detections(self, all_boxes, output_dir=None):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.

        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        cachedir = os.path.join(output_dir, "annotations_cache")
        if not os.path.exists(cachedir):
            os.makedirs(cachedir)
        cache_path = os.path.join(cachedir, 'annotations.pickle')
        
        if not os.path.exists(cache_path):
            all_gt_boxes = self._read_txt_annotations()
            self._write_ducha_results_file(all_gt_boxes, output_dir)
        else:
            f = open(cache_path, 'rb')
            all_gt_boxes = pickle.load(f)
            f.close()
        # self._write_ducha_results_file(all_boxes)
        IouTh = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True)
        mAPs = []
        for boxes in all_boxes:
            if isinstance(boxes, torch.Tensor):
                boxes = boxes.cpu().numpy()
        for iou in IouTh:
            mAP = self._do_python_eval(all_gt_boxes, all_boxes, output_dir, iou)
            mAPs.append(mAP)

        print("--------------------------------------------------------------")
        print("map_5095:", np.mean(mAPs))
        print("map_50:", mAPs[0])
        print("--------------------------------------------------------------")
        return mAPs[0], np.mean(mAPs)

    def _get_ducha_results_file_template(self, output_dir):
        filename = "eval_boxes.pickle"
        filedir = os.path.join(output_dir, "results")
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def _write_ducha_results_file(self, all_boxes, output_dir="output"):
        result_save_path = self._get_ducha_results_file_template(output_dir)
        f = open(result_save_path, 'wb')
        pickle.dump(all_boxes, f)
        f.close()

    def _read_txt_annotations(self):
        gt_all_boxes = []
        for i, img_path in enumerate(self.images_list):
            img = cv2.imread(img_path)
            img_info = (img.shape[0], img.shape[1])
            boxes = self.load_anno(i, img_info)
            gt_scores = np.ones([boxes.shape[0],1])
            gt_boxes = np.concatenate([boxes, gt_scores],1)
            gt_all_boxes.append(gt_boxes)

        return gt_all_boxes

    def _do_python_eval(self, all_gt_boxes, all_boxes, output_dir="output", iou=0.5):
    
        recs, precs, aps = det_eval(
           all_gt_boxes, all_boxes, DUCHA_CLASSES, iou
        )
        print("Eval IoU : {:.2f}".format(iou))
        if output_dir is not None and not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        for rec, prec, ap, cls in zip(recs, precs, aps, DUCHA_CLASSES):
            
            if iou == 0.5:
                print("AP for {} = {:.4f}".format(cls, ap))
            if output_dir is not None:
                with open(os.path.join(output_dir, cls + "_pr.pkl"), "wb") as f:
                    pickle.dump({"rec": rec, "prec": prec, "ap": ap}, f)

        if iou == 0.5:
            print("Mean AP = {:.4f}".format(np.mean(aps)))
            print("~~~~~~~~")
            print("Results:")
            for ap in aps:
                print("{:.3f}".format(ap))
            print("{:.3f}".format(np.mean(aps)))
            print("~~~~~~~~")
            print("")
            print("--------------------------------------------------------------")
            print("Results computed with the **unofficial** Python eval code.")
            print("Results should be very close to the official MATLAB eval code.")
            print("Recompute with `./tools/reval.py --matlab ...` for your paper.")
            print("-- Thanks, The Management")
            print("--------------------------------------------------------------")

        return np.mean(aps)