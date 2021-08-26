#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Code are based on
# https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py
# Copyright (c) Francisco Massa.
# Copyright (c) Ellis Brown, Max deGroot.
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import os.path
import pickle
from loguru import logger
import torch
import cv2
import numpy as np
import glob
from yolox.evaluators.ducha_eval import det_eval
from ..dataloading import get_yolox_datadir
from .datasets_wrapper import Dataset
from .ducha_classes import DUCHA_CLASSES
import copy

class DUCHADataset(Dataset):

    """
    VOC Detection Dataset Object

    input is image, target is annotation

    Args:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(
        self,
        data_dir=None,
        name="train",
        img_size=(416, 416),
        preproc=None,
        cache=False,
    ):
        super().__init__(img_size)
        if data_dir is None:
            data_dir = os.path.join(get_yolox_datadir(), "ducha_det/images/{}".format(name))
            self.data_dir = data_dir
        else:
            self.data_dir = os.path.join(data_dir, "images/{}".format(name))
        self.images_list = glob.glob(self.data_dir+ '/*.jpg')
        self.root = data_dir
        self.labels_list = []
        self.sizes_list = []
        for img in self.images_list:
            txt_path = img.replace("images","labels").replace(".jpg",".txt")
            size_txt_path = txt_path.replace("labels","sizes")
            assert os.path.exists(txt_path), "{} lack of label".format(img)
            self.labels_list.append(txt_path)
            self.sizes_list.append(size_txt_path)
        # self.images_list = self.images_list[:1000]
        # self.labels_list = self.labels_list[:1000]
        # self.sizes_list = self.sizes_list[:1000]
        self.ids = list(range(len(self.images_list)))
        self.class_ids = [0]
        self.name = name
        self.img_size = img_size
        self.preproc = preproc
        self.image_size_info = {}
        self.annotations = self._load_ducha_annotations()
        self.imgs = None
        if cache:
            self._cache_images()

    def __len__(self):
        return len(self.ids)

    def _load_ducha_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in range(len(self.ids))]

    def _cache_images(self):
        logger.warning(
            "\n********************************************************************************\n"
            "You are using cached images in RAM to accelerate training.\n"
            "This requires large system RAM.\n"
            "Make sure you have 60G+ RAM and 90G available disk space for training ducha.\n"
            "********************************************************************************\n"
        )
        max_h = self.img_size[0]
        max_w = self.img_size[1]
        cache_file = self.root + "/img_resized_cache_" + self.name + ".array"
        if not os.path.exists(cache_file):
            logger.info(
                "Caching images for the frist time. This might take about 3 minutes for VOC"
            )
            self.imgs = np.memmap(
                cache_file,
                shape=(len(self.ids), max_h, max_w, 3),
                dtype=np.uint8,
                mode="w+",
            )
            from tqdm import tqdm
            from multiprocessing.pool import ThreadPool

            NUM_THREADs = min(8, os.cpu_count())
            loaded_images = ThreadPool(NUM_THREADs).imap(
                lambda x: self.load_resized_img(x),
                range(len(self.annotations)),
            )
            pbar = tqdm(enumerate(loaded_images), total=len(self.annotations))
            for k, out in pbar:
                self.imgs[k][: out.shape[0], : out.shape[1], :] = out.copy()
            self.imgs.flush()
            pbar.close()
        else:
            logger.warning(
                "You are using cached imgs! Make sure your dataset is not changed!!"
            )

        logger.info("Loading cached imgs...")
        self.imgs = np.memmap(
            cache_file,
            shape=(len(self.ids), max_h, max_w, 3),
            dtype=np.uint8,
            mode="r",
        )

    def load_anno_from_ids(self, index):
        img_id = self.ids[index]
        label_path = self.labels_list[img_id]
        size_path = self.sizes_list[img_id]
        height = 0
        width = 0
        with open(size_path, 'r') as txt_read:
            for box_line in txt_read.readlines():
                height, width = map(int, box_line.split(" "))

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

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r
        resized_info = (int(height * r), int(width * r))

        return (res, (height, width), resized_info)

    def load_anno(self, index):
        return self.annotations[index]

    def load_resized_img(self, index):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)

        return resized_img

    def load_image(self, index):
        img_id = self.ids[index]
        img = cv2.imread(self.images_list[img_id], cv2.IMREAD_COLOR)
        assert img is not None
        return img

    def pull_item(self, index):
        """Returns the original image and target at an index for mixup

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            img, target
        """
        if self.imgs is not None:
            target, img_info, resized_info = self.annotations[index]
            pad_img = self.imgs[index]
            img = pad_img[: resized_info[0], : resized_info[1], :].copy()
        else:
            img = self.load_resized_img(index)
            target, img_info, _ = self.annotations[index]

        return img, target, img_info, index

    @Dataset.mosaic_getitem
    def __getitem__(self, index):
        img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)

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
        all_recs = []
        all_precs = []
        for boxes in all_boxes:
            if isinstance(boxes, torch.Tensor):
                boxes = boxes.cpu().numpy()
        for iou in IouTh:
            mAP, all_rec, all_prec = self._do_python_eval(all_gt_boxes, all_boxes, output_dir, iou)
            mAPs.append(mAP)
            all_recs.append(all_rec)
            all_precs.append(all_prec)

        print("--------------------------------------------------------------")
        print("map_5095:", np.mean(mAPs))
        print("map_50:", mAPs[0])
        print("rec_50:", all_recs[0])
        print("prec_50:", all_precs[0])
        print("--------------------------------------------------------------")
        return mAPs[0], np.mean(mAPs), all_recs[0], all_precs[0]

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
            boxes, ori_size, img_size = self.load_anno(i)
            cp_box = copy.deepcopy(boxes)
            cp_box[:,:4] *= (ori_size[0]/ img_size[0])
            gt_scores = np.ones([cp_box.shape[0],1])
            gt_boxes = np.concatenate([cp_box, gt_scores],1)
            gt_all_boxes.append(gt_boxes)

        return gt_all_boxes

    def _do_python_eval(self, all_gt_boxes, all_boxes, output_dir="output", iou=0.5):
    
        recs, precs, aps, all_rec, all_prec = det_eval(
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

        return np.mean(aps), all_rec, all_prec
