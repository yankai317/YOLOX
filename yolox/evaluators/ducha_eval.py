import numpy as np
import math
import sys, os
import random
import json

def intersect(box_a, box_b):
    A = box_a.shape[0]
    B = box_b.shape[0]

    box_a_expand = np.expand_dims(box_a[:, 2:], axis=1)
    box_a_expand = np.tile(box_a_expand, (1, B, 1))
    box_b_expand = np.expand_dims(box_b[:, 2:], axis=0)
    box_b_expand = np.tile(box_b_expand, (A, 1, 1))

    max_xy = np.minimum(box_a_expand, box_b_expand)

    box_a_expand = np.expand_dims(box_a[:, :2], axis=1)
    box_a_expand = np.tile(box_a_expand, (1, B, 1))
    box_b_expand = np.expand_dims(box_b[:, :2], axis=0)
    box_b_expand = np.tile(box_b_expand, (A, 1, 1))

    min_xy = np.maximum(box_a_expand, box_b_expand)
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=1e10)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    inter = intersect(box_a, box_b)

    A = inter.shape[0]
    B = inter.shape[1]

    area_a = np.expand_dims(((box_a[:, 2] - box_a[:, 0]) *
                             (box_a[:, 3] - box_a[:, 1])), axis=1)
    area_a = np.tile(area_a, (1, B))
    area_b = np.expand_dims(((box_b[:, 2] - box_b[:, 0]) *
                             (box_b[:, 3] - box_b[:, 1])), axis=0)
    area_b = np.tile(area_b, (A, 1))
    union = area_a + area_b - inter + 1e-4
    # print(union)
    return inter / union  # [A,B]


def voc_ap(rec, prec, use_07_metric=False):
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def cal_acc_recall(all_objs_gt, all_objs_dt, class_name, iou_threshold):
    tp_class = [[] for _ in range(len(class_name))]
    fp_class = [[] for _ in range(len(class_name))]
    gt_num = [0 for _ in range(len(class_name))]
    dt_num = [0 for _ in range(len(class_name))]
    det_conf = [[] for _ in range(len(class_name))]
    assert len(all_objs_gt) == len(all_objs_dt)
    for i, objs_gt in enumerate(all_objs_gt):
        objs_dt = all_objs_dt[i]

        if len(objs_gt) == 0 and len(objs_dt) == 0:
            continue
        elif len(objs_gt) == 0:
            for class_i in range(len(class_name)):
                dt_class_list = list(np.where(objs_dt[..., 4] == class_i))[0]
                dt_num[class_i] += len(dt_class_list)
                fp_class[class_i].extend(list(objs_dt[np.where(objs_dt[..., 4] == class_i)][..., 5]))
        elif len(objs_dt) == 0:
            for class_i in range(len(class_name)):
                gt_class_list = list(np.where(objs_gt[..., 4] == class_i))[0]
                gt_num[class_i] += len(gt_class_list)
        else:
            iou = jaccard(objs_gt[..., :4], objs_dt[..., :4])
            for class_i in range(len(class_name)):
                gt_valid_index = np.where(objs_gt[..., 4] == class_i)[0]
                gt_num[class_i] += len(gt_valid_index)
                dt_valid_index = np.where((objs_dt[..., 4] == class_i))[0]
                dt_num[class_i] += len(dt_valid_index)
                gt_invalid_index = np.where(objs_gt[..., 4] != class_i)[0]
                dt_invalid_index = np.where((objs_dt[..., 4] != class_i))[0]

                iou_class = iou.copy()
                iou_class[..., dt_invalid_index] = 0
                iou_class[gt_invalid_index, ...] = 0
                match_gt_indexs, match_dt_indexs = np.where(
                    (iou_class == np.max(iou_class, axis=0)) * (iou_class > iou_threshold))

                #  For match gt with dt ,filter bt boxs which match same gt box by choosing the max_iou one
                match_gt_index_max = {}
                match_dt_dict = {}
                for match_gt_index, match_dt_index in zip(match_gt_indexs, match_dt_indexs):
                    if not match_gt_index in match_gt_index_max.keys():
                        match_gt_index_max[match_gt_index] = iou_class[match_gt_index, match_dt_index]
                        match_dt_dict[match_gt_index] = match_dt_index
                    else:
                        if match_gt_index_max[match_gt_index] < iou_class[match_gt_index, match_dt_index]:
                            match_gt_index_max[match_gt_index] = iou_class[match_gt_index, match_dt_index]
                            match_dt_dict[match_gt_index] = match_dt_index

                match_dt_index_filter = set(match_dt_dict.values())
                match_gt_indexs_filter = set(match_gt_indexs)
                no_match_dt = list(set(dt_valid_index).difference(match_dt_index_filter))

                for match_gt_index in match_gt_indexs_filter:
                    tp = objs_dt[match_dt_dict[match_gt_index], 5]
                    tp_class[class_i].append(tp)

                for no_match_dt_index in no_match_dt:
                    fp = objs_dt[no_match_dt_index, 5]
                    fp_class[class_i].append(fp)

    rec_all = []
    prec_all = []
    fppi_all = []
    conf_all = []
    all_tp = 0
    all_fp = 0

    for class_i in range(len(class_name)):
        det_conf = []

        for conf in tp_class[class_i]:
            det_conf.append([conf, 1])

        for conf in fp_class[class_i]:
            det_conf.append([conf, 0])

        random.shuffle(det_conf)
        det_conf.sort(key=lambda x: float(x[0]), reverse=True)

        if len(det_conf) == 0:
            rec_all.append([0])
            prec_all.append([0])
            fppi_all.append([0])
            conf_all.append([0])
            continue
        tp = [0]
        fp = [0]
        last_det_conf = det_conf[0][0]
        det_conf_score = [det_conf[0][0]]
        count = 0
        for det_box in det_conf:
            if not det_box[0] == last_det_conf:
                tp.append(tp[count])
                fp.append(fp[count])
                last_det_conf = det_box[0]
                count += 1
                det_conf_score.append(det_box[0])

            tp[count] += det_box[1]
            fp[count] += 1 - det_box[1]
        all_tp += tp[-1]
        all_fp += fp[-1]
        rec = tp[:]
        for idx, val in enumerate(tp):
            rec[idx] = float(tp[idx] + 1e-10) / (gt_num[class_i] + 1e-10)
        # print("len-recall", len(rec))
        prec = tp[:]
        for idx, val in enumerate(tp):
            prec[idx] = float(tp[idx] + 1e-10) / (fp[idx] + tp[idx] + 1e-10)
        fppi = tp[:]
        for idx, val in enumerate(tp):
            fppi[idx] = float(fp[idx] + 1e-10) / (fp[idx] + tp[idx] + 1e-10)

        rec_all.append(rec)
        prec_all.append(prec)
        fppi_all.append(fppi)
        conf_all.append(det_conf_score)
    all_rec = float(all_tp + 1e-10) / (sum(gt_num) + 1e-10)
    all_prec = float(all_tp + 1e-10) / (sum(dt_num) + 1e-10)

    return rec_all, prec_all, all_rec, all_prec


def det_eval(gt_file, bt_file, class_name, iou_threshold):
    """
    :param gt_file:  str input file like 'test/gt.json'
    :param bt_file:  [str] models like ['test/mdoel_1.json','test/mdoel_2.json']
    :param class_name:  [str] classification names like ['person', 'non-motor', 'car', 'tricycle']
    :param models_name:  [str] classification names like ['pelee-fp32','pelee-int8-1']
    :param iou_threshold: float eval iou threshold like 0.5
    :return:
    """

    rec, prec, all_rec, all_prec = cal_acc_recall(gt_file, bt_file, class_name, iou_threshold)

    aps = []
    for i, name in enumerate(class_name):
        ap = voc_ap(rec[i], prec[i])
        aps.append(ap)

    return rec, prec, aps, all_rec, all_prec