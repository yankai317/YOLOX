#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import importlib
import os
import sys


def get_exp_by_file(exp_file):
    try:
        sys.path.append(os.path.dirname(exp_file))
        current_exp = importlib.import_module(os.path.basename(exp_file).split(".")[0])
        exp = current_exp.Exp()
    except Exception:
        raise ImportError("{} doesn't contains class named 'Exp'".format(exp_file))
    return exp


def get_exp_by_name(exp_name):
    import yolox

    yolox_path = os.path.dirname(os.path.dirname(yolox.__file__))
    filedict = {
        "yolox-s": "yolox_s.py",
        "yolox-m": "yolox_m.py",
        "yolox-l": "yolox_l.py",
        "yolox-x": "yolox_x.py",
        "yolox-tiny": "yolox_tiny.py",
        "yolox-nano": "nano.py",
        "yolov3": "yolov3.py",
        "yolox-rexnet15": "yolox_rexnet15.py",
        "yolox-rexnet13": "yolox_rexnet13.py",
        "yolox-rexnet13-nose": "yolox_rexnet13_nose.py",
        "yolox-rexnet10": "yolox_rexnet10.py",
        "yolox-rexnet10-lite": "yolox_rexnet10_lite.py",
        "yolov3_ducha": "yolov3_ducha.py",
        "yolox-m-ducha": "yolox_m_ducha.py",
        "yolox-rex10-ducha": "yolox_rex10_ducha.py",
        "yolox-rex10-fpn768-ducha":"yolox_rex10_fpn768_ducha.py",
        "yolox-rex13-ducha": "yolox_rex13_ducha.py",
        "yolox-rex16-lite-ducha": "yolox_rexnet16_lite_ducha.py",
        "yolox-s-ducha": "yolox_s_ducha.py",
        "yolox-m-multi-match": "yolox_m_multi_match.py",
        "yolox-m-person-head": "yolox_m_person_head.py",
        "yolox-m-ddod-ducha": "yolox_m_ddod_ducha.py",
        "yolox-pelee": "yolox_pelee.py",
        "yolox-rexnet11":"yolox_rexnet11.py",
        "yolox-rexnet15-lite":"yolox_rexnet15_lite.py",
        "yolox-rexnet13-lite":"yolox_rexnet13_lite.py",
        "yolox-rexnet16-lite":"yolox_rexnet16_lite.py",
        "yolox-rexnet12":"yolox_rexnet12.py"

    }
    filename = filedict[exp_name]
    exp_path = os.path.join(yolox_path, "exps", "default", filename)
    return get_exp_by_file(exp_path)


def get_exp(exp_file, exp_name):
    """
    get Exp object by file or name. If exp_file and exp_name
    are both provided, get Exp by exp_file.

    Args:
        exp_file (str): file path of experiment.
        exp_name (str): name of experiment. "yolo-s",
    """
    assert (
        exp_file is not None or exp_name is not None
    ), "plz provide exp file or exp name."
    if exp_file is not None:
        return get_exp_by_file(exp_file)
    else:
        return get_exp_by_name(exp_name)
