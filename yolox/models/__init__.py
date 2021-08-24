#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

from .darknet import CSPDarknet, Darknet
from .losses import IOUloss
from .yolo_fpn import YOLOFPN
from .yolo_head import YOLOXHead
from .yolo_head_ddod import YOLOXHeadDdod
from .yolo_pafpn import YOLOPAFPN
# from .yolo_pafpn_cot import YOLOPAFPNCOT
from .yolox import YOLOX
