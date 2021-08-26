#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

from .coco import COCODataset
from .coco_classes import COCO_CLASSES
from .datasets_wrapper import ConcatDataset, Dataset, MixConcatDataset
from .mosaicdetection import MosaicDetection
from .voc import VOCDetection
from .ducha import DUCHADataset
from .ducha_classes import DUCHA_CLASSES
# from .person_head import PersonHeadDataset
from .person_head_classes import PERSON_HEAD_CLASSES