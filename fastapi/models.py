#!/usr/bin/env python
# -*- coding: utf-8 -*-
from enum import Enum


class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"
