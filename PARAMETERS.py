# -*- coding: utf-8 -*-
# @Author: gehuama
# @Date:   2017-12-03 15:31:43
# @Last Modified by:   gehuama
# @Last Modified time: 2017-12-05 11:35:16

import sys,os
sys.path.insert(0, '/home/wb/Env/caffe/python')
import caffe
import cv2
from PIL import Image 
import numpy as np
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
import argparse

#INPUT_SIZE    = (768, 1024, 3)
INPUT_PATH     = './0120_03_01_02_c.bmp'
RGB_IMG_DIR    = './rgb_imgs/'
#OUTPUT_SIZE   = (768, 1024, 1) 

PATCH_DIR      = '/home/wb/RGB2NIR/transformed_0120_03_01_02/'
PATCH_SAVE_DIR = './patches/'

CROP_W         = 5
HSZ            = 32 - CROP_W
HALF_HSZ       = 32
PATCH_WID      = 65
PATCH_HEI      = 65

