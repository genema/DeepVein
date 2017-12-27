# -*- coding: utf-8 -*-
# @Author: gehuama
# @Date:   2017-12-03 15:31:43
# @Last Modified by:   gehuama
# @Last Modified time: 2017-12-23 13:47:48

import sys,os
# modifies this too 
sys.path.insert(0, '/home/wb/Env/caffe-ghma/python')
import caffe
import cv2
from PIL import Image 
import numpy as np
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
import argparse

# for general usage ,just modify the directory of rg images,all the rgb files in the directory will be transformed
RGB_IMG_DIR    = './rgb_imgs/'

# also modify the caffe root 
caffe_root = '/home/wb/Env/caffe-ghma/'

# these are parameters for developpers 
PATCH_DIR      = '/home/wb/RGB2NIR/transformed_0120_03_01_02/'
PATCH_SAVE_DIR = './patches/'
TRANSFORMED_IMG_SAVE_PATH = '/home/wb/RGB2NIR/transformed_images/'


CROP_W         = 5
HSZ            = 32 - CROP_W
HALF_HSZ       = 32
PATCH_WID      = 65
PATCH_HEI      = 65

