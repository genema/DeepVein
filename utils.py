import tensorflow as tf
import tensorlayer as tl
from tensorlayer.prepro import *

import scipy
import numpy as np

def get_imgs_fn(file_name, path):
    """ Input an image path and name, return an image array """
    # return scipy.misc.imread(path + file_name).astype(np.float)
    return scipy.misc.imread(path + file_name, mode='RGB')

def crop_sub_imgs_fn(x, is_random=False):
    # x = crop(x, wrg=65, hrg=65, is_random=is_random)
    x = x / (255. / 2.)
    x = x - 1.
    return x

def downsample_fn(x):
    # We obtained the LR images by downsampling the HR images using bicubic kernel with downsampling factor r = 4.
    # x = imresize(x, size=[65, 65], interp='bicubic', mode=None)
    x = x / (255. / 2.)
    x = x - 1.
    return x

def resize_fn(x, size=[260, 260], scale=1):
    x = imresize(x, size=size, interp='bicubic', mode=None)
    if scale:
        x = x / (255. / 2.)
        x = x - 1
    return x
