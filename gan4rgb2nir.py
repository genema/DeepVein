#! /usr/bin/python
# -*- coding: utf8 -*-

import os, time, pickle, random, time
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging, scipy

import tensorflow as tf
import tensorlayer as tl
from model import *
from utils import *
# from config import config, log_config

import argparse
import cv2


def read_all_imgs(img_list, path='', n_threads=20):
    """ Returns all images in array by given path and name of each image file. """
    imgs = []
    for idx in range(0, len(img_list), n_threads):
        b_imgs_list = img_list[idx : idx + n_threads]
        b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=path)
        # print(b_imgs.shape)
        imgs.extend(b_imgs)
        print('read %d from %s' % (len(imgs), path))
    return imgs


def read_one_img():
    img = tl.prepro.threading_data(['temp/temp.png'],fn=get_imgs_fn,path='')
    return img[0]


def evaluate():

    checkpoint_dir = "checkpoint"

    rgb_img = read_one_img()
    rgb_img = tl.prepro.threading_data(rgb_img, fn=downsample_fn)

    #rgb_img = (rgb_img / 127.5) - 1   # rescale to ［－1, 1]

    size = rgb_img.shape
    t_image = tf.placeholder('float32', [None, size[0], size[1], size[2]], name='input_image')

    net_g = SRGAN_g(t_image, is_train=False, reuse=False)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/g_srgan.npz', network=net_g)

    start_time = time.time()
    out = sess.run(net_g.outputs, {t_image: [rgb_img]})
    print("took: %4.4fs" % (time.time() - start_time))
    return resize_fn(out[0], [65, 65], 0) * 2

#im = evaluate()
#cv2.imwrite('./result.png', im)
