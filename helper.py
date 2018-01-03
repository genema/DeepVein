# -*- coding: utf-8 -*-
# @Author: gehuama
# @Date:   2017-11-30 19:13:12
# @Last Modified by:   gehuama
# @Last Modified time: 2018-01-03 10:36:31
# =========================================================================================================
# This is a script for splicing forearm patches splitted by a matlab algorithm, which 
# is written by my tutor and her fellows.
# What's more, I am sure I'll replace the old one with a python script, in order to form a more applicable 
# joint system.
# =========================================================================================================
from PARAMETERS import *


def ind2sub(array_shape, ind):
	cols = (int(ind) / array_shape[0])
	rows = (int(ind) % array_shape[0])
	return (rows, cols)

# in matlab ,the index is following the order of column?
def sub2ind(array_shape, rows, cols):
	return rows*array_shape[1] + cols

def cpy_pix(x0, x1, y0, y1, src, dst, is_color=1):
	for i in range(x0, x1):
		for j in range(y0, y1):
			if is_color:
				dst[i][j][0] = src[i-x0][j-y0][0]
			else:
				dst[i][j] = src[i-x0][j-y0]

def crop(src, wid, hei, crop_w=CROP_W):
	im = Image.open(src)
	im = im.crop((crop_w-1, crop_w-1, wid-1-crop_w, hei-1-crop_w))
	cv2_im = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
	return cv2_im

def write_file(file, data):
	with open(file, 'wba') as f:
		f.write(data)

def imread_with_pad(im_path):
	im = cv2.imread(im_path)
	orig_size = im.shape
	new_size = (im.shape[0]+PATCH_WID, im.shape[1]+PATCH_HEI, im.shape[2])
	new_im = np.zeros(new_size)
	new_im[HALF_HSZ:HALF_HSZ+im.shape[0], HALF_HSZ:HALF_HSZ+im.shape[1]] = im[:, :]
	return new_im, orig_size
