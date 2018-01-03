# -*- coding: utf-8 -*-
# @Author: gehuama
# @Date:   2017-12-03 12:00:20
# @Last Modified by:   gehuama
# @Last Modified time: 2018-01-03 15:28:54
# =========================================================================================================
# RGB to NIR all in one 
# Usage:
# command line
# 	python rgb2nir.py --step x
# =========================================================================================================

from PARAMETERS import *
from helper_caffe import caffe_init, rgb2nir
from helper import cpy_pix, imread_with_pad

import tensorflow as tf
import tensorlayer as tl
from model import *
from utils import *



OUTPUT_DIR = 'results/clarityLoss_results'

def dir_chk():
	for name in (OUTPUT_DIR, PATCH_SAVE_DIR):
		if not os.path.isdir(name):
			os.makedirs(name) 
		

def edge_detector(top, mid, bot):
	judge1 = bool(top[0]==0 and top[1]==0 and top[2]==0)
	judge2 = bool(mid[0]!=0 or mid[1]!=0 or mid[2]!=0)
	judge3 = bool(bot[0]==0 and bot[1]==0 and bot[2]==0)
	#print judge1, judge2, judge3
	if judge1 and judge2 and not judge3:
		return 0 # top
	elif judge2 and judge3 and not judge1:
		return 1 # bot
	else:
		return 2


def get_split_info(src, input_size=(768, 1024, 3)):
	result = dict() # col_y: [top_x, bot_x]
	print ' >> detecting and saving the edge coordinates...'
	for i in tqdm(range(1, input_size[0]-1)):
		for j in range(1, input_size[1]-1):
			# print i, j
			flag = edge_detector(src[i][j-1], src[i][j], src[i][j+1])
			if not i in result.keys():
				result[i] = [9999, 0]
			if flag == 0:
				if result[i][0] > j:
					result[i][0] = j
			elif flag == 1:
				if result[i][1] < j:
					result[i][1] = j
	return result


def get_edge(result, input_size):
	output = np.zeros((input_size[0], input_size[1], 1))
	print ' >> plotting the edge...'
	for i in tqdm(range(1, input_size[0])):
		if not result[i] == [9999, 0]:
			output[i][result[i][1]][0] = 255
			output[i][result[i][0]][0] = 255
	cv2.imwrite('./edge_of_forearm.bmp', output)


def get_single_patch(src, col, row, input_size):
	patch = src[col-HALF_HSZ:col+HALF_HSZ+1, row-HALF_HSZ:row+HALF_HSZ+1]
	index = col*input_size[0] + row
	return patch ,[col, row, index]


def transform_single(im_path, input_image, net, transformer, step=1):
	im_path = im_path + input_image
	im, orig_size = imread_with_pad(im_path)
	nir_img = np.zeros((im.shape[0], im.shape[1], 1))
	result = get_split_info(im, im.shape)
	print ' >> splitting original image and transforming'
	#print result
	for i in tqdm(result.keys()):
		if result[i] != [9999, 0]:
			# print ' >> Transforming and splicing patches'
			for j in range(result[i][0], result[i][1]+1, step):
				patch, info = get_single_patch(im, i, j, im.shape)
				# cv2.imwrite('{}{}.png'.format(PATCH_SAVE_DIR, ind), patch)
				patch = rgb2nir(net, transformer, patch)
				patch = patch[4:60, 4:60]
				patch *= 20
				#patch *= 60
				#patch *= 20
				cpy_pix(info[0]-HSZ, info[0]+HSZ, info[1]-HSZ, info[1]+HSZ, patch, nir_img, is_color=0)
	
	print ' >> {} successfully transformed and saved as ./{}/{}.bmp'.format(input_image, OUTPUT_DIR, input_image[:-4])
	cv2.imwrite('./{}/{}.bmp'.format(OUTPUT_DIR, input_image[:-4]), 
				nir_img[HALF_HSZ:HALF_HSZ+orig_size[0], HALF_HSZ:HALF_HSZ+orig_size[1]])


def forward_gan(im_path, input_image, t_image, net_g, sess, step=4):
	im, orig_size = imread_with_pad(im_path + input_image)
	nir_img = np.zeros((im.shape[0], im.shape[1], 1))
	result = get_split_info(im, im.shape)
	print ' >> splitting original image and transforming '

	for i in tqdm(result.keys()):
		if result[i] != [9999, 0]:
			for j in range(result[i][0], result[i][1]+1, step):
				patch, info = get_single_patch(im, i, j, im.shape)
				cv2.imwrite('temp/temp.png', patch)

				rgb_img = tl.prepro.threading_data(['temp/temp.png'],fn=get_imgs_fn,path='')[0]
				rgb_img = (rgb_img / 127.5) - 1   # rescale to ［－1, 1]
				out = sess.run(net_g.outputs, {t_image: [rgb_img]})
				patch = resize_fn(out[0], [65, 65], 0)
				patch = patch[8:56, 8:56]
				HSZ = 24
				cpy_pix(info[0]-HSZ, info[0]+HSZ, info[1]-HSZ, info[1]+HSZ, patch, nir_img, is_color=1)

	print ' >> {} successfully transformed and saved as ./{}/{}.bmp'.format(input_image, OUTPUT_DIR, input_image[:-4])
	cv2.imwrite('./{}/{}.bmp'.format(OUTPUT_DIR, input_image[:-4]), 
				nir_img[HALF_HSZ:HALF_HSZ+orig_size[0], HALF_HSZ:HALF_HSZ+orig_size[1]])


def type_caffe(args):
	input_list = os.listdir(RGB_IMG_DIR)
	if not len(input_list):
		raise Exception(" ERR : NO FILE ")
	net, transformer = caffe_init('models/rgb2nir/1230_exp/use_grey_nir_deploy.prototxt', 'models/rgb2nir/1230_exp/trained_models/1230_use_grey_nir_iter_50000.caffemodel')
	#net, transformer = caffe_init('models/rgb2nir/deploy.prototxt',
	#								'models/rgb2nir/__model_conv_5__iter_50000.caffemodel')
	#net, transformer = caffe_init('models/rgb2nir/1220_exp/with_clarityLoss_deploy.prototxt',
	#							'models/rgb2nir/1220_exp/trained_models/1220_clarityLoss__iter_70000.caffemodel')

	for input_image in input_list:
		transform_single(RGB_IMG_DIR, input_image, net, transformer, args.step)


def type_tf(args):
	input_list = os.listdir(RGB_IMG_DIR)
	if not len(input_list):
		raise Exception(" ERR : NO FILE ")
	t_image = tf.placeholder('float32', [None, 65, 65, 3], name='input_image')
	net_g   = SRGAN_g(t_image, is_train=False, reuse=False)
	sess    = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
	tl.layers.initialize_global_variables(sess)
	tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/g_srgan.npz', network=net_g)
	for input_image in input_list:
		forward_gan(RGB_IMG_DIR, input_image, t_image, net_g, sess, args.step)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-step', '--step', type=int, default=4, 
						help='larger to make transformation faster(1~30)')
	parser.add_argument('--type', type=str, default='cnn',
						help="switch 'gan' for tf-gan and 'cnn' for caffe-cnn")
	args = parser.parse_args()
	dir_chk()
	if args.type == 'cnn':
		type_caffe(args)
	elif args.type == 'gan':
		type_tf(args)
