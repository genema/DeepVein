# -*- coding: utf-8 -*-
# @Author: gehuama
# @Date:   2017-12-03 12:00:20
# @Last Modified by:   gehuama
# @Last Modified time: 2017-12-23 13:57:45
# =========================================================================================================
# RGB to NIR all in one 
# Usage:
# command line
# 	python rgb2nir.py --step x
# =========================================================================================================

from PARAMETERS import *
from helper_caffe import caffe_init, rgb2nir
from helper import cpy_pix, imread_with_pad


OUTPUT_DIR = 'clarityLoss_results'

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

def transform_single(im_path, net, transformer, step=1):
	im, orig_size = imread_with_pad(im_path)
	nir_img = np.zeros((im.shape[0], im.shape[1], 1))
	result = get_split_info(im, im.shape)
	print ' >> splitting original image and transforming'
	for i in tqdm(result.keys()):
		if result[i] != [9999, 0]:
			# print ' >> Transforming and splicing patches'
			for j in range(result[i][0], result[i][1]+1, step):
				patch, info = get_single_patch(im, i, j, im.shape)
				# cv2.imwrite('{}{}.png'.format(PATCH_SAVE_DIR, ind), patch)
				patch = rgb2nir(net, transformer, patch)
				patch = patch[4:60, 4:60]
				cpy_pix(info[0]-HSZ, info[0]+HSZ, info[1]-HSZ, info[1]+HSZ, patch, nir_img)
	
	print ' >> {} successfully transformed and saved as ./{}/{}.bmp'.format(input_image, OUTPUT_DIR, input_image[:-4])
	cv2.imwrite('./{}/{}.bmp'.format(OUTPUT_DIR, input_image[:-4]), 
				nir_img[HALF_HSZ:HALF_HSZ+orig_size[0], HALF_HSZ:HALF_HSZ+orig_size[1]])

def dir_chk():
	if not os.path.isdir(PATCH_SAVE_DIR):
		os.mkdir(PATCH_SAVE_DIR) 

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-step', '--step', type=int, default=1, 
						help='larger to make transformation faster(1~30)')
	args = parser.parse_args()
	dir_chk()
	#net, transformer = caffe_init('models/rgb2nir/deploy.prototxt',
	#								'models/rgb2nir/__model_conv_5__iter_50000.caffemodel')
	net, transformer = caffe_init('models/rgb2nir/1220_exp/with_clarityLoss_deploy.prototxt',
								'models/rgb2nir/1220_exp/trained_models/1220_clarityLoss__iter_70000.caffemodel')

	input_list = os.listdir(RGB_IMG_DIR)
	
	for input_image in input_list:
		transform_single(RGB_IMG_DIR + input_image, net, transformer, args.step)



