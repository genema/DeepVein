# -*- coding: utf-8 -*-
# @Author: gehuama
# @Date:   2017-11-25 16:28:47
# @Last Modified by:   gehuama
# @Last Modified time: 2017-12-04 14:43:48
# This is a prototype script for transforming RGB images to NIR-like images.
# Also include some visualization functions.

from PARAMETERS import *

TRANSFORMED_IMG_SAVE_PATH = '/home/wb/RGB2NIR/transformed_images/'
TEST_IMG_PATH = '/home/wb/Downloads/CNN_experiment/'
RGB_IMG_LIST = os.listdir(TEST_IMG_PATH)
caffe_root = '/home/wb/Env/caffe/'
counter = 0

def vis_feat_rgb(data):
  merged_img = np.dstack([data[0], data[1], data[2]])
  return merged_img


def caffe_init(net_cfg, net_weight, use_cpu_flag=0, gpu_id=0):
  if use_cpu_flag:
    caffe.set_mode_cpu()
  else:
    caffe.set_device(gpu_id)
    caffe.set_mode_gpu()
  net = caffe.Net(caffe_root + net_cfg,
                  caffe_root + net_weight,
                  caffe.TEST)
  transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
  transformer.set_transpose('data', (2,0,1))
  #transformer.set_raw_scale('data', 255)
  #transformer.set_channel_swap('data', (2,1,0))
  return net, transformer


def transform_rgb_img(net, transformer, img_name, plot_flag=0, show_mat=0, calc_l2=0, save_img=1):
  global counter
  image = caffe.io.load_image(TEST_IMG_PATH + img_name)
  image = transformer.preprocess('data', image)
  net.blobs['data'].data[...] = image
  net.forward()
  im          = vis_feat_rgb(net.blobs['gen_image'].data[0][:3])
  #im_orig     = Image.open(TEST_IMG_PATH + 'NIREnh_65/' + img_name)
  rgb_mean    = np.sum(im, 2)
  rgb_mean    /= 3
  rgb_max     = np.max(im, 2)
  im_mean_RGB = np.dstack([rgb_mean, rgb_mean, rgb_mean])
  im_max_RGB  = np.dstack([rgb_max, rgb_max, rgb_max])
  #im_orig_arr = np.array(im_orig)
  #im_orig_arr = im_orig_arr.astype(np.float32)
  #im_orig_arr /= 255 #because of the normalization operation in image input layer
  if plot_flag:
    plt.subplot(221)
    plt.imshow(im_orig)
    plt.title('NIR_IMG')
    plt.subplot(222)
    plt.imshow(im)
    plt.title('transformed_img')
    plt.subplot(223)
    plt.imshow(im_mean_RGB)
    plt.title('mean_RGB_transformed_img')
    plt.subplot(224)
    plt.imshow(im_max_RGB)
    plt.title('max_RGB_transformed_img')
  if show_mat:
    im_arr = np.array(im)
    im_arr = im_arr.astype(np.float32)
    print '>>>>>>>>>>>>original NIR image:'
    print im_orig_arr
    print '>>>>>>>>>>>>transformed image:'
    print im_arr
  if calc_l2:
    print '>> l2 loss: %.3f ' % (np.linalg.norm(im_orig_arr - im))
  if save_img:
    '''
    plt.imshow(im)
    plt.axis('off')
    plt.savefig(TRANSFORMED_IMG_SAVE_PATH + img_name)
    '''
    #Image.fromarray(im, 'RGB').save(TRANSFORMED_IMG_SAVE_PATH + img_name)
    cv2.imwrite(TRANSFORMED_IMG_SAVE_PATH + img_name, im)
    counter += 1
    print ' >> No.%d: %s transformed and saved successfully' % (counter, img_name)

def rgb2nir(net, transformer, im):
  #im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)  
  #im = im/255. 
  im = transformer.preprocess('data', im)
  net.blobs['data'].data[...] = im
  net.forward()
  im = vis_feat_rgb(net.blobs['gen_image'].data[0][:3])
  return im

'''
if __name__ == '__main__':
  net, transformer = caffe_init('models/rgb2nir/deploy.prototxt',
                                'models/rgb2nir/__model_conv_5__iter_50000.caffemodel')
  for name in RGB_IMG_LIST:
    transform_rgb_img(net, transformer, name, 0, 0, 0, 1)
  print ' >> done'
'''
