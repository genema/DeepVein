from easydict import EasyDict as edict
import json

config = edict()
config.TRAIN = edict()

## Adam
config.TRAIN.batch_size = 16
config.TRAIN.lr_init = 1e-4
config.TRAIN.beta1 = 0.9

## initialize G
config.TRAIN.n_epoch_init = 1
    # config.TRAIN.lr_decay_init = 0.1
    # config.TRAIN.decay_every_init = int(config.TRAIN.n_epoch_init / 2)

## adversarial learning (SRGAN)
config.TRAIN.n_epoch = 10
config.TRAIN.lr_decay = 0.1
config.TRAIN.decay_every = int(config.TRAIN.n_epoch / 2)

## train set location
config.TRAIN.nir_img_path = '/home/wb/RGB2NIR/Purged_Set/NIREnh_65/train/'
config.TRAIN.rgb_img_path = '/home/wb/RGB2NIR/Purged_Set/RGB_65/train/'

config.VALID = edict()
## test set location
config.VALID.nir_img_path = '/home/wb/RGB2NIR/Purged_Set/NIREnh_65/test/'
config.VALID.rgb_img_path = '/home/wb/RGB2NIR/Purged_Set/RGB_65/test/'
