# -*- coding:utf-8 -*-
import os.path
import pprint
import sys
import tensorflow as tf

sys.path.append(os.getcwd())
this_dir = os.path.dirname(__file__)

from lib.fast_rcnn.train import get_training_roidb, train_net, get_test_roidb
from lib.fast_rcnn.config import cfg_from_file, get_output_dir, get_log_dir
from lib.datasets.factory import get_imdb
from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg

# 在服务器
os.environ['CUDA_VISIBLE_DEVICES']='3'
# 在郝鑫本机
# CUDA_VISIBLE_DEVICES = 0
if __name__ == '__main__':
    yml_path = os.path.join(this_dir, 'text.yml')
    cfg_from_file(yml_path)
    cfg.TRAIN.restore = 1
    print('Using config:')
    pprint.pprint(cfg)
    imdb = get_imdb('voc_2007_trainval')
    imdb_test = get_imdb('voc_2007_test')
    print('Loaded dataset `{:s}` for training'.format(imdb.name))
    print('Loaded dataset `{:s}` for test'.format(imdb_test.name))
    roidb = get_training_roidb(imdb)
    roidb_test = get_test_roidb(imdb_test)

    output_dir = get_output_dir(imdb, None)
    log_dir = get_log_dir(imdb)
    print('Output will be saved to `{:s}`'.format(output_dir))
    print('Logs will be saved to `{:s}`'.format(log_dir))

    # device_name = '/gpu:'+str(CUDA_VISIBLE_DEVICES)
    # print(device_name)

    # 用vgg16训练
    training_flag = tf.placeholder(tf.bool)
    network = get_network('VGGnet_train',training_flag)
    pretrained_model = 'data/pretrain/VGG_imagenet.npy'

    # 用densenet作为骨干网络，不加载预训练权重
    # training_flag = tf.placeholder(tf.bool)
    # network = get_network('Densenet_train', training_flag)
    # pretrained_model = None

    train_net(network, training_flag, imdb, roidb,
              imdb_test,
              roidb_test,
              output_dir=output_dir,
              log_dir=log_dir,
              pretrained_model=pretrained_model,
              max_iters=int(cfg.TRAIN.max_steps),
              restore=bool(int(cfg.TRAIN.restore)))
