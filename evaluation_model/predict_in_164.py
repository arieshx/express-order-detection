#-*- coding:utf-8 -*-
"""
对164数据进行检测可视化
"""
from __future__ import print_function

import cv2
import glob
import os
import shutil
import sys

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())
from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg, cfg_from_file
from lib.fast_rcnn.test import test_ctpn
from lib.utils.timer import Timer
from lib.text_connector.detectors import TextDetector
from lib.text_connector.text_connect_cfg import Config as TextLineCfg

os.environ['CUDA_VISIBLE_DEVICES']='2'
def resize_im(im, scale, max_scale=None):
    f = float(scale) / min(im.shape[0], im.shape[1])
    if max_scale != None and f * max(im.shape[0], im.shape[1]) > max_scale:
        f = float(max_scale) / max(im.shape[0], im.shape[1])
    return cv2.resize(im, None, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR), f


def draw_boxes(img, image_name, boxes, scale, save_all_dir):
    base_name = image_name.split('/')[-1]
    with open(save_all_dir + '/res_{}.txt'.format(base_name.split('.')[0]), 'w') as f:
        for box in boxes:
            if box[8] >= 0.9:
                color = (0, 255, 0)
            else :
                color = (255, 0, 0)
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[6]), int(box[7])), color, 2)

            min_x = min(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
            min_y = min(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))
            max_x = max(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
            max_y = max(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))

            line = ','.join([str(min_x), str(min_y), str(max_x), str(max_y)]) + '\n'
            f.write(line)

    img = cv2.resize(img, None, None, fx=1.0 / scale, fy=1.0 / scale, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join(save_all_dir, base_name), img)
    # plt.imshow(img)
    # plt.show()

def draw_middle_boxes(img, boxes, scale):
    """
    将模型预测出来的文本线可视化，nms之前的结果
    :param img:
    :param image_name:
    :param boxes:
    :param scale:
    :param save_all_dir:
    :return:
    """
    for box in boxes:
        color = (0, 255, 0)
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)

    img = cv2.resize(img, None, None, fx=1.0 / scale, fy=1.0 / scale, interpolation=cv2.INTER_LINEAR)
    plt.imshow(img)
    plt.show()

def ctpn(sess, training_flag, net, image_name, save_all_dir):
    timer = Timer()
    timer.tic()

    img = cv2.imread(image_name)
    img, scale = resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
    scores, boxes = test_ctpn(sess, training_flag, net, img)

    textdetector = TextDetector()
    boxes1, boxes2, boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
    # img1 = img.copy()
    # draw_middle_boxes(img1, boxes1, scale)
    # img2 = img.copy()
    # draw_middle_boxes(img2, boxes2, scale)
    draw_boxes(img, image_name, boxes, scale, save_all_dir)
    timer.toc()
    print(('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0]))


def local_result(list_img_path, save_dir, ctpn_path, base_net):
    save_dir = os.path.join(cfg.ROOT_DIR, 'data', save_dir)
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    yml_path = os.path.join(cfg.ROOT_DIR, 'ctpn/text.yml')
    cfg_from_file(yml_path)


    # init session
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    # load network
    training_flag = tf.placeholder(tf.bool)
    net = get_network(base_net, training_flag)
    # load model
    print(('Loading network {:s}... '.format(base_net)), end=' ')
    saver = tf.train.Saver()

    try:
        ckpt = tf.train.get_checkpoint_state(ctpn_path)
        print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('done')
    except:
        raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)

    im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    for i in range(2):
        _, _ = test_ctpn(sess, training_flag, net, im)

    im_names = list_img_path

    # im_names = glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.png')) + \
    #            glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.jpg'))

    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(('Demo for {:s}'.format(im_name)))
        # if os.path.basename(im_name)!='30_5357176972_20180629182525_20180629202528_163.jpg':
        #     continue
        ctpn(sess,training_flag,  net, im_name, save_dir)


def ctpn_test():
    pass

def get_some_164():
    """
    获取一些164数据用来测试模型在164上的效果
    :return:
    """
    all_data_dir = '/media/haoxin/A1/all_data'
    list_164_img_path = glob.glob(all_data_dir + '/*164.jpg')[0: 10]
    dis_dir = '../data/164_jpg'
    for img_path in list_164_img_path:
        shutil.copy(img_path, dis_dir)

if __name__ == '__main__':
    # get_some_164()
    base_net = 'VGGnet_test' # VGGnet_test or Densenet_test
    list_img_path = glob.glob('../data/daily_error/*.jpg')
    save_dir = 'daily_error_result'
    ctpn_path = os.path.join(cfg.ROOT_DIR, 'output/ctpn_end2end/voc_2007_trainval')
    local_result(list_img_path, save_dir, ctpn_path, base_net)

# gt中表现有问题的图片
# 25_5746164554_20180701125111_20180701145113_163.jpg
# gt中图片，框重复
# 25_5583747546_20180630163309_20180630183316_163.jpg  查看省市县分开现象
# 25_5596205399_20180628200536_20180628220541_163.jpg
# 25_5630433101_20180630174719_20180630194724_161.   25_5746724107_20180629191951_20180629211955_161.jpg
# 25_5660232676_20180628162925_20180628182930_161.jpg  25_5666385980_20180626183822_20180626203827_163.jpg  25_5704429631_20180628180824_20180628200831_161.jpg
# 25_5636156079_20180626185631_20180626205634_161.jpg 短框
# 25_5753375165_20180630172357_20180630192401_161.jpg 将电话分开了。

# 30_5357176972_20180629182525_20180629202528_163.jpg  50_5758149814_20180628212902_20180629100000_163.jpg