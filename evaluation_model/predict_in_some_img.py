#-*- coding:utf-8 -*-

from __future__ import print_function

import cv2
import glob
import os
import shutil
import sys

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.getcwd()))
from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg, cfg_from_file
from lib.fast_rcnn.test import test_ctpn
from lib.utils.timer import Timer
from lib.text_connector.detectors import TextDetector
from lib.text_connector.text_connect_cfg import Config as TextLineCfg

os.environ['CUDA_VISIBLE_DEVICES']='0'
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
    plt.imshow(img)
    plt.show()

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


def local_result(img_path, save_dir, ctpn_path, base_net):
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
    
    ckpt = tf.train.get_checkpoint_state(ctpn_path)
    print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
    saver.restore(sess, ckpt.model_checkpoint_path)
    print('done')
    try:
        ckpt = tf.train.get_checkpoint_state(ctpn_path)
        print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('done')
    except Exception:
        print('Check your pretrained {:s}'.format(ckpt.model_checkpoint_path))

    im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    for i in range(2):
        _, _ = test_ctpn(sess, training_flag, net, im)

    for im_name in img_path:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(('Demo for {:s}'.format(im_name)))
        # if os.path.basename(im_name)!='30_5357176972_20180629182525_20180629202528_163.jpg':
        #     continue
        ctpn(sess, training_flag, net, im_name, save_dir)


def ctpn_test():
    pass

if __name__ == '__main__':
    base_net = 'Densenet_test' # VGGnet_test or Densenet_test
    img_path = glob.glob('../data/164_jpg/*.jpg')
    save_dir = 'art_rank_result'
    
    ctpn_path = os.path.join(cfg.ROOT_DIR, 'output_vgg/ctpn_end2end/voc_2007_trainval')
    print('ctpn_path:', ctpn_path)
    local_result(img_path, save_dir, ctpn_path, base_net)

