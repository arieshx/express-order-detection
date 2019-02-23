#-*- coding:utf-8 -*-

from __future__ import print_function

import cv2
import glob
import os
import shutil
import sys

import numpy as np
import tensorflow as tf

sys.path.append(os.getcwd())
from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg, cfg_from_file
from lib.fast_rcnn.test import test_ctpn
from lib.utils.timer import Timer
from lib.text_connector.detectors import TextDetector
from lib.text_connector.text_connect_cfg import Config as TextLineCfg


def resize_im(im, scale, max_scale=None):
    f = float(scale) / min(im.shape[0], im.shape[1])
    if max_scale != None and f * max(im.shape[0], im.shape[1]) > max_scale:
        f = float(max_scale) / max(im.shape[0], im.shape[1])
    return cv2.resize(im, None, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR), f


def draw_boxes(img, image_name, boxes, scale, save_all_dir):
    base_name = image_name.split('/')[-1]
    with open(save_all_dir + '/res_{}.txt'.format(base_name.split('.')[0]), 'w') as f:
        for box in boxes:
            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                continue
            if box[8] >= 0.9:
                color = (0, 255, 0)
            elif box[8] >= 0.8:
                color = (255, 0, 0)
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[4]), int(box[5])), color, 2)
            cv2.line(img, (int(box[6]), int(box[7])), (int(box[2]), int(box[3])), color, 2)
            cv2.line(img, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), color, 2)

            min_x = min(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
            min_y = min(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))
            max_x = max(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
            max_y = max(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))

            line = ','.join([str(min_x), str(min_y), str(max_x), str(max_y)]) + '\n'
            f.write(line)

    img = cv2.resize(img, None, None, fx=1.0 / scale, fy=1.0 / scale, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join(save_all_dir, base_name), img)


def ctpn(sess, net, image_name, save_all_dir):
    timer = Timer()
    timer.tic()

    img = cv2.imread(image_name)
    img, scale = resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
    scores, boxes = test_ctpn(sess, net, img)

    textdetector = TextDetector()
    boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
    draw_boxes(img, image_name, boxes, scale, save_all_dir)
    timer.toc()
    print(('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0]))


def local_result(save_dir):
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
    net = get_network("VGGnet_test")
    # load model
    print(('Loading network {:s}... '.format("VGGnet_test")), end=' ')
    saver = tf.train.Saver()

    _ = os.path.join(cfg.ROOT_DIR, 'output_old_auto/ctpn_end2end/voc_2007_trainval')
    try:
        ckpt = tf.train.get_checkpoint_state(_)
        print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('done')
    except:
        raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)

    im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    for i in range(2):
        _, _ = test_ctpn(sess, net, im)

    im_names = glob.glob(os.path.join(cfg.DATA_DIR, 'gt_set', '*.jpg'))

    # im_names = glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.png')) + \
    #            glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.jpg'))

    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(('Demo for {:s}'.format(im_name)))
        ctpn(sess, net, im_name, save_dir)




class rectangle(object):
    def __init__(self,x0=None,y0=None,x1=None,y1 = None):
        self.x0 = int(x0)
        self.x1 = int(x1)
        self.y0 = int(y0)
        self.y1 = int(y1)
        self.width = abs(self.x0-self.x1)
        self.height = abs(self.y0-self.y1)


def iou(rec1,rec2):
    """


    :param rec1:
    :param rec2:
    :return:
    """
    # 计算两个矩形的iou
    endx = max(rec1.x1 , rec2.x1);
    startx = min(rec1.x0, rec2.x0);
    width = rec1.width + rec2.width - (endx - startx);

    endy = max(rec1.y1 , rec2.y1);
    starty = min(rec1.y0, rec2.y0);
    height = rec1.height + rec2.height - (endy - starty);

    if width <= 0 or height <= 0:
        ratio = 0  # 重叠率为 0
    else:
        Area = width * height;  # 两矩形相交面积
        Area1 = rec1.width * rec1.height
        Area2 = rec2.width*rec2.height
        ratio = Area * 1. / (Area1 + Area2 - Area);
    # 计算两个矩形的水平和竖直方向的overlap
    vertical_overlap = height * 1. / (endy - starty)
    horizontal_overlap = width * 1. / (endx - startx)

    return ratio,vertical_overlap,horizontal_overlap

def is_qualified_two_img(list_local_frame, list_gt_frame):
    """

    :param list_local_frame:
    :param list_gt_frame:
    :return:
    """
    list_match_local = list()
    list_poss_local = list()
    list_match_pair = list()
    list_poss_pair = list()
    #list_poss_gt = list()
    for idx_gt,one_gt_frame in enumerate(list_local_frame):
        gt_rec = rectangle(one_gt_frame[0], one_gt_frame[1], one_gt_frame[2], one_gt_frame[3])

        list_chance_local_frame = list()
        list_chance_local_id = list()
        for idx_local, one_local_frame in enumerate(list_gt_frame):
            local_rec  = rectangle(one_local_frame[0], one_local_frame[1], one_local_frame[2], one_local_frame[3])

            ratio, vertical_overlap, horizontal_overlap = iou(gt_rec, local_rec)

            if ratio>0.1 and vertical_overlap>0.5 and horizontal_overlap>0.1 :# 认为该local框和gt框是同一行内的
                list_chance_local_id.append(idx_local)
                list_chance_local_frame.append(one_local_frame)
            else:continue
        #
        if len(list_chance_local_id) == 0:
            list_poss_pair.append({'gt_frame':one_gt_frame, 'local_frame':[]})
        elif len(list_chance_local_id) == 1:
            local_frame = list_chance_local_frame[0]
            local_rec = rectangle(local_frame[0], local_frame[1], local_frame[2], local_frame[3])
            ratio, vertical_overlap, horizontal_overlap = iou(gt_rec, local_rec)

            if horizontal_overlap > 0.5:  # 1:1,直接信任
                list_match_pair.append({'gt_frame':one_gt_frame, 'local_frame':[local_frame]})
                list_match_local.append(local_frame)
            else:
                list_poss_pair.append({'gt_frame':one_gt_frame, 'local_frame':[local_frame]})
                list_poss_local.append(local_frame)
        else:
            combine_one_line_frame = [min([_[0] for _ in list_chance_local_frame ]),
                                      min([_[1] for _ in list_chance_local_frame]),
                                      max([_[2] for _ in list_chance_local_frame]),
                                      max([_[3] for _ in list_chance_local_frame])
                                      ]
            combine_one_line_rec = rectangle(combine_one_line_frame[0], combine_one_line_frame[1], combine_one_line_frame[2], combine_one_line_frame[3])

            ratio, vertical_overlap, horizontal_overlap = iou(gt_rec, combine_one_line_rec)

            if ratio > 0.4 and horizontal_overlap > 0.6 :  # 1:n的信任条件
                list_match_pair.append({'gt_frame': one_gt_frame, 'local_frame': list_chance_local_frame})
                list_match_local.extend(list_chance_local_frame)
            else:
                list_poss_pair.append({'gt_frame': one_gt_frame, 'local_frame': list_chance_local_frame})
                list_poss_local.extend(list_chance_local_frame)

    #list_match_local = list(set(list_match_local))
    if len(list_match_local) == len(list_gt_frame) and len(list_poss_local) == 0 :
        flag = True
    else:
        flag = False
    return list_match_local, list_poss_local, list_match_pair, list_poss_pair, flag



def comapre_local_gt(save_dir, local_result_dir):

    local_compare_dir = os.path.join(cfg.ROOT_DIR, 'data',save_dir)
    success_dir = os.path.join(local_compare_dir,'success')
    fail_dir = os.path.join(local_compare_dir,'fail')
    if os.path.exists(success_dir):
        shutil.rmtree(success_dir)
    os.makedirs(success_dir)

    if os.path.exists(fail_dir):
        shutil.rmtree(fail_dir)
    os.makedirs(fail_dir)

    gt_result_dir = os.path.join(cfg.ROOT_DIR, 'data/gt_set/')
    local_result_dir = os.path.join(cfg.ROOT_DIR, 'data',local_result_dir)
    list_local_txt_path = glob.glob(os.path.join(local_result_dir, '*.txt'))
    list_gt_txt_path = glob.glob(gt_result_dir+'*.txt')
    correct_num = 0
    error_num = 0
    for i , local_txt_path in enumerate(list_local_txt_path):
        list_local_frame = list()
        list_gt_frame = list()
        txt_name = os.path.basename(local_txt_path)[4:]
        jpg_name = txt_name[:-4]+'.jpg'
        jpg_path = os.path.join(gt_result_dir, jpg_name)
        img = cv2.imread(jpg_path)
        gt_txt_path = os.path.join(gt_result_dir, txt_name)
        assert gt_txt_path in list_gt_txt_path,'gt txt must exist.'
        with open(local_txt_path, 'r') as f:
            for line in f.readlines():
                a = line.strip('\n').split(',')
                b = [int(_) for _ in a]
                assert len(a) == 4,'local frame  list lenght must be four'
                list_local_frame.append(b)
        with open(gt_txt_path, 'r') as f:
            for line in f.readlines():
                a = line.strip('\n').split(',')
                b = [int(_) for _ in a]
                assert len(a) == 4, 'gt frame  list lenght must be four'
                list_gt_frame.append(b)

        list_match_local, list_poss_local, list_match_pair, list_poss_pair, flag = is_qualified_two_img(list_local_frame, list_gt_frame)

        # draw in img
        for frame in list_local_frame:
            x0,y0,x1,y1 = frame[0], frame[1], frame[2], frame[3]
            cv2.rectangle(img,(x0,y0), (x1,y1),(0,255,0),1)
        for frame in list_gt_frame:
            x0, y0, x1, y1 = frame[0], frame[1], frame[2], frame[3]
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), 2)

        if flag:
            correct_num+=1
            jpg_vis_path = os.path.join(success_dir, jpg_name)
            cv2.imwrite(jpg_vis_path, img)
        else:
            for frame in list_poss_local:
                x0, y0, x1, y1 = frame[0], frame[1], frame[2], frame[3]
                cv2.rectangle(img, (x0, y0), (x1, y1), (125, 125, 0), 2)

            error_num+=1
            jpg_vis_path = os.path.join(fail_dir, jpg_name)
            cv2.imwrite(jpg_vis_path, img)
    pass_rate = correct_num*1./len(list_local_txt_path)
    print("通过率：",pass_rate)
    print("查看通过图片的目录：",success_dir,'\n查看不通过图片的目录:',fail_dir)

if __name__ == '__main__':
    #save_dir = 'gt_set_old_auto5'
    #local_result(save_dir)
    save_compare_dir = 'gt_set_esat_compare'
    local_result_dir = 'gt_set_east'
    comapre_local_gt(save_compare_dir,local_result_dir)
