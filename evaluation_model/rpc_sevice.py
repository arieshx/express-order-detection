#coding=utf-8


import cv2
import glob
import zerorpc
import os, sys
import numpy as np
import json
import time, shutil, base64

sys.path.append(os.path.dirname(os.getcwd()))

from matplotlib import pyplot as plt
from lib.fast_rcnn.config import cfg


def _img_to_str_base64(image):
    """ convert image to base64 string
    """
    img_encode = cv2.imencode('.jpg', image)[1]
    img_base64 = base64.b64encode(img_encode)
    return img_base64

# ----------------------------------------
# 连接两个检测 RPC 服务
# ----------------------------------------
c_det_east = zerorpc.Client()
c_det_east.connect("tcp://192.168.1.115:18000")  # kuaidi

c_det_frcnn = zerorpc.Client()
c_det_frcnn.connect("tcp://192.168.1.115:18765")  # kuaidi


def east_predict(list_img_path, save_dir):
    # 获得所有的图片
    count_max = len(list_img_path)
    print("There are imgs: ", count_max)

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)

    for i, img_path in enumerate(list_img_path):
        image = cv2.imread(img_path)

        # 传输给服务器的数据
        data = {'fname': img_path, 'img_str': _img_to_str_base64(image)}

        # test by EAST mode
        res_east_detect = c_det_east.detect(data, 0.8, False)  # Debug mode, test image is need labeled?
        bboxdict_east = res_east_detect['data']

        # draw east
        list_frame = list()
        for idx, inst in enumerate(bboxdict_east):
            x0, y0, x1, y1 = int(inst['x0']), int(inst['y0']), int(inst['x2']), int(inst['y2'])
            list_frame.append([x0,y0,x1,y1])

            cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 2)

        jpg_name = os.path.basename(img_path)
        save_img_path = os.path.join(save_dir, jpg_name )
        txt_name = 'res_'+jpg_name[:-4]+'.txt'
        save_txt_path = os.path.join(save_dir, txt_name)

        plt.imshow(image)
        plt.show()
        cv2.imwrite(save_img_path, image)
        with open(save_txt_path, 'w') as f:
            for frame in list_frame:
                str_frame = [str(_) for _ in frame]
                text = ','.join(str_frame)+'\n'
                f.write(text)

        if i%1000 == 0:
            print('east predict finish:', i)

def ctpn_predict(list_img_path, save_dir):
    # 获得所有的图片
    count_max = len(list_img_path)
    print("There are imgs: ", count_max)

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)

    for i, img_path in enumerate(list_img_path):
        image = cv2.imread(img_path)

        # 传输给服务器的数据
        data = {'fname': img_path, 'img_str': _img_to_str_base64(image)}

        res_frcnn_detect = c_det_frcnn.detect(data)  # Debug mode, test image is need labeled?  #报错

        bboxdict_frcnn = res_frcnn_detect['data']['bbox_list']

        # draw east
        list_frame = list()
        for idx, inst in enumerate(bboxdict_frcnn):
            rect = inst['bbox']
            x0, y0, x1, y1 = rect[0], rect[1], rect[2], rect[3]
            list_frame.append([x0, y0, x1, y1])

            cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 2)

        jpg_name = os.path.basename(img_path)
        save_img_path = os.path.join(save_dir, jpg_name)
        txt_name = 'res_' + jpg_name[:-4] + '.txt'
        save_txt_path = os.path.join(save_dir, txt_name)

        plt.imshow(image)
        plt.show()
        cv2.imwrite(save_img_path, image)
        with open(save_txt_path, 'w') as f:
            for frame in list_frame:
                str_frame = [str(_) for _ in frame]
                text = ','.join(str_frame) + '\n'
                f.write(text)

        if i % 1000 == 0:
            print('ctpn predict finish:', i)


if __name__ == '__main__':

    list_img_path = glob.glob(os.path.join(cfg.DATA_DIR, '164_jpg', '*.*'))[0:3]
    save_dir = os.path.join(cfg.DATA_DIR, 'art_rank_result')
    # east_predict(list_img_path, save_dir)
    ctpn_predict(list_img_path, save_dir)
