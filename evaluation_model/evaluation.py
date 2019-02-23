#-*- coding:utf-8 -*-

from __future__ import print_function

import cv2
import glob
import os
import shutil
import sys

sys.path.append(os.getcwd())

from lib.fast_rcnn.config import cfg



def resize_im(im, scale, max_scale=None):
    f = float(scale) / min(im.shape[0], im.shape[1])
    if max_scale != None and f * max(im.shape[0], im.shape[1]) > max_scale:
        f = float(max_scale) / max(im.shape[0], im.shape[1])
    return cv2.resize(im, None, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR), f




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

def iou_area(rec1, rec2):
    # 计算两个矩形的重叠面积
    endx = max(rec1.x1, rec2.x1);
    startx = min(rec1.x0, rec2.x0);
    width = rec1.width + rec2.width - (endx - startx);

    endy = max(rec1.y1, rec2.y1);
    starty = min(rec1.y0, rec2.y0);
    height = rec1.height + rec2.height - (endy - starty);

    if width <= 0 or height <= 0:
        Area = 0  # 重叠率为 0
    else:
        Area = width * height;  # 两矩形相交面积
    return Area, width




def is_qualified_two_img(img, list_local_frame, list_gt_frame):
    """

    :param list_local_frame:
    :param list_gt_frame:
    :return:
    """

    #
    list_match_local = list()
    list_poss_local = list()
    list_match_pair = list()
    list_poss_pair = list()
    # list_poss_gt = list()
    for idx_gt, one_gt_frame in enumerate(list_gt_frame):
        gt_rec = rectangle(one_gt_frame[0], one_gt_frame[1], one_gt_frame[2], one_gt_frame[3])

        list_chance_local_frame = list()
        list_chance_local_id = list()
        for idx_local, one_local_frame in enumerate(list_local_frame):
            local_rec = rectangle(one_local_frame[0], one_local_frame[1], one_local_frame[2], one_local_frame[3])

            ratio, vertical_overlap, horizontal_overlap = iou(gt_rec, local_rec)

            # img_copy = img.copy()
            if ratio > 0.1 and vertical_overlap > 0.5 and horizontal_overlap > 0.1:  # 认为该local框和gt框是同一行内的
                list_chance_local_id.append(idx_local)
                list_chance_local_frame.append(one_local_frame)


                # x0, y0, x1, y1 = gt_rec.x0, gt_rec.y0, gt_rec.x1, gt_rec.y1
                # cv2.rectangle(img_copy, (x0, y0), (x1, y1), (0, 0, 0), 2)
                # x0, y0, x1, y1 = local_rec.x0, local_rec.y0, local_rec.x1, local_rec.y1
                # cv2.rectangle(img_copy, (x0, y0), (x1, y1), (255, 0, 0), 2)
                # plt.imshow(img_copy)
                # plt.show()
            else:

                # x0, y0, x1, y1 = gt_rec.x0, gt_rec.y0, gt_rec.x1, gt_rec.y1
                # cv2.rectangle(img_copy, (x0, y0), (x1, y1), (0, 0, 0), 2)
                # x0, y0, x1, y1 = local_rec.x0, local_rec.y0, local_rec.x1, local_rec.y1
                # cv2.rectangle(img_copy, (x0, y0), (x1, y1), (0, 0, 255), 2)
                # plt.imshow(img_copy)
                # plt.show()
                continue
        # draw 判断是否一行的结果
        # del img_copy
        # img_copy = img.copy()
        # x0, y0, x1, y1 = gt_rec.x0, gt_rec.y0, gt_rec.x1, gt_rec.y1
        # cv2.rectangle(img_copy, (x0, y0), (x1, y1), (0, 0, 0), 2)
        # for local_frame in list_chance_local_frame:
        #     x0, y0, x1, y1 = local_frame[0], local_frame[1], local_frame[2], local_frame[3]
        #     cv2.rectangle(img_copy, (x0, y0), (x1, y1), (0, 0, 255), 2)
        #
        # plt.imshow(img_copy)
        # plt.show()
        #
        if len(list_chance_local_id) == 0:
            list_poss_pair.append({'gt_frame': one_gt_frame, 'local_frame': []})
        elif len(list_chance_local_id) == 1:
            local_frame = list_chance_local_frame[0]
            local_rec = rectangle(local_frame[0], local_frame[1], local_frame[2], local_frame[3])
            ratio, vertical_overlap, horizontal_overlap = iou(gt_rec, local_rec)

            overlap_gt_area, overlap_gt_horizontal_len = iou_area(gt_rec, local_rec)
            if ratio > 0.5 or (horizontal_overlap>0.5 and vertical_overlap>0.5) or overlap_gt_area>0.75*gt_rec.width*gt_rec.height or overlap_gt_horizontal_len>0.9*gt_rec.width:  # 1:1,直接信任
                list_match_pair.append({'gt_frame': one_gt_frame, 'local_frame': [local_frame]})
                list_match_local.append(local_frame)
            else:
                list_poss_pair.append({'gt_frame': one_gt_frame, 'local_frame': [local_frame]})
                list_poss_local.append(local_frame)
        else:
            overlap_gt_area = 0
            overlap_gt_horizontal_len = 0
            for frame in list_chance_local_frame:
                local_rec = rectangle(frame[0], frame[1], frame[2], frame[3])
                _, hon_len = iou_area(gt_rec, local_rec)
                overlap_gt_area += _
                overlap_gt_horizontal_len+=hon_len

            if overlap_gt_area > 0.7*gt_rec.width*gt_rec.height or overlap_gt_horizontal_len>0.7*gt_rec.width:
                list_match_pair.append({'gt_frame': one_gt_frame, 'local_frame': list_chance_local_frame})
                list_match_local.extend(list_chance_local_frame)
            else:
                list_poss_pair.append({'gt_frame': one_gt_frame, 'local_frame': list_chance_local_frame})
                list_poss_local.extend(list_chance_local_frame)

    if len(list_match_pair) == len(list_gt_frame) and len(list_poss_pair) == 0:
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

        list_match_local, list_poss_local, list_match_pair, list_poss_pair, flag = is_qualified_two_img(img.copy(), list_local_frame, list_gt_frame)

        # draw in img
        for frame in list_local_frame:
            x0,y0,x1,y1 = frame[0], frame[1], frame[2], frame[3]
            cv2.rectangle(img,(x0,y0), (x1,y1),(0,255,0),2)
        for frame in list_gt_frame:
            x0, y0, x1, y1 = frame[0], frame[1], frame[2], frame[3]
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), 1)

        if flag:
            correct_num+=1
            jpg_vis_path = os.path.join(success_dir, jpg_name)
            cv2.imwrite(jpg_vis_path, img)
        else:
            for _ in list_poss_pair:
                gt_frame = _['gt_frame']
                list_local_frame = _['local_frame']

                x0,y0,x1,y1 = gt_frame[0], gt_frame[1], gt_frame[2], gt_frame[3]
                cv2.rectangle(img, (x0,y0), (x1, y1), (0, 0, 0), 2)
                for local_frame in list_local_frame:
                    x0, y0, x1, y1 = local_frame[0], local_frame[1], local_frame[2], local_frame[3]
                    cv2.rectangle(img, (x0, y0), (x1, y1),(0,125,125), 1)

            error_num+=1
            jpg_vis_path = os.path.join(fail_dir, jpg_name)
            cv2.imwrite(jpg_vis_path, img)
    pass_rate = correct_num*1./len(list_local_txt_path)
    print("通过率：",pass_rate)
    print("查看通过图片的目录：",success_dir,'\n查看不通过图片的目录:',fail_dir)

if __name__ == '__main__':
    save_compare_dir = 'gt_set_densenet2_compare'
    local_result_dir = 'gt_set_densenet2'
    comapre_local_gt(save_compare_dir,local_result_dir)
