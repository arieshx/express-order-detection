# -*- coding:utf-8 -*-

from lib.fast_rcnn.config import cfg

import os, shutil, glob, cv2
"""
将east模型运行出的保存的结果转化成ctpn结果，然后方便计算准确率
"""
east_result_dir = os.path.join(cfg.ROOT_DIR, 'data', 'test_data_result')
save_dir = os.path.join(cfg.ROOT_DIR, 'data', 'gt_set_new_east')
if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
os.makedirs(save_dir)

list_jpg_path = glob.glob(os.path.join(east_result_dir, '[0-9]*.jpg'))
for jpg_path in list_jpg_path:
    txt_path = os.path.join(east_result_dir,'0719_kuaidi_split_512_512_'+os.path.basename(jpg_path)+'.txt')
    with open(txt_path, 'r') as f:
        east_lines = f.readlines()

    jpg_save_path = os.path.join(save_dir, os.path.basename(jpg_path))
    txt_save_path = os.path.join(save_dir, 'res_'+os.path.basename(jpg_path)[:-4]+'.txt')

    img = cv2.imread(jpg_path)

    with open(txt_save_path, 'w') as f:
        for east_line in east_lines:
            list_east_line = east_line.strip().split(',')
            list_txt = [int(int(list_east_line[0])/0.9), int(int(list_east_line[1])/0.9),int(int(list_east_line[4])/0.9), int(int(list_east_line[5])/0.9)]
            list_txt = [str(_) for _ in list_txt]
            x0,y0,x1,y1 = int(list_txt[0]), int(list_txt[1]), int(list_txt[2]), int(list_txt[3])
            txt_line = ','.join(list_txt)+'\n'

            cv2.rectangle(img, (x0,y0), (x1, y1), (0, 255, 0), 1)

            f.write(txt_line)

    cv2.imwrite(jpg_save_path, img)


