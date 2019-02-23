#-*- coding:utf-8 -*-

"""
从大数据集中筛选出目前east检测不对的数据,对模型进行训练
"""
import cv2, os , glob, base64, shutil
import matplotlib.pyplot as plt
from lib.fast_rcnn.config import cfg
import zerorpc
from evaluation_model import rpc_sevice, evaluation
import xml.etree.ElementTree as ET


def show(img_path):
    img = cv2.imread(img_path)
    plt.imshow(img)
    plt.show()


# ----------------------------------------
# 连接两个检测 RPC 服务
# ----------------------------------------
c_det_east = zerorpc.Client()
c_det_east.connect("tcp://192.168.1.115:18000")  # kuaidi

save_dir = '/media/haoxin/A1/all_data_result/east_result'
# if os.path.exists(save_dir):
#     shutil.rmtree(save_dir)
# os.mkdir(save_dir)

save_success_dir = os.path.join(save_dir, 'success')
save_fail_dir = os.path.join(save_dir, 'fail')
# os.mkdir(save_success_dir)
# os.mkdir(save_fail_dir)


imgs_dir = '/media/haoxin/A1/all_data'
xmls_dir = '/media/haoxin/A1/all_data_result/all_xmls'
list_xmls_path = glob.glob(os.path.join(xmls_dir, '*.xml'))[15000:]
print('there are imgs num:',len(list_xmls_path))

correct_num = 0
error_num = 0
for odx, xml_path in enumerate(list_xmls_path):
    img_path = os.path.join(imgs_dir, os.path.basename(xml_path)[5:-4]+'.jpg')
    if not os.path.exists(img_path):
        print('error, not exit', img_path)
    image = cv2.imread(img_path)
    # 传输给服务器的数据
    data = {'fname': img_path, 'img_str': rpc_sevice._img_to_str_base64(image)}

    # test by EAST mode
    res_east_detect = c_det_east.detect(data, 0.8, False)  # Debug mode, test image is need labeled?
    bboxdict_east = res_east_detect['data']

    # draw east
    list_frame = list()
    for idx, inst in enumerate(bboxdict_east):
        x0, y0, x1, y1 = int(inst['x0']), int(inst['y0']), int(inst['x2']), int(inst['y2'])
        list_frame.append([x0, y0, x1, y1])

        #cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 1)

    jpg_name = os.path.basename(img_path)

    # get gt frame
    list_gt_frame = list()
    e = ET.parse(xml_path).getroot()
    for bbox in e.findall('object'):
        inst_bbox = bbox.find('bndbox')
        xmin = int(inst_bbox.find('xmin').text)
        ymin = int(inst_bbox.find('ymin').text)
        xmax = int(inst_bbox.find('xmax').text)
        ymax = int(inst_bbox.find('ymax').text)
        list_gt_frame.append([xmin,ymin,xmax, ymax])

    list_local_frame = list_frame
    list_match_local, list_poss_local, list_match_pair, list_poss_pair, flag = evaluation.is_qualified_two_img(image, list_local_frame, list_gt_frame)

    # draw in img
    for frame in list_local_frame:
        x0, y0, x1, y1 = frame[0], frame[1], frame[2], frame[3]
        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 2)
    for frame in list_gt_frame:
        x0, y0, x1, y1 = frame[0], frame[1], frame[2], frame[3]
        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 0), 1)


    if flag:
        correct_num += 1
        jpg_vis_path = os.path.join(save_success_dir, jpg_name)
        cv2.imwrite(jpg_vis_path, image)

        txt_path = os.path.join(save_success_dir, jpg_name[:-4]+'.txt')
        with open(txt_path, 'w') as f:
            for local_frame in list_local_frame:
                local_frame = [str(_) for _ in local_frame]
                txt_line = ','.join(local_frame)+'\n'
                f.write(txt_line)
    else:
        for _ in list_poss_pair:
            gt_frame = _['gt_frame']
            list_local_frame = _['local_frame']

            x0, y0, x1, y1 = gt_frame[0], gt_frame[1], gt_frame[2], gt_frame[3]
            cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 0), 2)
            for local_frame in list_local_frame:
                x0, y0, x1, y1 = local_frame[0], local_frame[1], local_frame[2], local_frame[3]
                cv2.rectangle(image, (x0, y0), (x1, y1), (0, 125, 125), 1)

        error_num += 1
        jpg_vis_path = os.path.join(save_fail_dir, jpg_name)
        cv2.imwrite(jpg_vis_path, image)

        txt_path = os.path.join(save_fail_dir, jpg_name[:-4] + '.txt')
        with open(txt_path, 'w') as f:
            for local_frame in list_local_frame:
                local_frame = [str(_) for _ in local_frame]
                txt_line = ','.join(local_frame)+'\n'
                f.write(txt_line)

    if odx%100 == 0:
        print('finish num:', odx)

print('all finish, correct rate:', correct_num*1./len(list_xmls_path))