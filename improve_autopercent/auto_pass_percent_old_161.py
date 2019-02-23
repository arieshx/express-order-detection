# -*- coding: utf-8
# ----------------------------------------
# Automatic Pass Percent test in Deppon
# ----------------------------------------
# 2018/08

import cv2, glob
import zerorpc
import os, sys
import numpy as np
import json
import time, shutil, base64
import requests
from termcolor import colored
from matplotlib import pyplot as plt

# def _str_to_img_base64(str_image, FLAG_color=False):
#     """ convert base64 string to image
#     """
#     image = np.array(Image.open(StringIO(base64.b64decode(str_image))))
#     if len(image.shape) == 3 and FLAG_color: return image
#     if len(image.shape) == 2 and FLAG_color: return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
#     if len(image.shape) == 3 and FLAG_color == False: return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     if len(image.shape) == 2 and FLAG_color == False: return image


def _img_to_str_base64(image):
    """ convert image to base64 string
    """
    img_encode = cv2.imencode('.jpg', image)[1]
    img_base64 = base64.b64encode(img_encode)
    return img_base64


class rectangle(object):
    def __init__(self,x0=None,y0=None,x1=None,y1 = None):
        self.x0 = int(x0)
        self.x1 = int(x1)
        self.y0 = int(y0)
        self.y1 = int(y1)
        self.width = abs(self.x0-self.x1)
        self.height = abs(self.y0-self.y1)
        self.area = self.width*self.height


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


c_det_east = zerorpc.Client()
c_det_east.connect("tcp://192.168.1.115:18000")  # kuaidi

c_det_frcnn = zerorpc.Client()
c_det_frcnn.connect("tcp://192.168.1.115:18765")  # kuaidi


def draw_detect_result(image_path):
    """
    将某图检测结构可视化
    :param image_path:
    :return:
    """

    # ----------------------------------------
    # 连接两个检测 RPC 服务
    # ----------------------------------------

    image = cv2.imread(image_path)

    # 传输给服务器的数据
    data = {'fname': image_path, 'img_str': _img_to_str_base64(image)}
    # test by EAST mode
    res_east_detect = c_det_east.detect(data, 0.8, False)  # Debug mode, test image is need labeled?
    bboxdict_east = res_east_detect['data']

    # draw east
    list_frame = list()
    for idx, inst in enumerate(bboxdict_east):
        x0, y0, x1, y1 = int(inst['x0']), int(inst['y0']), int(inst['x2']), int(inst['y2'])
        list_frame.append([x0, y0, x1, y1])

        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 0), 1)

    res_frcnn_detect = c_det_frcnn.detect(data)  #
    bboxdict_frcnn = res_frcnn_detect['data']['bbox_list']

    # draw ctpn
    list_frame = list()
    for idx, inst in enumerate(bboxdict_frcnn):
        rect = inst['bbox']
        x0, y0, x1, y1 = rect[0], rect[1], rect[2], rect[3]
        list_frame.append([x0, y0, x1, y1])

        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 2)

    # 保存结果到 ../data/improve_auto_detect_result
    if DEBUG:
        plt.imshow(image)
        plt.show()

    jpg_name = os.path.basename(image_path)
    save_dir = '../data/improve_auto_detect_result'
    if not os.path.exists(save_dir):
        print(os.path.abspath(save_dir))
        os.mkdir(save_dir)
    save_img_path = os.path.join('../data/improve_auto_detect_result', jpg_name)
    cv2.imwrite(save_img_path, image)


def find_error_detect(image_path):
    """
    给一张图片，判断是否检查正确，错误则保存下来。
    :param image_path:
    :return:
    """
    save_path = './ctpn_detect_error.txt'

    image = cv2.imread(image_path)

    # 传输给服务器的数据
    data = {'fname': image_path, 'img_str': _img_to_str_base64(image)}

    # test by EAST mode
    res_east_detect = c_det_east.detect(data, 0.8, False)  # Debug mode, test image is need labeled?
    bboxdict_east = res_east_detect['data']
    boxes_east = list()
    for idx, inst in enumerate(bboxdict_east):
        x0, y0, x1, y1 = int(inst['x0']), int(inst['y0']), int(inst['x2']), int(inst['y2'])
        boxes_east.append([x0, y0, x1, y1])
    # test by CTPN mode
    res_frcnn_detect = c_det_frcnn.detect(data)  #
    bboxdict_frcnn = res_frcnn_detect['data']['bbox_list']
    boxes_ctpn = [_['bbox'] for _ in bboxdict_frcnn]  # (n[4])

    list_rec_ctpn = [rectangle(_[0], _[1], _[2], _[3]) for _ in boxes_ctpn]
    list_rec_east = [rectangle(_[0], _[1], _[2], _[3]) for _ in boxes_east]

    list_rec_onlyeast = list()
    for east_idx, one_east_rec in enumerate(list_rec_east):
        flag = 0
        for ctpn_idx, one_ctpn_rec in enumerate(list_rec_ctpn):
            overlap_area, iou_width = iou_area(one_east_rec, one_ctpn_rec)
            if overlap_area*1./one_east_rec.area > 0.8 or overlap_area*1./one_ctpn_rec.area > 0.8:  # 重叠面积到达两者其中一个的0.8以上，认为这两个框重合
                flag = 1
                break
            else:
                continue
        # 找晚所有的ctpn，都没很重合的，则这个east框可能是比ctpn多的
        if not flag:
            list_rec_onlyeast.append(one_east_rec)

    detect_right = 1
    # 得到了east比ctpn多的框，判断这个框是否处于最上面被ctpn滤掉的框，如果不是，说明ctpn检测错误
    for east_rec in list_rec_onlyeast:
        if east_rec.y0 <= 2 or east_rec.y0 > image.shape[0]-60:
            continue
        else:
            detect_right = 0
            break
    if not detect_right:  # 如果检查错误，则保存图片路径，返回错误
        with open(save_path, 'a') as f:
            f.write(image_path)
    return detect_right


def draw_svm_result(image_path, detect_svm):
    image = cv2.imread(image_path)
    for _ in detect_svm:
        predict_type = _['predict_type']
        box = _['area']
        x0, y0, x1, y1 = box[0], box[1], box[2], box[3]
        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 2)
        txt = str(predict_type)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, txt, (x0, y0), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
    if DEBUG:
        plt.imshow(image)
        plt.show()


def draw_gbc_result(image_path, detect_gbc):
    image = cv2.imread(image_path)
    for one_key in detect_gbc.keys():
        one_value = detect_gbc[one_key]
        for two_key in one_value.keys():
            two_value = one_value[two_key]
            text_inst = two_value['text_inst']
            box = text_inst['detect']['bbox']
            svm_type = text_inst['predict_type']
            gbc_type = two_value['cate']
            txt_line = str(svm_type) + '/' + str(gbc_type)
            x0, y0, x1, y1 = box[0], box[1], box[2], box[3]
            cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, txt_line, (x0, y0), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
    if DEBUG:
        plt.imshow(image)
        plt.show()


def rule_about_phone():
    import Levenshtein
    deppon_phone = deppon_res['receiveCustomerPhone']
    detect_phone = detect_res['receiveCustomerPhone']['text']
    is_same = 0
    diff = Levenshtein.distance(deppon_phone, detect_phone)
    if diff < 3 and diff !=0:
        # ocr error， 存起来，继续训练
        pass
    elif diff>=3:
        phone_list = detect_res['receiveCustomerPhone']['text_list']
        if len(phone_list) == 1:  # 不是拼起来的电话
            # ocr error, 存下来，
            pass  # 1.两个电话与一个电话做匹配，2.是一个电话与一个电话，ocr出错严重。
        else:  # s是拼起来的电话
            best_phone = min(phone_list, key = lambda x:Levenshtein.distance(deppon_phone, x['text']))
            for one_phone_info in phone_list:
                if one_phone_info is best_phone:
                    diff = Levenshtein.distance(deppon_phone, best_phone)
                    if diff:  # 有错误。ocr出错
                        pass  # 记下来这一条数据，切出来这个图片，做为ocr出错训练数据.
                    else:  # 起码第一条认对了，不是ocr出错。
                        pass  # 判断为通过
                else:
                    y_diff = abs(best_phone['text_area'][1] - one_phone_info['text_area'][1])
                    if y_diff > 2*(best_phone['text_area'][3]-best_phone['text_area'][1]):
                        pass # cls error
                        if one_phone_info['text']:  # 如果拼接起来的内容属于其他字段，则可以判断为是那一类字段错误分类为了电话字段。，内部再详细判断是哪一个分类器出错了。
                            pass  # 内部判断是哪一个分类器出错。是svm还是xgb分类树。

                    else:  # 位置相对真电话来说是靠近的，那么有可能真的是电话，也不能说是分类错误，也不能确认ocr是否出错，所以可以记到空闲字段，等待其他字段做匹配。
                        pass  # 可能是一个被抹掉的电话，也可能是另一个电话。


DEBUG = True

NUM_total = 0
NUM_pass = 0

URL_request = 'http://manhattanic.hexin.im/image/getList?_limit=50000&_page=11'
data = json.loads(requests.get(URL_request).text)['data']

detect_error_num = 0
cate_right_num = {
        'deliveryCustomerName': 0,
        'deliveryCustomerPhone': 0,
        'notificationMode': 0,
    }
#  针对161统计各字段的通过率
list_notificationMode_error = list()
for idx, inst in enumerate(data[0: 200]):
    image_path = inst['url']
    uid = inst['uid']
    if '_161.jpg' not in image_path: continue

    # check image type, 161, 162, 163, 164
    # detect_res = inst['detectResult']
    kuaidi_rpc = zerorpc.Client()
    kuaidi_rpc.connect('tcp://192.168.1.57:18888')
    splited_url = image_path.split('/')
    data = {'fname': splited_url[-1], 'img_str': 'http://192.168.1.115/{}/{}'.format(splited_url[-2], splited_url[-1])}
    areaCode = image_path[-7:-4]
    # detect = kuaidi_rpc.extract_predict(data, True, True, False, areaCode, False)
    # detect_res = json.loads(detect['data'])
    # detect_svm = json.loads(detect['detect'])
    # detect_gbc = json.loads(detect['detect_cate'])
    detect_res = inst['detectResult']
    deppon_res = inst['depponOcrResult']
    if detect_res is None: continue
    if deppon_res['isReturn'] == 'Y':
        continue

    print('*' * 50)
    print(image_path, '*' * 10, str(idx))
    if image_path != '/data/kuaidi01/kuaidi_online/deppon201807020456/30_5757778208_20180625230319_20180626003322_161.jpg':
        continue
    NUM_total += 1

    # print vis two result
    print('-' * 30)
    for name in sorted(deppon_res.keys()):
        if name in ['isReturn', 'waybillNo', 'returnReason']: continue
        print name, deppon_res[name]

    print('-' * 30)
    for name in sorted(detect_res.keys()):
        if type(detect_res[name]) == dict:
            print name, detect_res[name]['text']
        else:
            print name, detect_res[name]
    print('-' * 30)

    # draw svm result
    # draw_svm_result(image_path, detect_svm)
    # draw_gbc_result(image_path, detect_gbc)

    # 处理脏数据
    for name in deppon_res.keys():
        if name in ['returnReason', 'waybillNo', 'isReturn']: continue
        if deppon_res[name] in ['*', 'NULL']:
            deppon_res[name] = ''
        if type(deppon_res[name]) == int:
            deppon_res[name] = str(deppon_res[name])

    for name in detect_res.keys():
        if detect_res[name] != -1 and detect_res[name]['text'] in ['*']:
            detect_res[name]['text'] = ''

    # check 字段是否正确
    diff_key = list()
    for name in deppon_res.keys():
        # 164 badcase
        if name in ['collectionAccount', 'packageFeeCanvas', 'refundType', 'codAmount', 'accountName']: continue
        if name in ['returnReason', 'waybillNo']: continue
        if name == 'isReturn': continue

        # 检查签收短信通知
        if name == 'notificationMode':
            if (detect_res[name] == -1 and deppon_res[name]=='0') or (detect_res[name] != -1 and deppon_res[name]=='1'):
                cate_right_num[name] += 1
                continue
            else:
                FLAG_pass = 0
                diff_key.append(name)
                list_notificationMode_error.append(image_path)
            continue
        # check if two result is eqaul
        if detect_res[name] == -1:  # 该字段检测没有结果
            if deppon_res[name] == '':
                cate_right_num[name] +=1
                continue
            else:
                FLAG_pass = 0
                diff_key.append(name)
        else:
            assert type(detect_res[name]) == dict, 'error'+ str(detect_res[name])
            if detect_res[name]['text'] != deppon_res[name]:
                FLAG_pass = 0
                diff_key.append(name)
                # print('%s\t%s\t%s' % (name, detect_res[name]['text'], deppon_res[name]))
            else:
                cate_right_num[name] += 1
                continue

    if len(diff_key) == 0:
        FLAG_pass = 1
    else:
        FLAG_pass = 0
    if FLAG_pass == 1: NUM_pass += 1
    if not FLAG_pass:  # 如果没有通过，检查是什么地方的问题
        # 对错误样本判断检测是否出错了
        draw_detect_result(image_path)
        # 找到检测出错的样本，记录下来.
        detect_right = find_error_detect(image_path)
        if not detect_right:
            detect_error_num += 1
    # 打印出不通过条目
    for name in diff_key:
        if detect_res[name] == -1:
            print('%s\t%s\t%s' % (name, detect_res[name], deppon_res[name]))
        else:
            print('%s\t%s\t%s' % (name, detect_res[name]['text'], deppon_res[name]))
# stat final outputs
print('*' * 50)
print(NUM_pass, NUM_total, 1.0 * NUM_pass / NUM_total)
cate_result = [(_, cate_right_num[_], 1.0*cate_right_num[_]/NUM_total) for _ in cate_right_num]
for _ in cate_result:
    print(_)
print('list_notificationMode_error:', list_notificationMode_error)

# 在新模型上验证的图
# /data/kuaidi01/kuaidi_online/deppon201807020456/50_5634028722_20180625230149_20180626100000_163.jpg