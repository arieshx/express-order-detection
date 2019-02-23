# -*- coding:utf-8 -*-
import glob, os
import matplotlib.pyplot as plt
from tqdm import tqdm
"""
此文件的作用就是可视化出框的高度分布情况，来选择合适的anchor
"""


def statictic():
    """
    统计出所有的框的height并且保存下来
    :return:
    """
    # txt_dir = '/media/haoxin/A1/data/AdvancedEAST/txt_550'
    txt_dir = '/data/kuaidi01/dataset_detect/AdvancedEast_data/txt_all'
    list_txt_path = glob.glob(os.path.join(txt_dir, '*.txt'))
    all_lines = list()
    for txt_path in tqdm(list_txt_path):
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        all_lines.extend(lines)

    list_height = list()
    for line in tqdm(all_lines):
        ll = line.strip().split(',')
        height = int(float(ll[5]) - float(ll[1]))
        list_height.append(height)

    # save_height_txt_path = '/media/haoxin/A1/data/AdvancedEAST/height_distribute.txt'
    save_height_txt_path = '/data/kuaidi01/dataset_detect/AdvancedEast_data/height_distribute.txt'
    with open(save_height_txt_path, 'w') as f:
        str_height = [str(_) for _ in list_height]
        line = ','.join(str_height)
        f.write(line)

def vis_height_distribute():
    """
    将txt中的高度读出来并且可视化出来密度
    :return:
    """
    txt_path = '/media/haoxin/A1/data/AdvancedEAST/height_distribute2.txt'
    with open(txt_path, 'r') as f:
        a = f.readline()
        list_height_str = a.split(',')
        list_height = [int(_) for _ in list_height_str]

    plt.hist(list_height, bins=50, color='steelblue', normed=True)
    plt.show()

def search_big_frame():
    """
    可视化出的密度函数有很高的框，所以查看一下很高的框以及图片的高度,应该是164中有很高的框，那么，还要改变anchor吗。
    :return:
    """
    pass

vis_height_distribute()