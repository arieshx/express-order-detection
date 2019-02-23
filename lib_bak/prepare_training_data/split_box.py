import os
import numpy as np
import math
import shutil, glob
import xml.etree.ElementTree as ET
import copy

class Object:
    def __init__(self):
        self.name = None
        self.pose = "Unspecified"
        self.truncated = 0
        self.difficult = 0
        self.xmin = None
        self.ymin = None
        self.xmax = None
        self.ymax = None


def Gt_Object( parent , object ):

    name = ET.SubElement(parent, 'name')
    name.text = object.name
    pose = ET.SubElement(parent, 'pose')
    pose.text = object.pose
    truncated = ET.SubElement(parent, 'truncated')
    truncated.text = object.truncated
    difficult = ET.SubElement(parent, 'difficult')
    difficult.text = object.difficult
    bndbox = ET.SubElement(parent, 'bndbox')


    xmin = ET.SubElement(bndbox, 'xmin')
    xmin.text = object.xmin
    ymin = ET.SubElement(bndbox, 'ymin')
    ymin.text = object.ymin
    xmax = ET.SubElement(bndbox, 'xmax')
    xmax.text = object.xmax
    ymax = ET.SubElement(bndbox, 'ymax')
    ymax.text = object.ymax


def write_xml(e, list_fine_box, save_dir, xml_name):
    xml_path = os.path.join(save_dir, xml_name)

    annotation = copy.deepcopy(e)
    obgs = annotation.findall('object')
    for _ in obgs:
        annotation.remove(_)

    for idx, box in enumerate(list_fine_box):
        object = ET.SubElement(annotation, 'object')
        ot = Object()

        t_xmin = box[0]
        t_xmax = box[2]
        t_ymin = box[1]
        t_ymax = box[3]

        ot.xmin = str(t_xmin if t_xmin > 0 else 1)
        ot.xmax = str(t_xmax )
        ot.ymin = str(t_ymin if t_ymin > 0 else 1)
        ot.ymax = str(t_ymax )

        ot.truncated = "0"
        ot.difficult = "0"

        Gt_Object(object, ot)
    tree = ET.ElementTree(annotation)

    tree.write(xml_path, encoding="UTF-8")

#
this_dir = os.path.dirname(__file__)
# xml_dir = os.path.join(this_dir, 'Annotations_bak')
# save_dir = os.path.join(this_dir, 'Annotations_bak_split')
xml_dir = '/media/haoxin/A1/VOC200710w/Annotations10w_large'
save_dir = '/media/haoxin/A1/VOC200710w/Annotations10w_split'
if  os.path.exists(save_dir):
    shutil.rmtree(save_dir)
os.makedirs(save_dir)
list_xml_path = glob.glob(os.path.join(xml_dir, '*.xml'))


for odx, xml_path in enumerate(list_xml_path):
    xml_name = os.path.basename(xml_path)

    list_box = list()
    e = ET.parse(xml_path).getroot()

    for bbox in e.findall('object'):
        inst_bbox = bbox.find('bndbox')
        xmin = int(inst_bbox.find('xmin').text)
        ymin = int(inst_bbox.find('ymin').text)
        xmax = int(inst_bbox.find('xmax').text)
        ymax = int(inst_bbox.find('ymax').text)

        list_box.append([xmin, ymin, xmax, ymax])

    list_fine_box = list()
    for box in list_box:
        xmin,ymin,xmax,ymax = box[0], box[1], box[2], box[3]
        width = xmax - xmin
        height = ymax - ymin

        # reimplement
        step = 16.0
        x_left = []
        x_right = []
        x_left.append(xmin)
        x_left_start = int(math.ceil(xmin / 16.0) * 16.0)
        if x_left_start == xmin:
            x_left_start = xmin + 16
        for i in np.arange(x_left_start, xmax, 16):
            x_left.append(i)
        x_left = np.array(x_left)

        x_right.append(x_left_start - 1)
        for i in range(1, len(x_left) - 1):
            x_right.append(x_left[i] + 15)
        x_right.append(xmax)
        x_right = np.array(x_right)

        idx = np.where(x_left == x_right)
        x_left = np.delete(x_left, idx, axis=0)
        x_right = np.delete(x_right, idx, axis=0)

        for i in range(len(x_left)):
            list_fine_box.append([x_left[i], ymin, x_right[i], ymax])

    write_xml(e, list_fine_box, save_dir, xml_name)

    if odx % 1000 == 0:
        print('finishi num' + str(odx))
