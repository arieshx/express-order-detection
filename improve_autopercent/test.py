#! /usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os, glob, shutil
list_img_path = glob.glob('../data/daily_error/*.jpg')
for idx, img_path in enumerate(list_img_path):
    new_name = str(idx)+'_'+os.path.basename(img_path)[-7:]
    new_path = os.path.join(os.path.dirname(img_path), new_name)
    shutil.copy(img_path, new_path)