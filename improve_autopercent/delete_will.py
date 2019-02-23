import glob
import shutil
some_161 = glob.glob('/media/haoxin/A1/all_data/*161.jpg')
for i in some_161[0:20]:
    shutil.copy(i, '/media/haoxin/A1/gx_data')
some_163 = glob.glob('/media/haoxin/A1/all_data/*163.jpg')
for i in some_163[0:20]:
    shutil.copy(i, '/media/haoxin/A1/gx_data')

some_164 = glob.glob('/media/haoxin/A1/all_data/*164.jpg')
for i in some_164[0:20]:
    shutil.copy(i, '/media/haoxin/A1/gx_data')