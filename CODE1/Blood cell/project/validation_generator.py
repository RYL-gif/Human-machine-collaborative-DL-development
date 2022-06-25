import os
import cv2
import numpy as np
import random

root = '../training dataset'
savepath = '../val'
split = 0.2

files_dict = {}
for rt,dir,files in os.walk(root):
    if not files:
        continue
    cls = rt.split('/')[-1]
    files_dict[cls] = files

val_dict = {}
for cls in files_dict.keys():
    files = files_dict[cls]
    # random.shuffle(files)

    val_files = np.random.choice(files, int(len(files) * split), replace=False)
    val_dict[cls] = val_files

    for file in val_files:
        files_dict[cls].remove(file)

if not os.path.exists(savepath):
    os.mkdir(savepath)
for cls in val_dict:
    save_dir = os.path.join(savepath,cls)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for file in val_dict[cls]:
        image = cv2.imread(os.path.join(root,cls,file))
        cv2.imwrite(os.path.join(save_dir, file), image)


savepath = '../training dataset2'
if not os.path.exists(savepath):
    os.mkdir(savepath)
for cls in files_dict:
    save_dir = os.path.join(savepath,cls)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for file in files_dict[cls]:
        image = cv2.imread(os.path.join(root,cls,file))
        cv2.imwrite(os.path.join(save_dir, file), image)

