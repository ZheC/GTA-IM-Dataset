import cv2
import numpy as np
import glob
import shutil
import os.path as osp

from tqdm import tqdm


#rec_inds = glob.glob('D:\\datasets\\gtav-30\\2020*')  # mind this.
rec_inds = glob.glob('2020-06-09-16-09-56*')
for data_path in tqdm(rec_inds, desc='video', leave=True):
    ims = sorted(glob.glob(data_path + '/*.jpg'))
    if len(ims) == 0:
        shutil.rmtree(data_path)
        continue
    if osp.exists(osp.join(data_path, 'project.mp4')):
        continue

    img_array = []
    for filename in tqdm(ims, leave=True, desc='frame', position=1):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width // 4, height // 4)
        img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
        img_array.append(img)

    out = cv2.VideoWriter(data_path+'/project.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()