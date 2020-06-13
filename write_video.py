import cv2
import numpy as np
import glob
import shutil
import os.path as osp
import argparse
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("-pa", "--path", default="2020-06-10-21-47-45")
    parser.add_argument("-s", "--scale", default=4, type=int, help="down scale")
    parser.add_argument("-fr", "--frame_rate", default=5, type=int, help="frame_rate")
    args = parser.parse_args()

    rec_inds = glob.glob(args.path + '/')
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
            size = (width // args.scale, height // args.scale)
            img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
            img_array.append(img)

        ## frame rate 10    
        out = cv2.VideoWriter(data_path+'/project.mp4', cv2.VideoWriter_fourcc(*'mp4v'), args.frame_rate, size)
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()