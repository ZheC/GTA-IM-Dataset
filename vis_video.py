"""
GTA-IM Dataset
"""

import argparse
import glob
import os
import os.path as osp
import sys

import cv2
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-pa', '--path', default='2020-06-10-21-47-45')
    parser.add_argument('-s', '--scale', default=4, type=int, help='down scale')
    parser.add_argument(
        '-fr', '--frame_rate', default=5, type=int, help='frame_rate'
    )
    args = parser.parse_args()
    args.outpath = args.path + '/vis/'
    if not osp.exists(args.outpath):
        os.mkdir(args.outpath)

    ims = sorted(glob.glob(args.path + '/*.jpg'))
    if osp.exists(osp.join(args.outpath, 'video.mp4')):
        sys.exit()

    img_array = []
    for filename in tqdm(ims, desc='frame'):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width // args.scale, height // args.scale)
        img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
        img_array.append(img)

    out = cv2.VideoWriter(
        osp.join(args.outpath, 'video.mp4'),
        cv2.VideoWriter_fourcc(*'mp4v'),
        args.frame_rate,
        size,
    )
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
