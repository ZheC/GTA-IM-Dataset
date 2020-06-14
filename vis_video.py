"""
GTA-IM Dataset

Copyright (c) 2020, Zhe Cao, Hang Gao, Karttikeya Mangalam, Qi-Zhi Cai, Minh Vo, Jitendra Malik.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY Hang Gao ''AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL Hang Gao BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation
are those of the authors and should not be interpreted as representing
official policies, either expressed or implied, of Hang Gao.
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
