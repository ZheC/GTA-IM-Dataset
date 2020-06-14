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
import os
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np

from gta_utils import LIMBS, read_depthmap


def single_vis(args):
    joints_2d = np.load(args.path + '/info_frames.npz')['joints_2d']
    info = pickle.load(open(args.path + '/info_frames.pickle', 'rb'))
    if not os.path.exists(args.outpath):
        os.mkdir(args.outpath)
    for idx in range(30, len(info)):
        if os.path.exists(
            os.path.join(args.path, '{:05d}'.format(idx) + '.jpg')
        ):
            keypoint = joints_2d[idx]

            # root
            root_pos = (int(keypoint[14, 0]), int(keypoint[14, 1]))
            # color image
            frame = cv2.imread(
                os.path.join(args.path, '{:05d}'.format(idx) + '.jpg')
            )
            frame = cv2.circle(frame, tuple(root_pos), 10, (0, 0, 255), 20)
            # depth map
            infot = info[idx]
            cam_near_clip = infot['cam_near_clip']
            cam_far_clip = infot['cam_far_clip']
            fname = os.path.join(args.path, '{:05d}'.format(idx) + '.png')
            depthmap = read_depthmap(fname, cam_near_clip, cam_far_clip)

            # plot joints
            for i0, i1 in LIMBS:
                p1 = (int(keypoint[i0, 0]), int(keypoint[i0, 1]))
                p2 = (int(keypoint[i1, 0]), int(keypoint[i1, 1]))
                frame = cv2.line(frame, tuple(p1), tuple(p2), (0, 255, 0), 20)

            plt.figure(figsize=(32, 9))
            plt.subplot(121)
            plt.imshow(frame[:, :, ::-1])
            plt.axis('off')
            plt.subplot(122)
            # visualize the disparity
            plt.imshow(100.0 / depthmap[:, :, 0], cmap='plasma')
            plt.axis('off')
            plt.savefig(
                os.path.join(args.outpath, str(idx) + '_vis.jpg'),
                bbox_inches='tight',
            )
            plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-pa', '--path', default='2020-06-10-09-27-04/')
    args = parser.parse_args()
    args.outpath = args.path + '/vis/'
    single_vis(args)
