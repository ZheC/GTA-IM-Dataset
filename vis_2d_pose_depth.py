"""
GTA-IM Dataset
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
            if 'cam_far_clip' in infot.keys():
                cam_far_clip = infot['cam_far_clip']
            else:
                cam_far_clip = 800.    
            fname = os.path.join(args.path, '{:05d}'.format(idx) + '.png')
            depthmap = read_depthmap(fname, cam_near_clip, cam_far_clip)

            # plot joints
            for i0, i1 in LIMBS:
                p1 = (int(keypoint[i0, 0]), int(keypoint[i0, 1]))
                p2 = (int(keypoint[i1, 0]), int(keypoint[i1, 1]))
                frame = cv2.line(frame, tuple(p1), tuple(p2), (0, 255, 0), 20)


            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 9), sharey=True)
            ax1.imshow(frame[:, :, ::-1])
            ax1.axis('off')
            # visaulize the disparity
            ax2.imshow(100.0 / depthmap[:, :, 0], cmap='plasma')
            ax2.axis('off')
            # tight figure
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
            plt.margins(0,0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.savefig(
                os.path.join(args.outpath, str(idx) + '_vis.jpg'), 
                bbox_inches='tight',
                pad_inches=0)
            plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-pa', '--path', default='2020-06-10-09-27-04/')
    args = parser.parse_args()
    args.outpath = args.path + '/vis/'
    single_vis(args)
