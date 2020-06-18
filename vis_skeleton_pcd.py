"""
GTA-IM Dataset
"""

import argparse
import os
import pickle
import sys

import cv2
import numpy as np
import open3d as o3d
from open3d import (LineSet, PinholeCameraIntrinsic, Vector2iVector,
                    Vector3dVector, draw_geometries)

from gta_utils import LIMBS, read_depthmap

sys.path.append('./')


def create_skeleton_viz_data(nskeletons, njoints):
    lines = []
    colors = []
    for i in range(nskeletons):
        cur_lines = np.asarray(LIMBS)
        cur_lines += i * njoints
        lines.append(cur_lines)

        single_color = np.zeros([njoints, 3])
        single_color[:] = [0.0, float(i) / nskeletons, 1.0]
        colors.append(single_color[1:])

    lines = np.concatenate(lines, axis=0)
    colors = np.asarray(colors).reshape(-1, 3)
    return lines, colors


def vis_skeleton_pcd(rec_idx, f_id, fusion_window=20):
    info = pickle.load(open(rec_idx + '/info_frames.pickle', 'rb'))
    info_npz = np.load(rec_idx + '/info_frames.npz')

    pcd = o3d.geometry.PointCloud()
    global_pcd = o3d.geometry.PointCloud()
    # use nearby RGBD frames to create the environment point cloud
    for i in range(f_id - fusion_window // 2, f_id + fusion_window // 2, 10):
        fname = rec_idx + '/' + '{:05d}'.format(i) + '.png'
        if os.path.exists(fname):
            infot = info[i]
            cam_near_clip = infot['cam_near_clip']
            if 'cam_far_clip' in infot.keys():
                cam_far_clip = infot['cam_far_clip']
            else:
                cam_far_clip = 800. 
            depth = read_depthmap(fname, cam_near_clip, cam_far_clip)
            # delete points that are more than 20 meters away
            depth[depth > 20.0] = 0

            # obtain the human mask
            p = info_npz['joints_2d'][i, 0]
            fname = rec_idx + '/' + '{:05d}'.format(i) + '_id.png'
            id_map = cv2.imread(fname, cv2.IMREAD_ANYDEPTH)
            human_id = id_map[
                np.clip(int(p[1]), 0, 1079), np.clip(int(p[0]), 0, 1919)
            ]

            mask = id_map == human_id
            kernel = np.ones((3, 3), np.uint8)
            mask_dilation = cv2.dilate(
                mask.astype(np.uint8), kernel, iterations=1
            )
            depth = depth * (1 - mask_dilation[..., None])
            depth = o3d.geometry.Image(depth.astype(np.float32))
            # cv2.imshow('tt', mask.astype(np.uint8)*255)
            # cv2.waitKey(0)

            fname = rec_idx + '/' + '{:05d}'.format(i) + '.jpg'
            color_raw = o3d.io.read_image(fname)

            focal_length = info_npz['intrinsics'][f_id, 0, 0]
            rgbd_image = o3d.geometry.create_rgbd_image_from_color_and_depth(
                color_raw,
                depth,
                depth_scale=1.0,
                depth_trunc=15.0,
                convert_rgb_to_intensity=False,
            )
            pcd = o3d.geometry.create_point_cloud_from_rgbd_image(
                rgbd_image,
                o3d.camera.PinholeCameraIntrinsic(
                    PinholeCameraIntrinsic(
                        1920, 1080, focal_length, focal_length, 960.0, 540.0
                    )
                ),
            )
            depth_pts = np.asarray(pcd.points)

            depth_pts_aug = np.hstack(
                [depth_pts, np.ones([depth_pts.shape[0], 1])]
            )
            cam_extr_ref = np.linalg.inv(info_npz['world2cam_trans'][i])
            depth_pts = depth_pts_aug.dot(cam_extr_ref)[:, :3]
            pcd.points = Vector3dVector(depth_pts)

            global_pcd.points.extend(pcd.points)
            global_pcd.colors.extend(pcd.colors)

    # read gt pose in world coordinate, visualize nearby frame as well
    joints = info_npz['joints_3d_world'][(f_id - 30) : (f_id + 30) : 10]
    tl, jn, _ = joints.shape
    joints = joints.reshape(-1, 3)

    # create skeletons in open3d
    nskeletons = tl
    lines, colors = create_skeleton_viz_data(nskeletons, jn)
    line_set = LineSet()
    line_set.points = Vector3dVector(joints)
    line_set.lines = Vector2iVector(lines)
    line_set.colors = Vector3dVector(colors)

    vis_list = [global_pcd, line_set]
    for j in range(joints.shape[0]):
        # spine joints
        if j % jn == 11 or j % jn == 12 or j % jn == 13:
            continue
        transformation = np.identity(4)
        transformation[:3, 3] = joints[j]
        # head joint
        if j % jn == 0:
            r = 0.07
        else:
            r = 0.03

        sphere = o3d.geometry.create_mesh_sphere(radius=r)
        sphere.paint_uniform_color([0.0, float(j // jn) / nskeletons, 1.0])
        vis_list.append(sphere.transform(transformation))

    draw_geometries(vis_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-pa', '--path', default='2020-06-10-21-47-45')
    parser.add_argument(
        '-f', '--frame', default=180, type=int, help='frame to visualize'
    )
    parser.add_argument(
        '-fw',
        '--fusion-window',
        default=20,
        type=int,
        help='timesteps of RGB frames for fusing',
    )
    args = parser.parse_args()

    vis_skeleton_pcd(args.path + '/', args.frame, args.fusion_window)
