"""
GTA-IM Dataset
"""

import glob
import os
import pickle

import numba
import numpy as np


@numba.jit(nopython=True, nogil=True)
def rot_axis(angle, axis):
    cg = np.cos(angle)
    sg = np.sin(angle)
    if axis == 0:  # X
        v = [0, 4, 5, 7, 8]
    elif axis == 1:  # Y
        v = [4, 0, 6, 2, 8]
    else:  # Z
        v = [8, 0, 1, 3, 4]
    RX = np.zeros(9, dtype=numba.float64)
    RX[v[0]] = 1.0
    RX[v[1]] = cg
    RX[v[2]] = -sg
    RX[v[3]] = sg
    RX[v[4]] = cg
    return RX.reshape(3, 3)


@numba.jit(nopython=True, nogil=True)
def rotate(vector, angle, inverse=False):
    """
    Rotation of x, y, z axis
    Forward rotate order: Z, Y, X
    Inverse rotate order: X^T, Y^T,Z^T
    Input:
        vector: vector in 3D coordinates
        angle: rotation along X, Y, Z (raw data from GTA)
    Output:
        out: rotated vector
    """
    gamma, beta, alpha = angle[0], angle[1], angle[2]

    # Rotation matrices around the X (gamma), Y (beta), and Z (alpha) axis
    RX = rot_axis(gamma, 0)
    RY = rot_axis(beta, 1)
    RZ = rot_axis(alpha, 2)

    # Composed rotation matrix with (RX, RY, RZ)
    if inverse:
        return np.dot(np.dot(np.dot(RX.T, RY.T), RZ.T), vector)
    else:
        return np.dot(np.dot(np.dot(RZ, RY), RX), vector)


def angle2rot(rotation, inverse=False):
    return rotate(np.eye(3), rotation, inverse=inverse)


class Pose:
    def __init__(self, position, rotation):
        # relative position to the 1st frame: (X, Y, Z)
        # relative rotation to the previous frame: (r_x, r_y, r_z)
        self.position = position
        self.rotation = angle2rot(rotation)
        magic_rot = angle2rot(np.array([np.pi / 2, 0, 0]), inverse=True)
        self.rotation = self.rotation.dot(magic_rot)


def get_focal_length(cam_near_clip, cam_field_of_view):
    near_clip_height = (
        2 * cam_near_clip * np.tan(cam_field_of_view / 2.0 * (np.pi / 180.0))
    )

    # camera focal length
    return 1080.0 / near_clip_height * cam_near_clip


def get_cam_extr(cam_pos, cam_rot):
    cam_pos = np.array(cam_pos)
    cam_rot = np.array(cam_rot)

    pose = Pose(cam_pos, cam_rot / 180.0 * np.pi)
    cam_extr = np.eye(4)
    cam_extr[:3, :3] = pose.rotation
    cam_extr[:3, -1] = pose.position

    return cam_extr


if __name__ == '__main__':
    rec_inds = glob.glob('2020*')
    for data_path in rec_inds:
        if '.zip' in data_path:
            continue
        print(data_path)
        data_path += '/'
        info_path = data_path + 'realtimeinfo.gz'
        info = pickle.load(open(info_path, 'rb'))['frames']

        new_info = []
        joints_2d_seq = []
        joints_3d_cam_seq = []
        joints_3d_world_seq = []
        world2cam_trans = []
        intrinsics = []
        count = 0
        for i in range(len(info)):
            infot = info[i]
            # Change the image names
            prefix = data_path + str(infot['time'])
            if os.path.exists(prefix + '_final.jpg') and os.path.exists(
                prefix + '_depth.png'
            ):
                os.rename(
                    prefix + '_final.jpg',
                    data_path + '{:05d}'.format(count) + '.jpg',
                )
                os.rename(
                    prefix + '_depth.png',
                    data_path + '{:05d}'.format(count) + '.png',
                )
                os.rename(
                    prefix + '_id.png',
                    data_path + '{:05d}'.format(count) + '_id.png',
                )
                count = count + 1

                # 3d keypoints
                keypoint = [
                    infot['head'],
                    infot['neck'],
                    infot['right_clavicle'],
                    infot['right_shoulder'],
                    infot['right_elbow'],
                    infot['right_wrist'],
                    infot['left_clavicle'],
                    infot['left_shoulder'],
                    infot['left_elbow'],
                    infot['left_wrist'],
                    infot['spine0'],
                    infot['spine1'],
                    infot['spine2'],
                    infot['spine3'],
                    infot['spine4'],
                    infot['right_hip'],
                    infot['right_knee'],
                    infot['right_ankle'],
                    infot['left_hip'],
                    infot['left_knee'],
                    infot['left_ankle'],
                ]

                # camera parameters
                cam_near_clip = infot['cam_near_clip']
                cam_field_of_view = infot['cam_field_of_view']
                focal_length = get_focal_length(
                    cam_near_clip, cam_field_of_view
                )
                intrinsic = np.asarray(
                    [
                        [focal_length, 0, 960.0],
                        [0, focal_length, 540.0],
                        [0, 0, 1],
                    ]
                )
                cam_extr_ref = get_cam_extr(infot['cam_pos'], infot['cam_rot'])

                joints = np.asarray(keypoint)
                jn = joints.shape[0]

                joints_world = np.concatenate(
                    [joints, np.ones((jn, 1))], axis=-1
                )
                joints_cam = joints_world.dot(np.linalg.inv(cam_extr_ref.T))[
                    :, :3
                ]
                joints_2d = np.matmul(intrinsic, joints_cam.T)
                joints_2d = (
                    joints_2d[0] / joints_2d[2],
                    joints_2d[1] / joints_2d[2],
                )
                gta_pose_2d = np.asarray(joints_2d).T.reshape(jn, 2)
                joints_cam = joints_cam.reshape(jn, 3)

                joints_2d_seq.append(np.asarray(joints_2d).T)
                joints_3d_cam_seq.append(joints_cam)
                joints_3d_world_seq.append(joints)
                world2cam_trans.append(np.linalg.inv(cam_extr_ref.T))
                intrinsics.append(intrinsic)

                new_info.append(infot)

        np.savez(
            data_path + 'info_frames.npz',
            joints_2d=np.asarray(joints_2d_seq),
            joints_3d_cam=np.asarray(joints_3d_cam_seq),
            joints_3d_world=np.asarray(joints_3d_world_seq),
            world2cam_trans=np.asarray(world2cam_trans),
            intrinsics=np.asarray(intrinsics),
        )

        fn = open(data_path + 'info_frames.pickle', 'wb')
        pickle.dump(new_info, fn)
