"""
GTA-IM Dataset
"""

import cv2
import numpy as np

LIMBS = [
    (0, 1),  # head_center -> neck
    (1, 2),  # neck -> right_clavicle
    (2, 3),  # right_clavicle -> right_shoulder
    (3, 4),  # right_shoulder -> right_elbow
    (4, 5),  # right_elbow -> right_wrist
    (1, 6),  # neck -> left_clavicle
    (6, 7),  # left_clavicle -> left_shoulder
    (7, 8),  # left_shoulder -> left_elbow
    (8, 9),  # left_elbow -> left_wrist
    (1, 10),  # neck -> spine0
    (10, 11),  # spine0 -> spine1
    (11, 12),  # spine1 -> spine2
    (12, 13),  # spine2 -> spine3
    (13, 14),  # spine3 -> spine4
    (14, 15),  # spine4 -> right_hip
    (15, 16),  # right_hip -> right_knee
    (16, 17),  # right_knee -> right_ankle
    (14, 18),  # spine4 -> left_hip
    (18, 19),  # left_hip -> left_knee
    (19, 20),  # left_knee -> left_ankle
]


####################
# camera utils.
def get_focal_length(cam_near_clip, cam_field_of_view):
    near_clip_height = (
        2 * cam_near_clip * np.tan(cam_field_of_view / 2.0 * (np.pi / 180.0))
    )

    # camera focal length
    return 1080.0 / near_clip_height * cam_near_clip


def get_2d_from_3d(
    vertex,
    cam_coords,
    cam_rotation,
    cam_near_clip,
    cam_field_of_view,
    WIDTH=1920,
    HEIGHT=1080,
):
    WORLD_NORTH = np.array([0.0, 1.0, 0.0], 'double')
    WORLD_UP = np.array([0.0, 0.0, 1.0], 'double')
    WORLD_EAST = np.array([1.0, 0.0, 0.0], 'double')
    theta = (np.pi / 180.0) * cam_rotation
    cam_dir = rotate(WORLD_NORTH, theta)
    clip_plane_center = cam_coords + cam_near_clip * cam_dir
    camera_center = -cam_near_clip * cam_dir
    near_clip_height = (
        2 * cam_near_clip * np.tan(cam_field_of_view / 2.0 * (np.pi / 180.0))
    )
    near_clip_width = near_clip_height * WIDTH / HEIGHT

    cam_up = rotate(WORLD_UP, theta)
    cam_east = rotate(WORLD_EAST, theta)
    near_clip_to_target = vertex - clip_plane_center

    camera_to_target = near_clip_to_target - camera_center

    camera_to_target_unit_vector = camera_to_target * (
        1.0 / np.linalg.norm(camera_to_target)
    )

    view_plane_dist = cam_near_clip / cam_dir.dot(camera_to_target_unit_vector)

    new_origin = (
        clip_plane_center
        + (near_clip_height / 2.0) * cam_up
        - (near_clip_width / 2.0) * cam_east
    )

    view_plane_point = (
        view_plane_dist * camera_to_target_unit_vector
    ) + camera_center
    view_plane_point = (view_plane_point + clip_plane_center) - new_origin
    viewPlaneX = view_plane_point.dot(cam_east)
    viewPlaneZ = view_plane_point.dot(cam_up)
    screenX = viewPlaneX / near_clip_width
    screenY = -viewPlaneZ / near_clip_height

    # screenX and screenY between (0, 1)
    ret = np.array([screenX, screenY], 'double')
    return ret


def screen_x_to_view_plane(x, cam_near_clip, cam_field_of_view):
    # x in (0, 1)
    near_clip_height = (
        2 * cam_near_clip * np.tan(cam_field_of_view / 2.0 * (np.pi / 180.0))
    )
    near_clip_width = near_clip_height * 1920.0 / 1080.0

    viewPlaneX = x * near_clip_width

    return viewPlaneX


def generate_id_map(map_path):
    id_map = cv2.imread(map_path, -1)
    h, w, _ = id_map.shape
    id_map = np.concatenate(
        (id_map, np.zeros((h, w, 1), dtype=np.uint8)), axis=2
    )
    id_map.dtype = np.uint32
    return id_map


def get_depth(
    vertex, cam_coords, cam_rotation, cam_near_clip, cam_field_of_view
):
    WORLD_NORTH = np.array([0.0, 1.0, 0.0], 'double')
    theta = (np.pi / 180.0) * cam_rotation
    cam_dir = rotate(WORLD_NORTH, theta)
    clip_plane_center = cam_coords + cam_near_clip * cam_dir
    camera_center = -cam_near_clip * cam_dir

    near_clip_to_target = vertex - clip_plane_center

    camera_to_target = near_clip_to_target - camera_center
    camera_to_target_unit_vector = camera_to_target * (
        1.0 / np.linalg.norm(camera_to_target)
    )

    depth = np.linalg.norm(camera_to_target) * cam_dir.dot(
        camera_to_target_unit_vector
    )
    depth = depth - cam_near_clip

    return depth


def get_kitti_format_camera_coords(
    vertex, cam_coords, cam_rotation, cam_near_clip
):
    cam_dir, cam_up, cam_east = get_cam_dir_vecs(cam_rotation)

    clip_plane_center = cam_coords + cam_near_clip * cam_dir

    camera_center = -cam_near_clip * cam_dir

    near_clip_to_target = vertex - clip_plane_center

    camera_to_target = near_clip_to_target - camera_center
    camera_to_target_unit_vector = camera_to_target * (
        1.0 / np.linalg.norm(camera_to_target)
    )

    z = np.linalg.norm(camera_to_target) * cam_dir.dot(
        camera_to_target_unit_vector
    )
    y = -np.linalg.norm(camera_to_target) * cam_up.dot(
        camera_to_target_unit_vector
    )
    x = np.linalg.norm(camera_to_target) * cam_east.dot(
        camera_to_target_unit_vector
    )

    return np.array([x, y, z])


def get_cam_dir_vecs(cam_rotation):
    WORLD_NORTH = np.array([0.0, 1.0, 0.0], 'double')
    WORLD_UP = np.array([0.0, 0.0, 1.0], 'double')
    WORLD_EAST = np.array([1.0, 0.0, 0.0], 'double')
    theta = (np.pi / 180.0) * cam_rotation
    cam_dir = rotate(WORLD_NORTH, theta)
    cam_up = rotate(WORLD_UP, theta)
    cam_east = rotate(WORLD_EAST, theta)

    return cam_dir, cam_up, cam_east


def is_before_clip_plane(
    vertex,
    cam_coords,
    cam_rotation,
    cam_near_clip,
    cam_field_of_view,
    WIDTH=1920,
    HEIGHT=2080,
):
    WORLD_NORTH = np.array([0.0, 1.0, 0.0], 'double')
    theta = (np.pi / 180.0) * cam_rotation
    cam_dir = rotate(WORLD_NORTH, theta)
    clip_plane_center = cam_coords + cam_near_clip * cam_dir
    camera_center = -cam_near_clip * cam_dir

    near_clip_to_target = vertex - clip_plane_center

    camera_to_target = near_clip_to_target - camera_center

    camera_to_target_unit_vector = camera_to_target * (
        1.0 / np.linalg.norm(camera_to_target)
    )

    if cam_dir.dot(camera_to_target_unit_vector) > 0:
        return True
    else:
        return False


def get_clip_center_and_dir(cam_coords, cam_rotation, cam_near_clip):
    WORLD_NORTH = np.array([0.0, 1.0, 0.0], 'double')
    theta = (np.pi / 180.0) * cam_rotation
    cam_dir = rotate(WORLD_NORTH, theta)
    clip_plane_center = cam_coords + cam_near_clip * cam_dir
    return clip_plane_center, cam_dir


def rotate(a, t):
    d = np.zeros(3, 'double')
    d[0] = np.cos(t[2]) * (
        np.cos(t[1]) * a[0]
        + np.sin(t[1]) * (np.sin(t[0]) * a[1] + np.cos(t[0]) * a[2])
    ) - (np.sin(t[2]) * (np.cos(t[0]) * a[1] - np.sin(t[0]) * a[2]))
    d[1] = np.sin(t[2]) * (
        np.cos(t[1]) * a[0]
        + np.sin(t[1]) * (np.sin(t[0]) * a[1] + np.cos(t[0]) * a[2])
    ) + (np.cos(t[2]) * (np.cos(t[0]) * a[1] - np.sin(t[0]) * a[2]))
    d[2] = -np.sin(t[1]) * a[0] + np.cos(t[1]) * (
        np.sin(t[0]) * a[1] + np.cos(t[0]) * a[2]
    )
    return d


def get_intersect_point(center_pt, cam_dir, vertex1, vertex2):
    c1 = center_pt[0]
    c2 = center_pt[1]
    c3 = center_pt[2]
    a1 = cam_dir[0]
    a2 = cam_dir[1]
    a3 = cam_dir[2]
    x1 = vertex1[0]
    y1 = vertex1[1]
    z1 = vertex1[2]
    x2 = vertex2[0]
    y2 = vertex2[1]
    z2 = vertex2[2]

    k_up = a1 * (x1 - c1) + a2 * (y1 - c2) + a3 * (z1 - c3)
    k_down = a1 * (x1 - x2) + a2 * (y1 - y2) + a3 * (z1 - z2)
    k = k_up / k_down
    inter_point = (1 - k) * vertex1 + k * vertex2

    return inter_point


####################
# dataset utils.
def is_inside(x, y):
    return x >= 0 and x <= 1 and y >= 0 and y <= 1


def get_cut_edge(x1, y1, x2, y2):
    # (x1, y1) inside while (x2, y2) outside
    dx = x2 - x1
    dy = y2 - y1
    ratio_pool = []
    if x2 < 0:
        ratio = (x1 - 0) / (x1 - x2)
        ratio_pool.append(ratio)
    if x2 > 1:
        ratio = (1 - x1) / (x2 - x1)
        ratio_pool.append(ratio)
    if y2 < 0:
        ratio = (y1 - 0) / (y1 - y2)
        ratio_pool.append(ratio)
    if y2 > 1:
        ratio = (1 - y1) / (y2 - y1)
        ratio_pool.append(ratio)
    actual_ratio = min(ratio_pool)
    return x1 + actual_ratio * dx, y1 + actual_ratio * dy


def get_min_max_x_y_from_line(x1, y1, x2, y2):
    if is_inside(x1, y1) and is_inside(x2, y2):
        return min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)
    if (not is_inside(x1, y1)) and (not is_inside(x2, y2)):
        return None, None, None, None
    if is_inside(x1, y1) and not is_inside(x2, y2):
        x2, y2 = get_cut_edge(x1, y1, x2, y2)
        return min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)
    if is_inside(x2, y2) and not is_inside(x1, y1):
        x1, y1 = get_cut_edge(x2, y2, x1, y1)
        return min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)


def get_angle_in_2pi(unit_vec):
    theta = np.arccos(unit_vec[0])
    if unit_vec[1] > 0:
        return theta
    else:
        return 2 * np.pi - theta


####################
# math utils.
def vec_cos(a, b):
    prod = a.dot(b)
    prod = prod * 1.0 / np.linalg.norm(a) / np.linalg.norm(b)
    return prod


def compute_bbox_ratio(bbox2, bbox):
    # bbox2 is inside bbox
    s = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    s2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    return s2 * 1.0 / s


def compute_iou(boxA, boxB):
    if (
        boxA[0] > boxB[2]
        or boxB[0] > boxA[2]
        or boxA[1] > boxB[3]
        or boxB[1] > boxA[3]
    ):
        return 0
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def project2dline(
    p1,
    p2,
    cam_coords,
    cam_rotation,
    cam_near_clip=0.15,
    cam_field_of_view=50.0,
    WIDTH=1920,
    HEIGHT=2080,
):
    before1 = is_before_clip_plane(
        p1, cam_coords, cam_rotation, cam_near_clip, cam_field_of_view
    )
    before2 = is_before_clip_plane(
        p2, cam_coords, cam_rotation, cam_near_clip, cam_field_of_view
    )
    if not (before1 or before2):
        return None
    if before1 and before2:
        cp1 = get_2d_from_3d(
            p1,
            cam_coords,
            cam_rotation,
            cam_near_clip,
            cam_field_of_view,
            WIDTH,
            HEIGHT,
        )
        cp2 = get_2d_from_3d(
            p2,
            cam_coords,
            cam_rotation,
            cam_near_clip,
            cam_field_of_view,
            WIDTH,
            HEIGHT,
        )
        x1 = int(cp1[0] * WIDTH)
        x2 = int(cp2[0] * WIDTH)
        y1 = int(cp1[1] * HEIGHT)
        y2 = int(cp2[1] * HEIGHT)
        return [[x1, y1], [x2, y2]]
    center_pt, cam_dir = get_clip_center_and_dir(
        cam_coords, cam_rotation, cam_near_clip
    )
    if before1 and not before2:
        inter2 = get_intersect_point(center_pt, cam_dir, p1, p2)
        cp1 = get_2d_from_3d(
            p1,
            cam_coords,
            cam_rotation,
            cam_near_clip,
            cam_field_of_view,
            WIDTH,
            HEIGHT,
        )
        cp2 = get_2d_from_3d(
            inter2,
            cam_coords,
            cam_rotation,
            cam_near_clip,
            cam_field_of_view,
            WIDTH,
            HEIGHT,
        )
        x1 = int(cp1[0] * WIDTH)
        x2 = int(cp2[0] * WIDTH)
        y1 = int(cp1[1] * HEIGHT)
        y2 = int(cp2[1] * HEIGHT)
        return [[x1, y1], [x2, y2]]
    if before2 and not before1:
        inter1 = get_intersect_point(center_pt, cam_dir, p1, p2)
        cp2 = get_2d_from_3d(
            p2,
            cam_coords,
            cam_rotation,
            cam_near_clip,
            cam_field_of_view,
            WIDTH,
            HEIGHT,
        )
        cp1 = get_2d_from_3d(
            inter1,
            cam_coords,
            cam_rotation,
            cam_near_clip,
            cam_field_of_view,
            WIDTH,
            HEIGHT,
        )
        x1 = int(cp1[0] * WIDTH)
        x2 = int(cp2[0] * WIDTH)
        y1 = int(cp1[1] * HEIGHT)
        y2 = int(cp2[1] * HEIGHT)
        return [[x1, y1], [x2, y2]]


####################
# io utils.
def read_depthmap(name, cam_near_clip, cam_far_clip):
    depth = cv2.imread(name)
    depth = np.concatenate(
        (depth, np.zeros_like(depth[:, :, 0:1], dtype=np.uint8)), axis=2
    )
    depth.dtype = np.uint32
    depth = 0.05 * 1000 / depth.astype('float')
    depth = (
        cam_near_clip
        * cam_far_clip
        / (cam_near_clip + depth * (cam_far_clip - cam_near_clip))
    )
    return depth
