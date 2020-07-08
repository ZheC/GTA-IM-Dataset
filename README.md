# <b>GTA-IM Dataset</b> [[Website]](https://people.eecs.berkeley.edu/~zhecao/hmp/)

<div align=center>
<img src="assets/sample1.gif" width=32%>
<img src="assets/sample2.gif" width=32%>
<img src="assets/sample3.gif" width=32%>
</div>

<br>

**Long-term Human Motion Prediction with Scene Context** [[ECCV 2020 (Oral)]](https://arxiv.org/pdf/2007.03672.pdf)
<br>
[Zhe Cao](http://people.eecs.berkeley.edu/~zhecao/), [Hang Gao](http://people.eecs.berkeley.edu/~hangg/), [Karttikeya Mangalam](https://karttikeya.github.io/), [Qi-Zhi Cai](https://scholar.google.com/citations?user=oyh-YNwAAAAJ&hl=en), [Minh Vo](https://minhpvo.github.io/), [Jitendra Malik](https://people.eecs.berkeley.edu/~malik/). <br>

This repository maintains our GTA Indoor Motion dataset (GTA-IM) that emphasizes human-scene interactions in the indoor environments. We collect HD RGB-D image seuqences of 3D human motion from realistic game engine. The dataset has clean 3D human pose and camera pose annoations, and large diversity in human appearances, indoor environments, camera views, and human activities.

**Table of contents**<br>
1. [A demo for playing with our dataset.](#demo)<br>
2. [Instructions to request our full dataset.](#requesting-dataset)<br>
3. [Documentation on our dataset structure and contents.](#dataset-contents)<br>


## Demo

### (0) Getting Started
Clone this repository, and create local environment: `conda env create -f environment.yml`.

For your convinience, we provide a fragment of our data in `demo` directory. And in this section, you will be able to play with different parts of our data using maintained tool scripts.

### (1) 3D skeleton & point cloud
```bash
$ python vis_skeleton_pcd.py -h
usage: vis_skeleton_pcd.py [-h] [-pa PATH] [-f FRAME] [-fw FUSION_WINDOW]

# now visualize demo 3d skeleton and point cloud!
$ python vis_skeleton_pcd.py -pa demo -f 2720 -fw 80
```

You should be able to see a open3d viewer with our 3D skeleton and point cloud data, press 'h' in the viewer to see how to control the viewpoint:
<img src="assets/vis_skeleton_pcd.gif" width=100%>

Note that we use `open3d == 0.7.0`, the visualization code is not compatible with the newer version of open3d.

### (2) 2D skeleton & depth map
```bash
$ python vis_2d_pose_depth.py -h
usage: vis_2d_pose_depth.py [-h] [-pa PATH]

# now visualize 2d skeleton and depth map!
$ python vis_2d_pose_depth.py -pa demo
```

You should be able to find a created `demo/vis/` directory with `*_vis.jpg` that render to a movie strip like this:
<img src="assets/vis_2d_pose_depth.gif" width=80%>

### (3) RGB video
```bash
$ python vis_video.py -h
usage: vis_video.py [-h] [-pa PATH] [-s SCALE] [-fr FRAME_RATE]

# now visualize demo video!
$ python vis_video.py -pa demo -fr 15
```

You should be able to find a created `demo/vis/` directory with a `video.mp4`:

## Requesting Dataset

To obtain the Dataset, please send an email to [Zhe Cao](https://people.eecs.berkeley.edu/~zhecao/) (with the title "GTA-IM Dataset Download") stating:

- Your name, title and affilation
- Your intended use of the data
- The following statement:
    > With this email we declare that we will use the GTA-IM Dataset for non-commercial research purposes only. We also undertake to purchase a copy of Grand Theft Auto V. We will not redistribute the data in any form except in academic publications where necessary to present examples.

We will promptly reply with the download link.


## Dataset Contents

After you download data from our link and unzip, each sequence folder will contain the following files:

- `images`:
    - color images: `*.jpg`
    - depth images: `*.jpg`
    - instance masks: `*_id`.png

<br>

- `info_frames.pickle`: a pickle file contains camera information, 3d human poses (98 joints) in the global coordinate, weather condition, the character ID, and so on.

    ````python
    import pickle
    info = pickle.load(open(data_path + 'info_frames.pickle', 'rb'))
    print(info[0].keys())
    ````

<br>

- `info_frames.npz`: it contains five arrays. 21 joints out of 98 human joints are extraced to form the minimal skeleton. [Here](gen_npz.py) is how we generate it from raw captures.

    - `joints_2d`: 2d human poses on the HD image plane.
    - `joints_3d_cam`: 3d human poses in the current frame's camera coordinate
    - `joints_3d_world`: 3d human poses in the game/world coordinate
    - `world2cam_trans`: the world to camera transformation matrix for each frame
    - `intrinsics`: camera intrinsics

    <br>

    ````python
    import numpy as np
    info_npz = np.load(rec_idx+'info_frames.npz'); 
    print(info_npz.files)
    # 2d poses for frame 0
    print(npz['joints_2d'][0]) 
    ````


<br>

- `realtimeinfo.pickle`: a backup pickle file which contains all information from the data collection.

#### Joint Types

The human skeleton connection and joints index name:

```python
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
    (19, 20)  # left_knee -> left_ankle
]
```

## Important Note

This dataset is for non-commercial research purpose only. Due to public interest, I decided to reimplement the data generation pipeline from scratch to collect the GTA-IM dataset again. I do not use Facebook resources to reproduce the data. 

## Citation

We believe in open research and we will be happy if you find this data useful.
If you use it, please consider citing our [work](https://people.eecs.berkeley.edu/~zhecao/hmp/preprint.pdf).

```latex
@incollection{caoHMP2020,
  author = {Zhe Cao and
    Hang Gao and
    Karttikeya Mangalam and
    Qizhi Cai and
    Minh Vo and
    Jitendra Malik},
  title = {Long-term human motion prediction with scene context},
  booktitle = ECCV,
  year = {2020},
  }
```

## Acknowledgement

Our data collection pipeline was built upon [this plugin](https://github.com/philkr/gamehook_gtav) and [this tool](https://github.com/fabbrimatteo/JTA-Mods).

## LICENSE
Our project is released under [CC-BY-NC 4.0](https://github.com/ZheC/GTA-IM-Dataset/tree/master/LICENSE).
