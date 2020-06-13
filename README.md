# GTA-IM Dataset

We introduce the GTA Indoor Motion dataset (GTA-IM) that emphasizes human-scene interactions in the indoor environments. We collected a set of full-HD RGB-D sequence with clean human motion annotations. 

<div align=center>
<img src="https://github.com/ZheC/GTA-IM-Dataset/blob/master/gif/demo1.gif" width=32% style="margin-left:4%">
<img src="https://github.com/ZheC/GTA-IM-Dataset/blob/master/gif/demo2.gif" width=32% style="margin-right:4%">
<img src="https://github.com/ZheC/GTA-IM-Dataset/blob/master/gif/demo3.gif" width=32% style="margin-right:4%">
</div>

## Obtain the Dataset

To obtain the Dataset, please send an email to [Zhe Cao](https://people.eecs.berkeley.edu/~zhecao/) (with the title "GTA-IM Dataset Download") stating:

- Your name, title and affilation

- Your intended use of the data

- The following statement:
    > With this email we declare that we will use the GTA-IM Dataset for non-commercial research purposes only. We also undertake to purchase a copy of Grand Theft Auto V. We will not redistribute the data in any form except in academic publications where necessary to present examples.

We will promptly reply with the download link.


## `GTA-IM-Dataset` Contents

After the data download and unzip, each sequence folder will contain the following files:

- `images`: 

    - `color images`: `*.jpg`
    - `depth images`: `*.jpg`
    - `instance masks`: `*_id`.png


- `info_frames.pickle`: a pickle file contains camera information, 3d human poses (98 joints) in the global coordinate, weather condition, the character ID, and so on.

    ````python
    import pickle 
    info = pickle.load(open(data_path+'info_frames.pickle', 'rb'))
    print(info[0].keys())
    ````

- `info_frames.npz`: it contains five arrays. 21 joints out of 98 human joints are extraced to form the minimal skeleton. 

    - `joints_2d`: 2d human poses on the HD image plane.
    - `joints_3d_cam`: 3d human poses in the current frame's camera coordinate
    - `joints_3d_world`: 3d human poses in the game/world coordinate
    - `world2cam_trans`: the world to camera transformation matrix for each frame
    - `intrinsics`: camera intrinsics

    ````python
    import numpy as np 
    info = pickle.load(open(data_path+'info_frames.pickle', 'rb'))
    print(info[0].keys())
    ````

- `realtimeinfo.pickle`: a backup pickle file which contains all information from the data collection.

### Joint Types

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

### Data Visualization

- `vis_skeleton_mesh.py`: visaulize the 3D pose in 3D point cloud
    - requires open3d == 0.7.0.0, newer version does not work with the 3D skeleton visualization code
    - usage example: 
      ```bash
      python vis_skeleton_mesh.py -pa PATH_TO_DATA/ -f FRAME_INDEX
      ```

- `vis_2d_pose_depth.py`: visaulize the 2d pose in RGB image together with the depth map. This will create a subfolder `vis` to save the result.
    - usage example: 
        ````bash
		python vis_2dpose.py -pa PATH_TO_DATA/ 
        ````

- `write_video.py`: create a RGB video from the color images.
    - usage example: 
        ````bash
		python write_video.py -pa PATH_TO_DATA/ -s DOWN_SCALE_RATIO -fr FRAME_RATE
        ````

## Important Note

This dataset is for non-commercial research purpose only. Due to public interest, we decided to reimplement the data generation pipeline from scratch to collect the GTA-IM dataset again. We do not use FB resources to reproduce the data.

## Citation

We believe in open research and we are happy if you find this data useful.   
If you use it, please cite our [work](https://people.eecs.berkeley.edu/~zhecao/hmp/preprint.pdf).

```latex
@incollection{caoHMP2020,
  author = {Zhe Cao and
  Hang Gao and
  Karttikeya Mangalam and
  Qizhi Cai and
  Minh Vo and
  Jitendra Malik},
  title = {Long-term human motion prediction with scene context},
  booktitle = Arxiv,
  year = {2020},
  }
```

## Acknowledgement

Our data collection pipeline was built upon [this plugin](https://github.com/philkr/gamehook_gtav) and [this tool](https://github.com/fabbrimatteo/JTA-Mods).