# -*- coding: utf-8 -*-
# @Time    : 2022/6/24 10:59
# @Author  : Jiaan Chen

# Generate DVS PointCloud Dataset 3D labels from .h5 labels and save as .npy, (xyz)
# Only for MeanLabel

import h5py
import numpy as np
import os
import glob

# path of files generated using matlab
root_dir = 'F://DHP19EPC_dataset//test_MeanLabel//label//'
out_dir = 'F://DHP19EPC_dataset//test_MeanLabel_extract//3Dlabel//'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# camera projection matrices path
P_mat_dir = 'F://EventPointPose//P_matrices//'


def extract_frames_3Dlabels(label_file_dir, out_dir, image_info):
    image_h, image_w, num_joints = image_info[:]

    filename = os.path.basename(label_file_dir)
    sub = int(filename[filename.find('S') + len('S'): filename.find('session')].split('_')[0])
    session = int(filename[filename.find('session') + len('session'): filename.find('mov')].split('_')[0])
    mov = int(filename[filename.find('mov') + len('mov'): filename.find('label')].split('_')[0])

    labels_all = h5py.File(label_file_dir, 'r')

    vicon_xyz_all = labels_all['XYZ']  # JOINTS xyz

    data_len = len(vicon_xyz_all[:]) - 1  # delete last frame
    for frame_num in range(data_len):
        vicon_xyz = vicon_xyz_all[...][frame_num]

        label_name = "S{}_session{}_mov{}_frame{}_3Dlabel{}.npy".format(sub, session, mov, frame_num, "")
        out_path = out_dir + label_name

        np.save(out_path, vicon_xyz)

    labels_all.close()


image_info = [260, 346, 13]

h5labelfiles = sorted(glob.glob(os.path.join(root_dir, "*label.h5")))
n_labelfiles = len(h5labelfiles)
extract_now = 0

for h5labelfile in h5labelfiles:
    extract_frames_3Dlabels(h5labelfile, out_dir, image_info)
    extract_now += 1
    print('Extracting Label {} || {}'.format(extract_now, n_labelfiles))

