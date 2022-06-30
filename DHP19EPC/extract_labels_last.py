# -*- coding: utf-8 -*-
# @Time    : 2022/6/24 13:19
# @Author  : Jiaan Chen

# Generate DVS PointCloud LastLabel Dataset labels from .h5 labels and save as .npy, (xy)

import h5py
import numpy as np
import os
from os.path import join
import glob

# path of files generated using matlab
root_dir = 'F://DHP19EPC_dataset//test_LastLabel//label//'
out_dir = 'F://DHP19EPC_dataset//test_LastLabel_extract//label//'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# camera projection matrices path
P_mat_dir = 'F://EventPointPose//P_matrices//'


def extract_frames_labels(label_file_dir, out_dir, P_mat_cam_all, image_info):
    image_h, image_w, num_joints = image_info[:]

    filename = os.path.basename(label_file_dir)
    sub = int(filename[filename.find('S') + len('S'): filename.find('session')].split('_')[0])
    session = int(filename[filename.find('session') + len('session'): filename.find('mov')].split('_')[0])
    mov = int(filename[filename.find('mov') + len('mov'): filename.find('cam')].split('_')[0])
    cam = int(filename[filename.find('cam') + len('cam'): filename.find('label')].split('_')[0])

    labels_all = h5py.File(label_file_dir, 'r')

    vicon_xyz_all = labels_all['XYZ']  # JOINTS xyz

    data_len = len(vicon_xyz_all[:]) - 1  # delete last frame
    for frame_num in range(data_len):
        vicon_xyz = vicon_xyz_all[...][frame_num]
        vicon_xyz_homog = np.concatenate([vicon_xyz, np.ones([1, 13])], axis=0)
        coord_pix_all_cam2_homog = np.matmul(P_mat_cam_all[cam], vicon_xyz_homog)
        coord_pix_all_cam2_homog_norm = coord_pix_all_cam2_homog / coord_pix_all_cam2_homog[-1]
        u = coord_pix_all_cam2_homog_norm[0]
        v = image_h - coord_pix_all_cam2_homog_norm[1]  # flip v coordinate to match the image direction

        # mask is used to make sure that pixel positions are in frame range.
        mask = np.ones(u.shape).astype(np.float32)
        mask[u > image_w] = 0
        mask[u <= 0] = 0
        mask[v > image_h] = 0
        mask[v <= 0] = 0

        # pixel coordinates
        u = u.astype(np.int32)
        v = v.astype(np.int32)
        label_name = "S{}_session{}_mov{}_cam{}_frame{}_label{}.npy".format(sub, session, mov, cam, frame_num, "")

        out_path = out_dir + label_name

        np.save(out_path, [u, v, mask])

    labels_all.close()


image_info = [260, 346, 13]

# NB: the order of channels in the .aedat file (and in the saved .h5) is different from the camera index.
# The next cell takes care of this, loading the proper camera projection matrix.
P_mat_cam_all = []
P_mat_cam_all.append(np.load(join(P_mat_dir, 'P4.npy')))
P_mat_cam_all.append(np.load(join(P_mat_dir, 'P1.npy')))
P_mat_cam_all.append(np.load(join(P_mat_dir, 'P3.npy')))
P_mat_cam_all.append(np.load(join(P_mat_dir, 'P2.npy')))

h5labelfiles = sorted(glob.glob(os.path.join(root_dir, "*label.h5")))
n_labelfiles = len(h5labelfiles)
extract_now = 0

for h5labelfile in h5labelfiles:
    extract_frames_labels(h5labelfile, out_dir, P_mat_cam_all, image_info)
    extract_now += 1
    print('Extracting Label {} || {}'.format(extract_now, n_labelfiles))