# -*- coding: utf-8 -*-
# @Time    : 2022/6/24 12:56
# @Author  : Jiaan Chen

# Generate DVS PointCloud LastLabel Dataset from .h5 video and save as .npy

import h5py
import numpy as np
import os
import glob


# path of files generated using matlab
root_dir = 'F://DHP19EPC_dataset//test_LastLabel//data//'
out_dir = 'F://DHP19EPC_dataset//test_LastLabel_extract//data//'


if not os.path.exists(out_dir):
    os.makedirs(out_dir)


def extract_frames(file_dir, out_dir):
    filename = os.path.basename(file_dir)
    sub = int(filename[filename.find('S') + len('S'): filename.find('session')].split('_')[0])
    session = int(filename[filename.find('session') + len('session'): filename.find('mov')].split('_')[0])
    mov = int(filename[filename.find('mov') + len('mov'): filename.find('cam')].split('_')[0])
    cam = int(filename[filename.find('cam') + len('cam'): filename.find('h5')].split('.')[0])
    pcdata_all = h5py.File(file_dir, 'r')

    pc = pcdata_all['DVS'][...]

    data_len = len(pc) - 1  # delete last frame
    for frame_num in range(data_len):
        pc_data = pc[frame_num, :, :]
        frame_name = "S{}_session{}_mov{}_cam{}_frame{}{}.npy".format(sub, session, mov, cam, frame_num, "")
        out_path = out_dir + frame_name
        np.save(out_path, pc_data)

    pcdata_all.close()

dvs_frames = sorted(glob.glob(os.path.join(root_dir, "*.h5")))

n_files = len(dvs_frames)
extract_now = 0

for h5file in dvs_frames:
    extract_frames(h5file, out_dir)
    extract_now += 1
    print('Extracting Data {} || {}'.format(extract_now, n_files))
