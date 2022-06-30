# -*- coding: utf-8 -*-
# @Time    : 2022/6/24 10:00
# @Author  : Jiaan Chen

# Generate DVS PointCloud MeanLabel Dataset from .h5 video and save as .npy

import h5py
import numpy as np
import os
import glob


# path of files generated using matlab
root_dir = 'F://DHP19EPC_dataset//test_MeanLabel//data//'
out_dir = 'F://DHP19EPC_dataset//test_MeanLabel_extract//data//'


if not os.path.exists(out_dir):
    os.makedirs(out_dir)


def extract_frames(file_dir, out_dir):
    filename = os.path.basename(file_dir)
    sub = int(filename[filename.find('S') + len('S'): filename.find('session')].split('_')[0])
    session = int(filename[filename.find('session') + len('session'): filename.find('mov')].split('_')[0])
    mov = int(filename[filename.find('mov') + len('mov'): filename.find('h5')].split('.')[0])

    pcdata_all = h5py.File(file_dir, 'r')

    pc = pcdata_all['DVS'][...]
    pc_camPointNum = pcdata_all['CamPointNum'][...]

    data_len = len(pc_camPointNum) - 1  # delete last frame
    for frame_num in range(data_len):
        PointNum = pc_camPointNum[frame_num]
        PointNum_start = 0
        for cam in range(4):
            pc_data = pc[frame_num, PointNum_start:PointNum_start + int(PointNum[cam]), :]

            frame_name = "S{}_session{}_mov{}_frame{}_cam{}{}.npy".format(sub, session, mov, frame_num, cam, "")
            out_path = out_dir + frame_name
            PointNum_start = PointNum_start + int(PointNum[cam])
            np.save(out_path, pc_data)
    pcdata_all.close()


dvs_frames = sorted(glob.glob(os.path.join(root_dir, "*.h5")))

n_files = len(dvs_frames)
extract_now = 0

for h5file in dvs_frames:
    extract_frames(h5file, out_dir)
    extract_now += 1
    print('Extracting Data {} || {}'.format(extract_now, n_files))

