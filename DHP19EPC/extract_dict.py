# -*- coding: utf-8 -*-
# @Time    : 2022/6/24 10:52
# @Author  : Jiaan Chen

# Generate DVS PointCloud Dataset dict recording event numbers from .h5 video and save as .npy
# Only for MeanLabel

import h5py
import numpy as np
import os
import glob

# path of files generated using matlab
root_dir = 'F://DHP19EPC_dataset//test_MeanLabel//data//'
out_dir = 'F://DHP19EPC_dataset//test_MeanLabel_extract//'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

Point_Num_Dict = {}
Point_Num_Dict_PerVideo = {}


def extract_dict(file_dir):
    Point_List0 = []
    Point_List1 = []
    Point_List2 = []
    Point_List3 = []
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
        Point_List0.append(PointNum[0])
        Point_List1.append(PointNum[1])
        Point_List2.append(PointNum[2])
        Point_List3.append(PointNum[3])
        for cam in range(4):
            frame_name = "S{}_session{}_mov{}_frame{}_cam{}{}.npy".format(sub, session, mov, frame_num, cam, "")
            Point_Num_Dict[frame_name] = int(PointNum[cam])

    pcdata_all.close()
    Point_List0 = np.array(Point_List0)[:, np.newaxis]
    Point_List1 = np.array(Point_List1)[:, np.newaxis]
    Point_List2 = np.array(Point_List2)[:, np.newaxis]
    Point_List3 = np.array(Point_List3)[:, np.newaxis]
    Point_Num_Dict_PerVideo[filename] = np.concatenate((Point_List0, Point_List1, Point_List2, Point_List3), axis=1)


dvs_frames = sorted(glob.glob(os.path.join(root_dir, "*.h5")))

n_files = len(dvs_frames)
extract_now = 0

for h5file in dvs_frames:
    extract_dict(h5file)
    extract_now += 1
    print('Extracting Data dict {} || {}'.format(extract_now, n_files))

np.save(out_dir + 'Point_Num_Dict.npy', Point_Num_Dict)
np.save(out_dir + 'Point_Num_Dict_PerVideo.npy', Point_Num_Dict_PerVideo)