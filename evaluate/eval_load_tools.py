# -*- coding: utf-8 -*-
# @Time    : 2022/6/12 14:56
# @Author  : Jiaan Chen

import os
from os.path import join
import h5py
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
import time
from dataset.rasterized import RasEventCloud
from dataset.sample import random_sample_point


def get_data_and_label(root_dir, sub, session, mov, frame, cam, H=260, W=346):
    frame_name = "S{}_session{}_mov{}_frame{}_cam{}{}.npy".format(sub, session, mov, frame, cam, "")
    frame_lable_name = "S{}_session{}_mov{}_frame{}_cam{}_label{}.npy".format(sub, session, mov, frame, cam, "")

    frame_root_dir = root_dir + '/data/'
    label_root_dir = root_dir + '/label/'
    data_numpy = np.load(frame_root_dir + frame_name)

    label_numpy = np.load(label_root_dir + frame_lable_name)
    u, v, mask = label_numpy[:].astype(np.float)

    mask = np.ones(u.shape).astype(np.float32)
    mask[u > W] = 0
    mask[u <= 0] = 0
    mask[v > H] = 0
    mask[v <= 0] = 0

    gt_int = np.stack((v, u), axis=-1).astype(np.int32)

    return data_numpy, gt_int, mask


def get_3D_label(root_dir, sub, session, mov, frame):
    frame_3Dlable_name = "S{}_session{}_mov{}_frame{}_3Dlabel{}.npy".format(sub, session, mov, frame, "")
    label3D_root_dir = root_dir + '/3Dlabel/'
    label3D_numpy = np.load(label3D_root_dir + frame_3Dlable_name)

    return label3D_numpy


def RasEventCloud_preprocess(data, num_sample, sy=260, sx=346):

    if data.size == 0:
        data_temp = np.zeros((1, 5))
        data_sample, select_index = random_sample_point(data_temp, num_sample)

        return data_sample

    data_temp = data[:, 0:4].copy()
    EventCloudDHP = RasEventCloud(input_size=(4, sy, sx))

    data_temp = EventCloudDHP.convert(data_temp).numpy()[:, 1:]  # [x, y, t_avg, p_acc, event_cnt]

    data_sample, select_index = random_sample_point(data_temp, num_sample)
    data_temp = data_sample  # [num_sample, C]

    return data_temp
