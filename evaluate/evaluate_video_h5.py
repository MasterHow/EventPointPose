# -*- coding: utf-8 -*-
# @Time    : 2022/6/12 20:32
# @Author  : Jiaan Chen

import os
import h5py
import numpy as np
from os.path import join

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
sys.path.append("..")
import cv2
from tqdm import tqdm
import matplotlib.animation as animation
from eval_load_tools import get_data_and_label, get_3D_label
from eval_run_tools import init_point_model, run_point_modelh5_2D, run_all_h5_3D
from eval_plot_tools import init_3D_video, plotVideo3Dframe, show_skeleton
from tools.image_save import accumulate_image


# test data
root_data_dir = 'F://DHP19EPC_dataset//test_MeanLabel_extract//'
h5datadir = 'F://DHP19EPC_dataset//test_MeanLabel//data//'
h5labeldir = 'F://DHP19EPC_dataset//test_MeanLabel//label//'

P_mat_dir = '../P_matrices/'
cameras_pos = np.load(join(P_mat_dir, 'camera_positions.npy'))

P_mat_cam1 = np.load(join(P_mat_dir, 'P4.npy'))
P_mat_cam2 = np.load(join(P_mat_dir, 'P1.npy'))
P_mat_cam3 = np.load(join(P_mat_dir, 'P3.npy'))
P_mat_cam4 = np.load(join(P_mat_dir, 'P2.npy'))
P_mats = [P_mat_cam1, P_mat_cam2, P_mat_cam3, P_mat_cam4]

cam_num = [4, 1, 3, 2]

cam1 = 2
cam2 = 3

# centers of the 2 used cameras
Point0 = (np.stack((cameras_pos[cam_num[cam1] - 1], cameras_pos[cam_num[cam2] - 1])))  # 3D label

sub = 13
session = 1
mov = 1

video_name = 'sub{}_session{}_mov{}'.format(sub, session, mov)

h5label_name = "S{}_session{}_mov{}_label.h5".format(sub, session, mov)
h5label = h5py.File(join(h5labeldir, h5label_name), 'r')
h5data_name = "S{}_session{}_mov{}.h5".format(sub, session, mov)
h5data = h5py.File(join(h5datadir, h5data_name), 'r')

run_result = [[], []]  # 2D and 3D results

exp_name = 'PointNet'
model, args = init_point_model('PointNet')

fps = 8  # video frame rate to save

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
if not os.path.exists('DHP19_video'):
    os.makedirs('DHP19_video')
if not os.path.exists('DHP19_video/' + exp_name):
    os.makedirs('DHP19_video/' + exp_name)
videoWriter_point = cv2.VideoWriter('./DHP19_video/' + exp_name + '/' + video_name + '.mp4', fourcc, fps, (346*2, 260), True)

print('video_length: ', len(h5label['XYZ']) - 1)

fig_point, ax_point = init_3D_video()
ims_point = []
fig_cnn, ax_cnn = init_3D_video()
ims_cnn = []

pbar = tqdm(total=len(h5label['XYZ']) - 1)
for frame in range(len(h5label['XYZ']) - 1):
    pbar.update(1)
    video_info = [sub, session, mov, frame]

    data_numpy_origin1, gt1, gt_mask1 = get_data_and_label(root_data_dir, sub, session, mov, frame, cam1)
    data_numpy_origin2, gt2, gt_mask2 = get_data_and_label(root_data_dir, sub, session, mov, frame, cam2)

    data_numpy_origin1[:, 0] = data_numpy_origin1[:, 0] - 1
    data_numpy_origin1[:, 1] = data_numpy_origin1[:, 1]
    data_numpy_origin2[:, 0] = data_numpy_origin2[:, 0] - 1
    data_numpy_origin2[:, 1] = data_numpy_origin2[:, 1]

    image1 = accumulate_image(data_numpy_origin1.T, noise_show=False)
    image2 = accumulate_image(data_numpy_origin2.T, noise_show=False)

    data = [[data_numpy_origin1, gt1, gt_mask1], [data_numpy_origin2, gt2, gt_mask2]]
    label_3d = get_3D_label(root_data_dir, sub, session, mov, frame)

    Point_2Dresult = run_point_modelh5_2D(data, model, args)
    Point_3Dresult = run_all_h5_3D(Point_2Dresult[0], Point_2Dresult[1], P_mats, Point0, label_3d.T)

    run_result[0].append(Point_2Dresult[2])
    run_result[0].append(Point_2Dresult[3])
    run_result[1].append(Point_3Dresult[1])

    im_gt_point = plotVideo3Dframe(ax_point, label_3d.T, c='red')  # gt
    im_pred_point = plotVideo3Dframe(ax_point, Point_3Dresult[0], c='blue')  # pred
    im_point = im_gt_point + im_pred_point
    im_text_point = ax_point.text2D(0.5, 1, "MPJPE3D:{:.2f}".format(Point_3Dresult[1]), fontsize=20,
                                    transform=ax_point.transAxes).findobj()
    im_point = im_point + im_text_point
    ims_point.append(im_point)

    frame_with_pred1_point = show_skeleton(image1, Point_2Dresult[0][:, 1], Point_2Dresult[0][:, 0], Point_2Dresult[2])
    frame_with_pred2_point = show_skeleton(image2, Point_2Dresult[1][:, 1], Point_2Dresult[1][:, 0], Point_2Dresult[3])
    frame_with_gt1_point = show_skeleton(frame_with_pred1_point, gt1[:, 1], gt1[:, 0])
    frame_with_gt2_point = show_skeleton(frame_with_pred2_point, gt2[:, 1], gt2[:, 0])

    image_point = np.concatenate((frame_with_gt1_point, frame_with_gt2_point), axis=1)
    videoWriter_point.write(image_point)

pbar.close()
videoWriter_point.release()

print('*** Saving 3D GIF... ***')
ani_point = animation.ArtistAnimation(fig_point, ims_point, interval=100, repeat_delay=1000)
ani_point.save('./DHP19_video/' + exp_name + '/' + video_name + '_3D.gif', writer='pillow')

print('*** '+video_name+' ***')
print(exp_name + ' result ==> 2D:{}, 3D:{}'.format(np.mean(run_result[0]), np.mean(run_result[1])))

