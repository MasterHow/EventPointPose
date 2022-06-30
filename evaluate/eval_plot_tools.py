# -*- coding: utf-8 -*-
# @Time    : 2022/6/12 20:57
# @Author  : Jiaan Chen
# Modified from DHP19 "https://github.com/SensorsINI/DHP19"

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import cv2
from tools.image_save import accumulate_image


def show_skeleton(frame, u, v, mpjpe2d=None, mask=np.ones((13)), skel_only=False):
    color_pred = (255, 194, 0)  # skyblue
    color_gt = (170, 232, 238)  # yellow

    if len(frame.shape) == 2:
        # plot prediction first
        img = np.zeros((frame.shape[0], frame.shape[1], 3))
        if not skel_only:
            img[:, :, 0] = frame
            img[:, :, 1] = frame
            img[:, :, 2] = frame
        img = img.astype(np.uint8)
        color = color_pred
        if mpjpe2d:
            cv2.putText(img, 'MPJPE2D:{:.2f}'.format(mpjpe2d), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    elif len(frame.shape) == 3:
        # plot prediction second
        img = frame
        color = color_gt

    skeleton_parent_ids = [0, 0, 0, 1, 2, 1, 2, 3, 4, 5, 6, 9, 10]
    for points in range(0, 13):
        pos1 = (u[points], v[points])
        pos2 = (u[skeleton_parent_ids[points]], v[skeleton_parent_ids[points]])
        if mask[points]:
            cv2.circle(img, pos1, 2, color, -1)
            if mask[skeleton_parent_ids[points]]:
                cv2.line(img, pos1, pos2, color, 2, 5)

    return img


def skeleton(x, y, z):
    """ Definition of skeleton edges """
    # rename joints to identify name and axis
    x_head, x_shoulderR, x_shoulderL, x_elbowR, x_elbowL, x_hipR, x_hipL, x_handR, x_handL, x_kneeR, x_kneeL, x_footR, x_footL = \
        x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12]
    y_head, y_shoulderR, y_shoulderL, y_elbowR, y_elbowL, y_hipR, y_hipL, y_handR, y_handL, y_kneeR, y_kneeL, y_footR, y_footL = \
        y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12]
    z_head, z_shoulderR, z_shoulderL, z_elbowR, z_elbowL, z_hipR, z_hipL, z_handR, z_handL, z_kneeR, z_kneeL, z_footR, z_footL = \
        z[0], z[1], z[2], z[3], z[4], z[5], z[6], z[7], z[8], z[9], z[10], z[11], z[12]
    # definition of the lines of the skeleton graph
    skeleton = np.zeros((14, 3, 2))
    skeleton[0, :, :] = [[x_head, x_shoulderR], [y_head, y_shoulderR], [z_head, z_shoulderR]]
    skeleton[1, :, :] = [[x_head, x_shoulderL], [y_head, y_shoulderL], [z_head, z_shoulderL]]
    skeleton[2, :, :] = [[x_elbowR, x_shoulderR], [y_elbowR, y_shoulderR], [z_elbowR, z_shoulderR]]
    skeleton[3, :, :] = [[x_elbowL, x_shoulderL], [y_elbowL, y_shoulderL], [z_elbowL, z_shoulderL]]
    skeleton[4, :, :] = [[x_elbowR, x_handR], [y_elbowR, y_handR], [z_elbowR, z_handR]]
    skeleton[5, :, :] = [[x_elbowL, x_handL], [y_elbowL, y_handL], [z_elbowL, z_handL]]
    skeleton[6, :, :] = [[x_hipR, x_shoulderR], [y_hipR, y_shoulderR], [z_hipR, z_shoulderR]]
    skeleton[7, :, :] = [[x_hipL, x_shoulderL], [y_hipL, y_shoulderL], [z_hipL, z_shoulderL]]
    skeleton[8, :, :] = [[x_hipR, x_kneeR], [y_hipR, y_kneeR], [z_hipR, z_kneeR]]
    skeleton[9, :, :] = [[x_hipL, x_kneeL], [y_hipL, y_kneeL], [z_hipL, z_kneeL]]
    skeleton[10, :, :] = [[x_footR, x_kneeR], [y_footR, y_kneeR], [z_footR, z_kneeR]]
    skeleton[11, :, :] = [[x_footL, x_kneeL], [y_footL, y_kneeL], [z_footL, z_kneeL]]
    skeleton[12, :, :] = [[x_shoulderR, x_shoulderL], [y_shoulderR, y_shoulderL], [z_shoulderR, z_shoulderL]]
    skeleton[13, :, :] = [[x_hipR, x_hipL], [y_hipR, y_hipL], [z_hipR, z_hipL]]
    return skeleton


def plotSingle3Dframe(ax, y_true_pred, c='red', limits=[[-500, 500], [-500, 500], [0, 1500]], plot_lines=True):
    """ 3D plot of single frame. Can be both label or prediction """
    x = y_true_pred[:, 0]
    y = y_true_pred[:, 1]
    z = y_true_pred[:, 2]
    ax.scatter(x, y, z, zdir='z', s=20, c=c, marker='o', depthshade=True)
    # plot skeleton
    lines_skeleton = skeleton(x, y, z)
    if plot_lines:
        for l in range(len(lines_skeleton)):
            ax.plot(lines_skeleton[l, 0, :], lines_skeleton[l, 1, :], lines_skeleton[l, 2, :], c)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_aspect('auto')
    # set same scale for all the axis
    x_limits = limits[0]
    y_limits = limits[1]
    z_limits = limits[2]
    x_range = np.abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = np.abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = np.abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * np.max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plotVideo3Dframe(ax, y_true_pred, c='red', plot_lines=True):
    """ 3D plot of single frame. Can be both label or prediction """
    x = y_true_pred[:, 0]
    y = y_true_pred[:, 1]
    z = y_true_pred[:, 2]
    im1 = ax.scatter(x, y, z, zdir='z', s=20, c=c, marker='o', depthshade=True).findobj()
    im = im1
    # plot skeleton
    lines_skeleton = skeleton(x, y, z)
    if plot_lines:
        for l in range(len(lines_skeleton)):
            im2 = ax.plot(lines_skeleton[l, 0, :], lines_skeleton[l, 1, :], lines_skeleton[l, 2, :], c)
            im += im2

    return im


def plot3Dresults(label_3d, pred_3d, model_name, video_info, mpjpe, save=False):
    fs = 10
    fig = plt.figure('{}'.format(model_name), figsize=(fs, fs))
    ax = Axes3D(fig)
    plotSingle3Dframe(ax, label_3d, c='red')  # gt
    plotSingle3Dframe(ax, pred_3d, c='blue')  # pred

    if not save:
        plotSingle3Dframe(ax, label_3d, c='red')  # gt
        plotSingle3Dframe(ax, pred_3d, c='blue')  # pred
        plt.legend()
        plt.show()
    if save:
        fig.savefig(
            '../image_results/' + model_name + '/3D' + '/sub{}_session{}_mov{}_frame{}_{:.2f}.png'.format(video_info[0],
                                                                                                          video_info[1],
                                                                                                          video_info[2],
                                                                                                          video_info[3],
                                                                                                          mpjpe
                                                                                                          ),
            bbox_inches='tight')
    plt.close(fig)


def plot2Dresults(data, sample_gt, sample_pred, gt_mask, cam_num, model_name, video_info, mpjpe, skel_only=False,
                  joints_only=False, save=False, H=260, W=346):
    fs = 15
    fig = plt.figure('{}_sub{}_session{}_mov{}_frame{}_cam{}.png'.format(model_name, video_info[0],
                                                                         video_info[1],
                                                                         video_info[2],
                                                                         video_info[3],
                                                                         cam_num), figsize=(fs, fs))
    plt.axis('off')
    if not skel_only:
        image = accumulate_image(data.T)
        plt.imshow(image, cmap='gray')
    else:
        image = np.zeros([H, W])
        image[0, 0] = 1
        plt.imshow(image, cmap='gray')

    sample_gt_float = sample_gt.astype(np.float)
    # where mask is 0, set gt back to NaN
    sample_gt_float[gt_mask == 0] = np.nan
    sample_pred_float = sample_pred.astype(np.float)
    if joints_only:
        plt.plot(sample_gt_float[:, 1], sample_gt_float[:, 0], 'o', c='red', label='gt')
        plt.plot(sample_pred_float[:, 1], sample_pred_float[:, 0], 'o', c='blue', label='pred')
    else:
        skeleton_parent_ids = [0, 0, 0, 1, 2, 1, 2, 3, 4, 5, 6, 9, 10]
        for points in range(0, 13):
            gt_x = [sample_gt_float[points, 1], sample_gt_float[skeleton_parent_ids[points], 1]]
            gt_y = [sample_gt_float[points, 0],
                    sample_gt_float[skeleton_parent_ids[points], 0]]
            pred_x = [sample_pred_float[points, 1], sample_pred_float[skeleton_parent_ids[points], 1]]
            pred_y = [sample_pred_float[points, 0],
                      sample_pred_float[skeleton_parent_ids[points], 0]]
            if gt_mask[points]:
                if gt_mask[skeleton_parent_ids[points]]:
                    if points == 12:
                        plt.plot(gt_x, gt_y, '.-', markersize=13., linewidth=4., c='yellow', label='gt')  # gt
                        plt.plot(pred_x, pred_y, '.-', markersize=13., linewidth=4., c='deepskyblue',
                                 label='pred')
                    else:
                        plt.plot(gt_x, gt_y, '.-', markersize=13., linewidth=4., c='yellow')  # gt
                        plt.plot(pred_x, pred_y, '.-', markersize=13., linewidth=4., c='deepskyblue')  # pred
    if not save:
        plt.legend()
        plt.show()
    if save:
        fig.savefig(
            '../image_results/' + model_name + '/cam{}'.format(
                cam_num) + '/sub{}_session{}_mov{}_frame{}_cam{}_{:.2f}.png'.format(
                video_info[0],
                video_info[1],
                video_info[2],
                video_info[3],
                cam_num, mpjpe), bbox_inches='tight')
    plt.close(fig)


def init_3D_video():
    fs = 5
    fig = plt.figure(figsize=(fs, fs))
    # ax = Axes3D(fig)
    ax = fig.gca(projection='3d')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_aspect('auto')
    limits = [[-500, 500], [-500, 500], [0, 1500]]
    # set same scale for all the axis
    x_limits = limits[0]
    y_limits = limits[1]
    z_limits = limits[2]
    x_range = np.abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = np.abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = np.abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * np.max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    return fig, ax
