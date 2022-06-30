# -*- coding: utf-8 -*-
# @Time    : 2022/6/10 19:42
# @Author  : Jiaan Chen

import numpy as np
import torch
import cv2
import torchvision
import math

def show_skeleton(frame, u, v, mask):
    color_pred = (255, 0, 0)
    u = u.astype(np.int)
    v = v.astype(np.int)

    # plot prediction
    img = np.zeros((frame.shape[0], frame.shape[1], 3))
    img[:, :, 0] = frame
    img[:, :, 1] = frame
    img[:, :, 2] = frame
    color = color_pred

    skeleton_parent_ids = [0, 0, 0, 1, 2, 1, 2, 3, 4, 5, 6, 9, 10]
    for points in range(0, 13):
        pos1 = (u[points], v[points])
        pos2 = (u[skeleton_parent_ids[points]], v[skeleton_parent_ids[points]])
        if mask[points]:
            cv2.circle(img, pos1, 3, color, -1)  # plot key-points
            # cv2.putText(img, str(points), pos1, cv2.FONT_HERSHEY_COMPLEX,0.5,color,1)
            if mask[skeleton_parent_ids[points]]:
                cv2.line(img, pos1, pos2, color, 2, 8)  # plot skeleton

    return img


def save_batch_image_with_joints(batch_image, file_name, nrow=8, padding=2):
    """
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3]
    batch_joints_vis: [batch_size, num_joints, 1]
    https://github.com/microsoft/human-pose-estimation.pytorch/blob/master/lib/utils/vis.py
    """
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)

    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    cv2.imwrite(file_name, ndarr)


def save_debug_images(data, joints_pred, joints_vis, save_dir, sx=346, sy=260):
    B = data.shape[0]
    batch_image = np.zeros([B, sy, sx, 3])

    for i in range(B):
        image_cam_accumulate = accumulate_image(data[i])
        batch_image[i] = show_skeleton(image_cam_accumulate, joints_pred[i, :, 0], joints_pred[i, :, 1], joints_vis[i])

    batch_image = torch.from_numpy(batch_image).permute(0, 3, 1, 2)

    save_batch_image_with_joints(batch_image, save_dir + '_pred.jpg')


def accumulate_image(data, sx=346, sy=260, noise_show=False):
    """
    Parameters
    ----------
    data : [3, num_sample]: [x, y, t] or [4, num_sample]: [x, y, t, p]
    num_sample : int
    pred2D : [13, 2]: [x, y]

    Returns
    -------
    image : accumulate image
    """
    x = data[0, :]  # x
    y = data[1, :]  # y
    t = data[2, :]  # t
    if data.shape[0] == 4:
        p = data[3, :]  # p

    img_cam = np.zeros([sy, sx])
    num_sample = len(x)

    for idx in range(num_sample):
        coordx = int(x[idx])
        coordy = sy - int(y[idx]) - 1

        img_cam[coordy, coordx] = img_cam[coordy, coordx] + 1

    if noise_show:
        img_cam *= 255.0
        image_cam_accumulate = img_cam.astype(np.uint8)
    else:
        image_cam_accumulate = normalizeImage3Sigma(img_cam, sy, sx)
        image_cam_accumulate = image_cam_accumulate.astype(np.uint8)

    return image_cam_accumulate


def normalizeImage3Sigma(image, imageH=260, imageW=346):
    """followed by matlab dhp19 generate"""
    sum_img = np.sum(image)
    count_image = np.sum(image > 0)
    mean_image = sum_img / count_image
    var_img = np.var(image[image > 0])
    sig_img = np.sqrt(var_img)

    if sig_img < 0.1 / 255:
        sig_img = 0.1 / 255

    numSDevs = 3.0
    # Rectify polarity=true
    meanGrey = 0
    range_old = numSDevs * sig_img
    half_range = 0
    range_new = 255
    # Rectify polarity=false
    # meanGrey=127 / 255
    # range= 2*numSDevs * sig_img
    # halfrange = numSDevs * sig_img

    normalizedMat = np.zeros([imageH, imageW])
    for i in range(imageH):
        for j in range(imageW):
            l = image[i, j]
            if l == 0:
                normalizedMat[i, j] = meanGrey
            else:
                f = (l + half_range) * range_new / range_old
                if f > range_new:
                    f = range_new
                if f < 0:
                    f = 0
                normalizedMat[i, j] = np.floor(f)

    return normalizedMat
