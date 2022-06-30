# -*- coding: utf-8 -*-
# @Time    : 2022/6/12 16:27
# @Author  : Jiaan Chen
# Modified from DHP19 "https://github.com/SensorsINI/DHP19"

import os
from os.path import join
import h5py
import numpy as np
import torch


def project_uv_xyz_cam(uv, M):
    # adapted from: https://www.cc.gatech.edu/~hays/compvision/proj3/
    N = len(uv)
    uv_homog = np.hstack((uv, np.ones((N, 1))))
    M_inv = np.linalg.pinv(M)
    xyz = np.dot(M_inv, uv_homog.T).T
    x = xyz[:, 0] / xyz[:, 3]
    y = xyz[:, 1] / xyz[:, 3]
    z = xyz[:, 2] / xyz[:, 3]
    return x, y, z


def find_intersection(P0, P1):
    # from: https://stackoverflow.com/questions/52088966/nearest-intersection-point-to-many-lines-in-python/52089867
    """P0 and P1 are NxD arrays defining N lines.
    D is the dimension of the space. This function
    returns the least squares intersection of the N
    lines from the system given by eq. 13 in
    http://cal.cs.illinois.edu/~johannes/research/LS_line_intersect.pdf."""

    # generate all line direction vectors
    n = (P1 - P0) / np.linalg.norm(P1 - P0, axis=1)[:, np.newaxis]  # normalized
    # generate the array of all projectors
    projs = np.eye(n.shape[1]) - n[:, :, np.newaxis] * n[:, np.newaxis]  # I - n*n.T
    # see fig. 1

    # generate R matrix and q vector
    R = projs.sum(axis=0)
    q = (np.matmul(projs, P0[:, :, np.newaxis])).sum(axis=0)
    # solve the least squares problem for the
    # intersection point p: Rp = q
    p = np.linalg.lstsq(R, q, rcond=None)[0]
    return p.T


def get_pred_3d(pred1, pred2, P_mats, Point0, cam1, cam2, H=260, W=346):
    # initialize empty sample of 3D prediction
    pred_3d = np.zeros((13, 3))

    pred_2d_cam2 = pred1
    pred_2d_cam3 = pred2

    pred_2d_cam2_ = np.zeros(pred_2d_cam2.shape)
    pred_2d_cam3_ = np.zeros(pred_2d_cam3.shape)

    pred_2d_cam2_[:, 0] = pred_2d_cam2[:, 1]  # change order
    pred_2d_cam2_[:, 1] = H - pred_2d_cam2[:, 0]

    pred_2d_cam3_[:, 0] = pred_2d_cam3[:, 1]
    pred_2d_cam3_[:, 1] = H - pred_2d_cam3[:, 0]

    x_cam2_pred, y_cam2_pred, z_cam2_pred = project_uv_xyz_cam(pred_2d_cam2_, P_mats[cam1])
    x_cam3_pred, y_cam3_pred, z_cam3_pred = project_uv_xyz_cam(pred_2d_cam3_, P_mats[cam2])

    xyz_cam2 = np.stack((x_cam2_pred, y_cam2_pred, z_cam2_pred), axis=1)
    xyz_cam3 = np.stack((x_cam3_pred, y_cam3_pred, z_cam3_pred), axis=1)

    for joint_idx in range(13):
        # coordinates for both cameras of 2nd point of triangulation line.
        Point1 = np.stack((xyz_cam2[joint_idx, :], xyz_cam3[joint_idx, :]), axis=1).T
        intersection = find_intersection(Point0, Point1)
        pred_3d[joint_idx] = intersection[0]

    return pred_3d


def cal_2D_mpjpe(gt, gt_mask, pred):
    gt_float = gt.astype(np.float)
    # where mask is 0, set gt back to NaN
    gt_float[gt_mask == 0] = np.nan
    pred_float = pred.astype(np.float)
    dist_2d = np.linalg.norm((gt_float - pred_float), axis=-1)
    mpjpe2D = np.nanmean(dist_2d)

    return mpjpe2D


def cal_3D_mpjpe(label_3d, pred_3d):
    mpjpe_3d_joints = np.linalg.norm((label_3d - pred_3d), axis=-1)
    mpjpe_3d_sample = np.nanmean(mpjpe_3d_joints)

    return mpjpe_3d_sample