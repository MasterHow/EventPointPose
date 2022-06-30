# -*- coding: utf-8 -*-
# @Time    : 2022/6/10 19:50
# @Author  : Jiaan Chen

from __future__ import print_function
import numpy as np


def random_sample_point(xyz, npoint):
    """
    Input:
        xyz: pointcloud data tensor, [N, C], C => [x,y,t,p]
        npoint: number of points after sampled
        N > npoint or N < npoint or N == npoint
    Return:
        xyz_sample: sampled pointcloud data, [npoint, C], points after deleted
        centroids: sampled pointcloud index, [npoint], points after deleted
    """

    N, C = xyz.shape
    IndexRange = np.arange(N)
    if npoint <= N:
        IndexSelect = np.sort(np.random.choice(IndexRange, size=npoint, replace=False, p=None))
    else:
        IndexSelect = np.sort(np.random.choice(IndexRange, size=npoint, replace=True, p=None))

    xyz_sample = xyz[IndexSelect, :]

    return xyz_sample, IndexSelect
