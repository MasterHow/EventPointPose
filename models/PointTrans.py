# -*- coding: utf-8 -*-
# @Time    : 2022/6/10 18:54
# @Author  : Jiaan Chen, Hao Shi
# Modified from "https://github.com/qq456cvb/Point-Transformers/tree/master/models/Hengshuang"

import torch.nn as nn
from .PointTrans_tools import PointNetSetAbstraction, TransformerBlock


class TransitionDown(nn.Module):
    def __init__(self, k, nneighbor, channels):
        super().__init__()
        # k is the number of center points in furthest point sample
        self.sa = PointNetSetAbstraction(k, 0, nneighbor, channels[0], channels[1:], group_all=False, knn=True)

    def forward(self, xyz, points):
        return self.sa(xyz, points)


class Backbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        npoints = cfg.num_points
        nblocks = 4
        nneighbor = 16   # 16 for default
        d_points = 5  # dim for input points
        transformer_dim = 256
        self.fc1 = nn.Sequential(
            nn.Linear(d_points, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        self.transformer1 = TransformerBlock(32, transformer_dim, nneighbor)
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in range(nblocks):
            channel = 32 * 2 ** (i + 1)
            self.transition_downs.append(
                TransitionDown(npoints // 4 ** (i + 1), nneighbor, [channel // 2 + 3, channel, channel]))
            self.transformers.append(TransformerBlock(channel, transformer_dim, nneighbor))
        self.nblocks = nblocks

    def forward(self, x):
        xyz = x[..., :3]
        points = self.transformer1(xyz, self.fc1(x))[0]

        for i in range(self.nblocks):
            xyz, points = self.transition_downs[i](xyz, points)
            points = self.transformers[i](xyz, points)[0]

        return points


class Pose_PointTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.backbone = Backbone(args)
        self.args = args
        self.sizeH = args.sensor_sizeH
        self.sizeW = args.sensor_sizeW
        self.num_joints = args.num_joints
        nblocks = 4

        self.fc2 = nn.Sequential(
            nn.Linear(32 * 2 ** nblocks, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_joints * 128),
            nn.ReLU(),
        )

        self.mlp_head_x = nn.Linear(128, self.sizeW)
        self.mlp_head_y = nn.Linear(128, self.sizeH)

    def forward(self, x):
        batch_size = x.size(0)

        points = self.backbone(x)

        res = self.fc2(points.mean(1))
        res = res.view(batch_size, 13, -1)
        pred_x = self.mlp_head_x(res)
        pred_y = self.mlp_head_y(res)

        return pred_x, pred_y

