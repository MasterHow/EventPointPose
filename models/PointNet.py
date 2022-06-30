# -*- coding: utf-8 -*-
# @Time    : 2022/6/10 18:41
# @Author  : Hao Shi, Jiaan Chen

import torch
import torch.nn as nn
import torch.nn.functional as F


class Pose_PointNet(nn.Module):
    def __init__(self, args):
        super(Pose_PointNet, self).__init__()
        self.args = args
        self.sizeH = args.sensor_sizeH
        self.sizeW = args.sensor_sizeW
        self.num_joints = args.num_joints

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(1024)
        self.bn6 = nn.BatchNorm1d(1024)
        self.bn7 = nn.BatchNorm1d(self.num_joints * 128)

        self.dp1 = nn.Dropout(p=0.1)
        self.dp2 = nn.Dropout(p=0.1)

        self.conv1 = nn.Sequential(nn.Conv1d(5, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU())
        self.conv3 = nn.Sequential(nn.Conv1d(64, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU())
        self.conv4 = nn.Sequential(nn.Conv1d(128, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU())
        self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU())

        self.out_conv1 = nn.Sequential(nn.Linear(1024 * 2, 1024, bias=False),
                                       self.bn6,
                                       nn.LeakyReLU(),
                                       self.dp1)
        self.out_conv2 = nn.Sequential(nn.Linear(1024, self.num_joints * 128, bias=False),
                                       self.bn7,
                                       nn.LeakyReLU(),
                                       self.dp2)

        self.mlp_head_x = nn.Linear(128, self.sizeW)
        self.mlp_head_y = nn.Linear(128, self.sizeH)

    def forward(self, x):
        batch_size = x.size(0)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)

        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = self.out_conv1(x)
        x = self.out_conv2(x)

        x = x.view(batch_size, self.num_joints, -1)

        pred_x = self.mlp_head_x(x)
        pred_y = self.mlp_head_y(x)

        return pred_x, pred_y
