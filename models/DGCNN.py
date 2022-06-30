# -*- coding: utf-8 -*-
# @Time    : 2022/6/10 18:27
# @Author  : Jiaan Chen, Hao Shi
# Modified from "https://github.com/WangYueFt/dgcnn"

import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x, k=20, idx=None, cudanum=0):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)
    if cudanum == 0:
        device = torch.device('cuda:0')
    elif cudanum == 1:
        device = torch.device('cuda:1')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


class Pose_DGCNN(nn.Module):
    def __init__(self, args):
        super(Pose_DGCNN, self).__init__()
        self.args = args
        self.k = 20     # Num of nearest neighbors to use
        self.cuda_num = args.cuda_num
        self.sizeH = args.sensor_sizeH
        self.sizeW = args.sensor_sizeW
        self.num_joints = args.num_joints

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(10, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(1024 * 2, 1024, bias=False)
        self.bn6 = nn.BatchNorm1d(1024)
        self.dp1 = nn.Dropout(p=0.1)
        self.linear2 = nn.Linear(1024, self.num_joints * 128)
        self.bn7 = nn.BatchNorm1d(self.num_joints * 128)
        self.dp2 = nn.Dropout(p=0.1)

        self.mlp_head_x = nn.Linear(128, self.sizeW)
        self.mlp_head_y = nn.Linear(128, self.sizeH)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k, cudanum=self.cuda_num)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k, cudanum=self.cuda_num)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k, cudanum=self.cuda_num)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k, cudanum=self.cuda_num)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)

        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)

        x = x.view(batch_size, self.num_joints, -1)
        assert x.is_contiguous()

        pred_x = self.mlp_head_x(x)
        pred_y = self.mlp_head_y(x)

        return pred_x, pred_y
