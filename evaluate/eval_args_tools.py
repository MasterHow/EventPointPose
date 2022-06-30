# -*- coding: utf-8 -*-
# @Time    : 2022/6/12 15:47
# @Author  : Jiaan Chen

import argparse
import torch
import sys
import torch.cuda
from models import Pose_PointNet, Pose_DGCNN, Pose_PointTransformer


def point_args_init(model_name):
    parser = argparse.ArgumentParser(description='Event Point Cloud HPE')

    parser.add_argument('--model', type=str, default=model_name, metavar='N',
                        choices=['PointNet', 'DGCNN', 'PointTrans'],
                        help='Model to use, [PointNet, DGCNN, PointTrans]')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='num of event points to use(after sample)')
    parser.add_argument('--label', type=str, default='mean', metavar='N',
                        choices=['mean', 'last'],
                        help='label setting ablation, [MeanLabel, LastLabel]')
    parser.add_argument('--cuda_num', type=int, default=0, metavar='N',
                        help='cuda device number')
    parser.add_argument('--sensor_sizeH', type=int, default=260,
                        help='sensor_sizeH')
    parser.add_argument('--sensor_sizeW', type=int, default=346,
                        help='sensor_sizeW')
    parser.add_argument('--num_joints', type=int, default=13,
                        help='number of joints')
    parser.add_argument('--model_path', type=str,
                        default='../results/' + model_name + '/models/model.pth', metavar='N',
                        help='Pretrained model path')
    args = parser.parse_args()

    return args


def point_model_init(model_name, args):

    if args.cuda_num == 1:
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if model_name == 'PointNet':
        model = Pose_PointNet(args).to(device)
    elif model_name == 'DGCNN':
        model = Pose_DGCNN(args).to(device)
    elif model_name == 'PointTrans':
        model = Pose_PointTransformer(args).to(device)
    print('Loading model...')
    model.load_state_dict(torch.load(args.model_path, map_location='cuda:0'))

    model = model.eval()

    return model
