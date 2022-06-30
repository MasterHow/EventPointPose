# -*- coding: utf-8 -*-
# @Time    : 2022/6/12 15:09
# @Author  : Jiaan Chen

import os
import sys
import glob
import h5py
import numpy as np
import torch
from os.path import join
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import cv2
from eval_args_tools import point_args_init, point_model_init
from eval_3D_tools import cal_2D_mpjpe, cal_3D_mpjpe, get_pred_3d
from eval_load_tools import RasEventCloud_preprocess


def init_point_model(model_name):
    args = point_args_init(model_name)
    model = point_model_init(model_name, args)

    return model, args


def decode_batch_sa_simdr(output_x, output_y):

    max_val_x, preds_x = output_x.max(2, keepdim=True)
    max_val_y, preds_y = output_y.max(2, keepdim=True)

    output = torch.ones([output_x.size(0), preds_x.size(1), 2])
    output[:, :, 0] = torch.squeeze(preds_x)
    output[:, :, 1] = torch.squeeze(preds_y)

    output = output.cpu().numpy()
    preds = output.copy()

    return preds


def data_to_torch(data, args):
    if args.cuda_num == 1:
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data = torch.from_numpy(data).unsqueeze(dim=0).float().to(device)
    if args.model != 'PointTrans':
        data = data.permute([0, 2, 1])

    return data


def run_point_model(data, model, args):
    data = data_to_torch(data, args)
    output_x, output_y = model(data)

    decode_batch_pred = decode_batch_sa_simdr(output_x, output_y)
    batch_size = data.size()[0]
    pred = np.zeros((batch_size, 13, 2))

    pred[:, :, 1] = decode_batch_pred[:, :, 0]  # change order
    pred[:, :, 0] = decode_batch_pred[:, :, 1]

    pred = pred.squeeze()

    return pred.astype(np.int32)


def run_point_modelh5_2D(data, model, args):
    data_numpy_origin1, gt1, gt_mask1 = data[0].copy()
    data_numpy_origin2, gt2, gt_mask2 = data[1].copy()

    data_numpy_cloud1 = RasEventCloud_preprocess(data_numpy_origin1, args.num_points)
    data_numpy_cloud2 = RasEventCloud_preprocess(data_numpy_origin2, args.num_points)

    pred1 = run_point_model(data_numpy_cloud1, model, args)
    pred2 = run_point_model(data_numpy_cloud2, model, args)

    mpjpe2D_1 = cal_2D_mpjpe(gt1[np.newaxis, :], gt_mask1[np.newaxis, :], pred1)
    mpjpe2D_2 = cal_2D_mpjpe(gt2[np.newaxis, :], gt_mask2[np.newaxis, :], pred2)

    return [pred1, pred2, mpjpe2D_1, mpjpe2D_2]


def run_all_h5_3D(pred1, pred2, P_mats, Point0, label_3d):
    pred_3d = get_pred_3d(pred1, pred2, P_mats, Point0, 2, 3)

    mpjpe3D = cal_3D_mpjpe(label_3d, pred_3d)

    return [pred_3d, mpjpe3D]
