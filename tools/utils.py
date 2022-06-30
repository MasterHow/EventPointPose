# -*- coding: utf-8 -*-
# @Time    : 2022/6/10 19:20
# @Author  : Jiaan Chen

import os
import numpy as np
import torch
import torch.nn as nn


def init_dir(args):
    exp_name = args.name
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + exp_name):
        os.makedirs('checkpoints/' + exp_name)
    if not os.path.exists('checkpoints/' + exp_name + '/'+'models'):
        os.makedirs('checkpoints/' + exp_name + '/' + 'models')
    if args.save_image:
        if not os.path.exists('checkpoints/' + exp_name + '/'+'output_image/train'):
            os.makedirs('checkpoints/' + exp_name + '/' + 'output_image/train')
        if not os.path.exists('checkpoints/' + exp_name + '/'+'output_image/valid'):
            os.makedirs('checkpoints/' + exp_name + '/' + 'output_image/valid')


class KLDiscretLoss(nn.Module):
    """
    "https://github.com/leeyegy/SimDR"
    """
    def __init__(self):
        super(KLDiscretLoss, self).__init__()
        self.LogSoftmax = nn.LogSoftmax(dim=1)  # [B,LOGITS]
        self.criterion_ = nn.KLDivLoss(reduction='none')

    def criterion(self, dec_outs, labels):
        scores = self.LogSoftmax(dec_outs)
        loss = torch.mean(self.criterion_(scores, labels), dim=1)
        return loss

    def forward(self, output_x, output_y, target_x, target_y, target_weight):
        num_joints = output_x.size(1)
        # print(num_joints)
        loss = 0
        for idx in range(num_joints):
            coord_x_pred = output_x[:, idx]
            coord_y_pred = output_y[:, idx]
            coord_x_gt = target_x[:, idx]
            coord_y_gt = target_y[:, idx]
            weight = target_weight[:, idx]
            loss += (self.criterion(coord_x_pred, coord_x_gt).mul(weight).mean())
            loss += (self.criterion(coord_y_pred, coord_y_gt).mul(weight).mean())

        return loss / num_joints


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def decode_sa_simdr(output_x, output_y):

    output_x = torch.from_numpy(output_x).unsqueeze(0)
    output_y = torch.from_numpy(output_y).unsqueeze(0)

    max_val_x, preds_x = output_x.max(2, keepdim=True)
    max_val_y, preds_y = output_y.max(2, keepdim=True)

    output = torch.ones([output_x.size(0), preds_x.size(1), 2])
    output[:, :, 0] = torch.squeeze(preds_x)
    output[:, :, 1] = torch.squeeze(preds_y)

    output = output.cpu().numpy()
    preds = output.copy()

    pred = preds.squeeze(0)
    x = pred[:, 0]
    y = pred[:, 1]

    return x, y


def decode_batch_sa_simdr(output_x, output_y):

    max_val_x, preds_x = output_x.max(2, keepdim=True)
    max_val_y, preds_y = output_y.max(2, keepdim=True)

    output = torch.ones([output_x.size(0), preds_x.size(1), 2])
    output[:, :, 0] = torch.squeeze(preds_x)
    output[:, :, 1] = torch.squeeze(preds_y)

    output = output.cpu().numpy()
    preds = output.copy()

    return preds


def calc_dists(preds, target, normalize):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
            else:
                dists[c, n] = -1
    return dists


def dist_acc(dists, thr=0.5):
    """ Return percentage below threshold while ignoring values with a -1 """
    # dist => [batch,]
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1


def accuracy(output, target, hm_type='sa-simdr', thr=0.5, sx=346, sy=260):
    """
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    "https://github.com/leeyegy/SimDR"
    """
    idx = list(range(output.shape[1]))
    norm = 1.0
    if hm_type == 'sa-simdr':
        pred = output
        target = target
        h = sy
        w = sx
        norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10
    # dists => [batch, 13]
    dists = calc_dists(pred, target, norm)

    acc = np.zeros((len(idx) + 1))
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i + 1] = dist_acc(dists[idx[i]], thr)
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc
    return acc, avg_acc, cnt, pred


