# -*- coding: utf-8 -*-
# @Time    : 2022/6/10 14:18
# @Author  : Jiaan Chen, Hao Shi

from __future__ import print_function
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from dataset import DHP19EPC
from models import Pose_PointNet, Pose_PointTransformer, Pose_DGCNN
import numpy as np
from torch.utils.data import DataLoader
from tools.utils import init_dir, IOStream, decode_batch_sa_simdr, accuracy, KLDiscretLoss
from tools.geometry_function import get_pred_3d_batch, cal_2D_mpjpe, cal_3D_mpjpe
from tools.image_save import save_debug_images
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from os.path import join


def train(exp_name, args, io):
    P_mat_dir = './P_matrices/'

    if args.label == 'mean':
        # mean label
        # path of train data
        root_train_data_dir = 'F://DHP19EPC_dataset//train_MeanLabel_extract//'
        # path of valid data
        root_valid_data_dir = 'F://DHP19EPC_dataset//test_MeanLabel_extract//'

        train_dataset = DHP19EPC(
            args,
            root_data_dir=root_train_data_dir + 'data//',
            root_label_dir=root_train_data_dir + 'label//',
            root_3Dlabel_dir=root_train_data_dir + '3Dlabel//',
            root_dict_dir=root_train_data_dir + 'Point_Num_Dict.npy',
            min_EventNum=1024, Test3D=False
        )
        valid_dataset = DHP19EPC(
            args,
            root_data_dir=root_valid_data_dir + 'data//',
            root_label_dir=root_valid_data_dir + 'label//',
            root_3Dlabel_dir=root_valid_data_dir + '3Dlabel//',
            root_dict_dir=root_valid_data_dir + 'Point_Num_Dict.npy',
            min_EventNum=0, Test3D=True,
        )

    elif args.label == 'last':
        # last label
        # path of train data
        root_train_data_dir = 'F://DHP19EPC_dataset//train_LastLabel_extract//'
        # path of valid data
        root_valid_data_dir = 'F://DHP19EPC_dataset//test_LastLabel_extract//'

        train_dataset = DHP19EPC(
            args,
            root_data_dir=root_train_data_dir + 'data/',
            root_label_dir=root_train_data_dir + 'label/',
            Test3D=False,
        )
        valid_dataset = DHP19EPC(
            args,
            root_data_dir=root_valid_data_dir + 'data/',
            root_label_dir=root_valid_data_dir + 'label/',
            Test3D=False
        )

    train_loader = DataLoader(train_dataset, num_workers=8,
                              batch_size=args.train_batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, num_workers=8,
                              batch_size=args.valid_batch_size, shuffle=False, drop_last=False)

    if args.cuda_num == 1:
        device = torch.device("cuda:1" if args.cuda else "cpu")
    else:
        device = torch.device("cuda:0" if args.cuda else "cpu")

    # Try to load models
    if args.model == 'DGCNN':
        model = Pose_DGCNN(args).to(device)
    elif args.model == 'PointNet':
        model = Pose_PointNet(args).to(device)
    elif args.model == 'PointTrans':
        model = Pose_PointTransformer(args).to(device)
    else:
        raise Exception("Not implemented")

    opt = optim.Adam(model.parameters(), lr=0.0001)

    scheduler = MultiStepLR(opt, [15, 20], 0.1)

    criterion = KLDiscretLoss()

    LogWriter = SummaryWriter(log_dir='logs/%s/' % exp_name)
    global_train_steps = 0

    best_valid_MPJPE3D = 1e10
    best_valid_MPJPE2D = 1e10
    model.train()

    for epoch in range(args.epochs):
        pbar = tqdm(total=len(train_loader))

        scheduler.step()

        # ********Train********
        train_loss_list = []
        train_acc_cnt_all = 0.0
        train_acc_final = 0.0

        for i, (data, xlabel, ylabel, wlabel) in enumerate(train_loader):

            data, xlabel, ylabel, wlabel = data.to(device), xlabel.to(device), ylabel.to(device), wlabel.to(device)

            if args.model != 'PointTrans':
                data = data.permute(0, 2, 1)

            output_x, output_y = model(data.float())

            # KL loss
            loss = criterion(output_x, output_y, xlabel, ylabel, wlabel)

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss_list.append(loss.item())

            decode_batch_pred = decode_batch_sa_simdr(output_x, output_y)
            decode_batch_label = decode_batch_sa_simdr(xlabel, ylabel)

            acc, avg_acc, cnt, pred = accuracy(decode_batch_pred, decode_batch_label, hm_type='sa-simdr', thr=0.5)

            train_acc_cnt_all += cnt
            train_acc_final += avg_acc * cnt

            if args.save_image:
                if i % 2000 == 0:
                    save_dir = 'checkpoints/{}/output_image/train/train_{}'.format(exp_name, i)
                    if args.model == 'PointTrans':
                        data = data.permute(0, 2, 1)
                    save_debug_images(data, decode_batch_pred, wlabel, save_dir)

            if i % 20 == 0 and i > 0:
                outstr = 'Train Step %d | %d epoch, Loss: %.6f, Acc: %.6f' % (global_train_steps, epoch + 1,
                                                                              np.mean(train_loss_list),
                                                                              train_acc_final / train_acc_cnt_all)
                io.cprint(outstr)

                LogWriter.add_scalar('Train_loss', np.mean(train_loss_list), global_train_steps)
                LogWriter.add_scalar('Train_acc', train_acc_final / train_acc_cnt_all, global_train_steps)

                train_loss_list = []
                train_acc_cnt_all = 0.0
                train_acc_final = 0.0

            global_train_steps += 1

            pbar.update(1)

        pbar.close()

        print(outstr)

        # ********Valid********
        # if (epoch + 1) % 2 == 0 and (epoch >= args.epochs // 2) or (epoch == args.epochs - 1):
        if (epoch + 1) % 2 == 0 and (epoch >= 0) or (epoch == args.epochs - 1):

            cameras_pos = np.load(join(P_mat_dir, 'camera_positions.npy'))

            P_mat_cam1 = np.load(join(P_mat_dir, 'P4.npy'))
            P_mat_cam2 = np.load(join(P_mat_dir, 'P1.npy'))
            P_mat_cam3 = np.load(join(P_mat_dir, 'P3.npy'))
            P_mat_cam4 = np.load(join(P_mat_dir, 'P2.npy'))
            P_mats = [P_mat_cam1, P_mat_cam2, P_mat_cam3, P_mat_cam4]

            cam_num = [4, 1, 3, 2]

            cam1 = 2
            cam2 = 3

            # centers of the 2 used cameras
            Point0 = (np.stack((cameras_pos[cam_num[cam1] - 1], cameras_pos[cam_num[cam2] - 1])))  # 3D label

            valid_mpjpe2D_all = []
            valid_mpjpe3D_all = []
            valid_acc_cnt_all = 0.0
            valid_acc_final = 0.0
            model.eval()
            with torch.no_grad():
                pbar = tqdm(total=len(valid_loader))
                if args.label == 'mean':
                    for i, (data, xlabel, ylabel, wlabel, label3D) in enumerate(valid_loader):
                        data2 = data[0].to(device)
                        data3 = data[1].to(device)

                        batch_size = data2.size()[0]
                        if args.model != 'PointTrans':
                            data2 = data2.permute([0, 2, 1])
                            data3 = data3.permute([0, 2, 1])

                        data2 = data2.float().to(device)
                        data3 = data3.float().to(device)

                        xlabel2 = xlabel[0].to(device)
                        xlabel3 = xlabel[1].to(device)

                        ylabel2 = ylabel[0].to(device)
                        ylabel3 = ylabel[1].to(device)

                        wlabel2 = wlabel[0].to(device)
                        wlabel3 = wlabel[1].to(device)

                        label3D = label3D.to(device)

                        output_x2, output_y2 = model(data2)
                        output_x3, output_y3 = model(data3)

                        decode_batch_label2 = decode_batch_sa_simdr(xlabel2, ylabel2)
                        decode_batch_label3 = decode_batch_sa_simdr(xlabel3, ylabel3)

                        decode_batch_pred2 = decode_batch_sa_simdr(output_x2, output_y2)
                        decode_batch_pred3 = decode_batch_sa_simdr(output_x3, output_y3)

                        pred2 = np.zeros((batch_size, 13, 2))
                        pred3 = np.zeros((batch_size, 13, 2))
                        pred2[:, :, 1] = decode_batch_pred2[:, :, 0]  # exchange x,y order
                        pred2[:, :, 0] = decode_batch_pred2[:, :, 1]

                        pred3[:, :, 1] = decode_batch_pred3[:, :, 0]  # exchange x,y order
                        pred3[:, :, 0] = decode_batch_pred3[:, :, 1]

                        pred_3d = get_pred_3d_batch(pred2, pred3, label3D, P_mats, Point0, cam1, cam2)
                        pred_3d = torch.from_numpy(pred_3d).to(device)

                        Loss2D2 = cal_2D_mpjpe(decode_batch_label2, wlabel2.squeeze(dim=2).cpu(), decode_batch_pred2)
                        Loss2D3 = cal_2D_mpjpe(decode_batch_label3, wlabel3.squeeze(dim=2).cpu(), decode_batch_pred3)

                        valid_mpjpe2D_all.append(Loss2D2)
                        valid_mpjpe2D_all.append(Loss2D3)

                        loss3D = cal_3D_mpjpe(label3D.transpose(2, 1).cpu(), pred_3d.cpu())
                        valid_mpjpe3D_all.append(loss3D)

                        acc, avg_acc, cnt, pred = accuracy(decode_batch_pred2, decode_batch_label2, hm_type='sa-simdr',
                                                           thr=0.5)

                        valid_acc_cnt_all += cnt
                        valid_acc_final += avg_acc * cnt

                        acc, avg_acc, cnt, pred = accuracy(decode_batch_pred3, decode_batch_label3, hm_type='sa-simdr',
                                                           thr=0.5)

                        valid_acc_cnt_all += cnt
                        valid_acc_final += avg_acc * cnt

                        if args.save_image:
                            if i % 1000 == 0:
                                save_dir = 'checkpoints/{}/output_image/valid/valid_{}'.format(exp_name, i)
                                if args.model == 'PointTrans':
                                    data2 = data2.permute(0, 2, 1)
                                save_debug_images(data2, decode_batch_pred2, wlabel2, save_dir)

                        pbar.update(1)

                elif args.label == 'last':
                    for i, (data, xlabel, ylabel, wlabel) in enumerate(valid_loader):
                        data, xlabel, ylabel, wlabel = data.to(device), xlabel.to(device), ylabel.to(device), wlabel.to(
                            device)

                        if args.model != 'PointTrans':
                            data = data.permute(0, 2, 1)

                        output_x, output_y = model(data.float())

                        decode_batch_pred = decode_batch_sa_simdr(output_x, output_y)
                        decode_batch_label = decode_batch_sa_simdr(xlabel, ylabel)

                        acc, avg_acc, cnt, pred = accuracy(decode_batch_pred, decode_batch_label, hm_type='sa-simdr',
                                                           thr=0.5)

                        valid_acc_cnt_all += cnt
                        valid_acc_final += avg_acc * cnt

                        Loss2D = cal_2D_mpjpe(decode_batch_label, wlabel.squeeze(dim=2).cpu(), decode_batch_pred)
                        valid_mpjpe2D_all.append(Loss2D)

                        if args.save_image:
                            if i % 1000 == 0:
                                save_dir = 'checkpoints/{}/output_image/valid/valid_{}'.format(exp_name, i)
                                if args.model == 'PointTrans':
                                    data = data.permute(0, 2, 1)
                                save_debug_images(data, decode_batch_pred, wlabel, save_dir)

                        pbar.update(1)

                pbar.close()

            LogWriter.add_scalar('Valid_acc', valid_acc_final / valid_acc_cnt_all, epoch + 1)
            LogWriter.add_scalar('Valid_MPJPE2D', np.mean(valid_mpjpe2D_all), epoch + 1)

            if args.label == 'mean':
                LogWriter.add_scalar('Valid_MPJPE3D', np.mean(valid_mpjpe3D_all), epoch + 1)
                outstr = 'Valid %d epoch, Acc: %.6f, MPJPE2D: %.6f, MPJPE3D: %.6f' % (epoch + 1,
                                                                                valid_acc_final / valid_acc_cnt_all,
                                                                                np.mean(valid_mpjpe2D_all),
                                                                                np.mean(valid_mpjpe3D_all))
            elif args.label == 'last':
                outstr = 'Valid %d epoch, Acc: %.6f, MPJPE2D: %.6f' % (epoch + 1,
                                                                 valid_acc_final / valid_acc_cnt_all,
                                                                 np.mean(valid_mpjpe2D_all))
            io.cprint(outstr)

            if args.label == 'mean':
                if (np.mean(valid_mpjpe3D_all)) <= best_valid_MPJPE3D:
                    best_valid_MPJPE3D = np.mean(valid_mpjpe3D_all)
                    best_valid_MPJPE2D = np.mean(valid_mpjpe2D_all)
                    torch.save(model.state_dict(),
                               'checkpoints/{}/models/model.pth'.format(exp_name))
            elif args.label == 'last':
                if (np.mean(valid_mpjpe2D_all)) <= best_valid_MPJPE2D:
                    best_valid_MPJPE2D = np.mean(valid_mpjpe2D_all)
                    torch.save(model.state_dict(),
                               'checkpoints/{}/models/model.pth'.format(exp_name))

            model.train()

    print('Best model is saved!')
    print('MPJPE3D: {:.2f} || MPJPE2D: {:.2f}'.format(best_valid_MPJPE3D, best_valid_MPJPE2D))


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Event Point Cloud HPE')

    parser.add_argument('--model', type=str, default='PointNet', metavar='N',
                        choices=['PointNet', 'DGCNN', 'PointTrans'],
                        help='Model to use, [PointNet, DGCNN, PointTrans]')
    parser.add_argument('--train_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of train batch)')
    parser.add_argument('--valid_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of valid batch)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--num_points', type=int, default=7500,
                        help='Number of event points to use(after sample)')
    parser.add_argument('--sensor_sizeH', type=int, default=260,
                        help='sensor_sizeH')
    parser.add_argument('--sensor_sizeW', type=int, default=346,
                        help='sensor_sizeW')
    parser.add_argument('--num_joints', type=int, default=13,
                        help='number of joints')
    parser.add_argument('--label', type=str, default='mean', metavar='N',
                        choices=['mean', 'last'],
                        help='label setting ablation, [MeanLabel, LastLabel]')
    parser.add_argument('--name', type=str, default='Experiment1',
                        help='Name your exp')
    parser.add_argument('--cuda_num', type=int, default=0, metavar='N',
                        help='cuda device number')
    parser.add_argument('--save_image', action='store_true',
                        help='save image for debug')
    args = parser.parse_args()

    exp_name = args.name

    init_dir(args)

    io = IOStream('checkpoints/' + exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    train(exp_name, args, io)

    print('******** Finish ' + exp_name + ' ********')
