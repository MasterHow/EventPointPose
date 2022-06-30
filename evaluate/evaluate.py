# -*- coding: utf-8 -*-
# @Time    : 2022/6/12 18:45
# @Author  : Jiaan Chen


import numpy as np
from os.path import join
import torch
import sys

sys.path.append("..")
from dataset import DHP19EPC
from torch.utils.data import DataLoader
from tools.geometry_function import cal_2D_mpjpe, cal_3D_mpjpe, get_pred_3d_batch
from tools.utils import decode_batch_sa_simdr, accuracy
from eval_run_tools import init_point_model
from tqdm import tqdm


P_mat_dir = '../P_matrices/'

model, args = init_point_model('PointNet')

# path of valid data mean label
root_valid_data_dir = 'F://DHP19EPC_dataset//test_MeanLabel_extract//'
test = DHP19EPC(
            args,
            root_data_dir=root_valid_data_dir + 'data//',
            root_label_dir=root_valid_data_dir + 'label//',
            root_3Dlabel_dir=root_valid_data_dir + '3Dlabel//',
            root_dict_dir=root_valid_data_dir + 'Point_Num_Dict.npy',
            min_EventNum=0, Test3D=True,
        )

# path of valid data last label
# root_valid_data_dir = 'F://DHP19EPC_dataset//test_LastLabel_extract//'
# args.label = 'last'
# test = DHP19EPC(
#             args,
#             root_data_dir=root_valid_data_dir + 'data/',
#             root_label_dir=root_valid_data_dir + 'label/',
#             Test3D=False
#         )

num_workers = 0
test_loader = DataLoader(test, num_workers=num_workers, batch_size=16, shuffle=False, drop_last=False)

acc_cnt_all = 0
acc_final = 0
mpjpe2D2_all = []
mpjpe2D3_all = []
mpjpe2D_all = []
mpjpe3D_all = []

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

if args.cuda_num == 1:
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with torch.no_grad():
    pbar = tqdm(total=len(test_loader))

    for i, (data, xlabel, ylabel, wlabel, label3D) in enumerate(test_loader):
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

        # test the re-project error
        # decode_batch_pred2 = decode_batch_sa_simdr(xlabel2, ylabel2)
        # decode_batch_pred3 = decode_batch_sa_simdr(xlabel3, ylabel3)

        pred2 = np.zeros((batch_size, 13, 2))
        pred3 = np.zeros((batch_size, 13, 2))
        pred2[:, :, 1] = decode_batch_pred2[:, :, 0]  # change order
        pred2[:, :, 0] = decode_batch_pred2[:, :, 1]

        pred3[:, :, 1] = decode_batch_pred3[:, :, 0]  # change order
        pred3[:, :, 0] = decode_batch_pred3[:, :, 1]

        pred_3d = get_pred_3d_batch(pred2, pred3, label3D, P_mats, Point0, cam1, cam2)
        pred_3d = torch.from_numpy(pred_3d).to(device)

        Loss2D2 = cal_2D_mpjpe(decode_batch_label2, wlabel2.squeeze(dim=2).cpu(), decode_batch_pred2)
        mpjpe2D2_all.append(Loss2D2)

        Loss2D3 = cal_2D_mpjpe(decode_batch_label3, wlabel3.squeeze(dim=2).cpu(), decode_batch_pred3)
        mpjpe2D3_all.append(Loss2D3)

        mpjpe2D_all.append(Loss2D2)
        mpjpe2D_all.append(Loss2D3)

        loss3D = cal_3D_mpjpe(label3D.transpose(2, 1).cpu(), pred_3d.cpu())
        mpjpe3D_all.append(loss3D)

        acc, avg_acc, cnt, pred = accuracy(decode_batch_pred2, decode_batch_label2, hm_type='sa-simdr', thr=0.5)

        acc_cnt_all += cnt
        acc_final += avg_acc * cnt

        acc, avg_acc, cnt, pred = accuracy(decode_batch_pred3, decode_batch_label3, hm_type='sa-simdr', thr=0.5)

        acc_cnt_all += cnt
        acc_final += avg_acc * cnt

        pbar.update(1)
    pbar.close()

print('\n')
print("Model Name:{}".format(args.model))
print("Accuracy: {:.3f}".format(acc_final / acc_cnt_all))
print("Mean MPJPE2D cam2: {:.3f}".format(np.array(mpjpe2D2_all).mean()))
print("Mean MPJPE2D cam3: {:.3f}".format(np.array(mpjpe2D3_all).mean()))
print("Mean MPJPE2D cam2 and cam3: {:.3f}".format(np.array(mpjpe2D_all).mean()))
print("Mean MPJPE3D: {:.3f}".format(np.array(mpjpe3D_all).mean()))
