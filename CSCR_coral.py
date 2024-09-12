# -*- coding: utf-8 -*-

from __future__ import print_function, division
import os
import time

import argparse
import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CosineAnnealingLR
from torch.utils.data import DataLoader
from model import Net
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize, scale
import scipy.io
import h5py
import math
import copy
import random
from loss import CLoss

from measure import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def label_sim(inc_label):
    sim = np.matmul(inc_label, inc_label.T)
    label_cout = np.sum(inc_label, axis=1)
    sim_new = np.zeros_like(sim)
    np.seterr(divide='ignore', invalid='ignore')
    for i, j in enumerate(sim):
        sim_new[i] = 2 * sim[i] / (label_cout[i] + label_cout)
    np.fill_diagonal(sim_new, 1)
    sim_new = np.nan_to_num(sim_new)
    return sim_new


def class_rec_sample(Xv, sim, ind1, Wv):
    # nonzero_before = np.count_nonzero(Wv)
    # print(f"Exist sample in this view before rec:{nonzero_before}")
    WRv = np.zeros_like(Wv).astype(float)
    XRv = np.copy(Xv)
    np.fill_diagonal(sim, 0)
    nonzero_info = []
    # building similar label lists
    for row in sim:
        nonzero_indices = np.nonzero(row)[0]
        nonzero_elements = row[nonzero_indices]
        nonzero_count = len(nonzero_indices)
        nonzero_info.append((nonzero_indices, nonzero_count, nonzero_elements))
    for i, (indices, count, elements) in enumerate(nonzero_info):
        if Wv[i] == 0:  # missing sample
            if count == 0:  # missing label
                WRv[i] = 0
            else:  # existing label
                intersection_i = np.intersect1d(indices, ind1)
                if len(intersection_i) == 0:  # all sim label sample missing
                    WRv[i] = 0
                    continue
                indices_vector = np.where(np.isin(indices, intersection_i))[0]
                result_vector = elements[indices_vector]
                sum_weight = np.sum(result_vector)
                XRv[i] = np.sum(result_vector[:, np.newaxis] * Xv[intersection_i], axis=0) / sum_weight
                WRv[i] = sum_weight / len(indices_vector)
                # if math.isnan(WRv[i]):
                #     print("NaN")
        else:  # existing sample
            WRv[i] = 1
    # nonzero_after = np.count_nonzero(WRv)
    # print(f"Exist sample in this view:{nonzero_after}")

    return XRv, WRv


def wmse_loss(input, target, weight, reduction='mean'):
    ret = (torch.diag(weight).mm(target - input)) ** 2
    ret = torch.mean(ret)
    return ret


def do_metric(y_prob, label):
    y_predict = y_prob > 0.5
    ranking_loss = 1 - compute_ranking_loss(y_prob, label)
    # print(ranking_loss)
    one_error = compute_one_error(y_prob, label)
    # print(one_error)
    coverage = compute_coverage(y_prob, label)
    # print(coverage)
    hamming_loss = 1 - compute_hamming_loss(y_predict, label)
    # print(hamming_loss)
    precision = compute_average_precision(y_prob, label)
    # print(precision)
    macro_f1 = compute_macro_f1(y_predict, label)
    # print(macro_f1)
    micro_f1 = compute_micro_f1(y_predict, label)
    # print(micro_f1)
    auc = compute_auc(y_prob, label)
    auc_me = mlc_auc(y_prob, label)
    return np.array([hamming_loss, one_error, coverage, ranking_loss, precision, auc, auc_me, macro_f1, micro_f1])


def train(mul_X, mul_X_val, WE, WE_val, WR, yv_label, sim_label, device, args):
    # return None, torch.randn(9, 1)
    model = Net(
        n_stacks=4,
        n_input=args.n_input,
        n_z=args.Nlabel,
        Nlabel=args.Nlabel).to(device)
    loss_model = CLoss(args.alpha, args.normalize_loss, args.class_num, device)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Module):
            for mm in m.modules():
                if isinstance(mm, nn.Linear):
                    nn.init.xavier_uniform_(mm.weight)
                    nn.init.constant_(mm.bias, 0.0)
    num_X = mul_X[0].shape[0]
    num_X_val = mul_X_val[0].shape[0]
    print(f"All_train_num: {num_X}, val_num: {num_X_val}")
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    total_loss = 0
    ytest_Lab = np.zeros([mul_X_val[0].shape[0], args.Nlabel])
    ap_loss = []
    best_value_result = [0] * 10
    best_value_epoch = 0
    best_train_model = copy.deepcopy(model)
    for epoch in range(int(args.maxiter)):
        start_time = time.time()
        model.train()
        total_loss_last = total_loss
        total_loss = 0
        ytest_Lab_last = np.copy(ytest_Lab)
        index_array = np.arange(num_X)
        if args.AE_shuffle:
            np.random.shuffle(index_array)
        for batch_idx in range(int(np.ceil(num_X / args.batch_size))):
            idx = index_array[batch_idx * args.batch_size: min((batch_idx + 1) * args.batch_size, num_X)]
            mul_X_batch = []
            for iv, X in enumerate(mul_X):
                mul_X_batch.append(X[idx].to(device))

            wr = WR[idx].to(device)
            we = WE[idx].to(device)
            sub_target = Inc_label[idx].to(device)
            fan_sub_target = fan_Inc_label[idx].to(device)
            sub_obrT = obrT[idx].to(device)
            sub_sim_label = sim_label[idx]
            sub_sim_label = sub_sim_label[:, idx].to(device)
            optimizer.zero_grad()

            z_list, x_rec_list, y_list = model(mul_X_batch)

            loss_rec = 0
            for i, x_rec in enumerate(x_rec_list):
                loss_rec += wmse_loss(x_rec, mul_X_batch[i], wr[:, i])

            loss_con = 0
            for i in range(len(z_list)):
                for j in range(i + 1, len(z_list)):
                    loss_con += loss_model.semi_contrast_loss(z_list[i], z_list[j], wr[:, i], wr[:, j], sub_sim_label)

            loss_cla = 0
            for i in range(len(y_list)):
                loss_cla += torch.mean(torch.abs((sub_target.mul(torch.log(y_list[i] + 1e-10)) + fan_sub_target.mul(
                    torch.log(1 - y_list[i] + 1e-10))).mul(sub_obrT)) * wr[:, i][:, np.newaxis])

            fusion_loss = loss_cla + args.gamma * loss_rec + loss_con * args.beta

            total_loss += fusion_loss.item()
            fusion_loss.backward()
            optimizer.step()
        yp_prob = test(model, mul_X_val, WE_val, args, device)

        value_result = do_metric(yp_prob, yv_label)
        ap_loss.append([value_result[4], total_loss])
        total_loss = total_loss / (batch_idx + 1)
        spend_time = time.time() - start_time
        print("semi_epoch {} time={:.4f} loss={:.4f} hamming loss={:.4f} AP={:.4f} AUC={:.4f} auc_me={:.4f}"
              .format(epoch, spend_time, total_loss, value_result[0], value_result[4], value_result[5],
                      value_result[6]))

        if best_value_result[4] < value_result[4]:
            best_value_result = value_result
            best_train_model = copy.deepcopy(model)
            best_value_epoch = epoch

            # torch.save(model)
        ytest_Lab = yp_prob > 0.5
        del yp_prob
        delta_y = np.sum(ytest_Lab != ytest_Lab_last).astype(np.float32) / ytest_Lab.shape[0] / ytest_Lab.shape[1]
        if epoch > args.miniter and ((best_value_result[4] - value_result[4] > 0.03) or
                                     best_value_result[4] < args.min_AP or (
                                             abs(total_loss_last - total_loss) < 1e-5 or delta_y < args.tol)):
            print('Training stopped: epoch=%d, best_epoch=%d, best_AP=%.7f, min_AP=%.7f,total_loss=%.7f' % (
                epoch, best_value_epoch, best_value_result[4], args.min_AP, total_loss))
            break

    return best_train_model, best_value_result, ap_loss


def test(model, mul_X_test, WE_test, args, device):
    model.eval()
    num_X_test = mul_X_test[0].shape[0]
    tmp_q = torch.zeros([num_X_test, args.Nlabel]).to(device)
    index_array_test = np.arange(num_X_test)
    for batch_idx in range(int(np.ceil(num_X_test / args.batch_size))):
        idx = index_array_test[batch_idx * args.batch_size: min((batch_idx + 1) * args.batch_size, num_X_test)]
        mul_X_test_batch = []
        for iv, X in enumerate(mul_X_test):
            mul_X_test_batch.append(X[idx].to(device))

        we = WE_test[idx].to(device)
        _, _, y_list = model(mul_X_test_batch)
        y_temp = torch.zeros(size=(we.shape[0], tmp_q.shape[1])).to(device)
        for i, y_v in enumerate(y_list):
            y_temp += torch.diag(we[:, i]).mm(y_v) / torch.sum(we, dim=1)[:, None]
        tmp_q[idx] = y_temp
        del y_list

    yy_pred = tmp_q.data.cpu().numpy()
    yy_pred = np.nan_to_num(yy_pred)
    return yy_pred


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Adjust #
    parser.add_argument('--lr', type=float, default=0.1)        # 0.1
    parser.add_argument('--momentum', type=float, default=0.9)  # 0.9
    parser.add_argument('--gamma', type=float, default=1)       # 1
    parser.add_argument('--alpha', type=float, default=0.1)     # 0.1
    parser.add_argument('--beta', type=float, default=0.001)    # 0.001
    parser.add_argument('--maxiter', default=500, type=int)     # 500
    parser.add_argument('--miniter', default=200, type=int)     # 200
    parser.add_argument('--batch_size', default=128, type=int)  # 128
    parser.add_argument('--weight_decay', type=float, default=0.0001)  # 0.0001
    # Dataset #
    parser.add_argument('--dataset', type=str, default='corel5k_six_view')
    parser.add_argument('--dataPath', type=str, default='data/corel5k')
    parser.add_argument('--MaskRatios', type=float, default=0.5)
    parser.add_argument('--LabelMaskRatio', type=float, default=0.5)
    parser.add_argument('--TraindataRatio', type=float, default=0.7)  
    # Train #
    parser.add_argument('--preNum', type=int, default=10)
    parser.add_argument('--fnum', type=int, default=0)          # [0-9]
    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument('--AE_shuffle', type=bool, default=True)
    parser.add_argument('--normalize_loss', type=bool, default=True)
    parser.add_argument('--min_AP', default=0., type=float)
    parser.add_argument('--tol', default=1e-7, type=float)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    args.cuda = torch.cuda.is_available()
    # args.cuda = torch.has_mps
    print("use cuda: {}".format(args.cuda))
    print("Data loading...", end='')
    file_path = 'CSCR_' + args.dataset + '_BS_' + str(args.batch_size) + '_VMR_' + str(
        args.MaskRatios) + '_LMR_' + str(args.LabelMaskRatio) + '_TR_' + str(
        args.TraindataRatio) + '-best_AP' + '.txt'

    device = torch.device("cuda" if args.cuda else "cpu")
    best_AUC_me = 0
    best_AUC_mac = 0
    best_AP = 0

    data = scipy.io.loadmat(args.dataPath + '/' + args.dataset + '.mat')

    X = data['X'][0]

    view_num = X.shape[0]
    label = data['label']
    label = np.array(label, 'float32')
    args.class_num = label.shape[1]
    # print(args)

    hm_loss = np.zeros(args.preNum)
    one_error = np.zeros(args.preNum)
    coverage = np.zeros(args.preNum)
    rk_loss = np.zeros(args.preNum)
    AP_score = np.zeros(args.preNum)

    mac_auc = np.zeros(args.preNum)
    auc_me = np.zeros(args.preNum)
    mac_f1 = np.zeros(args.preNum)
    mic_f1 = np.zeros(args.preNum)
    print("Complete")
    print(
        f"Dataset: {args.dataset}, Number of instance: {label.shape[0]}, Number of class: {label.shape[1]}, Number of view: {len(X)}")
    for fnum in range(args.preNum):
        print("Label loading...", end='')
        start_label_loading = time.time()
        mul_X = [None] * view_num

        datafold = scipy.io.loadmat(args.dataPath + '/' + args.dataset + '_MaskRatios_' + str(
            args.MaskRatios) + '_LabelMaskRatio_' +
                                    str(args.LabelMaskRatio) + '_TraindataRatio_' + str(
            args.TraindataRatio) + '.mat')
        folds_data = datafold['folds_data']
        folds_label = datafold['folds_label']
        folds_sample_index = datafold['folds_sample_index']
        del datafold
        Ndata, args.Nlabel = label.shape
        # training data, val data and test data
        indexperm = np.array(folds_sample_index[0, fnum], 'int32')
        train_num = math.ceil(Ndata * args.TraindataRatio)
        val_test_num = Ndata - train_num
        val_num = math.ceil(val_test_num * 0.5)
        # val_test_index = indexperm[0, train_num:indexperm.shape[1]] - 1
        val_index = indexperm[0, train_num:train_num + val_num] - 1
        test_index = indexperm[0, train_num + val_num:indexperm.shape[1]] - 1
        # incomplete data index
        WE = np.array(folds_data[0, fnum], 'int32')
        # incomplete label construction
        # the label of val and test sample is all zero, the label of train sample is partial zero
        obrT = np.array(folds_label[0, fnum], 'int32')  # incomplete label index
        zero_rows = np.where(np.all(obrT == 0, axis=1))[0]
        if len(zero_rows) != val_test_num:
            raise ValueError("Dataset error.")
        if label.min() == -1:
            label = (label + 1) * 0.5
        Inc_label = label * obrT  # incomplete label matrix
        # sim_label = cosine_similarities_label(Inc_label)
        sim_label = label_sim(Inc_label)
        fan_Inc_label = 1 - Inc_label
        # incomplete data construction
        WR = np.zeros_like(WE).astype(float)
        label_loading_time = time.time() - start_label_loading
        print(f"Complete, time:{round(label_loading_time, 2)}")
        print(f"Labeled_train_num: {train_num}, Val_num: {len(val_index)}, Test_num: {len(test_index)}")
        for iv in range(view_num):
            start_missing_construction = time.time()
            print(f"Missing view-{iv} construction...", end='')
            mul_X[iv] = np.copy(X[iv])
            mul_X[iv] = mul_X[iv].astype(np.float32)
            WEiv = WE[:, iv]
            ind_1 = np.where(WEiv == 1)
            ind_1 = (np.array(ind_1)).reshape(-1)
            ind_0 = np.where(WEiv == 0)
            ind_0 = (np.array(ind_0)).reshape(-1)
            # print(f"Exist sample in view-{iv}:{np.count_nonzero(WEiv)}")
            mul_X[iv][ind_1, :] = StandardScaler().fit_transform(mul_X[iv][ind_1, :])
            mul_X[iv][ind_0, :] = 0
            mul_X[iv], WR[:, iv] = class_rec_sample(mul_X[iv], sim_label, ind_1, WEiv)
            clum = abs(mul_X[iv]).sum(0)
            ind_11 = np.array(np.where(clum != 0)).reshape(-1)
            new_X = np.copy(mul_X[iv][:, ind_11])
            mul_X[iv] = torch.Tensor(np.nan_to_num(np.copy(new_X)))
            missing_construction_time = time.time() - start_missing_construction
            print(f"Complete, time:{round(missing_construction_time, 2)}")
            del new_X, ind_0, ind_1, ind_11, clum

        WE = torch.Tensor(WE)
        WR = torch.Tensor(WR)

        mul_X_val = [xiv[val_index] for xiv in mul_X]
        mul_X_test = [xiv[test_index] for xiv in mul_X]
        WE_val = WE[val_index]
        WE_test = WE[test_index]
        obrT = torch.Tensor(obrT)
        Inc_label = torch.Tensor(Inc_label)
        sim_label = torch.Tensor(sim_label)
        fan_Inc_label = torch.Tensor(fan_Inc_label)
        args.n_input = [xiv.shape[1] for xiv in mul_X]
        yv_label = np.copy(label[val_index])
        yt_label = np.copy(label[test_index])

        model, _, ap_loss = train(mul_X, mul_X_val, WE, WE_val, WR, yv_label, sim_label, device, args)
        yp_prob = test(model, mul_X_test, WE_test, args, device)
        value_result = do_metric(yp_prob, yt_label)

        print(
            "final:hamming-loss" + ' ' + "one-error" + ' ' + "coverage" + ' ' + "ranking-loss" + ' ' + "average"
                                                                                                       "-precision" +
            ' ' + "macro-auc" + ' ' + "auc_me" + ' ' + "macro_f1" + ' ' + "micro_f1")
        print(value_result)

        hm_loss[fnum] = value_result[0]
        one_error[fnum] = value_result[1]
        coverage[fnum] = value_result[2]
        rk_loss[fnum] = value_result[3]
        AP_score[fnum] = value_result[4]
        mac_auc[fnum] = value_result[5]
        auc_me[fnum] = value_result[6]
        mac_f1[fnum] = value_result[7]
        mic_f1[fnum] = value_result[8]
    if AP_score.mean() > best_AP:
        best_AP = AP_score.mean()

    file_handle = open(file_path, mode='a')
    if os.path.getsize(file_path) == 0:
        file_handle.write(
            'mean_AP std_AP mean_hamming_loss std_hamming_loss mean_one_error std_one_error mean_coverage '
            'std_coverage mean_ranking_loss std_ranking_loss mean_AUC std_AUC mean_AUCme std_AUCme mean_macro_f1 '
            'std_macro_f1 mean_micro_f1 std_micro_f1 lrkl momentumKL alphakl betakl gammakl\n')

    file_handle.write(str(AP_score.mean()) + ' ' +
                      str(AP_score.std()) + ' ' +
                      str(hm_loss.mean()) + ' ' +
                      str(hm_loss.std()) + ' ' +
                      str(one_error.mean()) + ' ' +
                      str(one_error.std()) + ' ' +
                      str(coverage.mean()) + ' ' +
                      str(coverage.std()) + ' ' +
                      str(rk_loss.mean()) + ' ' +
                      str(rk_loss.std()) + ' ' +
                      str(mac_auc.mean()) + ' ' +
                      str(mac_auc.std()) + ' ' +
                      str(auc_me.mean()) + ' ' +
                      str(auc_me.std()) + ' ' +
                      str(mac_f1.mean()) + ' ' +
                      str(mac_f1.std()) + ' ' +
                      str(mic_f1.mean()) + ' ' +
                      str(mic_f1.std()) + ' ' +
                      str(args.lr) + ' ' +
                      str(args.momentum) + ' ' +
                      str(args.alpha) + ' ' +
                      str(args.beta) + ' ' +
                      str(args.gamma)
                      )
    
    file_handle.write('\n')
    file_handle.close()
