#!/usr/bin/env python3
# coding=utf-8
########################################################################
# Copyright (C) Baidu Ltd. All rights reserved. 
########################################################################

"""
Author: zharui
Date: 2023-07-05 01:19:10
LastEditTime: 2023-08-10 13:58:09
LastEditors: zharui@baidu.com
FilePath: /decoupledST/src/utils/DSTAGNN_utils.py
Description: 
"""
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch
import time
import argparse
from scipy.optimize import linprog
import multiprocessing as mp


import pdb
from tqdm import tqdm
np.seterr(divide='ignore', invalid='ignore')

def wasserstein_distance(p, q, D):
    A_eq = []
    for i in range(len(p)):
        A = np.zeros_like(D)
        A[i, :] = 1
        A_eq.append(A.reshape(-1))
    for i in range(len(q)):
        A = np.zeros_like(D)
        A[:, i] = 1
        A_eq.append(A.reshape(-1))
    A_eq = np.array(A_eq)
    b_eq = np.concatenate([p, q])
    D = np.array(D)
    D = D.reshape(-1)

    result = linprog(D, A_eq=A_eq[:-1], b_eq=b_eq[:-1])
    myresult = result.fun

    return myresult


def spatial_temporal_aware_distance(x, y):
    """ 每个样本取 L2 范数变成标量, 然后计算 x 和 y 样本分布的余弦距离
    x, y: [K, T] K: 样本数目; T: 时间长度
    """
    x, y = np.array(x), np.array(y)
    x_norm = (x**2).sum(axis=1, keepdims=True)**0.5 + 1e-10
    y_norm = (y**2).sum(axis=1, keepdims=True)**0.5 + 1e-10
    p = x_norm[:, 0] / x_norm.sum()
    q = y_norm[:, 0] / y_norm.sum()
    D = 1 - np.dot(x / x_norm, (y / y_norm).T)
    if np.isnan(D).sum() > 0:
        pdb.set_trace()
    return wasserstein_distance(p, q, D)


def spatial_temporal_similarity(x, y, normal, transpose):
    if normal:
        x = normalize(x)
        x = normalize(x)
    if transpose:
        x = np.transpose(x)
        y = np.transpose(y)
    
    try:
        return 1 - spatial_temporal_aware_distance(x, y)
    except:
        return 1e-4


def gen_data(data, ntr, N):
    '''
    if flag:
        data=pd.read_csv(fname)
    else:
        data=pd.read_csv(fname,header=None)
    '''
    data = np.reshape(data, [-1, 288, N])
    return data[0:ntr]


def normalize(a):
    mu = np.mean(a, axis=1, keepdims=True)
    std = np.std(a, axis=1, keepdims=True)
    return (a - mu) / std

import sys
sys.path.append("..")

from trainer.dataloader import Dataset_Traffic
from utils.data_utils import Single_Data_Container

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="yizhuang", type=str, help="specific fine-tune dataset")
    parser.add_argument("--data_path", default="../../data", type=str)
    parser.add_argument("--flow_file", default="2023_01_03_flow.csv", type=str)
    parser.add_argument("--rel_file", default="connect.csv", type=str)
    parser.add_argument("--nfeat_file", default="node_feature.csv", type=str)
    parser.add_argument("--look_back", default=24, type=int)
    parser.add_argument("--pred_len", default=12, type=int)
    parser.add_argument("--time_dim", default=4, type=int)
    parser.add_argument("--time_encode", default="raw", type=str)
    args = parser.parse_args()
    return args


def get_data(args):
    """ 单个城市数据加载器 """
    data_dict = {
        args.city: {
            "flow": args.flow_file,
            "rel": args.rel_file,
            "nfeat": args.nfeat_file
        }
    }

    data = Single_Data_Container(data_path=args.data_path,
                                 data_dict=data_dict,
                                 look_back=args.look_back,
                                 pred_len=args.pred_len,
                                 time_encode=args.time_encode)

    train_data = Dataset_Traffic(data, mode="train")

    res = []
    for idx in tqdm(range(train_data.__len__()), desc="collect training data"):
        if idx % args.look_back == 0:
            x, _, _, _, _ = train_data.__getitem__(idx)
            res.append(x.T)
    
    res = np.array(res)     # [K, T, N]
    return res, train_data


if __name__ == "__main__":
    import os
    import sys
    args = get_args()

    train_data, dataset = get_data(args)
    num_node = len(dataset.nids)
    print(num_node)
    
    # chose continue but disjoint samples to compute the STAG proposed in DSTAGNN.
    K, L, N = train_data.shape
    folds = int(288 / L)
    train_data = train_data[:int(K / folds) * folds]
    train_data = train_data.reshape(int(K / folds), folds, L, N)
    train_data = train_data.reshape(int(K / folds), -1, N)

    train_data = train_data.reshape(-1, N)
    train_data = dataset.scaler.inverse_transform(train_data)
    train_data = train_data.reshape(-1, 288, N)
    train_data[np.isnan(train_data)] = 0

    def calculate_similarity(args):
        """
        i, j: index
        data_i, data_j: [K, T]   
        """
        i, j, data_i, data_j = args
        similarity = spatial_temporal_similarity(data_i, data_j, normal=False, transpose=False)
        return (i, j, similarity)

    def update(*a):
        pbar.update()

    # 多进程加速相似度计算
    if not os.path.exists(os.path.join(args.data_path, args.city, f"stag.npy")):
        pool = mp.Pool(8)
        d = np.zeros([num_node, num_node])
        args_list = [(i, j, train_data[:, :, i], train_data[:, :, j]) for i in range(num_node) for j in range(i + 1, num_node)]
        pbar = tqdm(total=len(args_list))
        for result in pool.imap_unordered(calculate_similarity, args_list):
            d[result[0], result[1]] = result[2]
            pbar.update()
        pool.close()
        pbar.close()

        sta = d + d.T
        np.save(os.path.join(args.data_path, args.city, f"stag.npy"), sta)
        print("The calculation of time series is done!")

    adj = np.load(os.path.join(args.data_path, args.city, f"stag.npy"))     # 基于序列相似度计算的邻接矩阵

    A_adj = np.zeros([num_node, num_node])
    R_adj = np.zeros([num_node, num_node])

    adj_percent = 0.01
    top = int(num_node * adj_percent)

    # A_adj 只保留 top-1% weight 的邻居; R_adj 只保留 top normalized weight 的邻居
    for i in range(adj.shape[0]):
        a = adj[i, :].argsort()[0:top]
        for j in range(top):
            A_adj[i, a[j]] = 1
            R_adj[i, a[j]] = adj[i, a[j]]

    pdb.set_trace()

    print("Total route number: ", num_node)
    print("Sparsity of adj: ", len(A_adj.nonzero()[0]) / (num_node * num_node))

    pd.DataFrame(A_adj).to_csv(os.path.join(args.data_path, args.city, f"stag.csv"), index=False, header=None)
    pd.DataFrame(R_adj).to_csv(os.path.join(args.data_path, args.city, f"strg.csv"), index=False, header=None)

    print("The weighted matrix of temporal graph is generated!")

