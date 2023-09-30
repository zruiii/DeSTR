#!/usr/bin/env python3
# coding=utf-8
########################################################################
# Copyright (C) Baidu Ltd. All rights reserved. 
########################################################################

"""
Author: zharui
Date: 2023-07-11 12:37:12
LastEditTime: 2023-08-10 13:59:14
LastEditors: zharui@baidu.com
FilePath: /decoupledST/src/utils/STGODE_utils.py
Description: 
"""
import csv
import os
import numpy as np
from fastdtw import fastdtw
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import pdb
import torch
import pandas as pd

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

    K, L, N = train_data.shape
    folds = int(288 / L)
    train_data = train_data[:int(K / folds) * folds]
    train_data = train_data.reshape(int(K / folds), folds, L, N)
    train_data = train_data.reshape(int(K / folds), -1, N)

    train_data = train_data.reshape(-1, N)
    train_data = dataset.scaler.inverse_transform(train_data)
    
    # train_data = train_data.reshape(-1, 288, N)
    train_data[np.isnan(train_data)] = 0
    train_data = train_data.reshape(-1, num_node, 1)

    # data: (K, N, C) 样本: 节点: 特征  

    ### DTW Distance
    if not os.path.exists(os.path.join(args.data_path, args.city, "dtw_distance.npy")):
        data_mean = np.mean([train_data[:, :, 0][24*12*i: 24*12*(i+1)] for i in range(train_data.shape[0]//(24*12))], axis=0)
        data_mean = data_mean.squeeze().T 
        dtw_distance = np.zeros((num_node, num_node))
        for i in tqdm(range(num_node)):
            for j in range(i, num_node):
                dtw_distance[i][j] = fastdtw(data_mean[i], data_mean[j], radius=6)[0]
        for i in range(num_node):
            for j in range(i):
                dtw_distance[i][j] = dtw_distance[j][i]
    
        mean = np.mean(dtw_distance)
        std = np.std(dtw_distance)
        dtw_distance = (dtw_distance - mean) / std
        
        sigma = 0.1
        dtw_distance = np.exp(-dtw_distance ** 2 / sigma ** 2)
        dtw_matrix = np.zeros_like(dtw_distance)
        dtw_matrix[dtw_distance > 0.6] = 1

        np.save(os.path.join(args.data_path, args.city, "dtw_distance.npy"), dtw_matrix)
        print(f'average degree of semantic graph is {np.sum(dtw_matrix > 0)/2/num_node}')
