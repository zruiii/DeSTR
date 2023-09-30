#!/usr/bin/env python3
# coding=utf-8
########################################################################
# Copyright (C) Baidu Ltd. All rights reserved. 
########################################################################

"""
Author: zharui
Date: 2023-08-22 10:39:09
LastEditTime: 2023-09-02 16:48:23
LastEditors: zharui@baidu.com
FilePath: /decoupledST/src/trainer/imputation_dataloader.py
Description: 
For the imputation task, we random mask some records for the whole validation & test dataset to calculate the metrics.
and the window is sliding by the step of window size.
In the training process, we apply random masking on each batch of training data for masked imputation loss.
and the window is sliding by the step of 1.
"""

import pdb
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class Imputation_Dataset(Dataset):
    """ 数据加载器, 适用于多城市或者单城市 """
    def __init__(self, data, mode="train", mask_ratio=0.3):
        super().__init__()
        self.mode = mode

        self.look_back = data.look_back
        self.pred_len = data.pred_len

        # signal
        set_type = {"train": 0, "val": 1, "test": 2}
        border_start = data.start_points[set_type[mode]]
        border_end = data.end_points[set_type[mode]]

        self.flow_mask = data.flow_mask[border_start:border_end]                # 有效信号为True
        self.arr_flow = data.arr_flow[border_start:border_end]                  # 归一化后的信号
        self.org_flow = data.org_flow[border_start:border_end]                  # 原始信号
        self.scaler = data.scaler

        # NOTE: 仅用于测试
        # add mask for the whole data part (except missing values) to calculate metrics
        rng = np.random.RandomState(seed=2023)
        rand_idx = rng.randn(*self.flow_mask.shape)
        quantile = np.quantile(rand_idx[~self.flow_mask], mask_ratio)
        rand_idx[~self.flow_mask] = 1
        self.imp_mask = rand_idx < quantile
        
        # time
        self.time_feat = data.time_feat[border_start:border_end]
        
        # road network
        self.g = data.g
        self.adj = data.adj
        self.nids = data.nids
        self.nfeat = data.nfeat
    

    def __len__(self):
        if self.mode == "train":                                    # stride=1 滑移
            return len(self.arr_flow) - self.look_back + 1
        else:                                                       # stride=window 滑移
            return len(self.arr_flow) // self.look_back


    def __getitem__(self, index):
        """ 返回元组 (xt, xg, t, org, mask)
        xt: [B, N, L], 历史窗口内流量记录
        xg: [B, N, H], 流向的空间特征
        t:  [B, L, C], 历史窗口内每个时间戳特征
        org:  [B, N, L], 原始输入信号(No Scale & Mask)
        mask: [B, N, L], 用于补全的掩码
        """
        if self.mode != "train":
            index = index * self.look_back

        xt = self.arr_flow[index: index + self.look_back].copy()                       
        xt = torch.from_numpy(xt.T)
        xg = torch.from_numpy(self.nfeat.copy())                                      
        t = torch.from_numpy(self.time_feat[index:index + self.look_back])            
        org = self.org_flow[index: index + self.look_back]                            
        mask = self.imp_mask[index: index + self.look_back]                             

        org = torch.from_numpy(org.T)
        mask = torch.from_numpy(mask.T)
        
        return xt, xg, t, org, mask


class SimpleImputeBatch:
    def __init__(self, data):
        xt, xg, t, org, mask = zip(*data)
        self.xt = torch.stack(xt, dim=0)                     # [B, N, L]  归一化后的输入信号 (No Mask)
        self.xg = torch.stack(xg, dim=0)                     # [B, N, H]  路网节点特征
        self.t = torch.stack(t, dim=0)                       # [B, L, C]  时间戳特征
        self.org = torch.stack(org, dim=0)                   # [B, N, L]  原始输入信号(No Scale & Mask)
        self.mask = torch.stack(mask, dim=0)                 # [B, N, L]  Mask (loss for train; metrics for eval)

    def random_mask(self, mask_ratio):
        """ 针对单个batch数据进行掩码, 仅在训练阶段用于计算 Masked Imputation Loss """
        pre_mask = torch.isnan(self.xt)
        rand_idx = torch.randn_like(self.xt)
        rand_idx[pre_mask] = 1
        threshold = torch.quantile(rand_idx, mask_ratio)  # 计算百分位数

        mask = rand_idx < threshold
        mask = mask & ~pre_mask
        self.mask = mask


def collate_wrapper(mode="forecast", mask_ratio=0.3):
    if mode == "forecast":
        def collate_fn(batch):
            return SimpleImputeBatch(batch)
    else:
        def collate_fn(batch):
            batch = SimpleImputeBatch(batch)
            batch.random_mask(mask_ratio)
            return batch
    return collate_fn



if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from utils.data_utils import Multi_Data_Container, Single_Data_Container
    
    # Multi Data
    # data_dict = {
    #     "yizhuang": {
    #         "flow": "2023_01_03_flow.csv",
    #         "rel": "connect.csv",
    #         "nfeat": "node_feature.csv"
    #     },
    #     "guangzhou": {
    #         "flow": "2023_01_03_flow.csv",
    #         "rel": "connect.csv",
    #         "nfeat": "node_feature.csv"
    #     },
    #     "zhuzhou": {
    #         "flow": "2023_01_03_flow.csv",
    #         "rel": "connect.csv",
    #         "nfeat": "node_feature.csv"
    #     }
    # }

    # data = Multi_Data_Container(data_path="/home/users/zharui/decoupledST/data",
    #                             data_dict=data_dict,
    #                             look_back=24,
    #                             pred_len=12)
    
    # Single Data
    data_dict = {
        "guangzhou": {
            "flow": "2023_01_03_flow.csv",
            "rel": "connect.csv",
            "dtw": "dtw_distance.npy",
            "nfeat": "node_feature.csv"
        }
    }

    data = Single_Data_Container(data_path="/home/users/zharui/decoupledST/data",
                                data_dict=data_dict,
                                look_back=24,
                                pred_len=12)
    print(data.g.num_nodes(), data.g.num_edges())
    for mode in ["val", "train", "test"]:
        print(mode)
        miss_per_batch_x, miss_per_batch_y = [], []
        train_data = Imputation_Dataset(data, mode=mode)
        collate_fn = collate_wrapper(mode="imputation", mask_ratio=0.3)
        dataloader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_fn)
        print("整体缺失率: {:.2f}%".format((~train_data.flow_mask).sum() / np.prod(train_data.flow_mask.shape) * 100))
        
        for idx, batch in enumerate(dataloader):
            pdb.set_trace()

        #     miss_per_batch_x.append(torch.isnan(batch[0]).sum() / np.prod(batch[0].shape))
        #     miss_per_batch_y.append(torch.isnan(batch[2]).sum() / np.prod(batch[2].shape))
        #     # pdb.set_trace()
        
        # print("平均每个 batch 输入缺失率为: {:.2f}%; 标签缺失率为: {:.2f}%".format(np.mean(miss_per_batch_x) * 100,
        #                                                             np.mean(miss_per_batch_y) * 100))
    
