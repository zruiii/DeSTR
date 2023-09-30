#!/usr/bin/env python3
# coding=utf-8
########################################################################
# Copyright (C) Baidu Ltd. All rights reserved. 
########################################################################

"""
Author: zharui
Date: 2023-06-20 10:49:19
LastEditTime: 2023-09-28 09:46:04
LastEditors: zharui@baidu.com
FilePath: /decoupledST/src/trainer/forecast_dataloader.py
Description: 
"""

import pdb
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class Forecast_Dataset(Dataset):
    """ 数据加载器, 适用于多城市或者单城市 """
    def __init__(self, data, mode="train"):
        super().__init__()
        self.mode = mode

        self.look_back = data.look_back
        self.pred_len = data.pred_len

        # signal
        set_type = {"train": 0, "val": 1, "test": 2}
        border_start = data.start_points[set_type[mode]]
        border_end = data.end_points[set_type[mode]]

        self.flow_mask = data.flow_mask[border_start:border_end]                # 有效信号
        self.arr_flow = data.arr_flow[border_start:border_end]                  # 归一化后的信号
        self.org_flow = data.org_flow[border_start:border_end]                  # 原始信号
        self.scaler = data.scaler

        # time
        self.time_feat = data.time_feat[border_start:border_end]
        
        # road network
        self.g = data.g
        self.adj = data.adj
        self.nids = data.nids
        self.nfeat = data.nfeat
    
    
    def __len__(self):
        return len(self.arr_flow) - self.look_back - self.pred_len + 1

    def __getitem__(self, index):
        """ 返回元组 (x, y, t, m)
        xt: [B, N, L], 历史窗口内流量记录
        xg: [B, N, H], 流向的空间特征
        t: [B, L, C], 历史窗口内每个时间戳特征
        y: [B, N, P], 预测窗口内流量记录
        m: [B, N, P], 预测窗口内掩码
        """
        xt = self.arr_flow[index:index + self.look_back].copy()                         # [L, N]
        xt = torch.from_numpy(xt.T)
        xg = torch.from_numpy(self.nfeat.copy())                                        # [N, H]
        t = torch.from_numpy(self.time_feat[index:index + self.look_back])              # [B, L, C]

        if self.mode != "train":
            y = self.org_flow[index + self.look_back:index + self.look_back + self.pred_len]  # [P, N]
        else:
            y = self.arr_flow[index + self.look_back:index + self.look_back + self.pred_len]  # [P, N]
        m = self.flow_mask[index + self.look_back:index + self.look_back + self.pred_len]

        y = torch.from_numpy(y.T)
        m = torch.from_numpy(m.T)
        
        return xt, xg, t, y, m


class SimpleForecastBatch:
    def __init__(self, data):
        xt, xg, t, y, m = zip(*data)
        self.xt = torch.stack(xt, dim=0)       # [B, N, L]  输入信号
        self.xg = torch.stack(xg, dim=0)       # [B, N, H]  路网节点特征
        self.t = torch.stack(t, dim=0)         # [B, L, C]  时间戳特征
        self.y = torch.stack(y, dim=0)         # [B, N, L'] 未来信号标签
        self.m = torch.stack(m, dim=0)         # [B, N, L'] mask掉标签中的缺失值


def collate_fn(batch):
    return SimpleForecastBatch(batch)

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
        "yizhuang": {
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
    for mode in ["test"]:
        print(mode)
        miss_per_batch_x, miss_per_batch_y = [], []
        train_data = Forecast_Dataset(data, mode=mode)
        pdb.set_trace()
        dataloader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_fn)
        print("整体缺失率: {:.2f}%".format((~train_data.flow_mask).sum() / np.prod(train_data.flow_mask.shape) * 100))
        
        for idx, batch in enumerate(dataloader):
            pdb.set_trace()

        #     miss_per_batch_x.append(torch.isnan(batch[0]).sum() / np.prod(batch[0].shape))
        #     miss_per_batch_y.append(torch.isnan(batch[2]).sum() / np.prod(batch[2].shape))
        #     # pdb.set_trace()
        
        # print("平均每个 batch 输入缺失率为: {:.2f}%; 标签缺失率为: {:.2f}%".format(np.mean(miss_per_batch_x) * 100,
        #                                                             np.mean(miss_per_batch_y) * 100))
    
