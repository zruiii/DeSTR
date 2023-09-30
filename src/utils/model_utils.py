#!/usr/bin/env python3
# coding=utf-8
########################################################################
# Copyright (C) Baidu Ltd. All rights reserved.
########################################################################
"""
Author: zharui
Date: 2023-06-19 21:22:27
LastEditTime: 2023-07-02 21:46:06
LastEditors: zharui@baidu.com
FilePath: /aaai24/src_0702/utils/model_utils.py
Description: 
"""

import torch
import random
import copy
import pdb


def add_random_mask(x, mask_ratio=0.2, padding=0, ts_only=True):
    """ 对输入矩阵按行随机 mask 掉固定百分比的记录
    Input:
        x: [N, L, C] / ts: [N, L]
        mask_ratio: 新增缺失占比
        padding: 缺失部分填充值
    Return:
        x: [N, L, C] / ts: [N, L]
        add_mask: [N, L]
        org_ts: [N, L]
    """
    if not ts_only and len(x.shape) == 3:
        ts = x[..., 0]        # [B*N, L]
    else:
        ts = x
    
    pre_mask = torch.isnan(ts)
    
    rand_idx = torch.rand(*ts.shape)
    rand_idx[pre_mask] = 1
    threshold = torch.quantile(rand_idx, mask_ratio, dim=1)     # 计算百分位数
    
    add_mask = rand_idx < threshold[:, None]
    add_mask = add_mask.to(x.device) & ~pre_mask                # 相对输入而言新增的空缺

    org_ts = ts.clone()
    ts[pre_mask | add_mask] = padding

    if not ts_only: 
        x[..., 0] = ts
        return x, add_mask, org_ts
    else:
        return ts, add_mask, org_ts


def get_patch(x, patch_len, stride, dimension=1):
    """ 将 x 沿着指定 dimension 划分成 patch """
    total_len = x.shape[dimension]
    num_patch = (total_len - patch_len) // stride + 1
    squeeze_len = (num_patch - 1) * stride + patch_len
    x = torch.narrow(x, dimension, -squeeze_len, squeeze_len)
    x = x.unfold(dimension=dimension, size=patch_len, step=stride)
    return x


class EarlyStopping:
    """ 单卡训练早停机制 """
    def __init__(self, patience):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False

    def __call__(self, score, model, save_path, epoch=0):
        """EarlyStop function """
        if self.best_score is None or score < self.best_score:
            self.best_score = score
            self.counter = 0
            self.best_epoch = epoch
            self.save_checkpoint(model, save_path, epoch)
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience} with {epoch}')
            if self.counter >= self.patience:
                self.early_stop = True
    
    def save_checkpoint(self, model, save_path, epoch):
        print(f"Save model of epoch {epoch}")
        torch.save(model.state_dict(), save_path)


class DDPEarlyStopping:
    """ 分布式训练早停机制 """
    def __init__(self, patience, rank):
        self.patience = patience
        self.counter = 0
        self.rank = rank
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False

    def __call__(self, score, model, save_path, epoch=0):
        """ 每次 score 下降则保存模型, 如果触发早停则结束训练 """
        if self.best_score is None or score < self.best_score:
            self.best_score = score
            self.counter = 0
            self.best_epoch = epoch
            self.save_checkpoint(model, save_path, epoch)
        else:
            self.counter += 1
            self.save_checkpoint(model, save_path, epoch)

            print(f'EarlyStopping counter: {self.counter} out of {self.patience} with {epoch}')
            if self.counter >= self.patience:
                self.early_stop = True
                
    def save_checkpoint(self, model, save_path, epoch):
        """ 只在第0个设备上保存模型 """
        if self.rank == 0:
            print(f"Save model of epoch {epoch}")
            torch.save(model.module.state_dict(), save_path)
        