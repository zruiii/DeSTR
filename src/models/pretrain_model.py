#!/usr/bin/env python3
# coding=utf-8
########################################################################
# Copyright (C) Baidu Ltd. All rights reserved. 
########################################################################

"""
Author: zharui
Date: 2023-08-07 11:35:55
LastEditTime: 2023-09-12 14:19:09
LastEditors: zharui@baidu.com
FilePath: /decoupledST/src/models/pretrain_model.py
Description: 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pdb
import random
import argparse
import numpy as np

from einops import rearrange

from models.decoupled_encoder import DecoupledEncoder


class DeSTR(nn.Module):
    """ Masked AutoEncoder on both Spatial & Temporal dimension """
    def __init__(self, h_dim, seq_len, mask_ratio, mask_mode="spacetime") -> None:
        super().__init__()
        self.mask_ratio = mask_ratio
        self.mask_mode = mask_mode

        self.encoder = DecoupledEncoder(h_dim)

        self.decoder = nn.Sequential(nn.Linear(h_dim * 7, 512), 
                                     nn.ReLU(inplace=True), 
                                     nn.Linear(512, seq_len))
        
    def __repr__(self):
        return "DeSTR"

    def forward(self, xt, xg=None, t=None, nids=None, g=None):
        """
        xt: [B, N, L]
        xg: [B, N, H]
        t: [B, L, C]
        """
        B, N, L = xt.shape
        
        ###### random mask ######
        if self.mask_mode == "spacetime":
            mask_xt, mask = self.spacetime_mask(xt) 
        elif self.mask_mode == "space":
            mask_xt, mask = self.space_mask(xt)
        elif self.mask_mode == "time":
            mask_xt, mask = self.time_mask(xt)
        else:
            raise NotImplementedError

        ###### encoder ######
        z = self.encoder(xt=mask_xt, t=t, nids=nids, g=g)                     # [B, N, k+1, H]
        z = rearrange(z, 'b n l h -> b n (l h)', b=B, n=N)

        ###### decoder ######
        y = self.decoder(z)                                    # [B, N, L]

        ###### loss ######
        loss = self.get_reconstruct_loss(xt, y, mask)

        return loss
    

    def time_mask(self, x, padding=0):
        B, N, L = x.shape
        device = x.device
        x = x.reshape(B*N, L)
        pre_miss = torch.isnan(x)

        # Time Masking
        rand_idx = torch.randn(L).argsort(dim=-1)
        rand_idx = rand_idx[:round(L * self.mask_ratio)]
        time_mask = torch.zeros_like(x, dtype=torch.bool)
        time_mask[:, rand_idx] = True

        # New mask
        ts = x.clone()
        add_mask = time_mask & ~pre_miss              # 相对输入而言新增的空缺
        ts[pre_miss | add_mask] = padding

        ts = rearrange(ts, '(b n) l -> b n l', b=B, n=N)
        add_mask = rearrange(add_mask, '(b n) l -> b n l', b=B, n=N)

        return ts, add_mask
        

    def space_mask(self, x, padding=0):
        B, N, L = x.shape
        device = x.device
        x = x.reshape(B*N, L)
        pre_miss = torch.isnan(x)

        # Space Masking
        random_seq = torch.rand(B, N, device=device).argsort(dim=-1)
        random_seq = random_seq[:, :round(N * self.mask_ratio)]
        sensor_mask = F.one_hot(random_seq, num_classes=N).sum(dim=1)
        sensor_mask = (sensor_mask == 1)                                        # [B, N]
        sensor_mask = sensor_mask.unsqueeze(-1).expand(B, N, L)
        sensor_mask = sensor_mask.reshape(B*N, L)

        # New mask
        ts = x.clone()
        add_mask = sensor_mask & ~pre_miss              # 相对输入而言新增的空缺
        ts[pre_miss | add_mask] = padding

        ts = rearrange(ts, '(b n) l -> b n l', b=B, n=N)
        add_mask = rearrange(add_mask, '(b n) l -> b n l', b=B, n=N)

        return ts, add_mask


    def spacetime_mask(self, x, padding=0):
        B, N, L = x.shape
        device = x.device
        x = x.reshape(B*N, L)
        pre_miss = torch.isnan(x)

        # Spacetime Masking
        rand_idx = torch.rand_like(x)                                         # [B, N, L]
        rand_idx[pre_miss] = 1
        rand_idx = torch.argsort(rand_idx, dim=-1)[..., :round(L * self.mask_ratio)]
        spacetime_mask = torch.zeros_like(x, dtype=torch.bool)
        spacetime_mask.scatter_(-1, rand_idx, True)
        
        # New mask
        ts = x.clone()
        add_mask = spacetime_mask & ~pre_miss              # 相对输入而言新增的空缺
        ts[pre_miss | add_mask] = padding

        ts = rearrange(ts, '(b n) l -> b n l', b=B, n=N)
        add_mask = rearrange(add_mask, '(b n) l -> b n l', b=B, n=N)

        return ts, add_mask


    # TODO: 还可以添加对预测结构的平滑项作为惩罚
    def get_reconstruct_loss(self, labels, pred, mask=None, alpha=6):
        """ 获取重构的损失
        labels: 原始真实值
        pred:   预测值
        mask:   新增的掩码 True or False, 这部分重构权重调大点
        alpha:  掩码部分重构权重, 其他部分默认为1
        """
        pre_mask = ~torch.isnan(labels)
        valid_mask = pre_mask & ~mask

        # 非缺失值重构
        valid_pred = pred[valid_mask]
        valid_inputs = labels[valid_mask]
        valid_loss = torch.abs(valid_inputs - valid_pred).mean()

        # 新增缺失部分重构
        miss_loss = torch.tensor(0.0, device=labels.device)
        if mask is not None:
            labels = labels[mask].flatten()
            pred = pred[mask].flatten()
            miss_loss = torch.abs(labels - pred).mean()

        loss = alpha * miss_loss + valid_loss
        return loss


        




