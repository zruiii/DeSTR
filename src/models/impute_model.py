#!/usr/bin/env python3
# coding=utf-8
########################################################################
# Copyright (C) Baidu Ltd. All rights reserved. 
########################################################################

"""
Author: zharui
Date: 2023-06-20 14:13:11
LastEditTime: 2023-08-22 15:49:52
LastEditors: zharui@baidu.com
FilePath: /decoupledST/src/models/impute_model.py
Description: 
"""

import pdb
import torch
import torch.nn as nn
from einops import rearrange

from models.decoupled_encoder import DecoupledEncoder

class ImputeModel(nn.Module):
    """ time series forecasting model """
    def __init__(self, h_dim, seq_len) -> None:
        super().__init__()
        self.encoder = DecoupledEncoder(h_dim)

        # 2-Layer MLP as Prediction Head
        self.predicter = nn.Sequential(nn.Linear(128 * 7, seq_len * 32),
                                       nn.ReLU(),
                                       nn.Linear(seq_len * 32, seq_len))

    def __repr__(self):
        return "ImputeModel"

    def forward(self, xt, xg=None, t=None, nids=None, g=None):
        """
        x: [B, N, L]
        t: [B, L, C]
        """
        B, N, L = xt.shape
        xt[torch.isnan(xt)] = 0

        z = self.encoder(xt=xt, nids=nids, t=t, g=g)                       # [B, N, k+1, H]
        z = rearrange(z, 'b n l h -> b n (l h)', b=B, n=N)
        out = self.predicter(z)

        return out

    def get_loss(self, pred, org, mask, mask_imp_weight=0.5, obs_rec_weight=0.5):
        """
        pred: [B, N, L]     重构序列
        org:  [B, N, L]     完整输入
        mask: [B, N, L]     imputation mask
        """
        org_mask = ~torch.isnan(org)
        rec_mask = org_mask & ~mask
        obs_rec_loss = torch.abs(pred[rec_mask] - org[rec_mask]).mean()
        mask_imp_loss = torch.abs(pred[mask] - org[mask]).mean()
        if mask_imp_weight > 0 and obs_rec_weight > 0:      # Hybrid Loss
            loss = mask_imp_loss * mask_imp_weight + obs_rec_loss * obs_rec_weight
        elif mask_imp_weight == 0:                  # Only Masked Imputation
            loss = obs_rec_loss     
        elif obs_rec_weight == 0:                   # Only Observation Reconstruct
            loss = mask_imp_loss
        else:
            raise NotImplementedError
        
        return loss
