#!/usr/bin/env python3
# coding=utf-8
########################################################################
# Copyright (C) Baidu Ltd. All rights reserved. 
########################################################################

"""
Author: zharui
Date: 2023-09-05 14:57:06
LastEditTime: 2023-09-05 15:23:11
LastEditors: zharui@baidu.com
FilePath: /decoupledST/src/baselines/FCLSTM.py
Description: 
"""
import pdb
import torch
import torch.nn as nn


class FCLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FCLSTM, self).__init__()
        
        # Fully connected layer
        self.fc = nn.Linear(input_dim, hidden_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # Output layer
        self.linear = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, xt, xg=None, t=None, nids=None, g=None):
        B, N, L = xt.shape
        xt[torch.isnan(xt)] = 0
        
        xt = xt.view(-1, L).unsqueeze(-1)   # [B*N, L, 1]
        xt = self.fc(xt)                    # [B*N, L, H]
        
        h, _ = self.lstm(xt)                # [B*N, L, H]
        h = h[:, -1, :].reshape(B, N, -1)   # [B, N, H]
        out = self.linear(h)                # [B, N, P]
        
        return out

    def get_loss(self, pred, true, mask=None):
        """
        pred: [B, N, P]
        true: [B, N, P]
        mask: [B, N, P]
        """
        if mask is not None:
            pred = pred[mask].flatten()
            true = true[mask].flatten()
        return torch.abs(pred - true).mean()
    