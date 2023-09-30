#!/usr/bin/env python3
# coding=utf-8
########################################################################
# Copyright (C) Baidu Ltd. All rights reserved. 
########################################################################

"""
Author: zharui
Date: 2023-06-28 11:32:21
LastEditTime: 2023-09-15 19:27:15
LastEditors: zharui@baidu.com
FilePath: /decoupledST/src/baselines/MEAN.py
Description: 
"""

import torch
import torch.nn as nn


class MEAN(nn.Module):
    def __init__(self):
        super().__init__()
    

    def __repr__(self):
        return "MEAN"


    def forward(self, x):
        """
        x: [B, N, L]
        out: [B, N, P]
        """
        x[torch.isnan(x)] = 0
        out = x[..., -1:].mean(-1).unsqueeze(-1)
        out = out.repeat(1, 1, self.pred_len)

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
        return ((pred - true) ** 2).mean()

