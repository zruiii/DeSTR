#!/usr/bin/env python3
# coding=utf-8
########################################################################
# Copyright (C) Baidu Ltd. All rights reserved. 
########################################################################

"""
Author: zharui
Date: 2023-06-20 14:13:01
LastEditTime: 2023-09-19 19:28:07
LastEditors: zharui@baidu.com
FilePath: /decoupledST/src/models/forecast_model.py
Description: 
"""
import pdb
import torch
import torch.nn as nn
from einops import rearrange

from models.decoupled_encoder import DecoupledEncoder
from models import variants

class ForecastModel(nn.Module):
    """ time series forecasting model """
    def __init__(self, h_dim, pred_dim, variant=None) -> None:
        super().__init__()
        if variant == "stgg_wo_sg":
            self.encoder = variants.DecoupledEncoderV5(h_dim)
        elif variant == "stgg_to_trans":
            self.encoder = variants.DecoupledEncoderV4(h_dim)
        elif variant == "mcc_to_lstm":
            self.encoder = variants.DecoupledEncoderV3(h_dim)
        elif variant == "mcc_to_tcn":
            self.encoder = variants.DecoupledEncoderV2(h_dim)
        elif variant == "mcc_to_mlp":
            self.encoder = variants.DecoupledEncoderV1(h_dim)
        else:
            self.encoder = DecoupledEncoder(h_dim)

        # 2-Layer MLP as Prediction Head
        patches = 7 if variant != "stgg_to_trans" else 6
        self.predicter = nn.Sequential(nn.Linear(128 * patches, pred_dim * 32),
                                       nn.ReLU(),
                                       nn.Linear(pred_dim * 32, pred_dim))

    def __repr__(self):
        return "ForecastModel"

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




