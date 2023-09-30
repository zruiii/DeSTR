#!/usr/bin/env python3
# coding=utf-8
########################################################################
# Copyright (C) Baidu Ltd. All rights reserved. 
########################################################################

"""
Author: zharui
Date: 2023-07-06 12:09:19
LastEditTime: 2023-09-01 16:34:21
LastEditors: zharui@baidu.com
FilePath: /decoupledST/src/baselines/BRITS.py
Description: 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

import math
import numpy as np
import pdb

def parse_delta(masks, seq_len):
    masks = masks.int()
    deltas = torch.zeros_like(masks, dtype=torch.float32, device=masks.device)
    for h in range(seq_len):
        if h == 0:
            deltas[h, :] = 1
        else:
            deltas[h, :] = 1 + (1 - masks[h]) * deltas[h-1]

    return deltas


def binary_cross_entropy_with_logits(input, target, weight=None, size_average=True, reduce=True):
    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

    if weight is not None:
        loss = loss * weight

    if not reduce:
        return loss
    elif size_average:
        return loss.mean()
    else:
        return loss.sum()


class FeatureRegression(nn.Module):
    def __init__(self, input_size):
        super(FeatureRegression, self).__init__()
        self.build(input_size)

    def build(self, input_size):
        self.W = Parameter(torch.Tensor(input_size, input_size))
        self.b = Parameter(torch.Tensor(input_size))

        m = torch.ones(input_size, input_size) - torch.eye(input_size, input_size)
        self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        z_h = F.linear(x, self.W * Variable(self.m), self.b)
        return z_h


class TemporalDecay(nn.Module):
    def __init__(self, input_size, output_size, diag = False):
        super(TemporalDecay, self).__init__()
        self.diag = diag

        self.build(input_size, output_size)

    def build(self, input_size, output_size):
        self.W = Parameter(torch.Tensor(output_size, input_size))
        self.b = Parameter(torch.Tensor(output_size))

        if self.diag == True:
            assert(input_size == output_size)
            m = torch.eye(input_size, input_size)
            self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, d):
        if self.diag == True:
            gamma = F.relu(F.linear(d, self.W * Variable(self.m), self.b))
        else:
            gamma = F.relu(F.linear(d, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma


class RITS(nn.Module):
    def __init__(self, h_dim, seq_len, num_nodes):
        super(RITS, self).__init__()

        self.rnn_hid_size = h_dim
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.build()

    def build(self):
        self.rnn_cell = nn.LSTMCell(self.num_nodes * 2, self.rnn_hid_size)

        self.temp_decay_h = TemporalDecay(input_size = self.num_nodes, output_size = self.rnn_hid_size, diag = False)
        self.temp_decay_x = TemporalDecay(input_size = self.num_nodes, output_size = self.num_nodes, diag = True)

        self.hist_reg = nn.Linear(self.rnn_hid_size, self.num_nodes)
        self.feat_reg = FeatureRegression(self.num_nodes)

        self.weight_combine = nn.Linear(self.num_nodes * 2, self.num_nodes)

        self.dropout = nn.Dropout(p = 0.25)
        self.out = nn.Linear(self.rnn_hid_size, 1)


    def forward(self, xt, masks, deltas):
        """
        xt/masks/deltas: [B, L, C], C equals to num_nodes N
        imputations: [B, L, C]
        """
        B, L, C = xt.shape
        xt[torch.isnan(xt)] = 0
        h = Variable(torch.zeros((B, self.rnn_hid_size))).to(xt.device)
        c = Variable(torch.zeros((B, self.rnn_hid_size))).to(xt.device)

        x_loss = 0.0
        imputations = []

        for t in range(self.seq_len):
            x = xt[:, t, :]
            m = masks[:, t, :]
            d = deltas[:, t, :]

            gamma_h = self.temp_decay_h(d)              # [B, H]
            gamma_x = self.temp_decay_x(d)              # [B, C]

            h = h * gamma_h
            x_h = self.hist_reg(h)                      # [B, C]
            x_loss += torch.sum(torch.abs(x - x_h) * m) / (torch.sum(m) + 1e-5)

            x_c =  m * x +  (1 - m) * x_h
            z_h = self.feat_reg(x_c)
            x_loss += torch.sum(torch.abs(x - z_h) * m) / (torch.sum(m) + 1e-5)

            alpha = self.weight_combine(torch.cat([gamma_x, m], dim = 1))
            c_h = alpha * z_h + (1 - alpha) * x_h
            x_loss += torch.sum(torch.abs(x - c_h) * m) / (torch.sum(m) + 1e-5)

            c_c = m * x + (1 - m) * c_h
            inputs = torch.cat([c_c, m], dim = 1)
            h, c = self.rnn_cell(inputs, (h, c))

            imputations.append(c_c.unsqueeze(dim = 1))

        imputations = torch.cat(imputations, dim = 1)
        return imputations, x_loss


class BRITS(nn.Module):
    def __init__(self, h_dim, seq_len, num_nodes):
        super(BRITS, self).__init__()

        self.rnn_hid_size = h_dim
        self.seq_len = seq_len
        self.num_nodes = num_nodes

        self.build()
        self.loss = None
       

    def build(self):
        self.rits_f = RITS(self.rnn_hid_size, seq_len=self.seq_len, num_nodes=self.num_nodes)
        self.rits_b = RITS(self.rnn_hid_size, seq_len=self.seq_len, num_nodes=self.num_nodes)

    def forward(self, xt, xg=None, t=None, nids=None, g=None):
        """
        xt: [B, N, L]
        impuations: [B, N, L]
        """
        B, N, L = xt.shape
        forward_xt = xt.transpose(2, 1)                                         # [B, L, N]
        backward_xt = forward_xt.index_select(1, torch.arange(L-1, -1, -1, device=xt.device))     # [B, L, N]

        forward_mask = ~torch.isnan(forward_xt)
        backward_mask = ~torch.isnan(backward_xt)
        delta = parse_delta(forward_mask, L)

        ret_f, loss_f = self.rits_f(forward_xt, forward_mask.int(), delta)
        ret_b, loss_b = self.rits_b(backward_xt, backward_mask.int(), delta)
        ret_b = ret_b.index_select(1, torch.arange(L-1, -1, -1, device=xt.device))

        loss_c = self.get_consistency_loss(ret_f, ret_b)
        self.loss = loss_f + loss_b + loss_c
        impuations = (ret_f + ret_b) / 2

        return impuations.transpose(2, 1)


    def get_consistency_loss(self, pred_f, pred_b):
        loss = torch.abs(pred_f - pred_b).mean() * 1e-1
        return loss


    def get_loss(self, pred, org, mask, mask_imp_weight=0.5, obs_rec_weight=0.5):
        return self.loss


if __name__ == "__main__":
    xt = torch.randn(32, 10, 24)
    idx = torch.randn(32, 10, 24)
    xt[idx < 0] = torch.nan

    model = BRITS(64, seq_len=24, num_nodes=10)
    imputations = model(xt)
    loss = model.loss
    print(imputations.shape, loss.item())

