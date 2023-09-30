#!/usr/bin/env python3
# coding=utf-8
########################################################################
# Copyright (C) Baidu Ltd. All rights reserved. 
########################################################################

"""
Author: zharui
Date: 2023-07-06 12:09:19
LastEditTime: 2023-08-10 11:14:00
LastEditors: zharui@baidu.com
FilePath: /decoupledST/src/baselines/AGCRN.py
Description: 
"""
import time
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

# fixme: 这个模块很耗时: bs=128耗时0.2s，24步下来耗时2s 多，一个epoch 半个多小时
class AVWGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        nn.init.xavier_normal_(self.weights_pool)
        nn.init.xavier_normal_(self.bias_pool)
        
    def forward(self, x, node_embeddings):
        #x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        #output shape [B, N, C]
        start_time = time.time()
        node_num = node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        support_set = [torch.eye(node_num).to(supports.device), supports]
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  #N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)                       #N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, x)      #B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias     #b, N, dim_out
        # pdb.set_trace()
        return x_gconv

class AGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AVWGCN(dim_in+self.hidden_dim, 2*dim_out, cheb_k, embed_dim)
        self.update = AVWGCN(dim_in+self.hidden_dim, dim_out, cheb_k, embed_dim)

    def forward(self, x, state, node_embeddings):
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim
        # start_time = time.time()
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))
        # print(1, time.time() - start_time)          # 0.1s

        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings))         # fixme
        # print(2, time.time() - start_time)          # 0.1s

        h = r*state + (1-r)*hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)
    
class AVWDCRNN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super(AVWDCRNN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, init_state, node_embeddings):
        #shape of x: (B, T, N, D)
        #shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        start_time = time.time()
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
            # pdb.set_trace()
            # print("layer", i, time.time() - start_time)         # 每一层大概2s, 一个 batch 大概4.5s, 一个 epoch 大概半小时...

        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)      #(num_layers, B, N, hidden_dim)


class AGCRN(nn.Module):
    def __init__(self, num_nodes, args):
        super(AGCRN, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = 1
        self.hidden_dim = 64
        self.output_dim = 1
        self.horizon = args.pred_len
        self.num_layers = 1
        self.cheb_k = 2

        self.default_graph = True
        self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, self.hidden_dim), requires_grad=True)

        self.encoder = AVWDCRNN(self.num_nodes, self.input_dim, self.hidden_dim, self.cheb_k,
                                self.hidden_dim, self.num_layers)

        #predictor
        self.end_conv = nn.Conv2d(1, self.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

    def __repr__(self):
        return "AGCRN"
    
    def forward(self, xt, xg=None, t=None, nids=None, g=None):
        #source: B, L, N, 1
        #target: B, P, N, 1
        #supports = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec1.transpose(0,1))), dim=1)
        
        source = xt
        source[torch.isnan(source)] = 1e-5
        if len(source.shape) == 3:
            source = source.unsqueeze(-1).transpose(1, 2)          # [B, T, N, 1]
        
        init_state = self.encoder.init_hidden(source.shape[0])
        output, _ = self.encoder(source, init_state, self.node_embeddings)      #B, T, N, hidden
        output = output[:, -1:, :, :]                                   #B, 1, N, hidden
        
        #CNN based predictor
        output = self.end_conv((output))                         #B, T*C, N, 1
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_nodes)
        output = output.permute(0, 1, 3, 2)                             #B, T, N, C

        output = output.squeeze().transpose(1, 2)               # [B, N, P]
        # print(time.time() - start_time)
        return output
    
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
    