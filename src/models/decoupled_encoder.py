#!/usr/bin/env python3
# coding=utf-8
########################################################################
# Copyright (C) Baidu Ltd. All rights reserved. 
########################################################################

"""
Author: zharui
Date: 2023-07-24 20:16:36
LastEditTime: 2023-09-07 11:31:25
LastEditors: zharui@baidu.com
FilePath: /decoupledST/src/models/decoupled_encoder.py
Description: 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb
import math
from einops import rearrange

import dgl
from dgl.nn import GATv2Conv, GraphConv, EdgeWeightNorm


class PositionalEncoding(nn.Module):
    """ 1D Position Encoding """
    def __init__(self, d_model, max_len=100, learnable=False):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.learnable = learnable

        if learnable:
            self.positional_encoding = nn.Parameter(torch.zeros(max_len, d_model))
            nn.init.xavier_uniform_(self.positional_encoding)
        else:
            self.positional_encoding = self.get_fixed_encoding(max_len, d_model)


    def get_fixed_encoding(self, max_len, d_model):
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        sin_term = torch.sin(position * div_term)
        cos_term = torch.cos(position * div_term)
        encoding = torch.zeros(max_len, d_model)
        encoding[:, 0::2] = sin_term
        encoding[:, 1::2] = cos_term
        return encoding


    def forward(self, seq_len):
        if self.learnable:
            encoding = self.positional_encoding[:seq_len, :]
        else:
            encoding = self.positional_encoding[:seq_len, :]

        return encoding
    

########## Signal Encoder ##########
class MaskCausalConv(nn.Module):
    """ Maked Causal Convolution """
    def __init__(self, in_dim, out_dim, kernel_size, dropout=0.2):
        super().__init__()
        self.padding = kernel_size-1
        self.input_conv = nn.Conv1d(in_dim, out_dim, kernel_size, stride=1, padding=self.padding)
        
        self.mask_conv = nn.Conv1d(in_dim, out_dim, kernel_size, stride=1, padding=self.padding)
        torch.nn.init.constant_(self.mask_conv.weight, 1.0)  # 计算 window 内有多少有效值

        # mask params are not updated
        for param in self.mask_conv.parameters():
            param.requires_grad = False

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_dim, out_dim, 1)
        # self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x, mask):
        """
        x, mask: [N, 1, L], [N, 1, L]
        output, new_mask: [N, C, L']
        """
        # http://masc.cs.gmu.edu/wiki/partialconv
        # C(X) = W^T * X + b, C(0) = b, D(M) = 1 * M + 0 = sum(M)
        # W^T* (M .* X) / sum(M) + b = [C(M .* X) – C(0)] / D(M) + C(0)

        # Step 1: Convolution 
        output = self.input_conv(x * mask)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1).expand_as(output)
        else:
            output_bias = torch.zeros_like(output)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        # Step2: Chomp
        output = output[:, :, :-self.padding]
        output_bias = output_bias[:, :, :-self.padding]
        output_mask = output_mask[:, :, :-self.padding]
        
        # Step 3: Re-Norm & Mask 
        no_update_holes = output_mask == 0
        mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)

        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes, 0.0)

        new_mask = torch.ones_like(output, device=output.device)
        new_mask = new_mask.masked_fill_(no_update_holes, 0.0)

        # Step4: relu + dropout
        output = self.dropout(self.relu(output))

        # Step5: Residual
        res = self.downsample(x)
        res = res.masked_fill_(no_update_holes, 0.0)
        output = self.relu(output + res)
        
        return output, new_mask


class MultiConvNet(nn.Module):
    """ Extract Signal Feature with Multi-layer Masked Causal Convolution """
    def __init__(self, channels, kernel_size) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        self.num_layers = len(channels)-1

        for i in range(self.num_layers):
            self.layers.append(MaskCausalConv(in_dim=channels[i],
                                              out_dim=channels[i+1],
                                              kernel_size=kernel_size))
      
    def forward(self, x):
        """ 
        x: [N, 1, L]
        out: [N, C, k]
        """
        mask = torch.where(x == 0, torch.tensor(0.0, device=x.device), torch.tensor(1.0, device=x.device))
        for i in range(self.num_layers):
            x, mask = self.layers[i](x, mask)

        return x


########## Graph Encoder ##########
class GraphConvNet(nn.Module):
    def __init__(self, num_layers, h_dim):
        super(GraphConvNet, self).__init__()
        """
        heads: list of heads in each layer, last one is 1
        feat_drop: input feature dopout
        attn_drop: attention dropout
        negative_slope: the negative slope of leaky relu
        residual: whether to use residual connection
        """
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for l in range(num_layers):
            # using distance as edge weight w/o self-loop
            self.layers.append(GraphConv(in_feats=h_dim,
                                         out_feats=h_dim,
                                         norm='none',
                                         weight=True,
                                         bias=True,
                                         allow_zero_in_degree=True))
            

    def forward(self, g, inputs):
        # add self-loop, default edge weigth as 1.0 for self-loop
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)

        edge_weight = g.edata['w']
        norm = EdgeWeightNorm(norm='both')
        norm_edge_weight = norm(g, edge_weight)

        h = inputs
        for l in range(self.num_layers):
            h = self.layers[l](g, h, edge_weight=norm_edge_weight)
        
        return h


########## Time Encoder ##########
class Time2Vec(nn.Module):
    def __init__(self, in_dim, h_dim):
        super().__init__()
        self.w0 = nn.Parameter(torch.randn(in_dim, 1))
        self.b0 = nn.Parameter(torch.randn(1))
        self.w = nn.Parameter(torch.randn(in_dim, h_dim-1))
        self.b = nn.Parameter(torch.randn(h_dim - 1))
        self.f = torch.sin
    
    def forward(self, tau):
        """
        t: [B, L, C]
        res: [B, L, H]
        """
        v0 = torch.matmul(tau, self.w0) + self.b0
        v1 = self.f(torch.matmul(tau, self.w) + self.b)
        res = torch.cat((v0, v1), dim=-1)
        return res


########## Cross Attention Fusion ##########
class CrossAttenBlock(nn.Module):
    """ Cross-Attention Block """
    def __init__(self,
                 d_model,
                 nhead,
                 activation=F.relu,
                 norm_first=True,
                 dim_feedforward=1024,
                 dropout=0.2) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.ti_linear1 = nn.Linear(d_model, dim_feedforward)
        self.ti_linear2 = nn.Linear(dim_feedforward, d_model)
        self.sp_linear1 = nn.Linear(d_model, dim_feedforward)
        self.sp_linear2 = nn.Linear(dim_feedforward, d_model)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.activation = activation


    def forward(self, xt, xg):
        """
        xt: [B*N, L, H]
        xg: [B*N, 1, H]
        """
        if self.norm_first: 
            xg = xg + self._cs_block(self.norm1(xg), self.norm1(xt))
            xt = xt + self._sa_block(self.norm1(xt))

            xt = xt + self._ti_ff_block(self.norm2(xt))
            xg = xg + self._sp_ff_block(self.norm2(xg))
        else:
            xg = self.norm1(xg + self._cs_block(xt, xg))
            xt = self.norm1(xt + self._sa_block(xt))

            xt = self.norm2(xt + self._ti_ff_block(xt))
            xg = self.norm2(xg + self._sp_ff_block(xg))

        return xt, xg


    # self-attention block
    def _sa_block(self, xt):
        xt = self.self_attn(xt, xt, xt, need_weights=False)[0]
        return self.dropout1(xt)


    # cross-attention block
    def _cs_block(self, xg, xt):
        xt = xt.detach()                    # * stop gradient 可以试试不用
        xg = self.cross_attn(xg, xt, xt, need_weights=False)[0]
        return self.dropout1(xg)


    # temporal feed forward block
    def _ti_ff_block(self, x):
        x = self.ti_linear2(self.dropout(self.activation(self.ti_linear1(x))))
        return self.dropout2(x)
    

    # spatial feed forward block
    def _sp_ff_block(self, x):
        x = self.sp_linear2(self.dropout(self.activation(self.sp_linear1(x))))
        return self.dropout2(x)
    

class CrossAttenModule(nn.Module):
    """ Cross Attention Module """
    def __init__(self, d_model, nhead=8, num_layers=2, dropout=0.2) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.attn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.attn_layers.append(CrossAttenBlock(d_model, nhead, norm_first=True, dim_feedforward=2048, dropout=dropout))


    def forward(self, xt, xg):
        """
        xt: [B, N, k, H]
        xg: [B, N, 1, H]
        """
        B, N, L, k = xt.shape
        xt = xt.reshape(B * N, L, -1)               # [B*N, L, H]
        xg = xg.reshape(B * N, 1, -1)               # [B*N, 1, H]

        for l in range(self.num_layers):
            xt, xg = self.attn_layers[l](xt, xg)

        xt = rearrange(xt, '(b n) l h -> b n l h', b=B, n=N)
        xg = rearrange(xg, '(b n) l h -> b n l h', b=B, n=N)

        return xt, xg


########## Decoupled Encoder ##########
class DecoupledEncoder(nn.Module):
    def __init__(self, h_dim, time_dim=4, max_num_nodes=3000) -> None:
        """
        h_dim: 表征维度
        max_num_nodes: 最多维护5000个节点的初始表征
        """
        super().__init__()

        # Signal Encoder  
        self.patch_encoder = MultiConvNet(channels=[1, 32, 64, h_dim],
                                          kernel_size=2)                            # stacked stride-1 1x2 convolutions
        
        self.last_conv = nn.Conv1d(h_dim, h_dim, kernel_size=4, stride=4, padding=0)        # stride-p 1xp convolution

        # Temporal Encoder
        self.time_encoder = Time2Vec(time_dim, h_dim)
        self.time_conv = nn.Conv1d(h_dim, h_dim, kernel_size=4, stride=4, padding=0)

        # Graph Encoder
        self.graph_embed = nn.Embedding(max_num_nodes, h_dim)               # Normal Initialize
        self.graph_encoder = GraphConvNet(num_layers=2, h_dim=h_dim)

        # Cross Attention Encoder
        self.crossformer_encoder = CrossAttenModule(h_dim, nhead=4, num_layers=2, dropout=0.2)


    def forward(self, xt, xg=None, t=None, nids=None, g=None):
        """
        xt: [B, N, L]   历史流量信息
        xg: [B, N, H]   节点特征
        t: [B, L, C]    时间特征
        nid: [N,]       节点全局ID
        """
        B, N, L = xt.shape
        
        # Signal Feature
        xt = xt.reshape(B*N, -1).unsqueeze(1)                                             # [B*N, 1, L]
        singal_embed = self.patch_encoder(xt)                                             # [B*N, H, k]
        singal_embed = self.last_conv(singal_embed)                                       # [B*N, H, k]
        singal_embed = rearrange(singal_embed, '(b n) h k -> b n k h', b=B, n=N)          # [B, N, k, H]

        # add TPE
        time_embed = self.time_encoder(t).transpose(2, 1)
        time_embed = self.time_conv(time_embed).transpose(2, 1)
        time_embed = time_embed.unsqueeze(1).repeat(1, N, 1, 1)                           # [B, N, k, H]
        singal_embed += time_embed

        # Network Feature
        graph_embed = self.graph_embed(nids)                                              # [N, H]
        graph_embed = self.graph_encoder(g, graph_embed)
        graph_embed = graph_embed.unsqueeze(0).repeat(B, 1, 1)                            # [B, N, H]
        graph_embed = graph_embed.unsqueeze(2).repeat(1, 1, 1, 1)                         # [B, N, 1, H]

        # Cross-Modal Fusion
        singal_embed, graph_embed = self.crossformer_encoder(singal_embed, graph_embed)    
        patch_embed = torch.cat((singal_embed, graph_embed), dim=2)                       # [B, N, k+1, H]

        return patch_embed



