#!/usr/bin/env python3
# coding=utf-8
########################################################################
# Copyright (C) Baidu Ltd. All rights reserved. 
########################################################################

"""
Author: zharui
Date: 2023-07-04 20:42:53
LastEditTime: 2023-09-19 13:13:52
LastEditors: zharui@baidu.com
FilePath: /decoupledST/src/baselines/DSTAGNN.py
Description: 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb
import pandas as pd
import numpy as np
from scipy.sparse.linalg import eigs


def scaled_Laplacian(W):
    '''
    compute \tilde{L}
    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices
    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)
    '''
    assert W.shape[0] == W.shape[1]
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    lambda_max = eigs(L, k=1, which='LR')[0].real
    return (2 * L) / lambda_max - np.identity(W.shape[0])


def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}
    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)
    K: the maximum order of chebyshev polynomials
    Returns
    ----------
    cheb_polynomials: list(np.ndarray), length: K, from T_0 to T_{K-1}
    '''
    N = L_tilde.shape[0]
    cheb_polynomials = [np.identity(N), L_tilde.copy()]
    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])
    return cheb_polynomials


class Embedding(nn.Module):
    def __init__(self, nb_seq, d_Em, num_of_features, Etype):
        super(Embedding, self).__init__()
        self.nb_seq = nb_seq
        self.Etype = Etype
        self.num_of_features = num_of_features
        self.pos_embed = nn.Embedding(nb_seq, d_Em)
        self.norm = nn.LayerNorm(d_Em)

    def forward(self, x, batch_size):
        if self.Etype == 'T':
            pos = torch.arange(self.nb_seq, dtype=torch.long).to(x.device)
            pos = pos.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_of_features,
                                                   self.nb_seq)  # [seq_len] -> [batch_size, seq_len]
            embedding = x.permute(0, 2, 3, 1) + self.pos_embed(pos)
        else:
            pos = torch.arange(self.nb_seq, dtype=torch.long).to(x.device)
            pos = pos.unsqueeze(0).expand(batch_size, self.nb_seq)
            embedding = x + self.pos_embed(pos)
        Emx = self.norm(embedding)          # [B, 1, T, N]
        return Emx


class SScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(SScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.
        return scores
    

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, num_of_d):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.num_of_d =num_of_d

    def forward(self, Q, K, V, attn_mask, res_att):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k) + res_att  # scores : [batch_size, n_heads, len_q, len_k]
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.
        attn = F.softmax(scores, dim=3)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, scores


class MultiHeadAttention(nn.Module):
    def __init__(self, DEVICE, d_model, d_k ,d_v, n_heads, num_of_d):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.num_of_d = num_of_d
        self.DEVICE = DEVICE
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_Q, input_K, input_V, attn_mask, res_att):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, self.num_of_d, -1, self.n_heads, self.d_k).transpose(2, 3)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, self.num_of_d, -1, self.n_heads, self.d_k).transpose(2, 3)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, self.num_of_d, -1, self.n_heads, self.d_v).transpose(2, 3)  # V: [batch_size, n_heads, len_v(=len_k), d_v]
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1,
                                                  1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, res_attn = ScaledDotProductAttention(self.d_k, self.num_of_d)(Q, K, V, attn_mask, res_att)

        context = context.transpose(2, 3).reshape(batch_size, self.num_of_d, -1,
                                                  self.n_heads * self.d_v)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]

        return self.norm(output + residual), res_attn


class SMultiHeadAttention(nn.Module):
    def __init__(self, DEVICE, d_model, d_k ,d_v, n_heads):
        super(SMultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.DEVICE = DEVICE
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        
        # NOTE: 修改到 paper 一致
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)           # scores : [batch_size, n_heads, len_q, len_k]
        attn = F.softmax(scores + input_V, dim=3)
        
        # if attn_mask is not None:
        #     attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        # # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        # attn = SScaledDotProductAttention(self.d_k)(Q, K, attn_mask)
        
        return attn


class GTU(nn.Module):
    def __init__(self, in_channels, time_strides, kernel_size):
        super(GTU, self).__init__()
        self.in_channels = in_channels
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.con2out = nn.Conv2d(in_channels, 2 * in_channels, kernel_size=(1, kernel_size), stride=(1, time_strides))

    def forward(self, x):
        x_causal_conv = self.con2out(x)
        x_p = x_causal_conv[:, : self.in_channels, :, :]
        x_q = x_causal_conv[:, -self.in_channels:, :, :]
        x_gtu = torch.mul(self.tanh(x_p), self.sigmoid(x_q))
        return x_gtu
    

class cheb_conv_withSAt(nn.Module):
    '''
    K-order chebyshev graph convolution
    '''
    def __init__(self, K, cheb_polynomials, in_channels, out_channels, num_of_vertices):
        '''
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super(cheb_conv_withSAt, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = cheb_polynomials[0].device
        self.relu = nn.ReLU(inplace=True)
        self.Theta = nn.ParameterList(
            [nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE)) for _ in range(K)])
        self.mask = nn.ParameterList(
            [nn.Parameter(torch.FloatTensor(num_of_vertices,num_of_vertices).to(self.DEVICE)) for _ in range(K)])
    
    def forward(self, x, spatial_attention, adj_pa):
        '''
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        '''
        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape
        outputs = []
        for time_step in range(num_of_timesteps):
            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)
            output = torch.zeros(batch_size, num_of_vertices, self.out_channels, device=self.DEVICE)  # (b, N, F_out)
            for k in range(self.K):
                T_k = self.cheb_polynomials[k]  # (N,N)
                mask = self.mask[k]
                myspatial_attention = spatial_attention[:, k, :, :] + adj_pa.mul(mask)      # note: 这里爆内存不够, Full-batch 操作
                myspatial_attention = F.softmax(myspatial_attention, dim=1)
                T_k_with_at = T_k.mul(myspatial_attention)   # (N,N)*(N,N) = (N,N) 多行和为1, 按着列进行归一化
                theta_k = self.Theta[k]  # (in_channel, out_channel)
                rhs = T_k_with_at.permute(0, 2, 1).matmul(graph_signal)  # (N, N)(b, N, F_in) = (b, N, F_in) 因为是左乘，所以多行和为1变为多列和为1，即一行之和为1，进行左乘
                output = output + rhs.matmul(theta_k)  # (b, N, F_in)(F_in, F_out) = (b, N, F_out)
            outputs.append(output.unsqueeze(-1))  # (b, N, F_out, 1)
        return self.relu(torch.cat(outputs, dim=-1))  # (b, N, F_out, T)


class DSTAGNN_block(nn.Module):
    def __init__(self, DEVICE, num_of_d, in_channels, K, nb_chev_filter, nb_time_filter, time_strides,
                 cheb_polynomials, adj_pa, adj_TMD, num_of_vertices, num_of_timesteps, d_model, d_k, d_v, n_heads):
        super(DSTAGNN_block, self).__init__()

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True)

        self.adj_pa = torch.FloatTensor(adj_pa).to(DEVICE)         # STRG
        self.adj_TMD = torch.FloatTensor(adj_TMD).to(DEVICE)        # STAG

        self.pre_conv = nn.Conv2d(num_of_timesteps, d_model, kernel_size=(1, num_of_d))

        self.EmbedT = Embedding(num_of_timesteps, num_of_vertices, num_of_d, 'T')
        self.EmbedS = Embedding(num_of_vertices, d_model, num_of_d, 'S')

        self.TAt = MultiHeadAttention(DEVICE, num_of_vertices, d_k, d_v, n_heads, num_of_d)
        # NOTE: 修改
        self.SAt = SMultiHeadAttention(DEVICE, d_model, d_k, d_v, n_heads)

        self.cheb_conv_SAt = cheb_conv_withSAt(K, cheb_polynomials, in_channels, nb_chev_filter, num_of_vertices)

        self.gtu3 = GTU(nb_time_filter, time_strides, 7)
        self.gtu5 = GTU(nb_time_filter, time_strides, 9)
        self.gtu7 = GTU(nb_time_filter, time_strides, 11)
        self.pooling = torch.nn.MaxPool2d(kernel_size=(1, 2), stride=None, padding=0,
                                          return_indices=False, ceil_mode=False)

        self.residual_conv = nn.Conv2d(in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides))

        self.dropout = nn.Dropout(p=0.05)
        self.fcmy = nn.Sequential(
            nn.Linear(3 * num_of_timesteps - 12, num_of_timesteps),
            nn.Dropout(0.05),
        )
        self.ln = nn.LayerNorm(nb_time_filter)

        self.W_m = nn.Parameter(torch.FloatTensor(num_of_vertices, num_of_vertices))
        nn.init.xavier_normal_(self.W_m)

    def forward(self, x, res_att):
        '''
        :param x: (Batch_size, N, F_in, T)
        :param res_att: (Batch_size, N, F_in, T)
        :return: (Batch_size, N, nb_time_filter, T)
        '''
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape  # B,N,F,T

        # TAT
        # if num_of_features == 1:
        #     TEmx = self.EmbedT(x, batch_size)  # 添加 Pos Embedding // B,F,T,N
        # else:
        #     TEmx = x.permute(0, 2, 3, 1)

        TEmx = x.permute(0, 2, 3, 1)        # NOTE: Paper 中没有添加 Pos Embed
        
        TATout, re_At = self.TAt(TEmx, TEmx, TEmx, None, res_att)  # B,F,T,N; B,F,Ht,T,T

        x_TAt = self.pre_conv(TATout.permute(0, 2, 3, 1))[:, :, :, -1].permute(0, 2, 1)  # B,N,d_model

        # SAt
        SEmx_TAt = self.EmbedS(x_TAt, batch_size)  # B,N,d_model
        SEmx_TAt = self.dropout(SEmx_TAt)   # B,N,d_model
        
        # NOTE: 修改至和原文一致
        STAt = self.SAt(SEmx_TAt, SEmx_TAt, torch.mul(self.adj_pa, self.W_m), None)  # B,Hs,N,N

        # NOTE: Spatial Graph Convolution, 这一步复杂度极高
        spatial_gcn = self.cheb_conv_SAt(x, STAt, self.adj_TMD)  # B,N,F,T

        # convolution along the time axis
        X = spatial_gcn.permute(0, 2, 1, 3)  # B,F,N,T
        x_gtu = []
        
        # NOTE: 论文中说可以调整 GTU 的参数，使得 GTU 的输出拼接起来仍然等于原来的维度; 并且添加 Pooling
        # x_gtu.append(self.gtu3(X))  # B,F,N,T-2
        # x_gtu.append(self.gtu5(X))  # B,F,N,T-4
        # x_gtu.append(self.gtu7(X))  # B,F,N,T-6

        x_gtu.append(self.pooling(self.gtu3(X)))  # B,F,N,T-2
        x_gtu.append(self.pooling(self.gtu5(X)))  # B,F,N,T-4
        x_gtu.append(self.pooling(self.gtu7(X)))  # B,F,N,T-6

        time_conv = torch.cat(x_gtu, dim=-1)  # B,F,N,3T-1
        # time_conv = self.fcmy(time_conv)

        time_conv_output = self.relu(X + time_conv)  # B,F,N,T

        # residual shortcut
        if num_of_features == 1:
            x_residual = self.residual_conv(x.permute(0, 2, 1, 3))
        else:
            x_residual = x.permute(0, 2, 1, 3)
        x_residual = self.ln(F.relu(x_residual + time_conv_output).permute(0, 3, 2, 1)).permute(0, 2, 3, 1)

        return x_residual, re_At


class DSTAGNN_submodule(nn.Module):
    def __init__(self, DEVICE, num_of_d, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides,
                 cheb_polynomials, adj_pa, adj_TMD, num_for_predict, len_input, num_of_vertices, d_model, d_k, d_v, n_heads):
        '''
        :param nb_block:
        :param in_channels:
        :param K:
        :param nb_chev_filter:
        :param nb_time_filter:
        :param time_strides:
        :param cheb_polynomials:
        :param num_for_predict:
        '''

        super(DSTAGNN_submodule, self).__init__()

        self.BlockList = nn.ModuleList([DSTAGNN_block(DEVICE, num_of_d, in_channels, K,
                                                     nb_chev_filter, nb_time_filter, time_strides, cheb_polynomials,
                                                     adj_pa, adj_TMD, num_of_vertices, len_input, d_model, d_k, d_v, n_heads)])

        self.BlockList.extend([DSTAGNN_block(DEVICE, num_of_d * nb_time_filter, nb_chev_filter, K,
                                            nb_chev_filter, nb_time_filter, 1, cheb_polynomials,
                                            adj_pa, adj_TMD, num_of_vertices, len_input//time_strides, d_model, d_k, d_v, n_heads) for _ in range(nb_block-1)])

        self.final_conv = nn.Conv2d(int((len_input/time_strides) * nb_block), 128, kernel_size=(1, nb_time_filter))
        self.final_fc = nn.Linear(128, num_for_predict)
        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, xt, xg=None, t=None, nids=None, g=None):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        """
        for block in self.BlockList:
            x = block(x)
            
        output = self.final_conv(x.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1)
        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)
        """
        x = xt
        x[torch.isnan(x)] = 0
        if len(x.shape) == 3:
            x = x.unsqueeze(2)          # [B, N, 1, T]
            
        need_concat = []
        res_att = 0
        for block in self.BlockList:
            x, res_att = block(x, res_att)
            need_concat.append(x)

        final_x = torch.cat(need_concat, dim=-1)
        output1 = self.final_conv(final_x.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1)
        output = self.final_fc(output1)

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


def make_model(DEVICE, num_of_d, nb_block, in_channels, K,
               nb_chev_filter, nb_time_filter, time_strides, adj_mx, adj_pa,
               adj_TMD, num_for_predict, len_input, num_of_vertices, d_model, d_k, d_v, n_heads):
    '''

    :param DEVICE:
    :param nb_block:
    :param in_channels:
    :param K:
    :param nb_chev_filter:
    :param nb_time_filter:
    :param time_strides:
    :param num_for_predict:
    :param len_input
    :return:
    '''
    L_tilde = scaled_Laplacian(adj_mx)
    cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor).to(DEVICE) for i in cheb_polynomial(L_tilde, K)]
    model = DSTAGNN_submodule(DEVICE, num_of_d, nb_block, in_channels,
                             K, nb_chev_filter, nb_time_filter, time_strides, cheb_polynomials,
                             adj_pa, adj_TMD, num_for_predict, len_input, num_of_vertices, d_model, d_k, d_v, n_heads)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

    return model


def get_adjacency_matrix2(distance_df_filename, num_of_vertices,
                         type_='connectivity', id_filename=None):
    ''' 获取无权的地理邻接矩阵
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    type_: str, {connectivity, distance} 决定返回的邻接矩阵是否为加权矩阵, weight = 1. / distance

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''
    import csv

    A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                 dtype=np.float32)

    if id_filename:
        with open(id_filename, 'r') as f:
            id_dict = {int(i): idx
                       for idx, i in enumerate(f.read().strip().split('\n'))}
        with open(distance_df_filename, 'r') as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                A[id_dict[i], id_dict[j]] = 1
                A[id_dict[j], id_dict[i]] = 1
        return A

    # Fills cells in the matrix with distances.
    with open(distance_df_filename, 'r') as f:
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 3:
                continue
            i, j, distance = int(row[0]), int(row[1]), float(row[2])
            if type_ == 'connectivity':
                A[i, j] = 1
                # A[j, i] = 1
            elif type_ == 'distance':
                A[i, j] = 1 / distance
                A[j, i] = 1 / distance
            else:
                raise ValueError("type_ error, must be "
                                 "connectivity or distance!")
    return A


def load_weighted_adjacency_matrix2(file_path, num_v):
    df = pd.read_csv(file_path, header=None)
    df = df.to_numpy()
    df = np.int64(df > 0)
    id_mat = np.identity(num_v)
    mydf = df - id_mat
    return mydf



def load_weighted_adjacency_matrix(file_path, num_v):
    df = pd.read_csv(file_path, header=None)
    df = df.to_numpy()
    df = np.float64(df > 0)
    return df


def load_PA(file_path):
    df = pd.read_csv(file_path, header=None)
    df = df.to_numpy()
    df = np.float64(df>0)
    return df


def load_dstagnn(device, adj_mx, adj_TMD, adj_pa, d_model=128, len_input=24, 
                 num_for_predict=12, num_of_vertices=325):
    """
    adj_mx : 地理的邻接矩阵 (无权)
    adj_TMD : 基于时间序列相似度计算的邻接矩阵
    adj_pa : 归一化后的相似度矩阵
    """
    in_channels = 1
    num_of_d = in_channels
    nb_block = 4                # 2个block
    n_heads = 4
    K = 3                       # 1-hop 邻居 (节省内存)
    d_k = 32
    nb_chev_filter = 32
    nb_time_filter = 32
    time_strides = 1
    d_v = d_k
    graph_use = 'AG'

    if graph_use =='G':
        adj_merge = adj_mx
    else:
        adj_merge = adj_TMD

    net = make_model(device, num_of_d, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, 
                     adj_merge, adj_pa, adj_TMD, num_for_predict, len_input, num_of_vertices, d_model, d_k, d_v, n_heads)

    return net


if __name__ == "__main__":
    import pdb
    DEVICE = torch.device('cuda:7')
    in_channels = 1
    num_of_d = 1
    num_of_weeks = 0
    num_of_days = 0
    num_of_hours = 1
    nb_block = 4
    n_heads = 4
    K = 3
    d_k = 32
    d_model = 512
    nb_chev_filter = 32
    nb_time_filter = 32
    time_strides = 1
    num_for_predict = 12
    len_input = 12
    num_of_vertices = 307
    d_v = d_k
    dataset_name = 'PEMS04'
    graph_use = 'AG'
    
    adj_filename = "/home/users/zharui/aaai24/open_src/DSTAGNN-main/data/PEMS04/PEMS04.csv"           # from, to, cost: 不同地点之间的距离
    graph_signal_matrix_filename = "/home/users/zharui/aaai24/open_src/DSTAGNN-main/data/PEMS04/PEMS04.npz"
    stag_filename = "/home/users/zharui/aaai24/open_src/DSTAGNN-main/data/PEMS04/stag_001_PEMS04.csv"
    strg_filename = "/home/users/zharui/aaai24/open_src/DSTAGNN-main/data/PEMS04/strg_001_PEMS04.csv"

    # 地理邻接矩阵
    if dataset_name == 'PEMS04' or 'PEMS08' or 'PEMS07' or 'PEMS03':
        adj_mx = get_adjacency_matrix2(adj_filename, num_of_vertices, id_filename=None)
    else:
        adj_mx = load_weighted_adjacency_matrix2(adj_filename, num_of_vertices)
    
    # 基于时间序列相似度计算的相似度矩阵
    adj_TMD = load_weighted_adjacency_matrix(stag_filename, num_of_vertices)
    
    # 归一化后的相似度矩阵
    adj_pa = load_PA(strg_filename)

    if graph_use =='G':
        adj_merge = adj_mx
    else:
        adj_merge = adj_TMD

    net = make_model(DEVICE, num_of_d, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, adj_merge,
                    adj_pa, adj_TMD, num_for_predict, len_input, num_of_vertices, d_model, d_k, d_v, n_heads)

    pdb.set_trace()