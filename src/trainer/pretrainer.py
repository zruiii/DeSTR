#!/usr/bin/env python3
# coding=utf-8
########################################################################
# Copyright (C) Baidu Ltd. All rights reserved.
########################################################################
"""
Author: zharui
Date: 2023-06-07 14:48:48
LastEditTime: 2023-08-22 14:15:35
LastEditors: zharui@baidu.com
FilePath: /decoupledST/src/trainer/pretrainer.py
Description: 分布式训练
"""

import numpy as np
import random
import pdb
from tqdm import tqdm

import os
import time
import math
import torch
import torch.nn as nn
import torch.distributed as dist

from utils.train_utils import EarlyStopping, DDPEarlyStopping

class PreTrainer:
    """ 预训练模型训练器, 适用于单机单卡和单机多卡 """
    def __init__(self, model, device, train_data, val_data, 
                 lr, optimizer, logger, save_path, use_ddp=False,
                 min_lr=5e-5, warmup_epochs=10, max_epochs=100):
        """
        model: 模型
        device: GPU
        train_data: DataLoader
        optimizer:  优化器
        logger:  日志记录器
        use_ddp: 是否启用分布式训练
        """
        self.model = model
        self.device = device
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.logger = logger
        self.max_epochs = max_epochs
        self.save_path = save_path
        self.g = train_data.dataset.g.to(device)                            # 路网图结构
        self.nids = torch.tensor(train_data.dataset.nids).to(device)        # 目标节点 ID

        # ddp
        if use_ddp:
            self.world_size = torch.cuda.device_count()
        else:
            self.world_size = None
        self.use_ddp = use_ddp

        # warm up
        self.lr = lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs


    def _check_device(self):
        """ 确认当前设备是否为第0张卡 """
        if ((isinstance(self.device, int) and self.device == 0) or 
                (hasattr(self.device, 'index') and self.device.index == 0)):
            return True
        else:
            return False


    def _run_batch(self, batch):
        """ 训练单个 batch 样本
        xt: [B, N, L]
        xg: [B, N, H]
        t: [B, L, C]
        g: dgl.Graph
        """
        self.optimizer.zero_grad()
        loss = self.model(xt=batch.xt.to(self.device), 
                          xg=batch.xg.to(self.device), 
                          t=batch.t.to(self.device), 
                          nids=self.nids, g=self.g)
        loss.backward()
        self.optimizer.step()
        return loss


    def _run_train(self, epoch):
        """ 训练单个 epoch 样本 """
        self.model.train()
        if self.use_ddp:
            self.train_data.sampler.set_epoch(epoch)

        epoch_loss = []

        for it, batch in enumerate(self.train_data):

            # * Warmup: increased linearly and annealed to a minima using cosine schedule
            iteration = it / len(self.train_data) + epoch
            if iteration < self.warmup_epochs:
                lr_scale = iteration / self.warmup_epochs
                new_lr = self.lr * lr_scale
            else:
                new_lr = self.min_lr + (self.lr - self.min_lr) * 0.5 * \
                    (1. + math.cos(math.pi * (iteration - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr

            loss = self._run_batch(batch)

            # 所有进程损失的均值
            if self.use_ddp:
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                loss /= self.world_size
            
            print_every = 20
            epoch_loss.append(loss.item())

            if self._check_device() and (it + 1) % print_every == 0:
                avg_loss = sum(epoch_loss[-print_every:]) / print_every
                self.logger.info("Epoch {} | Iteration {} | Loss {:.4f}".format(epoch, it + 1, avg_loss))

        train_loss = np.mean(epoch_loss)
        return train_loss


    def _run_eval(self, epoch):
        """ 在验证集上测试效果 """
        self.model.eval()
        if self.use_ddp:
            self.val_data.sampler.set_epoch(epoch)

        with torch.no_grad():
            epoch_loss = []
            for batch in self.val_data:
                loss = self.model(xt=batch.xt.to(self.device), 
                                  xg=batch.xg.to(self.device),
                                  t=batch.t.to(self.device),
                                  nids=self.nids, g=self.g)

                # 所有进程损失的均值
                if self.use_ddp:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss /= self.world_size

                epoch_loss.append(loss.item())

        val_loss = np.mean(epoch_loss)
        return val_loss
            

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data)).xt)
        if self._check_device():
            self.logger.info("\n[GPU {}] Epoch: {} | Batchsize: {} | Steps: {} | LR: {}".
                             format(self.device, epoch, b_sz, len(self.train_data), self.optimizer.param_groups[0]['lr'] if epoch > 0 else 0.0))

        # ------------ Training ------------
        train_loss = self._run_train(epoch)

        # ------------ Evaluation ------------
        val_loss = self._run_eval(epoch)

        if self._check_device():
            self.logger.info("Epoch {} | Training Loss {:.4f} | Validation Loss {:.4f}".format(epoch, train_loss, val_loss))

        return train_loss, val_loss


    def _check_stop(self, rank, early_stopper):
        """ 在多个进程之间同步早停状态 """
        stop = torch.tensor(int(early_stopper.early_stop), dtype=torch.int, device=rank)
        dist.all_reduce(stop, op=dist.ReduceOp.SUM)
        stop = stop.item() > 0  # if one process decides to stop, all processes stop
        return stop


    def train(self, patience=10):
        if not self.use_ddp:
            early_stopper = EarlyStopping(patience=patience)
        else:
            early_stopper = DDPEarlyStopping(patience=patience, rank=self.device)

        for epoch in range(self.max_epochs):
            train_loss, val_loss = self._run_epoch(epoch)
            if self.device == 0:
                early_stopper(val_loss, self.model, self.save_path, epoch)
            
            stop = self._check_stop(self.device, early_stopper)
            if stop:
                break


