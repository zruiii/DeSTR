#!/usr/bin/env python3
# coding=utf-8
########################################################################
# Copyright (C) Baidu Ltd. All rights reserved. 
########################################################################

"""
Author: zharui
Date: 2023-08-18 11:51:31
LastEditTime: 2023-09-15 19:35:28
LastEditors: zharui@baidu.com
FilePath: /decoupledST/src/trainer/imputer.py
Description: 
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

from utils.train_utils import EarlyStopping

class Imputer:
    """ 
    训练类, 适用于补全任务, 仅单卡
    支持多个测试数据集
    """
    def __init__(self, model, device, train_data, val_datas, test_datas, optimizer, lr,
                 logger, save_path, min_lr=5e-5, warmup_epochs=10, max_epochs=100,
                 mask_imp_weight=0.5, obs_rec_weight=0.5):
        super().__init__()

        self.model = model
        self.device = device
        self.train_data = train_data
        self.val_datas = val_datas
        self.test_data = test_datas
        self.optimizer = optimizer
        self.logger = logger
        self.save_path = save_path
        self.max_epochs = max_epochs
        self.scaler = train_data.dataset.scaler                                     
        self.g = train_data.dataset.g.to(device)                            # 路网图结构
        self.nids = torch.tensor(train_data.dataset.nids).to(device)        # 目标节点 ID

        # warm up
        self.lr = lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs

        # loss
        self.mask_imp_weight = mask_imp_weight
        self.obs_rec_weight = obs_rec_weight

    # * Masked Imputation Loss & Observed Reconstruction Loss
    def _run_batch(self, batch):
        self.optimizer.zero_grad()

        xt = batch.xt.to(self.device)
        mask = batch.mask.to(self.device)
        mask_xt = xt.clone()
        mask_xt[mask] = torch.nan

        pred = self.model(xt=mask_xt, 
                          t=batch.t.to(self.device), 
                          nids=self.nids, g=self.g)

        loss = self.model.get_loss(pred=pred, org=xt,
                                   mask=batch.mask.to(self.device),
                                   mask_imp_weight=self.mask_imp_weight,
                                   obs_rec_weight=self.obs_rec_weight)

        loss.backward()
        self.optimizer.step()
        return loss


    def _run_train(self, epoch):
        self.model.train()

        epoch_loss = []
        start_time = time.time()
        self.logger.info("\nEpoch: {}; LR: {}".format(epoch, self.optimizer.param_groups[0]['lr'] if epoch>0 else 0.0))

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
            epoch_loss.append(loss.item())

            print_every = 20
            if (it + 1) % print_every == 0:
                time_cost = time.time() - start_time
                avg_loss = sum(epoch_loss[-print_every:]) / print_every
                self.logger.info("Epoch {} | Iteration {} | Loss {:.4f} | Time Cost {:.2f}".format(epoch, it+1, avg_loss, time_cost))
                start_time = time.time()
            
        train_loss = np.mean(epoch_loss)
        return train_loss
    

    def _run_eval(self, data_loader):
        self.model.eval()
        pred_li, target_li, mask_li = [], [], []
        with torch.no_grad():
            for _, batch in enumerate(data_loader):
                xt = batch.xt.to(self.device)
                mask = batch.mask.to(self.device)
                mask_xt = xt.clone()
                mask_xt[mask] = torch.nan

                pred = self.model(xt=mask_xt, 
                                  t=batch.t.to(self.device),
                                  nids=self.nids, g=self.g)            

                ######### Inversed Metrics #########
                B, N, L = pred.shape
                pred = pred.cpu().detach().numpy()
                target = batch.org.numpy()
                mask = batch.mask.numpy()

                target = target.transpose(0, 2, 1).reshape(B * L, N)
                pred = pred.transpose(0, 2, 1).reshape(B * L, N)
                mask = mask.transpose(0, 2, 1).reshape(B * L, N)
                pred = self.scaler.inverse_transform(pred)

                pred_li.append(pred)
                target_li.append(target)
                mask_li.append(mask)

            pred = np.concatenate(pred_li)
            target = np.concatenate(target_li)
            mask = np.concatenate(mask_li)
            inv_mae, inv_rmse, inv_smape = self._cal_metrics(pred, target, mask=mask)

        return inv_mae, inv_rmse, inv_smape


    def _run_epoch(self, epoch):
        # -------- training --------
        train_loss = self._run_train(epoch)
        self.logger.info("Epoch: {} | Training Loss {:.4f}".format(epoch, train_loss))

        val_mae, val_rmse = [], []
        for i in range(len(self.val_datas)):
            # -------- evaluation --------
            start_time = time.time()
            inv_mae, inv_rmse, inv_smape = self._run_eval(self.val_datas[i])
            time_cost = time.time() - start_time
            self.logger.info("Val_MAE {:.4f}, Val_RMSE {:.4f}, Val_SMAPE {:.4f} | Time Cost {:.2f}". 
                            format(inv_mae, inv_rmse, inv_smape, time_cost)) 

            # -------- Test --------
            start_time = time.time()
            test_inv_mae, test_inv_rmse, test_inv_smape = self._run_eval(self.test_data[i])
            time_cost = time.time() - start_time
            self.logger.info("Test MAE: {:.4f}, Test RMSE: {:.4f}, Test SMAPE: {:.4f} | Time Cost {:.2f}".
                            format(test_inv_mae, test_inv_rmse, test_inv_smape, time_cost))
            
            val_mae.append(inv_mae)
            val_rmse.append(inv_rmse)

        return val_mae, val_rmse
        
        
    def _cal_metrics(self, source, target, mask=None):
        if mask is not None:
            source = source[mask]
            target = target[mask]
        
        target = target.astype(np.int32)
        mae = np.abs(source - target).mean()
        rmse = np.sqrt(np.mean((source - target) ** 2))
        smape = np.abs(source - target) * 2 / (np.abs(source) + np.abs(target))
        smape = np.mean(smape) * 100
        return mae, rmse, smape
    

    def train(self, case=[30, 50, 70], patience=10):
        num_case = len(case)
        es_container = [EarlyStopping(patience=patience) for _ in range(num_case)]

        flag = 0
        for epoch in range(self.max_epochs):
            mae, rmse = self._run_epoch(epoch)
            for i, early_stop in enumerate(es_container):
                if not early_stop.early_stop:
                    save_path = self.save_path.rsplit('.', 1)[0] + f'_{int(case[i])}.pth'
                    early_stop(mae[i], self.model, save_path, epoch)
                    if early_stop.early_stop:
                        flag += 1

            if flag == num_case:
                break
        
        for i in range(num_case):
            save_path = self.save_path.rsplit('.', 1)[0] + f'_{int(case[i])}.pth'

            # -------- Test --------
            self.model.load_state_dict(torch.load(save_path, map_location=self.device))

            self.model.eval()
            with torch.no_grad():
                test_inv_mae, test_inv_rmse, test_inv_smape = self._run_eval(self.test_data[i])
            self.logger.info("Miss Ratio: {} Best Epoch: {} | Test MAE: {:.4f}, Test RMSE: {:.4f}, Test SMAPE: {:.4f}".
                            format(case[i], es_container[i].best_epoch, test_inv_mae, test_inv_rmse, test_inv_smape))

