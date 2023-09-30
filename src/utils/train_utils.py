#!/usr/bin/env python3
# coding=utf-8
########################################################################
# Copyright (C) Baidu Ltd. All rights reserved. 
########################################################################

"""
Author: zharui
Date: 2023-08-07 14:32:06
LastEditTime: 2023-09-12 16:25:43
LastEditors: zharui@baidu.com
FilePath: /decoupledST/src/utils/train_utils.py
Description: 
"""

import torch
import logging


class Logger:
    """ 日志记录 """
    def __init__(self, log_name, log_file, mode='a', rank=None):
        """
        log_name: 日志名称
        log_file: 日志路径
        """
        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(logging.DEBUG)
        self.rank = rank

        if isinstance(rank, int) and rank != 0:
            return 

        print("日志记录器已被初始化在第 {} 个设备上, 日志路径: {}".format(rank, log_file))
        # 创建文件处理器
        file_handler = logging.FileHandler(log_file, mode)
        file_handler.setLevel(logging.INFO)

        # 创建控制台处理器，使得日志可以输出到终端
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # 定义日志格式
        log_format = logging.Formatter('%(message)s')

        # 设置文件处理器和控制台处理器的格式
        file_handler.setFormatter(log_format)
        console_handler.setFormatter(log_format)

        # 将处理器添加到日志记录器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def debug(self, message):
        """ bug """
        if self.rank is None or self.rank == 0:
            self.logger.debug(message)

    def info(self, message):
        """ info """
        if self.rank is None or self.rank == 0:
            self.logger.info(message)

    def warning(self, message):
        """ warning """
        if self.rank is None or self.rank == 0:
            self.logger.warning(message)

    def error(self, message):
        """ error """
        if self.rank is None or self.rank == 0:
            self.logger.error(message)

    def critical(self, message):
        """ critical """
        if self.rank is None or self.rank == 0:
            self.logger.critical(message)


class EarlyStopping:
    """ 单卡训练早停机制 """
    def __init__(self, patience):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False

    def __call__(self, score, model, save_path, epoch=0):
        """EarlyStop function """
        if self.best_score is None or score < self.best_score:
            self.best_score = score
            self.counter = 0
            self.best_epoch = epoch
            self.save_checkpoint(model, save_path, epoch)
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience} with {epoch}')
            if self.counter >= self.patience:
                self.early_stop = True
    
    def save_checkpoint(self, model, save_path, epoch):
        print(f"Save model of epoch {epoch}")
        torch.save(model.state_dict(), save_path)


class DDPEarlyStopping:
    """ 分布式训练早停机制 """
    def __init__(self, patience, rank):
        self.patience = patience
        self.counter = 0
        self.rank = rank
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False

    def __call__(self, score, model, save_path, epoch=0):
        """ score下降 or 间隔固定轮次 => 保存模型, 如果触发早停则结束训练 """
        if self.best_score is None or score < self.best_score:
            self.best_score = score
            self.counter = 0
            self.best_epoch = epoch
            self.save_checkpoint(model, save_path, epoch=None)
        elif epoch % 10 == 0:
            self.save_checkpoint(model, save_path, epoch=epoch)
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience} with {epoch}')
            if epoch > 50 and self.counter >= self.patience:        # 至少训练50EP
                self.early_stop = True
                
    def save_checkpoint(self, model, save_path, epoch=None):
        """ 只在第0个设备上保存模型的 Encoder """
        if self.rank == 0:
            if epoch is not None: save_path = save_path.rsplit(".", 1)[0] + f"_ep{epoch:03}.pth"
            print(f"Save model of epoch {epoch} at {save_path}")
            torch.save(model.module.state_dict(), save_path)
   

# for PatchTST
def transfer_weights(weights_path, model, exclude_head=True, device='cpu'):
    # state_dict = model.state_dict()
    new_state_dict = torch.load(weights_path, map_location=device)
    #print('new_state_dict',new_state_dict)
    matched_layers = 0
    unmatched_layers = []
    for name, param in model.state_dict().items():        
        if exclude_head and 'head' in name: continue
        if name in new_state_dict:            
            matched_layers += 1
            input_param = new_state_dict[name]
            if input_param.shape == param.shape: param.copy_(input_param)
            else: unmatched_layers.append(name)
        else:
            unmatched_layers.append(name)
            pass # these are weights that weren't in the original model, such as a new head
    if matched_layers == 0: raise Exception("No shared weight names were found between the models")
    else:
        if len(unmatched_layers) > 0:
            print(f'check unmatched_layers: {unmatched_layers}')
        else:
            print(f"weights from {weights_path} successfully transferred!\n")
    model = model.to(device)
    return model

