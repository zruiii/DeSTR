#!/usr/bin/env python3
# coding=utf-8
########################################################################
# Copyright (C) Baidu Ltd. All rights reserved.
########################################################################
"""
Author: zharui
Date: 2023-06-07 14:48:48
LastEditTime: 2023-09-18 20:05:38
LastEditors: zharui@baidu.com
FilePath: /decoupledST/src/main_pretrain.py
Description: 分布式训练
"""
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, broadcast_object_list
import torch.distributed as dist

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as utils
from torch.utils.data import DataLoader

import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

import pdb
import json
import random
import argparse
import datetime
import numpy as np

import models
import baselines
from trainer.pretrainer import PreTrainer
from trainer.forecast_dataloader import Forecast_Dataset, collate_fn
from utils.data_utils import Multi_Data_Container
from utils.train_utils import Logger


def set_seed(seed: int):
    """ 设置种子 """
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed(seed)  # current GPU
    torch.cuda.manual_seed_all(seed)  # all GPU
    # torch.backends.cudnn.deterministic = True  # CuDNN Alogorithm
    # torch.backends.cudnn.benchmark = False  # Using the same CuDNN Alogorithm
    np.random.seed(seed)
    random.seed(seed)


def ddp_setup(rank: int, world_size: int):
    """ Constructing the process group """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12365"
    init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def get_args():
    """ 参数项 """
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--city", nargs="+", type=str, help="multi-city pretrain datasets")
    parser.add_argument("--data_path", default="../data", type=str)
    parser.add_argument("--flow_file", default="2023_01_03_flow.csv", type=str)
    parser.add_argument("--rel_file", default="connect.csv", type=str)
    parser.add_argument("--dtw_file", default="dtw_distance.npy", type=str)
    parser.add_argument("--nfeat_file", default="node_feature.csv", type=str)
    parser.add_argument("--look_back", default=24, type=int)
    parser.add_argument("--pred_len", default=0, type=int)
    parser.add_argument("--time_dim", default=4, type=int)
    parser.add_argument("--graph", default="DIST", type=str)
    parser.add_argument("--time_encode", default="raw", type=str)

    # Model
    parser.add_argument("--model", default="DeSTR", type=str)
    parser.add_argument("--h_dim", default=128, type=int)
    parser.add_argument("--mask_mode", default="spacetime", type=str, help="masking strategy.")
    parser.add_argument("--mask_ratio", default=0.2, type=float, help="mask ratio.")
    # parser.add_argument("--spatial_mask", default=0.2, type=float, help="spatial mask ratio.")
    # parser.add_argument("--temporal_mask", default=0.2, type=float, help="temporal mask ratio.")

    # Train
    parser.add_argument("--save_path", default="../save/pretrain", type=str)
    parser.add_argument("--use_ddp", action="store_true")
    parser.add_argument("--max_epochs", default=100, type=int)
    parser.add_argument("--warmup_epochs", default=20, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--min_lr", default=5e-4, type=float)
    parser.add_argument("--weight_decay", default=0.001, type=float)
    args = parser.parse_args()
    return args


def get_data(args):
    """ 单个城市数据加载器 """
    data_dict = dict()
    for city in args.city:
        data_dict.update({
            city: {
                "flow": args.flow_file,
                "rel": args.rel_file,
                "dtw": args.dtw_file,
                "nfeat": args.nfeat_file
            }
        })

    data = Multi_Data_Container(data_path=args.data_path,
                                data_dict=data_dict,
                                look_back=args.look_back,
                                pred_len=args.pred_len,
                                graph=args.graph,
                                time_encode=args.time_encode)

    train_data = Forecast_Dataset(data, mode="train")
    val_data = Forecast_Dataset(data, mode="val")

    if args.use_ddp:
        train_dataloader = DataLoader(train_data, args.batch_size, num_workers=args.num_workers, 
                                      sampler=DistributedSampler(train_data, shuffle=True),
                                      collate_fn=collate_fn)
        val_dataloader = DataLoader(val_data, args.batch_size, num_workers=args.num_workers, 
                                    sampler=DistributedSampler(val_data, shuffle=True),
                                    collate_fn=collate_fn)
    else:
        train_dataloader = DataLoader(train_data, args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
        val_dataloader = DataLoader(val_data, args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    
    return train_dataloader, val_dataloader 


def ddp_main(rank, world_size, args):
    """
    rank: 当前进程在进程组中排名
    world_size: 进程组中总的进程数
    args: 参数类
    """
    # ------------- Initialize the process group -------------
    ddp_setup(rank, world_size)
    set_seed(2023)

    # ------------- DataSet & model & Optimizer -------------
    train_dataloader, val_dataloader = get_data(args)
    num_nodes = len(train_dataloader.dataset.nids)
    
    if args.model == "DeSTR":
        DeSTR = getattr(models, args.model)
        model = DeSTR(h_dim=args.h_dim,
                      seq_len=args.look_back,
                      mask_ratio=args.mask_ratio,
                      mask_mode=args.mask_mode).to(rank)

    elif args.model == "PatchTST":
        PatchTST = getattr(baselines, args.model)
        model = PatchTST(c_in=num_nodes, patch_len=4, num_patch=6).to(rank)

    # pretrain_model = getattr(models, args.model)
    # model = pretrain_model(h_dim=args.h_dim,
    #                        seq_len=args.look_back,
    #                        spatial_mask=args.spatial_mask,
    #                        temporal_mask=args.temporal_mask).to(rank)
    
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    # lr = args.batch_size * args.lr / 256
    lr = args.lr
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)

    # ------------- Training -------------
    ct = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    logger = Logger(log_name="Pretrain Log", log_file=f"../log/pretrain/pretrain_{args.model}_{ct}.log", rank=rank)
    save_path = os.path.join(args.save_path, f"pretrain_{args.model}_{ct}.pth")
    logger.info(f'Running with arguments: {json.dumps(vars(args), indent=4)}')

    trainer = PreTrainer(model, rank, train_dataloader, val_dataloader, lr, optimizer, logger, save_path, 
                         use_ddp=True, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs)
    trainer.train()

    destroy_process_group()


def main(args):
    """
    args: 参数类
    """
    set_seed(2023)
    device = torch.device("cuda:0")

    # ------------- DataSet & model & Optimizer -------------
    train_dataloader, val_dataloader = get_data(args)
    num_nodes = len(train_dataloader.dataset.nids)

    if args.model == "DeSTR":
        DeSTR = getattr(models, args.model)
        model = DeSTR(h_dim=args.h_dim,
                      seq_len=args.look_back,
                      mask_ratio=args.mask_ratio,
                      mask_mode=args.mask_mode).to(device)

    elif args.model == "PatchTST":
        PatchTST = getattr(baselines, args.model)
        model = PatchTST(c_in=num_nodes, patch_len=4, num_patch=6).to(device)

    # lr = args.batch_size * args.lr / 256
    lr = args.lr
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)

    # ------------- Training -------------
    ct = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    logger = Logger(log_name="Pretrain Log", log_file=f"../log/pretrain/pretrain_{args.model}_{ct}.log", rank=None)
    save_path = os.path.join(args.save_path, f"pretrain_{args.model}_{ct}.pth")
    logger.info(f'Running with arguments: {json.dumps(vars(args), indent=4)}')
    
    trainer = PreTrainer(model, device, train_dataloader, val_dataloader, lr, optimizer, logger, save_path, 
                         use_ddp=False, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs)
    trainer.train()


if __name__ == "__main__":
    args = get_args()
    print(args)
    print("\n")

    if args.use_ddp:
        world_size = torch.cuda.device_count()
        print("Availabel Devices Number: {}".format(world_size))
        mp.spawn(ddp_main, args=(world_size, args), nprocs=world_size)
    else:
        main(args)

