#!/usr/bin/env python3
# coding=utf-8
########################################################################
# Copyright (C) Baidu Ltd. All rights reserved.
########################################################################
"""
Author: zharui
Date: 2023-06-07 14:48:48
LastEditTime: 2023-09-19 20:27:37
LastEditors: zharui@baidu.com
FilePath: /decoupledST/src/main_forecast.py
Description: 预测任务
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as utils
from torch.utils.data import DataLoader

import pdb
import os
import json
import random
import datetime
import argparse
import numpy as np
import pickle as pkl
import pandas as pd

import models
import baselines
from trainer.forecaster import Predictor
from trainer.forecast_dataloader import Forecast_Dataset, SimpleForecastBatch
from utils.train_utils import Logger, transfer_weights
from utils.data_utils import Single_Data_Container, get_normalized_adj

# import torch.multiprocessing as mp
# from torch.utils.data.distributed import DistributedSampler
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.distributed import init_process_group, destroy_process_group, broadcast_object_list
# import torch.distributed as dist


def set_seed(seed: int):
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed(seed)  # current GPU
    torch.cuda.manual_seed_all(seed)  # all GPU
    # torch.backends.cudnn.deterministic = True  # CuDNN Alogorithm
    # torch.backends.cudnn.benchmark = True  # Using the same CuDNN Alogorithm
    np.random.seed(seed)
    random.seed(seed)


def get_args():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--city", default="yizhuang", type=str, help="specific fine-tune dataset")
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
    parser.add_argument("--h_dim", default=128, type=int)
    parser.add_argument("--pretrain_model", default="DeSTR", type=str)
    parser.add_argument("--pretrain_path", default="../save/pretrain/xxx", type=str)
    parser.add_argument("--forecast_model", default="ForecastModel", type=str)
    parser.add_argument("--mask_ratio", default=0.2, type=float, help="spatial mask ratio.")
    parser.add_argument("--mask_mode", default="spacetime", type=str, help="temporal mask ratio.")
    parser.add_argument("--variant", default=None, type=str, help="variant model")
    
    # Train
    parser.add_argument("--save_path", default="../save/forecast", type=str)
    parser.add_argument("--max_epochs", default=100, type=int)
    parser.add_argument("--warmup_epochs", default=10, type=int)
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--min_lr", default=5e-4, type=float)
    parser.add_argument("--weight_decay", default=0.001, type=float)
    parser.add_argument("--load_from_pretrain", action="store_true", help="use pre-trained model")
    parser.add_argument('--note', type=str, default=None, help='Note for the current model training')
    
    args = parser.parse_args()
    return args


def get_data(args):
    """ 单个城市数据加载器 """
    data_dict = {
        args.city: {
            "flow": args.flow_file,
            "rel": args.rel_file,
            "dtw": args.dtw_file,
            "nfeat": args.nfeat_file
        }
    }

    data = Single_Data_Container(data_path=args.data_path,
                                 data_dict=data_dict,
                                 look_back=args.look_back,
                                 pred_len=args.pred_len,
                                 graph=args.graph,
                                 time_encode=args.time_encode)

    train_data = Forecast_Dataset(data, mode="train")
    val_data = Forecast_Dataset(data, mode="val")
    test_data = Forecast_Dataset(data, mode="test")

    def collate_fn(batch):
        return SimpleForecastBatch(batch)
    
    # train_dataloader = DataLoader(train_data, args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    # val_dataloader = DataLoader(val_data, args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn) 
    # test_dataloader = DataLoader(test_data, args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    train_dataloader = DataLoader(train_data, args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_data, args.batch_size, shuffle=True, collate_fn=collate_fn) 
    test_dataloader = DataLoader(test_data, args.batch_size, shuffle=False, collate_fn=collate_fn)

    return train_dataloader, val_dataloader, test_dataloader 


def get_model(args, device):
    # forecast model
    forecast_model = getattr(models, args.forecast_model)
    forecast_model = forecast_model(h_dim=args.h_dim, 
                                    pred_dim=args.pred_len,
                                    variant=args.variant).to(device)

    # transfer parameters from pre-train model
    if args.load_from_pretrain:
        pretrain_model = getattr(models, args.pretrain_model)
        pretrain_model = pretrain_model(h_dim=args.h_dim,
                                        seq_len=args.look_back,
                                        mask_ratio=args.mask_ratio,
                                        mask_mode=args.mask_mode)
        
        pretrain_model.load_state_dict(torch.load(args.pretrain_path, map_location=device))
        forecast_model.encoder.load_state_dict(pretrain_model.encoder.state_dict())

    return forecast_model


def get_patchtst(args, num_nodes):
    PatchTST = getattr(baselines, args.pretrain_model)
    PatchTST = PatchTST(c_in=num_nodes, patch_len=4, num_patch=6, head_type="forecast", target_dim=args.pred_len)
    transfer_weights(args.pretrain_path, PatchTST)
    for param in PatchTST.parameters():
        param.requires_grad = True  
    return PatchTST


def main(args):
    set_seed(2023)
    device = torch.device("cuda:0")

    # ------------- DataSet & Model & Optimizer -------------
    train_dataloader, val_dataloader, test_dataloader = get_data(args)
    num_nodes = len(train_dataloader.dataset.nids)

    if args.forecast_model == "ForecastModel":
        model = get_model(args, device)
    
    if args.forecast_model == "PatchTST":
        model = get_patchtst(args, num_nodes).to(device)

    elif args.forecast_model == "DLinear":
        DLinear = getattr(baselines, "DLinear")
        model = DLinear(args.look_back, args.pred_len, num_nodes).to(device)
    
    elif args.forecast_model == "TCN":
        TCN = getattr(baselines, "TCN")
        model = TCN(1, [32, 64, 128], args.pred_len).to(device)
    
    elif args.forecast_model == "FCLSTM":
        FCLSTM = getattr(baselines, "FCLSTM")
        model = FCLSTM(input_dim=1, hidden_dim=args.h_dim, output_dim=args.pred_len).to(device)

    elif args.forecast_model == "AGCRN":
        AGCRN = getattr(baselines, "AGCRN")
        model = AGCRN(num_nodes, args).to(device)

    elif args.forecast_model == "ASTGCN":
        adj_mx = train_dataloader.dataset.adj
        adj_mx[adj_mx > 0] = 1

        model = baselines.make_astgcn(device, adj_mx, num_for_predict=args.pred_len, 
                                      len_input=args.look_back, num_of_vertices=num_nodes).to(device)

    elif args.forecast_model == "DSTAGNN":
        adj_mx = train_dataloader.dataset.adj
        
        adj_TMD = pd.read_csv(os.path.join(args.data_path, args.city, "stag.csv"), header=None)
        adj_TMD = adj_TMD.to_numpy().astype(np.float32)

        adj_pa = pd.read_csv(os.path.join(args.data_path, args.city, "strg.csv"), header=None)
        adj_pa = adj_pa.to_numpy().astype(np.float32)

        model = baselines.load_dstagnn(device, adj_mx, adj_TMD, adj_pa, len_input=args.look_back, 
                                       num_for_predict=args.pred_len, num_of_vertices=num_nodes)
    
    elif args.forecast_model == "STGODE":
        sp_matrix = train_dataloader.dataset.adj
        dtw_matrix = np.load(os.path.join(args.data_path, args.city, "dtw_distance.npy"))
        A_se_hat = get_normalized_adj(dtw_matrix).to(device)
        A_sp_hat = get_normalized_adj(sp_matrix).to(device)
        
        STGODE = getattr(baselines, "STGODE")
        model = STGODE(num_nodes, 1, args.look_back, args.pred_len, A_sp_hat, A_se_hat).to(device)
    
    
    # ------------- Training -------------
    lr = args.lr
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)

    ct = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger = Logger(log_name="Forecast Log", log_file=f"../log/forecast/{args.city}_{args.forecast_model}_{ct}.log")
    save_path = os.path.join(args.save_path, f"{args.city}_{args.forecast_model}_{ct}.pth")
    
    logger.info(f'Running with arguments: {json.dumps(vars(args), indent=4)}')
    logger.info('params number(M) of %s: %.2f' % (args.forecast_model, n_parameters / 1.e6))

    trainer = Predictor(model, device, train_dataloader, val_dataloader, test_dataloader, optimizer, lr,
                        logger, save_path, min_lr=args.min_lr, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs)
    trainer.train()


if __name__ == "__main__":
    args = get_args()
    print(args)

    main(args)

   


