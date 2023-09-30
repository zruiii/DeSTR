#!/usr/bin/env python3
# coding=utf-8
########################################################################
# Copyright (C) Baidu Ltd. All rights reserved.
########################################################################
"""
Author: zharui
Date: 2023-06-07 14:48:48
LastEditTime: 2023-09-12 17:24:02
LastEditors: zharui@baidu.com
FilePath: /decoupledST/src/main_impute.py
Description: 补全任务
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
from trainer.imputer import Imputer
from trainer.imputation_dataloader import Imputation_Dataset, SimpleImputeBatch
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
    parser.add_argument("--train_mask", default=0.0, type=float, help="mask ratio for imputation training")
    parser.add_argument("--eval_mask", nargs='+', type=float, help="mask ratio for imputation evluation, such as [30, 50, 70]")

    # Model
    parser.add_argument("--h_dim", default=128, type=int)
    parser.add_argument("--pretrain_model", default="STMAE", type=str)
    parser.add_argument("--pretrain_path", default="../save/pretrain/xxx", type=str)
    parser.add_argument("--impute_model", default="ImputeModel", type=str)
    parser.add_argument("--mask_ratio", default=0.2, type=float, help="spatial mask ratio.")
    parser.add_argument("--mask_mode", default="spacetime", type=str, help="temporal mask ratio.")
    parser.add_argument("--load_from_pretrain", action="store_true", help="use pre-trained model")
    
    # Train
    parser.add_argument("--save_path", default="../save/imputation", type=str)
    parser.add_argument("--max_epochs", default=100, type=int)
    parser.add_argument("--warmup_epochs", default=10, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--min_lr", default=5e-4, type=float)
    parser.add_argument("--weight_decay", default=0.001, type=float)
    parser.add_argument("--mask_imp_weight", default=0.5, type=float)
    parser.add_argument("--obs_rec_weight", default=0.5, type=float)
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

    train_data = Imputation_Dataset(data, mode="train")
    val_data_li = [Imputation_Dataset(data, mode="val", mask_ratio=eval_mask*0.01) for eval_mask in args.eval_mask]
    test_data_li = [Imputation_Dataset(data, mode="test", mask_ratio=eval_mask*0.01) for eval_mask in args.eval_mask]

    def train_collate(batch):
        """ add random mask for each batch data """
        batch = SimpleImputeBatch(batch)
        batch.random_mask(args.train_mask)
        return batch
    
    def test_collate(batch):
        batch = SimpleImputeBatch(batch)
        return batch

    # train_dataloader = DataLoader(train_data, args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=train_collate) 
    # val_dataloaders = [DataLoader(val_data, args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=test_collate) for val_data in val_data_li]
    # test_dataloaders = [DataLoader(test_data, args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=test_collate) for test_data in test_data_li]

    train_dataloader = DataLoader(train_data, args.batch_size, shuffle=True, collate_fn=train_collate) 
    val_dataloaders = [DataLoader(val_data, args.batch_size, shuffle=True, collate_fn=test_collate) for val_data in val_data_li]
    test_dataloaders = [DataLoader(test_data, args.batch_size, shuffle=False, collate_fn=test_collate) for test_data in test_data_li]

    return train_dataloader, val_dataloaders, test_dataloaders


def get_model(args, device):
    # forecast model
    impute_model = getattr(models, args.impute_model)
    impute_model = impute_model(h_dim=args.h_dim, 
                                seq_len=args.look_back).to(device)

    # transfer parameters from pre-train model
    if args.load_from_pretrain:
        pretrain_model = getattr(models, args.pretrain_model)
        pretrain_model = pretrain_model(h_dim=args.h_dim,
                                        seq_len=args.look_back,
                                        mask_ratio=args.mask_ratio,
                                        mask_mode=args.mask_mode)
        
        pretrain_model.load_state_dict(torch.load(args.pretrain_path, map_location=device))
        impute_model.encoder.load_state_dict(pretrain_model.encoder.state_dict())

    return impute_model


def get_patchtst(args, num_nodes):
    PatchTST = getattr(baselines, args.pretrain_model)
    PatchTST = PatchTST(c_in=num_nodes, patch_len=4, num_patch=6, head_type="imputation", target_dim=args.look_back)
    transfer_weights(args.pretrain_path, PatchTST)
    for param in PatchTST.parameters():
        param.requires_grad = True  
    return PatchTST


def main(args):
    set_seed(2023)
    device = torch.device("cuda:0")

    # ------------- DataSet & Model & Optimizer -------------
    train_dataloader, val_dataloaders, test_dataloaders = get_data(args)
    num_nodes = len(train_dataloader.dataset.nids)
    adj = torch.FloatTensor(train_dataloader.dataset.adj)

    if args.impute_model == "ImputeModel":
        model = get_model(args, device)
    if args.impute_model == "PatchTST":
        model = get_patchtst(args, num_nodes).to(device)
    elif args.impute_model == "BRITS":
        BRITS = getattr(baselines, "BRITS")
        model = BRITS(h_dim=args.h_dim, seq_len=args.look_back, num_nodes=num_nodes).to(device)
    elif args.impute_model == "GRIN":
        GRIN = getattr(baselines, "GRIN")
        model = GRIN(adj=adj, h_dim=args.h_dim, d_ff=args.h_dim).to(device)
    elif args.impute_model == "SAITS":
        SAITS = getattr(baselines, "SAITS")
        model = SAITS(d_time=args.look_back, d_model=args.h_dim, d_feature=num_nodes, device=device).to(device)
    elif args.impute_model == "SPIN":
        SPIN = getattr(baselines, "SPIN")
        model = SPIN(input_size=1, hidden_size=args.h_dim, n_nodes=num_nodes, u_size=args.time_dim).to(device)

    # ------------- Training -------------
    lr = args.lr
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)

    ct = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger = Logger(log_name="Impuation Log", log_file=f"../log/imputation/{args.city}_{args.impute_model}_{ct}.log")
    save_path = os.path.join(args.save_path, f"{args.city}_{args.impute_model}_{ct}.pth")
    
    logger.info(f'Running with arguments: {json.dumps(vars(args), indent=4)}')
    logger.info('params number(M) of %s: %.2f' % (args.impute_model, n_parameters / 1.e6))

    trainer = Imputer(model, device, train_dataloader, val_dataloaders, test_dataloaders, optimizer, lr,
                      logger, save_path, min_lr=args.min_lr, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs,
                      mask_imp_weight=args.mask_imp_weight, obs_rec_weight=args.obs_rec_weight)
    trainer.train(case=args.eval_mask)


if __name__ == "__main__":
    args = get_args()
    print(args)

    main(args)

   


