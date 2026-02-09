from sys import set_asyncgen_hooks
import torch
import torch.nn as nn

torch.manual_seed(1)
from model import SDE_RUL_Adapter 
from data_process.data_processing2 import *
from visualize import *
from data_process.loader import *
from torch.utils.data import DataLoader
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from utils.logger import init_logger
import argparse
import time
import logging
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import pdb
import numpy as np
import random
from torch.utils.data.sampler import SubsetRandomSampler
from itertools import cycle

def Training(opt):
    PATH = opt.path + "-" + opt.dataset
    logger = init_logger(opt.save_path, opt, True)
    WRITER = SummaryWriter(log_dir=opt.save_path)

    dataset = opt.dataset
    num_epochs = opt.epoch
    batch_size = opt.batch_size
    train_seq_len = opt.train_seq_len
    LR = opt.LR
    smooth_param = opt.smooth_param
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    group_train, y_test, group_test, X_test = data_processing(dataset, smooth_param)
    print("data processed")

    train_dataset = SequenceDataset(mode='train', group=group_train, sequence_train=train_seq_len,
                                    patch_size=train_seq_len, apply_irregularity=True)

    test_dataset = SequenceDataset(mode='test', group=group_test, y_label=y_test, sequence_train=train_seq_len,
                                   patch_size=train_seq_len, apply_irregularity=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print("train loaded")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, drop_last=False)
    print("test loaded")

    if opt.path == '':
        PATH = "train-model-" + time.strftime("%m-%d-%H:%M:%S", time.localtime()) + ".pth"

    logger.cprint("------Train-------")
    logger.cprint("------" + PATH + "-------")

    print("Initializing SDE_RUL_Adapter...")
    train_model = SDE_RUL_Adapter(opt)

    if torch.cuda.is_available():
        train_model = train_model.to(device)

    for p in train_model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    criterion = torch.nn.MSELoss(reduction="mean")
    
    optimization = torch.optim.Adam(filter(lambda p: p.requires_grad, train_model.parameters()), lr=LR,
                                    weight_decay=opt.weight_decay)

    sde_kl_weight = 1e-4 

    best_test_rmse1 = 10000
    final_best_test_rmse = 10000

    for epoch in range(num_epochs):
        train_model.train()
        train_epoch_loss = 0
        iter_num = 0

        for X, y, mask, times in train_loader:
            iter_num += 1

            if torch.cuda.is_available():
                X = X.cuda()
                y = y.cuda()
                mask = mask.cuda()
                times = times.cuda()

            y_norm = y / 125.0 if y.max() > 1.0 else y
            y_pred = train_model.forward(X, obs_times=times, mask=mask, y_target=y_norm)

            mse_loss = criterion(y_pred.view(-1), y.view(-1))
            
            if hasattr(train_model, 'aux_loss'):
                total_loss = mse_loss + train_model.aux_loss
            else:
                total_loss = mse_loss

            optimization.zero_grad()
            total_loss.backward()
            optimization.step()

            train_epoch_loss += mse_loss.item()

            if iter_num % 10 == 0:
                train_model.eval()
                with torch.no_grad():
                    test_epoch_loss = 0
                    res = 0
                    for X_val, y_val, mask_val, times_val in test_loader:
                        if torch.cuda.is_available():
                            X_val = X_val.cuda()
                            y_val = y_val.cuda()
                            mask_val = mask_val.cuda()
                            times_val = times_val.cuda()

                        y_hat_recons = train_model.forward(X_val, obs_times=times_val, mask=mask_val)

                        y_hat_unscale = y_hat_recons * 125

                        subs = y_hat_unscale.view(-1) - y_val.view(-1)
                        subs = subs.cpu().detach().numpy()

                        for sub in subs:
                            if sub < 0:
                                res = res + np.exp(-sub / 13) - 1
                            else:
                                res = res + np.exp(sub / 10) - 1

                        loss = criterion(y_hat_unscale.view(-1), y_val.view(-1))
                        test_epoch_loss += loss.item()
                        
                    test_loss = torch.sqrt(torch.tensor(test_epoch_loss / len(test_loader)))
                    WRITER.add_scalar('Test loss', test_loss, epoch)

                    if iter_num % 100 == 0:
                         logger.cprint(f"Epoch {epoch} Iter {iter_num}: Train Loss {mse_loss.item():.4f}, Test RMSE {test_loss:.4f}")

                    if epoch >= 10 and test_loss < best_test_rmse1:
                        best_test_rmse1 = test_loss
                        best_score = res
                        cur_best = train_model.state_dict()
                        best_model_path = PATH + "_new_best.pth"
                        torch.save(cur_best, best_model_path)
                        logger.cprint(">> New Best (Batch): RMSE %1.5f, Score %1.5f" % (test_loss, res))

                train_model.train()

        train_epoch_loss = np.sqrt(train_epoch_loss / len(train_loader))
        WRITER.add_scalar('Train RMSE', train_epoch_loss, epoch)
        train_model.eval()
        with torch.no_grad():
            test_epoch_loss = 0
            res = 0

            for X_val, y_val, mask_val, times_val in test_loader:
                if torch.cuda.is_available():
                    X_val = X_val.cuda()
                    y_val = y_val.cuda()
                    mask_val = mask_val.cuda()
                    times_val = times_val.cuda()

                y_hat_recons = train_model.forward(X_val, obs_times=times_val, mask=mask_val)
                y_hat_unscale = y_hat_recons * 125

                subs = y_hat_unscale.view(-1) - y_val.view(-1)
                subs = subs.cpu().detach().numpy()

                for sub in subs:
                    if sub < 0:
                        res = res + np.exp(-sub / 13) - 1
                    else:
                        res = res + np.exp(sub / 10) - 1

                loss = criterion(y_hat_unscale.view(-1), y_val.view(-1))
                test_epoch_loss += loss.item()

            test_loss = torch.sqrt(torch.tensor(test_epoch_loss / len(test_loader)))

            if epoch >= 10 and test_loss < final_best_test_rmse:
                final_best_test_rmse = test_loss
                cur_best = train_model.state_dict()
                torch.save(cur_best, PATH + "final_new_best.pth")
                logger.cprint(">> New Final Best: RMSE %1.5f" % (test_loss))

        logger.cprint("Epoch: %d, Train RMSE: %1.5f, Test RMSE: %1.5f" % (epoch, train_epoch_loss, test_loss))

    return
