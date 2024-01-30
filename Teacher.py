import os
import sys
import csv
import timeit
import math
import random
import pickle
import numpy as np
import pandas as pd

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR

import sklearn
from sklearn import metrics

from src.models import Multimodal_Teacher
from src.Dataprocessing import Dataset
from src.utils import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataset_name = 'DrugApp_256'
dataset_path = './dataset'

# ablation = None
ablations = ['ECFP', 'Clinical', 'Property', 'Patent', None]
for ablation in ablations:
    if ablation is not None:
        perform_save_dir = f'./trained_model/Teacher_wo_{ablation}/'
    else:
        perform_save_dir = f'./trained_model/Teacher/'
    os.makedirs(perform_save_dir, exist_ok=True)

    latent_dim = 32
    enc_drop= 0.43
    clf_drop= 0.17

    lr     = 0.01
    epochs = 500
    num_seeds = 10
    split = 'drug'

    swa_start = 300
    swa_lr = 5e-5

    roc_ls = []
    prc_ls = []
    acc_ls = []
    pre_ls = []
    rec_ls = []
    f1_ls  = []
    ba_ls  = []

    roc_ls_t = []
    prc_ls_t = []
    acc_ls_t = []
    pre_ls_t = []
    rec_ls_t = []
    f1_ls_t  = []
    ba_ls_t  = []

    for seed in range(num_seeds):
        seed_everything(seed)

        train = pd.read_csv(f'./dataset/DrugApp_processed_data/DrugApp_256/train/DrugApp_seed_{seed}_train_minmax.csv')
        valid = pd.read_csv(f'./dataset/DrugApp_processed_data/DrugApp_256/validation/DrugApp_seed_{seed}_valid_minmax.csv')
        test = pd.read_csv(f'./dataset/DrugApp_processed_data/DrugApp_256/test/DrugApp_seed_{seed}_test_minmax.csv')

        train_dataset = Dataset(train, device)
        test_dataset  = Dataset(test, device)

        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        test_loader  = DataLoader(test_dataset,  batch_size=256, shuffle=False)

        # define test model
        model = Multimodal_Teacher(latent_dim, enc_drop=enc_drop, clf_drop=clf_drop, ablation=ablation).to(device)
        model_optim = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        model.apply(xavier_init)

        ce_fn = nn.CrossEntropyLoss()

        swa_model = AveragedModel(model)
        scheduler = CosineAnnealingLR(model_optim, T_max=100)
        swa_scheduler = SWALR(model_optim, swa_lr=swa_lr)

        #print('Start student model training')
        for epoch in range(epochs):
            model.train()
            for i, data in enumerate(train_loader, 0):
                vec, y = data
                _, output = model(vec)
                loss = ce_fn(output, y)
                model_optim.zero_grad()
                loss.backward()
                model_optim.step()

            if epoch > swa_start:
                swa_model.update_parameters(model)
                swa_scheduler.step()
            else:
                scheduler.step()

        torch.optim.swa_utils.update_bn(train_loader, swa_model)

        pred_list = []
        prob_list = []
        target_list = []

        model.eval()
        swa_model.eval()
        with torch.no_grad():
            for i, data in enumerate(train_loader, 0):
                vec, y = data
                _, output = swa_model(vec)
                pred = torch.argmax(F.softmax(output, dim=1), dim=1).detach().cpu()
                prob = F.softmax(output, dim=1)[:,1].detach().cpu()
                pred_list.append(pred)
                prob_list.append(prob)
                target_list.append(y)

            pred_list = torch.cat(pred_list, dim=0).numpy()
            prob_list = torch.cat(prob_list, dim=0).numpy()
            target_list = torch.cat(target_list, dim=0).cpu().numpy()

            fpr, tpr, thresholds = metrics.roc_curve(target_list, prob_list, pos_label=1)
            roc_ls.append(metrics.auc(fpr, tpr))
            precision, recall, _ = metrics.precision_recall_curve(target_list, prob_list, pos_label=1)
            prc_ls.append(metrics.auc(recall, precision))
            acc_ls.append(metrics.accuracy_score(target_list, pred_list))
            pre_ls.append(metrics.precision_score(target_list, pred_list, pos_label=1))
            rec_ls.append(metrics.recall_score(target_list, pred_list, pos_label=1))
            f1_ls.append(metrics.f1_score(target_list, pred_list, pos_label=1))
            ba_ls.append(metrics.balanced_accuracy_score(target_list, pred_list))

        pred_list = []
        prob_list = []
        target_list = []

        model.eval()
        swa_model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader, 0):
                vec, y = data
                _, output = swa_model(vec)
                pred = torch.argmax(F.softmax(output, dim=1), dim=1).detach().cpu()
                prob = F.softmax(output, dim=1)[:,1].detach().cpu()
                pred_list.append(pred)
                prob_list.append(prob)
                target_list.append(y)

            pred_list = torch.cat(pred_list, dim=0).numpy()
            prob_list = torch.cat(prob_list, dim=0).numpy()
            target_list = torch.cat(target_list, dim=0).cpu().numpy()

            fpr, tpr, thresholds = metrics.roc_curve(target_list, prob_list, pos_label=1)
            roc_ls_t.append(metrics.auc(fpr, tpr))
            precision, recall, _ = metrics.precision_recall_curve(target_list, prob_list, pos_label=1)
            prc_ls_t.append(metrics.auc(recall, precision))
            acc_ls_t.append(metrics.accuracy_score(target_list, pred_list))
            pre_ls_t.append(metrics.precision_score(target_list, pred_list, pos_label=1))
            rec_ls_t.append(metrics.recall_score(target_list, pred_list, pos_label=1))
            f1_ls_t.append(metrics.f1_score(target_list, pred_list, pos_label=1))
            ba_ls_t.append(metrics.balanced_accuracy_score(target_list, pred_list))

        torch.save(swa_model.state_dict(), f'{perform_save_dir}/Teacher_{seed}.pt')
        print(f'Model saved: {seed} seed')

    roc = pd.DataFrame(roc_ls, columns = ['AUCROC'])
    prc = pd.DataFrame(prc_ls, columns = ['AUCPRC'])
    acc = pd.DataFrame(acc_ls, columns = ['ACC'])
    pre = pd.DataFrame(pre_ls, columns = ['PRE'])
    rec = pd.DataFrame(rec_ls, columns = ['REC'])
    f1  = pd.DataFrame(f1_ls, columns = ['F1'])
    ba  = pd.DataFrame(ba_ls, columns = ['BA'])

    res = pd.concat([roc, prc, acc, ba, f1, pre, rec], axis=1)
    res.to_csv(f'{perform_save_dir}/train_perform.csv', sep = ',', index=None)

    roc_t = pd.DataFrame(roc_ls_t, columns = ['AUCROC'])
    prc_t = pd.DataFrame(prc_ls_t, columns = ['AUCPRC'])
    acc_t = pd.DataFrame(acc_ls_t, columns = ['ACC'])
    pre_t = pd.DataFrame(pre_ls_t, columns = ['PRE'])
    rec_t = pd.DataFrame(rec_ls_t, columns = ['REC'])
    f1_t  = pd.DataFrame(f1_ls_t, columns = ['F1'])
    ba_t  = pd.DataFrame(ba_ls_t, columns = ['BA'])

    res_t = pd.concat([roc_t, prc_t, acc_t, ba_t, f1_t, pre_t, rec_t], axis=1)
    res_t.to_csv(f'{perform_save_dir}/test_perform.csv', sep = ',', index=None)

    print(res_t['AUCROC'].mean())