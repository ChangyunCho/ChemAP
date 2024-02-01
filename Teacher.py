import os
import argparse
import numpy as np
import pandas as pd

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.models import Multimodal_Teacher
from src.Dataprocessing import Dataset
from src.utils import *

from sklearn import metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', help="processed dataset path", type=str, default='./dataset/processed_data')
    parser.add_argument('--save_path', help="model save path", type=str, default='./model/Teacher')
    parser.add_argument('--gpu', help="gpu device", type=int, default=0)
    parser.add_argument('--batch_size', help="batch size", type=int, default=256)
    parser.add_argument("--epochs", help="epochs", type=int, default=500)
    parser.add_argument("--swa_start", help="SWA start", type=int, default=300)
    parser.add_argument("--lr", help="Learning rate", default=0.01)
    parser.add_argument("--swa_lr", help="SWA Learning rate", default=5e-5)
    parser.add_argument("--latent_dim", help="latent dimension", default=32)
    parser.add_argument("--enc_drop", help="Encoder dropout rate", default=0.43)
    parser.add_argument("--clf_drop", help="Classifier dropout rate", default=0.17)
    parser.add_argument("--seed", type=int, default=7)
    arg = parser.parse_args()
    
    device = torch.device(f"cuda:{arg.gpu}" if torch.cuda.is_available() else "cpu")

    os.makedirs(arg.save_path, exist_ok=True)

    swa_start = 300
    swa_lr = 5e-5

    roc_ls_t = []
    prc_ls_t = []
    acc_ls_t = []
    pre_ls_t = []
    rec_ls_t = []
    f1_ls_t  = []
    ba_ls_t  = []

    seed_everything(arg.seed)

    train = pd.read_csv(f'{arg.data_path}/train/DrugApp_seed_{arg.seed}_train_minmax.csv')
    valid = pd.read_csv(f'{arg.data_path}/valid/DrugApp_seed_{arg.seed}_valid_minmax.csv')
    test = pd.read_csv(f'{arg.data_path}/test/DrugApp_seed_{arg.seed}_test_minmax.csv')

    train_dataset = Dataset(train, device)
    test_dataset  = Dataset(test, device)

    train_loader = DataLoader(train_dataset, batch_size=arg.batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=arg.batch_size, shuffle=False)

    # define test model
    model = Multimodal_Teacher(arg.latent_dim, 
                               enc_drop=arg.enc_drop, 
                               clf_drop=arg.clf_drop, 
                               ablation=None).to(device)
    model_optim = optim.AdamW(model.parameters(), 
                              lr=arg.lr, weight_decay=1e-3)
    model.apply(xavier_init)

    ce_fn = nn.CrossEntropyLoss()

    swa_model = AveragedModel(model)
    scheduler = CosineAnnealingLR(model_optim, T_max=100)
    swa_scheduler = SWALR(model_optim, swa_lr=arg.swa_lr)

    #print('Start student model training')
    for epoch in range(arg.epochs):
        model.train()
        for i, data in enumerate(train_loader, 0):
            vec, y = data
            _, output = model(vec)
            loss = ce_fn(output, y)
            model_optim.zero_grad()
            loss.backward()
            model_optim.step()

        if epoch > arg.swa_start:
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

    torch.save(swa_model.state_dict(), f'{arg.save_path}/Teacher_{arg.seed}.pt')
    print(f'Teacher model saved')

    roc_t = pd.DataFrame(roc_ls_t, columns = ['AUROC'])
    prc_t = pd.DataFrame(prc_ls_t, columns = ['AUPRC'])
    acc_t = pd.DataFrame(acc_ls_t, columns = ['ACC'])
    pre_t = pd.DataFrame(pre_ls_t, columns = ['PRE'])
    rec_t = pd.DataFrame(rec_ls_t, columns = ['REC'])
    f1_t  = pd.DataFrame(f1_ls_t, columns = ['F1'])
    ba_t  = pd.DataFrame(ba_ls_t, columns = ['BA'])

    res_t = pd.concat([roc_t, prc_t, acc_t, ba_t, f1_t, pre_t, rec_t], axis=1)
    res_t.to_csv(f'./results/Teacher_performances_testset_{arg.seed}.csv', sep = ',', index=None)

    print('Teacher AUROC: ', res_t['AUROC'].mean())
    
if __name__ == "__main__":
    main()