import os
import csv
import pickle
import numpy as np
import pandas as pd

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from torch.optim.swa_utils import AveragedModel

from src.models import Multimodal_Teacher, FP_Student
from src.Dataprocessing import Dataset
from src.loss_function import DistillationLoss
from src.utils import *

from sklearn import metrics

import optuna 
from optuna import Trial, visualization
from optuna.samplers import TPESampler

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def objective(trial: optuna.Trial):
#     alpha = trial.suggest_float('alpha', low=0.0, high=1.0, step=0.01)
#     beta = trial.suggest_float('beta', low=1.5, high=2.5, step=0.01)
#     gamma = trial.suggest_float('gamma', low=0.0, high=1.5, step=0.01)
#     tau = trial.suggest_categorical('tau', [0.5, 1, 2, 5, 10])
    
    #enc_h1 = trial.suggest_categorical('encoder_hidden_1', [256, 512, 1024, 2048])
    #enc_h2 = trial.suggest_categorical('encoder_hidden_2', [128, 256, 512])
    #enc_d   = trial.suggest_float('encoder Dropout', low=0.1, high=0.5, step=0.01)
    
    #pro_h1 = trial.suggest_categorical('projector_hidden_1', [128, 256, 512])
    #pro_d   = trial.suggest_float('projector Dropout', low=0.1, high=0.5, step=0.01)
    
    # train arguments
    
    num_seeds = 10
    split = 'drug'
    batch = 128
    epochs = 2000
    lr     = 0.005
    
    alpha = 0.33
    beta = 2.21
    gamma = 1.21
    tau = 10
    
    distillation = False
    
    # result save dir
    if distillation == False:
        perform_save_dir = f'./trained_model/ECFP_Student_wo_KD/'
    elif distillation == True:
        perform_save_dir = f'./trained_model/ECFP_Student/'
    os.makedirs(perform_save_dir, exist_ok=True)
    
    # model arguments
    enc_h1 = 1024
    enc_h2 = 128
    enc_d = 0.21
    pro_h1 = 256
    pro_d = 0.11 
    
    # performance records
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
        Smiles_vocab = Vocab()
        
        train = pd.read_csv(f'./dataset/DrugApp_processed_data/DrugApp_256/train/DrugApp_seed_{seed}_train_minmax.csv')
        valid = pd.read_csv(f'./dataset/DrugApp_processed_data/DrugApp_256/validation/DrugApp_seed_{seed}_valid_minmax.csv')
        test = pd.read_csv(f'./dataset/DrugApp_processed_data/DrugApp_256/test/DrugApp_seed_{seed}_test_minmax.csv')
        
        train_dataset = Dataset(train, device, model_type='ECFP_Student')
        valid_dataset = Dataset(valid, device, model_type='ECFP_Student')
        test_dataset  = Dataset(test, device, model_type='ECFP_Student')

        train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch, shuffle=True)
        test_loader  = DataLoader(test_dataset, batch_size=batch, shuffle=False)
    
        # load pretrained teacher model
        teacher_model = AveragedModel(Multimodal_Teacher(32, enc_drop=0.43, clf_drop=0.17)).to(device)
        teacher_model.load_state_dict(torch.load(f'./trained_model/Teacher/Teacher_{seed}.pt', map_location=device))
        
        # student Model
        fp_student = FP_Student(2048, enc_h1, enc_h2, pro_h1, enc_d, pro_d).to(device)
        fp_student.apply(xavier_init)

        optimizer = optim.AdamW([{'params':fp_student.parameters()}], lr=lr, weight_decay=1e-6)

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

        ce_fn = nn.CrossEntropyLoss()
        mse_fn = nn.MSELoss()
        dis_fn = DistillationLoss(reduction='batchmean', temperature=tau)

        #print('Start student model training')
        
        best_val_auc = 0.0
        patience = 0
        for epoch in range(epochs):
            fp_student.train()
            teacher_model.eval()
            for i, data in enumerate(train_loader, 0):
                vec, ecfp_2048, y = data
                t_embed, t_output = teacher_model(vec)
                fp_embed, fp_output = fp_student(ecfp_2048)
                # CE loss
                ce_loss = ce_fn(fp_output, y)
                # total loss
                if distillation is True:
                    # KD loss
                    mse_loss = mse_fn(fp_embed, t_embed)
                    soft_loss = dis_fn(fp_output, t_output)
                    loss = alpha*ce_loss + beta*mse_loss + gamma*soft_loss
                else:
                    loss = ce_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # early stopping with valid set
                fp_student.eval()
                teacher_model.eval()
                with torch.no_grad():
                    y_true = []
                    y_pred = []
                    for i, data in enumerate(valid_loader, 0):
                        _, ecfp_2048, y = data
                        _, output = fp_student(ecfp_2048)
                        pred = torch.argmax(F.softmax(output, dim=1), dim=1).detach().cpu().numpy()
                        y_pred.extend(pred)
                        y_true.extend(y.cpu().numpy())
                        
                    val_auc = metrics.roc_auc_score(y_true, y_pred)
                    if val_auc > best_val_auc:
                        best_val_auc = val_auc
                        patience = 0
                        best_model_state = fp_student.state_dict()
                    else:
                        patience += 1
                        
                    if patience >= 20:
                        break
            
            scheduler.step()
                
        ### model evalutaion
        fp_student.load_state_dict(best_model_state)
        
        pred_list = []
        prob_list = []
        target_list = []
        
        fp_student.eval()
        with torch.no_grad():
            for i, data in enumerate(train_loader, 0):
                _, ecfp_2048, y = data
                _, output = fp_student(ecfp_2048)
                
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
        
        fp_student.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader, 0):
                _, ecfp_2048, y = data
                _, output = fp_student(ecfp_2048)
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

        torch.save(fp_student.state_dict(), f'{perform_save_dir}ECFP_student_{seed}.pt')

    roc = pd.DataFrame(roc_ls, columns = ['AUCROC'])
    prc = pd.DataFrame(prc_ls, columns = ['AUCPRC'])
    acc = pd.DataFrame(acc_ls, columns = ['ACC'])
    pre = pd.DataFrame(pre_ls, columns = ['PRE'])
    rec = pd.DataFrame(rec_ls, columns = ['REC'])
    f1  = pd.DataFrame(f1_ls, columns = ['F1'])
    ba  = pd.DataFrame(ba_ls, columns = ['BA'])

    res = pd.concat([roc, prc, acc, ba, f1, pre, rec], axis=1)
    res.to_csv(f'{perform_save_dir}train_perform.csv', sep = ',', index=None)

    roc_t = pd.DataFrame(roc_ls_t, columns = ['AUCROC'])
    prc_t = pd.DataFrame(prc_ls_t, columns = ['AUCPRC'])
    acc_t = pd.DataFrame(acc_ls_t, columns = ['ACC'])
    pre_t = pd.DataFrame(pre_ls_t, columns = ['PRE'])
    rec_t = pd.DataFrame(rec_ls_t, columns = ['REC'])
    f1_t  = pd.DataFrame(f1_ls_t, columns = ['F1'])
    ba_t  = pd.DataFrame(ba_ls_t, columns = ['BA'])

    res_t = pd.concat([roc_t, prc_t, acc_t, ba_t, f1_t, pre_t, rec_t], axis=1)
    res_t.to_csv(f'{perform_save_dir}test_perform.csv', sep = ',', index=None)
    
    return res_t['AUCROC'].mean()

study = optuna.create_study(direction='maximize', sampler=TPESampler())

#study.optimize(lambda trial : objective(trial), timeout=72*60*60)
study.optimize(lambda trial : objective(trial), n_trials=1)

print('Best trial : score {}, \nparams {}'.format(study.best_trial.value, study.best_trial.params))
