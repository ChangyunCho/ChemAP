import os
import csv
import pickle
import numpy as np 
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.swa_utils import AveragedModel
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.models import Multimodal_Teacher, SMILES_BERT, SMILES_Student
from src.Dataprocessing import Dataset, SMILES_augmentation
from src.loss_function import DistillationLoss
from src.utils import *

from sklearn import metrics

import optuna 
from optuna import Trial, visualization
from optuna.samplers import TPESampler

# dataset path
dataset_name = 'DrugApp_256'
dataset_path = './dataset'
# pretrained chem-bert model path
pretrained_model = './model/pretrained_model.pt'

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def objective(trial: optuna.Trial):
    
#     alpha = trial.suggest_float('alpha', low=0.0, high=2.0, step=0.01)
#     beta = trial.suggest_float('beta', low=0.0, high=2.0, step=0.01)
#     gamma = trial.suggest_float('gamma', low=0.0, high=2.0, step=0.01)
#     tau = trial.suggest_categorical('tau', [0.5, 1, 2, 5, 10])
    
    # train arguments
    num_seeds = 10
    split = 'drug'
    batch = 128
    epochs = 100
    lr = 0.0001
    
    alpha = 1.04
    beta = 0.69
    gamma = 1.34
    tau = 2
    distillation = False
    
    # result save dir
    if distillation == False:
        perform_save_dir = f'./trained_model/SMILES_Student_wo_KD/'
    elif distillation == True:
        perform_save_dir = f'./trained_model/SMILES_Student/'
    os.makedirs(perform_save_dir, exist_ok=True)

    # model arguments
    seq = 256
    embed_size = 1024
    model_dim = 1024
    layers = 8 
    nhead = 16 
    drop_rate = 0
    num_workers = 0
    
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

        train_aug = SMILES_augmentation(train)

        train_dataset = Dataset(train_aug, device, model_type='SMILES_Student', vocab=Smiles_vocab, seq_len=256)
        valid_dataset = Dataset(valid, device, model_type='SMILES_Student', vocab=Smiles_vocab, seq_len=256)
        test_dataset  = Dataset(test, device, model_type='SMILES_Student', vocab=Smiles_vocab, seq_len=256)

        train_loader = DataLoader(train_dataset, batch_size=batch, num_workers=num_workers)
        valid_loader = DataLoader(valid_dataset, batch_size=batch, num_workers=num_workers)
        test_loader  = DataLoader(test_dataset, batch_size=batch, num_workers=num_workers)

        # chem-bert encoder Model 
        smiles_encoder = SMILES_BERT(len(Smiles_vocab), 
                                     max_len=seq, 
                                     nhead=nhead, 
                                     feature_dim=embed_size, 
                                     feedforward_dim=model_dim, 
                                     nlayers=layers, 
                                     adj=True, 
                                     dropout_rate=drop_rate)

        smiles_encoder.load_state_dict(torch.load(pretrained_model, map_location=device))
        for name, param in smiles_encoder.named_parameters():
            if 'layers.7' not in name:
                param.requires_grad_(False)
        smiles_student = SMILES_Student(smiles_encoder, model_dim).to(device)

        # load pretrained teacher model
        teacher_model = AveragedModel(Multimodal_Teacher(32, enc_drop=0.43, clf_drop=0.17)).to(device)
        teacher_model.load_state_dict(torch.load(f'./trained_model/Teacher/Teacher_{seed}.pt', map_location=device))

        optim = AdamW([{'params': smiles_student.parameters()}], lr=lr, weight_decay=1e-6)
        ce_fn = nn.CrossEntropyLoss()
        mse_fn = nn.MSELoss()
        dis_fn = DistillationLoss(reduction='batchmean', temperature=tau)

        for epoch in range(epochs):
            smiles_student.train()
            teacher_model.eval()
            for i, data in enumerate(train_loader):
                vec, smi_bert_input, smi_bert_adj, smi_bert_adj_mask, y = data
                position_num = torch.arange(256).repeat(smi_bert_input.size(0),1).to(device)

                smi_embed, smi_output = smiles_student(smi_bert_input,
                                                        position_num,
                                                        smi_bert_adj_mask,
                                                        smi_bert_adj)

                t_embed, t_output = teacher_model(vec)

                ce_loss = ce_fn(smi_output, y)

                if distillation is True:
                    soft_loss = dis_fn(smi_output, t_output)
                    mse_loss = mse_fn(smi_embed, t_embed)
                    loss = alpha*ce_loss + beta*mse_loss + gamma*soft_loss
                else:
                    loss = ce_loss

                optim.zero_grad()
                loss.backward()
                optim.step()            
        ######################## eval ################################
        #print("Start evaluation with last epoch model.")
        #print("Train set evaluation.")

        pred_list = []
        prob_list = []
        target_list = []

        smiles_student.eval()
        with torch.no_grad():
            for i, data in enumerate(train_loader):
                vec, smi_bert_input, smi_bert_adj, smi_bert_adj_mask, y = data
                position_num = torch.arange(256).repeat(smi_bert_input.size(0),1).to(device)

                smi_embed, smi_output = smiles_student(smi_bert_input,
                                                        position_num,
                                                        smi_bert_adj_mask,
                                                        smi_bert_adj)

                pred = torch.argmax(F.softmax(smi_output, dim=1), dim=1).detach().cpu()
                prob = F.softmax(smi_output, dim=1)[:,1].detach().cpu()
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

        #print("Test set evaluation.")
        pred_list = []
        prob_list = []
        target_list = []

        smiles_student.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                vec, smi_bert_input, smi_bert_adj, smi_bert_adj_mask, y = data
                position_num = torch.arange(256).repeat(smi_bert_input.size(0),1).to(device)

                smi_embed, smi_output = smiles_student(smi_bert_input,
                                                        position_num,
                                                        smi_bert_adj_mask,
                                                        smi_bert_adj)

                pred = torch.argmax(F.softmax(smi_output, dim=1), dim=1).detach().cpu()
                prob = F.softmax(smi_output, dim=1)[:,1].detach().cpu()
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

        torch.save(smiles_student.state_dict(), f'{perform_save_dir}SMILES_student_{seed}.pt')
        #print(f'Model saved: {seed} seed')
        print('AUCROC: ', metrics.auc(fpr, tpr))

    #### save the performance records
    res = pd.concat([pd.DataFrame(roc_ls, columns = ['AUCROC']),
                     pd.DataFrame(prc_ls, columns = ['AUCPRC']),
                     pd.DataFrame(acc_ls, columns = ['ACC']),
                     pd.DataFrame(ba_ls, columns = ['BA']),
                     pd.DataFrame(f1_ls, columns = ['F1']), 
                     pd.DataFrame(pre_ls, columns = ['PRE']),
                     pd.DataFrame(rec_ls, columns = ['REC'])], axis=1)

    res_t = pd.concat([pd.DataFrame(roc_ls_t, columns = ['AUCROC']),
                       pd.DataFrame(prc_ls_t, columns = ['AUCPRC']),
                       pd.DataFrame(acc_ls_t, columns = ['ACC']),
                       pd.DataFrame(ba_ls_t, columns = ['BA']),
                       pd.DataFrame(f1_ls_t, columns = ['F1']), 
                       pd.DataFrame(pre_ls_t, columns = ['PRE']),
                       pd.DataFrame(rec_ls_t, columns = ['REC'])], axis=1)

    res.to_csv(f'{perform_save_dir}train_perform.csv', sep = ',', index=None)
    res_t.to_csv(f'{perform_save_dir}test_perform.csv', sep = ',', index=None)


    #print(res_t['AUCROC'].mean())
    return res_t['AUCROC'].mean()

study = optuna.create_study(direction='maximize', sampler=TPESampler())

#study.optimize(lambda trial : objective(trial), timeout=72*60*60)
study.optimize(lambda trial : objective(trial), n_trials=1)

print('Best trial : score {}, \nparams {}'.format(study.best_trial.value, study.best_trial.params))