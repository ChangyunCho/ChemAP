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

from src.models import Multimodal_Teacher, FP_Student, SMILES_BERT, SMILES_Student
from src.Dataprocessing import Dataset, External_Dataset
from src.loss_function import DistillationLoss
from src.utils import *

from sklearn import metrics

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

infer_data = 'FDA'

num_seeds = 10

model_saved_dir = './trained_model'

distillation = False
    
# result save dir
if distillation == False:
    perform_save_dir = f'./trained_model/ENSEMBLE_Student_wo_KD_{infer_data}/'
    KD = '_wo_KD'
elif distillation == True:
    perform_save_dir = f'./trained_model/ENSEMBLE_Student_{infer_data}/'
    KD = ''
os.makedirs(perform_save_dir, exist_ok=True)

if infer_data == 'FDA':
    f = open(f'{perform_save_dir}FDA_prediction_final.csv', 'w', newline='')
    wr = csv.writer(f)
    wr.writerow(['model seed', 'FDA drug number', 'ECFP model pred', 'SMILES model pred', 'Ensemble model pred'])
    f.close()

# FP model arguments
fp_enc_h1 = 1024
fp_enc_h2 = 128
fp_enc_d = 0.21
fp_pro_h1 = 256
fp_pro_d = 0.11

# SMILES model arguments
seq = 256
embed_size = 1024
model_dim = 1024
layers = 8 
nhead = 16 
drop_rate = 0

# performance records
roc_ls_t = []
prc_ls_t = []
acc_ls_t = []
pre_ls_t = []
rec_ls_t = []
f1_ls_t  = []
ba_ls_t  = []

temp = []

for seed in range(num_seeds):
    seed_everything(seed)
    Smiles_vocab = Vocab()
    
    train = pd.read_csv(f'./dataset/DrugApp_processed_data/DrugApp_256/train/DrugApp_seed_{seed}_train_minmax.csv')
    train_dataset = Dataset(train, device, model_type='ENSEMBLE', vocab=Smiles_vocab, seq_len=256)
    train_loader  = DataLoader(train_dataset, batch_size=256, shuffle=False)
    
    if infer_data == 'Test':
        test = pd.read_csv(f'./dataset/DrugApp_processed_data/DrugApp_256/test/DrugApp_seed_{seed}_test_minmax.csv')
        test_dataset = Dataset(test, device, model_type='ENSEMBLE', vocab=Smiles_vocab, seq_len=256)
        
    elif infer_data == 'FDA':
        test_dataset = External_Dataset(Smiles_vocab, 256, device, nBits=2048, dataset='FDA', trainset=train)
    test_loader  = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # load trained predictors 
    # ECFP based student model
    ecfp_student = FP_Student(2048, fp_enc_h1, fp_enc_h2, fp_pro_h1, fp_enc_d, fp_pro_d).to(device)
    
    ecfp_student.load_state_dict(torch.load(f'{model_saved_dir}/ECFP_Student{KD}/ECFP_student_{seed}.pt', map_location=device))
    
    # SMILES based student model
    smiles_encoder = SMILES_BERT(len(Smiles_vocab), 
                                 max_len=seq, 
                                 nhead=nhead, 
                                 feature_dim=embed_size, 
                                 feedforward_dim=model_dim, 
                                 nlayers=layers, 
                                 adj=True,
                                 dropout_rate=drop_rate)
    smiles_student = SMILES_Student(smiles_encoder, model_dim).to(device)

    smiles_student.load_state_dict(torch.load(f'{model_saved_dir}/SMILES_Student{KD}/SMILES_student_{seed}.pt', map_location=device))
        
    # Inference
    ecfp_pred = []
    ecfp_prob = []
    smi_pred = []
    smi_prob = []
    target_list = []
    
    ecfp_student.eval()
    smiles_student.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            ecfp_2048, smi_bert_input, smi_bert_adj, smi_bert_adj_mask, y = data
            position_num = torch.arange(256).repeat(smi_bert_input.size(0),1).to(device)
            
            _, ecfp_output = ecfp_student(ecfp_2048)
            
            pred = torch.argmax(F.softmax(ecfp_output, dim=1), dim=1).detach().cpu()
            prob = F.softmax(ecfp_output, dim=1)[:,1].detach().cpu()
            ecfp_pred.append(pred)
            ecfp_prob.append(prob)

            _, smiles_output = smiles_student(smi_bert_input,
                                              position_num,
                                              smi_bert_adj_mask,
                                              smi_bert_adj)

            pred = torch.argmax(F.softmax(smiles_output, dim=1), dim=1).detach().cpu()
            prob = F.softmax(smiles_output, dim=1)[:,1].detach().cpu()
            smi_pred.append(pred)
            smi_prob.append(prob)
            
            target_list.append(y.cpu())
              
    target_list = torch.cat(target_list, dim=0).numpy()
    ecfp_pred = torch.cat(ecfp_pred, dim=0).numpy()
    ecfp_prob = torch.cat(ecfp_prob, dim=0).numpy()
    smi_pred  = torch.cat(smi_pred, dim=0).numpy()
    smi_prob  = torch.cat(smi_prob, dim=0).numpy()
    
    ens_prob  = (ecfp_prob + smi_prob)/2
    ens_pred  = (ens_prob > 0.5)*1
    #ens_pred  = (ens_prob > optimal_threshold)*1

    if infer_data == 'Test':
        fpr, tpr, thresholds = metrics.roc_curve(target_list, ens_prob, pos_label=1)
        roc_ls_t.append(metrics.auc(fpr, tpr))
        precision, recall, _ = metrics.precision_recall_curve(target_list, ens_prob, pos_label=1)
        prc_ls_t.append(metrics.auc(recall, precision))
        acc_ls_t.append(metrics.accuracy_score(target_list, ens_pred))
        pre_ls_t.append(metrics.precision_score(target_list, ens_pred, pos_label=1))
        rec_ls_t.append(metrics.recall_score(target_list, ens_pred, pos_label=1))
        f1_ls_t.append(metrics.f1_score(target_list, ens_pred, pos_label=1))
        ba_ls_t.append(metrics.balanced_accuracy_score(target_list, ens_pred))

    elif infer_data == 'FDA':
        f = open(f'{perform_save_dir}FDA_prediction_final.csv', 'a', newline='')
        wr = csv.writer(f)
        wr.writerow([seed, test_dataset.__len__(), sum(ecfp_pred), sum(smi_pred), sum(ens_pred)])
        f.close()
        dataset = test_dataset.GetDataset()
        dataset['ECFP_pred']=ecfp_pred
        dataset['SMILES_prd']=smi_pred
        dataset['ENS_pred']=ens_pred
        dataset.to_csv(f'{perform_save_dir}FDA_prediction_final_seed_{seed}.csv', sep=',', index=None)
            
if infer_data == 'Test':
    roc_t = pd.DataFrame(roc_ls_t, columns = ['AUCROC'])
    prc_t = pd.DataFrame(prc_ls_t, columns = ['AUCPRC'])
    acc_t = pd.DataFrame(acc_ls_t, columns = ['ACC'])
    pre_t = pd.DataFrame(pre_ls_t, columns = ['PRE'])
    rec_t = pd.DataFrame(rec_ls_t, columns = ['REC'])
    f1_t  = pd.DataFrame(f1_ls_t, columns = ['F1'])
    ba_t  = pd.DataFrame(ba_ls_t, columns = ['BA'])

    res_t = pd.concat([roc_t, prc_t, acc_t, ba_t, f1_t, pre_t, rec_t], axis=1)
    res_t.to_csv(f'{perform_save_dir}test_perform.csv', sep = ',', index=None)
    
    print('ENSEMBLE model performance saved')
