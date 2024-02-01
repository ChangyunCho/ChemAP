import os
import argparse
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', help="Data type", type=str, default='DrugApp')
    parser.add_argument('--data_path', help="processed dataset path", type=str, default='./dataset/processed_data')
    parser.add_argument('--custom_file', help="user file", type=str, default='./dataset/example.csv')
    parser.add_argument('--output', help="output path", type=str, default='example')
    parser.add_argument('--model_path', help="trained model path", type=str, default='./model/ChemAP')
    parser.add_argument('--fp_dim_1', help='2D fragment predictor hidden dim 1', type=int, default=1024)
    parser.add_argument('--fp_dim_2', help='2D fragment predictor hidden dim 2', type=int, default=128)
    parser.add_argument('--fp_dim_3', help='2D fragment predictor hidden dim 3', type=int, default=256)
    parser.add_argument('--fp_drop_1', help='2D fragment predictor dropout rate 1', type=float, default=0.21)
    parser.add_argument('--fp_drop_2', help='2D fragment predictor dropout rate 2', type=float, default=0.11)
    parser.add_argument("--KD", help="Knowledge distillation", default=True)
    parser.add_argument('--gpu', help="gpu device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    arg = parser.parse_args()
    
    device = torch.device(f"cuda:{arg.gpu}" if torch.cuda.is_available() else "cpu")
    
    if arg.data_type == 'DrugApp':
        print('Predict drug approval with DrugApp dataset')
    elif arg.data_type == 'FDA':
        print('Predict drug approval with 2023 FDA approved list')
    elif arg.data_type == 'custom':
        print('Predict drug approvl with custom drug list')
    else:
        print('Check data type')
    
    model_saved_dir = './model/ChemAP'
    
    perform_save_dir = f'./results'
    
    # result save dir
    if arg.KD == False:
        KD = '_wo_KD'
    elif arg.KD == True:
        KD = ''
        
    os.makedirs(perform_save_dir, exist_ok=True)

    if arg.data_type == 'FDA':
        f = open(f'{perform_save_dir}/FDA_2023_preds_statistic.csv', 'w', newline='')
        wr = csv.writer(f)
        wr.writerow(['model seed', 'FDA drug number', 'ChemAP pred'])
        f.close()

    # FP model arguments
    fp_enc_h1 = 1024
    fp_enc_h2 = 128
    fp_enc_d = 0.21
    fp_pro_h1 = 256
    fp_pro_d = 0.11

    # performance records
    roc_ls_t = []
    prc_ls_t = []
    acc_ls_t = []
    pre_ls_t = []
    rec_ls_t = []
    f1_ls_t  = []
    ba_ls_t  = []

    temp = []

    seed_everything(arg.seed)
    Smiles_vocab = Vocab()

    if arg.data_type == 'DrugApp':
        test = pd.read_csv(f'{arg.data_path}/test/DrugApp_seed_{arg.seed}_test_minmax.csv')
        test_dataset = Dataset(test, device, model_type='ChemAP', vocab=Smiles_vocab, seq_len=256)

    elif arg.data_type == 'FDA':
        # trainset load to remove drugs with high similarity
        train = pd.read_csv(f'{arg.data_path}/train/DrugApp_seed_{arg.seed}_train_minmax.csv')
        df = pd.read_csv('./dataset/FDA/FDA_2023_approved.csv').dropna().reset_index(drop=True)
        test_dataset = External_Dataset(Smiles_vocab, df, 'FDA', device, trainset=train, similarity_cut=0.7)
    elif arg.data_type == 'custom':
        df = pd.read_csv(arg.custom_file)
        test_dataset = External_Dataset(Smiles_vocab, df, 'custom', device, trainset=None)
    test_loader  = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # load trained predictors 
    # ECFP based student model
    ecfp_student = FP_Student(2048, arg.fp_dim_1, arg.fp_dim_2, arg.fp_dim_3, arg.fp_drop_1, arg.fp_drop_2).to(device)

    ecfp_student.load_state_dict(torch.load(f'{arg.model_path}/ECFP_predictor{KD}/ECFP_predictor_{arg.seed}.pt', map_location=device))

    # SMILES based student model
    smiles_encoder = SMILES_BERT(len(Smiles_vocab), 
                                 max_len=256, 
                                 nhead=16, 
                                 feature_dim=1024, 
                                 feedforward_dim=1024, 
                                 nlayers=8, 
                                 adj=True,
                                 dropout_rate=0)
    smiles_student = SMILES_Student(smiles_encoder, 1024).to(device)

    smiles_student.load_state_dict(torch.load(f'{arg.model_path}/SMILES_predictor{KD}/SMILES_predictor_{arg.seed}.pt', map_location=device))

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

    if arg.data_type == 'DrugApp':
        fpr, tpr, thresholds = metrics.roc_curve(target_list, ens_prob, pos_label=1)
        roc_ls_t.append(metrics.auc(fpr, tpr))
        precision, recall, _ = metrics.precision_recall_curve(target_list, ens_prob, pos_label=1)
        prc_ls_t.append(metrics.auc(recall, precision))
        acc_ls_t.append(metrics.accuracy_score(target_list, ens_pred))
        pre_ls_t.append(metrics.precision_score(target_list, ens_pred, pos_label=1))
        rec_ls_t.append(metrics.recall_score(target_list, ens_pred, pos_label=1))
        f1_ls_t.append(metrics.f1_score(target_list, ens_pred, pos_label=1))
        ba_ls_t.append(metrics.balanced_accuracy_score(target_list, ens_pred))
        
        roc_t = pd.DataFrame(roc_ls_t, columns = ['AUCROC'])
        prc_t = pd.DataFrame(prc_ls_t, columns = ['AUCPRC'])
        acc_t = pd.DataFrame(acc_ls_t, columns = ['ACC'])
        pre_t = pd.DataFrame(pre_ls_t, columns = ['PRE'])
        rec_t = pd.DataFrame(rec_ls_t, columns = ['REC'])
        f1_t  = pd.DataFrame(f1_ls_t, columns = ['F1'])
        ba_t  = pd.DataFrame(ba_ls_t, columns = ['BA'])

        res_t = pd.concat([roc_t, prc_t, acc_t, ba_t, f1_t, pre_t, rec_t], axis=1)
        res_t.to_csv(f'{perform_save_dir}/ChemAP_DrugApp_test_perform.csv', sep = ',', index=None)
        
        print('ChemAP model performance saved')

    elif arg.data_type == 'FDA':
        f = open(f'{perform_save_dir}/FDA_2023_preds_statistic.csv', 'a', newline='')
        wr = csv.writer(f)
        wr.writerow([arg.seed, test_dataset.__len__(), sum(ens_pred)])
        f.close()
        dataset = test_dataset.GetDataset()
        dataset['ChemAP_pred']=ens_pred
        dataset.to_csv(f'{perform_save_dir}/FDA_2023_preds.csv', sep=',', index=None)
    
    elif arg.data_type == 'custom':
        dataset = test_dataset.GetDataset()
        dataset['ChemAP_pred']=ens_pred
        dataset.to_csv(f'{perform_save_dir}/{arg.output}.csv', sep=',', index=None)

if __name__ == "__main__":
    main()
