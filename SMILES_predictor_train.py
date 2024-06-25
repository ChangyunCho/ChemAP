import os
import argparse
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', help="processed dataset path", type=str, default='./dataset/processed_data')
    parser.add_argument('--save_path', help="model save path", type=str, default='./model/ChemAP')
    parser.add_argument('--chembert_path', help="pretrained chembert path", type=str, default='./model/ChemBERT')
    parser.add_argument("--KD", help="Knowledge distillation", default=True)
    parser.add_argument("--teacher_path", help="pretrained multimodal teacher saved path", default='./model/Teacher')
    parser.add_argument("--t_dim", help="teacher latent dim", type=int, default=32)
    parser.add_argument("--t_enc_drop", help="teacher encoder dropout rate", type=float, default=0.43)
    parser.add_argument("--t_clf_drop", help="teacher classifier dropout rate", type=float, default=0.17)
    parser.add_argument('--gpu', help="gpu device", type=int, default=0)
    parser.add_argument('--batch_size', help="batchsize", type=int, default=128)
    parser.add_argument("--epochs", help="epochs", type=int, default=100)
    parser.add_argument("--lr", help="Learning rate", default=0.0001)
    parser.add_argument("--tau", help="temperature", default=2)
    parser.add_argument("--alpha", help="CE loss weight", default=1.04)
    parser.add_argument("--beta", help="feature KD loss weight", default=0.69)
    parser.add_argument("--gamma", help="logit KD loss weight", default=1.34)
    parser.add_argument("--seed", type=int, default=7)
    arg = parser.parse_args()
    
    device = torch.device(f"cuda:{arg.gpu}" if torch.cuda.is_available() else "cpu")
    
    # result save dir
    if arg.KD == False:
        model_save_dir = f'{arg.save_path}/SMILES_predictor_wo_KD'
    elif arg.KD == True:
        model_save_dir = f'{arg.save_path}/SMILES_predictor'
    os.makedirs(model_save_dir, exist_ok=True)
    
    # performance records
    roc_ls_t = []
    prc_ls_t = []
    acc_ls_t = []
    pre_ls_t = []
    rec_ls_t = []
    f1_ls_t  = []
    ba_ls_t  = []

    seed_everything(arg.seed)
    Smiles_vocab = Vocab()

    train = pd.read_csv(f'{arg.data_path}/train/DrugApp_seed_{arg.seed}_train_minmax.csv')
    valid = pd.read_csv(f'{arg.data_path}/valid/DrugApp_seed_{arg.seed}_valid_minmax.csv')
    test = pd.read_csv(f'{arg.data_path}/test/DrugApp_seed_{arg.seed}_test_minmax.csv')

    train_aug = SMILES_augmentation(train)

    train_dataset = Dataset(train_aug, device, 
                            model_type='SMILES_Student', vocab=Smiles_vocab, seq_len=256)
    valid_dataset = Dataset(valid, device, 
                            model_type='SMILES_Student', vocab=Smiles_vocab, seq_len=256)
    test_dataset  = Dataset(test, device, 
                            model_type='SMILES_Student', vocab=Smiles_vocab, seq_len=256)

    train_loader = DataLoader(train_dataset, batch_size=arg.batch_size, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=arg.batch_size, num_workers=0)
    test_loader  = DataLoader(test_dataset, batch_size=arg.batch_size, num_workers=0)

    # chem-bert encoder Model 
    smiles_encoder = SMILES_BERT(len(Smiles_vocab), 
                                 max_len=256, 
                                 nhead=16, 
                                 feature_dim=1024, 
                                 feedforward_dim=1024, 
                                 nlayers=8, 
                                 adj=True, 
                                 dropout_rate=0)

    smiles_encoder.load_state_dict(torch.load(f'{arg.chembert_path}/pretrained_model.pt', map_location=device))
    for name, param in smiles_encoder.named_parameters():
        if 'layers.7' not in name:
            param.requires_grad_(False)
    smiles_student = SMILES_Student(smiles_encoder, 1024).to(device)

    # load pretrained teacher model
    teacher_model = AveragedModel(Multimodal_Teacher(arg.t_dim, enc_drop=arg.t_enc_drop, clf_drop=arg.t_clf_drop)).to(device)
    teacher_model.load_state_dict(torch.load(f'{arg.teacher_path}/Teacher_{arg.seed}.pt', map_location=device))

    optim = AdamW([{'params': smiles_student.parameters()}], lr=arg.lr, weight_decay=1e-6)
    ce_fn = nn.CrossEntropyLoss()
    mse_fn = nn.MSELoss()
    dis_fn = DistillationLoss(reduction='batchmean', temperature=arg.tau)

    for epoch in range(arg.epochs):
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

            if arg.KD is True:
                soft_loss = dis_fn(smi_output, t_output)
                mse_loss = mse_fn(smi_embed, t_embed)
                loss = arg.alpha*ce_loss + arg.beta*mse_loss + arg.gamma*soft_loss
            else:
                loss = ce_loss

            optim.zero_grad()
            loss.backward()
            optim.step()            
            
    torch.save(smiles_student.state_dict(), f'{model_save_dir}/SMILES_predictor_{arg.seed}.pt')
    print('SMILES predictor saved')
    
#     ######################## eval ################################
#     print("Start SMILES predictor evaluation on the testset")
#     pred_list = []
#     prob_list = []
#     target_list = []

#     smiles_student.eval()
#     with torch.no_grad():
#         for i, data in enumerate(test_loader):
#             vec, smi_bert_input, smi_bert_adj, smi_bert_adj_mask, y = data
#             position_num = torch.arange(256).repeat(smi_bert_input.size(0),1).to(device)

#             smi_embed, smi_output = smiles_student(smi_bert_input,
#                                                     position_num,
#                                                     smi_bert_adj_mask,
#                                                     smi_bert_adj)

#             pred = torch.argmax(F.softmax(smi_output, dim=1), dim=1).detach().cpu()
#             prob = F.softmax(smi_output, dim=1)[:,1].detach().cpu()
#             pred_list.append(pred)
#             prob_list.append(prob)
#             target_list.append(y)

#     pred_list = torch.cat(pred_list, dim=0).numpy()
#     prob_list = torch.cat(prob_list, dim=0).numpy()
#     target_list = torch.cat(target_list, dim=0).cpu().numpy()

#     fpr, tpr, thresholds = metrics.roc_curve(target_list, prob_list, pos_label=1)
#     roc_ls_t.append(metrics.auc(fpr, tpr))
#     precision, recall, _ = metrics.precision_recall_curve(target_list, prob_list, pos_label=1)
#     prc_ls_t.append(metrics.auc(recall, precision))
#     acc_ls_t.append(metrics.accuracy_score(target_list, pred_list))
#     pre_ls_t.append(metrics.precision_score(target_list, pred_list, pos_label=1))
#     rec_ls_t.append(metrics.recall_score(target_list, pred_list, pos_label=1))
#     f1_ls_t.append(metrics.f1_score(target_list, pred_list, pos_label=1))
#     ba_ls_t.append(metrics.balanced_accuracy_score(target_list, pred_list))

#     print('SMILES predictor AUROC: ', metrics.auc(fpr, tpr))

if __name__ == "__main__":
    main()