import os
import argparse
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', help="processed dataset path", type=str, default='./dataset/processed_data')
    parser.add_argument('--save_path', help="model save path", type=str, default='./model/ChemAP')
    parser.add_argument('--enc_dim_1', help='Encoder hidden dim 1', type=int, default=1024)
    parser.add_argument('--enc_dim_2', help='Encoder hidden dim 2', type=int, default=128)
    parser.add_argument('--enc_dim_3', help='Encoder hidden dim 3', type=int, default=256)
    parser.add_argument('--enc_drop_1', help='Encoder dropout rate 1', type=float, default=0.21)
    parser.add_argument('--enc_drop_2', help='Encoder dropout rate 2', type=float, default=0.11)
    parser.add_argument("--KD", help="Knowledge distillation", default=True)
    parser.add_argument("--teacher_path", help="pretrained multimodal teacher saved path", default='./model/Teacher')
    parser.add_argument("--t_dim", help="teacher latent dim", type=int, default=32)
    parser.add_argument("--t_enc_drop", help="teacher encoder dropout rate", type=float, default=0.43)
    parser.add_argument("--t_clf_drop", help="teacher classifier dropout rate", type=float, default=0.17)
    parser.add_argument('--gpu', help="gpu device", type=int, default=0)
    parser.add_argument('--batch_size', help="batchsize", type=int, default=128)
    parser.add_argument("--epochs", help="epochs", type=int, default=2000)
    parser.add_argument("--lr", help="Learning rate", default=0.005)
    parser.add_argument("--tau", help="temperature", default=10)
    parser.add_argument("--alpha", help="CE loss weight", default=0.33)
    parser.add_argument("--beta", help="feature KD loss weight", default=2.21)
    parser.add_argument("--gamma", help="logit KD loss weight", default=1.21)
    parser.add_argument("--seed", type=int, default=7)
    arg = parser.parse_args()
    
    device = torch.device(f"cuda:{arg.gpu}" if torch.cuda.is_available() else "cpu")
    
    # result save dir
    if arg.KD == False:
        model_save_dir = f'{arg.save_path}/ECFP_predictor_wo_KD'
    elif arg.KD == True:
        model_save_dir = f'{arg.save_path}/ECFP_predictor'
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

    train_dataset = Dataset(train, device, model_type='ECFP_Student')
    valid_dataset = Dataset(valid, device, model_type='ECFP_Student')
    test_dataset  = Dataset(test, device, model_type='ECFP_Student')

    train_loader = DataLoader(train_dataset, batch_size=arg.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=arg.batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=arg.batch_size, shuffle=False)

    # load pretrained teacher model
    teacher_model = AveragedModel(Multimodal_Teacher(arg.t_dim, enc_drop=arg.t_enc_drop, clf_drop=arg.t_clf_drop)).to(device)
    teacher_model.load_state_dict(torch.load(f'{arg.teacher_path}/Teacher_{arg.seed}.pt', map_location=device))

    # student Model
    fp_student = FP_Student(2048, arg.enc_dim_1, arg.enc_dim_2, arg.enc_dim_3, arg.enc_drop_1, arg.enc_drop_2).to(device)
    fp_student.apply(xavier_init)

    optimizer = optim.AdamW([{'params':fp_student.parameters()}], lr=arg.lr, weight_decay=1e-6)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

    ce_fn = nn.CrossEntropyLoss()
    mse_fn = nn.MSELoss()
    dis_fn = DistillationLoss(reduction='batchmean', temperature=arg.tau)

    #print('Start student model training')

    best_val_auc = 0.0
    patience = 0
    for epoch in range(arg.epochs):
        fp_student.train()
        teacher_model.eval()
        for i, data in enumerate(train_loader, 0):
            vec, ecfp_2048, y = data
            t_embed, t_output = teacher_model(vec)
            fp_embed, fp_output = fp_student(ecfp_2048)
            # CE loss
            ce_loss = ce_fn(fp_output, y)
            # total loss
            if arg.KD is True:
                mse_loss = mse_fn(fp_embed, t_embed)
                soft_loss = dis_fn(fp_output, t_output)
                loss = arg.alpha*ce_loss + arg.beta*mse_loss + arg.gamma*soft_loss
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

    ### save model
    fp_student.load_state_dict(best_model_state)
    torch.save(fp_student.state_dict(), f'{model_save_dir}/ECFP_predictor_{arg.seed}.pt')
    print('2D fragment predictor saved'
    
#     ### model evalutaion
#     pred_list = []
#     prob_list = []
#     target_list = []

#     fp_student.eval()
#     with torch.no_grad():
#         for i, data in enumerate(test_loader, 0):
#             _, ecfp_2048, y = data
#             _, output = fp_student(ecfp_2048)
#             pred = torch.argmax(F.softmax(output, dim=1), dim=1).detach().cpu()
#             prob = F.softmax(output, dim=1)[:,1].detach().cpu()
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

#     print('2D fragment predictor AUROC: ', metrics.auc(fpr, tpr))
    
if __name__ == "__main__":
    main()
