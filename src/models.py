import math
import torch
from torch import nn
import torch.nn.functional as F

class Multimodal_Teacher(nn.Module):
    def __init__(self, latent_dim, enc_drop=0.3, clf_drop=0.3, ablation=None):
        super(Multimodal_Teacher, self).__init__()
        self.ablation = ablation
        if self.ablation is None:
            projector_dim = latent_dim*4
        else:
            projector_dim = latent_dim*3
            
        self.ecfp_enc = nn.Sequential(
            nn.Dropout(enc_drop),
            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.ELU(),
            nn.Dropout(enc_drop),
            nn.Linear(512, latent_dim)
        )
        self.clin_enc = nn.Sequential(
            nn.Dropout(enc_drop),
            nn.Linear(32, 512),
            nn.BatchNorm1d(512),
            nn.ELU(),
            nn.Dropout(enc_drop),
            nn.Linear(512, latent_dim)
        )
        self.prop_enc = nn.Sequential(
            nn.Dropout(enc_drop),
            nn.Linear(12, 256),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Dropout(enc_drop),
            nn.Linear(256, latent_dim)
        )
        self.ptnt_enc = nn.Sequential(
            nn.Dropout(enc_drop),
            nn.Linear(12, 256),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Dropout(enc_drop),
            nn.Linear(256, latent_dim)
        )
        self.layer1 = nn.Linear(projector_dim, 128)
        self.layer2 = nn.Linear(128, 8)
        self.layer3 = nn.Linear(8, 2)
        
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(8)
        
        self.drop = nn.Dropout(clf_drop)

    def forward(self, x):
        ecfp_x = x[:,:128]
        clin_x = x[:,128:160]
        prop_x = x[:,160:172]
        ptnt_x = x[:,172:]
        
        if self.ablation == None:
            ecfp = self.ecfp_enc(ecfp_x)
            clin = self.clin_enc(clin_x)
            prop = self.prop_enc(prop_x)
            ptnt = self.ptnt_enc(ptnt_x)
            x = torch.cat([ecfp, clin, prop, ptnt], dim=1)
        elif self.ablation == 'ECFP':
            clin = self.clin_enc(clin_x)
            prop = self.prop_enc(prop_x)
            ptnt = self.ptnt_enc(ptnt_x)
            x = torch.cat([clin, prop, ptnt], dim=1)
        elif self.ablation == 'Clinical':
            ecfp = self.ecfp_enc(ecfp_x)
            prop = self.prop_enc(prop_x)
            ptnt = self.ptnt_enc(ptnt_x)
            x = torch.cat([ecfp, prop, ptnt], dim=1)
        elif self.ablation == 'Property':
            ecfp = self.ecfp_enc(ecfp_x)
            clin = self.clin_enc(clin_x)
            ptnt = self.ptnt_enc(ptnt_x)
            x = torch.cat([ecfp, clin, ptnt], dim=1)
        elif self.ablation == 'Patent':
            ecfp = self.ecfp_enc(ecfp_x)
            clin = self.clin_enc(clin_x)
            prop = self.prop_enc(prop_x)
            x = torch.cat([ecfp, clin, prop], dim=1)
        else:
            print('Check the ablation type')
        x = self.layer1(x) 
        x = self.bn1(x)
        x = F.elu(x)
        x = self.drop(x) 
        x = self.layer2(x)
        x = self.bn2(x)
        emb = F.elu(x)
        x = self.layer3(emb) 
        
        return emb, x
    
class SMILES_BERT(nn.Module):
	def __init__(self, vocab_size, max_len=256, feature_dim=1024, nhead=4, feedforward_dim=1024, nlayers=6, adj=False, dropout_rate=0):
		super(SMILES_BERT, self).__init__()
		self.embedding = Smiles_embedding(vocab_size, feature_dim, max_len, adj=adj)
		trans_layer = nn.TransformerEncoderLayer(feature_dim, nhead, feedforward_dim, activation='gelu', dropout=dropout_rate)
		self.transformer_encoder = nn.TransformerEncoder(trans_layer, nlayers)

	def forward(self, src, pos_num, adj_mask=None, adj_mat=None):
		mask = (src == 0)
		mask = mask.type(torch.bool)

		x = self.embedding(src, pos_num, adj_mask, adj_mat)
		x = self.transformer_encoder(x.transpose(1,0), src_key_padding_mask=mask)
		x = x.transpose(1,0)
		return x

class Smiles_embedding(nn.Module):
	def __init__(self, vocab_size, embed_size, max_len, adj=False):
		super().__init__()
		self.token = nn.Embedding(vocab_size, embed_size, padding_idx=0)
		self.position = nn.Embedding(max_len, embed_size)
		self.max_len = max_len
		self.embed_size = embed_size
		if adj:
			self.adj = Adjacency_embedding(max_len, embed_size)

		self.embed_size = embed_size

	def forward(self, sequence, pos_num, adj_mask=None, adj_mat=None):
		x = self.token(sequence) + self.position(pos_num)
		if adj_mat is not None:
			# additional embedding matrix. need to modify
			#print(adj_mask.shape)
			x += adj_mask.unsqueeze(2) * self.adj(adj_mat).repeat(1, self.max_len).reshape(-1,self.max_len, self.embed_size)
		return x
    
class Adjacency_embedding(nn.Module):
	def __init__(self, input_dim, model_dim, bias=True):
		super(Adjacency_embedding, self).__init__()

		self.weight_h = nn.Parameter(torch.Tensor(input_dim, model_dim))
		self.weight_a = nn.Parameter(torch.Tensor(input_dim))
		if bias:
			self.bias = nn.Parameter(torch.Tensor(model_dim))
		else:
			self.register_parameter('bias', None)
		self.reset_parameters()

	def reset_parameters(self):
		stdv = 1. / math.sqrt(self.weight_h.size(1))
		stdv2 = 1. /math.sqrt(self.weight_a.size(0))
		self.weight_h.data.uniform_(-stdv, stdv)
		self.weight_a.data.uniform_(-stdv2, stdv2)
		if self.bias is not None:
			self.bias.data.uniform_(-stdv, stdv)

	def forward(self, input_mat):
		a_w = torch.matmul(input_mat, self.weight_h)
		out = torch.matmul(a_w.transpose(1,2), self.weight_a)

		if self.bias is not None:
			out += self.bias
		#print(out.shape)
		return out
    
class SMILES_Student(nn.Module):
    def __init__(self, encoder, latent_dim):
        super(SMILES_Student, self).__init__()
        self.encoder = encoder
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Linear(128, 8),
            nn.BatchNorm1d(8),
            nn.ELU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(8, 8),
            nn.BatchNorm1d(8),
            nn.ELU(),
            nn.Linear(8, 2)
        )
        
    def forward(self, x, pos_num, adj_mask=None, adj_mat=None):
        x   = self.encoder(x, pos_num, adj_mask, adj_mat)
        emb = self.projector(x[:,0,:])
        out = self.classifier(emb)
        return emb, out
    
class FP_Student(torch.nn.Module):
    def __init__(self, ecfp_dim, hd1, hd2, hd3, drop1=0.2, drop2=0.2):
        super(FP_Student, self).__init__()
        self.encoder_1 = nn.Sequential(
            nn.Dropout(drop1),
            nn.Linear(ecfp_dim, hd1),
            nn.BatchNorm1d(hd1),
            nn.ELU(),
            nn.Dropout(drop1),
            nn.Linear(hd1, hd2),
            nn.BatchNorm1d(hd2),
            nn.ELU(),
            nn.Linear(hd2, 32)
        )
        
        self.encoder_2 = nn.Sequential(
            nn.Linear(32, hd3),
            nn.BatchNorm1d(hd3),
            nn.Dropout(drop2),
            nn.ELU(),
            nn.Linear(hd3, 8),
            nn.BatchNorm1d(8),
            nn.Dropout(drop2),
            nn.ELU()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(8, 8),
            nn.BatchNorm1d(8),
            nn.ELU(),
            nn.Linear(8, 2)
        )

    def forward(self, x):
        x = self.encoder_1(x)
        emb = self.encoder_2(x)
        out = self.classifier(emb)
        return emb, out