import os 
import random
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

import torch
from torch import nn 
from torch.nn import functional as F

def seed_everything(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.set_rng_state(torch.cuda.get_rng_state())
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
    
def xavier_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
            
## Based on github.com/codertimo/BERT-pytorch
class Vocab(object):
	def __init__(self):
		self.pad_index = 0
		self.mask_index = 1
		self.unk_index = 2
		self.start_index = 3
		self.end_index = 4

		# check 'Na' later
		self.voca_list = ['<pad>', '<mask>', '<unk>', '<start>', '<end>'] + ['C', '[', '@', 'H', ']', '1', 'O', \
							'(', 'n', '2', 'c', 'F', ')', '=', 'N', '3', 'S', '/', 's', '-', '+', 'o', 'P', \
							 'R', '\\', 'L', '#', 'X', '6', 'B', '7', '4', 'I', '5', 'i', 'p', '8', '9', '%', '0', '.', ':', 'A']

		self.dict = {s: i for i, s in enumerate(self.voca_list)}

	def __len__(self):
		return len(self.voca_list)
    
def calculate_tanimoto_similarity(smiles1, smiles2):
    # Generate Morgan fingerprints for the molecules
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    if mol1 is None or mol2 is None:
        raise ValueError("Invalid SMILES input.")

    fingerprint1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
    fingerprint2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)

    # Calculate Tanimoto similarity
    similarity = DataStructs.TanimotoSimilarity(fingerprint1, fingerprint2)
    return similarity