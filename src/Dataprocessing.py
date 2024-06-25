import re
import functools
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix

from src.utils import *

class DatasetSplit():
    def __init__(self, df, split='random'):
        self.data = df
        self.split = split
        
    def data_split(self):       
        if self.split == 'Drug':
            drugs = self.data['SMILES'].unique()
            split1, split2 = int(np.floor(0.1 * len(drugs))), int(np.floor(0.2 * len(drugs)))
            np.random.shuffle(drugs)
            train = pd.merge(pd.DataFrame(drugs[split2:], columns=['SMILES']), self.data, on='SMILES')
            valid = pd.merge(pd.DataFrame(drugs[split1:split2], columns=['SMILES']), self.data, on='SMILES')
            test  = pd.merge(pd.DataFrame(drugs[:split1], columns=['SMILES']), self.data, on='SMILES')
            return train, valid, test
        
        elif self.split == 'random':
            indices = list(range(len(self.data)))
            split1, split2 = int(np.floor(0.1 * len(self.data))), int(np.floor(0.2 * len(self.data)))
            np.random.shuffle(indices)
            train_idx, valid_idx, test_idx = indices[split2:], indices[split1:split2], indices[:split1]
            train = pd.DataFrame(np.asarray(self.data)[train_idx], columns = list(self.data.columns))
            valid = pd.DataFrame(np.asarray(self.data)[valid_idx], columns = list(self.data.columns))
            test = pd.DataFrame(np.asarray(self.data)[test_idx], columns = list(self.data.columns))
            return train, valid, test

class Dataset(Dataset):
    def __init__(self, df, device, nBits=128, model_type='Teacher', vocab=None, seq_len=None):        
        self.device = device
        self.model_type = model_type
        
        self.ecfp = np.array([self.smiles_to_fingerprint(smiles, nBits=nBits) for smiles in df['SMILES'].values])
        if (self.model_type == 'ECFP_Student') or (self.model_type == 'ChemAP') or (self.model_type == 'All'):
            self.ecfp_2048 = np.array([self.smiles_to_fingerprint(smiles, nBits=2048) for smiles in df['SMILES'].values])
        self.clin = df.iloc[:,2:34].to_numpy(dtype=np.float32)
        self.ptnt = df.iloc[:,34:46].to_numpy(dtype=np.float32)
        self.prop = df.iloc[:,46:58].to_numpy(dtype=np.float32)
        self.label = np.asarray(df['Label'])
        
        self.vocab = vocab
        self.atom_vocab = ['C', 'O', 'n', 'c', 'F', 'N', 'S', 's', 'o', 'P', 'R', 'L', 'X', 'B', 'I', 'i', 'p', 'A']
        self.smiles_dataset = []
        self.adj_dataset = []
        self.seq_len = seq_len
        
        if (self.model_type == 'SMILES_Student') or (self.model_type == 'ChemAP') or (self.model_type == 'All'):
            smiles_list = np.asarray(df['SMILES'])
            for i in smiles_list:
                self.adj_dataset.append(i)
                self.smiles_dataset.append(self.replace_halogen(i))

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, idx):
        ecfp = torch.tensor(self.ecfp[idx]).type(torch.float).to(self.device)
        clin = torch.tensor(self.clin[idx]).type(torch.float).to(self.device)
        ptnt = torch.tensor(self.ptnt[idx]).type(torch.float).to(self.device)
        prop = torch.tensor(self.prop[idx]).type(torch.float).to(self.device)
        
        x = torch.cat([ecfp, clin, ptnt, prop], dim=0)
        y = torch.tensor(self.label[idx]).to(self.device)
            
        if self.model_type == 'Teacher':
            return x, y
        
        elif (self.model_type == 'SMILES_Student') or (self.model_type == 'ChemAP') or (self.model_type == 'All'):
            item = self.smiles_dataset[idx]
            input_token, input_adj_masking = self.CharToNum(item)
    
            input_data = [self.vocab.start_index] + input_token + [self.vocab.end_index]
            input_adj_masking = [0] + input_adj_masking + [0]
            
            smiles_bert_input = input_data[:self.seq_len]
            smiles_bert_adj_mask = input_adj_masking[:self.seq_len]
    
            padding = [0 for _ in range(self.seq_len - len(smiles_bert_input))]
            smiles_bert_input.extend(padding)
            smiles_bert_adj_mask.extend(padding)
    
            mol = Chem.MolFromSmiles(self.adj_dataset[idx])
    
            if mol != None:
                adj_mat = GetAdjacencyMatrix(mol)
                smiles_bert_adj_mat = torch.tensor(self.zero_padding(adj_mat, (self.seq_len, self.seq_len))).to(self.device)
            else:
                smiles_bert_adj_mat = torch.tensor(np.zeros((self.seq_len, self.seq_len), dtype=np.float32)).to(self.device)
            
            if self.model_type == 'SMILES_Student':
                return x, torch.tensor(smiles_bert_input).to(self.device), smiles_bert_adj_mat, torch.tensor(smiles_bert_adj_mask).to(self.device), y
            elif self.model_type == 'ChemAP':
                return torch.tensor(self.ecfp_2048[idx]).type(torch.float).to(self.device), torch.tensor(smiles_bert_input).to(self.device), smiles_bert_adj_mat, torch.tensor(smiles_bert_adj_mask).to(self.device), y
            elif self.model_type == 'All':
                return x, torch.tensor(self.ecfp_2048[idx]).type(torch.float).to(self.device), torch.tensor(smiles_bert_input).to(self.device), smiles_bert_adj_mat, torch.tensor(smiles_bert_adj_mask).to(self.device), y
        
        elif self. model_type == 'ECFP_Student':
            return x, torch.tensor(self.ecfp_2048[idx]).type(torch.float).to(self.device), y
    
    def smiles_to_fingerprint(self, smiles, nBits):
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return None
        # Generate a molecular fingerprint (e.g., Morgan fingerprint)
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(molecule, radius=2, nBits=nBits)
        fingerprint_arr = [int(bit) for bit in fingerprint.ToBitString()]
        return fingerprint_arr
    
    def CharToNum(self, smiles):
        tokens = [i for i in smiles]
        adj_masking = []

        for i, token in enumerate(tokens):
            if token in self.atom_vocab:
                adj_masking.append(1)
            else:
                adj_masking.append(0)

            tokens[i] = self.vocab.dict.get(token, self.vocab.unk_index)

        return tokens, adj_masking

    def replace_halogen(self,string):
        """Regex to replace Br and Cl with single letters"""
        br = re.compile('Br')
        cl = re.compile('Cl')
        sn = re.compile('Sn')
        na = re.compile('Na')
        string = br.sub('R', string)
        string = cl.sub('L', string)
        string = sn.sub('X', string)
        string = na.sub('A', string)
        return string

    def zero_padding(self, array, shape):
        if array.shape[0] > shape[0]:
            array = array[:shape[0],:shape[1]]
        padded = np.zeros(shape, dtype=np.float32)
        padded[:array.shape[0], :array.shape[1]] = array
        return padded
    
class External_Dataset(Dataset):
    def __init__(self, vocab, df, dataset, device, trainset=None, similarity_cut=0.7):
        if dataset == 'External':
            ### Process of removing drugs with structures similar to drugs in the training dataset for rigorous evaluation
            if trainset is not None:
                df['similarity'] = 0.0
                for i in range(len(df)):
                    fda_smi = df['SMILES'][i]
                    similarity = []
                    for j in range(len(trainset)):
                        train_smi = trainset['SMILES'][j]
                        similarity.append(calculate_tanimoto_similarity(fda_smi, train_smi))
                    df.loc[i, 'similarity'] = np.max(similarity)

                df = df[df['similarity'] <= similarity_cut].reset_index(drop=True)
            else:
                pass
            
        self.vocab = vocab
        self.atom_vocab = ['C', 'O', 'n', 'c', 'F', 'N', 'S', 's', 'o', 'P', 'R', 'L', 'X', 'B', 'I', 'i', 'p', 'A']
        self.seq_len = 256
        self.smiles_dataset = [self.replace_halogen(smiles) for smiles in df['SMILES'].values]
        self.adj_dataset = [smiles for smiles in df['SMILES'].values]

        self.smiles_list = np.asarray(df['SMILES'])
        self.ecfp_2048 = np.array([self.smiles_to_fingerprint(smiles, nBits=2048) for smiles in df['SMILES'].values])

        self.label = np.ones(len(df))
        
        self.df = df
        self.device = device
		
    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_dataset[idx]
        input_token, input_adj_masking = self.CharToNum(smiles)
    
        input_data = [self.vocab.start_index] + input_token + [self.vocab.end_index]
        input_adj_masking = [0] + input_adj_masking + [0]
            
        smiles_bert_input = input_data[:self.seq_len]
        smiles_bert_adj_mask = input_adj_masking[:self.seq_len]
    
        padding = [0 for _ in range(self.seq_len - len(smiles_bert_input))]
        smiles_bert_input.extend(padding)
        smiles_bert_adj_mask.extend(padding)

        mol = Chem.MolFromSmiles(self.adj_dataset[idx])
    
        if mol != None:
            adj_mat = GetAdjacencyMatrix(mol)
            smiles_bert_adj_mat = torch.tensor(self.zero_padding(adj_mat, (self.seq_len, self.seq_len))).to(self.device)
        else:
            smiles_bert_adj_mat = torch.tensor(np.zeros((self.seq_len, self.seq_len), dtype=np.float32)).to(self.device)

        ecfp_2048 = torch.tensor(self.ecfp_2048[idx]).type(torch.float).to(self.device)
        
        label = torch.tensor(self.label[idx]).to(self.device)

        return ecfp_2048, torch.tensor(smiles_bert_input).to(self.device), smiles_bert_adj_mat, torch.tensor(smiles_bert_adj_mask).to(self.device), label

    def GetDataset(self):
        return self.df
    
    def CharToNum(self, smiles):
        tokens = [i for i in smiles]
        adj_masking = []

        for i, token in enumerate(tokens):
            if token in self.atom_vocab:
                adj_masking.append(1)
            else:
                adj_masking.append(0)

            tokens[i] = self.vocab.dict.get(token, self.vocab.unk_index)

        return tokens, adj_masking

    def replace_halogen(self,string):
        """Regex to replace Br and Cl with single letters"""
        br = re.compile('Br')
        cl = re.compile('Cl')
        sn = re.compile('Sn')
        na = re.compile('Na')
        string = br.sub('R', string)
        string = cl.sub('L', string)
        string = sn.sub('X', string)
        string = na.sub('A', string)
        return string

    def zero_padding(self, array, shape):
        if array.shape[0] > shape[0]:
            array = array[:shape[0],:shape[1]]
        padded = np.zeros(shape, dtype=np.float32)
        padded[:array.shape[0], :array.shape[1]] = array
        return padded
    
    def smiles_to_fingerprint(self, smiles, nBits=128):
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return None
        # Generate a molecular fingerprint (e.g., Morgan fingerprint)
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(molecule, radius=2, nBits=nBits)
        fingerprint_arr = [int(bit) for bit in fingerprint.ToBitString()]
        return fingerprint_arr
    
def SMILES_augmentation(df):
    smiles = df['SMILES'].to_list()
    randomize_func = functools.partial(randomize_smiles, random_type='restricted')
    to_smiles_func = get_smi_func('restricted')  
    randomized_smiles = list(map((lambda smi: to_smiles_func(randomize_func(Chem.MolFromSmiles(smi)))), smiles))
    randomized_smiles = pd.concat([pd.DataFrame(randomized_smiles, columns=['SMILES']), df.iloc[:,1:]], axis=1).dropna()
    augmented = pd.concat([df, randomized_smiles], axis=0).reset_index(drop=True)
    return augmented

def randomize_smiles(mol, random_type="restricted"):
    """
    Returns a random SMILES given a SMILES of a molecule.
    :param mol: A Mol object
    :param random_type: The type (unrestricted, restricted) of randomization performed.
    :return : A random SMILES string of the same molecule or None if the molecule is invalid.
    """
    if not mol:
        return None

    if random_type == "unrestricted":
        return Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=False)
    if random_type == "restricted":
        new_atom_order = list(range(mol.GetNumAtoms()))
        random.shuffle(new_atom_order)
        random_mol = Chem.RenumberAtoms(mol, newOrder=new_atom_order)
        return Chem.MolToSmiles(random_mol, canonical=False, isomericSmiles=False)
    raise ValueError("Type '{}' is not valid".format(random_type))
    
def get_smi_func(smiles_type):
    """
    Returns a function pointer that converts a given SMILES string to SMILES of the given type.
    :param smiles_type: The SMILES type to convert VALUES=(deepsmiles.*, smiles, scaffold).
    :return : A function pointer.
    """
    if smiles_type.startswith("deepsmiles"):
        _, deepsmiles_type = smiles_type.split(".")
        return functools.partial(to_deepsmiles, converter=deepsmiles_type)
    elif smiles_type == "scaffold":
        return add_brackets_to_attachment_points
    else:
        return lambda x: x