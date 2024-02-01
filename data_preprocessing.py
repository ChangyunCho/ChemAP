import os
import argparse
import pandas as pd

from src.utils import seed_everything
from src.Dataprocessing import DatasetSplit

from sklearn.preprocessing import MinMaxScaler

def smiles_length(df):
    return len(df['SMILES'])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', help="dataset path", type=str, default = './dataset')
    parser.add_argument('--save_path', help="processed data save path", type=str, default = './dataset/processed_data')
    parser.add_argument("--split", help="data split type", type=str, default='Drug')
    parser.add_argument("--seed", type=int, default=7)
    arg = parser.parse_args()
    
    os.makedirs(f'{arg.save_path}/train', exist_ok=True)
    os.makedirs(f'{arg.save_path}/valid', exist_ok=True)
    os.makedirs(f'{arg.save_path}/test', exist_ok=True)

    total_dataset = pd.read_csv(f'{arg.data_path}/DrugApp/All_training_feature_vectors.csv')

    # remove SMILES length over 256
    total_dataset['length'] = total_dataset.apply(smiles_length, axis=1)
    total_dataset[total_dataset['length'] <= 256].drop(columns='length').reset_index(drop=True)

    # Dataset split (train, valid, test)    
    seed_everything(arg.seed)
    dataset = DatasetSplit(total_dataset, split=arg.split)
    train, valid, test = dataset.data_split()

    # without scaler for training ML models
    train.to_csv(f'{arg.save_path}/train/DrugApp_seed_{arg.seed}_train_no_scaler.csv', sep=',', index=None)
    valid.to_csv(f'{arg.save_path}/valid/DrugApp_seed_{arg.seed}_valid_no_scaler.csv', sep=',', index=None)
    test.to_csv(f'{arg.save_path}/test/DrugApp_seed_{arg.seed}_test_no_scaler.csv', sep=',', index=None)

    # feature scaling for training DL models   
    scaler = MinMaxScaler()
    scaler.fit(train.iloc[:,2:58])
    train = pd.concat([train.iloc[:,:2], pd.DataFrame(scaler.transform(train.iloc[:,2:58]))],axis=1)
    valid = pd.concat([valid.iloc[:,:2], pd.DataFrame(scaler.transform(valid.iloc[:,2:58]))],axis=1)
    test = pd.concat([test.iloc[:,:2], pd.DataFrame(scaler.transform(test.iloc[:,2:58]))],axis=1)
    
    train.to_csv(f'{arg.save_path}/train/DrugApp_seed_{arg.seed}_train_minmax.csv', sep=',', index=None)
    valid.to_csv(f'{arg.save_path}/valid/DrugApp_seed_{arg.seed}_valid_minmax.csv', sep=',', index=None)
    test.to_csv(f'{arg.save_path}/test/DrugApp_seed_{arg.seed}_test_minmax.csv', sep=',', index=None)

    print(f'Data processing with {arg.split} split is done')

if __name__ == "__main__":
    main()