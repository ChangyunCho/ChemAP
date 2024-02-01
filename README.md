ChemAP
=============
Chemical structure-based Drug Approval Prediction

![model1](img/model_overview.png)

Setup
-------------
First, clone this repository and move to the directory.

    git clone https://github.com/ChangyunCho/ChemAP.git
    cd ChemAP

Prerequisites
-------------
ChemAP training and evaluation were tested for the following python packages and versions.

  - `python`=3.7.11
  - `pytorch`=1.10.0
  - `rdkit`=2022.09.5
  - `numpy`=1.21.2
  - `pandas`=1.3.4
  - `scipy`=1.0.1
  
Example codes
-------------

# Usage for data processing
    python data_processing.py --data_path ./dataset --save_path ./dataset/processed_data --split Drug --seed 7

# Training multi-modal Teacher model
    python Teacher.py --seed 7

# Training ChemAP
ChemAP is consist with two chemical structure-based predictors.
Each predictor is trained individually, and the final drug approval prediction is made by soft-voting the drug approval probability of each model.

## Training SMILES-based predictor
For training SMILES-based predictor, pre-trained ChemBERT [Github link](https://github.com/HyunSeobKim/CHEM-BERT) model is required. 
First, download the pre-trained ChemBERT model using link [here](https://drive.google.com/file/d/1-8oAIwKowGy89w-ZjvCGSc1jsCWNS1Fw/view?usp=sharing).
Second, save the pre-trained model in the following directory './model/ChemBERT/'
    
    python SMILES_predictor.py --seed 7

## Training 2D fragment-based predictor
    python FP_predictor.py --seed 7

Usage for 2023 FDA approved drug list
-------------
    python ChemAP.py --data_type FDA --seed 7

Usage for user provided drug list
-------------
    python ChemAP.py --data_type custom --input_file example.csv --output example --seed 7