# ChemAP
Chemical structure-based Drug Approval Prediction

# Usage for data processing
python data_processing.py --data_path ./dataset --save_path ./dataset/processed_data --split Drug --seed 7

# Training multi-modal Teacher model
python Teacher.py --seed 7

# Training ChemAP
ChemAP is consist with two chemical structure-based predictors.
Each predictor is trained individually, and the final drug approval prediction is made by soft-voting the drug approval probability of each model.

### Training SMILES-based predictor
python SMILES_predictor.py --seed 7

### Training 2D fragment-based predictor
python FP_predictor.py --seed 7

# Usage for 2023 FDA approved drug list
python ChemAP.py --data_type FDA --seed 7

# Usage for user provided drug list
python ChemAP.py --data_type custom --input_file example.csv --output example --seed 7