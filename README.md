# ChemAP
Chemical structure-based Drug Approval Prediction

# Usage for data processing

# Training multi-modal Teacher model

# Training ChemAP
ChemAP is consist with two chemical structure-based predictors.
Each predictor is trained individually, and the final drug approval prediction is made by soft-voting the drug approval probability of each model.

### Training SMILES-based predictor
python SMILES_predictor.py

### Training 2D fragment-based predictor
python FP_predictor.py

# Usage for 2023 FDA approved drug list
python ChemAP.py --data_type FDA

# Usage for user provided drug list
python ChemAP.py --data_type custom --input_file example.csv