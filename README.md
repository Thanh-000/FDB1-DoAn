# Digital Financial Transaction Fraud Detection

This repository contains the working code and notebooks for a fraud-detection research project on:

- `PaySim`
- `IEEE-CIS Fraud Detection`

## Main contents

- [notebooks/MVS_XAI_Colab_IEEE_CIS.ipynb](./notebooks/MVS_XAI_Colab_IEEE_CIS.ipynb): main IEEE-CIS research notebook
- [notebooks/MVS_XAI_Colab_DataPrep_Phase1.ipynb](./notebooks/MVS_XAI_Colab_DataPrep_Phase1.ipynb): preprocessing and preparation notebook
- [scripts/MVS_XAI_Dashboard.py](./scripts/MVS_XAI_Dashboard.py): dashboard prototype

## Current architecture

The current project backbone is centered on the IEEE-CIS notebook and uses:

- `notebooks/`: Colab notebooks
- `scripts/`: support scripts and dashboard code
- `docs/`: notes and proposal files

The active modeling path is:

- fold-local preprocessing
- temporal and aggregate tabular features
- graph-derived motif features
- `XGBoost + LightGBM + CatBoost`
- `XGB` meta-learner
- UID post-processing
- threshold policy for balanced or high-recall operation

## Notes

- Raw datasets are intentionally excluded from Git because they are too large for GitHub.
- Colab is the preferred execution environment for the notebooks.

## Run on Colab

### Main IEEE-CIS notebook

1. Open [notebooks/MVS_XAI_Colab_IEEE_CIS.ipynb](./notebooks/MVS_XAI_Colab_IEEE_CIS.ipynb) in Google Colab.
2. Set `Runtime -> Change runtime type -> GPU`.
3. Upload or mount access to the IEEE-CIS dataset folder:
   - `ieee-fraud-detection/train_transaction.csv`
   - `ieee-fraud-detection/train_identity.csv`
   - `ieee-fraud-detection/test_transaction.csv`
   - `ieee-fraud-detection/test_identity.csv`
4. Update any dataset path variables in the notebook if your Drive structure is different.
5. Run cells from top to bottom.
