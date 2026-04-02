# Digital Financial Transaction Fraud Detection

This repository contains the working code and notebooks for a fraud-detection research project on:

- `PaySim`
- `IEEE-CIS Fraud Detection`

## Main contents

- [notebooks/MVS_XAI_Colab_IEEE_CIS.ipynb](./notebooks/MVS_XAI_Colab_IEEE_CIS.ipynb): main IEEE-CIS research notebook
- [notebooks/MVS_XAI_Colab_DataPrep_Phase1.ipynb](./notebooks/MVS_XAI_Colab_DataPrep_Phase1.ipynb): preprocessing and preparation notebook
- [notebooks/IEEE_TCN_Colab_Benchmark.ipynb](./notebooks/IEEE_TCN_Colab_Benchmark.ipynb): Colab notebook for standalone TCN benchmarking
- [notebooks/IEEE_SCARF_Colab_Benchmark.ipynb](./notebooks/IEEE_SCARF_Colab_Benchmark.ipynb): Colab notebook for standalone SCARF benchmarking
- [notebooks/IEEE_SCARF_Fusion_Colab_Benchmark.ipynb](./notebooks/IEEE_SCARF_Fusion_Colab_Benchmark.ipynb): Colab notebook for SCARF score-fusion benchmarking
- [notebooks/IEEE_SAINT_Colab_Benchmark.ipynb](./notebooks/IEEE_SAINT_Colab_Benchmark.ipynb): Colab notebook for standalone SAINT benchmarking
- [notebooks/IEEE_TABM_Colab_Benchmark.ipynb](./notebooks/IEEE_TABM_Colab_Benchmark.ipynb): Colab notebook for standalone TabM-style benchmarking
- [scripts/MVS_XAI_Dashboard.py](./scripts/MVS_XAI_Dashboard.py): dashboard prototype
- [run_ieee_tcn.py](./run_ieee_tcn.py): standalone TCN sequence benchmark entrypoint
- [run_ieee_scarf.py](./run_ieee_scarf.py): standalone SCARF benchmark entrypoint
- [run_ieee_scarf_fusion.py](./run_ieee_scarf_fusion.py): SCARF score-fusion benchmark entrypoint
- [run_ieee_saint.py](./run_ieee_saint.py): standalone SAINT benchmark entrypoint
- [run_ieee_tabm.py](./run_ieee_tabm.py): standalone TabM-style benchmark entrypoint

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
- The next architecture research phase is documented in [docs/research-roadmap.md](./docs/research-roadmap.md).
- The first candidate extension is specified in [docs/tcn-branch-design.md](./docs/tcn-branch-design.md).
- Experimental exclusions are recorded in [docs/negative-results.md](./docs/negative-results.md).
- The next paper-design options after TCN are documented in [docs/scarf-saint-design.md](./docs/scarf-saint-design.md).
- The controlled integration path for `SCARF` is documented in [docs/scarf-fusion-design.md](./docs/scarf-fusion-design.md).
- The next deep-tabular backbone candidate is documented in [docs/tabm-design.md](./docs/tabm-design.md).

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

### TCN benchmark notebook

1. Open [notebooks/IEEE_TCN_Colab_Benchmark.ipynb](./notebooks/IEEE_TCN_Colab_Benchmark.ipynb) in Google Colab.
2. Set `Runtime -> Change runtime type -> GPU`.
3. Keep the dataset zip in Drive, by default:
   - `/content/drive/MyDrive/MVS_XAI_Data/ieee-fraud-detection.zip`
4. Run cells from top to bottom.
5. The notebook will:
   - clone or pull the repo
   - extract the IEEE dataset zip
   - locate `train_transaction.csv` and `train_identity.csv`
   - run `run_ieee_tcn.py`

### SCARF benchmark notebook

1. Open [notebooks/IEEE_SCARF_Colab_Benchmark.ipynb](./notebooks/IEEE_SCARF_Colab_Benchmark.ipynb) in Google Colab.
2. Set `Runtime -> Change runtime type -> GPU`.
3. Keep the dataset zip in Drive, by default:
   - `/content/drive/MyDrive/MVS_XAI_Data/ieee-fraud-detection.zip`
4. Run cells from top to bottom.
5. The notebook will:
   - clone or pull the repo
   - extract the IEEE dataset zip
   - locate `train_transaction.csv` and `train_identity.csv`
   - run `run_ieee_scarf.py`

### SAINT benchmark notebook

1. Open [notebooks/IEEE_SAINT_Colab_Benchmark.ipynb](./notebooks/IEEE_SAINT_Colab_Benchmark.ipynb) in Google Colab.
2. Set `Runtime -> Change runtime type -> GPU`.
3. Keep the dataset zip in Drive, by default:
   - `/content/drive/MyDrive/MVS_XAI_Data/ieee-fraud-detection.zip`
4. Run cells from top to bottom.
5. The notebook will:
   - clone or pull the repo
   - extract the IEEE dataset zip
   - locate `train_transaction.csv` and `train_identity.csv`
   - run `run_ieee_saint.py`

### SCARF fusion notebook

1. Open [notebooks/IEEE_SCARF_Fusion_Colab_Benchmark.ipynb](./notebooks/IEEE_SCARF_Fusion_Colab_Benchmark.ipynb) in Google Colab.
2. Set `Runtime -> Change runtime type -> GPU`.
3. Keep the dataset zip in Drive, by default:
   - `/content/drive/MyDrive/MVS_XAI_Data/ieee-fraud-detection.zip`
4. Run cells from top to bottom.
5. The notebook will:
   - clone or pull the repo
   - extract the IEEE dataset zip
   - locate `train_transaction.csv` and `train_identity.csv`
   - run `run_ieee_scarf_fusion.py`

### TabM benchmark notebook

1. Open [notebooks/IEEE_TABM_Colab_Benchmark.ipynb](./notebooks/IEEE_TABM_Colab_Benchmark.ipynb) in Google Colab.
2. Set `Runtime -> Change runtime type -> GPU`.
3. Keep the dataset zip in Drive, by default:
   - `/content/drive/MyDrive/MVS_XAI_Data/ieee-fraud-detection.zip`
4. Run cells from top to bottom.
5. The notebook will:
   - clone or pull the repo
   - extract the IEEE dataset zip
   - locate `train_transaction.csv` and `train_identity.csv`
   - run `run_ieee_tabm.py`
