# Digital Financial Transaction Fraud Detection

This repository contains the working code and notebooks for a fraud-detection research project on:

- `PaySim`
- `IEEE-CIS Fraud Detection`

## Main contents

- [notebooks/MVS_XAI_Colab_IEEE_CIS.ipynb](./notebooks/MVS_XAI_Colab_IEEE_CIS.ipynb): main IEEE-CIS research notebook
- [notebooks/MVS_XAI_Colab_DataPrep_Phase1.ipynb](./notebooks/MVS_XAI_Colab_DataPrep_Phase1.ipynb): preprocessing and preparation notebook
- [scripts/Colab_Phase2_Sequence_Graph.py](./scripts/Colab_Phase2_Sequence_Graph.py): sequence/graph experimentation script
- [scripts/MVS_XAI_Dashboard.py](./scripts/MVS_XAI_Dashboard.py): dashboard prototype

## GNN branch

The repository also includes a standalone heterogeneous GNN prototype for IEEE-CIS:

- [gnn/ieee_gnn_graph.py](./gnn/ieee_gnn_graph.py)
- [gnn/ieee_gnn_model.py](./gnn/ieee_gnn_model.py)
- [gnn/ieee_gnn_train.py](./gnn/ieee_gnn_train.py)
- [gnn/ieee_gnn_runner.py](./gnn/ieee_gnn_runner.py)
- [run_ieee_gnn.py](./run_ieee_gnn.py): simple root entrypoint
- [notebooks/IEEE_GNN_Colab_Benchmark.ipynb](./notebooks/IEEE_GNN_Colab_Benchmark.ipynb)
- [docs/IEEE_GNN_COLAB.md](./docs/IEEE_GNN_COLAB.md)

## Repository layout

- `notebooks/`: Colab notebooks
- `gnn/`: standalone hetero GNN branch
- `scripts/`: support scripts and dashboard code
- `docs/`: notes, proposal files, and Colab instructions

## Notes

- Raw datasets are intentionally excluded from Git because they are too large for GitHub.
- Colab is the preferred execution environment for the notebooks and the GNN benchmark.

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

### GNN benchmark notebook

1. Open [notebooks/IEEE_GNN_Colab_Benchmark.ipynb](./notebooks/IEEE_GNN_Colab_Benchmark.ipynb) in Google Colab.
2. Set `Runtime -> Change runtime type -> GPU`.
3. In the config cell, set:
   - `REPO_DIR` to your repository path on Google Drive
   - `DATA_DIR` to the IEEE dataset folder, usually `ieee-fraud-detection`
4. Run the dependency install cell.
5. Run the benchmark cell. It calls:

```bash
python run_ieee_gnn.py --data-dir ieee-fraud-detection --fold-index 0 --n-splits 5 --epochs 30 --hidden-dim 64
```

### Colab references

- Detailed GNN Colab instructions: [docs/IEEE_GNN_COLAB.md](./docs/IEEE_GNN_COLAB.md)
- Root GNN entrypoint: [run_ieee_gnn.py](./run_ieee_gnn.py)
