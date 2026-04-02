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
