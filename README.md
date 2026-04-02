# Digital Financial Transaction Fraud Detection

This repository contains the working code and notebooks for a fraud-detection research project on:

- `PaySim`
- `IEEE-CIS Fraud Detection`

## Main contents

- [MVS_XAI_Colab_IEEE_CIS.ipynb](./MVS_XAI_Colab_IEEE_CIS.ipynb): main IEEE-CIS research notebook
- [MVS_XAI_Colab_DataPrep_Phase1.ipynb](./MVS_XAI_Colab_DataPrep_Phase1.ipynb): preprocessing and preparation notebook
- [Colab_Phase2_Sequence_Graph.py](./Colab_Phase2_Sequence_Graph.py): sequence/graph experimentation script
- [MVS_XAI_Dashboard.py](./MVS_XAI_Dashboard.py): dashboard prototype

## GNN branch

The repository also includes a standalone heterogeneous GNN prototype for IEEE-CIS:

- [ieee_gnn_graph.py](./ieee_gnn_graph.py)
- [ieee_gnn_model.py](./ieee_gnn_model.py)
- [ieee_gnn_train.py](./ieee_gnn_train.py)
- [ieee_gnn_runner.py](./ieee_gnn_runner.py)
- [IEEE_GNN_Colab_Benchmark.ipynb](./IEEE_GNN_Colab_Benchmark.ipynb)
- [IEEE_GNN_COLAB.md](./IEEE_GNN_COLAB.md)

## Notes

- Raw datasets are intentionally excluded from Git because they are too large for GitHub.
- Colab is the preferred execution environment for the notebooks and the GNN benchmark.
