# IEEE GNN on Colab

Use this flow to benchmark the standalone heterogeneous GNN branch on Google Colab instead of local Python.

## 1. Runtime

In Colab, set:

- `Runtime` -> `Change runtime type`
- `Hardware accelerator` -> `GPU`

## 2. Mount Drive and open repo

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
%cd "/content/drive/MyDrive/Digital Financial Transaction Fraud Detection Using Explainable Multi-Model Machine Learning on PaySim and IEEE-CIS"
!ls ieee_gnn_*.py
```

If your repo is stored in a different Drive path, change the `%cd` line accordingly.

## 3. Install dependencies

Colab usually already has `torch`. Install the missing graph stack and ML utilities:

```python
import sys
import torch

print("Python:", sys.version)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
```

```python
!pip -q install scikit-learn torch-geometric
```

## 4. Sanity check imports

```python
import torch
import torch_geometric
import sklearn
import pandas as pd

print("torch:", torch.__version__)
print("torch_geometric:", torch_geometric.__version__)
print("sklearn:", sklearn.__version__)
print("pandas:", pd.__version__)
print("CUDA:", torch.cuda.is_available())
```

## 5. Run a one-fold GNN benchmark

Start with the first temporal fold:

```python
!python ieee_gnn_runner.py \
  --data-dir ieee-fraud-detection \
  --fold-index 0 \
  --n-splits 5 \
  --epochs 30 \
  --hidden-dim 64
```

## 6. Run a stronger benchmark

If the first fold runs cleanly, try a slightly stronger setting:

```python
!python ieee_gnn_runner.py \
  --data-dir ieee-fraud-detection \
  --fold-index 0 \
  --n-splits 5 \
  --epochs 50 \
  --hidden-dim 96
```

## 7. What to send back

After the run, keep these lines:

- `[GNN] Train: ...`
- `[GNN] txn_feature_cols (...)`
- `train_auprc`
- `val_auprc`
- `val_roc_auc`

That is enough to decide whether the GNN branch is worth fusing into the main IEEE notebook.

## 8. Recommended next step after a clean run

If standalone GNN has useful signal:

1. Run fold `1` and `2` as well.
2. Compare `val_auprc` against the cleaned tabular temporal pipeline.
3. If competitive, add `gnn_score` back into the main meta-layer as the fourth learner.
