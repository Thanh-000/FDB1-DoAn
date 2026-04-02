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

## 3. Clone or update the repo from GitHub

```python
GITHUB_REPO = "https://github.com/Thanh-000/FDB1-DoAn.git"
REPO_PARENT = "/content/drive/MyDrive"
REPO_NAME = "FDB1-DoAn"
```

```python
import os

repo_path = os.path.join(REPO_PARENT, REPO_NAME)
if not os.path.exists(repo_path):
    !git clone "$GITHUB_REPO" "$repo_path"
else:
    %cd "$repo_path"
    !git pull

print("REPO_DIR:", repo_path)
```

```python
%cd "/content/drive/MyDrive/FDB1-DoAn"
!ls gnn
```

If your repo is stored in a different Drive path, change the `%cd` line accordingly.

## 4. Configure dataset zip and extract

Follow the same dataset pattern used by the main IEEE notebook:

```python
import os, zipfile, glob

ZIP_PATH = "/content/drive/MyDrive/MVS_XAI_Data/ieee-fraud-detection.zip"
EXTRACT_DIR = "/content/ieee-fraud-detection"

if not os.path.exists(EXTRACT_DIR):
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)

txn_files = glob.glob(f"{EXTRACT_DIR}/**/train_transaction.csv", recursive=True)
id_files = glob.glob(f"{EXTRACT_DIR}/**/train_identity.csv", recursive=True)

print("txn:", txn_files[:1])
print("id :", id_files[:1])

DATA_DIR = os.path.dirname(txn_files[0])
print("DATA_DIR =", DATA_DIR)
```

If your zip is stored elsewhere, only change `ZIP_PATH`.

## 5. Install dependencies

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

## 6. Sanity check imports

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

## 7. Run a one-fold GNN benchmark

Start with the first temporal fold:

```python
!python run_ieee_gnn.py \
  --data-dir "$DATA_DIR" \
  --fold-index 0 \
  --n-splits 5 \
  --epochs 30 \
  --hidden-dim 64
```

## 8. Run a stronger benchmark

If the first fold runs cleanly, try a slightly stronger setting:

```python
!python run_ieee_gnn.py \
  --data-dir "$DATA_DIR" \
  --fold-index 0 \
  --n-splits 5 \
  --epochs 50 \
  --hidden-dim 96
```

## 9. What to send back

After the run, keep these lines:

- `[GNN] Train: ...`
- `[GNN] txn_feature_cols (...)`
- `train_auprc`
- `val_auprc`
- `val_roc_auc`

That is enough to decide whether the GNN branch is worth fusing into the main IEEE notebook.

## 10. Recommended next step after a clean run

If standalone GNN has useful signal:

1. Run fold `1` and `2` as well.
2. Compare `val_auprc` against the cleaned tabular temporal pipeline.
3. If competitive, add `gnn_score` back into the main meta-layer as the fourth learner.
