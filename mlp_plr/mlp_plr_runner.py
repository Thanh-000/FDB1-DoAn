from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from mlp_plr.mlp_plr_model import MLPPLRModel


DEFAULT_MLP_PLR_FEATURES = [
    "TransactionAmt",
    "LogAmt",
    "Hour",
    "DayOfWeek",
    "Amt_cents",
    "C1_div_C14",
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C11",
    "C14",
    "D1",
    "D2",
    "D6",
    "D7",
    "D8",
    "D13",
    "D14",
    "TimeDays",
]

TIME_FEATURE = "TimeDays"


class ArrayDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x.astype(np.float32, copy=False))
        self.y = torch.from_numpy(y.astype(np.float32, copy=False))

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


@dataclass
class FoldTabularBundle:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    feature_cols: list[str]


def load_ieee_train(data_dir: str | Path) -> pd.DataFrame:
    data_dir = Path(data_dir)
    tx_path = data_dir / "train_transaction.csv"
    id_path = data_dir / "train_identity.csv"
    if not tx_path.exists() or not id_path.exists():
        raise FileNotFoundError(f"Missing IEEE train files in {data_dir}")
    df = pd.read_csv(tx_path)
    df_id = pd.read_csv(id_path)
    return df.merge(df_id, how="left", on="TransactionID")


def base_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    for c in cat_cols:
        df[c] = df[c].fillna("UNKNOWN")

    if "TransactionDT" in df.columns:
        df = df.sort_values("TransactionDT").reset_index(drop=True)
        df["Hour"] = (df["TransactionDT"] / 3600).astype(int) % 24
        df["DayOfWeek"] = (df["TransactionDT"] / 86400).astype(int) % 7
        df["TimeDays"] = (df["TransactionDT"] / 86400.0).astype(np.float32)

    if "TransactionAmt" in df.columns:
        df["LogAmt"] = np.log1p(df["TransactionAmt"])
        df["Amt_cents"] = np.round(df["TransactionAmt"] - np.floor(df["TransactionAmt"]), 2).astype(np.float32)

    if "C1" in df.columns and "C14" in df.columns:
        df["C1_div_C14"] = (df["C1"] / (df["C14"] + 1e-6)).astype(np.float32)

    num_cols = [c for c in DEFAULT_MLP_PLR_FEATURES if c in df.columns]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def build_fold_tabular(
    df_tr: pd.DataFrame,
    df_vl: pd.DataFrame,
    *,
    use_time_feature: bool = True,
) -> FoldTabularBundle:
    feature_cols = [c for c in DEFAULT_MLP_PLR_FEATURES if c in df_tr.columns and c in df_vl.columns]
    if not use_time_feature:
        feature_cols = [c for c in feature_cols if c != TIME_FEATURE]
    if not feature_cols:
        raise ValueError("No MLP-PLR feature columns available")

    medians = df_tr[feature_cols].median()
    x_tr = df_tr[feature_cols].fillna(medians)
    x_vl = df_vl[feature_cols].fillna(medians)

    scaler = StandardScaler()
    x_tr = scaler.fit_transform(x_tr).astype(np.float32)
    x_vl = scaler.transform(x_vl).astype(np.float32)

    return FoldTabularBundle(
        x_train=x_tr,
        y_train=df_tr["isFraud"].to_numpy(dtype=np.float32, copy=False),
        x_val=x_vl,
        y_val=df_vl["isFraud"].to_numpy(dtype=np.float32, copy=False),
        feature_cols=feature_cols,
    )


def train_fold_mlp_plr(
    data_dir: str | Path,
    *,
    fold_index: int = 0,
    n_splits: int = 5,
    epochs: int = 10,
    batch_size: int = 2048,
    hidden_dim: int = 256,
    n_blocks: int = 4,
    n_frequencies: int = 8,
    lr: float = 1e-3,
    use_time_feature: bool = True,
) -> dict[str, float]:
    df = base_preprocess(load_ieee_train(data_dir))
    y = df["isFraud"].to_numpy()
    cv = TimeSeriesSplit(n_splits=n_splits)
    splits = list(cv.split(np.zeros(len(y)), y))
    tr_i, vl_i = splits[fold_index]

    df_tr = df.iloc[tr_i].reset_index(drop=True).copy()
    df_vl = df.iloc[vl_i].reset_index(drop=True).copy()
    print(f"[MLP-PLR] Fold {fold_index + 1}/{n_splits}")
    print(f"[MLP-PLR] Train: {len(df_tr)}, Val: {len(df_vl)}")
    print(f"[MLP-PLR] use_time_feature: {use_time_feature}")

    bundle = build_fold_tabular(df_tr, df_vl, use_time_feature=use_time_feature)
    print(f"[MLP-PLR] feature_cols ({len(bundle.feature_cols)}): {bundle.feature_cols}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"[MLP-PLR] Device: cuda ({torch.cuda.get_device_name(0)})")
    else:
        print("[MLP-PLR] Device: cpu")

    train_ds = ArrayDataset(bundle.x_train, bundle.y_train)
    val_ds = ArrayDataset(bundle.x_val, bundle.y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=(device.type == "cuda"))
    train_loader_eval = DataLoader(train_ds, batch_size=batch_size, shuffle=False, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=(device.type == "cuda"))

    model = MLPPLRModel(
        num_features=len(bundle.feature_cols),
        hidden_dim=hidden_dim,
        n_blocks=n_blocks,
        n_frequencies=n_frequencies,
    ).to(device)

    pos = max(float(bundle.y_train.sum()), 1.0)
    neg = max(float(len(bundle.y_train) - bundle.y_train.sum()), 1.0)
    pos_weight = torch.tensor([neg / pos], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        losses = []
        for x, yb in train_loader:
            x = x.to(device, non_blocking=(device.type == "cuda"))
            yb = yb.to(device, non_blocking=(device.type == "cuda"))
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        print(f"[MLP-PLR] Epoch {epoch + 1}/{epochs} loss={np.mean(losses):.4f}")

    def _predict(loader: DataLoader) -> np.ndarray:
        model.eval()
        preds = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device, non_blocking=(device.type == "cuda"))
                preds.append(torch.sigmoid(model(x)).cpu().numpy())
        return np.concatenate(preds)

    train_probs = _predict(train_loader_eval)
    val_probs = _predict(val_loader)

    result = {
        "train_auprc": float(average_precision_score(bundle.y_train, train_probs)),
        "val_auprc": float(average_precision_score(bundle.y_val, val_probs)),
        "val_roc_auc": float(roc_auc_score(bundle.y_val, val_probs)),
        "val_f1": float(f1_score(bundle.y_val, val_probs >= 0.5)),
        "val_precision": float(precision_score(bundle.y_val, val_probs >= 0.5, zero_division=0)),
        "val_recall": float(recall_score(bundle.y_val, val_probs >= 0.5, zero_division=0)),
    }
    print("[MLP-PLR] Metrics")
    print(f"  train_auprc: {result['train_auprc']:.4f}")
    print(f"  val_auprc: {result['val_auprc']:.4f}")
    print(f"  val_roc_auc: {result['val_roc_auc']:.4f}")
    print(f"  val_f1@0.5: {result['val_f1']:.4f}")
    print(f"  val_precision@0.5: {result['val_precision']:.4f}")
    print(f"  val_recall@0.5: {result['val_recall']:.4f}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument("--n-frequencies", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no-time-feature", action="store_true")
    args = parser.parse_args()

    train_fold_mlp_plr(
        args.data_dir,
        fold_index=args.fold_index,
        n_splits=args.n_splits,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        n_blocks=args.n_blocks,
        n_frequencies=args.n_frequencies,
        lr=args.lr,
        use_time_feature=not args.no_time_feature,
    )


if __name__ == "__main__":
    main()
