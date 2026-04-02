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

from scarf.scarf_model import MLPEncoder, SCARFClassifier, SCARFPretrainModel, nt_xent_loss


DEFAULT_SCARF_FEATURES = [
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
]


class ArrayDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray | None = None):
        self.x = torch.from_numpy(x.astype(np.float32, copy=False))
        self.y = None if y is None else torch.from_numpy(y.astype(np.float32, copy=False))

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        if self.y is None:
            return self.x[idx]
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

    if "TransactionAmt" in df.columns:
        df["LogAmt"] = np.log1p(df["TransactionAmt"])
        df["Amt_cents"] = np.round(df["TransactionAmt"] - np.floor(df["TransactionAmt"]), 2).astype(np.float32)

    if "C1" in df.columns and "C14" in df.columns:
        df["C1_div_C14"] = (df["C1"] / (df["C14"] + 1e-6)).astype(np.float32)

    num_cols = [c for c in DEFAULT_SCARF_FEATURES if c in df.columns]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def build_fold_tabular(df_tr: pd.DataFrame, df_vl: pd.DataFrame) -> FoldTabularBundle:
    feature_cols = [c for c in DEFAULT_SCARF_FEATURES if c in df_tr.columns and c in df_vl.columns]
    if not feature_cols:
        raise ValueError("No SCARF feature columns available")

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


def corrupt_batch(x: torch.Tensor, corruption_rate: float = 0.6) -> torch.Tensor:
    bsz, dim = x.shape
    perm = torch.randperm(bsz, device=x.device)
    donor = x[perm]
    mask = torch.rand((bsz, dim), device=x.device) < corruption_rate
    return torch.where(mask, donor, x)


def pretrain_scarf(
    x_train: np.ndarray,
    input_dim: int,
    hidden_dim: int,
    emb_dim: int,
    proj_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    corruption_rate: float,
) -> MLPEncoder:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"[SCARF] Device: cuda ({torch.cuda.get_device_name(0)})")
    else:
        print("[SCARF] Device: cpu")

    ds = ArrayDataset(x_train)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, pin_memory=(device.type == "cuda"))

    model = SCARFPretrainModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        emb_dim=emb_dim,
        proj_dim=proj_dim,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        losses = []
        for x in loader:
            x = x.to(device, non_blocking=(device.type == "cuda"))
            x1 = corrupt_batch(x, corruption_rate=corruption_rate)
            x2 = corrupt_batch(x, corruption_rate=corruption_rate)
            opt.zero_grad()
            z1, z2 = model(x1, x2)
            loss = nt_xent_loss(z1, z2)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        print(f"[SCARF] Pretrain epoch {epoch + 1}/{epochs} loss={np.mean(losses):.4f}")
    return model.encoder.cpu()


def train_downstream(
    encoder: MLPEncoder,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    emb_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    freeze_encoder: bool,
) -> dict[str, float | np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)
    model = SCARFClassifier(encoder=encoder, emb_dim=emb_dim).to(device)

    if freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False

    train_ds = ArrayDataset(x_train, y_train)
    val_ds = ArrayDataset(x_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=(device.type == "cuda"))

    pos = max(float(y_train.sum()), 1.0)
    neg = max(float(len(y_train) - y_train.sum()), 1.0)
    pos_weight = torch.tensor([neg / pos], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)

    for epoch in range(epochs):
        model.train()
        losses = []
        for x, y in train_loader:
            x = x.to(device, non_blocking=(device.type == "cuda"))
            y = y.to(device, non_blocking=(device.type == "cuda"))
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        print(f"[SCARF] Downstream epoch {epoch + 1}/{epochs} loss={np.mean(losses):.4f}")

    def _predict(loader: DataLoader) -> np.ndarray:
        model.eval()
        preds = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device, non_blocking=(device.type == "cuda"))
                preds.append(torch.sigmoid(model(x)).cpu().numpy())
        return np.concatenate(preds)

    train_probs = _predict(train_loader)
    val_probs = _predict(val_loader)
    return {
        "train_probs": train_probs,
        "val_probs": val_probs,
        "train_auprc": float(average_precision_score(y_train, train_probs)),
        "val_auprc": float(average_precision_score(y_val, val_probs)),
        "val_roc_auc": float(roc_auc_score(y_val, val_probs)),
        "val_f1": float(f1_score(y_val, val_probs >= 0.5)),
        "val_precision": float(precision_score(y_val, val_probs >= 0.5, zero_division=0)),
        "val_recall": float(recall_score(y_val, val_probs >= 0.5, zero_division=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--pretrain-epochs", type=int, default=10)
    parser.add_argument("--downstream-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--emb-dim", type=int, default=128)
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--corruption-rate", type=float, default=0.6)
    parser.add_argument("--freeze-encoder", action="store_true")
    args = parser.parse_args()

    df = base_preprocess(load_ieee_train(args.data_dir))
    y = df["isFraud"].to_numpy()
    cv = TimeSeriesSplit(n_splits=args.n_splits)
    splits = list(cv.split(np.zeros(len(y)), y))
    tr_i, vl_i = splits[args.fold_index]

    df_tr = df.iloc[tr_i].reset_index(drop=True).copy()
    df_vl = df.iloc[vl_i].reset_index(drop=True).copy()
    print(f"[SCARF] Fold {args.fold_index + 1}/{args.n_splits}")
    print(f"[SCARF] Train: {len(df_tr)}, Val: {len(df_vl)}")

    bundle = build_fold_tabular(df_tr, df_vl)
    print(f"[SCARF] feature_cols ({len(bundle.feature_cols)}): {bundle.feature_cols}")

    encoder = pretrain_scarf(
        x_train=bundle.x_train,
        input_dim=len(bundle.feature_cols),
        hidden_dim=args.hidden_dim,
        emb_dim=args.emb_dim,
        proj_dim=args.proj_dim,
        epochs=args.pretrain_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        corruption_rate=args.corruption_rate,
    )

    result = train_downstream(
        encoder=encoder,
        x_train=bundle.x_train,
        y_train=bundle.y_train,
        x_val=bundle.x_val,
        y_val=bundle.y_val,
        emb_dim=args.emb_dim,
        epochs=args.downstream_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        freeze_encoder=args.freeze_encoder,
    )

    print("[SCARF] Metrics")
    print(f"  train_auprc: {result['train_auprc']:.4f}")
    print(f"  val_auprc: {result['val_auprc']:.4f}")
    print(f"  val_roc_auc: {result['val_roc_auc']:.4f}")
    print(f"  val_f1@0.5: {result['val_f1']:.4f}")
    print(f"  val_precision@0.5: {result['val_precision']:.4f}")
    print(f"  val_recall@0.5: {result['val_recall']:.4f}")


if __name__ == "__main__":
    main()
