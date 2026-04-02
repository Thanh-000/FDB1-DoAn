from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from sequence.sequence_tcn import TCNFraudModel


DEFAULT_SEQ_FEATURES = [
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


@dataclass
class FoldSequenceBundle:
    x_train: np.ndarray
    len_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    len_val: np.ndarray
    y_val: np.ndarray
    feature_cols: list[str]


class SequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, lengths: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x.astype(np.float32, copy=False))
        self.lengths = torch.from_numpy(lengths.astype(np.int64, copy=False))
        self.y = torch.from_numpy(y.astype(np.float32, copy=False))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.lengths[idx], self.y[idx]


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

    raw_cols = [
        "P_emaildomain",
        "R_emaildomain",
        "DeviceInfo",
        "DeviceType",
        "id_30",
        "id_31",
        "id_33",
        "ProductCD",
        "addr1",
        "addr2",
        "card1",
        "card2",
        "card3",
        "card5",
    ]
    for c in raw_cols:
        if c in df.columns:
            df[f"{c}_raw"] = df[c].astype(str)

    fp_parts = []
    for c in ["DeviceType_raw", "DeviceInfo_raw", "id_30_raw", "id_31_raw", "id_33_raw"]:
        if c in df.columns:
            fp_parts.append(df[c].fillna("UNKNOWN"))
        else:
            fp_parts.append(pd.Series(["UNKNOWN"] * len(df), index=df.index))
    df["device_fp_raw"] = fp_parts[0] + "|" + fp_parts[1] + "|" + fp_parts[2] + "|" + fp_parts[3] + "|" + fp_parts[4]

    if "TransactionDT" in df.columns:
        df = df.sort_values("TransactionDT").reset_index(drop=True)
        df["Hour"] = (df["TransactionDT"] / 3600).astype(int) % 24
        df["DayOfWeek"] = (df["TransactionDT"] / 86400).astype(int) % 7

    if "TransactionDT" in df.columns and "D1" in df.columns and "card1_raw" in df.columns and "addr1_raw" in df.columns:
        df["Day"] = (df["TransactionDT"] / 86400).astype(int)
        df["D1n"] = df["Day"] - df["D1"].fillna(0)
        df["AccountID"] = df["card1_raw"].astype(str) + "_" + df["addr1_raw"].astype(str) + "_" + df["D1n"].astype(str)
    elif "card1_raw" in df.columns and "card2_raw" in df.columns:
        df["AccountID"] = df["card1_raw"].astype(str) + "_" + df["card2_raw"].astype(str)
    elif "card1" in df.columns and "card2" in df.columns:
        df["AccountID"] = df["card1"].astype(str) + "_" + df["card2"].astype(str)
    else:
        df["AccountID"] = df.index.astype(str)

    if "TransactionAmt" in df.columns:
        df["LogAmt"] = np.log1p(df["TransactionAmt"])
        df["Amt_cents"] = np.round(df["TransactionAmt"] - np.floor(df["TransactionAmt"]), 2).astype(np.float32)

    if "C1" in df.columns and "C14" in df.columns:
        df["C1_div_C14"] = (df["C1"] / (df["C14"] + 1e-6)).astype(np.float32)

    num_cols = [c for c in DEFAULT_SEQ_FEATURES if c in df.columns]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _build_sequences_for_frame(df_hist: pd.DataFrame, df_target: pd.DataFrame, feature_cols: list[str], window: int) -> tuple[np.ndarray, np.ndarray]:
    combo = pd.concat(
        [
            df_hist.assign(_split="hist", _row=np.arange(len(df_hist))),
            df_target.assign(_split="target", _row=np.arange(len(df_target))),
        ],
        ignore_index=True,
    ).sort_values("TransactionDT").reset_index(drop=True)

    seq = np.zeros((len(df_target), window, len(feature_cols)), dtype=np.float32)
    lengths = np.zeros(len(df_target), dtype=np.int64)

    history: dict[str, list[np.ndarray]] = {}
    target_row_pos: dict[int, int] = {i: i for i in range(len(df_target))}

    for _, grp in combo.groupby("AccountID", sort=False):
        past: list[np.ndarray] = []
        for row in grp.itertuples(index=False):
            current = np.asarray([getattr(row, c) for c in feature_cols], dtype=np.float32)
            current = np.nan_to_num(current, nan=0.0, posinf=0.0, neginf=0.0)
            past.append(current)
            if len(past) > window:
                past = past[-window:]
            if getattr(row, "_split") == "target":
                idx = target_row_pos[getattr(row, "_row")]
                s = np.stack(past, axis=0)
                seq[idx, -len(s):, :] = s
                lengths[idx] = len(s)
        history.clear()
    return seq, lengths


def build_fold_sequences(df_tr: pd.DataFrame, df_vl: pd.DataFrame, window: int) -> FoldSequenceBundle:
    feature_cols = [c for c in DEFAULT_SEQ_FEATURES if c in df_tr.columns and c in df_vl.columns]
    if not feature_cols:
        raise ValueError("No sequence feature columns available")

    medians = df_tr[feature_cols].median()
    x_tr_num = df_tr[feature_cols].fillna(medians)
    x_vl_num = df_vl[feature_cols].fillna(medians)

    scaler = StandardScaler()
    x_tr_scaled = pd.DataFrame(scaler.fit_transform(x_tr_num), columns=feature_cols, index=df_tr.index)
    x_vl_scaled = pd.DataFrame(scaler.transform(x_vl_num), columns=feature_cols, index=df_vl.index)

    df_tr_seq = df_tr.copy()
    df_vl_seq = df_vl.copy()
    df_tr_seq.loc[:, feature_cols] = x_tr_scaled.values
    df_vl_seq.loc[:, feature_cols] = x_vl_scaled.values

    x_train, len_train = _build_sequences_for_frame(df_tr_seq, df_tr_seq, feature_cols, window)
    x_val, len_val = _build_sequences_for_frame(df_tr_seq, df_vl_seq, feature_cols, window)

    return FoldSequenceBundle(
        x_train=x_train,
        len_train=len_train,
        y_train=df_tr["isFraud"].to_numpy(dtype=np.float32, copy=False),
        x_val=x_val,
        len_val=len_val,
        y_val=df_vl["isFraud"].to_numpy(dtype=np.float32, copy=False),
        feature_cols=feature_cols,
    )


def _predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for x, lengths, _ in loader:
            x = x.to(device)
            lengths = lengths.to(device)
            logits = model(x, lengths)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds.append(probs)
    return np.concatenate(preds)


def train_fold_tcn(
    df_tr: pd.DataFrame,
    df_vl: pd.DataFrame,
    window: int = 10,
    hidden_dim: int = 64,
    epochs: int = 20,
    batch_size: int = 1024,
    lr: float = 1e-3,
    seed: int = 42,
) -> dict[str, float | list[str] | np.ndarray]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    bundle = build_fold_sequences(df_tr, df_vl, window)
    train_ds = SequenceDataset(bundle.x_train, bundle.len_train, bundle.y_train)
    val_ds = SequenceDataset(bundle.x_val, bundle.len_val, bundle.y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TCNFraudModel(input_dim=len(bundle.feature_cols), hidden_dim=hidden_dim).to(device)

    pos = max(float(bundle.y_train.sum()), 1.0)
    neg = max(float(len(bundle.y_train) - bundle.y_train.sum()), 1.0)
    pos_weight = torch.tensor([neg / pos], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        model.train()
        for x, lengths, y in train_loader:
            x = x.to(device)
            lengths = lengths.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x, lengths)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

    train_loader_eval = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    train_probs = _predict(model, train_loader_eval, device)
    val_probs = _predict(model, val_loader, device)

    return {
        "feature_cols": bundle.feature_cols,
        "train_probs": train_probs,
        "val_probs": val_probs,
        "train_auprc": float(average_precision_score(bundle.y_train, train_probs)),
        "val_auprc": float(average_precision_score(bundle.y_val, val_probs)),
        "val_roc_auc": float(roc_auc_score(bundle.y_val, val_probs)),
        "val_f1": float(f1_score(bundle.y_val, val_probs >= 0.5)),
        "val_precision": float(precision_score(bundle.y_val, val_probs >= 0.5, zero_division=0)),
        "val_recall": float(recall_score(bundle.y_val, val_probs >= 0.5, zero_division=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    df = base_preprocess(load_ieee_train(args.data_dir))
    y = df["isFraud"].to_numpy()
    cv = TimeSeriesSplit(n_splits=args.n_splits)
    splits = list(cv.split(np.zeros(len(y)), y))
    tr_i, vl_i = splits[args.fold_index]

    df_tr = df.iloc[tr_i].reset_index(drop=True).copy()
    df_vl = df.iloc[vl_i].reset_index(drop=True).copy()

    print(f"[TCN] Fold {args.fold_index + 1}/{args.n_splits}")
    print(f"[TCN] Train: {len(df_tr)}, Val: {len(df_vl)}")

    result = train_fold_tcn(
        df_tr=df_tr,
        df_vl=df_vl,
        window=args.window,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    print(f"[TCN] seq_feature_cols ({len(result['feature_cols'])}): {result['feature_cols']}")
    print("[TCN] Metrics")
    print(f"  train_auprc: {result['train_auprc']:.4f}")
    print(f"  val_auprc: {result['val_auprc']:.4f}")
    print(f"  val_roc_auc: {result['val_roc_auc']:.4f}")
    print(f"  val_f1@0.5: {result['val_f1']:.4f}")
    print(f"  val_precision@0.5: {result['val_precision']:.4f}")
    print(f"  val_recall@0.5: {result['val_recall']:.4f}")


if __name__ == "__main__":
    main()
