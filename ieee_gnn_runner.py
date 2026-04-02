from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler

from ieee_gnn_train import train_fold_gnn


RAW_COLS = [
    "P_emaildomain",
    "R_emaildomain",
    "DeviceInfo",
    "DeviceType",
    "id_30",
    "id_31",
    "id_33",
    "ProductCD",
    "addr1",
    "card1",
]


def load_ieee_dataframe(data_dir: Path) -> pd.DataFrame:
    tr = pd.read_csv(data_dir / "train_transaction.csv")
    identity_path = data_dir / "train_identity.csv"
    if identity_path.exists():
        ident = pd.read_csv(identity_path)
        df = tr.merge(ident, on="TransactionID", how="left")
    else:
        df = tr
    return df


def base_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        df[col] = df[col].fillna("UNKNOWN")

    for col in RAW_COLS:
        if col in df.columns:
            df[f"{col}_raw"] = df[col].astype(str)

    fp_parts = []
    for col in ["DeviceType_raw", "DeviceInfo_raw", "id_30_raw", "id_31_raw", "id_33_raw"]:
        if col in df.columns:
            fp_parts.append(df[col].fillna("UNKNOWN"))
        else:
            fp_parts.append(pd.Series(["UNKNOWN"] * len(df), index=df.index))
    df["device_fp_raw"] = fp_parts[0] + "|" + fp_parts[1] + "|" + fp_parts[2] + "|" + fp_parts[3] + "|" + fp_parts[4]

    if "TransactionDT" in df.columns:
        df = df.sort_values("TransactionDT").reset_index(drop=True)
        df["Hour"] = (df["TransactionDT"] / 3600).astype(int) % 24
        df["DayOfWeek"] = (df["TransactionDT"] / 86400).astype(int) % 7

    if "TransactionAmt" in df.columns:
        df["LogAmt"] = np.log1p(df["TransactionAmt"])
        df["Amt_cents"] = np.round(df["TransactionAmt"] - np.floor(df["TransactionAmt"]), 2).astype(np.float32)

    if {"TransactionDT", "D1", "card1_raw", "addr1_raw"}.issubset(df.columns):
        df["Day"] = (df["TransactionDT"] / 86400).astype(int)
        df["D1n"] = df["Day"] - df["D1"].fillna(0)
        df["AccountID"] = (
            df["card1_raw"].astype(str)
            + "_"
            + df["addr1_raw"].astype(str)
            + "_"
            + df["D1n"].astype(str)
        )
    elif {"card1_raw", "card2"}.issubset(df.columns):
        df["AccountID"] = df["card1_raw"].astype(str) + "_" + df["card2"].astype(str)
    else:
        df["AccountID"] = df.index.astype(str)

    if {"C1", "C14"}.issubset(df.columns):
        df["C1_div_C14"] = (df["C1"] / (df["C14"] + 1e-6)).astype(np.float32)

    return df


def choose_txn_feature_cols(df: pd.DataFrame) -> List[str]:
    candidates = [
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
    return [c for c in candidates if c in df.columns]


def prepare_fold_frames(df_tr: pd.DataFrame, df_vl: pd.DataFrame, txn_feature_cols: List[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_tr = df_tr.reset_index(drop=True).copy()
    df_vl = df_vl.reset_index(drop=True).copy()

    medians = df_tr[txn_feature_cols].median(numeric_only=True)
    df_tr[txn_feature_cols] = df_tr[txn_feature_cols].fillna(medians)
    df_vl[txn_feature_cols] = df_vl[txn_feature_cols].fillna(medians)

    scaler = MinMaxScaler()
    df_tr[txn_feature_cols] = scaler.fit_transform(df_tr[txn_feature_cols])
    df_vl[txn_feature_cols] = scaler.transform(df_vl[txn_feature_cols])

    for frame in (df_tr, df_vl):
        for col in txn_feature_cols:
            frame[col] = frame[col].astype(np.float32)

    return df_tr, df_vl


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a temporal hetero GNN fold on IEEE-CIS.")
    parser.add_argument("--data-dir", default="ieee-fraud-detection", help="Directory containing train_transaction.csv")
    parser.add_argument("--fold-index", type=int, default=0, help="0-based fold index for TimeSeriesSplit")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of walk-forward splits")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    df = base_preprocess(load_ieee_dataframe(data_dir))
    txn_feature_cols = choose_txn_feature_cols(df)
    if not txn_feature_cols:
        raise ValueError("No transaction feature columns available for the GNN runner.")

    y = df["isFraud"].to_numpy()
    cv = TimeSeriesSplit(n_splits=args.n_splits)
    splits = list(cv.split(np.zeros(len(y)), y))
    if args.fold_index < 0 or args.fold_index >= len(splits):
        raise ValueError(f"fold-index must be in [0, {len(splits)-1}]")

    tr_idx, vl_idx = splits[args.fold_index]
    df_tr, df_vl = prepare_fold_frames(df.iloc[tr_idx], df.iloc[vl_idx], txn_feature_cols)

    print(f"[GNN] Fold {args.fold_index + 1}/{len(splits)}")
    print(f"[GNN] Train: {len(df_tr)}, Val: {len(df_vl)}")
    print(f"[GNN] txn_feature_cols ({len(txn_feature_cols)}): {txn_feature_cols}")

    result = train_fold_gnn(
        df_tr=df_tr,
        df_vl=df_vl,
        txn_feature_cols=txn_feature_cols,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
    )

    print("[GNN] Metrics")
    for key, value in result.metrics.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
