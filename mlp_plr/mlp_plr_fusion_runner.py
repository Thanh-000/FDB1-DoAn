from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from mlp_plr.mlp_plr_model import MLPPLRModel
from mlp_plr.mlp_plr_runner import ArrayDataset, base_preprocess, build_fold_tabular, load_ieee_train


def _build_xgb_model(random_state: int = 42):
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=random_state,
        n_jobs=4,
    )


def _build_lgb_model(random_state: int = 42):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary",
        random_state=random_state,
        n_jobs=4,
        verbosity=-1,
    )


def _build_cat_model(random_state: int = 42):
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        iterations=200,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="PRAUC",
        random_seed=random_state,
        verbose=False,
    )


def _build_meta_model(random_state: int = 42):
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=random_state,
        n_jobs=4,
    )


def _fit_tabular_models(x_tr: np.ndarray, y_tr: np.ndarray):
    models = {
        "XGB": _build_xgb_model(),
        "LGB": _build_lgb_model(),
        "CatBoost": _build_cat_model(),
    }
    for model in models.values():
        model.fit(x_tr, y_tr)
    return models


def _predict_tabular(models: dict[str, object], x: np.ndarray) -> np.ndarray:
    return np.column_stack([models[name].predict_proba(x)[:, 1] for name in ["XGB", "LGB", "CatBoost"]]).astype(
        np.float32
    )


def _metric_dict(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    return {
        "auprc": float(average_precision_score(y_true, probs)),
        "roc_auc": float(roc_auc_score(y_true, probs)),
    }


def _train_predict_mlp_plr(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    *,
    epochs: int = 10,
    batch_size: int = 2048,
    hidden_dim: int = 256,
    n_blocks: int = 4,
    n_frequencies: int = 8,
    lr: float = 1e-3,
) -> np.ndarray:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"[MLP-PLR] Device: cuda ({torch.cuda.get_device_name(0)})")
    else:
        print("[MLP-PLR] Device: cpu")

    train_ds = ArrayDataset(x_train, y_train)
    val_ds = ArrayDataset(x_val, np.zeros(len(x_val), dtype=np.float32))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=(device.type == "cuda"))

    model = MLPPLRModel(
        num_features=x_train.shape[1],
        hidden_dim=hidden_dim,
        n_blocks=n_blocks,
        n_frequencies=n_frequencies,
    ).to(device)

    pos = max(float(y_train.sum()), 1.0)
    neg = max(float(len(y_train) - y_train.sum()), 1.0)
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

    model.eval()
    preds = []
    with torch.no_grad():
        for x, _ in val_loader:
            x = x.to(device, non_blocking=(device.type == "cuda"))
            preds.append(torch.sigmoid(model(x)).cpu().numpy())
    return np.concatenate(preds).astype(np.float32, copy=False)


def run_fold_mlp_plr_score_fusion(
    data_dir: str | Path,
    *,
    fold_index: int = 0,
    n_splits: int = 5,
    inner_splits: int = 2,
    epochs: int = 10,
    batch_size: int = 2048,
    hidden_dim: int = 256,
    n_blocks: int = 4,
    n_frequencies: int = 8,
    lr: float = 1e-3,
    use_time_feature: bool = False,
) -> dict[str, float]:
    df = base_preprocess(load_ieee_train(data_dir))
    y = df["isFraud"].to_numpy()
    outer_cv = TimeSeriesSplit(n_splits=n_splits)
    outer_splits = list(outer_cv.split(np.zeros(len(y)), y))
    tr_i, vl_i = outer_splits[fold_index]

    df_tr = df.iloc[tr_i].reset_index(drop=True).copy()
    df_vl = df.iloc[vl_i].reset_index(drop=True).copy()
    print(f"[MLP-PLR-FUSION] Fold {fold_index + 1}/{n_splits}")
    print(f"[MLP-PLR-FUSION] Train: {len(df_tr)}, Val: {len(df_vl)}")

    bundle_outer = build_fold_tabular(df_tr, df_vl, use_time_feature=use_time_feature)
    print(f"[MLP-PLR-FUSION] use_time_feature: {use_time_feature}")
    print(f"[MLP-PLR-FUSION] feature_cols ({len(bundle_outer.feature_cols)}): {bundle_outer.feature_cols}")

    y_tr = bundle_outer.y_train
    y_vl = bundle_outer.y_val

    inner_cv = TimeSeriesSplit(n_splits=inner_splits)
    inner_splits_list = list(inner_cv.split(np.zeros(len(df_tr)), y_tr))
    oof_base = np.zeros((len(df_tr), 3), dtype=np.float32)
    oof_mlp = np.zeros(len(df_tr), dtype=np.float32)

    for inner_idx, (itr, ivl) in enumerate(inner_splits_list, start=1):
        print(f"[MLP-PLR-FUSION] Inner {inner_idx}/{inner_splits}: train={len(itr)} val={len(ivl)}")
        inner_tr = df_tr.iloc[itr].reset_index(drop=True).copy()
        inner_vl = df_tr.iloc[ivl].reset_index(drop=True).copy()
        inner_bundle = build_fold_tabular(inner_tr, inner_vl, use_time_feature=use_time_feature)

        tab_models = _fit_tabular_models(inner_bundle.x_train, inner_bundle.y_train)
        oof_base[ivl] = _predict_tabular(tab_models, inner_bundle.x_val)

        oof_mlp[ivl] = _train_predict_mlp_plr(
            inner_bundle.x_train,
            inner_bundle.y_train,
            inner_bundle.x_val,
            epochs=epochs,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
            n_blocks=n_blocks,
            n_frequencies=n_frequencies,
            lr=lr,
        )

    final_tab_models = _fit_tabular_models(bundle_outer.x_train, y_tr)
    val_base = _predict_tabular(final_tab_models, bundle_outer.x_val)
    val_mlp = _train_predict_mlp_plr(
        bundle_outer.x_train,
        y_tr,
        bundle_outer.x_val,
        epochs=epochs,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        n_blocks=n_blocks,
        n_frequencies=n_frequencies,
        lr=lr,
    )

    meta_base = _build_meta_model()
    meta_base.fit(oof_base, y_tr)
    pred_base = meta_base.predict_proba(val_base)[:, 1]

    meta_fused = _build_meta_model()
    fused_oof = np.column_stack([oof_base, oof_mlp]).astype(np.float32)
    fused_val = np.column_stack([val_base, val_mlp]).astype(np.float32)
    meta_fused.fit(fused_oof, y_tr)
    pred_fused = meta_fused.predict_proba(fused_val)[:, 1]

    base_metrics = _metric_dict(y_vl, pred_base)
    fused_metrics = _metric_dict(y_vl, pred_fused)

    print("[MLP-PLR-FUSION] Baseline meta (Nx3)")
    print(f"  val_auprc: {base_metrics['auprc']:.4f}")
    print(f"  val_roc_auc: {base_metrics['roc_auc']:.4f}")
    print("[MLP-PLR-FUSION] Fused meta (Nx4 = +MLP-PLR score)")
    print(f"  val_auprc: {fused_metrics['auprc']:.4f}")
    print(f"  val_roc_auc: {fused_metrics['roc_auc']:.4f}")
    print(f"[MLP-PLR-FUSION] Delta AUPRC: {fused_metrics['auprc'] - base_metrics['auprc']:+.4f}")
    print(f"[MLP-PLR-FUSION] Delta ROC-AUC: {fused_metrics['roc_auc'] - base_metrics['roc_auc']:+.4f}")

    return {
        "baseline_val_auprc": base_metrics["auprc"],
        "baseline_val_roc_auc": base_metrics["roc_auc"],
        "fused_val_auprc": fused_metrics["auprc"],
        "fused_val_roc_auc": fused_metrics["roc_auc"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument("--n-frequencies", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--use-time-feature", action="store_true")
    args = parser.parse_args()

    run_fold_mlp_plr_score_fusion(
        data_dir=args.data_dir,
        fold_index=args.fold_index,
        n_splits=args.n_splits,
        inner_splits=args.inner_splits,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        n_blocks=args.n_blocks,
        n_frequencies=args.n_frequencies,
        lr=args.lr,
        use_time_feature=args.use_time_feature,
    )


if __name__ == "__main__":
    main()
