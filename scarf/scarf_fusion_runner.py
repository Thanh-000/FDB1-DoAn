from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from scarf.scarf_runner import base_preprocess, build_fold_tabular, load_ieee_train, run_fold_scarf


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


def run_fold_scarf_score_fusion(
    data_dir: str | Path,
    *,
    fold_index: int = 0,
    n_splits: int = 5,
    inner_splits: int = 2,
    pretrain_epochs: int = 10,
    downstream_epochs: int = 10,
    batch_size: int = 2048,
    hidden_dim: int = 256,
    emb_dim: int = 128,
    proj_dim: int = 128,
    lr: float = 1e-3,
    corruption_rate: float = 0.6,
    no_pretrain: bool = False,
) -> dict[str, float]:
    df = base_preprocess(load_ieee_train(data_dir))
    y = df["isFraud"].to_numpy()
    outer_cv = TimeSeriesSplit(n_splits=n_splits)
    outer_splits = list(outer_cv.split(np.zeros(len(y)), y))
    tr_i, vl_i = outer_splits[fold_index]

    df_tr = df.iloc[tr_i].reset_index(drop=True).copy()
    df_vl = df.iloc[vl_i].reset_index(drop=True).copy()
    print(f"[SCARF-FUSION] Fold {fold_index + 1}/{n_splits}")
    print(f"[SCARF-FUSION] Train: {len(df_tr)}, Val: {len(df_vl)}")

    bundle_outer = build_fold_tabular(df_tr, df_vl)
    print(f"[SCARF-FUSION] feature_cols ({len(bundle_outer.feature_cols)}): {bundle_outer.feature_cols}")

    y_tr = bundle_outer.y_train
    y_vl = bundle_outer.y_val

    inner_cv = TimeSeriesSplit(n_splits=inner_splits)
    inner_splits_list = list(inner_cv.split(np.zeros(len(df_tr)), y_tr))
    oof_base = np.zeros((len(df_tr), 3), dtype=np.float32)
    oof_scarf = np.zeros(len(df_tr), dtype=np.float32)

    for inner_idx, (itr, ivl) in enumerate(inner_splits_list, start=1):
        print(f"[SCARF-FUSION] Inner {inner_idx}/{inner_splits}: train={len(itr)} val={len(ivl)}")
        inner_tr = df_tr.iloc[itr].reset_index(drop=True).copy()
        inner_vl = df_tr.iloc[ivl].reset_index(drop=True).copy()
        inner_bundle = build_fold_tabular(inner_tr, inner_vl)

        tab_models = _fit_tabular_models(inner_bundle.x_train, inner_bundle.y_train)
        oof_base[ivl] = _predict_tabular(tab_models, inner_bundle.x_val)

        scarf_result = run_fold_scarf(
            inner_tr,
            inner_vl,
            pretrain_epochs=pretrain_epochs,
            downstream_epochs=downstream_epochs,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
            emb_dim=emb_dim,
            proj_dim=proj_dim,
            lr=lr,
            corruption_rate=corruption_rate,
            freeze_encoder=False,
            no_pretrain=no_pretrain,
        )
        oof_scarf[ivl] = scarf_result["val_probs"]

    final_tab_models = _fit_tabular_models(bundle_outer.x_train, y_tr)
    val_base = _predict_tabular(final_tab_models, bundle_outer.x_val)
    scarf_outer = run_fold_scarf(
        df_tr,
        df_vl,
        pretrain_epochs=pretrain_epochs,
        downstream_epochs=downstream_epochs,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        emb_dim=emb_dim,
        proj_dim=proj_dim,
        lr=lr,
        corruption_rate=corruption_rate,
        freeze_encoder=False,
        no_pretrain=no_pretrain,
    )
    val_scarf = scarf_outer["val_probs"].astype(np.float32, copy=False)

    meta_base = _build_meta_model()
    meta_base.fit(oof_base, y_tr)
    pred_base = meta_base.predict_proba(val_base)[:, 1]

    meta_fused = _build_meta_model()
    fused_oof = np.column_stack([oof_base, oof_scarf]).astype(np.float32)
    fused_val = np.column_stack([val_base, val_scarf]).astype(np.float32)
    meta_fused.fit(fused_oof, y_tr)
    pred_fused = meta_fused.predict_proba(fused_val)[:, 1]

    base_metrics = _metric_dict(y_vl, pred_base)
    fused_metrics = _metric_dict(y_vl, pred_fused)

    print("[SCARF-FUSION] Baseline meta (Nx3)")
    print(f"  val_auprc: {base_metrics['auprc']:.4f}")
    print(f"  val_roc_auc: {base_metrics['roc_auc']:.4f}")
    print("[SCARF-FUSION] Fused meta (Nx4 = +SCARF score)")
    print(f"  val_auprc: {fused_metrics['auprc']:.4f}")
    print(f"  val_roc_auc: {fused_metrics['roc_auc']:.4f}")
    print(f"[SCARF-FUSION] Delta AUPRC: {fused_metrics['auprc'] - base_metrics['auprc']:+.4f}")
    print(f"[SCARF-FUSION] Delta ROC-AUC: {fused_metrics['roc_auc'] - base_metrics['roc_auc']:+.4f}")

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
    parser.add_argument("--pretrain-epochs", type=int, default=10)
    parser.add_argument("--downstream-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--emb-dim", type=int, default=128)
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--corruption-rate", type=float, default=0.6)
    parser.add_argument("--no-pretrain", action="store_true")
    args = parser.parse_args()

    run_fold_scarf_score_fusion(
        data_dir=args.data_dir,
        fold_index=args.fold_index,
        n_splits=args.n_splits,
        inner_splits=args.inner_splits,
        pretrain_epochs=args.pretrain_epochs,
        downstream_epochs=args.downstream_epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        emb_dim=args.emb_dim,
        proj_dim=args.proj_dim,
        lr=args.lr,
        corruption_rate=args.corruption_rate,
        no_pretrain=args.no_pretrain,
    )


if __name__ == "__main__":
    main()
