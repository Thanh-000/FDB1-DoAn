from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from gnn.ieee_gnn_graph import FoldGraphBundle, build_fold_hetero_graph
from gnn.ieee_gnn_model import IEEEFraudHeteroGNN


@dataclass
class GNNEvalResult:
    val_probs: np.ndarray
    train_probs: np.ndarray
    metrics: Dict[str, float]


def train_fold_gnn(
    df_tr: pd.DataFrame,
    df_vl: pd.DataFrame,
    txn_feature_cols: List[str],
    epochs: int = 30,
    hidden_dim: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str | None = None,
) -> GNNEvalResult:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    bundle: FoldGraphBundle = build_fold_hetero_graph(df_tr, df_vl, txn_feature_cols)

    data = bundle.train_data.to(device)
    train_y = bundle.train_y.to(device)
    val_x = bundle.val_links.txn_x.to(device)
    val_rel = {k: v.to(device) for k, v in bundle.val_links.relation_index.items()}
    val_y = bundle.val_links.y.to(device)

    model = IEEEFraudHeteroGNN(
        metadata=bundle.metadata,
        entity_sizes=bundle.entity_sizes,
        relation_order=bundle.relation_order,
        txn_in_dim=len(txn_feature_cols),
        hidden_dim=hidden_dim,
    ).to(device)

    pos = float((train_y == 1).sum().item())
    neg = float((train_y == 0).sum().item())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits, _ = model.forward_train(data)
        loss = F.binary_cross_entropy_with_logits(logits, train_y, pos_weight=pos_weight)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        train_logits, memory = model.forward_train(data)
        val_logits = model.score_validation(val_x, val_rel, memory)

        train_probs = torch.sigmoid(train_logits).detach().cpu().numpy()
        val_probs = torch.sigmoid(val_logits).detach().cpu().numpy()

    metrics = {
        "train_auprc": average_precision_score(bundle.train_y.numpy(), train_probs),
        "val_auprc": average_precision_score(bundle.val_links.y.numpy(), val_probs),
        "val_roc_auc": roc_auc_score(bundle.val_links.y.numpy(), val_probs),
    }
    return GNNEvalResult(val_probs=val_probs, train_probs=train_probs, metrics=metrics)
