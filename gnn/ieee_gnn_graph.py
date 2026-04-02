from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np
import pandas as pd
import torch

try:
    from torch_geometric.data import HeteroData
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "torch_geometric is required for the IEEE GNN branch. "
        "Install it in Colab before using ieee_gnn_graph.py."
    ) from exc


ENTITY_SPECS: Mapping[str, str] = {
    "acct": "AccountID",
    "card1": "card1_raw",
    "addr1": "addr1_raw",
    "device": "device_fp_raw",
    "pemail": "P_emaildomain_raw",
    "product": "ProductCD_raw",
}

INVALID_TOKENS = {"UNKNOWN", "nan", "None", "-1", ""}


@dataclass
class ValidationLinks:
    txn_x: torch.Tensor
    y: torch.Tensor
    relation_index: Dict[str, torch.Tensor]


@dataclass
class FoldGraphBundle:
    train_data: HeteroData
    train_y: torch.Tensor
    val_links: ValidationLinks
    metadata: Tuple[List[str], List[Tuple[str, str, str]]]
    entity_sizes: Dict[str, int]
    relation_order: List[str]


def _normalize_token(value: object) -> str:
    token = str(value)
    return token if token not in INVALID_TOKENS else "UNKNOWN"


def _build_entity_maps(df_tr: pd.DataFrame, df_vl: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    maps: Dict[str, Dict[str, int]] = {}
    for node_type, col in ENTITY_SPECS.items():
        values = pd.concat([df_tr[col], df_vl[col]], ignore_index=True).astype(str).map(_normalize_token)
        uniq = pd.Index(values.unique())
        if "UNKNOWN" not in uniq:
            uniq = pd.Index(["UNKNOWN"]).append(uniq)
        maps[node_type] = {token: idx for idx, token in enumerate(uniq)}
    return maps


def _encode_entity_series(
    values: Iterable[object],
    mapping: Mapping[str, int],
) -> np.ndarray:
    arr = pd.Series(values, copy=False).astype(str).map(_normalize_token)
    unknown_idx = mapping.get("UNKNOWN", 0)
    return arr.map(mapping).fillna(unknown_idx).to_numpy(dtype=np.int64, copy=False)


def build_fold_hetero_graph(
    df_tr: pd.DataFrame,
    df_vl: pd.DataFrame,
    txn_feature_cols: List[str],
    label_col: str = "isFraud",
) -> FoldGraphBundle:
    """
    Build a causal-safe graph package for one fold.

    Train graph:
    - Contains only training transactions and entity nodes.
    - Supports full message passing on historical transactions.

    Validation package:
    - Uses frozen entity memory learned from the training graph.
    - Stores links from each validation transaction to entity ids, but does not
      allow validation transactions to message to each other.
    """

    df_tr = df_tr.reset_index(drop=True).copy()
    df_vl = df_vl.reset_index(drop=True).copy()

    entity_maps = _build_entity_maps(df_tr, df_vl)
    relation_order = list(ENTITY_SPECS.keys())

    data = HeteroData()
    data["txn"].x = torch.tensor(df_tr[txn_feature_cols].to_numpy(dtype=np.float32), dtype=torch.float32)
    train_y = torch.tensor(df_tr[label_col].to_numpy(dtype=np.float32), dtype=torch.float32)

    entity_sizes: Dict[str, int] = {}
    for node_type, mapping in entity_maps.items():
        num_nodes = len(mapping)
        entity_sizes[node_type] = num_nodes
        data[node_type].num_nodes = num_nodes

    train_txn_idx = np.arange(len(df_tr), dtype=np.int64)
    for node_type, col in ENTITY_SPECS.items():
        entity_idx = _encode_entity_series(df_tr[col], entity_maps[node_type])
        forward = torch.tensor(np.vstack([train_txn_idx, entity_idx]), dtype=torch.long)
        reverse = torch.tensor(np.vstack([entity_idx, train_txn_idx]), dtype=torch.long)
        data[("txn", f"to_{node_type}", node_type)].edge_index = forward
        data[(node_type, f"from_txn", "txn")].edge_index = reverse

    val_relation_index: Dict[str, torch.Tensor] = {}
    for node_type, col in ENTITY_SPECS.items():
        val_relation_index[node_type] = torch.tensor(
            _encode_entity_series(df_vl[col], entity_maps[node_type]),
            dtype=torch.long,
        )

    val_links = ValidationLinks(
        txn_x=torch.tensor(df_vl[txn_feature_cols].to_numpy(dtype=np.float32), dtype=torch.float32),
        y=torch.tensor(df_vl[label_col].to_numpy(dtype=np.float32), dtype=torch.float32),
        relation_index=val_relation_index,
    )

    return FoldGraphBundle(
        train_data=data,
        train_y=train_y,
        val_links=val_links,
        metadata=data.metadata(),
        entity_sizes=entity_sizes,
        relation_order=relation_order,
    )
