from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

try:
    from torch_geometric.nn import HeteroConv, SAGEConv
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "torch_geometric is required for the IEEE GNN branch. "
        "Install it in Colab before using ieee_gnn_model.py."
    ) from exc


class IEEEFraudHeteroGNN(nn.Module):
    """
    A practical hetero GNN branch for IEEE-CIS.

    Training:
    - full message passing over training transactions + entity nodes

    Validation:
    - frozen entity memory from the trained graph
    - no validation-to-validation message passing
    """

    def __init__(
        self,
        metadata: Tuple[List[str], List[Tuple[str, str, str]]],
        entity_sizes: Dict[str, int],
        relation_order: List[str],
        txn_in_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.metadata = metadata
        self.entity_sizes = entity_sizes
        self.relation_order = relation_order
        self.hidden_dim = hidden_dim

        self.txn_encoder = nn.Sequential(
            nn.Linear(txn_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.entity_embeddings = nn.ModuleDict(
            {
                node_type: nn.Embedding(num_nodes, hidden_dim)
                for node_type, num_nodes in entity_sizes.items()
            }
        )
        self.unknown_embeddings = nn.ParameterDict(
            {
                node_type: nn.Parameter(torch.zeros(hidden_dim))
                for node_type in relation_order
            }
        )

        conv_dict = {}
        for edge_type in metadata[1]:
            conv_dict[edge_type] = SAGEConv((-1, -1), hidden_dim)

        self.conv1 = HeteroConv(conv_dict, aggr="sum")
        self.conv2 = HeteroConv(conv_dict, aggr="sum")
        self.norms = nn.ModuleDict({node_type: nn.LayerNorm(hidden_dim) for node_type in metadata[0]})

        fusion_dim = hidden_dim * (1 + len(relation_order))
        self.val_head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.train_head = nn.Linear(hidden_dim, 1)

    def _initial_x_dict(self, txn_x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x_dict: Dict[str, torch.Tensor] = {"txn": self.txn_encoder(txn_x)}
        for node_type, emb in self.entity_embeddings.items():
            x_dict[node_type] = emb.weight
        return x_dict

    def encode_train_graph(self, data) -> Dict[str, torch.Tensor]:
        x_dict = self._initial_x_dict(data["txn"].x)
        x_dict = self.conv1(x_dict, data.edge_index_dict)
        x_dict = {k: self.norms[k](v).relu() for k, v in x_dict.items()}
        x_dict = self.conv2(x_dict, data.edge_index_dict)
        x_dict = {k: self.norms[k](v).relu() for k, v in x_dict.items()}
        return x_dict

    def forward_train(self, data) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x_dict = self.encode_train_graph(data)
        txn_logits = self.train_head(x_dict["txn"]).squeeze(-1)
        return txn_logits, x_dict

    def score_validation(self, txn_x: torch.Tensor, relation_index: Dict[str, torch.Tensor], entity_memory: Dict[str, torch.Tensor]) -> torch.Tensor:
        txn_h = self.txn_encoder(txn_x)
        parts = [txn_h]
        device = txn_h.device

        for node_type in self.relation_order:
            mem = entity_memory[node_type]
            idx = relation_index[node_type].to(device)
            unk = self.unknown_embeddings[node_type].to(device)
            gathered = mem[idx]
            unknown_mask = idx < 0
            if torch.any(unknown_mask):
                gathered = gathered.clone()
                gathered[unknown_mask] = unk
            parts.append(gathered)

        fused = torch.cat(parts, dim=-1)
        return self.val_head(fused).squeeze(-1)
