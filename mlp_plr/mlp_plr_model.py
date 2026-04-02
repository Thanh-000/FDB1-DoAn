import math

import torch
from torch import nn


class NumericalPLREmbeddings(nn.Module):
    def __init__(self, num_features: int, n_frequencies: int = 8):
        super().__init__()
        self.num_features = num_features
        self.n_frequencies = n_frequencies
        self.freq = nn.Parameter(torch.randn(num_features, n_frequencies) * 0.5)
        self.phase = nn.Parameter(torch.zeros(num_features, n_frequencies))
        self.linear_weight = nn.Parameter(torch.ones(num_features))
        self.linear_bias = nn.Parameter(torch.zeros(num_features))

    @property
    def out_dim(self) -> int:
        return self.num_features * (1 + 2 * self.n_frequencies)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_exp = x.unsqueeze(-1)
        angles = 2.0 * math.pi * x_exp * self.freq.unsqueeze(0) + self.phase.unsqueeze(0)
        sin_part = torch.sin(angles)
        cos_part = torch.cos(angles)
        linear = (x * self.linear_weight.unsqueeze(0) + self.linear_bias.unsqueeze(0)).unsqueeze(-1)
        emb = torch.cat([linear, sin_part, cos_part], dim=-1)
        return emb.reshape(x.shape[0], -1)


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.15):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop1(h)
        h = self.fc2(h)
        h = self.drop2(h)
        return x + h


class MLPPLRModel(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 256,
        n_blocks: int = 4,
        n_frequencies: int = 8,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.emb = NumericalPLREmbeddings(num_features, n_frequencies=n_frequencies)
        self.input = nn.Sequential(
            nn.Linear(self.emb.out_dim, hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[ResidualMLPBlock(hidden_dim, dropout=dropout) for _ in range(n_blocks)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.emb(x)
        h = self.input(h)
        h = self.blocks(h)
        h = self.norm(h)
        return self.head(h).squeeze(-1)
