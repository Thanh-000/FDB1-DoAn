import torch
from torch import nn


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


class RealMLPStyleModel(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 256,
        n_blocks: int = 4,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[ResidualMLPBlock(hidden_dim, dropout=dropout) for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input(x)
        h = self.blocks(h)
        h = self.norm(h)
        return self.head(h).squeeze(-1)
