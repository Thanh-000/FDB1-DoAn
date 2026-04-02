import torch
from torch import nn


class NumericalFeatureProjector(nn.Module):
    def __init__(self, num_features: int, d_token: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_features, d_token) * 0.02)
        self.bias = nn.Parameter(torch.zeros(num_features, d_token))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class TabMStyleModel(nn.Module):
    def __init__(
        self,
        num_features: int,
        d_token: int = 16,
        hidden_dim: int = 256,
        n_layers: int = 3,
        n_members: int = 4,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.projector = NumericalFeatureProjector(num_features=num_features, d_token=d_token)
        input_dim = num_features * d_token

        layers = []
        dim = input_dim
        for _ in range(n_layers):
            layers.extend(
                [
                    nn.Linear(dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(n_members)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.projector(x)
        flat = tokens.flatten(1)
        h = self.backbone(flat)
        logits = torch.cat([head(h) for head in self.heads], dim=1)
        return logits
