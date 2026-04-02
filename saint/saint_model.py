import torch
from torch import nn


class NumericalTokenizer(nn.Module):
    def __init__(self, num_features: int, d_token: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_features, d_token) * 0.02)
        self.bias = nn.Parameter(torch.zeros(num_features, d_token))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, num_features]
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class SAINTLikeModel(nn.Module):
    def __init__(
        self,
        num_features: int,
        d_token: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_mult: int = 4,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.tokenizer = NumericalTokenizer(num_features=num_features, d_token=d_token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        self.col_embedding = nn.Parameter(torch.randn(1, num_features + 1, d_token) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Sequential(
            nn.Linear(d_token, d_token),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x_tokens = torch.cat([cls, tokens], dim=1)
        x_tokens = x_tokens + self.col_embedding[:, : x_tokens.size(1), :]
        encoded = self.encoder(x_tokens)
        cls_out = self.norm(encoded[:, 0, :])
        return self.head(cls_out).squeeze(-1)
