import torch
from torch import nn


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNFraudModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        kernel_size: int = 3,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.tcn = nn.Sequential(
            TemporalBlock(hidden_dim, hidden_dim, kernel_size=kernel_size, dilation=1, dropout=dropout),
            TemporalBlock(hidden_dim, hidden_dim, kernel_size=kernel_size, dilation=2, dropout=dropout),
            TemporalBlock(hidden_dim, hidden_dim, kernel_size=kernel_size, dilation=4, dropout=dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # x: [batch, T, F]
        h = self.input_proj(x)  # [batch, T, H]
        h = h.transpose(1, 2)  # [batch, H, T]
        h = self.tcn(h)
        h = h.transpose(1, 2)  # [batch, T, H]

        last_idx = torch.clamp(lengths.long() - 1, min=0)
        batch_idx = torch.arange(h.size(0), device=h.device)
        last_h = h[batch_idx, last_idx]
        logits = self.head(last_h).squeeze(-1)
        return logits
