import torch
from torch import nn
import torch.nn.functional as F


class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, emb_dim: int = 128, dropout: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ProjectionHead(nn.Module):
    def __init__(self, emb_dim: int = 128, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, proj_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class SCARFPretrainModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, emb_dim: int = 128, proj_dim: int = 128, dropout: float = 0.10):
        super().__init__()
        self.encoder = MLPEncoder(input_dim=input_dim, hidden_dim=hidden_dim, emb_dim=emb_dim, dropout=dropout)
        self.projector = ProjectionHead(emb_dim=emb_dim, proj_dim=proj_dim)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))
        return z1, z2


class SCARFClassifier(nn.Module):
    def __init__(self, encoder: MLPEncoder, emb_dim: int = 128, dropout: float = 0.10):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(emb_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.head(z).squeeze(-1)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    n = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / temperature
    mask = torch.eye(2 * n, device=sim.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)

    targets = torch.arange(n, device=sim.device)
    targets = torch.cat([targets + n, targets], dim=0)
    return F.cross_entropy(sim, targets)
