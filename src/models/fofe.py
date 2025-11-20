# src/models/fofe.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class FastFOFEEncoder(nn.Module):
    def __init__(self, alpha: float = 0.7):
        super().__init__()
        self.alpha = alpha

    def forward(self, x_emb):
        # x_emb: (B, L, D)
        B, L, D = x_emb.shape
        device = x_emb.device
        dtype = x_emb.dtype

        idx = torch.arange(L, device=device, dtype=dtype)
        decay = self.alpha ** idx
        decay = decay.view(1, L, 1)

        scaled = x_emb * decay
        summed = torch.cumsum(scaled, dim=1)

        # Return final hidden state
        return summed[:, -1, :]


class FastBiFOFEEncoder(nn.Module):
    """
    Bidirectional GPU-optimized FOFE encoder.
    Concatenates left-to-right and right-to-left codes.
    """

    def __init__(self, alpha: float = 0.7):
        super().__init__()
        self.left = FastFOFEEncoder(alpha)
        self.right = FastFOFEEncoder(alpha)

    def forward(self, x_emb: torch.Tensor) -> torch.Tensor:
        # Left-to-right
        left_code = self.left(x_emb)

        # Right-to-left
        rev = torch.flip(x_emb, dims=[1])
        right_code = self.right(rev)

        return torch.cat([left_code, right_code], dim=-1)


class FOFENet(nn.Module):
    """Fast FOFE encoder + MLP classifier."""

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 128,
        alpha: float = 0.7,
        bidirectional: bool = True,
        hidden_dim: int = 256,
        num_classes: int = 2,
        pad_id: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.bidirectional = bidirectional

        if bidirectional:
            self.fofe = FastBiFOFEEncoder(alpha)
            fofe_dim = emb_dim * 2
        else:
            self.fofe = FastFOFEEncoder(alpha)
            fofe_dim = emb_dim

        self.fc1 = nn.Linear(fofe_dim, hidden_dim)
        self.fc_mid = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)        # (B, L, D)
        code = self.fofe(emb)          # (B, D or 2D)

        h = F.relu(self.fc1(code))
        h = self.dropout(h)
        h = F.relu(self.fc_mid(h))
        h = self.dropout(h)

        logits = self.fc2(h)
        return logits
