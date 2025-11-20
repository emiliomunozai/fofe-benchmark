# src/models/bilstm.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMNet(nn.Module):
    """BiLSTM encoder + MLP classifier for sequence classification."""

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 1,
        bidirectional: bool = True,
        num_classes: int = 2,
        pad_id: int = 0,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)

        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.bidirectional = bidirectional
        lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.fc1 = nn.Linear(lstm_out_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L) token IDs
        Returns:
            (B, num_classes) logits
        """
        emb = self.embedding(x)              # (B, L, E)
        out, _ = self.lstm(emb)              # (B, L, H*2)

        # Use last timestep (can be swapped for pooling)
        last_hidden = out[:, -1, :]          # (B, H*2)

        h = F.relu(self.fc1(last_hidden))    # (B, H)
        h = self.dropout(h)
        logits = self.fc2(h)                 # (B, C)

        return logits


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
