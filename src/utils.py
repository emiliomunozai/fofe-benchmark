import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

class PositionWiseCrossEntropy:
    def __init__(self):
        self.pad_index = 0
        self.criterion = nn.CrossEntropyLoss(reduction="none")

    def __call__(self, y_pred, y_target):
        """
        Compute the position-wise cross-entropy loss.
        """
        if y_pred.dim() != 3 or y_target.dim() != 2:
            raise ValueError(
                "Expected y_pred of shape (batch_size, mmax, num_classes) "
                "and y_target of shape (batch_size, mmax), "
                f"but got {y_pred.shape} "
                f"and {y_target.shape}")

        # Compute position-wise cross-entropy loss
        batch_size, mmax, _ = y_pred.size()
        loss = self.criterion(y_pred.view(-1, y_pred.size(-1)), y_target.view(-1))
        loss = loss.view(batch_size, mmax)
        total_loss = loss.mean()
        
        return total_loss

def collate_scalar_to_sequence(batch, max_seq_len):
    """
    Make objects in a batch the same size using padding 0 and ensure they
    are tensors
    """
    # Unzip
    xs, ys = zip(*batch)
    xs_padded = pad_sequence(xs, batch_first=True, padding_value=0)
    ys_padded = pad_sequence(ys, batch_first=True, padding_value=0)

    # Force global max_seq_len
    if xs_padded.size(1) < max_seq_len:
        xs_padded = F.pad(xs_padded, (0, max_seq_len - xs_padded.size(1)), value=0)
        ys_padded = F.pad(ys_padded, (0, max_seq_len - ys_padded.size(1)), value=0)
    else:
        xs_padded = xs_padded[:, :max_seq_len]
        ys_padded = ys_padded[:, :max_seq_len]

    return xs_padded, ys_padded