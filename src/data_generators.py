import torch
import random
from torch.utils.data import IterableDataset
from torch.nn.utils.rnn import pad_sequence
from torch import nn

class ExactSequenceGenerator(IterableDataset):
    def __init__(self, min_length: int, max_length: int, vocab_size: int):
        super().__init__()
        assert vocab_size > 2, "Vocabulary size must be greater than 2"
        self.min_length = min_length
        self.max_length = max_length
        self.vocab_size = vocab_size

    def __iter__(self):
        while True:
            # 1. Random length using torch
            length = int(torch.randint(self.min_length, self.max_length + 1, (1,)).item())
            x = torch.randint(2, self.vocab_size, (length,), dtype=torch.long)
            x = torch.cat([x, torch.tensor([1])])
            y = x.clone()

            yield x, y

def collate_scalar_to_sequence(batch):
    xs, ys = zip(*batch)
    lengths = torch.tensor([len(x) for x in xs])
    xs = pad_sequence(xs, batch_first=True, padding_value=0)
    ys = pad_sequence(ys, batch_first=True, padding_value=0)
    return xs, ys, lengths