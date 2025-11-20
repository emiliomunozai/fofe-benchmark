import torch
import math

class GRUCatSeq2CatSeqDecoder(torch.nn.Module):
    def __init__(
        self,
        embed_dim, 
        recurrence_dim, 
        num_layers, 
        bidirectional, 
        vocab_size):
        
        super().__init__()

        # Configurations
        self.embed_dim = embed_dim
        self.recurrence_dim = recurrence_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.vocab_size = vocab_size + 1

        # Layers
        self.embedding = torch.nn.Embedding(
            self.vocab_size, 
            embed_dim, 
            padding_idx=0)

        self.GRU = torch.nn.GRU(
            embed_dim, 
            recurrence_dim, 
            num_layers, 
            bidirectional=bidirectional, 
            batch_first=True)

        dir_factor = 2 if bidirectional else 1

        self.decoder = torch.nn.Linear(recurrence_dim * dir_factor, self.vocab_size)
    
    def forward(self, x):
        embed = self.embedding(x)           # (B, L, embed_dim)
        hidden_seq, _ = self.GRU(embed)     # (B, L, recurrence_dim * dir_factor)
        logits = self.decoder(hidden_seq)   # (B, L, vocab_size)
        return logits

class LSTMCatSeq2CatSeqDecoder(torch.nn.Module):
    def __init__(
        self,
        embed_dim, 
        hidden_dim, 
        num_layers, 
        bidirectional, 
        vocab_size):
        
        super().__init__()

        # Configurations
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.vocab_size = vocab_size + 1

        # Layers
        self.embedding = torch.nn.Embedding(
            self.vocab_size, 
            embed_dim, 
            padding_idx=0)

        self.rnn = torch.nn.LSTM(
            embed_dim, 
            hidden_dim, 
            num_layers, 
            bidirectional=bidirectional, 
            batch_first=True)

        dir_factor = 2 if bidirectional else 1

        self.decoder = torch.nn.Linear(hidden_dim * dir_factor, self.vocab_size)
    
    def forward(self, x):
        embed = self.embedding(x)           # (B, L, embed_dim)
        hidden_seq, _ = self.rnn(embed)     # (B, L, hidden_dim * dir_factor)
        logits = self.decoder(hidden_seq)   # (B, L, vocab_size)
        return logits

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, L, H)
        x = x + self.pe[:, :x.size(1)]
        return x

class TransformerCatSeq2CatSeqDecoder(torch.nn.Module):
    def __init__(
        self,
        embed_dim, 
        hidden_dim, 
        num_layers, 
        bidirectional, 
        vocab_size,
        head_dim=64,
        num_heads=4):
        
        super().__init__()

        # Configurations
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.vocab_size = vocab_size + 1
        self.head_dim = head_dim
        self.num_heads = num_heads

        # Layers

        # Embedding
        self.embedding = torch.nn.Embedding(self.vocab_size, embed_dim, padding_idx=0)

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(hidden_dim)

        # Transformer encoder
        nhead = num_heads if num_heads is not None else (hidden_dim // head_dim)
        assert nhead > 0 and hidden_dim % nhead == 0, f"hidden_dim must be divisible by nhead ({nhead})"

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )

        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj = torch.nn.Linear(embed_dim, hidden_dim) if embed_dim != hidden_dim else torch.nn.Identity()
        self.decoder = torch.nn.Linear(hidden_dim, self.vocab_size)
    
    def forward(self, x):
        embed = self.embedding(x)  # (B, L, embed_dim)
        embed = self.proj(embed)  # (B, L, hidden_dim)

        embed = self.positional_encoding(embed)  # Add positional encoding

        src_key_padding_mask = (x == 0)  # (B, L)

        if not self.bidirectional:
            L = x.size(1)
            src_mask = torch.triu(torch.ones((L, L), device=x.device) * float('-inf'), diagonal=1)
        else:
            src_mask = None

        hidden_seq = self.transformer(
            embed,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask
        )  # (B, L, hidden_dim)

        logits = self.decoder(hidden_seq)  # (B, L, vocab_size)
        return logits

class MLP(torch.nn.Module):
    def __init__(self, vocab_size, max_seq_len):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size + 1
        self.hidden_dim = 128
        self.embedding_dim = 10

        self.embedding = torch.nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)

        self.fc1 = torch.nn.Linear(self.embedding_dim * max_seq_len, self.hidden_dim)
        self.fc_out = torch.nn.Linear(self.hidden_dim, max_seq_len * self.vocab_size)

    def forward(self, x):

        # Input dimentions
        batch_size, seq_len = x.shape

        # Embedding
        embedding = self.embedding(x)  # (batch, max_seq_len, embedding_dim)       

        # Feed forward
        flat = embedding.view(batch_size, -1)  # (batch, max_seq_len * embedding_dim)
        h = F.relu(self.fc1(flat))
        out = self.fc_out(h)  # (batch, max_seq_len * vocab_size)
        out = out.view(batch_size, self.max_seq_len, self.vocab_size) # (batch, seq_len, vocab_size)
        return out
  
class SequenceAutoencoder(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, pad_idx=0):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = torch.nn.LSTM(embed_dim, hidden_dim, batch_first=True, num_layers=2)
        self.decoder = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.lstm(x)
        logits = self.decoder(output)
        return logits