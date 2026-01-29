"""BERT-style Transformer Encoder for sequence regression."""

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer models."""
    
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, L, D]
        L = x.size(1)
        return x + self.pe[:, :L]


class BertStyleRegressor(nn.Module):
    """
    BERT-style Transformer Encoder for sequence regression.
    
    Uses embedding layer, positional encoding, TransformerEncoder layers, 
    mean pooling, and a regression head.
    
    Input: token_ids [N, L] with optional padding_mask
    Output: [N, 1] regression values
    """
    
    def __init__(self, vocab_size: int, d_model: int = 128, nhead: int = 4, num_layers: int = 2, max_len: int = 1000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model, max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 1),
        )

    def forward(self, token_ids: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # token_ids: [N, L]
        x = self.embedding(token_ids)
        x = self.pos(x)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        pooled = x.mean(dim=1)
        return self.head(pooled)
