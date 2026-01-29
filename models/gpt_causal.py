import torch
import torch.nn as nn


def generate_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * float('-inf'), diagonal=1)
    return mask


class GPTStyleRegressor(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, nhead: int = 4, num_layers: int = 2, max_len: int = 1000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
            for _ in range(num_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 1),
        )

    def forward(self, token_ids: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        N, L = token_ids.size()
        device = token_ids.device
        pos = torch.arange(L, device=device).unsqueeze(0).expand(N, L)
        x = self.embedding(token_ids) + self.pos_embedding(pos)
        causal_mask = generate_causal_mask(L, device)
        for layer in self.layers:
            x = layer(x, src_mask=causal_mask, src_key_padding_mask=padding_mask)
        pooled = x[:, -1, :]
        return self.head(pooled)
