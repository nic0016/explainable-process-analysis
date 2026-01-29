import torch
import torch.nn as nn


class GatedResidualNetwork(nn.Module):
    def __init__(self, d_in: int, d_hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden, d_in),
        )
        self.gate = nn.Sequential(
            nn.Linear(d_in, d_in),
            nn.Sigmoid(),
        )
        self.ln = nn.LayerNorm(d_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        g = self.gate(x)
        return self.ln(x + g * y)


class LightTFTRegressor(nn.Module):
    def __init__(self, input_size: int, d_model: int = 128, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=d_model // 2, 
            num_layers=1, 
            bidirectional=True, 
            batch_first=True
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.temporal_attn = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.grn = GatedResidualNetwork(d_in=d_model, d_hidden=d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        y, _ = self.lstm(x)
        y = self.temporal_attn(y, src_key_padding_mask=padding_mask)
        pooled = y.mean(dim=1)
        pooled = self.grn(pooled)
        return self.head(pooled)
