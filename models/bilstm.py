"""Bidirectional LSTM for sequence regression."""

import torch
import torch.nn as nn


class BiLSTMRegressor(nn.Module):
    """
    Bidirectional LSTM for sequence regression.
    
    Processes sequences with bidirectional LSTM, mean pooling, and regression head.
    
    Input: [N, L, M] where L = sequence length, M = input_size (features per timestep)
    Output: [N, 1] regression values
    """
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x_tokens: torch.Tensor) -> torch.Tensor:
        # x_tokens: [N, L, M] where M is input_size (event channels as features)
        out, _ = self.lstm(x_tokens)
        pooled = out.mean(dim=1)
        return self.head(pooled)
