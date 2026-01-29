import torch
import torch.nn as nn


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out[..., : x.size(-1)] + x


class TemporalConvNet(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, num_blocks: int = 4, kernel_size: int = 3):
        super().__init__()
        self.stem = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        self.blocks = nn.Sequential(
            *[TemporalBlock(hidden_channels, kernel_size, dilation=2**i) for i in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x
