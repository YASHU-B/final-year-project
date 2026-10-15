"""Convolutional Block Attention Module (CBAM) for feature-map refinement."""

from __future__ import annotations

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Learn which feature channels are important (the 'what' attention)."""

    def __init__(self, channels: int, reduction_ratio: int = 16) -> None:
        super().__init__()
        hidden_channels = max(channels // reduction_ratio, 1)
        self.average_pool = nn.AdaptiveAvgPool2d(1)
        self.maximum_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
        )
        self.activation = nn.Sigmoid()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        average_attention = self.shared_mlp(self.average_pool(features))
        maximum_attention = self.shared_mlp(self.maximum_pool(features))
        return self.activation(average_attention + maximum_attention)


class SpatialAttention(nn.Module):
    """Learn which image locations are important (the 'where' attention)."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        if kernel_size not in (3, 7):
            raise ValueError("CBAM spatial-attention kernel_size must be 3 or 7.")
        padding = kernel_size // 2
        self.convolution = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.activation = nn.Sigmoid()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        average_features = features.mean(dim=1, keepdim=True)
        maximum_features, _ = features.max(dim=1, keepdim=True)
        combined_features = torch.cat((average_features, maximum_features), dim=1)
        return self.activation(self.convolution(combined_features))


class CBAM(nn.Module):
    """Sequential channel and spatial attention from Woo et al. (ECCV 2018)."""

    def __init__(
        self,
        channels: int,
        reduction_ratio: int = 16,
        spatial_kernel_size: int = 7,
    ) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        refined = self.channel_attention(features) * features
        return self.spatial_attention(refined) * refined
