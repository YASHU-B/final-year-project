"""Baseline classification models used in controlled ablation experiments."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from .cbam import CBAM


def build_efficientnet_b0_classifier(
    num_classes: int = 7,
    *,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> nn.Module:
    """Return an EfficientNet-B0 classifier adapted to HAM10000's seven classes.

    ImageNet weights supply general visual features. During training, the final
    classifier and then the convolutional backbone are fine-tuned on HAM10000.
    This baseline intentionally has no CBAM and uses no class balancing method.
    """
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)
    input_features = model.classifier[-1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(input_features, num_classes),
    )
    return model


class EfficientNetB0CBAM(nn.Module):
    """EfficientNet-B0 with one CBAM at its final 1280-channel feature map.

    This placement refines the backbone's high-level lesion representation,
    retains EfficientNet's pretrained feature extractor, and adds limited
    computational overhead compared with attaching CBAM to every MBConv block.
    """

    def __init__(
        self,
        num_classes: int = 7,
        *,
        pretrained: bool = True,
        dropout: float = 0.2,
        reduction_ratio: int = 16,
    ) -> None:
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        base_model = efficientnet_b0(weights=weights)
        self.features = base_model.features
        self.cbam = CBAM(channels=1280, reduction_ratio=reduction_ratio)
        self.average_pool = base_model.avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(1280, num_classes),
        )

    def forward(self, images):
        features = self.features(images)
        refined_features = self.cbam(features)
        pooled_features = self.average_pool(refined_features)
        return self.classifier(pooled_features.flatten(start_dim=1))


def build_efficientnet_b0_cbam_classifier(
    num_classes: int = 7,
    *,
    pretrained: bool = True,
    dropout: float = 0.2,
    reduction_ratio: int = 16,
) -> nn.Module:
    """Build the proposed EfficientNet-B0 + CBAM classifier."""
    return EfficientNetB0CBAM(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        reduction_ratio=reduction_ratio,
    )
