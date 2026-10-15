"""Loss functions for imbalanced multi-class skin-lesion classification."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as functional


def effective_number_class_weights(
    class_counts: Sequence[int],
    beta: float = 0.9999,
) -> torch.Tensor:
    """Return normalized inverse-effective-number class weights.

    The weighting is calculated exclusively from the training split. It gives
    minority classes greater influence without directly duplicating images.
    """
    counts = torch.as_tensor(class_counts, dtype=torch.float32)
    if torch.any(counts <= 0):
        raise ValueError("All classes must have at least one training sample.")
    if not 0.0 <= beta < 1.0:
        raise ValueError("beta must be in the interval [0, 1).")
    effective_numbers = 1.0 - torch.pow(torch.tensor(beta), counts)
    weights = (1.0 - beta) / effective_numbers
    return weights / weights.mean()


class FocalLoss(nn.Module):
    """Multi-class Focal Loss using softmax probabilities.

    FL = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Gamma=0 reduces the focusing term to class-weighted cross-entropy.
    """

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be non-negative.")
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probabilities = functional.log_softmax(logits, dim=1)
        log_probability_true_class = log_probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
        probability_true_class = log_probability_true_class.exp()
        focal_factor = torch.pow(1.0 - probability_true_class, self.gamma)
        loss = -focal_factor * log_probability_true_class
        if self.alpha is not None:
            loss = self.alpha[targets] * loss
        return loss.mean()
