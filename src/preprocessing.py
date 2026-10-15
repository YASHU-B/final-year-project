"""Image transformations for baseline and targeted-augmentation experiments."""

from __future__ import annotations

from torchvision import transforms

from .constants import IMAGENET_MEAN, IMAGENET_STD


def build_evaluation_transform(image_size: int = 224) -> transforms.Compose:
    """Build deterministic preprocessing for validation, test, and inference."""
    resize_size = round(image_size * 256 / 224)
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_base_training_transform(image_size: int = 224) -> transforms.Compose:
    """Build light augmentation applied to every training class."""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_targeted_training_transform(image_size: int = 224) -> transforms.Compose:
    """Build stronger, still clinically conservative transforms for rare classes.

    No vertical flip is used: it is mathematically valid for dermoscopic images,
    but excluding it keeps the augmentation policy easier to justify and audit.
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.80, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
