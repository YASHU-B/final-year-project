"""Project-wide constants for HAM10000 classification."""

from __future__ import annotations


CLASS_NAMES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
CLASS_TO_INDEX = {class_name: index for index, class_name in enumerate(CLASS_NAMES)}
INDEX_TO_CLASS = {index: class_name for class_name, index in CLASS_TO_INDEX.items()}

# ImageNet normalization is appropriate because EfficientNet-B0 is initialized
# from ImageNet-pretrained weights.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# These have fewer than 1,000 training examples in the initial grouped split.
# The final list is configurable in experiment YAML files.
DEFAULT_TARGETED_AUGMENTATION_CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "vasc")
