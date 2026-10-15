# Skin Lesion Research Project

**Working title:** Leakage-Aware, Class-Balanced, and Explainable EfficientNet-B0-CBAM Framework for Multi-Class Skin Lesion Classification

This repository contains the reproducible implementation and academic materials for a seven-class HAM10000 skin-lesion classification project. It is an academic research prototype, not a medical diagnostic system.

## Current status

Phase 1 (research validation) is complete. The next task is a metadata audit of the HAM10000 dataset before any train/validation/test split is created.

Phase 2 is complete: the dataset audit and lesion-grouped split are documented in
[`docs/dataset_protocol.md`](docs/dataset_protocol.md). The first training target
is the EfficientNet-B0 + cross-entropy baseline defined in
`configs/baseline_efficientnet_b0.yaml`.

To resume an interrupted baseline run, retain the checkpoint and run:

```bash
./.venv/bin/python -m src.train \
  --config configs/baseline_efficientnet_b0.yaml \
  --resume checkpoints/baseline_efficientnet_b0_cross_entropy/best_model.pt
```

## Layout

- `data/` — locally downloaded data and generated dataset splits; excluded from Git.
- `src/` — reusable PyTorch source modules.
- `configs/` — versioned experiment configurations.
- `notebooks/` — exploratory work only; training logic stays in `src/`.
- `results/` and `figures/` — generated metrics and visual outputs.
- `docs/` — report and presentation materials.
- `app/` — reserved for the later demonstration application.

## Safety note

Predictions and explanations produced by this project are for academic research only and must not be used as a medical diagnosis.
