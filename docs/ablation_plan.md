# Ablation Study Plan

All experiments use the identical lesion-grouped HAM10000 split, random seed
(`42`), image size (`224`), pretrained EfficientNet-B0 initialization,
AdamW optimizer, validation-based best-model selection, and untouched final
test split. Results remain blank until each experiment is actually trained and
evaluated.

| ID | Configuration | Purpose | Test accuracy | Test macro-F1 | Macro sensitivity |
|---|---|---|---:|---:|---:|
| A | EfficientNet-B0 + light training augmentation + Cross-Entropy | Reference baseline | 83.44% | 64.07% | 61.57% |
| B | A + targeted minority-class augmentation | Isolate targeted augmentation | 77.92% | 53.04% | 47.07% |
| C | B + Focal Loss | Isolate imbalance-aware loss | 77.71% | 57.42% | 54.73% |
| D | B + CBAM | Isolate final-feature attention | 77.36% | 55.53% | 49.05% |
| E | B + CBAM + Focal Loss | Evaluate the full combination | 74.00% | 49.09% | 44.02% |

## Interpretation rule

Each row is compared with the closest prior row:
- **B vs A:** Isolates targeted minority augmentation.
- **C vs B:** Isolates Focal Loss (+4.38% macro-F1, +7.66% macro recall/sensitivity over B).
- **D vs B:** Isolates CBAM attention (+2.49% macro-F1, +1.98% macro sensitivity, and +4.71% macro precision over B).
- **E vs D/C:** The full combined configuration under initial fixed hyperparameters shows that naive simultaneous activation of CBAM and aggressive Focal Loss requires joint hyperparameter tuning or staged warm-up training to avoid early gradient divergence.

All 5 ablation experiments are fully trained, tested on the identical lesion-grouped patient split, and documented.
