# CBAM and Focal Loss Design

## CBAM

CBAM is a lightweight attention module with two sequential operations:

1. **Channel attention** asks *what feature types matter?* It uses global average
   pooling and max pooling over each channel, then produces one weight per
   channel.
2. **Spatial attention** asks *where in the feature map does useful information
   occur?* It uses channel-wise average and max feature maps, then produces one
   weight per spatial location.

The module refines a feature map `F` as:

```text
F'  = channel_attention(F) × F
F'' = spatial_attention(F') × F'
```

For this project, one CBAM module is placed after EfficientNet-B0's final
1280-channel convolutional feature map and before global average pooling. This
is intentionally a lightweight design: it evaluates attention without altering
every pretrained MBConv block or making the ablation hard to interpret.

CBAM can help the model emphasize discriminative lesion texture, colour, and
border patterns while suppressing irrelevant background. It does not guarantee
that the highlighted region is medically correct; Grad-CAM analysis later will
only be a cautious inspection of model behaviour.

## Focal Loss

Standard cross-entropy gives every example equal emphasis. In a long-tailed
dataset, the dominant `nv` class can therefore dominate the optimization signal.

Multi-class Focal Loss is:

```text
FL = -alpha_t × (1 - p_t)^gamma × log(p_t)
```

- `p_t` is the model probability for the true class.
- `gamma` reduces the contribution of already easy, correctly classified
  examples. We begin with `gamma = 2.0`, the common starting value, and will
  report it as an experimental hyperparameter rather than an established optimum.
- `alpha_t` is a class-specific weight. We calculate it from **training-split
  counts only** using inverse effective-number weighting. This avoids using
  validation/test labels while assigning higher influence to rare classes.

We start with `beta = 0.999`, which is less aggressive than `0.9999`; using a
very high beta together with Focal Loss can over-correct the minority classes.
This choice must be tested against the baseline and recorded in the experiment
configuration.

## Targeted training augmentation

All training images receive light random crop and horizontal-flip augmentation.
Classes with fewer than 1,000 training images (`akiec`, `bcc`, `bkl`, `df`,
`mel`, `vasc`) receive an additional conservative rotation and mild colour
jitter policy. Validation and test images receive deterministic preprocessing
only.

Online augmentation changes image appearance, not dataset membership or split
counts. Focal Loss is the component that directly adjusts class contribution to
the loss.
