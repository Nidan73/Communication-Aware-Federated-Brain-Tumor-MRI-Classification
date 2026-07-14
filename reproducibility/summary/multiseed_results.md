# Multi-seed Non-IID Results

All means and sample standard deviations use unrounded seed-level values. Standard deviations use denominator n-1.

| Model | Seeds | Accuracy (%) mean ± SD | Macro-F1 (%) mean ± SD | State (MiB) | Communication (MiB) |
|---|---:|---:|---:|---:|---:|
| EfficientNet-B0 | 42,123,2025 | 97.25 ± 0.51 | 97.25 ± 0.51 | 15.47 | 2474.84 |
| ResNet50 | 42,123,2025 | 97.18 ± 0.28 | 97.18 ± 0.28 | 89.91 | 14385.65 |

## Seed-level values

| Model | Seed | Accuracy | Macro-F1 | Best round |
|---|---:|---:|---:|---:|
| EfficientNet-B0 | 42 | 0.976389 | 0.976392 | 20 |
| EfficientNet-B0 | 123 | 0.966667 | 0.966697 | 6 |
| EfficientNet-B0 | 2025 | 0.974306 | 0.974332 | 20 |
| ResNet50 | 42 | 0.972222 | 0.972270 | 15 |
| ResNet50 | 123 | 0.968750 | 0.968823 | 13 |
| ResNet50 | 2025 | 0.974306 | 0.974357 | 18 |

## Best-round stability

- EfficientNet-B0: mean 15.33; range 6–20.
- ResNet50: mean 15.33; range 13–18.
