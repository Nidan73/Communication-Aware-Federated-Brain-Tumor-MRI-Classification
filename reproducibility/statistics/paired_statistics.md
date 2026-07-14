# Paired EfficientNet-B0 vs ResNet50 Statistics

Differences are EfficientNet-B0 minus ResNet50. Exact two-sided McNemar tests and paired bootstrap resampling use aligned test-image indices.

Bootstrap stratified by true class: yes (class-stratified by true label; each production resample preserves 360 observations per class).
McNemar p-values are reported raw and with Holm adjustment across the three seeds. Fisher's method is not used.

| Seed | Both correct | Both wrong | Only EfficientNet correct | Only ResNet correct | Raw exact McNemar p | Holm-adjusted p | Accuracy diff (95% CI) | Macro-F1 diff (95% CI) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 1385 | 19 | 21 | 15 | 0.405032 | 1 | 0.42 pp [-0.42, 1.25] | 0.41 pp [-0.40, 1.23] |
| 123 | 1372 | 25 | 20 | 23 | 0.760792 | 1 | -0.21 pp [-1.11, 0.69] | -0.21 pp [-1.11, 0.69] |
| 2025 | 1391 | 25 | 12 | 12 | 1 | 1 | 0.00 pp [-0.69, 0.69] | -0.00 pp [-0.69, 0.68] |

## Fixed-test-set interpretation

- Seed 42: raw test—not statistically distinguishable on the fixed test set; familywise result—not statistically distinguishable after Holm adjustment on the fixed test set; accuracy confidence interval includes zero; macro-F1 confidence interval includes zero.
- Seed 123: raw test—not statistically distinguishable on the fixed test set; familywise result—not statistically distinguishable after Holm adjustment on the fixed test set; accuracy confidence interval includes zero; macro-F1 confidence interval includes zero.
- Seed 2025: raw test—not statistically distinguishable on the fixed test set; familywise result—not statistically distinguishable after Holm adjustment on the fixed test set; accuracy confidence interval includes zero; macro-F1 confidence interval includes zero.

These statements concern paired behavior on this fixed 1,440-image test set. They do not automatically establish model superiority or an overall statistical victory.

## Across-seed paired differences

| Seed | Accuracy difference | Macro-F1 difference |
|---:|---:|---:|
| 42 | 0.42 pp | 0.41 pp |
| 123 | -0.21 pp | -0.21 pp |
| 2025 | 0.00 pp | -0.00 pp |

Mean accuracy difference: 0.07 pp (sample SD 0.32 pp).

Mean macro-F1 difference: 0.07 pp (sample SD 0.32 pp).

Only three paired seed observations and three repeated fixed-test-set McNemar tests are available. Report raw and Holm-adjusted p-values, effect direction, and variability; do not claim an overall statistical victory. Fisher's method is not used.
