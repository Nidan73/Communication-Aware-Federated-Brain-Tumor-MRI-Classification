# Results Paragraphs

The experiments used a leakage-controlled cluster-aware split with 5,040 training images, 720 validation images, and 1,440 test images. All reported federated results in the final assets use the 20-round configuration and the same four-class label set: glioma, meningioma, no tumor, and pituitary.

In the centralized benchmark, the best validation macro-F1 was obtained by `efficientnet_b0_lr0p0003`. This model achieved 97.71% test accuracy and 97.71% test macro-F1 on the held-out cluster-aware test split. These centralized results are used as context for selecting a compact backbone for the federated experiments.

Across seeds 42, 123, and 2025 under the same 4-client, 20-round Non-IID FedAvg setting, EfficientNet-B0 achieved 97.25 ± 0.51% accuracy and 97.25 ± 0.51% macro-F1, compared with 97.18 ± 0.28% for both metrics with ResNet50. The mean paired differences were 0.07 percentage points for accuracy and macro-F1, with sample SDs of 0.32 percentage points. Exact McNemar tests were not significant for any seed, all Holm-adjusted p-values were 1.0, and every class-stratified paired-bootstrap 95% confidence interval included zero.

EfficientNet-B0 and ResNet50 therefore provided comparable classification performance, without a formal equivalence claim. EfficientNet-B0 reduced full-state communication payload per 20-round run from 14,385.65 MiB to 2,474.84 MiB, approximately 82.8% lower. This comparison is limited to three seeds, the evaluated dataset split, simulated clients, the fixed test set, and the stated communication accounting method.
