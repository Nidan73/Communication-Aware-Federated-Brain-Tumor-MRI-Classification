# Paper Results Summary

This folder contains publication-ready tables and figures generated only from completed saved outputs in `centralized_benchmark_results/`, `federated_results_20rounds/`, and `cluster_split_outputs/`. No model training, federated learning, or centralized benchmark rerun was performed while generating these assets.

## Dataset Split Used

The experiments use the leakage-controlled cluster-aware split saved under `cluster_split_outputs/`: 5,040 training images, 720 validation images, and 1,440 test images. Each split is balanced across the four classes: glioma, meningioma, no tumor, and pituitary.

## Final Federated Results

The primary paper results are the 20-round federated results from `federated_results_20rounds/`. The final federated table is saved as:

- `tables/final_federated_results_table.csv`
- `tables/final_federated_results_table.md`

Under the same 4-client, 20-round Non-IID federated setting, EfficientNet-B0 achieved 97.64% accuracy and 97.64% macro-F1, while ResNet50 achieved 97.22% accuracy and 97.23% macro-F1. EfficientNet-B0 reduced total communication from 14,385.65 MB to 2,474.84 MB, corresponding to about 82.8% lower communication.

The comparison should be interpreted cautiously: the results are specific to this cluster-aware split, the selected training configuration, and full-state federated communication accounting. The results do not establish state-of-the-art performance.

## Communication Findings

Communication cost was computed from the full model `state_dict` upload and download bytes per round. EfficientNet-B0 used 15.47 MB per state dictionary, whereas ResNet50 used 89.91 MB. Because both runs used the same number of clients and communication rounds, the total communication ratio follows the state-dict size ratio.

The communication comparison table is saved as:

- `tables/communication_comparison_table.csv`
- `tables/communication_comparison_table.md`

## Centralized Benchmark Context

The centralized benchmark table is generated from `centralized_benchmark_results/model_comparison_val.csv` and the selected final test row in `centralized_benchmark_results/final_test_metrics.csv`. The selected centralized model was `efficientnet_b0_lr0p0003`, with 97.71% test accuracy and 97.71% test macro-F1.

The centralized table is saved as:

- `tables/centralized_backbone_table.csv`
- `tables/centralized_backbone_table.md`

## Generated Figures

- `figures/fl_macro_f1_comparison.png`
- `figures/fl_accuracy_comparison.png`
- `figures/communication_comparison.png`
- `figures/communication_reduction_vs_resnet50.png`
- `figures/convergence_val_macro_f1.png`
- `figures/confusion_matrix_fedavg_efficientnet_b0_noniid_20r.png`
- `figures/confusion_matrix_fedavg_resnet50_noniid_20r.png`

All figures were generated with matplotlib and saved at 300 DPI.
