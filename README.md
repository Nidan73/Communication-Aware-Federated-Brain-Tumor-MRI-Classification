# Trading Payload for Comparable Accuracy

Code and results for **"Trading Payload for Comparable Accuracy: A Duplicate-Controlled, Communication-Aware Federated Study of Brain Tumor MRI Backbones"**, presented at the IEEE 3rd International Conference on Computing, Applications and Systems (COMPAS 2026), University of Dhaka, 9 to 10 October 2026. First author: Nidan Alam.

## Question

In federated learning every round sends the full model up to the server and back, so model size is a direct communication cost. Can a smaller backbone match a larger one on four-class brain tumor MRI classification once duplicate images are controlled?

## Data

Brain tumor MRI in four classes (glioma, meningioma, no tumor, pituitary), 7,200 images. Exact, pixel-level and near-duplicate images were grouped into 6,116 clusters (596 with more than one image) before splitting, so near-copies never cross splits (`cluster_split_outputs/CLUSTER_SPLIT_SUMMARY.md`):

| Split | Per class | Total |
| --- | --- | --- |
| Train | 1,260 | 5,040 |
| Validation | 180 | 720 |
| Test | 360 | 1,440 |

The images themselves are not included.

## Method

- **Backbone screening** (centralized, best validation macro-F1; `paper_assets_final/table_centralized_screening.md`): ResNet50 0.9750, DenseNet121 0.9778, EfficientNet-B0 0.9903, EfficientNet-B3 0.9847, ConvNeXt-Tiny 0.9889.
- **Federated training** (`configs/fl_final_20rounds.yaml`): FedAvg weighted by client sample count, 4 clients, 20 rounds, 3 local epochs, AdamW (learning rate 3e-4, weight decay 1e-4), batch size 32. The non-IID split gives each client one dominant class holding 70% of that class's samples.
- **Statistics** (`scripts/paired_model_statistics.py`): seeds 42, 123 and 2025; exact McNemar tests with Holm correction and class-stratified paired bootstrap on the same 1,440 test images.

## Results

Non-IID, mean and sample standard deviation over three seeds (`reproducibility/summary/multiseed_results.md`):

| Backbone | Accuracy | Macro-F1 | State dict | Payload per 20-round run |
| --- | --- | --- | --- | --- |
| EfficientNet-B0 | 97.25 +/- 0.51% | 97.25 +/- 0.51% | 15.47 MiB | 2,474.84 MiB |
| ResNet50 | 97.18 +/- 0.28% | 97.18 +/- 0.28% | 89.91 MiB | 14,385.65 MiB |

EfficientNet-B0 sends 82.8% less data than ResNet50 (`paper_anonymous_submission/tables/table_main_communication_comparison.md`). No seed shows a significant accuracy difference: the Holm-adjusted p-value is 1 for all three seeds and every 95% bootstrap interval includes zero (`reproducibility/statistics/paired_statistics.md`). The finding is comparable accuracy at a much smaller payload, not higher accuracy.

## How to run

Check the setup without training (manifests, class counts, client splits, model creation, one forward pass):

```bash
python scripts/run_federated_benchmark.py \
  --config configs/fl_efficientnet_b0.yaml configs/fl_resnet50_baseline.yaml \
  --dry-run
```

Train both backbones:

```bash
python scripts/run_federated_benchmark.py \
  --config configs/fl_efficientnet_b0.yaml configs/fl_resnet50_baseline.yaml
```

Multi-seed reproducibility and statistics:

```bash
python scripts/prepare_client_partitions.py --validate-only
python scripts/validate_reproducibility_outputs.py --partitions-only
python scripts/summarize_multiseed_results.py
python scripts/paired_model_statistics.py
pytest tests/
```

More detail: `README_RUN_FL.md` (federated runner) and `README_MULTISEED.md` (three-seed extension).

## Layout

- `scripts/`: federated runner, client partitions, statistics, figures, output validation.
- `configs/`: experiment configs, including per-seed configs in `configs/reproducibility/`.
- `cluster_split_outputs/`, `duplicate_audit_outputs/`: duplicate audit and leakage-controlled split manifests.
- `reproducibility/`: client partitions, multi-seed summaries and paired statistics.
- `paper_assets_final/`: tables and figures used in the paper.
- `tests/`: tests for the reproducibility tools.

The repository does not include datasets, Python environments, checkpoints, cached pretrained weights or large training outputs.
