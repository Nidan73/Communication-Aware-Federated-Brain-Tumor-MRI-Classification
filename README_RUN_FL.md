# Federated Benchmark Runner

This benchmark uses only the leakage-controlled cluster manifests:

- `cluster_split_outputs/train_manifest.csv`
- `cluster_split_outputs/val_manifest.csv`
- `cluster_split_outputs/test_manifest.csv`

It does not use the old `Dataset/Nickparvar/Training` or `Dataset/Nickparvar/Testing` folder split as the experimental split.

## Files

- `scripts/run_federated_benchmark.py`
- `configs/fl_efficientnet_b0.yaml`
- `configs/fl_resnet50_baseline.yaml`

## Dry Run

Use dry run first. It checks manifest loading, class counts, IID/Non-IID client splits, model creation, parameter counts, and one forward pass. It does not train.

```bash
python scripts/run_federated_benchmark.py \
  --config configs/fl_efficientnet_b0.yaml configs/fl_resnet50_baseline.yaml \
  --dry-run
```

## Main Run

Run both configs together so `final_comparison_table.csv` and `communication_costs.csv` include EfficientNet-B0 and ResNet50.

```bash
python scripts/run_federated_benchmark.py \
  --config configs/fl_efficientnet_b0.yaml configs/fl_resnet50_baseline.yaml
```

## Experiments

- `fedavg_efficientnet_b0_iid`
- `fedavg_efficientnet_b0_noniid`
- `fedprox_efficientnet_b0_noniid`
- `fedavg_resnet50_noniid`

Each run uses 4 clients, 10 communication rounds, 3 local epochs, batch size 32, AdamW, learning rate `0.0003`, weight decay `0.0001`, seed 42, and ImageNet-pretrained weights.

## Client Splits

- IID: each class is shuffled with seed 42 and split evenly across 4 clients.
- Non-IID: each class assigns 70% of samples to its dominant client and the remaining 30% across the other clients.

The runner saves:

- `federated_results/client_splits_iid.csv`
- `federated_results/client_splits_noniid.csv`

## Outputs

Outputs are written to `federated_results/`:

- `config_used.json`
- `progress_status.json`
- `progress_events.csv`
- `client_splits_iid.csv`
- `client_splits_noniid.csv`
- `metrics_per_round_<run_id>.csv`
- `classification_report_<run_id>.json`
- `confusion_matrix_<run_id>.csv`
- `predictions_<run_id>.csv`
- `communication_costs.csv`
- `final_comparison_table.csv`
- `FEDERATED_RESULTS_SUMMARY.md`
- `checkpoints/`

Communication cost counts full `state_dict` tensor bytes for every server-to-client download and every client-to-server upload in each round.

## Progress During Training

When output is piped through `tee`, the runner prints one clean progress line after every completed client-local epoch and disables noisy tqdm carriage-return bars. For the ResNet50 baseline config, the denominator is:

```text
10 rounds x 4 clients x 3 local epochs = 120 client-epochs
```

The latest status is also saved continuously:

```bash
cat federated_results/progress_status.json
```

The full event history is saved here:

```bash
tail -n 20 federated_results/progress_events.csv
```
