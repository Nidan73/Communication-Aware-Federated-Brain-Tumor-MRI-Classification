# Three-seed Non-IID Reproducibility Extension

This extension prepares four new runs only: EfficientNet-B0 and ResNet50 at seeds 123 and 2025. Seed 42 is reused from `federated_results_20rounds/`. No command below rebuilds the train/validation/test manifests.

The runner refuses to start training when its configured output directory already exists. Run experiments sequentially to avoid GPU-memory contention.

## 1. Activate the existing environment in fish

```fish
cd "/home/nidan73/Downloads/Brain Tumor paper"
source .venv/bin/activate.fish
```

Confirm PyTorch sees the CUDA GPU before any dry-run or training command:

```fish
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected output begins with `True`. Every training command below passes `--device cuda`; CPU is not used for training.

## 2. Validate saved client partitions

The three partition CSVs have already been generated once. This command verifies them without changing them:

```fish
python scripts/prepare_client_partitions.py --validate-only
python scripts/validate_reproducibility_outputs.py --partitions-only
```

Both model configs for seed 123 point to `reproducibility/client_partitions/noniid_seed123.csv`. Both seed-2025 configs point to `reproducibility/client_partitions/noniid_seed2025.csv`.

## 3. Validate and dry-run each configuration

Configuration-only validation does not load images or models:

```fish
python scripts/run_federated_benchmark.py --config configs/reproducibility/fedavg_efficientnet_b0_noniid_seed123.yaml --validate-config --device cpu
python scripts/run_federated_benchmark.py --config configs/reproducibility/fedavg_resnet50_noniid_seed123.yaml --validate-config --device cpu
python scripts/run_federated_benchmark.py --config configs/reproducibility/fedavg_efficientnet_b0_noniid_seed2025.yaml --validate-config --device cpu
python scripts/run_federated_benchmark.py --config configs/reproducibility/fedavg_resnet50_noniid_seed2025.yaml --validate-config --device cpu
```

Full dry-run checks load one image batch, instantiate the ImageNet-pretrained model, and perform one forward pass. They do not train or write result directories:

```fish
python scripts/run_federated_benchmark.py --config configs/reproducibility/fedavg_efficientnet_b0_noniid_seed123.yaml --dry-run --device cuda
python scripts/run_federated_benchmark.py --config configs/reproducibility/fedavg_resnet50_noniid_seed123.yaml --dry-run --device cuda
python scripts/run_federated_benchmark.py --config configs/reproducibility/fedavg_efficientnet_b0_noniid_seed2025.yaml --dry-run --device cuda
python scripts/run_federated_benchmark.py --config configs/reproducibility/fedavg_resnet50_noniid_seed2025.yaml --dry-run --device cuda
```

## 4. Run the four new experiments one at a time

Run these manually. Each command starts from ImageNet weights and writes a unique output directory:

```fish
python scripts/run_federated_benchmark.py --config configs/reproducibility/fedavg_efficientnet_b0_noniid_seed123.yaml --device cuda
python scripts/run_federated_benchmark.py --config configs/reproducibility/fedavg_resnet50_noniid_seed123.yaml --device cuda
python scripts/run_federated_benchmark.py --config configs/reproducibility/fedavg_efficientnet_b0_noniid_seed2025.yaml --device cuda
python scripts/run_federated_benchmark.py --config configs/reproducibility/fedavg_resnet50_noniid_seed2025.yaml --device cuda
```

Safe sequential fish function; later runs stop if an earlier run fails:

```fish
function run_multiseed_sequential
    python scripts/run_federated_benchmark.py --config configs/reproducibility/fedavg_efficientnet_b0_noniid_seed123.yaml --device cuda; or return 1
    python scripts/run_federated_benchmark.py --config configs/reproducibility/fedavg_resnet50_noniid_seed123.yaml --device cuda; or return 1
    python scripts/run_federated_benchmark.py --config configs/reproducibility/fedavg_efficientnet_b0_noniid_seed2025.yaml --device cuda; or return 1
    python scripts/run_federated_benchmark.py --config configs/reproducibility/fedavg_resnet50_noniid_seed2025.yaml --device cuda; or return 1
end
run_multiseed_sequential
```

## 5. Reuse or evaluate seed-42 checkpoints without retraining

Existing seed-42 prediction CSVs were found and packaged, so this command now verifies the packages and performs no inference. In a clean copy where a source prediction CSV is absent, the flag permits test-only checkpoint inference; it never trains:

```fish
python scripts/prepare_seed42_results.py --device cuda --evaluate-if-predictions-missing
python scripts/validate_reproducibility_outputs.py --seed42-only
```

## 6. Generate per-image prediction CSVs

Each new training command automatically evaluates its selected best checkpoint once and writes `predictions_<run_id>.csv`, sorted by stable `sample_id`. The seed-42 command above reuses the existing CSVs or performs checkpoint evaluation only when they are absent. No separate training pipeline is used.

Verify all six files after the four runs:

```fish
python scripts/validate_reproducibility_outputs.py
```

## 7. Produce mean and sample standard deviation

Run only after all six result packages exist:

```fish
python scripts/summarize_multiseed_results.py
```

Outputs:

- `reproducibility/summary/multiseed_results.csv`
- `reproducibility/summary/multiseed_results.md`
- `reproducibility/summary/multiseed_results.json`

## 8. Run paired statistical tests

```fish
python scripts/paired_model_statistics.py --bootstrap-resamples 10000 --bootstrap-seed 104729
```

Outputs:

- `reproducibility/statistics/paired_statistics.csv`
- `reproducibility/statistics/paired_statistics.md`
- `reproducibility/statistics/paired_statistics.json`

## 9. Validate all outputs

```fish
python scripts/validate_reproducibility_outputs.py
```

The validator checks all six model-seed combinations, seed-42 reuse provenance, same-seed partition identity, 1,440 unique aligned predictions, recalculated metrics, checkpoint state size, communication accounting, original-result hashes, and manifest overlap.

## Reproducibility boundary

Python, NumPy, PyTorch CPU/CUDA, DataLoader generators, and worker RNGs are seeded. Deterministic PyTorch operations are requested with warning-only behavior for unsupported operations. Perfect bit-for-bit reproducibility is not guaranteed across different GPU, CUDA, cuDNN, PyTorch, driver, operating-system, or hardware versions.
