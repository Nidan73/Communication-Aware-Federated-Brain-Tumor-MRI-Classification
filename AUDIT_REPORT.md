# Brain Tumor Federated Learning Audit Report

Audit date: 2026-06-28  
Scope: local dataset under `Dataset/Nickparvar`, the two notebooks, and the paper PDF in this workspace. No existing notebook/script was modified and no training was run.

## Executive Summary

- The local dataset is not the paper-described/original Nickparvar Kaggle split. It is a balanced 7,200-image folder set with 1,400 training images and 400 testing images per class.
- `Brain_Tumor_Final_Notebook.ipynb` is the executed notebook and matches the paper's 5-round/15-epoch experiment budget. `Brain_Tumor_Final_Notebook_v2.ipynb` changes the budget to 10 rounds/30 epochs and has no saved outputs.
- The FL implementation returns and averages full `state_dict()` objects, not gradients, deltas, or trainable-only tensors.
- Communication cost is calculated using trainable parameters only, but the aggregation payload is the full model state. Frozen layers are excluded from the cost calculation while still present in the communicated/aggregated state.
- The paper has several verification mismatches: dataset totals, test-set size, Table 2 F1-scores, and the Non-IID confusion matrix prose do not match the executed notebook outputs.

## Files Audited

- `Dataset/Nickparvar`
- `Brain_Tumor_Final_Notebook.ipynb`
- `Brain_Tumor_Final_Notebook_v2.ipynb`
- `Privacy_Preserving_Brain_Tumor_Classification_Using_Federated_Learning_with_ResNet50_and_Grad_CAM (3).pdf`

## 1. Dataset Structure

### Exact Folder Tree

```text
Dataset/Nickparvar/
|-- Testing/
|   |-- glioma/      (400 jpg)
|   |-- meningioma/  (400 jpg)
|   |-- notumor/     (400 jpg)
|   `-- pituitary/   (400 jpg)
`-- Training/
    |-- glioma/      (1400 jpg)
    |-- meningioma/  (1400 jpg)
    |-- notumor/     (1400 jpg)
    `-- pituitary/   (1400 jpg)
```

All files under `Dataset/Nickparvar` are `.jpg` files. Total image files found: 7,200.

### Class Counts

| Split | Glioma | Meningioma | No Tumor | Pituitary | Total |
|---|---:|---:|---:|---:|---:|
| Training | 1,400 | 1,400 | 1,400 | 1,400 | 5,600 |
| Testing | 400 | 400 | 400 | 400 | 1,600 |
| Total | 1,800 | 1,800 | 1,800 | 1,800 | 7,200 |

### Original Kaggle Split Check

This appears to be a custom balanced split, not the original/paper-described Nickparvar Kaggle split.

Evidence:

- Local folders are perfectly balanced: 1,400 train and 400 test images per class.
- The paper's dataset section claims 7,023 images with class totals Glioma 1,621, Meningioma 1,645, Pituitary 1,757, No Tumor 2,000, split into 5,618 train and 1,405 test.
- The executed notebook output also reports 7,200 total images, not 7,023.
- Some filenames indicate augmented/curated content, for example `Testing/meningioma/Te-aug-me_*.jpg`.

Mismatch flagged: dataset totals in the paper do not match the local dataset or the executed notebook.

## 2. Notebook and Code Configuration

### Train/Test/Validation Split Locations

No validation split is created anywhere in either notebook.

Train/test separation is folder-based:

- `Brain_Tumor_Final_Notebook.ipynb`, cell 11:
  - `datasets.ImageFolder(f'{DATASET_PATH}/Training', transform=train_transform)`
  - `datasets.ImageFolder(f'{DATASET_PATH}/Testing', transform=test_transform)`
- `Brain_Tumor_Final_Notebook_v2.ipynb`, cell 11: same.

Client splits are created from the training set only:

- IID split: both notebooks, cell 18, `partition_iid(dataset, num_clients)`
  - Shuffles all training indices with `np.random.shuffle(indices)`
  - Splits using `np.array_split(indices, num_clients)`
  - Uses `Subset(dataset, idx)`
- Non-IID split: both notebooks, cell 18, `partition_noniid(dataset, num_clients)`
  - For each class, shuffles class indices
  - Sends 70 percent of each class to a dominant client
  - Splits the remaining 30 percent across the other clients
  - Uses `Subset(dataset, idx)`

Additional v2 experiments reuse these functions:

- FedProx Non-IID: v2 cell 27 calls `partition_noniid`
- Focal Loss IID: v2 cell 30 calls `partition_iid`
- Statistical multi-seed IID: v2 cell 32 calls `partition_iid`
- DP-SGD IID: v2 cell 34 calls `partition_iid`

### Image Transforms and Augmentations

Defined in cell 11 of both notebooks.

Training transform:

- `Resize((256, 256))`
- `CenterCrop(224)`
- `RandomHorizontalFlip()`
- `RandomRotation(15)`
- `ColorJitter(brightness=0.2, contrast=0.2)`
- `ToTensor()`
- ImageNet normalization: mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`

Testing transform:

- `Resize((256, 256))`
- `CenterCrop(224)`
- `ToTensor()`
- Same ImageNet normalization

### ResNet50 Creation and Frozen Layers

Defined in cell 13 of both notebooks:

```python
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
for name, param in model.named_parameters():
    if 'layer4' not in name and 'fc' not in name:
        param.requires_grad = False
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, CONFIG['num_classes'])
)
```

Frozen:

- All parameters whose names do not contain `layer4` or `fc`.

Trainable:

- `layer4`
- Replaced final classifier `fc`

Important nuance: the freeze loop runs before replacing `fc`. The new `fc` layer is trainable by default.

### Main Configuration

| Setting | `Brain_Tumor_Final_Notebook.ipynb` | `Brain_Tumor_Final_Notebook_v2.ipynb` |
|---|---:|---:|
| Seed | 42 | 42 |
| Number of clients | 3 | 3 |
| Communication rounds | 5 | 10 |
| Local epochs | 3 | 3 |
| Central epochs | 15 | 30 |
| Batch size | 32 | 32 |
| Learning rate | 0.001 | 0.001 |
| Classes | 4 | 4 |
| Image size | 224 | 224 |

Configuration mismatch flagged: v1/paper use 5 FL rounds and 15 central epochs; v2 changes this to 10 FL rounds and 30 central epochs but has no saved outputs.

### Optimizer and Learning Rate

Optimizer locations:

- Centralized training: both notebooks, cell 15
- FedAvg local training: both notebooks, cell 18, `local_train`
- FedProx: v2 cell 26, `local_train_fedprox`
- Focal loss: v2 cell 30, `local_train_focal`
- DP-SGD: v2 cell 34, `local_train_dp`
- Statistical experiment: v2 cell 32

All use Adam over trainable parameters:

```python
optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=CONFIG['lr'])
```

Learning-rate check:

- `CONFIG['lr']` is `0.001` in both notebooks.
- I found no `0.002` training learning-rate setting.
- Conclusion: learning rate is consistent at `0.001`.

## 3. Federated Learning Implementation

### What Is Sent/Aggregated

The code sends full model states.

Evidence:

- `local_train` returns `model.state_dict()`.
- `local_train_fedprox` returns `model.state_dict()`.
- `local_train_focal` returns `model.state_dict()`.
- DP-SGD extracts and returns `model_dp._module.state_dict()`.
- `fedavg` iterates over every key in those `state_dict()` objects.

Therefore, the implementation aggregates full model weights/state, including frozen-layer tensors and buffers. It does not send:

- Gradients
- Model deltas
- Trainable parameters only

### Exact FedAvg Formula Used

Notebook implementation:

```python
def fedavg(global_model, client_weights, client_sizes):
    total = sum(client_sizes)
    g_state = copy.deepcopy(client_weights[0])
    for key in g_state.keys():
        g_state[key] = torch.zeros_like(g_state[key], dtype=torch.float32)
        for i, w in enumerate(client_weights):
            g_state[key] += (client_sizes[i] / total) * w[key].float()
    global_model.load_state_dict(g_state)
    return global_model
```

Mathematically:

```text
w_global[key] = sum_i (n_i / sum_j n_j) * w_i[key]
```

This is sample-size-weighted FedAvg over every `state_dict` key.

### IID Split Algorithm

Both notebooks:

```python
indices = list(range(len(dataset)))
np.random.shuffle(indices)
splits = np.array_split(indices, num_clients)
```

This is random but not stratified. With seed 42 and the local 5,600-image balanced training set, the client class counts are:

| IID Client | Total | Glioma | Meningioma | No Tumor | Pituitary |
|---|---:|---:|---:|---:|---:|
| Client 1 | 1,867 | 496 | 462 | 433 | 476 |
| Client 2 | 1,867 | 439 | 482 | 476 | 470 |
| Client 3 | 1,866 | 465 | 456 | 491 | 454 |

### Non-IID Split Algorithm

Both notebooks:

- For each class, shuffle all indices for that class.
- `dominant = int(0.7 * len(idx))`
- Send dominant 70 percent to `client_indices[class_id % num_clients]`.
- Split remaining 30 percent across the other two clients.

With 4 classes and 3 clients, client 1 receives the dominant shares for both class 0 and class 3, so the Non-IID clients are not equal-sized.

Client class counts with seed 42:

| Non-IID Client | Total | Glioma | Meningioma | No Tumor | Pituitary |
|---|---:|---:|---:|---:|---:|
| Client 1 | 2,380 | 979 | 211 | 211 | 979 |
| Client 2 | 1,611 | 211 | 979 | 210 | 211 |
| Client 3 | 1,609 | 210 | 210 | 979 | 210 |

### Communication Cost and Frozen Layers

Notebook formula:

```python
trainable_params = sum(p.numel() for p in global_model.parameters() if p.requires_grad)
cost_mb = (trainable_params * CONFIG['num_clients'] * 4 * 2) / (1024**2)
comm_costs.append(cost_mb * rnd)
```

Findings:

- Frozen layers are not included in `trainable_params`.
- The cost uses trainable parameters only: `layer4` plus the new `fc`.
- Trainable parameter count is 14,972,932.
- Per-round bidirectional cost is 342.703 MB.
- v1 cumulative reported costs are 342.7, 685.4, 1028.1, 1370.8, and 1713.5 MB.

Mismatch flagged:

- The aggregation code communicates/aggregates full `state_dict()` objects.
- The communication-cost code counts only trainable parameters.
- If full state is truly communicated, the reported communication cost is an underestimate.
- The paper formula says `P_train * N_clients * 4 bytes`, but the notebook uses `* 2` for upload plus download. The paper's reported values match the notebook's `* 2` behavior, not the printed formula.

## 4. Results Verification

Only `Brain_Tumor_Final_Notebook.ipynb` contains saved outputs. `Brain_Tumor_Final_Notebook_v2.ipynb` has no saved execution outputs, so its 10-round/30-epoch configuration cannot be verified from the workspace.

### Where Metrics Are Generated

Shared functions:

- Accuracy and predictions: cell 13, `evaluate`
- Precision/recall/F1 text: cell 13, `print_report`, using `classification_report`
- Confusion matrix: cell 13, `plot_confusion`, using `confusion_matrix`

Experiment outputs:

- Centralized report and confusion matrix: v1 cell 16
- FL IID report and confusion matrix: v1 cell 21
- FL Non-IID report and confusion matrix: v1 cell 24
- Final F1 comparison table: v1 cell 26, using `precision_recall_fscore_support`
- Accuracy comparison plot: v1 cell 27
- Communication cost plot: v1 cell 28

v2 has corresponding cells plus FedProx, focal loss, statistical testing, and DP-SGD reporting cells, but no saved outputs.

### Accuracy Comparison

| Metric | Executed notebook output | Paper reported value | Status |
|---|---:|---:|---|
| Centralized best accuracy | 95.31% | 95.31% | Match |
| Centralized final accuracy | 95.12% | 95.12% | Match |
| FL IID best accuracy | 95.19% | 95.20% | Rounding match |
| FL IID final accuracy | 94.81% | 94.80% | Rounding match |
| FL Non-IID best/final accuracy | 94.12% | 94.10% | Rounding match |

Internal notebook caveat:

- `fl_train` loads the best FL state before returning the model.
- Therefore the later FL IID classification report evaluates the best-round model at 95.19%, while the training log's final round accuracy is 94.81%.
- The paper's Table 1 correctly distinguishes best and final accuracy, but the notebook's final comparison table uses the best returned model for FL IID.

### Confusion Matrices

Notebook central confusion matrix:

| True \ Pred | Glioma | Meningioma | No Tumor | Pituitary | Row total |
|---|---:|---:|---:|---:|---:|
| Glioma | 335 | 47 | 17 | 1 | 400 |
| Meningioma | 1 | 392 | 2 | 5 | 400 |
| No Tumor | 1 | 1 | 398 | 0 | 400 |
| Pituitary | 0 | 2 | 1 | 397 | 400 |

Notebook FL IID confusion matrix:

| True \ Pred | Glioma | Meningioma | No Tumor | Pituitary | Row total |
|---|---:|---:|---:|---:|---:|
| Glioma | 345 | 15 | 39 | 1 | 400 |
| Meningioma | 4 | 381 | 12 | 3 | 400 |
| No Tumor | 0 | 0 | 400 | 0 | 400 |
| Pituitary | 1 | 2 | 0 | 397 | 400 |

Notebook FL Non-IID confusion matrix:

| True \ Pred | Glioma | Meningioma | No Tumor | Pituitary | Row total |
|---|---:|---:|---:|---:|---:|
| Glioma | 332 | 31 | 32 | 5 | 400 |
| Meningioma | 2 | 377 | 9 | 12 | 400 |
| No Tumor | 1 | 0 | 399 | 0 | 400 |
| Pituitary | 0 | 2 | 0 | 398 | 400 |

Confusion-matrix mismatch flags:

- Paper dataset/evaluation text says the test dataset has 1,405 images, but all notebook confusion matrices use 1,600 test images.
- Paper confusion prose also uses 400 samples per class, which totals 1,600, contradicting the paper's 1,405 test-set claim.
- Centralized diagonal counts match the notebook: 335, 392, 398, 397.
- FL IID diagonal counts match the notebook: 345, 381, 400, 397.
- FL Non-IID paper prose reports 318, 376, 396, 395, but the notebook matrix is 332, 377, 399, 398. This is a clear mismatch.

### Class-Wise F1-Score Comparison

Notebook F1 values below are computed from the saved confusion matrices.

| Method | Class | Notebook F1 | Paper Table 2 F1 | Status |
|---|---|---:|---:|---|
| Centralized | Glioma | 0.909 | 0.89 | Mismatch |
| Centralized | Meningioma | 0.931 | 0.97 | Mismatch |
| Centralized | No Tumor | 0.973 | 1.00 | Mismatch |
| Centralized | Pituitary | 0.989 | 0.99 | Match by rounding |
| FL IID | Glioma | 0.920 | 0.91 | Small mismatch |
| FL IID | Meningioma | 0.955 | 0.95 | Match by rounding/truncation |
| FL IID | No Tumor | 0.940 | 1.00 | Mismatch |
| FL IID | Pituitary | 0.991 | 0.99 | Match by rounding |
| FL Non-IID | Glioma | 0.903 | 0.88 | Mismatch |
| FL Non-IID | Meningioma | 0.931 | 0.93 | Match by rounding |
| FL Non-IID | No Tumor | 0.950 | 0.99 | Mismatch |
| FL Non-IID | Pituitary | 0.977 | 0.99 | Mismatch |

Major F1 issue:

- The paper says No Tumor has perfect F1 in Centralized and FL IID.
- The notebook supports perfect No Tumor recall for FL IID, but not perfect F1. FL IID No Tumor precision is reduced by false positives from Glioma and Meningioma, giving F1 = 0.940.

### Communication Cost Comparison

| Item | Notebook | Paper | Status |
|---|---:|---:|---|
| Round 1 cumulative cost | 342.7 MB | about 345 MB | Approximate match |
| Round 5 cumulative cost | 1713.5 MB | about 1,700 MB | Approximate match |
| Formula | `P_train * clients * 4 * 2` | `P_train * clients * 4` | Formula mismatch |
| Payload implied by FedAvg code | Full `state_dict()` | "model weights" | Cost model mismatch |

## 5. Reproducibility

### Fixed Seeds

Both notebooks define:

```python
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

Also:

- `g = torch.Generator(); g.manual_seed(42)`
- `seed_worker` seeds NumPy and Python random inside workers using `torch.initial_seed()`
- Main experiments call `set_seed(42)` before centralized, IID, and Non-IID runs.
- v2 statistical experiment defines `SEEDS = [42, 123, 256, 789, 999]`.

Reproducibility gaps:

- No split CSV files are saved.
- No prediction CSV files are saved.
- No numeric confusion-matrix CSV/JSON files are saved.
- v1 saves no metrics CSV.
- v2 contains CSV-saving code for `statistical_results.csv` and `full_comparison_results.csv`, but those cells have no outputs and no files exist locally.
- Checkpoints and plots are saved to `/content/...` inside Colab, not to the workspace and not consistently to `SAVE_DIR`.
- `SAVE_DIR` is created but not actually used for the main saved artifacts.
- `partition_noniid` creates DataLoaders without `generator=g` and without `worker_init_fn=seed_worker`.
- The generator `g` is created once; repeated partial reruns can depend on prior generator state unless it is reset before each DataLoader.
- The `labels = np.array([dataset[i][1] for i in range(len(dataset))])` step in `partition_noniid` loads every image and applies the training transform just to read labels. That is inefficient and can consume transform RNG state.

### Saved Artifacts Found Locally

No local experiment artifacts were found outside the notebooks, PDF, and dataset:

- No `.csv`
- No `.pth` or `.pt`
- No generated `.png`
- No prediction files
- No split manifests

The notebooks contain commands to save artifacts to `/content`, including:

- `centralized_best_model.pth`
- `centralized_model.pth`
- `fl_iid_model.pth`
- `fl_noniid_model.pth`
- `central_accuracy.png`
- `central_confusion_matrix.png`
- `fl_iid_accuracy.png`
- `fl_iid_confusion_matrix.png`
- `fl_noniid_accuracy.png`
- `fl_noniid_confusion_matrix.png`
- `comparison_accuracy.png`
- `communication_cost.png`
- `gradcam_visualization.png`

v2 additionally saves FedProx, focal-loss, DP-SGD, and CSV artifacts to `/content`, if those cells are run.

### Recommended Files to Save Before Rerunning

Before rerunning experiments, save these files under a timestamped results directory:

1. `config.json`
   - Notebook name/version
   - Dataset path
   - Class order
   - Seed(s)
   - Number of clients
   - Rounds
   - Local epochs
   - Central epochs
   - Batch size
   - Learning rate
   - Transform definitions
   - Freeze policy
   - Communication-cost formula

2. `dataset_manifest.csv`
   - File path
   - Class label
   - Folder split (`Training` or `Testing`)
   - File size
   - Optional hash

3. `dataset_counts.csv`
   - Class-wise counts for train/test
   - Total counts

4. `client_splits_iid_seed42.csv`
   - File path
   - Label
   - Client ID
   - Split type
   - Seed

5. `client_splits_noniid_seed42.csv`
   - Same fields as IID
   - Include dominant-class rule metadata

6. `metrics_per_epoch_central.csv`
   - Epoch
   - Train loss
   - Train accuracy
   - Test accuracy

7. `metrics_per_round_fl_iid.csv` and `metrics_per_round_fl_noniid.csv`
   - Round
   - Client train loss/accuracy
   - Global test accuracy
   - Best-so-far accuracy
   - Communication cost

8. `predictions_<method>.csv`
   - File path
   - True label
   - Predicted label
   - Class probabilities or logits

9. `classification_report_<method>.json`
   - Use `classification_report(..., output_dict=True)`

10. `confusion_matrix_<method>.csv`
    - Save the numeric matrix, not only the heatmap image.

11. Model checkpoints
    - `central_best.pth`
    - `central_final.pth`
    - `fl_iid_best.pth`
    - `fl_iid_final.pth`
    - `fl_noniid_best.pth`
    - `fl_noniid_final.pth`
    - Include optimizer state for centralized training if exact resumption is needed.

12. `environment.txt` or `requirements-freeze.txt`
    - Python version
    - PyTorch/torchvision versions
    - CUDA/cuDNN versions
    - scikit-learn, NumPy, pandas, matplotlib, seaborn versions

13. `paper_tables_generated.md` or `paper_tables_generated.csv`
    - Generate Table 1 and Table 2 directly from saved metrics to prevent manual transcription errors.

## High-Priority Fixes Before Any New Training

1. Decide which experiment budget is authoritative: v1/paper 5 rounds and 15 central epochs, or v2 10 rounds and 30 central epochs.
2. Decide which dataset is authoritative: local balanced 7,200 images or the paper-described 7,023-image Nickparvar distribution.
3. Save deterministic split manifests before training.
4. Align communication accounting with actual payload:
   - Either aggregate/send trainable parameters only, or calculate cost for full `state_dict()`.
5. Save numeric predictions, confusion matrices, and classification reports.
6. Regenerate paper tables from saved metrics and update the paper's dataset, F1, Non-IID matrix, and communication-formula text.
