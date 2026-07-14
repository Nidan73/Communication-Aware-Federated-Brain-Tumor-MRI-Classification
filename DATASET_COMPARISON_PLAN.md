# Dataset Comparison Plan

Prepared: 2026-06-28  
Scope: compare candidate datasets for the existing four-class ResNet50 federated-learning pipeline. No notebooks were modified and no model training was run.

## Recommendation

Recommended path: **B. use current + Mendeley external validation**.

Rationale:

- The current local dataset already matches the **current Kaggle Nickparvar Version 2** metadata: 7,200 images, balanced 1,400 train and 400 test images per class.
- Replacing the current dataset is unnecessary and would destabilize the current paper results.
- Mendeley DOI `10.17632/zwr4ntf94j.6` is a larger four-class classification dataset with a compatible train/test folder layout and CC BY 4.0 license. It is the best candidate for external validation if duplicate overlap is checked first.
- BraTS datasets are valuable future-work candidates, but they are segmentation-oriented 3D NIfTI datasets, not drop-in four-class 2D classification datasets.

Final decision:

| Option | Recommendation | Reason |
|---|---|---|
| A. Keep current dataset only | Acceptable fallback | Lowest effort, but weaker external validity. |
| B. Current + Mendeley external validation | Best choice | Strengthens paper while preserving current results. |
| C. Replace current dataset | Not recommended | High risk of changing results close to deadline. |
| D. Switch to BraTS | Not recommended for current paper | Requires a different task, data loader, model design, and metrics. |

## Source Summary

- Current local dataset: inspected from `Dataset/Nickparvar`.
- Kaggle current metadata: public Kaggle API/page for `masoudnickparvar/brain-tumor-mri-dataset`; metadata fetched 2026-06-28 without downloading data.
- Mendeley: official Mendeley Data page for DOI `10.17632/zwr4ntf94j.6`.
- BraTS: official Synapse challenge pages and BraTS challenge papers used only to assess future-work fit.

## 1. Current Local `Dataset/Nickparvar`

| Field | Assessment |
|---|---|
| Total images/cases | 7,200 image files |
| Classes | `glioma`, `meningioma`, `notumor`, `pituitary` |
| Image format | `.jpg` |
| Train/test split | Folder split: 5,600 train, 1,600 test |
| Class counts | Train: 1,400/class; Test: 400/class |
| License/citation | No local license file found. Treat as Kaggle Nickparvar-derived and cite Kaggle current dataset unless provenance says otherwise. |
| Fits four-class classification? | Yes, exactly. |
| Fits existing ResNet50 pipeline? | Yes, already used by the notebooks. |
| Expected code changes | None for current experiments. For reproducibility, save manifests/splits/metrics before reruns. |
| Duplicate overlap risk | Baseline dataset; overlap question applies when adding external datasets. |
| COMPAS suitability | High. Lowest execution risk. |

Key note: the local dataset is balanced at 7,200 images, not the 7,023-image split described in the paper text. However, it appears to match the current Kaggle Version 2 metadata.

## 2. Kaggle Nickparvar Brain Tumor MRI Dataset

Dataset: `masoudnickparvar/brain-tumor-mri-dataset`  
Title: `Brain Tumor MRI Dataset`  
Creator: Masoud Nickparvar  
Current version in fetched metadata: Version 2  
Last updated in fetched metadata: 2026-02-13  
License in fetched metadata: CC BY 4.0

| Field | Assessment |
|---|---|
| Total images/cases | 7,200 MRI images in current metadata |
| Classes | Glioma, Meningioma, Pituitary tumor, No tumor |
| Image format | MRI image files; metadata notes variable image sizes. Local mirror is `.jpg`. |
| Train/test split | `Training/` and `Testing/` folders |
| Current split counts | Training: 1,400/class; Testing: 400/class |
| License/citation | CC BY 4.0. Cite Kaggle dataset page and creator. |
| Fits four-class classification? | Yes, exactly. |
| Fits existing ResNet50 pipeline? | Yes. Folder names and class semantics match current pipeline. |
| Expected code changes | None if using the current local copy. If redownloading, verify class folder names and class order before training. |
| Duplicate overlap with current dataset | Very high, likely exact or near-exact overlap. The local dataset counts and tree match current Kaggle metadata exactly. |
| COMPAS suitability | High as the primary dataset, but not useful as an independent external validation source because it is likely the same dataset. |

Important distinction:

- Older/paper descriptions cite 7,023 images.
- Current Kaggle metadata reports Version 2 with 7,200 images and balanced classes.
- Therefore, the paper should be updated to cite the current Kaggle Version 2 counts if the local dataset remains unchanged.

## 3. Mendeley Brain Tumor MRI Dataset

Dataset: `Brain Tumor MRI Dataset (Glioma, Meningioma, Pituitary, No Tumor)`  
DOI: `10.17632/zwr4ntf94j.6`  
Published: 2026-04-17  
Version: 6  
License: CC BY 4.0

| Field | Assessment |
|---|---|
| Total images/cases | Official description says 11,148 images. Reproducibility text also says 11,145, but the listed class counts sum to 11,148. Use 11,148 and flag the source inconsistency. |
| Classes | Glioma, Meningioma, Pituitary, No Tumor |
| Image format | JPEG/PNG |
| Modality | T1-weighted contrast-enhanced MRI |
| Train/test split | 80/20 folder split: `train/` and `test/` |
| Train counts | Glioma 3,018; Pituitary 2,145; Meningioma 2,183; No Tumor 1,945; total 9,291 |
| Test counts | Glioma 603; Pituitary 429; Meningioma 436; No Tumor 389; total 1,857 |
| License/citation | CC BY 4.0. Cite DOI and listed contributors. |
| Fits four-class classification? | Yes, exactly. |
| Fits existing ResNet50 pipeline? | Mostly yes. `ImageFolder` can load it, but paths and folder names need a small adapter. |
| Expected code changes | Add external dataset path; create `external_test_loader`; normalize class names; confirm `class_to_idx`; allow JPEG/PNG; save separate Mendeley metrics. |
| Duplicate overlap with current dataset | Medium to high until checked. It has the same four public classes and may share images with Kaggle/Figshare/SARTAJ/Br35H-derived datasets. Must run hash and perceptual-hash duplicate checks before claiming independent validation. |
| COMPAS suitability | High for external validation if duplicate checking is done first. Medium if trying to retrain/rebuild the whole paper around it. |

Suggested use:

- Do not replace the current dataset with Mendeley.
- Use the Mendeley **test split only** as an external validation set after training the current protocol.
- If time permits, report a second experiment: train on current Kaggle/local train split and evaluate on current test plus Mendeley test.

Minimum code changes later, after approval:

```text
EXTERNAL_DATASET_PATH = ".../Brain_Tumor_MRI_Dataset"
external_test_dataset = ImageFolder(EXTERNAL_DATASET_PATH + "/test", transform=test_transform)
external_test_loader = DataLoader(external_test_dataset, batch_size=32, shuffle=False, ...)
```

Add a class-name normalization table:

| Mendeley folder | Existing class name |
|---|---|
| `Glioma` | `glioma` |
| `Meningioma` | `meningioma` |
| `No_Tumor` | `notumor` |
| `Pituitary` | `pituitary` |

## 4. BraTS Datasets as Future-Work Candidates

BraTS is not one dataset; it is a family of challenge datasets, mostly for tumor segmentation. Candidate examples include adult glioma, post-treatment glioma, meningioma, metastasis, pediatric tumor, and related tasks.

| Field | Assessment |
|---|---|
| Total images/cases | Varies by BraTS task. Example: BraTS 2024 post-treatment glioma paper reports about 2,200 cases contributed across seven sites. BraTS 2023 meningioma paper reports 1,000 training, 141 validation, and 283 hidden test cases. |
| Classes | Not four diagnosis classes. Usually tumor subregions or task-specific compartments. Example BraTS 2024 post-treatment glioma predicts ET, SNFH, NETC, and RC subregions. |
| Image format | 3D MRI volumes, commonly NIfTI `.nii.gz`, converted from DICOM and preprocessed. |
| Modalities | Usually multi-sequence MRI: T1, contrast-enhanced T1/T1-Gd/T1ce, T2, FLAIR. |
| Train/test split | Challenge-specific train/validation/test partitions, often controlled through Synapse or challenge infrastructure. |
| License/citation | Requires checking the specific BraTS task data-use terms on Synapse/official challenge pages. Do not assume CC BY 4.0. |
| Fits four-class classification? | No, not directly. BraTS is segmentation- and subtype/task-oriented, not the current four-class 2D classification setup. |
| Fits existing ResNet50 pipeline? | No. It would require a new medical-imaging loader and likely a new task definition. |
| Expected code changes | Major: NIfTI loading, 3D volume or slice extraction, modality selection/fusion, segmentation labels, new metrics, new model or large preprocessing pipeline. |
| Duplicate overlap with current dataset | Low direct image-duplicate risk, but also low task compatibility. |
| COMPAS suitability | Low for the current deadline. Better positioned as future work. |

Recommended framing:

- Mention BraTS as future work for moving from 2D image-level diagnosis to 3D multi-modal segmentation or federated segmentation.
- Do not switch the current paper to BraTS unless the project scope changes completely.

## Duplicate and Leakage Audit Plan

Before using Mendeley as external validation, run a duplicate audit.

Required checks:

1. Exact file hash:
   - Compute SHA-256 for every local and Mendeley image.
   - Flag exact duplicate files.

2. Pixel hash after canonical preprocessing:
   - Load with PIL.
   - Convert RGB.
   - Resize to a canonical size, e.g. 224 x 224.
   - Hash raw pixel bytes.
   - Flags duplicate images saved with different file compression.

3. Perceptual hash:
   - Use pHash or dHash.
   - Flag near-duplicates under a conservative Hamming threshold.
   - Manually inspect flagged pairs.

4. Split leakage report:
   - Current train vs current test
   - Current train vs Mendeley test
   - Current test vs Mendeley test
   - Mendeley train vs Mendeley test

5. Save all duplicate evidence:
   - `duplicate_exact_hashes.csv`
   - `duplicate_pixel_hashes.csv`
   - `duplicate_phash_candidates.csv`
   - `duplicate_audit_summary.md`

Decision rule:

- If Mendeley test has substantial overlap with current train/test, do not claim it as independent external validation.
- If overlap is small, remove duplicate/near-duplicate Mendeley samples before reporting external validation.

## Pipeline Fit Matrix

| Dataset | Four-class task | Drop-in with `ImageFolder` | Compatible with current transforms | Needs retraining? | Best use |
|---|---|---|---|---|---|
| Current local | Yes | Yes | Yes | No change to design | Primary experiments |
| Kaggle current | Yes | Yes | Yes | No if local copy is same | Citation/provenance baseline |
| Mendeley v6 | Yes | Mostly | Yes | Not for external validation; yes if used as replacement | External validation |
| BraTS | No | No | No | Yes, complete redesign | Future work |

## Expected Reporting Changes for the Paper

If choosing option B:

1. Dataset section:
   - State that the primary dataset is Kaggle Nickparvar Version 2 with 7,200 images.
   - State balanced train/test counts.
   - Add Mendeley v6 as an external validation dataset, not as training data.

2. Methods:
   - Add an external validation protocol:
     - train on primary training split only;
     - tune nothing on Mendeley;
     - evaluate once on Mendeley test split after model selection.

3. Results:
   - Keep current internal test metrics.
   - Add a separate table for external validation metrics.
   - Do not mix Mendeley images into FL client partitions unless explicitly running a new experiment.

4. Limitations:
   - Note potential public-dataset overlap risk and describe duplicate filtering.
   - Note that BraTS is future work for 3D segmentation and multi-institutional FL.

## Deadline-Oriented Action Plan

No-training preparation steps:

1. Freeze dataset provenance:
   - Save Kaggle metadata JSON.
   - Save Mendeley citation metadata.
   - Save local dataset manifest and class counts.

2. Prepare duplicate-audit scripts:
   - Exact SHA-256.
   - Pixel hash.
   - Perceptual hash.

3. Prepare Mendeley loader plan:
   - Add a small class-name mapping.
   - Use `test_transform`.
   - Save external predictions separately.

4. Prepare result tables:
   - Internal test metrics table.
   - External validation metrics table.
   - Duplicate audit summary table.

Training/evaluation steps only after approval:

1. Rerun or load approved current-model checkpoints.
2. Evaluate on current test set.
3. Evaluate the same trained models on cleaned Mendeley test set.
4. Save predictions, metrics JSON/CSV, confusion matrices, and class-wise reports.

## Final Recommendation

Choose **B. current + Mendeley external validation**.

Do not use Kaggle current as a second dataset because it likely duplicates the local dataset. Do not replace the current dataset because it would invalidate existing results close to the COMPAS deadline. Do not switch to BraTS unless the paper changes from four-class image classification to 3D segmentation.

The safest paper claim is:

> We use the current Kaggle Nickparvar Version 2 dataset as the primary benchmark and evaluate generalization on the independent Mendeley v6 four-class MRI dataset after duplicate-overlap screening. BraTS-style 3D segmentation is reserved for future work.

## Source Links

- Kaggle Nickparvar: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
- Kaggle public metadata endpoint: https://www.kaggle.com/api/v1/datasets/view/masoudnickparvar/brain-tumor-mri-dataset
- Mendeley DOI: https://doi.org/10.17632/zwr4ntf94j.6
- Mendeley dataset page: https://data.mendeley.com/datasets/zwr4ntf94j/6
- BraTS 2023 Synapse page: https://www.synapse.org/Synapse:syn51156910/wiki/622351
- BraTS 2024 post-treatment glioma paper: https://arxiv.org/abs/2405.18368
- BraTS 2023 meningioma paper: https://arxiv.org/abs/2405.09787
