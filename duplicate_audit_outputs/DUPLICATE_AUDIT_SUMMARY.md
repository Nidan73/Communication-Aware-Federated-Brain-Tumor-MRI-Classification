# Duplicate and Leakage Audit Summary

Generated: 2026-06-28

No model training was run. No dataset files were deleted, moved, or modified.

## Inputs

- Current dataset: `Dataset/Nickparvar`
- Mendeley v6 dataset: `datasets/Mendeley_v6/Epic and CSCR hospital Dataset`
- Mendeley split folders treated as `Train` and `Test`.
- `Dataset Splition.pdf` and other non-image files were ignored.

### Current Dataset Manifest Counts

| Split | glioma | meningioma | notumor | pituitary | Total |
|---|---:|---:|---:|---:|---:|
| Testing | 400 | 400 | 400 | 400 | 1600 |
| Training | 1400 | 1400 | 1400 | 1400 | 5600 |

### Mendeley Dataset Manifest Counts

| Split | glioma | meningioma | notumor | pituitary | Total |
|---|---:|---:|---:|---:|---:|
| Test | 603 | 436 | 389 | 429 | 1857 |
| Train | 3018 | 2183 | 1945 | 2145 | 9291 |

## Duplicate Counts

| Comparison | Exact SHA-256 | Pixel hash 224 RGB | dHash <= 5 candidates | dHash comparisons |
|---|---:|---:|---:|---:|
| `current_train_vs_current_test` | 0 | 121 | 5775 | 8960000 |
| `current_train_vs_mendeley_test` | 46 | 47 | 6831 | 10399200 |
| `current_test_vs_mendeley_test` | 7 | 7 | 2075 | 2971200 |
| `mendeley_train_vs_mendeley_test` | 0 | 0 | 7522 | 17253387 |

## Required Conclusions

1. Whether current train/test is clean:
   - Not clean under the configured checks: exact=0, pixel=121, dHash<=5 candidates=5775. Review the generated CSVs before claiming no leakage.

2. Whether Mendeley test can be used as independent external validation:
   - Not as-is. Current train vs Mendeley Test overlaps/candidates: exact=46, pixel=47, dHash<=5=6831. Current test vs Mendeley Test overlaps/candidates: exact=7, pixel=7, dHash<=5=2075.

3. How many exact, pixel-level, and near-duplicate overlaps exist:
   - See the Duplicate Counts table above and the generated per-comparison CSVs.

4. Whether duplicate Mendeley samples should be removed before external validation:
   - Yes. Remove all exact and pixel-level duplicates. Manually review dHash <= 5 candidates and remove confirmed near-duplicates before external validation.

5. Whether the paper can claim independent external validation:
   - Not until duplicate/near-duplicate Mendeley samples are removed or explicitly excluded and the cleaned-set audit is rerun.

6. Class-name mapping problems between datasets:
- None found. Both datasets use normalized class folders: `glioma`, `meningioma`, `notumor`, `pituitary`.

## Generated Files

- `current_dataset_manifest.csv`
- `mendeley_dataset_manifest.csv`
- `current_train_vs_current_test_exact_duplicates.csv`
- `current_train_vs_current_test_pixel_duplicates.csv`
- `current_train_vs_current_test_phash_candidates.csv`
- `current_train_vs_mendeley_test_exact_duplicates.csv`
- `current_train_vs_mendeley_test_pixel_duplicates.csv`
- `current_train_vs_mendeley_test_phash_candidates.csv`
- `current_test_vs_mendeley_test_exact_duplicates.csv`
- `current_test_vs_mendeley_test_pixel_duplicates.csv`
- `current_test_vs_mendeley_test_phash_candidates.csv`
- `mendeley_train_vs_mendeley_test_exact_duplicates.csv`
- `mendeley_train_vs_mendeley_test_pixel_duplicates.csv`
- `mendeley_train_vs_mendeley_test_phash_candidates.csv`

## Notes

- Exact duplicates use file SHA-256.
- Pixel duplicates use SHA-256 of RGB pixels after EXIF transpose and resize to 224 x 224.
- Near-duplicate candidates use 64-bit dHash after EXIF transpose, grayscale conversion, and resize to 9 x 8; suspicious threshold is Hamming distance <= 5.
- dHash candidates are conservative signals for manual review, not automatic proof of leakage.
