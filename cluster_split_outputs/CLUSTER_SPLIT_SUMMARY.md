# Cluster-Aware Split Summary

Generated: 2026-06-28

No model training was run. No notebooks were edited. No image files were moved, deleted, or modified.

## Dataset

- Input dataset: `Dataset/Nickparvar`
- Images used: all files under original `Training` and `Testing` folders
- Total images used: 7200

## Duplicate Graph

- Total clusters: 6116
- Duplicate clusters with size > 1: 596
- Largest cluster size: 14
- Exact duplicate edge pairs: 227
- Pixel duplicate edge pairs: 385
- Conservative near-duplicate edge pairs: 2146

## Final Split Counts by Class

| Final split | glioma | meningioma | notumor | pituitary | Total |
|---|---:|---:|---:|---:|---:|
| train | 1260 | 1260 | 1260 | 1260 | 5040 |
| val | 180 | 180 | 180 | 180 | 720 |
| test | 360 | 360 | 360 | 360 | 1440 |

## Leakage Verification

| Comparison | Exact | Pixel | Near duplicate | Total |
|---|---:|---:|---:|---:|
| `train_vs_val` | 0 | 0 | 0 | 0 |
| `train_vs_test` | 0 | 0 | 0 | 0 |
| `val_vs_test` | 0 | 0 | 0 | 0 |

- Cross-split leakage remaining: none found under the exact, pixel, and conservative near-duplicate graph criteria.
- Suitability for publication experiments: yes, this split is suitable as a leakage-controlled split, with the caveat that class balance is cluster-constrained rather than perfectly stratified.

## Split Method

- Nodes are images; edges connect exact SHA-256 duplicates, pixel-hash duplicates after RGB resize to 224 x 224, and same-class dHash/SSIM near-duplicates.
- Near-duplicate edges use grayscale 224 x 224 SSIM. The conservative condition is same class, dHash distance <= 5, and SSIM >= 0.92; this includes the stricter dHash <= 3 and SSIM >= 0.97 condition.
- Connected components define `cluster_id`; clusters, not individual images, were assigned to final splits.
- Target fractions were 70% train, 10% validation, and 20% test. Fixed seed: 42.

## Generated Files

- `full_current_manifest.csv`
- `image_clusters.csv`
- `cluster_summary.csv`
- `train_manifest.csv`
- `val_manifest.csv`
- `test_manifest.csv`
- `split_counts.csv`
- `split_leakage_check.csv`
- `CLUSTER_SPLIT_SUMMARY.md`
