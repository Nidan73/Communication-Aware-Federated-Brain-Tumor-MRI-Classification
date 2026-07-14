#!/usr/bin/env python3
"""Paired fixed-test-set analysis for EfficientNet-B0 versus ResNet50."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score

from reproducibility_common import (
    CLASSES,
    EXPECTED_TEST_SAMPLES,
    ROOT,
    SEEDS,
    read_json,
    resolve_project_path,
    run_id,
)


DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 104_729
EFFICIENTNET = "efficientnet_b0"
RESNET = "resnet50"


def prediction_path(results_root: Path, model_name: str, seed: int) -> Path:
    result_path = results_root / run_id(model_name, seed) / "result.json"
    result = read_json(result_path)
    path_value = result.get("artifacts", {}).get("predictions")
    if not path_value:
        raise ValueError(f"Prediction artifact missing from {result_path}")
    path = Path(path_value)
    return path if path.is_absolute() else resolve_project_path(path)


def load_predictions(path: Path, model_name: str, seed: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "sample_id",
        "image_path",
        "true_label",
        "predicted_label",
        "correct",
        "model",
        "seed",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    if set(frame["model"].astype(str)) != {model_name}:
        raise ValueError(f"{path} model column does not equal {model_name}")
    if set(pd.to_numeric(frame["seed"], errors="raise").astype(int)) != {seed}:
        raise ValueError(f"{path} seed column does not equal {seed}")
    return frame


def align_predictions(
    efficientnet: pd.DataFrame,
    resnet: pd.DataFrame,
    *,
    expected_samples: int = EXPECTED_TEST_SAMPLES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    for name, frame in (("EfficientNet-B0", efficientnet), ("ResNet50", resnet)):
        if len(frame) != expected_samples:
            raise ValueError(
                f"Prediction alignment failed: {name} has {len(frame)} rows; "
                f"expected exactly {expected_samples}."
            )
        if frame["sample_id"].duplicated().any():
            raise ValueError(f"Prediction alignment failed: {name} has duplicate sample IDs.")
    efficientnet = efficientnet.sort_values("sample_id").set_index("sample_id", drop=False)
    resnet = resnet.sort_values("sample_id").set_index("sample_id", drop=False)
    if set(efficientnet.index) != set(resnet.index):
        only_eff = set(efficientnet.index) - set(resnet.index)
        only_res = set(resnet.index) - set(efficientnet.index)
        raise ValueError(
            "Prediction alignment failed: sample ID sets differ; "
            f"only EfficientNet-B0={len(only_eff)}, only ResNet50={len(only_res)}."
        )
    resnet = resnet.loc[efficientnet.index]
    if not efficientnet["true_label"].equals(resnet["true_label"]):
        raise ValueError("Prediction alignment failed: true labels differ for matched sample IDs.")
    if not efficientnet["image_path"].equals(resnet["image_path"]):
        raise ValueError("Prediction alignment failed: image paths differ for matched sample IDs.")
    for name, frame in (("EfficientNet-B0", efficientnet), ("ResNet50", resnet)):
        calculated = (frame["true_label"] == frame["predicted_label"]).astype(int)
        observed = pd.to_numeric(frame["correct"], errors="raise").astype(int)
        if not calculated.equals(observed):
            raise ValueError(f"Prediction alignment failed: {name} correct column is inconsistent.")
    return efficientnet.reset_index(drop=True), resnet.reset_index(drop=True)


def exact_mcnemar_counts(
    efficientnet: pd.DataFrame, resnet: pd.DataFrame
) -> dict[str, int | float]:
    eff_correct = efficientnet["correct"].to_numpy(dtype=bool)
    res_correct = resnet["correct"].to_numpy(dtype=bool)
    both_correct = int(np.sum(eff_correct & res_correct))
    both_wrong = int(np.sum(~eff_correct & ~res_correct))
    only_eff = int(np.sum(eff_correct & ~res_correct))
    only_res = int(np.sum(~eff_correct & res_correct))
    discordant = only_eff + only_res
    p_value = 1.0 if discordant == 0 else float(
        binomtest(only_eff, n=discordant, p=0.5, alternative="two-sided").pvalue
    )
    return {
        "both_models_correct": both_correct,
        "both_models_wrong": both_wrong,
        "only_efficientnet_correct": only_eff,
        "only_resnet_correct": only_res,
        "discordant_total": discordant,
        "mcnemar_exact_two_sided_p": p_value,
    }


def holm_adjust_pvalues(p_values: list[float]) -> list[float]:
    """Return Holm step-down adjusted p-values in original input order."""
    if not p_values:
        return []
    values = np.asarray(p_values, dtype=float)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Holm adjustment requires finite p-values in [0, 1].")
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    multipliers = len(values) - np.arange(len(values))
    adjusted_sorted = np.minimum(1.0, sorted_values * multipliers)
    adjusted_sorted = np.maximum.accumulate(adjusted_sorted)
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted.tolist()


def apply_holm_adjustment(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    adjusted_values = holm_adjust_pvalues(
        [float(row["mcnemar_exact_two_sided_p"]) for row in seed_rows]
    )
    adjusted_rows: list[dict[str, Any]] = []
    for row, adjusted_p in zip(seed_rows, adjusted_values):
        updated = dict(row)
        updated["mcnemar_holm_adjusted_p"] = adjusted_p
        updated["mcnemar_holm_interpretation"] = (
            "statistically distinguishable after Holm adjustment on the fixed test set"
            if adjusted_p < 0.05
            else "not statistically distinguishable after Holm adjustment on the fixed test set"
        )
        adjusted_rows.append(updated)
    return adjusted_rows


def confusion_matrices_from_codes(codes: np.ndarray, indices: np.ndarray) -> np.ndarray:
    rows = indices.shape[0]
    offsets = np.arange(rows, dtype=np.int64)[:, None] * (len(CLASSES) ** 2)
    encoded = codes[indices] + offsets
    counts = np.bincount(
        encoded.ravel(),
        minlength=rows * (len(CLASSES) ** 2),
    )
    return counts.reshape(rows, len(CLASSES), len(CLASSES))


def macro_f1_from_confusions(confusions: np.ndarray) -> np.ndarray:
    true_positives = np.diagonal(confusions, axis1=1, axis2=2).astype(float)
    false_positives = confusions.sum(axis=1) - true_positives
    false_negatives = confusions.sum(axis=2) - true_positives
    denominator = 2 * true_positives + false_positives + false_negatives
    class_f1 = np.divide(
        2 * true_positives,
        denominator,
        out=np.zeros_like(true_positives),
        where=denominator != 0,
    )
    return class_f1.mean(axis=1)


def draw_paired_bootstrap_indices(
    true_labels: np.ndarray,
    *,
    resamples: int,
    rng: np.random.Generator,
    stratified: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw one shared index matrix for both models."""
    if true_labels.ndim != 1 or len(true_labels) == 0:
        raise ValueError("Bootstrap true labels must be a non-empty one-dimensional array.")
    if resamples <= 0:
        raise ValueError("Bootstrap resamples must be positive.")
    if stratified:
        class_blocks: list[np.ndarray] = []
        for class_index in range(len(CLASSES)):
            class_positions = np.flatnonzero(true_labels == class_index)
            if len(class_positions) == 0:
                raise ValueError(
                    f"Stratified bootstrap has no observations for class {CLASSES[class_index]}."
                )
            draws = rng.integers(
                0,
                len(class_positions),
                size=(resamples, len(class_positions)),
                dtype=np.int64,
            )
            class_blocks.append(class_positions[draws])
        sampled_indices = np.concatenate(class_blocks, axis=1)
    else:
        sampled_indices = rng.integers(
            0,
            len(true_labels),
            size=(resamples, len(true_labels)),
            dtype=np.int64,
        )
    return sampled_indices, sampled_indices


def paired_bootstrap(
    efficientnet: pd.DataFrame,
    resnet: pd.DataFrame,
    *,
    resamples: int,
    bootstrap_seed: int,
    chunk_size: int = 250,
    stratified: bool = True,
) -> dict[str, float | int | bool]:
    if resamples <= 0:
        raise ValueError("Bootstrap resamples must be positive.")
    label_to_index = {label: index for index, label in enumerate(CLASSES)}
    try:
        true = efficientnet["true_label"].map(label_to_index).to_numpy(dtype=np.int64)
        eff_pred = efficientnet["predicted_label"].map(label_to_index).to_numpy(dtype=np.int64)
        res_pred = resnet["predicted_label"].map(label_to_index).to_numpy(dtype=np.int64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Bootstrap inputs contain unexpected class labels.") from exc
    if np.any(pd.isna(efficientnet["true_label"].map(label_to_index))):
        raise ValueError("Bootstrap inputs contain unexpected true labels.")
    if np.any(pd.isna(efficientnet["predicted_label"].map(label_to_index))):
        raise ValueError("Bootstrap inputs contain unexpected EfficientNet-B0 predictions.")
    if np.any(pd.isna(resnet["predicted_label"].map(label_to_index))):
        raise ValueError("Bootstrap inputs contain unexpected ResNet50 predictions.")

    eff_correct = (eff_pred == true).astype(float)
    res_correct = (res_pred == true).astype(float)
    eff_codes = true * len(CLASSES) + eff_pred
    res_codes = true * len(CLASSES) + res_pred
    accuracy_differences = np.empty(resamples, dtype=float)
    macro_f1_differences = np.empty(resamples, dtype=float)
    rng = np.random.default_rng(bootstrap_seed)

    for start in range(0, resamples, chunk_size):
        stop = min(start + chunk_size, resamples)
        efficientnet_indices, resnet_indices = draw_paired_bootstrap_indices(
            true,
            resamples=stop - start,
            rng=rng,
            stratified=stratified,
        )
        accuracy_differences[start:stop] = (
            eff_correct[efficientnet_indices].mean(axis=1)
            - res_correct[resnet_indices].mean(axis=1)
        )
        eff_confusions = confusion_matrices_from_codes(eff_codes, efficientnet_indices)
        res_confusions = confusion_matrices_from_codes(res_codes, resnet_indices)
        macro_f1_differences[start:stop] = (
            macro_f1_from_confusions(eff_confusions)
            - macro_f1_from_confusions(res_confusions)
        )

    accuracy_ci = np.percentile(accuracy_differences, [2.5, 97.5])
    macro_f1_ci = np.percentile(macro_f1_differences, [2.5, 97.5])
    return {
        "bootstrap_resamples": resamples,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_stratified": stratified,
        "accuracy_difference_ci_lower": float(accuracy_ci[0]),
        "accuracy_difference_ci_upper": float(accuracy_ci[1]),
        "macro_f1_difference_ci_lower": float(macro_f1_ci[0]),
        "macro_f1_difference_ci_upper": float(macro_f1_ci[1]),
    }


def interval_interpretation(lower: float, upper: float) -> str:
    return "confidence interval includes zero" if lower <= 0 <= upper else "confidence interval excludes zero"


def analyze_seed(
    seed: int,
    efficientnet: pd.DataFrame,
    resnet: pd.DataFrame,
    *,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    expected_samples: int = EXPECTED_TEST_SAMPLES,
    stratified_bootstrap: bool = True,
) -> dict[str, Any]:
    efficientnet, resnet = align_predictions(
        efficientnet,
        resnet,
        expected_samples=expected_samples,
    )
    class_counts = (
        efficientnet["true_label"].value_counts().reindex(CLASSES, fill_value=0).astype(int)
    )
    if expected_samples == EXPECTED_TEST_SAMPLES and class_counts.tolist() != [360] * len(CLASSES):
        raise ValueError(
            "Production paired analysis requires exactly 360 true-label observations per class."
        )
    counts = exact_mcnemar_counts(efficientnet, resnet)
    stream_seed = bootstrap_seed + seed
    bootstrap = paired_bootstrap(
        efficientnet,
        resnet,
        resamples=bootstrap_resamples,
        bootstrap_seed=stream_seed,
        stratified=stratified_bootstrap,
    )
    eff_accuracy = float(accuracy_score(efficientnet["true_label"], efficientnet["predicted_label"]))
    res_accuracy = float(accuracy_score(resnet["true_label"], resnet["predicted_label"]))
    eff_macro_f1 = float(
        f1_score(
            efficientnet["true_label"],
            efficientnet["predicted_label"],
            labels=list(CLASSES),
            average="macro",
            zero_division=0,
        )
    )
    res_macro_f1 = float(
        f1_score(
            resnet["true_label"],
            resnet["predicted_label"],
            labels=list(CLASSES),
            average="macro",
            zero_division=0,
        )
    )
    mcnemar_text = (
        "statistically distinguishable on the fixed test set"
        if counts["mcnemar_exact_two_sided_p"] < 0.05
        else "not statistically distinguishable on the fixed test set"
    )
    return {
        "scope": "seed",
        "seed": seed,
        "test_samples": len(efficientnet),
        "true_class_counts": {class_name: int(class_counts[class_name]) for class_name in CLASSES},
        **counts,
        "efficientnet_accuracy": eff_accuracy,
        "resnet_accuracy": res_accuracy,
        "accuracy_difference_efficientnet_minus_resnet": eff_accuracy - res_accuracy,
        "efficientnet_macro_f1": eff_macro_f1,
        "resnet_macro_f1": res_macro_f1,
        "macro_f1_difference_efficientnet_minus_resnet": eff_macro_f1 - res_macro_f1,
        **bootstrap,
        "mcnemar_interpretation": mcnemar_text,
        "accuracy_ci_interpretation": interval_interpretation(
            float(bootstrap["accuracy_difference_ci_lower"]),
            float(bootstrap["accuracy_difference_ci_upper"]),
        ),
        "macro_f1_ci_interpretation": interval_interpretation(
            float(bootstrap["macro_f1_difference_ci_lower"]),
            float(bootstrap["macro_f1_difference_ci_upper"]),
        ),
    }


def multi_seed_summary(seed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    accuracy_differences = [
        float(row["accuracy_difference_efficientnet_minus_resnet"]) for row in seed_rows
    ]
    macro_f1_differences = [
        float(row["macro_f1_difference_efficientnet_minus_resnet"]) for row in seed_rows
    ]
    bootstrap_modes = {bool(row["bootstrap_stratified"]) for row in seed_rows}
    if len(bootstrap_modes) != 1:
        raise ValueError("All seed analyses must use the same bootstrap stratification mode.")
    return {
        "scope": "across_seeds",
        "seeds": list(SEEDS),
        "accuracy_differences": accuracy_differences,
        "mean_accuracy_difference": statistics.fmean(accuracy_differences),
        "sample_sd_accuracy_difference": statistics.stdev(accuracy_differences),
        "macro_f1_differences": macro_f1_differences,
        "mean_macro_f1_difference": statistics.fmean(macro_f1_differences),
        "sample_sd_macro_f1_difference": statistics.stdev(macro_f1_differences),
        "bootstrap_stratified": bootstrap_modes.pop(),
        "mcnemar_raw_p_values": {
            str(row["seed"]): float(row["mcnemar_exact_two_sided_p"]) for row in seed_rows
        },
        "mcnemar_holm_adjusted_p_values": {
            str(row["seed"]): float(row["mcnemar_holm_adjusted_p"]) for row in seed_rows
        },
        "mcnemar_multiple_testing_method": "Holm",
        "fisher_method_used": False,
        "interpretation": (
            "Only three paired seed observations and three repeated fixed-test-set McNemar tests "
            "are available. Report raw and Holm-adjusted p-values, effect direction, and "
            "variability; do not claim an overall statistical victory. Fisher's method is not used."
        ),
    }


def markdown_report(seed_rows: list[dict[str, Any]], across: dict[str, Any]) -> str:
    stratified = bool(across["bootstrap_stratified"])
    bootstrap_description = (
        "class-stratified by true label; each production resample preserves 360 observations per class"
        if stratified
        else "unstratified diagnostic mode; resampled class counts may vary"
    )
    lines = [
        "# Paired EfficientNet-B0 vs ResNet50 Statistics",
        "",
        "Differences are EfficientNet-B0 minus ResNet50. Exact two-sided McNemar tests and paired "
        "bootstrap resampling use aligned test-image indices.",
        "",
        f"Bootstrap stratified by true class: {'yes' if stratified else 'no'} ({bootstrap_description}).",
        "McNemar p-values are reported raw and with Holm adjustment across the three seeds. "
        "Fisher's method is not used.",
        "",
        "| Seed | Both correct | Both wrong | Only EfficientNet correct | Only ResNet correct | Raw exact McNemar p | Holm-adjusted p | Accuracy diff (95% CI) | Macro-F1 diff (95% CI) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in seed_rows:
        lines.append(
            f"| {row['seed']} | {row['both_models_correct']} | {row['both_models_wrong']} | "
            f"{row['only_efficientnet_correct']} | {row['only_resnet_correct']} | "
            f"{row['mcnemar_exact_two_sided_p']:.6g} | "
            f"{row['mcnemar_holm_adjusted_p']:.6g} | "
            f"{100 * row['accuracy_difference_efficientnet_minus_resnet']:.2f} pp "
            f"[{100 * row['accuracy_difference_ci_lower']:.2f}, {100 * row['accuracy_difference_ci_upper']:.2f}] | "
            f"{100 * row['macro_f1_difference_efficientnet_minus_resnet']:.2f} pp "
            f"[{100 * row['macro_f1_difference_ci_lower']:.2f}, {100 * row['macro_f1_difference_ci_upper']:.2f}] |"
        )
    lines.extend(["", "## Fixed-test-set interpretation", ""])
    for row in seed_rows:
        lines.append(
            f"- Seed {row['seed']}: raw test—{row['mcnemar_interpretation']}; "
            f"familywise result—{row['mcnemar_holm_interpretation']}; accuracy "
            f"{row['accuracy_ci_interpretation']}; macro-F1 {row['macro_f1_ci_interpretation']}."
        )
    lines.extend(
        [
            "",
            "These statements concern paired behavior on this fixed 1,440-image test set. They do "
            "not automatically establish model superiority or an overall statistical victory.",
            "",
            "## Across-seed paired differences",
            "",
            "| Seed | Accuracy difference | Macro-F1 difference |",
            "|---:|---:|---:|",
        ]
    )
    for row in seed_rows:
        lines.append(
            f"| {row['seed']} | {100 * row['accuracy_difference_efficientnet_minus_resnet']:.2f} pp | "
            f"{100 * row['macro_f1_difference_efficientnet_minus_resnet']:.2f} pp |"
        )
    lines.extend(
        [
            "",
            f"Mean accuracy difference: {100 * across['mean_accuracy_difference']:.2f} pp "
            f"(sample SD {100 * across['sample_sd_accuracy_difference']:.2f} pp).",
            "",
            f"Mean macro-F1 difference: {100 * across['mean_macro_f1_difference']:.2f} pp "
            f"(sample SD {100 * across['sample_sd_macro_f1_difference']:.2f} pp).",
            "",
            across["interpretation"],
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(
    seed_rows: list[dict[str, Any]],
    across: dict[str, Any],
    output_dir: Path,
    *,
    bootstrap_seed: int,
    bootstrap_resamples: int,
    stratified_bootstrap: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "paired_statistics.csv"
    markdown_path = output_dir / "paired_statistics.md"
    json_path = output_dir / "paired_statistics.json"
    existing = [path for path in (csv_path, markdown_path, json_path) if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite statistical outputs: {existing}")

    csv_rows = list(seed_rows) + [across]
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False, float_format="%.17g")
    markdown_path.write_text(markdown_report(seed_rows, across), encoding="utf-8")
    payload = {
        "schema_version": 1,
        "comparison": "EfficientNet-B0 minus ResNet50",
        "test_set_samples": EXPECTED_TEST_SAMPLES,
        "mcnemar": {
            "method": "exact two-sided binomial test on discordant pairs",
            "raw_p_values_reported": True,
            "multiple_testing_adjustment": "Holm across three seed-specific tests",
            "fisher_method_used": False,
            "overall_victory_claimed": False,
        },
        "bootstrap": {
            "method": "paired image-index resampling with 95% percentile interval",
            "stratified_by_true_class": stratified_bootstrap,
            "production_observations_per_class_per_resample": 360 if stratified_bootstrap else None,
            "base_seed": bootstrap_seed,
            "resamples_per_experiment_seed": bootstrap_resamples,
        },
        "per_seed": seed_rows,
        "across_seeds": across,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {markdown_path}")
    print(f"wrote {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired prediction-level model statistics.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=ROOT / "reproducibility/results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reproducibility/statistics",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument(
        "--no-stratified-bootstrap",
        dest="stratified_bootstrap",
        action="store_false",
        help="Disable true-class stratification for diagnostics only.",
    )
    parser.set_defaults(stratified_bootstrap=True)
    args = parser.parse_args()
    if args.bootstrap_resamples < 10_000:
        raise ValueError("Production paired analysis requires at least 10,000 bootstrap resamples.")

    seed_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        efficientnet = load_predictions(
            prediction_path(args.results_root, EFFICIENTNET, seed),
            EFFICIENTNET,
            seed,
        )
        resnet = load_predictions(
            prediction_path(args.results_root, RESNET, seed),
            RESNET,
            seed,
        )
        seed_rows.append(
            analyze_seed(
                seed,
                efficientnet,
                resnet,
                bootstrap_resamples=args.bootstrap_resamples,
                bootstrap_seed=args.bootstrap_seed,
                stratified_bootstrap=args.stratified_bootstrap,
            )
        )
    seed_rows = apply_holm_adjustment(seed_rows)
    across = multi_seed_summary(seed_rows)
    write_outputs(
        seed_rows,
        across,
        args.output_dir,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_resamples=args.bootstrap_resamples,
        stratified_bootstrap=args.stratified_bootstrap,
    )


if __name__ == "__main__":
    main()
