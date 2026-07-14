#!/usr/bin/env python3
"""Validate partitions, six model-seed results, metrics, and non-overwrite provenance."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)

from reproducibility_common import (
    CLASSES,
    EXPECTED_TEST_SAMPLES,
    MODELS,
    PROBABILITY_COLUMNS,
    ROOT,
    SEEDS,
    read_json,
    resolve_project_path,
    run_id,
    sha256_file,
    stable_sample_id,
)
from run_federated_benchmark import (
    load_manifest,
    state_dict_nbytes,
    validate_client_partition,
)


DEFAULT_BASELINE = ROOT / "reproducibility/baseline_existing_results_sha256.json"
EXISTING_RESULTS = ROOT / "federated_results_20rounds"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_close(actual: float, expected: float, label: str, tolerance: float = 1e-9) -> None:
    if not np.isclose(actual, expected, rtol=tolerance, atol=tolerance):
        raise ValueError(f"{label}: calculated {actual!r}, reported {expected!r}")


def resolve_artifact(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else resolve_project_path(path)


def capture_existing_results_baseline(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing baseline: {path}")
    files = sorted(item for item in EXISTING_RESULTS.rglob("*") if item.is_file())
    payload = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(EXISTING_RESULTS.relative_to(ROOT)),
        "files": {
            str(item.relative_to(EXISTING_RESULTS)): {
                "sha256": sha256_file(item),
                "size_bytes": item.stat().st_size,
            }
            for item in files
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"captured immutable baseline: {path} ({len(files)} files)")


def verify_existing_results_baseline(path: Path) -> None:
    baseline = read_json(path)
    expected = baseline.get("files", {})
    if not isinstance(expected, dict) or not expected:
        raise ValueError(f"Invalid baseline file list: {path}")
    current_files = {
        str(item.relative_to(EXISTING_RESULTS)): item
        for item in EXISTING_RESULTS.rglob("*")
        if item.is_file()
    }
    require(
        set(current_files) == set(expected),
        "Existing result file set changed since baseline capture.",
    )
    for relative_path, metadata in expected.items():
        item = current_files[relative_path]
        require(
            item.stat().st_size == int(metadata["size_bytes"]),
            f"Existing result size changed: {item}",
        )
        require(
            sha256_file(item) == metadata["sha256"],
            f"Existing result content changed: {item}",
        )
    print(f"verified unchanged existing results: {len(expected)} files")


def validate_manifests() -> dict[str, pd.DataFrame]:
    manifests = {
        split: pd.read_csv(ROOT / f"cluster_split_outputs/{split}_manifest.csv")
        for split in ("train", "val", "test")
    }
    expected_sizes = {"train": 5_040, "val": 720, "test": EXPECTED_TEST_SAMPLES}
    for split, frame in manifests.items():
        require(len(frame) == expected_sizes[split], f"Unexpected {split} manifest size.")
        require(frame["file_path"].is_unique, f"Duplicate file paths within {split} manifest.")
        require(set(frame["final_split"]) == {split}, f"Wrong final_split in {split} manifest.")
        require(set(frame["class_name"]) == set(CLASSES), f"Wrong classes in {split} manifest.")
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        path_overlap = set(manifests[left]["file_path"]) & set(manifests[right]["file_path"])
        cluster_overlap = set(manifests[left]["cluster_id"]) & set(manifests[right]["cluster_id"])
        require(not path_overlap, f"{left}/{right} sample overlap: {len(path_overlap)}")
        require(not cluster_overlap, f"{left}/{right} cluster overlap: {len(cluster_overlap)}")
    print("verified train/validation/test manifests: no sample or cluster overlap")
    return manifests


def validate_partitions(partition_root: Path) -> dict[int, tuple[Path, str]]:
    train_df = load_manifest(ROOT / "cluster_split_outputs/train_manifest.csv", "train")
    validated: dict[int, tuple[Path, str]] = {}
    for seed in SEEDS:
        path = partition_root / f"noniid_seed{seed}.csv"
        require(path.is_file(), f"Missing partition: {path}")
        partition = validate_client_partition(
            pd.read_csv(path),
            train_df,
            clients=4,
            seed=seed,
            dominant_fraction=0.70,
            source=path,
        )
        require(len(partition) == 5_040, f"Wrong partition size for seed {seed}")
        validated[seed] = (path.resolve(), sha256_file(path))
    print("verified three shared 70/30 Non-IID partitions")
    return validated


def validate_prediction_frame(
    frame: pd.DataFrame,
    *,
    path: Path,
    model_name: str,
    seed: int,
    best_round: int,
    test_manifest: pd.DataFrame,
) -> None:
    required = {
        "sample_id",
        "image_path",
        "true_label",
        "predicted_label",
        "correct",
        *PROBABILITY_COLUMNS.values(),
        "model",
        "seed",
        "best_round",
    }
    missing = required - set(frame.columns)
    require(not missing, f"{path} missing prediction columns: {sorted(missing)}")
    require(len(frame) == EXPECTED_TEST_SAMPLES, f"{path} must have 1,440 rows.")
    require(frame["sample_id"].is_unique, f"{path} has duplicate sample IDs.")
    require(frame["image_path"].is_unique, f"{path} has duplicate image paths.")
    require(frame["sample_id"].is_monotonic_increasing, f"{path} is not stably sample-ID sorted.")
    calculated_ids = frame["image_path"].map(stable_sample_id)
    require(calculated_ids.equals(frame["sample_id"]), f"{path} sample IDs are not path-stable hashes.")
    require(set(frame["model"].astype(str)) == {model_name}, f"{path} model mismatch.")
    require(set(pd.to_numeric(frame["seed"]).astype(int)) == {seed}, f"{path} seed mismatch.")
    require(
        set(pd.to_numeric(frame["best_round"]).astype(int)) == {best_round},
        f"{path} best-round mismatch.",
    )
    expected_labels = test_manifest.set_index("file_path")["class_name"]
    actual = frame.set_index("image_path")
    require(set(actual.index) == set(expected_labels.index), f"{path} test sample set mismatch.")
    require(
        actual["true_label"].sort_index().equals(expected_labels.sort_index()),
        f"{path} true labels differ from test manifest.",
    )
    calculated_correct = (frame["true_label"] == frame["predicted_label"]).astype(int)
    require(
        calculated_correct.equals(pd.to_numeric(frame["correct"]).astype(int)),
        f"{path} correct column mismatch.",
    )
    probabilities = frame[list(PROBABILITY_COLUMNS.values())].to_numpy(dtype=float)
    require(np.isfinite(probabilities).all(), f"{path} has non-finite probabilities.")
    require((probabilities >= 0).all() and (probabilities <= 1).all(), f"{path} probabilities outside [0,1].")
    require(np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5), f"{path} probabilities do not sum to one.")


def validate_reported_metrics(
    result: dict[str, Any],
    predictions: pd.DataFrame,
    *,
    report_path: Path,
    confusion_path: Path,
) -> None:
    y_true = predictions["true_label"]
    y_pred = predictions["predicted_label"]
    probabilities = predictions[list(PROBABILITY_COLUMNS.values())].to_numpy(dtype=float)
    normalized_probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    calculated = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(CLASSES), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=list(CLASSES), average="weighted", zero_division=0)),
        "loss": float(log_loss(y_true, normalized_probabilities, labels=list(CLASSES))),
    }
    for metric, value in calculated.items():
        tolerance = 1e-7 if metric == "loss" else 1e-12
        require_close(value, float(result["test_metrics"][metric]), f"{result['run_id']} {metric}", tolerance)

    calculated_report = classification_report(
        y_true,
        y_pred,
        labels=list(CLASSES),
        target_names=list(CLASSES),
        output_dict=True,
        zero_division=0,
    )
    saved_report = read_json(report_path)
    for label in (*CLASSES, "macro avg", "weighted avg"):
        for metric in ("precision", "recall", "f1-score", "support"):
            require_close(
                float(calculated_report[label][metric]),
                float(saved_report[label][metric]),
                f"{result['run_id']} report {label} {metric}",
                1e-10,
            )
    require_close(
        float(calculated_report["accuracy"]),
        float(saved_report["accuracy"]),
        f"{result['run_id']} report accuracy",
        1e-12,
    )

    calculated_confusion = confusion_matrix(y_true, y_pred, labels=list(CLASSES))
    saved_confusion = pd.read_csv(confusion_path, index_col=0).reindex(index=CLASSES, columns=CLASSES)
    require(
        np.array_equal(calculated_confusion, saved_confusion.to_numpy(dtype=int)),
        f"{result['run_id']} confusion matrix mismatch.",
    )


def validate_one_result(
    result_path: Path,
    *,
    model_name: str,
    seed: int,
    partition_path: Path,
    partition_hash: str,
    test_manifest: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    result = read_json(result_path)
    expected_run = run_id(model_name, seed)
    require(result.get("run_id") == expected_run, f"Run ID mismatch in {result_path}")
    require(result.get("model_name") == model_name, f"Model mismatch in {result_path}")
    require(result.get("algorithm") == "fedavg", f"Algorithm mismatch in {result_path}")
    require(result.get("partition") == "noniid", f"Partition mismatch in {result_path}")
    require(
        int(result["seed_metadata"]["experiment_seed"]) == seed,
        f"Experiment seed mismatch in {result_path}",
    )
    require(int(result["seed_metadata"]["partition_seed"]) == seed, f"Partition seed mismatch in {result_path}")
    require(int(result["seed_metadata"]["model_initialization_seed"]) == seed, f"Model seed mismatch in {result_path}")
    require(int(result["seed_metadata"]["dataloader_seed"]) == seed, f"DataLoader seed mismatch in {result_path}")
    result_partition = resolve_artifact(result["partition_file"]).resolve()
    require(result_partition == partition_path, f"Wrong partition path in {result_path}")
    require(result.get("partition_sha256") == partition_hash, f"Wrong partition hash in {result_path}")
    require(sha256_file(result_partition) == partition_hash, f"Partition changed after {result_path}")
    manifest_metadata = result.get("manifests", {})
    for split_name in ("train", "val", "test"):
        metadata = manifest_metadata.get(split_name, {})
        expected_manifest = (ROOT / f"cluster_split_outputs/{split_name}_manifest.csv").resolve()
        require(
            resolve_artifact(metadata.get("path", "")).resolve() == expected_manifest,
            f"Wrong {split_name} manifest path in {result_path}",
        )
        require(
            metadata.get("sha256") == sha256_file(expected_manifest),
            f"{split_name} manifest hash mismatch in {result_path}",
        )

    provenance = result.get("provenance", {})
    require(
        provenance.get("output_policy") == "fail_if_output_directory_exists",
        f"Missing no-overwrite output policy in {result_path}",
    )
    require(provenance.get("output_directory_created_new") is True, f"Output was not recorded as new: {result_path}")
    if seed == 42:
        require(result.get("training_performed") is False, "Seed 42 must not be retrained.")
        require(result.get("seed42_reused") is True, "Seed 42 reuse flag missing.")
        require(provenance.get("source") == "federated_results_20rounds", "Seed 42 source provenance mismatch.")
    else:
        require(result.get("training_performed") is True, f"Seed {seed} run not marked as trained.")
        require(result.get("seed42_reused") is False, f"Seed {seed} has wrong reuse flag.")

    environment = result.get("software_environment", {})
    require(environment, f"Software environment missing in {result_path}")
    require("bit-for-bit" in environment.get("determinism_note", ""), f"Determinism caveat missing in {result_path}")
    require(result.get("configuration_files"), f"Configuration file provenance missing in {result_path}")
    require(resolve_artifact(result["configuration_snapshot"]).is_file(), f"Configuration snapshot missing for {expected_run}")

    artifacts = result.get("artifacts", {})
    required_artifacts = (
        "round_history",
        "classification_report",
        "confusion_matrix",
        "predictions",
        "best_checkpoint",
        "environment",
    )
    for name in required_artifacts:
        require(name in artifacts, f"{name} path missing in {result_path}")
        require(resolve_artifact(artifacts[name]).is_file(), f"{name} file missing for {expected_run}")

    history_path = resolve_artifact(artifacts["round_history"])
    history = pd.read_csv(history_path)
    require(len(history) == 20, f"{expected_run} must have 20 history rows.")
    require(set(history["round"].astype(int)) == set(range(1, 21)), f"{expected_run} round history mismatch.")
    require({"val_accuracy", "val_macro_f1"}.issubset(history.columns), f"{expected_run} validation history incomplete.")
    best_round = int(result["selection"]["best_round"])
    calculated_best = int(history.loc[history["val_macro_f1"].astype(float).idxmax(), "round"])
    require(best_round == calculated_best, f"{expected_run} best round mismatch.")
    best_value = float(history.loc[history["round"].astype(int) == best_round, "val_macro_f1"].iloc[0])
    require_close(
        best_value,
        float(result["selection"]["best_validation_macro_f1"]),
        f"{expected_run} best validation macro-F1",
        1e-12,
    )

    predictions_path = resolve_artifact(artifacts["predictions"])
    predictions = pd.read_csv(predictions_path)
    validate_prediction_frame(
        predictions,
        path=predictions_path,
        model_name=model_name,
        seed=seed,
        best_round=best_round,
        test_manifest=test_manifest,
    )
    validate_reported_metrics(
        result,
        predictions,
        report_path=resolve_artifact(artifacts["classification_report"]),
        confusion_path=resolve_artifact(artifacts["confusion_matrix"]),
    )

    checkpoint_path = resolve_artifact(artifacts["best_checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_bytes = state_dict_nbytes(checkpoint["state_dict"])
    communication = result["communication"]
    require(checkpoint_bytes == int(communication["state_dict_bytes"]), f"{expected_run} checkpoint state size mismatch.")
    require(checkpoint_bytes == int(result["model"]["full_state_bytes"]), f"{expected_run} model state size mismatch.")
    require_close(
        checkpoint_bytes / (1024**2),
        float(communication["state_dict_mib"]),
        f"{expected_run} state MiB",
        1e-12,
    )
    clients = int(result["federated"]["clients"])
    rounds = int(result["federated"]["rounds"])
    require(clients == 4 and rounds == 20, f"{expected_run} schedule mismatch.")
    expected_total_mib = checkpoint_bytes * clients * rounds * 2 / (1024**2)
    require_close(
        expected_total_mib,
        float(communication["total_communication_mib"]),
        f"{expected_run} total communication",
        1e-12,
    )
    require(float(result["timing"]["elapsed_training_sec"]) > 0, f"{expected_run} elapsed training time missing.")
    print(f"verified {expected_run}")
    return result, predictions


def validate_all_results(
    results_root: Path,
    partitions: dict[int, tuple[Path, str]],
    test_manifest: pd.DataFrame,
    *,
    seeds: tuple[int, ...] = SEEDS,
    allow_partial: bool = False,
) -> int:
    expected_paths = {
        (model_name, seed): results_root / run_id(model_name, seed) / "result.json"
        for model_name in MODELS
        for seed in seeds
    }
    missing = [path for path in expected_paths.values() if not path.is_file()]
    if missing and not allow_partial:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing model-seed combinations:\n{formatted}")
    selected_paths = {
        key: path for key, path in expected_paths.items() if path.is_file()
    }
    if allow_partial and not selected_paths:
        raise FileNotFoundError(
            f"No completed result.json files found for selected seeds: {list(seeds)}"
        )
    if allow_partial and missing:
        print(f"partial validation: skipping {len(missing)} incomplete model-seed combinations")

    loaded: dict[tuple[str, int], tuple[dict[str, Any], pd.DataFrame]] = {}
    for (model_name, seed), result_path in selected_paths.items():
        partition_path, partition_hash = partitions[seed]
        loaded[(model_name, seed)] = validate_one_result(
            result_path,
            model_name=model_name,
            seed=seed,
            partition_path=partition_path,
            partition_hash=partition_hash,
            test_manifest=test_manifest,
        )

    paired_seeds: list[int] = []
    for seed in seeds:
        if not {
            ("efficientnet_b0", seed),
            ("resnet50", seed),
        }.issubset(loaded):
            continue
        eff_result, eff_predictions = loaded[("efficientnet_b0", seed)]
        res_result, res_predictions = loaded[("resnet50", seed)]
        require(
            eff_result["partition_file"] == res_result["partition_file"],
            f"Models use different partition paths for seed {seed}.",
        )
        require(
            eff_result["partition_sha256"] == res_result["partition_sha256"],
            f"Models use different partition hashes for seed {seed}.",
        )
        eff = eff_predictions.sort_values("sample_id").set_index("sample_id")
        res = res_predictions.sort_values("sample_id").set_index("sample_id")
        require(eff.index.equals(res.index), f"Prediction sample IDs differ for seed {seed}.")
        require(eff["true_label"].equals(res["true_label"]), f"True labels differ for seed {seed}.")
        paired_seeds.append(seed)
    if paired_seeds:
        print(
            "verified same-seed model partition fairness and prediction alignment for "
            f"seeds {paired_seeds}"
        )
    return len(loaded)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate multi-seed reproducibility outputs.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=ROOT / "reproducibility/results",
    )
    parser.add_argument(
        "--partition-root",
        type=Path,
        default=ROOT / "reproducibility/client_partitions",
    )
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--capture-baseline", action="store_true")
    parser.add_argument("--partitions-only", action="store_true")
    parser.add_argument("--seed42-only", action="store_true")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Validate completed model-seed outputs and skip combinations not finished yet.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        choices=SEEDS,
        action="append",
        help="Validate only this seed; may be repeated.",
    )
    args = parser.parse_args()
    if args.seed42_only and args.seed:
        parser.error("--seed42-only cannot be combined with --seed")

    if args.capture_baseline:
        capture_existing_results_baseline(args.baseline)
        return
    manifests = validate_manifests()
    partitions = validate_partitions(args.partition_root)
    if args.partitions_only:
        print("partition-only validation complete")
        return
    require(args.baseline.is_file(), f"Existing-results baseline missing: {args.baseline}")
    verify_existing_results_baseline(args.baseline)
    if args.seed42_only:
        seeds = (42,)
    elif args.seed:
        seeds = tuple(dict.fromkeys(args.seed))
    else:
        seeds = SEEDS
    validated_count = validate_all_results(
        args.results_root,
        partitions,
        manifests["test"],
        seeds=seeds,
        allow_partial=args.allow_partial,
    )
    if args.allow_partial:
        print(f"partial reproducibility validation complete: {validated_count} completed runs")
    else:
        print("all selected reproducibility outputs valid")


if __name__ == "__main__":
    main()
