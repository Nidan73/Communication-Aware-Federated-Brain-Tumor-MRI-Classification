#!/usr/bin/env python3
"""Package existing seed-42 artifacts without retraining or overwriting them."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

from reproducibility_common import (
    CLASSES,
    EXPECTED_TEST_SAMPLES,
    ROOT,
    project_relative,
    run_id,
    sha256_file,
    stable_sample_id,
)
from run_federated_benchmark import (
    collect_environment_metadata,
    create_model,
    evaluate,
    load_manifest,
    make_loader,
    make_transforms,
    save_predictions,
    seed_everything,
    write_json,
)


SOURCE_ROOT = ROOT / "federated_results_20rounds"
SOURCE_RUN_IDS = {
    "efficientnet_b0": "fedavg_efficientnet_b0_noniid_20r",
    "resnet50": "fedavg_resnet50_noniid_20r",
}


def requested_prediction_columns() -> list[str]:
    return [
        "sample_id",
        "image_path",
        "true_label",
        "predicted_label",
        "correct",
        "probability_glioma",
        "probability_meningioma",
        "probability_no_tumor",
        "probability_pituitary",
        "model",
        "seed",
        "best_round",
        "run_id",
    ]


def normalize_existing_predictions(
    source: Path,
    destination: Path,
    *,
    model_name: str,
    best_round: int,
) -> None:
    frame = pd.read_csv(source)
    required = {
        "file_path",
        "true_label",
        "pred_label",
        "correct",
        "prob_glioma",
        "prob_meningioma",
        "prob_notumor",
        "prob_pituitary",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{source} missing columns: {sorted(missing)}")
    normalized = pd.DataFrame(
        {
            "sample_id": frame["file_path"].map(stable_sample_id),
            "image_path": frame["file_path"],
            "true_label": frame["true_label"],
            "predicted_label": frame["pred_label"],
            "correct": frame["correct"].astype(int),
            "probability_glioma": frame["prob_glioma"],
            "probability_meningioma": frame["prob_meningioma"],
            "probability_no_tumor": frame["prob_notumor"],
            "probability_pituitary": frame["prob_pituitary"],
            "model": model_name,
            "seed": 42,
            "best_round": best_round,
            "run_id": run_id(model_name, 42),
        }
    ).sort_values("sample_id")
    if len(normalized) != EXPECTED_TEST_SAMPLES:
        raise ValueError(f"{source} has {len(normalized)} rows; expected {EXPECTED_TEST_SAMPLES}.")
    if normalized["sample_id"].duplicated().any():
        raise ValueError(f"{source} produces duplicate stable sample IDs.")
    normalized[requested_prediction_columns()].to_csv(destination, index=False)


def evaluate_checkpoint_if_needed(
    destination: Path,
    *,
    checkpoint_path: Path,
    model_name: str,
    best_round: int,
    device: torch.device,
) -> None:
    test_df = load_manifest(ROOT / "cluster_split_outputs/test_manifest.csv", "test")
    _, eval_transform = make_transforms(224)
    loader = make_loader(
        test_df,
        eval_transform,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        seed=20_042,
        device=device,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = create_model(model_name, len(CLASSES), "all", pretrained=False).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    metrics, y_true, y_pred, paths, probabilities = evaluate(
        model,
        loader,
        nn.CrossEntropyLoss(),
        device,
        desc=f"seed42-{model_name}-test",
    )
    del metrics
    save_predictions(
        destination,
        run_id(model_name, 42),
        model_name,
        42,
        best_round,
        y_true,
        y_pred,
        paths,
        probabilities,
    )


def package_model(
    model_name: str,
    *,
    device: torch.device,
    evaluate_if_missing: bool,
) -> None:
    source_run = SOURCE_RUN_IDS[model_name]
    target_run = run_id(model_name, 42)
    target_dir = ROOT / "reproducibility" / "results" / target_run
    if target_dir.exists():
        result_path = target_dir / "result.json"
        if result_path.is_file():
            existing = json.loads(result_path.read_text(encoding="utf-8"))
            if (
                existing.get("run_id") == target_run
                and existing.get("training_performed") is False
                and existing.get("seed42_reused") is True
            ):
                print(f"verified existing {target_run}; no overwrite, inference, or training")
                return
        raise FileExistsError(f"Refusing to overwrite incomplete seed-42 package: {target_dir}")

    comparison = pd.read_csv(SOURCE_ROOT / "final_comparison_table.csv").set_index("run_id")
    communication = pd.read_csv(SOURCE_ROOT / "communication_costs.csv").set_index("run_id")
    if source_run not in comparison.index or source_run not in communication.index:
        raise ValueError(f"Missing source summary row for {source_run}")
    result_row = comparison.loc[source_run]
    comm_row = communication.loc[source_run]
    best_round = int(result_row["best_round"])
    checkpoint_path = ROOT / str(result_row["checkpoint_path"])
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing seed-42 checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(checkpoint.get("round", -1)) != best_round:
        raise ValueError(f"Checkpoint best round mismatch for {source_run}")

    source_history = SOURCE_ROOT / f"metrics_per_round_{source_run}.csv"
    source_report = SOURCE_ROOT / f"classification_report_{source_run}.json"
    source_confusion = SOURCE_ROOT / f"confusion_matrix_{source_run}.csv"
    source_predictions = SOURCE_ROOT / f"predictions_{source_run}.csv"
    for source in (source_history, source_report, source_confusion):
        if not source.is_file():
            raise FileNotFoundError(f"Missing seed-42 source artifact: {source}")

    target_dir.mkdir(parents=True, exist_ok=False)
    history_path = target_dir / f"metrics_per_round_{target_run}.csv"
    report_path = target_dir / f"classification_report_{target_run}.json"
    confusion_path = target_dir / f"confusion_matrix_{target_run}.csv"
    predictions_path = target_dir / f"predictions_{target_run}.csv"
    history = pd.read_csv(source_history)
    history["run_id"] = target_run
    history.to_csv(history_path, index=False)
    shutil.copy2(source_report, report_path)
    shutil.copy2(source_confusion, confusion_path)

    if source_predictions.is_file():
        normalize_existing_predictions(
            source_predictions,
            predictions_path,
            model_name=model_name,
            best_round=best_round,
        )
        prediction_source = "reused_existing_prediction_csv"
    elif evaluate_if_missing:
        seed_everything(42)
        evaluate_checkpoint_if_needed(
            predictions_path,
            checkpoint_path=checkpoint_path,
            model_name=model_name,
            best_round=best_round,
            device=device,
        )
        prediction_source = "checkpoint_evaluation_only"
    else:
        raise FileNotFoundError(
            f"Prediction CSV missing for {source_run}. Re-run with "
            "--evaluate-if-predictions-missing to perform test inference only."
        )

    predictions = pd.read_csv(predictions_path)
    accuracy = float(accuracy_score(predictions["true_label"], predictions["predicted_label"]))
    macro_f1 = float(
        f1_score(
            predictions["true_label"],
            predictions["predicted_label"],
            labels=list(CLASSES),
            average="macro",
            zero_division=0,
        )
    )
    if abs(accuracy - float(result_row["test_accuracy"])) > 1e-12:
        raise ValueError(f"Recalculated accuracy mismatch for {source_run}")
    if abs(macro_f1 - float(result_row["test_macro_f1"])) > 1e-12:
        raise ValueError(f"Recalculated macro-F1 mismatch for {source_run}")

    partition_path = ROOT / "reproducibility/client_partitions/noniid_seed42.csv"
    source_config_path = SOURCE_ROOT / "config_used.json"
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    manifest_metadata = {
        split_name: {
            "path": f"cluster_split_outputs/{split_name}_manifest.csv",
            "sha256": sha256_file(ROOT / f"cluster_split_outputs/{split_name}_manifest.csv"),
        }
        for split_name in ("train", "val", "test")
    }
    config_snapshot = {
        "reuse_only": True,
        "training_performed": False,
        "seed": 42,
        "model_name": model_name,
        "source_run_id": source_run,
        "target_run_id": target_run,
        "partition_file": project_relative(partition_path),
        "checkpoint_path": project_relative(checkpoint_path),
        "source_configuration_file": "configs/fl_final_20rounds.yaml",
        "source_configuration_snapshot": source_config,
    }
    write_json(target_dir / "config_used.json", config_snapshot)
    environment = collect_environment_metadata(device)
    environment["original_training_environment_available"] = False
    environment["environment_scope"] = "seed-42 export/evaluation environment; original training environment unknown"
    write_json(target_dir / "environment.json", environment)
    write_json(
        target_dir / "partition_reference.json",
        {
            "path": project_relative(partition_path),
            "sha256": sha256_file(partition_path),
            "seed": 42,
            "partition_type": "noniid",
        },
    )

    training_elapsed = float(history["round_time_sec"].sum())
    result_payload = {
        "schema_version": 1,
        "run_id": target_run,
        "algorithm": "fedavg",
        "model_name": model_name,
        "partition": "noniid",
        "training_performed": False,
        "seed42_reused": True,
        "seed_metadata": {
            "experiment_seed": 42,
            "partition_seed": 42,
            "model_initialization_seed": 42,
            "dataloader_seed": 42,
            "historical_seed_note": (
                "Original benchmark seeded global RNGs with 42 once before its multi-experiment "
                "sequence; it did not reset model initialization immediately before each experiment."
            ),
            "historical_worker_seed_note": (
                "Original runner supplied deterministic generators to shuffled train loaders but "
                "did not supply an explicit worker_init_fn."
            ),
        },
        "partition_file": project_relative(partition_path),
        "partition_sha256": sha256_file(partition_path),
        "manifests": manifest_metadata,
        "configuration_files": ["configs/fl_final_20rounds.yaml"],
        "configuration_snapshot": project_relative(target_dir / "config_used.json"),
        "software_environment": environment,
        "model": {
            "pretrained_initialization": "imagenet",
            "centralized_checkpoint_used_for_initialization": False,
            "total_parameters": int(result_row["total_params"]),
            "trainable_parameters": int(result_row["trainable_params"]),
            "full_state_bytes": int(comm_row["state_dict_bytes"]),
            "full_state_mib": float(comm_row["state_dict_mb"]),
        },
        "federated": {
            "clients": int(comm_row["clients"]),
            "rounds": int(comm_row["rounds"]),
            "local_epochs": 3,
            "all_clients_per_round": True,
            "aggregation": "weighted_fedavg_by_client_sample_count",
        },
        "selection": {
            "metric": "validation_macro_f1",
            "best_round": best_round,
            "best_validation_macro_f1": float(result_row["best_val_macro_f1"]),
            "test_evaluated_after_selection": True,
        },
        "test_metrics": {
            "accuracy": float(result_row["test_accuracy"]),
            "macro_f1": float(result_row["test_macro_f1"]),
            "weighted_f1": float(result_row["test_weighted_f1"]),
            "loss": float(result_row["test_loss"]),
        },
        "communication": {
            "mode": "full_state_dict_upload_plus_download_per_round",
            "state_dict_bytes": int(comm_row["state_dict_bytes"]),
            "state_dict_mib": float(comm_row["state_dict_mb"]),
            "total_download_mib": float(comm_row["total_download_mb"]),
            "total_upload_mib": float(comm_row["total_upload_mb"]),
            "total_communication_mib": float(comm_row["total_comm_mb"]),
        },
        "timing": {
            "elapsed_training_sec": training_elapsed,
            "elapsed_total_sec": training_elapsed,
            "timing_note": "Sum of original saved per-round elapsed times.",
        },
        "artifacts": {
            "round_history": project_relative(history_path),
            "classification_report": project_relative(report_path),
            "confusion_matrix": project_relative(confusion_path),
            "predictions": project_relative(predictions_path),
            "best_checkpoint": project_relative(checkpoint_path),
            "environment": project_relative(target_dir / "environment.json"),
        },
        "provenance": {
            "source": "federated_results_20rounds",
            "source_run_id": source_run,
            "prediction_source": prediction_source,
            "source_checkpoint_sha256": sha256_file(checkpoint_path),
            "source_prediction_sha256": sha256_file(source_predictions) if source_predictions.exists() else None,
            "output_policy": "fail_if_output_directory_exists",
            "output_directory_created_new": True,
            "packaged_at_unix": time.time(),
        },
    }
    write_json(target_dir / "result.json", result_payload)
    print(f"packaged {target_run} from existing seed-42 artifacts; no training")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reuse existing seed-42 results without retraining.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(SOURCE_RUN_IDS),
        default=sorted(SOURCE_RUN_IDS),
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--evaluate-if-predictions-missing", action="store_true")
    args = parser.parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    for model_name in args.models:
        package_model(
            model_name,
            device=device,
            evaluate_if_missing=args.evaluate_if_predictions_missing,
        )


if __name__ == "__main__":
    main()
