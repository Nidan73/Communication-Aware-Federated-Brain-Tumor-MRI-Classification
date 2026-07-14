#!/usr/bin/env python3
"""Summarize three-seed Non-IID FedAvg results without rounding inputs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import pandas as pd

from reproducibility_common import MODEL_LABELS, MODELS, ROOT, SEEDS, read_json, run_id


def load_six_results(results_root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    loaded: dict[tuple[str, int], dict[str, Any]] = {}
    missing: list[Path] = []
    for model_name in MODELS:
        for seed in SEEDS:
            path = results_root / run_id(model_name, seed) / "result.json"
            if not path.is_file():
                missing.append(path)
                continue
            payload = read_json(path)
            if payload.get("model_name") != model_name:
                raise ValueError(f"Model mismatch in {path}")
            observed_seed = int(payload.get("seed_metadata", {}).get("experiment_seed", -1))
            if observed_seed != seed:
                raise ValueError(f"Seed mismatch in {path}: {observed_seed} != {seed}")
            loaded[(model_name, seed)] = payload
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing model-seed results:\n{formatted}")
    return loaded


def require_constant(values: list[float], label: str, model_name: str) -> float:
    first = values[0]
    if any(abs(value - first) > 1e-9 for value in values[1:]):
        raise ValueError(f"{label} differs across seeds for {model_name}: {values}")
    return first


def build_summary_rows(
    results: dict[tuple[str, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_name in MODELS:
        payloads = [results[(model_name, seed)] for seed in SEEDS]
        accuracies = [float(payload["test_metrics"]["accuracy"]) for payload in payloads]
        macro_f1s = [float(payload["test_metrics"]["macro_f1"]) for payload in payloads]
        best_rounds = [int(payload["selection"]["best_round"]) for payload in payloads]
        state_sizes = [float(payload["communication"]["state_dict_mib"]) for payload in payloads]
        communications = [
            float(payload["communication"]["total_communication_mib"]) for payload in payloads
        ]
        row: dict[str, Any] = {
            "model": model_name,
            "model_label": MODEL_LABELS[model_name],
            "seeds": ",".join(str(seed) for seed in SEEDS),
        }
        for seed, accuracy, macro_f1, best_round in zip(
            SEEDS, accuracies, macro_f1s, best_rounds
        ):
            row[f"accuracy_seed{seed}"] = accuracy
            row[f"macro_f1_seed{seed}"] = macro_f1
            row[f"best_round_seed{seed}"] = best_round
        row.update(
            {
                "mean_accuracy": statistics.fmean(accuracies),
                "sample_sd_accuracy": statistics.stdev(accuracies),
                "mean_macro_f1": statistics.fmean(macro_f1s),
                "sample_sd_macro_f1": statistics.stdev(macro_f1s),
                "mean_best_round": statistics.fmean(best_rounds),
                "best_round_min": min(best_rounds),
                "best_round_max": max(best_rounds),
                "state_mib": require_constant(state_sizes, "State size", model_name),
                "total_communication_mib": require_constant(
                    communications, "Communication", model_name
                ),
            }
        )
        rows.append(row)
    return rows


def markdown_report(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Multi-seed Non-IID Results",
        "",
        "All means and sample standard deviations use unrounded seed-level values. "
        "Standard deviations use denominator n-1.",
        "",
        "| Model | Seeds | Accuracy (%) mean ± SD | Macro-F1 (%) mean ± SD | State (MiB) | Communication (MiB) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model_label']} | {row['seeds']} | "
            f"{100 * row['mean_accuracy']:.2f} ± {100 * row['sample_sd_accuracy']:.2f} | "
            f"{100 * row['mean_macro_f1']:.2f} ± {100 * row['sample_sd_macro_f1']:.2f} | "
            f"{row['state_mib']:.2f} | {row['total_communication_mib']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Seed-level values",
            "",
            "| Model | Seed | Accuracy | Macro-F1 | Best round |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        for seed in SEEDS:
            lines.append(
                f"| {row['model_label']} | {seed} | {row[f'accuracy_seed{seed}']:.6f} | "
                f"{row[f'macro_f1_seed{seed}']:.6f} | {row[f'best_round_seed{seed}']} |"
            )
    lines.extend(
        [
            "",
            "## Best-round stability",
            "",
        ]
    )
    for row in rows:
        lines.append(
            f"- {row['model_label']}: mean {row['mean_best_round']:.2f}; "
            f"range {row['best_round_min']}–{row['best_round_max']}."
        )
    return "\n".join(lines) + "\n"


def write_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "multiseed_results.csv"
    markdown_path = output_dir / "multiseed_results.md"
    json_path = output_dir / "multiseed_results.json"
    existing = [path for path in (csv_path, markdown_path, json_path) if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite summary outputs: {existing}")
    pd.DataFrame(rows).to_csv(csv_path, index=False, float_format="%.17g")
    markdown_path.write_text(markdown_report(rows), encoding="utf-8")
    payload = {
        "schema_version": 1,
        "seeds": list(SEEDS),
        "standard_deviation": "sample (n-1)",
        "models": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {markdown_path}")
    print(f"wrote {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize seeds 42, 123, and 2025.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=ROOT / "reproducibility/results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reproducibility/summary",
    )
    args = parser.parse_args()
    results = load_six_results(args.results_root)
    rows = build_summary_rows(results)
    write_outputs(rows, args.output_dir)


if __name__ == "__main__":
    main()

