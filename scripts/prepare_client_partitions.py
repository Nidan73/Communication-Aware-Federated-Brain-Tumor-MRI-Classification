#!/usr/bin/env python3
"""Generate each shared Non-IID client assignment once, then verify it."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from run_federated_benchmark import (
    build_noniid_split,
    client_count_table,
    load_manifest,
    resolve_path,
    save_client_split,
    validate_client_partition,
)


DEFAULT_SEEDS = (42, 123, 2025)


def assignment_map(frame: pd.DataFrame) -> pd.Series:
    return frame.set_index("file_path")["client_id"].astype(int).sort_index()


def import_seed42_assignment(source: Path, train_df: pd.DataFrame) -> pd.DataFrame:
    if not source.is_file():
        raise FileNotFoundError(f"Existing seed-42 assignment not found: {source}")
    old = pd.read_csv(source)
    required = {"file_path", "class_name", "client_id"}
    missing = required - set(old.columns)
    if missing:
        raise ValueError(f"{source} missing columns: {sorted(missing)}")
    if old["file_path"].duplicated().any():
        raise ValueError(f"{source} contains duplicate file_path values.")
    if set(old["file_path"]) != set(train_df["file_path"]):
        raise ValueError(f"{source} does not exactly cover current train manifest.")

    imported = train_df.merge(
        old[["file_path", "client_id"]],
        on="file_path",
        how="left",
        validate="one_to_one",
    )
    imported["client_id"] = imported["client_id"].astype(int)
    imported["partition"] = "noniid"
    imported["partition_type"] = "noniid"
    imported["seed"] = 42

    regenerated = build_noniid_split(train_df, clients=4, seed=42, dominant_fraction=0.70)
    if not assignment_map(imported).equals(assignment_map(regenerated)):
        raise ValueError(
            "Existing seed-42 assignment differs from deterministic seed-42 splitter; "
            "refusing to invent replacement provenance."
        )
    return imported


def validate_one(path: Path, train_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    partition = validate_client_partition(
        pd.read_csv(path),
        train_df,
        clients=4,
        seed=seed,
        dominant_fraction=0.70,
        source=path,
    )
    counts = client_count_table(partition).to_dict(orient="records")
    print(f"verified {path}: {len(partition)} unique assignments")
    print(counts)
    return partition


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create or validate one shared 70/30 Non-IID partition CSV per seed."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("cluster_split_outputs/train_manifest.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reproducibility/client_partitions"),
    )
    parser.add_argument(
        "--seed42-source",
        type=Path,
        default=Path("federated_results_20rounds/client_splits_noniid.csv"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    unsupported = set(args.seeds) - set(DEFAULT_SEEDS)
    if unsupported:
        raise ValueError(f"Only planned seeds are allowed: {sorted(DEFAULT_SEEDS)}")
    train_df = load_manifest(resolve_path(args.manifest), "train")
    output_dir = resolve_path(args.output_dir)
    if not args.validate_only:
        output_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        output_path = output_dir / f"noniid_seed{seed}.csv"
        if output_path.exists():
            validate_one(output_path, train_df, seed)
            continue
        if args.validate_only:
            raise FileNotFoundError(f"Partition does not exist: {output_path}")

        if seed == 42:
            partition = import_seed42_assignment(resolve_path(args.seed42_source), train_df)
        else:
            partition = build_noniid_split(
                train_df,
                clients=4,
                seed=seed,
                dominant_fraction=0.70,
            )
        partition = validate_client_partition(
            partition,
            train_df,
            clients=4,
            seed=seed,
            dominant_fraction=0.70,
            source=f"generated seed {seed}",
        )
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite partition: {output_path}")
        save_client_split(partition, output_path)
        validate_one(output_path, train_df, seed)


if __name__ == "__main__":
    main()

