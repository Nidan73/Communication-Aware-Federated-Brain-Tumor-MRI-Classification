#!/usr/bin/env python3
"""Federated learning benchmark on leakage-controlled cluster manifests.

This script is intentionally manifest-driven. It never reads the old
Dataset/Nickparvar/Training or Dataset/Nickparvar/Testing split directly.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

from reproducibility_common import (
    CLASSES as COMMON_CLASSES,
    PROBABILITY_COLUMNS,
    ROOT,
    project_relative,
    sha256_file,
    stable_sample_id,
)

try:
    import yaml
except ImportError:  # pragma: no cover - handled at runtime with a clear error.
    yaml = None


CLASSES = list(COMMON_CLASSES)
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASSES)}


def show_progress_bars() -> bool:
    """Keep tqdm interactive, but avoid carriage-return-heavy logs under tee."""
    return sys.stderr.isatty()


def progress_iter(iterable, desc: str):
    return tqdm(iterable, desc=desc, leave=False, disable=not show_progress_bars())


def require_yaml() -> None:
    if yaml is None:
        raise RuntimeError("PyYAML is required to read config files. Install with: pip install pyyaml")


def seed_everything(seed: int) -> dict[str, Any]:
    """Seed supported RNGs and request deterministic behavior with warnings."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    return {
        "python_random_seed": seed,
        "numpy_seed": seed,
        "torch_cpu_seed": seed,
        "torch_cuda_seed": seed,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "deterministic_algorithms": True,
        "deterministic_algorithms_warn_only": True,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def set_seed(seed: int) -> None:
    """Backward-compatible alias for older imports."""
    seed_everything(seed)


def seed_dataloader_worker(worker_id: int) -> None:
    """Seed Python and NumPy from PyTorch's deterministic worker seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def collect_environment_metadata(device: torch.device) -> dict[str, Any]:
    gpu_name = None
    if device.type == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(device)
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "executable": sys.executable,
        "packages": {
            name: package_version(name)
            for name in ("torch", "torchvision", "numpy", "pandas", "scikit-learn", "scipy", "PyYAML")
        },
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "gpu_name": gpu_name,
        "determinism_note": (
            "Perfect bit-for-bit reproducibility is not guaranteed across different GPU, CUDA, "
            "cuDNN, PyTorch, driver, operating-system, or hardware versions."
        ),
    }


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def load_yaml_config(path: Path) -> dict[str, Any]:
    require_yaml()
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    data["_config_path"] = str(path)
    return data


def merge_configs(paths: list[Path]) -> dict[str, Any]:
    merged: dict[str, Any] = {"experiments": [], "_source_configs": []}
    for path in paths:
        cfg = load_yaml_config(path)
        merged["_source_configs"].append(str(path))
        for key, value in cfg.items():
            if key == "experiments":
                merged["experiments"].extend(value or [])
            elif key.startswith("_"):
                continue
            elif isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
    if not merged["experiments"]:
        raise ValueError("No experiments found in config file(s).")
    return merged


def markdown_table(df: pd.DataFrame) -> str:
    def fmt(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6f}"
        return str(value).replace("|", "\\|")

    columns = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(fmt(value) for value in row) + " |")
    return "\n".join(lines)


class ManifestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True).copy()
        self.transform = transform
        self.samples: list[tuple[Path, int, str]] = []
        for row in self.df.itertuples(index=False):
            rel_path = getattr(row, "file_path")
            class_name = getattr(row, "class_name")
            if class_name not in CLASS_TO_IDX:
                raise ValueError(f"Unexpected class name: {class_name!r}")
            self.samples.append((resolve_path(rel_path), CLASS_TO_IDX[class_name], rel_path))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label, rel_path = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
        return img, label, rel_path


def make_transforms(img_size: int):
    train_tf = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return train_tf, eval_tf


def load_manifest(path: Path, expected_split: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"file_path", "class_name", "cluster_id", "final_split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    bad_classes = sorted(set(df["class_name"]) - set(CLASSES))
    if bad_classes:
        raise ValueError(f"{path} has unexpected classes: {bad_classes}")
    bad_splits = sorted(set(df["final_split"]) - {expected_split})
    if bad_splits:
        raise ValueError(f"{path} has unexpected final_split values: {bad_splits}")
    df = df.copy().reset_index(drop=True)
    df["sample_id"] = df["file_path"].map(stable_sample_id)
    if df["sample_id"].duplicated().any():
        raise ValueError(f"{path} produces duplicate stable sample IDs.")
    return df


def class_count_table(df: pd.DataFrame, label: str) -> pd.DataFrame:
    counts = df["class_name"].value_counts().reindex(CLASSES, fill_value=0)
    return pd.DataFrame({"split": label, "class_name": counts.index, "count": counts.values})


def build_iid_split(train_df: pd.DataFrame, clients: int, seed: int) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    for class_name in CLASSES:
        class_rows = train_df[train_df["class_name"] == class_name].copy()
        indices = class_rows.index.to_numpy(copy=True)
        rng.shuffle(indices)
        for client_id, piece in enumerate(np.array_split(indices, clients)):
            part = train_df.loc[piece].copy()
            part["client_id"] = client_id
            part["partition"] = "iid"
            rows.append(part)
    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["client_id", "class_name", "file_path"]).reset_index(drop=True)


def build_noniid_split(train_df: pd.DataFrame, clients: int, seed: int, dominant_fraction: float) -> pd.DataFrame:
    if clients != len(CLASSES):
        raise ValueError("The 70/30 label-skew splitter expects one client per class.")
    rows = []
    rng = np.random.default_rng(seed)
    for dominant_client, class_name in enumerate(CLASSES):
        class_rows = train_df[train_df["class_name"] == class_name].copy()
        indices = class_rows.index.to_numpy(copy=True)
        rng.shuffle(indices)
        dominant_count = int(round(len(indices) * dominant_fraction))
        dominant_indices = indices[:dominant_count]
        remainder_indices = indices[dominant_count:]

        dominant_part = train_df.loc[dominant_indices].copy()
        dominant_part["client_id"] = dominant_client
        dominant_part["partition"] = "noniid"
        rows.append(dominant_part)

        other_clients = [cid for cid in range(clients) if cid != dominant_client]
        for offset, piece in enumerate(np.array_split(remainder_indices, len(other_clients))):
            part = train_df.loc[piece].copy()
            part["client_id"] = other_clients[offset]
            part["partition"] = "noniid"
            rows.append(part)

    out = pd.concat(rows, ignore_index=True)
    out["seed"] = seed
    out["partition_type"] = "noniid"
    return out.sort_values(["client_id", "class_name", "file_path"]).reset_index(drop=True)


def validate_client_partition(
    split_df: pd.DataFrame,
    train_df: pd.DataFrame,
    *,
    clients: int,
    seed: int,
    dominant_fraction: float,
    source: str | Path = "client partition",
) -> pd.DataFrame:
    """Validate and canonicalize one saved 70/30 Non-IID assignment."""
    source_label = str(source)
    split_df = split_df.copy()
    if "image_path" in split_df and "file_path" not in split_df:
        split_df = split_df.rename(columns={"image_path": "file_path"})
    if "class" in split_df and "class_name" not in split_df:
        split_df = split_df.rename(columns={"class": "class_name"})
    if "partition_type" not in split_df and "partition" in split_df:
        split_df["partition_type"] = split_df["partition"]
    if "partition" not in split_df and "partition_type" in split_df:
        split_df["partition"] = split_df["partition_type"]

    required = {
        "sample_id",
        "file_path",
        "class_name",
        "client_id",
        "seed",
        "partition_type",
        "partition",
    }
    missing = required - set(split_df.columns)
    if missing:
        raise ValueError(f"{source_label} missing columns: {sorted(missing)}")
    if len(split_df) != len(train_df):
        raise ValueError(f"{source_label} has {len(split_df)} rows; expected {len(train_df)}.")
    if split_df["sample_id"].duplicated().any():
        raise ValueError(f"{source_label} contains duplicate sample_id values.")
    if split_df["file_path"].duplicated().any():
        raise ValueError(f"{source_label} assigns at least one image more than once.")

    split_df["client_id"] = pd.to_numeric(split_df["client_id"], errors="raise").astype(int)
    split_df["seed"] = pd.to_numeric(split_df["seed"], errors="raise").astype(int)
    if set(split_df["seed"]) != {seed}:
        raise ValueError(f"{source_label} seed values do not equal expected seed {seed}.")
    if set(split_df["partition_type"]) != {"noniid"} or set(split_df["partition"]) != {"noniid"}:
        raise ValueError(f"{source_label} must contain only the 'noniid' partition type.")
    if set(split_df["client_id"]) != set(range(clients)):
        raise ValueError(f"{source_label} must contain client IDs 0 through {clients - 1}.")

    manifest = train_df[["sample_id", "file_path", "class_name"]].copy()
    if set(split_df["file_path"]) != set(manifest["file_path"]):
        missing_paths = set(manifest["file_path"]) - set(split_df["file_path"])
        extra_paths = set(split_df["file_path"]) - set(manifest["file_path"])
        raise ValueError(
            f"{source_label} does not exactly cover train manifest; "
            f"missing={len(missing_paths)}, extra={len(extra_paths)}."
        )
    expected = manifest.set_index("file_path")
    actual = split_df.set_index("file_path")
    if not actual["sample_id"].sort_index().equals(expected["sample_id"].sort_index()):
        raise ValueError(f"{source_label} sample IDs do not match stable manifest-derived IDs.")
    if not actual["class_name"].sort_index().equals(expected["class_name"].sort_index()):
        raise ValueError(f"{source_label} class labels do not match train manifest.")

    class_sizes = train_df["class_name"].value_counts().reindex(CLASSES)
    if class_sizes.isna().any() or class_sizes.nunique() != 1:
        raise ValueError("70/30 verifier requires equal non-empty class counts.")
    class_size = int(class_sizes.iloc[0])
    dominant_count = int(round(class_size * dominant_fraction))
    other_count = (class_size - dominant_count) // (clients - 1)
    expected_client_total = dominant_count + other_count * (clients - 1)
    counts = (
        split_df.groupby(["client_id", "class_name"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=range(clients), columns=CLASSES, fill_value=0)
    )
    for client_id, row in counts.iterrows():
        row_counts = [int(row[class_name]) for class_name in CLASSES]
        if sum(row_counts) != expected_client_total:
            raise ValueError(
                f"{source_label} client {client_id} has {sum(row_counts)} samples; "
                f"expected {expected_client_total}."
            )
        if row_counts.count(dominant_count) != 1 or row_counts.count(other_count) != clients - 1:
            raise ValueError(
                f"{source_label} client {client_id} must have one class with {dominant_count} "
                f"samples and {clients - 1} classes with {other_count} samples."
            )
    return split_df.sort_values(["client_id", "class_name", "file_path"]).reset_index(drop=True)


def load_saved_client_partition(
    path: Path,
    train_df: pd.DataFrame,
    *,
    clients: int,
    seed: int,
    dominant_fraction: float,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Saved client partition not found: {path}")
    return validate_client_partition(
        pd.read_csv(path),
        train_df,
        clients=clients,
        seed=seed,
        dominant_fraction=dominant_fraction,
        source=path,
    )


def client_count_table(split_df: pd.DataFrame) -> pd.DataFrame:
    table = (
        split_df.groupby(["partition", "client_id", "class_name"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=CLASSES, fill_value=0)
        .reset_index()
    )
    table["total"] = table[CLASSES].sum(axis=1)
    return table


def save_client_split(df: pd.DataFrame, path: Path) -> None:
    columns = [
        "partition",
        "partition_type",
        "seed",
        "client_id",
        "sample_id",
        "file_path",
        "class_name",
        "cluster_id",
        "original_split",
        "final_split",
    ]
    keep = [c for c in columns if c in df.columns]
    df[keep].to_csv(path, index=False)


def state_dict_nbytes(state_dict: dict[str, torch.Tensor]) -> int:
    return int(
        sum(value.numel() * value.element_size() for value in state_dict.values() if torch.is_tensor(value))
    )


def parameter_counts(model: nn.Module) -> tuple[int, int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    state_bytes = state_dict_nbytes(model.state_dict())
    return total, trainable, state_bytes


class ProgressTracker:
    event_fields = [
        "timestamp",
        "event_type",
        "run_id",
        "experiment_index",
        "total_experiments",
        "algorithm",
        "model_name",
        "partition",
        "round",
        "total_rounds",
        "client_id",
        "clients",
        "local_epoch",
        "local_epochs",
        "completed_client_epochs",
        "total_client_epochs",
        "percent_complete",
        "elapsed_sec",
        "eta_sec",
        "epoch_loss",
        "val_macro_f1",
        "message",
    ]

    def __init__(self, output_dir: Path, cfg: dict[str, Any]) -> None:
        self.output_dir = output_dir
        self.status_path = output_dir / "progress_status.json"
        self.events_path = output_dir / "progress_events.csv"
        self.start_time = time.time()
        self.completed_client_epochs = 0
        self.total_experiments = len(cfg["experiments"])
        self.total_rounds = int(cfg["federated"]["rounds"])
        self.clients = int(cfg["federated"]["clients"])
        self.local_epochs = int(cfg["federated"]["local_epochs"])
        self.total_client_epochs = (
            self.total_experiments * self.total_rounds * self.clients * self.local_epochs
        )
        self.current_experiment_index = 0
        self.current_exp: dict[str, Any] = {}
        with self.events_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.event_fields)
            writer.writeheader()
        self._record("run_started", message="Federated benchmark started.")

    def _timestamp(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S")

    def _elapsed(self) -> float:
        return time.time() - self.start_time

    def _percent(self) -> float:
        if self.total_client_epochs <= 0:
            return 100.0
        return 100.0 * self.completed_client_epochs / self.total_client_epochs

    def _eta(self) -> float:
        if self.completed_client_epochs <= 0:
            return 0.0
        rate = self._elapsed() / self.completed_client_epochs
        remaining = self.total_client_epochs - self.completed_client_epochs
        return max(0.0, rate * remaining)

    def _record(
        self,
        event_type: str,
        *,
        round_idx: int | None = None,
        client_id: int | None = None,
        local_epoch: int | None = None,
        epoch_loss: float | None = None,
        val_macro_f1: float | None = None,
        message: str = "",
    ) -> None:
        exp = self.current_exp
        row = {
            "timestamp": self._timestamp(),
            "event_type": event_type,
            "run_id": exp.get("run_id", ""),
            "experiment_index": self.current_experiment_index,
            "total_experiments": self.total_experiments,
            "algorithm": exp.get("algorithm", ""),
            "model_name": exp.get("model_name", ""),
            "partition": exp.get("partition", ""),
            "round": round_idx or "",
            "total_rounds": self.total_rounds,
            "client_id": client_id if client_id is not None else "",
            "clients": self.clients,
            "local_epoch": local_epoch or "",
            "local_epochs": self.local_epochs,
            "completed_client_epochs": self.completed_client_epochs,
            "total_client_epochs": self.total_client_epochs,
            "percent_complete": round(self._percent(), 4),
            "elapsed_sec": round(self._elapsed(), 3),
            "eta_sec": round(self._eta(), 3),
            "epoch_loss": "" if epoch_loss is None else round(float(epoch_loss), 8),
            "val_macro_f1": "" if val_macro_f1 is None else round(float(val_macro_f1), 8),
            "message": message,
        }
        with self.events_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.event_fields)
            writer.writerow(row)
        status = {
            **row,
            "status": event_type,
            "progress_status_path": str(self.status_path.relative_to(ROOT)),
            "progress_events_path": str(self.events_path.relative_to(ROOT)),
        }
        self.status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    def start_experiment(self, exp: dict[str, Any], experiment_index: int) -> None:
        self.current_exp = exp
        self.current_experiment_index = experiment_index
        message = (
            f"Starting experiment {experiment_index}/{self.total_experiments}: "
            f"{exp['run_id']}"
        )
        self._record("experiment_started", message=message)
        print(message, flush=True)

    def start_round(self, round_idx: int) -> None:
        exp = self.current_exp
        message = (
            f"Round {round_idx}/{self.total_rounds} started for {exp['run_id']} "
            f"| progress {self.completed_client_epochs}/{self.total_client_epochs} "
            f"({self._percent():.1f}%)"
        )
        self._record("round_started", round_idx=round_idx, message=message)
        print(message, flush=True)

    def finish_local_epoch(
        self,
        *,
        round_idx: int,
        client_id: int,
        local_epoch: int,
        epoch_loss: float,
    ) -> None:
        self.completed_client_epochs += 1
        exp = self.current_exp
        message = (
            f"Progress: {self.completed_client_epochs}/{self.total_client_epochs} "
            f"client-epochs ({self._percent():.1f}%) | "
            f"{exp['run_id']} | round {round_idx}/{self.total_rounds} | "
            f"client {client_id + 1}/{self.clients} | "
            f"local epoch {local_epoch}/{self.local_epochs} | "
            f"loss {epoch_loss:.4f} | ETA {self._eta() / 60:.1f} min"
        )
        self._record(
            "local_epoch_completed",
            round_idx=round_idx,
            client_id=client_id,
            local_epoch=local_epoch,
            epoch_loss=epoch_loss,
            message=message,
        )
        print(message, flush=True)

    def finish_round(self, round_idx: int, val_macro_f1: float) -> None:
        exp = self.current_exp
        message = (
            f"Round {round_idx}/{self.total_rounds} complete for {exp['run_id']} "
            f"| val_macro_f1 {val_macro_f1:.4f} | "
            f"progress {self.completed_client_epochs}/{self.total_client_epochs} "
            f"({self._percent():.1f}%)"
        )
        self._record(
            "round_completed",
            round_idx=round_idx,
            val_macro_f1=val_macro_f1,
            message=message,
        )
        print(message, flush=True)

    def finish_experiment(self, test_macro_f1: float, test_accuracy: float) -> None:
        exp = self.current_exp
        message = (
            f"Experiment complete: {exp['run_id']} | "
            f"test_accuracy {test_accuracy:.4f} | test_macro_f1 {test_macro_f1:.4f}"
        )
        self._record("experiment_completed", message=message)
        print(message, flush=True)

    def finish_run(self) -> None:
        self._record("run_completed", message="Federated benchmark complete.")
        print(
            f"Federated benchmark complete: {self.completed_client_epochs}/"
            f"{self.total_client_epochs} client-epochs (100.0%)",
            flush=True,
        )


def set_trainable_layers(model: nn.Module, trainable_layers: str | list[str] | None) -> None:
    if trainable_layers is None or trainable_layers == "all":
        for param in model.parameters():
            param.requires_grad = True
        return
    if trainable_layers == "classifier":
        trainable_layers = ["classifier", "fc"]
    for name, param in model.named_parameters():
        param.requires_grad = any(token in name for token in trainable_layers)


def create_model(
    model_name: str,
    num_classes: int,
    trainable_layers: str | list[str] | None = "all",
    pretrained: str | bool = "imagenet",
) -> nn.Module:
    use_imagenet = pretrained is True or str(pretrained).lower() in {"imagenet", "true", "yes"}
    if pretrained not in {True, False, "imagenet", "true", "false", "yes", "no", "none", None}:
        raise ValueError(f"Unsupported pretrained setting: {pretrained!r}")
    if model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if use_imagenet else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if use_imagenet else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    set_trainable_layers(model, trainable_layers)
    return model


def make_loader(
    df: pd.DataFrame,
    transform,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "generator": generator,
        "worker_init_fn": seed_dataloader_worker,
    }
    return DataLoader(ManifestDataset(df, transform), **kwargs)


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion, device: torch.device, desc: str):
    model.eval()
    total_loss = 0.0
    total = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    paths: list[str] = []
    probs: list[list[float]] = []
    for images, labels, batch_paths in progress_iter(loader, desc=desc):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        batch_probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)
        bs = labels.size(0)
        total_loss += loss.item() * bs
        total += bs
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
        paths.extend(batch_paths)
        probs.extend(batch_probs.cpu().tolist())
    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / max(total, 1)
    return metrics, y_true, y_pred, paths, probs


def aggregate_fedavg(client_states: list[dict[str, torch.Tensor]], sample_counts: list[int]) -> dict[str, torch.Tensor]:
    total_samples = float(sum(sample_counts))
    if total_samples <= 0:
        raise ValueError("Cannot aggregate with zero client samples.")
    aggregated: dict[str, torch.Tensor] = {}
    for key in client_states[0].keys():
        first = client_states[0][key]
        if torch.is_tensor(first) and torch.is_floating_point(first):
            dtype = torch.float64 if first.dtype == torch.float64 else torch.float32
            acc = torch.zeros_like(first, dtype=dtype)
            for state, count in zip(client_states, sample_counts):
                acc += state[key].to(dtype=dtype) * (count / total_samples)
            aggregated[key] = acc.to(dtype=first.dtype)
        else:
            aggregated[key] = first.clone() if torch.is_tensor(first) else copy.deepcopy(first)
    return aggregated


def train_local_model(
    global_model: nn.Module,
    client_loader: DataLoader,
    exp: dict[str, Any],
    cfg: dict[str, Any],
    device: torch.device,
    progress_tracker: ProgressTracker | None = None,
    round_idx: int | None = None,
    client_id: int | None = None,
) -> tuple[dict[str, torch.Tensor], int, float]:
    local_model = copy.deepcopy(global_model).to(device)
    local_model.train()
    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in local_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(exp.get("learning_rate", cfg["training"]["learning_rate"])),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    algorithm = exp.get("algorithm", "fedavg").lower()
    prox_mu = float(exp.get("prox_mu", cfg.get("fedprox", {}).get("mu", 0.0)))
    global_params = {
        name: param.detach().clone()
        for name, param in global_model.named_parameters()
        if param.requires_grad
    }

    total_loss = 0.0
    total_seen = 0
    local_epochs = int(cfg["federated"]["local_epochs"])
    for local_epoch in range(1, local_epochs + 1):
        epoch_loss = 0.0
        epoch_seen = 0
        for images, labels, _ in progress_iter(client_loader, desc="local"):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = local_model(images)
            loss = criterion(outputs, labels)
            if algorithm == "fedprox" and prox_mu > 0:
                prox_term = torch.zeros((), device=device)
                for name, param in local_model.named_parameters():
                    if param.requires_grad:
                        prox_term = prox_term + torch.sum((param - global_params[name]) ** 2)
                loss = loss + 0.5 * prox_mu * prox_term
            loss.backward()
            optimizer.step()
            bs = labels.size(0)
            total_loss += loss.item() * bs
            total_seen += bs
            epoch_loss += loss.item() * bs
            epoch_seen += bs
        if progress_tracker is not None and round_idx is not None and client_id is not None:
            progress_tracker.finish_local_epoch(
                round_idx=round_idx,
                client_id=client_id,
                local_epoch=local_epoch,
                epoch_loss=epoch_loss / max(epoch_seen, 1),
            )

    state = {key: value.detach().cpu() for key, value in local_model.state_dict().items()}
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return state, len(client_loader.dataset), total_loss / max(total_seen, 1)


def write_json(path: Path, payload: Any) -> None:
    def encode(value: Any):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    path.write_text(json.dumps(payload, indent=2, default=encode) + "\n", encoding="utf-8")


def save_predictions(
    path: Path,
    run_id: str,
    model_name: str,
    seed: int,
    best_round: int,
    y_true: list[int],
    y_pred: list[int],
    paths: list[str],
    probs: list[list[float]],
) -> None:
    rows = []
    for file_path, true_idx, pred_idx, prob_row in zip(paths, y_true, y_pred, probs):
        row = {
            "sample_id": stable_sample_id(file_path),
            "image_path": file_path,
            "true_label": CLASSES[true_idx],
            "predicted_label": CLASSES[pred_idx],
            "correct": int(true_idx == pred_idx),
            "model": model_name,
            "seed": seed,
            "best_round": best_round,
            "run_id": run_id,
        }
        for class_name, prob in zip(CLASSES, prob_row):
            row[PROBABILITY_COLUMNS[class_name]] = prob
        rows.append(row)
    predictions = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)
    if predictions["sample_id"].duplicated().any():
        raise ValueError("Prediction output contains duplicate stable sample IDs.")
    required_order = [
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
    predictions[required_order].to_csv(path, index=False)


def upsert_rows_by_run_id(path: Path, rows: list[dict[str, Any]]) -> pd.DataFrame:
    new_df = pd.DataFrame(rows)
    if path.exists():
        old_df = pd.read_csv(path)
        if "run_id" in old_df.columns and "run_id" in new_df.columns:
            old_df = old_df[~old_df["run_id"].isin(new_df["run_id"])]
        out = pd.concat([old_df, new_df], ignore_index=True, sort=False)
    else:
        out = new_df
    out.to_csv(path, index=False)
    return out


def run_experiment(
    exp: dict[str, Any],
    experiment_index: int,
    cfg: dict[str, Any],
    split_map: dict[str, pd.DataFrame],
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
    device: torch.device,
    progress_tracker: ProgressTracker | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    run_id = exp["run_id"]
    model_name = exp["model_name"]
    partition = exp["partition"]
    clients = int(cfg["federated"]["clients"])
    rounds = int(cfg["federated"]["rounds"])
    experiment_seed = int(exp.get("seed", cfg["training"]["seed"]))
    partition_seed = int(cfg["split"].get("partition_seed", experiment_seed))
    model_initialization_seed = int(exp.get("model_initialization_seed", experiment_seed))
    dataloader_seed = int(exp.get("dataloader_seed", experiment_seed))
    batch_size = int(cfg["training"]["batch_size"])
    num_workers = int(cfg["training"]["num_workers"])
    run_started = time.time()
    seed_settings = seed_everything(experiment_seed)
    partition_file_value = cfg["split"].get("partition_file")
    partition_path = resolve_path(partition_file_value).resolve() if partition_file_value else None
    partition_hash = sha256_file(partition_path) if partition_path else None
    environment = collect_environment_metadata(device)
    manifest_metadata = {
        split_name: {
            "path": project_relative(resolve_path(path_value)),
            "sha256": sha256_file(resolve_path(path_value)),
        }
        for split_name, path_value in cfg["manifests"].items()
    }
    loader_seeds = {
        "base": dataloader_seed,
        "clients": {str(client_id): dataloader_seed + client_id for client_id in range(clients)},
        "validation": dataloader_seed + 10_000,
        "test": dataloader_seed + 20_000,
    }
    seed_metadata = {
        "experiment_seed": experiment_seed,
        "partition_seed": partition_seed,
        "model_initialization_seed": model_initialization_seed,
        "dataloader_seed": dataloader_seed,
        "loader_seed_derivations": loader_seeds,
        "settings": seed_settings,
    }

    train_tf, eval_tf = make_transforms(int(cfg["training"]["image_size"]))
    client_split = split_map[partition]
    client_loaders = []
    for client_id in range(clients):
        client_df = client_split[client_split["client_id"] == client_id].reset_index(drop=True)
        client_loaders.append(
            make_loader(
                client_df,
                train_tf,
                batch_size,
                True,
                num_workers,
                loader_seeds["clients"][str(client_id)],
                device,
            )
        )
    val_loader = make_loader(
        val_df, eval_tf, batch_size, False, num_workers, loader_seeds["validation"], device
    )
    test_loader = make_loader(
        test_df, eval_tf, batch_size, False, num_workers, loader_seeds["test"], device
    )

    seed_everything(model_initialization_seed)
    model = create_model(
        model_name,
        len(CLASSES),
        exp.get("trainable_layers", "all"),
        exp.get("pretrained", "imagenet"),
    ).to(device)
    total_params, trainable_params, state_bytes = parameter_counts(model)
    criterion = nn.CrossEntropyLoss()
    best_val_macro_f1 = -1.0
    best_round = 0
    best_checkpoint = output_dir / "checkpoints" / f"{run_id}_best.pth"
    round_rows = []
    total_comm_bytes = 0
    training_started = time.time()

    if progress_tracker is not None:
        progress_tracker.start_experiment(exp, experiment_index)

    for round_idx in range(1, rounds + 1):
        if progress_tracker is not None:
            progress_tracker.start_round(round_idx)
        round_start = time.time()
        global_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        download_bytes = sum(state_dict_nbytes(global_state) for _ in range(clients))
        client_states = []
        sample_counts = []
        local_losses = []
        upload_bytes = 0

        for client_id, loader in enumerate(client_loaders):
            local_state, sample_count, local_loss = train_local_model(
                model,
                loader,
                exp,
                cfg,
                device,
                progress_tracker=progress_tracker,
                round_idx=round_idx,
                client_id=client_id,
            )
            client_states.append(local_state)
            sample_counts.append(sample_count)
            local_losses.append(local_loss)
            upload_bytes += state_dict_nbytes(local_state)

        new_state = aggregate_fedavg(client_states, sample_counts)
        model.load_state_dict(new_state)
        val_metrics, _, _, _, _ = evaluate(model, val_loader, criterion, device, desc="val")

        round_comm_bytes = download_bytes + upload_bytes
        total_comm_bytes += round_comm_bytes
        is_best = val_metrics["macro_f1"] > best_val_macro_f1
        if is_best:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_round = round_idx
            torch.save(
                {
                    "run_id": run_id,
                    "model_name": model_name,
                    "algorithm": exp["algorithm"],
                    "partition": partition,
                    "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "classes": CLASSES,
                    "round": round_idx,
                    "val_metrics": val_metrics,
                    "config": exp,
                    "seed_metadata": seed_metadata,
                    "partition_file": project_relative(partition_path) if partition_path else None,
                    "partition_sha256": partition_hash,
                },
                best_checkpoint,
            )

        round_rows.append(
            {
                "run_id": run_id,
                "round": round_idx,
                "algorithm": exp["algorithm"],
                "model_name": model_name,
                "partition": partition,
                "avg_local_loss": float(np.mean(local_losses)),
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_weighted_f1": val_metrics["weighted_f1"],
                "download_bytes": download_bytes,
                "upload_bytes": upload_bytes,
                "round_comm_mb": round_comm_bytes / (1024**2),
                "cumulative_comm_mb": total_comm_bytes / (1024**2),
                "is_best": int(is_best),
                "round_time_sec": time.time() - round_start,
            }
        )
        pd.DataFrame(round_rows).to_csv(output_dir / f"metrics_per_round_{run_id}.csv", index=False)
        if progress_tracker is not None:
            progress_tracker.finish_round(round_idx, val_metrics["macro_f1"])

    elapsed_training_sec = time.time() - training_started
    checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=False)
    model = create_model(
        model_name,
        len(CLASSES),
        exp.get("trainable_layers", "all"),
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    test_metrics, y_true, y_pred, paths, probs = evaluate(model, test_loader, criterion, device, desc="test")

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASSES))),
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )
    write_json(output_dir / f"classification_report_{run_id}.json", report)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    cm_df = pd.DataFrame(cm, index=CLASSES, columns=CLASSES)
    cm_df.index.name = "true_label"
    confusion_path = output_dir / f"confusion_matrix_{run_id}.csv"
    report_path = output_dir / f"classification_report_{run_id}.json"
    predictions_path = output_dir / f"predictions_{run_id}.csv"
    history_path = output_dir / f"metrics_per_round_{run_id}.csv"
    cm_df.to_csv(confusion_path)
    save_predictions(
        predictions_path,
        run_id,
        model_name,
        experiment_seed,
        best_round,
        y_true,
        y_pred,
        paths,
        probs,
    )

    result_row = {
        "run_id": run_id,
        "algorithm": exp["algorithm"],
        "model_name": model_name,
        "partition": partition,
        "seed": experiment_seed,
        "partition_seed": partition_seed,
        "model_initialization_seed": model_initialization_seed,
        "dataloader_seed": dataloader_seed,
        "best_round": best_round,
        "best_val_macro_f1": best_val_macro_f1,
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_weighted_f1": test_metrics["weighted_f1"],
        "test_loss": test_metrics["loss"],
        "total_params": total_params,
        "trainable_params": trainable_params,
        "state_dict_mb": state_bytes / (1024**2),
        "total_comm_mb": total_comm_bytes / (1024**2),
        "checkpoint_path": str(best_checkpoint.relative_to(ROOT)),
        "partition_file": project_relative(partition_path) if partition_path else "generated_in_run",
        "elapsed_training_sec": elapsed_training_sec,
    }
    comm_row = {
        "run_id": run_id,
        "algorithm": exp["algorithm"],
        "model_name": model_name,
        "partition": partition,
        "seed": experiment_seed,
        "rounds": rounds,
        "clients": clients,
        "state_dict_bytes": state_bytes,
        "state_dict_mb": state_bytes / (1024**2),
        "total_download_mb": (state_bytes * clients * rounds) / (1024**2),
        "total_upload_mb": (state_bytes * clients * rounds) / (1024**2),
        "total_comm_mb": total_comm_bytes / (1024**2),
    }
    elapsed_total_sec = time.time() - run_started
    result_payload = {
        "schema_version": 1,
        "run_id": run_id,
        "algorithm": exp["algorithm"],
        "model_name": model_name,
        "partition": partition,
        "training_performed": True,
        "seed42_reused": False,
        "seed_metadata": seed_metadata,
        "partition_file": project_relative(partition_path) if partition_path else None,
        "partition_sha256": partition_hash,
        "manifests": manifest_metadata,
        "configuration_files": [project_relative(path) for path in cfg.get("_source_configs", [])],
        "configuration_snapshot": project_relative(output_dir / "config_used.json"),
        "software_environment": environment,
        "model": {
            "pretrained_initialization": exp.get("pretrained", "imagenet"),
            "centralized_checkpoint_used_for_initialization": False,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "full_state_bytes": state_bytes,
            "full_state_mib": state_bytes / (1024**2),
        },
        "federated": {
            "clients": clients,
            "rounds": rounds,
            "local_epochs": int(cfg["federated"]["local_epochs"]),
            "all_clients_per_round": True,
            "aggregation": cfg["federated"].get("aggregation", "weighted_fedavg_by_client_sample_count"),
        },
        "selection": {
            "metric": "validation_macro_f1",
            "best_round": best_round,
            "best_validation_macro_f1": best_val_macro_f1,
            "test_evaluated_after_selection": True,
        },
        "test_metrics": {
            "accuracy": test_metrics["accuracy"],
            "macro_f1": test_metrics["macro_f1"],
            "weighted_f1": test_metrics["weighted_f1"],
            "loss": test_metrics["loss"],
        },
        "communication": {
            "mode": "full_state_dict_upload_plus_download_per_round",
            "state_dict_bytes": state_bytes,
            "state_dict_mib": state_bytes / (1024**2),
            "total_download_mib": (state_bytes * clients * rounds) / (1024**2),
            "total_upload_mib": (state_bytes * clients * rounds) / (1024**2),
            "total_communication_mib": total_comm_bytes / (1024**2),
        },
        "timing": {
            "elapsed_training_sec": elapsed_training_sec,
            "elapsed_total_sec": elapsed_total_sec,
        },
        "artifacts": {
            "round_history": project_relative(history_path),
            "classification_report": project_relative(report_path),
            "confusion_matrix": project_relative(confusion_path),
            "predictions": project_relative(predictions_path),
            "best_checkpoint": project_relative(best_checkpoint),
            "environment": project_relative(output_dir / "environment.json"),
        },
        "provenance": {
            "source": "new_multiseed_run",
            "output_policy": "fail_if_output_directory_exists",
            "output_directory_created_new": True,
        },
    }
    write_json(output_dir / "result.json", result_payload)
    if progress_tracker is not None:
        progress_tracker.finish_experiment(test_metrics["macro_f1"], test_metrics["accuracy"])
    return result_row, comm_row


def write_summary(output_dir: Path) -> None:
    comparison_path = output_dir / "final_comparison_table.csv"
    comm_path = output_dir / "communication_costs.csv"
    if not comparison_path.exists() or not comm_path.exists():
        return
    comparison = pd.read_csv(comparison_path)
    communication = pd.read_csv(comm_path)
    best = comparison.sort_values(["test_macro_f1", "best_val_macro_f1"], ascending=False).iloc[0]
    comm_compare_lines: list[str] = []
    if {"efficientnet_b0", "resnet50"}.issubset(set(communication["model_name"])):
        eff = communication[communication["model_name"] == "efficientnet_b0"].iloc[0]
        res = communication[communication["model_name"] == "resnet50"].iloc[0]
        state_ratio = res["state_dict_mb"] / eff["state_dict_mb"]
        total_ratio = res["total_comm_mb"] / eff["total_comm_mb"]
        comm_compare_lines = [
            "## EfficientNet-B0 vs ResNet50 Communication",
            "",
            f"- ResNet50 full-state payload is {state_ratio:.3f}x EfficientNet-B0.",
            f"- ResNet50 total communication is {total_ratio:.3f}x EfficientNet-B0 for the same client/round schedule.",
            "",
        ]
    lines = [
        "# Federated Results Summary",
        "",
        "Generated from saved CSV/JSON outputs.",
        "",
        "## Best Test Result",
        "",
        f"- Run ID: `{best['run_id']}`",
        f"- Algorithm: `{best['algorithm']}`",
        f"- Model: `{best['model_name']}`",
        f"- Partition: `{best['partition']}`",
        f"- Best round by validation macro-F1: {int(best['best_round'])}",
        f"- Test accuracy: {best['test_accuracy']:.6f}",
        f"- Test macro-F1: {best['test_macro_f1']:.6f}",
        f"- Test weighted-F1: {best['test_weighted_f1']:.6f}",
        "",
        "## Final Comparison",
        "",
        markdown_table(comparison.sort_values(["test_macro_f1", "best_val_macro_f1"], ascending=False)),
        "",
        "## Communication Costs",
        "",
        markdown_table(communication.sort_values("total_comm_mb")),
        "",
        *comm_compare_lines,
        "## Notes",
        "",
        "- All train/validation/test samples are loaded from `cluster_split_outputs/*.csv` manifests.",
        "- The old folder split is not used as the experimental split.",
        "- Communication cost counts full `state_dict` tensor bytes for every upload and download.",
        "- Test metrics are computed once per run after selecting that run's best global round by validation macro-F1.",
    ]
    (output_dir / "FEDERATED_RESULTS_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def dry_run(
    cfg: dict[str, Any],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    split_map: dict[str, pd.DataFrame],
    device: torch.device,
) -> None:
    print("Dry run checks only. No training will run.")
    print(pd.concat([class_count_table(train_df, "train"), class_count_table(val_df, "val"), class_count_table(test_df, "test")]))
    for partition, split_df in split_map.items():
        print(f"\nClient counts: {partition}")
        print(client_count_table(split_df))

    _, eval_tf = make_transforms(int(cfg["training"]["image_size"]))
    sample_loader = make_loader(
        train_df.head(int(cfg["training"]["batch_size"])),
        eval_tf,
        int(cfg["training"]["batch_size"]),
        False,
        0,
        int(cfg["training"]["seed"]),
        device,
    )
    images, _, _ = next(iter(sample_loader))
    images = images.to(device)
    with torch.no_grad():
        for exp in cfg["experiments"]:
            seed_everything(int(exp.get("model_initialization_seed", cfg["training"]["seed"])))
            model = create_model(
                exp["model_name"],
                len(CLASSES),
                exp.get("trainable_layers", "all"),
                exp.get("pretrained", "imagenet"),
            ).to(device)
            model.eval()
            total_params, trainable_params, state_bytes = parameter_counts(model)
            outputs = model(images)
            print(
                {
                    "run_id": exp["run_id"],
                    "model_name": exp["model_name"],
                    "output_shape": list(outputs.shape),
                    "total_params": total_params,
                    "trainable_params": trainable_params,
                    "state_dict_mb": state_bytes / (1024**2),
                }
            )


def validate_focused_configuration(cfg: dict[str, Any]) -> None:
    """Reject accidental drift in strict multi-seed experiment configs."""
    if not cfg.get("reproducibility", {}).get("strict_multiseed", False):
        return
    expected = {
        "clients": 4,
        "rounds": 20,
        "local_epochs": 3,
        "batch_size": 32,
        "optimizer": "AdamW",
        "learning_rate": 0.0003,
        "weight_decay": 0.0001,
        "noniid_dominant_fraction": 0.70,
    }
    observed = {
        "clients": int(cfg["federated"]["clients"]),
        "rounds": int(cfg["federated"]["rounds"]),
        "local_epochs": int(cfg["federated"]["local_epochs"]),
        "batch_size": int(cfg["training"]["batch_size"]),
        "optimizer": cfg["training"].get("optimizer"),
        "learning_rate": float(cfg["training"]["learning_rate"]),
        "weight_decay": float(cfg["training"]["weight_decay"]),
        "noniid_dominant_fraction": float(cfg["split"]["noniid_dominant_fraction"]),
    }
    mismatches = {key: (observed[key], value) for key, value in expected.items() if observed[key] != value}
    if mismatches:
        raise ValueError(f"Strict multi-seed settings changed: {mismatches}")
    if len(cfg["experiments"]) != 1:
        raise ValueError("Each strict multi-seed config must contain exactly one experiment.")
    exp = cfg["experiments"][0]
    if exp.get("algorithm", "").lower() != "fedavg" or exp.get("partition") != "noniid":
        raise ValueError("Strict multi-seed experiment must use FedAvg and the Non-IID partition.")
    if exp.get("model_name") not in {"efficientnet_b0", "resnet50"}:
        raise ValueError("Strict multi-seed experiment must use EfficientNet-B0 or ResNet50.")
    if str(exp.get("pretrained", "")).lower() != "imagenet":
        raise ValueError("Strict multi-seed experiment must start from ImageNet-pretrained weights.")
    if exp.get("centralized_checkpoint_initialization") is not False:
        raise ValueError("Centralized checkpoint initialization must remain disabled.")
    if float(exp.get("learning_rate", -1)) != 0.0003:
        raise ValueError("Experiment learning rate must remain 0.0003.")
    if not cfg["split"].get("partition_file"):
        raise ValueError("Strict multi-seed experiment must load a saved partition_file.")
    seed = int(cfg["training"]["seed"])
    if int(cfg["split"].get("partition_seed", -1)) != seed:
        raise ValueError("partition_seed must equal experiment seed.")
    if any(
        int(exp.get(key, -1)) != seed
        for key in ("seed", "model_initialization_seed", "dataloader_seed")
    ):
        raise ValueError("Experiment, model-initialization, and DataLoader seeds must match.")
    if float(cfg["federated"].get("participation_fraction", -1)) != 1.0:
        raise ValueError("All four clients must participate in every round.")
    if cfg["federated"].get("aggregation") != "weighted_fedavg_by_client_sample_count":
        raise ValueError("Aggregation must remain sample-weighted FedAvg.")
    expected_manifests = {
        "train": "cluster_split_outputs/train_manifest.csv",
        "val": "cluster_split_outputs/val_manifest.csv",
        "test": "cluster_split_outputs/test_manifest.csv",
    }
    if cfg.get("manifests") != expected_manifests:
        raise ValueError("Strict multi-seed configs must use the fixed train/validation/test manifests.")


def create_new_output_directory(path: Path) -> None:
    if path.exists():
        raise FileExistsError(
            f"Output directory already exists; refusing to overwrite any result: {path}"
        )
    path.mkdir(parents=True, exist_ok=False)
    (path / "checkpoints").mkdir(exist_ok=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Federated benchmark on cluster-aware manifests.")
    parser.add_argument("--config", nargs="+", type=Path, required=True, help="One or more YAML config files.")
    parser.add_argument(
        "--experiment",
        action="append",
        default=[],
        help="Run only the named run_id; may be repeated. Omit to run every config entry.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate data/model setup without training.")
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate config, manifests, and saved partitions without loading images or models.",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()

    cfg = merge_configs(args.config)
    if args.experiment:
        requested = set(args.experiment)
        available = {exp["run_id"] for exp in cfg["experiments"]}
        missing = requested - available
        if missing:
            raise ValueError(f"Requested experiments not found in config: {sorted(missing)}")
        cfg["experiments"] = [exp for exp in cfg["experiments"] if exp["run_id"] in requested]
    training_defaults = {
        "seed": 42,
        "image_size": 224,
        "batch_size": 32,
        "num_workers": 2,
        "learning_rate": 0.0003,
        "weight_decay": 1e-4,
    }
    cfg["training"] = {**training_defaults, **cfg.get("training", {})}
    cfg["federated"] = {
        "clients": 4,
        "rounds": 10,
        "local_epochs": 3,
        **cfg.get("federated", {}),
    }
    cfg["split"] = {"noniid_dominant_fraction": 0.70, **cfg.get("split", {})}
    seed_everything(int(cfg["training"]["seed"]))
    validate_focused_configuration(cfg)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "--device cuda requested, but torch.cuda.is_available() is False in this environment."
            )
        device = torch.device(args.device)

    manifests = cfg["manifests"]
    train_df = load_manifest(resolve_path(manifests["train"]), "train")
    val_df = load_manifest(resolve_path(manifests["val"]), "val")
    test_df = load_manifest(resolve_path(manifests["test"]), "test")

    clients = int(cfg["federated"]["clients"])
    seed = int(cfg["training"]["seed"])
    dominant_fraction = float(cfg["split"]["noniid_dominant_fraction"])
    required_partitions = {exp["partition"] for exp in cfg["experiments"]}
    split_map: dict[str, pd.DataFrame] = {}
    if "iid" in required_partitions:
        split_map["iid"] = build_iid_split(train_df, clients, seed)
    if "noniid" in required_partitions:
        partition_file = cfg["split"].get("partition_file")
        if partition_file:
            split_map["noniid"] = load_saved_client_partition(
                resolve_path(partition_file),
                train_df,
                clients=clients,
                seed=int(cfg["split"].get("partition_seed", seed)),
                dominant_fraction=dominant_fraction,
            )
        else:
            split_map["noniid"] = build_noniid_split(
                train_df,
                clients,
                seed,
                dominant_fraction,
            )

    if args.validate_config:
        print("Configuration validation only. No model loading or training will run.")
        print(
            {
                "experiments": [exp["run_id"] for exp in cfg["experiments"]],
                "output_dir": project_relative(resolve_path(cfg.get("output_dir", "federated_results"))),
                "train_samples": len(train_df),
                "validation_samples": len(val_df),
                "test_samples": len(test_df),
                "partitions": sorted(split_map),
            }
        )
        for partition_name, split_df in split_map.items():
            print(partition_name)
            print(client_count_table(split_df))
        return

    if args.dry_run:
        dry_run(cfg, train_df, val_df, test_df, split_map, device)
        return

    output_dir = resolve_path(cfg.get("output_dir", "federated_results"))
    create_new_output_directory(output_dir)
    write_json(output_dir / "config_used.json", cfg)
    write_json(output_dir / "environment.json", collect_environment_metadata(device))
    partition_file = cfg["split"].get("partition_file")
    if partition_file:
        resolved_partition = resolve_path(partition_file).resolve()
        write_json(
            output_dir / "partition_reference.json",
            {
                "path": project_relative(resolved_partition),
                "sha256": sha256_file(resolved_partition),
                "seed": int(cfg["split"].get("partition_seed", seed)),
                "partition_type": "noniid",
            },
        )
    else:
        for partition_name, split_df in split_map.items():
            save_client_split(split_df, output_dir / f"client_splits_{partition_name}.csv")
    progress_tracker = ProgressTracker(output_dir, cfg)

    for experiment_index, exp in enumerate(cfg["experiments"], start=1):
        result_row, comm_row = run_experiment(
            exp,
            experiment_index,
            cfg,
            split_map,
            val_df,
            test_df,
            output_dir,
            device,
            progress_tracker=progress_tracker,
        )
        upsert_rows_by_run_id(output_dir / "final_comparison_table.csv", [result_row])
        comm_df = upsert_rows_by_run_id(output_dir / "communication_costs.csv", [comm_row])
        eff_rows = comm_df[comm_df["model_name"] == "efficientnet_b0"]
        if not eff_rows.empty:
            eff_state_mb = eff_rows.iloc[0]["state_dict_mb"]
            eff_total_mb = eff_rows.iloc[0]["total_comm_mb"]
            comm_df["state_dict_ratio_vs_efficientnet_b0"] = comm_df["state_dict_mb"] / eff_state_mb
            comm_df["total_comm_ratio_vs_efficientnet_b0"] = comm_df["total_comm_mb"] / eff_total_mb
        comm_df.to_csv(output_dir / "communication_costs.csv", index=False)
        write_summary(output_dir)

    progress_tracker.finish_run()
    write_summary(output_dir)


if __name__ == "__main__":
    main()
