#!/usr/bin/env python3
"""Shared constants and small helpers for multi-seed reproducibility tools."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("glioma", "meningioma", "notumor", "pituitary")
SEEDS = (42, 123, 2025)
MODELS = ("efficientnet_b0", "resnet50")
MODEL_LABELS = {
    "efficientnet_b0": "EfficientNet-B0",
    "resnet50": "ResNet50",
}
PROBABILITY_COLUMNS = {
    "glioma": "probability_glioma",
    "meningioma": "probability_meningioma",
    "notumor": "probability_no_tumor",
    "pituitary": "probability_pituitary",
}
EXPECTED_TRAIN_SAMPLES = 5_040
EXPECTED_TEST_SAMPLES = 1_440
EXPECTED_CLIENTS = 4
EXPECTED_CLIENT_SAMPLES = 1_260
EXPECTED_DOMINANT_SAMPLES = 882
EXPECTED_OTHER_CLASS_SAMPLES = 126


def run_id(model_name: str, seed: int) -> str:
    if model_name not in MODELS:
        raise ValueError(f"Unsupported model: {model_name}")
    if seed not in SEEDS:
        raise ValueError(f"Unsupported seed: {seed}")
    return f"fedavg_{model_name}_noniid_seed{seed}"


def stable_sample_id(image_path: str) -> str:
    """Return an order-independent ID derived only from normalized image path."""
    normalized = Path(str(image_path).replace("\\", "/")).as_posix()
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_project_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def project_relative(path_like: str | Path) -> str:
    path = resolve_project_path(path_like).resolve()
    try:
        return str(path.relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def result_directory(model_name: str, seed: int) -> Path:
    return ROOT / "reproducibility" / "results" / run_id(model_name, seed)

