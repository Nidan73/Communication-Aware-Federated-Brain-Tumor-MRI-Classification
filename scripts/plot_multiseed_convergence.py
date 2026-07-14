#!/usr/bin/env python3
"""Plot three-seed validation macro-F1 histories for the paper."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 123, 2025)
ROUNDS = np.arange(1, 21)
MODELS = {
    "EfficientNet-B0": "efficientnet_b0",
    "ResNet50": "resnet50",
}
COLORS = {
    "EfficientNet-B0": "#1f77b4",
    "ResNet50": "#d95f02",
}


def history_path(model_name: str, seed: int) -> Path:
    run_id = f"fedavg_{model_name}_noniid_seed{seed}"
    return ROOT / "reproducibility" / "results" / run_id / f"metrics_per_round_{run_id}.csv"


def load_history(model_name: str, seed: int) -> np.ndarray:
    path = history_path(model_name, seed)
    if not path.is_file():
        raise FileNotFoundError(f"Missing history: {path}")
    frame = pd.read_csv(path)
    required = {"round", "val_macro_f1"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    rounds = pd.to_numeric(frame["round"], errors="raise").astype(int)
    if len(frame) != 20 or not rounds.is_unique or set(rounds) != set(ROUNDS):
        raise ValueError(f"{path} must contain each communication round 1 through 20 exactly once.")
    ordered = frame.assign(round=rounds).sort_values("round")
    values = pd.to_numeric(ordered["val_macro_f1"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError(f"{path} contains invalid validation macro-F1 values.")
    return values


def main() -> None:
    histories: dict[str, np.ndarray] = {}
    for label, model_name in MODELS.items():
        histories[label] = np.vstack([load_history(model_name, seed) for seed in SEEDS])

    fig, ax = plt.subplots(figsize=(7.15, 3.35), constrained_layout=True)
    for label, values in histories.items():
        mean = values.mean(axis=0)
        sample_sd = values.std(axis=0, ddof=1)
        color = COLORS[label]
        ax.plot(
            ROUNDS,
            mean,
            color=color,
            linewidth=1.8,
            marker="o",
            markersize=3.0,
            label=label,
        )
        ax.fill_between(
            ROUNDS,
            np.clip(mean - sample_sd, 0.90, 1.0),
            np.clip(mean + sample_sd, 0.90, 1.0),
            color=color,
            alpha=0.18,
            linewidth=0,
        )

    ax.set_xlim(1, 20)
    ax.set_ylim(0.90, 1.0)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_yticks(np.linspace(0.90, 1.0, 6))
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Validation macro-F1")
    ax.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.8)
    ax.legend(loc="lower right", frameon=True, fontsize=8.5)
    ax.tick_params(labelsize=8.5)

    output_paths = (
        ROOT / "paper_assets_final" / "fig_multiseed_convergence.pdf",
        ROOT / "paper_assets_final" / "fig_multiseed_convergence.png",
        ROOT / "paper_anonymous_submission" / "figures" / "fig_multiseed_convergence.pdf",
        ROOT / "paper_anonymous_submission" / "figures" / "fig_multiseed_convergence.png",
    )
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".pdf":
            fig.savefig(
                path,
                format="pdf",
                bbox_inches="tight",
                metadata={"Creator": "", "Producer": "", "CreationDate": None},
            )
        else:
            fig.savefig(path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    for path in output_paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Figure output was not created: {path}")
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
