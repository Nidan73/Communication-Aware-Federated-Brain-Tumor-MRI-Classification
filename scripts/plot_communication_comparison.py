#!/usr/bin/env python3
"""Regenerate the paper communication chart with explicit MiB units."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper_assets_final" / "table_main_communication_comparison.csv"
OUTPUTS = (
    ROOT / "paper_assets_final" / "fig_main_communication_comparison.pdf",
    ROOT / "paper_assets_final" / "fig_main_communication_comparison.png",
    ROOT / "paper_anonymous_submission" / "figures" / "fig_main_communication_comparison.pdf",
    ROOT / "paper_anonymous_submission" / "figures" / "fig_main_communication_comparison.png",
)


def load_values() -> tuple[list[str], list[float]]:
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected = {"EfficientNet-B0": 2474.84, "ResNet50": 14385.65}
    values = {row["Backbone"]: float(row["Full-state payload MiB/run"]) for row in rows}
    if values != expected:
        raise ValueError(f"Communication CSV values changed: {values!r}")
    labels = ["EfficientNet-B0", "ResNet50"]
    return labels, [values[label] for label in labels]


def main() -> None:
    labels, values = load_values()
    fig, ax = plt.subplots(figsize=(5.15, 3.35), constrained_layout=True)
    bars = ax.bar(labels, values, color=["#4c9f50", "#888888"], edgecolor="black", linewidth=0.5)
    ax.set_title("Full-State Payload per 20-Round Non-IID Run", fontsize=10)
    ax.set_ylabel("Full-state payload (MiB)")
    ax.set_ylim(0, 16000)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 350,
            f"{value:,.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    for path in OUTPUTS:
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

    for path in OUTPUTS:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Figure output was not created: {path}")
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
