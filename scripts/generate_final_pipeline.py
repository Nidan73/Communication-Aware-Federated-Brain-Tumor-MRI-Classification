#!/usr/bin/env python3
"""Generate the final editable vector workflow figure for the IEEE paper."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_anonymous_submission" / "figures"
PDF_PATH = OUT / "pipeline_overview_final.pdf"
SVG_PATH = OUT / "pipeline_overview_final.svg"
PNG_PATH = OUT / "pipeline_overview_final.png"
PREVIEW_PATH = OUT / "pipeline_overview_preview.png"

CANVAS_IN = (15.0, 4.6)
PAPER_WIDTH_IN = 7.16
MAIN_DPI = 300
PREVIEW_DPI = 300

VISIBLE_TEXT: list[str] = []

INK = "#17212b"
MUTED = "#52606d"
BLUE = "#2c5f86"
TEAL = "#236b69"
GREEN = "#3f7350"
GRAY = "#68737d"
PALE = ("#eef4f8", "#f3f5f6", "#eef5f2", "#f5f2ea")


def rounded(ax, x, y, w, h, *, face, edge=INK, radius=0.55, lw=1.15, z=1):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.18,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def label(ax, x, y, text, *, size=16.0, weight="normal", color=INK, ha="center", va="center", z=5):
    VISIBLE_TEXT.append(text)
    return ax.text(
        x,
        y,
        text,
        fontsize=size,
        fontweight=weight,
        color=color,
        ha=ha,
        va=va,
        zorder=z,
        linespacing=1.08,
    )


def arrow(ax, start, end, *, color=GRAY, lw=1.35, scale=13, both=False, z=3):
    style = "<|-|>" if both else "-|>"
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=scale,
        linewidth=lw,
        color=color,
        shrinkA=1.5,
        shrinkB=1.5,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def image_stack_icon(ax, x, y, scale=1.0):
    for dx, dy in ((0.0, 0.0), (0.38, 0.34), (0.76, 0.68)):
        ax.add_patch(Rectangle((x + dx, y + dy), 3.0 * scale, 2.15 * scale, facecolor="white", edgecolor=BLUE, lw=1.0, zorder=4))
    ax.add_patch(Polygon([(x + 1.0, y + 1.0), (x + 1.7, y + 1.65), (x + 2.35, y + 0.9)], closed=False, fill=False, edgecolor=BLUE, lw=1.0, zorder=5))
    ax.add_patch(Circle((x + 2.35, y + 1.72), 0.22, facecolor=BLUE, edgecolor="none", zorder=5))


def duplicate_icon(ax, x, y):
    ax.add_patch(Rectangle((x, y), 2.8, 2.0, facecolor="white", edgecolor=TEAL, lw=1.0, zorder=4))
    ax.add_patch(Rectangle((x + 0.55, y + 0.48), 2.8, 2.0, facecolor="#f8fbfa", edgecolor=TEAL, lw=1.0, zorder=4))
    ax.add_patch(Circle((x + 2.95, y + 0.45), 0.72, facecolor="none", edgecolor=TEAL, lw=1.25, zorder=5))
    ax.plot([x + 3.45, x + 4.05], [y - 0.05, y - 0.65], color=TEAL, lw=1.35, zorder=5)


def check_icon(ax, x, y):
    ax.add_patch(Circle((x, y), 1.35, facecolor="#e6f2ea", edgecolor=GREEN, lw=1.2, zorder=4))
    ax.plot([x - 0.62, x - 0.12, x + 0.75], [y, y - 0.48, y + 0.58], color=GREEN, lw=2.1, solid_capstyle="round", zorder=5)


def server_icon(ax, x, y):
    rounded(ax, x - 2.7, y - 1.9, 5.4, 3.8, face="#e7f0f4", edge=BLUE, radius=0.45, lw=1.25, z=4)
    for offset in (0.72, 0.0, -0.72):
        ax.plot([x - 1.8, x + 1.25], [y + offset, y + offset], color=BLUE, lw=1.0, zorder=5)
        ax.add_patch(Circle((x + 1.8, y + offset), 0.16, facecolor=TEAL, edgecolor="none", zorder=5))
    label(ax, x, y - 2.75, "Server", size=15.5, weight="bold")


def client_icon(ax, x, y, name):
    rounded(ax, x - 3.0, y - 1.25, 6.0, 2.5, face="white", edge=GRAY, radius=0.4, lw=1.0, z=4)
    label(ax, x, y, name, size=15.0, weight="bold")


def stage_one(ax, x, y, w, h):
    rounded(ax, x, y, w, h, face=PALE[0])
    label(ax, x + 1.2, y + h - 1.25, "1. Duplicate-Controlled\nData", size=16.5, weight="bold", ha="left", va="top")
    image_stack_icon(ax, x + 1.25, y + h - 7.1, scale=0.85)
    label(ax, x + 12.4, y + h - 5.65, "7,200 MRI images", size=15.8, weight="bold")
    arrow(ax, (x + 12.4, y + h - 6.55), (x + 12.4, y + h - 7.5), color=BLUE, scale=11)
    duplicate_icon(ax, x + 1.35, y + h - 11.6)
    label(ax, x + 13.8, y + h - 8.65, "Duplicate grouping", size=15.5, weight="bold")
    arrow(ax, (x + 13.8, y + h - 9.55), (x + 13.8, y + h - 10.45), color=BLUE, scale=11)
    label(ax, x + 13.8, y + h - 11.55, "6,116 image clusters", size=15.5, weight="bold")
    arrow(ax, (x + 13.8, y + h - 12.45), (x + 13.8, y + h - 13.35), color=BLUE, scale=11)
    label(ax, x + 13.8, y + h - 14.45, "Cluster-aware split", size=15.5, weight="bold")

    box_y = y + 5.6
    for bx, title, value in (
        (x + 1.05, "Train", "5,040"),
        (x + 8.95, "Val.", "720"),
        (x + 16.85, "Test", "1,440"),
    ):
        rounded(ax, bx, box_y, 6.2, 4.75, face="white", edge=BLUE, radius=0.38, lw=0.95, z=3)
        label(ax, bx + 3.1, box_y + 3.25, title, size=14.2, weight="bold")
        label(ax, bx + 3.1, box_y + 0.72, value, size=16.0, weight="bold", color=BLUE)
    label(ax, x + w / 2, y + 4.25, "4 balanced classes", size=15.0, weight="bold", color=BLUE)
    label(ax, x + w / 2, y + 2.65, "Exact hash  •  Pixel duplicate", size=14.5, color=MUTED)
    label(ax, x + w / 2, y + 1.35, "Same-class near-duplicate", size=14.5, color=MUTED)


def stage_two(ax, x, y, w, h):
    rounded(ax, x, y, w, h, face=PALE[1])
    label(ax, x + 1.2, y + h - 1.25, "2. Backbone\nScreening", size=16.5, weight="bold", ha="left", va="top")
    names = ("ResNet50", "DenseNet121", "EfficientNet-B0", "EfficientNet-B3", "ConvNeXt-Tiny")
    first_y = y + h - 7.15
    for index, name in enumerate(names):
        by = first_y - index * 2.75
        rounded(ax, x + 3.0, by, w - 6.0, 2.18, face="white", edge=GRAY, radius=0.32, lw=0.85, z=3)
        label(ax, x + w / 2, by + 1.09, name, size=15.2, weight="bold")
    arrow(ax, (x + w / 2, y + 8.25), (x + w / 2, y + 7.75), color=TEAL, lw=1.3, scale=10)
    rounded(ax, x + 2.6, y + 0.3, w - 5.2, 7.3, face="#e5f1ec", edge=GREEN, radius=0.5, lw=1.25, z=3)
    check_icon(ax, x + 5.0, y + 4.45)
    label(ax, x + 7.2, y + 5.75, "Selected", size=15.0, weight="bold", ha="left", color=GREEN)
    label(ax, x + 7.2, y + 4.15, "EfficientNet-B0", size=15.5, weight="bold", ha="left", color=GREEN)
    label(ax, x + w / 2, y + 2.75, "Best validation macro-F1", size=13.8)
    label(ax, x + w / 2, y + 1.25, "0.9903", size=15.0, weight="bold", color=GREEN)


def stage_three(ax, x, y, w, h):
    rounded(ax, x, y, w, h, face=PALE[2])
    label(ax, x + 1.2, y + h - 1.25, "3. Simulated Federated\nLearning", size=16.2, weight="bold", ha="left", va="top")
    cx, cy = x + w / 2, y + 15.2
    positions = (
        (x + 4.15, y + 20.2, "Client 1"),
        (x + w - 4.15, y + 20.2, "Client 2"),
        (x + 4.15, y + 10.5, "Client 3"),
        (x + w - 4.15, y + 10.5, "Client 4"),
    )
    server_icon(ax, cx, cy)
    for px, py, name in positions:
        arrow(ax, (cx, cy), (px, py), color=TEAL, lw=1.15, scale=11, both=True, z=2)
        client_icon(ax, px, py, name)
    label(ax, cx, y + 6.55, "4 virtual clients", size=15.5, weight="bold", color=TEAL)
    label(ax, cx, y + 4.95, "IID  •  70/30 Non-IID", size=14.8)
    label(ax, cx, y + 3.35, "FedAvg  •  FedProx", size=14.8, color=MUTED)
    label(ax, cx, y + 1.75, "20 rounds  •  3 local epochs", size=14.8, color=MUTED)


def stage_four(ax, x, y, w, h):
    rounded(ax, x, y, w, h, face=PALE[3])
    label(ax, x + 1.2, y + h - 1.65, "4. Accuracy & Payload", size=16.2, weight="bold", ha="left")
    label(ax, x + w / 2, y + h - 3.8, "Seeds 42, 123, 2025", size=15.2, weight="bold")
    label(ax, x + w / 2, y + h - 5.25, "Non-IID accuracy", size=14.5, color=MUTED)

    rounded(ax, x + 1.45, y + h - 9.7, w - 2.9, 3.75, face="white", edge=BLUE, radius=0.4, lw=1.0, z=3)
    label(ax, x + w / 2, y + h - 7.05, "EfficientNet-B0", size=15.0, weight="bold")
    label(ax, x + w / 2, y + h - 8.5, "97.25 ± 0.51%", size=15.8, weight="bold", color=BLUE)
    rounded(ax, x + 1.45, y + h - 13.9, w - 2.9, 3.75, face="white", edge=GRAY, radius=0.4, lw=1.0, z=3)
    label(ax, x + w / 2, y + h - 11.25, "ResNet50", size=15.0, weight="bold")
    label(ax, x + w / 2, y + h - 12.7, "97.18 ± 0.28%", size=15.8, weight="bold", color=GRAY)

    rounded(ax, x + 2.0, y + 9.4, w - 4.0, 2.7, face="#e5f1ec", edge=GREEN, radius=0.42, lw=1.1, z=3)
    label(ax, x + w / 2, y + 11.3, "82.8% lower payload", size=15.5, weight="bold", color=GREEN)
    label(ax, x + w / 2, y + 10.15, "per 20-round run", size=13.8, color=MUTED)

    bar_x = x + 1.8
    full_bar = w - 3.6
    compact_bar = full_bar * (2474.84 / 14385.65)
    label(ax, x + w / 2, y + 8.45, "EfficientNet-B0", size=13.6, weight="bold")
    ax.add_patch(Rectangle((bar_x, y + 7.55), compact_bar, 0.55, facecolor=GREEN, edgecolor=INK, lw=0.75, zorder=3))
    label(ax, x + w / 2, y + 6.85, "2,474.84 MiB", size=13.6, weight="bold")
    label(ax, x + w / 2, y + 5.7, "ResNet50", size=13.6, weight="bold")
    ax.add_patch(Rectangle((bar_x, y + 4.8), full_bar, 0.55, facecolor="#8a939a", edgecolor=INK, lw=0.75, zorder=3))
    label(ax, x + w / 2, y + 4.05, "14,385.65 MiB", size=13.6, weight="bold")


def verify_visible_content() -> None:
    joined = "\n".join(VISIBLE_TEXT)
    required = (
        "7,200",
        "6,116",
        "5,040",
        "720",
        "1,440",
        "EfficientNet-B0",
        "97.25 ± 0.51%",
        "97.18 ± 0.28%",
        "82.8%",
        "2,474.84 MiB",
        "14,385.65 MiB",
    )
    missing = [item for item in required if item not in joined]
    if missing:
        raise RuntimeError(f"Pipeline labels missing: {missing}")
    clients = re.findall(r"\bClient [1-4]\b", joined)
    if clients != ["Client 1", "Client 2", "Client 3", "Client 4"]:
        raise RuntimeError(f"Expected exactly four ordered client labels, found {clients}")
    forbidden = (
        "97.64",
        "97.22",
        "Nidan Alam",
        "Abu Jafar Sisty",
        "AIUB",
        "student.aiub.edu",
        "/home/",
        "GitHub",
    )
    found = [item for item in forbidden if item.lower() in joined.lower()]
    if found:
        raise RuntimeError(f"Forbidden pipeline text found: {found}")
    short_labels = {
        "Train", "Val.", "Test", "ResNet50", "DenseNet121", "EfficientNet-B0",
        "EfficientNet-B3", "ConvNeXt-Tiny", "Client 1", "Client 2", "Client 3", "Client 4",
        "Server", "IID  •  70/30 Non-IID", "FedAvg  •  FedProx",
    }
    prose = " ".join(text for text in VISIBLE_TEXT if text not in short_labels)
    words = [word for word in re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)*", prose) if word.lower() not in {"mib"}]
    if not 45 <= len(words) <= 55:
        raise RuntimeError(f"Pipeline prose word count outside target range: {len(words)}")
    print(f"verified_prose_word_count={len(words)}")


def verify_outputs() -> None:
    for path in (PDF_PATH, SVG_PATH, PNG_PATH, PREVIEW_PATH):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing pipeline output: {path}")
    root = ET.parse(SVG_PATH).getroot()
    svg_text = " ".join("".join(node.itertext()) for node in root.iter() if node.tag.endswith("text"))
    if "Client 1" not in svg_text or "82.8% lower" not in svg_text:
        raise RuntimeError("SVG does not preserve expected editable text nodes")
    with Image.open(PNG_PATH) as image:
        if image.width < 3000:
            raise RuntimeError(f"Pipeline PNG is too narrow: {image.width}px")
    with Image.open(PREVIEW_PATH) as image:
        expected = round(PAPER_WIDTH_IN * PREVIEW_DPI)
        if abs(image.width - expected) > 2:
            raise RuntimeError(f"Unexpected preview width: {image.width}px")
    svg_source = SVG_PATH.read_text(encoding="utf-8")
    if "/home/" in svg_source or "nidan" in svg_source.lower():
        raise RuntimeError("Unsafe local metadata found in SVG")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=CANVAS_IN)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 32)
    ax.axis("off")

    y, h = 4.6, 26.4
    xs = (0.7, 25.5, 50.3, 75.1)
    w = 24.1
    stage_one(ax, xs[0], y, w, h)
    stage_two(ax, xs[1], y, w, h)
    stage_three(ax, xs[2], y, w, h)
    stage_four(ax, xs[3], y, w, h)
    for left, right in zip(xs[:-1], xs[1:]):
        arrow(ax, (left + w + 0.05, 17.6), (right - 0.05, 17.6), color=INK, lw=1.5, scale=14, z=6)

    rounded(ax, 0.7, 0.55, 98.5, 2.75, face="#e8eef1", edge=INK, radius=0.5, lw=1.0, z=1)
    label(
        ax,
        50,
        1.93,
        "Comparable three-seed performance with 82.8% lower communication payload",
        size=18.0,
        weight="bold",
    )

    verify_visible_content()
    fig.subplots_adjust(left=0.006, right=0.994, bottom=0.015, top=0.99)
    safe_title = "Experimental workflow"
    fig.savefig(
        PDF_PATH,
        format="pdf",
        metadata={
            "Title": safe_title,
            "Author": "",
            "Subject": "",
            "Keywords": "",
            "Creator": "",
            "Producer": "",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(
        SVG_PATH,
        format="svg",
        metadata={"Title": safe_title, "Creator": "", "Description": "Four-stage experimental workflow", "Date": None},
    )
    fig.savefig(
        PNG_PATH,
        format="png",
        dpi=MAIN_DPI,
        metadata={"Software": None, "Title": safe_title, "Author": "", "Description": "Four-stage experimental workflow"},
    )
    plt.close(fig)

    with Image.open(PNG_PATH) as image:
        preview_width = round(PAPER_WIDTH_IN * PREVIEW_DPI)
        preview_height = round(preview_width * image.height / image.width)
        preview = image.resize((preview_width, preview_height), Image.Resampling.LANCZOS)
        preview.save(PREVIEW_PATH, format="PNG", dpi=(PREVIEW_DPI, PREVIEW_DPI), optimize=True)

    verify_outputs()
    with Image.open(PNG_PATH) as image:
        print(f"pipeline_png={image.width}x{image.height}")
    with Image.open(PREVIEW_PATH) as image:
        print(f"pipeline_preview={image.width}x{image.height}")
    print(f"pipeline_canvas_inches={CANVAS_IN[0]}x{CANVAS_IN[1]}")
    print("verified_clients=4")
    print("verified_required_labels=yes")
    print("verified_forbidden_labels_absent=yes")
    for path in (PDF_PATH, SVG_PATH, PNG_PATH, PREVIEW_PATH):
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
