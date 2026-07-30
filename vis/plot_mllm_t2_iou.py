#!/usr/bin/env python3
"""Plot MLLM T2 localization IoU on the frozen ClaimForge forged panel.

The plotted values use all 750 forged images in
``annotations/claimforge_mllm_benchmark1000_v2.jsonl``: 250 Mouse, 250 Cat,
and 250 Trash-can images. Missing or invalid units contribute zero IoU so
every model uses the same denominator.

Protocol provenance:
  * Formal strict-scope aggregation:
    ``results/mllm/balanced250_local1000_v2/*/localization_metrics.json``.
  * Doubao uses its normalized bbox_1000 protocol; boxes are converted to
    image pixels before rasterization and scoring.

Run from the repository root:

    python vis/plot_mllm_t2_iou.py

By default, the script writes PDF and PNG files next to itself.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt


MODELS = (
    "Qwen 3.7\nPlus",
    "GPT-5.6\nLuna",
    "Claude Opus\n4.8",
    "Doubao Seed\n2.1 Pro",
)

# Macro pixel IoU in percent. Bounding-box unions are rasterized and compared
# with the exact nonzero RGB difference between each source/forged PNG pair.
# Values are computed over the fixed 750-image panel, including zero for any
# missing/invalid unit (Qwen: one Cat; GPT: one Mouse).
SERIES = (
    ("Mouse", (4.6298, 8.4632, 15.0374, 8.9530), "#4477AA", ""),
    ("Cat", (0.8270, 1.5394, 30.8778, 3.5364), "#EE6677", ""),
    ("Trash-can", (0.0957, 0.1172, 1.3162, 0.3685), "#228833", ""),
    ("Overall", (1.8509, 3.3733, 15.7438, 4.2860), "#AA3377", "///"),
)


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def make_figure() -> plt.Figure:
    """Create the paper-ready grouped bar chart."""
    _configure_style()
    fig, ax = plt.subplots(figsize=(7.1, 3.2))

    group_centers = list(range(len(MODELS)))
    bar_width = 0.19
    offsets = (-1.5, -0.5, 0.5, 1.5)

    for offset, (label, values, color, hatch) in zip(offsets, SERIES):
        positions = [center + offset * bar_width for center in group_centers]
        bars = ax.bar(
            positions,
            values,
            width=bar_width,
            label=label,
            color=color,
            edgecolor="#333333",
            linewidth=0.45,
            hatch=hatch,
            zorder=3,
        )
        ax.bar_label(
            bars,
            labels=[f"{value:.2f}" for value in values],
            padding=2,
            fontsize=6.8,
            rotation=90,
        )

    ax.set_ylabel("Macro pixel IoU (%)")
    ax.set_xticks(group_centers, MODELS)
    ax.set_ylim(0, 35)
    ax.set_yticks(range(0, 36, 5))
    ax.grid(axis="y", color="#D6D6D6", linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=6)
    ax.tick_params(axis="y", width=0.7, length=3)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=4,
        frameon=False,
        columnspacing=1.5,
        handlelength=1.7,
    )

    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.22, top=0.83)
    return fig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot ClaimForge MLLM T2 macro pixel IoU."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="directory for generated files (default: script directory)",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("pdf", "png", "svg"),
        default=("pdf", "png"),
        help="output formats (default: pdf png)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="raster output resolution (default: 300)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="open an interactive preview after saving",
    )
    return parser.parse_args()


def main(formats: Sequence[str] | None = None) -> None:
    args = _parse_args()
    selected_formats = tuple(formats) if formats is not None else args.formats
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig = make_figure()
    stem = args.output_dir / "mllm_t2_macro_pixel_iou"
    for output_format in selected_formats:
        output_path = stem.with_suffix(f".{output_format}")
        save_kwargs = {
            "bbox_inches": "tight",
            "pad_inches": 0.03,
            "facecolor": "white",
        }
        if output_format == "png":
            save_kwargs["dpi"] = args.dpi
        fig.savefig(output_path, **save_kwargs)
        print(output_path)

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
