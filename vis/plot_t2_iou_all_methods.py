#!/usr/bin/env python3
"""Plot ClaimForge T2 IoU for all methods with valid localization outputs.

Every panel uses its frozen 250-image forged condition. Missing or invalid
units count as zero IoU. Complete open-source and MLLM results therefore cover
750 images across the three panels. Copyleaks also has 250 valid localization
results in each panel.

The figure deliberately excludes image-level-only commercial APIs. Resemble's
provider-rendered JPEG heatmap is also excluded because it is not a validated
single-channel edit-localization score map.

Run from the repository root:

    conda run -n utils python vis/plot_t2_iou_all_methods.py

The script writes PDF and PNG files next to itself by default.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


@dataclass(frozen=True)
class MethodResult:
    name: str
    family: str
    mouse: float | None
    cat: float | None
    trash_can: float | None


# Macro per-image pixel IoU in percent.
#
# Open-source values:
#   results/opensource/*/*balanced250*/{balanced250_metrics,metrics}.json
# CAT-Net values:
#   docs/CATNET_BALANCED250_FULL_RESULTS_2026-07-27.md
# MLLM values:
#   results/mllm/balanced250_local1000_v2/summary.json; the strict Local750
#   denominator counts missing, invalid, empty, and out-of-bounds boxes as
#   zero IoU before rasterized-box comparison with exact-difference GT.
# Copyleaks:
#   results/commercial/copyleaks/
#   claimforge_balanced250_main_table_20260727.summary.json
#   (remote commit 41ae0219); unconditional IoU counts empty masks as zero.
METHODS = (
    MethodResult("CAT-Net v2", "Open-source", 35.1662, 60.9690, 29.4434),
    MethodResult("TruFor", "Open-source", 37.7607, 59.3433, 27.5932),
    MethodResult("DINOv3-IML", "Open-source", 12.7626, 46.5361, 13.7689),
    MethodResult("IML-ViT", "Open-source", 10.1329, 46.6896, 5.3608),
    MethodResult("RelayFormer", "Open-source", 13.3243, 37.8043, 7.3644),
    MethodResult("Mesorch", "Open-source", 2.9108, 16.5259, 3.5048),
    MethodResult("MVSS-Net", "Open-source", 1.2611, 2.7081, 2.9532),
    MethodResult("MaskCLIP", "Open-source", 0.4058, 2.9718, 0.3641),
    MethodResult("PSCC-Net", "Open-source", 0.0040, 1.0806, 0.5053),
    MethodResult("HiFi-IFDL", "Open-source", 0.00305, 0.00726, 0.01509),
    MethodResult("Claude Opus 4.8", "MLLM", 15.0374, 30.8778, 1.3162),
    MethodResult("Doubao Seed 2.1 Pro", "MLLM", 8.9530, 3.5364, 0.3685),
    MethodResult("GPT-5.6 Luna", "MLLM", 8.4632, 1.5394, 0.1172),
    MethodResult("Qwen 3.7 Plus", "MLLM", 4.6298, 0.8270, 0.0957),
    MethodResult("Copyleaks Ultra", "Commercial API", 32.6857, 59.8741, 32.2788),
)

FAMILY_PALETTES = {
    "Open-source": (
        "#e3f2fd",
        "#bbdefb",
        "#90caf9",
        "#64b5f6",
        "#42a5f5",
        "#2196f3",
        "#1e88e5",
        "#1976d2",
        "#1565c0",
        "#0d47a1",
    ),
    "Commercial API": (
        "#edc4b3",
        "#e6b8a2",
        "#deab90",
        "#d69f7e",
        "#cd9777",
        "#c38e70",
        "#b07d62",
        "#9d6b53",
        "#8a5a44",
        "#774936",
    ),
    "MLLM": ("#a1cca5", "#8fb996", "#709775", "#415d43"),
}

# Match the Commercial API encoding to Alibaba's line color in Figure 3.
ALIBABA_LINE_COLOR = "#9F8D8D"

PANELS = (
    ("Mouse", "mouse"),
    ("Cat", "cat"),
    ("Trash-can", "trash_can"),
)

ROW_STEP = 0.72
BAR_HEIGHT = 0.40


def _assign_method_colors() -> dict[str, str]:
    family_indices = {family: 0 for family in FAMILY_PALETTES}
    colors: dict[str, str] = {}
    for method in METHODS:
        index = family_indices[method.family]
        # Rows are ordered from top to bottom; reverse each supplied palette so
        # every family progresses from its darkest shade to its lightest.
        palette = tuple(reversed(FAMILY_PALETTES[method.family]))
        if index >= len(palette):
            raise ValueError(f"not enough colors for family {method.family}")
        colors[method.name] = palette[index]
        family_indices[method.family] += 1
    return colors


METHOD_COLORS = _assign_method_colors()
# Copyleaks is the only commercial API with a validated T2 mask, so use the
# same Commercial API color in both its bars and the family legend.
METHOD_COLORS["Copyleaks Ultra"] = ALIBABA_LINE_COLOR


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "axes.linewidth": 0.65,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6.4,
            "legend.fontsize": 7,
            "mathtext.fontset": "stix",
            "mathtext.default": "regular",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def make_figure() -> plt.Figure:
    """Create a four-panel, paper-ready horizontal bar chart."""
    _configure_style()
    fig, axes = plt.subplots(
        1,
        len(PANELS),
        figsize=(7.5, 3.15),
        sharex=True,
        sharey=True,
        gridspec_kw={"wspace": 0.15},
    )

    y_positions = [index * ROW_STEP for index in range(len(METHODS))]
    family_boundaries = [
        (y_positions[index - 1] + y_positions[index]) / 2
        for index in range(1, len(METHODS))
        if METHODS[index - 1].family != METHODS[index].family
    ]
    x_max = 68

    for panel_index, ((title, field), ax) in enumerate(zip(PANELS, axes)):
        for y, method in zip(y_positions, METHODS):
            value = getattr(method, field)
            if value is None:
                ax.text(
                    1.0,
                    y,
                    "N/A",
                    va="center",
                    ha="left",
                    fontsize=5.8,
                    color="#777777",
                    fontstyle="italic",
                )
                continue

            ax.barh(
                y,
                value,
                height=BAR_HEIGHT,
                color=METHOD_COLORS[method.name],
                edgecolor="none",
                linewidth=0,
                zorder=3,
            )
            value_label = (
                r"$\mathbf{<0.01}$"
                if 0 < value < 0.01
                else rf"$\mathbf{{{value:.2f}}}$"
            )
            ax.text(
                value + 0.8,
                y,
                value_label,
                va="center",
                ha="left",
                fontsize=5.8,
                color="#222222",
            )

        ax.set_title(title, pad=4, fontsize=8, fontweight="bold")
        ax.set_xlim(0, x_max)
        ax.set_xticks(
            (0, 20, 40, 60),
            (r"$0$", r"$20$", r"$40$", r"$60$"),
        )
        ax.grid(axis="x", color="#D6D6D6", linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if panel_index > 0:
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", left=False)

        # Separate method families without adding extra panels or labels.
        for boundary in family_boundaries:
            ax.axhline(boundary, color="#AFAFAF", linewidth=0.6, zorder=1)

    axes[0].set_yticks(y_positions, [method.name for method in METHODS])
    axes[0].invert_yaxis()
    fig.supxlabel(
        "Macro pixel IoU (%)",
        y=0.015,
        fontsize=7.5,
        fontweight="bold",
    )

    legend_colors = {
        "Open-source": FAMILY_PALETTES["Open-source"][5],
        "MLLM": FAMILY_PALETTES["MLLM"][2],
        "Commercial API": ALIBABA_LINE_COLOR,
    }
    legend_handles = [
        Patch(
            facecolor=legend_colors[family],
            edgecolor="none",
            linewidth=0,
            label=family,
        )
        for family in ("Open-source", "MLLM", "Commercial API")
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.59, 0.992),
        ncol=3,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.6,
    )
    fig.subplots_adjust(left=0.225, right=0.995, bottom=0.11, top=0.875)
    return fig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot ClaimForge T2 IoU for localization-capable methods."
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
    stem = args.output_dir / "t2_macro_pixel_iou_all_methods"
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
