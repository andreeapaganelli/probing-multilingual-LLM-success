"""Shared figure style constants + per-language AUROC trajectory overlay plot.

STYLE is imported by the other AUROC-trajectory figure scripts. Running this
module directly plots the per-language same-language/transfer overlay for one
model.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LANGUAGES = [
    "Arabic", "Chinese", "English", "French", "Italian",
    "Russian", "Swahili", "Telugu", "Thai", "Turkish",
]
LANGUAGE_COLORS = {
    "Chinese":  "#e74c3c",
    "English":  "#3498db",
    "French":   "#2ecc71",
    "Italian":  "#f39c12",
    "Swahili":  "#9b59b6",
    "Arabic":   "#8c564b",
    "Russian":  "#e377c2",
    "Telugu":   "#7f7f7f",
    "Thai":     "#bcbd22",
    "Turkish":  "#17becf",
}

SHORT_NAME: dict[str, str] = {}

TITLE_FONTSIZE = 19
PANEL_TITLE_FONTSIZE = 18
AXIS_LABEL_FONTSIZE = 17
TICK_FONTSIZE = 15
LEGEND_FONTSIZE = 15
OUTPUT_DPI = 300

STYLE = {
    "paper": {
        "suffix": "",
        "figsize": (14.2, 6.0),
        "subplots_adjust": dict(left=0.07, right=0.97, top=0.91, bottom=0.25, wspace=0.10),
        "title_fontsize": TITLE_FONTSIZE,
        "panel_title_fontsize": PANEL_TITLE_FONTSIZE,
        "axis_label_fontsize": AXIS_LABEL_FONTSIZE,
        "tick_fontsize": TICK_FONTSIZE,
        "legend_fontsize": LEGEND_FONTSIZE,
        "legend_loc": "bottom",
        "line_width": 2.0,
        "mean_line_width": 3.8,
    },
    "half-page": {
        "suffix": "_half_page",
        "figsize": (14.2, 6.5),
        "subplots_adjust": dict(left=0.075, right=0.985, top=0.91, bottom=0.29, wspace=0.10),
        "title_fontsize": 30,
        "panel_title_fontsize": 27,
        "axis_label_fontsize": 26,
        "tick_fontsize": 23,
        "legend_fontsize": 22,
        "legend_loc": "bottom",
        "line_width": 3.0,
        "mean_line_width": 5.2,
    },
}


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")


def language_order(df: pd.DataFrame) -> list[str]:
    observed = list(df["test_lang"].dropna().unique())
    ordered = [language for language in LANGUAGES if language in observed]
    ordered.extend(sorted(language for language in observed if language not in ordered))
    return ordered


def plot_overlay(df: pd.DataFrame, model: str, output_path: Path, style: dict[str, object]) -> None:
    model_label = SHORT_NAME.get(model, model)
    df = df[df["layer"] > 0].copy()
    languages = language_order(df)
    layers = sorted(df["layer"].unique())

    fig, axes = plt.subplots(1, 2, figsize=style["figsize"], sharey=True)
    fig.subplots_adjust(**style["subplots_adjust"])
    ax_same, ax_trans = axes

    for language in languages:
        color = LANGUAGE_COLORS.get(language, "#444444")
        lang_df = df[df["test_lang"] == language]

        same = lang_df[lang_df["is_diagonal"]].set_index("layer")["auroc"].reindex(layers)
        transfer_mean = (
            lang_df[~lang_df["is_diagonal"]]
            .groupby("layer")["auroc"]
            .mean()
            .reindex(layers)
        )
        ax_same.plot(layers, same, color=color, linewidth=style["line_width"], alpha=0.88, label=language)
        ax_trans.plot(layers, transfer_mean, color=color, linewidth=style["line_width"], alpha=0.88, label=language)

    grand_same = df[df["is_diagonal"]].groupby("layer")["auroc"].mean().reindex(layers)
    grand_transfer = df[~df["is_diagonal"]].groupby("layer")["auroc"].mean().reindex(layers)
    ax_same.plot(layers, grand_same, color="black", linewidth=style["mean_line_width"], label="Mean", zorder=10)
    ax_trans.plot(layers, grand_transfer, color="black", linewidth=style["mean_line_width"], label="Mean", zorder=10)

    titles = [
        (ax_same, "Same-language probe"),
        (ax_trans, "Transfer probe mean"),
    ]
    for ax, title in titles:
        ax.set_xlabel("Layer", fontsize=style["axis_label_fontsize"])
        ax.set_title(title, fontsize=style["panel_title_fontsize"], pad=10)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.05))
        ax.set_xlim(layers[0] - 0.5, layers[-1] + 0.5)
        ax.tick_params(labelsize=style["tick_fontsize"])
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.grid(axis="x", linestyle=":", alpha=0.22)

    ax_same.set_ylabel("AUROC", fontsize=style["axis_label_fontsize"])
    ax_trans.set_ylabel("AUROC", fontsize=style["axis_label_fontsize"])
    handles, labels = ax_trans.get_legend_handles_labels()
    if style["legend_loc"] == "bottom":
        ncol_legend = (len(labels) + 1) // 2  # 2 rows
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            fontsize=style["legend_fontsize"],
            ncol=ncol_legend,
            framealpha=0.85,
        )
    else:
        ax_trans.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=style["legend_fontsize"],
            ncol=1,
            framealpha=0.85,
            title="Test language",
            title_fontsize=style["legend_fontsize"],
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "svg"):
        fig.savefig(output_path.with_suffix(f".{fmt}"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe-type", type=str, default="ridge",
        help="Probe variant (e.g. ridge, logistic, logistic_balanced). Controls transfer-csv if not set explicitly.",
    )
    parser.add_argument(
        "--transfer-csv",
        type=Path,
        default=None,
    )
    parser.add_argument("--model", type=str, default="Qwen3.5-4B")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures/auroc_trajectories"))
    parser.add_argument(
        "--variant",
        type=str,
        default="paper",
        choices=sorted(STYLE),
        help="Figure typography/layout variant. Use half-page for larger text in a smaller paper placement.",
    )
    args = parser.parse_args()

    if args.transfer_csv is None:
        args.transfer_csv = Path(
            f"outputs/transfer/MATH/{args.probe_type}/cross_lingual_per_layer_all.csv"
        )
    transfer_csv = args.transfer_csv if args.transfer_csv.is_absolute() else REPO_ROOT / args.transfer_csv
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir

    df = pd.read_csv(transfer_csv)
    df["is_diagonal"] = df["is_diagonal"].astype(bool)
    model_df = df[df["model"] == args.model].copy()
    if model_df.empty:
        sys.exit(f"No rows found for model {args.model!r} in {transfer_csv}")

    style = STYLE[args.variant]
    output_name = f"auroc_trajectory_overlay_{safe_filename(args.model)}{style['suffix']}.pdf"

    plot_overlay(
        model_df,
        args.model,
        output_dir / output_name,
        style,
    )


if __name__ == "__main__":
    main()
