"""Paper-ready per-model AUROC trajectories for same-language probes only."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.scripts.figures.plot_style import STYLE
from src.scripts.figures.plot_same_language_auroc_trajectories import MODEL_ORDER


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")


def _mean_std(series_by_layer: pd.DataFrame, layers: list[int]) -> tuple[pd.Series, pd.Series]:
    series_by_layer = series_by_layer.reindex(layers)
    return series_by_layer.mean(axis=1), series_by_layer.std(axis=1)


def _axis_limits(means: list[pd.Series], stds: list[pd.Series], margin: float = 0.05) -> tuple[float, float]:
    plotted_min = min(float((mean - std).min()) for mean, std in zip(means, stds))
    plotted_max = max(float((mean + std).max()) for mean, std in zip(means, stds))
    return max(0.0, plotted_min - margin), min(1.0, plotted_max + margin)


def _single_panel_style(style: dict[str, object]) -> dict[str, object]:
    width, height = style["figsize"]
    return {
        **style,
        "axis_label_fontsize": style["axis_label_fontsize"] + 3,
        "tick_fontsize": style["tick_fontsize"] + 3,
        "legend_fontsize": style["legend_fontsize"] + 3,
        "figsize": (width / 2.0, height),
        "subplots_adjust": dict(left=0.14, right=0.985, top=0.96, bottom=0.24),
    }


def plot_same_language(df: pd.DataFrame, model: str, output_path: Path, style: dict[str, object]) -> None:
    df = df[df["layer"] > 0].copy()
    layers = sorted(df["layer"].unique())
    if not layers:
        raise ValueError(f"No positive layers found to plot for {model}.")

    same_by_language = (
        df[df["is_diagonal"]]
        .pivot_table(index="layer", columns="test_lang", values="auroc", aggfunc="mean")
    )
    transfer_by_language = (
        df[~df["is_diagonal"]]
        .groupby(["layer", "test_lang"])["auroc"]
        .mean()
        .unstack("test_lang")
    )
    same_mean, same_std = _mean_std(same_by_language, layers)
    transfer_mean, transfer_std = _mean_std(transfer_by_language, layers)
    ymin, ymax = _axis_limits([same_mean, transfer_mean], [same_std, transfer_std])

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    single_style = _single_panel_style(style)
    fig, ax = plt.subplots(1, 1, figsize=single_style["figsize"])
    fig.subplots_adjust(**single_style["subplots_adjust"])

    color = "#2f6f9f"
    lower = (same_mean - same_std).clip(lower=0.0)
    upper = (same_mean + same_std).clip(upper=1.0)
    ax.fill_between(
        layers,
        lower,
        upper,
        color=color,
        alpha=0.18,
        linewidth=0,
        label="+/- 1 std. dev.",
    )
    ax.plot(layers, lower, color=color, linewidth=1.0, alpha=0.35)
    ax.plot(layers, upper, color=color, linewidth=1.0, alpha=0.35)
    ax.plot(
        layers,
        same_mean,
        color=color,
        linewidth=3.2,
        label="Mean AUROC",
        zorder=3,
    )

    ax.set_xlabel("Layer", fontsize=single_style["axis_label_fontsize"])
    ax.set_ylabel("AUROC", fontsize=single_style["axis_label_fontsize"])
    ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.05))
    ax.set_xlim(layers[0] - 0.5, layers[-1] + 0.5)
    ax.set_ylim(ymin, ymax)
    ax.tick_params(labelsize=single_style["tick_fontsize"])
    ax.grid(axis="y", linestyle="-", alpha=0.16)
    ax.grid(axis="x", linestyle=":", alpha=0.14)

    handles = [
        Line2D(
            [0],
            [0],
            color=color,
            lw=3.2,
            label="Mean AUROC",
        ),
        Patch(facecolor=color, alpha=0.22, edgecolor="none", label="+/- 1 std. dev."),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        fontsize=single_style["legend_fontsize"],
        ncol=2,
        frameon=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "svg"):
        fig.savefig(output_path.with_suffix(f".{fmt}"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transfer-csv",
        type=Path,
        action="append",
        required=True,
        help="Per-layer cross-lingual transfer CSV. Pass multiple times to combine model families.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures/auroc_trajectories"))
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODEL_ORDER,
        help="Model names to plot.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="paper",
        choices=sorted(STYLE),
        help="Typography/layout variant borrowed from the paper trajectory plots.",
    )
    args = parser.parse_args()

    frames = []
    for path in args.transfer_csv:
        csv_path = path if path.is_absolute() else REPO_ROOT / path
        frames.append(pd.read_csv(csv_path))
    df = pd.concat(frames, ignore_index=True)
    df["is_diagonal"] = df["is_diagonal"].astype(bool)

    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    style = STYLE[args.variant]

    for model in args.models:
        model_df = df[df["model"] == model].copy()
        if model_df.empty:
            print(f"Skipping {model}: no rows found")
            continue

        output_name = f"same_language_auroc_trajectory_{safe_filename(model)}"
        if args.variant != "paper":
            output_name += f"_{safe_filename(args.variant)}"
        plot_same_language(model_df, model, output_dir / f"{output_name}.pdf", style)


if __name__ == "__main__":
    main()
