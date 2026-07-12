"""All-model AUROC trajectories for same-language probes only."""

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

MODEL_ORDER = [
    "Qwen3-0.6B",
    "Qwen3-1.7B",
    "Qwen3-4B",
    "Qwen3-8B",
    "Qwen3.5-4B",
    "Qwen3.5-9B",
    "gpt-oss-20b_low",
    "gpt-oss-20b_high",
]


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")


def mean_std_by_layer(df: pd.DataFrame, layers: list[int]) -> tuple[pd.Series, pd.Series]:
    by_language = (
        df[df["is_diagonal"]]
        .pivot_table(index="layer", columns="test_lang", values="auroc", aggfunc="mean")
        .reindex(layers)
    )
    return by_language.mean(axis=1), by_language.std(axis=1)


def axis_limits(series: list[tuple[pd.Series, pd.Series]], margin: float = 0.05) -> tuple[float, float]:
    lower = min(float((mean - std).min()) for mean, std in series)
    upper = max(float((mean + std).max()) for mean, std in series)
    return max(0.0, lower - margin), min(1.0, upper + margin)


def plot_all_models(
    df: pd.DataFrame,
    output_path: Path,
    model_order: list[str],
    style: dict[str, object],
) -> None:
    df = df[df["layer"] > 0].copy()
    models = [model for model in model_order if model in set(df["model"])]
    if not models:
        raise ValueError("None of the requested models were present in the input data.")

    model_series: dict[str, tuple[list[int], pd.Series, pd.Series]] = {}
    for model in models:
        model_df = df[df["model"] == model]
        layers = sorted(model_df["layer"].unique())
        mean, std = mean_std_by_layer(model_df, layers)
        model_series[model] = (layers, mean, std)

    ymin, ymax = axis_limits([(mean, std) for _, mean, std in model_series.values()])

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(4, 2, figsize=(14.2, 14.6), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.975, bottom=0.075, wspace=0.12, hspace=0.55)
    axes_flat = [axes[row][col] for row in range(4) for col in range(2)]

    color = "#2f6f9f"
    for idx, model in enumerate(models):
        ax = axes_flat[idx]
        layers, mean, std = model_series[model]
        lower = (mean - std).clip(lower=0.0)
        upper = (mean + std).clip(upper=1.0)

        ax.fill_between(layers, lower, upper, color=color, alpha=0.18, linewidth=0)
        ax.plot(layers, lower, color=color, linewidth=1.0, alpha=0.35)
        ax.plot(layers, upper, color=color, linewidth=1.0, alpha=0.35)
        ax.plot(
            layers,
            mean,
            color=color,
            linewidth=3.2,
            zorder=3,
        )

        ax.set_title(model, fontsize=style["panel_title_fontsize"], pad=9)
        ax.set_xlabel("Layer", fontsize=style["axis_label_fontsize"])
        if idx % 2 == 0:
            ax.set_ylabel("AUROC", fontsize=style["axis_label_fontsize"])
        ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.05))
        ax.set_xlim(layers[0] - 0.5, layers[-1] + 0.5)
        ax.set_ylim(ymin, ymax)
        ax.tick_params(labelsize=style["tick_fontsize"])
        ax.grid(axis="y", linestyle="-", alpha=0.16)
        ax.grid(axis="x", linestyle=":", alpha=0.14)

    for idx in range(len(models), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    handles = [
        Line2D(
            [0],
            [0],
            color="#2f2f2f",
            lw=3.2,
            label="Mean AUROC",
        ),
        Patch(facecolor="#6a7fa5", alpha=0.22, edgecolor="none", label="+/- 1 std. dev."),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        fontsize=style["legend_fontsize"],
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
        help="Model names to plot, in panel order.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="paper",
        choices=sorted(STYLE),
        help="Typography variant borrowed from the paper trajectory plots.",
    )
    args = parser.parse_args()

    frames = []
    for path in args.transfer_csv:
        csv_path = path if path.is_absolute() else REPO_ROOT / path
        frames.append(pd.read_csv(csv_path))
    df = pd.concat(frames, ignore_index=True)
    df["is_diagonal"] = df["is_diagonal"].astype(bool)

    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    output_name = "same_language_auroc_trajectory_all_models"
    if args.variant != "paper":
        output_name += f"_{safe_filename(args.variant)}"
    plot_all_models(df, output_dir / f"{output_name}.pdf", args.models, STYLE[args.variant])


if __name__ == "__main__":
    main()
