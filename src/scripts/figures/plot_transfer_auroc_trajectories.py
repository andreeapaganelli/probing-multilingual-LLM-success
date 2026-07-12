"""Layer-wise AUROC trajectory (mean ± std across languages) for one model.

Left panel: same-language probes. Right panel: cross-lingual transfer mean.
"""

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


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")


def _mean_std(series_by_layer: pd.DataFrame, layers: list[int]) -> tuple[pd.Series, pd.Series]:
    series_by_layer = series_by_layer.reindex(layers)
    mean = series_by_layer.mean(axis=1)
    std = series_by_layer.std(axis=1)
    return mean, std


def _axis_limits(means: list[pd.Series], stds: list[pd.Series], margin: float = 0.05) -> tuple[float, float]:
    plotted_min = min(float((mean - std).min()) for mean, std in zip(means, stds))
    plotted_max = max(float((mean + std).max()) for mean, std in zip(means, stds))
    return max(0.0, plotted_min - margin), min(1.0, plotted_max + margin)


def plot_mean_std(df: pd.DataFrame, model: str, output_path: Path, style: dict[str, object]) -> None:
    df = df[df["layer"] > 0].copy()
    layers = sorted(df["layer"].unique())
    if not layers:
        raise ValueError("No positive layers found to plot.")

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

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=style["figsize"], sharey=True)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.90, bottom=0.18, wspace=0.12)

    panels = [
        (axes[0], same_mean, same_std, "Same-language probe", "#2f6f9f"),
        (axes[1], transfer_mean, transfer_std, "Transfer probe mean", "#7a5c9e"),
    ]
    ymin, ymax = _axis_limits([same_mean, transfer_mean], [same_std, transfer_std])
    for ax, mean, std, title, color in panels:
        lower = (mean - std).clip(lower=0.0)
        upper = (mean + std).clip(upper=1.0)
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
            mean,
            color=color,
            linewidth=3.2,
            label="Mean AUROC",
            zorder=3,
        )
        ax.set_xlabel("Layer", fontsize=style["axis_label_fontsize"])
        ax.set_title(title, fontsize=style["panel_title_fontsize"], pad=10)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.05))
        ax.set_xlim(layers[0] - 0.5, layers[-1] + 0.5)
        ax.set_ylim(ymin, ymax)
        ax.tick_params(labelsize=style["tick_fontsize"])
        ax.grid(axis="y", linestyle="-", alpha=0.16)
        ax.grid(axis="x", linestyle=":", alpha=0.14)

    axes[0].set_ylabel("AUROC", fontsize=style["axis_label_fontsize"])
    axes[1].set_ylabel("AUROC", fontsize=style["axis_label_fontsize"])

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
        default=Path("outputs/transfer/MATH/ridge_scaled/cross_lingual_per_layer_all.csv"),
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures/auroc_trajectories"))
    parser.add_argument(
        "--variant",
        type=str,
        default="paper",
        choices=sorted(STYLE),
        help="Figure typography/layout variant.",
    )
    args = parser.parse_args()

    transfer_csv = args.transfer_csv if args.transfer_csv.is_absolute() else REPO_ROOT / args.transfer_csv
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir

    df = pd.read_csv(transfer_csv)
    df["is_diagonal"] = df["is_diagonal"].astype(bool)
    model_df = df[df["model"] == args.model].copy()
    if model_df.empty:
        sys.exit(f"No rows found for model {args.model!r} in {transfer_csv}")

    output_name = f"transfer_auroc_trajectory_{safe_filename(args.model)}{STYLE[args.variant]['suffix']}.pdf"
    plot_mean_std(model_df, args.model, output_dir / output_name, STYLE[args.variant])


if __name__ == "__main__":
    main()
