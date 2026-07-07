"""Reliability diagram grid for the pooled multilingual probe.

The pooled probe is trained on a language-balanced 10%-per-language subsample
(same total training size as a single-language probe) and evaluated on every
language's test split at the layer selected by validation NLL.

Usage
-----
python -m src.scripts.figures.plot_reliability_pooled_multilingual --model Qwen3-4B
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.scripts.figures.reliability_utils import _load_test_data, _select_nll_layer

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

TITLE_FONTSIZE = 22
PANEL_TITLE_FONTSIZE = 19
AXIS_LABEL_FONTSIZE = 19
TICK_FONTSIZE = 16
ANNOTATION_FONTSIZE = 19
OUTPUT_DPI = 300


def _calibration_curve(y_pred: np.ndarray, y_true: np.ndarray, n_bins: int = 10, min_samples: int = 10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(y_pred, bins, right=False) - 1, 0, n_bins - 1)
    centers, mean_pred_out, mean_true_out = [], [], []
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() < min_samples:
            continue
        centers.append(bins[b] + (bins[b + 1] - bins[b]) / 2)
        mean_pred_out.append(float(y_pred[mask].mean()))
        mean_true_out.append(float(y_true[mask].mean()))
    return np.array(centers), np.array(mean_pred_out), np.array(mean_true_out)


def plot_cell(
    ax: plt.Axes,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    language: str,
    layer: int,
    n_bins: int,
    min_samples: int = 10,
) -> None:
    color = LANGUAGE_COLORS.get(language, "#333333")
    ax.set_facecolor("#ffffff")

    _, mean_pred, mean_true = _calibration_curve(y_pred, y_true, n_bins, min_samples)

    ax.plot([0, 1], [0, 1], "--", color="black", lw=1.0, alpha=0.55, zorder=1)
    ax.plot(mean_pred, mean_true, "o-", color=color, lw=3.4, ms=7, zorder=3)
    ax.fill_between(mean_pred, mean_pred, mean_true, color=color, alpha=0.15, zorder=2)
    ax.plot(
        y_pred,
        np.full_like(y_pred, -0.03),
        "|",
        color=color,
        alpha=0.25,
        ms=3,
        mew=0.5,
        transform=ax.get_xaxis_transform(),
        clip_on=False,
    )

    ece = float(np.mean(np.abs(mean_pred - mean_true))) if len(mean_pred) > 0 else float("nan")
    ax.text(
        0.97,
        0.04,
        f"ECE={ece:.3f}",
        ha="right",
        va="bottom",
        transform=ax.transAxes,
        fontsize=ANNOTATION_FONTSIZE,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.75),
    )
    ax.text(
        0.03,
        0.97,
        f"L{layer}",
        ha="left",
        va="top",
        transform=ax.transAxes,
        fontsize=ANNOTATION_FONTSIZE,
        color="#444444",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.75),
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)


def plot_model(
    model: str,
    probe: object,
    layer: int,
    success_df,
    activations_root: Path,
    languages: list[str],
    output_path: Path,
    n_bins: int = 10,
    min_samples: int = 10,
) -> None:
    n_lang = len(languages)
    ncols = 5
    nrows = (n_lang + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2.95 * ncols + 1.0, 3.9 * nrows),
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0.07,
        right=0.995,
        top=0.97,
        bottom=0.05,
        wspace=0.10,
        hspace=0.02,
    )

    for j, language in enumerate(languages):
        row, col = j // ncols, j % ncols
        ax = axes[row][col]
        ax.set_title(
            language,
            fontsize=PANEL_TITLE_FONTSIZE,
            color=LANGUAGE_COLORS.get(language, "black"),
            fontweight="bold",
            pad=10,
        )
        xy = _load_test_data(language, model, layer, success_df, activations_root)
        if xy is None:
            ax.set_facecolor("#fff0f0")
            ax.text(
                0.5,
                0.5,
                "no data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=ANNOTATION_FONTSIZE,
            )
            continue

        X_test, y_test = xy
        y_pred = np.clip(probe.predict(X_test).astype(np.float32), 0.0, 1.0)
        plot_cell(ax, y_pred, y_test, language, layer, n_bins, min_samples)
        if col != 0:
            ax.tick_params(axis="y", left=False, labelleft=False)
            ax.set_xticks([0.5, 1.0])
            ax.set_xticklabels(["0.5", "1.0"])

    # hide any unused axes
    for j in range(n_lang, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.supylabel("Empirical success rate", fontsize=AXIS_LABEL_FONTSIZE, x=0.01)
    fig.supxlabel("Predicted success probability", fontsize=AXIS_LABEL_FONTSIZE, y=0.01)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "svg"):
        fig.savefig(output_path.with_suffix(f".{fmt}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pooled-probe-dir",
        type=Path,
        default=Path("outputs/probes/pooled"),
    )
    parser.add_argument(
        "--activations-root",
        type=Path,
        default=Path("outputs/activations/MATH"),
    )
    parser.add_argument(
        "--success-dir",
        type=Path,
        default=Path("outputs/success_rates/MATH"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/figures/calibration"),
    )
    parser.add_argument("--model", type=str, default="Qwen3.5-4B")
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--min-samples", type=int, default=10,
                        help="Minimum number of samples required per bin. Default: 5")
    parser.add_argument("--languages", type=str, default=None)
    parser.add_argument(
        "--probe-type", type=str, default="ridge",
        help="Probe variant to load (e.g. ridge, ridge_scaled, logistic_balanced).",
    )
    parser.add_argument("--pool-label", type=str, default="_pooled",
                        help="Probe bundle stem to load (default: _pooled).")
    parser.add_argument("--output-prefix", type=str, default="reliability_pooled_multilingual",
                        help="Output filename prefix before _{model}.")
    parser.add_argument("--layer", type=int, default=None,
                        help="Override layer selection (skips NLL criterion).")
    args = parser.parse_args()

    def abs_path(path: Path) -> Path:
        return path if path.is_absolute() else REPO_ROOT / path

    probes_dir = abs_path(args.pooled_probe_dir) / f"probes_{args.probe_type}"
    if not probes_dir.exists():
        sys.exit(f"Probes directory not found: {probes_dir}")

    languages = (
        [language.strip() for language in args.languages.split(",") if language.strip()]
        if args.languages
        else LANGUAGES
    )

    from src.data import load_success_rates

    success_df = load_success_rates(
        abs_path(args.success_dir),
        models=[args.model],
        languages=languages,
    )
    output_dir = abs_path(args.output_dir)

    print(f"Model:      {args.model}")
    print(f"Languages:  {languages}")
    print(f"Output dir: {output_dir}\n")

    bundle_path = probes_dir / args.model / f"{args.pool_label}.joblib"
    if not bundle_path.exists():
        sys.exit(f"Probe bundle not found: {bundle_path}")

    bundle = joblib.load(bundle_path)
    n_layers = list(bundle["layers"].values())[0]["record"].get("n_layers", len(bundle["layers"]))
    if args.layer is not None:
        layer = args.layer
        print(f"  Layer override: L{layer} (frac={layer / n_layers:.2f}, n_layers={n_layers})")
    else:
        layer = _select_nll_layer(bundle, within_tolerance=False)
        print(f"  NLL-selected layer: L{layer} (frac={layer / n_layers:.2f}, n_layers={n_layers})")

    probe = bundle["layers"][layer]["probe"]
    del bundle
    gc.collect()

    plot_model(
        model=args.model,
        probe=probe,
        layer=layer,
        success_df=success_df,
        activations_root=abs_path(args.activations_root),
        languages=languages,
        output_path=output_dir / f"{args.output_prefix}_{args.model}.pdf",
        n_bins=args.n_bins,
        min_samples=args.min_samples,
    )


if __name__ == "__main__":
    main()
