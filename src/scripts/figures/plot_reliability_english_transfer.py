"""Reliability diagram grid for the English-trained transfer probe.

One reliability panel per language, matching the styling of
plot_reliability_pooled_multilingual.py. The layer is selected by the pooled
probe's validation NLL. Supports two directions:
  - english-probe-to-all: English probe evaluated on each language
  - all-probes-to-english: each language probe evaluated on English

Usage
-----
python -m src.scripts.figures.plot_reliability_english_transfer \\
    [--pooled-probe-dir outputs/probes/pooled] \\
    [--probes-dir       outputs/probes/language_specific/probes_ridge] \\
    [--activations-root outputs/activations/MATH] \\
    [--output-dir       outputs/figures/calibration] \\
    [--model            Qwen3.5-4B] \\
    [--direction        english-probe-to-all] \\
    [--n-bins           10]
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

from src.scripts.figures.reliability_utils import _select_nll_layer
from src.scripts.figures.reliability_utils import load_transfer_predictions

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

TITLE_FONTSIZE        = 22
PANEL_TITLE_FONTSIZE  = 19
AXIS_LABEL_FONTSIZE   = 19
TICK_FONTSIZE         = 16
ANNOTATION_FONTSIZE   = 19
OUTPUT_DPI            = 300


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
    train_language: str,
    test_language: str,
    layer: int,
    n_bins: int,
    min_samples: int = 10,
) -> None:
    color = LANGUAGE_COLORS.get(test_language, "#333333")
    ax.set_facecolor("#ffffff")

    _, mean_pred, mean_true = _calibration_curve(y_pred, y_true, n_bins, min_samples)

    ax.plot([0, 1], [0, 1], "--", color="black", lw=1.0, alpha=0.55, zorder=1)
    ax.plot(mean_pred, mean_true, "o-", color=color, lw=3.4, ms=7, zorder=3)
    ax.fill_between(mean_pred, mean_pred, mean_true, color=color, alpha=0.15, zorder=2)
    ax.plot(
        y_pred, np.full_like(y_pred, -0.03),
        "|", color=color, alpha=0.25, ms=3, mew=0.5,
        transform=ax.get_xaxis_transform(), clip_on=False,
    )

    ece = float(np.mean(np.abs(mean_pred - mean_true))) if len(mean_pred) > 0 else float("nan")
    ax.text(
        0.97, 0.04, f"ECE={ece:.3f}",
        ha="right", va="bottom", transform=ax.transAxes,
        fontsize=ANNOTATION_FONTSIZE,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.75),
    )
    ax.text(
        0.03, 0.97, f"L{layer}",
        ha="left", va="top", transform=ax.transAxes,
        fontsize=ANNOTATION_FONTSIZE, color="#444444",
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
    layer: int,
    probes_dir: Path,
    activations_root: Path,
    panel_train_languages: list[str],
    panel_test_languages: list[str],
    panel_titles: list[str],
    output_path: Path,
    title_prefix: str,
    n_bins: int = 10,
    min_samples: int = 10,
    success_dir: Path | None = None,
) -> None:
    n_lang = len(panel_test_languages)
    ncols = 5
    nrows = (n_lang + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(2.95 * ncols + 1.0, 3.9 * nrows),
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0.07, right=0.995,
        top=0.97, bottom=0.05,
        wspace=0.10, hspace=0.02,
    )

    for j, (train_lang, test_lang, title) in enumerate(
        zip(panel_train_languages, panel_test_languages, panel_titles)
    ):
        row, col = j // ncols, j % ncols
        ax = axes[row][col]
        ax.set_title(
            title,
            fontsize=PANEL_TITLE_FONTSIZE,
            color=LANGUAGE_COLORS.get(title, LANGUAGE_COLORS.get(test_lang, "black")),
            fontweight="bold",
            pad=10,
        )
        try:
            y_pred, y_true = load_transfer_predictions(
                model=model,
                train_lang=train_lang,
                test_lang=test_lang,
                layer=layer,
                probes_dir=probes_dir,
                activations_root=activations_root,
                success_dir=success_dir,
            )
        except Exception as exc:
            ax.set_facecolor("#fff0f0")
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=ANNOTATION_FONTSIZE)
            print(f"  [skip] {train_lang}→{test_lang}: {exc}")
            continue

        plot_cell(ax, y_pred, y_true, train_lang, test_lang, layer, n_bins, min_samples)
        if col != 0:
            ax.tick_params(axis="y", left=False, labelleft=False)
            ax.set_xticks([0.5, 1.0])
            ax.set_xticklabels(["0.5", "1.0"])

    for j in range(n_lang, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.supylabel("Empirical success rate", fontsize=AXIS_LABEL_FONTSIZE, x=0.01)
    fig.supxlabel("Predicted success probability", fontsize=AXIS_LABEL_FONTSIZE, y=0.01)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "svg"):
        fig.savefig(output_path.with_suffix(f".{fmt}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pooled-probe-dir", type=Path,
        default=Path("outputs/probes/pooled"))
    parser.add_argument(
        "--probe-type", type=str, default="ridge",
        help="Probe variant to load (e.g. ridge, logistic). Controls probes-dir if not set explicitly.",
    )
    parser.add_argument("--probes-dir", type=Path, default=None)
    parser.add_argument("--activations-root", type=Path,
        default=Path("outputs/activations/MATH"))
    parser.add_argument("--output-dir", type=Path,
        default=Path("outputs/figures/calibration"))
    parser.add_argument("--model", type=str, default="Qwen3.5-4B")
    parser.add_argument("--train-language", type=str, default="English")
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--min-samples", type=int, default=10,
                        help="Minimum number of samples required per bin. Default: 5")
    parser.add_argument("--test-languages", type=str, default=None)
    parser.add_argument(
        "--direction", type=str, default="english-probe-to-all",
        choices=["english-probe-to-all", "all-probes-to-english"],
    )
    parser.add_argument("--layer", type=int, default=None,
                        help="Override layer selection (skips NLL criterion).")
    parser.add_argument("--success-dir", type=Path, default=None,
                        help="Directory containing {model}_{language}.csv success rate files. "
                             "Defaults to outputs/success_rates/MATH.")
    args = parser.parse_args()
    if args.probes_dir is None:
        args.probes_dir = Path(f"outputs/probes/language_specific/probes_{args.probe_type}")

    def abs_path(p: Path) -> Path:
        return p if p.is_absolute() else REPO_ROOT / p

    panel_languages = (
        [l.strip() for l in args.test_languages.split(",") if l.strip()]
        if args.test_languages else list(LANGUAGES)
    )

    if args.layer is not None:
        layer = args.layer
        n_layers = None
    else:
        bundle_path = abs_path(args.probes_dir) / args.model / f"{args.train_language}.joblib"
        if not bundle_path.exists():
            sys.exit(f"Probe bundle not found: {bundle_path}")
        bundle = joblib.load(bundle_path)
        n_layers = list(bundle["layers"].values())[0]["record"].get("n_layers", len(bundle["layers"]))
        layer = _select_nll_layer(bundle, within_tolerance=False)
        del bundle; gc.collect()

    print(f"Model:          {args.model}")
    print(f"Direction:      {args.direction}")
    print(f"Panel languages:{panel_languages}")
    frac_str = f"{layer / n_layers:.2f}" if n_layers else "?"
    print(f"Layer:          L{layer} (frac={frac_str}, n_layers={n_layers})")
    print(f"Output dir:     {abs_path(args.output_dir)}\n")

    if args.direction == "english-probe-to-all":
        panel_train = [args.train_language] * len(panel_languages)
        panel_test  = panel_languages
        output_name = f"reliability_english_transfer_{args.model}.pdf"
        title_prefix = "English-probe transfer reliability"
    else:
        panel_train = panel_languages
        panel_test  = ["English"] * len(panel_languages)
        output_name = f"reliability_to_english_transfer_{args.model}.pdf"
        title_prefix = "English transfer reliability"

    plot_model(
        model=args.model,
        layer=layer,
        probes_dir=abs_path(args.probes_dir),
        activations_root=abs_path(args.activations_root),
        panel_train_languages=panel_train,
        panel_test_languages=panel_test,
        panel_titles=panel_languages,
        output_path=abs_path(args.output_dir) / output_name,
        title_prefix=title_prefix,
        n_bins=args.n_bins,
        min_samples=args.min_samples,
        success_dir=abs_path(args.success_dir) if args.success_dir is not None else None,
    )


if __name__ == "__main__":
    main()
