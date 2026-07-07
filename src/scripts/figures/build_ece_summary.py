"""Mean-ECE comparison of English-transfer vs. pooled probes (Tables 4.5-4.6).

For every model, computes the mean test-split ECE across the 10 target
languages for (a) the English-trained probe at its strict-NLL layer and
(b) the pooled multilingual probe at its strict-NLL layer, then writes
``ece_english_vs_pooled_summary.{csv,pdf,svg}``.

Usage:
    python -m src.scripts.figures.build_ece_summary
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import load_success_rates
from src.scripts.probes.evaluate_cross_lingual_transfer import load_test_data_multilayer
from src.scripts.figures.reliability_utils import load_transfer_predictions

LANGUAGES = [
    "Arabic", "Chinese", "English", "French", "Italian",
    "Russian", "Swahili", "Telugu", "Thai", "Turkish",
]

MODEL_ROWS = [
    ("Qwen3-0.6B", "Qwen3-0.6B"),
    ("Qwen3-1.7B", "Qwen3-1.7B"),
    ("Qwen3-4B", "Qwen3-4B"),
    ("Qwen3-8B", "Qwen3-8B"),
    ("Qwen3.5-4B", "Qwen3.5-4B"),
    ("Qwen3.5-9B", "Qwen3.5-9B"),
    ("gpt-oss-low", "gpt-oss-20b_low"),
    ("gpt-oss-high", "gpt-oss-20b_high"),
]


def abs_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def strict_nll_layer(bundle: dict) -> int:
    best_layer = None
    best_value = float("inf")
    for layer, payload in bundle["layers"].items():
        value = payload["record"].get("val_log_loss", float("nan"))
        if not np.isfinite(value):
            continue
        value = float(value)
        layer = int(layer)
        if value < best_value or (value == best_value and (best_layer is None or layer < best_layer)):
            best_layer = layer
            best_value = value
    if best_layer is None:
        raise ValueError("No finite validation NLL found in bundle.")
    return best_layer


def calibration_ece(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    n_bins: int,
    min_samples: int,
) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(y_pred, bins, right=False) - 1, 0, n_bins - 1)
    gaps = []
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if int(mask.sum()) < min_samples:
            continue
        gaps.append(abs(float(y_pred[mask].mean()) - float(y_true[mask].mean())))
    return float(np.mean(gaps)) if gaps else float("nan")


def compute_rows(args: argparse.Namespace) -> pd.DataFrame:
    english_probe_dir = abs_path(args.probes_dir) / f"probes_{args.probe_type}"
    pooled_probe_dir = abs_path(args.pooled_probe_dir) / f"probes_{args.probe_type}"
    activations_root = abs_path(args.activations_root)
    success_dir = abs_path(args.success_dir)

    rows = []
    for display_model, model in MODEL_ROWS:
        english_bundle = joblib.load(english_probe_dir / model / f"{args.train_language}.joblib")
        english_layer = strict_nll_layer(english_bundle)
        del english_bundle
        gc.collect()

        pooled_bundle = joblib.load(pooled_probe_dir / model / args.pool_label)
        pooled_layer = strict_nll_layer(pooled_bundle)
        pooled_probe = pooled_bundle["layers"][pooled_layer]["probe"]

        success_df = load_success_rates(success_dir, models=[model], languages=LANGUAGES)
        english_ece = []
        pooled_ece = []
        for language in LANGUAGES:
            english_pred, english_true = load_transfer_predictions(
                model=model,
                train_lang=args.train_language,
                test_lang=language,
                layer=english_layer,
                probes_dir=english_probe_dir,
                activations_root=activations_root,
                success_dir=success_dir,
            )
            pooled_xy = load_test_data_multilayer(
                language,
                model,
                {pooled_layer},
                success_df,
                activations_root,
            )
            pooled_x, pooled_true = pooled_xy[pooled_layer]
            pooled_pred = np.clip(pooled_probe.predict(pooled_x).astype(np.float32), 0.0, 1.0)
            english_ece.append(calibration_ece(english_pred, english_true, args.n_bins, args.min_samples))
            pooled_ece.append(calibration_ece(pooled_pred, pooled_true, args.n_bins, args.min_samples))

        english_mean = float(np.mean(english_ece))
        pooled_mean = float(np.mean(pooled_ece))
        rows.append(
            {
                "model": display_model,
                "english_layer": english_layer,
                "pooled_layer": pooled_layer,
                "english_ece": english_mean,
                "pooled_ece": pooled_mean,
                "absolute_delta": english_mean - pooled_mean,
                "relative_reduction": (english_mean - pooled_mean) / english_mean,
                "factor_lower": english_mean / pooled_mean,
            }
        )
        del pooled_bundle, pooled_probe
        gc.collect()
    return pd.DataFrame(rows)


def plot_summary(df: pd.DataFrame, output_base: Path) -> None:
    plot_df = df.sort_values("relative_reduction", ascending=True).reset_index(drop=True)
    y = np.arange(len(plot_df))

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    fig.subplots_adjust(left=0.22, right=0.98, top=0.88, bottom=0.18)

    english_color = "#9b4d3f"
    pooled_color = "#2f6f9f"
    line_color = "#b8b8b8"

    for row_index, row in plot_df.iterrows():
        ax.plot(
            [row["pooled_ece"], row["english_ece"]],
            [row_index, row_index],
            color=line_color,
            linewidth=2.2,
            zorder=1,
        )
    ax.scatter(plot_df["english_ece"], y, s=54, color=english_color, label="English-transfer", zorder=3)
    ax.scatter(plot_df["pooled_ece"], y, s=54, color=pooled_color, label="Pooled training", zorder=4)

    for row_index, row in plot_df.iterrows():
        ax.text(
            row["pooled_ece"] - 0.004,
            row_index,
            f"{row['relative_reduction'] * 100:.0f}%",
            ha="right",
            va="center",
            fontsize=9.5,
            color="#1f4f66",
        )

    mean_english = float(df["english_ece"].mean())
    mean_pooled = float(df["pooled_ece"].mean())
    mean_reduction = (mean_english - mean_pooled) / mean_english
    ax.axvline(mean_english, color=english_color, linestyle=":", linewidth=1.4, alpha=0.65)
    ax.axvline(mean_pooled, color=pooled_color, linestyle=":", linewidth=1.4, alpha=0.65)

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["model"], fontsize=10.5)
    ax.set_xlabel("Mean ECE across target languages (lower is better)", fontsize=11.5)
    ax.tick_params(axis="x", labelsize=10.5)
    ax.grid(axis="x", alpha=0.18)
    ax.set_xlim(0.0, max(float(plot_df["english_ece"].max()), float(plot_df["pooled_ece"].max())) + 0.04)
    ax.set_title(
        f"Pooled multilingual probes reduce calibration error by {mean_reduction * 100:.0f}% on average",
        fontsize=13.5,
        pad=12,
    )
    ax.legend(loc="lower right", frameon=False, fontsize=10.5, ncol=2)

    output_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "svg"):
        fig.savefig(output_base.with_suffix(f".{fmt}"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--probe-type", default="ridge_scaled")
    parser.add_argument("--pool-label", default="_pooled.joblib")
    parser.add_argument("--train-language", default="English")
    parser.add_argument(
        "--probes-dir",
        type=Path,
        default=Path("outputs/probes/language_specific"),
    )
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
    args = parser.parse_args()

    output_dir = abs_path(args.output_dir)
    df = compute_rows(args)
    output_base = output_dir / "ece_english_vs_pooled_summary"
    df.to_csv(output_base.with_suffix(".csv"), index=False)
    plot_summary(df, output_base)
    print(f"Saved -> {output_base.with_suffix('.pdf')}")
    print(f"Saved -> {output_base.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
