#!/usr/bin/env python
"""Plot matched-layer MATH/BFCL probe AUROC trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_OUTPUT_DIR = Path("outputs/bfcl/cross_task_transfer")
DEFAULT_MODELS = [
    "Qwen3-0.6B",
    "Qwen3-1.7B",
    "Qwen3-4B",
    "Qwen3-8B",
    "Qwen3.5-4B",
    "Qwen3.5-9B",
]

SERIES = [
    ("math_test_auroc", "Train: MATH; test: MATH", "#4c78a8"),
    ("math_to_bfcl_test_auroc", "Train: MATH; test: BFCL", "#f58518"),
    ("math_bfcl_to_bfcl_test_auroc", "Train: MATH+BFCL; test: BFCL", "#b279a2"),
    ("bfcl_to_bfcl_test_auroc", "Train: BFCL; test: BFCL", "#54a24b"),
]


def parse_models(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def load_matched_dataframe(output_dir: Path, models: list[str]) -> pd.DataFrame:
    math_path = output_dir / "recomputed_math_english_layer_auroc.csv"
    metrics_path = output_dir / "bfcl_cross_task_layer_metrics.csv"
    if not math_path.exists():
        raise FileNotFoundError(f"Missing MATH reference CSV: {math_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing layer metrics CSV: {metrics_path}")

    math_df = pd.read_csv(math_path)[["model", "layer", "math_test_auroc_recomputed"]].rename(
        columns={"math_test_auroc_recomputed": "math_test_auroc"}
    )
    metrics = pd.read_csv(metrics_path)
    transfer = metrics[
        (metrics["metric"] == "auroc")
        & (metrics["train_task"] == "MATH")
        & (metrics["eval_task"] == "BFCL")
        & (metrics["bfcl_eval_split"] == "test")
    ][["model", "layer", "value"]].rename(columns={"value": "math_to_bfcl_test_auroc"})
    bfcl = metrics[
        (metrics["metric"] == "auroc")
        & (metrics["train_task"] == "BFCL")
        & (metrics["eval_task"] == "BFCL")
    ][["model", "layer", "value"]].rename(columns={"value": "bfcl_to_bfcl_test_auroc"})
    mixed = metrics[
        (metrics["metric"] == "auroc")
        & (metrics["train_task"] == "MATH+BFCL")
        & (metrics["eval_task"] == "BFCL")
    ][["model", "layer", "value"]].rename(
        columns={"value": "math_bfcl_to_bfcl_test_auroc"}
    )

    merged = math_df.merge(transfer, on=["model", "layer"], how="inner").merge(
        bfcl, on=["model", "layer"], how="inner"
    ).merge(mixed, on=["model", "layer"], how="inner")
    merged = merged[merged["model"].isin(models)].copy()
    missing = merged[
        ["math_to_bfcl_test_auroc", "math_bfcl_to_bfcl_test_auroc", "bfcl_to_bfcl_test_auroc"]
    ].isna().any(axis=1)
    if missing.any():
        bad = merged.loc[missing, ["model", "layer"]].head(10).to_dict("records")
        raise ValueError(f"Missing matched-layer values for {bad}")
    return merged.sort_values(["model", "layer"]).reset_index(drop=True)


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in df.groupby("model"):
        item: dict[str, float | str] = {"model": model}
        for column, prefix in [
            ("math_test_auroc", "math"),
            ("math_to_bfcl_test_auroc", "math_to_bfcl"),
            ("math_bfcl_to_bfcl_test_auroc", "math_bfcl_to_bfcl"),
            ("bfcl_to_bfcl_test_auroc", "bfcl_to_bfcl"),
        ]:
            peak = group.loc[group[column].idxmax()]
            item[f"{prefix}_peak_auroc"] = float(peak[column])
            item[f"{prefix}_peak_layer"] = int(peak["layer"])
        rows.append(item)
    return pd.DataFrame(rows)


def plot_grid(df: pd.DataFrame, output_dir: Path, models: list[str]) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    value_columns = [column for column, _, _ in SERIES]
    fig, axes = plt.subplots(2, 3, figsize=(15.1, 8.4), sharey="row")
    axes_flat = axes.ravel()
    for ax, model in zip(axes_flat, models):
        group = df[df["model"] == model].sort_values("layer")
        for column, label, color in SERIES:
            ax.plot(
                group["layer"],
                group[column],
                label=label,
                color=color,
                marker="o",
                linewidth=2.0,
                markersize=4.0,
            )
            peak = group.loc[group[column].idxmax()]
            ax.scatter(
                peak["layer"],
                peak[column],
                color=color,
                s=64,
                marker="o",
                edgecolors="black",
                linewidths=0.8,
                zorder=5,
            )
        ax.axhline(0.5, color="#666666", linestyle="--", linewidth=0.9, dashes=(3.7, 1.6))
        ax.set_title(model, fontsize=11)
        ax.set_xlabel("Layer")
        ax.grid(alpha=0.25)

    for row_idx in range(axes.shape[0]):
        row_models = models[row_idx * axes.shape[1] : (row_idx + 1) * axes.shape[1]]
        row_values = df.loc[df["model"].isin(row_models), value_columns]
        ymin = max(0.0, float(row_values.min().min()) - 0.05)
        ymax = min(1.0, float(row_values.max().max()) + 0.05)
        for ax in axes[row_idx, :]:
            ax.set_ylim(ymin, ymax)

    for ax in axes[:, 0]:
        ax.set_ylabel("AUROC")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.suptitle("Layer-wise probe AUROC", fontsize=15, y=0.99)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.955),
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])

    output_base = figures_dir / "matched_layer_math_transfer_mixed_bfcltrained_auroc_grid"
    fig.savefig(output_base.with_suffix(".png"), dpi=300)
    fig.savefig(output_base.with_suffix(".pdf"))
    fig.savefig(output_base.with_suffix(".svg"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--models", type=parse_models, default=",".join(DEFAULT_MODELS))
    args = parser.parse_args()

    df = load_matched_dataframe(args.output_dir, args.models)
    out_csv = args.output_dir / "matched_layer_math_transfer_mixed_bfcltrained_auroc.csv"
    summary_csv = args.output_dir / "matched_layer_math_transfer_mixed_bfcltrained_summary.csv"
    df.to_csv(out_csv, index=False)
    build_summary(df).to_csv(summary_csv, index=False)
    plot_grid(df, args.output_dir, args.models)
    print(f"Saved matched-layer CSV, summary, and figure set under {args.output_dir}")


if __name__ == "__main__":
    main()
