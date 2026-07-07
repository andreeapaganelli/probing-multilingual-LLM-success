"""Evaluate and plot every Qwen3-4B ETD layer as a standalone predictor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.scripts.routing.evaluate_routing_policies import (
    add_input_pass_cost,
    add_relative_decision_cost,
    evaluate,
)
from src.scripts.etd.train_etd_probes import (
    ITEM_COLS,
    EncoderLayerStore,
    build_feature_matrix,
    safe_path_part,
)
from src.scripts.etd.train_layer_ensemble_probes import align_features


ANCHOR_MODEL = "gpt-oss-20b_high"
CONDITIONS = ("pooled", "english_transfer")
PLOT_EXCLUDE_LAYERS = (0,)


def pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    ordered = df.sort_values(["mean_total_cost_usd", "mean_success"], ascending=[True, False])
    keep = []
    best = -np.inf
    for idx, row in ordered.iterrows():
        success = float(row["mean_success"])
        if success > best + 1e-12:
            keep.append(idx)
            best = success
    return ordered.loc[keep]


def predict_layer(
    candidates: pd.DataFrame,
    activation_root: Path,
    condition_root: Path,
    encoder_model: str,
    layer: int,
) -> np.ndarray:
    store = EncoderLayerStore(activation_root, encoder_model, layer, cache_size=1)
    features, item_index = build_feature_matrix(candidates[ITEM_COLS], store)
    feature_rows = align_features(candidates, item_index)
    predictions = np.full(len(candidates), np.nan, dtype=np.float32)
    for target_model in sorted(candidates["model"].astype(str).unique()):
        mask = candidates["model"].astype(str).eq(target_model).to_numpy()
        bundle = joblib.load(
            condition_root
            / "probes"
            / f"layer_{layer:03d}"
            / f"{safe_path_part(target_model)}.joblib"
        )
        predictions[mask] = np.clip(
            bundle["probe"].predict(features[feature_rows[mask]]),
            0.0,
            1.0,
        )
    if np.isnan(predictions).any():
        raise ValueError(f"Layer {layer} predictions are incomplete")
    return predictions


def plot_condition(
    grid: pd.DataFrame,
    condition: str,
    encoder_model: str,
    max_layer: int,
    output_dir: Path,
    split: str = "test",
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    cmap = plt.get_cmap("viridis")
    min_layer = min(layer for layer in range(max_layer + 1) if layer not in PLOT_EXCLUDE_LAYERS)
    norm = plt.Normalize(min_layer, max_layer)
    selected_split = grid[
        grid["split"].eq(split) & ~grid["layer"].isin(PLOT_EXCLUDE_LAYERS)
    ]
    for layer, group in selected_split.groupby("layer", sort=True):
        frontier = pareto_front(group)
        ax.plot(
            100.0 - frontier["cost_savings_pct_vs_anchor"],
            frontier["mean_success"] * 100.0,
            color=cmap(norm(int(layer))),
            linewidth=0.85,
            alpha=0.82,
        )

    anchor_success = float(
        (
            selected_split["mean_success"]
            - selected_split["success_delta_pp_vs_anchor"] / 100.0
        ).median()
    )
    ax.scatter(100.0, anchor_success * 100.0, marker="*", s=145, color="black", zorder=5)
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.02)
    colorbar.set_label(f"{encoder_model} encoder layer", fontsize=16)
    colorbar.set_ticks(np.linspace(min_layer, max_layer, 7, dtype=int))
    colorbar.ax.tick_params(labelsize=13)
    ax.set_xlabel("Cost relative to GPT-OSS high (%)", fontsize=16)
    ax.set_ylabel(
        "Validation success (%)" if split == "val" else "Test success (%)",
        fontsize=16,
    )
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(True, alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    safe_encoder = encoder_model.lower().replace(".", "_").replace("-", "_")
    stem = f"{safe_encoder}_{condition}_all_single_layers_pareto"
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("outputs/etd/all_encoders_all_layers/Qwen3-4B"),
    )
    parser.add_argument("--encoder-model", default=None)
    parser.add_argument(
        "--parameter-grid-from",
        type=Path,
        default=Path(
            "outputs/routing/relative_decision_cost/frac_100/pooled/"
            "raw_policy_grid_val_test.csv"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/etd/all_encoders_all_layers/Qwen3-4B/single_layer_routing"),
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("outputs/figures/etd/all_encoders_all_layers/Qwen3-4B"),
    )
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    with (args.experiment_root / "pooled" / "run_config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    layers = [int(layer) for layer in config["layers"]]
    encoder_model = args.encoder_model or str(config["encoder_model"])
    activation_root = Path(config["activation_root"])
    candidates = pd.read_csv(config["source_candidates"]).reset_index(drop=True)
    reference_grid = pd.read_csv(args.parameter_grid_from)
    lambda_grid = sorted(reference_grid["lambda"].dropna().astype(float).unique())
    margin_grid = sorted(reference_grid["anchor_margin"].dropna().astype(float).unique())

    for condition in CONDITIONS:
        output_csv = args.output_root / f"{condition}_all_single_layers_grid.csv"
        if output_csv.exists():
            combined = pd.read_csv(output_csv)
            plot_condition(
                combined,
                condition,
                encoder_model,
                max(layers),
                args.figure_dir,
                args.split,
            )
            continue
        frames = []
        condition_root = args.experiment_root / condition
        for layer in layers:
            print(f"{condition}: evaluating layer {layer}", flush=True)
            layer_candidates = candidates.copy()
            layer_candidates["pred_success_raw"] = predict_layer(
                candidates,
                activation_root,
                condition_root,
                encoder_model,
                layer,
            )
            priced = add_input_pass_cost(layer_candidates, 0.0, 1.0 / 6.0)
            priced = add_relative_decision_cost(priced, ANCHOR_MODEL)
            grid = evaluate(
                priced,
                lambda_grid,
                margin_grid,
                anchor_model=ANCHOR_MODEL,
                encoder_model=encoder_model,
                output_cost_col="mean_output_cost_usd",
            )
            grid["condition"] = condition
            grid["layer"] = layer
            frames.append(grid)
        combined = pd.concat(frames, ignore_index=True)
        combined.to_csv(output_csv, index=False)
        plot_condition(
            combined,
            condition,
            encoder_model,
            max(layers),
            args.figure_dir,
            args.split,
        )

    with (args.output_root / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "source_experiment": str(args.experiment_root),
                "encoder_model": encoder_model,
                "layers": layers,
                "conditions": CONDITIONS,
                "policies": ["raw_utility", "raw_anchor"],
                "decision_cost": "training-mean one-generation cost relative to GPT-OSS high",
                "input_price_fraction": 1.0 / 6.0,
                "output_cost_col": "mean_output_cost_usd",
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
