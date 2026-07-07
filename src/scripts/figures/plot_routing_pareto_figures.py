"""Rebuild the routing accuracy–cost figures from the canonical policy grids.

Produces, under outputs/figures/routing/:
  routing_success_by_depth.{pdf,png}        max validation routing success vs. layer depth
  routing_success_by_depth_anchor.{pdf,png} same, anchor-aware policy
  pareto_direct_routing.{pdf,png}           direct probe routing accuracy–cost trade-off
  pareto_etd_by_layer_{condition}.{pdf,png} validation frontiers at every relative depth
  pareto_etd_routing.{pdf,png}              ETD routing (both policies), Qwen3-4B encoder
  pareto_etd_anchor_policy.{pdf,png}        ETD routing, anchor policy only
  pareto_direct_vs_etd_pooled.{pdf,png}     direct pooled vs. pooled ETD comparison

Inputs:
  outputs/routing/relative_decision_cost/frac_XXX/{pooled,english_transfer}/raw_policy_grid_val_test.csv
  outputs/etd/all_encoders_all_layers/Qwen3-4B/single_layer_routing/{pooled,english_transfer}_all_single_layers_grid.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.scripts.figures.plot_routing_pareto import (
    COLOR_ETD_ENGLISH,
    COLOR_ETD_POOLED,
    COLOR_POOLED,
    plot_all_depth_pareto,
    plot_auroc_and_routing_by_depth,
    plot_fixed_layer_both_policies,
)


DEPTH_ROOT = Path("outputs/routing/relative_decision_cost")
ETD_ROOT = Path("outputs/etd")
SINGLE_LAYER_ROOT = ETD_ROOT / "all_encoders_all_layers" / "Qwen3-4B" / "single_layer_routing"
COMBINED = ETD_ROOT / "qwen3_4b_both_conditions_grid.csv"
DEFAULT_POOLED_LAYER = 23
DEFAULT_ENGLISH_LAYER = 23
DEFAULT_ORIGINAL_POOLED_FRAC = 100


def load_depth_grid() -> pd.DataFrame:
    frames = []
    for frac in range(0, 101, 5):
        for condition in ("pooled", "english_transfer"):
            path = DEPTH_ROOT / f"frac_{frac:03d}" / condition / "raw_policy_grid_val_test.csv"
            frame = pd.read_csv(path)
            frame["condition"] = condition
            frame["target_frac"] = frac
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_single_layer_grid(path: Path, condition: str, layer: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[frame["layer"].eq(layer)].copy()
    if frame.empty:
        raise ValueError(f"No rows for layer {layer} in {path}")
    frame["condition"] = condition
    frame["target_frac"] = layer
    frame["encoder_layer"] = layer
    return frame.drop(columns=["layer"])


def load_original_pooled_final_layer(depth_grid: pd.DataFrame) -> pd.DataFrame:
    return depth_grid[
        depth_grid["condition"].eq("pooled")
        & depth_grid["target_frac"].eq(DEFAULT_ORIGINAL_POOLED_FRAC)
    ].copy()


def load_pooled_etd_for_comparison(layer: int) -> pd.DataFrame:
    frame = load_single_layer_grid(
        SINGLE_LAYER_ROOT / "pooled_all_single_layers_grid.csv",
        "pooled",
        layer,
    )
    frame["condition"] = "english_transfer"
    frame["target_frac"] = layer
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pooled-layer", type=int, default=DEFAULT_POOLED_LAYER)
    parser.add_argument("--english-layer", type=int, default=DEFAULT_ENGLISH_LAYER)
    parser.add_argument(
        "--skip-depth-figures",
        action="store_true",
        help="Skip the direct-routing depth figures (needs the full frac_000..100 sweep).",
    )
    args = parser.parse_args()

    # ── Direct probe routing (relative-depth sweep) ──────────────────────────
    if not args.skip_depth_figures:
        direct = load_depth_grid()
        plot_auroc_and_routing_by_depth(
            direct,
            "raw_utility",
            "routing_success_by_depth",
            display_split="val",
            selection_split="val",
        )
        plot_auroc_and_routing_by_depth(
            direct,
            "raw_anchor",
            "routing_success_by_depth_anchor",
            display_split="val",
            selection_split="val",
        )
        plot_fixed_layer_both_policies(
            direct,
            output_stem="pareto_direct_routing",
            pooled_label="Pooled multilingual probe (100% depth)",
            english_label="English-only transfer probe (70% depth)",
        )
        for condition in ("pooled", "english_transfer"):
            plot_all_depth_pareto(
                direct,
                condition,
                output_stem=f"pareto_etd_by_layer_{condition}",
                split="val",
            )

    # ── Encoder-target decoupled routing (Qwen3-4B encoder) ─────────────────
    pooled = load_single_layer_grid(
        SINGLE_LAYER_ROOT / "pooled_all_single_layers_grid.csv",
        "pooled",
        args.pooled_layer,
    )
    english = load_single_layer_grid(
        SINGLE_LAYER_ROOT / "english_transfer_all_single_layers_grid.csv",
        "english_transfer",
        args.english_layer,
    )
    combined = pd.concat([pooled, english], ignore_index=True)
    COMBINED.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(COMBINED, index=False)

    common = {
        "grid": combined,
        "pooled_frac": args.pooled_layer,
        "english_frac": args.english_layer,
        "pooled_label": "Pooled ETD",
        "english_label": "English-transfer ETD",
        "pooled_color": COLOR_ETD_POOLED,
        "english_color": COLOR_ETD_ENGLISH,
    }
    plot_fixed_layer_both_policies(
        **common,
        output_stem="pareto_etd_routing",
        legend_loc="lower right",
    )
    plot_fixed_layer_both_policies(
        **common,
        output_stem="pareto_etd_anchor_policy",
        policies=("raw_anchor",),
        legend_loc="lower right",
    )

    if not args.skip_depth_figures:
        comparison = pd.concat(
            [
                load_original_pooled_final_layer(direct),
                load_pooled_etd_for_comparison(args.pooled_layer),
            ],
            ignore_index=True,
        )
        plot_fixed_layer_both_policies(
            comparison,
            output_stem="pareto_direct_vs_etd_pooled",
            pooled_frac=DEFAULT_ORIGINAL_POOLED_FRAC,
            english_frac=args.pooled_layer,
            pooled_label="Original pooled (target-model encoders, final layer)",
            english_label=f"Pooled ETD (Qwen3-4B encoder, layer {args.pooled_layer}/36)",
            pooled_color=COLOR_POOLED,
            english_color=COLOR_ETD_POOLED,
            legend_loc="lower right",
        )

    print(
        f"Wrote {COMBINED} "
        f"(pooled layer {args.pooled_layer}, english layer {args.english_layer})"
    )


if __name__ == "__main__":
    main()
