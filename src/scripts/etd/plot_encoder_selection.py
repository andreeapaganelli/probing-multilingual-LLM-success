"""Plot the ETD encoder-selection Pareto frontiers (thesis Fig B.18-B.19).

Each encoder is shown at one selected layer, chosen from its per-layer routing
grids under ``outputs/etd/all_encoders_all_layers`` (written by
``run_all_encoders_all_layers.sh``):

  last            -- encoder's final layer (100% depth)
  matched_savings -- maximize cost savings subject to success >= anchor
  max_success     -- maximize success (layer 0 excluded)

The defaults reproduce the thesis figures ``etd_encoder_selection_validation``
and ``etd_encoder_selection_validation_zoom`` (pooled condition, max-success
layer selected on the validation split, both routing policies pooled).

Usage:
    python -m src.scripts.etd.plot_encoder_selection
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ALL_LAYERS_ROOT = Path("outputs/etd/all_encoders_all_layers")
FIGURE_ROOT = Path("outputs/figures/etd")

PALETTE = {
    "Qwen3-0.6B": "#0072B2",
    "Qwen3-1.7B": "#56B4E9",
    "Qwen3-4B": "#009E73",
    "Qwen3-8B": "#E69F00",
    "Qwen3.5-4B": "#CC79A7",
    "Qwen3.5-9B": "#D55E00",
    "gpt-oss-20b_high": "#000000",
    "gpt-oss-20b_low": "#7F7F7F",
}


def pareto_min_cost_max_success(df: pd.DataFrame) -> pd.DataFrame:
    """Return non-dominated rows for lower cost and higher success."""
    work = df.sort_values(
        ["mean_total_cost_usd", "mean_success"],
        ascending=[True, False],
        kind="mergesort",
    ).copy()
    best_so_far = -np.inf
    keep = []
    for row in work.itertuples():
        success = float(row.mean_success)
        keep.append(success > best_so_far + 1e-12)
        best_so_far = max(best_so_far, success)
    return work.loc[keep]


def anchor_success_rate(test: pd.DataFrame) -> float:
    return float((test["mean_success"] - test["success_delta_pp_vs_anchor"] / 100.0).iloc[0])


def summarize_layers(test: pd.DataFrame) -> pd.DataFrame:
    """One row per layer with best matched-savings and max-success hyperparameter points."""
    anchor = anchor_success_rate(test)
    rows: list[dict[str, object]] = []
    for layer, group in test.groupby("layer", sort=True):
        feasible = group[group["mean_success"] >= anchor - 1e-12]
        if feasible.empty:
            matched_row = None
            matched_savings = float("-inf")
        else:
            matched_row = feasible.sort_values(
                ["cost_savings_pct_vs_anchor", "mean_success"],
                ascending=[False, False],
                kind="mergesort",
            ).iloc[0]
            matched_savings = float(matched_row["cost_savings_pct_vs_anchor"])
        max_row = group.sort_values(
            ["mean_success", "cost_savings_pct_vs_anchor"],
            ascending=[False, False],
            kind="mergesort",
        ).iloc[0]
        rows.append(
            {
                "layer": int(layer),
                "matched_savings_pct": matched_savings,
                "matched_success": float(matched_row["mean_success"]) if matched_row is not None else np.nan,
                "max_success": float(max_row["mean_success"]),
                "max_success_delta_pp": float(max_row["success_delta_pp_vs_anchor"]),
            }
        )
    return pd.DataFrame(rows)


def pick_best_layer(summary: pd.DataFrame, criterion: str) -> int:
    if criterion == "last":
        return int(summary["layer"].max())
    if criterion == "matched_savings":
        picked = summary.sort_values(
            ["matched_savings_pct", "layer"],
            ascending=[False, True],
            kind="mergesort",
        ).iloc[0]
        return int(picked["layer"])
    if criterion == "max_success":
        filtered = summary[summary["layer"].ne(0)]
        if filtered.empty:
            raise ValueError("No non-zero layers available for max_success selection")
        picked = filtered.sort_values(
            ["max_success", "matched_savings_pct"],
            ascending=[False, False],
            kind="mergesort",
        ).iloc[0]
        return int(picked["layer"])
    raise ValueError(f"Unknown layer criterion: {criterion}")


def load_encoder_grid(path: Path, encoder: str, condition: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "encoder" not in df.columns:
        df.insert(0, "encoder", encoder)
    if "condition" not in df.columns:
        df.insert(1, "condition", condition)
    return df


def discover_encoder_grids(all_layers_root: Path, condition: str) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for encoder_dir in sorted(all_layers_root.iterdir()):
        if not encoder_dir.is_dir():
            continue
        grid = encoder_dir / "single_layer_routing" / f"{condition}_all_single_layers_grid.csv"
        if grid.exists():
            paths[encoder_dir.name] = grid
    if not paths:
        raise FileNotFoundError(f"No {condition} layer grids found under {all_layers_root}")
    return paths


def build_best_layer_selection(
    grid_paths: dict[str, Path],
    condition: str,
    criterion: str,
    policy: str,
    split: str = "test",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selection_rows: list[dict[str, object]] = []
    frontier_frames: list[pd.DataFrame] = []

    for encoder, path in grid_paths.items():
        grid = load_encoder_grid(path, encoder, condition)
        policy_mask = True if policy == "both_policies" else grid["policy"].eq(policy)
        selected_split = grid[grid["split"].eq(split) & policy_mask].copy()
        if selected_split.empty:
            raise ValueError(f"No {split} rows for encoder={encoder}, policy={policy}")

        summary = summarize_layers(selected_split)
        best_layer = pick_best_layer(summary, criterion)
        layer_summary = summary[summary["layer"].eq(best_layer)].iloc[0]
        layer_points = selected_split[selected_split["layer"].eq(best_layer)].copy()
        layer_points["selected_layer"] = best_layer
        frontier_frames.append(layer_points)

        with (path.parent / "run_config.json").open(encoding="utf-8") as handle:
            run_config = json.load(handle)
        layers = run_config.get("layers", [])
        depth_pct = round(100.0 * layers.index(best_layer) / max(len(layers) - 1, 1), 1) if layers else np.nan

        selection_rows.append(
            {
                "encoder": encoder,
                "condition": condition,
                "split": split,
                "policy": policy,
                "criterion": criterion,
                "best_layer": best_layer,
                "best_layer_depth_pct": depth_pct,
                "matched_savings_pct_at_layer": float(layer_summary["matched_savings_pct"]),
                "max_success_at_layer": float(layer_summary["max_success"]),
                "grid_path": str(path),
            }
        )

    selection = pd.DataFrame(selection_rows).sort_values("encoder")
    combined = pd.concat(frontier_frames, ignore_index=True)
    return selection, combined


def plot_best_layer_policy(
    df: pd.DataFrame,
    selection: pd.DataFrame,
    policy: str,
    output_dir: Path,
    output_stem: str,
    *,
    split: str = "test",
    zoom_interesting: bool = False,
) -> None:
    policy_mask = True if policy == "both_policies" else df["policy"].eq(policy)
    selected_split = df[df["split"].eq(split) & policy_mask].copy()
    if selected_split.empty:
        raise ValueError(f"No {split} rows for policy={policy}")

    layer_lookup = selection.set_index("encoder")["best_layer"].to_dict()
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    anchor_cost = float(
        (
            selected_split["mean_total_cost_usd"]
            / (1.0 - selected_split["cost_savings_pct_vs_anchor"] / 100.0)
        ).median()
    )
    for encoder, group in selected_split.groupby("encoder", sort=True):
        color = PALETTE.get(encoder, "0.5")
        layer = layer_lookup[encoder]
        frontier = pareto_min_cost_max_success(group)
        ax.plot(
            100.0 * frontier["mean_total_cost_usd"] / anchor_cost,
            frontier["mean_success"] * 100.0,
            marker="o",
            ms=3.2,
            lw=1.3,
            color=color,
            label=f"{encoder} (L{layer})",
        )

    anchor_success = float(
        (
            selected_split["mean_success"]
            - selected_split["success_delta_pp_vs_anchor"] / 100.0
        ).median()
    )
    ax.scatter(
        [100.0],
        [anchor_success * 100.0],
        marker="*",
        s=160,
        color="black",
        edgecolor="white",
        linewidth=0.7,
        zorder=11,
        label="Always gpt-oss high",
    )

    ax.set_xlabel("Cost relative to GPT-OSS high (%)")
    split_label = "Validation" if split == "val" else "Test"
    ax.set_ylabel(f"{split_label} success rate (%)")
    if zoom_interesting:
        max_success_pct = float(selected_split["mean_success"].max() * 100.0)
        y_min = anchor_success * 100.0 - 0.2
        y_max = max_success_pct + 0.15
        ax.set_ylim(y_min, y_max)
        visible = selected_split[
            (selected_split["mean_success"] * 100.0 >= y_min)
            & (selected_split["mean_success"] * 100.0 <= y_max)
        ]
        visible_x = 100.0 * visible["mean_total_cost_usd"] / anchor_cost
        x_min = min(float(visible_x.min()), 100.0)
        x_max = max(float(visible_x.max()), 100.0)
        x_pad = max((x_max - x_min) * 0.04, 0.1)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.grid(ls=":", lw=0.6, alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=7.0, framealpha=0.92, ncol=1, loc="upper left")
    fig.tight_layout()

    stem = f"{output_stem}_zoom" if zoom_interesting else output_stem
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"{stem}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)

    pareto_min_cost_max_success(selected_split).to_csv(
        output_dir / f"{stem}_frontier.csv",
        index=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-layers-root", type=Path, default=ALL_LAYERS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=FIGURE_ROOT)
    parser.add_argument("--condition", default="pooled", choices=["pooled", "english_transfer"])
    parser.add_argument(
        "--layer-criterion",
        default="max_success",
        choices=["last", "matched_savings", "max_success"],
    )
    parser.add_argument("--policy", default="both_policies")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--output-stem", default="etd_encoder_selection_validation")
    parser.add_argument(
        "--variants",
        default="default,zoom",
        help="Comma-separated plot variants: default (full range), zoom.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    grid_paths = discover_encoder_grids(args.all_layers_root, args.condition)
    variants = {v.strip() for v in args.variants.split(",") if v.strip()}

    selection, combined = build_best_layer_selection(
        grid_paths,
        args.condition,
        args.layer_criterion,
        args.policy,
        args.split,
    )
    selection_path = args.output_dir / f"{args.output_stem}_selected_layers.csv"
    selection.to_csv(selection_path, index=False)
    print(f"Wrote {selection_path}")
    print(
        selection[
            ["encoder", "best_layer", "best_layer_depth_pct", "matched_savings_pct_at_layer"]
        ].to_string(index=False)
    )

    if "default" in variants:
        plot_best_layer_policy(
            combined,
            selection,
            args.policy,
            args.output_dir,
            args.output_stem,
            split=args.split,
            zoom_interesting=False,
        )
        print(f"Wrote {args.output_dir / f'{args.output_stem}.png'}")
    if "zoom" in variants:
        plot_best_layer_policy(
            combined,
            selection,
            args.policy,
            args.output_dir,
            args.output_stem,
            split=args.split,
            zoom_interesting=True,
        )
        print(f"Wrote {args.output_dir / f'{args.output_stem}_zoom.png'}")


if __name__ == "__main__":
    main()
