"""Plot test Pareto frontiers for the Qwen3-4B ETD mean layer ensembles.

Compares the single best layer (23) against the 4 / 6 / 10 / all-37 layer
mean ensembles trained by ``run_layer_ensembles.sh`` (thesis Fig 4.12 and
B.20). Writes ``layer_ensemble_pareto_{condition}{_zoom}`` figures plus an
operating-point summary CSV.

Usage:
    python -m src.scripts.etd.plot_layer_ensemble_frontiers
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT_DIR = Path("outputs/figures/etd")
ENSEMBLE_ROOT = Path("outputs/etd/layer_ensembles/Qwen3-4B")
SINGLE_LAYER_ROOT = Path("outputs/etd/all_encoders_all_layers/Qwen3-4B/single_layer_routing")
SUMMARY_PATH = Path("outputs/etd/layer_ensembles/qwen3_4b_ensemble_test_summary.csv")

CONFIGURATIONS: dict[str, tuple[str, list[int]]] = {
    "Layer 23": ("single_23", [23]),
    "4 layers": ("four_even", [18, 24, 30, 36]),
    "6 layers": ("six", [21, 24, 27, 30, 33, 36]),
    "10 layers": ("ten", [9, 12, 15, 18, 21, 24, 27, 30, 33, 36]),
    "All 37": ("all", list(range(37))),
}
COLORS = {
    "pooled": {
        "Layer 23": "#a1d99b",
        "4 layers": "#66c2a4",
        "6 layers": "#31a354",
        "10 layers": "#006d2c",
        "All 37": "#00441b",
    },
    "english_transfer": {
        "Layer 23": "#fcbba1",
        "4 layers": "#fc9272",
        "6 layers": "#ef3b2c",
        "10 layers": "#a50f15",
        "All 37": "#67000d",
    },
}
LABEL_FONTSIZE = 16
TICK_FONTSIZE = 13
LEGEND_FONTSIZE = 11


def pareto(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values(
        ["mean_total_cost_usd", "mean_success"], ascending=[True, False]
    )
    keep: list[int] = []
    best_success = -np.inf
    for index, row in ordered.iterrows():
        success = float(row["mean_success"])
        if success > best_success + 1e-12:
            keep.append(index)
            best_success = success
    return ordered.loc[keep]


def grid_path(key: str, condition: str) -> Path:
    return (
        ENSEMBLE_ROOT
        / key
        / condition
        / "evaluation_relative_decision_cost/mean/raw_policy_grid_val_test.csv"
    )


def load_test_grid(condition: str, label: str, key: str) -> pd.DataFrame:
    if key == "single_23":
        frame = pd.read_csv(SINGLE_LAYER_ROOT / f"{condition}_all_single_layers_grid.csv")
        frame = frame[frame["layer"].eq(23)].copy()
    else:
        frame = pd.read_csv(grid_path(key, condition))
    frame = frame[frame["split"].eq("test")].copy()
    frame["configuration"] = label
    return frame


def summarize(frame: pd.DataFrame, condition: str, label: str, layers: list[int]) -> dict:
    anchor = float(
        (frame["mean_success"] - frame["success_delta_pp_vs_anchor"] / 100.0).median()
    )
    maximum = frame.sort_values(
        ["mean_success", "cost_savings_pct_vs_anchor"],
        ascending=[False, False],
        kind="mergesort",
    ).iloc[0]
    feasible = frame[frame["mean_success"] >= anchor - 1e-12]
    matched = None
    if not feasible.empty:
        matched = feasible.sort_values(
            ["cost_savings_pct_vs_anchor", "mean_success"],
            ascending=[False, False],
            kind="mergesort",
        ).iloc[0]
    return {
        "condition": condition,
        "configuration": label,
        "layers": ",".join(map(str, layers)),
        "anchor_success_pct": 100.0 * anchor,
        "max_success_pct": 100.0 * maximum["mean_success"],
        "savings_at_max_success_pct": maximum["cost_savings_pct_vs_anchor"],
        "matched_success_pct": np.nan if matched is None else 100.0 * matched["mean_success"],
        "matched_anchor_savings_pct": (
            np.nan if matched is None else matched["cost_savings_pct_vs_anchor"]
        ),
    }


def plot_condition(
    grids: dict[tuple[str, str], pd.DataFrame],
    summary: pd.DataFrame,
    condition: str,
    zoom: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(6.1, 4.4))
    anchor = float(
        summary.loc[summary["condition"].eq(condition), "anchor_success_pct"].iloc[0]
    )
    visible_costs: list[float] = []
    visible_successes: list[float] = []
    for label in CONFIGURATIONS:
        front = pareto(grids[(condition, label)])
        costs = 100.0 - front["cost_savings_pct_vs_anchor"]
        successes = 100.0 * front["mean_success"]
        ax.plot(
            costs,
            successes,
            color=COLORS[condition][label],
            linewidth=1.25,
            label=label,
        )
        visible = successes >= anchor - 0.03
        visible_costs.extend(costs[visible].tolist())
        visible_successes.extend(successes[visible].tolist())
    ax.scatter(100.0, anchor, marker="*", s=90, color="black", zorder=5, label="GPT-OSS high")
    if zoom:
        left_limit = max(0.0, min(visible_costs) - 1.5) if condition == "pooled" else 68.0
        lower_margin = 0.25 if condition == "pooled" else 4.0
        ax.set_xlim(left_limit, 101.5)
        ax.set_ylim(anchor - lower_margin, max(visible_successes) + 0.08)
    ax.set_xlabel("Cost relative to GPT-OSS high (%)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Test success (%)", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.grid(True, alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend_location = "upper left" if zoom else "lower right"
    ax.legend(loc=legend_location, frameon=False, fontsize=LEGEND_FONTSIZE)
    fig.tight_layout()
    zoom_suffix = "_zoom" if zoom else ""
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            OUTPUT_DIR / f"layer_ensemble_pareto_{condition}{zoom_suffix}.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    grids: dict[tuple[str, str], pd.DataFrame] = {}
    rows: list[dict] = []
    for condition in ("pooled", "english_transfer"):
        for label, (key, layers) in CONFIGURATIONS.items():
            frame = load_test_grid(condition, label, key)
            grids[(condition, label)] = frame
            rows.append(summarize(frame, condition, label, layers))

    summary = pd.DataFrame(rows)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)

    for condition in ("pooled", "english_transfer"):
        plot_condition(grids, summary, condition, zoom=False)
        plot_condition(grids, summary, condition, zoom=True)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
