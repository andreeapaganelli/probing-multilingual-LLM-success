"""Plotting library + CLI for the relative-depth routing Pareto figures.

Provides the plotting functions used by plot_routing_pareto_figures.py
(``plot_fixed_layer_both_policies``, ``plot_auroc_and_routing_by_depth``,
``plot_all_depth_pareto``, ...). Running this module directly plots the
layer-sweep summary figures under outputs/figures/routing/.

Usage (from repo root):
    python -m src.scripts.figures.plot_routing_pareto
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]

REPO_ROOT = Path(__file__).resolve().parents[3]
SWEEP_ROOT = REPO_ROOT / "outputs/routing/depth_sweep"
OUTPUT_DIR = REPO_ROOT / "outputs/figures/routing"

ANCHOR_SUCCESS = 0.8593
ANCHOR_LABEL = "gpt-oss-20b_high (anchor)"

FRACS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
SWEEP_IDS = [f"frac_{f:03d}" for f in FRACS]

# Colorblind-safe palette
COLOR_POOLED = "#1f77b4"
COLOR_ENGLISH = "#ff7f0e"
COLOR_ETD_POOLED = "#2ca02c"
COLOR_ETD_ENGLISH = "#d62728"
COLOR_UTILITY = "#0072B2"    # blue = raw_utility (no anchor stickiness)
COLOR_ANCHOR  = "#009E73"    # green = raw_anchor  (anchor stickiness)


# ── Data loading ─────────────────────────────────────────────────────────────

def load_summary() -> pd.DataFrame:
    return pd.read_csv(SWEEP_ROOT / "layer_sweep_summary.csv")


def load_policy_grid(sweep_id: str) -> pd.DataFrame:
    path = SWEEP_ROOT / "final" / sweep_id / "policy_grid_val_test.csv"
    df = pd.read_csv(path)
    df["sweep_id"] = sweep_id
    df["target_frac"] = int(sweep_id.split("_")[1])
    return df


def load_all_grids() -> pd.DataFrame:
    frames = [load_policy_grid(sid) for sid in SWEEP_IDS
              if (SWEEP_ROOT / "final" / sid / "policy_grid_val_test.csv").exists()]
    return pd.concat(frames, ignore_index=True)


def _policy_param(policy: str) -> str:
    """Return the hyperparameter column name swept for a given policy."""
    return "anchor_margin" if policy == "raw_anchor" else "lambda"


def val_matched_test(grid: pd.DataFrame, condition: str, policy: str) -> pd.DataFrame:
    """For each sweep point, find the val-best hyperparameter and return the test row."""
    param = _policy_param(policy)
    rows = []
    for frac, gdf in grid[grid["condition"] == condition].groupby("target_frac"):
        val = gdf[(gdf["split"] == "val") & (gdf["policy"] == policy)]
        test = gdf[(gdf["split"] == "test") & (gdf["policy"] == policy)]
        if val.empty or test.empty:
            continue
        best_param = val.loc[val["mean_success"].idxmax(), param]
        matched = test[test[param] == best_param]
        if matched.empty:
            continue
        r = matched.iloc[0].to_dict()
        r["target_frac"] = frac
        rows.append(r)
    return pd.DataFrame(rows).sort_values("target_frac")


# ── Pareto frontier helper ────────────────────────────────────────────────────

def pareto_frontier(df: pd.DataFrame) -> pd.DataFrame:
    """Non-dominated points maximising both success and savings."""
    best = (df.groupby("cost_savings_pct_vs_anchor", as_index=False)["mean_success"]
              .max()
              .sort_values("cost_savings_pct_vs_anchor", ascending=False))
    cum_max = best["mean_success"].cummax().shift(fill_value=-np.inf)
    return best[best["mean_success"] > cum_max].sort_values("cost_savings_pct_vs_anchor")


# ── Figure 1 & 2: line plots vs. depth ───────────────────────────────────────

def plot_vs_depth(summary: pd.DataFrame) -> None:
    fracs = summary["target_frac"].values * 100  # → 0..100

    pooled_suc = summary["pooled__val_match_test_success"].values * 100
    eng_suc = summary["english_transfer__best_test_success"].values * 100
    pooled_sav = summary["pooled__val_match_test_savings"].values
    eng_sav = summary["english_transfer__best_test_savings"].values

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # — success —
    ax = axes[0]
    ax.plot(fracs, pooled_suc, "o-", color=COLOR_POOLED, lw=1.8, ms=4,
            label="Pooled (val-matched)")
    ax.plot(fracs, eng_suc, "s--", color=COLOR_ENGLISH, lw=1.8, ms=4,
            label="English transfer (best test)")
    ax.axhline(ANCHOR_SUCCESS * 100, ls=":", color="black", lw=1.2, alpha=0.6,
               label=f"Anchor ({ANCHOR_SUCCESS*100:.1f}%)")
    ax.set_xlabel("Layer depth (%)")
    ax.set_ylabel("Test success rate (%)")
    ax.set_title("Success rate vs. layer depth")
    ax.set_xlim(-2, 102)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(axis="y", ls=":", alpha=0.4)

    # — savings —
    ax = axes[1]
    ax.plot(fracs, pooled_sav, "o-", color=COLOR_POOLED, lw=1.8, ms=4,
            label="Pooled (val-matched)")
    ax.plot(fracs, eng_sav, "s--", color=COLOR_ENGLISH, lw=1.8, ms=4,
            label="English transfer (best test)")
    ax.set_xlabel("Layer depth (%)")
    ax.set_ylabel("Cost savings vs. anchor (%)")
    ax.set_title("Cost savings vs. layer depth")
    ax.set_xlim(-2, 102)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(axis="y", ls=":", alpha=0.4)

    fig.suptitle("Routing performance across layer depth (5% intervals)", y=1.01, fontsize=11)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUTPUT_DIR / f"success_savings_vs_depth.{ext}",
                    bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  saved success_savings_vs_depth.{{pdf,png}}")


# ── Shared plot helpers ───────────────────────────────────────────────────────

# Six representative depths with distinct, colorblind-friendly named colors
REP_FRACS = [0, 20, 40, 60, 80, 100]
REP_COLORS = {
    0:   "#333333",   # near-black
    20:  "#E69F00",   # orange
    40:  "#56B4E9",   # sky blue
    60:  "#009E73",   # green
    80:  "#CC79A7",   # pink
    100: "#D55E00",   # vermillion
}
REP_LABELS = {f: f"{f}% depth" for f in REP_FRACS}
REP_LABELS[0]   = "0% (first layer)"
REP_LABELS[100] = "100% (last layer)"


def _style_ax(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(ls=":", lw=0.6, alpha=0.45, color="0.65")
    ax.set_axisbelow(True)


def _draw_pareto_frontier(ax, matched_rows: pd.DataFrame,
                          color: str = "0.55", lw: float = 1.2,
                          label: str = "_nolegend_") -> None:
    if matched_rows.empty:
        return
    sav = matched_rows["cost_savings_pct_vs_anchor"].values
    suc = matched_rows["mean_success"].values
    pf  = pareto_frontier(pd.DataFrame({"cost_savings_pct_vs_anchor": sav, "mean_success": suc}))
    if len(pf) > 1:
        ax.plot(pf["cost_savings_pct_vs_anchor"], pf["mean_success"] * 100,
                "--", color=color, lw=lw, zorder=3, label=label, alpha=0.7)


def _draw_rep_points(ax, matched_rows: pd.DataFrame, marker: str = "o",
                     zorder: int = 5, add_legend: bool = True) -> None:
    """Draw background grey dots for all depths, then overlay the 6 rep depths."""
    if matched_rows.empty:
        return
    fracs = matched_rows["target_frac"].values
    suc   = matched_rows["mean_success"].values * 100
    sav   = matched_rows["cost_savings_pct_vs_anchor"].values

    # Background: all non-representative depths as small grey dots
    for f, x, y in zip(fracs, sav, suc):
        if int(f) not in REP_FRACS:
            ax.scatter(x, y, s=18, color="0.78", marker=marker,
                       zorder=2, linewidths=0)

    # Foreground: representative depths with distinct colors
    for f, x, y in zip(fracs, sav, suc):
        f_int = int(f)
        if f_int in REP_FRACS:
            lbl = REP_LABELS[f_int] if add_legend else "_nolegend_"
            ax.scatter(x, y, s=80, color=REP_COLORS[f_int], marker=marker,
                       zorder=zorder, linewidths=0.7, edgecolors="white",
                       label=lbl)


# ── Figure: per-condition layer-depth curves (original style) ────────────────

def plot_layer_curves(grid: pd.DataFrame, condition: str,
                      condition_label: str, policy: str,
                      output_stem: str) -> None:
    """Replicate the original accuracy_cost_scatter style: one curve per representative
    depth, sweeping all lambda values, x = cost ratio vs anchor (%)."""
    test = grid[
        (grid["condition"] == condition) &
        (grid["split"] == "test") &
        (grid["policy"] == policy)
    ].copy()
    test["cost_ratio"] = 100.0 - test["cost_savings_pct_vs_anchor"]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    _style_ax(ax)

    # Draw non-representative depths first as light grey background
    for frac, gdf in test.groupby("target_frac"):
        if int(frac) not in REP_FRACS:
            gdf_s = gdf.sort_values("cost_ratio")
            ax.scatter(gdf_s["cost_ratio"], gdf_s["mean_success"],
                       s=4, color="0.82", alpha=0.35, linewidths=0, zorder=1)

    # Draw representative depths with distinct colors
    for frac in REP_FRACS:
        gdf = test[test["target_frac"] == frac].sort_values("cost_ratio")
        if gdf.empty:
            continue
        ax.scatter(gdf["cost_ratio"], gdf["mean_success"],
                   s=7, color=REP_COLORS[frac], alpha=0.55,
                   linewidths=0, zorder=3, label=REP_LABELS[frac])

    # Always-anchor reference
    ax.scatter([100.0], [ANCHOR_SUCCESS], marker="x", color="black",
               s=80, linewidths=2, zorder=5, label="Always anchor")

    pol_label = "No anchor stickiness" if policy == "raw_utility" else "With anchor stickiness"
    ax.set_xlabel("Cost ratio vs. anchor (%)", fontsize=10)
    ax.set_ylabel("Average empirical success", fontsize=10)
    ax.set_title(f"{condition_label} — {pol_label}", fontsize=10, pad=8)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout()

    for ext in ("pdf", "png"):
        fig.savefig(OUTPUT_DIR / f"{output_stem}.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  saved {output_stem}.{{pdf,png}}")


def plot_combined_comparison(grid: pd.DataFrame, policy: str,
                             output_stem: str = "pareto_combined") -> None:
    """The main thesis figure: pooled (left, tight cluster) vs English transfer
    (right, spread curves) — same style as the original liked plot."""
    conditions = [
        ("pooled",           "Pooled multilingual probe"),
        ("english_transfer", "English-only transfer probe"),
    ]

    test = grid[
        (grid["split"] == "test") &
        (grid["policy"] == policy)
    ].copy()
    test["cost_ratio"] = 100.0 - test["cost_savings_pct_vs_anchor"]

    # Shared y-axis range across both conditions
    y_lo = test["mean_success"].min() - 0.005
    y_hi = max(test["mean_success"].max(), ANCHOR_SUCCESS) + 0.005

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), sharey=True)
    fig.subplots_adjust(wspace=0.06)

    for i, (ax, (condition, cond_label)) in enumerate(zip(axes, conditions)):
        sub = test[test["condition"] == condition]
        _style_ax(ax)

        # Grey background: all non-representative depths
        for frac, gdf in sub.groupby("target_frac"):
            if int(frac) not in REP_FRACS:
                ax.scatter(gdf.sort_values("cost_ratio")["cost_ratio"],
                           gdf.sort_values("cost_ratio")["mean_success"],
                           s=4, color="0.82", alpha=0.35, linewidths=0, zorder=1)

        # Coloured representative depths
        for frac in REP_FRACS:
            gdf = sub[sub["target_frac"] == frac].sort_values("cost_ratio")
            if gdf.empty:
                continue
            lbl = REP_LABELS[frac] if i == 0 else "_nolegend_"
            ax.scatter(gdf["cost_ratio"], gdf["mean_success"],
                       s=7, color=REP_COLORS[frac], alpha=0.60,
                       linewidths=0, zorder=3, label=lbl)

        # Always-anchor marker
        ax.scatter([100.0], [ANCHOR_SUCCESS], marker="x", color="black",
                   s=100, linewidths=2.2, zorder=5,
                   label="Always anchor" if i == 0 else "_nolegend_")

        ax.set_xlabel("Cost ratio vs. anchor (%)", fontsize=10.5)
        ax.set_title(cond_label, fontsize=11, pad=8)
        ax.set_ylim(y_lo, y_hi)
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Average empirical success", fontsize=10.5)
    axes[0].legend(frameon=False, fontsize=9, loc="lower right",
                   handletextpad=0.3, labelspacing=0.35)

    pol_label = "no anchor stickiness" if policy == "raw_utility" else "with anchor stickiness"
    fig.suptitle(f"Routing performance across layer depth ({pol_label})",
                 fontsize=11.5, y=1.02)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUTPUT_DIR / f"{output_stem}.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  saved {output_stem}.{{pdf,png}}")


# ── Figure: policy comparison — line plots ───────────────────────────────────

def plot_policy_comparison(grid: pd.DataFrame) -> None:
    """4-panel figure: success & savings vs depth for both conditions, both policies."""
    conditions = [
        ("pooled",           "Pooled multilingual"),
        ("english_transfer", "English transfer"),
    ]
    policies = [
        ("raw_utility", "No anchor stickiness", COLOR_UTILITY, "o-"),
        ("raw_anchor",  "With anchor stickiness", COLOR_ANCHOR,  "s--"),
    ]
    col_titles = ["Success rate (%)", "Cost savings (%)"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True)
    fig.subplots_adjust(hspace=0.35, wspace=0.28)

    for row, (cond, cond_label) in enumerate(conditions):
        for col, (metric_label, ylabel) in enumerate([
            ("test_success", "Success rate (%)"),
            ("test_savings", "Cost savings (%)"),
        ]):
            ax = axes[row, col]
            _style_ax(ax)
            for policy, pol_label, color, ls in policies:
                matched = val_matched_test(grid, cond, policy)
                if matched.empty:
                    continue
                fracs = matched["target_frac"].values
                vals = (matched["mean_success"].values * 100
                        if metric_label == "test_success"
                        else matched["cost_savings_pct_vs_anchor"].values)
                ax.plot(fracs, vals, ls, color=color, lw=1.8, ms=4.5, label=pol_label)

            if metric_label == "test_success":
                ax.axhline(ANCHOR_SUCCESS * 100, ls=":", color="0.55", lw=1.1, alpha=0.8)

            ax.set_xlim(-2, 102)
            ax.set_xticks([0, 25, 50, 75, 100])
            ax.set_ylabel(ylabel, fontsize=9)
            if row == 0:
                ax.set_title(col_titles[col], fontsize=10, pad=5)
            if row == 1:
                ax.set_xlabel("Layer depth (%)", fontsize=9)

            # Row label on left column only
            if col == 0:
                ax.set_ylabel(f"{cond_label}\n{ylabel}", fontsize=9)

            if row == 0 and col == 0:
                ax.legend(fontsize=8.5, framealpha=0.9)

    fig.suptitle("Routing policy comparison across layer depth", fontsize=11, y=1.02)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUTPUT_DIR / f"policy_comparison_vs_depth.{ext}",
                    bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("  saved policy_comparison_vs_depth.{pdf,png}")


# ── Figure: policy comparison — side-by-side Pareto panels ───────────────────

def plot_policy_pareto_comparison(grid: pd.DataFrame, condition: str,
                                   condition_label: str, output_stem: str) -> None:
    """Side-by-side panels (one per policy), original scatter-curve style."""
    policies = [
        ("raw_utility", "No anchor stickiness"),
        ("raw_anchor",  "With anchor stickiness"),
    ]

    test = grid[
        (grid["condition"] == condition) &
        (grid["split"] == "test") &
        (grid["policy"].isin(["raw_utility", "raw_anchor"]))
    ].copy()
    test["cost_ratio"] = 100.0 - test["cost_savings_pct_vs_anchor"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    fig.subplots_adjust(wspace=0.06)

    for i, (ax, (policy, pol_label)) in enumerate(zip(axes, policies)):
        sub = test[test["policy"] == policy]
        _style_ax(ax)

        # Grey background: non-representative depths
        for frac, gdf in sub.groupby("target_frac"):
            if int(frac) not in REP_FRACS:
                ax.scatter(gdf.sort_values("cost_ratio")["cost_ratio"],
                           gdf.sort_values("cost_ratio")["mean_success"],
                           s=4, color="0.82", alpha=0.35, linewidths=0, zorder=1)

        # Coloured representative depths
        for frac in REP_FRACS:
            gdf = sub[sub["target_frac"] == frac].sort_values("cost_ratio")
            if gdf.empty:
                continue
            lbl = REP_LABELS[frac] if i == 0 else "_nolegend_"
            ax.scatter(gdf["cost_ratio"], gdf["mean_success"],
                       s=7, color=REP_COLORS[frac], alpha=0.55,
                       linewidths=0, zorder=3, label=lbl)

        ax.scatter([100.0], [ANCHOR_SUCCESS], marker="x", color="black",
                   s=80, linewidths=2, zorder=5,
                   label="Always anchor" if i == 0 else "_nolegend_")

        ax.set_xlabel("Cost ratio vs. anchor (%)", fontsize=10)
        ax.set_title(pol_label, fontsize=10, pad=6)
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Average empirical success", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8.5, loc="lower right")

    fig.suptitle(condition_label, fontsize=11, y=1.01)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUTPUT_DIR / f"{output_stem}.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  saved {output_stem}.{{pdf,png}}")


# ── Figure 5: test-matched savings (pooled) ──────────────────────────────────

def plot_test_matched_savings(grid: pd.DataFrame) -> None:
    """Bar chart: savings while matching anchor on test, pooled only."""
    rows = []
    for frac, gdf in grid[(grid["condition"] == "pooled") &
                           (grid["split"] == "test")].groupby("target_frac"):
        for policy in ("raw_utility", "raw_anchor"):
            sub = gdf[gdf["policy"] == policy]
            above = sub[sub["mean_success"] >= ANCHOR_SUCCESS]
            if above.empty:
                continue
            best = above.loc[above["cost_savings_pct_vs_anchor"].idxmax()]
            rows.append({"frac": frac, "policy": policy,
                         "savings": best["cost_savings_pct_vs_anchor"],
                         "success": best["mean_success"]})
    if not rows:
        return
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    x = np.arange(len(FRACS))
    w = 0.38
    for i, (policy, label, color) in enumerate([
        ("raw_utility", "No anchor (raw utility)", COLOR_POOLED),
        ("raw_anchor",  "With anchor stickiness", "#009E73"),
    ]):
        sub = df[df["policy"] == policy].set_index("frac")
        heights = [sub.loc[f, "savings"] if f in sub.index else 0 for f in FRACS]
        ax.bar(x + (i - 0.5) * w, heights, w, color=color, alpha=0.85, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{f}%" for f in FRACS], fontsize=7.5, rotation=45)
    ax.set_ylabel("Cost savings vs. anchor (%)")
    ax.set_title("Pooled probe: savings while matching anchor success (test)\n"
                 "Zero bars = layer did not achieve anchor-level success on test")
    ax.legend(fontsize=8)
    ax.grid(axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUTPUT_DIR / f"pooled_test_matched_savings.{ext}",
                    bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("  saved pooled_test_matched_savings.{pdf,png}")



# ── AUROC trajectory + routing success line plot ──────────────────────────────

def plot_auroc_and_routing_by_depth(grid: pd.DataFrame,
                                    policy: str = "raw_utility",
                                    output_stem: str = "auroc_routing_by_depth",
                                    display_split: str = "test",
                                    selection_split: str = "val") -> None:
    """Single-panel figure: val-matched routing success vs. layer depth,
    with zone shading (early / mid-to-late / last layers).
    Shows pooled probe as a flat reference and English transfer as the main line.

    X-axis: relative layer depth, excluding layer 0 (100% = last layer).
    Y-axis: success rate on ``display_split`` at the threshold selected on
    ``selection_split``.
    """
    from scipy.ndimage import uniform_filter1d

    param = _policy_param(policy)

    plot_grid = grid[grid["target_frac"].ne(0)].copy()
    rows = []
    for frac, gdf in plot_grid[plot_grid["split"] == display_split].groupby("target_frac"):
        for cond in ("pooled", "english_transfer"):
            selection_g = plot_grid[
                (plot_grid["target_frac"] == frac)
                & (plot_grid["split"] == selection_split)
                & (plot_grid["condition"] == cond)
                & (plot_grid["policy"] == policy)
            ]
            display_g = gdf[(gdf["condition"] == cond) & (gdf["policy"] == policy)]
            if selection_g.empty or display_g.empty:
                continue
            pick = selection_g.loc[selection_g["mean_success"].idxmax()]
            m = display_g[np.isclose(display_g["lambda"], float(pick["lambda"]))]
            if policy == "raw_anchor":
                m = m[np.isclose(m["anchor_margin"], float(pick["anchor_margin"]))]
            if m.empty:
                continue
            rows.append({"frac": frac, "condition": cond,
                         "success": m.iloc[0]["mean_success"] * 100})
    sdf = pd.DataFrame(rows)

    fracs = sorted(sdf["frac"].unique())
    eng   = sdf[sdf["condition"] == "english_transfer"].set_index("frac")["success"]
    pool  = sdf[sdf["condition"] == "pooled"].set_index("frac")["success"]
    eng_vals  = [eng.get(f, float("nan")) for f in fracs]
    pool_vals = [pool.get(f, float("nan")) for f in fracs]

    eng_arr = np.array(eng_vals, dtype=float)
    valid = ~np.isnan(eng_arr)
    eng_smooth = np.full_like(eng_arr, float("nan"))
    eng_smooth[valid] = uniform_filter1d(eng_arr[valid], size=5)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    _style_ax(ax)

    ax.axvspan(55, 90, color="0.96", zorder=0)
    for pct in (55, 90):
        ax.axvline(pct, color="0.55", ls="--", lw=1.0, zorder=2)

    split_grid = plot_grid[plot_grid["split"].eq(display_split)]
    anchor_success = float(
        (
            split_grid["mean_success"]
            - split_grid["success_delta_pp_vs_anchor"] / 100.0
        ).median()
    )
    ax.axhline(anchor_success * 100, color="0.4", ls=":", lw=1.1, zorder=2,
               label=f"Always anchor ({anchor_success*100:.1f}%)")

    ax.plot(fracs, pool_vals, "o-", color=COLOR_POOLED, lw=1.8, ms=4,
            alpha=0.80, zorder=4, label="Pooled multilingual probe")

    # English transfer — connected raw dots + smoothed trend on top
    ax.plot(fracs, eng_vals, "o-", color=COLOR_ENGLISH, lw=1.0, ms=4,
            alpha=0.45, zorder=3)
    ax.plot(fracs, eng_smooth, "-", color=COLOR_ENGLISH, lw=2.4, zorder=5,
            label="English-only transfer probe")

    ax.set_xlabel("Layer depth (%)", fontsize=10.5, labelpad=2)
    if selection_split == display_split:
        split_label = "Validation" if display_split == "val" else "Test"
        ylabel = f"Maximum {split_label.lower()} success (%)"
    else:
        ylabel = "Val-matched test success (%)"
    ax.set_ylabel(ylabel, fontsize=10.5)
    ax.set_title("Routing success by layer depth", fontsize=10.5, visible=False)
    ax.set_xlim(3, 102)
    ax.set_ylim(60, 88.5)
    ax.set_xticks([5, 25, 50, 75, 100])
    ax.set_xticklabels(["5%", "25%", "50%", "75%", "100%"], fontsize=9)
    ax.legend(frameon=True, framealpha=0.9, edgecolor="none", fontsize=9, loc="lower right")
    ax.grid(alpha=0.25, linestyle="-")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUTPUT_DIR / f"{output_stem}.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  saved {output_stem}.{{pdf,png}}")


def plot_all_depth_pareto(
    grid: pd.DataFrame,
    condition: str,
    output_stem: str,
    split: str = "val",
) -> None:
    """Plot one combined-policy Pareto curve per relative layer depth."""
    selected = grid[
        grid["condition"].eq(condition)
        & grid["split"].eq(split)
        & grid["policy"].isin(["raw_utility", "raw_anchor"])
        & grid["target_frac"].ne(0)
    ].copy()
    if selected.empty:
        raise ValueError(f"No rows for condition={condition}, split={split}")

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    cmap = plt.get_cmap("viridis")
    min_depth = float(selected["target_frac"].min())
    max_depth = float(selected["target_frac"].max())
    norm = plt.Normalize(min_depth, max_depth)
    for frac, group in selected.groupby("target_frac", sort=True):
        frontier = pareto_frontier(group)
        ax.plot(
            100.0 - frontier["cost_savings_pct_vs_anchor"],
            100.0 * frontier["mean_success"],
            color=cmap(norm(float(frac))),
            linewidth=0.9,
            alpha=0.85,
        )

    anchor_success = float(
        (
            selected["mean_success"]
            - selected["success_delta_pp_vs_anchor"] / 100.0
        ).median()
    )
    ax.scatter(100.0, 100.0 * anchor_success, marker="*", s=145, color="black", zorder=5)
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.02)
    colorbar.set_label("Layer depth (%)")
    colorbar.set_ticks([min_depth, 25, 50, 75, max_depth])
    ax.set_xlabel("Cost relative to GPT-OSS high (%)")
    split_label = "Validation" if split == "val" else "Test"
    ax.set_ylabel(f"{split_label} success (%)")
    ax.grid(True, alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUTPUT_DIR / f"{output_stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {output_stem}.{{png,pdf,svg}}")


# ── Figure: English vs. pooled Pareto at each condition's peak layer ─────────

def plot_peak_layer_pareto(grid: pd.DataFrame,
                           policy: str = "raw_utility",
                           output_stem: str = "peak_layer_pareto",
                           min_savings_pct: float = 20.0) -> None:
    """Pareto scatter comparing English-transfer and pooled probes,
    each plotted at the layer depth where it achieves the highest
    val-matched test success with at least `min_savings_pct` cost savings.

    This guards against degenerate operating points where the router
    always defers to the anchor (100% cost ratio, no real routing).

    X-axis: cost ratio vs. anchor (%).
    Y-axis: mean test success.
    """
    param = _policy_param(policy)

    conditions = [
        ("pooled",           "Pooled multilingual probe",   COLOR_POOLED,   "o"),
        ("english_transfer", "English-only transfer probe", COLOR_ENGLISH,  "s"),
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    _style_ax(ax)

    for cond, cond_label, color, marker in conditions:
        # 1. Find peak layer: highest val-matched test success
        #    requiring at least min_savings_pct cost savings
        peak_frac = None
        peak_success = -np.inf
        for frac, gdf in grid[grid["condition"] == cond].groupby("target_frac"):
            val  = gdf[(gdf["split"] == "val")  & (gdf["policy"] == policy)]
            test = gdf[(gdf["split"] == "test") & (gdf["policy"] == policy)]
            if val.empty or test.empty:
                continue
            bp = val.loc[val["mean_success"].idxmax(), param]
            m  = test[test[param] == bp]
            if m.empty:
                continue
            # Require meaningful savings
            if m.iloc[0]["cost_savings_pct_vs_anchor"] < min_savings_pct:
                continue
            s = m.iloc[0]["mean_success"]
            if s > peak_success:
                peak_success = s
                peak_frac = frac

        if peak_frac is None:
            print(f"  Warning: no peak found for {cond} / {policy} "
                  f"with ≥{min_savings_pct}% savings")
            continue

        print(f"  Peak layer for {cond} ({policy}): {peak_frac}% "
              f"→ val-matched test success {peak_success*100:.2f}%")

        # 2. Full sweep at peak layer (test split)
        sub = grid[
            (grid["condition"] == cond) &
            (grid["split"] == "test") &
            (grid["policy"] == policy) &
            (grid["target_frac"] == peak_frac)
        ].copy()
        sub["cost_ratio"] = 100.0 - sub["cost_savings_pct_vs_anchor"]
        sub = sub.sort_values("cost_ratio")

        ax.scatter(sub["cost_ratio"], sub["mean_success"] * 100,
                   s=12, color=color, alpha=0.6, linewidths=0, zorder=3,
                   label=f"{cond_label} (layer {peak_frac}%)")

        # Highlight the val-matched operating point
        val_df = grid[
            (grid["condition"] == cond) &
            (grid["split"] == "val") &
            (grid["policy"] == policy) &
            (grid["target_frac"] == peak_frac)
        ]
        if not val_df.empty:
            best_param_val = val_df.loc[val_df["mean_success"].idxmax(), param]
            op = sub[sub[param] == best_param_val]
            if not op.empty:
                ax.scatter(op["cost_ratio"], op["mean_success"] * 100,
                           s=100, color=color, marker=marker,
                           edgecolors="white", linewidths=1.4, zorder=5)

    # Always-anchor reference: horizontal line + right-edge marker
    ax.axhline(ANCHOR_SUCCESS * 100, color="0.35", ls=":", lw=1.2, zorder=2,
               label=f"Always anchor ({ANCHOR_SUCCESS*100:.1f}%)")

    pol_label = "no anchor stickiness" if policy == "raw_utility" else "with anchor stickiness"
    ax.set_xlabel("Cost ratio vs. anchor (%)", fontsize=10.5)
    ax.set_ylabel("Test success rate (%)", fontsize=10.5)
    ax.set_title(f"English transfer vs. pooled at peak layer ({pol_label})", fontsize=10.5, pad=8)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()

    for ext in ("pdf", "png"):
        fig.savefig(OUTPUT_DIR / f"{output_stem}.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  saved {output_stem}.{{pdf,png}}")


# ── Figure: fixed-layer comparison, both policies on one Pareto ──────────────

def plot_fixed_layer_both_policies(
        grid: pd.DataFrame,
        pooled_frac: int = 100,
        english_frac: int = 70,
        output_stem: str = "pareto_both_policies",
        pooled_label: str = "Pooled multilingual probe",
        english_label: str = "English-only transfer probe",
        pooled_color: str = COLOR_POOLED,
        english_color: str = COLOR_ENGLISH,
        policies: tuple[str, ...] = ("raw_utility", "raw_anchor"),
        legend_loc: str = "best") -> None:
    """Match the accuracy_cost_scatter_test.png reference design:
      s=8, alpha=0.45 scatter, y-axis raw fraction, no title.
    Four series: pooled & english-transfer, each for raw_utility & raw_anchor.
    Policy distinguished by opacity (utility=solid, anchor=faint).
    """
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    series = [
        # (condition, frac, color, label)
        ("english_transfer", english_frac, english_color, english_label),
        ("pooled",           pooled_frac,  pooled_color,  pooled_label),
    ]

    for cond, frac, color, label in series:
        # Combine both policies into one scatter
        sub = grid[
            (grid["condition"] == cond) &
            (grid["split"] == "test") &
            (grid["policy"].isin(policies)) &
            (grid["target_frac"] == frac)
        ].copy()
        if sub.empty:
            continue
        sub["cost_ratio"] = 100.0 - sub["cost_savings_pct_vs_anchor"]

        ax.scatter(sub["cost_ratio"], sub["mean_success"],
                   s=8, color=color, alpha=0.45,
                   linewidths=0, zorder=3, label=label)

    # Always-anchor reference line
    ax.axhline(ANCHOR_SUCCESS, color="0.55", ls="--", lw=0.8, zorder=4,
               label="Always anchor")

    ax.set_xlabel("Cost ratio vs anchor (%)")
    ax.set_ylabel("Average empirical success")
    ax.grid(alpha=0.25)
    leg = ax.legend(frameon=False, fontsize=8, loc=legend_loc)
    for handle in leg.legend_handles:
        try:
            handle.set_alpha(1.0)
            handle.set_sizes([20])
        except AttributeError:
            pass
    fig.tight_layout()

    for ext in ("pdf", "png"):
        fig.savefig(OUTPUT_DIR / f"{output_stem}.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  saved {output_stem}.{{pdf,png}}")


# ── AUROC-zone Pareto figure ──────────────────────────────────────────────────

# Zone boundaries (based on mean cross-lingual AUROC shape)
ZONE_BOUNDS = [(0.0, 0.50), (0.50, 0.80), (0.80, 1.001)]
ZONE_LABELS = []
ZONE_COLORS = ["#777777", "#2CA02C", "#9467BD"]   # grey, green, purple (zone labels)
ZONE_COLORS_LIGHT = ["#FFF8E1", "#E8F5E9", "#EDE7F6"]  # amber-50, green-50, deep-purple-50


def _zone_for_frac(frac: float) -> int:
    """Return 0/1/2 zone index for a relative layer depth fraction (0-1 scale)."""
    f = frac / 100.0  # target_frac is stored as 0-100
    for i, (lo, hi) in enumerate(ZONE_BOUNDS):
        if lo <= f < hi:
            return i
    return len(ZONE_BOUNDS) - 1


def load_cross_lingual_auroc() -> pd.DataFrame:
    """Load mean cross-lingual AUROC per layer fraction, averaged across models & languages."""
    path = REPO_ROOT / "outputs/transfer/MATH/ridge_scaled/cross_lingual_per_layer_all.csv"
    df = pd.read_csv(path)
    xling = df[(df["train_lang"] == "English") & (df["test_lang"] != "English")].copy()
    n_layers = df.groupby("model")["layer"].max() + 1
    xling["n_layers"] = xling["model"].map(n_layers)
    xling["layer_frac"] = xling["layer"] / (xling["n_layers"] - 1)
    # Mean AUROC per model per layer_frac, then mean across models
    per_model = xling.groupby(["model", "layer_frac"])["auroc"].mean().reset_index()
    # Bin into 100 evenly spaced points for a smooth curve
    bins = np.linspace(0, 1, 101)
    per_model["bin"] = pd.cut(per_model["layer_frac"], bins=bins,
                              labels=(bins[:-1] + bins[1:]) / 2,
                              include_lowest=True)
    mean_curve = per_model.groupby("bin", observed=True)["auroc"].mean().reset_index()
    mean_curve["bin"] = mean_curve["bin"].astype(float)
    return mean_curve.sort_values("bin")


def plot_auroc_zones_pareto(grid: pd.DataFrame, policy: str = "raw_utility",
                            output_stem: str = "auroc_zones_pareto") -> None:
    """Two-panel figure:
    Top — mean cross-lingual AUROC trajectory with zone colour bands.
    Bottom — English-transfer Pareto scatter, curves coloured by zone.
    """
    auroc_curve = load_cross_lingual_auroc()

    test = grid[
        (grid["condition"] == "english_transfer") &
        (grid["split"] == "test") &
        (grid["policy"] == policy)
    ].copy()
    test["cost_ratio"] = 100.0 - test["cost_savings_pct_vs_anchor"]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(7.5, 9.0),
        gridspec_kw={"height_ratios": [1, 2], "hspace": 0.08}
    )

    # ── Top: AUROC trajectory ──────────────────────────────────────────────
    _style_ax(ax_top)
    # Zone background shading
    for (lo, hi), color_light, label in zip(ZONE_BOUNDS, ZONE_COLORS_LIGHT, ZONE_LABELS):
        ax_top.axvspan(lo * 100, min(hi, 1.0) * 100, color=color_light,
                       alpha=0.65, zorder=0, label=label.split("\n")[0])
    # AUROC curve
    ax_top.plot(auroc_curve["bin"] * 100, auroc_curve["auroc"],
                color="0.2", lw=2.0, zorder=3)
    ax_top.set_ylabel("Cross-lingual AUROC\n(mean across models)", fontsize=9.5)
    ax_top.set_xlim(0, 100)
    ax_top.set_xticks([])
    ax_top.set_ylim(0.50, 0.78)
    ax_top.legend(frameon=False, fontsize=8.5, loc="lower right", ncol=3,
                  handlelength=1.2, handletextpad=0.4)
    # Vertical dashed lines at zone boundaries
    for pct in (35, 80):
        ax_top.axvline(pct, color="0.4", ls="--", lw=0.9, zorder=2)

    # ── Bottom: Pareto scatter ─────────────────────────────────────────────
    _style_ax(ax_bot)
    # Draw each depth's lambda-sweep curve, coloured by zone
    for frac, gdf in test.groupby("target_frac"):
        zone = _zone_for_frac(frac)
        color = ZONE_COLORS[zone]
        gdf_s = gdf.sort_values("cost_ratio")
        ax_bot.scatter(gdf_s["cost_ratio"], gdf_s["mean_success"],
                       s=7, color=color, alpha=0.50,
                       linewidths=0, zorder=2)

    # Dummy patches for the zone legend
    patches = [mpatches.Patch(color=c, label=l.split("\n")[0])
               for c, l in zip(ZONE_COLORS, ZONE_LABELS)]
    ax_bot.scatter([100.0], [ANCHOR_SUCCESS], marker="x", color="black",
                   s=100, linewidths=2.2, zorder=5, label="Always anchor")
    ax_bot.legend(handles=patches + [ax_bot.get_legend_handles_labels()[0][-1]],
                  frameon=False, fontsize=8.5, loc="lower right",
                  handlelength=1.2)

    pol_label = "No anchor stickiness" if policy == "raw_utility" else "With anchor stickiness"
    ax_bot.set_xlabel("Cost ratio vs. anchor (%)", fontsize=10.5)
    ax_bot.set_ylabel("Average empirical success", fontsize=10.5)
    ax_bot.set_xlim(0, 100)
    ax_bot.set_title(f"English-only transfer probe — {pol_label}", fontsize=10, pad=6)
    ax_bot.grid(alpha=0.25)

    # Vertical dashed lines aligned with top panel
    for pct in (50, 80):
        ax_bot.axvline(100 - pct, color="0.4", ls="--", lw=0.9, zorder=1, alpha=0.5)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUTPUT_DIR / f"{output_stem}.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  saved {output_stem}.{{pdf,png}}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output → {OUTPUT_DIR}")

    summary = load_summary()
    print("Loading all policy grids …")
    grid = load_all_grids()

    print("Plotting success & savings vs. depth …")
    plot_vs_depth(summary)

    print("Plotting combined pooled vs. English transfer comparison …")
    plot_combined_comparison(grid, "raw_utility", "pareto_combined_raw_utility")
    plot_combined_comparison(grid, "raw_anchor",  "pareto_combined_raw_anchor")

    print("Plotting per-condition individual scatter curves …")
    for condition, cond_label, stem, policy in [
        ("pooled",           "Pooled multilingual probe",    "pareto_pooled",        "raw_utility"),
        ("pooled",           "Pooled multilingual probe",    "pareto_pooled_anchor", "raw_anchor"),
        ("english_transfer", "English-only transfer probe",  "pareto_english",       "raw_utility"),
    ]:
        print(f"Plotting Pareto scatter: {cond_label} / {policy} …")
        plot_layer_curves(grid, condition, cond_label, policy, stem)

    print("Plotting policy comparison (line plots) …")
    plot_policy_comparison(grid)

    for condition, cond_label, stem in [
        ("pooled",           "Pooled multilingual probe",   "policy_pareto_comparison_pooled"),
        ("english_transfer", "English-only transfer probe", "policy_pareto_comparison_english"),
    ]:
        print(f"Plotting policy Pareto comparison: {cond_label} …")
        plot_policy_pareto_comparison(grid, condition, cond_label, stem)

    print("Plotting pooled test-matched savings …")
    plot_test_matched_savings(grid)

    print("Plotting AUROC + routing success vs. depth (zone shading) …")
    plot_auroc_and_routing_by_depth(grid, "raw_utility", "auroc_routing_by_depth")
    plot_auroc_and_routing_by_depth(grid, "raw_anchor",  "auroc_routing_by_depth_anchor")

    print("Plotting peak-layer Pareto comparison …")
    plot_peak_layer_pareto(grid, "raw_utility", "peak_layer_pareto",        min_savings_pct=20.0)
    plot_peak_layer_pareto(grid, "raw_anchor",  "peak_layer_pareto_anchor", min_savings_pct=10.0)

    print("Plotting fixed-layer both-policies Pareto …")
    plot_fixed_layer_both_policies(grid, pooled_frac=100, english_frac=70,
                                   output_stem="pareto_both_policies")

    print("Plotting AUROC-zone Pareto figure …")
    plot_auroc_zones_pareto(grid, "raw_utility", "auroc_zones_pareto_raw_utility")
    plot_auroc_zones_pareto(grid, "raw_anchor",  "auroc_zones_pareto_raw_anchor")

    print("Done.")


if __name__ == "__main__":
    main()
