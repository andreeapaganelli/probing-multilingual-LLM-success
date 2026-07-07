"""Build multi-layer ensemble router candidates and evaluate final routing.

Ensembles average each target model's probe scores across several layer
offsets counted back from its final layer (the ``back_XXX`` candidates of a
backward-offset depth sweep), then run the final router on the aggregated
scores. Produces the thesis layer-ensemble tables:

  layer_ensemble_summary.csv              one row per ensemble variant (Table 4.9)
  layer_ensemble_vs_selected_summary.csv  ensembles vs. the single-final-layer and
                                          table-selected single-layer baselines (Table 4.10)

Inputs:
  --source-root   backward-offset sweep root, from
                  ``python -m src.scripts.routing.run_depth_routing_sweep
                    --backward-step-layers 2 --output-root outputs/routing/backward_sweep``
  --table-selected-final   final router run at the per-language best layers
                  (``outputs/routing/simulation``), used for the ``table_selected`` baseline.

Usage:
    python -m src.scripts.etd.run_layer_ensemble_routing
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.scripts.routing.run_depth_routing_sweep import (
    ANCHOR_MARGIN_GRID,
    LAMBDA_GRID,
    MODELS,
)


ITEM_COLUMNS = ["language", "model", "problem_id", "base_problem_id", "split"]
CONDITIONS = ["pooled", "english_transfer"]


VARIANTS: dict[str, tuple[list[int], str]] = {
    "core_mean": ([0, 4, 8, 12, 20], "mean"),
    "core_median": ([0, 4, 8, 12, 20], "median"),
    "all_even_mean": (list(range(0, 25, 2)), "mean"),
    "all_even_median": (list(range(0, 25, 2)), "median"),
}


def run(cmd: list[str], cwd: Path, dry_run: bool = False) -> None:
    print("+", " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=cwd, check=True)


def aggregate(values: list[pd.Series], method: str) -> np.ndarray:
    matrix = np.vstack([series.to_numpy(dtype=float) for series in values])
    if method == "mean":
        return np.nanmean(matrix, axis=0)
    if method == "median":
        return np.nanmedian(matrix, axis=0)
    raise ValueError(f"Unknown ensemble method: {method}")


def build_ensemble_candidates(
    source_root: Path,
    output_root: Path,
    variant: str,
    offsets: list[int],
    method: str,
    condition: str,
    force: bool,
) -> Path:
    out_dir = output_root / "candidates" / variant / condition
    out_path = out_dir / "router_candidates.csv"
    if out_path.exists() and not force:
        return out_path

    frames = []
    score_columns = []
    layer_columns = []
    for offset in offsets:
        path = source_root / "candidates" / f"back_{offset:03d}" / condition / "router_candidates.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        df = df.sort_values(ITEM_COLUMNS).reset_index(drop=True)
        score_col = f"pred_offset_{offset:03d}"
        layer_col = f"layer_offset_{offset:03d}"
        score_columns.append(score_col)
        layer_columns.append(layer_col)
        if not frames:
            base = df.copy()
            base[score_col] = df["pred_success_raw"].astype(float)
            base[layer_col] = df["selected_layer"].astype(int)
            frames.append(base)
        else:
            ref = frames[0][ITEM_COLUMNS].reset_index(drop=True)
            cur = df[ITEM_COLUMNS].reset_index(drop=True)
            if not ref.equals(cur):
                raise ValueError(f"Candidate key mismatch for {condition} offset {offset}")
            frames[0][score_col] = df["pred_success_raw"].astype(float).to_numpy()
            frames[0][layer_col] = df["selected_layer"].astype(int).to_numpy()

    out = frames[0]
    out["pred_success_raw_single_layer"] = out["pred_success_raw"].astype(float)
    out["pred_success_raw"] = np.clip(
        aggregate([out[col] for col in score_columns], method),
        0.0,
        1.0,
    )
    out["ensemble_variant"] = variant
    out["ensemble_method"] = method
    out["ensemble_offsets"] = ",".join(str(offset) for offset in offsets)
    out["ensemble_n_layers"] = len(offsets)
    out["selected_layer"] = -1
    out["selected_layer_frac"] = np.nan
    out["probe_path"] = f"ensemble:{variant}"
    out["selected_val_auroc"] = np.nan
    out["selected_test_auroc"] = np.nan

    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    metadata = {
        "variant": variant,
        "condition": condition,
        "method": method,
        "offsets": offsets,
        "score_columns": score_columns,
        "layer_columns": layer_columns,
        "n_rows": len(out),
    }
    (out_dir / "ensemble_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out_path


def evaluate_variant(output_root: Path, variant: str, dry_run: bool) -> None:
    run(
        [
            sys.executable,
            "-m",
            "src.scripts.routing.run_router_simulation",
            "--routing-root",
            str((output_root / "candidates" / variant).relative_to(REPO_ROOT)),
            "--conditions",
            "pooled,english_transfer",
            "--models",
            ",".join(MODELS),
            "--anchor-model",
            "gpt-oss-20b_high",
            "--score-col",
            "pred_success_raw",
            "--accounting-cost-source",
            "fixed_model",
            "--lambda-grid",
            LAMBDA_GRID,
            "--anchor-margin-grid",
            ANCHOR_MARGIN_GRID,
            "--max-drop-pp-grid",
            "0,0.25,0.5,1,1.5,2,3,5",
            "--output-dir",
            str((output_root / "final" / variant).relative_to(REPO_ROOT)),
        ],
        REPO_ROOT,
        dry_run,
    )


def _condition_stats(ops: pd.DataFrame, grid: pd.DataFrame, condition: str) -> dict[str, object]:
    """Best-validation-selected and test-matched operating points for one condition."""
    stats: dict[str, object] = {}
    best = ops[
        (ops["condition"] == condition)
        & (ops["selection_type"] == "best_val_success")
    ].iloc[0]
    stats["best_test_success"] = float(best["test_success"])
    stats["best_test_savings"] = float(best["test_cost_savings_pct_vs_anchor"])
    stats["best_anchor_route_frac"] = float(best["test_anchor_route_frac"])
    stats["best_policy"] = str(best["policy"])
    stats["best_lambda"] = float(best["lambda"])
    if pd.notna(best["anchor_margin"]):
        stats["best_anchor_margin"] = float(best["anchor_margin"])

    test = grid[
        (grid["condition"] == condition)
        & (grid["split"] == "test")
        & (grid["policy"].isin(["raw_utility", "raw_anchor"]))
    ].copy()
    anchor = grid[
        (grid["condition"] == condition)
        & (grid["split"] == "test")
        & (grid["policy"] == "always:gpt-oss-20b_high")
    ].iloc[0]
    eligible = test[test["mean_success"] >= float(anchor["mean_success"])].copy()
    stats["test_anchor_success"] = float(anchor["mean_success"])
    stats["test_matched_count"] = int(len(eligible))
    if not eligible.empty:
        matched = eligible.sort_values(
            ["cost_savings_pct_vs_anchor", "mean_success"],
            ascending=[False, False],
        ).iloc[0]
        stats["test_matched_success"] = float(matched["mean_success"])
        stats["test_matched_savings"] = float(matched["cost_savings_pct_vs_anchor"])
        stats["test_matched_anchor_route_frac"] = float(matched["anchor_route_frac"])
        stats["test_matched_policy"] = str(matched["policy"])
        stats["test_matched_lambda"] = float(matched["lambda"])
        if pd.notna(matched["anchor_margin"]):
            stats["test_matched_anchor_margin"] = float(matched["anchor_margin"])
    return stats


def load_final_dir(final_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    ops_path = final_dir / "routing_operating_points.csv"
    if not ops_path.exists():
        # accept the pre-rename filename from older runs
        ops_path = final_dir / "operating_points_val_selected_test.csv"
    grid_path = final_dir / "policy_grid_val_test.csv"
    if not ops_path.exists() or not grid_path.exists():
        return None
    return pd.read_csv(ops_path), pd.read_csv(grid_path)


def collect_summary(output_root: Path) -> pd.DataFrame:
    rows = []
    for variant, (offsets, method) in VARIANTS.items():
        final_dir = output_root / "final" / variant
        loaded = load_final_dir(final_dir)
        if loaded is None:
            continue
        ops, grid = loaded
        row: dict[str, object] = {
            "variant": variant,
            "method": method,
            "offsets": ",".join(str(offset) for offset in offsets),
            "n_layers": len(offsets),
            "final_dir": str(final_dir.relative_to(REPO_ROOT)),
        }
        for condition in CONDITIONS:
            for key, value in _condition_stats(ops, grid, condition).items():
                row[f"{condition}__{key}"] = value
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(output_root / "layer_ensemble_summary.csv", index=False)
    return summary


def collect_vs_selected_summary(
    output_root: Path,
    baselines: dict[str, Path],
) -> pd.DataFrame:
    """Condensed comparison of the single-layer baselines and the ensembles."""
    short = {
        "best_test_success": "best_success",
        "best_test_savings": "best_savings",
        "best_anchor_route_frac": "best_anchor_frac",
        "test_matched_count": "testmatch_count",
        "test_matched_success": "testmatch_success",
        "test_matched_savings": "testmatch_savings",
        "test_matched_anchor_route_frac": "testmatch_anchor_frac",
    }
    rows = []
    entries = list(baselines.items()) + [
        (variant, output_root / "final" / variant) for variant in VARIANTS
    ]
    for variant, final_dir in entries:
        loaded = load_final_dir(final_dir)
        if loaded is None:
            print(f"[warn] missing final router outputs for {variant}: {final_dir}")
            continue
        ops, grid = loaded
        row: dict[str, object] = {"variant": variant}
        for condition in CONDITIONS:
            stats = _condition_stats(ops, grid, condition)
            for long_key, short_key in short.items():
                row[f"{condition}_{short_key}"] = stats.get(long_key, np.nan)
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(output_root / "layer_ensemble_vs_selected_summary.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("outputs/routing/backward_sweep"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/etd/layer_ensemble_routing"),
    )
    parser.add_argument(
        "--table-selected-final",
        type=Path,
        default=Path("outputs/routing/simulation"),
        help="Final router run at the per-language best layers (table_selected baseline).",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source_root = args.source_root if args.source_root.is_absolute() else REPO_ROOT / args.source_root
    output_root = args.output_root if args.output_root.is_absolute() else REPO_ROOT / args.output_root
    table_selected = (
        args.table_selected_final
        if args.table_selected_final.is_absolute()
        else REPO_ROOT / args.table_selected_final
    )
    output_root.mkdir(parents=True, exist_ok=True)

    for variant, (offsets, method) in VARIANTS.items():
        print(f"\n=== {variant}: {method} offsets={offsets} ===", flush=True)
        for condition in CONDITIONS:
            path = build_ensemble_candidates(
                source_root,
                output_root,
                variant,
                offsets,
                method,
                condition,
                args.force,
            )
            print(f"[candidates] {condition}: {path}", flush=True)
        final_path = output_root / "final" / variant / "policy_grid_val_test.csv"
        if args.force or not final_path.exists():
            evaluate_variant(output_root, variant, args.dry_run)
        else:
            print(f"[skip] {variant}: final router already exists", flush=True)

    if not args.dry_run:
        summary = collect_summary(output_root)
        print(f"\nWrote {output_root / 'layer_ensemble_summary.csv'}")
        cols = [
            "variant",
            "pooled__best_test_success",
            "pooled__best_test_savings",
            "pooled__test_matched_success",
            "pooled__test_matched_savings",
            "english_transfer__best_test_success",
            "english_transfer__best_test_savings",
            "english_transfer__test_matched_count",
        ]
        print(summary[cols].to_string(index=False))

        collect_vs_selected_summary(
            output_root,
            baselines={
                "single_last": source_root / "final" / "back_000",
                "table_selected": table_selected,
            },
        )
        print(f"Wrote {output_root / 'layer_ensemble_vs_selected_summary.csv'}")


if __name__ == "__main__":
    main()
