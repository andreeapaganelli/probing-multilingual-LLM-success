"""Run a shared layer-depth routing sweep for thesis router experiments.

For each normalized layer fraction available from the model depths, map every
model to its nearest real layer, rebuild pooled/English-transfer candidates,
run the final router, and collect a compact summary.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODELS = [
    "Qwen3-0.6B",
    "Qwen3-1.7B",
    "Qwen3-4B",
    "Qwen3-8B",
    "Qwen3.5-4B",
    "Qwen3.5-9B",
    "gpt-oss-20b_low",
    "gpt-oss-20b_high",
]

LANGUAGES = "Arabic,Chinese,English,French,Italian,Russian,Swahili,Telugu,Thai,Turkish"
MODELS_CSV = ",".join(MODELS)

LAMBDA_GRID = (
    "0,0.0025,0.005,0.0075,0.01,"
    "0.011,0.012,0.013,0.014,0.015,0.016,0.017,0.018,0.019,"
    "0.02,0.021,0.022,0.023,0.024,0.025,0.026,0.027,0.028,0.029,"
    "0.03,0.031,0.032,0.033,0.034,0.035,0.036,0.037,0.038,0.039,"
    "0.04,0.041,0.042,0.043,0.044,0.045,0.046,0.047,0.048,0.049,"
    "0.05,0.051,0.052,0.053,0.054,0.055,0.056,0.057,0.058,0.059,"
    "0.06,0.061,0.062,0.063,0.064,0.065,0.08,0.1,0.13,0.15,0.17,"
    "0.2,0.25,0.3,0.4,0.5"
)

ANCHOR_MARGIN_GRID = (
    "0,0.005,0.01,0.015,0.016,0.017,0.018,0.019,"
    "0.02,0.021,0.022,0.023,0.024,0.025,0.026,0.027,0.028,0.029,"
    "0.03,0.031,0.032,0.033,0.034,0.035,0.036,0.037,0.038,0.039,"
    "0.04,0.041,0.042,0.043,0.044,0.045,0.046,0.047,0.048,0.049,"
    "0.05,0.075,0.1"
)


def run(cmd: list[str], cwd: Path, dry_run: bool = False) -> None:
    print("+", " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=cwd, check=True)


def run_parallel(cmds: list[list[str]], cwd: Path, dry_run: bool = False) -> None:
    if not cmds:
        return
    for cmd in cmds:
        print("+", " ".join(cmd), flush=True)
    if dry_run:
        return
    procs = [subprocess.Popen(cmd, cwd=cwd) for cmd in cmds]
    failures: list[tuple[list[str], int]] = []
    for cmd, proc in zip(cmds, procs, strict=True):
        code = proc.wait()
        if code != 0:
            failures.append((cmd, code))
    if failures:
        for cmd, code in failures:
            print(f"[failed:{code}] {' '.join(cmd)}", flush=True)
        raise subprocess.CalledProcessError(failures[0][1], failures[0][0])


def load_model_depths(probe_root: Path, models: list[str]) -> dict[str, int]:
    depths: dict[str, int] = {}
    for model in models:
        bundle_path = probe_root / "probes_ridge_scaled" / model / "_pooled.joblib"
        bundle = joblib.load(bundle_path)
        layers = [int(layer) for layer in bundle["layers"].keys()]
        depths[model] = max(layers) + 1
    return depths


def shared_fraction_grid(depths: dict[str, int]) -> list[float]:
    values: set[float] = set()
    for n_layers in depths.values():
        denom = max(n_layers - 1, 1)
        for layer in range(n_layers):
            values.add(layer / denom)
    return sorted(values)


def layer_for_fraction(frac: float, n_layers: int) -> int:
    return int(round(frac * max(n_layers - 1, 0)))


def build_sweep_index(depths: dict[str, int], models: list[str]) -> pd.DataFrame:
    rows = []
    seen: set[tuple[int, ...]] = set()
    for frac in shared_fraction_grid(depths):
        layers = {model: layer_for_fraction(frac, depths[model]) for model in models}
        signature = tuple(layers[model] for model in models)
        if signature in seen:
            continue
        seen.add(signature)
        row = {
            "sweep_id": f"frac_{len(rows):03d}",
            "target_frac": frac,
        }
        for model in models:
            row[f"{model}__layer"] = layers[model]
            row[f"{model}__layer_frac"] = (
                layers[model] / max(depths[model] - 1, 1)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_fraction_index(depths: dict[str, int], fractions: list[float], models: list[str]) -> pd.DataFrame:
    rows = []
    seen: set[tuple[int, ...]] = set()
    for frac in fractions:
        layers = {model: layer_for_fraction(frac, depths[model]) for model in models}
        signature = tuple(layers[model] for model in models)
        if signature in seen:
            continue
        seen.add(signature)
        row = {
            "sweep_id": f"frac_{int(round(frac * 100)):03d}",
            "target_frac": frac,
        }
        for model in models:
            row[f"{model}__layer"] = layers[model]
            row[f"{model}__layer_frac"] = (
                layers[model] / max(depths[model] - 1, 1)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_backward_offset_index(
    depths: dict[str, int],
    models: list[str],
    step_layers: int,
    max_offset: int | None,
) -> pd.DataFrame:
    if step_layers <= 0:
        raise ValueError("--backward-step-layers must be positive")
    highest_valid_offset = min(depths[model] - 1 for model in models)
    if max_offset is not None:
        highest_valid_offset = min(highest_valid_offset, int(max_offset))
    rows = []
    for offset in range(0, highest_valid_offset + 1, step_layers):
        row = {
            "sweep_id": f"back_{offset:03d}",
            "target_frac": float("nan"),
            "layer_offset_from_last": offset,
        }
        for model in models:
            layer = depths[model] - 1 - offset
            row[f"{model}__layer"] = int(layer)
            row[f"{model}__layer_frac"] = layer / max(depths[model] - 1, 1)
        rows.append(row)
    return pd.DataFrame(rows)


def write_fixed_layer_csv(row: pd.Series, path: Path, models: list[str]) -> None:
    records = [
        {"model": model, "layer": int(row[f"{model}__layer"])}
        for model in models
    ]
    pd.DataFrame(records).to_csv(path, index=False)


def collect_summary(final_dir: Path, row: pd.Series, models: list[str]) -> dict[str, object]:
    ops = pd.read_csv(final_dir / "routing_operating_points.csv")
    grid = pd.read_csv(final_dir / "policy_grid_val_test.csv")
    out: dict[str, object] = {
        "sweep_id": row["sweep_id"],
        "target_frac": float(row["target_frac"]),
        "final_dir": str(final_dir.relative_to(REPO_ROOT)),
    }
    if "layer_offset_from_last" in row and pd.notna(row["layer_offset_from_last"]):
        out["layer_offset_from_last"] = int(row["layer_offset_from_last"])
    for model in models:
        out[f"{model}__layer"] = int(row[f"{model}__layer"])
    for condition in ["pooled", "english_transfer"]:
        best = ops[
            (ops["condition"] == condition)
            & (ops["selection_type"] == "best_val_success")
        ].iloc[0]
        out[f"{condition}__best_test_success"] = float(best["test_success"])
        out[f"{condition}__best_test_savings"] = float(best["test_cost_savings_pct_vs_anchor"])
        out[f"{condition}__best_anchor_route_frac"] = float(best["test_anchor_route_frac"])
        match0 = ops[
            (ops["condition"] == condition)
            & (ops["selection_type"] == "max_drop_pp")
            & (ops["selection_value"].astype(str) == "0")
        ]
        if not match0.empty and str(match0.iloc[0]["status"]) == "ok":
            row0 = match0.iloc[0]
            out[f"{condition}__val_match_test_success"] = float(row0["test_success"])
            out[f"{condition}__val_match_test_savings"] = float(row0["test_cost_savings_pct_vs_anchor"])
            out[f"{condition}__val_match_anchor_route_frac"] = float(row0["test_anchor_route_frac"])

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
        out[f"{condition}__test_anchor_success"] = float(anchor["mean_success"])
        out[f"{condition}__test_matched_count"] = int(len(eligible))
        if not eligible.empty:
            matched = eligible.sort_values(
                ["cost_savings_pct_vs_anchor", "mean_success"],
                ascending=[False, False],
            ).iloc[0]
            out[f"{condition}__test_matched_success"] = float(matched["mean_success"])
            out[f"{condition}__test_matched_savings"] = float(matched["cost_savings_pct_vs_anchor"])
            out[f"{condition}__test_matched_anchor_route_frac"] = float(matched["anchor_route_frac"])
            out[f"{condition}__test_matched_policy"] = str(matched["policy"])
            out[f"{condition}__test_matched_lambda"] = float(matched["lambda"])
            if pd.notna(matched["anchor_margin"]):
                out[f"{condition}__test_matched_anchor_margin"] = float(matched["anchor_margin"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/routing/depth_sweep"),
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--stop-index", type=int, default=None)
    parser.add_argument(
        "--fractions",
        type=str,
        default=None,
        help="Optional comma-separated normalized depths, e.g. 0.6,0.7,0.8,0.9. "
             "When set, only these shared depths are run.",
    )
    parser.add_argument(
        "--backward-step-layers",
        type=int,
        default=None,
        help=(
            "Optional layer step for a backward-from-final sweep. With 2, each "
            "assignment uses max_layer, max_layer-2, max_layer-4, ... for every model."
        ),
    )
    parser.add_argument(
        "--backward-max-offset",
        type=int,
        default=None,
        help=(
            "Largest offset from final layer for --backward-step-layers. Defaults "
            "to the shallowest model's final-layer index."
        ),
    )
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = args.output_root if args.output_root.is_absolute() else REPO_ROOT / args.output_root
    probe_root = REPO_ROOT / "outputs/probes/pooled"
    root.mkdir(parents=True, exist_ok=True)
    for subdir in ["fixed_layers", "configs", "candidates", "final"]:
        (root / subdir).mkdir(parents=True, exist_ok=True)

    models = MODELS
    models_csv = ",".join(models)
    depths = load_model_depths(probe_root, models)
    if args.backward_step_layers is not None:
        index = build_backward_offset_index(
            depths,
            models,
            step_layers=args.backward_step_layers,
            max_offset=args.backward_max_offset,
        )
    elif args.fractions:
        fractions = [float(item.strip()) for item in args.fractions.split(",") if item.strip()]
        index = build_fraction_index(depths, fractions, models)
    else:
        index = build_sweep_index(depths, models)
    index.to_csv(root / "layer_sweep_index.csv", index=False)
    print(
        f"Prepared {len(index)} unique shared layer assignments under {root} "
        f"(n_models={len(models)})",
        flush=True,
    )

    stop = args.stop_index if args.stop_index is not None else len(index)
    summary_rows: list[dict[str, object]] = []
    summary_path = root / "layer_sweep_summary.csv"
    if args.resume and summary_path.exists() and not args.force:
        existing = pd.read_csv(summary_path)
        done = set(existing["sweep_id"].astype(str))
        summary_rows.extend(existing.to_dict("records"))
    else:
        done = set()

    for idx, row in index.iloc[args.start_index:stop].iterrows():
        sweep_id = str(row["sweep_id"])
        if sweep_id in done and not args.force:
            print(f"[skip] {sweep_id}: already summarized", flush=True)
            continue
        print(f"\n=== {sweep_id} target_frac={float(row['target_frac']):.6f} ===", flush=True)
        fixed_csv = root / "fixed_layers" / f"{sweep_id}.csv"
        config_dir = root / "configs" / sweep_id
        routing_root = root / "candidates" / sweep_id
        final_dir = root / "final" / sweep_id
        write_fixed_layer_csv(row, fixed_csv, models)

        if args.force or not (config_dir / "pooled.csv").exists():
            run(
                [
                    sys.executable,
                    "-m",
                    "src.scripts.routing.build_routing_probe_configs",
                    "--output-dir",
                    str(config_dir.relative_to(REPO_ROOT)),
                    "--fixed-layers-csv",
                    str(fixed_csv.relative_to(REPO_ROOT)),
                ],
                REPO_ROOT,
                args.dry_run,
            )

        commands = [
            (
                "pooled",
                [
                    sys.executable,
                    "-m",
                    "src.scripts.routing.generate_router_candidates",
                    "--probe-dir",
                    "outputs/probes/pooled",
                    "--best-layers-path",
                    str((config_dir / "pooled.csv").relative_to(REPO_ROOT)),
                ],
            ),
            (
                "english_transfer",
                [
                    sys.executable,
                    "-m",
                    "src.scripts.routing.generate_router_candidates",
                    "--probe-dir",
                    "outputs/probes/language_specific",
                    "--best-layers-path",
                    str((config_dir / "english_transfer.csv").relative_to(REPO_ROOT)),
                ],
            ),
        ]
        candidate_cmds: list[list[str]] = []
        for condition, base_cmd in commands:
            out_dir = routing_root / condition
            if args.force or not (out_dir / "router_candidates.csv").exists():
                cmd = [
                    *base_cmd,
                    "--activations-root",
                    "outputs/activations/MATH",
                    "--answers-root",
                    "outputs/rollouts/MATH",
                    "--success-dir",
                    "outputs/success_rates/MATH",
                    "--output-dir",
                    str(out_dir.relative_to(REPO_ROOT)),
                    "--canonical-csv-path",
                    "data/MATH_translated.csv",
                    "--languages",
                    LANGUAGES,
                    "--models",
                    models_csv,
                    "--min-router-candidates",
                    str(len(models)),
                    "--require-all-candidates",
                    "--write-router-candidates",
                    "--candidates-only",
                ]
                candidate_cmds.append(cmd)
        run_parallel(candidate_cmds, REPO_ROOT, args.dry_run)

        if args.force or not (final_dir / "policy_grid_val_test.csv").exists():
            run(
                [
                    sys.executable,
                    "-m",
                    "src.scripts.routing.run_router_simulation",
                    "--routing-root",
                    str(routing_root.relative_to(REPO_ROOT)),
                    "--conditions",
                    "pooled,english_transfer",
                    "--models",
                    models_csv,
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
                    str(final_dir.relative_to(REPO_ROOT)),
                ],
                REPO_ROOT,
                args.dry_run,
            )

        if not args.dry_run:
            summary_rows = [r for r in summary_rows if r.get("sweep_id") != sweep_id]
            summary_rows.append(collect_summary(final_dir, row, models))
            sort_col = "layer_offset_from_last" if "layer_offset_from_last" in summary_rows[-1] else "target_frac"
            pd.DataFrame(summary_rows).sort_values(sort_col).to_csv(summary_path, index=False)
            print(f"[summary] wrote {summary_path}", flush=True)

    if not args.dry_run and summary_rows:
        summary = pd.DataFrame(summary_rows)
        sort_col = "layer_offset_from_last" if "layer_offset_from_last" in summary.columns else "target_frac"
        summary = summary.sort_values(sort_col)
        print("\nTop pooled test-matched savings:")
        cols = [
            "sweep_id",
            "target_frac",
            "pooled__test_matched_success",
            "pooled__test_matched_savings",
            "pooled__test_matched_anchor_route_frac",
            "pooled__best_test_success",
            "pooled__best_test_savings",
        ]
        print(
            summary.sort_values("pooled__test_matched_savings", ascending=False)
            .head(10)[cols]
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
