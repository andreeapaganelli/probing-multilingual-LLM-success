"""Recompute model-cost sensitivity for raw_utility and raw_anchor policies only.

This matches the paper relative-depth analysis: both policies use
``pred_success_raw``. ``raw_anchor`` is not a different probe; it applies anchor
stickiness to the raw-utility decision.

Because these policies compare raw scores for all candidate models, the
deployment-cost sensitivity charges one probe/input-read overhead for every
candidate model per item, plus the selected model's generation/output cost.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ITEM_COLS = ["split", "language", "base_problem_id"]
ANCHOR_MODEL = "gpt-oss-20b_high"


def parse_grid(spec: str) -> list[float]:
    spec = spec.strip()
    if spec.startswith("linspace:"):
        _, start, stop, count = spec.split(":")
        return [round(float(x), 8) for x in np.linspace(float(start), float(stop), int(count))]
    return [float(x.strip()) for x in spec.split(",") if x.strip()]


def parse_models(spec: str | None) -> list[str] | None:
    if spec is None or not spec.strip():
        return None
    models = [x.strip() for x in spec.split(",") if x.strip()]
    if ANCHOR_MODEL not in models:
        models.append(ANCHOR_MODEL)
    return models


def add_probe_pass_cost(
    candidates: pd.DataFrame,
    probe_cost_fraction: float,
    input_price_fraction: float,
) -> pd.DataFrame:
    out = candidates.copy()
    if "input_tokens" in out.columns:
        out["probe_pass_cost_usd"] = (
            out["input_tokens"].astype(float)
            * out["output_cost_per_million"].astype(float)
            * float(input_price_fraction)
            / 1_000_000.0
        )
        out["probe_cost_source"] = "input_tokens"
    else:
        out["probe_pass_cost_usd"] = out["tier_cost_usd"].astype(float) * float(probe_cost_fraction)
        out["probe_cost_source"] = "tier_cost_fraction_all_models"
    return out


def add_relative_decision_cost(candidates: pd.DataFrame) -> pd.DataFrame:
    """Use predicted one-generation cost relative to GPT-OSS high for routing."""
    out = candidates.copy()
    source = out[out["split"].astype(str).eq("probe_train")]
    if source.empty:
        source = out
    predicted_cost = source.groupby("model")["mean_output_cost_usd"].mean()
    if ANCHOR_MODEL not in predicted_cost.index:
        raise ValueError(f"Cost reference model {ANCHOR_MODEL!r} is absent")
    anchor_cost = float(predicted_cost.loc[ANCHOR_MODEL])
    if anchor_cost <= 0:
        raise ValueError("The anchor predicted generation cost must be positive")
    out["predicted_generation_cost_usd"] = out["model"].map(predicted_cost)
    if out["predicted_generation_cost_usd"].isna().any():
        raise ValueError("Missing training-mean generation cost for one or more candidate models")
    out["predicted_generation_cost_relative"] = out["predicted_generation_cost_usd"] / anchor_cost
    return out


def choose_policy(group: pd.DataFrame, policy: str, lambda_value: float, anchor_margin: float | None) -> pd.Series:
    work = group.copy()
    work["_utility"] = (
        work["pred_success_raw"].astype(float)
        - lambda_value * work["predicted_generation_cost_relative"].astype(float)
    )
    best = work.loc[work["_utility"].idxmax()]
    if policy == "raw_utility":
        return best
    if policy != "raw_anchor":
        raise ValueError(f"Unsupported policy: {policy}")
    if anchor_margin is None:
        raise ValueError("raw_anchor requires anchor_margin")
    anchor_rows = work[work["model"].eq(ANCHOR_MODEL)]
    if anchor_rows.empty:
        return best
    anchor = anchor_rows.iloc[0]
    if float(anchor["_utility"]) >= float(best["_utility"]) - float(anchor_margin):
        return anchor
    return best


def build_arrays(split_df: pd.DataFrame, output_cost_col: str) -> dict[str, Any]:
    models = (
        split_df.groupby("model", as_index=False)["tier_cost_usd"]
        .mean()
        .sort_values(["tier_cost_usd", "model"], kind="mergesort")["model"]
        .astype(str)
        .tolist()
    )
    if ANCHOR_MODEL in models:
        models = [m for m in models if m != ANCHOR_MODEL] + [ANCHOR_MODEL]
    item_index = split_df[ITEM_COLS].drop_duplicates().reset_index(drop=True)
    item_lookup = {tuple(row): i for i, row in enumerate(item_index[ITEM_COLS].itertuples(index=False, name=None))}
    model_lookup = {model: j for j, model in enumerate(models)}
    n_items, n_models = len(item_index), len(models)
    score = np.full((n_items, n_models), np.nan, dtype=np.float32)
    success = np.full((n_items, n_models), np.nan, dtype=np.float32)
    output_cost = np.full((n_items, n_models), np.nan, dtype=np.float64)
    decision_cost = np.full((n_items, n_models), np.nan, dtype=np.float32)
    probe_cost = np.full((n_items, n_models), 0.0, dtype=np.float64)

    cols = [
        *ITEM_COLS,
        "model",
        "pred_success_raw",
        "true_success",
        output_cost_col,
        "predicted_generation_cost_relative",
        "probe_pass_cost_usd",
    ]
    for row in split_df[cols].itertuples(index=False):
        i = item_lookup[(row.split, row.language, row.base_problem_id)]
        j = model_lookup[str(row.model)]
        score[i, j] = float(row.pred_success_raw)
        success[i, j] = float(row.true_success)
        output_cost[i, j] = float(getattr(row, output_cost_col))
        decision_cost[i, j] = float(row.predicted_generation_cost_relative)
        probe_cost[i, j] = float(row.probe_pass_cost_usd)
    return {
        "split": str(split_df["split"].iloc[0]),
        "models": models,
        "score": score,
        "success": success,
        "output_cost": output_cost,
        "decision_cost": decision_cost,
        "probe_overhead": np.nansum(probe_cost, axis=1),
    }


def summarize_indices(
    arrays: dict[str, Any],
    selected: np.ndarray,
    anchor_idx: int,
    anchor_cost: float,
    anchor_success: float,
    params: dict[str, Any],
) -> dict[str, Any]:
    models: list[str] = arrays["models"]
    row_idx = np.arange(len(selected))
    selected_success = arrays["success"][row_idx, selected]
    selected_output = arrays["output_cost"][row_idx, selected]
    overhead = arrays["probe_overhead"]
    total = selected_output + overhead
    counts = np.bincount(selected, minlength=len(models)) / max(len(selected), 1)
    mix = {models[i]: float(v) for i, v in enumerate(counts) if v > 0}
    return {
        **params,
        "n_items": int(len(selected)),
        "mean_success": float(np.nanmean(selected_success)),
        "success_delta_pp_vs_anchor": float((np.nanmean(selected_success) - anchor_success) * 100.0),
        "mean_selected_output_cost_usd": float(np.nanmean(selected_output)),
        "mean_all_model_probe_overhead_cost_usd": float(np.nanmean(overhead)),
        "mean_total_cost_usd": float(np.nanmean(total)),
        "cost_savings_pct_vs_anchor": float((1.0 - np.nanmean(total) / anchor_cost) * 100.0),
        "anchor_route_frac": float(np.mean(selected == anchor_idx)),
        "selected_model_mix": json.dumps(mix, sort_keys=True),
    }


def summarize(chosen: pd.DataFrame, anchor_cost: float, anchor_success: float, params: dict[str, Any]) -> dict[str, Any]:
    selected_cost = chosen["total_output_cost_usd"].astype(float)
    overhead = chosen["all_model_probe_overhead_cost_usd"].astype(float)
    total = selected_cost + overhead
    mix = chosen["model"].value_counts(normalize=True).to_dict()
    return {
        **params,
        "n_items": int(len(chosen)),
        "mean_success": float(chosen["true_success"].astype(float).mean()),
        "success_delta_pp_vs_anchor": float((chosen["true_success"].astype(float).mean() - anchor_success) * 100.0),
        "mean_selected_output_cost_usd": float(selected_cost.mean()),
        "mean_all_model_probe_overhead_cost_usd": float(overhead.mean()),
        "mean_total_cost_usd": float(total.mean()),
        "cost_savings_pct_vs_anchor": float((1.0 - total.mean() / anchor_cost) * 100.0),
        "anchor_route_frac": float(chosen["model"].eq(ANCHOR_MODEL).mean()),
        "selected_model_mix": json.dumps({str(k): float(v) for k, v in mix.items()}, sort_keys=True),
    }


def evaluate(
    candidates: pd.DataFrame,
    lambda_grid: list[float],
    anchor_margin_grid: list[float],
    output_cost_col: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, split_df in candidates[candidates["split"].isin(["val", "test"])].groupby("split", sort=False):
        arrays = build_arrays(split_df, output_cost_col)
        models = arrays["models"]
        anchor_idx = models.index(ANCHOR_MODEL)
        # Baseline "always anchor" reads the input only in the anchor model.
        anchor_probe = split_df[split_df["model"].eq(ANCHOR_MODEL)]["probe_pass_cost_usd"].to_numpy(dtype=float)
        anchor_cost = float(np.nanmean(arrays["output_cost"][:, anchor_idx] + anchor_probe))
        anchor_success = float(np.nanmean(arrays["success"][:, anchor_idx]))
        row_idx = np.arange(arrays["score"].shape[0])

        for lambda_value in lambda_grid:
            utility = arrays["score"] - lambda_value * arrays["decision_cost"]
            selected = np.nanargmax(utility, axis=1)
            rows.append(
                summarize_indices(
                    arrays,
                    selected,
                    anchor_idx,
                    anchor_cost,
                    anchor_success,
                    {
                        "policy": "raw_utility",
                        "split": split,
                        "lambda": lambda_value,
                        "anchor_margin": np.nan,
                    },
                )
            )

            for margin in anchor_margin_grid:
                best_utility = utility[row_idx, selected]
                anchor_utility = utility[:, anchor_idx]
                anchored = np.where(anchor_utility >= best_utility - float(margin), anchor_idx, selected)
                rows.append(
                    summarize_indices(
                        arrays,
                        anchored,
                        anchor_idx,
                        anchor_cost,
                        anchor_success,
                        {
                            "policy": "raw_anchor",
                            "split": split,
                            "lambda": lambda_value,
                            "anchor_margin": margin,
                        },
                    )
                )
    return pd.DataFrame(rows)


def select_by_loss_tolerance(grid: pd.DataFrame, tolerances: list[float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for policy, val in grid[grid["split"].eq("val")].groupby("policy", sort=False):
        test = grid[grid["split"].eq("test") & grid["policy"].eq(policy)]
        anchor_val_success = float((val["mean_success"] - val["success_delta_pp_vs_anchor"] / 100.0).iloc[0])
        for tol in tolerances:
            feasible = val[val["mean_success"] >= anchor_val_success - float(tol) / 100.0].copy()
            if feasible.empty:
                continue
            pick = feasible.sort_values(
                ["mean_total_cost_usd", "mean_success"],
                ascending=[True, False],
                kind="mergesort",
            ).iloc[0]
            mask = test["lambda"].astype(float).eq(float(pick["lambda"]))
            if policy == "raw_anchor":
                mask &= test["anchor_margin"].astype(float).eq(float(pick["anchor_margin"]))
            matched = test[mask]
            if matched.empty:
                continue
            row = matched.iloc[0].to_dict()
            row["loss_tolerance_pp"] = float(tol)
            row["selected_on_val_mean_success"] = float(pick["mean_success"])
            row["selected_on_val_cost_savings_pct_vs_anchor"] = float(pick["cost_savings_pct_vs_anchor"])
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument(
        "--input-tokens-from",
        type=Path,
        default=None,
        help="Optional candidate CSV supplying exact input_tokens by model/language/problem_id.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--probe-cost-fraction", type=float, required=True)
    parser.add_argument(
        "--input-price-fraction",
        type=float,
        default=1.0,
        help="Input-token price as a fraction of the model's output-token price.",
    )
    parser.add_argument(
        "--output-cost-col",
        default="total_output_cost_usd",
        choices=("total_output_cost_usd", "mean_output_cost_usd"),
        help="Candidate column used for the cost of the selected generation.",
    )
    parser.add_argument("--lambda-grid", default="0,0.01,0.02,0.05,0.10,0.20,0.35,0.50,0.75,1.0")
    parser.add_argument(
        "--anchor-margin-grid",
        default="0,0.005,0.01,0.015,0.02,0.025,0.03,0.04,0.05,0.075,0.10",
    )
    parser.add_argument("--loss-tolerances", default="0,0.5,1,2,3,5")
    parser.add_argument(
        "--parameter-grid-from",
        type=Path,
        default=None,
        help="Reuse the exact lambda and anchor-margin values from an existing policy-grid CSV.",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated candidate model subset. Anchor is added if omitted.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=False)
    if args.input_price_fraction < 0:
        raise ValueError("--input-price-fraction must be non-negative")
    candidates = pd.read_csv(args.candidates)
    if args.input_tokens_from is not None:
        token_cols = ["model", "language", "problem_id", "input_tokens"]
        tokens = pd.read_csv(args.input_tokens_from, usecols=token_cols).drop_duplicates()
        candidates = candidates.drop(columns=["input_tokens"], errors="ignore").merge(
            tokens,
            on=["model", "language", "problem_id"],
            how="left",
            validate="many_to_one",
        )
        if candidates["input_tokens"].isna().any():
            raise ValueError("Exact input-token table does not cover every candidate row")
    candidates = add_probe_pass_cost(
        candidates,
        args.probe_cost_fraction,
        args.input_price_fraction,
    )
    model_subset = parse_models(args.models)
    if model_subset is not None:
        missing = sorted(set(model_subset) - set(candidates["model"].astype(str).unique()))
        if missing:
            raise ValueError(f"Requested model(s) not found in candidates: {missing}")
        candidates = candidates[candidates["model"].astype(str).isin(model_subset)].copy()
    candidates = add_relative_decision_cost(candidates)
    lambda_grid = parse_grid(args.lambda_grid)
    anchor_margin_grid = parse_grid(args.anchor_margin_grid)
    if args.parameter_grid_from is not None:
        old_grid = pd.read_csv(args.parameter_grid_from, usecols=["lambda", "anchor_margin"])
        lambda_grid = sorted(old_grid["lambda"].dropna().astype(float).unique().tolist())
        anchor_margin_grid = sorted(old_grid["anchor_margin"].dropna().astype(float).unique().tolist())
    grid = evaluate(
        candidates,
        lambda_grid,
        anchor_margin_grid,
        args.output_cost_col,
    )
    selected = select_by_loss_tolerance(grid, parse_grid(args.loss_tolerances))
    grid.to_csv(args.output_dir / "raw_policy_grid_val_test.csv", index=False)
    selected.to_csv(args.output_dir / "raw_policy_selected_by_loss_tolerance.csv", index=False)
    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "candidates": str(args.candidates),
                "input_tokens_from": str(args.input_tokens_from) if args.input_tokens_from else None,
                "probe_cost_fraction": args.probe_cost_fraction,
                "input_price_fraction": args.input_price_fraction,
                "output_cost_col": args.output_cost_col,
                "parameter_grid_from": str(args.parameter_grid_from) if args.parameter_grid_from else None,
                "models": model_subset,
                "policies": ["raw_utility", "raw_anchor"],
                "score_col": "pred_success_raw",
                "decision_cost_col": "predicted_generation_cost_relative",
                "decision_cost_formula": "training-mean one-generation model cost / GPT-OSS-high cost",
                "probe_overhead_formula": "sum(probe_pass_cost_usd for all candidate models per item)",
                "probe_cost_source": str(candidates["probe_cost_source"].iloc[0]),
            },
            f,
            indent=2,
            sort_keys=True,
        )
    print(f"Wrote {args.output_dir / 'raw_policy_grid_val_test.csv'}")
    print(f"Wrote {args.output_dir / 'raw_policy_selected_by_loss_tolerance.csv'}")


if __name__ == "__main__":
    main()
