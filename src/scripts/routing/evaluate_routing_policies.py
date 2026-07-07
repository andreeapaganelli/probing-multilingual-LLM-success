"""Evaluate raw routing policies for encoder-target decoupled candidates.

Cost accounting:
  total = selected target output cost
        + one encoder input/prefill pass
        + one selected-target input/prefill pass

When the selected target is the encoder model, the encoder pass is reused and
the selected-target input pass is not charged a second time.
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


def parse_models(spec: str | None, anchor_model: str) -> list[str] | None:
    if spec is None or not spec.strip():
        return None
    models = [x.strip() for x in spec.split(",") if x.strip()]
    if anchor_model not in models:
        models.append(anchor_model)
    return models


def add_input_pass_cost(
    candidates: pd.DataFrame,
    probe_cost_fraction: float,
    input_price_fraction: float,
) -> pd.DataFrame:
    out = candidates.copy()
    if "input_tokens" in out.columns:
        out["input_pass_cost_usd"] = (
            out["input_tokens"].astype(float)
            * out["output_cost_per_million"].astype(float)
            * float(input_price_fraction)
            / 1_000_000.0
        )
        out["input_cost_source"] = "input_tokens"
    else:
        out["input_pass_cost_usd"] = out["tier_cost_usd"].astype(float) * float(probe_cost_fraction)
        out["input_cost_source"] = "tier_cost_fraction"
    return out


def add_relative_decision_cost(candidates: pd.DataFrame, anchor_model: str) -> pd.DataFrame:
    """Price routing decisions relative to the anchor's predicted generation cost."""
    out = candidates.copy()
    source = out[out["split"].astype(str).eq("probe_train")]
    if source.empty:
        source = out
    predicted_cost = source.groupby("model")["mean_output_cost_usd"].mean()
    if anchor_model not in predicted_cost.index:
        raise ValueError(f"Cost reference model {anchor_model!r} is absent from candidates")
    anchor_cost = float(predicted_cost.loc[anchor_model])
    if anchor_cost <= 0:
        raise ValueError("The anchor predicted generation cost must be positive")
    out["predicted_generation_cost_usd"] = out["model"].map(predicted_cost)
    if out["predicted_generation_cost_usd"].isna().any():
        raise ValueError("Missing training-mean generation cost for one or more candidate models")
    out["predicted_generation_cost_relative"] = out["predicted_generation_cost_usd"] / anchor_cost
    return out


def build_arrays(split_df: pd.DataFrame, encoder_model: str, output_cost_col: str) -> dict[str, Any]:
    models = (
        split_df.groupby("model", as_index=False)["tier_cost_usd"]
        .mean()
        .sort_values(["tier_cost_usd", "model"], kind="mergesort")["model"]
        .astype(str)
        .tolist()
    )
    if encoder_model in models:
        models = [m for m in models if m != encoder_model] + [encoder_model]

    item_index = split_df[ITEM_COLS].drop_duplicates().reset_index(drop=True)
    item_lookup = {tuple(row): i for i, row in enumerate(item_index[ITEM_COLS].itertuples(index=False, name=None))}
    model_lookup = {model: j for j, model in enumerate(models)}
    n_items, n_models = len(item_index), len(models)

    score = np.full((n_items, n_models), np.nan, dtype=np.float32)
    success = np.full((n_items, n_models), np.nan, dtype=np.float32)
    output_cost = np.full((n_items, n_models), np.nan, dtype=np.float64)
    input_cost = np.full((n_items, n_models), np.nan, dtype=np.float64)
    decision_cost = np.full((n_items, n_models), np.nan, dtype=np.float32)

    cols = [
        *ITEM_COLS,
        "model",
        "pred_success_raw",
        "true_success",
        output_cost_col,
        "input_pass_cost_usd",
        "predicted_generation_cost_relative",
    ]
    for row in split_df[cols].itertuples(index=False):
        i = item_lookup[(row.split, row.language, row.base_problem_id)]
        j = model_lookup[str(row.model)]
        score[i, j] = float(row.pred_success_raw)
        success[i, j] = float(row.true_success)
        output_cost[i, j] = float(getattr(row, output_cost_col))
        input_cost[i, j] = float(row.input_pass_cost_usd)
        decision_cost[i, j] = float(row.predicted_generation_cost_relative)

    encoder_idx = model_lookup[encoder_model]
    return {
        "split": str(split_df["split"].iloc[0]),
        "models": models,
        "encoder_idx": encoder_idx,
        "score": score,
        "success": success,
        "output_cost": output_cost,
        "input_cost": input_cost,
        "decision_cost": decision_cost,
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
    encoder_idx = int(arrays["encoder_idx"])
    selected_success = arrays["success"][row_idx, selected]
    selected_output = arrays["output_cost"][row_idx, selected]
    encoder_input = arrays["input_cost"][:, encoder_idx]
    selected_input = arrays["input_cost"][row_idx, selected]
    extra_selected_input = np.where(selected == encoder_idx, 0.0, selected_input)
    total = selected_output + encoder_input + extra_selected_input
    counts = np.bincount(selected, minlength=len(models)) / max(len(selected), 1)
    mix = {models[i]: float(v) for i, v in enumerate(counts) if v > 0}
    return {
        **params,
        "n_items": int(len(selected)),
        "mean_success": float(np.nanmean(selected_success)),
        "success_delta_pp_vs_anchor": float((np.nanmean(selected_success) - anchor_success) * 100.0),
        "mean_selected_output_cost_usd": float(np.nanmean(selected_output)),
        "mean_encoder_input_cost_usd": float(np.nanmean(encoder_input)),
        "mean_extra_selected_input_cost_usd": float(np.nanmean(extra_selected_input)),
        "mean_total_cost_usd": float(np.nanmean(total)),
        "cost_savings_pct_vs_anchor": float((1.0 - np.nanmean(total) / anchor_cost) * 100.0),
        "anchor_route_frac": float(np.mean(selected == anchor_idx)),
        "encoder_route_frac": float(np.mean(selected == encoder_idx)),
        "selected_model_mix": json.dumps(mix, sort_keys=True),
    }


def evaluate(
    candidates: pd.DataFrame,
    lambda_grid: list[float],
    anchor_margin_grid: list[float],
    anchor_model: str,
    encoder_model: str,
    output_cost_col: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, split_df in candidates[candidates["split"].isin(["val", "test"])].groupby("split", sort=False):
        arrays = build_arrays(split_df, encoder_model, output_cost_col)
        models = arrays["models"]
        anchor_idx = models.index(anchor_model)
        encoder_idx = int(arrays["encoder_idx"])
        anchor_cost = float(np.nanmean(arrays["output_cost"][:, anchor_idx] + arrays["input_cost"][:, anchor_idx]))
        anchor_success = float(np.nanmean(arrays["success"][:, anchor_idx]))
        row_idx = np.arange(arrays["score"].shape[0])

        for lambda_value in lambda_grid:
            utility = arrays["score"] - float(lambda_value) * arrays["decision_cost"]
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
                        "anchor_model": anchor_model,
                        "encoder_model": encoder_model,
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
                            "anchor_model": anchor_model,
                            "encoder_model": encoder_model,
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


def extract_matched_and_max(grid: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for policy, test in grid[grid["split"].eq("test")].groupby("policy", sort=False):
        anchor_success = test["mean_success"] - test["success_delta_pp_vs_anchor"] / 100.0
        matched = test.iloc[(test["success_delta_pp_vs_anchor"].abs()).argsort(kind="mergesort")[:1]]
        max_acc = test.sort_values(["mean_success", "cost_savings_pct_vs_anchor"], ascending=[False, False]).head(1)
        for kind, selected in [("matched_anchor_success", matched), ("max_accuracy", max_acc)]:
            row = selected.iloc[0].to_dict()
            row["selection_kind"] = kind
            row["anchor_success"] = float(anchor_success.loc[selected.index[0]])
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--probe-cost-fraction", type=float, required=True)
    parser.add_argument(
        "--input-price-fraction",
        type=float,
        default=1.0,
        help="Input-token price as a fraction of output_cost_per_million when exact input_tokens are available.",
    )
    parser.add_argument("--anchor-model", default=ANCHOR_MODEL)
    parser.add_argument("--encoder-model", default="gpt-oss-20b_high")
    parser.add_argument(
        "--output-cost-col",
        default="mean_output_cost_usd",
        choices=["mean_output_cost_usd", "total_output_cost_usd"],
        help=(
            "Output generation cost to charge for the selected target. "
            "Use mean_output_cost_usd for one deployed generation; "
            "total_output_cost_usd reproduces the old 5-rollout accounting."
        ),
    )
    parser.add_argument("--lambda-grid", default="0,0.01,0.02,0.05,0.10,0.20,0.35,0.50,0.75,1.0")
    parser.add_argument(
        "--anchor-margin-grid",
        default="0,0.005,0.01,0.015,0.02,0.025,0.03,0.04,0.05,0.075,0.10",
    )
    parser.add_argument("--loss-tolerances", default="0,0.5,1,2,3,5")
    parser.add_argument("--models", default=None, help="Comma-separated target model subset. Anchor is added if omitted.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=False)
    if args.input_price_fraction < 0:
        raise ValueError("--input-price-fraction must be non-negative")
    candidates = add_input_pass_cost(
        pd.read_csv(args.candidates),
        args.probe_cost_fraction,
        args.input_price_fraction,
    )
    model_subset = parse_models(args.models, args.anchor_model)
    if model_subset is not None:
        missing = sorted(set(model_subset) - set(candidates["model"].astype(str).unique()))
        if missing:
            raise ValueError(f"Requested model(s) not found in candidates: {missing}")
        candidates = candidates[candidates["model"].astype(str).isin(model_subset)].copy()
    if args.anchor_model not in set(candidates["model"].astype(str)):
        raise ValueError(f"Anchor model {args.anchor_model!r} not present in candidates")
    if args.encoder_model not in set(candidates["model"].astype(str)):
        raise ValueError(f"Encoder model {args.encoder_model!r} must also be present as a candidate for cost accounting")
    candidates = add_relative_decision_cost(candidates, args.anchor_model)
    if args.output_cost_col not in candidates.columns:
        raise ValueError(f"Output cost column {args.output_cost_col!r} not present in candidates")

    grid = evaluate(
        candidates,
        parse_grid(args.lambda_grid),
        parse_grid(args.anchor_margin_grid),
        anchor_model=args.anchor_model,
        encoder_model=args.encoder_model,
        output_cost_col=args.output_cost_col,
    )
    selected = select_by_loss_tolerance(grid, parse_grid(args.loss_tolerances))
    matched = extract_matched_and_max(grid)
    grid.to_csv(args.output_dir / "raw_policy_grid_val_test.csv", index=False)
    selected.to_csv(args.output_dir / "raw_policy_selected_by_loss_tolerance.csv", index=False)
    matched.to_csv(args.output_dir / "matched_anchor_and_max_accuracy_costs.csv", index=False)
    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "candidates": str(args.candidates),
                "probe_cost_fraction": args.probe_cost_fraction,
                "input_price_fraction": args.input_price_fraction,
                "anchor_model": args.anchor_model,
                "encoder_model": args.encoder_model,
                "models": model_subset,
                "policies": ["raw_utility", "raw_anchor"],
                "score_col": "pred_success_raw",
                "decision_cost_col": "predicted_generation_cost_relative",
                "decision_cost_formula": "training-mean one-generation model cost / anchor model cost",
                "output_cost_col": args.output_cost_col,
                "cost_formula": (
                    "selected output + encoder input pass + selected target input pass; "
                    "selected target input is reused when target == encoder"
                ),
                "input_cost_source": str(candidates["input_cost_source"].iloc[0]),
            },
            f,
            indent=2,
            sort_keys=True,
        )
    print(f"Wrote {args.output_dir / 'raw_policy_grid_val_test.csv'}")
    print(f"Wrote {args.output_dir / 'raw_policy_selected_by_loss_tolerance.csv'}")
    print(f"Wrote {args.output_dir / 'matched_anchor_and_max_accuracy_costs.csv'}")


if __name__ == "__main__":
    main()
