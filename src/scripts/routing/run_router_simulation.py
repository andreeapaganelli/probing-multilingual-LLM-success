"""Final thesis multi-model cost-aware routing experiment.

This is the operational counterpart to final_two_model_router.py. It keeps the
validation/test discipline, but allows the router to choose from a candidate
pool of models, which exposes a much wider accuracy-cost trade-off surface.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_ROUTING_ROOT = Path("outputs/routing/candidates")
DEFAULT_CONDITIONS = {
    "per_language": "Same-language probe",
    "pooled": "Pooled multilingual probe",
    "english_transfer": "English-only transfer probe",
}
MODEL_POOLS = {
    "all_8": [
        "Qwen3-0.6B",
        "Qwen3-1.7B",
        "Qwen3-4B",
        "Qwen3-8B",
        "Qwen3.5-4B",
        "Qwen3.5-9B",
        "gpt-oss-20b_low",
        "gpt-oss-20b_high",
    ],
    "qwen_plus_gpt_high": [
        "Qwen3-0.6B",
        "Qwen3-1.7B",
        "Qwen3-4B",
        "Qwen3-8B",
        "Qwen3.5-4B",
        "Qwen3.5-9B",
        "gpt-oss-20b_high",
    ],
    "premium_4": [
        "Qwen3.5-4B",
        "Qwen3-8B",
        "Qwen3.5-9B",
        "gpt-oss-20b_high",
    ],
    "qwen_only": [
        "Qwen3-0.6B",
        "Qwen3-1.7B",
        "Qwen3-4B",
        "Qwen3-8B",
        "Qwen3.5-4B",
        "Qwen3.5-9B",
    ],
}
DEFAULT_LAMBDA_GRID = "0,0.005,0.01,0.02,0.03,0.05,0.08,0.1,0.13,0.15,0.17,0.2,0.25,0.3,0.4,0.5"
DEFAULT_ANCHOR_MARGIN_GRID = "0,0.01,0.02,0.05"
DEFAULT_MAX_DROP_PP = "0,0.25,0.5,1,1.5,2,3,5"
ITEM_COLUMNS = ["language", "base_problem_id"]


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    label: str
    candidates_path: Path


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_float_grid(value: str) -> list[float]:
    spec = value.strip()
    if spec.startswith("linspace:"):
        parts = spec.split(":")
        if len(parts) != 4:
            raise ValueError("linspace grid must look like linspace:start:stop:count")
        start = float(parts[1])
        stop = float(parts[2])
        count = int(parts[3])
        return [round(float(x), 10) for x in np.linspace(start, stop, count)]
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_condition_specs(root: Path, value: str) -> list[ConditionSpec]:
    specs = []
    for name in parse_csv_list(value):
        specs.append(
            ConditionSpec(
                name=name,
                label=DEFAULT_CONDITIONS.get(name, name.replace("_", " ").title()),
                candidates_path=root / name / "router_candidates.csv",
            )
        )
    return specs


def resolve_model_pool(pool_name: str, models: str | None) -> list[str]:
    if models:
        return parse_csv_list(models)
    if pool_name not in MODEL_POOLS:
        raise ValueError(f"Unknown pool {pool_name}. Available: {sorted(MODEL_POOLS)}")
    return MODEL_POOLS[pool_name]


def load_candidates(
    path: Path,
    models: list[str],
    score_col: str,
    languages: set[str] | None,
    require_all_models: bool,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing candidates CSV: {path}")
    needed = {
        "language",
        "base_problem_id",
        "problem_id",
        "split",
        "model",
        "true_success",
        score_col,
        "cost_norm",
        "output_cost_per_million",
        "total_output_cost_usd",
        "total_output_tokens",
    }
    df = pd.read_csv(path, usecols=lambda col: col in needed)
    missing = sorted(needed - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    if languages is not None:
        df = df[df["language"].isin(languages)].copy()
    df = df[df["model"].isin(models)].copy()
    if df.empty:
        raise RuntimeError(f"No rows remain in {path} for selected models.")

    item_cols = ["split", *ITEM_COLUMNS]
    counts = df.groupby(item_cols, as_index=False)["model"].nunique().rename(columns={"model": "n_models"})
    required = len(models) if require_all_models else 2
    eligible = counts[counts["n_models"] >= required][item_cols]
    df = df.merge(eligible, on=item_cols, how="inner")
    return df.sort_values([*item_cols, "model"]).reset_index(drop=True)


def add_expected_costs(candidates: pd.DataFrame) -> pd.DataFrame:
    """Estimate deploy-time output cost from non-test output lengths.

    Actual test output length is only known after generation. For a deployment
    simulation, use model-level expected token counts estimated from probe_train
    rows, falling back to the global mean. This intentionally avoids
    language-conditioned accounting because language may not be available before
    routing.
    """
    out = candidates.copy()
    if "output_cost_per_million" not in out.columns:
        out["output_cost_per_million"] = np.where(
            out["total_output_tokens"].astype(float) > 0,
            out["total_output_cost_usd"].astype(float) * 1_000_000.0 / out["total_output_tokens"].astype(float),
            np.nan,
        )
    out["output_cost_per_million"] = out["output_cost_per_million"].astype(float)
    source = out[out["split"] == "probe_train"].copy()
    if source.empty:
        source = out[out["split"] == "val"].copy()
    if source.empty:
        source = out

    model_tokens = source.groupby("model")["total_output_tokens"].mean()
    global_tokens = float(source["total_output_tokens"].mean())

    model_prices = source.groupby("model")["output_cost_per_million"].mean()
    global_price = float(source["output_cost_per_million"].mean())

    expected_tokens = []
    expected_prices = []
    for model in out["model"]:
        expected_tokens.append(float(model_tokens.get(model, global_tokens)))
        expected_prices.append(float(model_prices.get(model, global_price)))
    out["expected_total_output_tokens"] = np.asarray(expected_tokens, dtype=float)
    out["expected_output_cost_per_million"] = np.asarray(expected_prices, dtype=float)
    out["expected_total_output_cost_usd"] = (
        out["expected_total_output_tokens"]
        * out["expected_output_cost_per_million"]
        / 1_000_000.0
    )
    out["fixed_model_cost"] = out["expected_total_output_cost_usd"]
    return out


def item_arrays(split_df: pd.DataFrame, models: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    for key, group in split_df.groupby(ITEM_COLUMNS, sort=True):
        g = group.set_index("model").reindex(models)
        if g["true_success"].isna().any():
            continue
        language, base_problem_id = key
        rows.append(
            {
                "language": language,
                "base_problem_id": base_problem_id,
                "true_success": g["true_success"].to_numpy(dtype=float),
                "score": g["_score"].to_numpy(dtype=float),
                "cost_norm": g["cost_norm"].to_numpy(dtype=float),
                "observed_cost_usd": g["total_output_cost_usd"].to_numpy(dtype=float),
                "observed_tokens": g["total_output_tokens"].to_numpy(dtype=float),
                "expected_cost_usd": g["expected_total_output_cost_usd"].to_numpy(dtype=float),
                "expected_tokens": g["expected_total_output_tokens"].to_numpy(dtype=float),
                "fixed_model_cost": g["fixed_model_cost"].to_numpy(dtype=float),
            }
        )
    items = pd.DataFrame(rows)
    arrays = {
        "success": np.stack(items["true_success"].to_numpy()),
        "score": np.stack(items["score"].to_numpy()),
        "cost_norm": np.stack(items["cost_norm"].to_numpy()),
        "observed_cost_usd": np.stack(items["observed_cost_usd"].to_numpy()),
        "observed_tokens": np.stack(items["observed_tokens"].to_numpy()),
        "expected_cost_usd": np.stack(items["expected_cost_usd"].to_numpy()),
        "expected_tokens": np.stack(items["expected_tokens"].to_numpy()),
        "fixed_model_cost": np.stack(items["fixed_model_cost"].to_numpy()),
    }
    return items, arrays


def summarize_indices(
    items: pd.DataFrame,
    arrays: dict[str, Any],
    selected: np.ndarray,
    models: list[str],
    anchor_idx: int,
    policy: str,
    split: str,
    accounting_cost_source: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selected = selected.astype(int)
    row_idx = np.arange(len(selected))
    success = arrays["success"][row_idx, selected]
    observed_cost = arrays["observed_cost_usd"][row_idx, selected]
    observed_tokens = arrays["observed_tokens"][row_idx, selected]
    expected_cost = arrays["expected_cost_usd"][row_idx, selected]
    expected_tokens = arrays["expected_tokens"][row_idx, selected]
    fixed_model_cost = arrays["fixed_model_cost"][row_idx, selected]
    if accounting_cost_source == "observed":
        cost = observed_cost
        anchor_cost = arrays["observed_cost_usd"][:, anchor_idx]
    elif accounting_cost_source == "expected":
        cost = expected_cost
        anchor_cost = arrays["expected_cost_usd"][:, anchor_idx]
    elif accounting_cost_source == "fixed_model":
        cost = fixed_model_cost
        anchor_cost = arrays["fixed_model_cost"][:, anchor_idx]
    else:
        raise ValueError(f"Unknown accounting cost source: {accounting_cost_source}")
    anchor_success = arrays["success"][:, anchor_idx]
    chosen_models = [models[i] for i in selected]
    mix = pd.Series(chosen_models).value_counts(normalize=True).to_dict()
    cost_contrib = (
        pd.DataFrame({"model": chosen_models, "cost": cost})
        .groupby("model")["cost"]
        .sum()
    )
    cost_contrib = (cost_contrib / cost_contrib.sum()).to_dict() if float(cost.sum()) > 0 else {}
    row = {
        "policy": policy,
        "split": split,
        "n_items": int(len(selected)),
        "mean_success": float(success.mean()),
        "mean_output_cost_usd": float(cost.mean()),
        "total_output_cost_usd": float(cost.sum()),
        "mean_output_tokens": (
            float(expected_tokens.mean())
            if accounting_cost_source == "expected"
            else float(observed_tokens.mean())
            if accounting_cost_source == "observed"
            else np.nan
        ),
        "total_output_tokens": (
            float(expected_tokens.sum())
            if accounting_cost_source == "expected"
            else float(observed_tokens.sum())
            if accounting_cost_source == "observed"
            else np.nan
        ),
        "accounting_cost_source": accounting_cost_source,
        "mean_observed_output_cost_usd": float(observed_cost.mean()),
        "total_observed_output_cost_usd": float(observed_cost.sum()),
        "mean_expected_output_cost_usd": float(expected_cost.mean()),
        "total_expected_output_cost_usd": float(expected_cost.sum()),
        "mean_fixed_model_cost": float(fixed_model_cost.mean()),
        "total_fixed_model_cost": float(fixed_model_cost.sum()),
        "mean_observed_output_tokens": float(observed_tokens.mean()),
        "mean_expected_output_tokens": float(expected_tokens.mean()),
        "cost_ratio_vs_anchor": float(cost.mean() / anchor_cost.mean()) if anchor_cost.mean() > 0 else np.nan,
        "cost_savings_pct_vs_anchor": float((1.0 - cost.mean() / anchor_cost.mean()) * 100.0)
        if anchor_cost.mean() > 0
        else np.nan,
        "success_delta_pp_vs_anchor": float((success.mean() - anchor_success.mean()) * 100.0),
        "anchor_route_frac": float(np.mean(selected == anchor_idx)),
        "selected_model_mix": json.dumps(mix, sort_keys=True),
        "selected_cost_contribution_mix": json.dumps(cost_contrib, sort_keys=True),
        "lambda": np.nan,
        "anchor_margin": np.nan,
        "threshold": np.nan,
    }
    if params:
        row.update(params)
    return row


def evaluate_split(
    candidates: pd.DataFrame,
    models: list[str],
    anchor_model: str,
    score_col: str,
    lambda_grid: list[float],
    anchor_margin_grid: list[float],
    accounting_cost_source: str,
    split: str,
) -> pd.DataFrame:
    split_df = candidates[candidates["split"] == split].copy()
    split_df["_score"] = split_df[score_col].astype(float)
    items, arrays = item_arrays(split_df, models)
    anchor_idx = models.index(anchor_model)
    rows: list[dict[str, Any]] = []
    n = len(items)
    row_idx = np.arange(n)

    for i, model in enumerate(models):
        rows.append(
            summarize_indices(
                items,
                arrays,
                np.full(n, i),
                models,
                anchor_idx,
                f"always:{model}",
                split,
                accounting_cost_source,
            )
        )

    oracle_utility = arrays["success"] - 1e-6 * arrays["cost_norm"]
    rows.append(
        summarize_indices(
            items,
            arrays,
            np.argmax(oracle_utility, axis=1),
            models,
            anchor_idx,
            "oracle_best_success_cheapest_tie",
            split,
            accounting_cost_source,
        )
    )

    for lambda_value in lambda_grid:
        utility = arrays["score"] - lambda_value * arrays["cost_norm"]
        selected = np.argmax(utility, axis=1)
        rows.append(
            summarize_indices(
                items,
                arrays,
                selected,
                models,
                anchor_idx,
                "raw_utility",
                split,
                accounting_cost_source,
                {"lambda": lambda_value},
            )
        )
        for margin in anchor_margin_grid:
            best_utility = utility[row_idx, selected]
            anchor_utility = utility[:, anchor_idx]
            anchored = np.where(anchor_utility >= best_utility - margin, anchor_idx, selected)
            rows.append(
                summarize_indices(
                    items,
                    arrays,
                    anchored,
                    models,
                    anchor_idx,
                    "raw_anchor",
                    split,
                    accounting_cost_source,
                    {"lambda": lambda_value, "anchor_margin": margin},
                )
            )

    out = pd.DataFrame(rows)
    out["n_languages"] = int(items["language"].nunique()) if not items.empty else 0
    return out


def evaluate_condition(
    spec: ConditionSpec,
    candidates: pd.DataFrame,
    models: list[str],
    anchor_model: str,
    score_col: str,
    lambda_grid: list[float],
    anchor_margin_grid: list[float],
    accounting_cost_source: str,
) -> pd.DataFrame:
    frames = [
        evaluate_split(
            candidates,
            models,
            anchor_model,
            score_col,
            lambda_grid,
            anchor_margin_grid,
            accounting_cost_source,
            split,
        )
        for split in ["val", "test"]
    ]
    out = pd.concat(frames, ignore_index=True)
    out["condition"] = spec.name
    out["condition_label"] = spec.label
    return out


def select_operating_points(
    curve: pd.DataFrame,
    max_drop_pp_grid: list[float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for condition, group in curve.groupby("condition", sort=True):
        label = str(group["condition_label"].iloc[0])
        val = group[group["split"] == "val"].copy()
        test = group[group["split"] == "test"].copy()
        anchor_val = val[val["policy"].str.startswith("always:", na=False)]
        anchor_val = anchor_val[anchor_val["cost_ratio_vs_anchor"].round(8).eq(1.0)].iloc[0]
        anchor_test = test[test["policy"].str.startswith("always:", na=False)]
        anchor_test = anchor_test[anchor_test["cost_ratio_vs_anchor"].round(8).eq(1.0)].iloc[0]
        deployable_val = val[val["policy"].isin(["raw_utility", "raw_anchor"])].copy()
        deployable_test = test[test["policy"].isin(["raw_utility", "raw_anchor"])].copy()

        best_val = deployable_val.sort_values(
            ["mean_success", "mean_output_cost_usd"],
            ascending=[False, True],
        ).iloc[0]
        mask = deployable_test["policy"].eq(best_val["policy"])
        for col in ["lambda", "anchor_margin", "threshold"]:
            val_value = best_val[col]
            if pd.isna(val_value):
                mask &= deployable_test[col].isna()
            else:
                mask &= np.isclose(deployable_test[col].astype(float), float(val_value), equal_nan=True)
        best_test = deployable_test[mask].iloc[0]
        rows.append(
            {
                "condition": condition,
                "condition_label": label,
                "selection_type": "best_val_success",
                "selection_value": "best",
                "status": "ok",
                "policy": best_val["policy"],
                "lambda": best_val["lambda"],
                "anchor_margin": best_val["anchor_margin"],
                "threshold": best_val["threshold"],
                "val_success": best_val["mean_success"],
                "val_cost_savings_pct_vs_anchor": best_val["cost_savings_pct_vs_anchor"],
                "test_success": best_test["mean_success"],
                "test_cost_savings_pct_vs_anchor": best_test["cost_savings_pct_vs_anchor"],
                "test_success_delta_pp_vs_anchor": best_test["success_delta_pp_vs_anchor"],
                "test_anchor_route_frac": best_test["anchor_route_frac"],
                "test_n_items": int(best_test["n_items"]),
                "anchor_test_success": anchor_test["mean_success"],
            }
        )

        for max_drop_pp in max_drop_pp_grid:
            min_success = float(anchor_val["mean_success"]) - max_drop_pp / 100.0
            eligible = deployable_val[deployable_val["mean_success"] >= min_success].copy()
            if eligible.empty:
                rows.append(
                    {
                        "condition": condition,
                        "condition_label": label,
                        "selection_type": "max_drop_pp",
                        "selection_value": f"{max_drop_pp:g}",
                        "status": "no_validation_policy",
                    }
                )
                continue
            val_row = eligible.sort_values(
                ["mean_output_cost_usd", "mean_success"],
                ascending=[True, False],
            ).iloc[0]
            mask = deployable_test["policy"].eq(val_row["policy"])
            for col in ["lambda", "anchor_margin", "threshold"]:
                val_value = val_row[col]
                if pd.isna(val_value):
                    mask &= deployable_test[col].isna()
                else:
                    mask &= np.isclose(deployable_test[col].astype(float), float(val_value), equal_nan=True)
            test_row = deployable_test[mask].iloc[0]
            rows.append(
                {
                    "condition": condition,
                    "condition_label": label,
                    "selection_type": "max_drop_pp",
                    "selection_value": f"{max_drop_pp:g}",
                    "status": "ok",
                    "policy": val_row["policy"],
                    "lambda": val_row["lambda"],
                    "anchor_margin": val_row["anchor_margin"],
                    "threshold": val_row["threshold"],
                    "val_success": val_row["mean_success"],
                    "val_cost_savings_pct_vs_anchor": val_row["cost_savings_pct_vs_anchor"],
                    "test_success": test_row["mean_success"],
                    "test_cost_savings_pct_vs_anchor": test_row["cost_savings_pct_vs_anchor"],
                    "test_success_delta_pp_vs_anchor": test_row["success_delta_pp_vs_anchor"],
                    "test_anchor_route_frac": test_row["anchor_route_frac"],
                    "test_n_items": int(test_row["n_items"]),
                    "anchor_test_success": anchor_test["mean_success"],
                }
            )
    return pd.DataFrame(rows)


def _matching_test_row(deployable_test: pd.DataFrame, val_row: pd.Series) -> pd.Series:
    mask = deployable_test["policy"].eq(val_row["policy"])
    for col in ["lambda", "anchor_margin", "threshold"]:
        val_value = val_row[col]
        if pd.isna(val_value):
            mask &= deployable_test[col].isna()
        else:
            mask &= np.isclose(deployable_test[col].astype(float), float(val_value), equal_nan=True)
    return deployable_test[mask].iloc[0]


def select_anchor_comparison(
    curve: pd.DataFrame,
    max_drop_pp_grid: list[float],
) -> pd.DataFrame:
    """Compare unanchored raw utility and anchored raw utility under matched validation rules."""
    rows: list[dict[str, Any]] = []
    family_specs = [
        ("without_anchor", ["raw_utility"]),
        ("with_anchor", ["raw_anchor"]),
    ]
    for condition, group in curve.groupby("condition", sort=True):
        label = str(group["condition_label"].iloc[0])
        val = group[group["split"] == "val"].copy()
        test = group[group["split"] == "test"].copy()
        anchor_val = val[val["policy"].str.startswith("always:", na=False)]
        anchor_val = anchor_val[anchor_val["cost_ratio_vs_anchor"].round(8).eq(1.0)].iloc[0]
        anchor_test = test[test["policy"].str.startswith("always:", na=False)]
        anchor_test = anchor_test[anchor_test["cost_ratio_vs_anchor"].round(8).eq(1.0)].iloc[0]

        for family_name, policies in family_specs:
            val_family = val[val["policy"].isin(policies)].copy()
            test_family = test[test["policy"].isin(policies)].copy()
            if family_name == "with_anchor":
                val_family = val_family[val_family["anchor_margin"].astype(float) > 0.0].copy()
                test_family = test_family[test_family["anchor_margin"].astype(float) > 0.0].copy()
            if val_family.empty:
                continue

            best_val = val_family.sort_values(
                ["mean_success", "mean_output_cost_usd"],
                ascending=[False, True],
            ).iloc[0]
            best_test = _matching_test_row(test_family, best_val)
            rows.append(
                {
                    "condition": condition,
                    "condition_label": label,
                    "router_family": family_name,
                    "selection_type": "best_val_success",
                    "selection_value": "best",
                    "status": "ok",
                    "policy": best_val["policy"],
                    "lambda": best_val["lambda"],
                    "anchor_margin": best_val["anchor_margin"],
                    "threshold": best_val["threshold"],
                    "val_success": best_val["mean_success"],
                    "val_cost_savings_pct_vs_anchor": best_val["cost_savings_pct_vs_anchor"],
                    "test_success": best_test["mean_success"],
                    "test_cost_savings_pct_vs_anchor": best_test["cost_savings_pct_vs_anchor"],
                    "test_success_delta_pp_vs_anchor": best_test["success_delta_pp_vs_anchor"],
                    "test_anchor_route_frac": best_test["anchor_route_frac"],
                    "test_n_items": int(best_test["n_items"]),
                    "anchor_test_success": anchor_test["mean_success"],
                }
            )

            for max_drop_pp in max_drop_pp_grid:
                min_success = float(anchor_val["mean_success"]) - max_drop_pp / 100.0
                eligible = val_family[val_family["mean_success"] >= min_success].copy()
                if eligible.empty:
                    rows.append(
                        {
                            "condition": condition,
                            "condition_label": label,
                            "router_family": family_name,
                            "selection_type": "max_drop_pp",
                            "selection_value": f"{max_drop_pp:g}",
                            "status": "no_validation_policy",
                        }
                    )
                    continue
                val_row = eligible.sort_values(
                    ["mean_output_cost_usd", "mean_success"],
                    ascending=[True, False],
                ).iloc[0]
                test_row = _matching_test_row(test_family, val_row)
                rows.append(
                    {
                        "condition": condition,
                        "condition_label": label,
                        "router_family": family_name,
                        "selection_type": "max_drop_pp",
                        "selection_value": f"{max_drop_pp:g}",
                        "status": "ok",
                        "policy": val_row["policy"],
                        "lambda": val_row["lambda"],
                        "anchor_margin": val_row["anchor_margin"],
                        "threshold": val_row["threshold"],
                        "val_success": val_row["mean_success"],
                        "val_cost_savings_pct_vs_anchor": val_row["cost_savings_pct_vs_anchor"],
                        "test_success": test_row["mean_success"],
                        "test_cost_savings_pct_vs_anchor": test_row["cost_savings_pct_vs_anchor"],
                        "test_success_delta_pp_vs_anchor": test_row["success_delta_pp_vs_anchor"],
                        "test_anchor_route_frac": test_row["anchor_route_frac"],
                        "test_n_items": int(test_row["n_items"]),
                        "anchor_test_success": anchor_test["mean_success"],
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    paired = out.pivot_table(
        index=["condition", "selection_type", "selection_value"],
        columns="router_family",
        values=["test_success", "test_cost_savings_pct_vs_anchor", "test_success_delta_pp_vs_anchor"],
        aggfunc="first",
    )
    paired.columns = [f"{metric}_{family}" for metric, family in paired.columns]
    paired = paired.reset_index()
    out = out.merge(paired, on=["condition", "selection_type", "selection_value"], how="left")
    if {
        "test_success_with_anchor",
        "test_success_without_anchor",
        "test_cost_savings_pct_vs_anchor_with_anchor",
        "test_cost_savings_pct_vs_anchor_without_anchor",
    }.issubset(out.columns):
        out["delta_success_pp_with_minus_without_anchor"] = (
            out["test_success_with_anchor"] - out["test_success_without_anchor"]
        ) * 100.0
        out["delta_savings_pct_with_minus_without_anchor"] = (
            out["test_cost_savings_pct_vs_anchor_with_anchor"]
            - out["test_cost_savings_pct_vs_anchor_without_anchor"]
        )
    return out


def build_thesis_table(
    curve: pd.DataFrame,
    operating: pd.DataFrame,
    selection_type: str,
    selection_value: str,
) -> pd.DataFrame:
    first_condition = str(curve["condition"].iloc[0])
    test = curve[(curve["split"] == "test") & (curve["condition"] == first_condition)]
    baseline_policies = [
        "always:Qwen3-0.6B",
        "always:Qwen3-1.7B",
        "always:Qwen3-4B",
        "always:Qwen3-8B",
        "always:Qwen3.5-4B",
        "always:Qwen3.5-9B",
        "always:gpt-oss-20b_low",
        "always:gpt-oss-20b_high",
        "oracle_best_success_cheapest_tie",
    ]
    rows: list[dict[str, Any]] = []
    for policy in baseline_policies:
        match = test[test["policy"] == policy]
        if match.empty:
            continue
        row = match.iloc[0]
        rows.append(
            {
                "router": policy,
                "condition": "baseline",
                "policy": policy,
                "test_success": row["mean_success"],
                "test_cost_savings_pct_vs_anchor": row["cost_savings_pct_vs_anchor"],
                "test_success_delta_pp_vs_anchor": row["success_delta_pp_vs_anchor"],
                "test_anchor_route_frac": row["anchor_route_frac"],
                "lambda": np.nan,
                "anchor_margin": np.nan,
                "threshold": np.nan,
                "test_n_items": row["n_items"],
            }
        )

    selected = operating[
        (operating["status"] == "ok")
        & (operating["selection_type"] == selection_type)
        & (operating["selection_value"] == selection_value)
    ]
    for _, row in selected.iterrows():
        rows.append(
            {
                "router": row["condition_label"],
                "condition": row["condition"],
                "policy": row["policy"],
                "test_success": row["test_success"],
                "test_cost_savings_pct_vs_anchor": row["test_cost_savings_pct_vs_anchor"],
                "test_success_delta_pp_vs_anchor": row["test_success_delta_pp_vs_anchor"],
                "test_anchor_route_frac": row["test_anchor_route_frac"],
                "lambda": row["lambda"],
                "anchor_margin": row["anchor_margin"],
                "threshold": row["threshold"],
                "test_n_items": row["test_n_items"],
            }
        )
    return pd.DataFrame(rows)


def per_language_rows(
    spec: ConditionSpec,
    candidates: pd.DataFrame,
    models: list[str],
    anchor_model: str,
    score_col: str,
    selected_policy: pd.Series,
    accounting_cost_source: str,
) -> pd.DataFrame:
    rows = []
    for language, lang_df in candidates[candidates["split"] == "test"].groupby("language", sort=True):
        curve = evaluate_split(
            lang_df,
            models,
            anchor_model,
            score_col,
            lambda_grid=[float(selected_policy["lambda"])],
            anchor_margin_grid=[] if selected_policy["policy"] != "raw_anchor" else [float(selected_policy["anchor_margin"])],
            accounting_cost_source=accounting_cost_source,
            split="test",
        )
        row = curve[curve["policy"] == selected_policy["policy"]].iloc[0]
        anchor = curve[curve["policy"] == f"always:{anchor_model}"].iloc[0]
        rows.append(
            {
                "condition": spec.name,
                "condition_label": spec.label,
                "language": language,
                "policy": selected_policy["policy"],
                "test_success": row["mean_success"],
                "always_anchor_success": anchor["mean_success"],
                "cost_savings_pct_vs_anchor": row["cost_savings_pct_vs_anchor"],
                "success_delta_pp_vs_anchor": row["success_delta_pp_vs_anchor"],
                "anchor_route_frac": row["anchor_route_frac"],
                "selected_model_mix": row["selected_model_mix"],
                "n_items": row["n_items"],
            }
        )
    return pd.DataFrame(rows)


def plot_curve(curve: pd.DataFrame, output_dir: Path) -> None:
    test = curve[
        (curve["split"] == "test")
        & (curve["policy"].isin(["raw_utility", "raw_anchor"]))
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for condition, group in test.groupby("condition", sort=True):
        label = str(group["condition_label"].iloc[0])
        pareto = group.sort_values(["cost_savings_pct_vs_anchor", "mean_success"])
        ax.scatter(
            100.0 - pareto["cost_savings_pct_vs_anchor"],
            pareto["mean_success"],
            s=8,
            alpha=0.45,
            label=label,
        )
    anchor = curve[(curve["split"] == "test") & curve["policy"].eq("always:gpt-oss-20b_high")]
    if not anchor.empty:
        ax.scatter(
            100.0 - anchor["cost_savings_pct_vs_anchor"],
            anchor["mean_success"],
            marker="x",
            color="black",
            s=50,
            label="Always anchor",
        )
    ax.set_xlabel("Cost ratio vs anchor (%)")
    ax.set_ylabel("Average empirical success")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_cost_scatter_test.png", dpi=300)
    fig.savefig(output_dir / "accuracy_cost_scatter_test.pdf")
    plt.close(fig)


def plot_anchor_comparison(comparison: pd.DataFrame, output_dir: Path) -> None:
    rows = comparison[
        (comparison["status"] == "ok")
        & (comparison["selection_type"] == "max_drop_pp")
        & (comparison["condition"].isin(["pooled", "english_transfer"]))
    ].copy()
    if rows.empty:
        return
    rows["selection_value_float"] = rows["selection_value"].astype(float)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), sharex=True)
    styles = {
        ("pooled", "with_anchor"): ("Pooled, with anchor", "tab:blue", "-"),
        ("pooled", "without_anchor"): ("Pooled, no anchor", "tab:blue", "--"),
        ("english_transfer", "with_anchor"): ("English, with anchor", "tab:orange", "-"),
        ("english_transfer", "without_anchor"): ("English, no anchor", "tab:orange", "--"),
    }
    for (condition, family), group in rows.groupby(["condition", "router_family"], sort=True):
        label, color, linestyle = styles.get(
            (condition, family),
            (f"{condition} {family}", "gray", "-"),
        )
        group = group.sort_values("selection_value_float")
        axes[0].plot(
            group["selection_value_float"],
            group["test_success"],
            label=label,
            color=color,
            linestyle=linestyle,
            marker="o",
            markersize=3,
        )
        axes[1].plot(
            group["selection_value_float"],
            group["test_cost_savings_pct_vs_anchor"],
            label=label,
            color=color,
            linestyle=linestyle,
            marker="o",
            markersize=3,
        )
    axes[0].set_ylabel("Test success")
    axes[1].set_ylabel("Cost savings vs anchor (%)")
    for ax in axes:
        ax.set_xlabel("Validation max-drop budget (pp)")
        ax.grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "anchor_vs_no_anchor_operating_points.png", dpi=300)
    fig.savefig(output_dir / "anchor_vs_no_anchor_operating_points.pdf")
    plt.close(fig)


def write_latex(path: Path, table: pd.DataFrame) -> None:
    if table.empty:
        path.write_text("", encoding="utf-8")
        return
    pretty = table.copy()
    for col in pretty.columns:
        if pd.api.types.is_float_dtype(pretty[col]):
            pretty[col] = pretty[col].map(lambda x: f"{float(x):.3f}" if pd.notna(x) else "")
    def esc(x: Any) -> str:
        text = str(x)
        for old, new in {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
        }.items():
            text = text.replace(old, new)
        return text
    columns = list(pretty.columns)
    lines = [r"\begin{tabular}{" + "l" * len(columns) + "}", r"\toprule"]
    lines.append(" & ".join(esc(c) for c in columns) + r" \\")
    lines.append(r"\midrule")
    for row in pretty.itertuples(index=False, name=None):
        lines.append(" & ".join(esc(x) for x in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routing-root", type=Path, default=DEFAULT_ROUTING_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/routing/simulation"))
    parser.add_argument("--conditions", type=str, default="per_language,pooled,english_transfer")
    parser.add_argument("--pool", type=str, default="all_8", choices=sorted(MODEL_POOLS))
    parser.add_argument("--models", type=str, default=None)
    parser.add_argument("--anchor-model", type=str, default="gpt-oss-20b_high")
    parser.add_argument("--score-col", type=str, default="pred_success_raw")
    parser.add_argument("--lambda-grid", type=str, default=DEFAULT_LAMBDA_GRID)
    parser.add_argument("--anchor-margin-grid", type=str, default=DEFAULT_ANCHOR_MARGIN_GRID)
    parser.add_argument("--max-drop-pp-grid", type=str, default=DEFAULT_MAX_DROP_PP)
    parser.add_argument("--languages", type=str, default=None)
    parser.add_argument("--require-all-models", action="store_true", default=True)
    parser.add_argument(
        "--accounting-cost-source",
        choices=["expected", "observed", "fixed_model"],
        default="expected",
        help=(
            "Cost source used for reported savings. expected estimates output length from "
            "non-test rows; observed uses realized output lengths for selected test examples; "
            "fixed_model uses one fixed cost per selected model, independent of output length."
        ),
    )
    parser.add_argument(
        "--main-selection-type",
        type=str,
        default="best_val_success",
        choices=["best_val_success", "max_drop_pp"],
        help="Operating-point selection type used in main_results_table.csv.",
    )
    parser.add_argument(
        "--main-selection-value",
        type=str,
        default="best",
        help="Operating-point selection value used in main_results_table.csv.",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    models = resolve_model_pool(args.pool, args.models)
    if args.anchor_model not in models:
        raise ValueError(f"Anchor model {args.anchor_model} must be in selected model pool.")
    conditions = parse_condition_specs(args.routing_root, args.conditions)
    languages = set(parse_csv_list(args.languages)) if args.languages else None
    lambda_grid = parse_float_grid(args.lambda_grid)
    anchor_margin_grid = parse_float_grid(args.anchor_margin_grid)
    max_drop_pp_grid = parse_float_grid(args.max_drop_pp_grid)

    curves = []
    candidate_cache: dict[str, pd.DataFrame] = {}
    coverage_rows = []
    for spec in conditions:
        candidates = load_candidates(
            spec.candidates_path,
            models=models,
            score_col=args.score_col,
            languages=languages,
            require_all_models=args.require_all_models,
        )
        candidates = add_expected_costs(candidates)
        candidate_cache[spec.name] = candidates
        coverage_rows.append(
            {
                "condition": spec.name,
                "condition_label": spec.label,
                "n_candidate_rows": len(candidates),
                "n_val_items": candidates[candidates["split"] == "val"][ITEM_COLUMNS].drop_duplicates().shape[0],
                "n_test_items": candidates[candidates["split"] == "test"][ITEM_COLUMNS].drop_duplicates().shape[0],
                "n_languages": candidates["language"].nunique(),
                "models": ",".join(models),
                "anchor_model": args.anchor_model,
                "candidates_path": str(spec.candidates_path),
            }
        )
        curves.append(
            evaluate_condition(
                spec,
                candidates,
                models=models,
                anchor_model=args.anchor_model,
                score_col=args.score_col,
                lambda_grid=lambda_grid,
                anchor_margin_grid=anchor_margin_grid,
                accounting_cost_source=args.accounting_cost_source,
            )
        )

    curve = pd.concat(curves, ignore_index=True)
    operating = select_operating_points(curve, max_drop_pp_grid=max_drop_pp_grid)
    anchor_comparison = select_anchor_comparison(curve, max_drop_pp_grid=max_drop_pp_grid)
    thesis = build_thesis_table(
        curve,
        operating,
        selection_type=args.main_selection_type,
        selection_value=args.main_selection_value,
    )

    language_frames = []
    selected = operating[
        (operating["status"] == "ok")
        & (operating["selection_type"] == args.main_selection_type)
        & (operating["selection_value"] == args.main_selection_value)
    ]
    for spec in conditions:
        match = selected[selected["condition"] == spec.name]
        if not match.empty:
            language_frames.append(
                per_language_rows(
                    spec,
                    candidate_cache[spec.name],
                    models,
                    args.anchor_model,
                    args.score_col,
                    match.iloc[0],
                    args.accounting_cost_source,
                )
            )
    per_language = pd.concat(language_frames, ignore_index=True) if language_frames else pd.DataFrame()

    curve.to_csv(args.output_dir / "policy_grid_val_test.csv", index=False)
    operating.to_csv(args.output_dir / "routing_operating_points.csv", index=False)
    anchor_comparison.to_csv(args.output_dir / "anchor_vs_no_anchor_operating_points.csv", index=False)
    thesis.to_csv(args.output_dir / "main_results_table.csv", index=False)
    per_language.to_csv(args.output_dir / "per_language_test.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(args.output_dir / "coverage.csv", index=False)
    write_latex(args.output_dir / "main_results_table.tex", thesis)
    if not args.no_plots:
        plot_curve(curve, args.output_dir)
        plot_anchor_comparison(anchor_comparison, args.output_dir)
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "routing_root": str(args.routing_root),
                "output_dir": str(args.output_dir),
                "conditions": [spec.name for spec in conditions],
                "pool": args.pool,
                "models": models,
                "anchor_model": args.anchor_model,
                "score_col": args.score_col,
                "lambda_grid": lambda_grid,
                "anchor_margin_grid": anchor_margin_grid,
                "max_drop_pp_grid": max_drop_pp_grid,
                "accounting_cost_source": args.accounting_cost_source,
                "main_selection_type": args.main_selection_type,
                "main_selection_value": args.main_selection_value,
                "selection_rule": (
                    "Only raw_utility and raw_anchor routers are evaluated. For each condition and "
                    "max_drop_pp, choose on validation the cheapest deployable policy whose validation "
                    "success is within max_drop_pp of always-anchor success; evaluate the selected "
                    "policy once on test."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote final multi-model router to {args.output_dir}")
    print(pd.DataFrame(coverage_rows).to_string(index=False))
    main = operating[
        (operating["status"] == "ok")
        & (operating["selection_type"] == args.main_selection_type)
        & (operating["selection_value"] == args.main_selection_value)
    ]
    print("\nMain selected operating point:")
    print(
        main[
            [
                "condition",
                "policy",
                "lambda",
                "anchor_margin",
                "threshold",
                "test_success",
                "test_cost_savings_pct_vs_anchor",
                "test_success_delta_pp_vs_anchor",
                "test_anchor_route_frac",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
