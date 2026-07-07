"""Generate per-(model, language, problem) router candidates from trained probes.

For every problem the script applies each candidate model's success probe to its
own pre-generation activations, attaches per-example output-token costs, and
writes router_candidates.csv — the input consumed by run_router_simulation.py
and the ETD experiments.

Usage
-----
python -m src.scripts.routing.generate_router_candidates \
    --probe-dir outputs/probes/pooled \
    --best-layers-path outputs/routing/probe_configs/pooled.csv \
    --write-router-candidates --candidates-only
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.scripts.probes.train_language_probes import (
    DEFAULT_SPLIT_FRACS,
    SHORT_NAME,
    build_coverage,
    difficulty_threshold_from_quantile,
    load_canonical_problem_ids,
    load_aligned_dataset,
    load_problem_difficulties,
    parse_csv_filter,
    parse_float_grid,
    probe_bundle_path,
    read_success_table,
    select_best_layers,
    write_json,
)
from src.scripts.routing.cost_model import (
    add_reference_tier_costs,
    attach_answer_costs,
    load_answer_token_costs,
)


DEFAULT_LAMBDA_GRID = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0]
DEFAULT_RESCUE_THRESHOLD_GRID = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
DEFAULT_SMALL_ROUTE_RATE_GRID = [0.05, 0.10, 0.15, 0.20, 0.30]
DEFAULT_CASCADE_THRESHOLD_GRID = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98]
DEFAULT_ROUTER_MODELS = [
    "Qwen3-4B",
    "gpt-oss-20b_low",
    "gpt-oss-20b_high",
]
DEFAULT_ROUTER_MODELS_CSV = ",".join(DEFAULT_ROUTER_MODELS)
DEFAULT_CASCADE_MODEL_ORDER = list(DEFAULT_ROUTER_MODELS)


def parse_lambda_grid(value: str) -> list[float]:
    values = parse_float_grid(value)
    if any(v < 0 for v in values):
        raise ValueError("Router lambda values must be non-negative.")
    return values


def parse_model_order(value: str | None) -> list[str]:
    if value is None:
        return list(DEFAULT_CASCADE_MODEL_ORDER)
    models = [item.strip() for item in value.split(",") if item.strip()]
    if not models:
        raise ValueError("--cascade-model-order must include at least one model.")
    return models


def load_probe_run_config(probe_dir: Path) -> dict[str, Any]:
    config_path = probe_dir / "run_config.json"
    if not config_path.exists():
        return {}
    with config_path.open(encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


def resolve_path(value: Path | None, config: dict[str, Any], key: str, fallback: str) -> Path:
    if value is not None:
        return value
    config_value = config.get(key)
    return Path(config_value) if config_value else Path(fallback)


def resolve_value(value: Any, config: dict[str, Any], key: str, fallback: Any) -> Any:
    return value if value is not None else config.get(key, fallback)


def load_best_layers(probe_dir: Path, best_layers_path: Path | None) -> pd.DataFrame:
    path = best_layers_path or probe_dir / "per_language_best_layers.csv"
    if path.exists():
        return pd.read_csv(path)

    layer_results_path = probe_dir / "per_language_layer_results.csv"
    if layer_results_path.exists():
        return select_best_layers(pd.read_csv(layer_results_path))

    rows = []
    for bundle_path in sorted((probe_dir / "probes").glob("*/*.joblib")):
        bundle = joblib.load(bundle_path)
        if not isinstance(bundle, dict) or "layers" not in bundle:
            continue
        for layer_payload in bundle["layers"].values():
            record = dict(layer_payload.get("record") or {})
            if record:
                record["probe_path"] = str(bundle_path)
                rows.append(record)
    if rows:
        return select_best_layers(pd.DataFrame(rows))

    raise FileNotFoundError(
        f"Could not find best-layer table at {path}, layer results at {layer_results_path}, "
        f"or reusable probe bundles under {probe_dir / 'probes'}"
    )


def build_router_candidates(
    ready_jobs: pd.DataFrame,
    success_df: pd.DataFrame,
    best_layers: pd.DataFrame,
    probe_dir: Path,
    args: argparse.Namespace,
    split_fracs: tuple[float, float, float],
    language_filter: set[str] | None,
    model_filter: set[str] | None,
    canonical_problem_ids: dict[str, set[str]],
) -> pd.DataFrame:
    """Use validation-selected best probes to produce per-candidate success predictions."""
    if best_layers.empty:
        return pd.DataFrame()

    best_by_pair = {
        (str(row.language), str(row.model)): row
        for row in best_layers.itertuples(index=False)
    }
    frames: list[pd.DataFrame] = []

    for _, row in ready_jobs.iterrows():
        key = (str(row["language"]), str(row["model"]))
        if key not in best_by_pair:
            continue

        selected = best_by_pair[key]
        bundle_path = Path(selected.probe_path)
        if not bundle_path.exists():
            bundle_path = probe_bundle_path(probe_dir, key[0], key[1])
        if not bundle_path.exists():
            print(f"    router skipped {key[0]} / {key[1]}: missing probe bundle")
            continue

        layer = int(selected.layer)
        dataset, status = load_aligned_dataset(
            row,
            success_df,
            split_fracs=split_fracs,
            random_seed=args.random_seed,
            min_examples=args.min_examples,
            min_positives_per_split=args.min_positives_per_split,
            auroc_threshold=args.auroc_threshold,
            canonical_problem_ids=canonical_problem_ids,
            alignment_min_corr=args.alignment_min_corr,
            alignment_max_mismatch_frac=args.alignment_max_mismatch_frac,
            cache_label_fallback_languages=args.cache_label_fallback_languages,
            problem_difficulties=args.problem_difficulties,
            max_difficulty=args.max_difficulty,
            activation_layers=[layer],
        )
        if dataset is None:
            print(f"    router skipped {key[0]} / {key[1]}: {status.get('status')}")
            continue
        activations = dataset["activations"].get(layer)
        if activations is None:
            activations = dataset["activations"].get(str(layer))
        if activations is None:
            print(f"    router skipped {key[0]} / {key[1]}: no layer {layer} in activations")
            continue

        bundle = joblib.load(bundle_path)
        layer_payload = bundle["layers"].get(layer)
        if layer_payload is None:
            layer_payload = bundle["layers"].get(str(layer))
        if layer_payload is None:
            print(f"    router skipped {key[0]} / {key[1]}: no layer {layer} in bundle")
            continue

        probe = layer_payload["probe"]
        X = activations[dataset["row_idx"]].astype(np.float32, copy=False)
        pred = np.clip(probe.predict(X), 0.0, 1.0)
        frames.append(
            pd.DataFrame(
                {
                    "language": dataset["language"],
                    "model": dataset["model"],
                    "model_label": SHORT_NAME.get(dataset["model"], dataset["model"]),
                    "problem_id": dataset["problem_ids"],
                    "base_problem_id": dataset["base_problem_ids"],
                    "difficulty": dataset["difficulty"],
                    "split": dataset["splits"],
                    "true_success": dataset["y"],
                    "pred_success_raw": pred.astype(np.float32),
                    "n_rollouts": dataset["n_rollouts"],
                    "n_correct": dataset["n_correct"],
                    "selected_layer": layer,
                    "selected_layer_frac": float(selected.layer_frac),
                    "selected_val_auroc": float(selected.val_auroc),
                    "selected_test_auroc": float(selected.test_auroc),
                    "probe_path": str(bundle_path),
                    "label_source": dataset["alignment_label_source"],
                    "alignment_overlap": dataset["alignment_overlap"],
                    "alignment_corr": dataset["alignment_corr"],
                    "alignment_mismatch_count": dataset["alignment_mismatch_count"],
                    "alignment_mismatch_frac": dataset["alignment_mismatch_frac"],
                }
            )
        )

    if not frames:
        return pd.DataFrame()

    candidates = pd.concat(frames, ignore_index=True)
    answer_costs = load_answer_token_costs(
        args.answers_root,
        language_filter=language_filter,
        model_filter=model_filter,
    )
    candidates = attach_answer_costs(candidates, answer_costs)
    candidates, _ = add_reference_tier_costs(candidates)
    candidates["cost_norm"] = candidates["tier_cost_norm"]
    return candidates


def filter_router_items(
    candidates: pd.DataFrame,
    min_candidates: int,
    require_all_candidates: bool,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    item_cols = ["split", "language", "base_problem_id"]
    counts = (
        candidates.groupby(item_cols, as_index=False)
        .agg(candidate_models=("model", "nunique"))
    )
    required = int(candidates["model"].nunique()) if require_all_candidates else min_candidates
    eligible = counts[counts["candidate_models"] >= required]
    filtered = candidates.merge(eligible, on=item_cols, how="inner")
    return filtered.reset_index(drop=True)


def route_by_utility(df: pd.DataFrame, score_col: str, lambda_value: float) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    work = df.copy()
    work["_router_utility"] = work[score_col] - lambda_value * work["cost_norm"]
    idx = work.groupby(["language", "base_problem_id"], sort=False)["_router_utility"].idxmax()
    chosen = work.loc[idx].copy()
    return chosen.drop(columns=["_router_utility"])


def route_by_utility_with_anchor(
    df: pd.DataFrame,
    score_col: str,
    lambda_value: float,
    anchor_model: str,
    margin: float,
) -> pd.DataFrame:
    """Prefer anchor model unless another model wins utility by a clear margin."""
    if df.empty:
        return df.copy()
    work = df.copy()
    work["_router_utility"] = work[score_col] - lambda_value * work["cost_norm"]

    item_cols = ["language", "base_problem_id"]
    best_idx = work.groupby(item_cols, sort=False)["_router_utility"].idxmax()
    best = work.loc[best_idx, [*item_cols, "_router_utility"]].rename(
        columns={"_router_utility": "_best_utility"}
    )
    best["_best_idx"] = best.index

    anchor_rows = work[work["model"] == anchor_model]
    if anchor_rows.empty:
        chosen = work.loc[best_idx].copy()
        return chosen.drop(columns=["_router_utility"])

    anchor_idx = anchor_rows.groupby(item_cols, sort=False)["_router_utility"].idxmax()
    anchor = work.loc[anchor_idx, [*item_cols, "_router_utility"]].rename(
        columns={"_router_utility": "_anchor_utility"}
    )
    anchor["_anchor_idx"] = anchor.index

    choices = best.merge(anchor, on=item_cols, how="left")
    use_anchor = (
        choices["_anchor_idx"].notna()
        & (choices["_anchor_utility"].astype(float) >= choices["_best_utility"].astype(float) - margin)
    )
    selected_idx = np.where(use_anchor, choices["_anchor_idx"], choices["_best_idx"]).astype(int)
    chosen = work.loc[selected_idx].copy()
    return chosen.drop(columns=["_router_utility"])


def route_by_anchor_advantage(
    df: pd.DataFrame,
    lambda_value: float,
    anchor_model: str,
    margin: float,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    item_cols = ["language", "base_problem_id"]
    work = df.copy()
    anchor_rows = work[work["model"] == anchor_model]
    if anchor_rows.empty:
        work["_router_utility"] = (
            work["pred_success_language_shrunk"].astype(float)
            - lambda_value * work["cost_norm"].astype(float)
        )
        idx = work.groupby(item_cols, sort=False)["_router_utility"].idxmax()
        return work.loc[idx].drop(columns=["_router_utility"]).copy()

    anchor_idx = anchor_rows.groupby(item_cols, sort=False)["cost_norm"].idxmin()
    anchor = work.loc[anchor_idx, [*item_cols, "cost_norm"]].rename(
        columns={"cost_norm": "_anchor_cost_norm"}
    )
    anchor["_anchor_idx"] = anchor.index

    scored = work.reset_index(names="_orig_idx").merge(anchor, on=item_cols, how="left")
    has_anchor = scored["_anchor_idx"].notna()
    scored["_anchor_route_score"] = np.where(
        has_anchor,
        scored["pred_anchor_advantage"].astype(float)
        + lambda_value * (scored["_anchor_cost_norm"].astype(float) - scored["cost_norm"].astype(float)),
        scored["pred_success_language_shrunk"].astype(float)
        - lambda_value * scored["cost_norm"].astype(float),
    )

    best_pos = scored.groupby(item_cols, sort=False)["_anchor_route_score"].idxmax()
    best = scored.loc[best_pos, [*item_cols, "_anchor_route_score", "_anchor_idx"]]
    best["_best_idx"] = scored.loc[best_pos, "_orig_idx"].to_numpy()
    use_best = best["_anchor_idx"].isna() | (best["_anchor_route_score"].astype(float) >= -margin)
    selected_idx = np.where(use_best, best["_best_idx"], best["_anchor_idx"]).astype(int)
    return work.loc[selected_idx].copy()


def route_by_non_anchor_threshold(
    df: pd.DataFrame,
    score_col: str,
    threshold: float,
    anchor_model: str,
) -> pd.DataFrame:
    """Route to the best non-anchor candidate only when its score clears a threshold."""
    if df.empty:
        return df.copy()

    item_cols = ["language", "base_problem_id"]
    work = df.copy()
    fallback = work.reset_index(names="_fallback_idx").copy()
    fallback["_fallback_rank"] = np.where(fallback["model"].astype(str).eq(anchor_model), 0, 1)
    fallback = (
        fallback.sort_values([*item_cols, "_fallback_rank", "cost_norm"], kind="mergesort")
        .drop_duplicates(item_cols, keep="first")
        [[*item_cols, "_fallback_idx"]]
    )

    non_anchor = work[work["model"] != anchor_model].copy()
    if non_anchor.empty:
        return work.loc[fallback["_fallback_idx"].to_numpy(dtype=int)].copy()

    # Prefer high rescue score, then cheaper candidates for equal scores.
    non_anchor["_threshold_route_score"] = (
        non_anchor[score_col].astype(float) + 1e-6 * (1.0 - non_anchor["cost_norm"].astype(float))
    )
    best_idx = non_anchor.groupby(item_cols, sort=False)["_threshold_route_score"].idxmax()
    best = non_anchor.loc[best_idx, [*item_cols, score_col]].rename(columns={score_col: "_best_score"})
    best["_best_idx"] = best.index

    choices = fallback.merge(best, on=item_cols, how="left")
    use_best = choices["_best_idx"].notna() & (choices["_best_score"].astype(float) >= threshold)
    selected_idx = np.where(use_best, choices["_best_idx"], choices["_fallback_idx"]).astype(int)
    return work.loc[selected_idx].copy()


def route_by_probe_cascade(
    df: pd.DataFrame,
    score_col: str,
    threshold: float,
    model_order: list[str],
    fallback_model: str,
) -> pd.DataFrame:
    """Probe models in order and stop at the first model above threshold.

    This simulates an internal-state-only cascade: each step uses the saved
    end-of-prompt probe score for that model, without considering generated
    outputs. If no probed model clears the threshold, the cascade falls back to
    the strongest/default model.
    """
    if df.empty:
        return df.copy()
    item_cols = ["language", "base_problem_id"]
    order_map = {model: rank for rank, model in enumerate(model_order)}
    work = df.copy()
    work["_cascade_order"] = work["model"].map(order_map)
    ordered = work[work["_cascade_order"].notna()].copy()

    if ordered.empty:
        return df.loc[df.groupby(item_cols, sort=False)["cost_norm"].idxmin()].copy()

    ordered["_cascade_order"] = ordered["_cascade_order"].astype(int)
    ordered = ordered.sort_values([*item_cols, "_cascade_order"], kind="mergesort")
    eligible = ordered[ordered[score_col].astype(float) >= threshold]
    chosen = eligible.groupby(item_cols, sort=False).head(1).copy()

    chosen_keys = chosen[item_cols].drop_duplicates()
    fallback_pool = work.merge(chosen_keys, on=item_cols, how="left", indicator=True)
    fallback_pool = fallback_pool[fallback_pool["_merge"].eq("left_only")].drop(columns=["_merge"])
    fallback_rows = fallback_pool[fallback_pool["model"] == fallback_model].copy()
    fallback_idx = fallback_rows.groupby(item_cols, sort=False)["cost_norm"].idxmin()
    missing_fallback_keys = (
        fallback_pool[item_cols].drop_duplicates()
        .merge(fallback_rows[item_cols].drop_duplicates(), on=item_cols, how="left", indicator=True)
    )
    missing_fallback_keys = missing_fallback_keys[missing_fallback_keys["_merge"].eq("left_only")][item_cols]
    if not missing_fallback_keys.empty:
        cheapest_pool = fallback_pool.merge(missing_fallback_keys, on=item_cols, how="inner")
        fallback_idx = pd.Index([*fallback_idx, *cheapest_pool.groupby(item_cols, sort=False)["cost_norm"].idxmin()])
    fallback = fallback_pool.loc[fallback_idx].copy() if len(fallback_idx) else work.iloc[0:0].copy()

    order_counts = ordered.groupby(item_cols, sort=False)["_cascade_order"].size().rename("_ordered_count")
    chosen["_cascade_models_checked"] = chosen["_cascade_order"] + 1
    fallback = fallback.merge(order_counts.reset_index(), on=item_cols, how="left")
    fallback["_cascade_models_checked"] = fallback["_ordered_count"].fillna(0).astype(float)
    fallback = fallback.drop(columns=["_ordered_count"], errors="ignore")

    result = pd.concat([chosen, fallback], ignore_index=True)
    result["_cascade_threshold"] = threshold
    return result.drop(columns=["_cascade_order"], errors="ignore")


def choose_oracle(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    work = df.copy()
    work["_oracle_utility"] = work["true_success"] - 1e-6 * work["cost_norm"]
    idx = work.groupby(["language", "base_problem_id"], sort=False)["_oracle_utility"].idxmax()
    return work.loc[idx].drop(columns=["_oracle_utility"]).copy()


def summarize_choices(
    chosen: pd.DataFrame,
    policy: str,
    split: str,
    lambda_value: float | None = None,
    anchor_margin: float | None = None,
    rescue_threshold: float | None = None,
    target_small_route_rate: float | None = None,
) -> dict[str, Any]:
    if chosen.empty:
        return {
            "policy": policy,
            "split": split,
            "lambda": lambda_value,
            "anchor_margin": anchor_margin,
            "rescue_threshold": rescue_threshold,
            "target_small_route_rate": target_small_route_rate,
            "n_items": 0,
            "mean_success": float("nan"),
            "cost": float("nan"),
            "total_output_cost_usd": float("nan"),
            "mean_output_cost_usd": float("nan"),
            "total_output_tokens": float("nan"),
            "mean_output_tokens": float("nan"),
            "mean_cost_norm": float("nan"),
        }
    counts = chosen["model"].value_counts(normalize=True).to_dict()
    total_cost = float(chosen["total_output_cost_usd"].sum())
    row = {
        "policy": policy,
        "split": split,
        "lambda": lambda_value,
        "anchor_margin": anchor_margin,
        "rescue_threshold": rescue_threshold,
        "target_small_route_rate": target_small_route_rate,
        "n_items": int(len(chosen)),
        "mean_success": float(chosen["true_success"].mean()),
        "cost": total_cost,
        "total_output_cost_usd": total_cost,
        "mean_output_cost_usd": float(chosen["total_output_cost_usd"].mean()),
        "total_output_tokens": float(chosen["total_output_tokens"].sum()),
        "mean_output_tokens": float(chosen["total_output_tokens"].mean()),
        "mean_cost_norm": float(chosen["cost_norm"].mean()),
        "selected_model_mix": json.dumps(counts, sort_keys=True),
    }
    if "_cascade_models_checked" in chosen.columns:
        row["mean_cascade_models_checked"] = float(chosen["_cascade_models_checked"].mean())
    return row


def make_calibration_features(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    features = pd.DataFrame(
        {
            "pred_success_raw": df["pred_success_raw"].astype(float),
            "cost_norm": df["cost_norm"].astype(float),
        },
        index=df.index,
    )
    features["pred_x_cost"] = features["pred_success_raw"] * features["cost_norm"]
    dummies = pd.get_dummies(df[["language", "model"]], prefix=["lang", "model"], dtype=float)
    features = pd.concat([features, dummies], axis=1)
    if columns is None:
        return features, list(features.columns)
    return features.reindex(columns=columns, fill_value=0.0), columns


def add_calibrated_predictions(candidates: pd.DataFrame, alpha: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    candidates = candidates.copy()
    candidates["pred_success_calibrated"] = candidates["pred_success_raw"]
    val = candidates[candidates["split"] == "val"]
    if val.empty or val["true_success"].nunique() <= 1:
        return candidates, {"status": "skipped", "reason": "insufficient_validation_variation"}

    X_val, columns = make_calibration_features(val)
    y_val = val["true_success"].to_numpy(dtype=np.float32)
    calibrator = Ridge(alpha=alpha)
    calibrator.fit(X_val, y_val)
    X_all, _ = make_calibration_features(candidates, columns=columns)
    candidates["pred_success_calibrated"] = np.clip(calibrator.predict(X_all), 0.0, 1.0)
    return candidates, {
        "status": "trained",
        "alpha": alpha,
        "n_validation_rows": int(len(val)),
        "n_features": int(len(columns)),
    }


def add_language_shrunk_predictions(
    candidates: pd.DataFrame,
    shrinkage_strength: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    candidates = candidates.copy()
    candidates["pred_success_language_shrunk"] = candidates["pred_success_calibrated"]
    val = candidates[candidates["split"] == "val"].copy()
    if val.empty:
        return candidates, {"status": "skipped", "reason": "missing_validation_rows"}

    strength = max(float(shrinkage_strength), 0.0)
    residual = val["true_success"].astype(float) - val["pred_success_calibrated"].astype(float)
    lang_stats = (
        pd.DataFrame({"language": val["language"], "residual": residual})
        .groupby("language", as_index=False)
        .agg(lang_residual_mean=("residual", "mean"), lang_n=("residual", "size"))
    )
    lang_stats["lang_weight"] = lang_stats["lang_n"] / (lang_stats["lang_n"] + strength)
    lang_stats["lang_correction"] = lang_stats["lang_weight"] * lang_stats["lang_residual_mean"]

    val_with_lang = val.merge(lang_stats[["language", "lang_correction"]], on="language", how="left")
    residual_after_lang = (
        val_with_lang["true_success"].astype(float)
        - val_with_lang["pred_success_calibrated"].astype(float)
        - val_with_lang["lang_correction"].fillna(0.0).astype(float)
    )
    lm_stats = (
        pd.DataFrame(
            {
                "language": val_with_lang["language"],
                "model": val_with_lang["model"],
                "residual": residual_after_lang,
            }
        )
        .groupby(["language", "model"], as_index=False)
        .agg(lm_residual_mean=("residual", "mean"), lm_n=("residual", "size"))
    )
    lm_stats["lm_weight"] = lm_stats["lm_n"] / (lm_stats["lm_n"] + strength)
    lm_stats["lm_correction"] = lm_stats["lm_weight"] * lm_stats["lm_residual_mean"]

    calibrated = candidates["pred_success_calibrated"].astype(float)
    adjusted = (
        candidates.merge(lang_stats[["language", "lang_correction"]], on="language", how="left")
        .merge(lm_stats[["language", "model", "lm_correction"]], on=["language", "model"], how="left")
    )
    candidates["pred_success_language_shrunk"] = np.clip(
        calibrated
        + adjusted["lang_correction"].fillna(0.0).to_numpy(dtype=float)
        + adjusted["lm_correction"].fillna(0.0).to_numpy(dtype=float),
        0.0,
        1.0,
    )
    return candidates, {
        "status": "trained",
        "shrinkage_strength": strength,
        "n_validation_rows": int(len(val)),
        "n_language_corrections": int(len(lang_stats)),
        "n_language_model_corrections": int(len(lm_stats)),
    }


def make_relative_value_features(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    item_cols = ["split", "language", "base_problem_id"]
    raw = df["pred_success_raw"].astype(float)
    calibrated = df["pred_success_calibrated"].astype(float)
    shrunk = df["pred_success_language_shrunk"].astype(float)
    cost = df["cost_norm"].astype(float)

    features = pd.DataFrame(
        {
            "pred_success_raw": raw,
            "pred_success_calibrated": calibrated,
            "pred_success_language_shrunk": shrunk,
            "cost_norm": cost,
            "raw_minus_item_mean": raw - raw.groupby([df[c] for c in item_cols]).transform("mean"),
            "calibrated_minus_item_mean": calibrated
            - calibrated.groupby([df[c] for c in item_cols]).transform("mean"),
            "shrunk_minus_item_mean": shrunk - shrunk.groupby([df[c] for c in item_cols]).transform("mean"),
            "cost_minus_item_min": cost - cost.groupby([df[c] for c in item_cols]).transform("min"),
        },
        index=df.index,
    )
    features["raw_x_cost"] = features["pred_success_raw"] * features["cost_norm"]
    features["calibrated_x_cost"] = features["pred_success_calibrated"] * features["cost_norm"]
    features["shrunk_x_cost"] = features["pred_success_language_shrunk"] * features["cost_norm"]
    dummies = pd.get_dummies(df[["language", "model"]], prefix=["lang", "model"], dtype=float)
    features = pd.concat([features, dummies], axis=1)
    if columns is None:
        return features, list(features.columns)
    return features.reindex(columns=columns, fill_value=0.0), columns


def add_relative_value_predictions(candidates: pd.DataFrame, alpha: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    candidates = candidates.copy()
    candidates["pred_success_relative_value"] = candidates["pred_success_calibrated"]
    val = candidates[candidates["split"] == "val"].copy()
    if val.empty or val["true_success"].nunique() <= 1:
        return candidates, {"status": "skipped", "reason": "insufficient_validation_variation"}

    item_cols = ["split", "language", "base_problem_id"]
    best_success = val.groupby(item_cols)["true_success"].transform("max")
    y_val = (val["true_success"] - best_success).to_numpy(dtype=np.float32)
    X_val, columns = make_relative_value_features(val)
    calibrator = Ridge(alpha=alpha)
    calibrator.fit(X_val, y_val)
    X_all, _ = make_relative_value_features(candidates, columns=columns)
    candidates["pred_success_relative_value"] = calibrator.predict(X_all).astype(np.float32)
    return candidates, {
        "status": "trained",
        "alpha": alpha,
        "n_validation_rows": int(len(val)),
        "n_features": int(len(columns)),
        "target": "candidate_true_success_minus_item_best_true_success",
    }


def make_anchor_advantage_features(
    df: pd.DataFrame,
    anchor_model: str,
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    item_cols = ["split", "language", "base_problem_id"]
    anchor_cols = [
        *item_cols,
        "pred_success_raw",
        "pred_success_calibrated",
        "pred_success_language_shrunk",
        "pred_success_relative_value",
        "cost_norm",
        "true_success",
    ]
    anchor = df[df["model"] == anchor_model][anchor_cols].rename(
        columns={
            "pred_success_raw": "anchor_pred_success_raw",
            "pred_success_calibrated": "anchor_pred_success_calibrated",
            "pred_success_language_shrunk": "anchor_pred_success_language_shrunk",
            "pred_success_relative_value": "anchor_pred_success_relative_value",
            "cost_norm": "anchor_cost_norm",
            "true_success": "anchor_true_success",
        }
    )
    work = df.merge(anchor, on=item_cols, how="left")
    has_anchor = work["anchor_cost_norm"].notna()

    for col in [
        "anchor_pred_success_raw",
        "anchor_pred_success_calibrated",
        "anchor_pred_success_language_shrunk",
        "anchor_pred_success_relative_value",
    ]:
        source = col.removeprefix("anchor_")
        work[col] = work[col].fillna(work[source])
    work["anchor_cost_norm"] = work["anchor_cost_norm"].fillna(work["cost_norm"])

    features = pd.DataFrame(index=work.index)
    for col in [
        "pred_success_raw",
        "pred_success_calibrated",
        "pred_success_language_shrunk",
        "pred_success_relative_value",
    ]:
        features[col] = work[col].astype(float)
        features[f"{col}_minus_anchor"] = work[col].astype(float) - work[f"anchor_{col}"].astype(float)
    features["cost_norm"] = work["cost_norm"].astype(float)
    features["anchor_cost_norm"] = work["anchor_cost_norm"].astype(float)
    features["cost_saving_vs_anchor"] = features["anchor_cost_norm"] - features["cost_norm"]
    features["is_anchor_model"] = (work["model"] == anchor_model).astype(float)
    dummies = pd.get_dummies(work[["language", "model"]], prefix=["lang", "model"], dtype=float)
    features = pd.concat([features, dummies], axis=1)
    if columns is None:
        return features, has_anchor, list(features.columns)
    return features.reindex(columns=columns, fill_value=0.0), has_anchor, columns


def add_anchor_advantage_predictions(
    candidates: pd.DataFrame,
    alpha: float,
    anchor_model: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    candidates = candidates.copy()
    candidates["pred_anchor_advantage"] = 0.0
    val = candidates[candidates["split"] == "val"].copy()
    if val.empty or anchor_model not in set(val["model"].astype(str)):
        return candidates, {"status": "skipped", "reason": "missing_validation_anchor"}

    X_val, has_anchor_val, columns = make_anchor_advantage_features(val, anchor_model)
    val_with_anchor = val.reset_index(drop=True)
    anchor_true = (
        val_with_anchor[["split", "language", "base_problem_id", "model", "true_success"]]
        .query("model == @anchor_model")
        .drop(columns=["model"])
        .rename(columns={"true_success": "anchor_true_success"})
    )
    val_with_anchor = val_with_anchor.merge(
        anchor_true,
        on=["split", "language", "base_problem_id"],
        how="left",
    )
    train_mask = has_anchor_val.to_numpy() & val_with_anchor["anchor_true_success"].notna().to_numpy()
    if train_mask.sum() == 0:
        return candidates, {"status": "skipped", "reason": "no_anchor_matched_validation_rows"}

    y_val = (
        val_with_anchor.loc[train_mask, "true_success"].astype(float)
        - val_with_anchor.loc[train_mask, "anchor_true_success"].astype(float)
    ).to_numpy(dtype=np.float32)
    calibrator = Ridge(alpha=alpha)
    calibrator.fit(X_val.loc[train_mask], y_val)
    X_all, has_anchor_all, _ = make_anchor_advantage_features(candidates, anchor_model, columns=columns)
    pred = calibrator.predict(X_all).astype(np.float32)
    pred[~has_anchor_all.to_numpy()] = 0.0
    candidates["pred_anchor_advantage"] = pred
    return candidates, {
        "status": "trained",
        "alpha": alpha,
        "anchor_model": anchor_model,
        "n_validation_rows": int(train_mask.sum()),
        "n_features": int(len(columns)),
        "target": "candidate_true_success_minus_anchor_true_success",
    }


def add_rescue_gain_predictions(
    candidates: pd.DataFrame,
    alpha: float,
    anchor_model: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Predict positive-only gain over the anchor model.

    This is intentionally different from generic success prediction: the target is
    max(candidate_success - anchor_success, 0), so the router learns the rescue
    seam where a non-anchor model can recover items the anchor misses.
    """
    candidates = candidates.copy()
    candidates["pred_rescue_gain"] = 0.0
    val = candidates[candidates["split"] == "val"].copy()
    if val.empty or anchor_model not in set(val["model"].astype(str)):
        return candidates, {"status": "skipped", "reason": "missing_validation_anchor"}

    X_val, has_anchor_val, columns = make_anchor_advantage_features(val, anchor_model)
    val_with_anchor = val.reset_index(drop=True)
    anchor_true = (
        val_with_anchor[["split", "language", "base_problem_id", "model", "true_success"]]
        .query("model == @anchor_model")
        .drop(columns=["model"])
        .rename(columns={"true_success": "anchor_true_success"})
    )
    val_with_anchor = val_with_anchor.merge(
        anchor_true,
        on=["split", "language", "base_problem_id"],
        how="left",
    )
    train_mask = has_anchor_val.to_numpy() & val_with_anchor["anchor_true_success"].notna().to_numpy()
    if train_mask.sum() == 0:
        return candidates, {"status": "skipped", "reason": "no_anchor_matched_validation_rows"}

    y_val = np.maximum(
        val_with_anchor.loc[train_mask, "true_success"].astype(float)
        - val_with_anchor.loc[train_mask, "anchor_true_success"].astype(float),
        0.0,
    ).to_numpy(dtype=np.float32)
    calibrator = Ridge(alpha=alpha)
    calibrator.fit(X_val.loc[train_mask], y_val)

    X_all, has_anchor_all, _ = make_anchor_advantage_features(candidates, anchor_model, columns=columns)
    pred = np.clip(calibrator.predict(X_all).astype(np.float32), 0.0, 1.0)
    pred[~has_anchor_all.to_numpy()] = 0.0
    pred[candidates["model"].astype(str).eq(anchor_model).to_numpy()] = 0.0
    candidates["pred_rescue_gain"] = pred
    return candidates, {
        "status": "trained",
        "alpha": alpha,
        "anchor_model": anchor_model,
        "n_validation_rows": int(train_mask.sum()),
        "n_features": int(len(columns)),
        "target": "max(candidate_true_success_minus_anchor_true_success, 0)",
    }


def evaluate_fixed_model_policies(candidates: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ["val", "test"]:
        split_df = candidates[candidates["split"] == split]
        for model, model_df in split_df.groupby("model"):
            rows.append(summarize_choices(model_df, f"always:{model}", split))
    return rows


def evaluate_validation_selected_policies(candidates: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    val = candidates[candidates["split"] == "val"]
    test = candidates[candidates["split"] == "test"]
    if val.empty or test.empty:
        return rows

    global_scores = val.groupby("model")["true_success"].mean()
    if not global_scores.empty:
        best_model = str(global_scores.idxmax())
        for split, split_df in [("val", val), ("test", test)]:
            rows.append(
                summarize_choices(
                    split_df[split_df["model"] == best_model],
                    f"best_single_model_by_val:{best_model}",
                    split,
                )
            )

    per_language = (
        val.groupby(["language", "model"], as_index=False)["true_success"]
        .mean()
        .sort_values(["language", "true_success"], ascending=[True, False])
        .drop_duplicates("language")
        [["language", "model"]]
    )
    for split, split_df in [("val", val), ("test", test)]:
        chosen = split_df.merge(per_language, on=["language", "model"], how="inner")
        rows.append(summarize_choices(chosen, "per_language_best_model_by_val", split))
    return rows


def append_validation_selected_router_rows(result: pd.DataFrame) -> pd.DataFrame:
    if result.empty:
        return result

    val = result[
        (result["split"] == "val")
        & result["policy"].astype(str).str.startswith(("probe_router", "probe_cascade"))
    ].copy()
    if val.empty:
        return result

    selections: list[tuple[str, pd.Series]] = []
    selections.append(
        (
            "val_selected_best_router_by_success",
            val.sort_values(["mean_success", "mean_output_cost_usd"], ascending=[False, True]).iloc[0],
        )
    )

    high_val = result[
        (result["split"] == "val")
        & result["policy"].astype(str).str.startswith("always:gpt-oss-20b_high", na=False)
    ]
    if not high_val.empty:
        high_success = float(high_val.iloc[0]["mean_success"])
        within_margin = val[(high_success - val["mean_success"]) <= 0.005].copy()
        if not within_margin.empty:
            selections.append(
                (
                    "val_selected_cheapest_router_within_0p5pp_high",
                    within_margin.sort_values(["mean_output_cost_usd", "mean_success"], ascending=[True, False]).iloc[0],
                )
            )

    extra_rows: list[pd.Series] = []
    for selected_policy, selected in selections:
        mask = result["policy"].eq(selected["policy"])
        for col in ["lambda", "anchor_margin", "rescue_threshold", "target_small_route_rate"]:
            if col not in result.columns:
                continue
            selected_value = selected.get(col)
            if pd.isna(selected_value):
                mask &= result[col].isna()
            else:
                mask &= result[col].eq(selected_value)
        for _, source_row in result[mask].iterrows():
            row = source_row.copy()
            row["policy"] = selected_policy
            row["selection_source_policy"] = selected["policy"]
            row["selection_source_lambda"] = selected.get("lambda")
            row["selection_source_anchor_margin"] = selected.get("anchor_margin")
            row["selection_source_rescue_threshold"] = selected.get("rescue_threshold")
            row["selection_source_target_small_route_rate"] = selected.get("target_small_route_rate")
            extra_rows.append(row)

    if not extra_rows:
        return result
    return pd.concat([result, pd.DataFrame(extra_rows)], ignore_index=True)


def non_anchor_score_threshold_for_rate(
    val_df: pd.DataFrame,
    score_col: str,
    anchor_model: str,
    target_rate: float,
) -> float | None:
    if val_df.empty:
        return None
    non_anchor = val_df[val_df["model"] != anchor_model].copy()
    if non_anchor.empty:
        return None
    idx = non_anchor.groupby(["language", "base_problem_id"], sort=False)[score_col].idxmax()
    best_scores = non_anchor.loc[idx, score_col].astype(float)
    if best_scores.empty:
        return None
    clipped_rate = min(max(float(target_rate), 0.0), 1.0)
    if clipped_rate <= 0.0:
        return float("inf")
    return float(best_scores.quantile(1.0 - clipped_rate))


def cascade_threshold_for_route_rate(
    val_df: pd.DataFrame,
    score_col: str,
    model_order: list[str],
    fallback_model: str,
    target_rate: float,
) -> float | None:
    if val_df.empty:
        return None
    item_cols = ["language", "base_problem_id"]
    eligible = val_df[
        val_df["model"].isin(model_order)
        & ~val_df["model"].astype(str).eq(fallback_model)
    ]
    if eligible.empty:
        return None
    non_fallback_scores = eligible.groupby(item_cols, sort=False)[score_col].max().astype(float)
    clipped_rate = min(max(float(target_rate), 0.0), 1.0)
    if clipped_rate <= 0.0:
        return float("inf")
    return float(non_fallback_scores.quantile(1.0 - clipped_rate))


def evaluate_probe_cascade_policies(
    candidates: pd.DataFrame,
    cascade_threshold_grid: list[float],
    small_route_rate_grid: list[float],
    model_order: list[str],
    fallback_model: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    score_specs = [
        ("probe_cascade_raw", "pred_success_raw"),
        ("probe_cascade_calibrated", "pred_success_calibrated"),
        ("probe_cascade_language_shrunk", "pred_success_language_shrunk"),
    ]
    available_models = set(candidates["model"].astype(str).unique())
    effective_order = [model for model in model_order if model in available_models]
    if fallback_model not in effective_order and fallback_model in available_models:
        effective_order.append(fallback_model)
    if not effective_order:
        return rows

    for split in ["val", "test"]:
        split_df = candidates[candidates["split"] == split]
        if split_df.empty:
            continue
        for policy, score_col in score_specs:
            for threshold in cascade_threshold_grid:
                rows.append(
                    summarize_choices(
                        route_by_probe_cascade(
                            split_df,
                            score_col=score_col,
                            threshold=threshold,
                            model_order=effective_order,
                            fallback_model=fallback_model,
                        ),
                        policy,
                        split,
                        rescue_threshold=threshold,
                    )
                )

    val = candidates[candidates["split"] == "val"]
    for policy, score_col in score_specs:
        for target_rate in small_route_rate_grid:
            threshold = cascade_threshold_for_route_rate(
                val,
                score_col=score_col,
                model_order=effective_order,
                fallback_model=fallback_model,
                target_rate=target_rate,
            )
            if threshold is None:
                continue
            for split in ["val", "test"]:
                split_df = candidates[candidates["split"] == split]
                if split_df.empty:
                    continue
                rows.append(
                    summarize_choices(
                        route_by_probe_cascade(
                            split_df,
                            score_col=score_col,
                            threshold=threshold,
                            model_order=effective_order,
                            fallback_model=fallback_model,
                        ),
                        f"{policy}_target_rate",
                        split,
                        rescue_threshold=threshold,
                        target_small_route_rate=target_rate,
                    )
                )
    return rows


def evaluate_non_anchor_threshold_policies(
    candidates: pd.DataFrame,
    anchor_model: str,
    rescue_threshold_grid: list[float],
    small_route_rate_grid: list[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    threshold_specs = [
        ("probe_router_rescue_gain_threshold", "pred_rescue_gain"),
        ("probe_router_small_success_threshold", "pred_success_calibrated"),
    ]

    for split in ["val", "test"]:
        split_df = candidates[candidates["split"] == split]
        if split_df.empty:
            continue
        for policy, score_col in threshold_specs:
            for threshold in rescue_threshold_grid:
                rows.append(
                    summarize_choices(
                        route_by_non_anchor_threshold(
                            split_df,
                            score_col=score_col,
                            threshold=threshold,
                            anchor_model=anchor_model,
                        ),
                        policy,
                        split,
                        rescue_threshold=threshold,
                    )
                )

    val = candidates[candidates["split"] == "val"]
    for policy, score_col in threshold_specs:
        for target_rate in small_route_rate_grid:
            threshold = non_anchor_score_threshold_for_rate(
                val,
                score_col=score_col,
                anchor_model=anchor_model,
                target_rate=target_rate,
            )
            if threshold is None:
                continue
            for split in ["val", "test"]:
                split_df = candidates[candidates["split"] == split]
                if split_df.empty:
                    continue
                rows.append(
                    summarize_choices(
                        route_by_non_anchor_threshold(
                            split_df,
                            score_col=score_col,
                            threshold=threshold,
                            anchor_model=anchor_model,
                        ),
                        f"{policy}_target_rate",
                        split,
                        rescue_threshold=threshold,
                        target_small_route_rate=target_rate,
                    )
                )
    return rows


def evaluate_router_policies(
    candidates: pd.DataFrame,
    lambda_grid: list[float],
    anchor_model: str,
    anchor_margin_grid: list[float],
    rescue_threshold_grid: list[float],
    small_route_rate_grid: list[float],
    cascade_threshold_grid: list[float],
    cascade_model_order: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.extend(evaluate_fixed_model_policies(candidates))
    rows.extend(evaluate_validation_selected_policies(candidates))
    rows.extend(
        evaluate_probe_cascade_policies(
            candidates,
            cascade_threshold_grid=cascade_threshold_grid,
            small_route_rate_grid=small_route_rate_grid,
            model_order=cascade_model_order,
            fallback_model=anchor_model,
        )
    )
    rows.extend(
        evaluate_non_anchor_threshold_policies(
            candidates,
            anchor_model=anchor_model,
            rescue_threshold_grid=rescue_threshold_grid,
            small_route_rate_grid=small_route_rate_grid,
        )
    )

    for split in ["val", "test"]:
        split_df = candidates[candidates["split"] == split]
        if split_df.empty:
            continue
        rows.append(summarize_choices(choose_oracle(split_df), "oracle_best_candidate", split))
        for lambda_value in lambda_grid:
            rows.append(
                summarize_choices(
                    route_by_utility(split_df, "pred_success_raw", lambda_value),
                    "probe_router_raw",
                    split,
                    lambda_value=lambda_value,
                )
            )
            rows.append(
                summarize_choices(
                    route_by_utility(split_df, "pred_success_calibrated", lambda_value),
                    "probe_router_language_calibrated",
                    split,
                    lambda_value=lambda_value,
                )
            )
            rows.append(
                summarize_choices(
                    route_by_utility(split_df, "pred_success_language_shrunk", lambda_value),
                    "probe_router_language_shrunk",
                    split,
                    lambda_value=lambda_value,
                )
            )
            rows.append(
                summarize_choices(
                    route_by_utility(split_df, "pred_success_relative_value", lambda_value),
                    "probe_router_relative_value",
                    split,
                    lambda_value=lambda_value,
                )
            )
            for margin in anchor_margin_grid:
                rows.append(
                    summarize_choices(
                        route_by_anchor_advantage(
                            split_df,
                            lambda_value=lambda_value,
                            anchor_model=anchor_model,
                            margin=margin,
                        ),
                        "probe_router_anchor_advantage",
                        split,
                        lambda_value=lambda_value,
                        anchor_margin=margin,
                    )
                )
                rows.append(
                    summarize_choices(
                        route_by_utility_with_anchor(
                            split_df,
                            "pred_success_raw",
                            lambda_value=lambda_value,
                            anchor_model=anchor_model,
                            margin=margin,
                        ),
                        "probe_router_raw_anchor",
                        split,
                        lambda_value=lambda_value,
                        anchor_margin=margin,
                    )
                )
                rows.append(
                    summarize_choices(
                        route_by_utility_with_anchor(
                            split_df,
                            "pred_success_calibrated",
                            lambda_value=lambda_value,
                            anchor_model=anchor_model,
                            margin=margin,
                        ),
                        "probe_router_language_calibrated_anchor",
                        split,
                        lambda_value=lambda_value,
                        anchor_margin=margin,
                    )
                )
                rows.append(
                    summarize_choices(
                        route_by_utility_with_anchor(
                            split_df,
                            "pred_success_relative_value",
                            lambda_value=lambda_value,
                            anchor_model=anchor_model,
                            margin=margin,
                        ),
                        "probe_router_relative_value_anchor",
                        split,
                        lambda_value=lambda_value,
                        anchor_margin=margin,
                    )
                )

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = append_validation_selected_router_rows(result)

    test_high = result[
        (result["split"] == "test")
        & result["policy"].str.startswith("always:gpt-oss-20b_high", na=False)
    ]
    if not test_high.empty:
        high_success = float(test_high.iloc[0]["mean_success"])
        high_cost = float(test_high.iloc[0]["mean_output_cost_usd"])
        result["success_delta_vs_gpt_high"] = result["mean_success"] - high_success
        result["cost_ratio_vs_gpt_high"] = result["mean_output_cost_usd"] / high_cost if high_cost > 0 else np.nan
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create multilingual routing tables from saved end-of-prompt probes."
    )
    parser.add_argument("--probe-dir", type=Path, default=Path("outputs/probes/language_specific"))
    parser.add_argument("--best-layers-path", type=Path, default=None)
    parser.add_argument("--activations-root", type=Path, default=None)
    parser.add_argument("--answers-root", type=Path, default=None)
    parser.add_argument("--success-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/routing/candidates"))
    parser.add_argument("--token-position", type=int, default=None)
    parser.add_argument("--languages", type=str, default=None, help="Comma-separated language filter.")
    parser.add_argument(
        "--models",
        type=str,
        default=DEFAULT_ROUTER_MODELS_CSV,
        help=(
            "Comma-separated model filter. Defaults to the three useful routing candidates "
            "(Qwen 7B, gpt-oss low, gpt-oss high), excluding Qwen 1.5B."
        ),
    )
    parser.add_argument("--split-fracs", type=float, nargs=3, default=None)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--auroc-threshold", type=float, default=None)
    parser.add_argument("--min-examples", type=int, default=None)
    parser.add_argument("--min-positives-per-split", type=int, default=None)
    parser.add_argument("--canonical-csv-path", type=Path, default=Path("data/MATH_translated.csv"))
    parser.add_argument(
        "--difficulty-max-quantile",
        type=float,
        default=None,
        help=(
            "If set, route only problems with difficulty at or below this global quantile. "
            "Defaults to the value saved by the probe run, when present."
        ),
    )
    parser.add_argument("--alignment-min-corr", type=float, default=0.20)
    parser.add_argument("--alignment-max-mismatch-frac", type=float, default=0.30)
    parser.add_argument("--cache-label-fallback-languages", type=str, default="English")
    parser.add_argument(
        "--lambda-grid",
        type=str,
        default=",".join(str(x) for x in DEFAULT_LAMBDA_GRID),
        help=(
            "Comma-separated non-negative cost penalties for routing. "
            "Utility is predicted_success - lambda * normalized_cost."
        ),
    )
    parser.add_argument(
        "--calibration-alpha",
        type=float,
        default=1.0,
        help="Ridge alpha for the language/model calibration layer trained on validation candidates.",
    )
    parser.add_argument(
        "--language-shrinkage-strength",
        type=float,
        default=500.0,
        help="Validation-row shrinkage strength for language and language/model residual calibration.",
    )
    parser.add_argument(
        "--min-router-candidates",
        type=int,
        default=len(DEFAULT_ROUTER_MODELS),
        help="Minimum number of candidate models required for an item to enter router evaluation.",
    )
    parser.add_argument(
        "--require-all-candidates",
        action="store_true",
        help="Filter to items where every available model has a candidate row.",
    )
    parser.add_argument(
        "--write-router-candidates",
        action="store_true",
        help="Write the full per-problem router candidate table. This can be large.",
    )
    parser.add_argument(
        "--candidates-only",
        action="store_true",
        help="Build (and optionally write) router candidates only; skip policy evaluation.",
    )
    parser.add_argument(
        "--anchor-model",
        type=str,
        default="gpt-oss-20b_high",
        help="Anchor model used by anchor policies that require clear utility margin to deviate.",
    )
    parser.add_argument(
        "--anchor-margin-grid",
        type=str,
        default="0.0,0.01,0.02,0.05",
        help=(
            "Comma-separated utility margins for anchored policies. "
            "Larger values make the router stick with the anchor model more often."
        ),
    )
    parser.add_argument(
        "--rescue-threshold-grid",
        type=str,
        default=",".join(str(x) for x in DEFAULT_RESCUE_THRESHOLD_GRID),
        help=(
            "Comma-separated thresholds for non-anchor rescue policies. "
            "A smaller model is used only when its rescue/confidence score is above the threshold."
        ),
    )
    parser.add_argument(
        "--small-route-rate-grid",
        type=str,
        default=",".join(str(x) for x in DEFAULT_SMALL_ROUTE_RATE_GRID),
        help=(
            "Comma-separated target non-anchor route rates. Thresholds are selected on validation "
            "and then evaluated on validation/test."
        ),
    )
    parser.add_argument(
        "--cascade-threshold-grid",
        type=str,
        default=",".join(str(x) for x in DEFAULT_CASCADE_THRESHOLD_GRID),
        help="Comma-separated probe-success thresholds for internal-state-only cascade policies.",
    )
    parser.add_argument(
        "--cascade-model-order",
        type=str,
        default=",".join(DEFAULT_CASCADE_MODEL_ORDER),
        help=(
            "Comma-separated cheap-to-expensive model order for probe-only cascades. "
            "The anchor model is used as fallback if no earlier model clears threshold."
        ),
    )
    args = parser.parse_args()

    config = load_probe_run_config(args.probe_dir)
    args.activations_root = resolve_path(
        args.activations_root, config, "activations_root", "outputs/activations/MATH"
    )
    args.answers_root = resolve_path(
        args.answers_root, config, "answers_root", "outputs/rollouts/MATH"
    )
    args.success_dir = resolve_path(
        args.success_dir,
        config,
        "success_dir",
        "outputs/success_rates/MATH",
    )
    args.token_position = int(resolve_value(args.token_position, config, "token_position", -1))
    args.random_seed = int(resolve_value(args.random_seed, config, "random_seed", 42))
    args.auroc_threshold = float(resolve_value(args.auroc_threshold, config, "auroc_threshold", 0.5))
    args.min_examples = int(resolve_value(args.min_examples, config, "min_examples", 100))
    args.min_positives_per_split = int(
        resolve_value(args.min_positives_per_split, config, "min_positives_per_split", 2)
    )
    split_fracs = tuple(float(x) for x in (args.split_fracs or config.get("split_fracs", DEFAULT_SPLIT_FRACS)))
    split_sum = math.fsum(split_fracs)
    if not np.isclose(split_sum, 1.0):
        raise ValueError(f"--split-fracs must sum to 1.0, got {split_sum}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lambda_grid = parse_lambda_grid(args.lambda_grid)
    anchor_margin_grid = parse_float_grid(args.anchor_margin_grid)
    if any(v < 0 for v in anchor_margin_grid):
        raise ValueError("--anchor-margin-grid values must be non-negative.")
    rescue_threshold_grid = parse_float_grid(args.rescue_threshold_grid)
    if any(v < 0 for v in rescue_threshold_grid):
        raise ValueError("--rescue-threshold-grid values must be non-negative.")
    small_route_rate_grid = parse_float_grid(args.small_route_rate_grid)
    if any(v < 0 or v > 1 for v in small_route_rate_grid):
        raise ValueError("--small-route-rate-grid values must be between 0 and 1.")
    cascade_threshold_grid = parse_float_grid(args.cascade_threshold_grid)
    if any(v < 0 or v > 1 for v in cascade_threshold_grid):
        raise ValueError("--cascade-threshold-grid values must be between 0 and 1.")
    cascade_model_order = parse_model_order(args.cascade_model_order)
    language_filter = parse_csv_filter(args.languages)
    model_filter = parse_csv_filter(args.models)
    args.cache_label_fallback_languages = parse_csv_filter(args.cache_label_fallback_languages) or set()
    canonical_problem_ids = load_canonical_problem_ids(args.canonical_csv_path)
    args.difficulty_max_quantile = resolve_value(
        args.difficulty_max_quantile,
        config,
        "difficulty_max_quantile",
        None,
    )
    args.problem_difficulties = load_problem_difficulties(args.canonical_csv_path)
    args.max_difficulty = difficulty_threshold_from_quantile(
        args.problem_difficulties,
        args.difficulty_max_quantile,
    )
    if args.max_difficulty is not None:
        print(
            f"Difficulty filter: <= {args.max_difficulty:.6f} "
            f"(quantile={args.difficulty_max_quantile})"
        )

    best_layers = load_best_layers(args.probe_dir, args.best_layers_path)
    if language_filter is not None:
        best_layers = best_layers[best_layers["language"].isin(language_filter)]
    if model_filter is not None:
        best_layers = best_layers[best_layers["model"].isin(model_filter)]
    if best_layers.empty:
        raise RuntimeError("No best-layer rows are available for routing.")

    effective_language_filter = language_filter or set(best_layers["language"].astype(str).unique())
    effective_model_filter = model_filter or set(best_layers["model"].astype(str).unique())

    success_df = read_success_table(args.success_dir)
    coverage = build_coverage(
        success_df,
        activations_root=args.activations_root,
        answers_root=args.answers_root,
        token_position=args.token_position,
        language_filter=effective_language_filter,
        model_filter=effective_model_filter,
    )
    ready_jobs = coverage[coverage["status"] == "ready"].copy()
    ready_jobs = ready_jobs.sort_values(["model", "language"]).reset_index(drop=True)

    coverage.to_csv(args.output_dir / "coverage_status.csv", index=False)
    best_layers.to_csv(args.output_dir / "router_best_layers.csv", index=False)

    print(f"Ready coverage rows: {len(ready_jobs)}")
    print(f"Best-layer rows: {len(best_layers)}")
    print("Building router candidates from validation-selected best layers ...")
    router_candidates = build_router_candidates(
        ready_jobs=ready_jobs,
        success_df=success_df,
        best_layers=best_layers,
        probe_dir=args.probe_dir,
        args=args,
        split_fracs=split_fracs,
        language_filter=effective_language_filter,
        model_filter=effective_model_filter,
        canonical_problem_ids=canonical_problem_ids,
    )
    router_candidates = filter_router_items(
        router_candidates,
        min_candidates=args.min_router_candidates,
        require_all_candidates=args.require_all_candidates,
    )
    if router_candidates.empty:
        raise RuntimeError("No eligible router candidate rows were built.")

    router_candidates, router_calibration = add_calibrated_predictions(
        router_candidates,
        alpha=args.calibration_alpha,
    )
    router_candidates, language_shrinkage_calibration = add_language_shrunk_predictions(
        router_candidates,
        shrinkage_strength=args.language_shrinkage_strength,
    )
    router_candidates, relative_value_calibration = add_relative_value_predictions(
        router_candidates,
        alpha=args.calibration_alpha,
    )
    router_candidates, anchor_advantage_calibration = add_anchor_advantage_predictions(
        router_candidates,
        alpha=args.calibration_alpha,
        anchor_model=args.anchor_model,
    )
    router_candidates, rescue_gain_calibration = add_rescue_gain_predictions(
        router_candidates,
        alpha=args.calibration_alpha,
        anchor_model=args.anchor_model,
    )
    if args.candidates_only:
        if args.write_router_candidates:
            router_candidates.to_csv(args.output_dir / "router_candidates.csv", index=False)
        print(f"Wrote candidates only → {args.output_dir}")
        return

    router_eval = evaluate_router_policies(
        router_candidates,
        lambda_grid=lambda_grid,
        anchor_model=args.anchor_model,
        anchor_margin_grid=anchor_margin_grid,
        rescue_threshold_grid=rescue_threshold_grid,
        small_route_rate_grid=small_route_rate_grid,
        cascade_threshold_grid=cascade_threshold_grid,
        cascade_model_order=cascade_model_order,
    )
    router_eval.to_csv(args.output_dir / "router_eval.csv", index=False)
    if args.write_router_candidates:
        router_candidates.to_csv(args.output_dir / "router_candidates.csv", index=False)

    router_summary = (
        router_candidates.groupby(["split", "language"], as_index=False)
        .agg(
            n_candidate_rows=("model", "size"),
            n_items=("base_problem_id", "nunique"),
            n_models_mean=("candidate_models", "mean"),
            mean_success=("true_success", "mean"),
            mean_output_cost_usd=("total_output_cost_usd", "mean"),
            mean_output_tokens=("total_output_tokens", "mean"),
        )
    )
    router_summary.to_csv(args.output_dir / "router_candidate_coverage.csv", index=False)

    write_json(
        args.output_dir / "run_config.json",
        {
            "probe_dir": str(args.probe_dir),
            "best_layers_path": str(args.best_layers_path) if args.best_layers_path else None,
            "activations_root": str(args.activations_root),
            "answers_root": str(args.answers_root),
            "success_dir": str(args.success_dir),
            "output_dir": str(args.output_dir),
            "canonical_csv_path": str(args.canonical_csv_path),
            "difficulty_max_quantile": args.difficulty_max_quantile,
            "max_difficulty": args.max_difficulty,
            "token_position": args.token_position,
            "languages": sorted(language_filter) if language_filter else None,
            "models": sorted(model_filter) if model_filter else None,
            "effective_languages": sorted(effective_language_filter) if effective_language_filter else None,
            "effective_models": sorted(effective_model_filter) if effective_model_filter else None,
            "split_fracs": split_fracs,
            "random_seed": args.random_seed,
            "auroc_threshold": args.auroc_threshold,
            "min_examples": args.min_examples,
            "min_positives_per_split": args.min_positives_per_split,
            "alignment_min_corr": args.alignment_min_corr,
            "alignment_max_mismatch_frac": args.alignment_max_mismatch_frac,
            "cache_label_fallback_languages": sorted(args.cache_label_fallback_languages),
            "lambda_grid": lambda_grid,
            "calibration_alpha": args.calibration_alpha,
            "language_shrinkage_strength": args.language_shrinkage_strength,
            "min_router_candidates": args.min_router_candidates,
            "require_all_candidates": args.require_all_candidates,
            "write_router_candidates": args.write_router_candidates,
            "anchor_model": args.anchor_model,
            "anchor_margin_grid": anchor_margin_grid,
            "rescue_threshold_grid": rescue_threshold_grid,
            "small_route_rate_grid": small_route_rate_grid,
            "cascade_threshold_grid": cascade_threshold_grid,
            "cascade_model_order": cascade_model_order,
            "router_calibration": router_calibration,
            "language_shrinkage_calibration": language_shrinkage_calibration,
            "relative_value_calibration": relative_value_calibration,
            "anchor_advantage_calibration": anchor_advantage_calibration,
            "rescue_gain_calibration": rescue_gain_calibration,
        },
    )

    print(
        f"Router candidates: rows={len(router_candidates)} "
        f"items={router_candidates[['split', 'language', 'base_problem_id']].drop_duplicates().shape[0]}"
    )
    print(f"Router metrics: {args.output_dir / 'router_eval.csv'}")

    test_routes = router_eval[router_eval["split"] == "test"].copy()
    if not test_routes.empty:
        ranked = test_routes.sort_values(
            ["mean_success", "mean_output_cost_usd"],
            ascending=[False, True],
        ).head(8)
        print("\nTop test routing rows:")
        print(
            ranked[
                ["policy", "lambda", "n_items", "mean_success", "mean_output_cost_usd", "mean_cost_norm"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
