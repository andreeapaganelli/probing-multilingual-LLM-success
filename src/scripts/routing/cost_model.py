"""Token-level cost model shared by the routing experiments.

Per-token output prices are proportional to parameter count for dense models
(C = 0.10 · P, in $/M tokens with P in billions). MoE models price active and
total parameters separately:

  C_m = α·P_active + β·P_total   ($/M tokens, params in billions)

  - Decode is memory-bound (roofline analysis, arXiv 2402.16363)
  - FLOPs ∝ active params; VRAM ∝ total params (Mixtral, arXiv 2401.04088)
  - MoE memory overhead is real but β < α (MoE-Lightning, arXiv 2411.11217)

gpt-oss-20b: P_active=3.6B, P_total=20B → C = 0.090×3.6 + 0.045×20 = $1.224/M.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.scripts.probes.train_language_probes import SHORT_NAME

DEFAULT_COST_REFERENCE_MODEL = "gpt-oss-20b_high"

MODEL_OUTPUT_COST_PER_MILLION = {
    "Qwen3-0.6B": 0.06,
    "Qwen/Qwen3-0.6B": 0.06,
    "Qwen3-1.7B": 0.17,
    "Qwen/Qwen3-1.7B": 0.17,
    "Qwen3-4B": 0.40,
    "Qwen/Qwen3-4B": 0.40,
    "Qwen3-8B": 0.80,
    "Qwen/Qwen3-8B": 0.80,
    "Qwen3.5-4B": 0.40,
    "Qwen/Qwen3.5-4B": 0.40,
    "Qwen3.5-9B": 0.90,
    "Qwen/Qwen3.5-9B": 0.90,
    "gpt-oss-20b_low": 1.224,
    "gpt-oss-20b_high": 1.224,
    "openai/gpt-oss-20b": 1.224,
}


def output_cost_per_million(model: str, record_model_name: str | None = None) -> float:
    for key in [model, record_model_name]:
        if key and key in MODEL_OUTPUT_COST_PER_MILLION:
            return MODEL_OUTPUT_COST_PER_MILLION[key]
    return 0.20


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_answer_token_costs(
    answers_root: Path,
    language_filter: set[str] | None,
    model_filter: set[str] | None,
) -> pd.DataFrame:
    """Build per-row USD output costs using the per-language baseline pricing convention."""
    rows: list[dict[str, Any]] = []
    known_languages = {
        "English", "Chinese", "Italian", "French", "Swahili",
        "Russian", "Turkish", "Arabic", "Thai", "Telugu",
    }
    answer_files: list[tuple[Path, str, str]] = []
    for path in sorted(answers_root.glob("*/*.jsonl")):
        answer_files.append((path, path.parent.name, path.stem))
    for path in sorted(answers_root.glob("*.jsonl")):
        parts = path.stem.split("_")
        for split_at in range(len(parts) - 1, 0, -1):
            language = "_".join(parts[split_at:])
            if language in known_languages:
                answer_files.append((path, language, "_".join(parts[:split_at])))
                break

    for path, language, model in answer_files:
        if language_filter is not None and language not in language_filter:
            continue
        if model_filter is not None and model not in model_filter:
            continue

        for record in iter_jsonl(path):
            total_tokens = 0
            counted_rollouts = 0
            record_model = record.get("model_name")
            output_per_million = output_cost_per_million(model, str(record_model) if record_model else None)
            for rollout in record.get("generated_solutions") or []:
                if not isinstance(rollout, dict):
                    continue
                try:
                    total_tokens += int(rollout.get("output_tokens") or 0)
                    counted_rollouts += 1
                except (TypeError, ValueError):
                    continue

            if counted_rollouts == 0:
                continue
            total_output_cost_usd = total_tokens * output_per_million / 1_000_000.0
            rows.append(
                {
                    "language": language,
                    "model": model,
                    "problem_id": str(record.get("problem_id")),
                    "total_output_tokens": float(total_tokens),
                    "mean_output_tokens": float(total_tokens / counted_rollouts),
                    "output_cost_per_million": float(output_per_million),
                    "total_output_cost_usd": float(total_output_cost_usd),
                    "mean_output_cost_usd": float(total_output_cost_usd / counted_rollouts),
                    "costed_rollouts": int(counted_rollouts),
                }
            )

    return pd.DataFrame(rows)


def attach_answer_costs(candidates: pd.DataFrame, answer_costs: pd.DataFrame) -> pd.DataFrame:
    candidates = candidates.copy()
    if answer_costs.empty:
        candidates["total_output_tokens"] = 1.0
        candidates["mean_output_tokens"] = 1.0
        candidates["output_cost_per_million"] = 1.0
        candidates["total_output_cost_usd"] = 1.0
        candidates["mean_output_cost_usd"] = 1.0
        candidates["costed_rollouts"] = 0
        return candidates

    answer_costs = (
        answer_costs.groupby(["language", "model", "problem_id"], as_index=False)
        .agg(
            total_output_tokens=("total_output_tokens", "mean"),
            mean_output_tokens=("mean_output_tokens", "mean"),
            output_cost_per_million=("output_cost_per_million", "mean"),
            total_output_cost_usd=("total_output_cost_usd", "mean"),
            mean_output_cost_usd=("mean_output_cost_usd", "mean"),
            costed_rollouts=("costed_rollouts", "max"),
        )
    )
    candidates = candidates.merge(
        answer_costs,
        on=["language", "model", "problem_id"],
        how="left",
    )

    total_by_model = answer_costs.groupby("model")["total_output_tokens"].mean()
    mean_by_model = answer_costs.groupby("model")["mean_output_tokens"].mean()
    output_cost_by_model = answer_costs.groupby("model")["output_cost_per_million"].mean()
    total_usd_by_model = answer_costs.groupby("model")["total_output_cost_usd"].mean()
    mean_usd_by_model = answer_costs.groupby("model")["mean_output_cost_usd"].mean()
    global_total = float(answer_costs["total_output_tokens"].mean())
    global_mean = float(answer_costs["mean_output_tokens"].mean())
    global_output_cost = float(answer_costs["output_cost_per_million"].mean())
    global_total_usd = float(answer_costs["total_output_cost_usd"].mean())
    global_mean_usd = float(answer_costs["mean_output_cost_usd"].mean())

    candidates["total_output_tokens"] = candidates["total_output_tokens"].fillna(
        candidates["model"].map(total_by_model).fillna(global_total)
    )
    candidates["mean_output_tokens"] = candidates["mean_output_tokens"].fillna(
        candidates["model"].map(mean_by_model).fillna(global_mean)
    )
    candidates["output_cost_per_million"] = candidates["output_cost_per_million"].fillna(
        candidates["model"].map(output_cost_by_model).fillna(global_output_cost)
    )
    candidates["total_output_cost_usd"] = candidates["total_output_cost_usd"].fillna(
        candidates["model"].map(total_usd_by_model).fillna(global_total_usd)
    )
    candidates["mean_output_cost_usd"] = candidates["mean_output_cost_usd"].fillna(
        candidates["model"].map(mean_usd_by_model).fillna(global_mean_usd)
    )
    candidates["costed_rollouts"] = candidates["costed_rollouts"].fillna(0).astype(int)
    return candidates


def compute_reference_tier_costs(candidates: pd.DataFrame) -> pd.DataFrame:
    """Compute model tiers from the mean cost of one training-set generation."""
    source = candidates[candidates["split"] == "probe_train"]
    if source.empty:
        source = candidates

    tier_costs = (
        source.groupby("model", as_index=False)
        .agg(
            tier_cost_usd=("mean_output_cost_usd", "mean"),
            tier_cost_tokens=("mean_output_tokens", "mean"),
            output_cost_per_million=("output_cost_per_million", "mean"),
            n_tier_cost_rows=("mean_output_cost_usd", "size"),
        )
        .sort_values("tier_cost_usd")
        .reset_index(drop=True)
    )
    reference_rows = tier_costs["model"].eq(DEFAULT_COST_REFERENCE_MODEL)
    reference_cost = (
        float(tier_costs.loc[reference_rows, "tier_cost_usd"].iloc[0])
        if reference_rows.any()
        else float(tier_costs["tier_cost_usd"].max())
    )
    if reference_cost <= 0:
        raise ValueError("The routing cost reference must have a positive predicted cost.")
    tier_costs["tier_cost_relative"] = tier_costs["tier_cost_usd"] / reference_cost
    # Compatibility alias for older candidate readers. This is relative, not min-max, cost.
    tier_costs["tier_cost_norm"] = tier_costs["tier_cost_relative"]
    tier_costs["tier_cost_rank"] = np.arange(1, len(tier_costs) + 1)
    tier_costs["model_label"] = tier_costs["model"].map(lambda m: SHORT_NAME.get(m, m))
    return tier_costs


def add_reference_tier_costs(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tier_costs = compute_reference_tier_costs(candidates)
    candidates = candidates.merge(
        tier_costs[
            [
                "model",
                "tier_cost_usd",
                "tier_cost_tokens",
                "tier_cost_relative",
                "tier_cost_norm",
                "tier_cost_rank",
            ]
        ],
        on="model",
        how="left",
    )
    return candidates, tier_costs
