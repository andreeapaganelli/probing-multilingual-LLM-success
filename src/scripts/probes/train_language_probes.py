from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.linalg import LinAlgWarning
from scipy.stats import ConstantInputWarning, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

from src.calibration_metrics import safe_brier, safe_ece, safe_log_loss

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.probes import BalancedLogisticProbe, BalancedRidgeProbe, LogisticProbe, ScaledRidgeProbe


SHORT_NAME = {
    "Qwen3-0.6B": "Qwen3-0.6B",
    "Qwen3-1.7B": "Qwen3-1.7B",
    "Qwen3-4B": "Qwen3-4B",
    "Qwen3-8B": "Qwen3-8B",
    "Qwen3.5-4B": "Qwen3.5-4B",
    "Qwen3.5-9B": "Qwen3.5-9B",
    "gpt-oss-20b_high": "GPT-OSS high",
    "gpt-oss-20b_low": "GPT-OSS low",
}

DEFAULT_ALPHA_GRID = [10.0, 100.0, 1_000.0, 10_000.0, 100_000.0]
DEFAULT_RIDGE_IMPROVED_ALPHA_GRID = [100.0, 1_000.0, 10_000.0, 100_000.0]
DEFAULT_SPLIT_FRACS = (0.70, 0.15, 0.15)
DEFAULT_ALIGNMENT_MIN_CORR = 0.20
DEFAULT_ALIGNMENT_MAX_MISMATCH_FRAC = 0.30
DEFAULT_BEST_LAYER_AUROC_TOL = 0.005
DEFAULT_BEST_LAYER_TOL_MODE = "absolute"  # "absolute" | "relative"
DEFAULT_BEST_LAYER_RELATIVE_TOL = 0.01    # 1 % of the best metric value
# "best": single layer that maximizes the selection metric on val (tie: shallower layer, then val_rmse).
# "band_earliest": near-best band from tolerance rules, then earliest layer in band (plateau-stable).
DEFAULT_BEST_LAYER_POLICY = "best"

# Supported probe types.
PROBE_TYPE_RIDGE = "ridge"
PROBE_TYPE_RIDGE_BALANCED = "ridge_balanced"
PROBE_TYPE_RIDGE_SCALED = "ridge_scaled"
PROBE_TYPE_RIDGE_SCALED_BALANCED = "ridge_scaled_balanced"
PROBE_TYPE_LOGISTIC = "logistic"
PROBE_TYPE_LOGISTIC_BALANCED = "logistic_balanced"
PROBE_TYPES = (
    PROBE_TYPE_RIDGE,
    PROBE_TYPE_RIDGE_BALANCED,
    PROBE_TYPE_RIDGE_SCALED,
    PROBE_TYPE_RIDGE_SCALED_BALANCED,
    PROBE_TYPE_LOGISTIC,
    PROBE_TYPE_LOGISTIC_BALANCED,
)


def parse_csv_filter(value: str | None) -> set[str] | None:
    if value is None:
        return None
    items = {item.strip() for item in value.split(",") if item.strip()}
    return items or None


def load_canonical_problem_ids(csv_path: Path) -> dict[str, set[str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing canonical dataset CSV: {csv_path}")
    by_language: dict[str, set[str]] = {}
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Canonical dataset has no header row: {csv_path}")
        language_columns = [col for col in reader.fieldnames if col not in {"difficulty", "answer", "ground_truth", "split"}]
        for language in language_columns:
            by_language[language] = set()
        for i, row in enumerate(reader):
            for language in language_columns:
                text = (row.get(language) or "").strip()
                if text:
                    by_language[language].add(f"{i}_{language}")
    return by_language


def load_problem_difficulties(csv_path: Path) -> dict[str, float]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing canonical dataset CSV: {csv_path}")
    difficulties: dict[str, float] = {}
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "difficulty" not in reader.fieldnames:
            return difficulties
        for i, row in enumerate(reader):
            val = (row.get("difficulty") or "").strip()
            if val:
                try:
                    difficulties[str(i)] = float(val)
                except ValueError:
                    pass
    return difficulties


def difficulty_threshold_from_quantile(
    difficulties: dict[str, float],
    quantile: float | None,
) -> float | None:
    if quantile is None:
        return None
    if not 0.0 < quantile <= 1.0:
        raise ValueError(f"--difficulty-max-quantile must be in (0, 1], got {quantile}")
    return float(np.quantile(np.array(list(difficulties.values()), dtype=np.float32), quantile))


def parse_float_grid(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one alpha value.")
    return values


def safe_path_part(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("._") or "unnamed"


def probe_type_suffix(probe_type: str) -> str:
    return "" if probe_type == PROBE_TYPE_RIDGE else f"_{safe_path_part(probe_type)}"


def probe_bundle_root(output_dir: Path, probe_type: str) -> Path:
    return output_dir / f"probes{probe_type_suffix(probe_type)}" if probe_type != PROBE_TYPE_RIDGE else output_dir / "probes_ridge"


def probe_bundle_path(
    output_dir: Path,
    language: str,
    model: str,
    probe_type: str = PROBE_TYPE_RIDGE,
) -> Path:
    return probe_bundle_root(output_dir, probe_type) / safe_path_part(model) / f"{safe_path_part(language)}.joblib"


def output_table_path(output_dir: Path, stem: str, probe_type: str) -> Path:
    return output_dir / f"{stem}{probe_type_suffix(probe_type)}.csv"


def output_config_path(output_dir: Path, probe_type: str) -> Path:
    return output_dir / f"run_config{probe_type_suffix(probe_type)}.json"


def stable_unit_interval(value: str, seed: int) -> float:
    payload = f"{seed}:{value}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:16]
    return int(digest, 16) / float(16**16)


def assign_grouped_splits(
    base_problem_ids: np.ndarray,
    split_fracs: tuple[float, float, float],
    random_seed: int,
) -> np.ndarray:
    train_frac, val_frac, _ = split_fracs
    split_by_group = {}
    for pid in pd.Series(base_problem_ids).astype(str).drop_duplicates():
        u = stable_unit_interval(pid, seed=random_seed)
        if u < train_frac:
            split_by_group[pid] = "probe_train"
        elif u < train_frac + val_frac:
            split_by_group[pid] = "val"
        else:
            split_by_group[pid] = "test"
    return np.array([split_by_group[str(pid)] for pid in base_problem_ids])


def safe_auroc(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    y_bin = (np.asarray(y_true) > threshold).astype(int)
    if len(y_bin) == 0 or y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
        return float("nan")
    return float(roc_auc_score(y_bin, scores))


def safe_spearman(y_true: np.ndarray, scores: np.ndarray) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConstantInputWarning)
        corr = spearmanr(y_true, scores).correlation
    return 0.0 if corr is None or np.isnan(corr) else float(corr)


def rmse(y_true: np.ndarray, scores: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, scores)))


def regression_metrics(y_true: np.ndarray, scores: np.ndarray, auroc_threshold: float) -> dict[str, float]:
    clipped = np.clip(scores, 0.0, 1.0)
    return {
        "r2": float(r2_score(y_true, clipped)) if len(np.unique(y_true)) > 1 else float("nan"),
        "mae": float(mean_absolute_error(y_true, clipped)),
        "rmse": rmse(y_true, clipped),
        "spearman": safe_spearman(y_true, clipped),
        "auroc": safe_auroc(y_true, clipped, threshold=auroc_threshold),
        "log_loss": safe_log_loss(y_true, clipped, threshold=auroc_threshold),
        "brier": safe_brier(y_true, clipped, threshold=auroc_threshold),
        "ece": safe_ece(y_true, clipped, threshold=auroc_threshold),
    }


def split_quality(y: np.ndarray, splits: np.ndarray, auroc_threshold: float) -> dict[str, int]:
    out = {}
    y_bin = (y > auroc_threshold).astype(int)
    for split in ["probe_train", "val", "test"]:
        mask = splits == split
        out[f"n_{split}"] = int(mask.sum())
        out[f"pos_{split}"] = int(y_bin[mask].sum())
        out[f"neg_{split}"] = int(mask.sum() - y_bin[mask].sum())
    return out


def read_success_table(directory: Path) -> pd.DataFrame:
    from src.data import load_success_rates
    return load_success_rates(directory)


def discover_activation_files(root: Path, token_position: int) -> pd.DataFrame:
    rows = []
    for path in sorted(root.glob(f"*/*/activations_token_{token_position}.joblib")):
        rows.append(
            {
                "language": path.parent.parent.name,
                "model": path.parent.name,
                "activation_path": path,
                "has_activation": True,
            }
        )
    return pd.DataFrame(rows)


def discover_answer_files(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.glob("*/*.jsonl")):
        rows.append(
            {
                "language": path.parent.name,
                "model": path.stem,
                "answers_path": path,
                "has_answers_file": True,
            }
        )
    known_languages = {
        "English", "French", "Italian", "Swahili", "Chinese",
        "Arabic", "Bengali", "Dutch", "German", "Hindi",
        "Japanese", "Korean", "Portuguese", "Russian", "Spanish",
        "Turkish", "Vietnamese", "Greek", "Hebrew", "Polish",
        "Telugu", "Thai",
    }
    for path in sorted(root.glob("*.jsonl")):
        parts = path.stem.split("_")
        for split_at in range(len(parts) - 1, 0, -1):
            language = "_".join(parts[split_at:])
            if language in known_languages:
                rows.append(
                    {
                        "language": language,
                        "model": "_".join(parts[:split_at]),
                        "answers_path": path,
                        "has_answers_file": True,
                    }
                )
                break
    columns = ["language", "model", "answers_path", "has_answers_file"]
    return pd.DataFrame(rows, columns=columns).drop_duplicates(["language", "model"])


def activation_layer_cache_path(activation_path: Path, layer: int) -> Path:
    return activation_path.with_name(f"layer_{int(layer)}.npy")


def activation_metadata_cache_path(activation_path: Path) -> Path:
    return activation_path.with_name(f"{activation_path.stem}_metadata.joblib")


def load_activation_cache(path: Path, layers: list[int] | tuple[int, ...] | set[int] | None = None) -> dict[str, Any]:
    """Load an activation cache, preferring pre-extracted layer .npy files when available."""
    if layers is not None:
        requested_layers = sorted({int(layer) for layer in layers})
        metadata_path = activation_metadata_cache_path(path)
        layer_paths = {
            layer: activation_layer_cache_path(path, layer)
            for layer in requested_layers
        }
        if metadata_path.exists() and all(layer_path.exists() for layer_path in layer_paths.values()):
            try:
                metadata = joblib.load(metadata_path)
                if isinstance(metadata, dict):
                    cache = dict(metadata)
                    cache["activations"] = {
                        layer: np.load(layer_path, mmap_mode="r")
                        for layer, layer_path in layer_paths.items()
                    }
                    return cache
            except Exception as exc:
                print(f"Warning: failed to load layer cache for {path}: {exc!r}; falling back to joblib.")

    cache = joblib.load(path)
    if layers is not None and isinstance(cache, dict) and "activations" in cache:
        requested = {int(layer) for layer in layers}
        activations = cache["activations"]
        selected_activations = {}
        for layer in requested:
            value = activations.get(layer)
            if value is None:
                value = activations.get(str(layer))
            if value is not None:
                selected_activations[layer] = value
        cache = dict(cache)
        cache["activations"] = selected_activations
    return cache


def build_coverage(
    success_df: pd.DataFrame,
    activations_root: Path,
    answers_root: Path,
    token_position: int,
    language_filter: set[str] | None,
    model_filter: set[str] | None,
) -> pd.DataFrame:
    activation_inventory = discover_activation_files(activations_root, token_position)
    answer_inventory = discover_answer_files(answers_root)

    if language_filter is not None:
        success_df = success_df[success_df["language"].isin(language_filter)]
        activation_inventory = activation_inventory[activation_inventory["language"].isin(language_filter)]
        answer_inventory = answer_inventory[answer_inventory["language"].isin(language_filter)]

    if model_filter is not None:
        success_df = success_df[success_df["model"].isin(model_filter)]
        activation_inventory = activation_inventory[activation_inventory["model"].isin(model_filter)]
        answer_inventory = answer_inventory[answer_inventory["model"].isin(model_filter)]

    success_pairs = (
        success_df.groupby(["language", "model"], as_index=False)
        .agg(scored_rows=("success_rate", "size"), mean_success_rate=("success_rate", "mean"))
    )
    success_pairs["has_scored_success"] = True

    all_pairs = pd.concat(
        [
            activation_inventory[["language", "model"]]
            if not activation_inventory.empty
            else pd.DataFrame(columns=["language", "model"]),
            answer_inventory[["language", "model"]]
            if not answer_inventory.empty
            else pd.DataFrame(columns=["language", "model"]),
            success_pairs[["language", "model"]]
            if not success_pairs.empty
            else pd.DataFrame(columns=["language", "model"]),
        ],
        ignore_index=True,
    ).drop_duplicates()

    if answer_inventory.empty:
        answer_inventory = pd.DataFrame(columns=["language", "model", "answers_path", "has_answers_file"])
    coverage = all_pairs.merge(activation_inventory, on=["language", "model"], how="left")
    coverage = coverage.merge(answer_inventory, on=["language", "model"], how="left")
    coverage = coverage.merge(success_pairs, on=["language", "model"], how="left")
    for col in ["has_activation", "has_answers_file", "has_scored_success"]:
        coverage[col] = coverage[col].fillna(False).astype(bool)
    coverage["status"] = np.select(
        [
            coverage["has_activation"] & coverage["has_scored_success"],
            coverage["has_activation"] & ~coverage["has_scored_success"],
            ~coverage["has_activation"] & coverage["has_scored_success"],
            coverage["has_answers_file"] & ~coverage["has_scored_success"],
        ],
        ["ready", "activation_no_scores", "scores_no_activation", "answers_unscored"],
        default="missing",
    )
    return coverage.sort_values(["language", "model"]).reset_index(drop=True)


def load_aligned_dataset(
    row: pd.Series,
    success: pd.DataFrame,
    split_fracs: tuple[float, float, float],
    random_seed: int,
    min_examples: int,
    min_positives_per_split: int,
    auroc_threshold: float,
    canonical_problem_ids: dict[str, set[str]] | None = None,
    alignment_min_corr: float = DEFAULT_ALIGNMENT_MIN_CORR,
    alignment_max_mismatch_frac: float = DEFAULT_ALIGNMENT_MAX_MISMATCH_FRAC,
    cache_label_fallback_languages: set[str] | None = None,
    problem_difficulties: dict[str, float] | None = None,
    max_difficulty: float | None = None,
    activation_layers: list[int] | tuple[int, ...] | set[int] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    language = str(row["language"])
    model = str(row["model"])
    path = Path(row["activation_path"])
    info = {"language": language, "model": model, "activation_path": str(path)}

    if not path.exists():
        return None, {**info, "status": "missing_activation_file"}

    scores = success[(success["language"] == language) & (success["model"] == model)].copy()
    if scores.empty:
        return None, {**info, "status": "missing_success_rows"}
    scores = scores.dropna(subset=["success_rate"]).drop_duplicates("problem_id", keep="last")

    try:
        cache = load_activation_cache(path, layers=activation_layers)
    except Exception as exc:
        return None, {**info, "status": "load_error", "error": repr(exc)}

    if not isinstance(cache, dict) or "activations" not in cache:
        return None, {**info, "status": "bad_cache_format"}
    activations = cache["activations"]
    problem_ids = cache.get("problem_ids")
    if problem_ids is None:
        return None, {**info, "status": "missing_problem_ids"}

    cache_df = pd.Series(problem_ids, name="problem_id").astype(str).reset_index().rename(columns={"index": "row_idx"})
    n_duplicate_ids = int(cache_df["problem_id"].duplicated().sum())
    if n_duplicate_ids > 0:
        return None, {
            **info,
            "status": "duplicate_cache_problem_ids",
            "n_cache": len(cache_df),
            "n_duplicate_problem_ids": n_duplicate_ids,
        }

    if canonical_problem_ids is not None and language in canonical_problem_ids:
        canonical_ids = canonical_problem_ids[language]
        missing_in_canonical = ~cache_df["problem_id"].isin(canonical_ids)
        n_missing_in_canonical = int(missing_in_canonical.sum())
        if n_missing_in_canonical > 0:
            return None, {
                **info,
                "status": "cache_problem_ids_not_in_canonical_dataset",
                "n_cache": len(cache_df),
                "n_ids_not_in_canonical": n_missing_in_canonical,
            }

    merged = cache_df.merge(
        scores[["problem_id", "base_problem_id", "success_rate", "n_rollouts", "n_correct"]],
        on="problem_id",
        how="left",
    )
    n_overlap_success = int(merged["success_rate"].notna().sum())
    if n_overlap_success < min_examples:
        return None, {
            **info,
            "status": "too_few_overlaps",
            "n_overlap": n_overlap_success,
            "n_cache": len(cache_df),
            "n_matched_labels": n_overlap_success,
            "n_missing_scores": int((merged["success_rate"].isna()).sum()),
        }

    merged["selected_success"] = merged["success_rate"]
    label_source = "success_table"
    alignment_corr = float("nan")
    alignment_mismatch_count = 0
    alignment_overlap = 0
    alignment_mismatch_frac = float("nan")

    cache_labels = cache.get("labels")
    if cache_labels is not None:
        cache_labels_arr = np.asarray(cache_labels, dtype=np.float32).reshape(-1)
        if len(cache_labels_arr) == len(merged):
            merged["cache_label"] = cache_labels_arr
            overlap_mask = merged["success_rate"].notna() & merged["cache_label"].notna()
            alignment_overlap = int(overlap_mask.sum())
            if alignment_overlap > 1:
                ext = merged.loc[overlap_mask, "success_rate"].to_numpy(dtype=np.float32)
                lbl = merged.loc[overlap_mask, "cache_label"].to_numpy(dtype=np.float32)
                with np.errstate(invalid="ignore"):
                    alignment_corr = float(np.corrcoef(ext, lbl)[0, 1])
                if np.isnan(alignment_corr):
                    alignment_corr = float("nan")
                alignment_mismatch_count = int(np.sum(np.abs(ext - lbl) > 1e-6))
                alignment_mismatch_frac = float(alignment_mismatch_count / alignment_overlap)

            suspicious_alignment = (
                alignment_overlap >= min_examples
                and (
                    (not np.isnan(alignment_corr) and alignment_corr < alignment_min_corr)
                    or (
                        not np.isnan(alignment_mismatch_frac)
                        and alignment_mismatch_frac > alignment_max_mismatch_frac
                    )
                )
            )
            if suspicious_alignment:
                if cache_label_fallback_languages is not None and language in cache_label_fallback_languages:
                    merged["selected_success"] = merged["cache_label"]
                    label_source = "cache_labels_fallback"
                else:
                    label_source = "success_table_suspicious_cache_labels"
        else:
            merged["cache_label"] = np.nan
    else:
        merged["cache_label"] = np.nan

    keep = merged["selected_success"].notna()
    n_filtered_by_difficulty = 0
    if problem_difficulties is not None and max_difficulty is not None:
        merged["difficulty"] = merged["base_problem_id"].astype(str).map(problem_difficulties)
        difficulty_keep = merged["difficulty"].notna() & (merged["difficulty"] <= max_difficulty)
        n_filtered_by_difficulty = int((keep & ~difficulty_keep).sum())
        keep = keep & difficulty_keep
    else:
        merged["difficulty"] = np.nan

    n_overlap = int(keep.sum())
    if n_overlap < min_examples:
        return None, {
            **info,
            "status": "too_few_overlaps",
            "n_overlap": n_overlap,
            "n_cache": len(cache_df),
            "n_matched_labels": n_overlap,
            "n_filtered_by_difficulty": n_filtered_by_difficulty,
            "label_source": label_source,
        }

    idx = merged.loc[keep, "row_idx"].to_numpy(dtype=int)
    y = merged.loc[keep, "selected_success"].to_numpy(dtype=np.float32)
    base_problem_ids = merged.loc[keep, "base_problem_id"].astype(str).to_numpy()
    splits = assign_grouped_splits(base_problem_ids, split_fracs=split_fracs, random_seed=random_seed)
    quality = split_quality(y, splits, auroc_threshold=auroc_threshold)

    if min(quality["n_probe_train"], quality["n_val"], quality["n_test"]) == 0:
        return None, {**info, "status": "empty_split", **quality, "n_overlap": n_overlap}

    for split in ["probe_train", "val", "test"]:
        if quality[f"pos_{split}"] < min_positives_per_split or quality[f"neg_{split}"] < min_positives_per_split:
            return None, {**info, "status": "degenerate_split", **quality, "n_overlap": n_overlap}

    status = {
        **info,
        "status": "ready",
        **quality,
        "n_overlap": n_overlap,
        "n_cache": len(cache_df),
        "n_missing_scores": int((~keep).sum()),
        "n_filtered_by_difficulty": n_filtered_by_difficulty,
        "n_matched_labels": n_overlap,
        "label_source": label_source,
        "alignment_overlap": alignment_overlap,
        "alignment_corr": alignment_corr,
        "alignment_mismatch_count": alignment_mismatch_count,
        "alignment_mismatch_frac": alignment_mismatch_frac,
    }

    return {
        "language": language,
        "model": model,
        "path": path,
        "activations": activations,
        "problem_ids": merged.loc[keep, "problem_id"].astype(str).to_numpy(),
        "row_idx": idx,
        "y": y,
        "splits": splits,
        "base_problem_ids": base_problem_ids,
        "difficulty": merged.loc[keep, "difficulty"].to_numpy(dtype=np.float32),
        "n_rollouts": merged.loc[keep, "n_rollouts"].to_numpy(dtype=np.float32),
        "n_correct": merged.loc[keep, "n_correct"].to_numpy(dtype=np.float32),
        "layers": sorted(activations.keys()),
        "quality": quality,
        "n_cache": len(cache_df),
        "n_overlap": n_overlap,
        "n_missing_scores": int((~keep).sum()),
        "n_filtered_by_difficulty": n_filtered_by_difficulty,
        "alignment_label_source": label_source,
        "alignment_overlap": alignment_overlap,
        "alignment_corr": alignment_corr,
        "alignment_mismatch_count": alignment_mismatch_count,
        "alignment_mismatch_frac": alignment_mismatch_frac,
    }, status


def default_alpha_grid_for_probe_type(probe_type: str) -> list[float]:
    if probe_type in {PROBE_TYPE_RIDGE_BALANCED, PROBE_TYPE_RIDGE_SCALED_BALANCED}:
        return list(DEFAULT_RIDGE_IMPROVED_ALPHA_GRID)
    return list(DEFAULT_ALPHA_GRID)


def class_balanced_sample_weight(y: np.ndarray, threshold: float) -> np.ndarray:
    y_bin = (np.asarray(y) > threshold).astype(int)
    n_pos = int(y_bin.sum())
    n_neg = int(len(y_bin) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return np.ones(len(y_bin), dtype=np.float32)
    w_pos = len(y_bin) / (2.0 * n_pos)
    w_neg = len(y_bin) / (2.0 * n_neg)
    return np.where(y_bin == 1, w_pos, w_neg).astype(np.float32)


def _make_probe(
    probe_type: str,
    alpha: float,
    threshold: float = 0.5,
) -> "Ridge | BalancedRidgeProbe | ScaledRidgeProbe | LogisticProbe | BalancedLogisticProbe":
    """Instantiate a probe given the type and the current alpha grid value.

    For logistic probes C = 1 / alpha, inverting the regularisation direction
    so the same alpha_grid can be reused for both probe types.
    ``threshold`` is used by BalancedRidgeProbe to binarise y when computing
    class-balanced sample weights.
    """
    if probe_type == PROBE_TYPE_LOGISTIC:
        return LogisticProbe(C=1.0 / alpha)
    if probe_type == PROBE_TYPE_LOGISTIC_BALANCED:
        return BalancedLogisticProbe(C=1.0 / alpha)
    if probe_type in {PROBE_TYPE_RIDGE_SCALED, PROBE_TYPE_RIDGE_SCALED_BALANCED}:
        return ScaledRidgeProbe(alpha=alpha)
    if probe_type == PROBE_TYPE_RIDGE_BALANCED:
        return BalancedRidgeProbe(alpha=alpha, threshold=threshold)
    return Ridge(alpha=alpha)


def _alpha_selection_score(metrics: dict[str, float], selection_metric: str) -> float:
    """Convert per-alpha validation metrics into a score where higher is better."""
    if selection_metric == "nll":
        raw = metrics["log_loss"]
        return -raw if not np.isnan(raw) else -metrics["rmse"]
    # default: auroc
    return metrics["auroc"] if not np.isnan(metrics["auroc"]) else -metrics["rmse"]


def fit_layer_probe(
    X: np.ndarray,
    y: np.ndarray,
    splits: np.ndarray,
    alpha_grid: list[float],
    auroc_threshold: float,
    probe_type: str = PROBE_TYPE_RIDGE,
    selection_metric: str = "auroc",
) -> tuple[dict[str, float], "Ridge | ScaledRidgeProbe | LogisticProbe | BalancedLogisticProbe"]:
    masks = {name: splits == name for name in ["probe_train", "val", "test"]}
    X_train = X[masks["probe_train"]].astype(np.float32, copy=False)
    y_train = y[masks["probe_train"]]
    X_val = X[masks["val"]].astype(np.float32, copy=False)
    y_val = y[masks["val"]]

    best = None
    for alpha in alpha_grid:
        probe = _make_probe(probe_type, alpha, threshold=auroc_threshold)
        if probe_type == PROBE_TYPE_RIDGE_SCALED_BALANCED:
            sample_weight = class_balanced_sample_weight(y_train, threshold=auroc_threshold)
            probe.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            probe.fit(X_train, y_train)
        val_pred = np.clip(probe.predict(X_val), 0.0, 1.0)
        metrics = regression_metrics(y_val, val_pred, auroc_threshold=auroc_threshold)
        score = _alpha_selection_score(metrics, selection_metric)
        if best is None or score > best["selection_score"]:
            best = {"alpha": alpha, "probe": probe, "selection_score": score}

    if best is None:
        raise RuntimeError("No probe was trained; alpha grid may be empty.")

    probe = best["probe"]
    record = {
        "best_alpha": float(best["alpha"]),
        "selection_score": float(best["selection_score"]),
        "selection_metric": selection_metric,
        "probe_type": probe_type,
    }
    for split_name, mask in masks.items():
        pred = np.clip(probe.predict(X[mask].astype(np.float32, copy=False)), 0.0, 1.0)
        metrics = regression_metrics(y[mask], pred, auroc_threshold=auroc_threshold)
        for metric_name, value in metrics.items():
            record[f"{split_name}_{metric_name}"] = value
    return record, probe


def select_best_layers(
    results: pd.DataFrame,
    auroc_tol: float = DEFAULT_BEST_LAYER_AUROC_TOL,
    selection_metric: str = "auroc",
    tol_mode: str = DEFAULT_BEST_LAYER_TOL_MODE,
    best_layer_policy: str = DEFAULT_BEST_LAYER_POLICY,
) -> pd.DataFrame:
    """Select the best layer per (language, model) from a sweep of per-layer probe metrics.

    Policy ``best``
    ---------------
    Choose the validation layer that **strictly maximizes** the selection score
    (AUROC or negated val log-loss). Ties break on smaller layer index, then lower
    val_rmse.

    Policy ``band_earliest``
    ------------------------
    All layers whose validation score falls within *tol* of the best score form a band.

    absolute (default for tol_mode)
        Band = layers where  score >= best_score - tol

    relative
        AUROC: band = score >= best_score * (1 - tol)
        NLL:   band = log_loss <= best_log_loss * (1 + tol)

    Then the **earliest** layer in the band wins (tie: lower val_rmse).
    """
    if results.empty:
        return pd.DataFrame()
    df = results.copy()
    if selection_metric == "nll":
        df["_selection"] = -df["val_log_loss"].fillna(np.inf)
    else:
        df["_selection"] = df["val_auroc"].fillna(-np.inf)
    tol = float(auroc_tol)
    policy = best_layer_policy
    rows = []
    for _, group in df.groupby(["language", "model"], sort=False):
        if np.isfinite(group["_selection"]).any():
            best_score = float(group["_selection"].max())
            if policy == "best":
                cand = group[np.isfinite(group["_selection"])].copy()
                cand = cand.sort_values(
                    ["_selection", "layer", "val_rmse"],
                    ascending=[False, True, True],
                )
                band = cand.iloc[[0]].copy()
                band_median_layer = float(band["layer"].iloc[0])
            else:
                if tol_mode == "relative":
                    if selection_metric == "nll":
                        best_nll = -best_score
                        threshold_nll = best_nll * (1.0 + tol)
                        band = group[group["_selection"] >= -threshold_nll].copy()
                    else:
                        threshold = best_score * (1.0 - tol)
                        band = group[group["_selection"] >= threshold].copy()
                else:
                    band = group[group["_selection"] >= best_score - tol].copy()
                band_median_layer = float(band["layer"].median())
                band = band.sort_values(["layer", "val_rmse"], ascending=[True, True])
            selected = band.iloc[0].copy()
            if policy == "best":
                sel_method = f"val_{selection_metric}_best_layer"
            else:
                sel_method = f"val_{selection_metric}_{tol_mode}_band_earliest_layer"
            selected["selection_method"] = sel_method
            selected["best_layer_selection_metric"] = selection_metric
            selected["best_layer_policy"] = policy
            selected["best_layer_tol"] = tol
            selected["best_layer_band_size"] = int(len(band))
            selected["best_layer_band_min_layer"] = int(band["layer"].min())
            selected["best_layer_band_max_layer"] = int(band["layer"].max())
            selected["best_layer_band_min_frac"] = float(band["layer_frac"].min())
            selected["best_layer_band_max_frac"] = float(band["layer_frac"].max())
            selected["best_layer_band_median_layer"] = band_median_layer
            selected["best_layer_band_median_frac"] = float(band["layer_frac"].median())
            selected["best_layer_metric_max"] = best_score if selection_metric == "auroc" else -best_score
            selected["best_layer_metric_min"] = (
                float(band["_selection"].min()) if selection_metric == "auroc"
                else -float(band["_selection"].max())
            )
        else:
            rmse_idx = group["val_rmse"].idxmin()
            selected = df.loc[rmse_idx].copy()
            selected["selection_method"] = "val_rmse_fallback"
            selected["best_layer_selection_metric"] = selection_metric
            selected["best_layer_policy"] = policy
            selected["best_layer_tol"] = tol
            selected["best_layer_band_size"] = 1
            selected["best_layer_band_min_layer"] = int(selected["layer"])
            selected["best_layer_band_max_layer"] = int(selected["layer"])
            selected["best_layer_band_min_frac"] = float(selected["layer_frac"])
            selected["best_layer_band_max_frac"] = float(selected["layer_frac"])
            selected["best_layer_band_median_layer"] = float(selected["layer"])
            selected["best_layer_band_median_frac"] = float(selected["layer_frac"])
            selected["best_layer_metric_max"] = float("nan")
            selected["best_layer_metric_min"] = float("nan")
        rows.append(selected.drop(labels=["_selection"], errors="ignore"))
    return pd.DataFrame(rows).reset_index(drop=True)


def rows_from_existing_bundle(path: Path, expected_probe_type: str) -> list[dict[str, Any]]:
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "layers" not in bundle:
        raise ValueError(f"Invalid probe bundle: {path}")
    metadata = bundle.get("metadata", {})
    found_probe_type = metadata.get("probe_type", PROBE_TYPE_RIDGE)
    if found_probe_type != expected_probe_type:
        raise ValueError(
            f"Probe bundle type mismatch: expected {expected_probe_type}, found {found_probe_type}"
        )
    rows = []
    for layer_payload in bundle["layers"].values():
        record = dict(layer_payload["record"])
        record["probe_path"] = str(path)
        record["probe_type"] = found_probe_type
        rows.append(record)
    return rows


def train_language_model_probes(
    row: pd.Series,
    success: pd.DataFrame,
    canonical_problem_ids: dict[str, set[str]],
    output_dir: Path,
    args: argparse.Namespace,
    alpha_grid: list[float],
    split_fracs: tuple[float, float, float],
    probe_type: str = PROBE_TYPE_RIDGE,
    selection_metric: str = "auroc",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bundle_path = probe_bundle_path(output_dir, str(row["language"]), str(row["model"]), probe_type)
    if bundle_path.exists() and not args.overwrite:
        try:
            records = rows_from_existing_bundle(bundle_path, expected_probe_type=probe_type)
            return records, {
                "language": str(row["language"]),
                "model": str(row["model"]),
                "activation_path": str(row["activation_path"]),
                "probe_path": str(bundle_path),
                "probe_type": probe_type,
                "status": "loaded_existing_probe",
            }
        except Exception as exc:
            print(f"Could not reuse {bundle_path}: {exc}; retraining.")

    dataset, status = load_aligned_dataset(
        row,
        success,
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
    )
    if dataset is None:
        return [], status

    records = []
    layer_payloads = {}
    layers = dataset["layers"]
    n_layers = len(layers)
    for layer in layers:
        X = dataset["activations"][layer][dataset["row_idx"]]
        metrics, probe = fit_layer_probe(
            X,
            dataset["y"],
            dataset["splits"],
            alpha_grid=alpha_grid,
            auroc_threshold=args.auroc_threshold,
            probe_type=probe_type,
            selection_metric=selection_metric,
        )
        d_model = int(X.shape[1])
        record = {
            "language": dataset["language"],
            "model": dataset["model"],
            "model_label": SHORT_NAME.get(dataset["model"], dataset["model"]),
            "layer": int(layer),
            "layer_frac": float(layer / (n_layers - 1)) if n_layers > 1 else 0.0,
            "n_layers": int(n_layers),
            "d_model": d_model,
            "n_examples": int(dataset["n_overlap"]),
            "n_missing_scores": int(dataset["n_missing_scores"]),
            "n_filtered_by_difficulty": int(dataset["n_filtered_by_difficulty"]),
            "probe_path": str(bundle_path),
            "probe_type": probe_type,
            **dataset["quality"],
            **metrics,
        }
        records.append(record)
        layer_payloads[int(layer)] = {"probe": probe, "record": record}

    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "metadata": {
                "language": dataset["language"],
                "model": dataset["model"],
                "activation_path": str(dataset["path"]),
                "probe_type": probe_type,
                "selection_metric": selection_metric,
                "alpha_grid": alpha_grid,
                "split_fracs": split_fracs,
                "random_seed": args.random_seed,
                "auroc_threshold": args.auroc_threshold,
                "min_examples": args.min_examples,
                "min_positives_per_split": args.min_positives_per_split,
                "difficulty_max_quantile": args.difficulty_max_quantile,
                "max_difficulty": args.max_difficulty,
                "n_filtered_by_difficulty": dataset["n_filtered_by_difficulty"],
                "alignment_label_source": dataset["alignment_label_source"],
                "alignment_overlap": dataset["alignment_overlap"],
                "alignment_corr": dataset["alignment_corr"],
                "alignment_mismatch_count": dataset["alignment_mismatch_count"],
                "alignment_mismatch_frac": dataset["alignment_mismatch_frac"],
                "n_cache": dataset["n_cache"],
                "n_overlap": dataset["n_overlap"],
                "n_missing_scores": dataset["n_missing_scores"],
                "quality": dataset["quality"],
            },
            "layers": layer_payloads,
        },
        bundle_path,
    )
    return records, {**status, "probe_path": str(bundle_path), "probe_type": probe_type}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train multilingual end-of-prompt probes and persist probe bundles."
    )
    parser.add_argument("--activations-root", type=Path, default=Path("outputs/activations/MATH"))
    parser.add_argument("--answers-root", type=Path, default=Path("outputs/rollouts/MATH"))
    parser.add_argument(
        "--success-dir",
        type=Path,
        default=Path("outputs/success_rates/MATH"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/probes/language_specific"))
    parser.add_argument("--token-position", type=int, default=-1)
    parser.add_argument("--languages", type=str, default=None, help="Comma-separated language filter.")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model filter.")
    parser.add_argument(
        "--alpha-grid",
        type=str,
        default=None,
        help=(
            "Comma-separated alpha values. Defaults depend on --probe-type; "
            "ridge_scaled_balanced uses a compact high-alpha grid unless overridden."
        ),
    )
    parser.add_argument("--split-fracs", type=float, nargs=3, default=DEFAULT_SPLIT_FRACS)
    parser.add_argument(
        "--best-layer-auroc-tol",
        type=float,
        default=DEFAULT_BEST_LAYER_AUROC_TOL,
        help=(
            "Tolerance for the near-best layer band. "
            "With --tol-mode absolute (default): absolute difference in metric units "
            "(e.g. 0.005 log-loss or AUROC units). "
            "With --tol-mode relative: fraction of the best score "
            "(e.g. 0.01 = accept layers within 1%% of best NLL). "
            f"Default: {DEFAULT_BEST_LAYER_AUROC_TOL}."
        ),
    )
    parser.add_argument(
        "--tol-mode",
        type=str,
        default=DEFAULT_BEST_LAYER_TOL_MODE,
        choices=["absolute", "relative"],
        help=(
            "How to interpret --best-layer-auroc-tol. "
            "'absolute': band threshold = best_score - tol (current behaviour). "
            "'relative': band threshold = best_score * (1 - tol) for AUROC, "
            "or best_nll * (1 + tol) for NLL — scales with model quality. "
            f"Default: {DEFAULT_BEST_LAYER_TOL_MODE}."
        ),
    )
    parser.add_argument(
        "--best-layer-policy",
        type=str,
        default=DEFAULT_BEST_LAYER_POLICY,
        choices=["best", "band_earliest"],
        help=(
            "How to pick the layer once per-layer validation metrics are computed. "
            "'best': the single layer with best val AUROC / val NLL (ties: shallower "
            "layer, then lower val_rmse). "
            "'band_earliest': near-best band controlled by "
            "--best-layer-auroc-tol / --tol-mode, then earliest layer in the band "
            "(useful when the metric plateau spans many depths). "
            f"Default: {DEFAULT_BEST_LAYER_POLICY}."
        ),
    )
    parser.add_argument(
        "--selection-metric",
        type=str,
        default="auroc",
        choices=["auroc", "nll"],
        help=(
            "Metric used to select the best regularization alpha and the best layer. "
            "'auroc' (default) preserves existing behaviour. "
            "'nll' uses binomial NLL (log loss), which rewards both calibration and discrimination "
            "and is better suited for routing via utility maximization."
        ),
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--auroc-threshold", type=float, default=0.5)
    parser.add_argument("--min-examples", type=int, default=100)
    parser.add_argument("--min-positives-per-split", type=int, default=2)
    parser.add_argument(
        "--canonical-csv-path",
        type=Path,
        default=Path("data/MATH_translated.csv"),
        help="Canonical multilingual dataset used to validate stable problem IDs.",
    )
    parser.add_argument(
        "--difficulty-max-quantile",
        type=float,
        default=None,
        help=(
            "If set, train/evaluate probes only on problems with difficulty at or below "
            "this global quantile of the canonical CSV. Use 0.5 for the easiest half."
        ),
    )
    parser.add_argument(
        "--alignment-min-corr",
        type=float,
        default=DEFAULT_ALIGNMENT_MIN_CORR,
        help="Minimum allowed corr(cache_labels, success_table) before alignment is flagged.",
    )
    parser.add_argument(
        "--alignment-max-mismatch-frac",
        type=float,
        default=DEFAULT_ALIGNMENT_MAX_MISMATCH_FRAC,
        help="Maximum allowed label mismatch fraction before alignment is flagged.",
    )
    parser.add_argument(
        "--cache-label-fallback-languages",
        type=str,
        default="",
        help="Comma-separated languages allowed to fallback to cache-embedded labels when alignment is suspicious.",
    )
    parser.add_argument(
        "--probe-type",
        type=str,
        default=PROBE_TYPE_RIDGE,
        choices=PROBE_TYPES,
        help=(
            "Probe model type. 'ridge' uses Ridge regression (default). "
            "'ridge_scaled' standardizes activations before Ridge regression. "
            "'logistic_balanced' uses class-balanced Logistic Regression "
            "to handle the GPT success-rate class imbalance."
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="Retrain even when a persisted bundle exists.")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=ConstantInputWarning)
    warnings.filterwarnings("ignore", category=LinAlgWarning)

    split_sum = math.fsum(args.split_fracs)
    if not np.isclose(split_sum, 1.0):
        raise ValueError(f"--split-fracs must sum to 1.0, got {split_sum}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    alpha_grid = (
        parse_float_grid(args.alpha_grid)
        if args.alpha_grid is not None
        else default_alpha_grid_for_probe_type(args.probe_type)
    )
    split_fracs = tuple(float(x) for x in args.split_fracs)

    success_df = read_success_table(args.success_dir)
    canonical_problem_ids = load_canonical_problem_ids(args.canonical_csv_path)
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
    args.cache_label_fallback_languages = parse_csv_filter(args.cache_label_fallback_languages) or set()
    language_filter = parse_csv_filter(args.languages)
    model_filter = parse_csv_filter(args.models)
    coverage = build_coverage(
        success_df,
        activations_root=args.activations_root,
        answers_root=args.answers_root,
        token_position=args.token_position,
        language_filter=language_filter,
        model_filter=model_filter,
    )
    ready_jobs = coverage[coverage["status"] == "ready"].copy()
    ready_jobs = ready_jobs.sort_values(["model", "language"]).reset_index(drop=True)

    all_records: list[dict[str, Any]] = []
    status_records: list[dict[str, Any]] = []

    print(f"Ready jobs: {len(ready_jobs)}")
    for i, row in ready_jobs.iterrows():
        print(f"[{i + 1:03d}/{len(ready_jobs):03d}] {row['language']} / {row['model']}")
        records, status = train_language_model_probes(
            row,
            success_df,
            canonical_problem_ids=canonical_problem_ids,
            output_dir=output_dir,
            args=args,
            alpha_grid=alpha_grid,
            split_fracs=split_fracs,
            probe_type=args.probe_type,
            selection_metric=args.selection_metric,
        )
        all_records.extend(records)
        status_records.append(status)
        if records:
            selected = select_best_layers(
                pd.DataFrame(records),
                auroc_tol=args.best_layer_auroc_tol,
                selection_metric=args.selection_metric,
                tol_mode=args.tol_mode,
                best_layer_policy=args.best_layer_policy,
            ).iloc[0]
            nll_str = (
                f" val NLL={selected['val_log_loss']:.4f}" if args.selection_metric == "nll" else ""
            )
            print(
                f"    layers={len(records)} examples={records[0]['n_examples']} "
                f"selected val AUROC={selected['val_auroc']:.3f}{nll_str} layer={int(selected['layer'])} "
                f"band={int(selected['best_layer_band_min_layer'])}-{int(selected['best_layer_band_max_layer'])} "
                f"test AUROC={selected['test_auroc']:.3f}"
            )
        else:
            print(f"    skipped: {status.get('status')} {status.get('error', '')}")

    probe_results = pd.DataFrame(all_records)
    ready_job_log = pd.DataFrame(status_records)
    non_ready_log = coverage[coverage["status"] != "ready"].copy()
    if not non_ready_log.empty:
        non_ready_log = non_ready_log[
            ["language", "model", "status", "has_activation", "has_answers_file", "has_scored_success"]
        ]
    full_status_log = pd.concat([ready_job_log, non_ready_log], ignore_index=True, sort=False)
    best_layers = select_best_layers(
        probe_results,
        auroc_tol=args.best_layer_auroc_tol,
        selection_metric=args.selection_metric,
        tol_mode=args.tol_mode,
        best_layer_policy=args.best_layer_policy,
    )

    coverage.to_csv(output_table_path(output_dir, "coverage_status", args.probe_type), index=False)
    if not probe_results.empty:
        probe_results.to_csv(output_table_path(output_dir, "per_language_layer_results", args.probe_type), index=False)
    if not best_layers.empty:
        best_layers.to_csv(output_table_path(output_dir, "per_language_best_layers", args.probe_type), index=False)
    if not full_status_log.empty:
        full_status_log.to_csv(output_table_path(output_dir, "status_log", args.probe_type), index=False)
    if not ready_job_log.empty:
        ready_job_log.to_csv(output_table_path(output_dir, "ready_job_load_log", args.probe_type), index=False)

    write_json(
        output_config_path(output_dir, args.probe_type),
        {
            "activations_root": str(args.activations_root),
            "answers_root": str(args.answers_root),
            "success_dir": str(args.success_dir),
            "canonical_csv_path": str(args.canonical_csv_path),
            "difficulty_max_quantile": args.difficulty_max_quantile,
            "max_difficulty": args.max_difficulty,
            "output_dir": str(args.output_dir),
            "probe_type": args.probe_type,
            "selection_metric": args.selection_metric,
            "token_position": args.token_position,
            "languages": sorted(language_filter) if language_filter else None,
            "models": sorted(model_filter) if model_filter else None,
            "alpha_grid": alpha_grid,
            "split_fracs": split_fracs,
            "best_layer_tol": args.best_layer_auroc_tol,
            "best_layer_tol_mode": args.tol_mode,
            "best_layer_policy": args.best_layer_policy,
            "random_seed": args.random_seed,
            "auroc_threshold": args.auroc_threshold,
            "min_examples": args.min_examples,
            "min_positives_per_split": args.min_positives_per_split,
            "alignment_min_corr": args.alignment_min_corr,
            "alignment_max_mismatch_frac": args.alignment_max_mismatch_frac,
            "cache_label_fallback_languages": sorted(args.cache_label_fallback_languages),
            "overwrite": args.overwrite,
        },
    )

    print(f"\nLayer result rows: {len(probe_results)}")
    print(f"Probe bundles: {probe_bundle_root(output_dir, args.probe_type)}")
    print(f"Result tables: {output_dir}")
    if not full_status_log.empty:
        print(full_status_log["status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
