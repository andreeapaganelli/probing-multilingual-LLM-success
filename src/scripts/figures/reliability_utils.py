"""Shared helpers for the reliability (calibration) figures.

Provides:
  - ``_select_nll_layer``: pick the probe layer by validation binomial NLL.
  - ``_load_test_data``: load (X_test, y_test) for one language/model/layer.
  - ``load_transfer_predictions``: predictions of a probe trained on one
    language evaluated on another language, restricted to the test split.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]


# ── layer selection ─────────────────────────────────────────────────────────────

def _select_nll_layer(
    bundle: dict,
    *,
    within_tolerance: bool = True,
    rel_tol: float = 0.01,
) -> int:
    """Layer with minimum val NLL on the probe's validation split.

    If ``within_tolerance`` is True: first (shallowest) layer whose val NLL is
    ≤ min_NLL × (1 + ``rel_tol``). Otherwise strict argmin of val NLL, breaking
    ties by smaller ``layer``.
    """
    records = [(layer, payload["record"]) for layer, payload in bundle["layers"].items()]
    layers_sorted = sorted(records, key=lambda x: x[0])
    nll_values = np.array([r["val_log_loss"] for _, r in layers_sorted])
    valid_mask = np.isfinite(nll_values)
    if not valid_mask.any():
        auroc_values = np.array([r.get("val_auroc", 0.0) for _, r in layers_sorted])
        return layers_sorted[int(np.argmax(auroc_values))][0]
    best_nll = float(nll_values[valid_mask].min())
    if not within_tolerance:
        best_layer: int | None = None
        best_val = float("inf")
        for layer, rec in layers_sorted:
            v = rec["val_log_loss"]
            if not np.isfinite(v):
                continue
            fv = float(v)
            if fv < best_val or (fv == best_val and (best_layer is None or layer < best_layer)):
                best_val = fv
                best_layer = int(layer)
        return (
            best_layer if best_layer is not None else layers_sorted[int(np.nanargmin(nll_values))][0]
        )
    threshold = best_nll * (1.0 + rel_tol)
    for layer, rec in layers_sorted:
        if np.isfinite(rec["val_log_loss"]) and float(rec["val_log_loss"]) <= threshold:
            return layer
    return layers_sorted[int(np.nanargmin(nll_values))][0]


# ── data loading ────────────────────────────────────────────────────────────────

def _load_test_data(
    language: str,
    model: str,
    layer: int,
    success_df: pd.DataFrame,
    activations_root: Path,
) -> tuple[np.ndarray, np.ndarray] | None:
    from src.scripts.probes.evaluate_cross_lingual_transfer import load_test_data_multilayer
    data = load_test_data_multilayer(language, model, {layer}, success_df, activations_root)
    return data.get(layer, None)


# ── split helpers (mirror the probe training code) ──────────────────────────────

def _stable_unit_interval(value: str, seed: int) -> float:
    payload = f"{seed}:{value}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:16]
    return int(digest, 16) / float(16**16)


def is_test(base_problem_id: str, train_frac: float, val_frac: float, seed: int) -> bool:
    u = _stable_unit_interval(str(base_problem_id), seed)
    return u >= train_frac + val_frac


def load_transfer_predictions(
    model: str,
    train_lang: str,
    test_lang: str,
    layer: int,
    probes_dir: Path,
    activations_root: Path,
    random_seed: int = 42,
    split_fracs: tuple[float, float, float] = (0.70, 0.15, 0.15),
    success_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Probe trained on train_lang, evaluated on test_lang at given layer.

    Returns (y_pred, y_true) restricted to the test split.
    The test split is defined by base_problem_id hashing — language-agnostic,
    so the same problems are held out for all languages.
    """
    # ── probe (trained on train_lang) ──
    probe_path = probes_dir / model / f"{train_lang}.joblib"
    bundle = joblib.load(probe_path)
    probe = bundle["layers"][layer]["probe"]

    # ── activations for test_lang ──
    act_path = activations_root / test_lang / model / "activations_token_-1.joblib"
    act_bundle = joblib.load(act_path)
    X = act_bundle["activations"][layer]               # (N, d_model)
    problem_ids: list[str] = list(act_bundle["problem_ids"])

    y_pred = np.clip(np.asarray(probe.predict(X), dtype=np.float32), 0.0, 1.0)

    # ── success rates for test_lang ──
    _sr_dir = success_dir if success_dir is not None else REPO_ROOT / "outputs" / "success_rates" / "MATH"
    sr_path = _sr_dir / f"{model}_{test_lang}.csv"
    sr_df = pd.read_csv(sr_path, usecols=["problem_id", "success_rate"])
    sr_map: dict[str, float] = dict(zip(sr_df["problem_id"], sr_df["success_rate"]))
    y_true = np.array([sr_map.get(pid, np.nan) for pid in problem_ids], dtype=np.float32)

    # ── filter to test split ──
    train_frac, val_frac, _ = split_fracs
    keep = [
        i for i, pid in enumerate(problem_ids)
        if not np.isnan(y_true[i])
        and is_test(pid.split("_")[0], train_frac, val_frac, random_seed)
    ]
    idx = np.array(keep, dtype=int)
    return y_pred[idx], y_true[idx]
