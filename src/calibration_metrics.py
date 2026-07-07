"""Shared calibration and probabilistic evaluation metrics.

Used by probe training scripts so the same ECE/Brier/NLL implementations
are consistent across the codebase.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss


def expected_calibration_error(
    y_true_binary: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    min_samples: int = 5,
) -> float:
    """Binned Expected Calibration Error (ECE).

    Partitions [0, 1] into equal-width bins and computes the
    weighted mean absolute gap between mean predicted confidence
    and empirical accuracy in each bin. Bins with fewer than
    min_samples examples are skipped.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            bin_mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        else:
            bin_mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        n_bin = int(np.sum(bin_mask))
        if n_bin >= min_samples:
            bin_acc = float(np.mean(y_true_binary[bin_mask]))
            bin_conf = float(np.mean(y_prob[bin_mask]))
            ece += (n_bin / len(y_prob)) * abs(bin_acc - bin_conf)
    return float(ece)


def safe_log_loss(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> float:
    """Binomial NLL (log loss) after binarizing y_true with threshold.

    Returns NaN when all labels are the same class (log loss is undefined).
    Clips scores away from 0/1 to avoid log(0).
    """
    y_bin = (np.asarray(y_true) > threshold).astype(int)
    if len(y_bin) == 0 or y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
        return float("nan")
    p = np.clip(scores, 1e-7, 1.0 - 1e-7)
    return float(log_loss(y_bin, p))


def safe_brier(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> float:
    """Brier score after binarizing y_true with threshold."""
    y_bin = (np.asarray(y_true) > threshold).astype(int)
    if len(y_bin) == 0:
        return float("nan")
    return float(brier_score_loss(y_bin, scores))


def safe_ece(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    n_bins: int = 10,
    min_samples: int = 5,
) -> float:
    """ECE after binarizing y_true with threshold."""
    y_bin = (np.asarray(y_true) > threshold).astype(int)
    if len(y_bin) == 0:
        return float("nan")
    return expected_calibration_error(y_bin, scores, n_bins=n_bins, min_samples=min_samples)
