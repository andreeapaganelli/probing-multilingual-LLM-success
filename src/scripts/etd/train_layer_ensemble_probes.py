"""Train and ensemble ETD probes from several layers of one encoder pass.

Each layer has an independently regularized Ridge head for every target model.
The heads' scalar predictions are combined by a mean, median, or convex weights
fit on validation data.  Extracting several layers does not require additional
encoder forward passes at deployment time.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

from src.scripts.etd.train_etd_probes import (
    ITEM_COLS,
    EncoderLayerStore,
    build_feature_matrix,
    parse_csv,
    parse_float_grid,
    rmse,
    safe_auroc,
    safe_path_part,
    train_probe,
)


DEFAULT_LAYERS = [9, 18, 25, 36]
DEFAULT_ALPHA_GRID = [1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0, 1_000_000.0]
VARIANTS = ("single_last", "mean", "median", "convex_val")


def fit_layer_target(
    target_model: str,
    candidates: pd.DataFrame,
    X: np.ndarray,
    feature_rows: np.ndarray,
    layer: int,
    alpha_grid: list[float],
    train_languages: list[str] | None,
    reuse_probes_from: Path | None,
    local_probe_path: Path,
) -> dict[str, Any]:
    target_mask = candidates["model"].astype(str).eq(target_model).to_numpy()
    fit_mask = target_mask.copy()
    if train_languages is not None:
        fit_mask &= candidates["language"].astype(str).isin(train_languages).to_numpy()
    train_mask = fit_mask & candidates["split"].eq("probe_train").to_numpy()
    val_mask = fit_mask & candidates["split"].eq("val").to_numpy()
    if not train_mask.any() or not val_mask.any():
        raise ValueError(f"Missing training/validation rows for target {target_model}")

    reuse_path = None
    if local_probe_path.exists():
        reuse_path = local_probe_path
    elif reuse_probes_from is not None:
        candidate = (
            reuse_probes_from
            / "probes"
            / f"layer_{layer:03d}"
            / f"{safe_path_part(target_model)}.joblib"
        )
        if candidate.exists():
            reuse_path = candidate

    if reuse_path is not None:
        reused = joblib.load(reuse_path)
        probe = reused["probe"]
        best_alpha = float(reused["alpha"])
        records = list(reused["alpha_grid"])
        best = pd.Series(next(row for row in records if float(row["alpha"]) == best_alpha))
        reused_probe = True
    else:
        X_train = X[feature_rows[train_mask]]
        y_train = candidates.loc[train_mask, "true_success"].to_numpy(dtype=float)
        X_val = X[feature_rows[val_mask]]
        y_val = candidates.loc[val_mask, "true_success"].to_numpy(dtype=float)
        trained = {}
        records = []
        with threadpool_limits(limits=1):
            for alpha in alpha_grid:
                probe = train_probe(X_train, y_train, alpha)
                trained[alpha] = probe
                val_pred = np.clip(probe.predict(X_val), 0.0, 1.0)
                records.append(
                    {
                        "encoder_layer": layer,
                        "target_model": target_model,
                        "alpha": alpha,
                        "val_auroc": safe_auroc(y_val, val_pred),
                        "val_rmse": rmse(y_val, val_pred),
                    }
                )
        ranked = pd.DataFrame(records)
        ranked["_auroc"] = ranked["val_auroc"].fillna(-np.inf)
        best = ranked.sort_values(
            ["_auroc", "val_rmse", "alpha"],
            ascending=[False, True, True],
            kind="mergesort",
        ).iloc[0]
        best_alpha = float(best["alpha"])
        probe = trained[best_alpha]
        reused_probe = False

    row_idx = np.flatnonzero(target_mask)
    predictions = np.clip(probe.predict(X[feature_rows[target_mask]]), 0.0, 1.0)
    return {
        "target_model": target_model,
        "row_idx": row_idx,
        "predictions": predictions,
        "probe": probe,
        "best_alpha": best_alpha,
        "best_val_auroc": float(best["val_auroc"]),
        "records": records,
        "reused_probe": reused_probe,
        "reuse_path": reuse_path,
    }


def fit_convex_weights(predictions: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, bool]:
    """Fit non-negative, sum-to-one weights by validation MSE."""
    n_layers = predictions.shape[1]
    initial = np.full(n_layers, 1.0 / n_layers)
    result = minimize(
        lambda weights: float(np.mean((predictions @ weights - y) ** 2)),
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_layers,
        constraints={"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0)},
        options={"maxiter": 1_000, "ftol": 1e-12},
    )
    if not result.success or not np.isfinite(result.x).all():
        return initial, False
    weights = np.maximum(result.x, 0.0)
    return weights / weights.sum(), True


def align_features(candidates: pd.DataFrame, item_index: pd.DataFrame) -> np.ndarray:
    keyed = item_index[ITEM_COLS + ["_feature_row"]]
    aligned = candidates[ITEM_COLS].merge(keyed, on=ITEM_COLS, how="left", validate="many_to_one")
    if aligned["_feature_row"].isna().any():
        raise ValueError(f"Missing activations for {int(aligned['_feature_row'].isna().sum())} candidate rows")
    return aligned["_feature_row"].to_numpy(dtype=int)


def split_metric_rows(
    candidates: pd.DataFrame,
    predictions: np.ndarray,
    target_model: str,
    variant: str,
) -> list[dict[str, Any]]:
    rows = []
    target_mask = candidates["model"].astype(str).eq(target_model).to_numpy()
    for split, split_rows in candidates.loc[target_mask].groupby("split", sort=False):
        idx = split_rows.index.to_numpy(dtype=int)
        y = split_rows["true_success"].to_numpy(dtype=float)
        score = predictions[idx]
        rows.append(
            {
                "variant": variant,
                "target_model": target_model,
                "split": split,
                "n": len(idx),
                "mean_true_success": float(np.mean(y)),
                "mean_pred_success": float(np.mean(score)),
                "rmse": rmse(y, score),
                "auroc": safe_auroc(y, score),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--activation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--encoder-model", default="Qwen3-4B")
    parser.add_argument("--layers", default=",".join(str(layer) for layer in DEFAULT_LAYERS))
    parser.add_argument(
        "--baseline-layer",
        type=int,
        default=None,
        help="Layer used by the single-layer baseline; defaults to the deepest ensemble layer.",
    )
    parser.add_argument("--target-models", default=None)
    parser.add_argument("--train-languages", default=None)
    parser.add_argument(
        "--reuse-probes-from",
        type=Path,
        default=None,
        help="Compatible experiment directory containing probes/layer_NNN/*.joblib.",
    )
    parser.add_argument("--alpha-grid", default=",".join(str(alpha) for alpha in DEFAULT_ALPHA_GRID))
    parser.add_argument("--val-threshold", type=float, default=0.5)
    parser.add_argument("--cache-size", type=int, default=1)
    parser.add_argument("--target-jobs", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=args.resume)
    candidates = pd.read_csv(args.candidates).reset_index(drop=True)
    target_models = parse_csv(args.target_models) or sorted(candidates["model"].astype(str).unique())
    train_languages = parse_csv(args.train_languages)
    layers = [int(value) for value in args.layers.split(",") if value.strip()]
    baseline_layer = args.baseline_layer if args.baseline_layer is not None else layers[-1]
    alpha_grid = parse_float_grid(args.alpha_grid)
    if len(layers) < 2 or len(set(layers)) != len(layers):
        raise ValueError("--layers must contain at least two distinct layer indices")
    if baseline_layer not in layers:
        raise ValueError("--baseline-layer must be included in --layers")

    n_rows = len(candidates)
    layer_predictions = np.full((len(layers), n_rows), np.nan, dtype=np.float32)
    alpha_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    probe_root = args.output_dir / "probes"
    probe_root.mkdir(parents=True, exist_ok=args.resume)

    for layer_pos, layer in enumerate(layers):
        print(f"\nBuilding {args.encoder_model} layer {layer} features...", flush=True)
        store = EncoderLayerStore(
            args.activation_root,
            args.encoder_model,
            layer,
            cache_size=args.cache_size,
        )
        X, item_index = build_feature_matrix(candidates[ITEM_COLS], store)
        feature_rows = align_features(candidates, item_index)

        layer_probe_dir = probe_root / f"layer_{layer:03d}"
        layer_probe_dir.mkdir(parents=True, exist_ok=True)
        results = Parallel(n_jobs=args.target_jobs, prefer="threads")(
            delayed(fit_layer_target)(
                target_model,
                candidates,
                X,
                feature_rows,
                layer,
                alpha_grid,
                train_languages,
                args.reuse_probes_from,
                layer_probe_dir / f"{safe_path_part(target_model)}.joblib",
            )
            for target_model in target_models
        )
        for result in results:
            target_model = str(result["target_model"])
            best_alpha = float(result["best_alpha"])
            records = list(result["records"])
            reused_probe = bool(result["reused_probe"])
            reuse_path = result["reuse_path"]
            layer_predictions[layer_pos, result["row_idx"]] = result["predictions"]
            joblib.dump(
                {
                    "probe": result["probe"],
                    "encoder_model": args.encoder_model,
                    "encoder_layer": layer,
                    "target_model": target_model,
                    "alpha": best_alpha,
                    "alpha_grid": records,
                    "train_languages": train_languages,
                    "reused_from": str(reuse_path) if reused_probe else None,
                },
                layer_probe_dir / f"{safe_path_part(target_model)}.joblib",
            )
            alpha_rows.extend(records)
            print(
                f"  {target_model}: alpha={best_alpha:g} "
                f"val_auroc={float(result['best_val_auroc']):.4f}"
                f"{' (reused)' if reused_probe else ''}",
                flush=True,
            )

        del X, item_index, feature_rows, store
        gc.collect()

    if np.isnan(layer_predictions).any():
        raise ValueError("Some layer predictions were not populated")

    variant_predictions = {
        "single_last": layer_predictions[layers.index(baseline_layer)],
        "mean": np.mean(layer_predictions, axis=0),
        "median": np.median(layer_predictions, axis=0),
        "convex_val": np.full(n_rows, np.nan, dtype=np.float32),
    }
    weight_rows = []
    for target_model in target_models:
        target_mask = candidates["model"].astype(str).eq(target_model).to_numpy()
        weight_mask = target_mask & candidates["split"].eq("val").to_numpy()
        if train_languages is not None:
            weight_mask &= candidates["language"].astype(str).isin(train_languages).to_numpy()
        val_matrix = layer_predictions[:, weight_mask].T.astype(float)
        y_val = candidates.loc[weight_mask, "true_success"].to_numpy(dtype=float)
        weights, converged = fit_convex_weights(val_matrix, y_val)
        target_idx = np.flatnonzero(target_mask)
        variant_predictions["convex_val"][target_idx] = np.clip(
            layer_predictions[:, target_idx].T @ weights, 0.0, 1.0
        )
        for layer, weight in zip(layers, weights):
            weight_rows.append(
                {
                    "target_model": target_model,
                    "encoder_layer": layer,
                    "weight": float(weight),
                    "optimizer_converged": converged,
                }
            )

    for variant, predictions in variant_predictions.items():
        variant_dir = args.output_dir / "candidates" / variant
        variant_dir.mkdir(parents=True, exist_ok=args.resume)
        out = candidates.copy()
        out["pred_success_raw_pre_ensemble"] = out["pred_success_raw"].astype(float)
        out["pred_success_raw"] = np.asarray(predictions, dtype=float)
        out["ensemble_variant"] = variant
        out["ensemble_layers"] = ",".join(str(layer) for layer in layers)
        out["ensemble_n_layers"] = 1 if variant == "single_last" else len(layers)
        out["selected_layer"] = baseline_layer if variant == "single_last" else -1
        out["selected_layer_frac"] = baseline_layer / layers[-1] if variant == "single_last" else np.nan
        out["score_source"] = f"encoder_target_decoupled_layer_{variant}"
        out.to_csv(variant_dir / "router_candidates.csv", index=False)
        for target_model in target_models:
            metric_rows.extend(split_metric_rows(candidates, predictions, target_model, variant))

    pd.DataFrame(alpha_rows).to_csv(args.output_dir / "alpha_selection_metrics.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(args.output_dir / "convex_validation_weights.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(args.output_dir / "prediction_metrics.csv", index=False)
    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "method": "encoder_target_decoupling_layer_ensemble",
                "encoder_model": args.encoder_model,
                "layers": layers,
                "baseline_layer": baseline_layer,
                "variants": VARIANTS,
                "target_models": target_models,
                "train_languages": train_languages,
                "alpha_grid": alpha_grid,
                "source_candidates": str(args.candidates),
                "activation_root": str(args.activation_root),
                "reuse_probes_from": str(args.reuse_probes_from) if args.reuse_probes_from else None,
                "deployment_forward_passes": 1,
                "target_jobs": args.target_jobs,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    print(f"\nWrote layer-ensemble experiment to {args.output_dir}")


if __name__ == "__main__":
    main()
