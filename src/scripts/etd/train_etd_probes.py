"""Train encoder-target decoupled probes from one encoder's activations.

The setup follows the LLM Router / Encoder-Target Decoupling idea: a single
encoder model reads the prompt and its prefill activations are used to estimate
whether each target model will solve the same problem. This script trains one
linear probe per target model and writes a router-candidate table with
``pred_success_raw`` replaced by those decoupled predictions.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.linalg import LinAlgWarning
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

warnings.filterwarnings("ignore", category=LinAlgWarning)

ITEM_COLS = ["language", "problem_id", "base_problem_id", "split"]
DEFAULT_ALPHA_GRID = [1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0, 1_000_000.0]


def safe_path_part(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value).strip("._") or "unnamed"


def parse_csv(value: str | None) -> list[str] | None:
    if value is None or not value.strip():
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_float_grid(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def safe_auroc(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> float:
    y_bin = (np.asarray(y_true) >= threshold).astype(int)
    if len(y_bin) == 0 or y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
        return float("nan")
    return float(roc_auc_score(y_bin, scores))


def rmse(y_true: np.ndarray, scores: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, scores)))


def load_activation_metadata(path: Path) -> dict[str, Any]:
    metadata_path = path / "activations_token_-1_metadata.joblib"
    if metadata_path.exists():
        return joblib.load(metadata_path)
    payload = joblib.load(path / "activations_token_-1.joblib")
    return {k: payload[k] for k in payload.keys() if k != "activations"}


def discover_layers(activation_dir: Path) -> list[int]:
    sidecar_layers = []
    for path in sorted(activation_dir.glob("layer_*.npy")):
        try:
            sidecar_layers.append(int(path.stem.split("_", 1)[1]))
        except ValueError:
            pass
    payload_path = activation_dir / "activations_token_-1.joblib"
    if payload_path.exists():
        payload = joblib.load(payload_path)
        if isinstance(payload, dict) and "activations" in payload:
            return sorted(int(layer) for layer in payload["activations"].keys())
    return sorted(sidecar_layers)


def resolve_layer(spec: str, sample_dir: Path) -> tuple[int, int]:
    layers = discover_layers(sample_dir)
    if not layers:
        raise ValueError(f"No activation layers found under {sample_dir}")
    if spec == "last":
        return layers[-1], layers[-1]
    layer = int(spec)
    if layer not in layers and not (sample_dir / f"layer_{layer}.npy").exists():
        raise ValueError(f"Layer {layer} not found under {sample_dir}; available layers: {layers[:5]}...{layers[-5:]}")
    return layer, layers[-1]


class EncoderLayerStore:
    def __init__(self, activation_root: Path, encoder_model: str, layer: int, cache_size: int = 2):
        self.activation_root = activation_root
        self.encoder_model = encoder_model
        self.layer = int(layer)
        self.cache_size = int(cache_size)
        self._cache: OrderedDict[str, tuple[np.ndarray, list[str]]] = OrderedDict()

    def _activation_dir(self, language: str) -> Path:
        return self.activation_root / language / self.encoder_model

    def load_language(self, language: str) -> tuple[np.ndarray, list[str]]:
        if language in self._cache:
            value = self._cache.pop(language)
            self._cache[language] = value
            return value

        act_dir = self._activation_dir(language)
        if not act_dir.exists():
            raise FileNotFoundError(f"Missing encoder activation directory: {act_dir}")
        metadata = load_activation_metadata(act_dir)
        problem_ids = [str(pid) for pid in metadata["problem_ids"]]
        sidecar = act_dir / f"layer_{self.layer}.npy"
        if sidecar.exists():
            X = np.load(sidecar, mmap_mode="r")
        else:
            payload = joblib.load(act_dir / "activations_token_-1.joblib")
            X = np.asarray(payload["activations"][self.layer], dtype=np.float32)
        if len(problem_ids) != X.shape[0]:
            raise ValueError(f"Activation/problem_id length mismatch for {act_dir}: {X.shape[0]} vs {len(problem_ids)}")

        value = (X, problem_ids)
        self._cache[language] = value
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return value


def build_feature_matrix(items: pd.DataFrame, store: EncoderLayerStore) -> tuple[np.ndarray, pd.DataFrame]:
    items = items[ITEM_COLS].drop_duplicates().reset_index(drop=True)
    features: list[np.ndarray] = []
    kept_parts: list[pd.DataFrame] = []
    for language, group in items.groupby("language", sort=False):
        X_lang, pids = store.load_language(str(language))
        pid_to_idx = {pid: idx for idx, pid in enumerate(pids)}
        idx = np.array([pid_to_idx.get(str(pid), -1) for pid in group["problem_id"]], dtype=int)
        ok = idx >= 0
        if not ok.all():
            missing = int((~ok).sum())
            print(f"warning: skipped {missing} items with no {store.encoder_model} activations for {language}")
        if ok.any():
            features.append(np.asarray(X_lang[idx[ok]], dtype=np.float32))
            kept_parts.append(group.loc[ok].copy())
    if not features:
        raise ValueError("No candidate rows could be matched to encoder activations.")
    kept = pd.concat(kept_parts, ignore_index=True)
    X = np.vstack(features)
    kept["_feature_row"] = np.arange(len(kept), dtype=int)
    return X, kept


def train_probe(X: np.ndarray, y: np.ndarray, alpha: float) -> Pipeline:
    probe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=float(alpha))),
        ]
    )
    probe.fit(X, y)
    return probe


def split_metrics(probe: Pipeline, X: np.ndarray, rows: pd.DataFrame, target_model: str) -> list[dict[str, Any]]:
    out = []
    for split, split_rows in rows.groupby("split", sort=False):
        idx = split_rows["_feature_row"].to_numpy(dtype=int)
        y = split_rows["true_success"].to_numpy(dtype=float)
        scores = np.clip(probe.predict(X[idx]), 0.0, 1.0)
        out.append(
            {
                "target_model": target_model,
                "split": split,
                "n": int(len(split_rows)),
                "mean_true_success": float(np.mean(y)),
                "mean_pred_success": float(np.mean(scores)),
                "rmse": rmse(y, scores),
                "auroc": safe_auroc(y, scores),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True, help="Base router_candidates.csv with target labels/costs.")
    parser.add_argument("--activation-root", type=Path, required=True, help="Root like outputs/activations/MATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--encoder-model", default="gpt-oss-20b_high")
    parser.add_argument("--encoder-layer", default="last", help="'last' or an integer layer index.")
    parser.add_argument("--target-models", default=None, help="Comma-separated target model subset; defaults to all models.")
    parser.add_argument(
        "--extend-probes-from",
        type=Path,
        default=None,
        help="Reuse an existing probes directory and fit only alpha values missing from its saved grid.",
    )
    parser.add_argument(
        "--train-languages",
        default=None,
        help="Comma-separated languages used to train/select probes. Defaults to pooled all-language training.",
    )
    parser.add_argument("--alpha-grid", default=",".join(str(x) for x in DEFAULT_ALPHA_GRID))
    parser.add_argument("--val-threshold", type=float, default=0.5)
    parser.add_argument("--cache-size", type=int, default=2)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=False)
    candidates = pd.read_csv(args.candidates)
    target_models = parse_csv(args.target_models) or sorted(candidates["model"].astype(str).unique())
    train_languages = parse_csv(args.train_languages)
    missing = sorted(set(target_models) - set(candidates["model"].astype(str).unique()))
    if missing:
        raise ValueError(f"Requested target model(s) not found in candidates: {missing}")
    candidates = candidates[candidates["model"].astype(str).isin(target_models)].copy()

    sample_language = str(candidates["language"].iloc[0])
    encoder_layer, max_layer = resolve_layer(
        args.encoder_layer,
        args.activation_root / sample_language / args.encoder_model,
    )
    store = EncoderLayerStore(args.activation_root, args.encoder_model, encoder_layer, cache_size=args.cache_size)

    print(f"Building feature matrix from {args.encoder_model} layer {encoder_layer}...")
    X, item_index = build_feature_matrix(candidates[ITEM_COLS], store)
    keyed = item_index[ITEM_COLS + ["_feature_row"]]
    work = candidates.merge(keyed, on=ITEM_COLS, how="inner", validate="many_to_one")
    if work.empty:
        raise ValueError("No candidate rows remained after joining encoder features.")

    alpha_grid = parse_float_grid(args.alpha_grid)
    probe_dir = args.output_dir / "probes"
    probe_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, Any]] = []
    prediction_parts: list[pd.DataFrame] = []

    for target_model in target_models:
        rows = work[work["model"].astype(str).eq(target_model)].copy()
        fit_rows = rows
        if train_languages is not None:
            fit_rows = fit_rows[fit_rows["language"].astype(str).isin(train_languages)]
        train_rows = fit_rows[fit_rows["split"].eq("probe_train")]
        val_rows = fit_rows[fit_rows["split"].eq("val")]
        if train_rows.empty or val_rows.empty:
            print(f"warning: skipped {target_model}: missing probe_train or val rows")
            continue
        X_train = X[train_rows["_feature_row"].to_numpy(dtype=int)]
        y_train = train_rows["true_success"].to_numpy(dtype=float)
        X_val = X[val_rows["_feature_row"].to_numpy(dtype=int)]
        y_val = val_rows["true_success"].to_numpy(dtype=float)

        alpha_records = []
        probes: dict[float, Pipeline] = {}
        existing_probe_path = None
        if args.extend_probes_from is not None:
            existing_probe_path = args.extend_probes_from / f"{safe_path_part(target_model)}.joblib"
            if existing_probe_path.exists():
                existing = joblib.load(existing_probe_path)
                existing_alpha = float(existing["alpha"])
                probes[existing_alpha] = existing["probe"]
                alpha_records.extend(dict(record) for record in existing["alpha_grid"])
        evaluated_alphas = {float(record["alpha"]) for record in alpha_records}
        for alpha in alpha_grid:
            if float(alpha) in evaluated_alphas:
                continue
            probe = train_probe(X_train, y_train, alpha)
            probes[alpha] = probe
            val_pred = np.clip(probe.predict(X_val), 0.0, 1.0)
            alpha_records.append(
                {
                    "alpha": alpha,
                    "val_auroc": safe_auroc(y_val, val_pred, threshold=args.val_threshold),
                    "val_rmse": rmse(y_val, val_pred),
                }
            )
        alpha_df = pd.DataFrame(alpha_records)
        alpha_df["_sort_auroc"] = alpha_df["val_auroc"].fillna(-np.inf)
        best = alpha_df.sort_values(["_sort_auroc", "val_rmse", "alpha"], ascending=[False, True, True]).iloc[0]
        best_alpha = float(best["alpha"])
        if best_alpha not in probes:
            raise ValueError(
                f"Selected alpha {best_alpha:g} for {target_model} is not available as a fitted probe; "
                "the existing grid winner may be inconsistent with its saved records."
            )
        probe = probes[best_alpha]

        target_probe_path = probe_dir / f"{safe_path_part(target_model)}.joblib"
        joblib.dump(
            {
                "probe": probe,
                "target_model": target_model,
                "encoder_model": args.encoder_model,
                "encoder_layer": int(encoder_layer),
                "encoder_layer_frac": float(encoder_layer / max_layer) if max_layer else np.nan,
                "alpha": best_alpha,
                "alpha_grid": alpha_records,
                "source_candidates": str(args.candidates),
                "train_languages": train_languages,
                "extended_from": str(existing_probe_path) if existing_probe_path and existing_probe_path.exists() else None,
            },
            target_probe_path,
        )

        metric_rows.extend(
            {
                **row,
                "encoder_model": args.encoder_model,
                "encoder_layer": int(encoder_layer),
                "encoder_layer_frac": float(encoder_layer / max_layer) if max_layer else np.nan,
                "alpha": best_alpha,
                "probe_path": str(target_probe_path),
            }
            for row in split_metrics(probe, X, rows, target_model)
        )

        pred = rows.copy()
        pred["pred_success_raw_original"] = pred["pred_success_raw"].astype(float)
        pred["pred_success_raw"] = np.clip(probe.predict(X[pred["_feature_row"].to_numpy(dtype=int)]), 0.0, 1.0)
        pred["selected_layer"] = int(encoder_layer)
        pred["selected_layer_frac"] = float(encoder_layer / max_layer) if max_layer else np.nan
        pred["selected_val_auroc"] = float(best["val_auroc"])
        pred["probe_path"] = str(target_probe_path)
        pred["label_source"] = "encoder_target_decoupling"
        pred["encoder_model"] = args.encoder_model
        pred["target_model"] = target_model
        pred["score_source"] = "encoder_target_decoupled_raw"
        prediction_parts.append(pred.drop(columns=["_feature_row"]))
        print(f"{target_model}: alpha={best_alpha:g} val_auroc={float(best['val_auroc']):.4f} val_rmse={float(best['val_rmse']):.4f}")

    if not prediction_parts:
        raise ValueError("No target probes were trained.")

    out_candidates = pd.concat(prediction_parts, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    out_candidates.to_csv(args.output_dir / "router_candidates.csv", index=False)
    metrics.to_csv(args.output_dir / "target_probe_metrics.csv", index=False)
    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "method": "encoder_target_decoupling",
                "encoder_model": args.encoder_model,
                "encoder_layer": int(encoder_layer),
                "encoder_layer_frac": float(encoder_layer / max_layer) if max_layer else None,
                "target_models": target_models,
                "source_candidates": str(args.candidates),
                "activation_root": str(args.activation_root),
                "alpha_grid": alpha_grid,
                "train_languages": train_languages,
                "extend_probes_from": str(args.extend_probes_from) if args.extend_probes_from else None,
                "score_column": "pred_success_raw",
            },
            f,
            indent=2,
            sort_keys=True,
        )
    print(f"Wrote {args.output_dir / 'router_candidates.csv'}")
    print(f"Wrote {args.output_dir / 'target_probe_metrics.csv'}")


if __name__ == "__main__":
    main()
