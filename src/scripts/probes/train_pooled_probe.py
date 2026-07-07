"""Train the pooled multilingual probe (language-balanced 10% subsample).

The pooled probe is trained on 10% of each language's probe_train split, so it
sees the same total number of training examples (~2100) as a single-language
probe, drawn uniformly from all 10 languages (~210 per language). This is the
pooled probe used throughout the thesis.

The output directory mirrors the language-specific probe format so it can be
fed directly into the routing experiment via --best-layers-path.

Usage
-----
python -m src.scripts.probes.train_pooled_probe
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scipy.stats import ConstantInputWarning
from src.scripts.probes.train_language_probes import (
    build_coverage,
    default_alpha_grid_for_probe_type,
    fit_layer_probe,
    load_aligned_dataset,
    load_canonical_problem_ids,
    parse_csv_filter,
    read_success_table,
    select_best_layers,
    PROBE_TYPE_RIDGE,
    PROBE_TYPES,
)

LANGUAGES = [
    "Arabic", "Chinese", "English", "French", "Italian",
    "Russian", "Swahili", "Telugu", "Thai", "Turkish",
]
MODELS = [
    "Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B",
    "Qwen3.5-4B", "Qwen3.5-9B", "gpt-oss-20b_low", "gpt-oss-20b_high",
]
DEFAULT_SPLIT_FRACS = (0.7, 0.15, 0.15)


def stack_pool_subsampled(
    datasets: list[dict[str, Any]],
    layer: int,
    train_frac: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack datasets, subsampling probe_train to train_frac per language."""
    Xs, ys, sps = [], [], []
    for ds in datasets:
        layer_acts = ds["activations"].get(layer)
        if layer_acts is None:
            layer_acts = ds["activations"].get(str(layer))
        if layer_acts is None:
            raise KeyError(f"Layer {layer} missing for {ds['language']}/{ds['model']}")
        X = np.asarray(layer_acts[ds["row_idx"]], dtype=np.float32)
        y = np.asarray(ds["y"], dtype=np.float32)
        sp = np.asarray(ds["splits"])

        train_mask = sp == "probe_train"
        train_idx = np.where(train_mask)[0]
        n_keep = max(1, int(round(len(train_idx) * train_frac)))
        keep_train = rng.choice(train_idx, size=n_keep, replace=False)
        keep_train.sort()

        non_train_idx = np.where(~train_mask)[0]
        keep_idx = np.concatenate([keep_train, non_train_idx])
        keep_idx.sort()

        Xs.append(X[keep_idx])
        ys.append(y[keep_idx])
        sps.append(sp[keep_idx])

    return (
        np.concatenate(Xs, axis=0),
        np.concatenate(ys, axis=0),
        np.concatenate(sps, axis=0),
    )


def train_model(
    model: str,
    datasets: list[dict[str, Any]],
    bundle_path: Path,
    alpha_grid: list[float],
    train_frac: float,
    auroc_threshold: float,
    probe_type: str,
    random_seed: int,
    overwrite: bool,
) -> list[dict[str, Any]]:
    if bundle_path.exists() and not overwrite:
        print(f"  {model}: bundle exists, loading for CSV generation")
        existing = joblib.load(bundle_path)
        layers = sorted(existing["layers"].keys())
        n_layers = max(layers) + 1
        d_model = None
        layer_records = []
        for layer in layers:
            lp = existing["layers"][layer]
            record = lp.get("record") or {}
            if d_model is None and "record" in lp and lp["record"] and "d_model" in lp["record"]:
                d_model = lp["record"]["d_model"]
            base = {
                "layer": layer,
                "layer_frac": float(layer / (n_layers - 1)) if n_layers > 1 else 0.0,
                "n_layers": n_layers,
                "d_model": d_model or 0,
                "model": model,
                "probe_path": str(bundle_path),
                "pool_languages": ",".join(sorted(ds["language"] for ds in datasets)),
                **{k: v for k, v in record.items() if k not in ("model",)},
            }
            for ds in datasets:
                layer_records.append({"language": ds["language"], **base})
        return layer_records

    rng = np.random.default_rng(random_seed)
    layers = sorted(set(datasets[0]["layers"]))
    for ds in datasets[1:]:
        layers = sorted(set(layers) & set(ds["layers"]))

    n_layers = max(layers) + 1
    d_model = None
    layer_payloads: dict[int, dict] = {}
    layer_records: list[dict] = []

    # Count train examples at first layer for reporting
    X0, y0, sp0 = stack_pool_subsampled(datasets, layers[0], train_frac, np.random.default_rng(random_seed))
    n_train = int((sp0 == "probe_train").sum())
    n_val   = int((sp0 == "val").sum())
    n_test  = int((sp0 == "test").sum())
    print(f"  {model}: n_train={n_train}, n_val={n_val}, n_test={n_test} across {len(datasets)} languages")

    for layer in layers:
        rng_layer = np.random.default_rng(random_seed + layer)
        X, y, splits = stack_pool_subsampled(datasets, layer, train_frac, rng_layer)
        if d_model is None:
            d_model = X.shape[1]
        record, probe = fit_layer_probe(
            X, y, splits,
            alpha_grid=alpha_grid,
            auroc_threshold=auroc_threshold,
            probe_type=probe_type,
            selection_metric="nll",
        )
        layer_payloads[layer] = {"probe": probe, "record": record}
        base = {
            "layer": layer,
            "layer_frac": float(layer / (n_layers - 1)) if n_layers > 1 else 0.0,
            "n_layers": n_layers,
            "d_model": int(d_model),
            "model": model,
            "probe_path": str(bundle_path),
            "pool_languages": ",".join(sorted(ds["language"] for ds in datasets)),
            "n_probe_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
            **{k: v for k, v in record.items() if k not in ("model",)},
        }
        # emit one row per language so select_best_layers works per (language, model)
        for ds in datasets:
            layer_records.append({"language": ds["language"], **base})

    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"metadata": {"model": model, "pool_mode": "pooled_balanced"}, "layers": layer_payloads}, bundle_path)
    print(f"  Saved bundle → {bundle_path}")
    return layer_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activations-root", type=Path, default=Path("outputs/activations/MATH"))
    parser.add_argument("--answers-root", type=Path, default=Path("outputs/rollouts/MATH"))
    parser.add_argument("--success-dir", type=Path, default=Path("outputs/success_rates/MATH"))
    parser.add_argument("--canonical-csv-path", type=Path, default=Path("data/MATH_translated.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/probes/pooled"))
    parser.add_argument("--models", type=str, default=",".join(MODELS))
    parser.add_argument("--train-frac", type=float, default=0.10,
                        help="Fraction of each language's probe_train split to use.")
    parser.add_argument("--token-position", type=int, default=-1)
    parser.add_argument("--split-fracs", type=float, nargs=3, default=list(DEFAULT_SPLIT_FRACS))
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--auroc-threshold", type=float, default=0.5)
    parser.add_argument("--min-examples", type=int, default=10)
    parser.add_argument("--probe-type", type=str, default=PROBE_TYPE_RIDGE, choices=PROBE_TYPES)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=ConstantInputWarning)

    canonical_problem_ids = load_canonical_problem_ids(args.canonical_csv_path)
    split_fracs = tuple(args.split_fracs)
    alpha_grid = default_alpha_grid_for_probe_type(args.probe_type)

    all_layer_records: list[dict] = []

    models = sorted(parse_csv_filter(args.models) or set())
    if models:
        success_df = read_success_table(args.success_dir)
        coverage = build_coverage(
            success_df,
            activations_root=args.activations_root,
            answers_root=args.answers_root,
            token_position=args.token_position,
            language_filter=set(LANGUAGES),
            model_filter=set(models),
        )
        ready = coverage[coverage["status"] == "ready"].copy()

        for model in models:
            print(f"\n=== {model} ===")
            datasets = []
            for lang in LANGUAGES:
                row = ready[(ready["model"] == model) & (ready["language"] == lang)]
                if row.empty:
                    print(f"  {lang}: not ready, skipping")
                    continue
                ds, status = load_aligned_dataset(
                    row.iloc[0],
                    success_df,
                    split_fracs=split_fracs,
                    random_seed=args.random_seed,
                    min_examples=args.min_examples,
                    min_positives_per_split=0,
                    auroc_threshold=args.auroc_threshold,
                    canonical_problem_ids=canonical_problem_ids,
                    alignment_min_corr=0.2,
                    alignment_max_mismatch_frac=0.3,
                    cache_label_fallback_languages={"English"},
                )
                if ds is None:
                    print(f"  {lang}: {status}")
                    continue
                ds["language"] = lang
                datasets.append(ds)

            if not datasets:
                continue
            bundle_path = args.output_dir / "probes_ridge" / model / "_pooled.joblib"
            recs = train_model(model, datasets, bundle_path, alpha_grid, args.train_frac,
                               args.auroc_threshold, args.probe_type, args.random_seed, args.overwrite)
            all_layer_records.extend(recs)

    if not all_layer_records:
        print("No records produced.")
        return

    layer_df = pd.DataFrame(all_layer_records)
    best = select_best_layers(layer_df)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    layer_df.to_csv(args.output_dir / "per_language_layer_results.csv", index=False)
    best.to_csv(args.output_dir / "per_language_best_layers.csv", index=False)
    print(f"\nBest layers saved → {args.output_dir / 'per_language_best_layers.csv'}")
    print(best[["model","layer","val_auroc"]].to_string(index=False))


if __name__ == "__main__":
    main()
