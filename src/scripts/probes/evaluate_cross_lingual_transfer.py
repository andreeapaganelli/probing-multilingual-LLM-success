"""Cross-lingual probe transfer evaluation.

Two experiments:
  - monolingual_best_layer: apply each train-language probe at the layer that was
    selected as best during its own monolingual training.
  - fixed_layer: apply all probes at a single fixed layer per model (the rounded
    median of the monolingual best layers across all languages).

For each (train_lang, test_lang, model) triple the script:
  1. Loads the train-lang probe at the chosen layer.
  2. Loads the test-lang activations at the same layer.
  3. Reconstructs the test split using the same deterministic hash used during
     probe training (seed=42, base_problem_id, 70/15/15 split).
  4. Applies the probe and reports AUROC on the test split.

Memory strategy: each activation file is ~1.6 GB. For each test language the
file is loaded exactly once; all needed layers are extracted in a single pass
before the full cache is discarded.

Outputs (one CSV per experiment, written to --output-dir):
  cross_lingual_monolingual_best_layer.csv
  cross_lingual_fixed_layer.csv
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.calibration_metrics import safe_brier, safe_ece, safe_log_loss
from src.probes import BalancedLogisticProbe, BalancedRidgeProbe, LogisticProbe, ScaledRidgeProbe  # noqa: F401 — needed for joblib unpickling

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


# ── split helpers (must match train_language_probes.py exactly) ──

def _stable_unit_interval(value: str, seed: int) -> float:
    payload = f"{seed}:{value}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:16]
    return int(digest, 16) / float(16**16)


def _assign_splits(
    base_problem_ids: np.ndarray,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> np.ndarray:
    mapping: dict[str, str] = {}
    for pid in pd.Series(base_problem_ids).astype(str).drop_duplicates():
        u = _stable_unit_interval(pid, seed)
        if u < train_frac:
            mapping[pid] = "probe_train"
        elif u < train_frac + val_frac:
            mapping[pid] = "val"
        else:
            mapping[pid] = "test"
    return np.array([mapping[str(p)] for p in base_problem_ids])


def _safe_auroc(y: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> float:
    y_bin = (y > threshold).astype(int)
    if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
        return float("nan")
    return float(roc_auc_score(y_bin, scores))


def _regression_metrics(y: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict:
    """MSE, calibration bias, and probabilistic calibration metrics."""
    mse = float(np.mean((scores - y) ** 2))
    bias = float(np.mean(scores) - np.mean(y))
    return {
        "mse": mse,
        "bias": bias,
        "mean_pred": float(np.mean(scores)),
        "mean_true": float(np.mean(y)),
        "log_loss": safe_log_loss(y, scores, threshold=threshold),
        "brier": safe_brier(y, scores, threshold=threshold),
        "ece": safe_ece(y, scores, threshold=threshold),
    }


def _probe_type_suffix(probe_type: str) -> str:
    return "" if probe_type == PROBE_TYPE_RIDGE else f"_{probe_type}"


def _default_probes_root(probe_output_dir: Path, probe_type: str) -> Path:
    return probe_output_dir / f"probes{_probe_type_suffix(probe_type)}" if probe_type != PROBE_TYPE_RIDGE else probe_output_dir / "probes_ridge"


def _default_best_layers_csv(probe_output_dir: Path, probe_type: str) -> Path:
    return probe_output_dir / f"per_language_best_layers{_probe_type_suffix(probe_type)}.csv"


def _default_transfer_output_dir(probe_type: str) -> Path:
    return Path("outputs/transfer/MATH") / probe_type


# ── data loading ──

def load_test_data_multilayer(
    language: str,
    model: str,
    layers: set[int],
    success_df: pd.DataFrame,
    activations_root: Path,
    exclude_base_problem_ids: set[str] | None = None,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Load a single activation file and return test-split data for all requested layers.

    Returns a dict {layer: (X_test, y_test)}.  Missing or degenerate layers are
    omitted from the dict.  The full activation cache is freed after extraction.
    """
    act_path = activations_root / language / model / "activations_token_-1.joblib"
    if not act_path.exists():
        return {}

    cache = joblib.load(act_path)
    available_layers = set(cache["activations"].keys())

    # Align problem IDs with success labels once
    problem_ids = pd.Series(cache["problem_ids"], name="problem_id").astype(str)
    cache_df = problem_ids.reset_index().rename(columns={"index": "row_idx"})

    scores = success_df[
        (success_df["language"] == language) & (success_df["model"] == model)
    ][["problem_id", "base_problem_id", "success_rate"]].copy()
    scores["problem_id"] = scores["problem_id"].astype(str)
    scores = scores.dropna(subset=["success_rate"]).drop_duplicates("problem_id", keep="last")

    merged = cache_df.merge(scores, on="problem_id", how="left")
    keep = merged["success_rate"].notna()

    result: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    if keep.sum() == 0:
        del cache
        gc.collect()
        return result

    idx = merged.loc[keep, "row_idx"].to_numpy(dtype=int)
    y = merged.loc[keep, "success_rate"].to_numpy(dtype=np.float32)
    base_ids = merged.loc[keep, "base_problem_id"].astype(str).to_numpy()

    splits = _assign_splits(base_ids)
    test_mask = splits == "test"
    if exclude_base_problem_ids:
        test_mask = test_mask & ~pd.Series(base_ids).isin(exclude_base_problem_ids).to_numpy()
    idx_test = idx[test_mask]
    y_test = y[test_mask]

    if len(y_test) == 0:
        del cache
        gc.collect()
        return result

    for layer in layers:
        if layer not in available_layers:
            continue
        X_test = cache["activations"][layer][idx_test].astype(np.float32)
        result[layer] = (X_test, y_test)

    # Free the full cache (can be 1.6 GB+) before returning
    del cache
    gc.collect()
    return result


def load_probe(bundle_path: Path, layer: int):
    """Load a single probe object from a bundle at a specific layer."""
    bundle = joblib.load(bundle_path)
    if layer not in bundle["layers"]:
        return None
    probe = bundle["layers"][layer]["probe"]
    del bundle
    return probe


def load_all_probes(bundle_path: Path) -> dict[int, object]:
    """Load all probes from a bundle. Returns {layer: probe}."""
    bundle = joblib.load(bundle_path)
    probes = {layer: ld["probe"] for layer, ld in bundle["layers"].items()}
    del bundle
    return probes


def load_excluded_base_problem_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    if not path.exists():
        raise FileNotFoundError(f"Missing exclusion CSV: {path}")
    df = pd.read_csv(path, low_memory=False)
    candidate_columns = [
        "base_problem_id",
        "test_base_problem_id",
        "nearest_test_base_problem_id",
    ]
    column = next((col for col in candidate_columns if col in df.columns), None)
    if column is None:
        raise ValueError(
            f"Exclusion CSV must contain one of {candidate_columns}; found {list(df.columns)}"
        )
    return set(df[column].dropna().astype(str))


# ── per-layer-all experiment ──

def run_per_layer_all_experiment(
    models: list[str],
    languages: list[str],
    success_df: pd.DataFrame,
    activations_root: Path,
    probes_root: Path,
    exclude_base_problem_ids: set[str] | None = None,
) -> pd.DataFrame:
    """Compute AUROC for every (train_lang, test_lang, layer) triple.

    Same memory strategy as the oracle: load each activation file once,
    extract all layers, then evaluate all train probes against it.
    Output has one row per (model, train_lang, test_lang, layer).
    """
    rows = []

    for model in models:
        sample_bundle_path = probes_root / model / f"{languages[0]}.joblib"
        if not sample_bundle_path.exists():
            print(f"  Skipping {model} — no probe bundle found.")
            continue
        sample_bundle = joblib.load(sample_bundle_path)
        all_layers: set[int] = set(sample_bundle["layers"].keys())
        del sample_bundle
        print(f"\n=== {model}  ({len(all_layers)} layers) ===")

        print("    Pre-loading test data for all layers...")
        test_data: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
        for lang in languages:
            test_data[lang] = load_test_data_multilayer(
                lang, model, all_layers, success_df, activations_root, exclude_base_problem_ids
            )
            ok = sum(1 for v in test_data[lang].values() if v is not None)
            print(f"      {lang}: {ok}/{len(all_layers)} layers loaded")

        for train_lang in languages:
            bundle_path = probes_root / model / f"{train_lang}.joblib"
            if not bundle_path.exists():
                continue
            probes = load_all_probes(bundle_path)

            for layer, probe in sorted(probes.items()):
                for test_lang in languages:
                    layer_data = test_data[test_lang].get(layer)
                    if layer_data is None:
                        continue
                    X_test, y_test = layer_data
                    scores = np.clip(probe.predict(X_test), 0.0, 1.0)
                    auroc = _safe_auroc(y_test, scores)
                    reg = _regression_metrics(y_test, scores)
                    rows.append({
                        "experiment": "per_layer_all",
                        "model": model,
                        "train_lang": train_lang,
                        "test_lang": test_lang,
                        "layer": int(layer),
                        "auroc": auroc,
                        "n_test": int(len(y_test)),
                        "is_diagonal": train_lang == test_lang,
                        **reg,
                    })

        del test_data
        gc.collect()

    return pd.DataFrame(rows)


# ── oracle experiment ──

def run_oracle_experiment(
    models: list[str],
    languages: list[str],
    success_df: pd.DataFrame,
    activations_root: Path,
    probes_root: Path,
    exclude_base_problem_ids: set[str] | None = None,
) -> pd.DataFrame:
    """For each (train_lang, test_lang) pair find the layer maximising AUROC.

    All 29 layers are tried. This is an oracle upper bound — in practice you
    would not have access to the test labels to pick the layer.
    """
    rows = []

    for model in models:
        # Discover which layers exist by inspecting one bundle
        sample_bundle_path = probes_root / model / f"{languages[0]}.joblib"
        if not sample_bundle_path.exists():
            print(f"  Skipping {model} — no probe bundle found.")
            continue
        sample_bundle = joblib.load(sample_bundle_path)
        all_layers: set[int] = set(sample_bundle["layers"].keys())
        del sample_bundle
        print(f"\n=== {model}  ({len(all_layers)} layers: {sorted(all_layers)}) ===")

        # Load test data for all layers in one pass per language
        print("    Pre-loading test data for all layers...")
        test_data: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
        for lang in languages:
            test_data[lang] = load_test_data_multilayer(
                lang, model, all_layers, success_df, activations_root, exclude_base_problem_ids
            )
            ok = sum(1 for v in test_data[lang].values() if v is not None)
            print(f"      {lang}: {ok}/{len(all_layers)} layers loaded")

        # Evaluate every (train_lang, test_lang, layer) triple
        for train_lang in languages:
            bundle_path = probes_root / model / f"{train_lang}.joblib"
            if not bundle_path.exists():
                continue
            probes = load_all_probes(bundle_path)

            for test_lang in languages:
                best_auroc = float("nan")
                best_layer = None
                best_scores = None
                best_y = None
                for layer, probe in probes.items():
                    layer_data = test_data[test_lang].get(layer)
                    if layer_data is None:
                        continue
                    X_test, y_test = layer_data
                    scores = np.clip(probe.predict(X_test), 0.0, 1.0)
                    auroc = _safe_auroc(y_test, scores)
                    if np.isnan(best_auroc) or (not np.isnan(auroc) and auroc > best_auroc):
                        best_auroc = auroc
                        best_layer = layer
                        best_scores = scores
                        best_y = y_test

                reg = _regression_metrics(best_y, best_scores) if best_scores is not None else {"mse": float("nan"), "bias": float("nan"), "mean_pred": float("nan"), "mean_true": float("nan")}
                rows.append({
                    "experiment": "oracle_layer",
                    "model": model,
                    "train_lang": train_lang,
                    "test_lang": test_lang,
                    "layer": int(best_layer) if best_layer is not None else -1,
                    "auroc": best_auroc,
                    "n_test": int(len(best_y)) if best_y is not None else 0,
                    "is_diagonal": train_lang == test_lang,
                    **reg,
                })

        del test_data
        gc.collect()

    return pd.DataFrame(rows)


# ── main experiment runner ──

def run_experiment(
    experiment: str,
    models: list[str],
    languages: list[str],
    best_layers: pd.DataFrame,
    success_df: pd.DataFrame,
    activations_root: Path,
    probes_root: Path,
    exclude_base_problem_ids: set[str] | None = None,
) -> pd.DataFrame:
    """Run one experiment and return a results DataFrame."""

    fixed_layer_per_model = (
        best_layers.groupby("model")["layer"].median().round().astype(int).to_dict()
    )
    best_layer_lookup = best_layers.set_index(["language", "model"])["layer"].to_dict()

    rows = []

    for model in models:
        fixed_layer = fixed_layer_per_model.get(model)
        print(f"\n=== {model}  (fixed layer: {fixed_layer}) ===")

        # Determine all layers needed across all train languages for this model
        if experiment == "fixed_layer":
            needed_layers: set[int] = {fixed_layer}
        else:
            needed_layers = {
                best_layer_lookup[(lang, model)]
                for lang in languages
                if (lang, model) in best_layer_lookup
            }
        print(f"    Needed layers: {sorted(needed_layers)}")

        # Load test-split data for every test language (one file load per language)
        print("    Pre-loading test data...")
        test_data: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
        for lang in languages:
            test_data[lang] = load_test_data_multilayer(
                lang, model, needed_layers, success_df, activations_root, exclude_base_problem_ids
            )
            ok = sum(1 for v in test_data[lang].values() if v is not None)
            print(f"      {lang}: {ok}/{len(needed_layers)} layers loaded")

        # Evaluate each (train_lang, test_lang) pair
        for train_lang in languages:
            bundle_path = probes_root / model / f"{train_lang}.joblib"
            if not bundle_path.exists():
                continue

            probe_layer = (
                fixed_layer
                if experiment == "fixed_layer"
                else best_layer_lookup.get((train_lang, model))
            )
            if probe_layer is None:
                continue

            probe = load_probe(bundle_path, probe_layer)
            if probe is None:
                continue

            for test_lang in languages:
                layer_data = test_data[test_lang].get(probe_layer)
                if layer_data is None:
                    continue
                X_test, y_test = layer_data
                scores = np.clip(probe.predict(X_test), 0.0, 1.0)
                auroc = _safe_auroc(y_test, scores)
                reg = _regression_metrics(y_test, scores)
                rows.append({
                    "experiment": experiment,
                    "model": model,
                    "train_lang": train_lang,
                    "test_lang": test_lang,
                    "layer": int(probe_layer),
                    "auroc": auroc,
                    "n_test": int(len(y_test)),
                    "is_diagonal": train_lang == test_lang,
                    **reg,
                })

        # Free test data for this model before moving to the next
        del test_data
        gc.collect()

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-lingual probe transfer evaluation.")
    parser.add_argument(
        "--experiment",
        choices=["monolingual_best_layer", "fixed_layer", "both", "oracle_layer", "per_layer_all"],
        default="both",
    )
    parser.add_argument("--activations-root", type=Path, default=Path("outputs/activations/MATH"))
    parser.add_argument("--probe-type", type=str, default=PROBE_TYPE_RIDGE, choices=PROBE_TYPES)
    parser.add_argument("--probe-output-dir", type=Path, default=Path("outputs/probes/language_specific"))
    parser.add_argument("--probes-root", type=Path, default=None)
    parser.add_argument("--best-layers-csv", type=Path, default=None)
    parser.add_argument("--success-dir", type=Path, default=Path("outputs/success_rates/MATH"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--languages", type=str, default=None, help="Comma-separated language filter.")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model filter.")
    parser.add_argument(
        "--exclude-base-ids-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV of base_problem_id/test_base_problem_id values to remove "
            "from every reconstructed test split."
        ),
    )
    args = parser.parse_args()

    args.probes_root = args.probes_root or _default_probes_root(args.probe_output_dir, args.probe_type)
    args.best_layers_csv = args.best_layers_csv or _default_best_layers_csv(args.probe_output_dir, args.probe_type)
    args.output_dir = args.output_dir or _default_transfer_output_dir(args.probe_type)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    best_layers = pd.read_csv(args.best_layers_csv)
    from src.data import load_success_rates
    success_df = load_success_rates(args.success_dir)

    languages = (
        [l.strip() for l in args.languages.split(",") if l.strip()]
        if args.languages
        else sorted(best_layers["language"].unique().tolist())
    )
    models = (
        [m.strip() for m in args.models.split(",") if m.strip()]
        if args.models
        else sorted(best_layers["model"].unique().tolist())
    )

    print(f"Languages ({len(languages)}): {languages}")
    print(f"Models ({len(models)}): {models}")
    exclude_base_problem_ids = load_excluded_base_problem_ids(args.exclude_base_ids_csv)
    if exclude_base_problem_ids:
        print(f"Excluding {len(exclude_base_problem_ids)} base_problem_id values from test splits.")

    if args.experiment == "per_layer_all":
        print(f"\n{'='*60}\nRunning experiment: per_layer_all\n{'='*60}")
        results = run_per_layer_all_experiment(
            models, languages, success_df, args.activations_root, args.probes_root,
            exclude_base_problem_ids,
        )
        out_path = args.output_dir / "cross_lingual_per_layer_all.csv"
        if out_path.exists() and not results.empty:
            existing = pd.read_csv(out_path)
            existing = existing[~existing["model"].isin(results["model"].unique())]
            results = pd.concat([existing, results], ignore_index=True)
        results.to_csv(out_path, index=False)
        print(f"\nSaved {len(results)} rows → {out_path}")
        return

    if args.experiment == "oracle_layer":
        print(f"\n{'='*60}\nRunning experiment: oracle_layer\n{'='*60}")
        results = run_oracle_experiment(
            models, languages, success_df, args.activations_root, args.probes_root,
            exclude_base_problem_ids,
        )
        out_path = args.output_dir / "cross_lingual_oracle_layer.csv"
        if out_path.exists() and not results.empty:
            existing = pd.read_csv(out_path)
            existing = existing[~existing["model"].isin(results["model"].unique())]
            results = pd.concat([existing, results], ignore_index=True)
        results.to_csv(out_path, index=False)
        print(f"\nSaved {len(results)} rows → {out_path}")
        if not results.empty:
            diag = results[results["is_diagonal"]]["auroc"].mean()
            off_diag = results[~results["is_diagonal"]]["auroc"].mean()
            print(f"  Diagonal mean AUROC:     {diag:.3f}")
            print(f"  Off-diagonal mean AUROC: {off_diag:.3f}")
            print(f"  Transfer gap:            {diag - off_diag:.3f}")
            print("\n  Per-model summary:")
            for model, grp in results.groupby("model"):
                d = grp[grp["is_diagonal"]]["auroc"].mean()
                o = grp[~grp["is_diagonal"]]["auroc"].mean()
                print(f"    {model:35s}  diag={d:.3f}  off-diag={o:.3f}  gap={d-o:.3f}")
        return

    experiments = (
        ["monolingual_best_layer", "fixed_layer"]
        if args.experiment == "both"
        else [args.experiment]
    )

    for exp in experiments:
        print(f"\n{'='*60}")
        print(f"Running experiment: {exp}")
        print(f"{'='*60}")
        results = run_experiment(
            exp, models, languages, best_layers, success_df,
            args.activations_root, args.probes_root, exclude_base_problem_ids,
        )
        out_path = args.output_dir / f"cross_lingual_{exp}.csv"
        # Merge with any existing results so partial runs don't erase other models.
        if out_path.exists() and not results.empty:
            existing = pd.read_csv(out_path)
            existing = existing[~existing["model"].isin(results["model"].unique())]
            results = pd.concat([existing, results], ignore_index=True)
        results.to_csv(out_path, index=False)
        print(f"\nSaved {len(results)} rows → {out_path}")

        if not results.empty:
            diag = results[results["is_diagonal"]]["auroc"].mean()
            off_diag = results[~results["is_diagonal"]]["auroc"].mean()
            print(f"  Diagonal (monolingual) mean AUROC:  {diag:.3f}")
            print(f"  Off-diagonal (transfer) mean AUROC: {off_diag:.3f}")
            print(f"  Transfer gap:                        {diag - off_diag:.3f}")

            print("\n  Per-model summary:")
            for model, grp in results.groupby("model"):
                d = grp[grp["is_diagonal"]]["auroc"].mean()
                o = grp[~grp["is_diagonal"]]["auroc"].mean()
                print(f"    {model:35s}  diag={d:.3f}  off-diag={o:.3f}  gap={d-o:.3f}")


if __name__ == "__main__":
    main()
