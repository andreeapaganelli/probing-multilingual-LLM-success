"""Build merged best-layers CSVs used as probe configs by the routing experiments.

Produces three configs in --output-dir:
  per_language.csv      one row per (model, language): the language-specific probe
                        at its own best layer.
  pooled.csv            one row per (model, language): the pooled multilingual probe
                        (language-balanced 10% subsample) applied to every language.
  english_transfer.csv  one row per (model, language): the English-trained probe
                        applied to every language.

Usage
-----
python -m src.scripts.routing.build_routing_probe_configs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LANGUAGES = [
    "Arabic", "Chinese", "English", "French", "Italian",
    "Russian", "Swahili", "Telugu", "Thai", "Turkish",
]


def load_per_language(best_layers_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(best_layers_csv)
    if "probe_type" in df.columns:
        df["probe_type"] = "ridge_scaled"
    return df.drop_duplicates(subset=["model", "language"], keep="last")


def load_pooled(best_layers_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(best_layers_csv)
    if "probe_type" in df.columns:
        df["probe_type"] = "ridge_scaled"
    base = df.drop_duplicates(subset=["model"], keep="last")
    rows = []
    for _, row in base.iterrows():
        for language in LANGUAGES:
            out = row.copy()
            out["language"] = language
            rows.append(out)
    return pd.DataFrame(rows)


def build_english_transfer(per_language: pd.DataFrame) -> pd.DataFrame:
    english_rows = per_language[per_language["language"] == "English"].copy()
    english_by_model = english_rows.set_index("model")
    rows = []
    for _, row in per_language.iterrows():
        model = row["model"]
        if model not in english_by_model.index:
            raise KeyError(f"No English probe row for model {model}")
        eng = english_by_model.loc[model]
        out = row.copy()
        out["probe_path"] = eng["probe_path"]
        out["layer"] = eng["layer"]
        if "layer_frac" in eng.index:
            out["layer_frac"] = eng["layer_frac"]
        rows.append(out)
    return pd.DataFrame(rows)


def load_fixed_layers(path: Path | None) -> dict[str, int]:
    if path is None:
        return {}
    df = pd.read_csv(path)
    required = {"model", "layer"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Fixed layer CSV is missing columns: {sorted(missing)}")
    return {str(row.model): int(row.layer) for row in df.itertuples(index=False)}


def _bundle_record(probe_path: str, layer: int) -> dict:
    bundle_path = REPO_ROOT / probe_path
    bundle = joblib.load(bundle_path)
    layer_payload = bundle["layers"].get(layer)
    if layer_payload is None:
        layer_payload = bundle["layers"].get(str(layer))
    if layer_payload is None:
        raise KeyError(f"{bundle_path} has no layer {layer}")
    return dict(layer_payload.get("record") or {})


def apply_fixed_layers(df: pd.DataFrame, fixed_layers: dict[str, int]) -> pd.DataFrame:
    if not fixed_layers:
        return df
    out = df.copy()
    for idx, row in out.iterrows():
        model = str(row["model"])
        if model not in fixed_layers:
            continue
        layer = fixed_layers[model]
        rec = _bundle_record(str(row["probe_path"]), layer)
        out.at[idx, "layer"] = layer
        for col, value in rec.items():
            if col in {"probe_path", "language"}:
                continue
            if col in out.columns:
                out.at[idx, col] = value
        if "layer_frac" in out.columns:
            if "layer_frac" in rec:
                out.at[idx, "layer_frac"] = rec["layer_frac"]
            elif "n_layers" in rec and int(rec["n_layers"]) > 1:
                out.at[idx, "layer_frac"] = layer / (int(rec["n_layers"]) - 1)
        if "selection_method" in out.columns:
            out.at[idx, "selection_method"] = "shared_fixed_layer"
        if "best_layer_selection_metric" in out.columns:
            out.at[idx, "best_layer_selection_metric"] = "shared"
        if "best_layer_policy" in out.columns:
            out.at[idx, "best_layer_policy"] = "fixed"
        if "best_layer_tol" in out.columns:
            out.at[idx, "best_layer_tol"] = 0.0
        for col in ("best_layer_band_size",):
            if col in out.columns:
                out.at[idx, col] = 1
        for col in ("best_layer_band_min_layer", "best_layer_band_max_layer", "best_layer_band_median_layer"):
            if col in out.columns:
                out.at[idx, col] = layer
        for col in ("best_layer_band_min_frac", "best_layer_band_max_frac", "best_layer_band_median_frac"):
            if col in out.columns and "layer_frac" in out.columns:
                out.at[idx, col] = out.at[idx, "layer_frac"]
        if "best_layer_metric_max" in out.columns:
            out.at[idx, "best_layer_metric_max"] = rec.get("val_log_loss", pd.NA)
        if "best_layer_metric_min" in out.columns:
            out.at[idx, "best_layer_metric_min"] = rec.get("val_log_loss", pd.NA)
    return out


def verify_paths(df: pd.DataFrame, label: str) -> None:
    missing = []
    for _, row in df.iterrows():
        path = REPO_ROOT / row["probe_path"]
        if not path.exists():
            missing.append((row["model"], row["language"], str(row["probe_path"])))
    if missing:
        msg = f"{label}: {len(missing)} missing probe bundles (first 5):\n"
        msg += "\n".join(f"  {m}" for m in missing[:5])
        raise FileNotFoundError(msg)
    print(f"{label}: {len(df)} rows, all probe_path exist ({df['model'].nunique()} models)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--per-language-best-layers",
        type=Path,
        default=Path("outputs/probes/language_specific/per_language_best_layers.csv"),
        help="Best-layers CSV produced by train_language_probes.py.",
    )
    parser.add_argument(
        "--pooled-best-layers",
        type=Path,
        default=Path("outputs/probes/pooled/per_language_best_layers.csv"),
        help="Best-layers CSV produced by train_pooled_probe.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/routing/probe_configs"),
    )
    parser.add_argument(
        "--fixed-layers-csv",
        type=Path,
        default=None,
        help="Optional CSV with model,layer columns. When set, pooled and English-transfer "
             "configs use these shared per-model layers.",
    )
    args = parser.parse_args()

    def abs_path(p: Path) -> Path:
        return p if p.is_absolute() else REPO_ROOT / p

    out_dir = abs_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_language = load_per_language(abs_path(args.per_language_best_layers))
    pooled = load_pooled(abs_path(args.pooled_best_layers))
    english = build_english_transfer(per_language)
    fixed_layers = load_fixed_layers(
        args.fixed_layers_csv if args.fixed_layers_csv is None or args.fixed_layers_csv.is_absolute()
        else REPO_ROOT / args.fixed_layers_csv
    )
    pooled = apply_fixed_layers(pooled, fixed_layers)
    english = apply_fixed_layers(english, fixed_layers)

    verify_paths(per_language, "per_language")
    verify_paths(pooled, "pooled")
    verify_paths(english, "english_transfer")

    per_language.to_csv(out_dir / "per_language.csv", index=False)
    pooled.to_csv(out_dir / "pooled.csv", index=False)
    english.to_csv(out_dir / "english_transfer.csv", index=False)
    print(f"Wrote configs to {out_dir}")


if __name__ == "__main__":
    main()
