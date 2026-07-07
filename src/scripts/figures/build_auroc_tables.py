"""Build the layer-wise AUROC summary tables from the cross-lingual transfer CSV.

Produces the thesis Tables 4.2-4.4 and B.1 (percentages are derived from the
rounded 3-decimal AUROC values, matching the thesis):

  peak_same_language_auroc_by_language.csv  peak same-language AUROC per (model, language)
  peak_same_language_auroc_by_model.csv     peak of the mean same-language trajectory per model
  peak_transfer_auroc_by_model.csv          peak mean cross-lingual transfer AUROC per model
  same_language_vs_transfer_auroc.csv       same-language vs. transfer AUROC at the
                                            transfer-optimal layer

Usage:
    python -m src.scripts.figures.build_auroc_tables
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

MODEL_ORDER = [
    "Qwen3-0.6B",
    "Qwen3-1.7B",
    "Qwen3-4B",
    "Qwen3-8B",
    "Qwen3.5-4B",
    "Qwen3.5-9B",
    "gpt-oss-20b_low",
    "gpt-oss-20b_high",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transfer-csv",
        type=Path,
        default=Path("outputs/transfer/MATH/ridge_scaled/cross_lingual_per_layer_all.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/figures/auroc_trajectories"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.transfer_csv)
    order = [m for m in MODEL_ORDER if m in set(df["model"])]
    same = df[df["is_diagonal"]]
    off = df[~df["is_diagonal"]]

    # Table B.1 — peak same-language AUROC per (model, language)
    rows = []
    for (model, language), group in same.groupby(["model", "train_lang"]):
        best = group.loc[group["auroc"].idxmax()]
        rows.append(
            {
                "model": model,
                "language": language,
                "peak_layer": int(best["layer"]),
                "peak_auroc": round(float(best["auroc"]), 3),
                "n_test": int(best["n_test"]),
            }
        )
    by_language = pd.DataFrame(rows)
    by_language["model"] = pd.Categorical(by_language["model"], order)
    by_language = by_language.sort_values(["language", "model"]).reset_index(drop=True)
    by_language.to_csv(args.output_dir / "peak_same_language_auroc_by_language.csv", index=False)

    # Table 4.2 — peak of the mean same-language trajectory per model
    rows = []
    for model in order:
        group = same[same["model"].eq(model)]
        trajectory = group.groupby("layer")["auroc"].mean()
        layer = int(trajectory.idxmax())
        max_layer = int(group["layer"].max())
        rows.append(
            {
                "model": model,
                "mean_same_language_auroc": round(float(trajectory.max()), 3),
                "layer": layer,
                "max_layer": max_layer,
                "depth_pct": round(100.0 * layer / max_layer, 1),
                "n_languages": group["train_lang"].nunique(),
                "n_test_per_language": int(group["n_test"].iloc[0]),
            }
        )
    pd.DataFrame(rows).to_csv(
        args.output_dir / "peak_same_language_auroc_by_model.csv", index=False
    )

    # Table 4.3 — peak mean cross-lingual transfer AUROC per model
    rows = []
    for model in order:
        group = off[off["model"].eq(model)]
        trajectory = group.groupby("layer")["auroc"].mean()
        layer = int(trajectory.idxmax())
        max_layer = int(group["layer"].max())
        drop = round(float(trajectory.max()) - float(trajectory.loc[max_layer]), 3)
        peak = round(float(trajectory.max()), 3)
        final = round(float(trajectory.loc[max_layer]), 3)
        rows.append(
            {
                "Model": model,
                "Peak transfer AUROC": peak,
                "Peak layer": layer,
                "Max layer": max_layer,
                "% depth": round(100.0 * layer / max_layer, 1),
                "Final-layer transfer AUROC": final,
                "Drop from peak to final": drop,
                "Drop %": round(100.0 * drop / peak, 1),
            }
        )
    pd.DataFrame(rows).to_csv(args.output_dir / "peak_transfer_auroc_by_model.csv", index=False)

    # Table 4.4 — same-language vs. transfer AUROC at the transfer-optimal layer
    rows = []
    for model in order:
        group = off[off["model"].eq(model)]
        trajectory = group.groupby("layer")["auroc"].mean()
        layer = int(trajectory.idxmax())
        same_raw = float(same[same["model"].eq(model) & same["layer"].eq(layer)]["auroc"].mean())
        transfer_raw = float(trajectory.loc[layer])
        gap = round(same_raw - transfer_raw, 3)
        same_auroc = round(same_raw, 3)
        transfer_auroc = round(transfer_raw, 3)
        rows.append(
            {
                "Model": model,
                "Transfer-optimal layer": layer,
                "Same-language AUROC at that layer": same_auroc,
                "Transfer AUROC at that layer": transfer_auroc,
                "Gap": gap,
                "Gap %": round(100.0 * gap / same_auroc, 1),
            }
        )
    pd.DataFrame(rows).to_csv(args.output_dir / "same_language_vs_transfer_auroc.csv", index=False)
    print(f"Wrote 4 tables to {args.output_dir}")


if __name__ == "__main__":
    main()
