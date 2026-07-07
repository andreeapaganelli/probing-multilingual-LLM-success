"""Build per-(model, language) success rate CSVs for the MATH_translated dataset.

Reads rollout JSONL files from outputs/rollouts/MATH/ and evaluates each
rollout with the shared answer-matching logic in src/eval.py, then writes one
CSV per (model, language) pair to outputs/success_rates/MATH/.

File naming convention:
  {ModelTag}_{Language}.csv   e.g. Qwen3-4B_English.csv

Each file contains the standard 8-column schema:
  problem_id, base_problem_id, language, model, success_rate,
  n_rollouts, n_correct, has_truncated_rollouts

Input JSONL filename convention: {ModelTag}_{Language}.jsonl
  e.g. Qwen3-4B_English.jsonl  →  model=Qwen3-4B, language=English
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval import evaluate_rollout_text, parse_mcq_options, infer_correct_letter


def parse_model_language(stem: str) -> tuple[str, str] | None:
    """Parse 'ModelTag_Language' stem → (model_tag, language).

    Handles model tags that may contain underscores by matching against
    a known set of language names.
    """
    known_languages = {
        "English", "Chinese", "Italian", "French", "Swahili",
        "Russian", "Turkish", "Arabic", "Thai", "Telugu",
    }
    parts = stem.split("_")
    for split_at in range(len(parts) - 1, 0, -1):
        language = "_".join(parts[split_at:])
        model_tag = "_".join(parts[:split_at])
        if language in known_languages:
            return model_tag, language
    return None


def process_jsonl(path: Path, model_tag: str, language: str) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            problem_id = str(record.get("problem_id", ""))
            ground_truth = str(record.get("ground_truth", ""))
            problem_text = str(record.get("problem_text", ""))
            options = parse_mcq_options(problem_text)
            correct_letter = infer_correct_letter(ground_truth, options) if options else None

            solutions = record.get("generated_solutions", [])
            scores = []
            truncated_flags = []
            for sol in solutions:
                if isinstance(sol, dict):
                    text = str(sol.get("text", ""))
                    is_trunc = bool(sol.get("truncated", False))
                else:
                    text = str(sol)
                    is_trunc = False
                score = evaluate_rollout_text(
                    text=text,
                    ground_truth=ground_truth,
                    options=options,
                    correct_letter=correct_letter,
                )
                scores.append(score)
                truncated_flags.append(is_trunc)

            n_rollouts = len(scores)
            n_correct = sum(scores)
            success_rate = n_correct / n_rollouts if n_rollouts > 0 else 0.0
            has_truncated = any(truncated_flags)

            base_problem_id = problem_id.rsplit("_", 1)[0] if "_" in problem_id else problem_id

            rows.append({
                "problem_id": problem_id,
                "base_problem_id": base_problem_id,
                "language": language,
                "model": model_tag,
                "success_rate": success_rate,
                "n_rollouts": n_rollouts,
                "n_correct": n_correct,
                "has_truncated_rollouts": int(has_truncated),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--answers-dir",
        type=Path,
        default=Path("outputs/rollouts/MATH"),
        help="Directory containing {ModelTag}_{Language}.jsonl files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/success_rates/MATH"),
    )
    parser.add_argument(
        "--languages",
        type=str,
        default=None,
        help="Comma-separated language filter, e.g. English,French.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model tag filter, e.g. Qwen3-4B.",
    )
    args = parser.parse_args()

    language_filter = {s.strip() for s in args.languages.split(",")} if args.languages else None
    model_filter = {s.strip() for s in args.models.split(",")} if args.models else None

    jsonl_files = sorted(args.answers_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"No .jsonl files found in {args.answers_dir}")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "problem_id", "base_problem_id", "language", "model",
        "success_rate", "n_rollouts", "n_correct", "has_truncated_rollouts",
    ]

    total_written = 0
    for path in jsonl_files:
        parsed = parse_model_language(path.stem)
        if parsed is None:
            print(f"  Skipping (cannot parse model/language): {path.name}")
            continue
        model_tag, language = parsed
        if language_filter and language not in language_filter:
            continue
        if model_filter and model_tag not in model_filter:
            continue
        print(f"  Processing {path.name} → model={model_tag}, language={language}")
        rows = process_jsonl(path, model_tag, language)
        mean_sr = sum(r["success_rate"] for r in rows) / len(rows) if rows else 0.0
        print(f"    {len(rows)} problems, mean SR={mean_sr:.3f}")

        out_path = args.output_dir / f"{model_tag}_{language}.csv"
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        total_written += len(rows)

    print(f"\nWrote {total_written} rows across individual CSVs → {args.output_dir}")


if __name__ == "__main__":
    main()
