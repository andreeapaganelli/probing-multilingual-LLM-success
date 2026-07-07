"""Add exact prompt/input token counts to a router-candidate table.

The script reads the original rollout JSONL files, reconstructs the same chat
prompt used for generation, tokenizes that prompt with the corresponding model
tokenizer, and writes a new candidate CSV with an ``input_tokens`` column.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.scripts.rollouts.generate_rollouts import format_prompt


def load_answer_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing rollout JSONL: {path}")
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            records[str(row["problem_id"])] = row
    return records


def count_prompt_tokens(
    records: dict[str, dict[str, Any]],
    problem_ids: list[str],
    tokenizer: Any,
    batch_size: int,
) -> list[int]:
    prompts = []
    for pid in problem_ids:
        rec = records[pid]
        prompt = format_prompt(
            str(rec["problem_text"]),
            tokenizer,
            rec.get("reasoning_effort"),
            bool(rec.get("enable_thinking", False)),
            language=str(rec.get("language", "English")),
        )
        prompts.append(prompt)

    out: list[int] = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        encoded = tokenizer(batch, add_special_tokens=False, padding=False)
        out.extend(len(ids) for ids in encoded["input_ids"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--answers-dir", type=Path, default=Path("outputs/rollouts/MATH"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    candidates = pd.read_csv(args.candidates)
    required = {"model", "language", "problem_id"}
    missing_cols = sorted(required - set(candidates.columns))
    if missing_cols:
        raise ValueError(f"Candidates missing required columns: {missing_cols}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[tuple[str, str, str], int] = {}
    tokenizer_cache: dict[str, Any] = {}
    metadata_rows: list[dict[str, Any]] = []

    groups = candidates[["model", "language", "problem_id"]].drop_duplicates()
    for (model, language), group in groups.groupby(["model", "language"], sort=True):
        model = str(model)
        language = str(language)
        answers_path = args.answers_dir / f"{model}_{language}.jsonl"
        records = load_answer_records(answers_path)
        problem_ids = group["problem_id"].astype(str).tolist()
        missing = sorted(set(problem_ids) - set(records))
        if missing:
            raise ValueError(f"{answers_path} is missing {len(missing)} problem ids, first={missing[:5]}")

        sample = records[problem_ids[0]]
        model_name = str(sample.get("model_name") or model)
        if model_name not in tokenizer_cache:
            tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer = tokenizer_cache[model_name]
        token_counts = count_prompt_tokens(records, problem_ids, tokenizer, args.batch_size)
        for pid, n_tokens in zip(problem_ids, token_counts, strict=True):
            counts[(model, language, pid)] = int(n_tokens)
        metadata_rows.append(
            {
                "model": model,
                "language": language,
                "model_name": model_name,
                "n_items": len(problem_ids),
                "mean_input_tokens": float(pd.Series(token_counts).mean()),
                "max_input_tokens": int(max(token_counts)),
            }
        )
        print(f"{model}/{language}: n={len(problem_ids)} mean_input_tokens={metadata_rows[-1]['mean_input_tokens']:.1f}")

    out = candidates.copy()
    out["input_tokens"] = [
        counts[(str(row.model), str(row.language), str(row.problem_id))]
        for row in out[["model", "language", "problem_id"]].itertuples(index=False)
    ]
    out.to_csv(args.output, index=False)
    pd.DataFrame(metadata_rows).to_csv(args.output.with_name(args.output.stem + "_token_count_summary.csv"), index=False)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.output.with_name(args.output.stem + '_token_count_summary.csv')}")


if __name__ == "__main__":
    main()
