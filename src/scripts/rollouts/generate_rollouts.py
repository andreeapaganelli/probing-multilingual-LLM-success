#!/usr/bin/env python3
"""
Generate 5 rollout answers per problem using vLLM for batched inference.

Reads problems from a CSV file with columns: ground_truth, split, <language>, ....

Models run:
  Qwen/Qwen3-0.6B, Qwen/Qwen3-1.7B, Qwen/Qwen3-4B, Qwen/Qwen3-8B,
  Qwen/Qwen3.5-4B, Qwen/Qwen3.5-9B, openai/gpt-oss-20b (high + low reasoning)

NOTE: openai/gpt-oss-20b requires a special vLLM build.
Install it with:
  pip install "vllm==0.10.1+gptoss" \\
    --extra-index-url https://wheels.vllm.ai/gpt-oss/

Output: one JSONL file per (model, language), e.g.:
  <output_dir>/Qwen3-4B_English.jsonl
  <output_dir>/gpt-oss-20b_low_French.jsonl

Each JSONL line has the schema:
  {
    "problem_id":             str,
    "problem_text":           str,
    "ground_truth":           str,
    "model_name":             str,
    "language":               str,
    "reasoning_effort":       "high" | "low" | null,
    "max_model_len_used":     int,
    "has_truncated_rollouts": bool,   # true → problem will be re-run on next invocation
    "generated_solutions": [
      {
        "rollout_index": int,
        "text":          str,
        "output_tokens": int,
        "finish_reason": "stop" | "length",
        "truncated":     bool,
        "loop_stripped": bool   # true → text was trimmed after 3rd repeated \boxed{}
      },
      ...  # 5 entries
    ]
  }

Usage:
  # Single language
  python -m src.scripts.rollouts.generate_rollouts \\
      --csv_path data/MATH_translated.csv \\
      --model all --output_dir outputs/rollouts/MATH

  # All languages (one model load per model, loops languages internally)
  python -m src.scripts.rollouts.generate_rollouts \\
      --csv_path data/MATH_translated.csv \\
      --language all --model all --output_dir outputs/rollouts/MATH

  # Smoke test (2 problems per language)
  python -m src.scripts.rollouts.generate_rollouts \\
      --csv_path data/MATH_translated.csv \\
      --language all --model Qwen3-0.6B --smoke-test
"""

import argparse
import csv
import gc
import json
import logging
import os
import re
import sys

from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer
from vllm import LLM, SamplingParams

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Loop detection
# ──────────────────────────────────────────────────────────────────────────────

# How many times the same \boxed{…} value must appear before we treat it as a
# repetition loop and trim the tail.
LOOP_MAX_REPEATS = 3

_BOXED_OPEN = re.compile(r"\\boxed\{")


def _boxed_end(text: str, open_brace: int) -> int:
    """Return position *after* the matching closing brace of text[open_brace]=='{'.
    Returns -1 if the brace is never closed within text."""
    depth = 1
    i = open_brace + 1
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return i if depth == 0 else -1


def strip_boxed_loop(text: str, max_repeats: int = LOOP_MAX_REPEATS) -> tuple[str, bool]:
    r"""Strip a \boxed{} repetition loop from model output.

    Scans for repeated identical \boxed{…} expressions (handles nested braces).
    When any value appears max_repeats times the text is trimmed right after
    that occurrence — keeping the final answer and discarding the loop tail.

    Returns:
        (output_text, was_stripped) — output_text is unchanged if no loop found.
    """
    counts: dict[str, int] = {}
    for m in _BOXED_OPEN.finditer(text):
        end = _boxed_end(text, m.end() - 1)   # m.end()-1 is the index of '{'
        if end < 0:
            continue
        val = text[m.start():end]              # full '\boxed{…}' expression
        counts[val] = counts.get(val, 0) + 1
        if counts[val] >= max_repeats:
            return text[:end], True
    return text, False


# ──────────────────────────────────────────────────────────────────────────────
# Model registry
# ──────────────────────────────────────────────────────────────────────────────

MODEL_CONFIGS = [
    {
        "model_name": "Qwen/Qwen3-0.6B",
        "output_tag": "Qwen3-0.6B",
        "reasoning_effort": None,
        "enable_thinking": False,
    },
    {
        "model_name": "Qwen/Qwen3-1.7B",
        "output_tag": "Qwen3-1.7B",
        "reasoning_effort": None,
        "enable_thinking": False,
    },
    {
        "model_name": "Qwen/Qwen3-4B",
        "output_tag": "Qwen3-4B",
        "reasoning_effort": None,
        "enable_thinking": False,
    },
    {
        "model_name": "Qwen/Qwen3-8B",
        "output_tag": "Qwen3-8B",
        "reasoning_effort": None,
        "enable_thinking": False,
    },
    {
        "model_name": "Qwen/Qwen3.5-4B",
        "output_tag": "Qwen3.5-4B",
        "reasoning_effort": None,
        "enable_thinking": False,
    },
    {
        "model_name": "Qwen/Qwen3.5-9B",
        "output_tag": "Qwen3.5-9B",
        "reasoning_effort": None,
        "enable_thinking": False,
    },
    {
        "model_name": "openai/gpt-oss-20b",
        "output_tag": "gpt-oss-20b_high",
        "reasoning_effort": "high",
        "enable_thinking": False,
        "template_kwargs": {"reasoning_effort": "high"},
    },
    {
        "model_name": "openai/gpt-oss-20b",
        "output_tag": "gpt-oss-20b_low",
        "reasoning_effort": "low",
        "enable_thinking": False,
        "template_kwargs": {"reasoning_effort": "low"},
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# Prompt constants
# ──────────────────────────────────────────────────────────────────────────────

# Per-language (system_prompt, user_suffix) pairs.
# English is the fallback for any language not explicitly listed.
LANGUAGE_PROMPTS: dict[str, tuple[str, str]] = {
    "English":    (
        "Please reason step by step, and put your final answer within \\boxed{}.",
        "\nLet's think step by step and output the final answer within \\boxed{}.",
    ),
    "French":     (
        "Veuillez raisonner étape par étape et mettre votre réponse finale dans \\boxed{}.",
        "\nRéfléchissons étape par étape et indiquons la réponse finale dans \\boxed{}.",
    ),
    "Italian":    (
        "Per favore, ragiona passo per passo e inserisci la risposta finale dentro \\boxed{}.",
        "\nPensiamo passo per passo e scriviamo la risposta finale dentro \\boxed{}.",
    ),
    "Russian":    (
        "Пожалуйста, рассуждайте шаг за шагом и поместите окончательный ответ в \\boxed{}.",
        "\nДавайте рассуждать шаг за шагом и запишем окончательный ответ в \\boxed{}.",
    ),
    "Chinese":    (
        "请逐步推理，并将最终答案放在 \\boxed{} 中。",
        "\n让我们逐步思考，并将最终答案放在 \\boxed{} 中。",
    ),
    "Arabic":     (
        "يرجى الاستدلال خطوة بخطوة، وضع الإجابة النهائية داخل \\boxed{}.",
        "\nلنفكر خطوة بخطوة ونكتب الإجابة النهائية داخل \\boxed{}.",
    ),
    "Thai":       (
        "กรุณาให้เหตุผลทีละขั้นตอน และใส่คำตอบสุดท้ายไว้ใน \\boxed{}",
        "\nมาคิดทีละขั้นตอนและแสดงคำตอบสุดท้ายใน \\boxed{}",
    ),
    "Swahili":    (
        "Tafadhali fikiri hatua kwa hatua, na weka jibu lako la mwisho ndani ya \\boxed{}.",
        "\nHebu tufikirie hatua kwa hatua na kutoa jibu la mwisho ndani ya \\boxed{}.",
    ),
    "Telugu":     (
        "దయచేసి దశలవారీగా వాదించండి మరియు మీ చివరి సమాధానాన్ని \\boxed{} లో ఉంచండి.",
        "\nదశలవారీగా ఆలోచించి చివరి సమాధానాన్ని \\boxed{} లో రాద్దాం.",
    ),
    "Turkish":    (
        "Lütfen adım adım akıl yürütün ve nihai cevabınızı \\boxed{} içine yazın.",
        "\nAdım adım düşünelim ve nihai cevabı \\boxed{} içinde verelim.",
    ),
}

N_ROLLOUTS = 5
MAX_MODEL_LEN_CAP = 16384  # upper bound; use model's default if smaller
TEMPERATURE = 0.7
TOP_P = 0.8
# Measured from data/MATH_translated.csv across all languages (Qwen3-0.6B tokenizer).
MAX_PROMPT_LEN = 2500
GENERATION_BUFFER = 500  # safety margin so prompt + output never exceeds max_model_len


# ──────────────────────────────────────────────────────────────────────────────
# Model context-length resolution
# ──────────────────────────────────────────────────────────────────────────────


def get_model_max_len(model_name: str, cap: int = MAX_MODEL_LEN_CAP) -> int:
    """Return min(cap, model's max_position_embeddings).

    Uses the model's native context length when it is smaller than the cap.
    """
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model_default = getattr(config, "max_position_embeddings", cap)
    actual = min(cap, model_default)
    log.info(
        "  max_position_embeddings=%d → max_model_len=%d", model_default, actual
    )
    return actual


# ──────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ──────────────────────────────────────────────────────────────────────────────


def load_problems(csv_path: str, language: str) -> list:
    """Load problems for a single language from a CSV.

    CSV must have columns: ground_truth, split, <language>, ....

    Returns a list of dicts with keys: problem_id, problem_text, ground_truth.
    """
    log.info("Loading problems from CSV %s  language=%s …", csv_path, language)
    all_problems = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            text = row.get(language, "").strip()
            if not text:
                log.warning("Row %d: empty %s translation — skipping.", i, language)
                continue
            all_problems.append({
                "problem_id":   f"{i}_{language}",
                "problem_text": text,
                "ground_truth": row["ground_truth"].strip(),
            })
    log.info("Loaded %d problems for language=%s", len(all_problems), language)
    return all_problems


def load_problems_all_languages(csv_path: str, languages: list) -> dict:
    """Load problems for all requested languages in one CSV pass.

    Returns dict mapping language -> list of problem dicts.
    """
    log.info("Loading problems from CSV %s  (%d languages) …", csv_path, len(languages))
    problems_per_language: dict = {lang: [] for lang in languages}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            gt = row["ground_truth"].strip()
            for lang in languages:
                text = row.get(lang, "").strip()
                if not text:
                    log.warning("Row %d: empty %s translation — skipping.", i, lang)
                    continue
                problems_per_language[lang].append({
                    "problem_id":   f"{i}_{lang}",
                    "problem_text": text,
                    "ground_truth": gt,
                    "hf_split":     "train",
                })
    for lang in languages:
        log.info("  %s: %d problems", lang, len(problems_per_language[lang]))
    return problems_per_language


# ──────────────────────────────────────────────────────────────────────────────
# Prompt formatting
# ──────────────────────────────────────────────────────────────────────────────


def format_prompt(
    problem_text: str,
    tokenizer: AutoTokenizer,
    reasoning_effort: str | None,
    enable_thinking: bool,
    language: str = "English",
) -> str:
    """Format a problem into a model-specific chat prompt string."""
    system_prompt, user_suffix = LANGUAGE_PROMPTS.get(language, LANGUAGE_PROMPTS["English"])
    system_content = system_prompt
    if reasoning_effort is not None:
        # Harmony / gpt-oss format: prepend reasoning effort directive
        system_content = f"Reasoning: {reasoning_effort}\n\n{system_content}"

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": problem_text + user_suffix},
    ]

    base_kwargs = {"tokenize": False, "add_generation_prompt": True}

    # Try to pass enable_thinking explicitly so Qwen3's chat template injects
    # </think> right after the assistant token when thinking is disabled (without
    # this the model defaults to thinking mode regardless of the flag).
    try:
        return tokenizer.apply_chat_template(
            messages, enable_thinking=enable_thinking, **base_kwargs
        )
    except TypeError:
        # Tokenizer does not support enable_thinking.
        # Fall back: append /think or /no_think hint for models that read it.
        if enable_thinking:
            messages[-1]["content"] += "\n/think"
        return tokenizer.apply_chat_template(messages, **base_kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# Checkpointing helpers
# ──────────────────────────────────────────────────────────────────────────────


def load_completed_ids(output_path: str, keep_truncated: bool = False) -> set:
    """Return problem_ids that should be skipped on the next invocation.

    By default, problems where any rollout was truncated (finish_reason == "length")
    are NOT included so they will be regenerated on the next run.  If
    keep_truncated=True, any existing row counts as complete, including rows with
    truncated rollouts.
    """
    if not os.path.exists(output_path):
        return set()
    completed = set()
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if keep_truncated or not rec.get("has_truncated_rollouts", False):
                    completed.add(rec["problem_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return completed


def load_existing_rollouts(output_path: str) -> dict:
    """Load all existing records keyed by problem_id. Last record wins on duplicates."""
    if not os.path.exists(output_path):
        return {}
    records = {}
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records[rec["problem_id"]] = rec
            except (json.JSONDecodeError, KeyError):
                pass
    return records


# ──────────────────────────────────────────────────────────────────────────────
# Generation for one model config
# ──────────────────────────────────────────────────────────────────────────────


SMOKE_TEST_N_PROBLEMS = 2


def _run_language(
    cfg: dict,
    problems: list,
    output_dir: str,
    batch_size: int,
    language: str,
    llm,
    tokenizer,
    sampling_params,
    max_model_len: int,
    keep_truncated: bool = False,
) -> None:
    """Run inference for a single language using an already-loaded model.

    Only generates rollouts that are missing or truncated. Non-truncated rollouts
    from previous runs are preserved and merged with new ones. After each batch the
    file is atomically rewritten so progress is never lost.
    """
    model_name       = cfg["model_name"]
    output_tag       = cfg["output_tag"]
    reasoning_effort = cfg["reasoning_effort"]
    enable_thinking  = cfg["enable_thinking"]

    output_path = os.path.join(output_dir, f"{output_tag}_{language}.jsonl")

    # Load all existing records; complete ones go straight to all_records.
    existing = load_existing_rollouts(output_path)
    all_records = {}   # pid -> final merged record
    # (problem, existing_rec | None, truncated_slot_indices)
    pending: list[tuple[dict, dict | None, list[int]]] = []

    for p in problems:
        pid = p["problem_id"]
        if pid in existing:
            rec = existing[pid]
            if keep_truncated:
                all_records[pid] = rec
                continue
            trunc_indices = [
                i for i, s in enumerate(rec.get("generated_solutions", []))
                if s.get("truncated", False)
            ]
            if not trunc_indices:
                all_records[pid] = rec
            else:
                pending.append((p, rec, trunc_indices))
        else:
            pending.append((p, None, list(range(N_ROLLOUTS))))

    if not pending:
        log.info("[%s/%s] All %d problems already complete. Skipping.", output_tag, language, len(problems))
        return

    n_fresh   = sum(1 for _, rec, _ in pending if rec is None)
    n_partial = len(pending) - n_fresh
    log.info(
        "[%s/%s] %d problems pending: %d fresh, %d partial re-runs (%d already complete).",
        output_tag, language, len(pending), n_fresh, n_partial, len(all_records),
    )

    log.info("[%s/%s] Formatting %d prompts …", output_tag, language, len(pending))
    prompts = [
        format_prompt(p["problem_text"], tokenizer, reasoning_effort, enable_thinking, language=language)
        for p, _, _ in tqdm(pending, desc=f"formatting/{language}", leave=False)
    ]

    os.makedirs(output_dir, exist_ok=True)
    n_batches = (len(pending) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(n_batches), desc=f"{output_tag}/{language}", unit="batch"):
        start        = batch_idx * batch_size
        batch_items  = pending[start : start + batch_size]
        batch_prompts = prompts[start : start + batch_size]

        # Each problem may need a different number of new rollouts.
        batch_sp = [
            SamplingParams(
                temperature=sampling_params.temperature,
                top_p=sampling_params.top_p,
                max_tokens=sampling_params.max_tokens,
                n=len(trunc_indices),
            )
            for _, _, trunc_indices in batch_items
        ]

        outputs = llm.generate(batch_prompts, batch_sp)

        for (problem, existing_rec, trunc_indices), vllm_output in zip(batch_items, outputs):
            new_slots = []
            for slot, completion in zip(trunc_indices, vllm_output.outputs):
                text, loop_stripped = strip_boxed_loop(completion.text)
                if loop_stripped:
                    finish_reason = "stop"
                    truncated = False
                    orig_len = len(completion.text)
                    output_tokens = max(1, round(
                        len(completion.token_ids) * len(text) / orig_len
                    )) if orig_len else len(completion.token_ids)
                else:
                    finish_reason = completion.finish_reason
                    truncated = completion.finish_reason == "length"
                    output_tokens = len(completion.token_ids)
                new_slots.append({
                    "rollout_index": slot,
                    "text":          text,
                    "output_tokens": output_tokens,
                    "finish_reason": finish_reason,
                    "truncated":     truncated,
                    "loop_stripped": loop_stripped,
                })

            if existing_rec is None:
                solutions = new_slots
            else:
                solutions = list(existing_rec["generated_solutions"])
                for sol in new_slots:
                    solutions[sol["rollout_index"]] = sol

            has_truncated = any(s["truncated"] for s in solutions)
            if has_truncated:
                n_trunc = sum(s["truncated"] for s in solutions)
                log.warning(
                    "[%s/%s] problem_id=%s: %d/%d rollout(s) still truncated at %d tokens — "
                    "will be re-run on next invocation unless --keep-truncated is used.",
                    output_tag, language, problem["problem_id"], n_trunc, N_ROLLOUTS, max_model_len,
                )

            all_records[problem["problem_id"]] = {
                "problem_id":             problem["problem_id"],
                **( {"hf_split": problem["hf_split"]} if "hf_split" in problem else {} ),
                "problem_text":           problem["problem_text"],
                "ground_truth":           problem["ground_truth"],
                "model_name":             model_name,
                "language":               language,
                "reasoning_effort":       reasoning_effort,
                "max_model_len_used":     max_model_len,
                "has_truncated_rollouts": has_truncated,
                "generated_solutions":    solutions,
            }

        # Atomic checkpoint: rewrite the full file after every batch so a crash
        # never leaves a partial or inconsistent file.
        tmp_path = output_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            for rec in all_records.values():
                f.write(json.dumps(rec) + "\n")
        os.replace(tmp_path, output_path)

    log.info("[%s/%s] Saved to %s", output_tag, language, output_path)


def run_model(
    cfg: dict,
    problems_per_language: dict,
    output_dir: str,
    batch_size: int,
    smoke_test: bool = False,
    languages: list | None = None,
    max_model_len_override: int | None = None,
    keep_truncated: bool = False,
) -> None:
    """Load one model and generate rollouts for one or more languages.

    problems_per_language maps language -> list of problem dicts.
    When smoke_test=True, each language is capped to SMOKE_TEST_N_PROBLEMS.
    """
    if languages is None:
        languages = list(problems_per_language.keys())

    model_name = cfg["model_name"]
    output_tag = cfg["output_tag"]

    if smoke_test:
        output_dir = os.path.join(output_dir, "smoke_test")

    # ── Resolve context length (once per model) ────────────────────────────────
    log.info("[%s] Resolving max_model_len …", output_tag)
    cap = max_model_len_override if max_model_len_override is not None else MAX_MODEL_LEN_CAP
    max_model_len = get_model_max_len(model_name, cap=cap)

    # ── Tokenizer (once per model) ─────────────────────────────────────────────
    log.info("[%s] Loading tokenizer …", output_tag)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # ── vLLM model (once per model) ───────────────────────────────────────────
    log.info("[%s] Initialising vLLM (max_model_len=%d) …", output_tag, max_model_len)
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        dtype="auto",               # let vLLM pick native precision (critical for gpt-oss-20b MXFP4)
        gpu_memory_utilization=0.92,
        enable_prefix_caching=True,
        max_num_seqs=min(batch_size, 50),
        max_model_len=max_model_len,
    )

    max_tokens = max_model_len - MAX_PROMPT_LEN - GENERATION_BUFFER
    log.info("[%s] max_tokens=%d (max_model_len=%d − prompt=%d − buffer=%d)",
             output_tag, max_tokens, max_model_len, MAX_PROMPT_LEN, GENERATION_BUFFER)
    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=max_tokens,
        n=N_ROLLOUTS,
    )

    # ── Loop over languages without reloading the model ───────────────────────
    for language in languages:
        problems = problems_per_language[language]
        if smoke_test:
            problems = problems[:SMOKE_TEST_N_PROBLEMS]
            log.info("[%s/%s] Smoke-test mode: using %d problems.", output_tag, language, len(problems))

        _run_language(
            cfg=cfg,
            problems=problems,
            output_dir=output_dir,
            batch_size=batch_size,
            language=language,
            llm=llm,
            tokenizer=tokenizer,
            sampling_params=sampling_params,
            max_model_len=max_model_len,
            keep_truncated=keep_truncated,
        )

    # ── Free GPU memory before next model ─────────────────────────────────────
    # Note: when running all models sequentially, consider submitting separate
    # SLURM jobs (one per --model tag) to guarantee full GPU memory release.
    import torch
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate math rollouts with vLLM (5 samples per problem).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    valid_tags = [c["output_tag"] for c in MODEL_CONFIGS]
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to a CSV with columns: ground_truth, split, <language>, ....",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        help=(
            f"Model to run. 'all' runs all models sequentially. "
            f"Valid tags: {', '.join(valid_tags)}."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/rollouts/MATH",
        help="Directory for JSONL output files (created if absent).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Problems per vLLM batch (controls checkpoint granularity; vLLM still auto-batches internally).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            f"Run only {SMOKE_TEST_N_PROBLEMS} problems per language to verify the pipeline end-to-end. "
            "Outputs are written to <output_dir>/smoke_test/ and do not interfere with a real run."
        ),
    )
    parser.add_argument(
        "--language",
        type=str,
        default="English",
        help=(
            "Language column to use. "
            "Pass 'all' to run every language in one model-load pass, "
            "or a comma-separated list (e.g. French,Italian,Swahili) to run a subset. "
            f"Supported: {', '.join(LANGUAGE_PROMPTS)}."
        ),
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help=(
            f"Override the model context length cap (default: {MAX_MODEL_LEN_CAP}). "
            "Smaller values reduce KV cache memory and improve throughput."
        ),
    )
    parser.add_argument(
        "--keep-truncated",
        action="store_true",
        help=(
            "Treat existing rows with truncated rollouts as complete, so reruns "
            "do not regenerate them."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.model == "all":
        configs_to_run = MODEL_CONFIGS
    else:
        configs_to_run = [c for c in MODEL_CONFIGS if c["output_tag"] == args.model]
        if not configs_to_run:
            valid = [c["output_tag"] for c in MODEL_CONFIGS]
            log.error("Unknown --model %r. Valid choices: %s", args.model, valid)
            sys.exit(1)

    all_languages = args.language == "all"

    if all_languages:
        languages_to_run = list(LANGUAGE_PROMPTS.keys())
    elif "," in args.language:
        languages_to_run = [l.strip() for l in args.language.split(",")]
        unknown = [l for l in languages_to_run if l not in LANGUAGE_PROMPTS]
        if unknown:
            log.error("Unknown language(s): %s. Supported: %s", unknown, list(LANGUAGE_PROMPTS))
            sys.exit(1)
    else:
        if args.language not in LANGUAGE_PROMPTS:
            log.error("Unknown --language %r. Supported: 'all', %s", args.language, list(LANGUAGE_PROMPTS))
            sys.exit(1)
        languages_to_run = [args.language]

    if len(languages_to_run) > 1:
        problems_per_language = load_problems_all_languages(args.csv_path, languages_to_run)
    else:
        problems_per_language = {languages_to_run[0]: load_problems(args.csv_path, languages_to_run[0])}

    for cfg in configs_to_run:
        run_model(
            cfg,
            problems_per_language,
            args.output_dir,
            args.batch_size,
            smoke_test=args.smoke_test,
            languages=languages_to_run,
            max_model_len_override=args.max_model_len,
            keep_truncated=args.keep_truncated,
        )

    log.info("All done.")


if __name__ == "__main__":
    main()
