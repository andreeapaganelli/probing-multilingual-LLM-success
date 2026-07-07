"""
Extract end-of-prompt residual-stream activations for the multilingual MATH dataset.

Reads problems from data/MATH_translated.csv, formats them into chat prompts
(identical to generate_rollouts.py), then extracts the hidden state at every
layer at token_position=-1 (last token before generation).

Outputs:
  {output_dir}/{language}/{model_tag}/activations_token_{pos}.joblib
  → {"activations": {layer_idx: np.ndarray(N, d_model)},
     "problem_ids": [str, ...],
     "language": str}

Works for all architectures including MoE (gpt-oss-20b).
For gpt-oss-20b_high / gpt-oss-20b_low the base model openai/gpt-oss-20b is
loaded once; the output_tag only affects the prompt format (reasoning_effort).

Usage (single language):
    python -m src.scripts.extract.extract_math_activations \\
        --model Qwen3-4B \\
        --language French \\
        --output_dir outputs/activations/MATH

Usage (SLURM array — one task per language):
    python -m src.scripts.extract.extract_math_activations \\
        --model Qwen3-4B \\
        --language_index $SLURM_ARRAY_TASK_ID \\
        --output_dir outputs/activations/MATH
"""

import argparse
import csv
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry — must match generate_rollouts.py
# ---------------------------------------------------------------------------

MODEL_CONFIGS = [
    {
        "model_name":      "Qwen/Qwen3-0.6B",
        "output_tag":      "Qwen3-0.6B",
        "reasoning_effort": None,
        "enable_thinking": False,
    },
    {
        "model_name":      "Qwen/Qwen3-1.7B",
        "output_tag":      "Qwen3-1.7B",
        "reasoning_effort": None,
        "enable_thinking": False,
    },
    {
        "model_name":      "Qwen/Qwen3-4B",
        "output_tag":      "Qwen3-4B",
        "reasoning_effort": None,
        "enable_thinking": False,
    },
    {
        "model_name":      "Qwen/Qwen3-8B",
        "output_tag":      "Qwen3-8B",
        "reasoning_effort": None,
        "enable_thinking": False,
    },
    {
        "model_name":      "Qwen/Qwen3.5-4B",
        "output_tag":      "Qwen3.5-4B",
        "reasoning_effort": None,
        "enable_thinking": False,
    },
    {
        "model_name":      "Qwen/Qwen3.5-9B",
        "output_tag":      "Qwen3.5-9B",
        "reasoning_effort": None,
        "enable_thinking": False,
    },
    {
        "model_name":      "openai/gpt-oss-20b",
        "output_tag":      "gpt-oss-20b_high",
        "reasoning_effort": "high",
        "enable_thinking": False,
    },
    {
        "model_name":      "openai/gpt-oss-20b",
        "output_tag":      "gpt-oss-20b_low",
        "reasoning_effort": "low",
        "enable_thinking": False,
    },
]


LANGUAGES = [
    "English", "Chinese", "Italian", "French", "Swahili",
    "Russian", "Turkish", "Arabic", "Thai", "Telugu",
]


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

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[2]
CSV_PATH = str(_REPO_ROOT / "data" / "MATH_translated.csv")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_problems(csv_path: str, language: str) -> list:
    """Return list of {problem_id, language, problem_text, ground_truth, difficulty}."""
    problems = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            text = row.get(language, "").strip()
            if not text:
                log.warning("Row %d: empty %s translation — skipping.", i, language)
                continue
            problems.append({
                "problem_id":   f"{i}_{language}",
                "language":     language,
                "problem_text": text,
                "ground_truth": row["ground_truth"].strip(),
                "difficulty":   float(row["difficulty"]) if row.get("difficulty") else None,
            })
    log.info("Loaded %d problems for language=%s", len(problems), language)
    return problems


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def format_prompt(
    problem_text: str,
    tokenizer: AutoTokenizer,
    reasoning_effort: Optional[str],
    enable_thinking: bool,
    language: str = "English",
) -> str:
    system_prompt, user_suffix = LANGUAGE_PROMPTS.get(language, LANGUAGE_PROMPTS["English"])
    system_content = system_prompt
    if reasoning_effort is not None:
        system_content = f"Reasoning: {reasoning_effort}\n\n{system_content}"

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": problem_text + user_suffix},
    ]
    base_kwargs = {"tokenize": False, "add_generation_prompt": True}

    try:
        return tokenizer.apply_chat_template(
            messages, enable_thinking=enable_thinking, **base_kwargs
        )
    except TypeError:
        if enable_thinking:
            messages[-1]["content"] += "\n/think"
        return tokenizer.apply_chat_template(messages, **base_kwargs)


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_activations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: List[str],
    batch_size: int = 8,
    max_length: int = 3000,
    token_position: int = -1,
) -> dict:
    """
    Extract residual-stream hidden states at `token_position` for every layer.

    Returns dict mapping layer_index -> np.ndarray of shape (N, d_model).
    """
    input_device = next(model.parameters()).device
    all_hidden: dict[int, list] = {}

    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting"):
        batch_texts = texts[i : i + batch_size]
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids      = enc["input_ids"].to(input_device)
        attention_mask = enc["attention_mask"].to(input_device)

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = out.hidden_states  # tuple: (embedding, layer_1, ..., layer_N)

        # Resolve the target token per example, accounting for right-padding
        # (real tokens occupy [0, length); last real token is at index length-1)
        lengths = attention_mask.sum(dim=1)  # (B,)
        B = input_ids.shape[0]
        T = input_ids.shape[1]

        for layer_idx, hs in enumerate(hidden_states):
            if token_position < 0:
                idx = (lengths + token_position).clamp(min=0, max=T - 1)
            else:
                idx = torch.full((B,), token_position, device=hs.device).clamp(
                    min=0, max=T - 1
                )
            feats    = hs[torch.arange(B, device=hs.device), idx]  # (B, D)
            feats_np = feats.float().cpu().numpy()

            if layer_idx not in all_hidden:
                all_hidden[layer_idx] = []
            all_hidden[layer_idx].append(feats_np)

    return {k: np.concatenate(v, axis=0) for k, v in all_hidden.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract end-of-prompt activations for the multilingual MATH dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    valid_tags = [c["output_tag"] for c in MODEL_CONFIGS]

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=f"Model output tag. Valid: {', '.join(valid_tags)}.",
    )

    lang_group = parser.add_mutually_exclusive_group(required=True)
    lang_group.add_argument(
        "--language",
        type=str,
        help=(
            f"Language to process, comma-separated languages, or 'all' to run all languages sequentially. "
            f"Valid: {', '.join(LANGUAGES)}."
        ),
    )
    lang_group.add_argument(
        "--language_index",
        type=int,
        help=(
            f"0-based index into LANGUAGES (for SLURM array jobs). "
            f"Mapping: {', '.join(f'{i}={l}' for i, l in enumerate(LANGUAGES))}."
        ),
    )

    parser.add_argument(
        "--csv_path",
        type=str,
        default=CSV_PATH,
        help="Path to the translated dataset CSV (default: data/MATH_translated.csv).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/activations/MATH",
        help="Root output directory. Files go to {output_dir}/[{dataset_tag}/]{language}/{model_tag}/.",
    )
    parser.add_argument(
        "--dataset-tag",
        type=str,
        default=None,
        help="Optional dataset label inserted into the output path, e.g. 'MATH'.",
    )
    parser.add_argument("--batch_size",     type=int,   default=8)
    parser.add_argument("--max_length",     type=int,   default=3000)
    parser.add_argument(
        "--token_position",
        type=int,
        default=-1,
        help="Token index to extract (default -1 = last token of prompt).",
    )
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        action="store_true",
        default=None,
        help=(
            "Force chat templates that support it to leave thinking enabled. "
            "For Qwen3 this probes at the assistant-generation boundary."
        ),
    )
    thinking_group.add_argument(
        "--disable-thinking",
        "--no-thinking",
        dest="enable_thinking",
        action="store_false",
        help=(
            "Force chat templates that support it to disable thinking. "
            "For Qwen3 this probes after the injected empty <think></think> block."
        ),
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Attention backend. GPT-OSS currently requires eager in Transformers.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate activation cache even if the output file already exists.",
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        default=None,
        help=(
            "Maximum GPU memory to allocate, e.g. '16GiB'. "
            "Overflow is placed on CPU RAM. "
            "Useful when sharing the GPU with a concurrent vLLM process."
        ),
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force the model entirely onto CPU (ignores --max-gpu-memory).",
    )
    return parser.parse_args()


def run_language(
    cfg: dict,
    language: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    args,
) -> None:
    problems = load_problems(args.csv_path, language)
    if not problems:
        log.warning("No problems found for language=%s, skipping.", language)
        return

    base = os.path.join(args.output_dir, args.dataset_tag) if args.dataset_tag else args.output_dir
    out_dir  = os.path.join(base, language, cfg["output_tag"])
    out_path = os.path.join(out_dir, f"activations_token_{args.token_position}.joblib")

    if os.path.exists(out_path) and not args.overwrite:
        log.info("[%s/%s] Cache already exists, skipping: %s",
                 language, cfg["output_tag"], out_path)
        return

    os.makedirs(out_dir, exist_ok=True)

    log.info("[%s/%s] Formatting %d prompts …",
             language, cfg["output_tag"], len(problems))
    prompts = [
        format_prompt(
            p["problem_text"], tokenizer,
            cfg["reasoning_effort"], cfg["enable_thinking"],
            language=language,
        )
        for p in tqdm(problems, desc="formatting", leave=False)
    ]

    log.info("[%s/%s] Extracting activations (batch=%d, max_len=%d, pos=%d) …",
             language, cfg["output_tag"],
             args.batch_size, args.max_length, args.token_position)
    activations = extract_activations(
        model=model,
        tokenizer=tokenizer,
        texts=prompts,
        batch_size=args.batch_size,
        max_length=args.max_length,
        token_position=args.token_position,
    )

    n_layers     = len(activations)
    sample_shape = activations[0].shape
    log.info("[%s/%s] %d layers, shape per layer: %s",
             language, cfg["output_tag"], n_layers, sample_shape)

    payload = {
        "activations": activations,
        "problem_ids": [p["problem_id"] for p in problems],
        "language":    language,
        "enable_thinking": bool(cfg["enable_thinking"]),
    }
    joblib.dump(payload, out_path)
    log.info("[%s/%s] Saved to %s", language, cfg["output_tag"], out_path)


def main() -> None:
    args = parse_args()

    # Resolve model config
    matching = [c for c in MODEL_CONFIGS if c["output_tag"] == args.model]
    if not matching:
        log.error("Unknown --model %r. Valid: %s",
                  args.model, [c["output_tag"] for c in MODEL_CONFIGS])
        sys.exit(1)
    cfg = matching[0]
    if args.enable_thinking is not None:
        cfg = {**cfg, "enable_thinking": bool(args.enable_thinking)}
    log.info(
        "Using prompt thinking mode for %s: enable_thinking=%s",
        cfg["output_tag"],
        cfg["enable_thinking"],
    )

    # Resolve language(s)
    if args.language_index is not None:
        if not (0 <= args.language_index < len(LANGUAGES)):
            log.error("--language_index %d out of range [0, %d).",
                      args.language_index, len(LANGUAGES))
            sys.exit(1)
        languages = [LANGUAGES[args.language_index]]
    elif args.language == "all":
        languages = LANGUAGES
    else:
        languages = [lang.strip() for lang in args.language.split(",") if lang.strip()]
        unknown_languages = [lang for lang in languages if lang not in LANGUAGES]
        if unknown_languages:
            log.error("Unknown --language value(s) %r. Valid: %s", unknown_languages, LANGUAGES)
            sys.exit(1)

    # Check whether all caches already exist before loading the model
    _base = os.path.join(args.output_dir, args.dataset_tag) if args.dataset_tag else args.output_dir
    all_cached = (not args.overwrite) and all(
        os.path.exists(
            os.path.join(
                _base, lang, cfg["output_tag"],
                f"activations_token_{args.token_position}.joblib",
            )
        )
        for lang in languages
    )
    if all_cached:
        log.info("All caches already exist — nothing to do.")
        return

    # Load tokenizer and model once for all languages
    model_name = cfg["model_name"]
    log.info("Loading tokenizer for %s …", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log.info("Loading model %s (dtype=%s) …", model_name, args.dtype)
    dtype_map = {
        "float16":  torch.float16,
        "bfloat16": torch.bfloat16,
        "float32":  torch.float32,
    }
    torch_dtype = dtype_map.get(args.dtype, "auto")
    if args.cpu:
        device_map = "cpu"
        max_memory = None
    else:
        device_map = "auto"
        max_memory = {0: args.max_gpu_memory, "cpu": "128GiB"} if args.max_gpu_memory else None
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        max_memory=max_memory,
        low_cpu_mem_usage=True,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    text_config = getattr(model.config, "text_config", None)
    hidden_size = getattr(model.config, "hidden_size", None)
    if hidden_size is None and text_config is not None:
        hidden_size = getattr(text_config, "hidden_size", None)
    log.info("  d_model=%s", hidden_size if hidden_size is not None else "unknown")

    for language in languages:
        run_language(cfg, language, model, tokenizer, args)

    log.info("All done.")


if __name__ == "__main__":
    main()
