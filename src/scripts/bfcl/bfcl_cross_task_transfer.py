from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.bfcl import (
    BFCL_DEFAULT_CATEGORIES,
    assign_split,
    bfcl_ast_correct,
    format_bfcl_chat_prompt,
    format_bfcl_qwen_manual_prompt,
    load_processed_bfcl_examples,
    prepare_bfcl_data,
)
from src.extraction import strip_thinking_suffix
from src.scripts.extract.extract_activations import extract_activations
from src.scripts.probes.train_language_probes import (
    PROBE_TYPE_RIDGE_SCALED,
    assign_grouped_splits,
    default_alpha_grid_for_probe_type,
    fit_layer_probe,
    parse_float_grid,
    probe_bundle_path,
    regression_metrics,
    safe_path_part,
)
from src.probes import ScaledRidgeProbe


MODEL_CONFIGS = [
    {
        "model_name": "Qwen/Qwen3-0.6B",
        "output_tag": "Qwen3-0.6B",
        "reasoning_effort": None,
        "enable_thinking": None,
    },
    {
        "model_name": "Qwen/Qwen3-1.7B",
        "output_tag": "Qwen3-1.7B",
        "reasoning_effort": None,
        "enable_thinking": None,
    },
    {
        "model_name": "Qwen/Qwen3-4B",
        "output_tag": "Qwen3-4B",
        "reasoning_effort": None,
        "enable_thinking": None,
    },
    {
        "model_name": "Qwen/Qwen3-8B",
        "output_tag": "Qwen3-8B",
        "reasoning_effort": None,
        "enable_thinking": None,
    },
    {
        "model_name": "Qwen/Qwen3.5-4B",
        "output_tag": "Qwen3.5-4B",
        "reasoning_effort": None,
        "enable_thinking": None,
    },
    {
        "model_name": "Qwen/Qwen3.5-9B",
        "output_tag": "Qwen3.5-9B",
        "reasoning_effort": None,
        "enable_thinking": None,
    },
    {
        "model_name": "openai/gpt-oss-20b",
        "output_tag": "gpt-oss-20b_low",
        "reasoning_effort": "low",
        "enable_thinking": False,
        "use_fc_template": True,
        # Registry key used when running the official BFCL FC handler;
        # determines the subdirectory name in all_categories_32k_5gen/.
        "run_stem": "openai_gpt-oss-20b_low-FC",
    },
    {
        "model_name": "openai/gpt-oss-20b",
        "output_tag": "gpt-oss-20b_high",
        "reasoning_effort": "high",
        "enable_thinking": False,
        "use_fc_template": True,
        "run_stem": "openai_gpt-oss-20b_high-FC",
    },
]


def model_config(tag: str) -> dict[str, Any]:
    for cfg in MODEL_CONFIGS:
        if cfg["output_tag"] == tag or cfg["model_name"] == tag:
            return cfg
    raise ValueError(f"Unknown model {tag!r}. Known tags: {[c['output_tag'] for c in MODEL_CONFIGS]}")


def parse_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [x.strip() for x in value.split(",") if x.strip()]
    return items or None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_array(example_ids: list[str], split_by_id: dict[str, str]) -> np.ndarray:
    mapping = {"train": "probe_train", "validation": "val", "test": "test"}
    return np.asarray([mapping[split_by_id[str(pid)]] for pid in example_ids])


def bfcl_id_aliases(example_id: str) -> list[str]:
    """Return equivalent BFCL ids across local processed data and run CSVs."""
    example_id = str(example_id)
    if example_id.startswith("simple_python_"):
        return [example_id, f"simple_{example_id.removeprefix('simple_python_')}"]
    if example_id.startswith("simple_"):
        return [example_id, f"simple_python_{example_id.removeprefix('simple_')}"]
    return [example_id]


def safe_auroc(y: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> float:
    y_bin = (np.asarray(y) > threshold).astype(int)
    if len(y_bin) == 0 or y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
        return float("nan")
    return float(roc_auc_score(y_bin, scores))


def bfcl_generation_path(output_dir: Path, model: str) -> Path:
    return output_dir / f"{safe_path_part(model)}.jsonl"


def bfcl_eval_path(output_dir: Path, model: str) -> Path:
    return output_dir / f"{safe_path_part(model)}_per_example.csv"


def bfcl_activation_path(root: Path, model: str, token_position: int) -> Path:
    return root / safe_path_part(model) / f"activations_token_{token_position}.joblib"


def math_activation_path(root: Path, model: str, token_position: int) -> Path:
    return root / "English" / safe_path_part(model) / f"activations_token_{token_position}.joblib"


def stage_prepare(args: argparse.Namespace) -> None:
    categories = parse_csv(args.categories) or list(BFCL_DEFAULT_CATEGORIES)
    examples = prepare_bfcl_data(
        args.processed_path,
        categories=categories,
        cache_dir=str(args.cache_dir),
        local_data_root=args.local_data_root,
        seed=args.random_seed,
        max_examples=args.max_examples,
    )
    counts = Counter(ex["split"] for ex in examples)
    print(f"Saved {len(examples)} BFCL examples to {args.processed_path}")
    print("Splits:", dict(counts))


def stage_generate(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    cfg = model_config(args.model)
    examples = load_processed_bfcl_examples(args.processed_path, max_examples=args.max_examples)
    base_model = strip_thinking_suffix(cfg["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    prompts = [
        format_bfcl_chat_prompt(
            ex,
            tokenizer,
            enable_thinking=cfg.get("enable_thinking", False),
            reasoning_effort=cfg.get("reasoning_effort"),
            use_fc_template=cfg.get("use_fc_template", False),
        )
        for ex in examples
    ]
    llm = LLM(
        model=base_model,
        trust_remote_code=True,
        dtype="auto",
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
        max_model_len=args.max_model_len,
        max_num_seqs=min(args.batch_size, 64),
        tensor_parallel_size=args.tensor_parallel_size,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n=args.n_generations,
    )
    rows = []
    for start in range(0, len(prompts), args.batch_size):
        batch_prompts = prompts[start : start + args.batch_size]
        batch_examples = examples[start : start + args.batch_size]
        outputs = llm.generate(batch_prompts, sampling_params)
        for ex, out in zip(batch_examples, outputs):
            rows.append(
                {
                    "example_id": ex["example_id"],
                    "category": ex["category"],
                    "split": ex["split"],
                    "user_prompt": ex["user_prompt"],
                    "functions": ex["functions"],
                    "ground_truth": ex["ground_truth"],
                    "model": cfg["output_tag"],
                    "model_name": cfg["model_name"],
                    "generated_solutions": [
                        {
                            "rollout_index": i,
                            "text": completion.text,
                            "finish_reason": completion.finish_reason,
                            "output_tokens": len(completion.token_ids),
                        }
                        for i, completion in enumerate(out.outputs)
                    ],
                }
            )
    path = bfcl_generation_path(args.generations_dir, cfg["output_tag"])
    write_jsonl(path, rows)
    print(f"Saved {len(rows)} BFCL generation rows to {path}")


def stage_evaluate(args: argparse.Namespace) -> None:
    path = bfcl_generation_path(args.generations_dir, args.model)
    rows = read_jsonl(path)
    per_generation = []
    per_example = []
    for row in rows:
        generations = row.get("generated_solutions", [])
        correct = []
        for gen in generations:
            is_correct = bfcl_ast_correct(gen.get("text", ""), row.get("ground_truth"))
            correct.append(int(is_correct))
            per_generation.append(
                {
                    "example_id": row["example_id"],
                    "model": row["model"],
                    "category": row["category"],
                    "split": row["split"],
                    "rollout_index": gen.get("rollout_index"),
                    "correct": int(is_correct),
                    "text": gen.get("text", ""),
                }
            )
        k = int(sum(correct))
        n = int(len(correct))
        per_example.append(
            {
                "example_id": row["example_id"],
                "model": row["model"],
                "category": row["category"],
                "split": row["split"],
                "k_correct": k,
                "n_generations": n,
                "empirical_success": float(k / n) if n else float("nan"),
            }
        )
    args.evaluations_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_generation).to_csv(
        args.evaluations_dir / f"{safe_path_part(args.model)}_per_generation.csv",
        index=False,
    )
    out_path = bfcl_eval_path(args.evaluations_dir, args.model)
    pd.DataFrame(per_example).to_csv(out_path, index=False)
    print(f"Saved BFCL per-example success to {out_path}")


def stage_extract(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = model_config(args.model)
    base_model = strip_thinking_suffix(cfg["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    examples = load_processed_bfcl_examples(args.processed_path, max_examples=args.max_examples)
    if args.prompt_handler != "qwen-manual":
        raise ValueError(
            "Final BFCL extraction only supports --prompt-handler qwen-manual so "
            "Qwen3 and Qwen3.5 use the same generation prompt format."
        )
    if not cfg["model_name"].startswith("Qwen/Qwen3"):
        raise ValueError("qwen-manual extraction is restricted to Qwen3/Qwen3.5 models.")
    prompts = [format_bfcl_qwen_manual_prompt(ex) for ex in examples]
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=args.dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    activations = extract_activations(
        model=model,
        tokenizer=tokenizer,
        texts=prompts,
        batch_size=args.batch_size,
        max_length=args.max_length,
        token_position=args.token_position,
        capture_token_only=True,
    )
    path = bfcl_activation_path(args.activations_root, cfg["output_tag"], args.token_position)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "activations": activations,
        "problem_ids": [ex["example_id"] for ex in examples],
        "categories": [ex["category"] for ex in examples],
        "splits": [ex["split"] for ex in examples],
        "model": cfg["output_tag"],
        "task": "BFCL",
        "prompt_handler": args.prompt_handler,
        "token_position": args.token_position,
        "max_length": args.max_length,
        "attention_implementation": "sdpa",
        "capture_token_only": True,
    }
    success_path = bfcl_eval_path(args.evaluations_dir, cfg["output_tag"])
    if success_path.exists():
        success = pd.read_csv(success_path).set_index("example_id")["empirical_success"].to_dict()
        payload["labels"] = np.asarray(
            [success.get(ex["example_id"], np.nan) for ex in examples],
            dtype=np.float32,
        )
    joblib.dump(payload, path)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Saved BFCL activations to {path}")


def load_bfcl_arrays(args: argparse.Namespace, model: str) -> dict[str, Any]:
    act_path = bfcl_activation_path(args.activations_root, model, args.token_position)
    if not act_path.exists():
        raise FileNotFoundError(
            f"Missing BFCL activation cache for {model}: {act_path}. "
            "Run the extract stage first."
        )
    eval_path = bfcl_eval_path(args.evaluations_dir, model)
    if not eval_path.exists():
        raise FileNotFoundError(
            f"Missing BFCL per-example evaluation CSV for {model}: {eval_path}. "
            "Run evaluate or import-runs first."
        )
    cache = joblib.load(act_path)
    eval_df = pd.read_csv(eval_path)
    y_by_id: dict[str, float] = {}
    split_by_id: dict[str, str] = {}
    for row in eval_df.itertuples(index=False):
        for alias in bfcl_id_aliases(str(row.example_id)):
            y_by_id[alias] = float(row.empirical_success)
            split_by_id[alias] = str(row.split)
    ids = [str(x) for x in cache["problem_ids"]]
    y = np.asarray([y_by_id.get(pid, np.nan) for pid in ids], dtype=np.float32)
    keep = ~np.isnan(y)
    splits = split_array(ids, split_by_id)
    return {
        "cache": cache,
        "ids": np.asarray(ids)[keep],
        "keep_idx": np.flatnonzero(keep),
        "y": y[keep],
        "splits": splits[keep],
        "categories": np.asarray(cache.get("categories", ["unknown"] * len(ids)))[keep],
        "activations": cache["activations"],
    }


def evaluate_math_on_bfcl(args: argparse.Namespace, model: str) -> list[dict[str, Any]]:
    data = load_bfcl_arrays(args, model)
    if args.transfer_eval_split == "all":
        eval_mask = np.ones(len(data["y"]), dtype=bool)
    else:
        eval_mask = data["splits"] == args.transfer_eval_split
    if not eval_mask.any():
        raise ValueError(
            f"No BFCL examples found for --transfer-eval-split={args.transfer_eval_split!r}."
        )
    bundle_path = probe_bundle_path(args.math_probe_dir, "English", model, args.math_probe_type)
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"Missing MATH-trained {args.math_probe_type} probe for {model}: {bundle_path}. "
            "Set --math-probe-dir to the directory containing probes_ridge_scaled/."
        )
    bundle = joblib.load(bundle_path)
    rows = []
    n_layers = len(bundle["layers"])
    for layer, payload in bundle["layers"].items():
        layer = int(layer)
        if layer not in data["activations"]:
            continue
        X = np.asarray(data["activations"][layer])[data["keep_idx"]][eval_mask]
        pred = np.clip(payload["probe"].predict(X.astype(np.float32, copy=False)), 0.0, 1.0)
        auroc = safe_auroc(data["y"][eval_mask], pred, threshold=args.auroc_threshold)
        rows.append(
            {
                "model": model,
                "train_task": "MATH",
                "eval_task": "BFCL",
                "layer": layer,
                "layer_frac": float(layer / (n_layers - 1)) if n_layers > 1 else 0.0,
                "metric": "auroc",
                "value": auroc,
                "n_examples": int(eval_mask.sum()),
                "bfcl_eval_split": args.transfer_eval_split,
                "probe_path": str(bundle_path),
            }
        )
    return rows


def train_bfcl_on_bfcl(args: argparse.Namespace, model: str) -> list[dict[str, Any]]:
    data = load_bfcl_arrays(args, model)
    alpha_grid = parse_float_grid(args.alpha_grid) if args.alpha_grid else default_alpha_grid_for_probe_type(args.probe_type)
    rows = []
    layer_payloads = {}
    layers = sorted(int(k) for k in data["activations"].keys())
    n_layers = len(layers)
    for layer in layers:
        X = np.asarray(data["activations"][layer])[data["keep_idx"]]
        record, probe = fit_layer_probe(
            X,
            data["y"],
            data["splits"],
            alpha_grid=alpha_grid,
            auroc_threshold=args.auroc_threshold,
            probe_type=args.probe_type,
            selection_metric="auroc",
        )
        out = {
            "model": model,
            "train_task": "BFCL",
            "eval_task": "BFCL",
            "layer": layer,
            "layer_frac": float(layer / (n_layers - 1)) if n_layers > 1 else 0.0,
            "metric": "auroc",
            "value": record["test_auroc"],
            "n_examples": int(len(data["y"])),
            "n_probe_train": int((data["splits"] == "probe_train").sum()),
            "n_val": int((data["splits"] == "val").sum()),
            "n_test": int((data["splits"] == "test").sum()),
            "probe_type": args.probe_type,
            **record,
        }
        rows.append(out)
        layer_payloads[layer] = {"probe": probe, "record": out}
    bundle_path = args.output_dir / f"bfcl_probes_{safe_path_part(args.probe_type)}" / safe_path_part(model) / "English.joblib"
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "metadata": {
                "task": "BFCL",
                "language": "English",
                "model": model,
                "probe_type": args.probe_type,
                "alpha_grid": alpha_grid,
                "auroc_threshold": args.auroc_threshold,
            },
            "layers": layer_payloads,
        },
        bundle_path,
    )
    return rows


def load_math_arrays(args: argparse.Namespace, model: str) -> dict[str, Any]:
    act_path = math_activation_path(args.math_activations_root, model, args.token_position)
    if not act_path.exists():
        raise FileNotFoundError(
            f"Missing English MATH activation cache for {model}: {act_path}."
        )
    success_path = args.math_success_dir / f"{safe_path_part(model)}_English.csv"
    if not success_path.exists():
        raise FileNotFoundError(
            f"Missing English MATH success CSV for {model}: {success_path}."
        )

    cache = joblib.load(act_path)
    success_df = pd.read_csv(success_path)
    if "problem_id" not in success_df.columns or "success_rate" not in success_df.columns:
        raise ValueError(f"MATH success CSV has unexpected columns: {success_path}")

    ids = pd.Series(cache["problem_ids"], name="problem_id").astype(str).reset_index()
    merged = ids.merge(
        success_df[["problem_id", "base_problem_id", "success_rate"]],
        on="problem_id",
        how="left",
    )
    keep = merged["success_rate"].notna().to_numpy()
    base_problem_ids = merged.loc[keep, "base_problem_id"].astype(str).to_numpy()
    splits = assign_grouped_splits(
        base_problem_ids,
        split_fracs=(0.70, 0.15, 0.15),
        random_seed=args.random_seed,
    )
    return {
        "cache": cache,
        "ids": merged.loc[keep, "problem_id"].astype(str).to_numpy(),
        "keep_idx": merged.loc[keep, "index"].to_numpy(dtype=int),
        "y": merged.loc[keep, "success_rate"].to_numpy(dtype=np.float32),
        "splits": splits,
        "activations": cache["activations"],
    }


def domain_balanced_weights(n_math: int, n_bfcl: int) -> np.ndarray:
    n_total = n_math + n_bfcl
    if n_math == 0 or n_bfcl == 0:
        return np.ones(n_total, dtype=np.float32)
    return np.concatenate(
        [
            np.full(n_math, n_total / (2.0 * n_math), dtype=np.float32),
            np.full(n_bfcl, n_total / (2.0 * n_bfcl), dtype=np.float32),
        ]
    )


def train_mixed_on_bfcl(args: argparse.Namespace, model: str) -> list[dict[str, Any]]:
    """Train ridge-scaled probes on English MATH + BFCL train; evaluate BFCL test."""
    if args.probe_type != PROBE_TYPE_RIDGE_SCALED:
        raise ValueError("--train-mixed currently supports --probe-type ridge_scaled only.")

    math_data = load_math_arrays(args, model)
    bfcl_data = load_bfcl_arrays(args, model)
    alpha_grid = parse_float_grid(args.alpha_grid) if args.alpha_grid else default_alpha_grid_for_probe_type(args.probe_type)

    math_train = math_data["splits"] == "probe_train"
    bfcl_train = bfcl_data["splits"] == "probe_train"
    bfcl_val = bfcl_data["splits"] == "val"
    bfcl_test = bfcl_data["splits"] == "test"
    if not bfcl_val.any() or not bfcl_test.any():
        raise ValueError(f"BFCL val/test split is empty for {model}.")

    rows = []
    layer_payloads = {}
    layers = sorted(set(int(k) for k in math_data["activations"].keys()) & set(int(k) for k in bfcl_data["activations"].keys()))
    n_layers = len(layers)

    for layer in layers:
        X_math = np.asarray(math_data["activations"][layer])[math_data["keep_idx"]].astype(np.float32, copy=False)
        X_bfcl = np.asarray(bfcl_data["activations"][layer])[bfcl_data["keep_idx"]].astype(np.float32, copy=False)

        X_train = np.vstack([X_math[math_train], X_bfcl[bfcl_train]])
        y_train = np.concatenate([math_data["y"][math_train], bfcl_data["y"][bfcl_train]])
        sample_weight = domain_balanced_weights(int(math_train.sum()), int(bfcl_train.sum()))

        best = None
        for alpha in alpha_grid:
            probe = ScaledRidgeProbe(alpha=alpha)
            probe.fit(X_train, y_train, sample_weight=sample_weight)
            val_pred = np.clip(probe.predict(X_bfcl[bfcl_val]), 0.0, 1.0)
            metrics = regression_metrics(bfcl_data["y"][bfcl_val], val_pred, auroc_threshold=args.auroc_threshold)
            score = metrics["auroc"] if not np.isnan(metrics["auroc"]) else -metrics["rmse"]
            if best is None or score > best["selection_score"]:
                best = {"alpha": alpha, "probe": probe, "selection_score": score}

        if best is None:
            raise RuntimeError("No mixed probe was trained; alpha grid may be empty.")

        probe = best["probe"]
        record: dict[str, Any] = {
            "best_alpha": float(best["alpha"]),
            "selection_score": float(best["selection_score"]),
            "selection_metric": "bfcl_val_auroc",
            "probe_type": args.probe_type,
        }
        split_masks = {"probe_train": bfcl_train, "val": bfcl_val, "test": bfcl_test}
        for split_name, mask in split_masks.items():
            pred = np.clip(probe.predict(X_bfcl[mask]), 0.0, 1.0)
            metrics = regression_metrics(bfcl_data["y"][mask], pred, auroc_threshold=args.auroc_threshold)
            for metric_name, value in metrics.items():
                record[f"{split_name}_{metric_name}"] = value

        out = {
            "model": model,
            "train_task": "MATH+BFCL",
            "eval_task": "BFCL",
            "layer": layer,
            "layer_frac": float(layer / (n_layers - 1)) if n_layers > 1 else 0.0,
            "metric": "auroc",
            "value": record["test_auroc"],
            "n_examples": int(bfcl_test.sum()),
            "n_math_train": int(math_train.sum()),
            "n_bfcl_train": int(bfcl_train.sum()),
            "n_val": int(bfcl_val.sum()),
            "n_test": int(bfcl_test.sum()),
            **record,
        }
        rows.append(out)
        layer_payloads[layer] = {"probe": probe, "record": out}

    bundle_path = args.output_dir / f"mixed_probes_{safe_path_part(args.probe_type)}" / safe_path_part(model) / "English_MATH_plus_BFCL.joblib"
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "metadata": {
                "task": "MATH+BFCL",
                "language": "English",
                "model": model,
                "probe_type": args.probe_type,
                "alpha_grid": alpha_grid,
                "auroc_threshold": args.auroc_threshold,
                "math_activations_root": str(args.math_activations_root),
                "math_success_dir": str(args.math_success_dir),
            },
            "layers": layer_payloads,
        },
        bundle_path,
    )
    return rows


def bfcl_baselines(args: argparse.Namespace, model: str) -> list[dict[str, Any]]:
    data = load_bfcl_arrays(args, model)
    examples = {ex["example_id"]: ex for ex in load_processed_bfcl_examples(args.processed_path)}
    lengths = np.asarray([len(examples[pid]["user_prompt"].split()) for pid in data["ids"]], dtype=np.float32)
    rows = [
        {
            "model": model,
            "train_task": "baseline",
            "eval_task": "BFCL",
            "layer": -1,
            "layer_frac": float("nan"),
            "metric": "prompt_length_auroc",
            "value": safe_auroc(data["y"], lengths, threshold=args.auroc_threshold),
            "n_examples": int(len(data["y"])),
        },
        {
            "model": model,
            "train_task": "baseline",
            "eval_task": "BFCL",
            "layer": -1,
            "layer_frac": float("nan"),
            "metric": "random_auroc",
            "value": 0.5,
            "n_examples": int(len(data["y"])),
        },
    ]
    masks = {name: data["splits"] == name for name in ["probe_train", "test"]}
    if masks["probe_train"].sum() and masks["test"].sum() and len(set(data["categories"])) > 1:
        cat_to_idx = {cat: i for i, cat in enumerate(sorted(set(data["categories"])))}
        X = np.zeros((len(data["categories"]), len(cat_to_idx)), dtype=np.float32)
        for i, cat in enumerate(data["categories"]):
            X[i, cat_to_idx[cat]] = 1.0
        y_bin = (data["y"] > args.auroc_threshold).astype(int)
        if len(np.unique(y_bin[masks["probe_train"]])) > 1:
            clf = LogisticRegression(max_iter=1000)
            clf.fit(X[masks["probe_train"]], y_bin[masks["probe_train"]])
            scores = clf.predict_proba(X[masks["test"]])[:, 1]
            value = safe_auroc(data["y"][masks["test"]], scores, threshold=args.auroc_threshold)
        else:
            dummy = DummyClassifier(strategy="prior")
            dummy.fit(X[masks["probe_train"]], y_bin[masks["probe_train"]])
            scores = dummy.predict_proba(X[masks["test"]])[:, 1]
            value = safe_auroc(data["y"][masks["test"]], scores, threshold=args.auroc_threshold)
        rows.append(
            {
                "model": model,
                "train_task": "baseline",
                "eval_task": "BFCL",
                "layer": -1,
                "layer_frac": float("nan"),
                "metric": "category_only_auroc",
                "value": value,
                "n_examples": int(masks["test"].sum()),
            }
        )
    return rows


def load_math_reference(args: argparse.Namespace, models: list[str]) -> list[dict[str, Any]]:
    candidates = [
        args.math_probe_dir / f"per_language_layer_results_{safe_path_part(args.math_probe_type)}.csv",
        args.math_probe_dir / "per_language_layer_results.csv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return []
    df = pd.read_csv(path)
    df = df[(df["language"] == "English") & (df["model"].isin(models))]
    rows = []
    for row in df.itertuples(index=False):
        rows.append(
            {
                "model": row.model,
                "train_task": "MATH",
                "eval_task": "MATH",
                "layer": int(row.layer),
                "layer_frac": float(row.layer_frac),
                "metric": "auroc",
                "value": float(getattr(row, "test_auroc")),
                "n_examples": int(getattr(row, "n_examples", 0)),
            }
        )
    return rows


def compact_summary(layer_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    auroc = layer_df[layer_df["metric"] == "auroc"].copy()
    for model, group in auroc.groupby("model"):
        item: dict[str, Any] = {"model": model}
        for train_task, eval_task, col in [
            ("MATH", "MATH", "math_to_math"),
            ("MATH", "BFCL", "math_to_bfcl"),
            ("MATH+BFCL", "BFCL", "math_bfcl_to_bfcl"),
            ("BFCL", "BFCL", "bfcl_to_bfcl"),
        ]:
            g = group[(group["train_task"] == train_task) & (group["eval_task"] == eval_task)]
            if g.empty or g["value"].isna().all():
                item[f"{col}_peak_auroc"] = float("nan")
                item[f"{col}_peak_layer"] = float("nan")
                item[f"{col}_final_layer_auroc"] = float("nan")
                item[f"{col}_peak_to_final_drop"] = float("nan")
                continue
            peak = g.sort_values(["value", "layer"], ascending=[False, True]).iloc[0]
            final = g.sort_values("layer").iloc[-1]
            item[f"{col}_peak_auroc"] = float(peak["value"])
            item[f"{col}_peak_layer"] = int(peak["layer"])
            item[f"{col}_final_layer_auroc"] = float(final["value"])
            item[f"{col}_peak_to_final_drop"] = float(peak["value"] - final["value"])
        item["cross_task_gap"] = item.get("bfcl_to_bfcl_peak_auroc", math.nan) - item.get("math_to_bfcl_peak_auroc", math.nan)
        item["best_transfer_layer"] = item.get("math_to_bfcl_peak_layer", math.nan)
        rows.append(item)
    return pd.DataFrame(rows)


def plot_results(layer_df: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    df = layer_df[layer_df["metric"] == "auroc"].copy()
    if df.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    for model, group in df.groupby("model"):
        fig, ax = plt.subplots(figsize=(7, 4))
        plotted_values = []
        upper_left_values = []
        for train_task, eval_task, label in [
            ("MATH", "MATH", "MATH->MATH"),
            ("MATH", "BFCL", "MATH->BFCL"),
            ("MATH+BFCL", "BFCL", "MATH+BFCL->BFCL"),
            ("BFCL", "BFCL", "BFCL->BFCL"),
        ]:
            g = group[(group["train_task"] == train_task) & (group["eval_task"] == eval_task)]
            if g.empty:
                continue
            g = g.sort_values("layer")
            g = g[g["layer"] != 0]
            if g.empty:
                continue
            ax.plot(g["layer"], g["value"], marker="o", linewidth=1.5, markersize=3, label=label)
            plotted_values.extend(g["value"].dropna().astype(float).tolist())
            upper_left_values.extend(g[g["layer"] <= 8]["value"].dropna().astype(float).tolist())
        ax.axhline(0.5, color="0.5", linestyle="--", linewidth=1)
        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel("AUROC", fontsize=12)
        ax.grid(alpha=0.25)
        if plotted_values:
            needs_legend_headroom = (
                model in {"Qwen3-0.6B", "Qwen3-8B"}
                and upper_left_values
                and max(upper_left_values) > max(plotted_values) - 0.08
            )
            y_top_padding = 0.10 if needs_legend_headroom else 0.05
            ax.set_ylim(min(plotted_values) - 0.05, max(plotted_values) + y_top_padding)
        ax.legend(
            frameon=True,
            loc="upper left",
            ncol=1,
            fontsize=9,
            handlelength=1.5,
            columnspacing=1.2,
            facecolor="white",
            edgecolor="none",
            framealpha=0.85,
        )
        fig.tight_layout()
        fig.savefig(output_dir / f"bfcl_transfer_auroc_{safe_path_part(model)}.png", dpi=200, bbox_inches="tight")
        fig.savefig(output_dir / f"bfcl_transfer_auroc_{safe_path_part(model)}.pdf", bbox_inches="tight")
        plt.close(fig)


def stage_evaluate_probes(args: argparse.Namespace) -> None:
    models = parse_csv(args.models) or [args.model]
    rows: list[dict[str, Any]] = []
    for model in models:
        rows.extend(evaluate_math_on_bfcl(args, model))
        if args.train_mixed:
            rows.extend(train_mixed_on_bfcl(args, model))
        if args.train_bfcl:
            rows.extend(train_bfcl_on_bfcl(args, model))
        if args.include_baselines:
            rows.extend(bfcl_baselines(args, model))
    if args.include_math_reference:
        rows.extend(load_math_reference(args, models))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    layer_df = pd.DataFrame(rows)
    layer_path = args.output_dir / "bfcl_cross_task_layer_metrics.csv"
    layer_df.to_csv(layer_path, index=False)
    summary = compact_summary(layer_df)
    summary_path = args.output_dir / "bfcl_cross_task_summary.csv"
    summary.to_csv(summary_path, index=False)
    plot_results(layer_df, args.output_dir / "figures")
    print(f"Saved layer metrics to {layer_path}")
    print(f"Saved compact summary to {summary_path}")


def _derive_run_stem(cfg: dict) -> str:
    """Derive the directory name used in bfcl-runs for a MODEL_CONFIGS entry.

    Uses ``run_stem`` when present (e.g. GPT-OSS FC variants), otherwise
    builds from ``model_name`` with the low/high suffix appended when the
    output_tag disambiguates two entries sharing the same model_name.
    """
    if "run_stem" in cfg:
        return cfg["run_stem"]
    stem = cfg["model_name"].replace("/", "_")
    tag = cfg["output_tag"]
    for suffix in ("_low", "_high"):
        if tag.endswith(suffix):
            stem = stem + suffix
            break
    return stem


def _candidate_run_stems(stem: str) -> list[str]:
    stems = [stem]
    handler_suffixes = {
        "Qwen_Qwen3.5-4B": ["qwen3-matched", "chat-template-no-thinking"],
        "Qwen_Qwen3.5-9B": [
            "qwen3-matched",
            "chat-template-no-thinking-recommended-sampling",
            "chat-template-no-thinking",
        ],
    }
    for suffix in handler_suffixes.get(stem, []):
        stems.append(f"{stem}-{suffix}")
    return stems


def _load_category_csv(
    src_dir: Path,
    stem: str,
    category: str,
    postprocessed: bool = False,
) -> pd.DataFrame | None:
    """Return the empirical-success DataFrame for one model+category, or None."""
    if postprocessed:
        # Legacy postprocessed layout: stem/stem_cat_5gen_empirical_success_postprocessed.csv
        globs = [
            src_dir / stem / f"*_{category}_*empirical_success_postprocessed.csv",
            src_dir / f"*_{category}_*empirical_success_postprocessed.csv",
        ]
    else:
        # New all_categories layout: stem/category/stem_cat_Ngen_empirical_success.csv
        # Also handles legacy: stem/stem_cat_5gen_empirical_success.csv
        globs = [
            src_dir / stem / category / f"*_{category}_*empirical_success.csv",
            src_dir / stem / f"*_{category}_*empirical_success.csv",
        ]
    for pattern in globs:
        matches = sorted(pattern.parent.glob(pattern.name)) if pattern.parent.exists() else []
        if matches:
            return pd.read_csv(matches[0])
    return None


def stage_import_runs(args: argparse.Namespace) -> None:
    """Convert pre-existing bfcl-runs empirical_success CSVs into the format
    expected by the extract / evaluate-probes stages.

    Two directory layouts are supported:

    **Legacy** (``--runs-dir``)::

        <runs_dir>/
          <stem>/<stem>_<category>_5gen_empirical_success.csv

    **All-categories** (``--all-categories-dir``)::

        <all_categories_dir>/
          <stem>/<category>/<stem>_<category>_Ngen_empirical_success.csv

    ``--categories`` (comma-separated) selects which categories to import;
    all are concatenated into a single evaluation CSV per model.
    When ``--gpt-oss-postprocessed-dir`` is given, GPT-OSS entries are read
    from that directory instead (supports the legacy postprocessed format).
    """
    categories = [c.strip() for c in args.categories.split(",") if c.strip()]

    args.evaluations_dir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    selected_models = set(parse_csv(args.models) or [])
    for cfg in MODEL_CONFIGS:
        if selected_models and cfg["output_tag"] not in selected_models:
            continue
        stem = _derive_run_stem(cfg)
        output_tag = cfg["output_tag"]
        is_gpt_oss = "gpt-oss" in stem

        frames: list[pd.DataFrame] = []
        for category in categories:
            candidate_stems = _candidate_run_stems(stem)
            df = None
            # Decide which source directory to use.
            for candidate_stem in candidate_stems:
                if is_gpt_oss and args.gpt_oss_postprocessed_dir:
                    df = _load_category_csv(
                        args.gpt_oss_postprocessed_dir, candidate_stem, category, postprocessed=True
                    )
                    src_label = str(args.gpt_oss_postprocessed_dir)
                elif args.all_categories_dir:
                    df = _load_category_csv(args.all_categories_dir, candidate_stem, category)
                    # Fall back to legacy runs_dir if not found in all_categories_dir
                    if df is None and args.runs_dir:
                        df = _load_category_csv(args.runs_dir, candidate_stem, category)
                    src_label = str(args.all_categories_dir)
                else:
                    df = _load_category_csv(args.runs_dir, candidate_stem, category)
                    src_label = str(args.runs_dir)
                if df is not None:
                    break

            if df is None:
                print(f"  [skip] {output_tag}  cat={category}: no CSV in {src_label}")
                continue
            frames.append(df)

        if not frames:
            print(f"  [skip] {output_tag}: no data found for any requested category")
            continue

        df = pd.concat(frames, ignore_index=True)
        df = df.rename(columns={"model": "_src_model"})
        df["model"] = output_tag
        df["split"] = df["example_id"].apply(lambda eid: assign_split(str(eid)))

        out_cols = ["example_id", "model", "category", "split",
                    "k_correct", "n_generations", "empirical_success"]
        for col in out_cols:
            if col not in df.columns:
                raise ValueError(f"Missing column {col!r} in data for {output_tag}")

        out_path = bfcl_eval_path(args.evaluations_dir, output_tag)
        if args.append and out_path.exists():
            existing = pd.read_csv(out_path)
            df = pd.concat([existing, df[out_cols]], ignore_index=True)
            df = df.drop_duplicates(subset=["example_id"], keep="last")
        df[out_cols].to_csv(out_path, index=False)
        cats_found = sorted(df["category"].unique().tolist())
        mean_succ = df["empirical_success"].mean()
        print(
            f"  {output_tag:30s}  n={len(df):4d}  cats={cats_found}"
            f"  mean_success={mean_succ:.4f}  → {out_path}"
        )
        written.append(output_tag)

    print(f"\nimport-runs: wrote {len(written)} evaluation files to {args.evaluations_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="English-only BFCL cross-task transfer experiment.")
    sub = parser.add_subparsers(dest="stage", required=True)

    import_runs = sub.add_parser(
        "import-runs",
        help="Import pre-existing bfcl-runs empirical_success CSVs into the evaluations directory.",
    )
    import_runs.add_argument(
        "--runs-dir",
        type=lambda p: Path(p) if p else None,
        default=None,
        help="Legacy per-model directory with empirical_success CSVs (simple_python).",
    )
    import_runs.add_argument(
        "--models",
        type=str,
        default="Qwen3-0.6B,Qwen3-1.7B,Qwen3-4B,Qwen3-8B,Qwen3.5-4B,Qwen3.5-9B",
        help="Comma-separated aligned Qwen models to import.",
    )
    import_runs.add_argument(
        "--all-categories-dir",
        type=lambda p: Path(p) if p else None,
        default=None,
        help="Root of the all_categories_32k_5gen run tree produced by "
             "run_bfcl_all_categories.sh.  Takes precedence over --runs-dir "
             "when both are given.",
    )
    import_runs.add_argument(
        "--gpt-oss-postprocessed-dir",
        type=lambda p: Path(p) if p else None,
        default=None,
        help="Directory with postprocessed GPT-OSS FC CSVs (correct scores). "
             "Pass empty string to disable.",
    )
    import_runs.add_argument(
        "--categories",
        type=str,
        default="simple_python",
        help="Comma-separated list of BFCL categories to import, e.g. "
             "'simple_python' or 'simple_python,multiple,parallel,parallel_multiple'. "
             "All selected categories are concatenated into one CSV per model.",
    )
    import_runs.add_argument("--evaluations-dir", type=Path, default=Path("outputs/bfcl/evaluations"))
    import_runs.add_argument(
        "--append",
        action="store_true",
        help="Append categories to an existing model evaluation file.",
    )
    import_runs.set_defaults(func=stage_import_runs)

    prepare = sub.add_parser("prepare", help="Prepare BFCL examples from Hugging Face JSON files.")
    prepare.add_argument("--processed-path", type=Path, default=Path("data/bfcl/processed_bfcl.jsonl"))
    prepare.add_argument("--cache-dir", type=Path, default=Path("data/hf_cache"))
    prepare.add_argument(
        "--local-data-root",
        type=Path,
        default=None,
        help="Optional local bfcl_eval/data directory to read BFCL JSONL files from.",
    )
    prepare.add_argument("--categories", type=str, default=",".join(BFCL_DEFAULT_CATEGORIES))
    prepare.add_argument("--random-seed", type=int, default=42)
    prepare.add_argument("--max-examples", type=int, default=None)
    prepare.set_defaults(func=stage_prepare)

    generate = sub.add_parser("generate", help="Generate BFCL function-call outputs with vLLM.")
    generate.add_argument("--processed-path", type=Path, default=Path("data/bfcl/processed_bfcl.jsonl"))
    generate.add_argument("--generations-dir", type=Path, default=Path("outputs/bfcl/generations"))
    generate.add_argument("--model", type=str, required=True)
    generate.add_argument("--max-examples", type=int, default=None)
    generate.add_argument("--n-generations", type=int, default=5)
    generate.add_argument("--batch-size", type=int, default=16)
    generate.add_argument("--temperature", type=float, default=0.7)
    generate.add_argument("--top-p", type=float, default=0.95)
    generate.add_argument("--max-tokens", type=int, default=512)
    generate.add_argument("--max-model-len", type=int, default=4096)
    generate.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    generate.add_argument("--tensor-parallel-size", type=int, default=1)
    generate.set_defaults(func=stage_generate)

    evaluate = sub.add_parser("evaluate", help="Evaluate raw BFCL generations with an isolated AST wrapper.")
    evaluate.add_argument("--generations-dir", type=Path, default=Path("outputs/bfcl/generations"))
    evaluate.add_argument("--evaluations-dir", type=Path, default=Path("outputs/bfcl/evaluations"))
    evaluate.add_argument("--model", type=str, required=True)
    evaluate.set_defaults(func=stage_evaluate)

    extract = sub.add_parser("extract", help="Extract final-prompt-token BFCL activations.")
    extract.add_argument("--processed-path", type=Path, default=Path("data/bfcl/processed_bfcl.jsonl"))
    extract.add_argument("--evaluations-dir", type=Path, default=Path("outputs/bfcl/evaluations"))
    extract.add_argument("--activations-root", type=Path, default=Path("outputs/bfcl/activations"))
    extract.add_argument("--model", type=str, required=True)
    extract.add_argument("--max-examples", type=int, default=None)
    extract.add_argument("--batch-size", type=int, default=1)
    extract.add_argument("--max-length", type=int, default=32768)
    extract.add_argument("--token-position", type=int, default=-1)
    extract.add_argument(
        "--prompt-handler",
        choices=["qwen-manual"],
        default="qwen-manual",
        help="Use the official BFCL QwenHandler prompt path used for generation.",
    )
    extract.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    extract.set_defaults(func=stage_extract)

    probes = sub.add_parser("evaluate-probes", help="Evaluate MATH probes on BFCL and optionally train BFCL probes.")
    probes.add_argument("--processed-path", type=Path, default=Path("data/bfcl/processed_bfcl.jsonl"))
    probes.add_argument("--evaluations-dir", type=Path, default=Path("outputs/bfcl/evaluations"))
    probes.add_argument("--activations-root", type=Path, default=Path("outputs/bfcl/activations"))
    probes.add_argument("--math-activations-root", type=Path, default=Path("outputs/activations/MATH"))
    probes.add_argument("--math-success-dir", type=Path, default=Path("outputs/success_rates/MATH"))
    probes.add_argument(
        "--math-probe-dir",
        type=Path,
        default=Path("outputs/probes/language_specific"),
        help="Directory containing MATH-trained probe bundles, e.g. probes_ridge_scaled/<model>/English.joblib.",
    )
    probes.add_argument("--output-dir", type=Path, default=Path("outputs/bfcl/cross_task_transfer"))
    probes.add_argument("--model", type=str, default="Qwen3-4B")
    probes.add_argument(
        "--models",
        type=str,
        default="Qwen3-0.6B,Qwen3-1.7B,Qwen3-4B,Qwen3-8B,Qwen3.5-4B,Qwen3.5-9B",
    )
    probes.add_argument("--math-probe-type", type=str, default=PROBE_TYPE_RIDGE_SCALED)
    probes.add_argument("--probe-type", type=str, default=PROBE_TYPE_RIDGE_SCALED)
    probes.add_argument("--alpha-grid", type=str, default=None)
    probes.add_argument("--token-position", type=int, default=-1)
    probes.add_argument("--auroc-threshold", type=float, default=0.5)
    probes.add_argument("--random-seed", type=int, default=42)
    probes.add_argument(
        "--transfer-eval-split",
        type=str,
        default="test",
        choices=["probe_train", "val", "test", "all"],
        help="BFCL split used when evaluating MATH-trained probes on BFCL.",
    )
    probes.add_argument("--train-bfcl", action="store_true")
    probes.add_argument("--train-mixed", action="store_true")
    probes.add_argument("--include-baselines", action="store_true")
    probes.add_argument("--include-math-reference", action="store_true")
    probes.set_defaults(func=stage_evaluate_probes)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
