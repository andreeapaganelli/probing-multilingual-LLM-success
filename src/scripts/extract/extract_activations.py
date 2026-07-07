"""
Extract residual-stream activations using raw HuggingFace models.

Generic dataset-loader-based extraction (used for the BFCL branch; the
multilingual MATH extraction lives in extract_math_activations.py).
Works for all architectures including MoE (gpt-oss-20b).
Outputs are .joblib caches:
  {"activations": {layer_idx: np.ndarray(N, d_model)}}

For gpt-oss-20b reasoning-effort variants (_low/_high), the base model
`openai/gpt-oss-20b` is loaded once and the variant suffix only affects
which dataset config (and thus which labels) are used.

Usage:
    python -m src.scripts.extract.extract_activations \
        --model Qwen/Qwen3-4B \
        --dataset bfcl \
        --output-dir outputs/bfcl/activations/Qwen3-4B
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import get_loader
from src.extraction import strip_thinking_suffix


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
    capture_token_only: bool = False,
) -> dict[int, np.ndarray]:
    """
    Extract residual-stream activations at `token_position` for every layer.

    Returns dict mapping layer_index -> np.ndarray of shape (N, d_model).
    """
    input_device = next(model.parameters()).device
    all_hidden: dict[int, list[np.ndarray]] = {}

    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting"):
        batch_texts = texts[i : i + batch_size]
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc["input_ids"].to(input_device)
        attention_mask = enc["attention_mask"].to(input_device)

        # Resolve token position per example (handle padding)
        lengths = attention_mask.sum(dim=1)  # (B,)
        B = input_ids.shape[0]
        T = input_ids.shape[1]
        if token_position < 0:
            idx = (lengths + token_position).clamp(min=0, max=T - 1)
        else:
            idx = torch.full((B,), token_position, device=input_device).clamp(
                min=0, max=T - 1
            )

        if capture_token_only:
            backbone = getattr(model, "model", None)
            if backbone is None or not all(
                hasattr(backbone, name) for name in ("embed_tokens", "layers", "norm")
            ):
                raise TypeError("Token-only extraction requires a Qwen-style model backbone.")

            captured: dict[int, np.ndarray] = {}
            handles = []

            def capture(layer_idx: int):
                def hook(_module, _inputs, output):
                    hs = output[0] if isinstance(output, tuple) else output
                    row = torch.arange(B, device=hs.device)
                    local_idx = idx.to(hs.device)
                    captured[layer_idx] = hs[row, local_idx].detach().float().cpu().numpy()
                return hook

            # HF hidden_states semantics are embeddings, intermediate decoder
            # outputs, and the final normalized decoder output.
            handles.append(backbone.embed_tokens.register_forward_hook(capture(0)))
            for layer_idx, layer in enumerate(backbone.layers[:-1], start=1):
                handles.append(layer.register_forward_hook(capture(layer_idx)))
            handles.append(backbone.norm.register_forward_hook(capture(len(backbone.layers))))
            try:
                backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            finally:
                for handle in handles:
                    handle.remove()

            for layer_idx in range(len(backbone.layers) + 1):
                if layer_idx not in captured:
                    raise RuntimeError(f"Activation hook did not capture layer {layer_idx}.")
                all_hidden.setdefault(layer_idx, []).append(captured[layer_idx])
        else:
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            for layer_idx, hs in enumerate(out.hidden_states):
                row = torch.arange(B, device=hs.device)
                feats = hs[row, idx.to(hs.device)]
                all_hidden.setdefault(layer_idx, []).append(feats.float().cpu().numpy())

    # Concatenate batches
    return {k: np.concatenate(v, axis=0) for k, v in all_hidden.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract activations using HuggingFace")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model id (e.g. Qwen/Qwen3-4B or openai/gpt-oss-20b_low)",
    )
    parser.add_argument("--dataset", type=str, default="bfcl", help="Dataset key (bfcl)")
    parser.add_argument(
        "--dataset-config",
        type=str,
        default=None,
        help="Optional dataset config name (unused for bfcl).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Data directory override for the dataset loader.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save activation caches",
    )
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--test-split", type=str, default="test")
    parser.add_argument(
        "--extra-splits",
        type=str,
        nargs="*",
        default=[],
        help="Additional splits to extract (e.g. --extra-splits validation).",
    )
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-test-examples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=3000)
    parser.add_argument(
        "--token-position",
        type=int,
        default=-1,
        help="Token position to extract activations from. -1 = last token.",
    )
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        action="store_true",
        default=None,
        help="Force chat templates that support it to leave thinking enabled.",
    )
    thinking_group.add_argument(
        "--disable-thinking",
        "--no-thinking",
        dest="enable_thinking",
        action="store_false",
        help="Force chat templates that support it to disable thinking.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_split_names = [args.train_split, args.test_split] + args.extra_splits
    all_caches = {
        s: os.path.join(args.output_dir, f"activations_{s}_token_{args.token_position}.joblib")
        for s in all_split_names
    }

    if all(os.path.exists(p) for p in all_caches.values()):
        print(f"All caches already exist in {args.output_dir} — skipping extraction.")
        return

    # ---- Load tokenizer (needed before dataset for prompt formatting) ----
    base_model = strip_thinking_suffix(args.model)
    print(f"Loading tokenizer for {base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Load dataset ----
    print(f"Loading dataset {args.dataset}...")
    loader_kwargs = {}
    if args.dataset_config is not None:
        loader_kwargs["config_name"] = args.dataset_config
    if args.data_dir is not None:
        loader_kwargs["data_dir"] = args.data_dir
    # Pass tokenizer settings so prompts match generation exactly.
    # reasoning_effort is encoded in the model tag suffix (_high / _low).
    model_tag = args.model.split("/")[-1]
    if model_tag.endswith("_high"):
        loader_kwargs["reasoning_effort"] = "high"
    elif model_tag.endswith("_low"):
        loader_kwargs["reasoning_effort"] = "low"
    loader_kwargs["enable_thinking"] = (
        bool(args.enable_thinking) if args.enable_thinking is not None else False
    )
    loader_kwargs["tokenizer"] = tokenizer
    print(f"  {args.dataset} prompt enable_thinking={loader_kwargs['enable_thinking']}")
    loader = get_loader(args.dataset, **loader_kwargs)

    max_examples_map = {
        args.train_split: args.max_train_examples,
        args.test_split: args.max_test_examples,
    }
    split_texts = {}
    for s in all_split_names:
        texts, _ = loader.load_probe_fields(split=s, max_examples=max_examples_map.get(s))
        split_texts[s] = texts
    print("  " + ", ".join(f"{s}={len(split_texts[s])}" for s in all_split_names))

    # ---- Load model ----
    print(f"Loading model {base_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=args.dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    d_model = model.config.hidden_size
    print(f"  d_model={d_model}")

    # ---- Extract activations ----
    for split_name, texts, cache_path in [
        (s, split_texts[s], all_caches[s]) for s in all_split_names
    ]:
        if os.path.exists(cache_path):
            print(f"Cache exists for {split_name}, skipping: {cache_path}")
            continue

        print(f"\nExtracting {split_name} activations ({len(texts)} examples)...")
        activations = extract_activations(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            batch_size=args.batch_size,
            max_length=args.max_length,
            token_position=args.token_position,
        )
        n_layers = len(activations)
        sample_shape = activations[0].shape
        print(f"  {n_layers} layers, shape per layer: {sample_shape}")

        joblib.dump(
            {
                "activations": activations,
                "enable_thinking": bool(args.enable_thinking)
                if args.enable_thinking is not None
                else False,
            },
            cache_path,
        )
        print(f"  Saved to {cache_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
