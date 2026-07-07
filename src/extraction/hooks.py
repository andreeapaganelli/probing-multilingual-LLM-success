"""Hook-based single-layer activation extraction.

Registers a forward hook on one decoder layer at a time to minimize VRAM usage.
Safe for 32 GB cards even with large models.
"""

import time
from typing import Any, Callable, List, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_decoder_layers(model: Any):
    """Return the nn.ModuleList of decoder layers."""
    candidates = [
        getattr(getattr(model, "model", None), "layers", None),
        getattr(getattr(model, "transformer", None), "h", None),
        getattr(getattr(model, "gpt_neox", None), "layers", None),
    ]
    for layers in candidates:
        if layers is not None:
            return layers
    raise ValueError("Could not find decoder layers on model for hook-based extraction.")


def resolve_hook_module(model: Any, target_layer: int):
    """Return the nn.Module to hook for the given target_layer index.

    target_layer == 0  -> embedding layer
    target_layer == k  -> decoder block k-1 (1-indexed)
    """
    if target_layer < 0:
        raise ValueError("target_layer must be >= 0")
    if target_layer == 0:
        return model.get_input_embeddings()
    layers = get_decoder_layers(model)
    layer_idx = target_layer - 1
    if layer_idx < 0 or layer_idx >= len(layers):
        raise IndexError(
            f"target_layer={target_layer} out of range. "
            f"Expected [0, {len(layers)}] where 0 is embeddings and 1..L are decoder layers."
        )
    return layers[layer_idx]


@torch.no_grad()
def extract_activations_at_last_token(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: List[str],
    target_layer: int,
    batch_size: int = 1,
    max_length: int = 4096,
    checkpoint_interval_seconds: float = 900.0,
    checkpoint_callback: Optional[Callable[[dict], None]] = None,
    resume_from: int = 0,
    existing_acts: Optional[list[np.ndarray]] = None,
) -> np.ndarray:
    """Extract residual-stream activations at the LAST non-padding token using a
    forward-hook on a single layer.

    Supports periodic checkpointing and resuming from a previous checkpoint.
    Returns np.ndarray of shape (N, d_model).
    """
    input_device = next(model.parameters()).device
    all_acts: list[np.ndarray] = list(existing_acts) if existing_acts else []

    hook_module = resolve_hook_module(model, target_layer)
    captured: dict[str, torch.Tensor | None] = {"hs": None}

    def _capture_hook(_module, _inputs, output):
        if isinstance(output, tuple):
            captured["hs"] = output[0]
        else:
            captured["hs"] = output

    handle = hook_module.register_forward_hook(_capture_hook)

    effective_bs = max(1, int(batch_size))
    last_checkpoint_time = time.monotonic()

    try:
        cursor = resume_from
        progress = tqdm(
            total=len(texts),
            initial=resume_from,
            desc="Extracting activations",
        )

        while cursor < len(texts):
            current_bs = min(effective_bs, len(texts) - cursor)
            batch_texts = texts[cursor : cursor + current_bs]

            try:
                enc = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                )
                input_ids = enc["input_ids"].to(input_device)
                attention_mask = enc["attention_mask"].to(input_device)

                captured["hs"] = None
                _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                    use_cache=False,
                )

                layer_hs = captured["hs"]
                if layer_hs is None:
                    raise RuntimeError("Hook did not capture hidden states for target layer.")

                lengths = attention_mask.sum(dim=1)
                B = input_ids.shape[0]
                T = input_ids.shape[1]
                idx = (lengths - 1).clamp(min=0, max=T - 1)
                feats = layer_hs[torch.arange(B, device=layer_hs.device), idx]
                all_acts.append(feats.float().cpu().numpy())

                del layer_hs, feats, input_ids, attention_mask
                captured["hs"] = None
                torch.cuda.empty_cache()

                cursor += current_bs
                progress.update(current_bs)

                now = time.monotonic()
                if checkpoint_callback is not None and (
                    (now - last_checkpoint_time) >= checkpoint_interval_seconds
                    or cursor >= len(texts)
                ):
                    checkpoint_callback({
                        "all_acts": all_acts,
                        "processed_examples": cursor,
                        "total_examples": len(texts),
                    })
                    last_checkpoint_time = now

            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                if effective_bs <= 1:
                    raise
                new_bs = max(1, effective_bs // 2)
                effective_bs = new_bs
                print(f"OOM at batch_size={current_bs}; retrying with batch_size={new_bs}.")

    finally:
        progress.close()
        handle.remove()

    if not all_acts:
        return np.array([])
    return np.concatenate(all_acts, axis=0)
