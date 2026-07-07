#!/usr/bin/env bash
# Train and evaluate encoder-target decoupled (ETD) probes for every encoder
# model at every layer, in both pooled and English-transfer conditions.
set -euo pipefail

ROOT="outputs/etd/all_encoders_all_layers"
FIG_ROOT="outputs/figures/etd/all_encoders_all_layers"
CANDIDATE_ROOT="outputs/routing/candidates_with_input_tokens"
ACTIVATIONS="outputs/activations/MATH"
PARAM_GRID="outputs/routing/relative_decision_cost/frac_100/pooled/raw_policy_grid_val_test.csv"
ALPHAS="1,10,100,1000,10000,100000,1000000"

run_encoder() {
  local encoder="$1"
  local max_layer="$2"
  local layers
  local english_baseline
  layers="$(seq -s, 0 "${max_layer}")"
  english_baseline="$(( (7 * max_layer + 5) / 10 ))"
  local encoder_root="${ROOT}/${encoder}"

  for condition in pooled english_transfer; do
    local output_dir="${encoder_root}/${condition}"
    if [[ -f "${output_dir}/run_config.json" ]]; then
      continue
    fi
    local extra_args=()
    local baseline_layer="${max_layer}"
    if [[ "${condition}" == "english_transfer" ]]; then
      extra_args+=(--train-languages English)
      baseline_layer="${english_baseline}"
    fi
    if [[ -d "${output_dir}" ]]; then
      extra_args+=(--resume)
    fi
    python -m src.scripts.etd.train_layer_ensemble_probes \
      --candidates "${CANDIDATE_ROOT}/${encoder}/router_candidates_with_input_tokens.csv" \
      --activation-root "${ACTIVATIONS}" \
      --output-dir "${output_dir}" \
      --encoder-model "${encoder}" \
      --layers "${layers}" \
      --baseline-layer "${baseline_layer}" \
      --alpha-grid "${ALPHAS}" \
      --target-jobs 8 \
      "${extra_args[@]}"
  done

  python -m src.scripts.etd.evaluate_etd_single_layers \
    --experiment-root "${encoder_root}" \
    --parameter-grid-from "${PARAM_GRID}" \
    --output-root "${encoder_root}/single_layer_routing" \
    --figure-dir "${FIG_ROOT}/${encoder}"
}

run_encoder "Qwen3-0.6B" 28
run_encoder "Qwen3-1.7B" 28
run_encoder "Qwen3-4B" 36
run_encoder "Qwen3-8B" 36
run_encoder "Qwen3.5-4B" 32
run_encoder "Qwen3.5-9B" 32
run_encoder "gpt-oss-20b_low" 24
run_encoder "gpt-oss-20b_high" 24
