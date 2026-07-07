#!/usr/bin/env bash
# Evaluate the relative-decision-cost routing policies for every layer-depth
# fraction of the depth sweep (inputs priced at 1/6 of output tokens).
#
# Requires:
#   - the depth sweep candidates/final grids (run_depth_routing_sweep.py)
#   - exact input-token counts for the Qwen3-4B candidates (annotate_input_tokens.py):
#       outputs/routing/candidates_with_input_tokens/Qwen3-4B/router_candidates_with_input_tokens.csv
set -euo pipefail

SWEEP_ROOT="outputs/routing/depth_sweep"
OUT_ROOT="outputs/routing/relative_decision_cost"
INPUT_TOKENS="outputs/routing/candidates_with_input_tokens/Qwen3-4B/router_candidates_with_input_tokens.csv"

for frac in $(seq 0 5 100); do
  sweep_id=$(printf "frac_%03d" "${frac}")
  for condition in pooled english_transfer; do
    output_dir="${OUT_ROOT}/${sweep_id}/${condition}"
    if [[ -f "${output_dir}/run_config.json" ]]; then
      continue
    fi
    python -m src.scripts.routing.evaluate_depth_routing_costs \
      --candidates "${SWEEP_ROOT}/candidates/${sweep_id}/${condition}/router_candidates.csv" \
      --input-tokens-from "${INPUT_TOKENS}" \
      --parameter-grid-from "${SWEEP_ROOT}/final/${sweep_id}/policy_grid_val_test.csv" \
      --output-dir "${output_dir}" \
      --probe-cost-fraction 0 \
      --input-price-fraction 0.16666666666666666 \
      --output-cost-col mean_output_cost_usd
  done
done
