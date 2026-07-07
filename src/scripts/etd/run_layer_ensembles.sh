#!/usr/bin/env bash
# Train the Qwen3-4B ETD mean layer ensembles used in the thesis (4 / 6 / 10 /
# all-37 layer sets), reusing the per-layer probes from
# run_all_encoders_all_layers.sh, and evaluate each ensemble's routing policies
# under the relative-decision-cost model. Afterwards run
#   python -m src.scripts.etd.plot_layer_ensemble_frontiers
# to produce the layer_ensemble_pareto_* figures.
set -euo pipefail

ROOT="outputs/etd/layer_ensembles"
CANDIDATES="outputs/routing/candidates_with_input_tokens"
ACTIVATIONS="outputs/activations/MATH"
ALL_ENCODERS="outputs/etd/all_encoders_all_layers"
PARAM_GRID="outputs/routing/relative_decision_cost/frac_100/pooled/raw_policy_grid_val_test.csv"
ALPHAS="1,10,100,1000,10000,100000,1000000"
LAMBDAS="$(python -c "import pandas as pd; d=pd.read_csv('${PARAM_GRID}'); print(','.join(map(str, sorted(d['lambda'].dropna().unique()))))")"
MARGINS="$(python -c "import pandas as pd; d=pd.read_csv('${PARAM_GRID}'); print(','.join(map(str, sorted(d['anchor_margin'].dropna().unique()))))")"

run_config() {
  local encoder="$1"
  local config="$2"
  local layers="$3"
  local pooled_baseline="$4"
  local english_baseline="$5"

  for condition in pooled english_transfer; do
    local output_dir="${ROOT}/${encoder}/${config}/${condition}"
    local baseline="${pooled_baseline}"
    local extra_args=()
    if [[ "${condition}" == "english_transfer" ]]; then
      baseline="${english_baseline}"
      extra_args+=(--train-languages English)
    fi
    if [[ ! -f "${output_dir}/run_config.json" ]]; then
      python -m src.scripts.etd.train_layer_ensemble_probes \
        --candidates "${CANDIDATES}/${encoder}/router_candidates_with_input_tokens.csv" \
        --activation-root "${ACTIVATIONS}" \
        --output-dir "${output_dir}" \
        --encoder-model "${encoder}" \
        --layers "${layers}" \
        --baseline-layer "${baseline}" \
        --alpha-grid "${ALPHAS}" \
        --reuse-probes-from "${ALL_ENCODERS}/${encoder}/${condition}" \
        --target-jobs 8 \
        "${extra_args[@]}"
    fi

    local evaluation="${output_dir}/evaluation_relative_decision_cost/mean"
    if [[ ! -f "${evaluation}/run_config.json" ]]; then
      python -m src.scripts.routing.evaluate_routing_policies \
        --candidates "${output_dir}/candidates/mean/router_candidates.csv" \
        --output-dir "${evaluation}" \
        --probe-cost-fraction 0 \
        --input-price-fraction 0.16666666666666666 \
        --encoder-model "${encoder}" \
        --output-cost-col mean_output_cost_usd \
        --lambda-grid "${LAMBDAS}" \
        --anchor-margin-grid "${MARGINS}" \
        --loss-tolerances "0,0.5,1,2,3,5"
    fi
  done
}

run_config "Qwen3-4B" four_even "18,24,30,36" 36 24
run_config "Qwen3-4B" six "21,24,27,30,33,36" 36 24
run_config "Qwen3-4B" ten "9,12,15,18,21,24,27,30,33,36" 36 24
run_config "Qwen3-4B" all "$(seq -s, 0 36)" 36 24
