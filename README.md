# Probing Multilingual LLM Success

Code and results for the M.Sc. thesis **"Layer-wise Transferability of Pre-generation Success Probes in Multilingual Large Language Models"** (Andrea Paganelli, Politecnico di Milano / University of Queensland, A.Y. 2025–26).

**TL;DR** — A model's future success on a math problem can be predicted from its hidden activations *before it generates a single answer token*. This repo studies whether those success probes transfer across 10 languages, at which layer depth the transferable signal lives, whether the transferred scores stay calibrated, and whether they can drive cost-aware model routing.

- Success is linearly decodable from pre-generation activations for every evaluated model, and probes transfer across languages with only a small AUROC gap — strongest in **middle-to-late layers**, declining toward the final layers.
- English-trained probes transfer *discriminatively* but are often poorly **calibrated** on other languages; a pooled multilingual probe (balanced 10% per language, same training size) fixes the score scale.
- Pooled multilingual probes support **cost-aware routing**: with a shared Qwen3-4B encoder (encoder-target decoupling), routing matches the strongest model's success at **~26% lower cost**.

All final figures and tables live in [`results/`](results/README.md), organized by thesis section.

## Setup

| | |
|---|---|
| Models | Qwen3-0.6B / 1.7B / 4B / 8B, Qwen3.5-4B / 9B, gpt-oss-20b (low + high reasoning effort) |
| Languages | English, Chinese, Italian, French, Swahili, Russian, Turkish, Arabic, Thai, Telugu |
| Data | 3,000 [MATH](https://github.com/hendrycks/math) problems translated into the 9 non-English languages with DeepL; [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html) function calling (English) for the cross-dataset check |
| Probes | Ridge regression (feature-standardized) on the final-prompt-token residual stream, one probe per layer, predicting the empirical 5-rollout success rate |

```bash
pip install -r requirements.txt
```

`generate_rollouts.py` needs vLLM with GPU inference for all 8 models including gpt-oss-20b (supported natively from vLLM 0.20+; older vLLM releases required a dedicated `+gptoss` wheel). Rollout generation and activation extraction need a GPU; probe training and figures run on CPU. `torch`/`vllm` wheels are CUDA-build-specific — the pinned versions were run against CUDA 13.0; adjust to match your driver if needed (see the [PyTorch](https://pytorch.org/get-started/locally/) and [vLLM](https://docs.vllm.ai/en/latest/getting_started/installation.html) install docs).

## Data

Nothing under `data/` is tracked. To reproduce you need:

- **`data/MATH_translated.csv`** — the multilingual MATH benchmark. One row per problem: `ground_truth`, `split` (the deterministic probe_train/val/test assignment, sha256 of the problem index, 70/15/15, seed 42), then one problem column per language starting with `English`. It was built by translating 3,000 MATH problems with DeepL document translation (Asymptote `[asy]` blocks kept in English). Contact the author or rebuild it from the Hendrycks MATH dataset with any translation provider following the thesis §3.2.3.
- **`data/bfcl/processed_bfcl.jsonl`** — BFCL examples processed with `python -m src.scripts.bfcl.bfcl_cross_task_transfer prepare` (downloads the BFCL data through the `bfcl_eval` package into `data/hf_cache/`).

## Pipeline

All commands run from the repo root. Every stage writes into the (gitignored) `outputs/` tree:

```
outputs/
├── rollouts/MATH/                 {Model}_{Language}.jsonl rollouts (5 per problem)
├── success_rates/MATH/            {Model}_{Language}.csv empirical success rates
├── activations/MATH/              {Language}/{Model}/activations_token_-1.joblib
├── transfer/MATH/                 cross-lingual AUROC tables
├── probes/
│   ├── language_specific/         one probe bundle per (model, language)
│   └── pooled/                    the pooled multilingual probe (balanced 10%/language)
├── routing/
│   ├── probe_configs/  candidates/  simulation/
│   ├── depth_sweep/               probes + router runs at every relative layer depth
│   ├── backward_sweep/            same, at fixed layer offsets back from the final layer
│   ├── candidates_with_input_tokens/
│   └── relative_decision_cost/    accuracy–cost policy grids per depth
├── etd/
│   ├── all_encoders_all_layers/   ETD probes for every encoder × layer (+ single-layer routing grids)
│   ├── layer_ensembles/           Qwen3-4B mean layer ensembles (4/6/10/all-37 layer sets)
│   └── layer_ensemble_routing/    back-offset score ensembles + final router tables
├── bfcl/                          generations / evaluations / activations / cross_task_transfer
└── figures/                       generated figures
```

### 1. Generate rollouts (GPU, vLLM)

```bash
python -m src.scripts.rollouts.generate_rollouts \
    --csv_path data/MATH_translated.csv --language all --model all
```

### 2. Label correctness → empirical success rates

```bash
python -m src.scripts.labelling.build_math_success_rates
```

### 3. Extract pre-generation activations (GPU)

```bash
python -m src.scripts.extract.extract_math_activations \
    --model Qwen3-4B --language all        # repeat per model
```

### 4. Train probes

```bash
# language-specific probes (one per model × language × layer)
python -m src.scripts.probes.train_language_probes --probe-type ridge_scaled
# pooled multilingual probe (balanced 10% per language)
python -m src.scripts.probes.train_pooled_probe --probe-type ridge_scaled
```

### 5. Cross-lingual transfer + AUROC figures

```bash
python -m src.scripts.probes.evaluate_cross_lingual_transfer --probe-type ridge_scaled
python -m src.scripts.figures.plot_transfer_auroc_trajectories --model Qwen3-4B
python -m src.scripts.figures.plot_same_language_auroc_trajectories \
    --transfer-csv outputs/transfer/MATH/ridge_scaled/cross_lingual_per_layer_all.csv
python -m src.scripts.figures.build_auroc_tables    # Tables 4.2-4.4, B.1
```

### 6. Calibration figures

```bash
python -m src.scripts.figures.plot_reliability_english_transfer --model Qwen3-4B --probe-type ridge_scaled
python -m src.scripts.figures.plot_reliability_pooled_multilingual --model Qwen3-4B --probe-type ridge_scaled
python -m src.scripts.figures.build_ece_summary     # Tables 4.5-4.6
```

### 7. Routing

```bash
# merged probe configs, then the depth sweep (candidates + router at every depth)
python -m src.scripts.routing.build_routing_probe_configs
python -m src.scripts.routing.run_depth_routing_sweep
# exact prompt-token counts for the cost model
python -m src.scripts.routing.annotate_input_tokens \
    --candidates outputs/routing/depth_sweep/candidates/frac_100/pooled/router_candidates.csv \
    --output outputs/routing/candidates_with_input_tokens/Qwen3-4B/router_candidates_with_input_tokens.csv
# relative-decision-cost policy grids at every depth
bash src/scripts/routing/run_relative_depth_routing.sh
```

### 8. Encoder-target decoupling (ETD)

```bash
bash src/scripts/etd/run_all_encoders_all_layers.sh   # every encoder × every layer
python -m src.scripts.etd.plot_encoder_selection      # encoder-selection frontiers (Fig B.18–B.19)
bash src/scripts/etd/run_layer_ensembles.sh           # Qwen3-4B 4/6/10/all-37 layer ensembles
python -m src.scripts.etd.plot_layer_ensemble_frontiers   # Fig 4.12, B.20
# layer-ensemble routing tables (Tables 4.9–4.10; needs a backward-offset depth sweep)
python -m src.scripts.routing.run_depth_routing_sweep \
    --backward-step-layers 2 --output-root outputs/routing/backward_sweep
python -m src.scripts.etd.run_layer_ensemble_routing
```

### 9. Routing figures

```bash
python -m src.scripts.figures.plot_routing_pareto_figures
```

### BFCL branch (English function calling)

```bash
python -m src.scripts.bfcl.bfcl_cross_task_transfer prepare
python -m src.scripts.bfcl.bfcl_cross_task_transfer generate --model Qwen3-4B   # or run the official harness via run_official_bfcl.sh
python -m src.scripts.bfcl.bfcl_cross_task_transfer evaluate --model Qwen3-4B
python -m src.scripts.bfcl.bfcl_cross_task_transfer extract  --model Qwen3-4B
python -m src.scripts.bfcl.bfcl_cross_task_transfer probes   --model Qwen3-4B   # trains MATH→BFCL / MATH+BFCL / BFCL probes and plots the transfer AUROC figures
```

## Repository layout

```
src/
├── eval.py                boxed-answer extraction, normalization, correctness matching
├── data.py                dataset loaders (BFCL) + success-rate table loader
├── bfcl.py                BFCL data prep, prompt formatting, AST correctness
├── probes.py              ridge / logistic probe wrappers (joblib-safe)
├── calibration_metrics.py ECE / Brier / NLL
├── extraction/            hidden-state extraction hooks + thinking-suffix utils
└── scripts/
    ├── rollouts/  labelling/  extract/    stages 1–3
    ├── probes/                            stages 4–5 (training + cross-lingual transfer)
    ├── routing/   etd/                    stages 7–8 (cost-aware routing)
    ├── figures/                           all thesis figures
    └── bfcl/                              BFCL branch
results/                                   final thesis figures + tables (see results/README.md)
```

## License

Code in `src/` is released under the [MIT License](LICENSE). The figures and tables in `results/` are excerpts from the thesis and remain under the author's copyright; the MATH problem text is not redistributed by this repository (see [Data](#data)).
