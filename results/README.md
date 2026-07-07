# Results

Final results of the thesis **"Layer-wise Transferability of Pre-generation Success Probes in Multilingual Large Language Models"** (Politecnico di Milano, A.Y. 2025–26).

Evaluated models: `Qwen3-0.6B`, `Qwen3-1.7B`, `Qwen3-4B`, `Qwen3-8B`, `Qwen3.5-4B`, `Qwen3.5-9B`, `gpt-oss-20b_low`, `gpt-oss-20b_high` (the last two are the same checkpoint run at low/high reasoning effort).
Languages: English, Chinese, Italian, French, Swahili, Russian, Turkish, Arabic, Thai, Telugu (3,000 MATH problems translated into each).

The folders follow the structure of the thesis Results chapter. Each table below maps an artifact to its thesis figure/table and to the command that produces it (see the top-level README for the full pipeline).

## 1_empirical_success — base model performance

| Artifact | Thesis | Description |
|---|---|---|
| `math_success_rates_by_model_language.csv` | Table 4.1 | Empirical MATH success rate per (model, language), averaged over 5 rollouts per problem. |
| `bfcl_success_rates.csv` | Table 4.11 | Overall BFCL success rate per Qwen-family model. |
| `bfcl_success_rates_by_category.csv` | Table B.4 | BFCL success rate per subset (simple, multiple, parallel, live, ...). |

Produced by `src/scripts/labelling/build_math_success_rates.py` and the BFCL evaluation in `src/scripts/bfcl/bfcl_cross_task_transfer.py`.

## 2_layerwise_auroc — layer-wise success prediction

| Artifact | Thesis | Description |
|---|---|---|
| `same_language_auroc_trajectory_<model>.pdf` | Fig 4.1, B.1–B.5 (left) | Layer-wise same-language probe AUROC (mean ± std across the 10 languages). |
| `transfer_auroc_trajectory_<model>.pdf` | Fig 4.2, B.1–B.5 | Same-language vs. cross-lingual transfer AUROC trajectories side by side. |
| `same_language_auroc_trajectory_all_models.pdf` | — | All 8 models in one grid. |
| `peak_same_language_auroc_by_model.csv` | Table 4.2 | Peak of the mean same-language AUROC trajectory per model. |
| `peak_transfer_auroc_by_model.csv` | Table 4.3 | Peak cross-lingual transfer AUROC (mean over all 90 source→target pairs) per model, with final-layer drop. |
| `same_language_vs_transfer_auroc.csv` | Table 4.4 | Same-language vs. transfer AUROC at the transfer-optimal layer. |
| `peak_same_language_auroc_by_language.csv` | Table B.1 | Peak same-language AUROC per (language, model). |

Produced by `src/scripts/probes/evaluate_cross_lingual_transfer.py` (data), `src/scripts/figures/plot_same_language_auroc_trajectories*.py` and `plot_transfer_auroc_trajectories.py` (figures), and `src/scripts/figures/build_auroc_tables.py` (tables).

## 3_calibration — reliability of probe scores

| Artifact | Thesis | Description |
|---|---|---|
| `reliability_english_transfer_<model>.pdf` | Fig 4.3–4.4, B.6–B.11 | Reliability diagrams (one panel per target language) for the **English-trained** probe applied to every language. |
| `reliability_pooled_multilingual_<model>.pdf` | Fig 4.5–4.6, B.12–B.17 | Same for the **pooled multilingual** probe (language-balanced 10%-per-language training). |
| `ece_english_vs_pooled_summary.csv` / `.pdf` | Tables 4.5–4.6 | Mean ECE per model for English-transfer vs. pooled probes and the relative improvement. |

Produced by `src/scripts/figures/plot_reliability_english_transfer.py` and `plot_reliability_pooled_multilingual.py` (figures) and `src/scripts/figures/build_ece_summary.py` (summary).

## 4_routing — cost-aware model routing

| Artifact | Thesis | Description |
|---|---|---|
| `routing_success_by_depth.pdf` | Fig 4.7 | Maximum validation routing success vs. probe layer depth (pooled vs. English-transfer). |
| `pareto_direct_routing.pdf` | Fig 4.8 | Accuracy–cost trade-off for direct probe routing (every candidate model prefills). |
| `pareto_etd_by_layer_{pooled,english_transfer}.pdf` | Fig 4.9 | Validation frontiers of ETD routing across Qwen3-4B encoder layers. |
| `pareto_etd_routing.pdf`, `pareto_etd_anchor_policy.pdf` | Fig 4.10 | Encoder-target decoupled routing (Qwen3-4B encoder, layer 23), both policies / anchor policy. |
| `pareto_direct_vs_etd_pooled.pdf` | Fig 4.11 | Direct pooled routing (target-model encoders) vs. pooled ETD (shared Qwen3-4B encoder). |
| `layer_ensemble_pareto_{pooled,english_transfer}.pdf` (+ `_zoom`) | Fig 4.12, B.20 | ETD layer-ensemble frontiers (layer 23 / 4 / 6 / 10 / all-37 layer sets). |
| `etd_encoder_selection_validation.pdf` (+ `_zoom`) | Fig B.18–B.19 | Validation frontiers across all encoder choices at their best validation-selected layer. |
| `routing_operating_points.csv` | Tables 4.7–4.8, B.2–B.3 | Validation-selected operating points and model-selection mixes on the test split. |
| `layer_ensemble_summary.csv`, `layer_ensemble_vs_selected_summary.csv` | Tables 4.9–4.10 | Layer-ensemble operating points vs. the single-layer baselines. |

Costs are relative to always selecting the anchor `gpt-oss-20b_high`; input tokens are priced at 1/6 of output tokens. The depth figures, ETD paretos, and Fig 4.9 come from `src/scripts/figures/plot_routing_pareto_figures.py`; the encoder-selection figures from `src/scripts/etd/plot_encoder_selection.py`; the layer-ensemble frontiers from `src/scripts/etd/plot_layer_ensemble_frontiers.py`; `routing_operating_points.csv` from `src/scripts/routing/run_router_simulation.py`; the layer-ensemble tables from `src/scripts/etd/run_layer_ensemble_routing.py`.

## 5_bfcl — cross-dataset evaluation on function calling

| Artifact | Thesis | Description |
|---|---|---|
| `bfcl_transfer_auroc_<model>.pdf` | Fig 4.13, B.21–B.23 | Layer-wise BFCL AUROC under three training configurations: MATH→BFCL (direct transfer), MATH+BFCL→BFCL (joint), BFCL→BFCL (in-task). |
| `bfcl_transfer_auroc_all_models.pdf` | — | Matched-layer transfer grid across all Qwen-family models. |

Produced by `src/scripts/bfcl/bfcl_cross_task_transfer.py`.
