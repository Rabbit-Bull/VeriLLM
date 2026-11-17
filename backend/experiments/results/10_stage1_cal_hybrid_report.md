# VeriLLM Phase 8 Step 7: Stage 1 Calibration & Hybrid-Strategy Study Report

- **Generated At (UTC)**: 2026-11-12T17:20:39.573524+00:00
- **Val-Dev Manifest**: `09_val_pilot_30sources_manifest.json` (SHA-256: `75a9b015123a42f3...`)
- **Val-Dev Raw Cache**: `09_val_pilot_raw_cache.jsonl` (SHA-256: `75f37245451b4457...`)
- **Total Val-Dev Sources**: 30 | **Responses**: 177 | **Sentences**: 1259
- **Source-Grouped Nested CV**: 5 outer folds × 4 inner folds
- **H3 Status**: **EXCLUDED FROM STAGE 1 CACHE-ONLY EVALUATION** (`09_val_pilot_raw_cache.jsonl` does not store per-chunk DeBERTa scores for secondary passages).

---

## Pillar 1 — Probability Calibration Analysis (5-Fold Source-Grouped OOF)

All calibration models are evaluated strictly out-of-fold across 5 outer source folds. Each unique `source_id` is assigned to exactly one outer fold identically across Direct and Retrieval regimes.

### Regime: `direct` — Custom DeBERTa 3-Class Calibration

| Model / Scaling Method | 3-Class NLL | 3-Class Brier | Top-1 ECE | Top-1 MCE | Classwise ECE (CW-ECE) | Binary Hallucination ECE | Argmax Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **C0: Raw Uncalibrated Softmax** | 0.3732 | 0.1857 | 0.0425 | 0.2327 | 0.0334 | 0.0460 | 88.64% |
| **C1: Scalar Temperature Scaling** | 0.3577 | 0.1837 | 0.0144 | 0.0461 | 0.0334 | 0.0421 | 88.64% |
| **C2: Regularized Prior-Bias + Temp** | **0.3412** | **0.1741** | **0.0153** | **0.2667** | **0.0167** | **0.0232** | 88.96% |

#### MiniCheck Binary Support Calibration (`direct`)

| MiniCheck Estimator | Binary NLL | Brier Score | ECE (Uniform) | MCE (Uniform) | ECE (Equal Mass) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Raw MiniCheck Support Score | 0.6002 | 0.1869 | 0.2291 | 0.5440 | 0.2296 |
| Platt-Calibrated MiniCheck | **0.2973** | **0.0909** | **0.0289** | **0.2791** | **0.0190** |

### Regime: `retrieval` — Custom DeBERTa 3-Class Calibration

| Model / Scaling Method | 3-Class NLL | 3-Class Brier | Top-1 ECE | Top-1 MCE | Classwise ECE (CW-ECE) | Binary Hallucination ECE | Argmax Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **C0: Raw Uncalibrated Softmax** | 0.6274 | 0.3438 | 0.1169 | 0.3989 | 0.1013 | 0.1449 | 77.28% |
| **C1: Scalar Temperature Scaling** | 0.5578 | 0.3219 | 0.0377 | 0.1317 | 0.1254 | 0.1880 | 77.28% |
| **C2: Regularized Prior-Bias + Temp** | **0.3884** | **0.2040** | **0.0342** | **0.1921** | **0.0210** | **0.0320** | 88.16% |

#### MiniCheck Binary Support Calibration (`retrieval`)

| MiniCheck Estimator | Binary NLL | Brier Score | ECE (Uniform) | MCE (Uniform) | ECE (Equal Mass) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Raw MiniCheck Support Score | 1.5159 | 0.3909 | 0.4493 | 0.7496 | 0.4493 |
| Platt-Calibrated MiniCheck | **0.3542** | **0.1070** | **0.0304** | **0.1226** | **0.0377** |

---

## Pillar 2 & 3 — Claim Classification, Response-Level & 6-LLM Evaluation

Reported across both pre-registered objectives:
- **OBJ_CONSTRAINED**: `Macro-F1 - 0.005 * max(0, MAE_conf_pp - MAE_max_pp)` subject to coverage >= 0.95 (direct) / 0.90 (retrieval) and recall floors.
- **OBJ_MACRO_F1_ONLY**: Unconstrained Macro-F1 subject only to coverage floor.

### Regime: `direct` | Objective: `OBJ_CONSTRAINED`

> **Winning Strategy**: **`H2_deberta_cal_c2`** (Utility: `0.6372`, Macro-F1: `63.72%`, Conf-MAE: `9.59 pp`, Coverage: `100.00%`, All Constraints Met: `True`)

| Strategy | Utility | Macro-F1 | W-F1 | Accuracy | Conf-MAE (pp) | Non-Sup MAE (pp) | GFI Spearman | Coverage | Holm p (vs H0) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **H2_deberta_cal_c2** | 0.6372 | 63.72% | 88.45% | 89.20% | 9.59 | 9.59 | 0.456 | 100.00% | 0.0014 |
| H2b_deberta_cal_plus_nli_rule4 | 0.6372 | 63.72% | 88.45% | 89.20% | 9.59 | 9.59 | 0.456 | 100.00% | 0.0014 |
| H4_mc_deberta_cascade | 0.6372 | 63.72% | 88.45% | 89.20% | 9.59 | 9.59 | 0.456 | 100.00% | 0.0014 |
| H6_selective_abstention | 0.6306 | 63.06% | 88.24% | 88.88% | 9.89 | 9.59 | 0.456 | 99.52% | 0.0014 |
| H5_meta_calibrated_logreg | 0.6252 | 62.52% | 88.70% | 89.92% | 8.94 | 8.94 | 0.486 | 100.00% | 0.0014 |
| H2_deberta_cal_c1 | 0.6114 | 61.16% | 87.44% | 88.00% | 10.04 | 10.04 | 0.418 | 100.00% | 0.0014 |
| H1_mc_cal | 0.4384 | 44.07% | 84.41% | 86.16% | 10.46 | 10.46 | 0.409 | 100.00% | 0.0014 |
| H0_baseline | 0.3709 | 44.15% | 75.96% | 71.20% | 24.13 | 24.13 | 0.503 | 100.00% | Ref |

### Regime: `direct` | Objective: `OBJ_MACRO_F1_ONLY`

> **Winning Strategy**: **`H2_deberta_cal_c2`** (Utility: `0.6333`, Macro-F1: `63.33%`, Conf-MAE: `9.83 pp`, Coverage: `100.00%`, All Constraints Met: `True`)

| Strategy | Utility | Macro-F1 | W-F1 | Accuracy | Conf-MAE (pp) | Non-Sup MAE (pp) | GFI Spearman | Coverage | Holm p (vs H0) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **H2_deberta_cal_c2** | 0.6333 | 63.33% | 88.20% | 88.88% | 9.83 | 9.83 | 0.440 | 100.00% | 0.0014 |
| H2b_deberta_cal_plus_nli_rule4 | 0.6333 | 63.33% | 88.20% | 88.88% | 9.83 | 9.83 | 0.440 | 100.00% | 0.0014 |
| H4_mc_deberta_cascade | 0.6303 | 63.03% | 88.12% | 88.88% | 9.67 | 9.67 | 0.447 | 100.00% | 0.0014 |
| H6_selective_abstention | 0.6267 | 62.67% | 87.98% | 88.56% | 10.13 | 9.83 | 0.440 | 99.52% | 0.0014 |
| H5_meta_calibrated_logreg | 0.6252 | 62.52% | 88.70% | 89.92% | 8.94 | 8.94 | 0.486 | 100.00% | 0.0014 |
| H2_deberta_cal_c1 | 0.6116 | 61.16% | 87.44% | 88.00% | 10.04 | 10.04 | 0.418 | 100.00% | 0.0014 |
| H0_baseline | 0.4415 | 44.15% | 75.96% | 71.20% | 24.13 | 24.13 | 0.503 | 100.00% | Ref |
| H1_mc_cal | 0.4331 | 43.31% | 83.81% | 85.20% | 11.00 | 11.00 | 0.402 | 100.00% | 0.0014 |

### Regime: `retrieval` | Objective: `OBJ_CONSTRAINED`

> **Winning Strategy**: **`H0_baseline`** (Utility: `0.2255`, Macro-F1: `33.13%`, Conf-MAE: `39.17 pp`, Coverage: `91.68%`, All Constraints Met: `False`)

| Strategy | Utility | Macro-F1 | W-F1 | Accuracy | Conf-MAE (pp) | Non-Sup MAE (pp) | GFI Spearman | Coverage | Holm p (vs H0) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **H0_baseline** | 0.2255 | 33.13% | 56.65% | 47.36% | 39.17 | 46.91 | 0.371 | 91.68% | Ref |
| H2_deberta_cal_c1 | 0.3604 | 36.04% | 63.80% | 51.92% | 14.52 | 35.43 | 0.290 | 66.32% | 0.0890 |
| H6_selective_abstention | 0.3107 | 31.07% | 66.70% | 56.72% | 13.43 | 28.38 | 0.222 | 66.16% | 0.0760 |
| H2_deberta_cal_c2 | 0.3103 | 31.03% | 66.69% | 56.72% | 13.55 | 28.38 | 0.222 | 66.32% | 0.0760 |
| H4_mc_deberta_cascade | 0.3103 | 31.03% | 66.69% | 56.72% | 13.55 | 28.38 | 0.222 | 66.32% | 0.0760 |
| H2b_deberta_cal_plus_nli_rule4 | 0.3087 | 30.87% | 66.28% | 56.16% | 14.27 | 28.98 | 0.196 | 66.32% | 0.0760 |
| H5_meta_calibrated_logreg | 0.2805 | 28.05% | 67.28% | 58.56% | 12.91 | 26.68 | 0.213 | 66.32% | 0.0490 |
| H1_mc_cal | 0.2638 | 26.38% | 66.54% | 58.08% | 13.15 | 26.97 | 0.214 | 66.32% | 0.0708 |

### Regime: `retrieval` | Objective: `OBJ_MACRO_F1_ONLY`

> **Winning Strategy**: **`H0_baseline`** (Utility: `0.3313`, Macro-F1: `33.13%`, Conf-MAE: `39.17 pp`, Coverage: `91.68%`, All Constraints Met: `False`)

| Strategy | Utility | Macro-F1 | W-F1 | Accuracy | Conf-MAE (pp) | Non-Sup MAE (pp) | GFI Spearman | Coverage | Holm p (vs H0) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **H0_baseline** | 0.3313 | 33.13% | 56.65% | 47.36% | 39.17 | 46.91 | 0.371 | 91.68% | Ref |
| H2_deberta_cal_c1 | 0.3604 | 36.04% | 63.80% | 51.92% | 14.52 | 35.43 | 0.290 | 66.32% | 0.0890 |
| H6_selective_abstention | 0.3107 | 31.07% | 66.70% | 56.72% | 13.43 | 28.38 | 0.222 | 66.16% | 0.0760 |
| H2_deberta_cal_c2 | 0.3103 | 31.03% | 66.69% | 56.72% | 13.55 | 28.38 | 0.222 | 66.32% | 0.0760 |
| H4_mc_deberta_cascade | 0.3103 | 31.03% | 66.69% | 56.72% | 13.55 | 28.38 | 0.222 | 66.32% | 0.0760 |
| H2b_deberta_cal_plus_nli_rule4 | 0.3087 | 30.87% | 66.28% | 56.16% | 14.27 | 28.98 | 0.196 | 66.32% | 0.0760 |
| H5_meta_calibrated_logreg | 0.2805 | 28.05% | 67.28% | 58.56% | 12.91 | 26.68 | 0.213 | 66.32% | 0.0490 |
| H1_mc_cal | 0.2638 | 26.38% | 66.54% | 58.08% | 13.15 | 26.97 | 0.214 | 66.32% | 0.0708 |

---

## Pillar 4 — Efficiency, Latency & Stage 2 Execution Framework

- **Stage 1 Execution Mode**: Pure cache-only offline nested cross-validation (`0` neural inference calls).
- **H3 Secondary-Passage Status**: Excluded from Stage 1 cache-only evaluation; marked `requires_stage2_secondary_chunk_inference`.
- **Stage 2 Val-Lock Safety Guard**: `--stage2-run` strictly requires `--approve-stage2-inference` before running any neural models.
- **Immutable Frozen Bundle**: Serialized to `10_stage1_frozen_bundle.json` using Val-Dev 5-fold OOF tuning; never updated using Val-Lock data.
