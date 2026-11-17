# VeriLLM Phase 8 Step 7: Stage 2 Val-Lock Confirmation Study Report

- **Generated At (UTC)**: 2026-11-13T02:54:14.326236+00:00
- **Val-Lock Manifest**: `10_stage2_vallock_manifest.json` (SHA-256: `639bd7ce18199f49...`)
- **Stage 1 Frozen Bundle**: `10_stage1_frozen_bundle.json` (SHA-256: `3c3978215b721737...`)
- **Val-Lock Raw Cache**: `10_stage2_vallock_raw_cache.jsonl` (SHA-256: `7160fd60ebabb2ee...`)
- **Total Val-Lock Sources**: 30 | **Responses**: 180 | **Sentences**: 1238
- **Evaluation Mode**: Zero parameters fitted or tuned on Val-Lock. Strictly frozen bundle evaluation.
- **Disjointness Audit**: 30 Val-Lock sources strictly disjoint from 30 Val-Dev sources and 0 exact training context overlaps.

---

## Pillar 1 — Probability Calibration Confirmation on Held-Out Val-Lock

All calibration models were fitted strictly on the 30 Val-Dev sources during Stage 1 and frozen in `10_stage1_frozen_bundle.json`. Here, they are evaluated out-of-sample on the 30 held-out Val-Lock sources with zero refitting.

### Regime: `direct` — Custom DeBERTa 3-Class Calibration on Val-Lock

| Model / Scaling Method | 3-Class NLL | 3-Class Brier | Top-1 ECE | Top-1 MCE | Classwise ECE (CW-ECE) | Binary Hallucination ECE | Argmax Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **C0: Raw Uncalibrated Softmax** | 0.2910 | 0.1495 | 0.0326 | 0.1195 | 0.0274 | 0.0455 | 91.23% |
| **C1: Scalar Temperature Scaling** | 0.2886 | 0.1491 | 0.0208 | 0.1880 | 0.0426 | 0.0632 | 91.23% |
| **C2: Regularized Prior-Bias + Temp** | **0.2647** | **0.1349** | **0.0282** | **0.1099** | **0.0236** | **0.0353** | 90.99% |

#### MiniCheck Binary Support Calibration (`direct`)

| MiniCheck Estimator | Binary NLL | Brier Score | ECE (Uniform) | MCE (Uniform) | ECE (Equal Mass) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Raw MiniCheck Support Score | 0.6483 | 0.1965 | 0.2372 | 0.6434 | 0.2372 |
| Platt-Calibrated MiniCheck | **0.2961** | **0.0917** | **0.0356** | **0.2732** | **0.0400** |

### Regime: `retrieval` — Custom DeBERTa 3-Class Calibration on Val-Lock

| Model / Scaling Method | 3-Class NLL | 3-Class Brier | Top-1 ECE | Top-1 MCE | Classwise ECE (CW-ECE) | Binary Hallucination ECE | Argmax Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **C0: Raw Uncalibrated Softmax** | 0.5696 | 0.3216 | 0.1058 | 0.2020 | 0.1057 | 0.1580 | 79.06% |
| **C1: Scalar Temperature Scaling** | 0.5108 | 0.2988 | 0.0535 | 0.0958 | 0.1347 | 0.2006 | 79.06% |
| **C2: Regularized Prior-Bias + Temp** | **0.3258** | **0.1690** | **0.0281** | **0.1538** | **0.0219** | **0.0308** | 89.61% |

#### MiniCheck Binary Support Calibration (`retrieval`)

| MiniCheck Estimator | Binary NLL | Brier Score | ECE (Uniform) | MCE (Uniform) | ECE (Equal Mass) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Raw MiniCheck Support Score | 1.5119 | 0.3834 | 0.4286 | 0.7984 | 0.4305 |
| Platt-Calibrated MiniCheck | **0.3359** | **0.0994** | **0.0483** | **0.1140** | **0.0502** |

---

## Pillar 2 & 3 — Frozen Strategy Confirmation on Val-Lock

Evaluated on held-out Val-Lock using frozen parameters across both pre-registered objectives:
- **OBJ_CONSTRAINED**: Constrained utility subject to coverage and recall floors.
- **OBJ_MACRO_F1_ONLY**: Unconstrained Macro-F1 subject only to coverage floor.

### Regime: `direct` | Objective: `OBJ_CONSTRAINED`

> **Stage 1 Pre-Registered Frozen Winner**: **`H5_meta_calibrated_logreg`**

| Strategy | Macro-F1 | W-F1 | Accuracy | Conf-MAE (pp) | Non-Sup MAE (pp) | GFI Spearman | Coverage | Holm p (vs H0) | McNemar 3-cls p | McNemar Bin p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| H0_baseline | 37.38% | 72.18% | 64.85% | 29.40 | 29.40 | 0.443 | 100.00% | Ref | Ref | Ref |
| H1_mc_cal | 40.99% | 83.76% | 85.06% | 11.66 | 11.66 | 0.316 | 100.00% | 0.0014 | N/A | N/A |
| H2_deberta_cal_c1 | 71.71% | 90.53% | 90.34% | 6.98 | 6.98 | 0.597 | 100.00% | 0.0014 | N/A | N/A |
| H2_deberta_cal_c2 | 70.14% | 90.50% | 90.42% | 6.84 | 6.84 | 0.591 | 100.00% | 0.0014 | N/A | N/A |
| H2b_deberta_cal_plus_nli_rule4 | 70.14% | 90.50% | 90.42% | 6.84 | 6.84 | 0.591 | 100.00% | 0.0014 | N/A | N/A |
| H4_mc_deberta_cascade | 70.14% | 90.50% | 90.42% | 6.84 | 6.84 | 0.591 | 100.00% | 0.0014 | N/A | N/A |
| **H5_meta_calibrated_logreg** | 64.23% | 90.00% | 90.99% | 6.94 | 6.94 | 0.574 | 100.00% | 0.0014 | N/A | N/A |
| H6_selective_abstention | 70.14% | 90.50% | 90.42% | 6.84 | 6.84 | 0.591 | 100.00% | 0.0014 | N/A | N/A |

### Regime: `direct` | Objective: `OBJ_MACRO_F1_ONLY`

> **Stage 1 Pre-Registered Frozen Winner**: **`H5_meta_calibrated_logreg`**

| Strategy | Macro-F1 | W-F1 | Accuracy | Conf-MAE (pp) | Non-Sup MAE (pp) | GFI Spearman | Coverage | Holm p (vs H0) | McNemar 3-cls p | McNemar Bin p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| H0_baseline | 37.38% | 72.18% | 64.85% | 29.40 | 29.40 | 0.443 | 100.00% | Ref | Ref | Ref |
| H1_mc_cal | 40.99% | 83.76% | 85.06% | 11.66 | 11.66 | 0.316 | 100.00% | 0.0014 | N/A | N/A |
| H2_deberta_cal_c1 | 71.71% | 90.53% | 90.34% | 6.98 | 6.98 | 0.597 | 100.00% | 0.0014 | N/A | N/A |
| H2_deberta_cal_c2 | 70.14% | 90.50% | 90.42% | 6.84 | 6.84 | 0.591 | 100.00% | 0.0014 | N/A | N/A |
| H2b_deberta_cal_plus_nli_rule4 | 70.14% | 90.50% | 90.42% | 6.84 | 6.84 | 0.591 | 100.00% | 0.0014 | N/A | N/A |
| H4_mc_deberta_cascade | 70.14% | 90.50% | 90.42% | 6.84 | 6.84 | 0.591 | 100.00% | 0.0014 | N/A | N/A |
| **H5_meta_calibrated_logreg** | 64.23% | 90.00% | 90.99% | 6.94 | 6.94 | 0.574 | 100.00% | 0.0014 | N/A | N/A |
| H6_selective_abstention | 70.14% | 90.50% | 90.42% | 6.84 | 6.84 | 0.591 | 100.00% | 0.0014 | N/A | N/A |

### Regime: `retrieval` | Objective: `OBJ_CONSTRAINED`

> **Stage 1 Pre-Registered Frozen Winner**: **`H0_baseline`**

| Strategy | Macro-F1 | W-F1 | Accuracy | Conf-MAE (pp) | Non-Sup MAE (pp) | GFI Spearman | Coverage | Holm p (vs H0) | McNemar 3-cls p | McNemar Bin p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **H0_baseline** | 30.36% | 58.28% | 47.81% | 37.96 | 45.31 | 0.289 | 91.40% | Ref | Ref | Ref |
| H1_mc_cal | 27.41% | 68.52% | 59.09% | 11.14 | 26.37 | 0.273 | 65.99% | 0.0132 | N/A | N/A |
| H2_deberta_cal_c1 | 33.50% | 62.98% | 50.57% | 15.74 | 37.20 | 0.402 | 65.99% | 0.1416 | N/A | N/A |
| H2_deberta_cal_c2 | 35.23% | 68.48% | 57.22% | 11.66 | 29.86 | 0.340 | 65.99% | 0.0250 | N/A | N/A |
| H2b_deberta_cal_plus_nli_rule4 | 35.22% | 67.21% | 55.52% | 12.89 | 31.36 | 0.318 | 65.99% | 0.0250 | N/A | N/A |
| H4_mc_deberta_cascade | 35.23% | 68.48% | 57.22% | 11.66 | 29.86 | 0.340 | 65.99% | 0.0250 | N/A | N/A |
| H5_meta_calibrated_logreg | 29.13% | 69.02% | 59.33% | 11.04 | 26.45 | 0.297 | 65.99% | 0.0112 | N/A | N/A |
| H6_selective_abstention | 35.43% | 68.52% | 57.22% | 11.36 | 29.86 | 0.340 | 65.67% | 0.0250 | N/A | N/A |

### Regime: `retrieval` | Objective: `OBJ_MACRO_F1_ONLY`

> **Stage 1 Pre-Registered Frozen Winner**: **`H0_baseline`**

| Strategy | Macro-F1 | W-F1 | Accuracy | Conf-MAE (pp) | Non-Sup MAE (pp) | GFI Spearman | Coverage | Holm p (vs H0) | McNemar 3-cls p | McNemar Bin p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **H0_baseline** | 30.36% | 58.28% | 47.81% | 37.96 | 45.31 | 0.289 | 91.40% | Ref | Ref | Ref |
| H1_mc_cal | 27.41% | 68.52% | 59.09% | 11.14 | 26.37 | 0.273 | 65.99% | 0.0132 | N/A | N/A |
| H2_deberta_cal_c1 | 33.50% | 62.98% | 50.57% | 15.74 | 37.20 | 0.402 | 65.99% | 0.1416 | N/A | N/A |
| H2_deberta_cal_c2 | 35.23% | 68.48% | 57.22% | 11.66 | 29.86 | 0.340 | 65.99% | 0.0250 | N/A | N/A |
| H2b_deberta_cal_plus_nli_rule4 | 35.22% | 67.21% | 55.52% | 12.89 | 31.36 | 0.318 | 65.99% | 0.0250 | N/A | N/A |
| H4_mc_deberta_cascade | 35.23% | 68.48% | 57.22% | 11.66 | 29.86 | 0.340 | 65.99% | 0.0250 | N/A | N/A |
| H5_meta_calibrated_logreg | 29.13% | 69.02% | 59.33% | 11.04 | 26.45 | 0.297 | 65.99% | 0.0112 | N/A | N/A |
| H6_selective_abstention | 35.43% | 68.52% | 57.22% | 11.36 | 29.86 | 0.340 | 65.67% | 0.0250 | N/A | N/A |

---

## Pillar 4 — Runtime Profile & Resource Consumption

- **Total Wall-Clock Time**: 25785.03 seconds
- **Inference Execution Time**: 25763.83 seconds (1238 missing sentences evaluated)
- **Evaluation & Bootstrap Time**: 19.54 seconds
- **Peak Process Memory**: 3506.48 MB (peak_working_set_mb)
- **Current Process RSS at Completion**: 1675.73 MB
- **Average Inference Latency**: 20810.85 ms/sentence
- **Platform**: `Windows-10-10.0.26200-SP0` | Python: `3.11.9`
- **Test Split Boundary Safety**: Verified zero access to `ragtruth_test.jsonl`.
- **Frozen Bundle Invariant**: Zero parameters modified, retrained, or re-tuned on Val-Lock.
