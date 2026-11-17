# Phase 8 — Corrected Cache-Only Analysis (30 Validation Sources)

- **Generated UTC**: `2026-11-12T16:15:07.448419+00:00` (cache_only=`True`)
- **Dataset**: `ragtruth_val.jsonl` ONLY (`ragtruth_test.jsonl` untouched)
- **Code & Runtime Provenance**: git=`486e4c4e813b07de4a343db560d8bc07e8e872ca` (dirty=`True`), device=`cpu`, python=`3.11.9`, torch=`2.15.0+cpu`, transformers=`5.22.2`, sklearn=`1.9.1`, numpy=`2.4.6`
- **Models Evaluated (from cache)**: MiniCheck=`bespokelabs/MiniCheck (roberta-large)`, NLI=`cross-encoder/nli-deberta-v3-xsmall`, Retriever=`sentence-transformers/all-MiniLM-L6-v2`, CustomDeBERTa=`D:\drive e\VeriLLM\backend\models\verillm_deberta` (sha256[:16]=`{"config.json": "930719b13d5bf3c4", "model.safetensors": "c2bc95e10f5b12fc", "tokenizer.json": "ccab0d2b1c2aec65", "tokenizer_config.json": "3dd09fcbf87e99e2"}`)
- **Manifest & Selection Provenance**: manifest=`backend\experiments\results\09_val_pilot_30sources_manifest.json`, locked_manifest=`backend\experiments\results\09_val_pilot_15sources_manifest.json`, method=`two_phase_locked_prefix_plus_uniform_followup`, seed=`42`, bootstrap_seed=`42`, n_boot=`2000`, resampling_unit=`source_id`
- **Selection Rationale**: Two-phase selection: preserved 15 locked source(s) from prior manifest (stage2_minority_stratified_source_allocation) and sampled remaining sources uniformly without replacement per task (seed=42, 10 sources/task total). Combined set satisfied all minority adequacy checks.
- **Pilot Scope**: `30` sources, `177` complete responses, `1259` total rows (`1250` 3-class + `9` `MIXED_HALLUCINATION` diagnostic)

> **Methodological Scope & Caveats**: (1) **Exploratory inference only**: `30` validation sources provide exploratory precision rather than definitive population-level confidence intervals, and some selection strata contain only `2–4` sources (high leverage up to `2.40x`), making stratum-weighted bootstrap CIs unstable. (2) **Primary vs. supplementary clustering**: `source_id` (`K=30`) is the primary independent unit; response-clustered (`R=177`) checks are supplementary (responses share a source), and sentence-level McNemar (`n=1,250`) is legacy and anti-conservative because it ignores intra-source/intra-response correlation. (3) **Uncalibrated DeBERTa outputs & `arm3a` warning**: CustomDeBERTa's class-weighted softmax outputs are **uncalibrated normalized logits**, not calibrated probabilities; fixed legacy NLI thresholds inside `arm3a_mc_plus_custom_deberta` are unvalidated on these outputs, so `arm3a` must NOT be used to justify a production decision prior to out-of-fold calibration. (4) **Retrieval `NLI-xsmall` dependency**: in Condition 2 (`retrieval`), `ClaimVerifier`'s Rule 4 secondary-evidence conflict resolution relies on `NLI-xsmall` (`cross-encoder/nli-deberta-v3-xsmall`), not a pure NLI-free retriever.

## 1. Pilot Class Distribution, Stratum Weights & Adequacy Audit

| Slice | Total | SUPPORTED_PROXY | CONTRADICTED | UNSUPPORTED | MIXED_HALLUCINATION |
|---|---:|---:|---:|---:|---:|
| Full `ragtruth_val.jsonl` (375 src) | 15792 | 13970 | 581 | 1167 | 74 |
| **Selected Pilot** (30 src) | 1259 | 1089 | 46 | 115 | 9 |
| Pilot `Summary` | 344 | 318 | 6 | 20 | 0 |
| Pilot `QA` | 378 | 302 | 17 | 59 | 0 |
| Pilot `Data2txt` | 537 | 469 | 23 | 36 | 9 |

### Stratum Sampling Summary & Horvitz-Thompson Weights

| Task | Stratum | Val Sources | Pilot Sources | Rel. Stratum Weight | Small-Cell Flag |
|---|---|---:|---:|---:|---|
| `Data2txt` | `CONTRA_PRESENT` | 125 | 8 | 1.2500 | ok |
| `Data2txt` | `NO_CONTRA` | 7 | 2 | 0.2800 | SMALL STRATUM (<5) |
| `QA` | `CONTRA_PRESENT` | 35 | 7 | 0.4000 | ok |
| `QA` | `NO_CONTRA` | 90 | 3 | 2.4000 | SMALL STRATUM (<5) |
| `Summary` | `CONTRA_PRESENT` | 50 | 4 | 1.0000 | SMALL STRATUM (<5) |
| `Summary` | `NO_CONTRA` | 68 | 6 | 0.9067 | ok |

## 2. Train-Validation Leakage & Duplicate Separation

- `source_id` overlap with `ragtruth_train.jsonl`: **0**
- Exact `(context_text, sentence_text)` duplicates in train: **0** (IDs: `[]`)
- Repeated `sentence_text` strings appearing in train (total / string-only diff context): **44** / **44**
- Within-pilot extra pair duplicates / repeated sentence strings: **17** / **32**

## 3. Recorded Warm-Up Latency Benchmark & 30-Source Runtime

- Cold model load seconds (recorded during initial Phase 8 run): `{'claim_verifier_minicheck_plus_nli': 3.716, 'custom_deberta': 4.871, 'passage_retriever': 0.978, 'total_model_init': 9.566}`
- Measured on warm-up slice (`16` sentences, `3` sources): MiniCheck=7016.41 ms/pair, NLI-xsmall=1007.64 ms/pair, CustomDeBERTa=1008.74 ms/pair, Chunk+Retrieve=66.97 ms/pair, Dual-regime estimate=18132.56 ms/sentence
- Scaled dual-regime estimate for `1259` sentences (`30` sources): **380.48 min** (actual tracked inference wall time across 30 sources: **20424.52s = 340.41 min**; stale 15-source benchmark entry was `192.21 min`).

## 4. Condition 1 — Direct Source Context Ablation

- Diagnostics: Direct pairs >512 tokens = `754` (59.89%), Retrieval top-1 >512 tokens = `0`, Retrieval mean sim = `0.5185`, <0.25 floor = `99`, Rule 4 `NLI-xsmall` secondary-conflict triggers = `11`

### A. Claim-Level 3-Class & Binary Detection (direct)

| Arm | Coverage | Pen. Macro-F1 | Sup-Weighted F1 | Sel. Macro-F1 | Pen. Acc | CONTRA F1 | SUP_P F1 | UNSUP F1 | Bin-F1 (Strict) | AUROC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `arm0a_nli_only` | 100.0% (1250/1250) | 18.05% | 24.22% | 18.05% | 21.84% | 9.71% | 25.38% | 19.06% | 25.02% | 0.600 | 0.173 |
| `arm0b_minicheck_only` | 100.0% (1250/1250) | 39.63% | 75.76% | 39.63% | 71.44% | 0.00% | 83.19% | 35.69% | 44.86% | 0.839 | 0.445 |
| `arm1_baseline` | 100.0% (1250/1250) | 44.15% | 75.96% | 44.15% | 71.20% | 13.70% | 82.82% | 35.94% | 43.76% | N/A (discrete) | N/A (discrete) |
| `arm2_custom_deberta_only` | 100.0% (1250/1250) | 61.22% | 87.70% | 61.22% | 88.64% | 42.11% | 93.85% | 47.72% | 49.82% | 0.840 | 0.557 |
| `arm3a_mc_plus_custom_deberta` | 100.0% (1250/1250) | 60.68% | 87.84% | 60.68% | 89.04% | 41.10% | 94.15% | 46.81% | 49.81% | N/A (discrete) | N/A (discrete) |
| `arm3b_baseline_support_deberta_disambig` | 100.0% (1250/1250) | 53.15% | 76.94% | 53.15% | 71.92% | 41.10% | 82.82% | 35.55% | 43.76% | N/A (discrete) | N/A (discrete) |
| `arm3c_selective_agreement_gate` | 71.0% (888/1250) | 56.31% | 77.26% | 73.23% | 66.48% | 41.10% | 82.12% | 45.71% | 22.47% | N/A (discrete) | N/A (discrete) |

### B. Per-Task, HT-Reweighted & Length/Duplicate Slices (direct)

| Arm | Summary Macro-F1 (Acc) | QA Macro-F1 (Acc) | Data2txt Macro-F1 (Acc) | HT-Weighted Macro-F1 (Sup-W F1) | <=512 tok Macro-F1 | >512 tok Macro-F1 | No Exact Train-Dup Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `arm0a_nli_only` | 3.70% (5.8%) | 39.82% (54.0%) | 9.20% (9.3%) | 17.45% (21.95%) | 30.36% | 9.64% | 18.05% |
| `arm0b_minicheck_only` | 42.46% (82.0%) | 44.65% (72.5%) | 33.57% (63.8%) | 40.27% (73.92%) | 44.42% | 35.97% | 39.63% |
| `arm1_baseline` | 42.02% (81.1%) | 52.57% (76.7%) | 37.42% (60.8%) | 43.89% (73.85%) | 51.55% | 38.65% | 44.15% |
| `arm2_custom_deberta_only` | 39.81% (92.7%) | 46.06% (82.5%) | 70.19% (90.3%) | 65.06% (86.88%) | 55.62% | 65.56% | 61.22% |
| `arm3a_mc_plus_custom_deberta` | 44.12% (93.3%) | 45.93% (82.8%) | 68.44% (90.7%) | 64.48% (86.95%) | 55.11% | 65.04% | 60.68% |
| `arm3b_baseline_support_deberta_disambig` | 41.90% (81.1%) | 46.93% (76.5%) | 52.06% (62.7%) | 57.08% (75.20%) | 53.60% | 53.39% | 53.15% |
| `arm3c_selective_agreement_gate` | 37.59% (76.7%) | 43.83% (66.9%) | 62.58% (59.5%) | 59.63% (75.20%) | 53.25% | 58.66% | 56.31% |

### C. Per-Class P / R / F1 & 3x4 Confusion Matrices (direct)

| Arm | Class | P | R | F1 | Support | Pred: CONTRA | Pred: SUP_P | Pred: UNSUP | Pred: UNVER |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `arm0a_nli_only` | `CONTRADICTED` | 8.77% | 10.87% | 9.71% | 46 | 5 | 3 | 38 | 0 |
| `arm0a_nli_only` | `SUPPORTED_PROXY` | 96.95% | 14.60% | 25.38% | 1089 | 48 | 159 | 882 | 0 |
| `arm0a_nli_only` | `UNSUPPORTED` | 10.59% | 94.78% | 19.06% | 115 | 4 | 2 | 109 | 0 |
| `arm0b_minicheck_only` | `CONTRADICTED` | 0.00% | 0.00% | 0.00% | 46 | 0 | 11 | 35 | 0 |
| `arm0b_minicheck_only` | `SUPPORTED_PROXY` | 96.37% | 73.19% | 83.19% | 1089 | 0 | 797 | 292 | 0 |
| `arm0b_minicheck_only` | `UNSUPPORTED` | 22.70% | 83.48% | 35.69% | 115 | 0 | 19 | 96 | 0 |
| `arm1_baseline` | `CONTRADICTED` | 18.52% | 10.87% | 13.70% | 46 | 5 | 13 | 28 | 0 |
| `arm1_baseline` | `SUPPORTED_PROXY` | 96.00% | 72.82% | 82.82% | 1089 | 19 | 793 | 277 | 0 |
| `arm1_baseline` | `UNSUPPORTED` | 23.17% | 80.00% | 35.94% | 115 | 3 | 20 | 92 | 0 |
| `arm2_custom_deberta_only` | `CONTRADICTED` | 53.33% | 34.78% | 42.11% | 46 | 16 | 29 | 1 | 0 |
| `arm2_custom_deberta_only` | `SUPPORTED_PROXY` | 91.83% | 95.96% | 93.85% | 1089 | 10 | 1045 | 34 | 0 |
| `arm2_custom_deberta_only` | `UNSUPPORTED` | 57.32% | 40.87% | 47.72% | 115 | 4 | 64 | 47 | 0 |
| `arm3a_mc_plus_custom_deberta` | `CONTRADICTED` | 55.56% | 32.61% | 41.10% | 46 | 15 | 29 | 2 | 0 |
| `arm3a_mc_plus_custom_deberta` | `SUPPORTED_PROXY` | 91.65% | 96.79% | 94.15% | 1089 | 8 | 1054 | 27 | 0 |
| `arm3a_mc_plus_custom_deberta` | `UNSUPPORTED` | 60.27% | 38.26% | 46.81% | 115 | 4 | 67 | 44 | 0 |
| `arm3b_baseline_support_deberta_disambig` | `CONTRADICTED` | 55.56% | 32.61% | 41.10% | 46 | 15 | 13 | 18 | 0 |
| `arm3b_baseline_support_deberta_disambig` | `SUPPORTED_PROXY` | 96.00% | 72.82% | 82.82% | 1089 | 8 | 793 | 288 | 0 |
| `arm3b_baseline_support_deberta_disambig` | `UNSUPPORTED` | 22.92% | 79.13% | 35.55% | 115 | 4 | 20 | 91 | 0 |
| `arm3c_selective_agreement_gate` | `CONTRADICTED` | 55.56% | 32.61% | 41.10% | 46 | 15 | 12 | 1 | 18 |
| `arm3c_selective_agreement_gate` | `SUPPORTED_PROXY` | 96.88% | 71.26% | 82.12% | 1089 | 8 | 776 | 19 | 286 |
| `arm3c_selective_agreement_gate` | `UNSUPPORTED` | 66.67% | 34.78% | 45.71% | 115 | 4 | 13 | 40 | 58 |

### D. Primary Paired Comparisons vs `arm1_baseline` — Source-Clustered (`K=30`) (direct)

Family-wise **Holm–Bonferroni** adjustment (`m=6` candidate arms) applied per test family within this regime.

| Candidate Arm | Unw. Δ 3-Cls Acc | Clusters (+/-/=) | Unw. Sign-Flip Raw p | Unw. Sign-Flip Holm p | Exact Sign Raw p (Holm) | Wt. Δ 3-Cls Acc | Wt. Sign-Flip Raw p | Wt. Sign-Flip Holm p | Bin Acc Δ (Unw. Holm p) |
|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| `arm0a_nli_only` | -48.80pp | 1/28/1 | 4.9998e-05 | 0.0003 | 1.118e-07 (6.7055e-07) | -44.66pp | 4.9998e-05 | 0.0003 | -47.96pp (0.0003) |
| `arm0b_minicheck_only` | +0.26pp | 7/8/15 | 0.8563 | 0.8563 | 1 (1) | +0.64pp | 0.712 | 0.712 | +0.48pp (1) |
| `arm2_custom_deberta_only` | +15.62pp | 26/3/1 | 4.9998e-05 | 0.0003 | 1.524e-05 (6.0946e-05) | +15.95pp | 9.9995e-05 | 0.0004 | +13.63pp (0.0003) |
| `arm3a_mc_plus_custom_deberta` | +16.00pp | 27/2/1 | 4.9998e-05 | 0.0003 | 1.624e-06 (8.1211e-06) | +16.31pp | 4.9998e-05 | 0.0003 | +14.09pp (0.0003) |
| `arm3b_baseline_support_deberta_disambig` | +0.58pp | 7/2/21 | 0.0828 | 0.1656 | 0.179687 (0.3594) | +1.06pp | 0.0184 | 0.0368 | +0.00pp (1) |
| `arm3c_selective_agreement_gate` | -6.08pp | 2/21/7 | 0.0002 | 0.0006 | 6.604e-05 (0.0001981) | -8.22pp | 0.00025 | 0.00075 | -8.07pp (0.0003) |

#### Source-Clustered Bootstrap 95% CIs — Unweighted (`K=30`, Primary Exploratory) (direct)

| Candidate Arm | Δ Macro-F1 (95% CI) | Δ Sup-Weighted F1 (95% CI) | Δ Pen. Acc (95% CI) | Δ CONTRA F1 (95% CI) | Δ UNSUP F1 (95% CI) | Δ Bin-F1 (95% CI) |
|---|---:|---:|---:|---:|---:|---:|
| `arm0a_nli_only` | -26.10pp [-30.54, -21.75] | -51.74pp [-61.53, -42.20] | -49.36pp [-55.29, -43.40] | -3.99pp [-9.02, -0.46] | -16.88pp [-20.55, -12.45] | -18.74pp [-22.61, -14.58] |
| `arm0b_minicheck_only` | -4.52pp [-7.88, -1.22] | -0.20pp [-1.69, +1.81] | +0.24pp [-2.01, +3.11] | -13.70pp [-23.88, -3.12] | -0.25pp [-2.10, +1.90] | +1.10pp [-0.64, +3.10] |
| `arm2_custom_deberta_only` | +17.07pp [+10.78, +23.30] | +11.74pp [+8.57, +15.22] | +17.44pp [+13.55, +21.96] | +28.41pp [+14.74, +44.58] | +11.78pp [-1.21, +24.49] | +6.06pp [-3.83, +16.00] |
| `arm3a_mc_plus_custom_deberta` | +16.53pp [+10.63, +22.73] | +11.88pp [+8.93, +15.33] | +17.84pp [+14.05, +22.35] | +27.40pp [+13.04, +44.38] | +10.87pp [-1.20, +23.09] | +6.05pp [-3.59, +16.13] |
| `arm3b_baseline_support_deberta_disambig` | +9.00pp [+4.15, +14.68] | +0.97pp [+0.37, +1.63] | +0.72pp [+0.17, +1.25] | +27.40pp [+13.04, +44.38] | -0.39pp [-1.46, +0.69] | +0.00pp [+0.00, +0.00] |
| `arm3c_selective_agreement_gate` | +12.16pp [+6.48, +18.30] | +1.29pp [-0.31, +2.64] | -4.72pp [-7.05, -2.62] | +27.40pp [+13.04, +44.38] | +9.78pp [-3.12, +22.37] | -21.29pp [-26.80, -15.67] |

#### Source-Clustered Bootstrap 95% CIs — Stratum-Weighted (Exploratory & Small-Stratum Unstable) (direct)

> **Small-Stratum Instability Note**: Reweighting by validation stratum populations amplifies small strata (`Data2txt/NO_CONTRA` `n=2`, `QA/NO_CONTRA` `n=3`, `Summary/CONTRA_PRESENT` `n=4`, max relative weight `2.40`), widening bootstrap variance. Do not treat these 30-source weighted intervals as definitive population CIs.

| Candidate Arm | Δ Macro-F1 (95% CI) | Δ Sup-Weighted F1 (95% CI) | Δ Pen. Acc (95% CI) | Δ CONTRA F1 (95% CI) | Δ UNSUP F1 (95% CI) | Δ Bin-F1 (95% CI) |
|---|---:|---:|---:|---:|---:|---:|
| `arm0a_nli_only` | -26.44pp [-31.50, -21.19] | -51.90pp [-63.56, -40.95] | -47.51pp [-55.72, -40.06] | -3.86pp [-8.58, -0.29] | -17.07pp [-21.33, -11.34] | -19.19pp [-22.98, -14.77] |
| `arm0b_minicheck_only` | -3.61pp [-5.97, -0.59] | +0.07pp [-2.10, +2.87] | +0.69pp [-2.44, +4.38] | -11.90pp [-20.62, -1.54] | +0.57pp [-1.18, +3.14] | +1.38pp [-0.46, +3.78] |
| `arm2_custom_deberta_only` | +21.17pp [+15.50, +27.59] | +13.02pp [+8.83, +17.98] | +18.55pp [+13.11, +24.99] | +40.63pp [+27.20, +56.62] | +10.67pp [-0.45, +25.70] | +7.60pp [-1.86, +19.88] |
| `arm3a_mc_plus_custom_deberta` | +20.59pp [+15.12, +26.66] | +13.10pp [+9.23, +17.68] | +18.87pp [+13.77, +24.90] | +39.90pp [+25.42, +57.21] | +9.37pp [-0.84, +22.20] | +7.43pp [-1.45, +18.25] |
| `arm3b_baseline_support_deberta_disambig` | +13.19pp [+8.41, +19.05] | +1.35pp [+0.74, +1.99] | +1.18pp [+0.55, +1.81] | +39.90pp [+25.42, +57.21] | -0.32pp [-1.57, +0.83] | +0.00pp [+0.00, +0.00] |
| `arm3c_selective_agreement_gate` | +15.74pp [+10.79, +21.55] | +1.34pp [-0.73, +3.23] | -5.81pp [-9.02, -2.56] | +39.90pp [+25.42, +57.21] | +8.47pp [-1.60, +20.99] | -21.39pp [-26.67, -14.46] |

### E. Supplementary Response-Clustered (`R=177`) & Legacy Sentence McNemar (direct)

Response-clustered (`R=177`) tests are supplementary because responses sharing a `source_id` are correlated. Sentence-level McNemar (`n=1,250`) assumes independent sentences and is **anti-conservative** (kept only for legacy comparison).

| Candidate Arm | Resp Δ 3-Cls Acc | Resp Sign-Flip Holm p | Resp Wilcoxon Holm p | Resp Boot Δ Conf-MAE (95% CI) | Resp Boot Δ GFI-ρ (95% CI) | Legacy McNemar 3-Cls (b/c, net) | Legacy McNemar 3-Cls Raw p (Holm p) | Legacy McNemar Bin Raw p (Holm p) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `arm0a_nli_only` | -48.28pp | 0.0003 | 1.3457e-25 | +49.88pp [+44.39, +55.78] | -0.441 [-0.617, -0.297] | 635/18 (-617, baseline_better) | 3.165e-162 (1.8987e-161) | 1.937e-150 (1.1622e-149) |
| `arm0b_minicheck_only` | -0.02pp | 0.9754 | 0.9673 | -0.20pp [-2.60, +2.30] | +0.027 [-0.029, +0.090] | 29/32 (+3, candidate_better) | 0.798153 (0.7982) | 0.418754 (0.8375) |
| `arm2_custom_deberta_only` | +17.96pp | 0.0003 | 1.8612e-12 | -14.24pp [-19.37, -9.55] | -0.088 [-0.308, +0.128] | 73/291 (+218, candidate_better) | 6.209e-32 (2.4838e-31) | 7.316e-25 (2.1947e-24) |
| `arm3a_mc_plus_custom_deberta` | +18.17pp | 0.0003 | 5.2298e-13 | -14.55pp [-19.71, -9.67] | -0.059 [-0.272, +0.149] | 62/285 (+223, candidate_better) | 2.699e-35 (1.3495e-34) | 9.850e-28 (4.9249e-27) |
| `arm3b_baseline_support_deberta_disambig` | +0.46pp | 0.287 | 0.419 | +0.00pp [+0.00, +0.00] | +0.000 [+0.000, +0.000] | 6/15 (+9, candidate_better) | 0.0783539 (0.1567) | 1 (1) |
| `arm3c_selective_agreement_gate` | -5.39pp | 0.0003 | 6.1728e-07 | -14.77pp [-19.97, -10.06] | -0.009 [-0.035, +0.015] | 73/14 (-59, baseline_better) | 8.584e-11 (2.5753e-10) | 5.170e-26 (2.0680e-25) |

### F. Response-Level & Generator-LLM (`6` LLMs) Evaluation (direct)

Confirmed hallucination rate `(CONTRADICTED + UNSUPPORTED) / N` is reported separately from abstained `UNVERIFIABLE / N`; their sum is `(100 - GFI)%` (`non_supported_rate`).

| Arm | Resp Cov | True Conf Hall % | Pred Conf Hall % | Conf MAE (pp) | Conf Spearman ρ | Pred (100-GFI) % | Pred UNVER % | Excl-Mixed MAE (ρ) | 6-LLM Mean-Hall ρ | 6-LLM Pooled ρ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `arm0a_nli_only` | 100.0% | 13.41% | 87.14% | 74.01 | 0.062 | 87.14% | 0.00% | 74.62 (0.057) | 0.600 | -0.257 |
| `arm0b_minicheck_only` | 100.0% | 13.41% | 35.29% | 23.93 | 0.531 | 35.29% | 0.00% | 24.54 (0.508) | 0.943 | 0.829 |
| `arm1_baseline` | 100.0% | 13.41% | 35.13% | 24.13 | 0.503 | 35.13% | 0.00% | 24.74 (0.479) | 0.943 | 0.886 |
| `arm2_custom_deberta_only` | 100.0% | 13.41% | 8.63% | 9.89 | 0.415 | 8.63% | 0.00% | 10.18 (0.372) | 0.771 | 0.600 |
| `arm3a_mc_plus_custom_deberta` | 100.0% | 13.41% | 7.85% | 9.58 | 0.444 | 7.85% | 0.00% | 9.87 (0.400) | 0.771 | 0.600 |
| `arm3b_baseline_support_deberta_disambig` | 100.0% | 13.41% | 35.13% | 24.13 | 0.503 | 35.13% | 0.00% | 24.74 (0.479) | 0.943 | 0.886 |
| `arm3c_selective_agreement_gate` | 69.5% | 13.41% | 6.63% | 9.36 | 0.442 | 37.12% | 30.49% | 9.66 (0.390) | 0.943 | 0.657 |

#### Generator-LLM Breakdown (`arm1_baseline` vs `arm2_custom_deberta_only`) (direct)

| Generator Model | Resp (Sent) | True Hall Mean % | Baseline Conf Mean % (GFI %) | Baseline UNVER Pooled % | DeBERTa Conf Mean % (GFI %) | DeBERTa UNVER Pooled % |
|---|---:|---:|---:|---:|---:|---:|
| `gpt-3.5-turbo-0613` | 28 (260) | 5.87% | 19.36% (80.6%) | 0.00% | 2.29% (97.7%) | 0.00% |
| `gpt-4-0613` | 29 (163) | 3.38% | 25.64% (74.4%) | 0.00% | 3.52% (96.5%) | 0.00% |
| `llama-2-13b-chat` | 30 (217) | 19.18% | 43.07% (56.9%) | 0.00% | 16.12% (83.9%) | 0.00% |
| `llama-2-70b-chat` | 30 (217) | 14.59% | 36.14% (63.9%) | 0.00% | 7.84% (92.2%) | 0.00% |
| `llama-2-7b-chat` | 30 (215) | 20.53% | 45.48% (54.5%) | 0.00% | 9.81% (90.2%) | 0.00% |
| `mistral-7B-instruct` | 30 (187) | 16.09% | 39.70% (60.3%) | 0.00% | 11.57% (88.4%) | 0.00% |

### G. MIXED_HALLUCINATION Separate Diagnostic (direct)

| Arm | n_mixed | Confirmed Flagged (`CONTRA+UNSUP`) | Abstained (`UNVERIFIABLE`) | Falsely `SUPPORTED_PROXY` |
|---|---:|---:|---:|---:|
| `arm0a_nli_only` | 9 | 100.0% (9) | 0.0% (0) | 0.0% (0) |
| `arm0b_minicheck_only` | 9 | 88.9% (8) | 0.0% (0) | 11.1% (1) |
| `arm1_baseline` | 9 | 88.9% (8) | 0.0% (0) | 11.1% (1) |
| `arm2_custom_deberta_only` | 9 | 88.9% (8) | 0.0% (0) | 11.1% (1) |
| `arm3a_mc_plus_custom_deberta` | 9 | 88.9% (8) | 0.0% (0) | 11.1% (1) |
| `arm3b_baseline_support_deberta_disambig` | 9 | 88.9% (8) | 0.0% (0) | 11.1% (1) |
| `arm3c_selective_agreement_gate` | 9 | 88.9% (8) | 0.0% (0) | 11.1% (1) |

## 5. Condition 2 — Chunked Retrieval End-to-End Ablation

- Diagnostics: Direct pairs >512 tokens = `754` (59.89%), Retrieval top-1 >512 tokens = `0`, Retrieval mean sim = `0.5185`, <0.25 floor = `99`, Rule 4 `NLI-xsmall` secondary-conflict triggers = `11`
- **Retrieval Caveat**: Rule 4 secondary-evidence conflict resolution inside `ClaimVerifier` invokes `NLI-xsmall` (`cross-encoder/nli-deberta-v3-xsmall`); candidate arms (`arm3a` uses uncalibrated DeBERTa logits with legacy NLI thresholds) do not represent a calibrated or NLI-free retrieval pipeline.

### A. Claim-Level 3-Class & Binary Detection (retrieval)

| Arm | Coverage | Pen. Macro-F1 | Sup-Weighted F1 | Sel. Macro-F1 | Pen. Acc | CONTRA F1 | SUP_P F1 | UNSUP F1 | Bin-F1 (Strict) | AUROC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `arm0a_nli_only` | 100.0% (1250/1250) | 20.94% | 29.68% | 20.94% | 25.36% | 11.97% | 31.52% | 19.33% | 24.98% | 0.589 | 0.155 |
| `arm0b_minicheck_only` | 100.0% (1250/1250) | 29.34% | 57.42% | 29.34% | 49.28% | 0.00% | 63.29% | 24.73% | 32.31% | 0.716 | 0.214 |
| `arm1_baseline` | 91.7% (1146/1250) | 33.13% | 56.65% | 34.94% | 47.36% | 12.05% | 61.82% | 25.53% | 27.69% | N/A (discrete) | N/A (discrete) |
| `arm2_custom_deberta_only` | 100.0% (1250/1250) | 50.63% | 79.71% | 50.63% | 77.28% | 30.95% | 86.56% | 34.37% | 36.90% | 0.779 | 0.393 |
| `arm3a_mc_plus_custom_deberta` | 91.7% (1146/1250) | 44.61% | 75.29% | 47.18% | 70.08% | 19.72% | 82.22% | 31.88% | 26.29% | N/A (discrete) | N/A (discrete) |
| `arm3b_baseline_support_deberta_disambig` | 91.7% (1146/1250) | 35.65% | 56.93% | 37.96% | 47.68% | 19.44% | 61.82% | 25.70% | 27.69% | N/A (discrete) | N/A (discrete) |
| `arm3c_selective_agreement_gate` | 60.2% (753/1250) | 37.28% | 57.09% | 50.58% | 43.92% | 19.44% | 61.44% | 30.95% | 15.31% | N/A (discrete) | N/A (discrete) |

### B. Per-Task, HT-Reweighted & Length/Duplicate Slices (retrieval)

| Arm | Summary Macro-F1 (Acc) | QA Macro-F1 (Acc) | Data2txt Macro-F1 (Acc) | HT-Weighted Macro-F1 (Sup-W F1) | <=512 tok Macro-F1 | >512 tok Macro-F1 | No Exact Train-Dup Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `arm0a_nli_only` | 15.58% (24.1%) | 33.82% (42.6%) | 13.79% (13.8%) | 20.09% (26.93%) | 32.27% | 12.40% | 20.94% |
| `arm0b_minicheck_only` | 29.97% (56.1%) | 36.83% (58.2%) | 23.16% (38.4%) | 30.40% (56.43%) | 35.70% | 25.09% | 29.34% |
| `arm1_baseline` | 29.58% (54.9%) | 44.72% (59.8%) | 26.05% (33.5%) | 33.60% (54.82%) | 44.03% | 25.35% | 33.13% |
| `arm2_custom_deberta_only` | 38.36% (83.7%) | 45.31% (79.6%) | 51.56% (71.4%) | 54.99% (79.86%) | 53.39% | 49.06% | 50.63% |
| `arm3a_mc_plus_custom_deberta` | 37.77% (81.4%) | 43.32% (71.2%) | 43.33% (61.9%) | 47.78% (74.50%) | 49.40% | 41.67% | 44.61% |
| `arm3b_baseline_support_deberta_disambig` | 29.64% (55.2%) | 39.11% (59.3%) | 30.92% (34.5%) | 38.37% (55.33%) | 43.24% | 30.98% | 35.65% |
| `arm3c_selective_agreement_gate` | 31.34% (52.3%) | 38.91% (53.2%) | 32.98% (31.8%) | 39.69% (55.61%) | 45.15% | 32.01% | 37.28% |

### C. Per-Class P / R / F1 & 3x4 Confusion Matrices (retrieval)

| Arm | Class | P | R | F1 | Support | Pred: CONTRA | Pred: SUP_P | Pred: UNSUP | Pred: UNVER |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `arm0a_nli_only` | `CONTRADICTED` | 9.86% | 15.22% | 11.97% | 46 | 7 | 7 | 32 | 0 |
| `arm0a_nli_only` | `SUPPORTED_PROXY` | 94.50% | 18.92% | 31.52% | 1089 | 58 | 206 | 825 | 0 |
| `arm0a_nli_only` | `UNSUPPORTED` | 10.82% | 90.43% | 19.33% | 115 | 6 | 5 | 104 | 0 |
| `arm0b_minicheck_only` | `CONTRADICTED` | 0.00% | 0.00% | 0.00% | 46 | 0 | 7 | 39 | 0 |
| `arm0b_minicheck_only` | `SUPPORTED_PROXY` | 96.43% | 47.11% | 63.29% | 1089 | 0 | 513 | 576 | 0 |
| `arm0b_minicheck_only` | `UNSUPPORTED` | 14.35% | 89.57% | 24.73% | 115 | 0 | 12 | 103 | 0 |
| `arm1_baseline` | `CONTRADICTED` | 13.51% | 10.87% | 12.05% | 46 | 5 | 7 | 24 | 10 |
| `arm1_baseline` | `SUPPORTED_PROXY` | 95.76% | 45.64% | 61.82% | 1089 | 30 | 497 | 476 | 86 |
| `arm1_baseline` | `UNSUPPORTED` | 15.25% | 78.26% | 25.53% | 115 | 2 | 15 | 90 | 8 |
| `arm2_custom_deberta_only` | `CONTRADICTED` | 34.21% | 28.26% | 30.95% | 46 | 13 | 29 | 4 | 0 |
| `arm2_custom_deberta_only` | `SUPPORTED_PROXY` | 91.77% | 81.91% | 86.56% | 1089 | 22 | 892 | 175 | 0 |
| `arm2_custom_deberta_only` | `UNSUPPORTED` | 25.42% | 53.04% | 34.37% | 115 | 3 | 51 | 61 | 0 |
| `arm3a_mc_plus_custom_deberta` | `CONTRADICTED` | 28.00% | 15.22% | 19.72% | 46 | 7 | 25 | 4 | 10 |
| `arm3a_mc_plus_custom_deberta` | `SUPPORTED_PROXY` | 91.36% | 74.75% | 82.22% | 1089 | 18 | 814 | 171 | 86 |
| `arm3a_mc_plus_custom_deberta` | `UNSUPPORTED` | 23.91% | 47.83% | 31.88% | 115 | 0 | 52 | 55 | 8 |
| `arm3b_baseline_support_deberta_disambig` | `CONTRADICTED` | 26.92% | 15.22% | 19.44% | 46 | 7 | 7 | 22 | 10 |
| `arm3b_baseline_support_deberta_disambig` | `SUPPORTED_PROXY` | 95.76% | 45.64% | 61.82% | 1089 | 19 | 497 | 487 | 86 |
| `arm3b_baseline_support_deberta_disambig` | `UNSUPPORTED` | 15.31% | 80.00% | 25.70% | 115 | 0 | 15 | 92 | 8 |
| `arm3c_selective_agreement_gate` | `CONTRADICTED` | 26.92% | 15.22% | 19.44% | 46 | 7 | 7 | 4 | 28 |
| `arm3c_selective_agreement_gate` | `SUPPORTED_PROXY` | 96.84% | 45.00% | 61.44% | 1089 | 19 | 490 | 165 | 415 |
| `arm3c_selective_agreement_gate` | `UNSUPPORTED` | 23.53% | 45.22% | 30.95% | 115 | 0 | 9 | 52 | 54 |

### D. Primary Paired Comparisons vs `arm1_baseline` — Source-Clustered (`K=30`) (retrieval)

Family-wise **Holm–Bonferroni** adjustment (`m=6` candidate arms) applied per test family within this regime.

| Candidate Arm | Unw. Δ 3-Cls Acc | Clusters (+/-/=) | Unw. Sign-Flip Raw p | Unw. Sign-Flip Holm p | Exact Sign Raw p (Holm) | Wt. Δ 3-Cls Acc | Wt. Sign-Flip Raw p | Wt. Sign-Flip Holm p | Bin Acc Δ (Unw. Holm p) |
|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| `arm0a_nli_only` | -21.67pp | 1/28/1 | 4.9998e-05 | 0.0003 | 1.118e-07 (4.4703e-07) | -20.53pp | 4.9998e-05 | 0.0003 | -20.82pp (0.0003) |
| `arm0b_minicheck_only` | +1.55pp | 19/6/5 | 0.1037 | 0.2074 | 0.0146333 (0.02927) | +2.81pp | 0.0084 | 0.0168 | +2.45pp (0.0373) |
| `arm2_custom_deberta_only` | +27.75pp | 29/0/1 | 4.9998e-05 | 0.0003 | 3.725e-09 (2.2352e-08) | +28.28pp | 4.9998e-05 | 0.0003 | +26.22pp (0.0003) |
| `arm3a_mc_plus_custom_deberta` | +20.76pp | 29/1/0 | 4.9998e-05 | 0.0003 | 5.774e-08 (2.8871e-07) | +19.46pp | 4.9998e-05 | 0.0003 | +19.01pp (0.0003) |
| `arm3b_baseline_support_deberta_disambig` | +0.23pp | 6/2/22 | 0.2697 | 0.2697 | 0.289062 (0.2891) | +0.42pp | 0.0477 | 0.0477 | +0.00pp (1) |
| `arm3c_selective_agreement_gate` | -4.49pp | 2/19/9 | 4.9998e-05 | 0.0003 | 0.000221252 (0.0006638) | -6.14pp | 4.9998e-05 | 0.0003 | -6.24pp (0.0003) |

#### Source-Clustered Bootstrap 95% CIs — Unweighted (`K=30`, Primary Exploratory) (retrieval)

| Candidate Arm | Δ Macro-F1 (95% CI) | Δ Sup-Weighted F1 (95% CI) | Δ Pen. Acc (95% CI) | Δ CONTRA F1 (95% CI) | Δ UNSUP F1 (95% CI) | Δ Bin-F1 (95% CI) |
|---|---:|---:|---:|---:|---:|---:|
| `arm0a_nli_only` | -12.19pp [-14.79, -9.47] | -26.97pp [-30.61, -23.01] | -22.00pp [-25.96, -18.09] | -0.08pp [-4.65, +4.67] | -6.20pp [-8.61, -3.58] | -2.71pp [-4.95, -0.47] |
| `arm0b_minicheck_only` | -3.79pp [-6.96, -0.57] | +0.77pp [-0.59, +2.26] | +1.92pp [+0.40, +3.46] | -12.05pp [-21.69, -2.70] | -0.80pp [-2.25, +0.93] | +4.62pp [+2.94, +6.44] |
| `arm2_custom_deberta_only` | +17.49pp [+12.51, +22.98] | +23.07pp [+18.98, +27.46] | +29.92pp [+25.70, +34.30] | +18.90pp [+7.77, +32.17] | +8.83pp [+2.14, +16.35] | +9.21pp [+2.85, +15.97] |
| `arm3a_mc_plus_custom_deberta` | +11.48pp [+7.42, +15.86] | +18.64pp [+15.14, +22.40] | +22.72pp [+19.06, +26.36] | +7.67pp [-1.44, +17.93] | +6.35pp [+0.87, +12.81] | -1.39pp [-6.08, +3.65] |
| `arm3b_baseline_support_deberta_disambig` | +2.52pp [-0.67, +6.16] | +0.29pp [-0.06, +0.68] | +0.32pp [+0.00, +0.68] | +7.40pp [-1.84, +17.87] | +0.17pp [-0.56, +1.03] | +0.00pp [+0.00, +0.00] |
| `arm3c_selective_agreement_gate` | +4.15pp [+0.35, +8.19] | +0.45pp [-0.48, +1.39] | -3.44pp [-5.16, -1.90] | +7.40pp [-1.84, +17.87] | +5.42pp [+0.12, +11.29] | -12.38pp [-15.88, -8.52] |

#### Source-Clustered Bootstrap 95% CIs — Stratum-Weighted (Exploratory & Small-Stratum Unstable) (retrieval)

> **Small-Stratum Instability Note**: Reweighting by validation stratum populations amplifies small strata (`Data2txt/NO_CONTRA` `n=2`, `QA/NO_CONTRA` `n=3`, `Summary/CONTRA_PRESENT` `n=4`, max relative weight `2.40`), widening bootstrap variance. Do not treat these 30-source weighted intervals as definitive population CIs.

| Candidate Arm | Δ Macro-F1 (95% CI) | Δ Sup-Weighted F1 (95% CI) | Δ Pen. Acc (95% CI) | Δ CONTRA F1 (95% CI) | Δ UNSUP F1 (95% CI) | Δ Bin-F1 (95% CI) |
|---|---:|---:|---:|---:|---:|---:|
| `arm0a_nli_only` | -13.51pp [-15.72, -10.98] | -27.89pp [-31.94, -23.64] | -21.87pp [-26.21, -17.37] | -1.54pp [-4.67, +1.89] | -7.36pp [-10.56, -3.38] | -2.85pp [-5.54, +0.14] |
| `arm0b_minicheck_only` | -3.21pp [-6.21, +0.30] | +1.61pp [+0.18, +3.06] | +3.17pp [+1.76, +4.50] | -11.97pp [-20.98, -1.44] | -0.03pp [-2.04, +2.67] | +5.84pp [+4.04, +7.80] |
| `arm2_custom_deberta_only` | +21.39pp [+16.61, +27.52] | +25.04pp [+20.40, +29.66] | +31.68pp [+26.39, +36.83] | +27.66pp [+16.83, +41.91] | +9.55pp [+0.94, +18.87] | +12.48pp [+5.47, +19.79] |
| `arm3a_mc_plus_custom_deberta` | +14.18pp [+10.08, +18.62] | +19.68pp [+15.45, +23.96] | +22.92pp [+17.85, +28.01] | +14.16pp [+5.75, +24.38] | +6.80pp [-0.20, +14.60] | -0.33pp [-6.10, +5.68] |
| `arm3b_baseline_support_deberta_disambig` | +4.77pp [+1.77, +8.29] | +0.51pp [+0.17, +0.88] | +0.53pp [+0.19, +0.88] | +14.05pp [+5.00, +24.55] | +0.25pp [-0.75, +1.26] | +0.00pp [+0.00, +0.00] |
| `arm3c_selective_agreement_gate` | +6.08pp [+2.53, +9.99] | +0.79pp [-0.25, +1.60] | -4.00pp [-6.80, -1.70] | +14.05pp [+5.00, +24.55] | +4.42pp [-1.84, +11.09] | -12.88pp [-17.68, -7.78] |

### E. Supplementary Response-Clustered (`R=177`) & Legacy Sentence McNemar (retrieval)

Response-clustered (`R=177`) tests are supplementary because responses sharing a `source_id` are correlated. Sentence-level McNemar (`n=1,250`) assumes independent sentences and is **anti-conservative** (kept only for legacy comparison).

| Candidate Arm | Resp Δ 3-Cls Acc | Resp Sign-Flip Holm p | Resp Wilcoxon Holm p | Resp Boot Δ Conf-MAE (95% CI) | Resp Boot Δ GFI-ρ (95% CI) | Legacy McNemar 3-Cls (b/c, net) | Legacy McNemar 3-Cls Raw p (Holm p) | Legacy McNemar Bin Raw p (Holm p) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `arm0a_nli_only` | -20.48pp | 0.0003 | 2.3153e-19 | +29.61pp [+25.50, +33.74] | -0.060 [-0.165, +0.062] | 302/27 (-275, baseline_better) | 5.669e-60 (2.8347e-59) | 1.570e-51 (7.8483e-51) |
| `arm0b_minicheck_only` | +1.00pp | 0.4806 | 0.5257 | +7.26pp [+5.00, +10.30] | +0.006 [-0.052, +0.062] | 39/63 (+24, candidate_better) | 0.0222977 (0.0446) | 0.0003907 (0.0007814) |
| `arm2_custom_deberta_only` | +29.26pp | 0.0003 | 2.2453e-21 | -20.99pp [-25.20, -16.79] | -0.026 [-0.201, +0.169] | 48/422 (+374, candidate_better) | 9.048e-76 (5.4289e-75) | 1.202e-64 (7.2113e-64) |
| `arm3a_mc_plus_custom_deberta` | +22.32pp | 0.0003 | 7.9283e-20 | -21.80pp [-25.99, -17.72] | -0.126 [-0.257, +0.004] | 44/328 (+284, candidate_better) | 8.111e-55 (3.2444e-54) | 3.520e-44 (1.4079e-43) |
| `arm3b_baseline_support_deberta_disambig` | +0.24pp | 0.4806 | 0.5257 | +0.00pp [+0.00, +0.00] | +0.000 [+0.000, +0.000] | 2/6 (+4, candidate_better) | 0.289062 (0.2891) | 1 (1) |
| `arm3c_selective_agreement_gate` | -3.86pp | 0.0003 | 5.2455e-06 | -22.19pp [-26.36, -17.99] | +0.014 [-0.003, +0.037] | 48/5 (-43, baseline_better) | 7.077e-10 (2.1232e-09) | 5.421e-20 (1.6263e-19) |

### F. Response-Level & Generator-LLM (`6` LLMs) Evaluation (retrieval)

Confirmed hallucination rate `(CONTRADICTED + UNSUPPORTED) / N` is reported separately from abstained `UNVERIFIABLE / N`; their sum is `(100 - GFI)%` (`non_supported_rate`).

| Arm | Resp Cov | True Conf Hall % | Pred Conf Hall % | Conf MAE (pp) | Conf Spearman ρ | Pred (100-GFI) % | Pred UNVER % | Excl-Mixed MAE (ρ) | 6-LLM Mean-Hall ρ | 6-LLM Pooled ρ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `arm0a_nli_only` | 100.0% | 13.41% | 82.05% | 68.78 | 0.311 | 82.05% | 0.00% | 69.39 (0.269) | 0.943 | 0.886 |
| `arm0b_minicheck_only` | 100.0% | 13.41% | 58.84% | 46.43 | 0.377 | 58.84% | 0.00% | 47.04 (0.365) | 0.886 | 0.943 |
| `arm1_baseline` | 91.8% | 13.41% | 50.93% | 39.17 | 0.430 | 59.09% | 8.16% | 39.78 (0.412) | 0.771 | 0.886 |
| `arm2_custom_deberta_only` | 100.0% | 13.41% | 22.58% | 18.18 | 0.345 | 22.58% | 0.00% | 18.79 (0.310) | 0.714 | 0.600 |
| `arm3a_mc_plus_custom_deberta` | 91.8% | 13.41% | 21.02% | 17.37 | 0.316 | 29.18% | 8.16% | 17.98 (0.288) | 0.771 | 0.543 |
| `arm3b_baseline_support_deberta_disambig` | 91.8% | 13.41% | 50.93% | 39.17 | 0.430 | 59.09% | 8.16% | 39.78 (0.412) | 0.771 | 0.886 |
| `arm3c_selective_agreement_gate` | 60.4% | 13.41% | 20.41% | 16.98 | 0.314 | 60.04% | 39.63% | 17.59 (0.284) | 0.714 | 0.600 |

#### Generator-LLM Breakdown (`arm1_baseline` vs `arm2_custom_deberta_only`) (retrieval)

| Generator Model | Resp (Sent) | True Hall Mean % | Baseline Conf Mean % (GFI %) | Baseline UNVER Pooled % | DeBERTa Conf Mean % (GFI %) | DeBERTa UNVER Pooled % |
|---|---:|---:|---:|---:|---:|---:|
| `gpt-3.5-turbo-0613` | 28 (260) | 5.87% | 32.79% (57.3%) | 9.62% | 10.82% (89.2%) | 0.00% |
| `gpt-4-0613` | 29 (163) | 3.38% | 39.65% (50.8%) | 7.98% | 15.55% (84.4%) | 0.00% |
| `llama-2-13b-chat` | 30 (217) | 19.18% | 55.02% (34.3%) | 11.98% | 26.92% (73.1%) | 0.00% |
| `llama-2-70b-chat` | 30 (217) | 14.59% | 56.66% (39.7%) | 4.61% | 24.96% (75.0%) | 0.00% |
| `llama-2-7b-chat` | 30 (215) | 20.53% | 61.95% (29.8%) | 9.77% | 25.61% (74.4%) | 0.00% |
| `mistral-7B-instruct` | 30 (187) | 16.09% | 57.92% (34.9%) | 5.88% | 30.60% (69.4%) | 0.00% |

### G. MIXED_HALLUCINATION Separate Diagnostic (retrieval)

| Arm | n_mixed | Confirmed Flagged (`CONTRA+UNSUP`) | Abstained (`UNVERIFIABLE`) | Falsely `SUPPORTED_PROXY` |
|---|---:|---:|---:|---:|
| `arm0a_nli_only` | 9 | 100.0% (9) | 0.0% (0) | 0.0% (0) |
| `arm0b_minicheck_only` | 9 | 100.0% (9) | 0.0% (0) | 0.0% (0) |
| `arm1_baseline` | 9 | 77.8% (7) | 22.2% (2) | 0.0% (0) |
| `arm2_custom_deberta_only` | 9 | 100.0% (9) | 0.0% (0) | 0.0% (0) |
| `arm3a_mc_plus_custom_deberta` | 9 | 77.8% (7) | 22.2% (2) | 0.0% (0) |
| `arm3b_baseline_support_deberta_disambig` | 9 | 77.8% (7) | 22.2% (2) | 0.0% (0) |
| `arm3c_selective_agreement_gate` | 9 | 77.8% (7) | 22.2% (2) | 0.0% (0) |

