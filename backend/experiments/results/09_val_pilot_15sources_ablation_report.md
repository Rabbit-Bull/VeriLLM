# Phase 8 Step 4 — Validation Ablation Pilot Report (15 Sources)

- **Generated UTC**: `2026-11-12T07:36:32.022175+00:00`
- **Dataset**: `ragtruth_val.jsonl` ONLY (`ragtruth_test.jsonl` untouched)
- **Manifest Selection Method**: `stage2_minority_stratified_source_allocation` (`seed=42`)
- **Selection Rationale**: Stage 1 uniform selection failed minority adequacy checks (total CONTRADICTED=7 < 25; Summary CONTRADICTED=1 < 3; QA CONTRADICTED=0 < 3). Revised deterministically (seed=42) by reserving k_contra=4 random source(s) per task from the CONTRA_PRESENT stratum and filling remaining slots uniformly from NO_CONTRA sources while keeping all responses and sentences per source intact.
- **Pilot Scope**: `15` sources, `89` complete responses, `636` sentences

## 1. Pilot Class Distribution & Adequacy Audit

| Slice | Total | SUPPORTED_PROXY | CONTRADICTED | UNSUPPORTED | MIXED_HALLUCINATION |
|---|---:|---:|---:|---:|---:|
| Full `ragtruth_val.jsonl` (375 src) | 15792 | 13970 | 581 | 1167 | 74 |
| **Selected Pilot** (15 src) | 636 | 536 | 34 | 62 | 4 |
| Pilot `Summary` | 168 | 157 | 6 | 5 | 0 |
| Pilot `QA` | 189 | 138 | 14 | 37 | 0 |
| Pilot `Data2txt` | 279 | 241 | 14 | 20 | 4 |

## 2. Train-Validation Leakage & Duplicate Separation

- `source_id` overlap with `ragtruth_train.jsonl`: **0**
- Exact `(context_text, sentence_text)` duplicates in train: **0** (IDs: `[]`)
- Repeated `sentence_text` strings appearing in train (total / string-only diff context): **16** / **16**
- Within-pilot extra pair duplicates / repeated sentence strings: **14** / **20**

## 3. Actual Local CPU Warm-Up Latency Benchmark

- Cold model load seconds: `{'claim_verifier_minicheck_plus_nli': 3.716, 'custom_deberta': 4.871, 'passage_retriever': 0.978, 'total_model_init': 9.566}`
- Measured on warm-up slice (`16` sentences, `3` sources): MiniCheck=7016.41 ms/pair, NLI-xsmall=1007.64 ms/pair, CustomDeBERTa=1008.74 ms/pair, Chunk+Retrieve=66.97 ms/pair
- Actual pilot execution wall time: **10331.6s** (estimated dual-regime runtime was 192.21 min; projected 30-source runtime ≈ 384.41 min)

## 4. Condition 1 — Direct Source Context Ablation

- Diagnostics: Direct pairs >512 tokens = `347` (54.56%), Retrieval top-1 >512 tokens = `0`, Retrieval mean sim = `0.5181`, <0.25 floor = `48`, Rule 4 conflicts = `6`

| Arm | Coverage | Pen. Macro-F1 | Sel. Macro-F1 | Pen. Acc | CONTRA F1 | SUP_P F1 | UNSUP F1 | Bin-F1 (Strict) | AUROC | PR-AUC | Resp Spearman ρ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `arm0a_nli_only` | 100.0% (632/632) | 18.73% | 18.73% | 23.42% | 8.33% | 27.24% | 20.61% | 29.06% | 0.641 | 0.245 | 0.210 |
| `arm0b_minicheck_only` | 100.0% (632/632) | 40.24% | 40.24% | 70.73% | 0.00% | 83.12% | 37.59% | 49.37% | 0.842 | 0.476 | 0.588 |
| `arm1_baseline` | 100.0% (632/632) | 44.32% | 44.32% | 71.52% | 10.26% | 83.28% | 39.42% | 49.20% | 0.842 | 0.476 | 0.552 |
| `arm2_custom_deberta_only` | 100.0% (632/632) | 61.31% | 61.31% | 88.29% | 38.30% | 93.64% | 52.00% | 51.70% | 0.864 | 0.642 | 0.443 |
| `arm3a_mc_plus_custom_deberta` | 100.0% (632/632) | 59.08% | 59.08% | 87.82% | 39.13% | 93.42% | 44.68% | 47.14% | 0.864 | 0.642 | 0.426 |
| `arm3b_baseline_support_deberta_disambig` | 100.0% (632/632) | 53.79% | 53.79% | 72.31% | 39.13% | 83.28% | 38.95% | 49.20% | 0.792 | 0.530 | 0.552 |
| `arm3c_selective_agreement_gate` | 70.9% (448/632) | 56.00% | 75.57% | 66.77% | 39.13% | 83.23% | 45.65% | 24.18% | 0.864 | 0.642 | 0.556 |

### Statistical Comparisons vs `arm1_baseline` (direct)

| Candidate Arm | Δ Macro-F1 (95% CI) | Δ CONTRA F1 (95% CI) | Δ UNSUP F1 (95% CI) | McNemar 3-cls p | McNemar Bin p |
|---|---:|---:|---:|---:|---:|
| `arm0a_nli_only` | -25.59pp [-32.18, -18.82] | -1.92pp [-7.24, +0.00] | -18.81pp [-22.95, -14.16] | 0 | 0 |
| `arm0b_minicheck_only` | -4.08pp [-8.99, +0.00] | -10.26pp [-24.24, +0.00] | -1.83pp [-4.04, +0.29] | 0.4049 | 1 |
| `arm2_custom_deberta_only` | +17.00pp [+9.41, +24.03] | +28.04pp [+10.30, +47.86] | +12.58pp [-6.40, +32.47] | 0 | 0 |
| `arm3a_mc_plus_custom_deberta` | +14.76pp [+8.31, +21.28] | +28.87pp [+11.36, +48.81] | +5.26pp [-12.16, +22.30] | 0 | 0 |
| `arm3b_baseline_support_deberta_disambig` | +9.47pp [+3.58, +16.41] | +28.87pp [+11.36, +48.81] | -0.46pp [-1.88, +0.56] | 0.2266 | 1 |
| `arm3c_selective_agreement_gate` | +11.69pp [+5.03, +18.26] | +28.87pp [+11.36, +48.81] | +6.24pp [-11.54, +24.41] | 9.25e-06 | 0 |

### MIXED_HALLUCINATION Separate Diagnostic (direct)

| Arm | n_mixed | Flagged Non-Supported Rate | Falsely SUPPORTED_PROXY Rate |
|---|---:|---:|---:|
| `arm0a_nli_only` | 4 | 100.0% (4) | 0.0% |
| `arm0b_minicheck_only` | 4 | 100.0% (4) | 0.0% |
| `arm1_baseline` | 4 | 100.0% (4) | 0.0% |
| `arm2_custom_deberta_only` | 4 | 100.0% (4) | 0.0% |
| `arm3a_mc_plus_custom_deberta` | 4 | 100.0% (4) | 0.0% |
| `arm3b_baseline_support_deberta_disambig` | 4 | 100.0% (4) | 0.0% |
| `arm3c_selective_agreement_gate` | 4 | 100.0% (4) | 0.0% |

## 5. Condition 2 — Chunked Retrieval End-to-End Ablation

- Diagnostics: Direct pairs >512 tokens = `347` (54.56%), Retrieval top-1 >512 tokens = `0`, Retrieval mean sim = `0.5181`, <0.25 floor = `48`, Rule 4 conflicts = `6`

| Arm | Coverage | Pen. Macro-F1 | Sel. Macro-F1 | Pen. Acc | CONTRA F1 | SUP_P F1 | UNSUP F1 | Bin-F1 (Strict) | AUROC | PR-AUC | Resp Spearman ρ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `arm0a_nli_only` | 100.0% (632/632) | 21.63% | 21.63% | 26.27% | 11.59% | 32.10% | 21.21% | 28.57% | 0.572 | 0.172 | 0.381 |
| `arm0b_minicheck_only` | 100.0% (632/632) | 29.06% | 29.06% | 47.63% | 0.00% | 61.89% | 25.29% | 35.39% | 0.708 | 0.251 | 0.427 |
| `arm1_baseline` | 91.9% (581/632) | 32.66% | 34.34% | 45.89% | 10.71% | 60.23% | 27.03% | 30.11% | 0.708 | 0.251 | 0.406 |
| `arm2_custom_deberta_only` | 100.0% (632/632) | 49.75% | 49.75% | 76.27% | 29.09% | 85.88% | 34.29% | 36.52% | 0.777 | 0.438 | 0.332 |
| `arm3a_mc_plus_custom_deberta` | 91.9% (581/632) | 42.98% | 45.43% | 69.30% | 17.39% | 81.96% | 29.59% | 24.41% | 0.777 | 0.438 | 0.211 |
| `arm3b_baseline_support_deberta_disambig` | 91.9% (581/632) | 34.45% | 36.56% | 46.04% | 16.67% | 60.23% | 26.46% | 30.11% | 0.706 | 0.309 | 0.406 |
| `arm3c_selective_agreement_gate` | 57.4% (363/632) | 35.69% | 50.13% | 41.77% | 16.67% | 60.10% | 30.30% | 14.49% | 0.758 | 0.359 | 0.411 |

### Statistical Comparisons vs `arm1_baseline` (retrieval)

| Candidate Arm | Δ Macro-F1 (95% CI) | Δ CONTRA F1 (95% CI) | Δ UNSUP F1 (95% CI) | McNemar 3-cls p | McNemar Bin p |
|---|---:|---:|---:|---:|---:|
| `arm0a_nli_only` | -11.02pp [-14.12, -8.35] | +0.88pp [-2.34, +2.88] | -5.82pp [-9.78, -1.60] | 0 | 0 |
| `arm0b_minicheck_only` | -3.60pp [-7.75, +0.34] | -10.71pp [-22.95, +0.00] | -1.74pp [-3.47, +0.33] | 0.1352 | 0.002602 |
| `arm2_custom_deberta_only` | +17.10pp [+10.78, +23.64] | +18.38pp [+5.48, +34.04] | +7.26pp [-1.07, +17.21] | 0 | 0 |
| `arm3a_mc_plus_custom_deberta` | +10.32pp [+5.64, +14.77] | +6.68pp [-4.02, +18.18] | +2.56pp [-3.66, +9.92] | 0 | 0 |
| `arm3b_baseline_support_deberta_disambig` | +1.79pp [-1.83, +5.73] | +5.95pp [-4.98, +16.99] | -0.57pp [-1.45, +0.36] | 1 | 1 |
| `arm3c_selective_agreement_gate` | +3.03pp [-1.41, +7.17] | +5.95pp [-4.98, +16.99] | +3.28pp [-2.96, +10.57] | 8.7e-07 | 0 |

### MIXED_HALLUCINATION Separate Diagnostic (retrieval)

| Arm | n_mixed | Flagged Non-Supported Rate | Falsely SUPPORTED_PROXY Rate |
|---|---:|---:|---:|
| `arm0a_nli_only` | 4 | 100.0% (4) | 0.0% |
| `arm0b_minicheck_only` | 4 | 100.0% (4) | 0.0% |
| `arm1_baseline` | 4 | 100.0% (4) | 0.0% |
| `arm2_custom_deberta_only` | 4 | 100.0% (4) | 0.0% |
| `arm3a_mc_plus_custom_deberta` | 4 | 100.0% (4) | 0.0% |
| `arm3b_baseline_support_deberta_disambig` | 4 | 100.0% (4) | 0.0% |
| `arm3c_selective_agreement_gate` | 4 | 100.0% (4) | 0.0% |

