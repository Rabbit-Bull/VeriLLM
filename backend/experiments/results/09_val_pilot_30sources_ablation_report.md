# Phase 8 Step 4 — Validation Ablation Pilot Report (30 Sources)

- **Generated UTC**: `2026-11-12T13:22:58.261791+00:00`
- **Dataset**: `ragtruth_val.jsonl` ONLY (`ragtruth_test.jsonl` untouched)
- **Manifest Selection Method**: `stage1_uniform_task_stratified` (`seed=42`)
- **Selection Rationale**: Uniform task-stratified selection (seed=42, 10 sources/task) satisfied all minority-class adequacy thresholds.
- **Pilot Scope**: `30` sources, `177` complete responses, `1259` sentences

## 1. Pilot Class Distribution & Adequacy Audit

| Slice | Total | SUPPORTED_PROXY | CONTRADICTED | UNSUPPORTED | MIXED_HALLUCINATION |
|---|---:|---:|---:|---:|---:|
| Full `ragtruth_val.jsonl` (375 src) | 15792 | 13970 | 581 | 1167 | 74 |
| **Selected Pilot** (30 src) | 1259 | 1089 | 46 | 115 | 9 |
| Pilot `Summary` | 344 | 318 | 6 | 20 | 0 |
| Pilot `QA` | 378 | 302 | 17 | 59 | 0 |
| Pilot `Data2txt` | 537 | 469 | 23 | 36 | 9 |

## 2. Train-Validation Leakage & Duplicate Separation

- `source_id` overlap with `ragtruth_train.jsonl`: **0**
- Exact `(context_text, sentence_text)` duplicates in train: **0** (IDs: `[]`)
- Repeated `sentence_text` strings appearing in train (total / string-only diff context): **44** / **44**
- Within-pilot extra pair duplicates / repeated sentence strings: **17** / **32**

## 3. Actual Local CPU Warm-Up Latency Benchmark

- Cold model load seconds: `{'claim_verifier_minicheck_plus_nli': 3.716, 'custom_deberta': 4.871, 'passage_retriever': 0.978, 'total_model_init': 9.566}`
- Measured on warm-up slice (`16` sentences, `3` sources): MiniCheck=7016.41 ms/pair, NLI-xsmall=1007.64 ms/pair, CustomDeBERTa=1008.74 ms/pair, Chunk+Retrieve=66.97 ms/pair
- Actual pilot execution wall time: **20424.52s** (estimated dual-regime runtime was 192.21 min; projected 30-source runtime ≈ 384.41 min)

## 4. Condition 1 — Direct Source Context Ablation

- Diagnostics: Direct pairs >512 tokens = `754` (59.89%), Retrieval top-1 >512 tokens = `0`, Retrieval mean sim = `0.5185`, <0.25 floor = `99`, Rule 4 conflicts = `11`

| Arm | Coverage | Pen. Macro-F1 | Sel. Macro-F1 | Pen. Acc | CONTRA F1 | SUP_P F1 | UNSUP F1 | Bin-F1 (Strict) | AUROC | PR-AUC | Resp Spearman ρ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `arm0a_nli_only` | 100.0% (1250/1250) | 18.05% | 18.05% | 21.84% | 9.71% | 25.38% | 19.06% | 25.02% | 0.600 | 0.173 | 0.062 |
| `arm0b_minicheck_only` | 100.0% (1250/1250) | 39.63% | 39.63% | 71.44% | 0.00% | 83.19% | 35.69% | 44.86% | 0.839 | 0.445 | 0.531 |
| `arm1_baseline` | 100.0% (1250/1250) | 44.15% | 44.15% | 71.20% | 13.70% | 82.82% | 35.94% | 43.76% | 0.839 | 0.445 | 0.503 |
| `arm2_custom_deberta_only` | 100.0% (1250/1250) | 61.22% | 61.22% | 88.64% | 42.11% | 93.85% | 47.72% | 49.82% | 0.840 | 0.557 | 0.415 |
| `arm3a_mc_plus_custom_deberta` | 100.0% (1250/1250) | 60.68% | 60.68% | 89.04% | 41.10% | 94.15% | 46.81% | 49.81% | 0.840 | 0.557 | 0.444 |
| `arm3b_baseline_support_deberta_disambig` | 100.0% (1250/1250) | 53.15% | 53.15% | 71.92% | 41.10% | 82.82% | 35.55% | 43.76% | 0.786 | 0.499 | 0.503 |
| `arm3c_selective_agreement_gate` | 71.0% (888/1250) | 56.31% | 73.23% | 66.48% | 41.10% | 82.12% | 45.71% | 22.47% | 0.840 | 0.557 | 0.494 |

### Per-Task Penalized Macro-F1 & Accuracy (direct)

| Arm | Summary Macro-F1 (Acc) | QA Macro-F1 (Acc) | Data2txt Macro-F1 (Acc) | Reweighted Macro-F1 | <=512 tok Macro-F1 | >512 tok Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| `arm0a_nli_only` | 3.70% (5.8%) | 39.82% (54.0%) | 9.20% (9.3%) | 17.45% | 30.36% | 9.64% |
| `arm0b_minicheck_only` | 42.46% (82.0%) | 44.65% (72.5%) | 33.57% (63.8%) | 40.27% | 44.42% | 35.97% |
| `arm1_baseline` | 42.02% (81.1%) | 52.57% (76.7%) | 37.42% (60.8%) | 43.89% | 51.55% | 38.65% |
| `arm2_custom_deberta_only` | 39.81% (92.7%) | 46.06% (82.5%) | 70.19% (90.3%) | 65.06% | 55.62% | 65.56% |
| `arm3a_mc_plus_custom_deberta` | 44.12% (93.3%) | 45.93% (82.8%) | 68.44% (90.7%) | 64.48% | 55.11% | 65.04% |
| `arm3b_baseline_support_deberta_disambig` | 41.90% (81.1%) | 46.93% (76.5%) | 52.06% (62.7%) | 57.08% | 53.60% | 53.39% |
| `arm3c_selective_agreement_gate` | 37.59% (76.7%) | 43.83% (66.9%) | 62.58% (59.5%) | 59.63% | 53.25% | 58.66% |

### Per-Class P / R / F1 & 3x4 Confusion Matrices (direct)

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

### Statistical Comparisons vs `arm1_baseline` (direct)

| Candidate Arm | Δ Macro-F1 (95% CI) | Δ CONTRA F1 (95% CI) | Δ UNSUP F1 (95% CI) | McNemar 3-cls p | McNemar Bin p |
|---|---:|---:|---:|---:|---:|
| `arm0a_nli_only` | -26.10pp [-30.54, -21.75] | -3.99pp [-9.02, -0.46] | -16.88pp [-20.55, -12.45] | 0 | 0 |
| `arm0b_minicheck_only` | -4.52pp [-7.88, -1.22] | -13.70pp [-23.88, -3.12] | -0.25pp [-2.10, +1.90] | 0.7982 | 0.4188 |
| `arm2_custom_deberta_only` | +17.07pp [+10.78, +23.30] | +28.41pp [+14.74, +44.58] | +11.78pp [-1.21, +24.49] | 0 | 0 |
| `arm3a_mc_plus_custom_deberta` | +16.53pp [+10.63, +22.74] | +27.40pp [+13.04, +44.38] | +10.87pp [-1.20, +23.09] | 0 | 0 |
| `arm3b_baseline_support_deberta_disambig` | +9.00pp [+4.15, +14.68] | +27.40pp [+13.04, +44.38] | -0.39pp [-1.46, +0.69] | 0.07835 | 1 |
| `arm3c_selective_agreement_gate` | +12.16pp [+6.48, +18.30] | +27.40pp [+13.04, +44.38] | +9.78pp [-3.12, +22.37] | 0 | 0 |

### MIXED_HALLUCINATION Separate Diagnostic (direct)

| Arm | n_mixed | Flagged Non-Supported Rate | Falsely SUPPORTED_PROXY Rate |
|---|---:|---:|---:|
| `arm0a_nli_only` | 9 | 100.0% (9) | 0.0% |
| `arm0b_minicheck_only` | 9 | 88.9% (8) | 11.1% |
| `arm1_baseline` | 9 | 88.9% (8) | 11.1% |
| `arm2_custom_deberta_only` | 9 | 88.9% (8) | 11.1% |
| `arm3a_mc_plus_custom_deberta` | 9 | 88.9% (8) | 11.1% |
| `arm3b_baseline_support_deberta_disambig` | 9 | 88.9% (8) | 11.1% |
| `arm3c_selective_agreement_gate` | 9 | 88.9% (8) | 11.1% |

## 5. Condition 2 — Chunked Retrieval End-to-End Ablation

- Diagnostics: Direct pairs >512 tokens = `754` (59.89%), Retrieval top-1 >512 tokens = `0`, Retrieval mean sim = `0.5185`, <0.25 floor = `99`, Rule 4 conflicts = `11`

| Arm | Coverage | Pen. Macro-F1 | Sel. Macro-F1 | Pen. Acc | CONTRA F1 | SUP_P F1 | UNSUP F1 | Bin-F1 (Strict) | AUROC | PR-AUC | Resp Spearman ρ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `arm0a_nli_only` | 100.0% (1250/1250) | 20.94% | 20.94% | 25.36% | 11.97% | 31.52% | 19.33% | 24.98% | 0.589 | 0.155 | 0.311 |
| `arm0b_minicheck_only` | 100.0% (1250/1250) | 29.34% | 29.34% | 49.28% | 0.00% | 63.29% | 24.73% | 32.31% | 0.716 | 0.214 | 0.377 |
| `arm1_baseline` | 91.7% (1146/1250) | 33.13% | 34.94% | 47.36% | 12.05% | 61.82% | 25.53% | 27.69% | 0.716 | 0.214 | 0.371 |
| `arm2_custom_deberta_only` | 100.0% (1250/1250) | 50.63% | 50.63% | 77.28% | 30.95% | 86.56% | 34.37% | 36.90% | 0.779 | 0.393 | 0.345 |
| `arm3a_mc_plus_custom_deberta` | 91.7% (1146/1250) | 44.61% | 47.18% | 70.08% | 19.72% | 82.22% | 31.88% | 26.29% | 0.779 | 0.393 | 0.245 |
| `arm3b_baseline_support_deberta_disambig` | 91.7% (1146/1250) | 35.65% | 37.96% | 47.68% | 19.44% | 61.82% | 25.70% | 27.69% | 0.713 | 0.280 | 0.371 |
| `arm3c_selective_agreement_gate` | 60.2% (753/1250) | 37.28% | 50.58% | 43.92% | 19.44% | 61.44% | 30.95% | 15.31% | 0.757 | 0.312 | 0.384 |

### Per-Task Penalized Macro-F1 & Accuracy (retrieval)

| Arm | Summary Macro-F1 (Acc) | QA Macro-F1 (Acc) | Data2txt Macro-F1 (Acc) | Reweighted Macro-F1 | <=512 tok Macro-F1 | >512 tok Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| `arm0a_nli_only` | 15.58% (24.1%) | 33.82% (42.6%) | 13.79% (13.8%) | 20.09% | 32.27% | 12.40% |
| `arm0b_minicheck_only` | 29.97% (56.1%) | 36.83% (58.2%) | 23.16% (38.4%) | 30.40% | 35.70% | 25.09% |
| `arm1_baseline` | 29.58% (54.9%) | 44.72% (59.8%) | 26.05% (33.5%) | 33.60% | 44.03% | 25.35% |
| `arm2_custom_deberta_only` | 38.36% (83.7%) | 45.31% (79.6%) | 51.56% (71.4%) | 54.99% | 53.39% | 49.06% |
| `arm3a_mc_plus_custom_deberta` | 37.77% (81.4%) | 43.32% (71.2%) | 43.33% (61.9%) | 47.78% | 49.40% | 41.67% |
| `arm3b_baseline_support_deberta_disambig` | 29.64% (55.2%) | 39.11% (59.3%) | 30.92% (34.5%) | 38.37% | 43.24% | 30.98% |
| `arm3c_selective_agreement_gate` | 31.34% (52.3%) | 38.91% (53.2%) | 32.98% (31.8%) | 39.69% | 45.15% | 32.01% |

### Per-Class P / R / F1 & 3x4 Confusion Matrices (retrieval)

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

### Statistical Comparisons vs `arm1_baseline` (retrieval)

| Candidate Arm | Δ Macro-F1 (95% CI) | Δ CONTRA F1 (95% CI) | Δ UNSUP F1 (95% CI) | McNemar 3-cls p | McNemar Bin p |
|---|---:|---:|---:|---:|---:|
| `arm0a_nli_only` | -12.19pp [-14.79, -9.47] | -0.08pp [-4.65, +4.67] | -6.20pp [-8.61, -3.58] | 0 | 0 |
| `arm0b_minicheck_only` | -3.79pp [-6.96, -0.57] | -12.05pp [-21.69, -2.70] | -0.80pp [-2.25, +0.93] | 0.0223 | 0.0003907 |
| `arm2_custom_deberta_only` | +17.49pp [+12.51, +22.98] | +18.90pp [+7.77, +32.17] | +8.83pp [+2.14, +16.35] | 0 | 0 |
| `arm3a_mc_plus_custom_deberta` | +11.48pp [+7.42, +15.86] | +7.67pp [-1.44, +17.93] | +6.35pp [+0.87, +12.81] | 0 | 0 |
| `arm3b_baseline_support_deberta_disambig` | +2.52pp [-0.67, +6.16] | +7.40pp [-1.84, +17.87] | +0.17pp [-0.56, +1.03] | 0.2891 | 1 |
| `arm3c_selective_agreement_gate` | +4.15pp [+0.35, +8.19] | +7.40pp [-1.84, +17.87] | +5.42pp [+0.12, +11.29] | 0 | 0 |

### MIXED_HALLUCINATION Separate Diagnostic (retrieval)

| Arm | n_mixed | Flagged Non-Supported Rate | Falsely SUPPORTED_PROXY Rate |
|---|---:|---:|---:|
| `arm0a_nli_only` | 9 | 100.0% (9) | 0.0% |
| `arm0b_minicheck_only` | 9 | 100.0% (9) | 0.0% |
| `arm1_baseline` | 9 | 100.0% (9) | 0.0% |
| `arm2_custom_deberta_only` | 9 | 100.0% (9) | 0.0% |
| `arm3a_mc_plus_custom_deberta` | 9 | 100.0% (9) | 0.0% |
| `arm3b_baseline_support_deberta_disambig` | 9 | 100.0% (9) | 0.0% |
| `arm3c_selective_agreement_gate` | 9 | 100.0% (9) | 0.0% |

