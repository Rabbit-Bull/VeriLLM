# VeriLLM Phase 6: Scoring & Quantification Engine Report

## 1. Apollo 11 Demonstration Response Factuality Score

- **Response ID**: `apollo11_candidate_response`
- **Reference Document**: `sample_apollo11_mission.pdf`
- **Total Propositions**: `9` (8 Empirical Facts, 1 Subjective Opinion Excluded)
- **Evidence Support Rate (ESR / GFI)**: **62.50%** (5 / 8)
- **Confirmed Contradiction Rate (CCR / HSI)**: **12.50%** (1 / 8)
- **Unsupported Rate (USR)**: **25.00%** (2 / 8)
- **Unverifiable Rate (UVR)**: **0.00%** (0 / 8)
- **Combined Uncertainty Rate (UR)**: **25.00%** (2 / 8)
- **Contradiction-Penalized Factuality (CPFS)**: **50.00%** ($\max(0, (5 - 1)/8) \times 100\%$)
- **Mathematical Invariants Verified**: **`True`** ($62.50\% + 12.50\% + 25.00\% + 0.00\% = 100.00\%$)

## 2. Multi-Model Benchmark Aggregation

| Model Identifier | Evaluated Responses | Factual Facts (N) | Non-Factual | Support Rate (Micro) | Contradiction Rate (Micro) | Uncertainty Rate (Micro) | Penalized Score (CPFS) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`GPT-4o-GroundTruth`** | 2 | 10 | 0 | **80.00%** | 0.00% | 20.00% | **80.00%** |
| **`Legacy-7B-Baseline`** | 2 | 5 | 2 | **20.00%** | 60.00% | 20.00% | **0.00%** |

## 3. Methodological Notes on Aggregation

- Dataset evaluated over 4 responses containing 17 total propositions.
- Micro-pooled metrics aggregate all 15 empirical factual claims across responses before calculating rates, preventing Simpson's paradox.
- Macro-averaged metrics report the unweighted average across the 3 responses that contained factual claims.
- 1 responses had zero testable factual claims and were excluded from factuality rate denominators.
