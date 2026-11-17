# VeriLLM Phase 5: Claim Verification Pipeline Demonstration Report

- **Reference Document**: `sample_apollo11_mission.pdf` (3 pages, 5 chunks)
- **Fact-Checking Baselines**: `MiniCheck-RoBERTa-Large` & `cross-encoder/nli-deberta-v3-xsmall` (CPU Runtime)
- **Total Latency**: 10.28s (1141.8ms per claim)

## 1. Verified Claims Breakdown

| Claim ID | Proposition | Classification | MiniCheck Prob | NLI (Contra/Entail/Neut) | Citation | Reasoning |
|:---|:---|:---:|:---:|:---:|:---|:---|
| `claim_001` | The Apollo 11 mission was launched on a Saturn V rocket from Kennedy Space Center on July 16, 1969. | **`SUPPORTED`** | 0.9868 | 0.00 / 0.98 / 0.02 | sample_apollo11_mission.pdf P1 | Evidence in 'sample_apollo11_mission.pdf' (Page 1) entails the claim. [MiniCheck support p=0.9868, NLI entailment=0.9757, contradiction=0.0005] |
| `claim_002` | Neil Armstrong and Buzz Aldrin landed the lunar module Eagle on July 20, 1969,. | **`SUPPORTED`** | 0.5015 | 0.09 / 0.88 / 0.03 | sample_apollo11_mission.pdf P2 | Evidence in 'sample_apollo11_mission.pdf' (Page 2) entails the claim. [MiniCheck support p=0.5015, NLI entailment=0.8798, contradiction=0.0854] |
| `claim_003` | they collected 47.5 pounds of lunar rock. | **`SUPPORTED`** | 0.9896 | 0.36 / 0.07 / 0.57 | sample_apollo11_mission.pdf P2 | Evidence in 'sample_apollo11_mission.pdf' (Page 2) entails the claim. [MiniCheck support p=0.9896, NLI entailment=0.0692, contradiction=0.3566] |
| `claim_004` | Michael Collins walked on the Moon with Armstrong while Buzz Aldrin flew the Command Module. | **`CONTRADICTED`** | 0.0326 | 1.00 / 0.00 / 0.00 | sample_apollo11_mission.pdf P2 | Evidence in 'sample_apollo11_mission.pdf' (Page 2) directly refutes the claim. [NLI contradiction=0.9958, MiniCheck support p=0.0326, NLI neutral=0.0038] |
| `claim_005` | The astronauts safely returned to Earth. | **`SUPPORTED`** | 0.9868 | 0.01 / 0.96 / 0.03 | sample_apollo11_mission.pdf P3 | Evidence in 'sample_apollo11_mission.pdf' (Page 3) entails the claim. [MiniCheck support p=0.9868, NLI entailment=0.9584, contradiction=0.0069] |
| `claim_006` | The astronauts splashed down in the Pacific Ocean on July 24, 1969. | **`SUPPORTED`** | 0.9887 | 0.00 / 0.99 / 0.01 | sample_apollo11_mission.pdf P3 | Evidence in 'sample_apollo11_mission.pdf' (Page 3) entails the claim. [MiniCheck support p=0.9887, NLI entailment=0.9861, contradiction=0.0019] |
| `claim_007` | The mission was watched by over 650 million television viewers worldwide. | **`UNSUPPORTED`** | 0.0058 | 0.01 / 0.00 / 0.99 | sample_apollo11_mission.pdf P2 | Retrieved passage in 'sample_apollo11_mission.pdf' (Page 2) discusses related context but does not contain factual assertions confirming or refuting the claim. [NLI neutral=0.9948, MiniCheck support p=0.0058, NLI contradiction=0.0050] |
| `claim_008` | Buzz Aldrin also planted an olive branch on Mars. | **`UNSUPPORTED`** | 0.0044 | 0.02 / 0.00 / 0.98 | sample_apollo11_mission.pdf P2 | Retrieved passage in 'sample_apollo11_mission.pdf' (Page 2) discusses related context but does not contain factual assertions confirming or refuting the claim. [NLI neutral=0.9845, MiniCheck support p=0.0044, NLI contradiction=0.0153] |
| `claim_009` | In my opinion, Apollo 11 is humanity's greatest triumph. | **`NON_FACTUAL`** | - | - | sample_apollo11_mission.pdf P1 | Statement is subjective, opinion-based, or conversational meta-text and does not assert an empirical factual proposition. |

## 2. Factuality & Hallucination Metrics

- **Total Extracted Claims**: `9`
- **Empirical Factual Claims (Denominator $N$)**: `8`
- **Non-Factual Claims Excluded**: `1`
- **Evidence Support Rate**: **62.50%** (5/8)
- **Confirmed Contradiction Rate**: **12.50%** (1/8)
- **Unsupported Rate**: **25.00%** (2/8)
- **Unverifiable Rate**: **0.00%** (0/8)
- **Combined Uncertainty Rate**: **25.00%**

> *Metrics evaluated over 8 factual claims (1 non-factual statements excluded from factuality denominator).*
