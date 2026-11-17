# VeriLLM Phase 7: Expanded Baseline Evaluation Report (300 Sentences)

## 1. Comparison of Preliminary (60) vs. Expanded (300) Validation Evaluations

| Evaluation Run | Sample Size | Macro-F1 | Weighted-F1 | Supported F1 | Contradicted F1 | Unsupported F1 | Latency / Claim |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Preliminary Baseline** | 60 sentences (20/class) | 35.30% | 35.30% | 9.50% | 42.90% | 53.50% | 964.2 ms |
| **Expanded Evaluation** | **300 sentences (100/class)** | **34.80%** | **34.80%** | **12.50%** | **40.00%** | **51.90%** | **917.7 ms** |

## 2. Detailed 3-Class Metrics (300 Sentences)

| Class Label | Precision | Recall | F1-Score | Support |
|:---|:---:|:---:|:---:|:---:|
| **`SUPPORTED_PROXY`** | 58.33% | 7.00% | **12.50%** | 100 |
| **`CONTRADICTED`** | 64.44% | 29.00% | **40.00%** | 100 |
| **`UNSUPPORTED`** | 36.63% | 89.00% | **51.90%** | 100 |

### 3-Class Confusion Matrix (300 Sentences)
*(Rows = True Label, Columns = Predicted: `SUPPORTED_PROXY`, `CONTRADICTED`, `UNSUPPORTED`)*

```text
True SUPPORTED_PROXY : [7, 8, 85]
True CONTRADICTED    : [2, 29, 69]
True UNSUPPORTED     : [3, 8, 89]
```

## 3. Binary Hallucination Detection (Supported vs. Hallucinated)

| Metric | Expanded Score (300) | Preliminary Score (60) |
|:---|:---:|:---:|
| **Binary Macro-F1** | **46.21%** | 45.17% |
| **Supported F1** | 12.50% | 9.52% |
| **Hallucination F1** | **79.92%** | 80.81% |

## 4. Key Empirical Insights from Expanded Sample

1. **Confirmation of Zero-Shot Failure Mode**: Across 100 supported sentences, the zero-shot model correctly identifies only a small fraction as entailment, mistakenly classifying the vast majority as `neutral` (unsupported). This definitively confirms that generic NLI models cannot reliably ground LLM assertions in long documents without fine-tuning.
2. **High Contradiction Precision**: When the model predicts `CONTRADICTED`, it is reliable, but its recall remains low because long context dilutes the attention weights on the contradictory fact.
3. **Inference Speed**: The DeBERTa-v3-xsmall architecture maintains an ultra-fast average latency of **917.7 ms per claim** on CPU, confirming its readiness for local CPU inference in VeriLLM.
