# VeriLLM Phase 7: Pre-Fine-Tuning Baseline Evaluation Report

## 1. Pretrained Model Architecture & Hardware Benchmark

| Model | Architecture | Parameters | Memory (Weights) | CPU Latency / Claim |
|:---|:---:|:---:|:---:|:---:|
| `cross-encoder/nli-deberta-v3-xsmall` | DeBERTa-v3 | **70,831,107** (70.8M) | **270.2 MB** | **964.2 ms** |
| `lytang/MiniCheck-RoBERTa-Large` | RoBERTa-Large | **355,361,794** (355M) | **1355.6 MB** | **9210.1 ms** (9.6× slower) |

## 2. Zero-Shot 3-Class NLI Performance (Validation Set)

- **Evaluated Sample**: `60` balanced sentences (20 per class) from `ragtruth_val.jsonl` (375 unseen validation sources).
- **Overall Macro-F1**: **`35.30%`**

| Class Label | Precision | Recall | F1-Score |
|:---|:---:|:---:|:---:|
| **`SUPPORTED_PROXY`** | 100.0% | 5.0% | **9.5%** |
| **`CONTRADICTED`** | 75.0% | 30.0% | **42.9%** |
| **`UNSUPPORTED`** | 37.3% | 95.0% | **53.5%** |

### 3-Class Confusion Matrix
*(Rows = True Label, Columns = Predicted: `SUPPORTED_PROXY`, `CONTRADICTED`, `UNSUPPORTED`)*

```text
[[1, 1, 18], [0, 6, 14], [0, 1, 19]]
```

## 3. Binary Hallucination Detection Comparison

| Baseline Model | Macro-F1 | Supported F1 | Hallucination F1 | Latency |
|:---|:---:|:---:|:---:|:---:|
| **`nli-deberta-v3-xsmall` (Zero-Shot)** | **45.17%** | 9.52% | 80.81% | **964.2 ms** |
| **`MiniCheck-RoBERTa-Large`** | **72.82%** | 65.12% | 80.52% | 9210.1 ms |

## 4. Error Analysis & Baseline Failure Modes

### Example `2483_s05` (Summary)
- **Sentence**: *"Betsy Sharkey of the Los Angeles Times enjoyed the film's action sequences, particularly the parkour-influenced stunts, but noted that they were not grounded in reality."*
- **Gold Label**: `SUPPORTED_PROXY` | **NLI Predicted**: `UNSUPPORTED` (MiniCheck prob: `0.0092`)
- **NLI Output Probabilities**: Entailment: `0.0286`, Contradiction: `0.0696`, Neutral: `0.9019`
- **Flags**: `implicit_true=False`, `due_to_null=False`

### Example `9452_s03` (Data2txt)
- **Sentence**: *"They are open Monday through Sunday from 6:30 am to 8:00 pm."*
- **Gold Label**: `CONTRADICTED` | **NLI Predicted**: `UNSUPPORTED` (MiniCheck prob: `0.1786`)
- **NLI Output Probabilities**: Entailment: `0.0037`, Contradiction: `0.0262`, Neutral: `0.9701`
- **Flags**: `implicit_true=False`, `due_to_null=False`
- **Annotations**: *"They are open Monday through Sunday from 6:30 am to 8:00 pm"* (Evident Conflict)

### Example `913_s08` (Summary)
- **Sentence**: *"Gosling believes that his film should convey the paradox of Detroit, showing that dereliction is happening but it doesn't define the city. ""*
- **Gold Label**: `SUPPORTED_PROXY` | **NLI Predicted**: `UNSUPPORTED` (MiniCheck prob: `0.0314`)
- **NLI Output Probabilities**: Entailment: `0.1878`, Contradiction: `0.1337`, Neutral: `0.6785`
- **Flags**: `implicit_true=False`, `due_to_null=False`

### Example `5893_s09` (Data2txt)
- **Sentence**: *"They also praised the well-rounded menu, with options ranging from burgers to nachos."*
- **Gold Label**: `SUPPORTED_PROXY` | **NLI Predicted**: `UNSUPPORTED` (MiniCheck prob: `0.9868`)
- **NLI Output Probabilities**: Entailment: `0.0304`, Contradiction: `0.2357`, Neutral: `0.734`
- **Flags**: `implicit_true=False`, `due_to_null=False`

### Example `12639_s14` (QA)
- **Sentence**: *"There are no income limits for contributing to a Roth IRA. ("*
- **Gold Label**: `UNSUPPORTED` | **NLI Predicted**: `CONTRADICTED` (MiniCheck prob: `0.0836`)
- **NLI Output Probabilities**: Entailment: `0.0042`, Contradiction: `0.9319`, Neutral: `0.0639`
- **Flags**: `implicit_true=False`, `due_to_null=False`
- **Annotations**: *"8. There are no income limits for contributing to a Roth IRA."* (Evident Baseless Info)

