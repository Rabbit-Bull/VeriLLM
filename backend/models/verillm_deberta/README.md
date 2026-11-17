---
language:
  - en
pipeline_tag: text-classification
library_name: transformers
tags:
  - deberta-v3
  - hallucination-detection
  - natural-language-inference
  - ragtruth
  - document-grounded-verification
---

# VeriLLM DeBERTa

**VeriLLM DeBERTa** is a 70.83-million parameter cross-encoder model fine-tuned for sentence-level, document-grounded hallucination verification. It was trained on the [RAGTruth](https://arxiv.org/abs/2401.00396) benchmark and serves as one classification component within the broader [VeriLLM](https://github.com/Rabbit-Bull/verillm) hybrid verification framework.

- **Model Repository:** [Rabbit-Bull/verillm-deberta](https://huggingface.co/Rabbit-Bull/verillm-deberta)
- **Pinned Revision:** `71259e48c794e89f92075ddf848813c93327461c`
- **Base Model:** `cross-encoder/nli-deberta-v3-xsmall` (`DebertaV2ForSequenceClassification`)
- **Parameters:** 70,831,107 (70.83M)
- **Required Model Files:** `config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`
- **Primary Task:** Document-Grounded Factuality & Hallucination Classification (Premise–Hypothesis Cross-Encoder)
- **Framework Compatibility:** Hugging Face `transformers`, PyTorch

---

## Model Description

VeriLLM DeBERTa evaluates the factual consistency of a candidate sentence (**hypothesis**) with respect to an authoritative reference context (**premise**). The model uses a cross-encoder architecture that jointly encodes the reference passage and the candidate statement through bidirectional cross-attention, producing unnormalized logits across three natural language inference categories.

The checkpoint is designed for CPU-compatible inference without requiring GPU acceleration.

### Training Dataset & Labels

The model was fine-tuned on the sentence-level annotations of the **RAGTruth** benchmark across three discrete training classes:

| Class Index | Label | Definition |
|:---:|:---|:---|
| `0` | `CONTRADICTED` | The candidate sentence conflicts with its reference evidence. |
| `1` | `SUPPORTED_PROXY` | No annotated hallucination span was found in the aligned sentence within the reference context. *(See Limitations regarding proxy interpretation).* |
| `2` | `UNSUPPORTED` | The candidate sentence lacks sufficient supporting evidence in the reference context. |

*Note: In the RAGTruth corpus, sentences without annotated hallucination spans serve as empirical proxies for support; this does not guarantee universal factual correctness. For the reported three-class test evaluation, 37 sentences with `MIXED` annotations in the test split were excluded from the three-class evaluation protocol.*

---

## Important Distinction: Training Labels vs. VeriLLM Production Categories

It is critical to distinguish between the **three raw classification labels** produced by this standalone model and the **five verification categories** exposed in the full VeriLLM application pipeline:

1. **Standalone Model Output (3 Classes):**
   - When loaded directly via Hugging Face Transformers, this checkpoint outputs raw logits across `[CONTRADICTED, SUPPORTED_PROXY, UNSUPPORTED]`.

2. **VeriLLM Full Framework Output (5 Categories):**
   - In production, VeriLLM embeds this model inside an end-to-end hybrid architecture that incorporates proposition decomposition, dense passage retrieval, probability calibration, and epistemic abstention:
     - **`SUPPORTED`**: Calibrated support probability exceeds the decision threshold ($\tau_{\text{sup}}$).
     - **`CONTRADICTED`**: Calibrated contradiction probability meets the contradiction threshold ($\tau_{\text{contra}}$).
     - **`UNSUPPORTED`**: Evidence is present, but fails to entail the proposition.
     - **`UNVERIFIABLE`**: Produced **upstream** when evidence retrieval fails or passage similarity falls below the relevance floor ($\tau_{\text{floor}} = 0.20$), triggering safe epistemic abstention prior to cross-encoder inference.
     - **`NON_FACTUAL`**: Produced **upstream** by linguistic filtering for subjective opinions, greetings, speculations, or conversational pleasantries that contain no empirical assertions.

This standalone checkpoint does **not** emit `UNVERIFIABLE` or `NON_FACTUAL` labels directly; those are system-level decisions produced by the VeriLLM pipeline orchestration.

---

## Evaluation Results

The model was evaluated on the RAGTruth test split containing **17,906** three-class sentences; 37 MIXED sentences from the test split were excluded from this three-class evaluation.

### Benchmark Metrics

| Metric | Score |
|:---|:---:|
| **Test Accuracy** | **92.86%** |
| **Weighted F1** | **92.76%** |
| **Macro-F1** | **69.99%** |

### Per-Class Performance Breakdown

| Class | F1-Score | Evaluation Note |
|:---|:---:|:---|
| **`SUPPORTED_PROXY`** | **96.20%** | F1-score on the majority class (~90% base rate) |
| **`CONTRADICTED`** | **61.67%** | Moderate classification performance on contradiction examples (F1: 61.67%) |
| **`UNSUPPORTED`** | **52.09%** | Classification performance on ungrounded/extrinsic assertions (F1: 52.09%) |

The difference between Macro-F1 (69.99%) and Weighted F1 (92.76%) reflects the class imbalance of the dataset, where supported sentences constitute the majority of generation outputs. Class imbalance can significantly affect minority-class performance.

---

## Limitations and Disclaimers

Users should observe the following limitations:

1. **Not a Universal Fact-Checker:** This model does not store parametric world knowledge and cannot determine absolute truth in the absence of reference context. It is designed for **reference-document-grounded verification**.
2. **Reference-Document Dependent:** The model's predictions are entirely conditioned on the provided premise. If the reference passage is inaccurate, incomplete, or biased, the model's verification will reflect those flaws.
3. **`SUPPORTED_PROXY` Is Not a Guarantee of Truth:** The label `SUPPORTED_PROXY` indicates only that no annotated hallucination span was found in the aligned sentence relative to the reference evidence. It does not establish absolute real-world truth or correctness outside the reference document.
4. **`UNSUPPORTED` Does Not Imply Falsehood:** An `UNSUPPORTED` prediction indicates that the provided reference passage lacks sufficient supporting evidence to verify the hypothesis. The statement itself may be objectively true in the real world.
5. **Class Imbalance and Minority-Class Performance:** The underlying dataset is heavily imbalanced toward supported claims (~90% base rate). Class imbalance can affect minority-class classification performance, resulting in lower F1 scores for `CONTRADICTED` (61.67%) and `UNSUPPORTED` (52.09%) relative to `SUPPORTED_PROXY` (96.20%). Applications sensitive to minority-class errors may benefit from probability calibration or threshold tuning.
6. **Component of a Hybrid System:** This model is one component of VeriLLM's hybrid pipeline and should not be interpreted as the entire production verification system. Raw softmax scores from this checkpoint do not incorporate passage relevance gating, proposition decomposition, or calibrated risk thresholds.
7. **Out-of-Distribution Generalization:** Results on RAGTruth evaluate performance on that benchmark's specific tasks (summarization, question answering, and data-to-text across selected LLMs). They do not establish universal out-of-distribution generalization across arbitrary unseen domains, non-English text, or arbitrary LLM architectures without task-specific evaluation.

---

## How to Use

The model can be loaded directly using the Hugging Face `transformers` library. The premise (reference evidence) and hypothesis (candidate sentence) are passed as a paired input sequence.

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model and tokenizer from Hugging Face Hub (pinned revision)
model_name = "Rabbit-Bull/verillm-deberta"
revision = "71259e48c794e89f92075ddf848813c93327461c"
tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
model = AutoModelForSequenceClassification.from_pretrained(model_name, revision=revision)

# Define reference evidence (premise) and candidate statement (hypothesis)
premise = (
    "The James Webb Space Telescope was launched on December 25, 2021, "
    "aboard an Ariane 5 rocket from Kourou, French Guiana."
)
hypothesis = "The James Webb Space Telescope was launched in December 2021."

# Tokenize as a paired sequence (Premise [SEP] Hypothesis)
inputs = tokenizer(
    premise,
    hypothesis,
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt"
)

# Run model inference
model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=-1).item()

# Map to class label
predicted_label = model.config.id2label[predicted_class_id]
print(f"Predicted Class: {predicted_label} (Index: {predicted_class_id})")
```

### Label Mapping

The model configuration provides the following standard `id2label` mapping:

```python
{
    0: "CONTRADICTED",
    1: "SUPPORTED_PROXY",
    2: "UNSUPPORTED"
}
```

---

## Citation & References

If you use this model in your research or application, please cite the underlying RAGTruth benchmark and the VeriLLM project:

```bibtex
@article{yuan2024ragtruth,
  title={RAGTruth: A Hallucination Detection Benchmark for Retrieval-Augmented Generation},
  author={Yuan, Chengshuai and Gan, Jiangnan and Song, Zhaoxuan and others},
  journal={arXiv preprint arXiv:2401.00396},
  year={2024}
}
```

For full details regarding the hybrid verification framework, probability calibration, and application pipeline, see the [VeriLLM Documentation and Source Code](https://github.com/Rabbit-Bull/verillm).
