# VeriLLM Phase 7: Model Selection & Sequence-Length Architecture Decision

**Project**: VeriLLM — A Hybrid, Model-Agnostic Framework for Fine-Grained LLM Hallucination Detection, Quantification and Root Cause Analysis  
**Date**: November 9, 2026  
**Document Status**: Official Architecture & Methodology Decision  

---

## 1. Executive Summary

This document formally records the architectural, algorithmic, and hardware decisions made for VeriLLM's own machine learning component in Phase 7. Based on empirical analysis of the RAGTruth benchmark (123,678 sentences across 2,965 sources) and CPU profiling on the host development machine, we establish:
1. **Model Selection**: Fine-tuning a lightweight **DeBERTa-v3 cross-encoder** (`cross-encoder/nli-deberta-v3-xsmall` / `microsoft/deberta-v3-small`, 44M–70M parameters) rather than large encoder models (e.g. RoBERTa-Large with 355M parameters).
2. **Classification Paradigm**: Fine-grained **3-class classification** (`SUPPORTED_PROXY`, `CONTRADICTED`, `UNSUPPORTED`) using class-weighted cross-entropy loss to address the 88.8% majority class imbalance.
3. **Sequence-Length Strategy**: **Targeted context windowing (350 words)** combined with **100% preservation of the candidate proposition**, strictly preventing the catastrophic assertion truncation observed in naive 512-token truncation.

---

## 2. Model Selection: DeBERTa-v3 vs. RoBERTa-Large vs. Generative LLMs

### 2.1 Empirical Hardware & Latency Benchmarks (Local CPU Profile)

We profiled the two primary pretrained candidate models on the host machine (Intel UHD 620, CPU-only PyTorch):

| Metric | `cross-encoder/nli-deberta-v3-xsmall` | `lytang/MiniCheck-RoBERTa-Large` | Factor Difference |
|:---|:---:|:---:|:---:|
| **Architecture** | DeBERTa-v3 (Disentangled Attention) | RoBERTa-Large (Standard Bi-Encoder) | — |
| **Total Parameters** | **70,831,107 (70.83 M)** | **355,361,794 (355.36 M)** | **5.02× smaller** |
| **Memory Footprint (Weights)** | **270.20 MB** | **1,355.60 MB** | **5.02× lighter** |
| **Vocabulary Size** | 128,000 (SPM) | 50,265 (BPE) | 2.55× richer |
| **Batched CPU Latency** | **964.2 ms / claim** | **~9,500 – 11,000 ms / claim** | **~10× – 11× faster** |
| **RAM Spikes During Forward Pass** | < 0.6 GB | > 2.8 GB | Low risk vs High OOM risk |

### 2.2 Theoretical & Algorithmic Advantages of DeBERTa-v3

1. **Disentangled Attention Mechanism**:
   Standard transformers represent each token by summing its content embedding and position embedding into a single vector. In contrast, DeBERTa represents each word with two distinct vectors: content and relative position. Attention matrices are computed via disentangled matrices ($Q_c K_c^T + Q_c K_p^T + Q_p K_c^T$). For grounded verification, where the structural relationship between a subject in the premise document and a predicate in the LLM response is critical, disentangled attention provides superior alignment fidelity at a fraction of the parameter scale.
2. **Replaced Token Detection (RTD) Pretraining**:
   DeBERTa-v3 replaced the standard Masked Language Modeling (MLM) objective with RTD, training an auxiliary discriminator to detect synthesized tokens. This yields dense gradient feedback across all sequence positions during pretraining, resulting in higher sample efficiency and better zero-shot NLI performance.
3. **Local Deployment Feasibility**:
   A fact-checking pipeline that takes 10 seconds per sentence is unusable in production workflows (a 10-sentence LLM response would require nearly 2 minutes to verify). DeBERTa-v3 processes sentences in < 1 second on standard laptop hardware without specialized accelerators.

---

## 3. Sequence-Length Strategy & Context Windowing

### 3.1 Empirical Sequence Distributions in RAGTruth

From our inspection of all 123,678 sentences across the 2,965 RAGTruth sources:
- **Candidate Sentences**:
  - Median: **18 words**
  - 95th Percentile: **33 words**
  - Maximum: 267 words
- **Reference Context Documents**:
  - Median: **334 words**
  - 95th Percentile: **943 words**
  - Maximum: 2,891 words
- **Combined Pairs ($Context + Sentence$)**:
  - **20.15% of all sentence-document pairs exceed 512 words** (and 7.2% exceed 800 words).

### 3.2 The Failure Mode of Naive Head-Truncation

When formatting cross-encoder inputs as:
$$\text{Input} = \text{[CLS]} \circ \text{Context} \circ \text{[SEP]} \circ \text{Candidate Sentence} \circ \text{[SEP]}$$
Standard Hugging Face tokenizers with `truncation=True` and `max_length=512` truncate tokens from the **tail** of the sequence by default. Because the candidate sentence is placed at the end of the sequence, naive truncation **chops off the candidate assertion itself**, causing the model to verify an incomplete sentence fragment or an empty string against the document.

### 3.3 VeriLLM Context-Windowing Policy

To guarantee 100% preservation of the candidate proposition while strictly respecting transformer computational limits:
1. **Assertion-First Guarantee**: The candidate sentence is never truncated unless its own token count exceeds 128 tokens (which occurred in 0 instances in our validation set).
2. **Context Windowing to 350 Words**:
   - The premise context is windowed to the first 350 whitespace-delimited words for single-pass tasks.
   - 350 context words $\approx$ 420 subword tokens + 35 sentence tokens + 3 special tokens = 458 tokens $\le$ 512 token ceiling.
3. **Phase 4 Dense Passage Integration**:
   - For long multi-page documents (e.g. PDFs), VeriLLM does not feed raw full-text documents. Instead, the Phase 4 dense retriever (`all-MiniLM-L6-v2`) retrieves the top-$k$ relevant sliding chunks (chunk size: 250 tokens, overlap: 50 tokens) based on cosine similarity to the extracted claim. This guarantees that only high-salience context passages enter the cross-encoder context window.

---

## 4. Class Imbalance & Loss Weighting Strategy

### 4.1 RAGTruth Class Distribution

| Class Label | Sentence Count | Frequency | Cross-Entropy Loss Weight ($w_c$) |
|:---|:---:|:---:|:---:|
| `SUPPORTED_PROXY` | 109,814 | **88.80%** | **0.38** |
| `UNSUPPORTED` | 8,778 | **7.10%** | **4.69** |
| `CONTRADICTED` | 4,526 | **3.66%** | **9.11** |
| `MIXED_HALLUCINATION` | 560 | **0.45%** | *Excluded from 3-class base model* |

### 4.2 Inverse Frequency Loss Function

Without loss adjustment, standard cross-entropy loss suffers severe majority-class collapse: a model predicting `SUPPORTED_PROXY` for 100% of inputs achieves 88.8% accuracy while failing completely at hallucination detection (0% recall on contradictions and baseless info).

To prevent collapse, we compute balanced inverse class weights:
$$w_c = \frac{N_{\text{total}}}{K \cdot N_c}$$
Normalized such that $\sum_{c=1}^K w_c = K$.
In the training pipeline (`train.py` and `colab_ragtruth_training.ipynb`), the loss function is parameterized as:
$$\mathcal{L} = -\sum_{c=1}^3 w_c \cdot y_c \log(\hat{y}_c)$$
where contradictions receive a **9.11× loss gradient boost** and baseless claims receive a **4.69× boost** relative to supported sentences.

---

## 5. Summary of Engineering Artifacts

1. `dataset_preprocessor.py`: Implemented source-level split isolation, label mapping, quality filtering, and context windowing.
2. `train.py`: PyTorch training script with gradient scaling, linear warmup, class-weighted loss, and Macro-F1 checkpointing.
3. `colab_ragtruth_training.ipynb`: Self-contained GPU notebook for Google Colab (T4/A100) execution.
4. `baseline_evaluator.py`: Zero-shot baseline benchmark on unseen validation set comparing DeBERTa-v3 and MiniCheck-RoBERTa-Large.

