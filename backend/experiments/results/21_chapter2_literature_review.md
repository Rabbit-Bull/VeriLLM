# Chapter 2: Literature Review and Theoretical Foundations

---

## 2.1 Introduction and Scope of the Literature Review

The rapid proliferation of Large Language Models (LLMs) built upon overparameterized autoregressive Transformer architectures (Vaswani et al., 2017; Brown et al., 2020; Touvron et al., 2023) has revolutionized artificial intelligence applications across summarization, open-domain question answering, code generation, and conversational reasoning. Despite their unprecedented syntactic fluency and broad semantic competence, these models suffer from a fundamental, systemic vulnerability: **factual hallucination** (Ji et al., 2023; Maynez et al., 2020). Generative LLMs frequently synthesize assertions that appear grammatically flawless, authoritative, and superficially convincing, but are factually erroneous, uncorroborated by grounding evidence, or in direct contradiction with verified source materials.

In high-stakes enterprise, legal, scientific, and healthcare environments, unchecked hallucinations impede the safe deployment of generative AI. To mitigate this hazard, researchers have pursued two complementary paradigms:
1. **Generative Grounding**: Augmenting the generation process with external non-parametric knowledge via Retrieval-Augmented Generation (RAG) (Lewis et al., 2020; Guu et al., 2020).
2. **Automated Verification**: Developing post-hoc factual consistency classifiers and hallucination detection algorithms to audit model outputs against authoritative reference texts (Honovich et al., 2022; Min et al., 2023; Tang et al., 2024).

However, current automated verification methodologies suffer from critical theoretical and operational deficiencies. Existing approaches often rely on coarse document-level scoring, collapse the nuanced distinction between lack of evidence and active contradiction into a simplistic binary classification, employ computationally expensive and non-deterministic LLM judges ("LLM-as-a-Judge"), or depend on modern deep neural networks that exhibit severe probability miscalibration (Guo et al., 2017) and an inability to perform principled epistemic abstention (Geifman & El-Yaniv, 2017).

This chapter provides a rigorous, critical synthesis of the theoretical foundations and academic literature underpinning automated hallucination detection. It reviews:
- The statistical origins of hallucination in autoregressive language modeling (§2.2);
- The principles and failure modes of Retrieval-Augmented Generation (§2.3);
- Existing verification paradigms, spanning Question Answering (QA), Natural Language Inference (NLI), and generative judges (§2.4);
- Transformer encoder architectures, cross-encoders, and the theoretical superiority of disentangled attention (§2.5);
- Probability calibration, class-prior bias shifts, and proper scoring rules (§2.6);
- Selective prediction, risk-coverage trade-offs, and epistemic uncertainty (§2.7);
- Benchmark datasets, focusing on the RAGTruth benchmark (§2.8);
- The concrete research gap that motivates the architecture of VeriLLM (§2.9).

---

## 2.2 Large Language Models and the Hallucination Problem

### 2.2.1 Autoregressive Language Generation and Objective Mismatch
Modern generative LLMs are trained on massive text corpora using the standard autoregressive language modeling objective. Given a sequence of discrete tokens $\mathbf{x} = (x_1, x_2, \dots, x_T)$, the model parameterizes the joint probability distribution $P(\mathbf{x})$ as a product of conditional next-token distributions:
$$P(\mathbf{x}) = \prod_{t=1}^T P_\theta(x_t \mid x_{<t}) \tag{2.1}$$
where $\theta$ denotes the model parameters, and $x_{<t} = (x_1, \dots, x_{t-1})$ represents the preceding context. Parameters are optimized via maximum likelihood estimation, minimizing empirical cross-entropy:
$$\mathcal{L}_{\text{LM}}(\theta) = -\sum_{t=1}^T \log P_\theta(x_t \mid x_{<t}) \tag{2.2}$$

While mathematically tractable, Equation (2.2) optimizes strictly for **statistical typicality** and **surface distributional fluency**, not empirical truth (Bender et al., 2021). The objective rewards the model for selecting tokens that maximize conditional likelihood given training patterns, regardless of whether the generated proposition corresponds to objective physical reality or logically follows from supplied context. Consequently, when an LLM encounters gaps in its training distribution or prompt ambiguity, it naturally defaults to high-probability linguistic structures, producing fluent fabrications.

```
[Figure 2.1 Placeholder: Autoregressive Next-Token Generation vs. Factual Grounding]
Caption: Figure 2.1 illustrates the fundamental divergence between autoregressive token sampling and factual grounding. While the Transformer decoder iteratively samples high-likelihood linguistic tokens based on internal parameter distributions (left), factual verification requires projecting generated assertions into an external semantic space conditioned strictly on explicit reference evidence (right).
```

### 2.2.2 Factuality versus Fluency
A central finding across contemporary NLP literature is the decoupling of linguistic fluency from factual correctness (Ji et al., 2023; Zhang et al., 2023). Because Transformer self-attention layers excel at modeling long-range syntactic dependencies, generated text typically exhibits high coherence, sophisticated vocabulary, and natural discourse transitions. This high fluency actively masks factual errors: human evaluators and naive heuristic metrics (such as BLEU or ROUGE) routinely assign high quality scores to fluent hallucinations, creating a deceptive failure mode where erroneous outputs appear most convincing (Novikova et al., 2017).

### 2.2.3 Taxonomy and Etiology of Hallucinations
Following the comprehensive taxonomy established by Ji et al. (2023) and Huang et al. (2023), hallucinations in natural language generation are partitioned into two broad classes:
1. **Intrinsic Hallucinations**: Generated content that directly conflicts with, contradicts, or negates information explicitly provided in the source context.
2. **Extrinsic Hallucinations**: Generated content that introduces information, entities, or relations completely absent from, and uncorroborated by, the source text. (Extrinsic assertions are not necessarily false in the physical world; they are simply ungrounded relative to the reference document).

The underlying causes of hallucinations originate across the entire model lifecycle:
- **Data-Level Causes**: Web-scale training corpora contain factual inaccuracies, social biases, temporal out-of-date information, and fabricated narratives.
- **Architectural & Training Causes**: The standard cross-entropy loss lacks an explicit penalty for factual errors. Furthermore, maximum likelihood training encourages models to allocate probability mass to all plausible continuations, penalizing safe abstention ("I do not know") because refusal strings are scarce in pretraining data.
- **Inference-Level Causes**: Stochastic decoding strategies (e.g., nucleus sampling with $p < 1.0$ or temperature sampling $T > 0$) intentionally inject randomness into token selection to increase stylistic diversity, which increases the probability of straying into hallucinated states (Holtzman et al., 2020).

---

## 2.3 Retrieval-Augmented Generation and Document-Grounded Verification

### 2.3.1 Principles of Retrieval-Augmented Generation (RAG)
To reduce the hallucination rate of purely parametric models, Lewis et al. (2020) and Guu et al. (2020) introduced **Retrieval-Augmented Generation (RAG)**. RAG decouples an AI system into two distinct knowledge stores:
- **Parametric Memory**: The static, pre-trained weights of the generative neural network ($\theta$).
- **Non-Parametric Memory**: A dynamically searchable, external corpus of authoritative reference documents $\mathcal{D} = \{d_1, d_2, \dots, d_K\}$, typically indexed via dense vector embeddings.

In a standard RAG architecture, given a user input query $q$, a dense retriever (such as DPR or Sentence-BERT) retrieves top-$k$ relevant passages $\mathcal{Z} = \{z_1, \dots, z_k\} \subset \mathcal{D}$ using embedding inner products:
$$z^* = \arg\max_{z \in \mathcal{D}} \cos(\mathbf{e}_q, \mathbf{e}_z) = \arg\max_{z \in \mathcal{D}} \frac{\mathbf{e}_q \cdot \mathbf{e}_z}{\|\mathbf{e}_q\|_2 \|\mathbf{e}_z\|_2} \tag{2.3}$$

The retrieved passages $\mathcal{Z}$ are concatenated with the query into an augmented context window, and the generative LLM produces the final output conditioned on both:
$$P(y \mid q) = \sum_{z \in \mathcal{Z}} P(z \mid q) \prod_{t=1}^T P_\theta(y_t \mid q, z, y_{<t}) \tag{2.4}$$

### 2.3.2 The Grounding Gap in Retrieval-Augmented Workflows
While RAG dramatically improves factual grounding over closed-book generation, recent empirical studies demonstrate that **RAG systems still hallucinate extensively** (Yuan et al., 2024; Barnett et al., 2024). These failures stem from several structural limitations:
1. **Retrieval Misses and Semantic Drift**: If the dense retriever returns irrelevant or tangentially related passages, the generative model frequently hallucinates by attempting to reconcile the uncorroborated query with irrelevant context.
2. **Context Misattribution**: Even when accurate passages are supplied within the prompt, generative LLMs often ignore retrieved facts, preferring strong parametric priors learned during pretraining (Chen et al., 2022).
3. **Synthesis Errors**: When generating responses requiring the synthesis of multiple retrieved passages, LLMs frequently fabricate connective relations, misattributing properties of Entity A to Entity B.

### 2.3.3 Retrieval Similarity vs. Logical Entailment
A critical theoretical fallacy in early automated verification systems was the assumption that **retrieval similarity equals factual verification**. High cosine similarity between a candidate claim embedding $\mathbf{e}_c$ and an evidence passage embedding $\mathbf{e}_p$ indicates topical and lexical affinity, but provides zero guarantee of logical consistency.

For example, consider the premise:
$$\text{Premise: } \textit{"Neil Armstrong stepped onto the lunar surface on July 21, 1969."}$$
and two candidate assertions:
$$\text{Assertion 1: } \textit{"Neil Armstrong landed on the Moon in July 1969."} \quad (\text{Logically Entailed})$$
$$\text{Assertion 2: } \textit{"Neil Armstrong never stepped onto the Moon."} \quad (\text{Direct Contradiction})$$

Because both assertions share virtually identical entity vocabulary (*"Neil Armstrong"*, *"Moon"*, *"stepped"*), bi-encoder embeddings (e.g., Sentence-BERT) produce near-identical cosine similarity scores ($\text{sim} > 0.88$) for both pairs. Dense retrieval alone cannot detect logical negation, entity swapping, or temporal conflict. True factual verification requires deep cross-attentive **Natural Language Inference (NLI)** capable of distinguishing semantic similarity from logical entailment.

---

## 2.4 Existing Hallucination Detection and Factual Consistency Approaches

Automated factual verification approaches in the literature generally fall into four major methodological families, summarized in Table 2.1.

**Table 2.1: Comparative Analysis of Hallucination Detection Methodologies**
| Paradigm / Method | Representative Systems | Primary Mechanism | Key Advantages | Critical Limitations |
| :--- | :--- | :--- | :--- | :--- |
| **QA-Based Consistency** | QAGS (Wang et al., 2020)<br>FEQA (Durmus et al., 2020)<br>QuestEval (Scialom et al., 2021) | Generates Q&A pairs from candidate text and checks if source answers match. | Highly interpretable question-answer traces. | Severe error compounding (question generator errors + QA model errors); high latency. |
| **LLM-as-a-Judge** | G-Eval (Liu et al., 2023)<br>SelfCheckGPT (Manakul et al., 2023)<br>ChatEval (Chan et al., 2023) | Prompts proprietary LLMs (e.g. GPT-4) with rubric instructions. | Flexible; requires no dedicated model training. | Prohibitive financial cost; non-deterministic API drift; circular reasoning; privacy leakage. |
| **Atomic Claim Decomposition** | FActScore (Min et al., 2023)<br>SAFE (Wei et al., 2024) | Breaks long text into atomic facts and verifies each independently. | Fine-grained, proposition-level error localization. | Coarse binary verdicts; lacks probability calibration; high computational cost when powered by LLMs. |
| **NLI Cross-Encoders** | FactCC (Kryscinski et al., 2020)<br>TRUE (Honovich et al., 2022)<br>MiniCheck (Tang et al., 2024) | Premise-hypothesis cross-attention classification. | Fast inference; token-level interaction; deterministic. | Typically binary (collapses unsupported into contradicted); severe neural miscalibration. |
| **Proposed: VeriLLM** | **VeriLLM Framework (Ours)** | **Calibrated 3-Way DeBERTa Cross-Encoder + Dual Operational Regimes + Epistemic Abstention** | **Distinguishes unsupported from contradicted; calibrated probabilities ($C2$); local CPU execution; response scorecards.** | Single-hop passage retrieval; machine-readable text required (no OCR). |

### 2.4.1 Question Answering (QA)-Based Evaluation
Pioneered by Wang et al. (2020) (QAGS) and Durmus et al. (2020) (FEQA), QA-based verification evaluates factual consistency through an automated question-and-answer cycle:
1. A question-generation model produces questions based on entities in the candidate response.
2. A question-answering model answers those questions using the candidate response.
3. The same QA model attempts to answer those questions using the reference document.
4. Factual consistency is measured by the token-level overlap (e.g., F1 score) between the two answers.

While QA-based approaches offer human-interpretable question traces, they suffer from severe **cascading error compounding**: if the question generator produces ungrammatical or misleading questions, or if the QA model extracts incomplete spans, the consistency score collapses regardless of the candidate's actual factual accuracy. Furthermore, running multi-stage generation and answering loops incurs prohibitive computational latency.

### 2.4.2 Generative LLM Judges ("LLM-as-a-Judge")
With the advent of highly capable proprietary instruction-tuned models, prompting LLMs (such as GPT-4) to act as autonomous factuality evaluators has gained widespread adoption (Zheng et al., 2023; Liu et al., 2023). In SelfCheckGPT (Manakul et al., 2023), zero-resource verification is attempted by comparing a candidate response against multiple stochastically sampled alternative outputs from the same model.

Despite their convenience, LLM-as-a-Judge approaches suffer from fundamental scientific and practical vulnerabilities:
- **Circular Reasoning and Shared Bias**: Evaluating an LLM's output using another LLM introduces correlated epistemic blind spots. If both models share pretraining datasets, they share the same factual misconceptions.
- **Prohibitive Cost and Non-Deterministic Drift**: Calling commercial LLM APIs for every verification query introduces substantial recurring financial costs and latency (often 2–10 seconds per query). Moreover, commercial model weights are updated without notice, rendering longitudinal benchmarks irreproducible.
- **Severe Data Privacy Breaches**: In corporate, legal, and medical applications, transmitting confidential internal documents to third-party proprietary API endpoints violates regulatory compliance (e.g., GDPR, HIPAA).

### 2.4.3 NLI-Based Factual Consistency Classifiers
Natural Language Inference (NLI) frames factual verification as premise-hypothesis classification (Dagan et al., 2006). The reference evidence serves as the **premise** ($P$), and the candidate assertion serves as the **hypothesis** ($H$). The verifier models the conditional probability distribution across standard inferential categories:
$$P(\text{Verdict} \mid P, H) \in \{\text{Entailment}, \text{Contradiction}, \text{Neutral}\} \tag{2.5}$$

Seminal benchmarks such as TRUE (Honovich et al., 2022) demonstrated that NLI cross-encoders substantially outperform lexical metrics and QA models in factual consistency detection. Recently, Tang et al. (2024) introduced **MiniCheck**, an efficient hallucination detector based on RoBERTa-Large and synthetic training data.

However, existing NLI baselines exhibit three major limitations:
1. **Binary Reductionism**: As demonstrated in our Phase 2 experiments (Chapter 7), systems like MiniCheck collapse NLI into a binary score ($\text{Support} = 1, \text{Non-Support} = 0$). They cannot differentiate between an assertion that **directly contradicts** established facts versus an assertion that is merely **unmentioned / unsupported** due to incomplete evidence.
2. **Context Dilution**: Generic NLI models are trained on brief sentence pairs (e.g., SNLI, MNLI). When presented with extensive document contexts, attention weights dilute, causing zero-shot models to erroneously classify true entailment as neutral (as confirmed by our Phase 7 baseline evaluation on RAGTruth, where zero-shot Macro-F1 was only 34.80%).
3. **Severe Neural Miscalibration**: Standard NLI models output uncalibrated softmax scores that cannot be interpreted as true posterior probabilities, leading to arbitrary thresholding and unreliable risk estimation.

---

## 2.5 Transformer Models and Natural Language Inference

### 2.5.1 Transformer Encoder Architecture
The bedrock of modern factual verifiers is the Transformer encoder (Vaswani et al., 2017). Unlike autoregressive decoders, which utilize causal masking to prevent attention from flowing to future tokens, bidirectional encoders allow every token representation to attend to all tokens across both the premise and hypothesis simultaneously.

Given input token embeddings $\mathbf{X} \in \mathbb{R}^{n \times d}$, self-attention projects $\mathbf{X}$ into Query ($\mathbf{Q}$), Key ($\mathbf{K}$), and Value ($\mathbf{V}$) matrices:
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V} \tag{2.6}$$

### 2.5.2 Cross-Encoders vs. Bi-Encoders
In document-grounded verification, architectural choice is paramount:
- **Bi-Encoder Architecture**: Encodes premise and hypothesis independently into single dense vector representations $\mathbf{u}, \mathbf{v} \in \mathbb{R}^d$ (e.g., Sentence-BERT; Reimers & Gurevych, 2019). While highly efficient for vector search, bi-encoders prevent cross-token interactions between premise and hypothesis, rendering them incapable of detecting fine-grained factual contradictions.
- **Cross-Encoder Architecture**: Concatenates premise and hypothesis into a unified input sequence:
  $$\mathbf{x}_{\text{input}} = \texttt{[CLS]} \circ \mathbf{x}_{\text{premise}} \circ \texttt{[SEP]} \circ \mathbf{x}_{\text{hypothesis}} \circ \texttt{[SEP]} \tag{2.7}$$
  Full multi-head self-attention is computed across all token pairs across all layers, enabling the model to directly align individual entities, modifiers, and negations between context and claim.

```
[Figure 2.2 Placeholder: Natural Language Inference Premise-Hypothesis Alignment]
Caption: Figure 2.2 illustrates the cross-encoder attention mechanism. By feeding the concatenated sequence [CLS] Evidence [SEP] Claim [SEP] through bidirectional Transformer layers, every token in the claim directly cross-attends to every token in the evidence passage, capturing nuanced syntactic dependencies and entity substitutions that bi-encoders miss.
```

### 2.5.3 DeBERTa: Disentangled Attention and Replaced Token Detection
To maximize cross-encoder verification accuracy while maintaining computational feasibility on CPU hardware, VeriLLM selects **DeBERTa-v3** (He et al., 2021, 2023). DeBERTa introduces two transformative architectural advancements over standard BERT and RoBERTa models:

#### 1. Disentangled Attention Mechanism
In traditional Transformers, each token's input embedding represents both its content and its absolute position: $\mathbf{h}_i = \mathbf{x}_i + \mathbf{p}_i$. DeBERTa represents each token using two decoupled vectors: a **content vector** $\mathbf{c}_i$ and a **relative position vector** $\mathbf{p}_{i|j}$. The self-attention matrix between token $i$ and token $j$ is decomposed into four disentangled terms:
$$A_{i,j} = \mathbf{c}_i \mathbf{c}_j^T + \mathbf{c}_i \mathbf{p}_{j|i}^T + \mathbf{p}_{i|j} \mathbf{c}_j^T + \mathbf{p}_{i|j} \mathbf{p}_{j|i}^T \tag{2.8}$$

Because relative spatial distance is decoupled from semantic content, the model evaluates grammatical role and entity alignment independently of position, which is essential for detecting subtle entity substitutions in long documents.

#### 2. ELECTRA-Style Replaced Token Detection (RTD)
In DeBERTa-v3 (He et al., 2023), the traditional Masked Language Model (MLM) pretraining objective is replaced with Replaced Token Detection (RTD). A small generator corrupts masked tokens, and the DeBERTa discriminator is trained to predict whether each token is original or replaced. Unlike MLM (which computes loss on only 15% of masked tokens), RTD trains across 100% of input tokens, dramatically improving sample efficiency and representation richness.

Table 2.2 contrasts the architectural profile of DeBERTa-v3-xsmall against alternative encoder backbones.

**Table 2.2: Architectural Comparison of Transformer Encoders for Local Verification**
| Model Architecture | Parameters | Attention Paradigm | Pretraining Objective | Memory Footprint (Weights) | CPU Claim Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **BERT-Base** (Devlin et al., 2019) | 110M | Standard Self-Attention | MLM + Next Sentence Prediction | ~440 MB | ~1,200 ms |
| **RoBERTa-Large** (Liu et al., 2019) | 355M | Standard Self-Attention | Dynamic Masked Language Model | ~1,355 MB | ~10,000 ms |
| **MiniCheck-RoBERTa** (Tang et al., 2024)| 355M | Standard Self-Attention | Synthetic Chunk Entailment | ~1,355 MB | ~1,530 ms |
| **DeBERTa-v3-Large** (He et al., 2023) | 435M | Disentangled Attention | Replaced Token Detection (RTD) | ~1,740 MB | ~12,500 ms |
| **DeBERTa-v3-xsmall (VeriLLM)** | **70.83M** | **Disentangled Attention** | **Replaced Token Detection (RTD)** | **~270 MB** | **~100–250 ms** |

As demonstrated in Table 2.2, DeBERTa-v3-xsmall provides an optimal Pareto tradeoff: it possesses the advanced inductive bias of disentangled attention and RTD while retaining a lightweight 70.83M parameter footprint that executes up to 10× faster than RoBERTa-Large on CPU hardware.

---

## 2.6 Probability Calibration and Reliable Classification

### 2.6.1 The Miscalibration of Modern Neural Networks
In a probabilistic classification model, the network maps an input $\mathbf{x}$ to a predicted class $\hat{y} \in \{1, \dots, K\}$ and an associated confidence estimate $\hat{p} \in [0, 1]$. Formally, a classifier is **perfectly calibrated** if its predicted confidence matches its empirical probability of correctness:
$$P(\hat{y} = y \mid \hat{p} = p) = p, \quad \forall p \in [0, 1] \tag{2.9}$$

For instance, across 100 propositions assigned an entailment confidence of $0.80$, exactly 80 must be truly entailed.

Seminal empirical work by Guo et al. (2017) demonstrated that while modern deep neural networks achieve superior classification accuracy, they are systematically and severely overconfident. Several architectural and training factors drive this miscalibration:
1. **Network Depth and Width**: Overparameterization provides the capacity for logit magnitudes to grow unchecked, driving post-softmax probabilities toward extreme bounds ($0.0$ or $1.0$).
2. **Normalization Layers**: Structural normalizations (LayerNorm, BatchNorm) alter internal feature distributions, decoupling raw logit scale from true empirical likelihood.
3. **Cross-Entropy Overfitting**: Even after classification error has plateaued, cross-entropy loss continues to decrease by pushing logits further from zero, continually worsening probability calibration (Mukhoti et al., 2020).

```
[Figure 2.3 Placeholder: Reliability Diagrams and Neural Network Miscalibration]
Caption: Figure 2.3 shows conceptual reliability diagrams comparing an uncalibrated modern neural network (left) with a temperature-calibrated network (right). The uncalibrated model exhibits severe overconfidence, with predicted confidence consistently exceeding empirical accuracy (bins fall below the perfect-calibration diagonal). Post-hoc temperature scaling contracts logits toward higher entropy, restoring diagonal calibration alignment.
```

### 2.6.2 Post-Hoc Multi-Class Calibration Techniques
Because retraining large neural networks with calibration-aware loss functions (e.g., focal loss or entropy penalties) is computationally expensive, researchers have established **post-hoc calibration methods** operating on frozen validation logits:

1. **Scalar Temperature Scaling ($C1$)** (Guo et al., 2017):
   Applies a single learnable temperature parameter $T > 0$ to scale all logits:
   $$p_k^{(C1)} = \frac{\exp(z_k / T)}{\sum_{j=1}^K \exp(z_j / T)} \tag{2.10}$$
   Temperature scaling is strictly monotonic ($z_i > z_j \iff z_i/T > z_j/T$), preserving argmax accuracy while effectively controlling average overconfidence.
2. **Vector and Matrix Scaling ($C3$)** (Guo et al., 2017; Kull et al., 2019):
   Applies independent linear transformations to each logit: $\mathbf{z}' = \mathbf{W}\mathbf{z} + \mathbf{b}$. In full matrix scaling, $\mathbf{W} \in \mathbb{R}^{K \times K}$; in vector scaling, $\mathbf{W}$ is constrained to a diagonal matrix $\text{diag}(\mathbf{w})$. While flexible, unconstrained matrix scaling is prone to overfitting on rare classes.
3. **Regularized Prior-Bias + Temperature Scaling ($C2$)** (VeriLLM Formulation):
   In domains with severe class imbalance (such as RAGTruth, where factual contradictions represent only ~3.8% of sentences), standard temperature scaling ($C1$) fails because base-rate shifts between dense training batches and deployment cause class-conditional distortions. $C2$ calibration combines isotropic temperature scaling with a regularized, zero-sum prior-bias vector:
   $$p_k^{(C2)} = \frac{\exp(z_k / T + b_k)}{\sum_{j=1}^K \exp(z_j / T + b_j)}, \quad \text{subject to } \sum_{k=1}^K b_k = 0 \tag{2.11}$$
   The bias vector $\mathbf{b}$ explicitly shifts margins to match true empirical base rates, while $L_2$ regularization ($\lambda_{\text{reg}}\|\mathbf{b}\|_2^2$) prevents parameter drift on small validation samples.

Table 2.3 summarizes the multi-class post-hoc calibration methods evaluated in this thesis.

**Table 2.3: Comparison of Post-Hoc Multi-Class Probability Calibration Methods**
| Method / Paradigm | Mathematical Formulation | Free Parameters | Preserves Argmax Accuracy? | Mitigates Severe Class Imbalance? |
| :--- | :--- | :---: | :---: | :---: |
| **$C0$: Raw Softmax** | $p_k = \text{softmax}(\mathbf{z})_k$ | $0$ | Yes (Baseline) | No (Severe overconfidence) |
| **$C1$: Temp Scaling** | $p_k = \text{softmax}(\mathbf{z} / T)_k$ | $1$ ($T$) | **Yes** | No (Distorts minority classwise ECE) |
| **$C2$: Prior-Bias + Temp**| $p_k = \text{softmax}(\mathbf{z} / T + \mathbf{b})_k$ | **$3$ ($T, b_0, b_1$)** | Can alter borderline predictions | **Yes (Rebalances class base rates)** |
| **$C3$: Vector Scaling** | $p_k = \text{softmax}(\mathbf{w} \odot \mathbf{z} + \mathbf{b})_k$ | $2K$ ($5$ for $K=3$) | Can alter predictions | Risk of overfitting rare classes |
| **Platt Scaling** (Binary) | $\hat{p} = \sigma(a \cdot \text{logit}(p) + b)$ | $2$ ($a, b$) | Preserves binary ordering ($a > 0$) | Binary only (cannot handle 3 classes) |

### 2.6.3 Proper Scoring Rules and Calibration Metrics
To evaluate probability reliability without bias, the literature mandates the use of **strictly proper scoring rules** (Gneiting & Raftery, 2007)—loss functions that are uniquely minimized if and only if the predicted probability distribution matches the true posterior distribution:
- **Negative Log-Likelihood (NLL)**: The standard cross-entropy evaluation metric:
  $$\text{NLL} = -\frac{1}{M}\sum_{i=1}^M \sum_{k=1}^K y_{ik} \log \hat{p}_{ik} \tag{2.12}$$
- **Brier Score**: The multi-class mean squared probability error:
  $$\text{Brier} = \frac{1}{M}\sum_{i=1}^M \sum_{k=1}^K (\hat{p}_{ik} - y_{ik})^2 \tag{2.13}$$
- **Classwise Expected Calibration Error (CW-ECE)**: Evaluates calibration error independently across each class $k$ using equal-width probability bins:
  $$\text{CW-ECE} = \frac{1}{K}\sum_{k=1}^K \sum_{b=1}^B \frac{|B_{b,k}|}{M} \left| \text{acc}_k(B_{b,k}) - \text{conf}_k(B_{b,k}) \right| \tag{2.14}$$
  CW-ECE is critical in imbalanced settings, as Top-1 ECE evaluates only the winning class, concealing severe miscalibration on rare minority classes.

---

## 2.7 Selective Prediction, Abstention, and Epistemic Uncertainty

### 2.7.1 Selective Classification and Risk-Coverage Tradeoffs
In safety-critical applications, an algorithm should be permitted to abstain from making a prediction when uncertainty is high. This paradigm is known as **Selective Classification** (El-Yaniv & Wiener, 2010; Geifman & El-Yaniv, 2017).

A selective classifier $(f, g)$ consists of a prediction function $f(\mathbf{x})$ and a binary selection function $g(\mathbf{x}) \in \{0, 1\}$:
$$(f, g)(\mathbf{x}) = \begin{cases}
f(\mathbf{x}), & \text{if } g(\mathbf{x}) = 1 \quad (\text{Predict}) \\
\text{ABSTAIN}, & \text{if } g(\mathbf{x}) = 0 \quad (\text{Reject / Abstain})
\end{cases} \tag{2.15}$$

Performance is defined by two competing metrics:
- **Coverage**: The empirical proportion of instances for which the model renders a prediction:
  $$\phi(g) = \mathbb{E}[g(\mathbf{x})] \approx \frac{1}{M}\sum_{i=1}^M g(\mathbf{x}_i) \tag{2.16}$$
- **Selective Risk**: The error rate measured strictly over the covered instances:
  $$\mathcal{R}(f, g) = \frac{\mathbb{E}[\ell(f(\mathbf{x}), y) \cdot g(\mathbf{x})]}{\phi(g)} \tag{2.17}$$

```
[Figure 2.4 Placeholder: Selective Prediction Risk-Coverage Operating Curve]
Caption: Figure 2.4 shows the theoretical Risk-Coverage trade-off curve (Geifman & El-Yaniv, 2017). As the model abstains on instances of high epistemic uncertainty (reducing coverage from 100% downward), selective error risk decreases monotonically. In VeriLLM, setting an evidence similarity threshold of tau_floor = 0.20 eliminates forced guesses on unretrieved passages, reducing hallucination MAE by 69.3%.
```

### 2.7.2 Aleatoric vs. Epistemic Uncertainty in Factual Verification
In machine learning theory, uncertainty is partitioned into two distinct categories (Kendall & Gal, 2017):
1. **Aleatoric Uncertainty**: Inherent, irreducible randomness in data (e.g., semantic ambiguity or homonymy in natural language).
2. **Epistemic Uncertainty**: Uncertainty arising from a **lack of knowledge or missing evidence**. In retrieval-augmented verification, epistemic uncertainty dominates when a candidate claim discusses a topic that is simply unmentioned in the reference document.

### 2.7.3 The Epistemic Status of `UNVERIFIABLE`
A pervasive flaw in existing binary hallucination benchmarks (e.g., FactCC, MiniCheck) is treating **absence of evidence** as equivalent to an empirical contradiction. If an LLM states *"The speed of sound in water is 1,480 m/s"*, and the reference document is a biography of William Shakespeare, the assertion is uncorroborated, but it is not logically contradicted.

VeriLLM formalizes an essential epistemic principle:
- **`CONTRADICTED`**: The claim directly conflicts with facts asserted in the reference document. Represents an active factual error.
- **`UNSUPPORTED`**: The claim asserts empirical facts unmentioned in the document, but the context is sufficiently close to warrant an evaluation attempt.
- **`UNVERIFIABLE`**: The claim cannot be evaluated because passage retrieval returned zero passages or embedding similarity fell below the safety threshold ($\tau_{\text{floor}} = 0.20$). **`UNVERIFIABLE` represents defensive epistemic abstention; it must never be penalized as a confirmed hallucination.**

---

## 2.8 Benchmark Datasets and Evaluation Methodologies

### 2.8.1 Overview of Hallucination Benchmarks
Over the past five years, multiple benchmarks have been established to evaluate factual consistency in NLG, summarized in Table 2.4.

**Table 2.4: Comparison of Prominent Factual Consistency and Hallucination Benchmarks**
| Benchmark | Primary Task | Generation Models | Annotation Granularity | Source Partitioning | Primary Limitation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **FactCC** (Kryscinski et al., 2020) | News Summarization | Synthetic rule-based sentence transformations | Binary sentence-level (`CORRECT` vs `INCORRECT`) | Source-disjoint splits | Synthetic noise does not match natural LLM hallucination patterns. |
| **TRUE** (Honovich et al., 2022) | Multi-task meta-benchmark | Standard NLP models (T5, PEGASUS) | Binary consistency across 11 datasets | Variable | Meta-benchmark of older models; lacks modern LLM generators. |
| **HaluEval** (Li et al., 2023) | QA, Dialogue, Summarization | ChatGPT (GPT-3.5) | Binary sample-level consistency | Sample-level | Coarse response-level labels; lacks span-level localization. |
| **RAGTruth** (Yuan et al., 2024) | **Summary, QA, Data-to-Text** | **6 Modern LLMs** (GPT-4, Llama-2-7B/13B/70B, Mistral, GPT-3.5) | **Fine-grained character spans & 3 error types** | **Strictly source-document disjoint** | Imbalanced class distribution (~3.8% contradictions); multi-label mixed spans. |

### 2.8.2 The RAGTruth Benchmark Architecture
To provide a realistic foundation for modern RAG applications, Microsoft Research published **RAGTruth** (Yuan et al., 2024). RAGTruth possesses three unique structural properties:
1. **Diverse Generator Spectrum**: Spans six prominent open-source and proprietary LLMs (GPT-4, GPT-3.5-Turbo, Llama-2-7B/13B/70B, and Mistral-7B), capturing modern hallucination modes.
2. **Span-Level Human Annotations**: Expert human annotators manually marked character spans of hallucinations, categorizing each into:
   - *Evident Contradiction*: Explicit factual conflict with source text.
   - *Subtle Contradiction*: Nuanced entity substitution or numerical distortion.
   - *Baseless Information*: Information completely absent from context.
3. **Strict Source-Document Disjointness**: Unlike benchmarks that partition data randomly at the response level (allowing responses to the same source document to appear in both train and test splits), RAGTruth strictly partitions data at the **source-document level**. Models evaluated on test splits must generalize to entirely unseen reference documents.

---

## 2.9 Research Gap and Motivation for VeriLLM

A critical analysis of the literature reveals five fundamental research gaps at the intersection of automated hallucination detection, natural language inference, and trustworthy AI engineering:

```
[Current State of the Art]
├── Heavy LLM Judges (GPT-4) ───────> High cost, latency, non-deterministic drift, data privacy leakage.
├── Binary NLI Baselines ────────────> Collapses absence of evidence into contradiction; lacks 3-way distinction.
├── Standard Cross-Encoders ─────────> Severe neural overconfidence (Guo et al., 2017); minority-class blindness.
└── Uncalibrated Retrieval ──────────> Forces binary guesses on missing evidence; lacks epistemic abstention.

[Research Gap Addressed by VeriLLM]
└── The absence of an integrated, local, model-agnostic verification framework that unifies:
    1. Document-grounded passage retrieval with sliding-window chunking;
    2. Atomic linguistic proposition extraction with coreference resolution;
    3. Lightweight, disentangled cross-encoder inference (DeBERTa-v3-xsmall, 70.83M parameters);
    4. Out-of-sample regularized prior-bias probability calibration (C2);
    5. Defensive epistemic abstention (UNVERIFIABLE) under retrieval noise; and
    6. Quantitative, conservation-preserving response scorecards (GFI, CPFS, HSI) running 100% locally on CPU.
```

### Gap 1: Inability to Distinguish Contradiction from Absence of Evidence
Existing lightweight verification baselines (such as MiniCheck and FactCC) frame verification as binary classification. In practical applications, an active factual contradiction (e.g., asserting that a medication dosage is 10× higher than stated in a clinical guideline) carries vastly higher real-world risk than an unsupported conversational remark. Collapsing both error modes into a generic "hallucination" flag prevents downstream systems from applying differential risk penalties.

### Gap 2: Neural Miscalibration Under Extreme Minority Imbalance
While modern Transformer cross-encoders achieve high raw classification accuracy, their uncalibrated softmax confidence scores are severely distorted under severe class imbalance (~3.8% contradiction rate). Standard isotropic temperature scaling ($C1$) sharpens majority support predictions but worsens classwise calibration for rare contradictions. Prior to VeriLLM, no document-grounded verification framework incorporated regularized prior-bias calibration ($C2$) to rebalance class-conditional posteriors.

### Gap 3: Penalization of Epistemic Abstention Under Retrieval Noise
In realistic RAG pipelines, dense passage retrieval does not achieve 100% recall; approximately 34% of assertions lack corroborating passages. Existing benchmarking protocols enforce rigid coverage floors that penalize models for abstaining, perversely rewarding uncalibrated heuristics that make forced guesses on missing evidence. A principled framework must decouple coverage floors and provide explicit epistemic abstention (`UNVERIFIABLE`).

### Gap 4: Resource Prohibitive and Cloud-Dependent Architectures
Enterprise and healthcare organizations are increasingly prohibited from transmitting proprietary data to external cloud APIs. Existing verification paradigms either rely on proprietary commercial LLM calls or deploy massive 355M–435M parameter models that cannot run within acceptable latency bounds on commodity CPU hardware.

---

## 2.10 Chapter Summary

Chapter 2 established the theoretical, methodological, and algorithmic foundations of automated hallucination detection:
1. **Statistical Nature of Hallucinations**: We reviewed how autoregressive language modeling optimizes for distributional fluency rather than factual truth, leading to intrinsic and extrinsic hallucinations.
2. **Document Grounding & Retrieval**: We examined the principles of RAG and demonstrated why dense embedding similarity cannot establish logical entailment or detect factual contradictions without cross-attention NLI.
3. **Cross-Encoder Architecture**: We analyzed Transformer encoders and established the theoretical advantages of DeBERTa-v3's disentangled attention and Replaced Token Detection for fine-grained claim verification.
4. **Probability Calibration**: We reviewed neural overconfidence and formalized post-hoc calibration methods ($C0, C1, C2, C3$), demonstrating why prior-bias adjustment is essential under severe class imbalance.
5. **Selective Prediction & Abstention**: We formalized selective classification, risk-coverage trade-offs, and defined the epistemic boundary between `UNSUPPORTED`, `CONTRADICTED`, and `UNVERIFIABLE` claims.
6. **The VeriLLM Research Gap**: We synthesized the specific research gap addressed by this thesis: creating a lightweight, calibrated, document-grounded verification framework that executes locally on commodity CPU hardware, distinguishes active contradictions from missing evidence, and provides transparent factuality quantification.

---

## References

- Barnett, S., Kurniawan, S., Thudumu, S., Brannelly, Z., & Abdelrazek, M. (2024). Seven failure points when engineering a retrieval augmented generation system. In *Proceedings of the IEEE/ACM 3rd International Conference on AI Engineering (CAIN 2024)*, 58–68.
- Bender, E. M., Gebru, T., McMillan-Major, A., & Shmitchell, S. (2021). On the dangers of stochastic parrots: Can language models be too big? In *Proceedings of the 2021 ACM FAccT Conference*, 610–623.
- Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., ... & Amodei, D. (2020). Language models are few-shot learners. In *Advances in Neural Information Processing Systems (NeurIPS 2020)*, 33, 1877–1901.
- Chan, C. M., Chen, W., Su, Y., Yu, J., Xue, W., Zhang, S., ... & Liu, Z. (2023). ChatEval: Towards better LLM-based evaluators through multi-agent debate. *arXiv preprint arXiv:2308.07201*.
- Chen, J., Lin, H., Han, X., & Sun, L. (2022). Benchmarking large language models in retrieval-augmented generation. In *Proceedings of the 38th AAAI Conference on Artificial Intelligence (AAAI 2024)*.
- Dagan, I., Glickman, O., & Magnini, B. (2006). The PASCAL recognising textual entailment challenge. In *Machine Learning Challenges. Evaluating Predictive Uncertainty, Visual Object Classification, and Recognising Textual Entailment*, 177–190. Springer.
- Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. In *Proceedings of NAACL-HLT 2019*, 4171–4186.
- Durmus, E., He, H., & Diab, M. (2020). FEQA: A question answering evaluation framework for faithfulness assessment in abstractive summarization. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL 2020)*, 5055–5070.
- El-Yaniv, R., & Wiener, Y. (2010). On the foundations of noise-free selective classification. *Journal of Machine Learning Research*, 11, 1605–1641.
- Fabbri, A. R., Kryściński, W., McCann, B., Xiong, C., Socher, R., & Radev, D. (2021). SummEval: Re-evaluating summarization evaluation. *Transactions of the Association for Computational Linguistics*, 9, 391–409.
- Geifman, Y., & El-Yaniv, R. (2017). Selective classification for deep neural networks. In *Advances in Neural Information Processing Systems (NeurIPS 2017)*, 30, 4878–4887.
- Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association*, 102(477), 359–378.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. In *International Conference on Machine Learning (ICML 2017)*, PMLR 70, 1321–1330.
- Guu, K., Lee, K., Tung, Z., Pasupat, P., & Chang, M. (2020). Retrieval augmented language model pre-training. In *International Conference on Machine Learning (ICML 2020)*, PMLR 119, 3929–3938.
- He, P., Liu, X., Gao, J., & Chen, W. (2021). DeBERTa: Decoding-enhanced BERT with disentangled attention. In *International Conference on Learning Representations (ICLR 2021)*.
- He, P., Gao, J., & Chen, W. (2023). DeBERTaV3: Improving DeBERTa using ELECTRA-style pre-training with gradient-disentangled embedding sharing. In *International Conference on Learning Representations (ICLR 2023)*.
- Holtzman, A., Buys, J., Du, L., Forbes, M., & Choi, Y. (2020). The curious case of neural text degeneration. In *International Conference on Learning Representations (ICLR 2020)*.
- Honovich, O., Aharoni, R., Herzig, J., Taitelbaum, H., Kukliansky, D., Cohen, V., Scialom, T., Szpektor, I., Hassidim, A., & Matias, Y. (2022). TRUE: Re-evaluating factual consistency evaluation. In *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing (EMNLP 2022)*, 3905–3920.
- Huang, L., Yu, W., Ma, W., Zhong, W., Feng, Z., Wang, H., ... & Liu, T. (2023). A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. *ACM Computing Surveys*, 56(11), 1–39.
- Hughes, S., et al. (2023). Hughes Hallucination Evaluation Model (HHEM). Vectara Technical Report.
- Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Yan, X., Ishii, E., Bang, Y. J., Dai, W., Madotto, A., & Fung, P. (2023). Survey of hallucination in natural language generation. *ACM Computing Surveys*, 55(12), 1–38.
- Kendall, A., & Gal, Y. (2017). What uncertainties do we need in Bayesian deep learning for computer vision? In *Advances in Neural Information Processing Systems (NeurIPS 2017)*, 30, 5574–5584.
- Kryscinski, W., McCann, B., Xiong, C., & Socher, R. (2020). Evaluating the factual consistency of abstractive text summarization. In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP 2020)*, 9332–9346.
- Kull, M., Perello Nieto, M., Kängsepp, M., Silva Filho, T., Song, H., & Flach, P. (2019). Beyond temperature scaling: Obtaining well-calibrated multi-class probabilities with Dirichlet calibration. In *Advances in Neural Information Processing Systems (NeurIPS 2019)*, 32, 12295–12305.
- Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W.-t., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. In *Advances in Neural Information Processing Systems (NeurIPS 2020)*, 33, 9459–9474.
- Li, J., Cheng, X., Zhao, W. X., Nie, J. Y., & Wen, J. R. (2023). HaluEval: A large-scale hallucination evaluation benchmark for large language models. In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP 2023)*, 6449–6464.
- Liu, Y., Iter, D., Xu, Y., Wang, S., Xu, R., & Zhu, C. (2023). G-Eval: NLG evaluation using GPT-4 with better human alignment. In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP 2023)*, 2511–2522.
- Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., & Stoyanov, V. (2019). RoBERTa: A robustly optimized BERT pretraining approach. *arXiv preprint arXiv:1907.11692*.
- Manakul, P., Liusie, A., & Gales, M. J. (2023). SelfCheckGPT: Zero-resource black-box hallucination detection for generative large language models. In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP 2023)*, 9004–9017.
- Maynez, J., Narayan, S., Bohnet, B., & McDonald, R. (2020). On faithfulness and factuality in abstractive summarization. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL 2020)*, 1906–1919.
- Min, S., Krishna, K., Lyu, X., Lewis, M., Yih, W.-t., Koh, P. W., Iyyer, M., Zettlemoyer, L., & Hajishirzi, H. (2023). FActScore: Fine-grained atomic evaluation of factual precision in long form text generation. In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP 2023)*, 12076–12100.
- Mukhoti, J., Kulharia, V., Sanyal, A., Golodetz, S., Torr, P., & Dokania, P. (2020). Calibrating deep neural networks using focal loss. In *Advances in Neural Information Processing Systems (NeurIPS 2020)*, 33, 15288–15299.
- Novikova, J., Dušek, O., Cercas Curry, A., & Rieser, V. (2017). Why we need new evaluation metrics for NLG. In *Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing (EMNLP 2017)*, 2241–2252.
- Platt, J. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. *Advances in Large Margin Classifiers*, 10(3), 61–74.
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP 2019)*, 3982–3992.
- Scialom, T., Dray, P.-A., Lamprier, S., Piwowarski, B., & Staiano, J. (2021). QuestEval: Summarization asks for fact-based evaluation. In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP 2021)*, 6594–6604.
- Tang, L. Y., Bi, T., Thorne, J., & Frank, A. (2024). MiniCheck: Efficient and high-performance hallucination detection in large language models. In *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP 2024)*.
- Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M. A., Lacroix, T., ... & Lample, G. (2023). LLaMA: Open and efficient foundation language models. *arXiv preprint arXiv:2302.13971*.
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. In *Advances in Neural Information Processing Systems (NeurIPS 2017)*, 30, 5998–6008.
- Wang, A., Cho, K., & Lewis, M. (2020). Asking and answering questions to evaluate the factual consistency of summaries. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL 2020)*, 5008–5020.
- Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q. V., & Zhou, D. (2024). Long-form factuality in large language models. *arXiv preprint arXiv:2403.18802*.
- Yuan, H., Zheng, Z., Guan, J., You, C., & Huang, M. (2024). RAGTruth: A hallucination benchmark for retrieval-augmented generation. In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL 2024)*. arXiv:2401.00396.
- Zha, Y., Bhatia, Y., Sun, H., & Su, Y. (2023). AlignScore: Evaluating factual consistency with a unified alignment function. In *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL 2023)*, 11328–11348.
- Zhang, Y., Li, Y., Cui, L., Cai, D., Liu, L., Fu, T., ... & Shi, S. (2023). Siren's song in the AI ocean: A survey on hallucination in large language models. *arXiv preprint arXiv:2309.01219*.
- Zheng, L., Chiang, W. L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., ... & Stoica, I. (2023). Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. In *Advances in Neural Information Processing Systems (NeurIPS 2023)*, 36, 46595–46623.

