"""
VeriLLM Phase 7: Baseline Evaluator and Model Benchmark Script.
Evaluates the existing zero-shot baselines (cross-encoder/nli-deberta-v3-xsmall and MiniCheck-RoBERTa-Large)
on the stratified RAGTruth validation set using batched inference.
Measures model parameters, memory usage, inference latency, confusion matrices, and failure cases.
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Any

# Ensure backend root is on sys.path
CURRENT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = CURRENT_DIR.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Force offline execution
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

from ml_engine.claim_verifier.minicheck_verifier import MiniCheckVerifier
from ml_engine.claim_verifier.nli_verifier import NLIVerifier


def load_stratified_val_sample(sample_size_per_class: int = 30, seed: int = 42) -> List[Dict[str, Any]]:
    """Sample a balanced validation set across the 3 main classes (excluding mixed)."""
    val_file = BACKEND_DIR / "data" / "processed" / "ragtruth_val.jsonl"
    by_class = defaultdict(list)
    
    with open(val_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            lbl = item["primary_label"]
            if lbl in ("SUPPORTED_PROXY", "CONTRADICTED", "UNSUPPORTED"):
                by_class[lbl].append(item)

    rng = random.Random(seed)
    sampled = []
    for lbl in ["SUPPORTED_PROXY", "CONTRADICTED", "UNSUPPORTED"]:
        pool = by_class[lbl]
        k = min(len(pool), sample_size_per_class)
        sampled.extend(rng.sample(pool, k))

    rng.shuffle(sampled)
    print(f"Loaded stratified validation sample: {len(sampled)} examples ({sample_size_per_class} per class).", flush=True)
    return sampled


def benchmark_model_specs():
    """Verify actual model parameter counts, memory, and disk footprint."""
    print("\n" + "=" * 80, flush=True)
    print(" Benchmarking Available Pretrained Models (Actual Hardware Footprint)", flush=True)
    print("=" * 80, flush=True)

    # 1. NLI Cross-Encoder
    nli_verifier = NLIVerifier("cross-encoder/nli-deberta-v3-xsmall")
    model_nli = nli_verifier.model.model
    nli_params = sum(p.numel() for p in model_nli.parameters())
    nli_trainable = sum(p.numel() for p in model_nli.parameters() if p.requires_grad)
    nli_mem_mb = sum(p.numel() * p.element_size() for p in model_nli.parameters()) / (1024 * 1024)

    print("Model 1: cross-encoder/nli-deberta-v3-xsmall (DeBERTa-v3 architecture)", flush=True)
    print(f"  - Total Parameters     : {nli_params:,} ({nli_params/1e6:.2f} Million)", flush=True)
    print(f"  - Trainable Parameters : {nli_trainable:,}", flush=True)
    print(f"  - Memory (Weights)     : {nli_mem_mb:.2f} MB", flush=True)
    print(f"  - Vocabulary Size      : {nli_verifier.model.tokenizer.vocab_size}", flush=True)

    # 2. MiniCheck-RoBERTa-Large
    mc_verifier = MiniCheckVerifier("roberta-large")
    mc_model = mc_verifier.scorer.model.model
    mc_params = sum(p.numel() for p in mc_model.parameters())
    mc_mem_mb = sum(p.numel() * p.element_size() for p in mc_model.parameters()) / (1024 * 1024)

    print("\nModel 2: lytang/MiniCheck-RoBERTa-Large (RoBERTa-Large architecture)", flush=True)
    print(f"  - Total Parameters     : {mc_params:,} ({mc_params/1e6:.2f} Million)", flush=True)
    print(f"  - Memory (Weights)     : {mc_mem_mb:.2f} MB", flush=True)

    return {
        "nli_model": {
            "name": "cross-encoder/nli-deberta-v3-xsmall",
            "total_parameters": nli_params,
            "weight_memory_mb": round(nli_mem_mb, 2)
        },
        "minicheck_model": {
            "name": "lytang/MiniCheck-RoBERTa-Large",
            "total_parameters": mc_params,
            "weight_memory_mb": round(mc_mem_mb, 2)
        }
    }


def evaluate_baselines():
    specs = benchmark_model_specs()
    samples = load_stratified_val_sample(sample_size_per_class=100, seed=42)

    nli_verifier = NLIVerifier("cross-encoder/nli-deberta-v3-xsmall")
    mc_verifier = MiniCheckVerifier("roberta-large")

    print("\n" + "=" * 80, flush=True)
    print(" Running Batched Baseline Evaluation on Stratified Validation Set...", flush=True)
    print("=" * 80, flush=True)

    # Prepare inputs with windowed context
    docs = []
    claims = []
    pairs = []
    y_true_3class = []
    y_true_binary = []

    for ex in samples:
        gold_3class = ex["primary_label"]  # SUPPORTED_PROXY, CONTRADICTED, UNSUPPORTED
        gold_binary = "SUPPORTED" if gold_3class == "SUPPORTED_PROXY" else "HALLUCINATED"
        y_true_3class.append(gold_3class)
        y_true_binary.append(gold_binary)

        ctx = ex["context_text"]
        sent = ex["sentence_text"]

        # Preserve sentence completely; window context if excessively long
        ctx_words = ctx.split()
        if len(ctx_words) > 350:
            ctx = " ".join(ctx_words[:350])

        docs.append(ctx)
        claims.append(sent)
        pairs.append((ctx, sent))

    total_samples = len(samples)

    # 1. Evaluate NLI Cross-Encoder in batches
    print(f"[1/2] Evaluating NLI Cross-Encoder ({total_samples} pairs in batches of 16)...", flush=True)
    t0_nli = time.time()
    nli_results = nli_verifier.predict_pairs(pairs)
    total_nli_time = time.time() - t0_nli
    avg_nli_lat = (total_nli_time / total_samples) * 1000
    print(f"  -> NLI evaluation completed in {total_nli_time:.2f}s ({avg_nli_lat:.1f} ms/pair).", flush=True)

    # 2. Evaluate MiniCheck in batches
    print(f"[2/2] Evaluating MiniCheck-RoBERTa-Large ({total_samples} pairs in batches)...", flush=True)
    t0_mc = time.time()
    mc_results = mc_verifier.score_pairs(docs, claims)
    total_mc_time = time.time() - t0_mc
    avg_mc_lat = (total_mc_time / total_samples) * 1000
    print(f"  -> MiniCheck evaluation completed in {total_mc_time:.2f}s ({avg_mc_lat:.1f} ms/pair).", flush=True)

    # Parse predictions
    y_pred_nli_3class = []
    y_pred_nli_binary = []
    y_pred_mc_binary = []
    failure_cases = []

    for idx, (ex, (p_entail, p_contra, p_neut), (mc_pred, mc_prob)) in enumerate(zip(samples, nli_results, mc_results)):
        gold_3class = ex["primary_label"]

        # 3-class NLI prediction
        if p_contra >= max(p_entail, p_neut):
            pred_3c = "CONTRADICTED"
            pred_nli_bin = "HALLUCINATED"
        elif p_entail >= max(p_contra, p_neut):
            pred_3c = "SUPPORTED_PROXY"
            pred_nli_bin = "SUPPORTED"
        else:
            pred_3c = "UNSUPPORTED"
            pred_nli_bin = "HALLUCINATED"

        y_pred_nli_3class.append(pred_3c)
        y_pred_nli_binary.append(pred_nli_bin)

        # MiniCheck binary prediction
        pred_mc_bin = "SUPPORTED" if mc_prob >= 0.50 else "HALLUCINATED"
        y_pred_mc_binary.append(pred_mc_bin)

        # Record failure cases
        if pred_3c != gold_3class and len(failure_cases) < 10:
            failure_cases.append({
                "example_id": ex["example_id"],
                "task_type": ex["task_type"],
                "gold_label": gold_3class,
                "nli_predicted": pred_3c,
                "nli_probs": {
                    "entailment": round(p_entail, 4),
                    "contradiction": round(p_contra, 4),
                    "neutral": round(p_neut, 4)
                },
                "minicheck_support_prob": round(mc_prob, 4),
                "sentence": ex["sentence_text"],
                "has_implicit_true": ex.get("has_implicit_true", False),
                "has_due_to_null": ex.get("has_due_to_null", False),
                "annotations": ex.get("overlapping_spans", [])
            })

    # -------------------------------------------------------------
    # Compute Metrics
    # -------------------------------------------------------------
    labels_3class = ["SUPPORTED_PROXY", "CONTRADICTED", "UNSUPPORTED"]
    p_3c, r_3c, f1_3c, _ = precision_recall_fscore_support(y_true_3class, y_pred_nli_3class, labels=labels_3class, average=None)
    macro_f1_3c = precision_recall_fscore_support(y_true_3class, y_pred_nli_3class, average="macro")[2]
    cm_3c = confusion_matrix(y_true_3class, y_pred_nli_3class, labels=labels_3class).tolist()

    # Binary Metrics
    binary_labels = ["SUPPORTED", "HALLUCINATED"]
    p_nli_bin, r_nli_bin, f1_nli_bin, _ = precision_recall_fscore_support(y_true_binary, y_pred_nli_binary, labels=binary_labels, average=None)
    macro_f1_nli_bin = precision_recall_fscore_support(y_true_binary, y_pred_nli_binary, average="macro")[2]
    cm_nli_bin = confusion_matrix(y_true_binary, y_pred_nli_binary, labels=binary_labels).tolist()

    p_mc_bin, r_mc_bin, f1_mc_bin, _ = precision_recall_fscore_support(y_true_binary, y_pred_mc_binary, labels=binary_labels, average=None)
    macro_f1_mc_bin = precision_recall_fscore_support(y_true_binary, y_pred_mc_binary, average="macro")[2]
    cm_mc_bin = confusion_matrix(y_true_binary, y_pred_mc_binary, labels=binary_labels).tolist()

    print("\n" + "=" * 80, flush=True)
    print(" BASELINE EVALUATION RESULTS ON RAGTRUTH VALIDATION SET", flush=True)
    print("=" * 80, flush=True)
    print(f"Latency: NLI Cross-Encoder: {avg_nli_lat:.1f} ms/claim | MiniCheck-RoBERTa-Large: {avg_mc_lat:.1f} ms/claim ({avg_mc_lat/avg_nli_lat:.1f}x slower)", flush=True)
    print("\n--- Model 1: NLI Cross-Encoder (3-Class Fine-Grained) ---", flush=True)
    print(f"Macro-F1: {macro_f1_3c*100:.2f}%", flush=True)
    for lbl, p, r, f in zip(labels_3class, p_3c, r_3c, f1_3c):
        print(f"  - {lbl:<16}: Precision={p*100:.1f}%, Recall={r*100:.1f}%, F1={f*100:.1f}%", flush=True)
    print(f"Confusion Matrix (Rows=True, Cols=Pred: {labels_3class}):", flush=True)
    for row in cm_3c:
        print(f"  {row}", flush=True)

    print("\n--- Binary Hallucination Detection Comparison ---", flush=True)
    print(f"1. NLI DeBERTa (Zero-Shot) Binary Macro-F1    : {macro_f1_nli_bin*100:.2f}% (Hallucination F1: {f1_nli_bin[1]*100:.2f}%)", flush=True)
    print(f"2. MiniCheck-RoBERTa-Large Binary Macro-F1    : {macro_f1_mc_bin*100:.2f}% (Hallucination F1: {f1_mc_bin[1]*100:.2f}%)", flush=True)

    # -------------------------------------------------------------
    # Export Reports
    # -------------------------------------------------------------
    results_dir = BACKEND_DIR / "experiments" / "results"
    out_json = results_dir / "07_baseline_evaluation_results.json"
    results_data = {
        "model_specs": specs,
        "sample_size": total_samples,
        "latencies_ms": {
            "nli_deberta_v3_xsmall": round(avg_nli_lat, 2),
            "minicheck_roberta_large": round(avg_mc_lat, 2)
        },
        "nli_3class_metrics": {
            "macro_f1": round(macro_f1_3c * 100, 2),
            "classes": {
                lbl: {
                    "precision": round(p * 100, 2),
                    "recall": round(r * 100, 2),
                    "f1": round(f * 100, 2)
                }
                for lbl, p, r, f in zip(labels_3class, p_3c, r_3c, f1_3c)
            },
            "confusion_matrix": cm_3c,
            "labels_order": labels_3class
        },
        "binary_comparison": {
            "nli_deberta": {
                "macro_f1": round(macro_f1_nli_bin * 100, 2),
                "supported_f1": round(f1_nli_bin[0] * 100, 2),
                "hallucination_f1": round(f1_nli_bin[1] * 100, 2),
                "confusion_matrix": cm_nli_bin
            },
            "minicheck": {
                "macro_f1": round(macro_f1_mc_bin * 100, 2),
                "supported_f1": round(f1_mc_bin[0] * 100, 2),
                "hallucination_f1": round(f1_mc_bin[1] * 100, 2),
                "confusion_matrix": cm_mc_bin
            }
        },
        "failure_cases": failure_cases
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)

    out_md = results_dir / "07_baseline_evaluation_report.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# VeriLLM Phase 7: Pre-Fine-Tuning Baseline Evaluation Report\n\n")
        f.write("## 1. Pretrained Model Architecture & Hardware Benchmark\n\n")
        f.write("| Model | Architecture | Parameters | Memory (Weights) | CPU Latency / Claim |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|\n")
        f.write(f"| `cross-encoder/nli-deberta-v3-xsmall` | DeBERTa-v3 | **{specs['nli_model']['total_parameters']:,}** (70.8M) | **{specs['nli_model']['weight_memory_mb']} MB** | **{avg_nli_lat:.1f} ms** |\n")
        f.write(f"| `lytang/MiniCheck-RoBERTa-Large` | RoBERTa-Large | **{specs['minicheck_model']['total_parameters']:,}** (355M) | **{specs['minicheck_model']['weight_memory_mb']} MB** | **{avg_mc_lat:.1f} ms** ({avg_mc_lat/avg_nli_lat:.1f}× slower) |\n\n")

        f.write("## 2. Zero-Shot 3-Class NLI Performance (Validation Set)\n\n")
        f.write(f"- **Evaluated Sample**: `300` balanced sentences (100 per class) from `ragtruth_val.jsonl` (375 unseen validation sources).\n")
        f.write(f"- **Overall Macro-F1**: **`{macro_f1_3c*100:.2f}%`**\n\n")
        f.write("| Class Label | Precision | Recall | F1-Score |\n")
        f.write("|:---|:---:|:---:|:---:|\n")
        for lbl, p, r, f_val in zip(labels_3class, p_3c, r_3c, f1_3c):
            f.write(f"| **`{lbl}`** | {p*100:.1f}% | {r*100:.1f}% | **{f_val*100:.1f}%** |\n")

        f.write(f"\n### 3-Class Confusion Matrix\n")
        f.write(f"*(Rows = True Label, Columns = Predicted: `SUPPORTED_PROXY`, `CONTRADICTED`, `UNSUPPORTED`)*\n\n")
        f.write(f"```text\n{cm_3c}\n```\n\n")

        f.write("## 3. Binary Hallucination Detection Comparison\n\n")
        f.write("| Baseline Model | Macro-F1 | Supported F1 | Hallucination F1 | Latency |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|\n")
        f.write(f"| **`nli-deberta-v3-xsmall` (Zero-Shot)** | **{macro_f1_nli_bin*100:.2f}%** | {f1_nli_bin[0]*100:.2f}% | {f1_nli_bin[1]*100:.2f}% | **{avg_nli_lat:.1f} ms** |\n")
        f.write(f"| **`MiniCheck-RoBERTa-Large`** | **{macro_f1_mc_bin*100:.2f}%** | {f1_mc_bin[0]*100:.2f}% | {f1_mc_bin[1]*100:.2f}% | {avg_mc_lat:.1f} ms |\n\n")

        f.write("## 4. Error Analysis & Baseline Failure Modes\n\n")
        for fc in failure_cases[:5]:
            f.write(f"### Example `{fc['example_id']}` ({fc['task_type']})\n")
            f.write(f"- **Sentence**: *\"{fc['sentence']}\"*\n")
            f.write(f"- **Gold Label**: `{fc['gold_label']}` | **NLI Predicted**: `{fc['nli_predicted']}` (MiniCheck prob: `{fc['minicheck_support_prob']}`)\n")
            f.write(f"- **NLI Output Probabilities**: Entailment: `{fc['nli_probs']['entailment']}`, Contradiction: `{fc['nli_probs']['contradiction']}`, Neutral: `{fc['nli_probs']['neutral']}`\n")
            f.write(f"- **Flags**: `implicit_true={fc['has_implicit_true']}`, `due_to_null={fc['has_due_to_null']}`\n")
            if fc['annotations']:
                f.write(f"- **Annotations**: *\"{fc['annotations'][0]['text']}\"* ({fc['annotations'][0]['label_type']})\n")
            f.write("\n")

    print(f"\n[Saved] Baseline results to: {out_json}", flush=True)
    print(f"[Saved] Baseline report to: {out_md}", flush=True)


if __name__ == "__main__":
    evaluate_baselines()
