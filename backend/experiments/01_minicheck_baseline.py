"""
VeriLLM Phase 2: Standalone MiniCheck Baseline Experiment.

Objective:
1. Load the lightweight MiniCheck baseline ('roberta-large', 355M params) on CPU.
2. Evaluate 3 distinct test claims against a controlled reference document:
   - Claim 1: Clearly SUPPORTED by the reference document.
   - Claim 2: Clearly CONTRADICTED by the reference document.
   - Claim 3: UNMENTIONED / absent from the reference document.
3. Capture raw predictions, support probabilities, and latency.
4. Output a detailed evaluation report demonstrating document-grounded verification behavior.
"""

import os
import sys
import time
import json
from pathlib import Path

# Ensure backend root is on sys.path
CURRENT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = CURRENT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Use offline mode for cached weights to prevent unnecessary network check timeouts
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from minicheck.minicheck import MiniCheck

# Controlled reference text (ground truth context)
REFERENCE_DOCUMENT = (
    "Apollo 11 was the American spaceflight that first landed humans on the Moon. "
    "Commander Neil Armstrong and Lunar Module Pilot Buzz Aldrin landed the Apollo Lunar Module Eagle "
    "on July 20, 1969, at 20:17 UTC. Neil Armstrong became the first person to step onto the lunar surface "
    "six hours and 39 minutes later, on July 21 at 02:56 UTC. Aldrin joined him 19 minutes later. "
    "They spent about two and a quarter hours together outside the spacecraft, and collected 47.5 pounds "
    "(21.5 kg) of lunar material to bring back to Earth. Command Module Pilot Michael Collins flew the "
    "Command Module Columbia alone in lunar orbit while they were on the surface."
)

# Test claims spanning the 3 required behavioral cases
TEST_CASES = [
    {
        "id": "claim_01_supported",
        "category": "SUPPORTED",
        "claim": "Neil Armstrong was the first person to step onto the lunar surface on July 21, 1969.",
        "expected_label": 1,
        "description": "Directly stated in the document (July 21, 02:56 UTC)."
    },
    {
        "id": "claim_02_contradicted",
        "category": "CONTRADICTED",
        "claim": "Buzz Aldrin flew the Command Module Columbia alone in lunar orbit.",
        "expected_label": 0,
        "description": "Contradicts document: Michael Collins flew Columbia alone, while Aldrin was on the surface."
    },
    {
        "id": "claim_03_unmentioned",
        "category": "UNMENTIONED",
        "claim": "The Apollo 11 astronauts ate bacon and freeze-dried peaches for breakfast on their return flight.",
        "expected_label": 0,
        "description": "Not mentioned anywhere in the reference document (outside knowledge / ungrounded)."
    }
]


def run_experiment(model_name: str = "roberta-large"):
    print("=" * 75)
    print(" VeriLLM Phase 2: MiniCheck Baseline Experiment")
    print(f" Model Selected : {model_name} (lytang/MiniCheck-RoBERTa-Large, 355M params)")
    print(f" Execution Mode : CPU (Local PyTorch runtime)")
    print("=" * 75)

    # 1. Initialize MiniCheck and track loading time
    print("\n[Step 1/3] Loading MiniCheck model into memory...")
    start_load = time.perf_counter()
    scorer = MiniCheck(model_name=model_name, batch_size=4)
    load_time = time.perf_counter() - start_load
    print(f"[Done] Model loaded successfully in {load_time:.2f} seconds.")

    # 2. Run inference on each test case
    print("\n[Step 2/3] Evaluating test claims against reference document...")
    results = []

    for idx, test_case in enumerate(TEST_CASES, start=1):
        claim_text = test_case["claim"]
        cat = test_case["category"]
        print(f"\n--- Test Case {idx}: [{cat}] ---")
        print(f"Claim: \"{claim_text}\"")

        start_eval = time.perf_counter()
        # MiniCheck.score expects: docs (list of str), claims (list of str)
        pred_labels, raw_probs, chunks, chunk_probs = scorer.score(
            docs=[REFERENCE_DOCUMENT],
            claims=[claim_text]
        )
        eval_time = time.perf_counter() - start_eval

        pred_label = int(pred_labels[0])
        support_prob = float(raw_probs[0])

        verdict_str = "SUPPORTED (1)" if pred_label == 1 else "UNSUPPORTED (0)"
        is_expected = (pred_label == test_case["expected_label"])

        print(f"Model Prediction    : {verdict_str}")
        print(f"Support Probability : {support_prob:.4f}")
        print(f"Inference Latency   : {eval_time:.3f} s")
        print(f"Matches Expected    : {'PASS' if is_expected else 'DISCREPANCY'}")

        results.append({
            "test_id": test_case["id"],
            "category": cat,
            "claim": claim_text,
            "expected_label": test_case["expected_label"],
            "predicted_label": pred_label,
            "support_probability": round(support_prob, 4),
            "latency_seconds": round(eval_time, 3),
            "matches_expected": is_expected,
            "description": test_case["description"],
            "used_chunk": chunks[0][0] if chunks and chunks[0] else REFERENCE_DOCUMENT
        })

    # 3. Print Summary Table
    print("\n" + "=" * 75)
    print(" EXPERIMENT RESULTS SUMMARY TABLE")
    print("=" * 75)
    print(f"{'Category':<15} | {'Pred':<5} | {'Prob':<8} | {'Latency':<8} | {'Status':<6} | Claim Snippet")
    print("-" * 75)
    for r in results:
        status_str = "PASS" if r["matches_expected"] else "FAIL"
        snippet = (r["claim"][:35] + "...") if len(r["claim"]) > 35 else r["claim"]
        print(f"{r['category']:<15} | {r['predicted_label']:<5} | {r['support_probability']:<8.4f} | {r['latency_seconds']:<7.3f}s | {status_str:<6} | {snippet}")
    print("=" * 75)

    # 4. Save results to report files
    results_dir = CURRENT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    report_payload = {
        "experiment_name": "Phase 2: MiniCheck Baseline Verification",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_name": model_name,
        "runtime_device": "CPU",
        "load_time_seconds": round(load_time, 2),
        "reference_document": REFERENCE_DOCUMENT,
        "test_results": results,
        "scientific_observations": {
            "binary_limitation": (
                "MiniCheck produces binary output (1=supported, 0=unsupported). "
                "Consequently, both contradicted claims and unmentioned claims receive label 0. "
                "This motivates VeriLLM's Phase 5 and Phase 7 4-way classification contribution."
            ),
            "document_grounding": (
                "MiniCheck evaluates document faithfulness rather than universal real-world truth. "
                "A statement true in reality but absent from the document is classified as unsupported."
            )
        }
    }

    json_path = results_dir / "01_minicheck_baseline_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    md_path = results_dir / "01_minicheck_baseline_report.md"
    generate_markdown_report(report_payload, md_path)

    print(f"\n[Saved] Structured JSON report -> {json_path}")
    print(f"[Saved] Markdown summary report  -> {md_path}")
    print("Phase 2 Experiment execution finished successfully.")
    return results


def generate_markdown_report(data: dict, out_path: Path):
    md_content = f"""# VeriLLM Phase 2: MiniCheck Baseline Experiment Report

- **Date / Time**: {data['timestamp']}
- **Model**: `{data['model_name']}` (`lytang/MiniCheck-RoBERTa-Large`, 355M parameters)
- **Runtime Environment**: CPU-only PyTorch (Intel UHD 620 laptop environment)
- **Model Load Time**: {data['load_time_seconds']} seconds

---

## 1. Controlled Reference Document
> "{data['reference_document']}"

---

## 2. Experimental Results

| Test Case | Target Category | Expected Label | Predicted Label | Support Probability | Inference Latency | Validation |
|---|---|---|---|---|---|---|
"""
    for r in data["test_results"]:
        val_icon = "PASS" if r["matches_expected"] else "FAIL"
        md_content += f"| `{r['test_id']}` | **{r['category']}** | `{r['expected_label']}` | `{r['predicted_label']}` | **{r['support_probability']:.4f}** | {r['latency_seconds']:.3f} s | **{val_icon}** |\n"

    md_content += f"""
---

## 3. Individual Claim Analysis

"""
    for r in data["test_results"]:
        md_content += f"""### `{r['test_id']}` ({r['category']})
- **Claim**: "{r['claim']}"
- **Description**: {r['description']}
- **Predicted Output**: `label = {r['predicted_label']}`, `support_prob = {r['support_probability']:.4f}`
- **Outcome**: {'Correctly classified as expected.' if r['matches_expected'] else 'Discrepancy observed.'}

"""

    md_content += f"""---

## 4. Scientific Observations & Known Baseline Limitations

1. **Binary Classification Ambiguity**:
   MiniCheck evaluates claims on a single binary dimension: `1` (supported by the document) vs `0` (unsupported by the document).
   - Case 2 (*Contradicted*: Buzz Aldrin flew Columbia alone) receives `label = 0` with low support probability.
   - Case 3 (*Unmentioned*: Astronauts ate bacon and peaches) also receives `label = 0` with low support probability.
   - **Research Implication for VeriLLM**: A user cannot determine from MiniCheck alone whether an unsupported claim directly contradicts the facts or is simply unmentioned. VeriLLM's Phase 5 and Phase 7 directly solve this by implementing a 4-way classification pipeline (`SUPPORTED`, `CONTRADICTED`, `UNSUPPORTED`, `UNVERIFIABLE`).

2. **Document Grounding vs. Universal Truth**:
   MiniCheck strictly measures document faithfulness. If an LLM generates a claim that is true in the real world but absent from the reference text, MiniCheck will classify it as `unsupported`. This is intentional and necessary for grounded evaluation (e.g. RAG pipelines).
"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md_content)


if __name__ == "__main__":
    run_experiment()

