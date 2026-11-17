"""
VeriLLM Phase 5 Demonstration Script: End-to-End Claim Verification Pipeline.
Integrates Phase 3 (Claim Extractor), Phase 4 (Evidence Retriever), and Phase 5 (Claim Verifier)
to demonstrate 4-way factuality classification (SUPPORTED, CONTRADICTED, UNSUPPORTED, UNVERIFIABLE)
with full citation provenance, MiniCheck scores, 3-way NLI probabilities, and factuality metrics.
"""

import os
import sys
import json
import time
from pathlib import Path

# Ensure backend root is on sys.path
CURRENT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = CURRENT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Force offline execution for cached models
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from ml_engine.claim_extractor import ClaimExtractor
from ml_engine.evidence_retriever import (
    parse_pdf,
    SlidingWindowChunker,
    PassageRetriever
)
from ml_engine.claim_verifier import ClaimVerifier, VerificationCategory


def run_pipeline_demo():
    print("=" * 90)
    print(" VeriLLM Phase 5: End-to-End Claim Verification Pipeline Demonstration")
    print("=" * 90)

    results_dir = CURRENT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = results_dir / "sample_apollo11_mission.pdf"

    if not pdf_path.exists():
        print(f"[Error] Reference PDF not found at: {pdf_path}")
        print("Please run backend/experiments/03_evidence_retrieval_demo.py first.")
        return

    # -------------------------------------------------------------
    # Step 1: Ingest and index reference document
    # -------------------------------------------------------------
    print(f"\n[Step 1] Ingesting reference document '{pdf_path.name}'...")
    doc = parse_pdf(pdf_path)
    print(f"  -> Extracted {doc.total_pages} pages ({doc.total_words} words).")

    print("\n[Step 2] Chunking and embedding reference document...")
    chunker = SlidingWindowChunker(chunk_size_words=60, overlap_words=15)
    chunks = chunker.chunk_pdf_pages(doc.pages, document_name=doc.filename)
    
    retriever = PassageRetriever(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        top_k=2
    )
    retriever.index_chunks(chunks)
    print(f"  -> Indexed {len(chunks)} chunks into dense passage retriever.")

    # -------------------------------------------------------------
    # Step 2: Complex candidate LLM response with all 4 categories
    # -------------------------------------------------------------
    candidate_llm_response = (
        "The Apollo 11 mission was launched on a Saturn V rocket from Kennedy Space Center on July 16, 1969. "
        "Neil Armstrong and Buzz Aldrin landed the lunar module Eagle on July 20, 1969, and they collected 47.5 pounds of lunar rock. "
        "Michael Collins walked on the Moon with Armstrong while Buzz Aldrin flew the Command Module. "
        "The astronauts safely returned to Earth and splashed down in the Pacific Ocean on July 24, 1969. "
        "Did you know the mission was watched by over 650 million television viewers worldwide? "
        "Buzz Aldrin also planted an olive branch on Mars. "
        "In my opinion, Apollo 11 is humanity's greatest triumph."
    )

    print("\n[Step 3] Extracting atomic claims using Phase 3 ClaimExtractor...")
    extractor = ClaimExtractor(
        resolve_coreference=True,
        decompose_compound=True,
        include_non_factual=True
    )
    extraction_result = extractor.extract_claims(candidate_llm_response)
    print(f"  -> Raw response length: {len(candidate_llm_response)} chars, {extraction_result.total_sentences} sentences.")
    print(f"  -> Extracted {len(extraction_result.claims)} atomic claims ({extraction_result.factual_claims_count} factual, {extraction_result.non_factual_count} non-factual).")

    # -------------------------------------------------------------
    # Step 3: Retrieve relevant evidence passages
    # -------------------------------------------------------------
    print("\n[Step 4] Retrieving top evidence passages using Phase 4 PassageRetriever...")
    evidence_matches = retriever.retrieve_for_claims(extraction_result.claims, top_k=2)
    print(f"  -> Matched evidence passages for all {len(evidence_matches)} claims.")

    # -------------------------------------------------------------
    # Step 4: Verify claims using Phase 5 ClaimVerifier
    # -------------------------------------------------------------
    print("\n[Step 5] Executing Phase 5 ClaimVerifier (MiniCheck + DeBERTa-v3 NLI + Decision Matrix)...")
    verifier = ClaimVerifier(
        minicheck_model="roberta-large",
        nli_model="cross-encoder/nli-deberta-v3-xsmall"
    )

    t0 = time.time()
    report = verifier.verify_claims(
        claims=extraction_result.claims,
        evidence_matches=evidence_matches,
        document_name=doc.filename
    )
    duration = time.time() - t0
    print(f"  -> Verified {report.total_claims_processed} claims in {duration:.3f}s ({duration/report.total_claims_processed*1000:.1f}ms/claim).")

    # -------------------------------------------------------------
    # Step 5: Display Verification Results Table
    # -------------------------------------------------------------
    print("\n" + "=" * 90)
    print(f"{'ID':<10} | {'Classification':<14} | {'MC Prob':<8} | {'NLI (C/E/N)':<15} | Claim Text")
    print("=" * 90)

    category_colors = {
        VerificationCategory.SUPPORTED: "[SUPPORTED]   ",
        VerificationCategory.CONTRADICTED: "[CONTRADICTED]",
        VerificationCategory.UNSUPPORTED: "[UNSUPPORTED] ",
        VerificationCategory.UNVERIFIABLE: "[UNVERIFIABLE]",
        VerificationCategory.NON_FACTUAL: "[NON_FACTUAL] "
    }

    serialized_claims = []

    for vc in report.claims:
        label = category_colors.get(vc.classification, str(vc.classification))
        mc_prob_str = f"{vc.scores.minicheck_probability:.4f}" if vc.scores.minicheck_probability is not None else "N/A"
        
        if vc.scores.nli_contradiction_prob is not None:
            nli_str = f"{vc.scores.nli_contradiction_prob:.2f}/{vc.scores.nli_entailment_prob:.2f}/{vc.scores.nli_neutral_prob:.2f}"
        else:
            nli_str = "N/A"

        print(f"{vc.claim_id:<10} | {label:<14} | {mc_prob_str:<8} | {nli_str:<15} | \"{vc.claim_text}\"")
        if vc.primary_evidence:
            chunk = vc.primary_evidence.chunk
            print(f"{'':<10}   Citation  : {chunk.document_name} (Page {chunk.page_number}, chunk: {chunk.chunk_id}, sim: {vc.primary_evidence.similarity_score:.4f})")
        print(f"{'':<10}   Reasoning : {vc.reasoning_trace}")
        print()

        serialized_claims.append({
            "claim_id": vc.claim_id,
            "claim_text": vc.claim_text,
            "classification": vc.classification.value,
            "is_factual": vc.is_factual,
            "scores": vc.scores.model_dump(),
            "decision_rule": vc.decision_rule,
            "reasoning_trace": vc.reasoning_trace,
            "primary_evidence": {
                "document_name": vc.primary_evidence.chunk.document_name,
                "page_number": vc.primary_evidence.chunk.page_number,
                "chunk_id": vc.primary_evidence.chunk.chunk_id,
                "similarity_score": round(vc.primary_evidence.similarity_score, 4),
                "text_snippet": vc.primary_evidence.chunk.text[:120] + "..."
            } if vc.primary_evidence else None
        })

    # -------------------------------------------------------------
    # Step 6: Factuality & Hallucination Quantification Metrics
    # -------------------------------------------------------------
    m = report.metrics
    print("=" * 90)
    print(" FACTUALITY AND HALLUCINATION QUANTIFICATION METRICS")
    print("=" * 90)
    print(f"  Total Claims Extracted        : {m.total_claims}")
    print(f"  Empirical Factual Claims (N)  : {m.factual_claims}")
    print(f"  Non-Factual Claims Excluded   : {m.non_factual_claims} (opinions, greetings, meta-text)")
    print("-" * 90)
    supp_str = f"{m.evidence_support_rate:.2f}%" if m.evidence_support_rate is not None else "N/A"
    contra_str = f"{m.confirmed_contradiction_rate:.2f}%" if m.confirmed_contradiction_rate is not None else "N/A"
    unsup_str = f"{m.unsupported_rate:.2f}%" if m.unsupported_rate is not None else "N/A"
    unver_str = f"{m.unverifiable_rate:.2f}%" if m.unverifiable_rate is not None else "N/A"
    uncert_str = f"{m.uncertainty_rate:.2f}%" if m.uncertainty_rate is not None else "N/A"

    print(f"  Evidence Support Rate         : {supp_str} ({m.supported_count} / {m.factual_claims})")
    print(f"  Confirmed Contradiction Rate  : {contra_str} ({m.contradicted_count} / {m.factual_claims})")
    print(f"  Unsupported Rate (Absence)    : {unsup_str} ({m.unsupported_count} / {m.factual_claims})")
    print(f"  Unverifiable Rate             : {unver_str} ({m.unverifiable_count} / {m.factual_claims})")
    print(f"  Uncertainty Rate (Total)      : {uncert_str} (({m.unsupported_count} + {m.unverifiable_count}) / {m.factual_claims})")
    print(f"  Metric Denominator Trace      : {m.denominator_description}")
    print("=" * 90)

    # -------------------------------------------------------------
    # Step 7: Export Structured Results and Report
    # -------------------------------------------------------------
    out_json = results_dir / "04_claim_verification_demo_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "04_claim_verification_pipeline_demo",
            "document": doc.filename,
            "models_used": {
                "document_verifier": "lytang/MiniCheck-RoBERTa-Large (355M params)",
                "nli_cross_encoder": "cross-encoder/nli-deberta-v3-xsmall (70.8M params)",
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2 (22.7M params)"
            },
            "candidate_response": candidate_llm_response,
            "metrics": m.model_dump(),
            "claims": serialized_claims,
            "execution_duration_seconds": round(duration, 3)
        }, f, indent=2)
    print(f"\n[Saved] Structured JSON results to: {out_json}")

    out_md = results_dir / "04_claim_verification_demo_report.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# VeriLLM Phase 5: Claim Verification Pipeline Demonstration Report\n\n")
        f.write(f"- **Reference Document**: `{doc.filename}` ({doc.total_pages} pages, {len(chunks)} chunks)\n")
        f.write(f"- **Fact-Checking Baselines**: `MiniCheck-RoBERTa-Large` & `cross-encoder/nli-deberta-v3-xsmall` (CPU Runtime)\n")
        f.write(f"- **Total Latency**: {duration:.2f}s ({duration/report.total_claims_processed*1000:.1f}ms per claim)\n\n")
        f.write("## 1. Verified Claims Breakdown\n\n")
        f.write("| Claim ID | Proposition | Classification | MiniCheck Prob | NLI (Contra/Entail/Neut) | Citation | Reasoning |\n")
        f.write("|:---|:---|:---:|:---:|:---:|:---|:---|\n")
        for sc in serialized_claims:
            p_scores = sc["scores"]
            mc_val = f"{p_scores['minicheck_probability']:.4f}" if p_scores['minicheck_probability'] is not None else "-"
            nli_val = f"{p_scores['nli_contradiction_prob']:.2f} / {p_scores['nli_entailment_prob']:.2f} / {p_scores['nli_neutral_prob']:.2f}" if p_scores['nli_contradiction_prob'] is not None else "-"
            cit = f"{sc['primary_evidence']['document_name']} P{sc['primary_evidence']['page_number']}" if sc['primary_evidence'] else "None"
            f.write(f"| `{sc['claim_id']}` | {sc['claim_text']} | **`{sc['classification']}`** | {mc_val} | {nli_val} | {cit} | {sc['reasoning_trace']} |\n")
        f.write("\n## 2. Factuality & Hallucination Metrics\n\n")
        f.write(f"- **Total Extracted Claims**: `{m.total_claims}`\n")
        f.write(f"- **Empirical Factual Claims (Denominator $N$)**: `{m.factual_claims}`\n")
        f.write(f"- **Non-Factual Claims Excluded**: `{m.non_factual_claims}`\n")
        f.write(f"- **Evidence Support Rate**: **{supp_str}** ({m.supported_count}/{m.factual_claims})\n")
        f.write(f"- **Confirmed Contradiction Rate**: **{contra_str}** ({m.contradicted_count}/{m.factual_claims})\n")
        f.write(f"- **Unsupported Rate**: **{unsup_str}** ({m.unsupported_count}/{m.factual_claims})\n")
        f.write(f"- **Unverifiable Rate**: **{unver_str}** ({m.unverifiable_count}/{m.factual_claims})\n")
        f.write(f"- **Combined Uncertainty Rate**: **{uncert_str}**\n\n")
        f.write(f"> *{m.denominator_description}*\n")
    print(f"[Saved] Markdown report to: {out_md}")


if __name__ == "__main__":
    run_pipeline_demo()

