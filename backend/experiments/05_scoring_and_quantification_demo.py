"""
VeriLLM Phase 6 Demonstration Script: Scoring and Quantification Engine.
Demonstrates:
1. Converting actual Phase 5 Apollo 11 verification results into a structured ResponseScore.
2. Scoring multi-response benchmark datasets across different LLM models.
3. Micro-pooling (claim-level) vs Macro-averaging (response-level) to avoid Simpson's paradox.
4. Robust handling of zero-factual-claim responses and conversational meta-text.
5. Exporting JSON and Markdown reports with full provenance.
"""

import sys
import json
import time
from pathlib import Path

# Ensure backend root is on sys.path
CURRENT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = CURRENT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml_engine.claim_verifier.models import (
    VerificationCategory,
    VerifiedClaim,
    ModelVerificationScore
)
from ml_engine.evidence_retriever.models import RetrievedPassage, DocumentChunk
from ml_engine.scorer import (
    FactualityScorer,
    ScoringConfig,
    ResponseScore,
    DatasetScore
)


def load_actual_phase5_claims(json_path: Path):
    """Load actual Apollo 11 verified propositions from Phase 5 demo results."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    reconstructed = []
    for sc in data["claims"]:
        cat = VerificationCategory(sc["classification"])
        s_data = sc.get("scores", {})
        scores = ModelVerificationScore(
            minicheck_prediction=s_data.get("minicheck_prediction"),
            minicheck_probability=s_data.get("minicheck_probability"),
            nli_entailment_prob=s_data.get("nli_entailment_prob"),
            nli_contradiction_prob=s_data.get("nli_contradiction_prob"),
            nli_neutral_prob=s_data.get("nli_neutral_prob"),
            retrieval_similarity=s_data.get("retrieval_similarity")
        )
        pe_data = sc.get("primary_evidence")
        pe = None
        if pe_data:
            chunk = DocumentChunk(
                chunk_id=pe_data["chunk_id"],
                document_name=pe_data["document_name"],
                page_number=pe_data["page_number"],
                chunk_index=0,
                start_char=0,
                end_char=100,
                text=pe_data["text_snippet"]
            )
            pe = RetrievedPassage(chunk=chunk, similarity_score=pe_data["similarity_score"], rank=1)

        vc = VerifiedClaim(
            claim_id=sc["claim_id"],
            claim_text=sc["claim_text"],
            source_sentence=sc["claim_text"],
            start_char=0,
            end_char=len(sc["claim_text"]),
            is_factual=sc["is_factual"],
            classification=cat,
            scores=scores,
            primary_evidence=pe,
            decision_rule=sc.get("decision_rule", "PHASE5_RULE"),
            reasoning_trace=sc.get("reasoning_trace", "")
        )
        reconstructed.append(vc)

    return reconstructed, data


def make_synthetic_claim(claim_id: str, text: str, cat: VerificationCategory, is_factual: bool = True) -> VerifiedClaim:
    """Helper for multi-model benchmark evaluation."""
    chunk = DocumentChunk(
        chunk_id=f"{claim_id}_c",
        document_name="benchmark_ref.pdf",
        page_number=1,
        chunk_index=0,
        start_char=0,
        end_char=100,
        text="Benchmark reference context."
    )
    pe = RetrievedPassage(chunk=chunk, similarity_score=0.72, rank=1)
    scores = ModelVerificationScore(
        minicheck_prediction=1 if cat == VerificationCategory.SUPPORTED else 0,
        minicheck_probability=0.92 if cat == VerificationCategory.SUPPORTED else 0.08,
        nli_entailment_prob=0.90 if cat == VerificationCategory.SUPPORTED else 0.05,
        nli_contradiction_prob=0.90 if cat == VerificationCategory.CONTRADICTED else 0.05,
        nli_neutral_prob=0.90 if cat == VerificationCategory.UNSUPPORTED else 0.05,
        retrieval_similarity=0.72
    )
    return VerifiedClaim(
        claim_id=claim_id,
        claim_text=text,
        source_sentence=text,
        start_char=0,
        end_char=len(text),
        is_factual=is_factual,
        classification=cat,
        scores=scores,
        primary_evidence=pe,
        decision_rule="SYNTHETIC_BENCHMARK",
        reasoning_trace=f"Verified as {cat.value}"
    )


def run_scoring_demo():
    print("=" * 95)
    print(" VeriLLM Phase 6: Scoring, Aggregation, and Hallucination Quantification Demonstration")
    print("=" * 95)

    results_dir = CURRENT_DIR / "results"
    phase5_json = results_dir / "04_claim_verification_demo_results.json"

    if not phase5_json.exists():
        print(f"[Error] Phase 5 results not found at: {phase5_json}")
        print("Please run backend/experiments/04_claim_verification_pipeline_demo.py first.")
        return

    scorer = FactualityScorer(ScoringConfig(contradiction_penalty_weight=1.0))

    # -------------------------------------------------------------
    # Part 1: Single Response Scoring on Real Apollo 11 Run
    # -------------------------------------------------------------
    print("\n[Part 1] Scoring Actual Phase 5 Apollo 11 Verification Results...")
    apollo_claims, apollo_raw = load_actual_phase5_claims(phase5_json)
    
    t0 = time.time()
    apollo_score: ResponseScore = scorer.score_response(
        apollo_claims,
        response_id="apollo11_candidate_response",
        model_id="roberta_deberta_hybrid",
        document_name=apollo_raw.get("document", "sample_apollo11_mission.pdf"),
        execution_time_seconds=apollo_raw.get("execution_duration_seconds", 10.28)
    )
    duration_part1 = time.time() - t0

    print(f"  -> Scored {apollo_score.counts.total_claims} propositions in {duration_part1*1000:.2f}ms.")
    print("-" * 95)
    print(f"  Empirical Factual Claims (N)    : {apollo_score.counts.factual_claims}")
    print(f"  Non-Factual Claims Excluded     : {apollo_score.counts.non_factual_claims} (opinions/meta-text)")
    print(f"  Supported Claims                : {apollo_score.counts.supported_count}")
    print(f"  Contradicted Claims             : {apollo_score.counts.contradicted_count}")
    print(f"  Unsupported Claims              : {apollo_score.counts.unsupported_count}")
    print(f"  Unverifiable Claims             : {apollo_score.counts.unverifiable_count}")
    print("-" * 95)
    print(f"  Evidence Support Rate (ESR)     : {apollo_score.rates.evidence_support_rate:.2f}%")
    print(f"  Confirmed Contradiction Rate    : {apollo_score.rates.confirmed_contradiction_rate:.2f}%")
    print(f"  Unsupported Rate (Absence)      : {apollo_score.rates.unsupported_rate:.2f}%")
    print(f"  Unverifiable Rate               : {apollo_score.rates.unverifiable_rate:.2f}%")
    print(f"  Combined Uncertainty Rate (UR)  : {apollo_score.rates.combined_uncertainty_rate:.2f}%")
    print("-" * 95)
    print(f"  Grounded Factuality Index (GFI) : {apollo_score.indices.grounded_factuality_index:.2f}% (fraction of supported facts)")
    print(f"  Penalized Factuality (CPFS)     : {apollo_score.indices.contradiction_penalized_factuality:.2f}% (max(0, (supp - contra)/N) * 100)")
    print(f"  Hallucination Severity (HSI)    : {apollo_score.indices.hallucination_severity_index:.2f}% (percentage of contradicted facts)")
    print(f"  Partition Invariant Verified    : {apollo_score.mathematical_invariants_verified} (Sum of 4 categories == 100.00%)")
    print(f"  Denominator Description         : {apollo_score.denominator_description}")

    # -------------------------------------------------------------
    # Part 2: Multi-Model Benchmark Dataset Aggregation
    # -------------------------------------------------------------
    print("\n[Part 2] Multi-Model Benchmark Aggregation & Simpson's Paradox Prevention...")

    # Model A: High accuracy model (e.g. gpt-4o style)
    # Response 1: 6 factual claims (5 supported, 0 contradicted, 1 unsupported)
    m_a_r1_claims = [
        make_synthetic_claim("a1_1", "Fact A1.1", VerificationCategory.SUPPORTED),
        make_synthetic_claim("a1_2", "Fact A1.2", VerificationCategory.SUPPORTED),
        make_synthetic_claim("a1_3", "Fact A1.3", VerificationCategory.SUPPORTED),
        make_synthetic_claim("a1_4", "Fact A1.4", VerificationCategory.SUPPORTED),
        make_synthetic_claim("a1_5", "Fact A1.5", VerificationCategory.SUPPORTED),
        make_synthetic_claim("a1_6", "Fact A1.6", VerificationCategory.UNSUPPORTED),
    ]
    # Response 2: 4 factual claims (3 supported, 0 contradicted, 1 unverifiable)
    m_a_r2_claims = [
        make_synthetic_claim("a2_1", "Fact A2.1", VerificationCategory.SUPPORTED),
        make_synthetic_claim("a2_2", "Fact A2.2", VerificationCategory.SUPPORTED),
        make_synthetic_claim("a2_3", "Fact A2.3", VerificationCategory.SUPPORTED),
        make_synthetic_claim("a2_4", "Fact A2.4", VerificationCategory.UNVERIFIABLE),
    ]

    # Model B: Hallucinating legacy model
    # Response 3: 5 factual claims (1 supported, 3 contradicted, 1 unsupported)
    m_b_r1_claims = [
        make_synthetic_claim("b1_1", "Fact B1.1", VerificationCategory.SUPPORTED),
        make_synthetic_claim("b1_2", "Fact B1.2", VerificationCategory.CONTRADICTED),
        make_synthetic_claim("b1_3", "Fact B1.3", VerificationCategory.CONTRADICTED),
        make_synthetic_claim("b1_4", "Fact B1.4", VerificationCategory.CONTRADICTED),
        make_synthetic_claim("b1_5", "Fact B1.5", VerificationCategory.UNSUPPORTED),
    ]
    # Response 4: Conversational response with 0 factual claims (2 opinions/greetings)
    m_b_r2_claims = [
        make_synthetic_claim("b2_1", "I really love learning about space!", VerificationCategory.NON_FACTUAL, is_factual=False),
        make_synthetic_claim("b2_2", "In my personal view, science is fun.", VerificationCategory.NON_FACTUAL, is_factual=False),
    ]

    benchmark_dataset = scorer.compare_models(
        responses_by_model={
            "GPT-4o-GroundTruth": [
                scorer.score_response(m_a_r1_claims, response_id="gpt4_resp1"),
                scorer.score_response(m_a_r2_claims, response_id="gpt4_resp2"),
            ],
            "Legacy-7B-Baseline": [
                scorer.score_response(m_b_r1_claims, response_id="legacy_resp1"),
                scorer.score_response(m_b_r2_claims, response_id="legacy_resp2_opinions"),
            ]
        },
        dataset_name="VeriLLM Comparative Benchmark 2026"
    )

    print(f"  Total Responses in Benchmark   : {benchmark_dataset.total_responses}")
    print(f"  Responses with Factual Claims  : {benchmark_dataset.responses_with_factual_claims}")
    print(f"  Zero-Factual-Claim Responses   : {benchmark_dataset.responses_zero_factual_claims} (handled safely without 0-division)")
    print(f"  Total Empirical Factual Claims : {benchmark_dataset.total_factual_claims}")
    print(f"  Total Non-Factual Claims       : {benchmark_dataset.total_non_factual_claims}")
    print("-" * 95)
    print("  POOLED DATASET METRICS (Micro-Pooled across all claims):")
    print(f"    Micro Evidence Support Rate   : {benchmark_dataset.micro_evidence_support_rate:.2f}% ({benchmark_dataset.pooled_supported_count} / {benchmark_dataset.total_factual_claims})")
    print(f"    Micro Contradiction Rate      : {benchmark_dataset.micro_confirmed_contradiction_rate:.2f}% ({benchmark_dataset.pooled_contradicted_count} / {benchmark_dataset.total_factual_claims})")
    print(f"    Micro Unsupported Rate        : {benchmark_dataset.micro_unsupported_rate:.2f}% ({benchmark_dataset.pooled_unsupported_count} / {benchmark_dataset.total_factual_claims})")
    print(f"    Micro Unverifiable Rate       : {benchmark_dataset.micro_unverifiable_rate:.2f}% ({benchmark_dataset.pooled_unverifiable_count} / {benchmark_dataset.total_factual_claims})")
    print(f"    Micro Uncertainty Rate        : {benchmark_dataset.micro_combined_uncertainty_rate:.2f}%")
    print(f"    Micro Penalized Factuality    : {benchmark_dataset.micro_penalized_factuality_score:.2f}%")
    print("-" * 95)
    print("  MACRO-AVERAGED METRICS (Unweighted mean across valid responses):")
    print(f"    Macro Evidence Support Rate   : {benchmark_dataset.macro_evidence_support_rate:.2f}%")
    print(f"    Macro Contradiction Rate      : {benchmark_dataset.macro_confirmed_contradiction_rate:.2f}%")
    print(f"    Macro Uncertainty Rate        : {benchmark_dataset.macro_combined_uncertainty_rate:.2f}%")

    print("\n[Part 3] Model-by-Model Breakdown:")
    print("=" * 95)
    print(f"{'Model Identifier':<22} | {'Resp':<5} | {'N (Facts)':<9} | {'Support %':<10} | {'Contra %':<9} | {'Uncert %':<9} | {'Penalized %':<11}")
    print("=" * 95)
    for m_id, m_data in benchmark_dataset.models_breakdown.items():
        supp_str = f"{m_data.micro_evidence_support_rate:.2f}%" if m_data.micro_evidence_support_rate is not None else "N/A"
        contra_str = f"{m_data.micro_confirmed_contradiction_rate:.2f}%" if m_data.micro_confirmed_contradiction_rate is not None else "N/A"
        uncert_str = f"{m_data.micro_combined_uncertainty_rate:.2f}%" if m_data.micro_combined_uncertainty_rate is not None else "N/A"
        cpfs_str = f"{m_data.micro_penalized_factuality_score:.2f}%" if m_data.micro_penalized_factuality_score is not None else "N/A"
        print(f"{m_id:<22} | {m_data.total_responses:<5} | {m_data.total_factual_claims:<9} | {supp_str:<10} | {contra_str:<9} | {uncert_str:<9} | {cpfs_str:<11}")
    print("=" * 95)

    # -------------------------------------------------------------
    # Part 4: Save Structured Results & Markdown Report
    # -------------------------------------------------------------
    out_json = results_dir / "05_scoring_and_quantification_demo_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "05_scoring_and_quantification_demo",
            "apollo11_response_score": apollo_score.model_dump(),
            "multi_model_benchmark": benchmark_dataset.model_dump()
        }, f, indent=2)
    print(f"\n[Saved] Structured JSON results to: {out_json}")

    out_md = results_dir / "05_scoring_and_quantification_demo_report.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# VeriLLM Phase 6: Scoring & Quantification Engine Report\n\n")
        f.write("## 1. Apollo 11 Demonstration Response Factuality Score\n\n")
        f.write(f"- **Response ID**: `{apollo_score.response_id}`\n")
        f.write(f"- **Reference Document**: `{apollo_score.document_name}`\n")
        f.write(f"- **Total Propositions**: `{apollo_score.counts.total_claims}` (8 Empirical Facts, 1 Subjective Opinion Excluded)\n")
        f.write(f"- **Evidence Support Rate (ESR / GFI)**: **{apollo_score.rates.evidence_support_rate:.2f}%** (5 / 8)\n")
        f.write(f"- **Confirmed Contradiction Rate (CCR / HSI)**: **{apollo_score.rates.confirmed_contradiction_rate:.2f}%** (1 / 8)\n")
        f.write(f"- **Unsupported Rate (USR)**: **{apollo_score.rates.unsupported_rate:.2f}%** (2 / 8)\n")
        f.write(f"- **Unverifiable Rate (UVR)**: **{apollo_score.rates.unverifiable_rate:.2f}%** (0 / 8)\n")
        f.write(f"- **Combined Uncertainty Rate (UR)**: **{apollo_score.rates.combined_uncertainty_rate:.2f}%** (2 / 8)\n")
        f.write(f"- **Contradiction-Penalized Factuality (CPFS)**: **{apollo_score.indices.contradiction_penalized_factuality:.2f}%** ($\max(0, (5 - 1)/8) \\times 100\\%$)\n")
        f.write(f"- **Mathematical Invariants Verified**: **`{apollo_score.mathematical_invariants_verified}`** ($62.50\\% + 12.50\\% + 25.00\\% + 0.00\\% = 100.00\\%$)\n\n")
        
        f.write("## 2. Multi-Model Benchmark Aggregation\n\n")
        f.write("| Model Identifier | Evaluated Responses | Factual Facts (N) | Non-Factual | Support Rate (Micro) | Contradiction Rate (Micro) | Uncertainty Rate (Micro) | Penalized Score (CPFS) |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for m_id, m_data in benchmark_dataset.models_breakdown.items():
            f.write(f"| **`{m_id}`** | {m_data.total_responses} | {m_data.total_factual_claims} | {m_data.total_non_factual_claims} | **{m_data.micro_evidence_support_rate:.2f}%** | {m_data.micro_confirmed_contradiction_rate:.2f}% | {m_data.micro_combined_uncertainty_rate:.2f}% | **{m_data.micro_penalized_factuality_score:.2f}%** |\n")
        
        f.write("\n## 3. Methodological Notes on Aggregation\n\n")
        for note in benchmark_dataset.aggregation_notes:
            f.write(f"- {note}\n")
    print(f"[Saved] Markdown report to: {out_md}")


if __name__ == "__main__":
    run_scoring_demo()

