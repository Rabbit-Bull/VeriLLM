"""
VeriLLM Unit Test Suite: Scoring and Quantification Engine.
Tests response-level scores, dataset-level micro/macro aggregations,
prevention of Simpson's paradox, zero-claim edge cases, mathematical invariants,
and actual Phase 5 Apollo 11 demonstration outputs.
"""

import sys
import json
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = CURRENT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml_engine.claim_verifier.models import (
    VerificationCategory,
    VerifiedClaim,
    ModelVerificationScore,
    VerificationReport,
    VerificationMetrics
)
from ml_engine.evidence_retriever.models import RetrievedPassage, DocumentChunk
from ml_engine.scorer import (
    FactualityScorer,
    ResponseScorer,
    DatasetScorer,
    ScoringConfig,
    ResponseScore,
    DatasetScore
)


def _make_dummy_claim(
    claim_id: str,
    text: str,
    category: VerificationCategory,
    is_factual: bool = True
) -> VerifiedClaim:
    """Helper to construct a VerifiedClaim without invoking heavy ML models."""
    chunk = DocumentChunk(
        chunk_id=f"{claim_id}_chunk",
        document_name="test_doc.pdf",
        page_number=1,
        chunk_index=0,
        start_char=0,
        end_char=100,
        text="Sample test evidence."
    )
    passage = RetrievedPassage(chunk=chunk, similarity_score=0.75, rank=1)
    scores = ModelVerificationScore(
        minicheck_prediction=1 if category == VerificationCategory.SUPPORTED else 0,
        minicheck_probability=0.95 if category == VerificationCategory.SUPPORTED else 0.05,
        nli_entailment_prob=0.90 if category == VerificationCategory.SUPPORTED else 0.05,
        nli_contradiction_prob=0.90 if category == VerificationCategory.CONTRADICTED else 0.05,
        nli_neutral_prob=0.90 if category == VerificationCategory.UNSUPPORTED else 0.05,
        retrieval_similarity=0.75
    )
    return VerifiedClaim(
        claim_id=claim_id,
        claim_text=text,
        source_sentence=text,
        start_char=0,
        end_char=len(text),
        is_factual=is_factual,
        classification=category,
        scores=scores,
        primary_evidence=passage,
        decision_rule="TEST_RULE",
        reasoning_trace=f"Test trace for {category.value}"
    )


def test_single_response_scoring_basic():
    """Verify standard 4-category calculation, indices (GFI, CPFS, HSI), and invariants."""
    claims = [
        _make_dummy_claim("c1", "Claim one is true.", VerificationCategory.SUPPORTED),
        _make_dummy_claim("c2", "Claim two is true.", VerificationCategory.SUPPORTED),
        _make_dummy_claim("c3", "Claim three is false.", VerificationCategory.CONTRADICTED),
        _make_dummy_claim("c4", "Claim four is unmentioned.", VerificationCategory.UNSUPPORTED),
    ]

    scorer = FactualityScorer()
    res: ResponseScore = scorer.score_response(claims, response_id="resp_basic", model_id="test_llm")

    assert res.response_id == "resp_basic"
    assert res.model_id == "test_llm"
    assert res.has_factual_claims is True
    assert res.is_empty_response is False

    # Counts
    assert res.counts.total_claims == 4
    assert res.counts.factual_claims == 4
    assert res.counts.supported_count == 2
    assert res.counts.contradicted_count == 1
    assert res.counts.unsupported_count == 1
    assert res.counts.unverifiable_count == 0

    # Rates
    assert res.rates.evidence_support_rate == 50.00
    assert res.rates.confirmed_contradiction_rate == 25.00
    assert res.rates.unsupported_rate == 25.00
    assert res.rates.unverifiable_rate == 0.00
    assert res.rates.combined_uncertainty_rate == 25.00

    # Overall Indices
    assert res.indices.grounded_factuality_index == 50.00  # GFI = ESR
    # CPFS: (2 - 1*1)/4 * 100 = 25.00%
    assert res.indices.contradiction_penalized_factuality == 25.00
    assert res.indices.hallucination_severity_index == 25.00

    # Partition Invariant
    assert res.mathematical_invariants_verified is True
    assert round(res.rates.evidence_support_rate + res.rates.confirmed_contradiction_rate + res.rates.unsupported_rate + res.rates.unverifiable_rate, 2) == 100.00


def test_single_response_with_non_factual_and_opinions():
    """Ensure non-factual statements are strictly excluded from rates and denominators."""
    claims = [
        _make_dummy_claim("c1", "Paris is in France.", VerificationCategory.SUPPORTED, is_factual=True),
        _make_dummy_claim("c2", "Paris is in Germany.", VerificationCategory.CONTRADICTED, is_factual=True),
        _make_dummy_claim("c3", "Paris was founded in 5000 BC.", VerificationCategory.UNVERIFIABLE, is_factual=True),
        _make_dummy_claim("c4", "Paris is the loveliest city in the world.", VerificationCategory.NON_FACTUAL, is_factual=False),
        _make_dummy_claim("c5", "Hello there!", VerificationCategory.NON_FACTUAL, is_factual=False),
    ]

    scorer = FactualityScorer()
    res = scorer.score_response(claims, response_id="resp_opinions")

    assert res.counts.total_claims == 5
    assert res.counts.factual_claims == 3
    assert res.counts.non_factual_claims == 2
    assert res.counts.non_factual_count == 2

    # 1/3 = 33.33% each
    assert res.rates.evidence_support_rate == 33.33
    assert res.rates.confirmed_contradiction_rate == 33.33
    assert res.rates.unverifiable_rate == 33.33
    assert res.rates.unsupported_rate == 0.00
    assert res.rates.combined_uncertainty_rate == 33.33

    # CPFS: max(0, (1 - 1*1)/3) = 0.00%
    assert res.indices.contradiction_penalized_factuality == 0.00
    assert res.mathematical_invariants_verified is True


def test_single_response_zero_factual_claims():
    """Verify clean N=0 behavior with None rates and no zero division."""
    claims = [
        _make_dummy_claim("c1", "I love this book.", VerificationCategory.NON_FACTUAL, is_factual=False),
        _make_dummy_claim("c2", "In my opinion, AI is amazing.", VerificationCategory.NON_FACTUAL, is_factual=False),
    ]

    scorer = FactualityScorer()
    res = scorer.score_response(claims, response_id="resp_zero_facts")

    assert res.has_factual_claims is False
    assert res.is_empty_response is False
    assert res.counts.total_claims == 2
    assert res.counts.factual_claims == 0
    assert res.counts.non_factual_claims == 2

    assert res.rates.evidence_support_rate is None
    assert res.rates.confirmed_contradiction_rate is None
    assert res.rates.unsupported_rate is None
    assert res.rates.unverifiable_rate is None
    assert res.rates.combined_uncertainty_rate is None

    assert res.indices.grounded_factuality_index is None
    assert res.indices.contradiction_penalized_factuality is None
    assert res.indices.hallucination_severity_index is None
    assert res.mathematical_invariants_verified is True


def test_single_response_empty_input():
    """Verify response with zero claims handles gracefully."""
    scorer = FactualityScorer()
    res = scorer.score_response([], response_id="resp_empty")

    assert res.is_empty_response is True
    assert res.has_factual_claims is False
    assert res.counts.total_claims == 0
    assert res.counts.factual_claims == 0
    assert res.rates.evidence_support_rate is None


def test_duplicate_claims_detection_and_deduplication():
    """Verify proposition normalization, duplicate counting, and deduplication configuration."""
    claims = [
        _make_dummy_claim("c1", "Water boils at 100 degrees Celsius.", VerificationCategory.SUPPORTED),
        _make_dummy_claim("c2", "water boils at 100 degrees celsius.", VerificationCategory.SUPPORTED),  # Duplicate
        _make_dummy_claim("c3", "Water freezes at 0 degrees Celsius.", VerificationCategory.SUPPORTED),
    ]

    # Without deduplication (default)
    scorer_default = FactualityScorer(ScoringConfig(deduplicate_claims=False))
    res1 = scorer_default.score_response(claims)
    assert res1.contains_duplicates is True
    assert res1.counts.duplicate_claims_count == 1
    assert res1.counts.factual_claims == 3
    assert res1.rates.evidence_support_rate == 100.00

    # With deduplication enabled
    scorer_dedup = FactualityScorer(ScoringConfig(deduplicate_claims=True))
    res2 = scorer_dedup.score_response(claims)
    assert res2.contains_duplicates is True
    assert res2.counts.duplicate_claims_count == 1
    assert res2.counts.factual_claims == 2
    assert res2.rates.evidence_support_rate == 100.00


def test_contradiction_penalty_weight_parameterization():
    """Verify that CPFS respects lambda penalty weight and bounds at 0.0."""
    claims = [
        _make_dummy_claim("c1", "Fact 1", VerificationCategory.SUPPORTED),
        _make_dummy_claim("c2", "Fact 2", VerificationCategory.SUPPORTED),
        _make_dummy_claim("c3", "Fact 3", VerificationCategory.SUPPORTED),
        _make_dummy_claim("c4", "Fact 4", VerificationCategory.CONTRADICTED),
    ]

    # lambda = 1.0 -> (3 - 1)/4 = 50.00%
    s1 = FactualityScorer(ScoringConfig(contradiction_penalty_weight=1.0)).score_response(claims)
    assert s1.indices.contradiction_penalized_factuality == 50.00

    # lambda = 2.0 -> (3 - 2)/4 = 25.00%
    s2 = FactualityScorer(ScoringConfig(contradiction_penalty_weight=2.0)).score_response(claims)
    assert s2.indices.contradiction_penalized_factuality == 25.00

    # lambda = 3.0 -> (3 - 3)/4 = 0.00%
    s3 = FactualityScorer(ScoringConfig(contradiction_penalty_weight=3.0)).score_response(claims)
    assert s3.indices.contradiction_penalized_factuality == 0.00

    # lambda = 5.0 -> max(0, (3 - 5)/4) = 0.00% (lower bounded at 0.0)
    s4 = FactualityScorer(ScoringConfig(contradiction_penalty_weight=5.0)).score_response(claims)
    assert s4.indices.contradiction_penalized_factuality == 0.00


def test_multi_response_micro_vs_macro_aggregation_simpsons_paradox():
    """
    Directly test prevention of Simpson's paradox when aggregating across responses
    with unequal factual claim denominators.
    """
    scorer = FactualityScorer()

    # Response A: 1 claim, 1 Supported (100% support rate)
    resp_a_claims = [
        _make_dummy_claim("a1", "Claim A1", VerificationCategory.SUPPORTED)
    ]
    resp_a = scorer.score_response(resp_a_claims, response_id="resp_a", model_id="llm_x")
    assert resp_a.rates.evidence_support_rate == 100.00

    # Response B: 9 claims, 1 Supported, 8 Contradicted (11.11% support rate)
    resp_b_claims = [
        _make_dummy_claim("b1", "Claim B1", VerificationCategory.SUPPORTED)
    ] + [
        _make_dummy_claim(f"b{i}", f"Claim B{i}", VerificationCategory.CONTRADICTED)
        for i in range(2, 10)
    ]
    resp_b = scorer.score_response(resp_b_claims, response_id="resp_b", model_id="llm_x")
    assert resp_b.rates.evidence_support_rate == 11.11

    # Aggregate dataset
    dataset_score = scorer.score_dataset([resp_a, resp_b], dataset_name="Simpsons Paradox Test")

    # Macro-average = (100.00 + 11.11) / 2 = ~55.55%
    assert abs(dataset_score.macro_evidence_support_rate - 55.55) <= 0.05

    # Micro-pooled = (1 + 1) / (1 + 9) = 2 / 10 = 20.00%
    # This proves that Micro-pooling avoids the fallacy of falsely reporting 55.56% when 8 out of 10 claims are contradicted!
    assert dataset_score.micro_evidence_support_rate == 20.00
    assert dataset_score.pooled_supported_count == 2
    assert dataset_score.pooled_contradicted_count == 8
    assert dataset_score.total_factual_claims == 10
    assert dataset_score.mathematical_invariants_verified is True


def test_dataset_scoring_with_mixed_and_zero_claim_responses():
    """Verify dataset aggregation safely handles zero-claim and non-factual responses."""
    scorer = FactualityScorer()

    # Resp 1: 2 supported, 2 contradicted
    r1 = scorer.score_response([
        _make_dummy_claim("r1_c1", "Text 1", VerificationCategory.SUPPORTED),
        _make_dummy_claim("r1_c2", "Text 2", VerificationCategory.SUPPORTED),
        _make_dummy_claim("r1_c3", "Text 3", VerificationCategory.CONTRADICTED),
        _make_dummy_claim("r1_c4", "Text 4", VerificationCategory.CONTRADICTED),
    ], response_id="r1")

    # Resp 2: Opinion only (N = 0)
    r2 = scorer.score_response([
        _make_dummy_claim("r2_c1", "I love space.", VerificationCategory.NON_FACTUAL, is_factual=False),
    ], response_id="r2")

    # Resp 3: Empty response
    r3 = scorer.score_response([], response_id="r3")

    ds = scorer.score_dataset([r1, r2, r3], dataset_name="Mixed Dataset")

    assert ds.total_responses == 3
    assert ds.responses_with_factual_claims == 1
    assert ds.responses_zero_factual_claims == 2
    assert ds.empty_responses == 1
    assert ds.total_factual_claims == 4
    assert ds.total_non_factual_claims == 1

    # Since r1 is the only response with factual claims, micro and macro should equal r1's rates
    assert ds.micro_evidence_support_rate == 50.00
    assert ds.micro_confirmed_contradiction_rate == 50.00
    assert ds.macro_evidence_support_rate == 50.00
    assert ds.macro_confirmed_contradiction_rate == 50.00
    assert ds.mathematical_invariants_verified is True


def test_cross_model_comparison():
    """Verify comparative evaluation between different generative LLM models."""
    scorer = FactualityScorer()

    # Model A: 2 responses, highly factual
    m_a_r1 = [_make_dummy_claim("a1", "Fact A1", VerificationCategory.SUPPORTED)]
    m_a_r2 = [_make_dummy_claim("a2", "Fact A2", VerificationCategory.SUPPORTED)]

    # Model B: 2 responses, hallucinating
    m_b_r1 = [_make_dummy_claim("b1", "Fact B1", VerificationCategory.CONTRADICTED)]
    m_b_r2 = [_make_dummy_claim("b2", "Fact B2", VerificationCategory.UNSUPPORTED)]

    comparison = scorer.compare_models({
        "gpt-4o": [scorer.score_response(m_a_r1), scorer.score_response(m_a_r2)],
        "legacy-llm": [scorer.score_response(m_b_r1), scorer.score_response(m_b_r2)]
    })

    assert "gpt-4o" in comparison.models_breakdown
    assert "legacy-llm" in comparison.models_breakdown

    gpt4_metrics = comparison.models_breakdown["gpt-4o"]
    assert gpt4_metrics.total_responses == 2
    assert gpt4_metrics.micro_evidence_support_rate == 100.00
    assert gpt4_metrics.micro_confirmed_contradiction_rate == 0.00

    legacy_metrics = comparison.models_breakdown["legacy-llm"]
    assert legacy_metrics.total_responses == 2
    assert legacy_metrics.micro_evidence_support_rate == 0.00
    assert legacy_metrics.micro_confirmed_contradiction_rate == 50.00
    assert legacy_metrics.micro_unsupported_rate == 50.00


def test_phase5_actual_apollo11_integration():
    """Verify scoring engine against actual Phase 5 Apollo 11 demonstration outputs."""
    results_path = BACKEND_DIR / "experiments" / "results" / "04_claim_verification_demo_results.json"
    assert results_path.exists(), f"Phase 5 results missing at: {results_path}"

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Reconstruct VerifiedClaim objects from actual Phase 5 demo output
    reconstructed_claims = []
    for sc in data["claims"]:
        cat_enum = VerificationCategory(sc["classification"])
        scores_data = sc.get("scores", {})
        scores = ModelVerificationScore(
            minicheck_prediction=scores_data.get("minicheck_prediction"),
            minicheck_probability=scores_data.get("minicheck_probability"),
            nli_entailment_prob=scores_data.get("nli_entailment_prob"),
            nli_contradiction_prob=scores_data.get("nli_contradiction_prob"),
            nli_neutral_prob=scores_data.get("nli_neutral_prob"),
            retrieval_similarity=scores_data.get("retrieval_similarity")
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
            classification=cat_enum,
            scores=scores,
            primary_evidence=pe,
            decision_rule=sc.get("decision_rule", "PHASE5_RULE"),
            reasoning_trace=sc.get("reasoning_trace", "")
        )
        reconstructed_claims.append(vc)

    scorer = FactualityScorer()
    score = scorer.score_response(
        reconstructed_claims,
        response_id="apollo11_demo",
        model_id="candidate_apollo11_llm",
        document_name=data.get("document", "sample_apollo11_mission.pdf"),
        execution_time_seconds=data.get("execution_duration_seconds", 10.28)
    )

    # Exact metrics matching Phase 5
    assert score.counts.total_claims == 9
    assert score.counts.factual_claims == 8
    assert score.counts.non_factual_claims == 1
    assert score.counts.supported_count == 5
    assert score.counts.contradicted_count == 1
    assert score.counts.unsupported_count == 2
    assert score.counts.unverifiable_count == 0

    assert score.rates.evidence_support_rate == 62.50
    assert score.rates.confirmed_contradiction_rate == 12.50
    assert score.rates.unsupported_rate == 25.00
    assert score.rates.unverifiable_rate == 0.00
    assert score.rates.combined_uncertainty_rate == 25.00

    # Grounded Factuality Index = 62.50%
    assert score.indices.grounded_factuality_index == 62.50
    # Contradiction-Penalized Factuality = (5 - 1)/8 * 100 = 50.00%
    assert score.indices.contradiction_penalized_factuality == 50.00
    # Hallucination Severity Index = 12.50%
    assert score.indices.hallucination_severity_index == 12.50

    # Partition Invariant
    assert score.mathematical_invariants_verified is True


def test_json_serialization_compatibility():
    """Verify that models serialize cleanly to Python dict and JSON string."""
    claims = [
        _make_dummy_claim("c1", "Claim one.", VerificationCategory.SUPPORTED),
        _make_dummy_claim("c2", "Claim two.", VerificationCategory.CONTRADICTED),
    ]
    scorer = FactualityScorer()
    res = scorer.score_response(claims, response_id="resp_json_test")

    # Serialize to dict and JSON
    res_dict = res.model_dump()
    res_json = res.model_dump_json()

    assert isinstance(res_dict, dict)
    assert isinstance(res_json, str)
    assert "grounded_factuality_index" in res_dict["indices"]
    assert "evidence_support_rate" in res_dict["rates"]
    assert res_dict["rates"]["evidence_support_rate"] == 50.00
