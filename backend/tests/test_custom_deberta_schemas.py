"""Focused schema tests for Phase 8 Step 2 Custom DeBERTa score extensions."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import pytest

CURRENT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = CURRENT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml_engine.claim_verifier.models import (
    ModelVerificationScore,
    VerificationCategory,
    VerifiedClaim,
)
from ml_engine.scorer.models import ClaimSummary
from ml_engine.scorer.response_scorer import ResponseScorer


def test_model_verification_score_without_new_fields() -> None:
    """Older callers omitting Custom DeBERTa fields keep working with None defaults."""
    empty_score = ModelVerificationScore()
    assert empty_score.minicheck_prediction is None
    assert empty_score.minicheck_probability is None
    assert empty_score.nli_entailment_prob is None
    assert empty_score.nli_contradiction_prob is None
    assert empty_score.nli_neutral_prob is None
    assert empty_score.retrieval_similarity is None
    assert empty_score.custom_deberta_supported_prob is None
    assert empty_score.custom_deberta_contradiction_prob is None
    assert empty_score.custom_deberta_unsupported_prob is None
    assert empty_score.custom_deberta_predicted_label is None

    legacy_score = ModelVerificationScore(
        minicheck_prediction=1,
        minicheck_probability=0.98,
        nli_entailment_prob=0.91,
        nli_contradiction_prob=0.02,
        nli_neutral_prob=0.07,
        retrieval_similarity=0.81,
    )
    assert legacy_score.minicheck_prediction == 1
    assert legacy_score.minicheck_probability == 0.98
    assert legacy_score.nli_entailment_prob == 0.91
    assert legacy_score.nli_contradiction_prob == 0.02
    assert legacy_score.nli_neutral_prob == 0.07
    assert legacy_score.retrieval_similarity == 0.81
    assert legacy_score.custom_deberta_supported_prob is None
    assert legacy_score.custom_deberta_contradiction_prob is None
    assert legacy_score.custom_deberta_unsupported_prob is None
    assert legacy_score.custom_deberta_predicted_label is None


def test_claim_summary_without_new_fields() -> None:
    """Legacy ClaimSummary instantiation succeeds with Custom DeBERTa fields defaulted to None."""
    summary = ClaimSummary(
        claim_id="c_001",
        claim_text="Apollo 11 landed on the Moon.",
        is_factual=True,
        classification="SUPPORTED",
        citation="apollo.pdf (Page 1, chunk: c1)",
        minicheck_probability=0.97,
        nli_entailment_prob=0.94,
        nli_contradiction_prob=0.01,
        nli_neutral_prob=0.05,
        reasoning_trace="Entailed by primary evidence.",
    )
    assert summary.citation == "apollo.pdf (Page 1, chunk: c1)"
    assert summary.minicheck_probability == 0.97
    assert summary.nli_entailment_prob == 0.94
    assert summary.nli_contradiction_prob == 0.01
    assert summary.nli_neutral_prob == 0.05
    assert summary.custom_deberta_supported_prob is None
    assert summary.custom_deberta_contradiction_prob is None
    assert summary.custom_deberta_unsupported_prob is None
    assert summary.custom_deberta_predicted_label is None


def test_both_models_accept_all_custom_deberta_fields() -> None:
    """Both ModelVerificationScore and ClaimSummary store all four Custom DeBERTa fields."""
    score = ModelVerificationScore(
        minicheck_prediction=1,
        minicheck_probability=0.95,
        nli_entailment_prob=0.88,
        nli_contradiction_prob=0.04,
        nli_neutral_prob=0.08,
        retrieval_similarity=0.77,
        custom_deberta_supported_prob=0.92,
        custom_deberta_contradiction_prob=0.03,
        custom_deberta_unsupported_prob=0.05,
        custom_deberta_predicted_label="SUPPORTED_PROXY",
    )
    assert score.custom_deberta_supported_prob == 0.92
    assert score.custom_deberta_contradiction_prob == 0.03
    assert score.custom_deberta_unsupported_prob == 0.05
    assert score.custom_deberta_predicted_label == "SUPPORTED_PROXY"

    summary = ClaimSummary(
        claim_id="c_002",
        claim_text="Neil Armstrong walked on the Moon.",
        is_factual=True,
        classification="SUPPORTED",
        minicheck_probability=0.95,
        nli_entailment_prob=0.88,
        nli_contradiction_prob=0.04,
        nli_neutral_prob=0.08,
        custom_deberta_supported_prob=0.92,
        custom_deberta_contradiction_prob=0.03,
        custom_deberta_unsupported_prob=0.05,
        custom_deberta_predicted_label="SUPPORTED_PROXY",
        reasoning_trace="Supported across signals.",
    )
    assert summary.custom_deberta_supported_prob == 0.92
    assert summary.custom_deberta_contradiction_prob == 0.03
    assert summary.custom_deberta_unsupported_prob == 0.05
    assert summary.custom_deberta_predicted_label == "SUPPORTED_PROXY"


def test_json_serialization_and_roundtrip_for_both_models() -> None:
    """JSON serialization includes exact keys and round-trips cleanly."""
    score = ModelVerificationScore(
        minicheck_prediction=0,
        minicheck_probability=0.04,
        nli_entailment_prob=0.01,
        nli_contradiction_prob=0.96,
        nli_neutral_prob=0.03,
        retrieval_similarity=0.69,
        custom_deberta_supported_prob=0.08,
        custom_deberta_contradiction_prob=0.87,
        custom_deberta_unsupported_prob=0.05,
        custom_deberta_predicted_label="CONTRADICTED",
    )
    score_json = score.model_dump_json()
    score_payload = json.loads(score_json)
    assert score_payload["custom_deberta_supported_prob"] == pytest.approx(0.08)
    assert score_payload["custom_deberta_contradiction_prob"] == pytest.approx(0.87)
    assert score_payload["custom_deberta_unsupported_prob"] == pytest.approx(0.05)
    assert score_payload["custom_deberta_predicted_label"] == "CONTRADICTED"
    assert ModelVerificationScore.model_validate_json(score_json) == score

    summary = ClaimSummary(
        claim_id="c_003",
        claim_text="Michael Collins walked on the Moon.",
        is_factual=True,
        classification="CONTRADICTED",
        minicheck_probability=0.04,
        nli_entailment_prob=0.01,
        nli_contradiction_prob=0.96,
        nli_neutral_prob=0.03,
        custom_deberta_supported_prob=0.08,
        custom_deberta_contradiction_prob=0.87,
        custom_deberta_unsupported_prob=0.05,
        custom_deberta_predicted_label="CONTRADICTED",
        reasoning_trace="Contradicted by Apollo evidence.",
    )
    summary_json = summary.model_dump_json()
    summary_payload = json.loads(summary_json)
    assert summary_payload["custom_deberta_supported_prob"] == pytest.approx(0.08)
    assert summary_payload["custom_deberta_contradiction_prob"] == pytest.approx(0.87)
    assert summary_payload["custom_deberta_unsupported_prob"] == pytest.approx(0.05)
    assert summary_payload["custom_deberta_predicted_label"] == "CONTRADICTED"
    assert ClaimSummary.model_validate_json(summary_json) == summary


def test_existing_defaults_categories_and_scorer_compatibility_unchanged() -> None:
    """Existing fields, categories, VerifiedClaim defaults, and ResponseScorer remain unchanged."""
    assert [c.value for c in VerificationCategory] == [
        "SUPPORTED",
        "CONTRADICTED",
        "UNSUPPORTED",
        "UNVERIFIABLE",
        "NON_FACTUAL",
    ]

    expected_score_fields = {
        "minicheck_prediction",
        "minicheck_probability",
        "nli_entailment_prob",
        "nli_contradiction_prob",
        "nli_neutral_prob",
        "retrieval_similarity",
        "custom_deberta_supported_prob",
        "custom_deberta_contradiction_prob",
        "custom_deberta_unsupported_prob",
        "custom_deberta_predicted_label",
    }
    assert set(ModelVerificationScore.model_fields.keys()) == expected_score_fields
    assert ModelVerificationScore.model_fields["custom_deberta_supported_prob"].annotation == Optional[float]
    assert ModelVerificationScore.model_fields["custom_deberta_contradiction_prob"].annotation == Optional[float]
    assert ModelVerificationScore.model_fields["custom_deberta_unsupported_prob"].annotation == Optional[float]
    assert ModelVerificationScore.model_fields["custom_deberta_predicted_label"].annotation == Optional[str]

    expected_summary_fields = {
        "claim_id",
        "claim_text",
        "is_factual",
        "classification",
        "citation",
        "minicheck_probability",
        "nli_entailment_prob",
        "nli_contradiction_prob",
        "nli_neutral_prob",
        "custom_deberta_supported_prob",
        "custom_deberta_contradiction_prob",
        "custom_deberta_unsupported_prob",
        "custom_deberta_predicted_label",
        "reasoning_trace",
    }
    assert set(ClaimSummary.model_fields.keys()) == expected_summary_fields

    vc = VerifiedClaim(
        claim_id="vc_1",
        claim_text="I think Apollo 11 was amazing.",
        source_sentence="I think Apollo 11 was amazing.",
        start_char=0,
        end_char=30,
        is_factual=False,
        classification=VerificationCategory.NON_FACTUAL,
        decision_rule="NON_FACTUAL_ASSERTION",
        reasoning_trace="Subjective remark.",
    )
    assert vc.offset_type == "EXACT"
    assert vc.scores == ModelVerificationScore()

    scored = ResponseScorer().score_response([vc], response_id="resp_schema_check")
    assert scored.counts.non_factual_count == 1
    assert scored.rates.evidence_support_rate is None
    assert scored.claims[0].custom_deberta_predicted_label is None

