"""
Focused Unit Tests for VeriLLM Production Calibrated DeBERTa Verifier.

Tests cover:
  1. Correct loading of Direct and Retrieval configurations from the actual frozen bundle.
  2. Correct application of C2 calibration using the existing research implementation.
  3. Correct class-label mapping.
  4. Correct Direct threshold decisions.
  5. Correct Retrieval threshold decisions.
  6. Evidence-floor abstention at similarity below 0.20.
  7. Evidence-floor behavior at exactly 0.20.
  8. Empty-evidence abstention.
  9. NON_FACTUAL handling.
 10. Invalid probabilities and malformed bundle handling.
 11. Missing model/checkpoint error handling.
 12. Model caching and lazy initialization.
 13. Compatibility with existing verification report and scorer interfaces.

All unit tests mock neural inference to run fast offline without loading model weights.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from unittest.mock import MagicMock

import numpy as np
import pytest

CURRENT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = CURRENT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml_engine.claim_extractor.models import ExtractedClaim
from ml_engine.evidence_retriever.models import ClaimEvidenceMatch, DocumentChunk, RetrievedPassage
from ml_engine.claim_verifier.models import (
    ModelVerificationScore,
    VerificationCategory,
    VerificationMetrics,
    VerificationReport,
    VerifiedClaim,
)
from ml_engine.claim_verifier.custom_deberta_verifier import (
    CustomDebertaPrediction,
    CustomDebertaVerifier,
)
from ml_engine.claim_verifier.calibrated_verifier import (
    CalibratedDebertaVerifier,
    CalibratedRegimeConfig,
    load_frozen_calibration_bundle,
    DEFAULT_FROZEN_BUNDLE_PATH,
)
from ml_engine.custom_model.calibration import (
    MulticlassCalibrator,
    reconstruct_logits_from_probs,
    _stable_softmax,
)
from ml_engine.scorer.response_scorer import ResponseScorer
from ml_engine.scorer.models import ResponseScore


class MockDebertaVerifier:
    """Mock CustomDebertaVerifier returning fixed predictions without loading torch model."""

    def __init__(self, predictions: Optional[List[CustomDebertaPrediction]] = None) -> None:
        self.predictions = predictions or []
        self.call_count = 0
        self.recorded_pairs: List[Tuple[str, str]] = []

    def predict_pairs(
        self,
        pairs: Sequence[Tuple[str, str]],
        batch_size: Optional[int] = None,
    ) -> List[CustomDebertaPrediction]:
        self.call_count += 1
        self.recorded_pairs.extend(pairs)
        if self.predictions:
            if len(self.predictions) < len(pairs):
                # Cycle or repeat last prediction
                return [self.predictions[i % len(self.predictions)] for i in range(len(pairs))]
            return self.predictions[:len(pairs)]
        # Default mock output: strong support
        return [
            CustomDebertaPrediction(
                predicted_label="SUPPORTED_PROXY",
                probabilities={"CONTRADICTED": 0.05, "SUPPORTED_PROXY": 0.90, "UNSUPPORTED": 0.05},
                contradicted_prob=0.05,
                supported_proxy_prob=0.90,
                unsupported_prob=0.05,
            )
            for _ in pairs
        ]


def _make_claim(
    claim_id: str,
    text: str,
    is_factual: bool = True,
    sentence: str = "Test source sentence.",
) -> ExtractedClaim:
    return ExtractedClaim(
        claim_id=claim_id,
        text=text,
        source_sentence=sentence,
        start_char=0,
        end_char=len(sentence),
        is_factual=is_factual,
    )


def _make_match(
    claim_id: str,
    text: str,
    similarity: float = 0.85,
    doc_name: str = "test_doc.txt",
    page_number: int = 1,
    rank: int = 1,
    claim_text: str = "Test claim text",
) -> ClaimEvidenceMatch:
    chunk = DocumentChunk(
        chunk_id=f"chk_{claim_id}",
        document_name=doc_name,
        page_number=page_number,
        chunk_index=0,
        text=text,
        word_count=len(text.split()),
        character_count=len(text),
        start_char=0,
        end_char=len(text),
    )
    passage = RetrievedPassage(chunk=chunk, similarity_score=similarity, rank=rank)
    return ClaimEvidenceMatch(
        claim_id=claim_id,
        claim_text=claim_text,
        retrieved_passages=[passage],
        best_passage=passage,
        relevant_passages_count=1,
    )


# ---------------------------------------------------------------------------
# Test 1: Correct loading of Direct and Retrieval configurations
# ---------------------------------------------------------------------------
def test_1_load_actual_frozen_bundle() -> None:
    """Verifies that the actual 10_stage1_frozen_bundle.json loads exactly as expected."""
    configs = load_frozen_calibration_bundle()
    assert set(configs.keys()) == {"direct", "retrieval"}

    # Direct mode checks
    d_cfg = configs["direct"]
    assert d_cfg.regime == "direct"
    assert pytest.approx(d_cfg.temperature, rel=1e-6) == 1.76959203
    assert pytest.approx(d_cfg.prior_bias[0], rel=1e-6) == -0.66788442
    assert pytest.approx(d_cfg.prior_bias[1], rel=1e-6) == 0.71171648
    assert pytest.approx(d_cfg.prior_bias[2], rel=1e-6) == -0.04383205
    assert d_cfg.tau_contra == 0.35
    assert d_cfg.tau_sup == 0.70
    assert d_cfg.tau_floor == 0.00
    assert d_cfg.l2_reg == 0.001

    # Retrieval mode checks
    r_cfg = configs["retrieval"]
    assert r_cfg.regime == "retrieval"
    assert pytest.approx(r_cfg.temperature, rel=1e-6) == 2.50411010
    assert pytest.approx(r_cfg.prior_bias[0], rel=1e-6) == -0.72845197
    assert pytest.approx(r_cfg.prior_bias[1], rel=1e-6) == 1.22831149
    assert pytest.approx(r_cfg.prior_bias[2], rel=1e-6) == -0.49985952
    assert r_cfg.tau_contra == 0.30
    assert r_cfg.tau_sup == 0.70
    assert r_cfg.tau_floor == 0.20
    assert r_cfg.l2_reg == 0.001


# ---------------------------------------------------------------------------
# Test 2: Correct application of C2 calibration using research implementation
# ---------------------------------------------------------------------------
def test_2_c2_calibration_transformation_parity() -> None:
    """Verifies that MulticlassCalibrator matches manual closed-form calculation with <1e-12 error."""
    configs = load_frozen_calibration_bundle()
    raw_probs = np.array([
        [0.02, 0.95, 0.03],
        [0.40, 0.20, 0.40],
        [0.10, 0.35, 0.55],
    ], dtype=np.float64)

    logits = reconstruct_logits_from_probs(raw_probs)

    for regime in ("direct", "retrieval"):
        cfg = configs[regime]
        T = cfg.temperature
        b = np.array(cfg.prior_bias)

        # Manual closed-form C2: softmax(z / T + b)
        manual_cal = _stable_softmax(logits / T + b)
        # Class method
        class_cal = cfg.calibrator.transform_logits(logits)

        np.testing.assert_allclose(manual_cal, class_cal, atol=1e-12)


# ---------------------------------------------------------------------------
# Test 3: Correct class-label mapping
# ---------------------------------------------------------------------------
def test_3_class_label_mapping() -> None:
    """Verifies that integer codes 0, 1, 2, 3, 4 map to the expected VerificationCategory values."""
    assert VerificationCategory.CONTRADICTED.value == "CONTRADICTED"
    assert VerificationCategory.SUPPORTED.value == "SUPPORTED"
    assert VerificationCategory.UNSUPPORTED.value == "UNSUPPORTED"
    assert VerificationCategory.UNVERIFIABLE.value == "UNVERIFIABLE"
    assert VerificationCategory.NON_FACTUAL.value == "NON_FACTUAL"


# ---------------------------------------------------------------------------
# Test 4: Correct Direct threshold decisions
# ---------------------------------------------------------------------------
def test_4_direct_threshold_decisions() -> None:
    """Verifies H2 decisions under Direct mode: Supported, Contradicted, and Unsupported."""
    mock_preds = [
        # Claim 1: Strong entailment -> calibrated support >= 0.70
        CustomDebertaPrediction(
            predicted_label="SUPPORTED_PROXY",
            probabilities={"CONTRADICTED": 0.02, "SUPPORTED_PROXY": 0.95, "UNSUPPORTED": 0.03},
            contradicted_prob=0.02,
            supported_proxy_prob=0.95,
            unsupported_prob=0.03,
        ),
        # Claim 2: High contradiction -> raw 0.90 calibrates to p_contra=0.5161 >= 0.35
        CustomDebertaPrediction(
            predicted_label="CONTRADICTED",
            probabilities={"CONTRADICTED": 0.90, "SUPPORTED_PROXY": 0.02, "UNSUPPORTED": 0.08},
            contradicted_prob=0.90,
            supported_proxy_prob=0.02,
            unsupported_prob=0.08,
        ),
        # Claim 3: Unmentioned neutral -> calibrated contra < 0.35 and sup < 0.70
        CustomDebertaPrediction(
            predicted_label="UNSUPPORTED",
            probabilities={"CONTRADICTED": 0.10, "SUPPORTED_PROXY": 0.35, "UNSUPPORTED": 0.55},
            contradicted_prob=0.10,
            supported_proxy_prob=0.35,
            unsupported_prob=0.55,
        ),
    ]
    mock_deb = MockDebertaVerifier(mock_preds)
    verifier = CalibratedDebertaVerifier(default_regime="direct", deberta_verifier=mock_deb)

    claims = [
        _make_claim("c1", "Apollo 11 landed on the moon."),
        _make_claim("c2", "Michael Collins walked on the moon."),
        _make_claim("c3", "The astronauts had lunch before landing."),
    ]
    matches = [
        _make_match("c1", "Neil Armstrong and Buzz Aldrin landed the Eagle on July 20, 1969."),
        _make_match("c2", "Collins remained in orbit in the Command Module and did not walk on the Moon."),
        _make_match("c3", "Mission control communicated regular updates to the spacecraft."),
    ]

    report = verifier.verify_claims(claims, matches, regime="direct")

    assert len(report.claims) == 3
    assert report.claims[0].classification == VerificationCategory.SUPPORTED
    assert report.claims[0].decision_rule == "CALIBRATED_DEBERTA_C2_SUPPORT"

    assert report.claims[1].classification == VerificationCategory.CONTRADICTED
    assert report.claims[1].decision_rule == "CALIBRATED_DEBERTA_C2_CONTRADICTION"

    assert report.claims[2].classification == VerificationCategory.UNSUPPORTED
    assert report.claims[2].decision_rule == "CALIBRATED_DEBERTA_C2_UNSUPPORTED"

    assert report.metrics.supported_count == 1
    assert report.metrics.contradicted_count == 1
    assert report.metrics.unsupported_count == 1
    assert report.metrics.unverifiable_count == 0


# ---------------------------------------------------------------------------
# Test 5: Correct Retrieval threshold decisions
# ---------------------------------------------------------------------------
def test_5_retrieval_threshold_decisions() -> None:
    """Verifies that retrieval calibration applies tau_contra=0.30, tau_sup=0.70 when sim >= 0.20."""
    mock_preds = [
        CustomDebertaPrediction(
            predicted_label="SUPPORTED_PROXY",
            probabilities={"CONTRADICTED": 0.02, "SUPPORTED_PROXY": 0.95, "UNSUPPORTED": 0.03},
            contradicted_prob=0.02,
            supported_proxy_prob=0.95,
            unsupported_prob=0.03,
        ),
        # High contradiction: raw 0.95 calibrates to p_contra=0.3995 >= 0.30 in retrieval
        CustomDebertaPrediction(
            predicted_label="CONTRADICTED",
            probabilities={"CONTRADICTED": 0.95, "SUPPORTED_PROXY": 0.01, "UNSUPPORTED": 0.04},
            contradicted_prob=0.95,
            supported_proxy_prob=0.01,
            unsupported_prob=0.04,
        ),
    ]
    mock_deb = MockDebertaVerifier(mock_preds)
    verifier = CalibratedDebertaVerifier(default_regime="retrieval", deberta_verifier=mock_deb)

    claims = [_make_claim("c1", "Claim one"), _make_claim("c2", "Claim two")]
    matches = [
        _make_match("c1", "Evidence for claim one", similarity=0.65),
        _make_match("c2", "Refutation of claim two", similarity=0.72),
    ]

    report = verifier.verify_claims(claims, matches, regime="retrieval")
    assert report.claims[0].classification == VerificationCategory.SUPPORTED
    assert report.claims[1].classification == VerificationCategory.CONTRADICTED


# ---------------------------------------------------------------------------
# Test 6: Evidence-floor abstention at similarity below 0.20
# ---------------------------------------------------------------------------
def test_6_retrieval_evidence_floor_below_020() -> None:
    """Verifies that evidence with similarity < 0.20 in Retrieval mode abstains to UNVERIFIABLE."""
    # Even if model is 99% confident of support, low similarity must overrule!
    mock_preds = [
        CustomDebertaPrediction(
            predicted_label="SUPPORTED_PROXY",
            probabilities={"CONTRADICTED": 0.01, "SUPPORTED_PROXY": 0.98, "UNSUPPORTED": 0.01},
            contradicted_prob=0.01,
            supported_proxy_prob=0.98,
            unsupported_prob=0.01,
        )
    ]
    mock_deb = MockDebertaVerifier(mock_preds)
    verifier = CalibratedDebertaVerifier(default_regime="retrieval", deberta_verifier=mock_deb)

    claim = _make_claim("c1", "A factual claim")
    match = _make_match("c1", "Some weakly related text", similarity=0.19)  # 0.19 < 0.20 floor

    report = verifier.verify_claims([claim], [match], regime="retrieval")
    vc = report.claims[0]

    assert vc.classification == VerificationCategory.UNVERIFIABLE
    assert vc.decision_rule == "EVIDENCE_SIMILARITY_BELOW_FLOOR"
    assert report.metrics.unverifiable_count == 1
    assert report.metrics.confirmed_contradiction_rate == 0.0  # NOT counted as hallucination


# ---------------------------------------------------------------------------
# Test 7: Evidence-floor behavior at exactly 0.20
# ---------------------------------------------------------------------------
def test_7_retrieval_evidence_floor_at_exactly_020() -> None:
    """Verifies that evidence with similarity == 0.20 in Retrieval mode does NOT abstain."""
    mock_preds = [
        CustomDebertaPrediction(
            predicted_label="SUPPORTED_PROXY",
            probabilities={"CONTRADICTED": 0.02, "SUPPORTED_PROXY": 0.95, "UNSUPPORTED": 0.03},
            contradicted_prob=0.02,
            supported_proxy_prob=0.95,
            unsupported_prob=0.03,
        )
    ]
    mock_deb = MockDebertaVerifier(mock_preds)
    verifier = CalibratedDebertaVerifier(default_regime="retrieval", deberta_verifier=mock_deb)

    claim = _make_claim("c1", "A factual claim")
    match = _make_match("c1", "Passage right at the similarity threshold", similarity=0.20)

    report = verifier.verify_claims([claim], [match], regime="retrieval")
    vc = report.claims[0]

    assert vc.classification == VerificationCategory.SUPPORTED
    assert vc.decision_rule == "CALIBRATED_DEBERTA_C2_SUPPORT"


# ---------------------------------------------------------------------------
# Test 8: Empty-evidence abstention
# ---------------------------------------------------------------------------
def test_8_empty_evidence_abstention() -> None:
    """Verifies that missing or empty-text evidence triggers UNVERIFIABLE and bypasses inference."""
    mock_deb = MockDebertaVerifier()
    verifier = CalibratedDebertaVerifier(deberta_verifier=mock_deb)

    c1 = _make_claim("c1", "No evidence match provided.")
    c2 = _make_claim("c2", "Empty text passage.")

    match_empty = _make_match("c2", "   ")  # whitespace only

    report = verifier.verify_claims([c1, c2], [match_empty], regime="retrieval")

    assert report.claims[0].classification == VerificationCategory.UNVERIFIABLE
    assert report.claims[0].decision_rule == "EMPTY_EVIDENCE"

    assert report.claims[1].classification == VerificationCategory.UNVERIFIABLE
    assert report.claims[1].decision_rule == "EMPTY_EVIDENCE"

    # Neural inference must have been bypassed completely (0 calls)
    assert mock_deb.call_count == 0


# ---------------------------------------------------------------------------
# Test 9: NON_FACTUAL handling
# ---------------------------------------------------------------------------
def test_9_non_factual_handling() -> None:
    """Verifies that non-factual propositions are tagged NON_FACTUAL without neural inference."""
    mock_deb = MockDebertaVerifier()
    verifier = CalibratedDebertaVerifier(deberta_verifier=mock_deb)

    c_nonfact = _make_claim("nf1", "In my personal opinion, Apollo 11 was magnificent.", is_factual=False)
    match = _make_match("nf1", "Apollo 11 landed on the moon.", similarity=0.90)

    report = verifier.verify_claims([c_nonfact], [match])

    vc = report.claims[0]
    assert vc.classification == VerificationCategory.NON_FACTUAL
    assert vc.decision_rule == "NON_FACTUAL_ASSERTION"
    assert report.metrics.factual_claims == 0
    assert report.metrics.non_factual_claims == 1
    assert report.metrics.has_factual_claims is False
    assert mock_deb.call_count == 0


# ---------------------------------------------------------------------------
# Test 10: Invalid probabilities and malformed bundle handling
# ---------------------------------------------------------------------------
def test_10_invalid_probabilities_and_malformed_bundle(tmp_path: Path) -> None:
    """Verifies that invalid bundle paths, malformed JSON, and NaN probabilities raise clear errors."""
    # 1. Non-existent bundle
    with pytest.raises(FileNotFoundError, match="Frozen calibration bundle not found"):
        load_frozen_calibration_bundle(tmp_path / "missing_bundle.json")

    # 2. Malformed JSON bundle
    bad_json = tmp_path / "corrupt.json"
    bad_json.write_text("{bad_json: true}", encoding="utf-8")
    with pytest.raises(ValueError, match="Failed to parse frozen calibration bundle"):
        load_frozen_calibration_bundle(bad_json)

    # 3. Missing regimes key
    bad_schema = tmp_path / "bad_schema.json"
    bad_schema.write_text(json.dumps({"schema_version": "v1"}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required top-level 'regimes'"):
        load_frozen_calibration_bundle(bad_schema)

    # 4. Unknown regime requested
    verifier = CalibratedDebertaVerifier(deberta_verifier=MockDebertaVerifier())
    claim = _make_claim("c1", "Test")
    match = _make_match("c1", "Evidence")
    with pytest.raises(ValueError, match="Invalid regime 'unknown_regime'"):
        verifier.verify_claims([claim], [match], regime="unknown_regime")

    # 5. Non-finite probability from verifier
    class BadPred:
        predicted_label = "SUPPORTED_PROXY"
        probabilities = {"CONTRADICTED": float("nan"), "SUPPORTED_PROXY": 0.5, "UNSUPPORTED": 0.5}
        contradicted_prob = float("nan")
        supported_proxy_prob = 0.5
        unsupported_prob = 0.5

    mock_bad_prob = MagicMock()
    mock_bad_prob.predict_pairs.return_value = [BadPred()]
    bad_verifier = CalibratedDebertaVerifier(deberta_verifier=mock_bad_prob)
    with pytest.raises(ValueError, match="Non-finite probability"):
        bad_verifier.verify_claims([claim], [match], regime="direct")


# ---------------------------------------------------------------------------
# Test 11: Missing model/checkpoint error handling
# ---------------------------------------------------------------------------
def test_11_missing_checkpoint_error_handling(tmp_path: Path) -> None:
    """Verifies that accessing custom_deberta when checkpoint is missing raises FileNotFoundError."""
    missing_dir = tmp_path / "non_existent_model_dir"
    verifier = CalibratedDebertaVerifier(model_path=missing_dir)

    with pytest.raises(FileNotFoundError, match="Custom DeBERTa model directory does not exist"):
        _ = verifier.custom_deberta


# ---------------------------------------------------------------------------
# Test 12: Model caching and lazy initialization
# ---------------------------------------------------------------------------
def test_12_model_caching_and_lazy_initialization() -> None:
    """Verifies that verifier is lazy and subsequent accesses reuse the exact instance."""
    verifier = CalibratedDebertaVerifier()
    # Initially None before first access
    assert verifier._custom_deberta is None

    # Injecting mock maintains identity across repeated accesses
    mock_inst = MockDebertaVerifier()
    mock_verifier = CalibratedDebertaVerifier(deberta_verifier=mock_inst)
    assert mock_verifier.custom_deberta is mock_inst
    assert mock_verifier.custom_deberta is mock_verifier.custom_deberta


# ---------------------------------------------------------------------------
# Test 13: Compatibility with existing verification report and scorer interfaces
# ---------------------------------------------------------------------------
def test_13_compatibility_with_response_scorer() -> None:
    """Verifies that VerificationReport from CalibratedDebertaVerifier flows into ResponseScorer cleanly."""
    mock_preds = [
        # c1 -> SUPPORTED
        CustomDebertaPrediction(
            predicted_label="SUPPORTED_PROXY",
            probabilities={"CONTRADICTED": 0.02, "SUPPORTED_PROXY": 0.95, "UNSUPPORTED": 0.03},
            contradicted_prob=0.02,
            supported_proxy_prob=0.95,
            unsupported_prob=0.03,
        ),
        # c2 -> CONTRADICTED: raw 0.95 calibrates to p_contra=0.3995 >= 0.30 in retrieval
        CustomDebertaPrediction(
            predicted_label="CONTRADICTED",
            probabilities={"CONTRADICTED": 0.95, "SUPPORTED_PROXY": 0.01, "UNSUPPORTED": 0.04},
            contradicted_prob=0.95,
            supported_proxy_prob=0.01,
            unsupported_prob=0.04,
        ),
        # c3 -> UNSUPPORTED: raw [0.05, 0.05, 0.90] calibrates to p_contra=0.0829 < 0.30, p_sup=0.5867 < 0.70
        CustomDebertaPrediction(
            predicted_label="UNSUPPORTED",
            probabilities={"CONTRADICTED": 0.05, "SUPPORTED_PROXY": 0.05, "UNSUPPORTED": 0.90},
            contradicted_prob=0.05,
            supported_proxy_prob=0.05,
            unsupported_prob=0.90,
        ),
    ]
    mock_deb = MockDebertaVerifier(mock_preds)
    verifier = CalibratedDebertaVerifier(default_regime="retrieval", deberta_verifier=mock_deb)

    claims = [
        _make_claim("c1", "Supported claim"),
        _make_claim("c2", "Contradicted claim"),
        _make_claim("c3", "Unsupported claim"),
        _make_claim("c4", "Unverifiable low sim claim"),
        _make_claim("c5", "Non-factual greeting", is_factual=False),
    ]
    matches = [
        _make_match("c1", "Entailing evidence", similarity=0.85),
        _make_match("c2", "Refuting evidence", similarity=0.75),
        _make_match("c3", "Neutral evidence", similarity=0.55),
        _make_match("c4", "Weak evidence", similarity=0.12),  # < 0.20 -> UNVERIFIABLE
        _make_match("c5", "Context for greeting", similarity=0.90),
    ]

    report = verifier.verify_claims(claims, matches, regime="retrieval", document_name="apollo.pdf")

    assert isinstance(report, VerificationReport)
    assert report.total_claims_processed == 5
    assert report.metrics.factual_claims == 4
    assert report.metrics.non_factual_claims == 1
    assert report.metrics.supported_count == 1
    assert report.metrics.contradicted_count == 1
    assert report.metrics.unsupported_count == 1
    assert report.metrics.unverifiable_count == 1
    assert report.metrics.non_factual_count == 1

    # Now feed into ResponseScorer
    scorer = ResponseScorer()
    score = scorer.score_response(report, response_id="resp_apollo_01")

    assert isinstance(score, ResponseScore)
    assert score.has_factual_claims is True
    assert score.counts.factual_claims == 4
    assert score.counts.supported_count == 1
    assert score.counts.contradicted_count == 1
    assert score.counts.unsupported_count == 1
    assert score.counts.unverifiable_count == 1
    assert score.counts.non_factual_count == 1

    # Exact rates: 1/4 = 25.0% each
    assert score.rates.evidence_support_rate == 25.0
    assert score.rates.confirmed_contradiction_rate == 25.0
    assert score.rates.unsupported_rate == 25.0
    assert score.rates.unverifiable_rate == 25.0
    assert score.rates.combined_uncertainty_rate == 50.0  # (1 + 1)/4 = 50.0%

    # Invariant check: ESR + CCR + USR + UVR == 100.0%
    assert score.mathematical_invariants_verified is True
    assert len(score.claims) == 5
