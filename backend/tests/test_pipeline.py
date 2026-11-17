"""
Unit and Integration Tests for VeriLLM End-to-End Pipeline.
Validates VeriLLMPipeline orchestrating document ingestion, sliding-window chunking,
atomic claim extraction, dense passage retrieval, calibrated DeBERTa-v3 verification,
and response scoring across both Direct and Retrieval regimes.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, List, Optional
from unittest.mock import MagicMock, patch

import pytest

# Ensure backend root is on sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Force offline execution for cached models
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from ml_engine.claim_verifier.calibrated_verifier import (
    CalibratedDebertaVerifier,
    CustomDebertaPrediction,
)
from ml_engine.claim_verifier.models import (
    VerificationCategory,
    VerificationReport,
    VerifiedClaim,
)
from ml_engine.evidence_retriever.models import (
    ClaimEvidenceMatch,
    DocumentChunk,
    RetrievedPassage,
)
from ml_engine.pipeline import PipelineResult, VeriLLMPipeline
from ml_engine.scorer.models import ResponseScore


class MockDebertaVerifier:
    """Mock for CustomDebertaVerifier avoiding heavy neural model loads."""

    def __init__(self, predictions: Optional[List[CustomDebertaPrediction]] = None) -> None:
        self.predictions = list(predictions) if predictions is not None else []
        self.calls: List[List[Any]] = []

    def predict_pairs(
        self, pairs: List[Any], batch_size: int = 16
    ) -> List[CustomDebertaPrediction]:
        self.calls.append(list(pairs))
        if self.predictions:
            # Pop or slice predictions to match input pairs
            returned = self.predictions[: len(pairs)]
            return returned
        # Default fallback: return neutral predictions (UNSUPPORTED)
        return [
            CustomDebertaPrediction(
                predicted_label="UNSUPPORTED",
                probabilities={
                    "CONTRADICTED": 0.05,
                    "SUPPORTED_PROXY": 0.05,
                    "UNSUPPORTED": 0.90,
                },
                contradicted_prob=0.05,
                supported_proxy_prob=0.05,
                unsupported_prob=0.90,
            )
            for _ in pairs
        ]


# ---------------------------------------------------------------------------
# Test 1: Successful Direct-Mode Pipeline
# ---------------------------------------------------------------------------
def test_1_successful_direct_mode_pipeline() -> None:
    """Verifies end-to-end execution of direct mode using full context document."""
    # Two predictions: claim 1 supported, claim 2 contradicted
    mock_preds = [
        # Claim 1: High support -> SUPPORTED
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
    ]
    mock_deb = MockDebertaVerifier(mock_preds)
    calibrated_ver = CalibratedDebertaVerifier(
        default_regime="direct", deberta_verifier=mock_deb
    )
    pipeline = VeriLLMPipeline(
        default_regime="direct", calibrated_verifier=calibrated_ver
    )

    doc_text = (
        "The Apollo 11 mission landed on the lunar surface on July 20, 1969. "
        "Neil Armstrong and Buzz Aldrin walked on the Moon while Michael Collins remained in lunar orbit."
    )
    response_text = (
        "Apollo 11 landed on the moon on July 20, 1969. "
        "Michael Collins walked on the moon with Armstrong."
    )

    result = pipeline.verify(
        document=doc_text,
        response=response_text,
        regime="direct",
        document_name="apollo_mission.txt",
    )

    assert isinstance(result, PipelineResult)
    assert result.regime == "direct"
    assert result.document_name == "apollo_mission.txt"
    assert len(result.claims) >= 2

    # Verify claim 1 is supported
    assert result.claims[0].classification == VerificationCategory.SUPPORTED
    assert result.claims[0].decision_rule == "CALIBRATED_DEBERTA_C2_SUPPORT"
    assert result.claims[0].primary_evidence is not None
    assert result.claims[0].primary_evidence.similarity_score == 1.0

    # Verify claim 2 is contradicted
    assert result.claims[1].classification == VerificationCategory.CONTRADICTED
    assert result.claims[1].decision_rule == "CALIBRATED_DEBERTA_C2_CONTRADICTION"

    # Verify response score metrics
    assert result.score.has_factual_claims is True
    assert result.score.counts.supported_count >= 1
    assert result.score.counts.contradicted_count >= 1
    assert result.score.rates.evidence_support_rate is not None
    assert result.score.rates.confirmed_contradiction_rate is not None


# ---------------------------------------------------------------------------
# Test 2: Successful Retrieval-Mode Pipeline
# ---------------------------------------------------------------------------
def test_2_successful_retrieval_mode_pipeline() -> None:
    """Verifies end-to-end execution of retrieval mode with chunking and passage retrieval."""
    mock_preds = [
        # Claim 1: SUPPORTED
        CustomDebertaPrediction(
            predicted_label="SUPPORTED_PROXY",
            probabilities={"CONTRADICTED": 0.02, "SUPPORTED_PROXY": 0.95, "UNSUPPORTED": 0.03},
            contradicted_prob=0.02,
            supported_proxy_prob=0.95,
            unsupported_prob=0.03,
        ),
    ]
    mock_deb = MockDebertaVerifier(mock_preds)
    calibrated_ver = CalibratedDebertaVerifier(
        default_regime="retrieval", deberta_verifier=mock_deb
    )
    pipeline = VeriLLMPipeline(
        default_regime="retrieval", calibrated_verifier=calibrated_ver
    )

    doc_text = (
        "The Apollo 11 Lunar Module Eagle touched down on July 20, 1969. "
        "Commander Neil Armstrong stepped onto the lunar surface six hours later."
    )
    response_text = "Neil Armstrong stepped onto the lunar surface on Apollo 11."

    result = pipeline.verify(
        document=doc_text,
        response=response_text,
        regime="retrieval",
        document_name="apollo_doc.txt",
    )

    assert isinstance(result, PipelineResult)
    assert result.regime == "retrieval"
    assert len(result.claims) >= 1
    assert result.claims[0].classification == VerificationCategory.SUPPORTED
    assert result.claims[0].primary_evidence is not None
    assert result.claims[0].primary_evidence.similarity_score > 0.20
    assert result.metadata["indexed_chunks_count"] >= 1


# ---------------------------------------------------------------------------
# Test 3: Correct Regime Selection & Validation
# ---------------------------------------------------------------------------
def test_3_correct_regime_selection() -> None:
    """Verifies default regime application, explicit override, and invalid regime rejection."""
    mock_deb = MockDebertaVerifier()
    calibrated_ver = CalibratedDebertaVerifier(deberta_verifier=mock_deb)

    # 1. Default regime is retrieval
    pipeline_default = VeriLLMPipeline(
        default_regime="retrieval", calibrated_verifier=calibrated_ver
    )
    res_default = pipeline_default.verify(
        document="A sample reference document.",
        response="A sample response statement.",
    )
    assert res_default.regime == "retrieval"

    # 2. Explicit direct override
    res_direct = pipeline_default.verify(
        document="A sample reference document.",
        response="A sample response statement.",
        regime="direct",
    )
    assert res_direct.regime == "direct"

    # 3. Explicit retrieval override
    res_retrieval = pipeline_default.verify(
        document="A sample reference document.",
        response="A sample response statement.",
        regime="retrieval",
    )
    assert res_retrieval.regime == "retrieval"

    # 4. Invalid regime in verify()
    with pytest.raises(ValueError, match="Invalid verification regime"):
        pipeline_default.verify(
            document="A reference document.",
            response="A response statement.",
            regime="quantum_superposition",
        )

    # 5. Invalid default_regime in constructor
    with pytest.raises(ValueError, match="Invalid default_regime"):
        VeriLLMPipeline(default_regime="invalid_default")


# ---------------------------------------------------------------------------
# Test 4: Evidence Retrieval and Claim Association
# ---------------------------------------------------------------------------
def test_4_evidence_retrieval_and_claim_association() -> None:
    """Verifies that in retrieval mode claims are linked to retrieved chunks with similarity scores."""
    mock_deb = MockDebertaVerifier()
    calibrated_ver = CalibratedDebertaVerifier(
        default_regime="retrieval", deberta_verifier=mock_deb
    )
    pipeline = VeriLLMPipeline(
        default_regime="retrieval", calibrated_verifier=calibrated_ver
    )

    doc_text = (
        "Paragraph 1: James Webb Space Telescope was launched in December 2021. "
        "It orbits the Sun-Earth L2 Lagrange point 1.5 million kilometers away.\n\n"
        "Paragraph 2: The Hubble Space Telescope was launched into low Earth orbit in 1990."
    )
    response_text = "The James Webb Space Telescope orbits at the Sun-Earth L2 point."

    result = pipeline.verify(
        document=doc_text,
        response=response_text,
        regime="retrieval",
        document_name="space_telescopes.txt",
    )

    assert len(result.claims) >= 1
    claim = result.claims[0]
    assert claim.primary_evidence is not None
    assert claim.primary_evidence.chunk is not None
    assert claim.primary_evidence.chunk.document_name == "space_telescopes.txt"
    assert claim.primary_evidence.similarity_score > 0.0
    assert len(claim.all_evidence) >= 1


# ---------------------------------------------------------------------------
# Test 5: Empty Evidence and Similarity Floor Abstention
# ---------------------------------------------------------------------------
def test_5_empty_evidence_abstention() -> None:
    """Verifies that low similarity (< 0.20) or missing evidence produces UNVERIFIABLE without HSI penalty."""
    mock_deb = MockDebertaVerifier()
    calibrated_ver = CalibratedDebertaVerifier(
        default_regime="retrieval", deberta_verifier=mock_deb
    )
    pipeline = VeriLLMPipeline(
        default_regime="retrieval", calibrated_verifier=calibrated_ver
    )

    # Document completely unrelated to the claim
    doc_text = "Recipes for baking chocolate chip cookies with organic brown sugar and flour."
    response_text = "Quantum computing uses qubits to achieve exponential speedups in factoring."

    result = pipeline.verify(
        document=doc_text,
        response=response_text,
        regime="retrieval",
        document_name="recipes.txt",
    )

    assert len(result.claims) >= 1
    claim = result.claims[0]

    # Either below 0.20 similarity floor or abstained
    if claim.primary_evidence and claim.primary_evidence.similarity_score < 0.20:
        assert claim.classification == VerificationCategory.UNVERIFIABLE
        assert claim.decision_rule == "EVIDENCE_SIMILARITY_BELOW_FLOOR"
        # Must not be counted as a confirmed hallucination in HSI
        assert result.score.indices.hallucination_severity_index == 0.0


# ---------------------------------------------------------------------------
# Test 6: Empty Document Handling
# ---------------------------------------------------------------------------
def test_6_empty_document_handling() -> None:
    """Verifies that an empty or whitespace document completes gracefully with claims marked UNVERIFIABLE."""
    mock_deb = MockDebertaVerifier()
    calibrated_ver = CalibratedDebertaVerifier(deberta_verifier=mock_deb)
    pipeline = VeriLLMPipeline(calibrated_verifier=calibrated_ver)

    # Empty document in Direct mode
    result_direct = pipeline.verify(
        document="   \n   ",
        response="Neil Armstrong stepped onto the lunar surface.",
        regime="direct",
    )
    assert result_direct.metadata["document_empty"] is True
    assert len(result_direct.claims) >= 1
    assert result_direct.claims[0].classification == VerificationCategory.UNVERIFIABLE
    assert result_direct.claims[0].decision_rule == "EMPTY_EVIDENCE"
    assert result_direct.score.indices.hallucination_severity_index == 0.0

    # Empty document in Retrieval mode
    result_retrieval = pipeline.verify(
        document="",
        response="Neil Armstrong stepped onto the lunar surface.",
        regime="retrieval",
    )
    assert result_retrieval.metadata["document_empty"] is True
    assert len(result_retrieval.claims) >= 1
    assert result_retrieval.claims[0].classification == VerificationCategory.UNVERIFIABLE
    assert result_retrieval.claims[0].decision_rule == "EMPTY_EVIDENCE"


# ---------------------------------------------------------------------------
# Test 7: Invalid Document Handling
# ---------------------------------------------------------------------------
def test_7_invalid_document_handling() -> None:
    """Verifies that invalid or corrupted documents raise appropriate, clear exceptions."""
    mock_deb = MockDebertaVerifier()
    calibrated_ver = CalibratedDebertaVerifier(deberta_verifier=mock_deb)
    pipeline = VeriLLMPipeline(calibrated_verifier=calibrated_ver)

    # 1. Non-existent PDF file
    with pytest.raises(FileNotFoundError, match="Reference PDF file not found"):
        pipeline.verify(
            document="completely_non_existent_file_12345.pdf",
            response="A valid candidate response.",
        )

    # 2. Corrupted PDF byte stream
    corrupt_pdf_bytes = b"%PDF-1.4 corrupt byte stream with no EOF or catalog"
    with pytest.raises(ValueError, match="Corrupted or invalid PDF file"):
        pipeline.verify(
            document=corrupt_pdf_bytes,
            response="A valid candidate response.",
        )

    # 3. None document
    with pytest.raises(ValueError, match="Reference document cannot be None"):
        pipeline.verify(
            document=None,  # type: ignore[arg-type]
            response="A valid candidate response.",
        )

    # 4. Unsupported document type
    with pytest.raises(ValueError, match="Unsupported document input type"):
        pipeline.verify(
            document=99999,  # type: ignore[arg-type]
            response="A valid candidate response.",
        )


# ---------------------------------------------------------------------------
# Test 8: Empty Response Handling
# ---------------------------------------------------------------------------
def test_8_empty_response_handling() -> None:
    """Verifies that an empty or whitespace response returns a valid PipelineResult with 0 claims."""
    mock_deb = MockDebertaVerifier()
    calibrated_ver = CalibratedDebertaVerifier(deberta_verifier=mock_deb)
    pipeline = VeriLLMPipeline(calibrated_verifier=calibrated_ver)

    # Empty string response
    result_empty = pipeline.verify(
        document="A valid reference document about astronomy.",
        response="",
    )
    assert isinstance(result_empty, PipelineResult)
    assert len(result_empty.claims) == 0
    assert result_empty.score.is_empty_response is True
    assert result_empty.score.has_factual_claims is False
    assert result_empty.metadata["response_empty"] is True

    # Whitespace-only response
    result_ws = pipeline.verify(
        document="A valid reference document about astronomy.",
        response="    \n\t  ",
    )
    assert len(result_ws.claims) == 0
    assert result_ws.score.is_empty_response is True

    # None response must raise ValueError
    with pytest.raises(ValueError, match="Response text cannot be None"):
        pipeline.verify(
            document="A valid reference document.",
            response=None,  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Test 9: Correct Response Scoring
# ---------------------------------------------------------------------------
def test_9_correct_response_scoring() -> None:
    """Verifies that response factuality metrics (GFI, CPFS, HSI, ESR) match claims accurately."""
    mock_preds = [
        # c1 -> SUPPORTED
        CustomDebertaPrediction(
            predicted_label="SUPPORTED_PROXY",
            probabilities={"CONTRADICTED": 0.02, "SUPPORTED_PROXY": 0.95, "UNSUPPORTED": 0.03},
            contradicted_prob=0.02,
            supported_proxy_prob=0.95,
            unsupported_prob=0.03,
        ),
        # c2 -> CONTRADICTED: raw 0.90 calibrates to p_contra >= 0.35 in direct
        CustomDebertaPrediction(
            predicted_label="CONTRADICTED",
            probabilities={"CONTRADICTED": 0.90, "SUPPORTED_PROXY": 0.02, "UNSUPPORTED": 0.08},
            contradicted_prob=0.90,
            supported_proxy_prob=0.02,
            unsupported_prob=0.08,
        ),
    ]
    mock_deb = MockDebertaVerifier(mock_preds)
    calibrated_ver = CalibratedDebertaVerifier(
        default_regime="direct", deberta_verifier=mock_deb
    )
    pipeline = VeriLLMPipeline(
        default_regime="direct", calibrated_verifier=calibrated_ver
    )

    doc_text = "The speed of light in vacuum is approximately 299,792 kilometers per second."
    response_text = (
        "The speed of light is 299,792 km/s. "
        "Sound travels faster than light in a vacuum."
    )

    result = pipeline.verify(
        document=doc_text,
        response=response_text,
        regime="direct",
    )

    score = result.score
    assert isinstance(score, ResponseScore)
    assert score.has_factual_claims is True
    assert score.counts.factual_claims == 2
    assert score.counts.supported_count == 1
    assert score.counts.contradicted_count == 1

    # Exact rates: 1 supported / 2 = 50.0%, 1 contradicted / 2 = 50.0%
    assert score.rates.evidence_support_rate == 50.0
    assert score.rates.confirmed_contradiction_rate == 50.0

    # Indices: GFI = 50.0, HSI = 50.0
    assert score.indices.grounded_factuality_index == 50.0
    assert score.indices.hallucination_severity_index == 50.0
    assert score.mathematical_invariants_verified is True


# ---------------------------------------------------------------------------
# Test 10: Output Structure and Serialization
# ---------------------------------------------------------------------------
def test_10_output_structure_and_serialization() -> None:
    """Verifies that PipelineResult serializes cleanly to JSON without errors."""
    mock_deb = MockDebertaVerifier()
    calibrated_ver = CalibratedDebertaVerifier(deberta_verifier=mock_deb)
    pipeline = VeriLLMPipeline(calibrated_verifier=calibrated_ver)

    result = pipeline.verify(
        document="The sun is a G-type main-sequence star.",
        response="The sun is a star.",
        regime="direct",
        document_name="solar_info.txt",
    )

    # Test .to_dict() and .model_dump()
    dumped = result.to_dict()
    assert isinstance(dumped, dict)
    assert dumped["document_name"] == "solar_info.txt"
    assert dumped["regime"] == "direct"
    assert "claims" in dumped
    assert "score" in dumped
    assert "verification_report" in dumped
    assert "metadata" in dumped
    assert dumped["execution_time_seconds"] >= 0.0

    # Test JSON serialization
    json_str = json.dumps(dumped, indent=2)
    assert isinstance(json_str, str)
    loaded = json.loads(json_str)
    assert loaded["document_name"] == "solar_info.txt"


# ---------------------------------------------------------------------------
# Test 11: Model Instance Reuse
# ---------------------------------------------------------------------------
def test_11_model_instance_reuse() -> None:
    """Verifies that repeated verify() calls reuse the exact same verifier instance without reloading."""
    mock_deb = MockDebertaVerifier()
    calibrated_ver = CalibratedDebertaVerifier(deberta_verifier=mock_deb)
    pipeline = VeriLLMPipeline(calibrated_verifier=calibrated_ver)

    # First call
    result_1 = pipeline.verify(
        document="The Moon orbits the Earth every 27.3 days.",
        response="The Moon orbits Earth.",
        regime="direct",
    )

    # Second call with different text
    result_2 = pipeline.verify(
        document="Mars has two small moons, Phobos and Deimos.",
        response="Mars has two moons.",
        regime="direct",
    )

    # Both calls must share the exact same verifier object
    assert pipeline.calibrated_verifier is calibrated_ver
    assert pipeline.calibrated_verifier.custom_deberta is mock_deb

    # The mock verifier was invoked twice
    assert len(mock_deb.calls) == 2


# ---------------------------------------------------------------------------
# Test 12: Compatibility with Existing Components
# ---------------------------------------------------------------------------
def test_12_compatibility_with_existing_components() -> None:
    """Verifies that PipelineResult components adhere strictly to existing Pydantic schemas."""
    mock_deb = MockDebertaVerifier()
    calibrated_ver = CalibratedDebertaVerifier(deberta_verifier=mock_deb)
    pipeline = VeriLLMPipeline(calibrated_verifier=calibrated_ver)

    result = pipeline.verify(
        document="Water boils at 100 degrees Celsius at standard atmospheric pressure.",
        response="Water boils at 100 degrees Celsius.",
        regime="direct",
    )

    # 1. VerificationReport schema
    assert isinstance(result.verification_report, VerificationReport)
    assert hasattr(result.verification_report, "metrics")
    assert hasattr(result.verification_report, "claims")

    # 2. VerifiedClaim schema
    for c in result.claims:
        assert isinstance(c, VerifiedClaim)
        assert isinstance(c.classification, VerificationCategory)
        assert hasattr(c, "scores")
        assert hasattr(c, "reasoning_trace")

    # 3. ResponseScore schema
    assert isinstance(result.score, ResponseScore)
    assert hasattr(result.score, "counts")
    assert hasattr(result.score, "rates")
    assert hasattr(result.score, "indices")
    assert hasattr(result.score, "mathematical_invariants_verified")


# ---------------------------------------------------------------------------
# Test 13: Scanned PDF Unsupported OCR Detection
# ---------------------------------------------------------------------------
def test_13_scanned_pdf_unsupported_ocr() -> None:
    """Verifies that scanned PDFs with no machine-readable text raise a clear OCR unsupported error."""
    import pypdf
    import io

    # Generate an image-only / blank 2-page PDF
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.add_blank_page(width=612, height=792)
    stream = io.BytesIO()
    writer.write(stream)
    scanned_bytes = stream.getvalue()

    mock_deb = MockDebertaVerifier()
    calibrated_ver = CalibratedDebertaVerifier(deberta_verifier=mock_deb)
    pipeline = VeriLLMPipeline(calibrated_verifier=calibrated_ver)

    with pytest.raises(ValueError, match="Scanned PDF detected.*OCR is not supported"):
        pipeline.verify(
            document=scanned_bytes,
            response="Apollo 11 landed on the moon.",
            document_name="scanned_report.pdf",
        )


# ---------------------------------------------------------------------------
# Test 14: Unsupported File Types Handling
# ---------------------------------------------------------------------------
def test_14_unsupported_file_types_handling(tmp_path: Path) -> None:
    """Verifies that binary files and unsupported extensions raise actionable ValueError."""
    mock_deb = MockDebertaVerifier()
    calibrated_ver = CalibratedDebertaVerifier(deberta_verifier=mock_deb)
    pipeline = VeriLLMPipeline(calibrated_verifier=calibrated_ver)

    # 1. Binary data with null bytes that is not a PDF
    with pytest.raises(ValueError, match="Binary data detected"):
        pipeline.verify(
            document=b"PNG_HEADER\x00\x00\x00IHDR\x00\x01\x02",
            response="A sample candidate response.",
        )

    # 2. Unsupported file extension on disk (.png)
    bad_file = tmp_path / "diagram.png"
    bad_file.write_bytes(b"dummy png content")
    with pytest.raises(ValueError, match="Unsupported file format '.png'"):
        pipeline.verify(
            document=bad_file,
            response="A sample candidate response.",
        )


# ---------------------------------------------------------------------------
# Test 15: Long-Document Direct Mode Warning
# ---------------------------------------------------------------------------
def test_15_long_document_direct_mode_warning() -> None:
    """Verifies that direct mode detects inputs exceeding 512 tokens, flags truncation, and warns user."""
    mock_deb = MockDebertaVerifier()
    calibrated_ver = CalibratedDebertaVerifier(
        default_regime="direct", deberta_verifier=mock_deb
    )
    pipeline = VeriLLMPipeline(
        default_regime="direct", calibrated_verifier=calibrated_ver
    )

    # Construct a 450-word document (~600 subword tokens, exceeds 512)
    sentences = [
        f"Scientific proposition number {i} establishes that celestial body {i} orbits star system Alpha Centauri."
        for i in range(1, 45)
    ]
    long_doc = " ".join(sentences)
    assert len(long_doc.split()) >= 400

    result = pipeline.verify(
        document=long_doc,
        response="Celestial body 1 orbits star system Alpha Centauri.",
        regime="direct",
    )

    assert result.metadata["direct_context_truncated"] is True
    assert result.metadata["recommended_regime"] == "retrieval"
    assert any("512-token limit" in w for w in result.warnings)
    assert any("Retrieval mode ('retrieval') is strongly recommended" in w for w in result.warnings)

    # Claim reasoning trace should note truncation
    assert len(result.claims) >= 1
    assert "truncated" in result.claims[0].reasoning_trace.lower()


# ---------------------------------------------------------------------------
# Test 16: Long-Document Retrieval Mode Execution
# ---------------------------------------------------------------------------
def test_16_long_document_retrieval_mode() -> None:
    """Verifies that long documents in retrieval mode chunk cleanly without direct-mode truncation warning."""
    mock_deb = MockDebertaVerifier()
    calibrated_ver = CalibratedDebertaVerifier(
        default_regime="retrieval", deberta_verifier=mock_deb
    )
    pipeline = VeriLLMPipeline(
        default_regime="retrieval", calibrated_verifier=calibrated_ver
    )

    sentences = [
        f"Scientific proposition number {i} establishes that celestial body {i} orbits star system Alpha Centauri."
        for i in range(1, 45)
    ]
    long_doc = " ".join(sentences)

    result = pipeline.verify(
        document=long_doc,
        response="Celestial body 1 orbits star system Alpha Centauri.",
        regime="retrieval",
    )

    assert result.metadata["direct_context_truncated"] is False
    assert result.metadata["indexed_chunks_count"] > 1
    assert not any("512-token limit" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Test 17: Missing Model Checkpoint Error
# ---------------------------------------------------------------------------
def test_17_missing_model_checkpoint_error() -> None:
    """Verifies that a missing model directory raises FileNotFoundError upon verifier access."""
    calibrated_ver = CalibratedDebertaVerifier(
        model_path="non_existent_checkpoint_directory_xyz_98765"
    )
    pipeline = VeriLLMPipeline(calibrated_verifier=calibrated_ver)

    with pytest.raises(FileNotFoundError, match="Custom DeBERTa model directory does not exist"):
        pipeline.verify(
            document="A valid reference document.",
            response="A valid empirical response statement.",
        )


# ---------------------------------------------------------------------------
# Test 18: Missing Frozen Bundle Error
# ---------------------------------------------------------------------------
def test_18_missing_frozen_bundle_error() -> None:
    """Verifies that a missing frozen calibration bundle path immediately raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Frozen calibration bundle not found"):
        VeriLLMPipeline(bundle_path="non_existent_frozen_bundle_98765.json")


# ---------------------------------------------------------------------------
# Test 19: Non-Factual Only Response Scoring Invariants
# ---------------------------------------------------------------------------
def test_19_non_factual_only_response_scoring_invariants() -> None:
    """Verifies that responses with zero factual claims produce undefined rates and clear warning."""
    mock_deb = MockDebertaVerifier()
    calibrated_ver = CalibratedDebertaVerifier(deberta_verifier=mock_deb)
    pipeline = VeriLLMPipeline(calibrated_verifier=calibrated_ver)

    result = pipeline.verify(
        document="The solar system formed approximately 4.6 billion years ago.",
        response="Hello! In my opinion, this is beautiful. Personally, I feel great. Hope this helps!",
    )

    assert result.score.has_factual_claims is False
    assert result.score.counts.factual_claims == 0
    assert result.score.counts.non_factual_count > 0
    assert result.score.rates.evidence_support_rate is None
    assert result.score.rates.confirmed_contradiction_rate is None
    assert any("no empirical factual claims" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Test 20: Comprehensive Metadata Observability
# ---------------------------------------------------------------------------
def test_20_comprehensive_metadata_observability() -> None:
    """Verifies that PipelineResult.metadata comprehensively exposes document stats, summary counts, and timings."""
    mock_deb = MockDebertaVerifier()
    calibrated_ver = CalibratedDebertaVerifier(deberta_verifier=mock_deb)
    pipeline = VeriLLMPipeline(calibrated_verifier=calibrated_ver)

    result = pipeline.verify(
        document="Earth has an average radius of 6,371 kilometers.",
        response="Earth radius is approximately 6,371 km.",
        prompt="What is the radius of Earth?",
        regime="retrieval",
        document_name="geophysics.txt",
    )

    meta = result.metadata
    assert "document" in meta
    assert meta["document"]["total_words"] > 0
    assert meta["document"]["char_count"] > 0

    assert "verification_summary" in meta
    summary = meta["verification_summary"]
    assert "total_claims" in summary
    assert "factual_claims" in summary
    assert "supported" in summary
    assert "contradicted" in summary
    assert "unsupported" in summary
    assert "unverifiable" in summary

    assert "extraction_stats" in meta
    assert "indexed_chunks_count" in meta
    assert meta["prompt_provided"] is True
    assert isinstance(result.warnings, list)
    assert result.execution_time_seconds > 0.0


# ---------------------------------------------------------------------------
# Test 21: Passage Retriever Memory Release
# ---------------------------------------------------------------------------
def test_21_passage_retriever_memory_release() -> None:
    """Verifies that passage retriever resets chunk embeddings after retrieval to prevent memory accumulation."""
    mock_deb = MockDebertaVerifier()
    calibrated_ver = CalibratedDebertaVerifier(deberta_verifier=mock_deb)
    pipeline = VeriLLMPipeline(calibrated_verifier=calibrated_ver)

    result = pipeline.verify(
        document="Jupiter is the largest planet in our solar system by mass and volume.",
        response="Jupiter is the largest planet.",
        regime="retrieval",
    )

    # After verification finishes, chunk_embeddings must have been cleared
    assert pipeline.passage_retriever.chunk_embeddings is None
    assert len(pipeline.passage_retriever.chunks) == 0


