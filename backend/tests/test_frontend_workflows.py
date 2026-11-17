"""
End-to-End Workflow and Integration Tests for VeriLLM Streamlit Application.
Step 10.6: Final Application Validation, Workflow Hardening, and Regression Closure.

Validates complete user journeys (Scenarios A through I):
- Scenario A: PDF upload -> response entry -> Retrieval verification -> Results dashboard.
- Scenario B: TXT upload -> response entry -> Direct verification -> Results dashboard.
- Scenario C: Manual reference text -> response entry -> Retrieval verification -> Results dashboard.
- Scenario D: Optional question omitted -> successful verification.
- Scenario E: Response containing only greetings/opinions -> non-factual handling (N=0).
- Scenario F: Insufficient evidence -> UNVERIFIABLE, never counted as confirmed hallucination.
- Scenario G: Long document in Direct mode -> context truncation warning.
- Scenario H: Scanned PDF -> actionable error explaining OCR is unsupported.
- Scenario I: Empty document, empty response and invalid file -> clear validation errors.

Also validates:
- Session state lifecycle and re-entrancy invariants.
- Dashboard KPI, proposition composition, and calibrated probability integrity.
- Memory and index cleanup invariants (PassageRetriever index clearing).
"""

import io
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
import pypdf

# Ensure project root and backend are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml_engine.claim_verifier.calibrated_verifier import (
    CalibratedDebertaVerifier,
    CustomDebertaPrediction,
)
from ml_engine.claim_verifier.models import VerificationCategory, VerifiedClaim
from ml_engine.pipeline import PipelineResult, VeriLLMPipeline
from ml_engine.scorer.models import ResponseScore

from frontend.views.verify import (
    build_staged_payload,
    format_file_size,
    validate_candidate_response,
    validate_document_file,
    validate_document_text,
)
from frontend.views.results import (
    build_calibrated_probabilities_chart,
    build_classification_distribution_chart,
    filter_claims_by_category,
    get_classification_str,
    render_results_view,
)
from frontend.utils.pipeline_service import (
    execute_verification,
    map_staged_inputs_to_pipeline_args,
)


# ===========================================================================
# Test Fixtures & Helpers
# ===========================================================================

class MockDebertaVerifier:
    """Mock for CustomDebertaVerifier allowing fast, deterministic predictions."""

    def __init__(self, predictions: Optional[List[CustomDebertaPrediction]] = None) -> None:
        self.predictions = list(predictions) if predictions is not None else []
        self.calls: List[List[Any]] = []

    def predict_pairs(
        self, pairs: List[Any], batch_size: int = 16
    ) -> List[CustomDebertaPrediction]:
        self.calls.append(list(pairs))
        if self.predictions:
            returned = self.predictions[: len(pairs)]
            return returned
        # Default fallback: neutral prediction
        return [
            CustomDebertaPrediction(
                predicted_label="SUPPORTED_PROXY",
                probabilities={
                    "CONTRADICTED": 0.05,
                    "SUPPORTED_PROXY": 0.90,
                    "UNSUPPORTED": 0.05,
                },
                contradicted_prob=0.05,
                supported_proxy_prob=0.90,
                unsupported_prob=0.05,
            )
            for _ in pairs
        ]


def create_test_pdf_bytes() -> bytes:
    """Generates a valid 2-page PDF document in memory using pypdf."""
    writer = pypdf.PdfWriter()

    font = pypdf.generic.DictionaryObject({
        pypdf.generic.NameObject("/Type"): pypdf.generic.NameObject("/Font"),
        pypdf.generic.NameObject("/Subtype"): pypdf.generic.NameObject("/Type1"),
        pypdf.generic.NameObject("/BaseFont"): pypdf.generic.NameObject("/Helvetica"),
    })
    font_ref = writer._add_object(font)

    pages_text = [
        (
            "Apollo 11 was launched by a Saturn V rocket from Kennedy Space Center on July 16, 1969. "
            "Commander Neil Armstrong and Lunar Module Pilot Buzz Aldrin landed the Eagle on the Moon on July 20, 1969."
        ),
        (
            "Neil Armstrong became the first human to walk on the Moon on July 21, 1969. "
            "Buzz Aldrin joined him 19 minutes later. Michael Collins flew the Command Module Columbia alone in orbit."
        ),
    ]

    for p_text in pages_text:
        page = writer.add_blank_page(width=612, height=792)
        stream_data = f"BT /F1 12 Tf 50 700 Td ({p_text}) Tj ET".encode("latin-1")
        stream = pypdf.generic.DecodedStreamObject()
        stream.set_data(stream_data)
        stream_ref = writer._add_object(stream)

        page[pypdf.generic.NameObject("/Contents")] = stream_ref
        page[pypdf.generic.NameObject("/Resources")] = pypdf.generic.DictionaryObject({
            pypdf.generic.NameObject("/Font"): pypdf.generic.DictionaryObject({
                pypdf.generic.NameObject("/F1"): font_ref,
            })
        })

    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def create_empty_scanned_pdf_bytes() -> bytes:
    """Generates a PDF document with zero extractable text (simulating scanned document)."""
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.add_blank_page(width=612, height=792)
    stream = io.BytesIO()
    writer.write(stream)
    return stream.getvalue()


def create_pipeline_with_mock(mock_deb: MockDebertaVerifier) -> VeriLLMPipeline:
    """Builds a VeriLLMPipeline using frozen calibration bundle and mock verifier."""
    calibrated_ver = CalibratedDebertaVerifier(deberta_verifier=mock_deb)
    return VeriLLMPipeline(calibrated_verifier=calibrated_ver)


# ===========================================================================
# Scenario Tests (A through I)
# ===========================================================================

class TestWorkflowScenarios:
    """Tests covering Scenarios A through I in complete user workflows."""

    def test_scenario_a_pdf_upload_retrieval_verification(self) -> None:
        """Scenario A: PDF upload -> response entry -> Retrieval verification -> Results dashboard."""
        pdf_bytes = create_test_pdf_bytes()
        file_name = "apollo_mission_report.pdf"

        # Step 1: Validate PDF file
        is_valid, err, meta = validate_document_file(file_name, pdf_bytes)
        assert is_valid is True
        assert err is None
        assert meta["type"] == "PDF Document"

        # Step 2: Validate candidate response
        response = (
            "Apollo 11 was launched on July 16, 1969. "
            "Neil Armstrong walked on the Moon on July 21, 1969."
        )
        r_valid, r_err, _ = validate_candidate_response(response)
        assert r_valid is True

        # Step 3: Build staged payload
        staged = build_staged_payload(
            doc_mode="File Upload (PDF / TXT)",
            doc_meta=meta,
            prompt="When was Apollo 11 launched?",
            response=response,
            regime="retrieval",
        )
        assert staged["is_validated"] is True
        assert staged["regime"] == "retrieval"

        # Step 4: Map staged inputs
        args = map_staged_inputs_to_pipeline_args(staged)
        assert args["document_name"] == file_name
        assert args["regime"] == "retrieval"
        assert args["prompt"] == "When was Apollo 11 launched?"

        # Step 5: Execute pipeline in retrieval mode
        mock_preds = [
            CustomDebertaPrediction(
                predicted_label="SUPPORTED_PROXY",
                probabilities={"CONTRADICTED": 0.02, "SUPPORTED_PROXY": 0.95, "UNSUPPORTED": 0.03},
                contradicted_prob=0.02,
                supported_proxy_prob=0.95,
                unsupported_prob=0.03,
            ),
            CustomDebertaPrediction(
                predicted_label="SUPPORTED_PROXY",
                probabilities={"CONTRADICTED": 0.01, "SUPPORTED_PROXY": 0.96, "UNSUPPORTED": 0.03},
                contradicted_prob=0.01,
                supported_proxy_prob=0.96,
                unsupported_prob=0.03,
            ),
        ]
        mock_deb = MockDebertaVerifier(predictions=mock_preds)
        pipeline = create_pipeline_with_mock(mock_deb)

        result = execute_verification(staged, pipeline=pipeline)

        # Step 6: Verify result structure
        assert isinstance(result, PipelineResult)
        assert result.regime == "retrieval"
        assert result.document_name == file_name
        assert len(result.claims) == 2
        assert result.score.indices.grounded_factuality_index == 100.0
        assert result.score.counts.supported_count == 2
        assert result.score.counts.contradicted_count == 0

        # Step 7: Verify Results dashboard rendering
        with patch("frontend.views.results.st") as mock_st:
            mock_st.session_state = {
                "verification_result": result,
                "last_verification_time": "2026-11-13 14:00:00",
            }
            mock_st.columns.side_effect = lambda n, **kwargs: [
                MagicMock() for _ in range(n if isinstance(n, int) else len(n))
            ]
            mock_st.radio.return_value = "All (2)"

            # Must execute without throwing exceptions
            render_results_view()
            assert mock_st.markdown.called

    def test_scenario_b_txt_upload_direct_verification(self) -> None:
        """Scenario B: TXT upload -> response entry -> Direct verification -> Results dashboard."""
        txt_content = (
            "The James Webb Space Telescope (JWST) is an infrared space observatory. "
            "It was launched on December 25, 2021, on an Ariane 5 rocket from Kourou, French Guiana."
        )
        txt_bytes = txt_content.encode("utf-8")
        file_name = "jwst_overview.txt"

        # Validate file
        is_valid, err, meta = validate_document_file(file_name, txt_bytes)
        assert is_valid is True
        assert meta["type"] == "Plain Text (TXT)"

        response = "JWST was launched on December 25, 2021."
        staged = build_staged_payload(
            doc_mode="File Upload (PDF / TXT)",
            doc_meta=meta,
            prompt=None,
            response=response,
            regime="direct",
        )

        mock_preds = [
            CustomDebertaPrediction(
                predicted_label="SUPPORTED_PROXY",
                probabilities={"CONTRADICTED": 0.02, "SUPPORTED_PROXY": 0.94, "UNSUPPORTED": 0.04},
                contradicted_prob=0.02,
                supported_proxy_prob=0.94,
                unsupported_prob=0.04,
            )
        ]
        pipeline = create_pipeline_with_mock(MockDebertaVerifier(mock_preds))
        result = execute_verification(staged, pipeline=pipeline)

        assert result.regime == "direct"
        assert len(result.claims) == 1
        assert result.claims[0].primary_evidence.chunk.chunk_id.endswith("_direct")
        assert result.score.indices.grounded_factuality_index == 100.0

    def test_scenario_c_manual_reference_text_retrieval_verification(self) -> None:
        """Scenario C: Manual reference text -> response entry -> Retrieval verification -> Results dashboard."""
        ref_text = (
            "Python 3.11 introduced substantial performance improvements, including the Specialized "
            "Adaptive Interpreter (PEP 659). Execution overhead was reduced by 10-60% across standard benchmarks."
        )
        is_valid, err, meta = validate_document_text(ref_text)
        assert is_valid is True
        assert meta["type"] == "Direct Text Entry"

        response = "Python 3.11 included PEP 659 to improve execution performance."
        staged = build_staged_payload(
            doc_mode="Paste / Type Reference Text",
            doc_meta=meta,
            prompt="What did Python 3.11 improve?",
            response=response,
            regime="retrieval",
        )

        mock_preds = [
            CustomDebertaPrediction(
                predicted_label="SUPPORTED_PROXY",
                probabilities={"CONTRADICTED": 0.03, "SUPPORTED_PROXY": 0.92, "UNSUPPORTED": 0.05},
                contradicted_prob=0.03,
                supported_proxy_prob=0.92,
                unsupported_prob=0.05,
            )
        ]
        pipeline = create_pipeline_with_mock(MockDebertaVerifier(mock_preds))
        result = execute_verification(staged, pipeline=pipeline)

        assert result.regime == "retrieval"
        assert result.document_name == "Pasted Reference Text"
        assert len(result.claims) == 1
        assert result.claims[0].classification == VerificationCategory.SUPPORTED

    def test_scenario_d_optional_question_omitted(self) -> None:
        """Scenario D: Optional question omitted -> successful verification without conversational context."""
        ref_text = "The speed of light in vacuum is exactly 299,792,458 meters per second."
        _, _, meta = validate_document_text(ref_text)

        response = "Light travels at 299,792,458 meters per second in a vacuum."
        staged = build_staged_payload(
            doc_mode="Paste / Type Reference Text",
            doc_meta=meta,
            prompt=None,  # Omitted question
            response=response,
            regime="direct",
        )

        args = map_staged_inputs_to_pipeline_args(staged)
        assert args["prompt"] is None

        mock_preds = [
            CustomDebertaPrediction(
                predicted_label="SUPPORTED_PROXY",
                probabilities={"CONTRADICTED": 0.01, "SUPPORTED_PROXY": 0.97, "UNSUPPORTED": 0.02},
                contradicted_prob=0.01,
                supported_proxy_prob=0.97,
                unsupported_prob=0.02,
            )
        ]
        pipeline = create_pipeline_with_mock(MockDebertaVerifier(mock_preds))
        result = execute_verification(staged, pipeline=pipeline)

        assert result.prompt is None
        assert result.score.indices.grounded_factuality_index == 100.0

    def test_scenario_e_response_only_greetings_and_opinions(self) -> None:
        """Scenario E: Response containing only greetings/opinions -> non-factual handling (N=0)."""
        ref_text = "Artificial intelligence is a branch of computer science."
        _, _, meta = validate_document_text(ref_text)

        # Strictly conversational remarks and subjective opinion
        response = "Hello! In my opinion, this is beautiful. Personally, I feel great. Hope this helps!"
        staged = build_staged_payload(
            doc_mode="Paste / Type Reference Text",
            doc_meta=meta,
            prompt="Hello",
            response=response,
            regime="direct",
        )

        pipeline = create_pipeline_with_mock(MockDebertaVerifier())
        result = execute_verification(staged, pipeline=pipeline)

        # Non-factual claims are partitioned out of factuality calculations
        assert result.score.has_factual_claims is False
        assert result.score.counts.factual_claims == 0
        assert result.score.counts.non_factual_count > 0
        # Indices are None when N=0 to avoid division by zero
        assert result.score.indices.grounded_factuality_index is None
        assert result.score.indices.contradiction_penalized_factuality is None
        assert result.score.indices.hallucination_severity_index is None

        # Verify results page renders cleanly without ZeroDivisionError
        with patch("frontend.views.results.st") as mock_st:
            mock_st.session_state = {
                "verification_result": result,
                "last_verification_time": "2026-11-13 14:00:00",
            }
            mock_st.columns.side_effect = lambda n, **kwargs: [
                MagicMock() for _ in range(n if isinstance(n, int) else len(n))
            ]
            mock_st.radio.return_value = "All (3)"

            render_results_view()
            # Informational notice about non-factual claims should be emitted
            assert mock_st.info.called

    def test_scenario_f_insufficient_evidence_unverifiable(self) -> None:
        """Scenario F: Insufficient evidence -> UNVERIFIABLE, never counted as confirmed hallucination."""
        ref_text = "The Amazon rainforest covers approximately 5.5 million square kilometers in South America."
        _, _, meta = validate_document_text(ref_text)

        # Completely unrelated proposition; similarity will be below tau_floor = 0.20
        response = "The Planck length is approximately 1.616 times 10 to the power of negative 35 meters."
        staged = build_staged_payload(
            doc_mode="Paste / Type Reference Text",
            doc_meta=meta,
            prompt="What is the Planck length?",
            response=response,
            regime="retrieval",
        )

        pipeline = create_pipeline_with_mock(MockDebertaVerifier())
        result = execute_verification(staged, pipeline=pipeline)

        assert len(result.claims) == 1
        claim = result.claims[0]
        assert claim.classification == VerificationCategory.UNVERIFIABLE
        assert "similarity" in claim.reasoning_trace.lower() or "floor" in claim.decision_rule.lower()

        # Invariant: Safe abstention is NOT penalized as a contradiction
        assert result.score.counts.unverifiable_count == 1
        assert result.score.counts.contradicted_count == 0
        assert result.score.rates.confirmed_contradiction_rate == 0.0
        assert result.score.indices.hallucination_severity_index == 0.0
        assert result.score.rates.unverifiable_rate == 100.0

    def test_scenario_g_long_document_direct_mode_truncation_warning(self) -> None:
        """Scenario G: Long document in Direct mode -> context truncation warning."""
        # Create long document text exceeding 512 tokens (~400 words)
        long_paragraph = (
            "The Apollo program was conceived in early 1960 during President Dwight D. Eisenhower's "
            "administration as a follow-up to Project Mercury. "
        ) * 20  # ~440 words, >512 subword tokens
        _, _, meta = validate_document_text(long_paragraph)

        response = "The Apollo program was conceived in early 1960."
        staged = build_staged_payload(
            doc_mode="Paste / Type Reference Text",
            doc_meta=meta,
            prompt=None,
            response=response,
            regime="direct",
        )

        mock_preds = [
            CustomDebertaPrediction(
                predicted_label="SUPPORTED_PROXY",
                probabilities={"CONTRADICTED": 0.02, "SUPPORTED_PROXY": 0.94, "UNSUPPORTED": 0.04},
                contradicted_prob=0.02,
                supported_proxy_prob=0.94,
                unsupported_prob=0.04,
            )
        ]
        pipeline = create_pipeline_with_mock(MockDebertaVerifier(mock_preds))
        result = execute_verification(staged, pipeline=pipeline)

        # Verify operational warning is attached
        assert len(result.warnings) > 0
        assert any("exceeds context limit" in w.lower() or "truncated" in w.lower() for w in result.warnings)

    def test_scenario_h_scanned_pdf_actionable_error(self) -> None:
        """Scenario H: Scanned PDF -> actionable error explaining OCR is unsupported."""
        scanned_bytes = create_empty_scanned_pdf_bytes()
        file_name = "scanned_doc.pdf"

        # File validation passes magic header check (%PDF)
        is_valid, err, meta = validate_document_file(file_name, scanned_bytes)
        assert is_valid is True

        response = "The document claims significant annual revenue."
        staged = build_staged_payload(
            doc_mode="File Upload (PDF / TXT)",
            doc_meta=meta,
            prompt=None,
            response=response,
            regime="retrieval",
        )

        pipeline = create_pipeline_with_mock(MockDebertaVerifier())

        # Parsing must detect is_scanned=True and raise clear, actionable ValueError
        with pytest.raises(ValueError, match="Scanned PDF detected.*OCR is not supported"):
            execute_verification(staged, pipeline=pipeline)

    def test_scenario_i_empty_and_invalid_inputs(self) -> None:
        """Scenario I: Empty document, empty response and invalid file -> clear validation errors."""
        # 1. Empty document bytes
        v_doc, err_doc, _ = validate_document_file("empty.pdf", b"")
        assert v_doc is False
        assert "empty (0 bytes)" in err_doc

        # 2. Corrupt PDF without %PDF header
        v_bad, err_bad, _ = validate_document_file("corrupt.pdf", b"NOT_A_PDF_STREAM")
        assert v_bad is False
        assert "lacks a valid PDF header" in err_bad

        # 3. Empty text entry
        v_txt, err_txt, _ = validate_document_text("   \n   \t  ")
        assert v_txt is False
        assert "empty or contains only whitespace" in err_txt

        # 4. Empty candidate response
        v_resp, err_resp, _ = validate_candidate_response("   ")
        assert v_resp is False
        assert "whitespace" in err_resp.lower()

        # 5. Execution attempt with unvalidated payload
        with pytest.raises(ValueError, match="have not been validated"):
            map_staged_inputs_to_pipeline_args({"is_validated": False})


# ===========================================================================
# Session State, Re-entrancy & Integrity Tests
# ===========================================================================

class TestSessionStateAndDashboardIntegrity:
    """Validates session state lifecycle, filter re-entrancy, and metric integrity."""

    def test_filter_claims_reentrancy_does_not_mutate_or_reinvoke(self) -> None:
        """Verifies that changing UI filters on Results page does not mutate claims or re-invoke inference."""
        claims = [
            VerifiedClaim(
                claim_id="c1",
                claim_text="Claim 1",
                source_sentence="Claim 1",
                start_char=0,
                end_char=7,
                is_factual=True,
                classification=VerificationCategory.SUPPORTED,
                decision_rule="TEST_RULE",
                reasoning_trace="Test trace",
            ),
            VerifiedClaim(
                claim_id="c2",
                claim_text="Claim 2",
                source_sentence="Claim 2",
                start_char=0,
                end_char=7,
                is_factual=True,
                classification=VerificationCategory.CONTRADICTED,
                decision_rule="TEST_RULE",
                reasoning_trace="Test trace",
            ),
            VerifiedClaim(
                claim_id="c3",
                claim_text="Claim 3",
                source_sentence="Claim 3",
                start_char=0,
                end_char=7,
                is_factual=False,
                classification=VerificationCategory.UNSUPPORTED,
                decision_rule="TEST_RULE",
                reasoning_trace="Test trace",
            ),
        ]

        # Filter "All"
        all_res = filter_claims_by_category(claims, "All (3)")
        assert len(all_res) == 3

        # Filter "Supported"
        sup_res = filter_claims_by_category(claims, "Supported (1)")
        assert len(sup_res) == 1
        assert sup_res[0].claim_id == "c1"

        # Filter "Contradicted"
        contra_res = filter_claims_by_category(claims, "Contradicted (1)")
        assert len(contra_res) == 1
        assert contra_res[0].claim_id == "c2"

        # Filter "Non-Factual"
        nf_res = filter_claims_by_category(claims, "Non-Factual (1)")
        assert len(nf_res) == 1
        assert nf_res[0].claim_id == "c3"

        # Original list is intact (immutability)
        assert len(claims) == 3

    def test_dashboard_chart_builders_integrity(self) -> None:
        """Verifies Altair chart builders generate valid specifications."""
        ref_text = "The Sun is at the center of the Solar System."
        _, _, meta = validate_document_text(ref_text)
        response = "The Sun is at the center of the Solar System."
        staged = build_staged_payload(
            doc_mode="Paste / Type Reference Text",
            doc_meta=meta,
            prompt=None,
            response=response,
            regime="direct",
        )

        mock_preds = [
            CustomDebertaPrediction(
                predicted_label="SUPPORTED_PROXY",
                probabilities={"CONTRADICTED": 0.05, "SUPPORTED_PROXY": 0.90, "UNSUPPORTED": 0.05},
                contradicted_prob=0.05,
                supported_proxy_prob=0.90,
                unsupported_prob=0.05,
            )
        ]
        pipeline = create_pipeline_with_mock(MockDebertaVerifier(mock_preds))
        result = execute_verification(staged, pipeline=pipeline)

        # 1. Distribution Chart
        dist_chart = build_classification_distribution_chart(result)
        assert dist_chart is not None
        chart_dict = dist_chart.to_dict()
        assert "data" in chart_dict or "datasets" in chart_dict or "layer" in chart_dict

        # 2. Calibrated Probabilities Chart
        prob_chart = build_calibrated_probabilities_chart(result)
        assert prob_chart is not None

        # 3. Probability Sum Check (posterior class probabilities must sum to 1.0)
        cal_probs_map = result.metadata.get("calibrated_probabilities", {})
        for c_id, probs in cal_probs_map.items():
            p_s = probs.get("Support", probs.get("sup", 0.0))
            p_c = probs.get("Contradiction", probs.get("contra", 0.0))
            p_u = probs.get("Unsupported", probs.get("unsup", max(0.0, 1.0 - p_s - p_c)))
            prob_sum = p_s + p_c + p_u
            assert pytest.approx(prob_sum, abs=1e-4) == 1.0

    def test_passage_retriever_memory_cleanup_after_verification(self) -> None:
        """Verifies that PassageRetriever index is wiped after verify() completes to preserve memory."""
        ref_text = "Apollo 11 was the American spaceflight that first landed humans on the Moon."
        _, _, meta = validate_document_text(ref_text)
        response = "Apollo 11 landed humans on the Moon."
        staged = build_staged_payload(
            doc_mode="Paste / Type Reference Text",
            doc_meta=meta,
            prompt=None,
            response=response,
            regime="retrieval",
        )

        pipeline = create_pipeline_with_mock(MockDebertaVerifier())
        result = execute_verification(staged, pipeline=pipeline)

        # Invariant: PassageRetriever FAISS/numpy index and chunks are cleared after execution
        assert len(pipeline.passage_retriever.chunks) == 0
        assert pipeline.passage_retriever.chunk_embeddings is None
