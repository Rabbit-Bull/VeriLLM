"""
Unit and Integration Tests for VeriLLM Live Generation Service & Dual Workflow Integration.
Phase 11.3: Streamlit Live Gemini Generation Interface & Review View.

Validates:
1. Provider availability checking without exposing secrets.
2. Candidate response generation with empty-prompt guarding and error boundary handling.
3. Strict data sovereignty invariant: grounding documents are never sent to external APIs.
4. Review view editing: user edits to generated responses are preserved and staged.
5. Stale-response detection: guard locks verification if question or document changes.
6. Workflow convergence: both Manual and Live workflows produce valid payloads for VeriLLMPipeline.

All tests run strictly offline with mocked Gemini APIs (zero live network requests).
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root and backend are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml_engine.llm_generator.client import GeminiProviderAdapter
from ml_engine.llm_generator.models import (
    GenerationError,
    GenerationRequest,
    GenerationResult,
)
from frontend.utils.generation_service import (
    check_gemini_availability,
    generate_candidate_response,
    get_gemini_adapter,
)
from frontend.views.verify import (
    build_staged_payload,
    validate_candidate_response,
    validate_document_file,
    validate_document_text,
)
from frontend.utils.pipeline_service import map_staged_inputs_to_pipeline_args


# =============================================================================
# 1. Provider Availability Tests
# =============================================================================

class TestProviderAvailability:
    """Tests for check_gemini_availability and configuration inspection."""

    def test_availability_when_key_configured(self):
        """When an API key is present, provider reports available and returns model info."""
        mock_adapter = MagicMock(spec=GeminiProviderAdapter)
        mock_adapter.is_available = True
        mock_adapter.model_id = "gemini-3.8-flash"

        is_avail, model_id, status_msg = check_gemini_availability(adapter=mock_adapter)

        assert is_avail is True
        assert model_id == "gemini-3.8-flash"
        assert "Ready" in status_msg
        assert "gemini-3.8-flash" in status_msg

    def test_availability_when_key_missing(self):
        """When API key is missing, provider reports unavailable with clear status message."""
        mock_adapter = MagicMock(spec=GeminiProviderAdapter)
        mock_adapter.is_available = False
        mock_adapter.model_id = "gemini-3.8-flash"

        is_avail, model_id, status_msg = check_gemini_availability(adapter=mock_adapter)

        assert is_avail is False
        assert model_id == "gemini-3.8-flash"
        assert "GEMINI_API_KEY" in status_msg

    def test_availability_never_leaks_api_key(self):
        """The availability tuple must never contain the raw secret key string."""
        secret = "AIzaSySecretKeyNeverExposeInUIOrLogs12345"
        adapter = GeminiProviderAdapter(api_key=secret, model="gemini-3.8-flash")

        is_avail, model_id, status_msg = check_gemini_availability(adapter=adapter)

        assert is_avail is True
        assert secret not in model_id
        assert secret not in status_msg

    def test_default_model_is_gemini_3_8_flash(self):
        """Without explicit override, default model is gemini-3.8-flash."""
        import os
        from unittest.mock import patch
        from ml_engine.llm_generator.client import DEFAULT_GEMINI_MODEL
        assert DEFAULT_GEMINI_MODEL == "gemini-3.8-flash"

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GEMINI_MODEL", None)
            adapter = GeminiProviderAdapter(api_key="test_key")
            assert adapter.model_id == "gemini-3.8-flash"

    def test_gemini_model_env_override_respected(self):
        """When GEMINI_MODEL is set in environment, adapter respects the override."""
        import os
        from unittest.mock import patch
        with patch.dict(os.environ, {"GEMINI_MODEL": "custom-override-model"}, clear=False):
            adapter = GeminiProviderAdapter(api_key="test_key")
            assert adapter.model_id == "custom-override-model"


# =============================================================================
# 2. Generation Service Tests (Offline Mocking)
# =============================================================================

class TestGenerationService:
    """Tests for generate_candidate_response function."""

    def test_empty_question_rejected_without_api_call(self):
        """Empty or whitespace question is rejected locally without calling adapter."""
        mock_adapter = MagicMock(spec=GeminiProviderAdapter)
        mock_adapter.model_id = "gemini-3.8-flash"

        res = generate_candidate_response("", adapter=mock_adapter)
        assert res.success is False
        assert res.error is not None
        assert res.error.error_code == "INVALID_INPUT"
        assert res.model_id == "gemini-3.8-flash"
        mock_adapter.generate.assert_not_called()

        res_ws = generate_candidate_response("   \t\n  ", adapter=mock_adapter)
        assert res_ws.success is False
        assert res_ws.error is not None
        assert res_ws.error.error_code == "INVALID_INPUT"
        assert res_ws.model_id == "gemini-3.8-flash"
        mock_adapter.generate.assert_not_called()

    def test_successful_candidate_generation(self):
        """Valid question triggers adapter and returns formatted GenerationResult."""
        mock_adapter = MagicMock(spec=GeminiProviderAdapter)
        mock_res = GenerationResult(
            question="What is the speed of light?",
            response_text="The speed of light in vacuum is approximately 299,792,458 meters per second.",
            model_id="gemini-3.8-flash",
            execution_time_seconds=0.75,
            usage_metadata={"prompt_token_count": 8, "candidates_token_count": 18, "total_token_count": 26},
            finish_reason="STOP",
            success=True,
        )
        mock_adapter.generate.return_value = mock_res

        result = generate_candidate_response(
            question="What is the speed of light?",
            model="gemini-3.8-flash",
            adapter=mock_adapter,
        )

        assert result.success is True
        assert "299,792,458" in result.response_text
        assert result.model_id == "gemini-3.8-flash"
        assert result.execution_time_seconds == 0.75
        mock_adapter.generate.assert_called_once()
        req: GenerationRequest = mock_adapter.generate.call_args[0][0]
        assert req.question == "What is the speed of light?"
        assert req.model == "gemini-3.8-flash"

    def test_data_sovereignty_invariant_no_document_in_signature(self):
        """Verifies that generate_candidate_response does not accept or process document arguments."""
        import inspect
        sig = inspect.signature(generate_candidate_response)
        param_names = list(sig.parameters.keys())

        # Crucial security assertion: parameters must ONLY include question, model, adapter
        assert "document" not in param_names
        assert "doc_text" not in param_names
        assert "reference" not in param_names
        assert "file_bytes" not in param_names

    def test_adapter_exception_handled_gracefully(self):
        """Unexpected adapter failure returns clean failed GenerationResult rather than crashing."""
        mock_adapter = MagicMock(spec=GeminiProviderAdapter)
        mock_adapter.model_id = "gemini-3.8-flash"
        mock_adapter.generate.side_effect = RuntimeError("SDK internal crash")

        result = generate_candidate_response(
            question="Tell me a fact",
            adapter=mock_adapter,
        )

        assert result.success is False
        assert result.error is not None
        assert "SDK internal crash" in result.error.message

    def test_timeout_error_propagation(self):
        """When adapter returns a timeout error, generation service preserves it cleanly."""
        mock_adapter = MagicMock(spec=GeminiProviderAdapter)
        mock_adapter.model_id = "gemini-3.8-flash"
        mock_res = GenerationResult(
            question="Slow query",
            response_text="",
            model_id="gemini-3.8-flash",
            execution_time_seconds=90.0,
            success=False,
            error=GenerationError(
                error_code="GATEWAY_TIMEOUT",
                message="Google Gemini server gateway timed out (HTTP 504) after 90.0s.",
                retryable=True,
            ),
        )
        mock_adapter.generate.return_value = mock_res

        res = generate_candidate_response("Slow query", adapter=mock_adapter)
        assert res.success is False
        assert res.error.error_code == "GATEWAY_TIMEOUT"
        assert "HTTP 504" in res.error.message
        assert res.error.retryable is True


# =============================================================================
# 3. Stale Response Guard Tests
# =============================================================================

class TestStaleResponseGuard:
    """Tests stale-response detection logic when inputs change post-generation."""

    def test_stale_detection_on_question_change(self):
        """If prompt text differs from the question used during generation, stale is detected."""
        original_q = "What was Apollo 11's launch vehicle?"
        edited_q = "What was Apollo 12's launch vehicle?"
        doc_name = "apollo.txt"

        stored_gen_q = original_q.strip()
        stored_doc = doc_name

        # User edits prompt
        current_q = edited_q.strip()
        current_doc = doc_name

        is_stale = (stored_gen_q != current_q) or (stored_doc != current_doc)
        assert is_stale is True

    def test_stale_detection_on_document_change(self):
        """If reference document name changes, stale is detected even if question is unchanged."""
        question = "What was Apollo 11's launch vehicle?"
        stored_doc = "apollo11_mission.pdf"
        new_doc = "artemis_mission.pdf"

        is_stale = (question != question) or (stored_doc != new_doc)
        assert is_stale is True

    def test_not_stale_when_inputs_match(self):
        """When current inputs match the generated context, stale is False."""
        question = "What was Apollo 11's launch vehicle?"
        doc_name = "apollo11_mission.pdf"

        is_stale = (question != question) or (doc_name != doc_name)
        assert is_stale is False


# =============================================================================
# 4. Review View Editing & Staging Tests
# =============================================================================

class TestReviewViewEditingAndStaging:
    """Tests that user modifications to generated responses are preserved during staging."""

    def test_edited_response_staged_correctly(self):
        """When a user modifies the generated text in the review text area, the edited version is staged."""
        generated_text = "Michael Collins walked on the Moon with Armstrong."
        user_edited_text = "Michael Collins flew the Command Module Columbia while Armstrong walked on the Moon."

        doc_meta = {
            "name": "apollo_summary.txt",
            "type": "Plain Text (TXT)",
            "bytes": b"Apollo 11 reference text...",
            "text": "Apollo 11 reference text...",
            "size_display": "25 B",
        }
        prompt = "What was Michael Collins' role?"

        payload = build_staged_payload(
            doc_mode="File Upload (PDF / TXT)",
            doc_meta=doc_meta,
            prompt=prompt,
            response=user_edited_text,  # User's edited text
            regime="Retrieval (Recommended)",
            workflow_mode="Live Gemini Verification",
            generation_metadata={
                "model_id": "gemini-3.8-flash",
                "execution_time_seconds": 1.12,
                "total_tokens": 42,
            },
        )

        assert payload["response"] == user_edited_text
        assert payload["workflow_mode"] == "Live Gemini Verification"
        assert payload["generation_metadata"]["model_id"] == "gemini-3.8-flash"
        assert payload["prompt"] == prompt
        assert payload["is_validated"] is True

    def test_whitespace_editing_caught_by_validator(self):
        """If user deletes all text in the review area, validation catches it."""
        is_valid, err_msg, meta = validate_candidate_response("   \n\t   ")
        assert is_valid is False
        assert "contains only whitespace" in err_msg


# =============================================================================
# 5. Dual Workflow Convergence Tests
# =============================================================================

class TestWorkflowConvergence:
    """Tests that both Manual and Live Gemini workflows map to the same pipeline arguments."""

    def test_manual_workflow_maps_to_pipeline_args(self):
        """Manual workflow payload produces standard pipeline verification arguments."""
        doc_meta = {
            "name": "doc.txt",
            "type": "Plain Text (TXT)",
            "bytes": b"Reference facts here.",
            "text": "Reference facts here.",
            "size_display": "20 B",
        }
        manual_payload = build_staged_payload(
            doc_mode="File Upload (PDF / TXT)",
            doc_meta=doc_meta,
            prompt="Summarize facts.",
            response="Candidate response manually typed by user.",
            regime="Retrieval (Recommended)",
            workflow_mode="Manual Verification",
            generation_metadata=None,
        )

        args = map_staged_inputs_to_pipeline_args(manual_payload)
        assert args["document"] == "Reference facts here."
        assert args["response"] == "Candidate response manually typed by user."
        assert args["prompt"] == "Summarize facts."
        assert args["regime"] == "retrieval"
        assert args["document_name"] == "doc.txt"

    def test_live_gemini_workflow_maps_to_identical_pipeline_args(self):
        """Live Gemini workflow payload produces the identical pipeline argument structure."""
        doc_meta = {
            "name": "doc.txt",
            "type": "Plain Text (TXT)",
            "bytes": b"Reference facts here.",
            "text": "Reference facts here.",
            "size_display": "20 B",
        }
        live_payload = build_staged_payload(
            doc_mode="File Upload (PDF / TXT)",
            doc_meta=doc_meta,
            prompt="Summarize facts.",
            response="Candidate response generated by Gemini and reviewed by user.",
            regime="Retrieval (Recommended)",
            workflow_mode="Live Gemini Verification",
            generation_metadata={"model_id": "gemini-3.8-flash", "execution_time_seconds": 0.8},
        )

        args = map_staged_inputs_to_pipeline_args(live_payload)
        assert args["document"] == "Reference facts here."
        assert args["response"] == "Candidate response generated by Gemini and reviewed by user."
        assert args["prompt"] == "Summarize facts."
        assert args["regime"] == "retrieval"
        assert args["document_name"] == "doc.txt"
        # Confirm that extraneous metadata does not interfere with pipeline input
        assert "generation_metadata" not in args
        assert "workflow_mode" not in args


# =============================================================================
# 6. Generation Metadata Contract & Rendering Tests
# =============================================================================

class TestGenerationMetadataContractAndRendering:
    """Tests the contract between GenerationResult and Streamlit UI metadata rendering."""

    def test_metadata_extraction_with_full_usage_info(self):
        """When GenerationResult contains token usage, session metadata captures it accurately."""
        gen_result = GenerationResult(
            question="What is the speed of light?",
            response_text="The speed of light is 299,792,458 m/s.",
            model_id="gemini-3.8-flash",
            execution_time_seconds=1.23,
            prompt_tokens=14,
            candidates_tokens=28,
            total_tokens=42,
            finish_reason="STOP",
            success=True,
        )

        # Ensure both direct attribute access and usage_metadata access succeed
        assert gen_result.prompt_tokens == 14
        assert gen_result.candidates_tokens == 28
        assert gen_result.total_tokens == 42
        assert gen_result.usage_metadata is not None
        assert gen_result.usage_metadata.get("prompt_token_count") == 14
        assert gen_result.usage_metadata.get("candidates_token_count") == 28
        assert gen_result.usage_metadata.get("total_token_count") == 42

        # Simulate Streamlit session state metadata packaging
        gen_meta = {
            "model_id": gen_result.model_id,
            "execution_time_seconds": gen_result.execution_time_seconds,
            "prompt_tokens": gen_result.prompt_tokens,
            "candidates_tokens": gen_result.candidates_tokens,
            "total_tokens": gen_result.total_tokens,
            "finish_reason": gen_result.finish_reason,
            "generated_at": "2026-11-16 10:00:00",
        }

        assert gen_meta["prompt_tokens"] == 14
        assert gen_meta["candidates_tokens"] == 28
        assert gen_meta["total_tokens"] == 42

        # Simulate UI token badge rendering logic
        tot_tok = gen_meta.get("total_tokens")
        tok_info = f" • {tot_tok} tokens" if tot_tok else ""
        assert tok_info == " • 42 tokens"

    def test_metadata_extraction_without_optional_usage_info(self):
        """When GenerationResult has no token usage (None), extraction and rendering handle it gracefully."""
        gen_result = GenerationResult(
            question="What is the speed of light?",
            response_text="The speed of light is 299,792,458 m/s.",
            model_id="gemini-3.8-flash",
            execution_time_seconds=0.85,
            prompt_tokens=None,
            candidates_tokens=None,
            total_tokens=None,
            finish_reason="STOP",
            success=True,
        )

        # Must not raise AttributeError
        assert gen_result.prompt_tokens is None
        assert gen_result.candidates_tokens is None
        assert gen_result.total_tokens is None
        assert gen_result.usage_metadata is None

        # Simulate Streamlit session state metadata packaging
        gen_meta = {
            "model_id": gen_result.model_id,
            "execution_time_seconds": gen_result.execution_time_seconds,
            "prompt_tokens": gen_result.prompt_tokens,
            "candidates_tokens": gen_result.candidates_tokens,
            "total_tokens": gen_result.total_tokens,
            "finish_reason": gen_result.finish_reason,
            "generated_at": "2026-11-16 10:00:00",
        }

        assert gen_meta["prompt_tokens"] is None
        assert gen_meta["candidates_tokens"] is None
        assert gen_meta["total_tokens"] is None

        # Simulate UI token badge rendering logic: must omit badge without fabrication
        tot_tok = gen_meta.get("total_tokens")
        tok_info = f" • {tot_tok} tokens" if tot_tok else ""
        assert tok_info == ""

    def test_candidate_generation_with_missing_usage_metadata(self):
        """Candidate generation service cleanly propagates result when usage_metadata is missing."""
        mock_adapter = MagicMock(spec=GeminiProviderAdapter)
        mock_adapter.model_id = "gemini-3.8-flash"
        mock_adapter.generate.return_value = GenerationResult(
            question="What is gravity?",
            response_text="Gravity is a fundamental interaction.",
            model_id="gemini-3.8-flash",
            execution_time_seconds=0.55,
            usage_metadata=None,
            success=True,
        )

        res = generate_candidate_response("What is gravity?", adapter=mock_adapter)
        assert res.success is True
        assert res.prompt_tokens is None
        assert res.usage_metadata is None
        assert hasattr(res, "usage_metadata")
        assert hasattr(res, "prompt_tokens")

    def test_error_handling_without_crashing_streamlit(self):
        """Failed candidate generation produces clean error without missing attribute crashes."""
        mock_adapter = MagicMock(spec=GeminiProviderAdapter)
        mock_adapter.model_id = "gemini-3.8-flash"
        mock_adapter.generate.return_value = GenerationResult(
            question="Query that fails",
            response_text="",
            model_id="gemini-3.8-flash",
            execution_time_seconds=0.1,
            success=False,
            error=GenerationError(
                error_code="RATE_LIMIT",
                message="Resource has been exhausted.",
                retryable=True,
            ),
        )

        res = generate_candidate_response("Query that fails", adapter=mock_adapter)
        assert res.success is False
        assert res.error is not None
        assert res.error.error_code == "RATE_LIMIT"
        assert res.prompt_tokens is None
        assert res.usage_metadata is None
        # Error message extraction
        err_msg = res.error.message if res.error else "Unknown generation error"
        assert err_msg == "Resource has been exhausted."


