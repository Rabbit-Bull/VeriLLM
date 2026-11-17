"""
Integration tests for frontend/utils/pipeline_service.py.
Tests input mapping for PDF, TXT, and direct-text modes, regime normalization,
pipeline invocation, exception handling, and real end-to-end execution with verillm_deberta.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure project root and backend are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml_engine.pipeline import PipelineResult, VeriLLMPipeline
from frontend.utils.pipeline_service import (
    execute_verification,
    map_staged_inputs_to_pipeline_args,
)


class TestMapStagedInputs:
    """Tests converting staged session-state payloads to VeriLLMPipeline.verify() arguments."""

    def test_map_pdf_file_input(self):
        pdf_bytes = b"%PDF-1.4\nsome pdf stream content\n%%EOF"
        prepared_inputs = {
            "doc_mode": "File Upload (PDF / TXT)",
            "doc_name": "annual_report.pdf",
            "doc_type": "PDF Document",
            "doc_bytes": pdf_bytes,
            "doc_text": None,
            "doc_size_display": "38 B",
            "prompt": "What were the revenues?",
            "response": "The annual revenue grew by 15 percent in fiscal year 2023.",
            "regime": "retrieval",
            "is_validated": True,
            "staged_at": "2026-11-13 12:00:00",
        }

        args = map_staged_inputs_to_pipeline_args(prepared_inputs)

        assert args["document"] == pdf_bytes
        assert args["document_name"] == "annual_report.pdf"
        assert args["response"] == "The annual revenue grew by 15 percent in fiscal year 2023."
        assert args["prompt"] == "What were the revenues?"
        assert args["regime"] == "retrieval"

    def test_map_txt_file_with_text(self):
        txt_content = "Mars is the fourth planet from the Sun."
        prepared_inputs = {
            "doc_mode": "File Upload (PDF / TXT)",
            "doc_name": "mars.txt",
            "doc_type": "Plain Text (TXT)",
            "doc_bytes": txt_content.encode("utf-8"),
            "doc_text": txt_content,
            "doc_size_display": "38 B",
            "prompt": None,
            "response": "Mars is the fourth planet from the Sun.",
            "regime": "direct",
            "is_validated": True,
            "staged_at": "2026-11-13 12:00:00",
        }

        args = map_staged_inputs_to_pipeline_args(prepared_inputs)

        assert args["document"] == txt_content
        assert args["document_name"] == "mars.txt"
        assert args["response"] == "Mars is the fourth planet from the Sun."
        assert args["prompt"] is None
        assert args["regime"] == "direct"

    def test_map_direct_text_entry(self):
        raw_text = "Jupiter has 95 officially recognized moons."
        prepared_inputs = {
            "doc_mode": "Direct Text Entry",
            "doc_name": "Pasted Reference Text",
            "doc_type": "Direct Text Entry",
            "doc_bytes": None,
            "doc_text": raw_text,
            "doc_size_display": "7 words",
            "prompt": "   How many moons does Jupiter have?   ",
            "response": "Jupiter has 95 moons.",
            "regime": "Retrieval (Recommended)",
            "is_validated": True,
            "staged_at": "2026-11-13 12:00:00",
        }

        args = map_staged_inputs_to_pipeline_args(prepared_inputs)

        assert args["document"] == raw_text
        assert args["document_name"] == "Pasted Reference Text"
        assert args["prompt"] == "How many moons does Jupiter have?"
        assert args["regime"] == "retrieval"

    def test_missing_prepared_inputs_raises_value_error(self):
        with pytest.raises(ValueError, match="No staged inputs found"):
            map_staged_inputs_to_pipeline_args({})

        with pytest.raises(ValueError, match="No staged inputs found"):
            map_staged_inputs_to_pipeline_args(None)  # type: ignore[arg-type]

    def test_unvalidated_inputs_raises_value_error(self):
        with pytest.raises(ValueError, match="have not been validated"):
            map_staged_inputs_to_pipeline_args({
                "is_validated": False,
                "response": "test",
            })

    def test_empty_response_raises_value_error(self):
        with pytest.raises(ValueError, match="Candidate LLM response cannot be empty"):
            map_staged_inputs_to_pipeline_args({
                "is_validated": True,
                "doc_mode": "Direct Text Entry",
                "doc_text": "Ground text",
                "response": "   \n\t   ",
            })

    def test_empty_doc_bytes_raises_value_error(self):
        with pytest.raises(ValueError, match="empty \\(0 bytes\\)"):
            map_staged_inputs_to_pipeline_args({
                "is_validated": True,
                "doc_mode": "File Upload (PDF / TXT)",
                "doc_name": "empty.pdf",
                "doc_type": "PDF Document",
                "doc_bytes": b"",
                "response": "Valid response",
            })

    def test_empty_direct_text_raises_value_error(self):
        with pytest.raises(ValueError, match="Reference document text cannot be empty"):
            map_staged_inputs_to_pipeline_args({
                "is_validated": True,
                "doc_mode": "Direct Text Entry",
                "doc_name": "Pasted Reference Text",
                "doc_text": "   ",
                "response": "Valid response",
            })


class TestExecuteVerificationMocked:
    """Tests executing verification with an injected/mock pipeline."""

    def test_execute_verification_success(self):
        mock_pipeline = MagicMock(spec=VeriLLMPipeline)
        mock_result = MagicMock(spec=PipelineResult)
        mock_result.document_name = "test.txt"
        mock_result.regime = "retrieval"
        mock_pipeline.verify.return_value = mock_result

        prepared_inputs = {
            "doc_mode": "Direct Text Entry",
            "doc_name": "test.txt",
            "doc_type": "Direct Text Entry",
            "doc_bytes": None,
            "doc_text": "Earth orbits the Sun.",
            "doc_size_display": "4 words",
            "prompt": None,
            "response": "Earth is in orbit around the Sun.",
            "regime": "retrieval",
            "is_validated": True,
            "staged_at": "2026-11-13 12:00:00",
        }

        result = execute_verification(prepared_inputs, pipeline=mock_pipeline)

        assert result == mock_result
        mock_pipeline.verify.assert_called_once_with(
            document="Earth orbits the Sun.",
            response="Earth is in orbit around the Sun.",
            prompt=None,
            regime="retrieval",
            document_name="test.txt",
        )

    def test_execute_verification_propagates_pipeline_error(self):
        mock_pipeline = MagicMock(spec=VeriLLMPipeline)
        mock_pipeline.verify.side_effect = ValueError("Scanned PDF detected in 'scanned.pdf'")

        prepared_inputs = {
            "doc_mode": "File Upload (PDF / TXT)",
            "doc_name": "scanned.pdf",
            "doc_type": "PDF Document",
            "doc_bytes": b"%PDF-1.4...",
            "doc_text": None,
            "doc_size_display": "100 B",
            "prompt": None,
            "response": "Some response.",
            "regime": "retrieval",
            "is_validated": True,
            "staged_at": "2026-11-13 12:00:00",
        }

        with pytest.raises(ValueError, match="Scanned PDF detected"):
            execute_verification(prepared_inputs, pipeline=mock_pipeline)


class TestRealLocalPipelineIntegration:
    """Executes a real local end-to-end verification run using the actual fine-tuned DeBERTa model."""

    @pytest.mark.slow
    def test_real_end_to_end_verification(self):
        """Runs genuine neural verification with verillm_deberta and frozen calibration."""
        reference_doc = (
            "The Apollo 11 spacecraft launched from Kennedy Space Center on July 16, 1969. "
            "Commander Neil Armstrong and Lunar Module Pilot Buzz Aldrin landed the Apollo "
            "Lunar Module Eagle on the Moon on July 20, 1969. Michael Collins remained in lunar orbit."
        )
        candidate_response = (
            "Apollo 11 launched on July 16, 1969. "
            "Neil Armstrong and Buzz Aldrin landed on the Moon on July 20, 1969."
        )

        prepared_inputs = {
            "doc_mode": "Direct Text Entry",
            "doc_name": "apollo11_factsheet.txt",
            "doc_type": "Direct Text Entry",
            "doc_bytes": None,
            "doc_text": reference_doc,
            "doc_size_display": "43 words",
            "prompt": "When did Apollo 11 land on the Moon?",
            "response": candidate_response,
            "regime": "retrieval",
            "is_validated": True,
            "staged_at": "2026-11-13 12:00:00",
        }

        # Initialize real pipeline
        pipeline = VeriLLMPipeline(default_regime="retrieval")
        result = execute_verification(prepared_inputs, pipeline=pipeline)

        assert isinstance(result, PipelineResult)
        assert result.document_name == "apollo11_factsheet.txt"
        assert result.regime == "retrieval"
        assert result.prompt == "When did Apollo 11 land on the Moon?"
        assert result.execution_time_seconds > 0.0

        # Validate claims extraction and classification
        assert len(result.claims) >= 1
        assert result.score.counts.factual_claims >= 1
        assert result.score.indices.grounded_factuality_index is not None
        assert 0.0 <= result.score.indices.grounded_factuality_index <= 100.0

        # Supported claims should be verified by the fine-tuned model
        assert result.score.counts.supported_count >= 1
        assert result.score.counts.contradicted_count == 0

