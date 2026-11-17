"""
Unit tests for VeriLLM Frontend Verify Response Input Interface.
Tests document validation (PDF/TXT), plain-text mode, candidate response validation,
regime selection, and payload staging logic.
"""

import sys
from pathlib import Path

# Ensure project root and backend are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import pytest
from unittest.mock import MagicMock
from frontend.views.verify import (
    format_file_size,
    validate_document_file,
    validate_document_text,
    validate_candidate_response,
    build_staged_payload,
    _init_session_state,
)


class TestFileSizeFormatting:
    """Tests for human-readable file size conversion."""

    def test_format_bytes(self):
        assert format_file_size(0) == "0 B"
        assert format_file_size(512) == "512 B"
        assert format_file_size(1023) == "1023 B"

    def test_format_kilobytes(self):
        assert format_file_size(1024) == "1.0 KB"
        assert format_file_size(2048) == "2.0 KB"
        assert format_file_size(15360) == "15.0 KB"

    def test_format_megabytes(self):
        assert format_file_size(1024 * 1024) == "1.00 MB"
        assert format_file_size(int(2.5 * 1024 * 1024)) == "2.50 MB"


class TestDocumentFileValidation:
    """Tests for uploaded document file validation (PDF and TXT)."""

    def test_valid_pdf_file(self):
        pdf_bytes = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF"
        is_valid, err_msg, meta = validate_document_file("research_paper.pdf", pdf_bytes)

        assert is_valid is True
        assert err_msg is None
        assert meta["name"] == "research_paper.pdf"
        assert meta["type"] == "PDF Document"
        assert meta["bytes"] == pdf_bytes
        assert meta["text"] is None
        assert "B" in meta["size_display"] or "KB" in meta["size_display"]

    def test_invalid_pdf_missing_magic_header(self):
        invalid_pdf_bytes = b"NOT_A_REAL_PDF_HEADER\nSome random content"
        is_valid, err_msg, meta = validate_document_file("corrupt.pdf", invalid_pdf_bytes)

        assert is_valid is False
        assert "lacks a valid PDF header" in err_msg
        assert meta["type"] == "PDF Document"

    def test_valid_txt_file(self):
        content = "The Apollo program was the third United States human spaceflight program."
        txt_bytes = content.encode("utf-8")
        is_valid, err_msg, meta = validate_document_file("apollo.txt", txt_bytes)

        assert is_valid is True
        assert err_msg is None
        assert meta["name"] == "apollo.txt"
        assert meta["type"] == "Plain Text (TXT)"
        assert meta["text"] == content
        assert meta["words"] == len(content.split())

    def test_txt_file_with_invalid_utf8(self):
        bad_bytes = b"Invalid utf-8 sequence \x80\x81\xff\xfe"
        is_valid, err_msg, meta = validate_document_file("bad_encoding.txt", bad_bytes)

        assert is_valid is False
        assert "could not be decoded as UTF-8" in err_msg

    def test_empty_file_rejected(self):
        is_valid, err_msg, meta = validate_document_file("empty.pdf", b"")
        assert is_valid is False
        assert "empty (0 bytes)" in err_msg

        is_valid_txt, err_txt, _ = validate_document_file("empty.txt", b"")
        assert is_valid_txt is False
        assert "empty (0 bytes)" in err_txt

    def test_whitespace_only_txt_file_rejected(self):
        is_valid, err_msg, meta = validate_document_file("spaces.txt", b"   \n\t  \n  ")
        assert is_valid is False
        assert "contains only whitespace" in err_msg

    def test_unsupported_file_extension(self):
        is_valid, err_msg, meta = validate_document_file("image.png", b"\x89PNG\r\n\x1a\n")
        assert is_valid is False
        assert "Unsupported file extension '.png'" in err_msg
        assert "Only .pdf and .txt are accepted" in err_msg

        is_valid_doc, err_doc, _ = validate_document_file("notes.docx", b"PK\x03\x04")
        assert is_valid_doc is False
        assert "Unsupported file extension '.docx'" in err_doc


class TestDocumentTextValidation:
    """Tests for direct plain-text document input."""

    def test_valid_plain_text(self):
        text = "Voyager 1 is a space probe launched by NASA on September 5, 1977."
        is_valid, err_msg, meta = validate_document_text(text)

        assert is_valid is True
        assert err_msg is None
        assert meta["type"] == "Direct Text Entry"
        assert meta["words"] == len(text.split())
        assert meta["chars"] == len(text)
        assert meta["text"] == text

    def test_empty_plain_text_rejected(self):
        is_valid, err_msg, meta = validate_document_text("")
        assert is_valid is False
        assert "empty" in err_msg
        assert meta["words"] == 0

    def test_whitespace_only_plain_text_rejected(self):
        is_valid, err_msg, meta = validate_document_text("   \n\n\t   ")
        assert is_valid is False
        assert "whitespace" in err_msg
        assert meta["words"] == 0


class TestCandidateResponseValidation:
    """Tests for candidate LLM response input validation."""

    def test_valid_response(self):
        response = (
            "Apollo 11 landed on the Moon on July 20, 1969. "
            "Neil Armstrong was the commander and Buzz Aldrin was the lunar module pilot."
        )
        is_valid, err_msg, meta = validate_candidate_response(response)

        assert is_valid is True
        assert err_msg is None
        assert meta["words"] == len(response.split())
        assert meta["chars"] == len(response)
        assert meta["clean_text"] == response

    def test_empty_response_rejected(self):
        is_valid, err_msg, meta = validate_candidate_response("")
        assert is_valid is False
        assert "Candidate LLM response is required" in err_msg
        assert meta["words"] == 0

    def test_whitespace_only_response_rejected(self):
        is_valid, err_msg, meta = validate_candidate_response("   \t  \n  ")
        assert is_valid is False
        assert "contains only whitespace" in err_msg


class TestPayloadStaging:
    """Tests for staged payload construction and schema completeness."""

    def test_build_staged_payload_retrieval(self):
        doc_meta = {
            "name": "mission_brief.pdf",
            "type": "PDF Document",
            "bytes": b"%PDF-1.4...",
            "text": None,
            "size_display": "14.5 KB",
        }
        prompt = "When did Apollo 11 land?"
        response = "Apollo 11 landed on the Moon on July 20, 1969."

        payload = build_staged_payload(
            doc_mode="File Upload (PDF / TXT)",
            doc_meta=doc_meta,
            prompt=prompt,
            response=response,
            regime="Retrieval (Recommended)",
        )

        assert payload["doc_mode"] == "File Upload (PDF / TXT)"
        assert payload["doc_name"] == "mission_brief.pdf"
        assert payload["doc_type"] == "PDF Document"
        assert payload["doc_bytes"] == b"%PDF-1.4..."
        assert payload["doc_text"] is None
        assert payload["prompt"] == prompt
        assert payload["response"] == response
        assert payload["regime"] == "retrieval"
        assert payload["is_validated"] is True
        assert "staged_at" in payload
        # Ensure no fabricated scores are present
        assert "score" not in payload
        assert "claims" not in payload

    def test_build_staged_payload_direct_mode(self):
        raw_text = "Voyager 1 is in interstellar space."
        doc_meta = {
            "name": "Pasted Reference Text",
            "type": "Direct Text Entry",
            "bytes": None,
            "text": raw_text,
            "size_display": "6 words",
        }
        response = "Voyager 1 has exited the solar system."

        payload = build_staged_payload(
            doc_mode="Direct Text Entry",
            doc_meta=doc_meta,
            prompt=None,  # No prompt
            response=response,
            regime="Direct Context",
        )

        assert payload["doc_mode"] == "Direct Text Entry"
        assert payload["doc_name"] == "Pasted Reference Text"
        assert payload["doc_bytes"] is None
        assert payload["doc_text"] == raw_text
        assert payload["prompt"] is None
        assert payload["response"] == response
        assert payload["regime"] == "direct"
        assert payload["is_validated"] is True

    def test_empty_prompt_normalized_to_none(self):
        doc_meta = {
            "name": "doc.txt",
            "type": "Plain Text (TXT)",
            "bytes": b"content",
            "text": "content",
            "size_display": "7 B",
        }
        payload = build_staged_payload(
            doc_mode="File Upload (PDF / TXT)",
            doc_meta=doc_meta,
            prompt="   \t   ",  # Whitespace prompt
            response="Test response",
            regime="retrieval",
        )
        assert payload["prompt"] is None


class TestSessionStateInitialization:
    """Tests that session state initializes properly."""

    def test_init_session_state(self, monkeypatch):
        mock_state = {}
        monkeypatch.setattr("streamlit.session_state", mock_state)

        _init_session_state()

        assert mock_state["doc_mode"] == "File Upload (PDF / TXT)"
        assert mock_state["raw_doc_text"] == ""
        assert mock_state["question_text"] == ""
        assert mock_state["response_text"] == ""
        assert mock_state["regime_choice"] == "Retrieval (Recommended)"
        assert mock_state["uploader_version"] == 0
        assert mock_state["prepared_inputs"] is None
        # Live generation keys
        assert mock_state["workflow_mode"] == "Manual Verification"
        assert mock_state["generated_response_text"] == ""
        assert mock_state["generation_metadata"] is None
        assert mock_state["generation_error"] is None
        assert mock_state["generation_error_code"] is None
        assert mock_state["generated_for_question"] is None
        assert mock_state["generated_for_doc_name"] is None
