"""
VeriLLM Pipeline Service.
Bridges the Streamlit frontend with the production VeriLLMPipeline backend.
Manages cached model initialization, input mapping, error boundary handling,
and factual verification execution.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Ensure backend and project root are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import streamlit as st
from ml_engine.pipeline import PipelineResult, VeriLLMPipeline

logger = logging.getLogger("verillm.frontend.pipeline_service")


@st.cache_resource(show_spinner=False)
def get_pipeline() -> VeriLLMPipeline:
    """Initializes and returns a cached singleton instance of VeriLLMPipeline.

    Cached via st.cache_resource so the fine-tuned DeBERTa checkpoint and SentenceTransformer
    models are loaded only once into CPU memory and reused across sessions and reruns.
    The pipeline is stateless across inference calls and safe to share.
    """
    logger.info("Initializing VeriLLMPipeline singleton via st.cache_resource...")
    return VeriLLMPipeline(default_regime="retrieval")


def map_staged_inputs_to_pipeline_args(
    prepared_inputs: Dict[str, Any]
) -> Dict[str, Any]:
    """Converts staged UI input payload into exact arguments accepted by VeriLLMPipeline.verify().

    Args:
        prepared_inputs: Canonical payload dictionary from st.session_state['prepared_inputs'].

    Returns:
        Dictionary of keyword arguments ready to be passed to pipeline.verify(**args).

    Raises:
        ValueError: If prepared_inputs is invalid, missing required fields, or unvalidated.
    """
    if not prepared_inputs:
        raise ValueError("No staged inputs found. Please provide document and candidate response.")

    if not prepared_inputs.get("is_validated"):
        raise ValueError("Staged inputs have not been validated. Please check your inputs.")

    doc_mode = prepared_inputs.get("doc_mode", "")
    doc_bytes = prepared_inputs.get("doc_bytes")
    doc_text = prepared_inputs.get("doc_text")
    doc_name = prepared_inputs.get("doc_name") or "document.txt"
    doc_type = prepared_inputs.get("doc_type", "")
    response = prepared_inputs.get("response", "")
    prompt = prepared_inputs.get("prompt")
    regime = prepared_inputs.get("regime", "retrieval")

    if not response or not response.strip():
        raise ValueError("Candidate LLM response cannot be empty or whitespace.")

    # Resolve document input based on mode and format
    if doc_mode == "File Upload (PDF / TXT)":
        if doc_bytes is None:
            raise ValueError(f"Uploaded file '{doc_name}' is missing byte content.")
        if len(doc_bytes) == 0:
            raise ValueError(f"Uploaded file '{doc_name}' is empty (0 bytes).")

        if doc_type == "PDF Document" or doc_name.lower().endswith(".pdf"):
            # Pass raw bytes (pipeline._parse_reference_document handles b"%PDF")
            document_arg: Union[str, bytes] = doc_bytes
        else:
            # Plain text file upload
            if doc_text is not None and doc_text.strip():
                document_arg = doc_text
            else:
                document_arg = doc_bytes
    else:
        # Direct text entry mode
        if not doc_text or not doc_text.strip():
            raise ValueError("Reference document text cannot be empty or whitespace.")
        document_arg = doc_text

    clean_regime = regime.lower().strip()
    if clean_regime not in ("retrieval", "direct"):
        clean_regime = "retrieval"

    clean_prompt = prompt.strip() if (prompt and prompt.strip()) else None

    return {
        "document": document_arg,
        "document_name": doc_name,
        "response": response.strip(),
        "prompt": clean_prompt,
        "regime": clean_regime,
    }


def execute_verification(
    prepared_inputs: Dict[str, Any],
    pipeline: Optional[VeriLLMPipeline] = None,
) -> PipelineResult:
    """Executes factual verification against the real VeriLLM pipeline.

    Args:
        prepared_inputs: Staged inputs from st.session_state['prepared_inputs'].
        pipeline: Optional pre-instantiated pipeline instance (defaults to get_pipeline()).

    Returns:
        PipelineResult containing verified claims, metrics, factuality indices, and provenance.

    Raises:
        ValueError: On invalid inputs, unsupported formats, or empty documents.
        FileNotFoundError: If model weights or frozen calibration bundle are missing.
        RuntimeError: If neural inference fails.
    """
    args = map_staged_inputs_to_pipeline_args(prepared_inputs)

    active_pipeline = pipeline or get_pipeline()
    logger.info(
        "Invoking VeriLLMPipeline.verify(document_name=%s, regime=%s, prompt=%s)",
        args["document_name"],
        args["regime"],
        bool(args["prompt"]),
    )

    result = active_pipeline.verify(
        document=args["document"],
        response=args["response"],
        prompt=args["prompt"],
        regime=args["regime"],
        document_name=args["document_name"],
    )

    return result

