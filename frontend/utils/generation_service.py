"""
VeriLLM Generation Service.
Bridges the Streamlit frontend with the backend GeminiProviderAdapter.
Provides candidate LLM response generation from user prompts while ensuring
grounding reference documents remain local and are never transmitted to external APIs.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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

logger = logging.getLogger("verillm.frontend.generation_service")


def get_gemini_adapter(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    timeout_seconds: Optional[float] = None,
    api_version: Optional[str] = None,
) -> GeminiProviderAdapter:
    """Instantiates a GeminiProviderAdapter resolving configuration from environment or .env."""
    return GeminiProviderAdapter(api_key=api_key, model=model, timeout_seconds=timeout_seconds, api_version=api_version)


def check_gemini_availability(
    adapter: Optional[GeminiProviderAdapter] = None,
) -> Tuple[bool, str, str]:
    """Checks whether the Gemini provider is configured and ready for live generation.

    Returns:
        Tuple of (is_available: bool, active_model_id: str, status_message: str).
        Never exposes the raw API key.
    """
    act_adapter = adapter or get_gemini_adapter()
    is_avail = act_adapter.is_available
    model_id = act_adapter.model_id

    if is_avail:
        status_msg = f"Ready • Model: {model_id} ({act_adapter.endpoint_info.get('api_version', 'v1')})"
    else:
        status_msg = "GEMINI_API_KEY environment variable is not configured."

    return is_avail, model_id, status_msg


def generate_candidate_response(
    question: str,
    model: Optional[str] = None,
    adapter: Optional[GeminiProviderAdapter] = None,
    timeout_seconds: Optional[float] = None,
    api_version: Optional[str] = None,
) -> GenerationResult:
    """Invokes Gemini to generate a candidate response for a user question.

    Crucial Architectural Invariant:
    This function accepts ONLY the question string. Reference documents are never passed
    to this service or sent to the Google Gemini API.

    Args:
        question: User prompt or query string to be answered.
        model: Optional model identifier override.
        adapter: Optional pre-instantiated GeminiProviderAdapter (allows mock injection).
        timeout_seconds: Optional request timeout in seconds override.
        api_version: Optional API version override ('v1' or 'v1beta').

    Returns:
        GenerationResult containing generated text, model ID, latency, and error tracking.
    """
    clean_q = question.strip() if question else ""
    act_adapter = adapter or get_gemini_adapter(model=model, timeout_seconds=timeout_seconds, api_version=api_version)
    raw_model = getattr(act_adapter, "model_id", None)
    effective_model = model or (raw_model if isinstance(raw_model, str) and raw_model.strip() else DEFAULT_GEMINI_MODEL)

    if not clean_q:
        return GenerationResult(
            question=question or "",
            response_text="",
            model_id=effective_model,
            execution_time_seconds=0.0,
            success=False,
            error=GenerationError(
                error_code="INVALID_INPUT",
                message="User question cannot be empty or whitespace.",
                retryable=False,
            ),
        )

    try:
        logger.info("Requesting candidate LLM response for query: '%s...'", clean_q[:40])
        req = GenerationRequest(question=clean_q, model=effective_model)
        return act_adapter.generate(req)
    except Exception as exc:
        logger.error("Unexpected error in generate_candidate_response: %s", exc)
        return GenerationResult(
            question=clean_q,
            response_text="",
            model_id=effective_model,
            execution_time_seconds=0.0,
            success=False,
            error=GenerationError(
                error_code="UNEXPECTED_ERROR",
                message=str(exc),
                retryable=False,
            ),
        )
