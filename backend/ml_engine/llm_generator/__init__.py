"""
VeriLLM Live LLM Generation Module.
Provides official Google Gen AI SDK integration for generating candidate responses
from user questions without sending reference documents or evaluating truthfulness.
"""

from .models import (
    GenerationError,
    GenerationRequest,
    GenerationResult,
)
from .client import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_TIMEOUT_SECONDS,
    GeminiAuthenticationError,
    GeminiConfigurationError,
    GeminiEmptyResponseError,
    GeminiInvalidModelError,
    GeminiNetworkError,
    GeminiProviderAdapter,
    GeminiProviderError,
    GeminiQuotaError,
    GeminiSafetyBlockError,
    GeminiServiceUnavailableError,
    GeminiTimeoutError,
)

__all__ = [
    "DEFAULT_GEMINI_MODEL",
    "DEFAULT_TIMEOUT_SECONDS",
    "GenerationError",
    "GenerationRequest",
    "GenerationResult",
    "GeminiAuthenticationError",
    "GeminiConfigurationError",
    "GeminiEmptyResponseError",
    "GeminiInvalidModelError",
    "GeminiNetworkError",
    "GeminiProviderAdapter",
    "GeminiProviderError",
    "GeminiQuotaError",
    "GeminiSafetyBlockError",
    "GeminiServiceUnavailableError",
    "GeminiTimeoutError",
]

