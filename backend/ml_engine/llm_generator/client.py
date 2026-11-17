"""
VeriLLM Gemini Provider Adapter.
Wraps the official Google Gen AI SDK (google-genai) to generate candidate LLM responses
from user questions without transmitting reference documents or determining factuality.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any, Callable, Dict, Optional, Union

import dotenv
import httpx
from google import genai
from google.genai import errors, types

from .models import GenerationError, GenerationRequest, GenerationResult

logger = logging.getLogger("verillm.llm_generator.gemini")

# Default model, version and retry configuration
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
DEFAULT_TIMEOUT_SECONDS = 90.0
DEFAULT_API_VERSION = "v1"
SUPPORTED_API_VERSIONS = ("v1", "v1beta")
MAX_503_RETRIES = 2
INITIAL_RETRY_BACKOFF_SECONDS = 1.0
BACKOFF_MULTIPLIER = 2.0


# ===========================================================================
# Custom Exception Hierarchy
# ===========================================================================

class GeminiProviderError(Exception):
    """Base exception for all Gemini provider operations."""

    def __init__(self, message: str, error_code: str = "GEMINI_ERROR", retryable: bool = False) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.retryable = retryable
        self.details: Optional[str] = None
        self.metadata: Dict[str, Any] = {}


class GeminiConfigurationError(GeminiProviderError):
    """Raised when GEMINI_API_KEY is missing or invalidly configured."""

    def __init__(self, message: str = "GEMINI_API_KEY environment variable is not configured.") -> None:
        super().__init__(message, error_code="MISSING_API_KEY", retryable=False)


class GeminiAuthenticationError(GeminiProviderError):
    """Raised when API credentials are rejected by Google (HTTP 401/403)."""

    def __init__(self, message: str = "Invalid or unauthorized Gemini API key.") -> None:
        super().__init__(message, error_code="AUTH_FAILURE", retryable=False)


class GeminiQuotaError(GeminiProviderError):
    """Raised when Gemini rate limits or quotas are exhausted (HTTP 429)."""

    def __init__(self, message: str = "Gemini rate limit or quota exceeded.") -> None:
        super().__init__(message, error_code="QUOTA_EXCEEDED", retryable=True)


class GeminiTimeoutError(GeminiProviderError):
    """Raised when request duration exceeds the timeout threshold."""

    def __init__(
        self,
        message: str = "Gemini API request timed out.",
        error_code: str = "TIMEOUT",
    ) -> None:
        super().__init__(message, error_code=error_code, retryable=True)


class GeminiServiceUnavailableError(GeminiProviderError):
    """Raised when the Gemini model is experiencing temporary high demand (HTTP 503)."""

    def __init__(
        self,
        message: str = "The selected Gemini model is currently experiencing high demand and is temporarily unavailable (HTTP 503).",
    ) -> None:
        super().__init__(message, error_code="SERVICE_UNAVAILABLE", retryable=True)


class GeminiInvalidModelError(GeminiProviderError):
    """Raised when the specified model ID is not found or unsupported (HTTP 404)."""

    def __init__(
        self,
        message: str = "The specified Gemini model was not found or is unsupported.",
    ) -> None:
        super().__init__(message, error_code="INVALID_MODEL", retryable=False)


class GeminiNetworkError(GeminiProviderError):
    """Raised on connection failure or network disruption."""

    def __init__(self, message: str = "Failed to connect to Google Gemini API endpoints.") -> None:
        super().__init__(message, error_code="NETWORK_ERROR", retryable=True)


class GeminiSafetyBlockError(GeminiProviderError):
    """Raised when a candidate response is blocked by Google safety filters."""

    def __init__(self, message: str = "Response was blocked by Gemini safety filters.") -> None:
        super().__init__(message, error_code="SAFETY_BLOCKED", retryable=False)


class GeminiEmptyResponseError(GeminiProviderError):
    """Raised when the API returns an empty text string."""

    def __init__(self, message: str = "Gemini API returned an empty response.") -> None:
        super().__init__(message, error_code="EMPTY_RESPONSE", retryable=False)



# ===========================================================================
# Provider Adapter
# ===========================================================================

class GeminiProviderAdapter:
    """Official Google Gen AI SDK adapter for candidate text generation.

    Crucial Architectural Invariants:
    1. Operates solely as an untrusted candidate text generator.
    2. Accepts ONLY questions / prompts; does NOT accept or transmit reference documents.
    3. Never evaluates or scores its own responses.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        api_version: Optional[str] = None,
        max_retries: int = MAX_503_RETRIES,
        retry_backoff_seconds: Optional[float] = None,
        sleep_fn: Optional[Callable[[float], None]] = None,
    ) -> None:
        """Initializes the Gemini provider adapter.

        Args:
            api_key: Optional explicit API key. If omitted, resolves from environment/dotenv.
            model: Optional model identifier. Defaults to GEMINI_MODEL env var or 'gemini-3.8-flash'.
            timeout_seconds: Maximum duration in seconds for generation requests (default 90.0).
                             Can be overridden via GEMINI_TIMEOUT_SECONDS environment variable.
            api_version: Optional Gemini API version ('v1' or 'v1beta'). Defaults to GEMINI_API_VERSION or 'v1'.
            max_retries: Maximum number of retries for transient HTTP 503 service unavailability (default 2).
            retry_backoff_seconds: Initial backoff delay in seconds between retries (default 1.0).
            sleep_fn: Optional sleep function override (for testing without real time delays).
        """
        self._api_key = self._resolve_api_key(api_key)
        self.default_model = self._resolve_model(model)
        if timeout_seconds is not None:
            self.timeout_seconds = float(timeout_seconds)
        else:
            env_timeout = os.environ.get("GEMINI_TIMEOUT_SECONDS") or os.environ.get("GEMINI_TIMEOUT")
            if env_timeout:
                try:
                    self.timeout_seconds = float(env_timeout.strip())
                except ValueError:
                    logger.warning(
                        "Invalid GEMINI_TIMEOUT_SECONDS='%s', falling back to %.1f",
                        env_timeout,
                        DEFAULT_TIMEOUT_SECONDS,
                    )
                    self.timeout_seconds = DEFAULT_TIMEOUT_SECONDS
            else:
                self.timeout_seconds = DEFAULT_TIMEOUT_SECONDS

        # Resolve API version (default 'v1', override via GEMINI_API_VERSION or GOOGLE_GENAI_API_VERSION)
        raw_version = (
            api_version
            or os.environ.get("GEMINI_API_VERSION")
            or os.environ.get("GOOGLE_GENAI_API_VERSION")
            or DEFAULT_API_VERSION
        )
        clean_version = raw_version.strip().lower() if isinstance(raw_version, str) else DEFAULT_API_VERSION
        if clean_version == "beta":
            clean_version = "v1beta"
        if clean_version not in SUPPORTED_API_VERSIONS:
            logger.warning(
                "Unrecognized API version '%s', falling back to default '%s'",
                raw_version,
                DEFAULT_API_VERSION,
            )
            self.api_version = DEFAULT_API_VERSION
        else:
            self.api_version = clean_version

        self.max_retries = int(max_retries)
        if retry_backoff_seconds is not None:
            self.retry_backoff_seconds = float(retry_backoff_seconds)
        else:
            env_backoff = os.environ.get("GEMINI_RETRY_BACKOFF_SECONDS")
            if env_backoff:
                try:
                    self.retry_backoff_seconds = float(env_backoff.strip())
                except ValueError:
                    self.retry_backoff_seconds = INITIAL_RETRY_BACKOFF_SECONDS
            else:
                self.retry_backoff_seconds = INITIAL_RETRY_BACKOFF_SECONDS

        self._sleep_fn = sleep_fn if sleep_fn is not None else time.sleep

    def _sanitize_message(self, raw_msg: str) -> str:
        """Sanitizes error text to prevent leaking API keys, headers, or internal tokens."""
        if not raw_msg:
            return ""
        sanitized = str(raw_msg)
        if self._api_key:
            sanitized = sanitized.replace(self._api_key, "[REDACTED_API_KEY]")
        # Redact Google API key pattern (AIzaSy...)
        sanitized = re.sub(r"AIza[0-9A-Za-z-_]{35}", "[REDACTED_API_KEY]", sanitized)
        # Redact potential key headers or params
        sanitized = re.sub(r"(x-goog-api-key[:\s=]+)\S+", r"\g<1>[REDACTED]", sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r"(key=)[^\s&'\"]+", r"\g<1>[REDACTED]", sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r"(bearer\s+)[^\s'\"]+", r"\g<1>[REDACTED]", sanitized, flags=re.IGNORECASE)
        return sanitized

    @staticmethod
    def _resolve_api_key(explicit_key: Optional[str] = None) -> Optional[str]:
        """Resolves GEMINI_API_KEY from argument, environment, or .env file."""
        if explicit_key and explicit_key.strip():
            return explicit_key.strip()

        # Check existing environment
        env_key = os.environ.get("GEMINI_API_KEY")
        if env_key and env_key.strip():
            return env_key.strip()

        # Attempt to load from .env in project root or current working dir
        try:
            dotenv.load_dotenv(override=False)
            env_key = os.environ.get("GEMINI_API_KEY")
            if env_key and env_key.strip():
                return env_key.strip()
        except Exception as exc:
            logger.debug("Failed loading .env file: %s", exc)

        return None

    @staticmethod
    def _resolve_model(explicit_model: Optional[str] = None) -> str:
        """Resolves model identifier following strict configuration precedence:
        1. Explicit constructor argument, when provided (non-empty string).
        2. GEMINI_MODEL environment variable (from environment or .env).
        3. Existing default model ('gemini-3.8-flash') as fallback.
        """
        if explicit_model and isinstance(explicit_model, str) and explicit_model.strip():
            return explicit_model.strip()

        # Check existing environment
        env_model = os.environ.get("GEMINI_MODEL")
        if env_model and isinstance(env_model, str) and env_model.strip():
            return env_model.strip()

        # Attempt to load from .env in project root without overriding current environment
        try:
            dotenv.load_dotenv(override=False)
            env_model = os.environ.get("GEMINI_MODEL")
            if env_model and isinstance(env_model, str) and env_model.strip():
                return env_model.strip()
        except Exception as exc:
            logger.debug("Failed loading .env for GEMINI_MODEL: %s", exc)

        return DEFAULT_GEMINI_MODEL

    @property
    def is_available(self) -> bool:
        """Returns True if a non-empty API key is configured and available."""
        return bool(self._api_key and self._api_key.strip())

    @property
    def model_id(self) -> str:
        """Returns the active default model identifier."""
        return self.default_model

    @property
    def endpoint_info(self) -> Dict[str, str]:
        """Returns non-sensitive metadata about the target API endpoint and SDK configuration."""
        return {
            "model_id": self.default_model,
            "base_url": "https://generativelanguage.googleapis.com/",
            "api_version": self.api_version,
            "timeout_seconds": f"{self.timeout_seconds:.1f}",
            "timeout_ms": str(int(self.timeout_seconds * 1000)),
        }

    def _create_client(self) -> genai.Client:
        """Creates an authenticated google.genai.Client instance with configured timeout and API version."""
        if not self._api_key:
            raise GeminiConfigurationError("GEMINI_API_KEY is not configured.")

        # HttpOptions.timeout expects integer milliseconds
        timeout_ms = int(self.timeout_seconds * 1000)
        http_options = types.HttpOptions(
            timeout=timeout_ms,
            api_version=self.api_version,
        )

        return genai.Client(
            api_key=self._api_key,
            http_options=http_options,
        )

    def _execute_single_generation(
        self,
        req: GenerationRequest,
        active_model: str,
        start_time: float,
        client: genai.Client,
        raise_on_error: bool = False,
    ) -> GenerationResult:
        """Executes a single generation attempt against Google Gemini with comprehensive error mapping."""
        try:
            timeout_ms = int(self.timeout_seconds * 1000)
            call_http_options = types.HttpOptions(
                timeout=timeout_ms,
                api_version=self.api_version,
            )

            config = types.GenerateContentConfig(
                temperature=req.temperature,
                max_output_tokens=req.max_output_tokens,
                system_instruction=req.system_instruction,
                http_options=call_http_options,
            )

            logger.info(
                "Executing Gemini generation (model=%s, api_version=%s, timeout=%.1fs)",
                active_model,
                self.api_version,
                self.timeout_seconds,
            )

            response = client.models.generate_content(
                model=active_model,
                contents=req.question,
                config=config,
            )

            elapsed = round(time.perf_counter() - start_time, 4)

            # Validate response content
            if response is None:
                raise GeminiEmptyResponseError("Gemini API returned None as response.")

            # Check candidate finish reasons
            finish_reason: Optional[str] = None
            if response.candidates:
                first_cand = response.candidates[0]
                if hasattr(first_cand, "finish_reason") and first_cand.finish_reason is not None:
                    finish_reason = str(first_cand.finish_reason)
                    if "SAFETY" in finish_reason.upper() or "BLOCK" in finish_reason.upper():
                        raise GeminiSafetyBlockError(
                            f"Response was blocked by Google safety filters (finish_reason={finish_reason})."
                        )

            generated_text = response.text
            if generated_text is None or not generated_text.strip():
                raise GeminiEmptyResponseError("Gemini returned an empty response string.")

            # Extract token usage metadata safely
            prompt_tokens = None
            candidates_tokens = None
            total_tokens = None
            if hasattr(response, "usage_metadata") and response.usage_metadata is not None:
                usage = response.usage_metadata
                prompt_tokens = getattr(usage, "prompt_token_count", None)
                candidates_tokens = getattr(usage, "candidates_token_count", None)
                total_tokens = getattr(usage, "total_token_count", None)

            return GenerationResult(
                question=req.question,
                response_text=generated_text.strip(),
                model_id=active_model,
                execution_time_seconds=elapsed,
                success=True,
                finish_reason=finish_reason,
                prompt_tokens=prompt_tokens,
                candidates_tokens=candidates_tokens,
                total_tokens=total_tokens,
                metadata={"api_version": self.api_version},
            )

        except GeminiProviderError as g_err:
            elapsed = round(time.perf_counter() - start_time, 4)
            g_err.details = str(g_err)
            err = GenerationError(
                error_code=g_err.error_code,
                message=g_err.message,
                details=str(g_err),
                retryable=g_err.retryable,
            )
            if raise_on_error:
                raise
            return GenerationResult(
                question=req.question,
                response_text="",
                model_id=active_model,
                execution_time_seconds=elapsed,
                success=False,
                error=err,
                metadata={"api_version": self.api_version},
            )

        except errors.APIError as api_err:
            elapsed = round(time.perf_counter() - start_time, 4)
            code = getattr(api_err, "code", 500)
            msg_str = self._sanitize_message(str(api_err))

            # Extract response-level diagnostic information safely
            resp_obj = getattr(api_err, "response", None)
            resp_headers = getattr(resp_obj, "headers", {}) if resp_obj else {}
            server_header = resp_headers.get("server") if hasattr(resp_headers, "get") else None
            content_type = resp_headers.get("content-type") if hasattr(resp_headers, "get") else None
            via_header = resp_headers.get("via") if hasattr(resp_headers, "get") else None
            date_header = resp_headers.get("date") if hasattr(resp_headers, "get") else None
            raw_url = getattr(resp_obj, "url", None)
            sanitized_url = self._sanitize_message(str(raw_url)) if raw_url else None

            diag_metadata: Dict[str, Any] = {
                "http_status_code": code,
                "api_status": getattr(api_err, "status", None),
                "api_version": self.api_version,
                "server_header": server_header,
                "content_type": content_type,
                "via_header": via_header,
                "date_header": date_header,
                "request_url": sanitized_url,
            }

            if code in (401, 403) or "API_KEY_INVALID" in msg_str or "PERMISSION_DENIED" in msg_str:
                typed_exc = GeminiAuthenticationError(
                    f"Authentication failed (HTTP {code}): Invalid or unauthorized Gemini API key."
                )
            elif code == 404 or "NOT_FOUND" in msg_str or "not found" in msg_str.lower():
                typed_exc = GeminiInvalidModelError(
                    f"Gemini model not found or unsupported (HTTP {code}): {msg_str}"
                )
            elif code == 503 or "UNAVAILABLE" in msg_str or "high demand" in msg_str.lower() or "overloaded" in msg_str.lower():
                typed_exc = GeminiServiceUnavailableError(
                    f"The selected Gemini model is currently experiencing high demand and is temporarily unavailable (HTTP 503): {msg_str}"
                )
            elif code == 429 or "RESOURCE_EXHAUSTED" in msg_str or "quota" in msg_str.lower():
                typed_exc = GeminiQuotaError(
                    f"Rate limit or quota exceeded (HTTP {code}). Please wait or use manual verification."
                )
            elif code == 504 or "GATEWAY_TIMEOUT" in msg_str.upper() or "GATEWAY TIMEOUT" in msg_str.upper():
                typed_exc = GeminiTimeoutError(
                    f"Google Gemini server gateway timed out (HTTP 504) after {self.timeout_seconds:.1f}s. "
                    "The upstream model service took too long to complete.",
                    error_code="GATEWAY_TIMEOUT",
                )
            elif code == 408 or "DEADLINE_EXCEEDED" in msg_str:
                typed_exc = GeminiTimeoutError(
                    f"Google Gemini server request timed out (HTTP {code}) after {self.timeout_seconds:.1f}s.",
                    error_code="SERVER_TIMEOUT",
                )
            elif "timeout" in msg_str.lower():
                typed_exc = GeminiTimeoutError(
                    f"Gemini API request timed out (HTTP {code}) after {self.timeout_seconds:.1f}s: {msg_str}",
                    error_code="TIMEOUT",
                )
            else:
                retryable = code >= 500
                typed_exc = GeminiProviderError(
                    f"Gemini API error (HTTP {code}): {msg_str}",
                    error_code="API_ERROR",
                    retryable=retryable,
                )

            # Preserve diagnostic annotations without removing base msg_str
            diag_snippets = []
            if server_header:
                diag_snippets.append(f"Server: {server_header}")
            if sanitized_url:
                diag_snippets.append(f"URL: {sanitized_url}")
            err_details = f"{msg_str} [{', '.join(diag_snippets)}]" if diag_snippets else msg_str

            typed_exc.details = err_details
            typed_exc.metadata = diag_metadata

            err = GenerationError(
                error_code=typed_exc.error_code,
                message=typed_exc.message,
                details=err_details,
                retryable=typed_exc.retryable,
            )
            if raise_on_error:
                raise typed_exc
            return GenerationResult(
                question=req.question,
                response_text="",
                model_id=active_model,
                execution_time_seconds=elapsed,
                success=False,
                error=err,
                metadata=diag_metadata,
            )

        except (TimeoutError, httpx.TimeoutException) as t_err:
            elapsed = round(time.perf_counter() - start_time, 4)
            typed_exc = GeminiTimeoutError(
                f"Client generation request timed out locally after {self.timeout_seconds:.1f} seconds. "
                "No response was received from the server.",
                error_code="CLIENT_TIMEOUT",
            )
            typed_exc.details = self._sanitize_message(str(t_err))
            typed_exc.metadata = {
                "error_source": "CLIENT_HTTPX_TIMEOUT",
                "timeout_seconds": self.timeout_seconds,
            }
            err = GenerationError(
                error_code=typed_exc.error_code,
                message=typed_exc.message,
                details=typed_exc.details,
                retryable=True,
            )
            if raise_on_error:
                raise typed_exc
            return GenerationResult(
                question=req.question,
                response_text="",
                model_id=active_model,
                execution_time_seconds=elapsed,
                success=False,
                error=err,
                metadata=typed_exc.metadata,
            )

        except (ConnectionError, httpx.NetworkError) as c_err:
            elapsed = round(time.perf_counter() - start_time, 4)
            typed_exc = GeminiNetworkError(
                f"Failed to connect to Google API endpoints: {self._sanitize_message(str(c_err))}"
            )
            typed_exc.details = self._sanitize_message(str(c_err))
            err = GenerationError(
                error_code=typed_exc.error_code,
                message=typed_exc.message,
                details=typed_exc.details,
                retryable=True,
            )
            if raise_on_error:
                raise typed_exc
            return GenerationResult(
                question=req.question,
                response_text="",
                model_id=active_model,
                execution_time_seconds=elapsed,
                success=False,
                error=err,
            )

        except Exception as unk_err:
            elapsed = round(time.perf_counter() - start_time, 4)
            err_str = self._sanitize_message(str(unk_err))
            # Detect timeout or connection errors inside other library wrappers
            if "timeout" in err_str.lower() or "deadline" in err_str.lower():
                typed_exc = GeminiTimeoutError(
                    f"Request timed out after {self.timeout_seconds:.1f} seconds: {err_str}",
                    error_code="TIMEOUT",
                )
            elif "connect" in err_str.lower() or "resolution" in err_str.lower():
                typed_exc = GeminiNetworkError(
                    f"Connection failure: {err_str}"
                )
            else:
                typed_exc = GeminiProviderError(
                    f"Unexpected generation failure: {err_str}",
                    error_code="UNKNOWN_ERROR",
                    retryable=False,
                )

            typed_exc.details = err_str
            err = GenerationError(
                error_code=typed_exc.error_code,
                message=typed_exc.message,
                details=err_str,
                retryable=typed_exc.retryable,
            )
            if raise_on_error:
                raise typed_exc
            return GenerationResult(
                question=req.question,
                response_text="",
                model_id=active_model,
                execution_time_seconds=elapsed,
                success=False,
                error=err,
            )

    def generate(
        self,
        request: Union[str, GenerationRequest],
        raise_on_error: bool = False,
    ) -> GenerationResult:
        """Generates a candidate response for a user question with bounded retries for HTTP 503.

        Args:
            request: A user question string or a typed GenerationRequest object.
            raise_on_error: If True, raises a typed GeminiProviderError on failure.
                            If False, returns a GenerationResult with success=False and error populated.

        Returns:
            GenerationResult containing generated text, model metadata, latency, and tokens.
        """
        start_time = time.perf_counter()

        # 1. Normalize request input
        if isinstance(request, str):
            clean_q = request.strip() if request else ""
            if not clean_q:
                err = GenerationError(
                    error_code="INVALID_INPUT",
                    message="Question cannot be empty or whitespace.",
                    retryable=False,
                )
                if raise_on_error:
                    raise GeminiProviderError(err.message, error_code=err.error_code)
                return GenerationResult(
                    question=request,
                    response_text="",
                    model_id=self.default_model,
                    execution_time_seconds=0.0,
                    success=False,
                    error=err,
                    metadata={"retry_attempts": 0, "max_retries": self.max_retries},
                )
            req = GenerationRequest(question=clean_q, model=self.default_model)
        elif isinstance(request, GenerationRequest):
            req = request
        else:
            raise TypeError(f"Expected str or GenerationRequest, got {type(request).__name__}")

        active_model = req.model or self.default_model

        # 2. Check API key availability
        if not self.is_available:
            err = GenerationError(
                error_code="MISSING_API_KEY",
                message=(
                    "Gemini API key is not configured. Set the GEMINI_API_KEY environment "
                    "variable or add it to a local .env file to enable live response generation."
                ),
                retryable=False,
            )
            if raise_on_error:
                raise GeminiConfigurationError(err.message)
            return GenerationResult(
                question=req.question,
                response_text="",
                model_id=active_model,
                execution_time_seconds=0.0,
                success=False,
                error=err,
                metadata={"retry_attempts": 0, "max_retries": self.max_retries},
            )

        # 3. Execute bounded retry loop for transient HTTP 503 (max 2 retries)
        client = self._create_client()
        backoff = self.retry_backoff_seconds
        last_result: Optional[GenerationResult] = None
        last_exc: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                res = self._execute_single_generation(
                    req=req,
                    active_model=active_model,
                    start_time=start_time,
                    client=client,
                    raise_on_error=True,
                )
                res.metadata["retry_attempts"] = attempt
                res.metadata["max_retries"] = self.max_retries
                return res

            except GeminiServiceUnavailableError as exc:
                last_exc = exc
                elapsed = round(time.perf_counter() - start_time, 4)
                last_result = GenerationResult(
                    question=req.question,
                    response_text="",
                    model_id=active_model,
                    execution_time_seconds=elapsed,
                    success=False,
                    error=GenerationError(
                        error_code=exc.error_code,
                        message=exc.message,
                        details=getattr(exc, "details", str(exc)),
                        retryable=exc.retryable,
                    ),
                    metadata={
                        **(getattr(exc, "metadata", {}) or {}),
                        "retry_attempts": attempt,
                        "max_retries": self.max_retries,
                    },
                )

                if attempt < self.max_retries:
                    logger.warning(
                        "Gemini model returned HTTP 503 (high demand). Retrying attempt %d of %d in %.2fs...",
                        attempt + 1,
                        self.max_retries,
                        backoff,
                    )
                    if self._sleep_fn and backoff > 0:
                        self._sleep_fn(backoff)
                    backoff *= BACKOFF_MULTIPLIER
                    continue
                else:
                    logger.error(
                        "Gemini model returned HTTP 503 across all %d attempts (%d retries). Concluding with SERVICE_UNAVAILABLE.",
                        attempt + 1,
                        self.max_retries,
                    )
                    break

            except (GeminiProviderError, Exception) as exc:
                # Permanent errors (AUTH_FAILURE, INVALID_MODEL, CLIENT_TIMEOUT, etc.): DO NOT RETRY
                last_exc = exc
                elapsed = round(time.perf_counter() - start_time, 4)
                err_code = getattr(exc, "error_code", "UNKNOWN_ERROR")
                err_msg = getattr(exc, "message", str(exc))
                err_retry = getattr(exc, "retryable", False)
                err_details = getattr(exc, "details", str(exc))
                err_meta = getattr(exc, "metadata", {}) or {}

                last_result = GenerationResult(
                    question=req.question,
                    response_text="",
                    model_id=active_model,
                    execution_time_seconds=elapsed,
                    success=False,
                    error=GenerationError(
                        error_code=err_code,
                        message=err_msg,
                        details=err_details,
                        retryable=err_retry,
                    ),
                    metadata={
                        **err_meta,
                        "retry_attempts": attempt,
                        "max_retries": self.max_retries,
                    },
                )
                break

        if raise_on_error and last_exc:
            raise last_exc

        assert last_result is not None
        return last_result

