"""
Unit tests for Gemini Provider Adapter (Step 11.2).
Validates configuration, question mapping, timeouts, and error handling offline using mocked SDK clients.
"""

from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure backend root is on sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from google.genai import errors, types

from ml_engine.llm_generator.client import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_TIMEOUT_SECONDS,
    GeminiAuthenticationError,
    GeminiConfigurationError,
    GeminiEmptyResponseError,
    GeminiNetworkError,
    GeminiProviderAdapter,
    GeminiProviderError,
    GeminiQuotaError,
    GeminiSafetyBlockError,
    GeminiTimeoutError,
)
from ml_engine.llm_generator.models import (
    GenerationError,
    GenerationRequest,
    GenerationResult,
)


# ===========================================================================
# 1. Successful Generation Tests
# ===========================================================================

class TestGeminiSuccessfulGeneration:
    """Tests for successful candidate text generation and metadata mapping."""

    @patch("ml_engine.llm_generator.client.genai.Client")
    def test_successful_response_generation(self, mock_client_cls: MagicMock) -> None:
        """Verifies that valid responses from Gemini are parsed into typed GenerationResult objects."""
        # Setup mock client and response
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_candidate = MagicMock()
        mock_candidate.finish_reason = "STOP"

        mock_usage = MagicMock()
        mock_usage.prompt_token_count = 12
        mock_usage.candidates_token_count = 34
        mock_usage.total_token_count = 46

        mock_response = MagicMock()
        mock_response.text = "Apollo 11 landed on the Moon on July 20, 1969."
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = mock_usage

        mock_client.models.generate_content.return_value = mock_response

        # Execute adapter
        adapter = GeminiProviderAdapter(api_key="test_api_key_12345")
        result = adapter.generate("When did Apollo 11 land on the Moon?")

        # Assertions
        assert isinstance(result, GenerationResult)
        assert result.success is True
        assert result.question == "When did Apollo 11 land on the Moon?"
        assert result.response_text == "Apollo 11 landed on the Moon on July 20, 1969."
        assert result.model_id == DEFAULT_GEMINI_MODEL
        assert result.finish_reason == "STOP"
        assert result.prompt_tokens == 12
        assert result.candidates_tokens == 34
        assert result.total_tokens == 46
        assert result.execution_time_seconds >= 0.0
        assert result.error is None

        # Verify client initialization arguments (timeout in milliseconds: 90000)
        mock_client_cls.assert_called_once()
        _, kwargs = mock_client_cls.call_args
        assert kwargs["api_key"] == "test_api_key_12345"
        assert isinstance(kwargs["http_options"], types.HttpOptions)
        assert kwargs["http_options"].timeout == 90000

    @patch("ml_engine.llm_generator.client.genai.Client")
    def test_custom_model_and_parameter_mapping(self, mock_client_cls: MagicMock) -> None:
        """Verifies that custom model identifiers and GenerationRequest configurations are mapped accurately."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.text = "Light travels at 299,792,458 m/s."
        mock_response.candidates = [MagicMock(finish_reason="STOP")]
        mock_response.usage_metadata = None
        mock_client.models.generate_content.return_value = mock_response

        adapter = GeminiProviderAdapter(api_key="test_key", model="gemini-2.5-pro", timeout_seconds=45.0)
        assert adapter.model_id == "gemini-2.5-pro"

        req = GenerationRequest(
            question="What is the speed of light?",
            model="gemini-2.5-pro",
            temperature=0.2,
            max_output_tokens=256,
            system_instruction="Be concise.",
        )

        result = adapter.generate(req)

        assert result.success is True
        assert result.model_id == "gemini-2.5-pro"

        # Verify generate_content arguments
        mock_client.models.generate_content.assert_called_once()
        call_kwargs = mock_client.models.generate_content.call_args[1]
        assert call_kwargs["model"] == "gemini-2.5-pro"
        assert call_kwargs["contents"] == "What is the speed of light?"

        config = call_kwargs["config"]
        assert isinstance(config, types.GenerateContentConfig)
        assert config.temperature == 0.2
        assert config.max_output_tokens == 256
        assert config.system_instruction == "Be concise."


# ===========================================================================
# 2. Configuration & Missing Key Tests
# ===========================================================================

class TestGeminiConfiguration:
    """Tests for API key resolution and missing credential handling."""

    def test_missing_api_key_returns_safe_error(self) -> None:
        """Verifies that missing GEMINI_API_KEY returns a structured error without crashing."""
        with patch.dict(os.environ, {}, clear=True), patch("dotenv.load_dotenv", return_value=False):
            adapter = GeminiProviderAdapter(api_key=None)
            assert adapter.is_available is False

            result = adapter.generate("What is the capital of France?")

            assert result.success is False
            assert result.response_text == ""
            assert result.error is not None
            assert result.error.error_code == "MISSING_API_KEY"
            assert "GEMINI_API_KEY" in result.error.message
            assert result.error.retryable is False

    def test_missing_api_key_raise_on_error(self) -> None:
        """Verifies that missing API key raises GeminiConfigurationError when raise_on_error=True."""
        with patch.dict(os.environ, {}, clear=True), patch("dotenv.load_dotenv", return_value=False):
            adapter = GeminiProviderAdapter(api_key=None)

            with pytest.raises(GeminiConfigurationError) as exc_info:
                adapter.generate("Test query", raise_on_error=True)

            assert "GEMINI_API_KEY" in str(exc_info.value)
            assert exc_info.value.error_code == "MISSING_API_KEY"

    def test_environment_variable_resolution(self) -> None:
        """Verifies that GEMINI_API_KEY and GEMINI_MODEL are resolved from os.environ."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "env_key_abc", "GEMINI_MODEL": "custom-gemini-model"}):
            adapter = GeminiProviderAdapter()
            assert adapter.is_available is True
            assert adapter.model_id == "custom-gemini-model"
            assert adapter._api_key == "env_key_abc"

    def test_timeout_configuration_and_env_override(self) -> None:
        """Verifies default 90.0s timeout and GEMINI_TIMEOUT_SECONDS env override."""
        # Default timeout without env var
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GEMINI_TIMEOUT_SECONDS", None)
            os.environ.pop("GEMINI_TIMEOUT", None)
            adapter = GeminiProviderAdapter(api_key="key")
            assert adapter.timeout_seconds == 90.0

        # Override via GEMINI_TIMEOUT_SECONDS
        with patch.dict(os.environ, {"GEMINI_TIMEOUT_SECONDS": "120.5"}):
            adapter = GeminiProviderAdapter(api_key="key")
            assert adapter.timeout_seconds == 120.5

        # Explicit argument overrides env var
        with patch.dict(os.environ, {"GEMINI_TIMEOUT_SECONDS": "120.5"}):
            adapter = GeminiProviderAdapter(api_key="key", timeout_seconds=45.0)
            assert adapter.timeout_seconds == 45.0


# ===========================================================================
# 3. Error Handling & Resilience Tests
# ===========================================================================

class TestGeminiErrorHandling:
    """Tests for API error categorization, timeouts, safety blocks, and network disruptions."""

    @patch("ml_engine.llm_generator.client.genai.Client")
    def test_authentication_failure_403(self, mock_client_cls: MagicMock) -> None:
        """Verifies that HTTP 403 / PERMISSION_DENIED maps to AUTH_FAILURE."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        err = errors.ClientError(
            code=403,
            response_json={"error": {"message": "API key not valid. Please pass a valid API key.", "status": "PERMISSION_DENIED"}},
        )
        mock_client.models.generate_content.side_effect = err

        adapter = GeminiProviderAdapter(api_key="bad_key")
        result = adapter.generate("Test question")

        assert result.success is False
        assert result.error is not None
        assert result.error.error_code == "AUTH_FAILURE"
        assert result.error.retryable is False

        with pytest.raises(GeminiAuthenticationError):
            adapter.generate("Test question", raise_on_error=True)

    @patch("ml_engine.llm_generator.client.genai.Client")
    def test_quota_exceeded_429(self, mock_client_cls: MagicMock) -> None:
        """Verifies that HTTP 429 / RESOURCE_EXHAUSTED maps to QUOTA_EXCEEDED with retryable=True."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        err = errors.ClientError(
            code=429,
            response_json={"error": {"message": "Resource has been exhausted (e.g. check quota).", "status": "RESOURCE_EXHAUSTED"}},
        )
        mock_client.models.generate_content.side_effect = err

        adapter = GeminiProviderAdapter(api_key="valid_key")
        result = adapter.generate("Explain black holes.")

        assert result.success is False
        assert result.error is not None
        assert result.error.error_code == "QUOTA_EXCEEDED"
        assert result.error.retryable is True

        with pytest.raises(GeminiQuotaError):
            adapter.generate("Explain black holes.", raise_on_error=True)

    @patch("ml_engine.llm_generator.client.genai.Client")
    def test_request_timeout(self, mock_client_cls: MagicMock) -> None:
        """Verifies that request timeouts map to CLIENT_TIMEOUT with retryable=True."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_client.models.generate_content.side_effect = TimeoutError("The read operation timed out")

        adapter = GeminiProviderAdapter(api_key="valid_key", timeout_seconds=90.0)
        result = adapter.generate("Long query")

        assert result.success is False
        assert result.error is not None
        assert result.error.error_code == "CLIENT_TIMEOUT"
        assert result.error.retryable is True
        assert "90.0 seconds" in result.error.message

        with pytest.raises(GeminiTimeoutError):
            adapter.generate("Long query", raise_on_error=True)

    @patch("ml_engine.llm_generator.client.genai.Client")
    def test_server_gateway_timeout_504(self, mock_client_cls: MagicMock) -> None:
        """Verifies that HTTP 504 API error maps to GATEWAY_TIMEOUT with clear server diagnosis."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_client.models.generate_content.side_effect = errors.APIError(
            code=504,
            response_json={
                "error": {
                    "code": 504,
                    "message": "The server encountered a temporary error and could not complete your request.",
                    "status": "Gateway Timeout",
                }
            },
        )

        adapter = GeminiProviderAdapter(api_key="valid_key", timeout_seconds=90.0)
        result = adapter.generate("Complex prompt")

        assert result.success is False
        assert result.error is not None
        assert result.error.error_code == "GATEWAY_TIMEOUT"
        assert "HTTP 504" in result.error.message
        assert "gateway timed out" in result.error.message.lower()
        assert result.error.retryable is True

    @patch("ml_engine.llm_generator.client.genai.Client")
    def test_client_side_httpx_timeout_distinction(self, mock_client_cls: MagicMock) -> None:
        """Verifies that client socket/read timeouts map to CLIENT_TIMEOUT rather than server 504."""
        import httpx
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_client.models.generate_content.side_effect = httpx.ReadTimeout("Client read timed out")

        adapter = GeminiProviderAdapter(api_key="valid_key", timeout_seconds=90.0)
        result = adapter.generate("Query")

        assert result.success is False
        assert result.error is not None
        assert result.error.error_code == "CLIENT_TIMEOUT"
        assert "Client generation request timed out locally" in result.error.message
        assert result.error.retryable is True

    @patch("ml_engine.llm_generator.client.genai.Client")
    def test_error_sanitizes_api_key(self, mock_client_cls: MagicMock) -> None:
        """Verifies that raw API keys are never leaked into error messages or details."""
        raw_secret = "AIzaSyAbc123456789012345678901234567"
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_client.models.generate_content.side_effect = RuntimeError(
            f"Endpoint error with x-goog-api-key: {raw_secret} on https://generativelanguage.googleapis.com?key={raw_secret}"
        )

        adapter = GeminiProviderAdapter(api_key=raw_secret)
        result = adapter.generate("Prompt")

        assert result.success is False
        assert raw_secret not in result.error.message
        assert raw_secret not in result.error.details
        assert "[REDACTED" in result.error.details

    @patch("ml_engine.llm_generator.client.genai.Client")
    def test_network_failure(self, mock_client_cls: MagicMock) -> None:
        """Verifies that socket and network connection failures map to NETWORK_ERROR."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_client.models.generate_content.side_effect = ConnectionError("Failed to resolve host")

        adapter = GeminiProviderAdapter(api_key="valid_key")
        result = adapter.generate("Hello world")

        assert result.success is False
        assert result.error is not None
        assert result.error.error_code == "NETWORK_ERROR"
        assert result.error.retryable is True

        with pytest.raises(GeminiNetworkError):
            adapter.generate("Hello world", raise_on_error=True)

    @patch("ml_engine.llm_generator.client.genai.Client")
    def test_safety_blocked_response(self, mock_client_cls: MagicMock) -> None:
        """Verifies that finish_reason=SAFETY maps to SAFETY_BLOCKED."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_candidate = MagicMock()
        mock_candidate.finish_reason = "SAFETY"

        mock_response = MagicMock()
        mock_response.text = None
        mock_response.candidates = [mock_candidate]

        mock_client.models.generate_content.return_value = mock_response

        adapter = GeminiProviderAdapter(api_key="valid_key")
        result = adapter.generate("Potentially sensitive topic")

        assert result.success is False
        assert result.error is not None
        assert result.error.error_code == "SAFETY_BLOCKED"
        assert result.error.retryable is False

        with pytest.raises(GeminiSafetyBlockError):
            adapter.generate("Potentially sensitive topic", raise_on_error=True)

    @patch("ml_engine.llm_generator.client.genai.Client")
    def test_empty_response_handling(self, mock_client_cls: MagicMock) -> None:
        """Verifies that whitespace or empty text from the model maps to EMPTY_RESPONSE."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.text = "   \n  \t "
        mock_response.candidates = [MagicMock(finish_reason="STOP")]

        mock_client.models.generate_content.return_value = mock_response

        adapter = GeminiProviderAdapter(api_key="valid_key")
        result = adapter.generate("Generate nothing")

        assert result.success is False
        assert result.error is not None
        assert result.error.error_code == "EMPTY_RESPONSE"

        with pytest.raises(GeminiEmptyResponseError):
            adapter.generate("Generate nothing", raise_on_error=True)

    def test_invalid_empty_question_rejected(self) -> None:
        """Verifies that empty questions are rejected immediately before invoking the API."""
        adapter = GeminiProviderAdapter(api_key="valid_key")

        res1 = adapter.generate("")
        assert res1.success is False
        assert res1.error.error_code == "INVALID_INPUT"

        res2 = adapter.generate("   \n\t  ")
        assert res2.success is False
        assert res2.error.error_code == "INVALID_INPUT"


# ===========================================================================
# 4. Strict Privacy & Architectural Invariant Tests
# ===========================================================================

class TestArchitecturalInvariants:
    """Verifies that the Gemini adapter respects all non-interference and privacy boundaries."""

    def test_no_reference_document_parameter_exists(self) -> None:
        """Verifies that GeminiProviderAdapter.generate() does NOT accept reference documents."""
        sig = inspect.signature(GeminiProviderAdapter.generate)
        param_names = list(sig.parameters.keys())

        # Crucial architectural assertion: Only 'request' and 'raise_on_error' are accepted
        assert "document" not in param_names
        assert "reference" not in param_names
        assert "context" not in param_names
        assert param_names == ["self", "request", "raise_on_error"]

    def test_generation_request_schema_has_no_document_field(self) -> None:
        """Verifies that GenerationRequest does not define document or context fields."""
        fields = list(GenerationRequest.model_fields.keys())
        assert "document" not in fields
        assert "context" not in fields
        assert "reference" not in fields


# ===========================================================================
# 5. Schema Contract & Usage Metadata Synchronization Tests
# ===========================================================================

class TestMetadataContractAndUsageSynchronization:
    """Verifies that GenerationResult enforces typed contract compatibility for usage metadata."""

    def test_generation_result_with_usage_metadata_dict(self) -> None:
        """Constructing GenerationResult with usage_metadata dict populates individual token fields."""
        res = GenerationResult(
            question="What is entropy?",
            response_text="Entropy is a measure of molecular disorder.",
            model_id="gemini-3.8-flash",
            execution_time_seconds=0.45,
            usage_metadata={
                "prompt_token_count": 8,
                "candidates_token_count": 16,
                "total_token_count": 24,
            },
            finish_reason="STOP",
            success=True,
        )

        assert res.prompt_tokens == 8
        assert res.candidates_tokens == 16
        assert res.total_tokens == 24
        assert res.usage_metadata == {
            "prompt_token_count": 8,
            "candidates_token_count": 16,
            "total_token_count": 24,
        }
        assert hasattr(res, "usage_metadata")

    def test_generation_result_with_individual_token_counts(self) -> None:
        """Constructing GenerationResult with token integers populates usage_metadata dict."""
        res = GenerationResult(
            question="What is entropy?",
            response_text="Entropy is a measure of molecular disorder.",
            model_id="gemini-3.8-flash",
            execution_time_seconds=0.45,
            prompt_tokens=10,
            candidates_tokens=20,
            total_tokens=30,
            finish_reason="STOP",
            success=True,
        )

        assert res.prompt_tokens == 10
        assert res.candidates_tokens == 20
        assert res.total_tokens == 30
        assert res.usage_metadata is not None
        assert res.usage_metadata["prompt_token_count"] == 10
        assert res.usage_metadata["candidates_token_count"] == 20
        assert res.usage_metadata["total_token_count"] == 30

    def test_generation_result_without_token_counts(self) -> None:
        """Constructing GenerationResult without tokens leaves both individual fields and usage_metadata as None."""
        res = GenerationResult(
            question="What is entropy?",
            response_text="Entropy is a measure of molecular disorder.",
            model_id="gemini-3.8-flash",
            execution_time_seconds=0.45,
            success=True,
        )

        assert res.prompt_tokens is None
        assert res.candidates_tokens is None
        assert res.total_tokens is None
        assert res.usage_metadata is None
        # Must not raise AttributeError when accessed
        assert getattr(res, "usage_metadata") is None

    def test_generation_result_with_partial_token_counts(self) -> None:
        """Constructing GenerationResult with partial token counts populates usage_metadata with None for missing."""
        res = GenerationResult(
            question="What is entropy?",
            response_text="Entropy is a measure of molecular disorder.",
            model_id="gemini-3.8-flash",
            total_tokens=45,
            success=True,
        )

        assert res.prompt_tokens is None
        assert res.candidates_tokens is None
        assert res.total_tokens == 45
        assert res.usage_metadata == {
            "prompt_token_count": None,
            "candidates_token_count": None,
            "total_token_count": 45,
        }

    @patch("ml_engine.llm_generator.client.genai.Client")
    def test_adapter_generate_populates_both_tokens_and_usage_metadata(self, mock_client_cls: MagicMock) -> None:
        """Verifies that GeminiProviderAdapter.generate() produces a GenerationResult with synchronized token counts."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_usage = MagicMock()
        mock_usage.prompt_token_count = 15
        mock_usage.candidates_token_count = 35
        mock_usage.total_token_count = 50

        mock_response = MagicMock()
        mock_response.text = "Newton's second law is F = ma."
        mock_response.candidates = [MagicMock(finish_reason="STOP")]
        mock_response.usage_metadata = mock_usage

        mock_client.models.generate_content.return_value = mock_response

        adapter = GeminiProviderAdapter(api_key="valid_key")
        result = adapter.generate("State Newton's second law")

        assert result.success is True
        assert result.prompt_tokens == 15
        assert result.candidates_tokens == 35
        assert result.total_tokens == 50
        assert result.usage_metadata == {
            "prompt_token_count": 15,
            "candidates_token_count": 35,
            "total_token_count": 50,
        }

    @patch("ml_engine.llm_generator.client.genai.Client")
    def test_adapter_generate_handles_missing_usage_metadata_gracefully(self, mock_client_cls: MagicMock) -> None:
        """Verifies that GeminiProviderAdapter.generate() handles response with no usage_metadata without error."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.text = "Newton's second law is F = ma."
        mock_response.candidates = [MagicMock(finish_reason="STOP")]
        mock_response.usage_metadata = None

        mock_client.models.generate_content.return_value = mock_response

        adapter = GeminiProviderAdapter(api_key="valid_key")
        result = adapter.generate("State Newton's second law")

        assert result.success is True
        assert result.prompt_tokens is None
        assert result.candidates_tokens is None
        assert result.total_tokens is None
        assert result.usage_metadata is None


# ===========================================================================
# 6. Diagnostic Metadata & Endpoint Info Tests
# ===========================================================================

class TestDiagnosticsAndEndpointInfo:
    """Verifies that diagnostic metadata, server headers, and endpoint info are captured safely."""

    def test_adapter_endpoint_info(self) -> None:
        """Verifies that endpoint_info returns non-sensitive metadata about the target API."""
        adapter = GeminiProviderAdapter(api_key="secret_test_key_12345", timeout_seconds=90.0)
        info = adapter.endpoint_info

        assert info["model_id"] == DEFAULT_GEMINI_MODEL
        assert info["base_url"] == "https://generativelanguage.googleapis.com/"
        assert info["api_version"] in ("v1", "v1beta")
        assert info["timeout_seconds"] == "90.0"
        assert info["timeout_ms"] == "90000"
        assert "secret_test_key_12345" not in str(info)

    @patch("ml_engine.llm_generator.client.genai.Client")
    def test_server_gateway_timeout_captures_diagnostic_metadata(self, mock_client_cls: MagicMock) -> None:
        """Verifies that HTTP 504 preserves server header and target URL in metadata."""
        import httpx
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        raw_secret = "AIzaSyTestKeyNeverExposeInReports98765"
        mock_resp = httpx.Response(
            status_code=504,
            headers={
                "server": "scaffolding on HTTPServer2",
                "content-type": "application/json; charset=UTF-8",
                "date": "Mon, 16 Nov 2026 04:00:00 GMT",
            },
            json={
                "error": {
                    "code": 504,
                    "message": "The upstream server took too long to respond.",
                    "status": "DEADLINE_EXCEEDED",
                }
            },
            request=httpx.Request(
                "POST",
                f"https://generativelanguage.googleapis.com/v1/models/gemini-3.8-flash:generateContent?key={raw_secret}",
            ),
        )

        mock_client.models.generate_content.side_effect = errors.APIError(
            code=504,
            response_json=mock_resp.json(),
            response=mock_resp,
        )

        adapter = GeminiProviderAdapter(api_key=raw_secret, timeout_seconds=90.0)
        result = adapter.generate("Test prompt")

        assert result.success is False
        assert result.error is not None
        assert result.error.error_code == "GATEWAY_TIMEOUT"
        # Verify metadata extraction
        assert result.metadata["http_status_code"] == 504
        assert result.metadata["server_header"] == "scaffolding on HTTPServer2"
        assert result.metadata["content_type"] == "application/json; charset=UTF-8"
        # Verify secret is redacted in request_url and error details
        assert raw_secret not in result.metadata["request_url"]
        assert raw_secret not in result.error.details
        assert "[REDACTED" in result.metadata["request_url"]
        assert "Server: scaffolding on HTTPServer2" in result.error.details

    def test_diagnostic_script_mocked_success(self) -> None:
        """Verifies that 17_diagnose_gemini_request runs cleanly on success."""
        import importlib
        diag_mod = importlib.import_module("backend.experiments.17_diagnose_gemini_request")

        with patch.object(diag_mod, "GeminiProviderAdapter") as mock_adapter_cls:
            mock_adapter = MagicMock()
            mock_adapter.is_available = True
            mock_adapter.model_id = "gemini-3.8-flash"
            mock_adapter.endpoint_info = {
                "base_url": "https://generativelanguage.googleapis.com/",
                "api_version": "v1",
                "timeout_seconds": "90.0",
                "timeout_ms": "90000",
            }
            mock_adapter.generate.return_value = GenerationResult(
                question="Reply with exactly: Hello from Gemini.",
                response_text="Hello from Gemini.",
                model_id="gemini-3.8-flash",
                execution_time_seconds=1.5,
                prompt_tokens=8,
                candidates_tokens=5,
                total_tokens=13,
                finish_reason="STOP",
                success=True,
            )
            mock_adapter_cls.return_value = mock_adapter

            code = diag_mod.run_diagnosis()
            assert code == 0

    def test_diagnostic_script_mocked_failure(self) -> None:
        """Verifies that 17_diagnose_gemini_request reports diagnostics cleanly on failure."""
        import importlib
        diag_mod = importlib.import_module("backend.experiments.17_diagnose_gemini_request")
        from ml_engine.llm_generator.models import GenerationError

        with patch.object(diag_mod, "GeminiProviderAdapter") as mock_adapter_cls:
            mock_adapter = MagicMock()
            mock_adapter.is_available = True
            mock_adapter.model_id = "gemini-3.8-flash"
            mock_adapter.endpoint_info = {
                "base_url": "https://generativelanguage.googleapis.com/",
                "api_version": "v1",
                "timeout_seconds": "90.0",
                "timeout_ms": "90000",
            }
            mock_adapter.generate.return_value = GenerationResult(
                question="Reply with exactly: Hello from Gemini.",
                response_text="",
                model_id="gemini-3.8-flash",
                execution_time_seconds=90.0,
                success=False,
                error=GenerationError(
                    error_code="GATEWAY_TIMEOUT",
                    message="Google Gemini server gateway timed out (HTTP 504) after 90.0s.",
                    details="504 DEADLINE_EXCEEDED [Server: scaffolding on HTTPServer2]",
                    retryable=True,
                ),
                metadata={
                    "http_status_code": 504,
                    "server_header": "scaffolding on HTTPServer2",
                    "content_type": "application/json",
                    "request_url": "https://generativelanguage.googleapis.com/v1/models/gemini-3.8-flash:generateContent?[REDACTED]",
                },
            )
            mock_adapter_cls.return_value = mock_adapter

            code = diag_mod.run_diagnosis()
            assert code == 1


# ===========================================================================
# 7. HTTP 503 Service Unavailability & Bounded Retry Tests
# ===========================================================================

class TestServiceUnavailableAndBoundedRetry:
    """Verifies resilience to transient HTTP 503 errors and strict bounded retries."""

    @patch("ml_engine.llm_generator.client.genai.Client")
    def test_503_followed_by_successful_generation(self, mock_client_cls: MagicMock) -> None:
        """Transient HTTP 503 on first attempt retries and succeeds on second attempt."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        err_503 = errors.ServerError(
            code=503,
            response_json={
                "error": {
                    "code": 503,
                    "message": "This model is currently experiencing high demand. Spikes in demand are usually temporary. Please try again later.",
                    "status": "UNAVAILABLE",
                }
            },
        )

        mock_candidate = MagicMock()
        mock_candidate.finish_reason = "STOP"
        mock_success = MagicMock()
        mock_success.text = "Hello from Gemini."
        mock_success.candidates = [mock_candidate]
        mock_success.usage_metadata = MagicMock(prompt_token_count=6, candidates_token_count=4, total_token_count=10)

        # Attempt 0 fails with 503, Attempt 1 succeeds
        mock_client.models.generate_content.side_effect = [err_503, mock_success]

        sleep_calls = []
        adapter = GeminiProviderAdapter(
            api_key="test_key_123",
            timeout_seconds=90.0,
            retry_backoff_seconds=0.5,
            sleep_fn=lambda s: sleep_calls.append(s),
        )

        result = adapter.generate("Reply with exactly: Hello from Gemini.")

        assert result.success is True
        assert result.response_text == "Hello from Gemini."
        assert mock_client.models.generate_content.call_count == 2
        assert len(sleep_calls) == 1
        assert sleep_calls[0] == 0.5
        assert result.metadata["retry_attempts"] == 1
        assert result.metadata["max_retries"] == 2

    @patch("ml_engine.llm_generator.client.genai.Client")
    def test_503_exhausts_all_permitted_attempts(self, mock_client_cls: MagicMock) -> None:
        """Persistent HTTP 503 retries up to max_retries (2 retries = 3 attempts total) and stops."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        err_503 = errors.ServerError(
            code=503,
            response_json={
                "error": {
                    "code": 503,
                    "message": "This model is currently experiencing high demand. Spikes in demand are usually temporary. Please try again later.",
                    "status": "UNAVAILABLE",
                }
            },
        )
        mock_client.models.generate_content.side_effect = err_503

        sleep_calls = []
        adapter = GeminiProviderAdapter(
            api_key="test_key_123",
            timeout_seconds=90.0,
            retry_backoff_seconds=1.0,
            sleep_fn=lambda s: sleep_calls.append(s),
        )

        result = adapter.generate("Reply with exactly: Hello from Gemini.")

        assert result.success is False
        assert result.error is not None
        assert result.error.error_code == "SERVICE_UNAVAILABLE"
        assert result.error.retryable is True
        assert "temporarily unavailable" in result.error.message.lower() or "high demand" in result.error.message.lower()
        # Original Google message preserved in details
        assert "high demand" in result.error.details.lower()
        assert "spikes in demand are usually temporary" in result.error.details.lower()
        # Verify attempt counts
        assert mock_client.models.generate_content.call_count == 3  # 1 initial + 2 retries
        assert sleep_calls == [1.0, 2.0]  # Exponential backoff: 1.0s then 2.0s
        assert result.metadata["retry_attempts"] == 2
        assert result.metadata["max_retries"] == 2

        # Verify raise_on_error raises GeminiServiceUnavailableError
        mock_client.models.generate_content.reset_mock()
        mock_client.models.generate_content.side_effect = err_503
        from ml_engine.llm_generator.client import GeminiServiceUnavailableError
        with pytest.raises(GeminiServiceUnavailableError) as exc_info:
            adapter.generate("Reply with exactly: Hello from Gemini.", raise_on_error=True)
        assert exc_info.value.error_code == "SERVICE_UNAVAILABLE"
        assert exc_info.value.retryable is True

    @patch("ml_engine.llm_generator.client.genai.Client")
    def test_no_retry_for_authentication_errors(self, mock_client_cls: MagicMock) -> None:
        """Authentication errors (HTTP 401/403) are permanent and must NOT be retried."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        err_401 = errors.ClientError(
            code=401,
            response_json={"error": {"code": 401, "message": "API key invalid", "status": "UNAUTHENTICATED"}},
        )
        mock_client.models.generate_content.side_effect = err_401

        sleep_calls = []
        adapter = GeminiProviderAdapter(
            api_key="bad_key",
            timeout_seconds=90.0,
            retry_backoff_seconds=1.0,
            sleep_fn=lambda s: sleep_calls.append(s),
        )

        result = adapter.generate("Test query")

        assert result.success is False
        assert result.error is not None
        assert result.error.error_code == "AUTH_FAILURE"
        assert result.error.retryable is False
        # Crucial assertion: exactly 1 call, zero sleep calls, zero retries
        assert mock_client.models.generate_content.call_count == 1
        assert len(sleep_calls) == 0
        assert result.metadata["retry_attempts"] == 0

    @patch("ml_engine.llm_generator.client.genai.Client")
    def test_no_retry_for_invalid_model_errors(self, mock_client_cls: MagicMock) -> None:
        """Invalid model errors (HTTP 404) are permanent and must NOT be retried."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        err_404 = errors.ClientError(
            code=404,
            response_json={"error": {"code": 404, "message": "models/nonexistent-model is not found", "status": "NOT_FOUND"}},
        )
        mock_client.models.generate_content.side_effect = err_404

        sleep_calls = []
        adapter = GeminiProviderAdapter(
            api_key="valid_key",
            model="nonexistent-model",
            timeout_seconds=90.0,
            retry_backoff_seconds=1.0,
            sleep_fn=lambda s: sleep_calls.append(s),
        )

        result = adapter.generate("Test query")

        assert result.success is False
        assert result.error is not None
        assert result.error.error_code == "INVALID_MODEL"
        assert result.error.retryable is False
        # Crucial assertion: exactly 1 call, zero sleep calls, zero retries
        assert mock_client.models.generate_content.call_count == 1
        assert len(sleep_calls) == 0
        assert result.metadata["retry_attempts"] == 0

    @patch("ml_engine.llm_generator.client.genai.Client")
    def test_503_not_misclassified_as_timeout_or_quota(self, mock_client_cls: MagicMock) -> None:
        """HTTP 503 must map strictly to SERVICE_UNAVAILABLE, never TIMEOUT or QUOTA_EXCEEDED."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        err_503 = errors.ServerError(
            code=503,
            response_json={"error": {"code": 503, "message": "Overloaded", "status": "UNAVAILABLE"}},
        )
        mock_client.models.generate_content.side_effect = err_503

        adapter = GeminiProviderAdapter(api_key="valid_key", retry_backoff_seconds=0.0)
        result = adapter.generate("Test query")

        assert result.error.error_code == "SERVICE_UNAVAILABLE"
        assert result.error.error_code != "TIMEOUT"
        assert result.error.error_code != "CLIENT_TIMEOUT"
        assert result.error.error_code != "GATEWAY_TIMEOUT"
        assert result.error.error_code != "QUOTA_EXCEEDED"


# ===========================================================================
# 8. API Version Configuration & SDK Dispatch Tests
# ===========================================================================

class TestApiVersionConfigurationAndSDKDispatch:
    """Verifies configurable API version support (v1 vs v1beta) and SDK options dispatch."""

    def test_default_api_version_is_v1(self) -> None:
        """Without explicit override, default API version is 'v1'."""
        with patch.dict(os.environ, {}, clear=True):
            adapter = GeminiProviderAdapter(api_key="test_key_123")
            assert adapter.api_version == "v1"
            assert adapter.endpoint_info["api_version"] == "v1"
            assert adapter.model_id == "gemini-3.8-flash"
            assert adapter.timeout_seconds == 90.0

    def test_env_override_gemini_api_version_v1beta(self) -> None:
        """GEMINI_API_VERSION environment variable selects 'v1beta'."""
        with patch.dict(os.environ, {"GEMINI_API_VERSION": "v1beta"}):
            adapter = GeminiProviderAdapter(api_key="test_key_123")
            assert adapter.api_version == "v1beta"
            assert adapter.endpoint_info["api_version"] == "v1beta"
            assert adapter.model_id == "gemini-3.8-flash"
            assert adapter.timeout_seconds == 90.0

    def test_explicit_argument_overrides_env_var(self) -> None:
        """Explicit api_version argument takes precedence over environment variable."""
        with patch.dict(os.environ, {"GEMINI_API_VERSION": "v1"}):
            adapter = GeminiProviderAdapter(api_key="test_key_123", api_version="v1beta")
            assert adapter.api_version == "v1beta"
            assert adapter.endpoint_info["api_version"] == "v1beta"

    @patch("ml_engine.llm_generator.client.genai.Client")
    def test_selected_api_version_reaches_sdk_client(self, mock_client_cls: MagicMock) -> None:
        """Selected API version is passed inside types.HttpOptions during Client creation."""
        adapter = GeminiProviderAdapter(api_key="test_key_123", api_version="v1beta", timeout_seconds=90.0)
        _ = adapter._create_client()

        mock_client_cls.assert_called_once()
        _, kwargs = mock_client_cls.call_args
        http_opts = kwargs.get("http_options")
        assert isinstance(http_opts, types.HttpOptions)
        assert http_opts.api_version == "v1beta"
        assert http_opts.timeout == 90000  # 90.0s in milliseconds

    @patch("ml_engine.llm_generator.client.genai.Client")
    def test_selected_api_version_reaches_generate_call_config(self, mock_client_cls: MagicMock) -> None:
        """Selected API version is passed in GenerateContentConfig http_options during generation."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_cand = MagicMock()
        mock_cand.finish_reason = "STOP"
        mock_resp = MagicMock()
        mock_resp.text = "Hello from Gemini v1beta."
        mock_resp.candidates = [mock_cand]
        mock_resp.usage_metadata = MagicMock(prompt_token_count=5, candidates_token_count=5, total_token_count=10)
        mock_client.models.generate_content.return_value = mock_resp

        adapter = GeminiProviderAdapter(api_key="test_key_123", api_version="v1beta", timeout_seconds=90.0)
        result = adapter.generate("Test prompt")

        assert result.success is True
        assert result.model_id == "gemini-3.8-flash"
        assert result.metadata.get("api_version") == "v1beta"

        # Verify call arguments
        mock_client.models.generate_content.assert_called_once()
        call_kwargs = mock_client.models.generate_content.call_args[1]
        assert call_kwargs["model"] == "gemini-3.8-flash"
        config = call_kwargs["config"]
        assert isinstance(config, types.GenerateContentConfig)
        assert config.http_options.api_version == "v1beta"
        assert config.http_options.timeout == 90000

    def test_unrecognized_version_falls_back_to_v1(self) -> None:
        """Unrecognized API version strings log warning and fall back to 'v1' safely."""
        adapter = GeminiProviderAdapter(api_key="test_key_123", api_version="unrecognized-xyz-version")
        assert adapter.api_version == "v1"
        assert adapter.endpoint_info["api_version"] == "v1"


# ===========================================================================
# 9. Model Resolution Precedence & Dispatch Tests
# ===========================================================================

class TestModelResolutionPrecedenceAndDispatch:
    """Verifies that model selection strictly adheres to configuration precedence:
    1. Explicit constructor argument
    2. GEMINI_MODEL environment variable
    3. Default fallback ('gemini-3.8-flash')
    and that the resolved model reaches the generateContent API call and diagnostics.
    """

    def test_default_model_is_gemini_3_8_flash(self) -> None:
        """Without explicit argument or env var, resolves to 'gemini-3.8-flash'."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GEMINI_MODEL", None)
            adapter = GeminiProviderAdapter(api_key="test_key_123")
            assert adapter.model_id == "gemini-3.8-flash"
            assert adapter.endpoint_info["model_id"] == "gemini-3.8-flash"

    def test_env_override_gemini_model_gemini_3_7_flash(self) -> None:
        """GEMINI_MODEL environment variable selects 'gemini-3.7-flash'."""
        with patch.dict(os.environ, {"GEMINI_MODEL": "gemini-3.7-flash"}):
            adapter = GeminiProviderAdapter(api_key="test_key_123")
            assert adapter.model_id == "gemini-3.7-flash"
            assert adapter.endpoint_info["model_id"] == "gemini-3.7-flash"

    def test_explicit_model_argument_overrides_env_var(self) -> None:
        """Explicit model constructor argument takes precedence over GEMINI_MODEL env var."""
        with patch.dict(os.environ, {"GEMINI_MODEL": "gemini-3.7-flash"}):
            adapter = GeminiProviderAdapter(api_key="test_key_123", model="gemini-custom-override")
            assert adapter.model_id == "gemini-custom-override"
            assert adapter.endpoint_info["model_id"] == "gemini-custom-override"

    def test_empty_or_whitespace_model_argument_falls_back_to_env(self) -> None:
        """Empty string or whitespace model argument falls back to GEMINI_MODEL env var."""
        with patch.dict(os.environ, {"GEMINI_MODEL": "gemini-3.7-flash"}):
            adapter_empty = GeminiProviderAdapter(api_key="test_key_123", model="")
            assert adapter_empty.model_id == "gemini-3.7-flash"

            adapter_ws = GeminiProviderAdapter(api_key="test_key_123", model="   ")
            assert adapter_ws.model_id == "gemini-3.7-flash"

    @patch("ml_engine.llm_generator.client.genai.Client")
    def test_env_selected_model_reaches_generate_content_api_call(self, mock_client_cls: MagicMock) -> None:
        """Verifies that GEMINI_MODEL=gemini-3.7-flash reaches generate_content model argument."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_cand = MagicMock()
        mock_cand.finish_reason = "STOP"
        mock_resp = MagicMock()
        mock_resp.text = "Hello from gemini-3.7-flash."
        mock_resp.candidates = [mock_cand]
        mock_resp.usage_metadata = MagicMock(prompt_token_count=6, candidates_token_count=8, total_token_count=14)
        mock_client.models.generate_content.return_value = mock_resp

        with patch.dict(os.environ, {"GEMINI_MODEL": "gemini-3.7-flash", "GEMINI_API_VERSION": "v1beta"}):
            adapter = GeminiProviderAdapter(api_key="test_key_123")
            result = adapter.generate("Test prompt")

            assert result.success is True
            assert result.model_id == "gemini-3.7-flash"
            assert result.response_text == "Hello from gemini-3.7-flash."

            # Verify generate_content was called with the exact model identifier
            mock_client.models.generate_content.assert_called_once()
            call_kwargs = mock_client.models.generate_content.call_args[1]
            assert call_kwargs["model"] == "gemini-3.7-flash"
            assert call_kwargs["contents"] == "Test prompt"

    def test_diagnostic_script_reflects_env_gemini_model_3_7_flash(self) -> None:
        """Verifies that 17_diagnose_gemini_request dynamically targets gemini-3.7-flash from env."""
        import importlib
        diag_mod = importlib.import_module("backend.experiments.17_diagnose_gemini_request")

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key_123", "GEMINI_MODEL": "gemini-3.7-flash", "GEMINI_API_VERSION": "v1beta"}):
            with patch.object(diag_mod, "GeminiProviderAdapter") as mock_adapter_cls:
                mock_adapter = MagicMock()
                mock_adapter.is_available = True
                mock_adapter.model_id = "gemini-3.7-flash"
                mock_adapter.api_version = "v1beta"
                mock_adapter.endpoint_info = {
                    "base_url": "https://generativelanguage.googleapis.com/",
                    "api_version": "v1beta",
                    "timeout_seconds": "90.0",
                    "timeout_ms": "90000",
                }
                mock_adapter.generate.return_value = GenerationResult(
                    question="Reply with exactly: Hello from Gemini.",
                    response_text="Hello from Gemini.",
                    model_id="gemini-3.7-flash",
                    execution_time_seconds=1.2,
                    prompt_tokens=8,
                    candidates_tokens=5,
                    total_tokens=13,
                    finish_reason="STOP",
                    success=True,
                    metadata={"api_version": "v1beta"},
                )
                mock_adapter_cls.return_value = mock_adapter

                code = diag_mod.run_diagnosis()
                assert code == 0

                # Ensure adapter was initialized without hardcoded gemini-3.8-flash
                mock_adapter_cls.assert_called_once_with(model=None)






