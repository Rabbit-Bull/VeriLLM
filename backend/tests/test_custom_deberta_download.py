"""Unit tests for VeriLLM DeBERTa local-first automatic download and caching.

Verifies:
1. Existing valid local default model -> no download.
2. Missing default model -> download is requested with exact parameters.
3. Incomplete default model -> download is requested.
4. Explicit valid model path -> no download.
5. Explicit missing model path -> existing FileNotFoundError, no download.
6. Offline mode with missing default model -> clear error, no download.
7. Download failure -> clear propagated diagnostic identifying repo and revision.
8. Downloaded directory is validated before model loading.
9. Existing process-wide cache behavior remains unchanged.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

CURRENT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = CURRENT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import ml_engine.claim_verifier.custom_deberta_verifier as cdv_module
from ml_engine.claim_verifier.custom_deberta_verifier import (
    DEFAULT_HF_REPO_ID,
    DEFAULT_HF_REVISION,
    DEFAULT_MODEL_DIR,
    REQUIRED_MODEL_FILES,
    CustomDebertaVerifier,
    _is_model_dir_complete,
    _is_offline_mode,
    _resolve_model_dir,
    _validate_model_dir_artifacts,
)


def _populate_mock_checkpoint(directory: Path) -> None:
    """Helper to create dummy valid model artifacts in a directory."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(
        json.dumps({
            "id2label": {"0": "CONTRADICTED", "1": "SUPPORTED_PROXY", "2": "UNSUPPORTED"},
            "label2id": {"CONTRADICTED": 0, "SUPPORTED_PROXY": 1, "UNSUPPORTED": 2},
        }),
        encoding="utf-8",
    )
    (directory / "model.safetensors").write_bytes(b"dummy_weights")
    (directory / "tokenizer.json").write_text(json.dumps({}), encoding="utf-8")
    (directory / "tokenizer_config.json").write_text(json.dumps({}), encoding="utf-8")


# ---------------------------------------------------------------------------
# Test 1: Existing valid local default model -> no download
# ---------------------------------------------------------------------------
def test_1_existing_valid_local_default_model_no_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies that an existing complete default model directory is used without download."""
    mock_download = MagicMock()
    monkeypatch.setattr(cdv_module.huggingface_hub, "snapshot_download", mock_download)

    resolved = _resolve_model_dir(model_path=None)

    assert resolved == DEFAULT_MODEL_DIR.resolve()
    mock_download.assert_not_called()


# ---------------------------------------------------------------------------
# Test 2: Missing default model -> download is requested
# ---------------------------------------------------------------------------
def test_2_missing_default_model_download_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies that a missing default model directory triggers download with exact parameters."""
    target_dir = tmp_path / "downloaded_model"
    monkeypatch.setattr(cdv_module, "DEFAULT_MODEL_DIR", target_dir)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    def mock_download_impl(**kwargs: Any) -> str:
        _populate_mock_checkpoint(target_dir)
        return str(target_dir)

    mock_download = MagicMock(side_effect=mock_download_impl)
    monkeypatch.setattr(cdv_module.huggingface_hub, "snapshot_download", mock_download)

    resolved = _resolve_model_dir(model_path=None)

    assert resolved == target_dir.resolve()
    mock_download.assert_called_once_with(
        repo_id=DEFAULT_HF_REPO_ID,
        revision=DEFAULT_HF_REVISION,
        local_dir=str(target_dir.resolve()),
        allow_patterns=list(REQUIRED_MODEL_FILES),
    )


# ---------------------------------------------------------------------------
# Test 3: Incomplete default model -> download is requested
# ---------------------------------------------------------------------------
def test_3_incomplete_default_model_download_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies that an incomplete default model directory triggers download."""
    target_dir = tmp_path / "incomplete_model"
    target_dir.mkdir(parents=True)
    # Only write config.json (missing weights and tokenizer)
    (target_dir / "config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(cdv_module, "DEFAULT_MODEL_DIR", target_dir)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    def mock_download_impl(**kwargs: Any) -> str:
        _populate_mock_checkpoint(target_dir)
        return str(target_dir)

    mock_download = MagicMock(side_effect=mock_download_impl)
    monkeypatch.setattr(cdv_module.huggingface_hub, "snapshot_download", mock_download)

    resolved = _resolve_model_dir(model_path=None)

    assert resolved == target_dir.resolve()
    mock_download.assert_called_once()


# ---------------------------------------------------------------------------
# Test 4: Explicit valid model path -> no download
# ---------------------------------------------------------------------------
def test_4_explicit_valid_model_path_no_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies that providing an explicit valid model path never triggers a download."""
    custom_dir = tmp_path / "explicit_custom_model"
    _populate_mock_checkpoint(custom_dir)

    mock_download = MagicMock()
    monkeypatch.setattr(cdv_module.huggingface_hub, "snapshot_download", mock_download)

    resolved = _resolve_model_dir(model_path=custom_dir)

    assert resolved == custom_dir.resolve()
    mock_download.assert_not_called()


# ---------------------------------------------------------------------------
# Test 5: Explicit missing model path -> FileNotFoundError, no download
# ---------------------------------------------------------------------------
def test_5_explicit_missing_model_path_raises_file_not_found(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies that an explicit non-existent path raises FileNotFoundError without downloading."""
    non_existent = tmp_path / "non_existent_explicit"
    mock_download = MagicMock()
    monkeypatch.setattr(cdv_module.huggingface_hub, "snapshot_download", mock_download)

    with pytest.raises(FileNotFoundError, match="Custom DeBERTa model directory does not exist"):
        _resolve_model_dir(model_path=non_existent)

    mock_download.assert_not_called()


# ---------------------------------------------------------------------------
# Test 6: Offline mode with missing default model -> clear error, no download
# ---------------------------------------------------------------------------
def test_6_offline_mode_with_missing_default_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies that offline mode raises a clear diagnostic and suppresses download."""
    missing_dir = tmp_path / "missing_offline_default"
    monkeypatch.setattr(cdv_module, "DEFAULT_MODEL_DIR", missing_dir)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    mock_download = MagicMock()
    monkeypatch.setattr(cdv_module.huggingface_hub, "snapshot_download", mock_download)

    with pytest.raises(FileNotFoundError) as exc_info:
        _resolve_model_dir(model_path=None)

    error_msg = str(exc_info.value)
    assert "offline mode is active" in error_msg
    assert "HF_HUB_OFFLINE=1" in error_msg
    assert DEFAULT_HF_REPO_ID in error_msg
    mock_download.assert_not_called()


# ---------------------------------------------------------------------------
# Test 7: Download failure -> clear propagated diagnostic
# ---------------------------------------------------------------------------
def test_7_download_failure_raises_clear_propagated_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies that download network/permission failures raise a clear RuntimeError."""
    missing_dir = tmp_path / "failing_download_dir"
    monkeypatch.setattr(cdv_module, "DEFAULT_MODEL_DIR", missing_dir)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    mock_download = MagicMock(side_effect=ConnectionError("Upstream connection refused"))
    monkeypatch.setattr(cdv_module.huggingface_hub, "snapshot_download", mock_download)

    with pytest.raises(RuntimeError) as exc_info:
        _resolve_model_dir(model_path=None)

    error_msg = str(exc_info.value)
    assert "Failed to download fine-tuned DeBERTa model from Hugging Face repository" in error_msg
    assert DEFAULT_HF_REPO_ID in error_msg
    assert DEFAULT_HF_REVISION in error_msg
    assert "Upstream connection refused" in error_msg
    assert "model_path" in error_msg


# ---------------------------------------------------------------------------
# Test 8: Downloaded directory is validated before model loading
# ---------------------------------------------------------------------------
def test_8_downloaded_directory_validated_before_model_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies that post-download artifact validation catches incomplete downloads."""
    target_dir = tmp_path / "partially_downloaded"
    monkeypatch.setattr(cdv_module, "DEFAULT_MODEL_DIR", target_dir)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    # Mock download creates the folder but fails to write config.json
    def bad_download(**kwargs: Any) -> str:
        target_dir.mkdir(parents=True, exist_ok=True)
        return str(target_dir)

    monkeypatch.setattr(cdv_module.huggingface_hub, "snapshot_download", MagicMock(side_effect=bad_download))

    with pytest.raises(FileNotFoundError, match="Missing config.json"):
        _resolve_model_dir(model_path=None)


# ---------------------------------------------------------------------------
# Test 9: Existing process-wide cache behavior remains unchanged
# ---------------------------------------------------------------------------
def test_9_existing_process_wide_cache_behavior_remains_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies that CustomDebertaVerifier caching and local_files_only loading remain intact."""
    CustomDebertaVerifier.clear_cache()

    tok_loads: List[Dict[str, Any]] = []
    mod_loads: List[Dict[str, Any]] = []

    orig_tok = cdv_module.AutoTokenizer.from_pretrained
    orig_mod = cdv_module.AutoModelForSequenceClassification.from_pretrained

    def spy_tok(path: str, **kwargs: Any) -> Any:
        tok_loads.append({"path": path, "kwargs": dict(kwargs)})
        return orig_tok(path, **kwargs)

    def spy_mod(path: str, **kwargs: Any) -> Any:
        mod_loads.append({"path": path, "kwargs": dict(kwargs)})
        return orig_mod(path, **kwargs)

    monkeypatch.setattr(cdv_module.AutoTokenizer, "from_pretrained", staticmethod(spy_tok))
    monkeypatch.setattr(cdv_module.AutoModelForSequenceClassification, "from_pretrained", staticmethod(spy_mod))

    try:
        v1 = CustomDebertaVerifier(device="cpu")
        v2 = CustomDebertaVerifier(device="cpu")

        assert v1.tokenizer is v2.tokenizer
        assert v1.model is v2.model
        assert len(tok_loads) == 1
        assert len(mod_loads) == 1
        assert tok_loads[0]["kwargs"].get("local_files_only") is True
        assert mod_loads[0]["kwargs"].get("local_files_only") is True
    finally:
        CustomDebertaVerifier.clear_cache()

