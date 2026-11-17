"""Dedicated verifier wrapper for the fine-tuned RAGTruth DeBERTa classifier.

This module loads the locally stored ``verillm_deberta`` sequence classifier
strictly from disk without network access and scores batches of
``(evidence_context, claim)`` pairs.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import huggingface_hub
import torch
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

# Anchor the default checkpoint path to this source file rather than CWD so
# callers work reliably from any working directory.
DEFAULT_MODEL_DIR: Path = (
    Path(__file__).resolve().parent.parent.parent / "models" / "verillm_deberta"
)

DEFAULT_HF_REPO_ID: str = "Rabbit-Bull/verillm-deberta"
DEFAULT_HF_REVISION: str = "71259e48c794e89f92075ddf848813c93327461c"

REQUIRED_MODEL_FILES: Tuple[str, ...] = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
)

EXPECTED_ID2LABEL: Dict[int, str] = {
    0: "CONTRADICTED",
    1: "SUPPORTED_PROXY",
    2: "UNSUPPORTED",
}

EXPECTED_LABEL2ID: Dict[str, int] = {
    "CONTRADICTED": 0,
    "SUPPORTED_PROXY": 1,
    "UNSUPPORTED": 2,
}

DEFAULT_MAX_LENGTH: int = 512
DEFAULT_BATCH_SIZE: int = 16

_WEIGHT_FILENAMES: Tuple[str, ...] = (
    "model.safetensors",
    "pytorch_model.bin",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)

_TOKENIZER_FILENAMES: Tuple[str, ...] = (
    "tokenizer.json",
    "spm.model",
    "vocab.txt",
)

# Process-wide, resolved-path-keyed caches to avoid reloading the checkpoint
# across multiple CustomDebertaVerifier instantiations.
_CACHE_LOCK = threading.Lock()
_TOKENIZER_CACHE: Dict[str, Any] = {}
_MODEL_CACHE: Dict[Tuple[str, str], Any] = {}


LabelName = Literal["CONTRADICTED", "SUPPORTED_PROXY", "UNSUPPORTED"]


class CustomDebertaPrediction(BaseModel):
    """Structured prediction returned for a single ``(evidence_context, claim)`` pair.

    Note:
        The probability fields contain normalized softmax outputs produced by
        a classifier trained with class weighting on RAGTruth. They represent
        relative class scores for a fixed 3-class model rather than calibrated
        probabilities that a claim is factually true.
    """

    predicted_label: LabelName
    probabilities: Dict[str, float]
    contradicted_prob: float = Field(..., ge=0.0, le=1.0)
    supported_proxy_prob: float = Field(..., ge=0.0, le=1.0)
    unsupported_prob: float = Field(..., ge=0.0, le=1.0)


def _clamp_unit_float(value: float) -> float:
    """Clamp a finite float to ``[0.0, 1.0]`` against float32 rounding noise."""
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"Model produced non-finite probability value: {value!r}")
    return min(1.0, max(0.0, numeric))


def _is_offline_mode() -> bool:
    """Return True if Hugging Face Hub or Transformers offline mode is enabled."""
    hf_offline = os.environ.get("HF_HUB_OFFLINE", "").strip().lower()
    tf_offline = os.environ.get("TRANSFORMERS_OFFLINE", "").strip().lower()
    return hf_offline in ("1", "true", "yes") or tf_offline in ("1", "true", "yes")


def _is_model_dir_complete(model_dir: Path) -> bool:
    """Return True if model_dir exists, is a directory, and contains all required artifacts."""
    if not model_dir.exists() or not model_dir.is_dir():
        return False
    return all((model_dir / filename).is_file() for filename in REQUIRED_MODEL_FILES)


def _validate_model_dir_artifacts(model_dir: Path) -> None:
    """Verify that model_dir exists, is a directory, and contains required artifacts.

    Raises:
        FileNotFoundError: If the directory or any required file is missing.
    """
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Custom DeBERTa model directory does not exist: {model_dir}"
        )
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"Custom DeBERTa model path is not a directory: {model_dir}"
        )

    config_file = model_dir / "config.json"
    if not config_file.is_file():
        raise FileNotFoundError(
            f"Missing config.json in custom DeBERTa model directory: {model_dir}"
        )

    if not any((model_dir / name).is_file() for name in _WEIGHT_FILENAMES):
        raise FileNotFoundError(
            f"Missing model weight file in {model_dir}. "
            f"Expected one of: {', '.join(_WEIGHT_FILENAMES)}"
        )

    if not any((model_dir / name).is_file() for name in _TOKENIZER_FILENAMES):
        raise FileNotFoundError(
            f"Missing tokenizer artifact in {model_dir}. "
            f"Expected one of: {', '.join(_TOKENIZER_FILENAMES)}"
        )


def _download_default_model_from_hub(target_dir: Path) -> None:
    """Download required model artifacts from Hugging Face Hub to target_dir.

    Raises:
        RuntimeError: If downloading fails due to network, repository, or permissions issues.
    """
    logger.info(
        "Default custom DeBERTa model missing or incomplete at %s. "
        "Downloading from Hugging Face repository '%s' (revision: %s)...",
        target_dir,
        DEFAULT_HF_REPO_ID,
        DEFAULT_HF_REVISION,
    )
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        huggingface_hub.snapshot_download(
            repo_id=DEFAULT_HF_REPO_ID,
            revision=DEFAULT_HF_REVISION,
            local_dir=str(target_dir),
            allow_patterns=list(REQUIRED_MODEL_FILES),
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download fine-tuned DeBERTa model from Hugging Face repository "
            f"'{DEFAULT_HF_REPO_ID}' (revision '{DEFAULT_HF_REVISION}') to '{target_dir}'. "
            f"Cause: {exc}. Please verify your network connection, ensure the repository is accessible "
            f"at https://huggingface.co/{DEFAULT_HF_REPO_ID}, or supply an explicit local model path via model_path."
        ) from exc


def _resolve_model_dir(model_path: Optional[Union[str, Path]]) -> Path:
    """Resolve checkpoint directory path with local-first precedence and Hugging Face fallback.

    Case A: Explicit model path supplied:
      - Validates the local path immediately.
      - Does not attempt any download.
      - Raises FileNotFoundError if directory or files are missing.

    Case B: Default model directory already contains valid files:
      - Uses local directory immediately without contacting Hugging Face.
      - Fully offline-compatible.

    Case C: Default model directory is missing or incomplete:
      - If HF_HUB_OFFLINE=1 or TRANSFORMERS_OFFLINE=1: raises FileNotFoundError.
      - Otherwise: downloads required artifacts via huggingface_hub.snapshot_download.
      - Validates downloaded artifacts before returning.

    Returns:
        Resolved Path to validated local checkpoint directory.

    Raises:
        FileNotFoundError: If local path is missing/incomplete in offline or explicit modes.
        RuntimeError: If automatic downloading from Hugging Face fails.
    """
    # Case A — Explicit model path supplied
    if model_path is not None:
        model_dir = Path(model_path).expanduser().resolve()
        _validate_model_dir_artifacts(model_dir)
        return model_dir

    # Resolve default directory dynamically to respect monkeypatching in tests
    default_dir = Path(DEFAULT_MODEL_DIR).expanduser().resolve()

    # Case B — Default model directory already contains valid files
    if _is_model_dir_complete(default_dir):
        _validate_model_dir_artifacts(default_dir)
        return default_dir

    # Case C — Default model directory is missing or incomplete
    if _is_offline_mode():
        raise FileNotFoundError(
            f"Custom DeBERTa model directory is missing or incomplete at: {default_dir}. "
            f"Automatic downloading from Hugging Face Hub ('{DEFAULT_HF_REPO_ID}') is disabled "
            f"because offline mode is active (HF_HUB_OFFLINE=1 or TRANSFORMERS_OFFLINE=1). "
            f"To resolve, provide an existing local model directory via model_path or disable offline mode."
        )

    # Download from Hugging Face Hub
    _download_default_model_from_hub(default_dir)

    # Validate downloaded artifacts
    _validate_model_dir_artifacts(default_dir)
    return default_dir


def _resolve_device(device: Optional[Union[str, torch.device]]) -> torch.device:
    """Select CPU or CUDA device cleanly without touching unavailable CUDA contexts."""
    if device is None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested CUDA device '{device}', but CUDA is not available."
            )
        if resolved.index is None:
            resolved = torch.device(f"cuda:{torch.cuda.current_device()}")

    return resolved


def _verify_label_mapping(config: Any) -> None:
    """Strictly verify checkpoint id2label and label2id against expected Phase 7 IDs."""
    raw_id2label = getattr(config, "id2label", None)
    raw_label2id = getattr(config, "label2id", None)

    if not isinstance(raw_id2label, dict) or not isinstance(raw_label2id, dict):
        raise ValueError(
            "Checkpoint config must define dictionary 'id2label' and 'label2id' mappings."
        )

    try:
        normalized_id2label = {int(k): str(v) for k, v in raw_id2label.items()}
        normalized_label2id = {str(k): int(v) for k, v in raw_label2id.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid label mapping types in checkpoint config: id2label={raw_id2label!r}, "
            f"label2id={raw_label2id!r}"
        ) from exc

    if normalized_id2label != EXPECTED_ID2LABEL:
        raise ValueError(
            "Checkpoint id2label mapping mismatch. "
            f"Expected {EXPECTED_ID2LABEL}, got {normalized_id2label}."
        )

    if normalized_label2id != EXPECTED_LABEL2ID:
        raise ValueError(
            "Checkpoint label2id mapping mismatch. "
            f"Expected {EXPECTED_LABEL2ID}, got {normalized_label2id}."
        )


def _validate_positive_int(value: int, name: str) -> int:
    """Ensure integer hyperparameters are strictly positive and not booleans."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return value


def _validate_pairs(pairs: Sequence[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Validate batch inputs as a sequence of non-empty ``(context, claim)`` strings."""
    if pairs is None or isinstance(pairs, (str, bytes)) or not isinstance(pairs, Iterable):
        raise ValueError(
            "Input pairs must be a sequence or iterable of (evidence_context, claim) tuples."
        )

    validated: List[Tuple[str, str]] = []
    for index, pair in enumerate(pairs):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError(
                f"Input pair at index {index} must be a 2-tuple of (evidence_context, claim), "
                f"got {pair!r}."
            )

        evidence_context, claim = pair
        if not isinstance(evidence_context, str) or not isinstance(claim, str):
            raise ValueError(
                f"Input pair at index {index} must contain string values for "
                f"(evidence_context, claim), got ({type(evidence_context).__name__}, "
                f"{type(claim).__name__})."
            )

        if not evidence_context.strip() or not claim.strip():
            raise ValueError(
                f"Input pair at index {index} must have non-empty, non-whitespace "
                "evidence_context and claim strings."
            )

        validated.append((evidence_context, claim))

    return validated


def _load_cached_artifacts(model_dir: Path, device: torch.device) -> Tuple[Any, Any]:
    """Load or reuse cached tokenizer and eval-mode model keyed by resolved path."""
    path_key = str(model_dir)
    model_key = (path_key, str(device))

    with _CACHE_LOCK:
        tokenizer = _TOKENIZER_CACHE.get(path_key)
        model = _MODEL_CACHE.get(model_key)

        if tokenizer is None:
            logger.info("Loading custom DeBERTa tokenizer locally from %s", path_key)
            tokenizer = AutoTokenizer.from_pretrained(path_key, local_files_only=True)
            _TOKENIZER_CACHE[path_key] = tokenizer

        if model is None:
            logger.info(
                "Loading custom DeBERTa model locally from %s onto %s", path_key, device
            )
            loaded_model = AutoModelForSequenceClassification.from_pretrained(
                path_key, local_files_only=True
            )
            _verify_label_mapping(loaded_model.config)
            loaded_model.to(device)
            loaded_model.eval()
            _MODEL_CACHE[model_key] = loaded_model
            model = loaded_model
        else:
            _verify_label_mapping(model.config)
            model.eval()

    return tokenizer, model


class CustomDebertaVerifier:
    """Offline inference wrapper for VeriLLM's fine-tuned RAGTruth DeBERTa model.

    Each call scores ``(evidence_context, claim)`` pairs using
    ``truncation="only_first"``, ``max_length=512``, and dynamic padding under
    ``model.eval()`` and ``torch.inference_mode()``.

    Note:
        Returned softmax values are normalized logits from a class-weighted
        training run on RAGTruth and are NOT calibrated truth probabilities.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
        max_length: int = DEFAULT_MAX_LENGTH,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self._model_dir: Path = _resolve_model_dir(model_path)
        self._device: torch.device = _resolve_device(device)
        self._max_length: int = _validate_positive_int(max_length, "max_length")
        self._batch_size: int = _validate_positive_int(batch_size, "batch_size")
        self._tokenizer, self._model = _load_cached_artifacts(
            self._model_dir, self._device
        )

    @property
    def model_dir(self) -> Path:
        """Resolved absolute path to the local checkpoint directory."""
        return self._model_dir

    @property
    def device(self) -> torch.device:
        """Torch device hosting the loaded classification model."""
        return self._device

    @property
    def max_length(self) -> int:
        """Maximum tokenized sequence length applied during pair encoding."""
        return self._max_length

    @property
    def batch_size(self) -> int:
        """Default mini-batch size used by ``predict_pairs``."""
        return self._batch_size

    @property
    def tokenizer(self) -> Any:
        """Cached HuggingFace tokenizer loaded from ``model_dir``."""
        return self._tokenizer

    @property
    def model(self) -> Any:
        """Cached HuggingFace sequence classification model in eval mode."""
        return self._model

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the process-wide tokenizer and model caches."""
        with _CACHE_LOCK:
            _TOKENIZER_CACHE.clear()
            _MODEL_CACHE.clear()

    @classmethod
    def cache_info(cls) -> Dict[str, int]:
        """Return counts of cached tokenizers and models."""
        with _CACHE_LOCK:
            return {
                "tokenizers": len(_TOKENIZER_CACHE),
                "models": len(_MODEL_CACHE),
            }

    def predict_pair(
        self, evidence_context: str, claim: str
    ) -> CustomDebertaPrediction:
        """Score a single ``(evidence_context, claim)`` pair."""
        return self.predict_pairs([(evidence_context, claim)], batch_size=1)[0]

    def predict_pairs(
        self,
        pairs: Sequence[Tuple[str, str]],
        batch_size: Optional[int] = None,
    ) -> List[CustomDebertaPrediction]:
        """Score a batch of ``(evidence_context, claim)`` pairs offline.

        Returns the predicted label and softmax scores for ``CONTRADICTED``,
        ``SUPPORTED_PROXY``, and ``UNSUPPORTED`` in the input order. These softmax
        scores are normalized model logits, not calibrated truth probabilities.
        """
        effective_batch_size = (
            self._batch_size
            if batch_size is None
            else _validate_positive_int(batch_size, "batch_size")
        )
        validated_pairs = _validate_pairs(pairs)
        if not validated_pairs:
            return []

        results: List[CustomDebertaPrediction] = []
        self._model.eval()

        with torch.inference_mode():
            for start in range(0, len(validated_pairs), effective_batch_size):
                batch = validated_pairs[start : start + effective_batch_size]
                batch_contexts = [context for context, _ in batch]
                batch_claims = [claim for _, claim in batch]

                encoded = self._tokenizer(
                    batch_contexts,
                    batch_claims,
                    truncation="only_first",
                    max_length=self._max_length,
                    padding=True,
                    return_tensors="pt",
                )
                model_inputs = {
                    key: tensor.to(self._device) for key, tensor in encoded.items()
                }
                outputs = self._model(**model_inputs)
                logits = outputs.logits

                if logits.ndim != 2 or logits.shape[-1] != len(EXPECTED_ID2LABEL):
                    raise ValueError(
                        f"Expected logits shape (batch_size, {len(EXPECTED_ID2LABEL)}), "
                        f"got {tuple(logits.shape)}."
                    )

                probs_tensor = torch.softmax(logits.detach().to(torch.float32), dim=-1).cpu()
                pred_indices = torch.argmax(probs_tensor, dim=-1).tolist()

                for row_probs, pred_idx in zip(probs_tensor.tolist(), pred_indices):
                    contradicted_prob = _clamp_unit_float(row_probs[0])
                    supported_proxy_prob = _clamp_unit_float(row_probs[1])
                    unsupported_prob = _clamp_unit_float(row_probs[2])
                    predicted_label = EXPECTED_ID2LABEL[int(pred_idx)]

                    results.append(
                        CustomDebertaPrediction(
                            predicted_label=predicted_label,  # type: ignore[arg-type]
                            probabilities={
                                "CONTRADICTED": contradicted_prob,
                                "SUPPORTED_PROXY": supported_proxy_prob,
                                "UNSUPPORTED": unsupported_prob,
                            },
                            contradicted_prob=contradicted_prob,
                            supported_proxy_prob=supported_proxy_prob,
                            unsupported_prob=unsupported_prob,
                        )
                    )

        return results
