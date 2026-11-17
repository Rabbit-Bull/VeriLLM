"""
VeriLLM MiniCheck Verifier Wrapper.
Provides a clean, cached interface to the official MiniCheck baseline model.
Handles offline caching, batch scoring, and hardware-safe execution on CPU.
"""

from contextlib import contextmanager
import os
import sys
import threading
from pathlib import Path
from typing import Any, Iterator, List, Optional, Tuple

import torch

# Ensure offline execution is favored if models are already cached
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

try:
    import minicheck.inference as _minicheck_inference
    from minicheck.minicheck import MiniCheck
except ImportError:
    _minicheck_inference = None
    MiniCheck = None

_LOAD_LOCK = threading.Lock()


class _CpuSafeAutoModelProxy:
    """Scoped proxy stripping ``device_map`` during MiniCheck loading on CPU-only hosts."""

    def __init__(self, target_cls: Any) -> None:
        self._target_cls = target_cls

    def from_pretrained(self, *args: Any, **kwargs: Any) -> Any:
        # MiniCheck hardcodes device_map="auto"; on CPU-only machines Accelerate
        # may assign layers to "disk" without offload_folder when free RAM is low.
        load_kwargs = dict(kwargs)
        load_kwargs.pop("device_map", None)
        model = self._target_cls.from_pretrained(*args, **load_kwargs)
        if hasattr(model, "to"):
            model = model.to("cpu")
        return model

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target_cls, name)


@contextmanager
def _cpu_safe_minicheck_loading() -> Iterator[None]:
    """Prevent Accelerate disk offloading on CPU-only hosts while preserving CUDA behavior."""
    if torch.cuda.is_available() or _minicheck_inference is None:
        yield
        return

    orig_seq_cls = getattr(_minicheck_inference, "AutoModelForSequenceClassification", None)
    orig_seq2seq_cls = getattr(_minicheck_inference, "AutoModelForSeq2SeqLM", None)

    if orig_seq_cls is not None:
        _minicheck_inference.AutoModelForSequenceClassification = _CpuSafeAutoModelProxy(
            orig_seq_cls
        )
    if orig_seq2seq_cls is not None:
        _minicheck_inference.AutoModelForSeq2SeqLM = _CpuSafeAutoModelProxy(
            orig_seq2seq_cls
        )

    try:
        yield
    finally:
        if orig_seq_cls is not None:
            _minicheck_inference.AutoModelForSequenceClassification = orig_seq_cls
        if orig_seq2seq_cls is not None:
            _minicheck_inference.AutoModelForSeq2SeqLM = orig_seq2seq_cls


class MiniCheckVerifier:
    """
    Singleton wrapper around MiniCheck for document-grounded factual verification.
    """
    _instance: Optional["MiniCheckVerifier"] = None
    _scorer: Optional[Any] = None

    def __init__(self, model_name: str = "roberta-large", cache_dir: Optional[str] = None):
        """
        Args:
            model_name: MiniCheck checkpoint ('roberta-large', 355M params).
            cache_dir: Optional cache directory path.
        """
        self.model_name = model_name
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/huggingface/hub")
        self._ensure_loaded()

    def _ensure_loaded(self):
        """Lazy-loads the MiniCheck model if not already initialized."""
        if MiniCheckVerifier._scorer is None:
            with _LOAD_LOCK:
                if MiniCheckVerifier._scorer is None:
                    if MiniCheck is None:
                        raise ImportError(
                            "MiniCheck is not installed. Please install it via "
                            "pip install git+https://github.com/Liyan06/MiniCheck.git@main"
                        )
                    with _cpu_safe_minicheck_loading():
                        MiniCheckVerifier._scorer = MiniCheck(
                            model_name=self.model_name,
                            cache_dir=self.cache_dir
                        )
        self.scorer = MiniCheckVerifier._scorer

    def score_pairs(
        self, 
        docs: List[str], 
        claims: List[str]
    ) -> List[Tuple[int, float]]:
        """
        Evaluates a batch of (document, claim) pairs.

        Args:
            docs: List of premise documents or evidence passage texts.
            claims: List of factual claims/propositions.

        Returns:
            List of (pred_label, support_probability) tuples where:
                pred_label: 1 if supported, 0 if unsupported/contradicted.
                support_probability: Float between 0.0 and 1.0.
        """
        if not docs or not claims or len(docs) != len(claims):
            return []

        try:
            pred_labels, raw_probs, chunks, chunk_probs = self.scorer.score(
                docs=docs,
                claims=claims
            )
            results = []
            for pred, prob in zip(pred_labels, raw_probs):
                results.append((int(pred), float(prob)))
            return results
        except Exception as e:
            # Fallback for inference failures
            return [(0, 0.0) for _ in claims]
