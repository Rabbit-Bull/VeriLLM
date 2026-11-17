"""Unit tests for the dedicated CustomDebertaVerifier wrapper."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import pytest
import torch

CURRENT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = CURRENT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import ml_engine.claim_verifier.custom_deberta_verifier as cdv_module
from ml_engine.claim_verifier.custom_deberta_verifier import (
    DEFAULT_MODEL_DIR,
    EXPECTED_ID2LABEL,
    EXPECTED_LABEL2ID,
    CustomDebertaPrediction,
    CustomDebertaVerifier,
)

VALID_LABELS = ("CONTRADICTED", "SUPPORTED_PROXY", "UNSUPPORTED")


def _assert_prediction_invariants(prediction: CustomDebertaPrediction) -> None:
    assert isinstance(prediction, CustomDebertaPrediction)
    assert prediction.predicted_label in VALID_LABELS
    assert set(prediction.probabilities.keys()) == set(VALID_LABELS)

    for label in VALID_LABELS:
        score = prediction.probabilities[label]
        assert math.isfinite(score)
        assert 0.0 <= score <= 1.0

    assert prediction.contradicted_prob == prediction.probabilities["CONTRADICTED"]
    assert prediction.supported_proxy_prob == prediction.probabilities["SUPPORTED_PROXY"]
    assert prediction.unsupported_prob == prediction.probabilities["UNSUPPORTED"]

    total_prob = sum(prediction.probabilities.values())
    assert total_prob == pytest.approx(1.0, abs=1e-5)

    highest_label = max(prediction.probabilities, key=prediction.probabilities.__getitem__)
    assert prediction.predicted_label == highest_label


@pytest.fixture(scope="module")
def verifier() -> CustomDebertaVerifier:
    """Shared verifier loaded from the local Phase 7 checkpoint."""
    return CustomDebertaVerifier()


def test_default_model_path_is_file_anchored_and_cwd_independent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected_dir = (BACKEND_DIR / "models" / "verillm_deberta").resolve()
    assert DEFAULT_MODEL_DIR.resolve() == expected_dir

    monkeypatch.chdir(tmp_path)
    loaded = CustomDebertaVerifier(device="cpu")

    assert loaded.model_dir == expected_dir
    assert loaded.device.type == "cpu"
    assert loaded.max_length == 512
    assert loaded.model.training is False


def test_checkpoint_label_mapping_verified(verifier: CustomDebertaVerifier) -> None:
    id2label = {int(k): str(v) for k, v in verifier.model.config.id2label.items()}
    label2id = {str(k): int(v) for k, v in verifier.model.config.label2id.items()}

    assert id2label == EXPECTED_ID2LABEL
    assert label2id == EXPECTED_LABEL2ID


def test_label_mapping_mismatch_raises_clear_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad_dir = tmp_path / "mismatched_checkpoint"
    bad_dir.mkdir()
    (bad_dir / "config.json").write_text("{}", encoding="utf-8")
    (bad_dir / "model.safetensors").write_bytes(b"placeholder")
    (bad_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    class _DummyModel:
        def __init__(self) -> None:
            self.config = SimpleNamespace(
                id2label={0: "ENTAILMENT", 1: "NEUTRAL", 2: "CONTRADICTION"},
                label2id={"ENTAILMENT": 0, "NEUTRAL": 1, "CONTRADICTION": 2},
            )

        def to(self, _device: torch.device) -> "_DummyModel":
            return self

        def eval(self) -> "_DummyModel":
            return self

    CustomDebertaVerifier.clear_cache()
    try:
        monkeypatch.setattr(
            cdv_module.AutoTokenizer,
            "from_pretrained",
            staticmethod(lambda *args, **kwargs: object()),
        )
        monkeypatch.setattr(
            cdv_module.AutoModelForSequenceClassification,
            "from_pretrained",
            staticmethod(lambda *args, **kwargs: _DummyModel()),
        )

        with pytest.raises(ValueError, match="id2label mapping mismatch"):
            CustomDebertaVerifier(model_path=bad_dir, device="cpu")

        assert CustomDebertaVerifier.cache_info()["models"] == 0
    finally:
        CustomDebertaVerifier.clear_cache()


def test_single_pair_inference_and_probability_invariants(
    verifier: CustomDebertaVerifier,
) -> None:
    supported = verifier.predict_pair(
        evidence_context="Apollo 11 landed on the Moon on July 20, 1969.",
        claim="Apollo 11 landed on the Moon in July 1969.",
    )
    contradicted = verifier.predict_pair(
        evidence_context="The Eiffel Tower is located in Paris, France.",
        claim="The Eiffel Tower is located in Tokyo, Japan.",
    )
    unsupported = verifier.predict_pair(
        evidence_context="The Eiffel Tower was built in Paris for the 1889 World's Fair.",
        claim="The Eiffel Tower is painted bright purple every winter.",
    )

    for pred in (supported, contradicted, unsupported):
        _assert_prediction_invariants(pred)

    assert supported.predicted_label == "SUPPORTED_PROXY"
    assert contradicted.predicted_label == "CONTRADICTED"
    assert unsupported.predicted_label == "UNSUPPORTED"


def test_batched_inference_matches_single_pair_inference(
    verifier: CustomDebertaVerifier,
) -> None:
    pairs: List[Tuple[str, str]] = [
        (
            "Marie Curie won Nobel Prizes in Physics and Chemistry.",
            "Marie Curie won two Nobel Prizes.",
        ),
        (
            "Mount Everest has an elevation of 8,848.86 meters.",
            "Mount Everest is 12,000 meters tall.",
        ),
        (
            "Photosynthesis converts light energy into chemical energy in plants.",
            "Photosynthesis occurs faster on Mars than on Earth.",
        ),
    ]

    batched = verifier.predict_pairs(pairs, batch_size=2)
    singles = [verifier.predict_pair(ctx, clm) for ctx, clm in pairs]

    assert len(batched) == len(pairs)
    for batch_pred, single_pred in zip(batched, singles):
        _assert_prediction_invariants(batch_pred)
        assert batch_pred.predicted_label == single_pred.predicted_label
        for label in VALID_LABELS:
            assert batch_pred.probabilities[label] == pytest.approx(
                single_pred.probabilities[label], abs=1e-4
            )


def test_process_wide_path_keyed_caching(monkeypatch: pytest.MonkeyPatch) -> None:
    CustomDebertaVerifier.clear_cache()

    tokenizer_calls: List[Dict[str, Any]] = []
    model_calls: List[Dict[str, Any]] = []

    orig_tok_load = cdv_module.AutoTokenizer.from_pretrained
    orig_mod_load = cdv_module.AutoModelForSequenceClassification.from_pretrained

    def spy_tok(path: str, **kwargs: Any) -> Any:
        tokenizer_calls.append({"path": path, "kwargs": dict(kwargs)})
        return orig_tok_load(path, **kwargs)

    def spy_mod(path: str, **kwargs: Any) -> Any:
        model_calls.append({"path": path, "kwargs": dict(kwargs)})
        return orig_mod_load(path, **kwargs)

    monkeypatch.setattr(cdv_module.AutoTokenizer, "from_pretrained", staticmethod(spy_tok))
    monkeypatch.setattr(
        cdv_module.AutoModelForSequenceClassification,
        "from_pretrained",
        staticmethod(spy_mod),
    )

    first = CustomDebertaVerifier(device="cpu")
    second = CustomDebertaVerifier(device="cpu")

    assert first.tokenizer is second.tokenizer
    assert first.model is second.model
    assert len(tokenizer_calls) == 1
    assert len(model_calls) == 1
    assert tokenizer_calls[0]["kwargs"].get("local_files_only") is True
    assert model_calls[0]["kwargs"].get("local_files_only") is True
    assert CustomDebertaVerifier.cache_info() == {"tokenizers": 1, "models": 1}


def test_long_context_truncation_only_first_preserves_claim(
    verifier: CustomDebertaVerifier, monkeypatch: pytest.MonkeyPatch
) -> None:
    long_context = " ".join(f"backgroundword{i}" for i in range(650))
    claim = "Perseverance landed in Jezero Crater on February 18, 2021."

    captured: Dict[str, Any] = {}
    real_tokenizer = verifier.tokenizer

    def spy_tokenizer(*args: Any, **kwargs: Any) -> Any:
        encoded = real_tokenizer(*args, **kwargs)
        captured["kwargs"] = dict(kwargs)
        captured["input_ids"] = encoded["input_ids"].clone()
        return encoded

    monkeypatch.setattr(verifier, "_tokenizer", spy_tokenizer)

    prediction = verifier.predict_pair(long_context, claim)
    _assert_prediction_invariants(prediction)

    assert captured["kwargs"]["truncation"] == "only_first"
    assert captured["kwargs"]["max_length"] == 512
    assert captured["kwargs"]["padding"] is True
    assert captured["kwargs"]["return_tensors"] == "pt"

    input_ids = captured["input_ids"][0].tolist()
    assert len(input_ids) == 512

    claim_token_ids = real_tokenizer.encode(claim, add_special_tokens=False)
    sep_token_id = real_tokenizer.sep_token_id
    assert input_ids[-1] == sep_token_id
    # Second segment ends with claim tokens immediately before the trailing [SEP].
    assert input_ids[-1 - len(claim_token_ids) : -1] == claim_token_ids


def test_missing_model_directory_and_files_raise_file_not_found(tmp_path: Path) -> None:
    missing_dir = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        CustomDebertaVerifier(model_path=missing_dir)

    not_a_dir = tmp_path / "plain_file.txt"
    not_a_dir.write_text("x", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="not a directory"):
        CustomDebertaVerifier(model_path=not_a_dir)

    missing_config = tmp_path / "missing_config"
    missing_config.mkdir()
    with pytest.raises(FileNotFoundError, match="config.json"):
        CustomDebertaVerifier(model_path=missing_config)

    missing_weights = tmp_path / "missing_weights"
    missing_weights.mkdir()
    (missing_weights / "config.json").write_text(json.dumps({}), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="weight file"):
        CustomDebertaVerifier(model_path=missing_weights)

    missing_tokenizer = tmp_path / "missing_tokenizer"
    missing_tokenizer.mkdir()
    (missing_tokenizer / "config.json").write_text(json.dumps({}), encoding="utf-8")
    (missing_tokenizer / "model.safetensors").write_bytes(b"placeholder")
    with pytest.raises(FileNotFoundError, match="tokenizer artifact"):
        CustomDebertaVerifier(model_path=missing_tokenizer)


def test_empty_and_invalid_inputs_handled_cleanly(verifier: CustomDebertaVerifier) -> None:
    assert verifier.predict_pairs([]) == []

    invalid_inputs: List[Any] = [
        None,
        "single string instead of pairs",
        [("only one element",)],
        [("context", "claim", "extra")],
        [(123, "claim")],
        [("context", None)],
        [("   ", "valid claim")],
        [("valid context", "\n\t ")],
    ]
    for bad_input in invalid_inputs:
        with pytest.raises(ValueError):
            verifier.predict_pairs(bad_input)

    for bad_batch_size in (0, -2, True):
        with pytest.raises(ValueError, match="batch_size"):
            verifier.predict_pairs([("ctx", "clm")], batch_size=bad_batch_size)

    for bad_init_kwargs in (
        {"max_length": 0},
        {"max_length": False},
        {"batch_size": 0},
        {"batch_size": True},
    ):
        with pytest.raises(ValueError):
            CustomDebertaVerifier(**bad_init_kwargs)

