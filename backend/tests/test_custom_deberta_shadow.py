"""Focused Phase 8 Step 3 tests for Custom DeBERTa shadow-mode integration."""

from __future__ import annotations

import logging
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pytest
import torch

CURRENT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = CURRENT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import ml_engine.claim_verifier.custom_deberta_verifier as cdv_module
import ml_engine.claim_verifier.verifier as verifier_module
from ml_engine.claim_extractor.models import ExtractedClaim
from ml_engine.claim_verifier.custom_deberta_verifier import (
    DEFAULT_MODEL_DIR,
    EXPECTED_ID2LABEL,
    EXPECTED_LABEL2ID,
    CustomDebertaPrediction,
    CustomDebertaVerifier,
)
from ml_engine.claim_verifier.models import VerificationCategory, VerifiedClaim
from ml_engine.claim_verifier.verifier import ClaimVerifier
from ml_engine.evidence_retriever.models import (
    ClaimEvidenceMatch,
    DocumentChunk,
    RetrievedPassage,
)
from ml_engine.scorer.response_scorer import ResponseScorer

VALID_LABELS = ("CONTRADICTED", "SUPPORTED_PROXY", "UNSUPPORTED")


def _local_checkpoint_present() -> bool:
    return (
        DEFAULT_MODEL_DIR.is_dir()
        and (DEFAULT_MODEL_DIR / "config.json").is_file()
        and (DEFAULT_MODEL_DIR / "model.safetensors").is_file()
        and (DEFAULT_MODEL_DIR / "tokenizer.json").is_file()
    )


def _make_claim(
    claim_id: str,
    text: str,
    *,
    is_factual: bool = True,
) -> ExtractedClaim:
    return ExtractedClaim(
        claim_id=claim_id,
        text=text,
        source_sentence=text,
        start_char=0,
        end_char=len(text),
        claim_start_char=0,
        claim_end_char=len(text),
        offset_type="EXACT",
        is_factual=is_factual,
    )


def _make_match(
    claim: ExtractedClaim,
    passages: Optional[List[Tuple[str, float]]] = None,
) -> ClaimEvidenceMatch:
    retrieved: List[RetrievedPassage] = []
    for rank, (passage_text, similarity) in enumerate(passages or [], start=1):
        chunk = DocumentChunk(
            chunk_id=f"{claim.claim_id}_chunk_{rank}",
            document_name="reference_doc.pdf",
            page_number=rank,
            chunk_index=rank - 1,
            text=passage_text,
            start_char=0,
            end_char=len(passage_text),
            word_count=len(passage_text.split()),
        )
        retrieved.append(
            RetrievedPassage(chunk=chunk, similarity_score=similarity, rank=rank)
        )

    best = retrieved[0] if retrieved else None
    return ClaimEvidenceMatch(
        claim_id=claim.claim_id,
        claim_text=claim.text,
        retrieved_passages=retrieved,
        best_passage=best,
        best_similarity_score=best.similarity_score if best else 0.0,
    )


def _make_prediction(
    predicted_label: str,
    contradicted: float,
    supported: float,
    unsupported: float,
) -> CustomDebertaPrediction:
    return CustomDebertaPrediction(
        predicted_label=predicted_label,  # type: ignore[arg-type]
        probabilities={
            "CONTRADICTED": contradicted,
            "SUPPORTED_PROXY": supported,
            "UNSUPPORTED": unsupported,
        },
        contradicted_prob=contradicted,
        supported_proxy_prob=supported,
        unsupported_prob=unsupported,
    )


class _StubMiniCheck:
    def __init__(self, scores_by_claim: Dict[str, Tuple[int, float]]) -> None:
        self.scores_by_claim = scores_by_claim
        self.calls: List[Tuple[List[str], List[str]]] = []

    def score_pairs(
        self, docs: List[str], claims: List[str]
    ) -> List[Tuple[int, float]]:
        self.calls.append((list(docs), list(claims)))
        return [self.scores_by_claim.get(claim, (0, 0.1)) for claim in claims]


class _StubNLI:
    def __init__(
        self,
        primary_scores_by_claim: Dict[str, Tuple[float, float, float]],
        pair_overrides: Optional[Dict[Tuple[str, str], Tuple[float, float, float]]] = None,
    ) -> None:
        self.primary_scores_by_claim = primary_scores_by_claim
        self.pair_overrides = pair_overrides or {}
        self.calls: List[List[Tuple[str, str]]] = []

    def predict_pairs(
        self, pairs: List[Tuple[str, str]]
    ) -> List[Tuple[float, float, float]]:
        self.calls.append(list(pairs))
        results: List[Tuple[float, float, float]] = []
        for evidence_text, claim_text in pairs:
            if (evidence_text, claim_text) in self.pair_overrides:
                results.append(self.pair_overrides[(evidence_text, claim_text)])
            else:
                results.append(
                    self.primary_scores_by_claim.get(claim_text, (0.05, 0.05, 0.90))
                )
        return results


def _attach_baseline_stubs(
    verifier: ClaimVerifier,
    mc_scores: Dict[str, Tuple[int, float]],
    nli_scores: Dict[str, Tuple[float, float, float]],
    pair_overrides: Optional[Dict[Tuple[str, str], Tuple[float, float, float]]] = None,
) -> None:
    verifier.minicheck = _StubMiniCheck(mc_scores)  # type: ignore[assignment]
    verifier.nli = _StubNLI(nli_scores, pair_overrides=pair_overrides)  # type: ignore[assignment]


def test_shadow_disabled_by_default_does_not_initialize_or_invoke_custom_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_calls: List[Any] = []

    class _FailIfInstantiated:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            init_calls.append((args, kwargs))
            raise AssertionError("CustomDebertaVerifier should not be instantiated when disabled.")

    monkeypatch.setattr(verifier_module, "CustomDebertaVerifier", _FailIfInstantiated)

    verifier = ClaimVerifier()
    assert verifier.use_custom_model is False
    assert verifier.custom_deberta is None

    claim = _make_claim("c1", "Apollo 11 landed on the Moon.")
    match = _make_match(claim, [("Apollo 11 landed on the Moon on July 20, 1969.", 0.88)])
    _attach_baseline_stubs(
        verifier,
        {"Apollo 11 landed on the Moon.": (1, 0.96)},
        {"Apollo 11 landed on the Moon.": (0.94, 0.02, 0.04)},
    )

    report = verifier.verify_claims([claim], [match], document_name="apollo.pdf")
    scored = ResponseScorer().score_response(report, response_id="r_default_off")

    assert init_calls == []
    assert verifier.custom_deberta is None
    assert report.claims[0].classification == VerificationCategory.SUPPORTED
    assert report.claims[0].scores.custom_deberta_supported_prob is None
    assert report.claims[0].scores.custom_deberta_contradiction_prob is None
    assert report.claims[0].scores.custom_deberta_unsupported_prob is None
    assert report.claims[0].scores.custom_deberta_predicted_label is None
    assert scored.claims[0].custom_deberta_predicted_label is None


def test_lazy_initialization_waits_for_eligible_claims_and_reuses_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_instances: List[Any] = []
    predict_calls: List[List[Tuple[str, str]]] = []

    class _SpyCustomDeberta:
        def __init__(self, model_path: Optional[Union[str, Path]] = None) -> None:
            self.model_path = model_path
            created_instances.append(self)

        def predict_pairs(
            self, pairs: Sequence[Tuple[str, str]]
        ) -> List[CustomDebertaPrediction]:
            predict_calls.append(list(pairs))
            return [
                _make_prediction("SUPPORTED_PROXY", 0.02, 0.95, 0.03)
                for _ in pairs
            ]

    monkeypatch.setattr(verifier_module, "CustomDebertaVerifier", _SpyCustomDeberta)

    verifier = ClaimVerifier(use_custom_model=True)
    assert created_instances == []
    assert verifier.custom_deberta is None

    opinion = _make_claim("op1", "This mission was inspiring.", is_factual=False)
    no_ev = _make_claim("ne1", "Saturn V used liquid oxygen.", is_factual=True)
    _attach_baseline_stubs(verifier, {}, {})

    # 1. Empty claims -> still no load
    empty_report = verifier.verify_claims([], [])
    assert empty_report.claims == []
    assert created_instances == []

    # 2. Only non-factual / missing-evidence claims -> still no load
    ineligible_report = verifier.verify_claims(
        [opinion, no_ev],
        [_make_match(opinion, []), _make_match(no_ev, [])],
    )
    assert len(ineligible_report.claims) == 2
    assert created_instances == []
    assert predict_calls == []

    # 3. Eligible factual claim with non-empty evidence -> lazy init occurs once
    factual = _make_claim("f1", "Neil Armstrong stepped onto the Moon.")
    _attach_baseline_stubs(
        verifier,
        {factual.text: (1, 0.94)},
        {factual.text: (0.91, 0.03, 0.06)},
    )
    eligible_match = _make_match(
        factual, [("Neil Armstrong stepped onto the lunar surface in 1969.", 0.85)]
    )
    first_report = verifier.verify_claims([factual], [eligible_match])
    assert len(created_instances) == 1
    assert verifier.custom_deberta is created_instances[0]
    assert len(predict_calls) == 1
    assert first_report.claims[0].scores.custom_deberta_predicted_label == "SUPPORTED_PROXY"

    # 4. Subsequent eligible batch reuses the same instance without re-instantiation
    second_report = verifier.verify_claims([factual], [eligible_match])
    assert len(created_instances) == 1
    assert len(predict_calls) == 2
    assert second_report.claims[0].scores.custom_deberta_supported_prob == pytest.approx(0.95)


def test_shadow_on_vs_off_baseline_and_response_scorer_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    c_sup = _make_claim("c_sup", "Armstrong collected lunar samples.")
    c_con = _make_claim("c_con", "Michael Collins walked on the Moon.")
    c_unsup = _make_claim("c_unsup", "The crew played chess on Mars.")
    c_no_ev = _make_claim("c_no_ev", "Rocket fuel was refined in Florida.")
    c_conf = _make_claim("c_conf", "The landing happened on July 20.")
    c_nonfact = _make_claim("c_nonfact", "What an incredible achievement!", is_factual=False)

    claims = [c_sup, c_con, c_unsup, c_no_ev, c_conf, c_nonfact]
    matches = [
        _make_match(c_sup, [("Armstrong collected 47.5 pounds of lunar rock.", 0.88)]),
        _make_match(c_con, [("Collins remained alone in lunar orbit.", 0.79)]),
        _make_match(c_unsup, [("Apollo 11 splashed down in the Pacific Ocean.", 0.62)]),
        _make_match(c_no_ev, []),
        _make_match(
            c_conf,
            [
                ("The landing took place on July 20, 1969.", 0.82),
                ("The landing was delayed until July 25, 1969.", 0.74),
            ],
        ),
        _make_match(c_nonfact, [("Apollo 11 launched in 1969.", 0.60)]),
    ]

    mc_map = {
        c_sup.text: (1, 0.96),
        c_con.text: (0, 0.04),
        c_unsup.text: (0, 0.08),
        c_conf.text: (1, 0.85),
    }
    nli_map = {
        c_sup.text: (0.93, 0.02, 0.05),
        c_con.text: (0.02, 0.95, 0.03),
        c_unsup.text: (0.06, 0.04, 0.90),
        c_conf.text: (0.88, 0.04, 0.08),
    }
    pair_overrides = {
        ("The landing took place on July 20, 1969.", c_conf.text): (0.92, 0.02, 0.06),
        ("The landing was delayed until July 25, 1969.", c_conf.text): (0.03, 0.91, 0.06),
    }

    shadow_lookup = {
        c_sup.text: _make_prediction("SUPPORTED_PROXY", 0.02, 0.94, 0.04),
        c_con.text: _make_prediction("CONTRADICTED", 0.91, 0.04, 0.05),
        c_unsup.text: _make_prediction("UNSUPPORTED", 0.07, 0.06, 0.87),
        c_conf.text: _make_prediction("CONTRADICTED", 0.81, 0.12, 0.07),
    }

    recorded_pairs: List[Tuple[str, str]] = []

    class _MockCustomDeberta:
        def __init__(self, model_path: Optional[Union[str, Path]] = None) -> None:
            pass

        def predict_pairs(
            self, pairs: Sequence[Tuple[str, str]]
        ) -> List[CustomDebertaPrediction]:
            recorded_pairs.extend(pairs)
            return [shadow_lookup[claim_text] for _, claim_text in pairs]

    monkeypatch.setattr(verifier_module, "CustomDebertaVerifier", _MockCustomDeberta)

    baseline_verifier = ClaimVerifier(use_custom_model=False)
    _attach_baseline_stubs(baseline_verifier, mc_map, nli_map, pair_overrides)

    shadow_verifier = ClaimVerifier(use_custom_model=True)
    _attach_baseline_stubs(shadow_verifier, mc_map, nli_map, pair_overrides)

    baseline_report = baseline_verifier.verify_claims(
        claims, matches, document_name="apollo_report.pdf"
    )
    shadow_report = shadow_verifier.verify_claims(
        claims, matches, document_name="apollo_report.pdf"
    )

    # Secondary conflict passages & non-factual / missing-evidence claims excluded from shadow batch
    assert [claim_text for _, claim_text in recorded_pairs] == [
        c_sup.text,
        c_con.text,
        c_unsup.text,
        c_conf.text,
    ]

    # Exact verification parity across every baseline attribute & metric
    assert shadow_report.metrics == baseline_report.metrics
    assert shadow_report.total_claims_processed == baseline_report.total_claims_processed
    assert shadow_report.document_name == baseline_report.document_name

    for base_vc, shad_vc in zip(baseline_report.claims, shadow_report.claims):
        assert shad_vc.claim_id == base_vc.claim_id
        assert shad_vc.classification == base_vc.classification
        assert shad_vc.decision_rule == base_vc.decision_rule
        assert shad_vc.reasoning_trace == base_vc.reasoning_trace
        assert shad_vc.is_factual == base_vc.is_factual
        assert shad_vc.primary_evidence == base_vc.primary_evidence
        assert shad_vc.all_evidence == base_vc.all_evidence
        assert shad_vc.scores.minicheck_prediction == base_vc.scores.minicheck_prediction
        assert shad_vc.scores.minicheck_probability == base_vc.scores.minicheck_probability
        assert shad_vc.scores.nli_entailment_prob == base_vc.scores.nli_entailment_prob
        assert shad_vc.scores.nli_contradiction_prob == base_vc.scores.nli_contradiction_prob
        assert shad_vc.scores.nli_neutral_prob == base_vc.scores.nli_neutral_prob
        assert shad_vc.scores.retrieval_similarity == base_vc.scores.retrieval_similarity

        if base_vc.claim_text in shadow_lookup:
            expected = shadow_lookup[base_vc.claim_text]
            assert shad_vc.scores.custom_deberta_supported_prob == pytest.approx(
                expected.supported_proxy_prob
            )
            assert shad_vc.scores.custom_deberta_contradiction_prob == pytest.approx(
                expected.contradicted_prob
            )
            assert shad_vc.scores.custom_deberta_unsupported_prob == pytest.approx(
                expected.unsupported_prob
            )
            assert shad_vc.scores.custom_deberta_predicted_label == expected.predicted_label
        else:
            assert shad_vc.scores.custom_deberta_supported_prob is None
            assert shad_vc.scores.custom_deberta_contradiction_prob is None
            assert shad_vc.scores.custom_deberta_unsupported_prob is None
            assert shad_vc.scores.custom_deberta_predicted_label is None

    # ResponseScorer metrics parity +ClaimSummary shadow field forwarding
    scorer = ResponseScorer()
    base_score = scorer.score_response(baseline_report, response_id="resp_parity")
    shadow_score = scorer.score_response(shadow_report, response_id="resp_parity")

    assert shadow_score.counts == base_score.counts
    assert shadow_score.rates == base_score.rates
    assert shadow_score.indices == base_score.indices
    assert shadow_score.has_factual_claims == base_score.has_factual_claims
    assert shadow_score.mathematical_invariants_verified == base_score.mathematical_invariants_verified
    assert shadow_score.denominator_description == base_score.denominator_description

    for base_sum, shad_sum in zip(base_score.claims, shadow_score.claims):
        assert shad_sum.classification == base_sum.classification
        assert shad_sum.minicheck_probability == base_sum.minicheck_probability
        assert shad_sum.nli_entailment_prob == base_sum.nli_entailment_prob
        assert shad_sum.nli_contradiction_prob == base_sum.nli_contradiction_prob
        assert shad_sum.nli_neutral_prob == base_sum.nli_neutral_prob
        assert shad_sum.reasoning_trace == base_sum.reasoning_trace

        if shad_sum.claim_text in shadow_lookup:
            expected = shadow_lookup[shad_sum.claim_text]
            assert shad_sum.custom_deberta_supported_prob == pytest.approx(
                expected.supported_proxy_prob
            )
            assert shad_sum.custom_deberta_contradiction_prob == pytest.approx(
                expected.contradicted_prob
            )
            assert shad_sum.custom_deberta_unsupported_prob == pytest.approx(
                expected.unsupported_prob
            )
            assert shad_sum.custom_deberta_predicted_label == expected.predicted_label
        else:
            assert shad_sum.custom_deberta_predicted_label is None


def test_adversarial_disagreement_never_changes_baseline_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    c_sup = _make_claim("adv_1", "Apollo 11 launched in 1969.")
    c_con = _make_claim("adv_2", "Apollo 11 launched in 1985.")
    c_unsup = _make_claim("adv_3", "Astronauts ate pizza on the Moon.")
    claims = [c_sup, c_con, c_unsup]
    matches = [
        _make_match(c_sup, [("Apollo 11 launched on July 16, 1969.", 0.89)]),
        _make_match(c_con, [("Apollo 11 launched on July 16, 1969.", 0.87)]),
        _make_match(c_unsup, [("Apollo 11 landed on July 20, 1969.", 0.61)]),
    ]

    mc_map = {
        c_sup.text: (1, 0.99),
        c_con.text: (0, 0.01),
        c_unsup.text: (0, 0.03),
    }
    nli_map = {
        c_sup.text: (0.96, 0.01, 0.03),
        c_con.text: (0.01, 0.97, 0.02),
        c_unsup.text: (0.02, 0.03, 0.95),
    }

    adversarial_shadow = {
        c_sup.text: _make_prediction("CONTRADICTED", 0.99, 0.005, 0.005),
        c_con.text: _make_prediction("SUPPORTED_PROXY", 0.005, 0.99, 0.005),
        c_unsup.text: _make_prediction("SUPPORTED_PROXY", 0.01, 0.98, 0.01),
    }

    class _AdversarialCustomDeberta:
        def __init__(self, model_path: Optional[Union[str, Path]] = None) -> None:
            pass

        def predict_pairs(
            self, pairs: Sequence[Tuple[str, str]]
        ) -> List[CustomDebertaPrediction]:
            return [adversarial_shadow[claim_text] for _, claim_text in pairs]

    monkeypatch.setattr(
        verifier_module, "CustomDebertaVerifier", _AdversarialCustomDeberta
    )

    off_verifier = ClaimVerifier(use_custom_model=False)
    on_verifier = ClaimVerifier(use_custom_model=True)
    _attach_baseline_stubs(off_verifier, mc_map, nli_map)
    _attach_baseline_stubs(on_verifier, mc_map, nli_map)

    off_report = off_verifier.verify_claims(claims, matches)
    on_report = on_verifier.verify_claims(claims, matches)

    assert [c.classification for c in on_report.claims] == [
        VerificationCategory.SUPPORTED,
        VerificationCategory.CONTRADICTED,
        VerificationCategory.UNSUPPORTED,
    ]
    assert [c.classification for c in on_report.claims] == [
        c.classification for c in off_report.claims
    ]
    assert on_report.metrics == off_report.metrics

    # Shadow fields faithfully capture the extreme disagreement while baseline stays untouched
    assert on_report.claims[0].scores.custom_deberta_predicted_label == "CONTRADICTED"
    assert on_report.claims[0].scores.custom_deberta_contradiction_prob == pytest.approx(0.99)
    assert on_report.claims[1].scores.custom_deberta_predicted_label == "SUPPORTED_PROXY"
    assert on_report.claims[1].scores.custom_deberta_supported_prob == pytest.approx(0.99)
    assert on_report.claims[2].scores.custom_deberta_predicted_label == "SUPPORTED_PROXY"

    scorer = ResponseScorer()
    assert scorer.score_response(on_report).indices == scorer.score_response(off_report).indices


def test_safe_copies_prevent_in_place_mutation_of_strategy_and_input_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_strategy_outputs: List[VerifiedClaim] = []
    captured_strategy_scores: List[Any] = []

    class _MockCustomDeberta:
        def __init__(self, model_path: Optional[Union[str, Path]] = None) -> None:
            pass

        def predict_pairs(
            self, pairs: Sequence[Tuple[str, str]]
        ) -> List[CustomDebertaPrediction]:
            return [_make_prediction("SUPPORTED_PROXY", 0.05, 0.90, 0.05) for _ in pairs]

    monkeypatch.setattr(verifier_module, "CustomDebertaVerifier", _MockCustomDeberta)

    verifier = ClaimVerifier(use_custom_model=True)
    claim = _make_claim("copy_1", "Saturn V launched Apollo 11.")
    match = _make_match(claim, [("Saturn V launched Apollo 11 from Kennedy Space Center.", 0.91)])
    claim_snapshot = claim.model_dump()
    match_snapshot = match.model_dump()

    _attach_baseline_stubs(
        verifier,
        {claim.text: (1, 0.97)},
        {claim.text: (0.95, 0.02, 0.03)},
    )

    orig_eval = verifier.decision_strategy.evaluate_claim

    def _spy_evaluate_claim(*args: Any, **kwargs: Any) -> VerifiedClaim:
        result = orig_eval(*args, **kwargs)
        captured_strategy_outputs.append(result)
        captured_strategy_scores.append(result.scores)
        return result

    monkeypatch.setattr(verifier.decision_strategy, "evaluate_claim", _spy_evaluate_claim)

    report = verifier.verify_claims([claim], [match])

    assert len(captured_strategy_outputs) == 1
    assert report.claims[0] is not captured_strategy_outputs[0]
    assert report.claims[0].scores is not captured_strategy_scores[0]
    # Strategy's original returned instances remain completely unmutated
    assert captured_strategy_outputs[0].scores.custom_deberta_predicted_label is None
    assert captured_strategy_scores[0].custom_deberta_supported_prob is None
    # Output report carries shadow fields safely
    assert report.claims[0].scores.custom_deberta_predicted_label == "SUPPORTED_PROXY"
    assert report.claims[0].scores.custom_deberta_supported_prob == pytest.approx(0.90)
    # Input models remain unmutated
    assert claim.model_dump() == claim_snapshot
    assert match.model_dump() == match_snapshot


def test_missing_default_checkpoint_logs_warning_and_continues_baseline_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    missing_default = tmp_path / "missing_default_verillm_deberta"
    monkeypatch.setattr(cdv_module, "DEFAULT_MODEL_DIR", missing_default)

    verifier = ClaimVerifier(use_custom_model=True, custom_model_path=None)
    claim = _make_claim("miss_1", "Apollo 11 landed on July 20, 1969.")
    match = _make_match(claim, [("Apollo 11 landed on the Moon on July 20, 1969.", 0.88)])
    _attach_baseline_stubs(
        verifier,
        {claim.text: (1, 0.95)},
        {claim.text: (0.92, 0.03, 0.05)},
    )

    with caplog.at_level(logging.WARNING, logger=verifier_module.__name__):
        report = verifier.verify_claims([claim], [match])

    assert "checkpoint" in caplog.text.lower()
    assert verifier.custom_deberta is None
    assert report.claims[0].classification == VerificationCategory.SUPPORTED
    assert report.claims[0].scores.custom_deberta_predicted_label is None
    assert report.metrics.evidence_support_rate == 100.00

    # Subsequent calls continue baseline-only cleanly without error
    caplog.clear()
    second_report = verifier.verify_claims([claim], [match])
    assert second_report.claims[0].classification == VerificationCategory.SUPPORTED
    assert second_report.claims[0].scores.custom_deberta_predicted_label is None
    assert caplog.text == ""


def test_explicit_invalid_checkpoint_path_raises_file_not_found(
    tmp_path: Path,
) -> None:
    bad_path = tmp_path / "does_not_exist_custom_deberta"
    verifier = ClaimVerifier(use_custom_model=True, custom_model_path=bad_path)
    claim = _make_claim("bad_path_1", "Apollo 11 landed on the Moon.")
    match = _make_match(claim, [("Apollo 11 landed on the Moon in 1969.", 0.88)])
    _attach_baseline_stubs(
        verifier,
        {claim.text: (1, 0.95)},
        {claim.text: (0.92, 0.03, 0.05)},
    )

    with pytest.raises(FileNotFoundError, match="does not exist"):
        verifier.verify_claims([claim], [match])


def test_corrupt_model_label_mismatch_and_inference_failures_propagate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claim = _make_claim("err_1", "Apollo 11 landed on the Moon.")
    match = _make_match(claim, [("Apollo 11 landed on the Moon in 1969.", 0.88)])

    # 1. Label mismatch (ValueError) must propagate even on default path
    class _LabelMismatchInit:
        def __init__(self, model_path: Optional[Union[str, Path]] = None) -> None:
            raise ValueError("Checkpoint id2label mapping mismatch.")

    monkeypatch.setattr(verifier_module, "CustomDebertaVerifier", _LabelMismatchInit)
    verifier_mismatch = ClaimVerifier(use_custom_model=True)
    _attach_baseline_stubs(verifier_mismatch, {claim.text: (1, 0.95)}, {claim.text: (0.92, 0.03, 0.05)})
    with pytest.raises(ValueError, match="id2label mapping mismatch"):
        verifier_mismatch.verify_claims([claim], [match])

    # 2. Corrupt checkpoint load (OSError/RuntimeError) must propagate
    fake_dir = tmp_path / "corrupt_checkpoint"
    fake_dir.mkdir()
    (fake_dir / "config.json").write_text("{}", encoding="utf-8")
    (fake_dir / "model.safetensors").write_bytes(b"corrupt")
    (fake_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(verifier_module, "CustomDebertaVerifier", CustomDebertaVerifier)
    CustomDebertaVerifier.clear_cache()
    try:
        monkeypatch.setattr(
            cdv_module.AutoTokenizer,
            "from_pretrained",
            staticmethod(lambda *args, **kwargs: object()),
        )

        def _raise_corrupt(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("Corrupted safetensors header")

        monkeypatch.setattr(
            cdv_module.AutoModelForSequenceClassification,
            "from_pretrained",
            staticmethod(_raise_corrupt),
        )

        verifier_corrupt = ClaimVerifier(use_custom_model=True, custom_model_path=fake_dir)
        _attach_baseline_stubs(verifier_corrupt, {claim.text: (1, 0.95)}, {claim.text: (0.92, 0.03, 0.05)})
        with pytest.raises(RuntimeError, match="Corrupted safetensors header"):
            verifier_corrupt.verify_claims([claim], [match])
    finally:
        CustomDebertaVerifier.clear_cache()

    # 3. Runtime inference failure must propagate
    class _FailingInferenceDeberta:
        def __init__(self, model_path: Optional[Union[str, Path]] = None) -> None:
            pass

        def predict_pairs(
            self, pairs: Sequence[Tuple[str, str]]
        ) -> List[CustomDebertaPrediction]:
            raise RuntimeError("Unexpected inference failure")

    monkeypatch.setattr(verifier_module, "CustomDebertaVerifier", _FailingInferenceDeberta)
    verifier_infer_fail = ClaimVerifier(use_custom_model=True)
    _attach_baseline_stubs(verifier_infer_fail, {claim.text: (1, 0.95)}, {claim.text: (0.92, 0.03, 0.05)})
    with pytest.raises(RuntimeError, match="Unexpected inference failure"):
        verifier_infer_fail.verify_claims([claim], [match])

    # 4. Mismatched prediction count from custom verifier must raise ValueError
    class _ShortInferenceDeberta:
        def __init__(self, model_path: Optional[Union[str, Path]] = None) -> None:
            pass

        def predict_pairs(
            self, pairs: Sequence[Tuple[str, str]]
        ) -> List[CustomDebertaPrediction]:
            return []

    monkeypatch.setattr(verifier_module, "CustomDebertaVerifier", _ShortInferenceDeberta)
    verifier_short = ClaimVerifier(use_custom_model=True)
    _attach_baseline_stubs(verifier_short, {claim.text: (1, 0.95)}, {claim.text: (0.92, 0.03, 0.05)})
    with pytest.raises(ValueError, match="predictions"):
        verifier_short.verify_claims([claim], [match])

    # Verify package-level export exposes CustomDebertaVerifier cleanly
    from ml_engine.claim_verifier import CustomDebertaVerifier as ExportedVerifier
    assert ExportedVerifier is CustomDebertaVerifier


def test_path_keyed_cache_reused_across_multiple_claim_verifiers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_dir = tmp_path / "cached_deberta"
    fake_dir.mkdir()
    (fake_dir / "config.json").write_text("{}", encoding="utf-8")
    (fake_dir / "model.safetensors").write_bytes(b"weights")
    (fake_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    load_counts = {"tok": 0, "mod": 0}

    class _DummyTokenizer:
        def __call__(
            self,
            contexts: List[str],
            claims: List[str],
            **kwargs: Any,
        ) -> Dict[str, torch.Tensor]:
            batch_len = len(contexts)
            return {"input_ids": torch.ones((batch_len, 4), dtype=torch.long)}

    class _DummyModel:
        def __init__(self) -> None:
            self.config = SimpleNamespace(
                id2label=dict(EXPECTED_ID2LABEL),
                label2id=dict(EXPECTED_LABEL2ID),
            )

        def to(self, _device: torch.device) -> "_DummyModel":
            return self

        def eval(self) -> "_DummyModel":
            return self

        def __call__(self, input_ids: torch.Tensor) -> SimpleNamespace:
            logits = torch.tensor([[0.1, 2.5, 0.2]] * input_ids.shape[0], dtype=torch.float32)
            return SimpleNamespace(logits=logits)

    def _mock_tok(*args: Any, **kwargs: Any) -> _DummyTokenizer:
        load_counts["tok"] += 1
        return _DummyTokenizer()

    def _mock_mod(*args: Any, **kwargs: Any) -> _DummyModel:
        load_counts["mod"] += 1
        return _DummyModel()

    CustomDebertaVerifier.clear_cache()
    try:
        monkeypatch.setattr(verifier_module, "CustomDebertaVerifier", CustomDebertaVerifier)
        monkeypatch.setattr(cdv_module.AutoTokenizer, "from_pretrained", staticmethod(_mock_tok))
        monkeypatch.setattr(
            cdv_module.AutoModelForSequenceClassification,
            "from_pretrained",
            staticmethod(_mock_mod),
        )

        v1 = ClaimVerifier(use_custom_model=True, custom_model_path=fake_dir)
        v2 = ClaimVerifier(use_custom_model=True, custom_model_path=fake_dir)

        claim = _make_claim("cache_1", "Apollo 11 launched in 1969.")
        match = _make_match(claim, [("Apollo 11 launched on July 16, 1969.", 0.90)])
        mc = {claim.text: (1, 0.96)}
        nli = {claim.text: (0.94, 0.02, 0.04)}
        _attach_baseline_stubs(v1, mc, nli)
        _attach_baseline_stubs(v2, mc, nli)

        r1 = v1.verify_claims([claim], [match])
        r2 = v2.verify_claims([claim], [match])

        assert load_counts == {"tok": 1, "mod": 1}
        assert CustomDebertaVerifier.cache_info() == {"tokenizers": 1, "models": 1}
        assert v1.custom_deberta is not None and v2.custom_deberta is not None
        assert v1.custom_deberta.tokenizer is v2.custom_deberta.tokenizer
        assert v1.custom_deberta.model is v2.custom_deberta.model
        assert r1.claims[0].scores.custom_deberta_predicted_label == "SUPPORTED_PROXY"
        assert r2.claims[0].scores.custom_deberta_predicted_label == "SUPPORTED_PROXY"
    finally:
        CustomDebertaVerifier.clear_cache()


@pytest.mark.skipif(
    not _local_checkpoint_present() or os.environ.get("VERILLM_SKIP_REAL_MODEL_TESTS") == "1",
    reason="Optional smoke test skipped when local Custom DeBERTa checkpoint is absent or disabled.",
)
def test_real_local_checkpoint_shadow_smoke() -> None:
    """Optional smoke test verifying real local DeBERTa loads and populates 3-class shadow scores."""
    c_sup = _make_claim("smoke_1", "Apollo 11 landed on the Moon in July 1969.")
    c_con = _make_claim("smoke_2", "The Eiffel Tower is located in Tokyo, Japan.")
    c_unsup = _make_claim("smoke_3", "The Eiffel Tower is painted bright purple every winter.")
    claims = [c_sup, c_con, c_unsup]
    matches = [
        _make_match(c_sup, [("Apollo 11 landed on the Moon on July 20, 1969.", 0.90)]),
        _make_match(c_con, [("The Eiffel Tower is located in Paris, France.", 0.85)]),
        _make_match(c_unsup, [("The Eiffel Tower was built in Paris for the 1889 World's Fair.", 0.72)]),
    ]

    # Stub only MiniCheck/NLI so this optional smoke test avoids loading MiniCheck RoBERTa-large.
    shadow_verifier = ClaimVerifier(use_custom_model=True)
    _attach_baseline_stubs(
        shadow_verifier,
        {
            c_sup.text: (1, 0.97),
            c_con.text: (0, 0.03),
            c_unsup.text: (0, 0.05),
        },
        {
            c_sup.text: (0.95, 0.02, 0.03),
            c_con.text: (0.01, 0.97, 0.02),
            c_unsup.text: (0.04, 0.05, 0.91),
        },
    )

    report = shadow_verifier.verify_claims(claims, matches, document_name="smoke_doc.pdf")
    scored = ResponseScorer().score_response(report, response_id="resp_smoke")

    assert len(report.claims) == 3
    for vc, summary in zip(report.claims, scored.claims):
        probs = [
            vc.scores.custom_deberta_contradiction_prob,
            vc.scores.custom_deberta_supported_prob,
            vc.scores.custom_deberta_unsupported_prob,
        ]
        assert all(p is not None and math.isfinite(p) and 0.0 <= p <= 1.0 for p in probs)
        assert sum(p for p in probs if p is not None) == pytest.approx(1.0, abs=1e-5)
        assert vc.scores.custom_deberta_predicted_label in VALID_LABELS
        assert summary.custom_deberta_predicted_label == vc.scores.custom_deberta_predicted_label
        assert summary.custom_deberta_supported_prob == pytest.approx(
            vc.scores.custom_deberta_supported_prob
        )
        assert summary.custom_deberta_contradiction_prob == pytest.approx(
            vc.scores.custom_deberta_contradiction_prob
        )
        assert summary.custom_deberta_unsupported_prob == pytest.approx(
            vc.scores.custom_deberta_unsupported_prob
        )
