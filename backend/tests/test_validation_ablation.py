"""
Unit and contract tests for Phase 8 Step 4 validation ablation utilities.

Tests run completely offline using synthetic/mocked data and NEVER read
ragtruth_test.jsonl or any real model weights.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml_engine.claim_verifier.models import (
    ModelVerificationScore,
    VerificationCategory,
    VerificationMetrics,
    VerificationReport,
    VerifiedClaim,
)
from ml_engine.custom_model.validation_ablation import (
    ARM_NAMES,
    MIXED_LABEL,
    audit_duplicate_and_leakage,
    build_train_overlap_index,
    compute_arm_metrics,
    compute_response_level_evaluation,
    evaluate_direct_context_records,
    evaluate_minority_adequacy,
    evaluate_retrieval_pipeline_for_source,
    exact_mcnemar_test,
    load_validation_records,
    predict_arm_for_record,
    select_pilot_sources,
    source_clustered_bootstrap_diff,
)
from ml_engine.evidence_retriever.chunker import SlidingWindowChunker
from ml_engine.evidence_retriever.models import ClaimEvidenceMatch, RetrievedPassage


def _make_record(
    example_id: str,
    source_id: str,
    response_id: str,
    task_type: str,
    primary_label: str,
    sentence_text: str,
    context_text: str,
    split: str = "val",
    model: str = "gpt-4",
) -> dict:
    return {
        "example_id": example_id,
        "response_id": response_id,
        "source_id": source_id,
        "task_type": task_type,
        "model": model,
        "split": split,
        "primary_label": primary_label,
        "sentence_text": sentence_text,
        "sentence_start": 0,
        "sentence_end": len(sentence_text),
        "context_text": context_text,
        "has_hallucination": primary_label != "SUPPORTED_PROXY",
        "max_overlap_ratio": 1.0 if primary_label != "SUPPORTED_PROXY" else 0.0,
        "has_implicit_true": False,
        "has_due_to_null": False,
        "overlapping_spans": [],
    }


def test_split_isolation_rejects_test_paths_and_non_val_records(tmp_path: Path):
    test_named = tmp_path / "ragtruth_test.jsonl"
    test_named.write_text(
        json.dumps(
            _make_record("x1", "s1", "r1", "QA", "SUPPORTED_PROXY", "hi.", "ctx", split="val")
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="forbidden test-like"):
        load_validation_records(test_named)

    # Even a differently named file containing a test/train record must fail
    bad_split = tmp_path / "val_candidate.jsonl"
    bad_split.write_text(
        json.dumps(
            _make_record("x1", "s1", "r1", "QA", "SUPPORTED_PROXY", "hi.", "ctx", split="test")
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Split isolation violation"):
        load_validation_records(bad_split)

    # Train index also rejects test-like paths and non-train split lines
    with pytest.raises(ValueError, match="forbidden test-like"):
        build_train_overlap_index(test_named)

    bad_train = tmp_path / "train_candidate.jsonl"
    bad_train.write_text(
        json.dumps(
            _make_record("x1", "s1", "r1", "QA", "SUPPORTED_PROXY", "hi.", "ctx", split="val")
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="expected split='train'"):
        build_train_overlap_index(bad_train)


def test_duplicate_and_leakage_audit_separates_exact_pair_vs_sentence_string(tmp_path: Path):
    train_file = tmp_path / "ragtruth_train.jsonl"
    train_recs = [
        _make_record("t1", "tr_s1", "tr_r1", "Summary", "SUPPORTED_PROXY", "Paris is in France.", "Context Alpha.", split="train"),
        _make_record("t2", "tr_s2", "tr_r2", "QA", "SUPPORTED_PROXY", "No data available.", "Context Beta.", split="train"),
    ]
    train_file.write_text("\n".join(json.dumps(r) for r in train_recs) + "\n", encoding="utf-8")
    train_index = build_train_overlap_index(train_file)

    val_recs = [
        # Exact (context, sentence) duplicate with train (t1) despite different source_id
        _make_record("v1", "val_s1", "val_r1", "Summary", "SUPPORTED_PROXY", "Paris is in France.", "Context Alpha.", split="val"),
        # Sentence string match ONLY (different context)
        _make_record("v2", "val_s2", "val_r2", "QA", "UNSUPPORTED", "No data available.", "Context Gamma.", split="val"),
        # Completely novel
        _make_record("v3", "val_s3", "val_r3", "Data2txt", "CONTRADICTED", "Brand new claim.", "Context Delta.", split="val"),
    ]

    audit = audit_duplicate_and_leakage(val_recs, train_index=train_index)
    assert audit["source_id_overlap_count"] == 0
    assert audit["exact_pair_duplicate_with_train_count"] == 1
    assert audit["exact_pair_duplicate_example_ids"] == ["v1"]
    assert audit["sentence_string_in_train_total"] == 2
    assert audit["sentence_string_only_overlap_count"] == 1

    # Source ID collision must fail loudly
    leaking = val_recs + [
        _make_record("v4", "tr_s1", "val_r4", "Summary", "SUPPORTED_PROXY", "Other.", "Other ctx.", split="val")
    ]
    with pytest.raises(ValueError, match="Source leakage detected"):
        audit_duplicate_and_leakage(leaking, train_index=train_index)


def test_deterministic_source_grouped_pilot_revises_for_minority_adequacy():
    val_records = []
    # Build 6 sources per task (Summary, QA, Data2txt). Only 2 sources per task carry CONTRADICTED,
    # and we build them so pure random sampling may miss minority thresholds, exercising Stage 2.
    idx = 0
    for task in ("Summary", "QA", "Data2txt"):
        for s_num in range(6):
            sid = f"{task}_{s_num}"
            ctx = f"Source passage for {sid} with several factual entities and numbers."
            # Sources 0 and 1 have CONTRADICTED + UNSUPPORTED; others only SUPPORTED_PROXY
            labels = (
                ["CONTRADICTED", "UNSUPPORTED", "SUPPORTED_PROXY", "SUPPORTED_PROXY", MIXED_LABEL]
                if s_num < 3
                else ["SUPPORTED_PROXY", "SUPPORTED_PROXY", "SUPPORTED_PROXY", "UNSUPPORTED", "SUPPORTED_PROXY"]
            )
            for r_num in range(2):
                rid = f"{sid}_resp_{r_num}"
                for j, lbl in enumerate(labels):
                    idx += 1
                    val_records.append(
                        _make_record(
                            example_id=f"ex_{idx:04d}",
                            source_id=sid,
                            response_id=rid,
                            task_type=task,
                            primary_label=lbl,
                            sentence_text=f"Claim {idx} for {rid}.",
                            context_text=ctx,
                        )
                    )

    manifest_a = select_pilot_sources(
        val_records,
        sources_per_task=4,
        seed=42,
        min_total_contradicted=12,
        min_total_unsupported=12,
        min_contradicted_per_task=4,
        min_unsupported_per_task=4,
    )
    manifest_b = select_pilot_sources(
        val_records,
        sources_per_task=4,
        seed=42,
        min_total_contradicted=12,
        min_total_unsupported=12,
        min_contradicted_per_task=4,
        min_unsupported_per_task=4,
    )

    assert manifest_a["selected_source_ids"] == manifest_b["selected_source_ids"]
    assert manifest_a["total_sources"] == 12
    assert manifest_a["final_adequacy"]["is_adequate"] is True
    assert manifest_a["final_adequacy"]["counts_by_label"]["CONTRADICTED"] >= 12

    # Check complete sources/responses invariant: every sentence for every picked source is accounted for
    selected_set = set(manifest_a["selected_source_ids"])
    expected_sent_count = sum(1 for r in val_records if r["source_id"] in selected_set)
    assert manifest_a["total_sentences"] == expected_sent_count

    # Check locked_source_ids prefix expansion preserves initial sources
    expanded = select_pilot_sources(
        val_records,
        sources_per_task=5,
        seed=42,
        min_total_contradicted=12,
        min_total_unsupported=12,
        min_contradicted_per_task=4,
        min_unsupported_per_task=4,
        locked_source_ids=manifest_a["selected_source_ids"],
    )
    assert set(manifest_a["selected_source_ids"]).issubset(set(expanded["selected_source_ids"]))
    assert expanded["total_sources"] == 15


def test_metrics_handle_unverifiable_and_isolate_mixed_hallucination():
    records = [
        _make_record("1", "s1", "r1", "QA", "SUPPORTED_PROXY", "a", "ctx"),
        _make_record("2", "s1", "r1", "QA", "SUPPORTED_PROXY", "b", "ctx"),
        _make_record("3", "s1", "r1", "QA", "CONTRADICTED", "c", "ctx"),
        _make_record("4", "s1", "r1", "QA", "UNSUPPORTED", "d", "ctx"),
        _make_record("5", "s1", "r1", "QA", MIXED_LABEL, "e", "ctx"),
    ]
    preds = [
        "SUPPORTED_PROXY",  # correct
        "UNVERIFIABLE",     # abstention on SUPPORTED_PROXY -> reduces coverage & recall
        "CONTRADICTED",     # correct
        "UNSUPPORTED",      # correct
        "CONTRADICTED",     # MIXED -> separate diagnostic only!
    ]
    scores = [0.1, 0.4, 0.9, 0.85, 0.95]

    m = compute_arm_metrics(records, preds, continuous_scores=scores)
    assert m["n_total_3class"] == 4
    assert m["n_unverifiable"] == 1
    assert m["n_covered"] == 3
    assert m["coverage"] == pytest.approx(0.75)
    assert m["abstention_rate"] == pytest.approx(0.25)
    assert m["abstention_true_label_breakdown"] == {
        "CONTRADICTED": 0,
        "SUPPORTED_PROXY": 1,
        "UNSUPPORTED": 0,
    }
    # Penalized accuracy = 3/4 = 0.75; Selective accuracy = 3/3 = 1.0
    assert m["penalized_3class"]["accuracy"] == pytest.approx(0.75)
    assert m["selective_3class"]["accuracy"] == pytest.approx(1.0)
    assert m["selective_3class"]["covered_of_total"] == "3/4"
    # SUPPORTED_PROXY penalized recall = 1/2 = 0.5, prec = 1/1 = 1.0 -> F1 = 2/3
    assert m["penalized_3class"]["per_class"]["SUPPORTED_PROXY"]["recall"] == pytest.approx(0.5)
    assert m["penalized_3class"]["per_class"]["SUPPORTED_PROXY"]["f1"] == pytest.approx(2.0 / 3.0)
    assert m["penalized_3class"]["confusion_3x4"]["SUPPORTED_PROXY"]["UNVERIFIABLE"] == 1
    # Mixed diagnostic kept separate
    assert m["mixed_hallucination_diagnostic"]["n_mixed"] == 1
    assert m["mixed_hallucination_diagnostic"]["flagged_confirmed_hallucination_count"] == 1
    assert m["mixed_hallucination_diagnostic"]["flagged_confirmed_hallucination_rate"] == pytest.approx(1.0)


def test_direct_and_retrieval_runners_and_all_non_averaging_arms():
    rec1 = _make_record(
        "e1", "src1", "resp1", "Summary", "CONTRADICTED",
        "The sky is green.", "Evidence states clearly that the sky is blue."
    )
    rec2 = _make_record(
        "e2", "src1", "resp1", "Summary", "SUPPORTED_PROXY",
        "The sky is blue.", "Evidence states clearly that the sky is blue."
    )

    class MockMiniCheck:
        def score_pairs(self, docs, claims):
            return [(0, 0.1) if "green" in c else (1, 0.95) for c in claims]

    class MockNLI:
        def predict_pairs(self, pairs):
            out = []
            for _, c in pairs:
                if "green" in c:
                    out.append((0.05, 0.85, 0.10))
                else:
                    out.append((0.90, 0.05, 0.05))
            return out

    class MockCustomDeberta:
        def predict_pairs(self, pairs):
            out = []
            for _, c in pairs:
                if "green" in c:
                    out.append(
                        SimpleNamespace(
                            predicted_label="CONTRADICTED",
                            supported_proxy_prob=0.05,
                            contradicted_prob=0.90,
                            unsupported_prob=0.05,
                        )
                    )
                else:
                    out.append(
                        SimpleNamespace(
                            predicted_label="SUPPORTED_PROXY",
                            supported_proxy_prob=0.92,
                            contradicted_prob=0.04,
                            unsupported_prob=0.04,
                        )
                    )
            return out

    class MockPassageRetriever:
        def index_chunks(self, chunks):
            self._chunks = chunks

        def retrieve_for_claims(self, claims):
            matches = []
            for c in claims:
                p = RetrievedPassage(chunk=self._chunks[0], similarity_score=0.80, rank=1)
                matches.append(
                    ClaimEvidenceMatch(
                        claim_id=c.claim_id,
                        claim_text=c.text,
                        retrieved_passages=[p],
                        best_passage=p,
                        best_similarity_score=0.80,
                    )
                )
            return matches

    class MockClaimVerifier:
        def __init__(self):
            self.nli = MockNLI()
            self.decision_strategy = SimpleNamespace(conflict_threshold=0.70)

        def verify_claims(self, claims, evidence_matches, document_name=None):
            verified = []
            for c, m in zip(claims, evidence_matches):
                is_green = "green" in c.text
                vc = VerifiedClaim(
                    claim_id=c.claim_id,
                    claim_text=c.text,
                    source_sentence=c.source_sentence,
                    start_char=c.start_char,
                    end_char=c.end_char,
                    is_factual=True,
                    classification=(
                        VerificationCategory.CONTRADICTED
                        if is_green
                        else VerificationCategory.SUPPORTED
                    ),
                    decision_rule="Rule 2" if is_green else "Rule 5",
                    reasoning_trace="mock",
                    primary_evidence=m.best_passage,
                    all_evidence=m.retrieved_passages,
                    scores=ModelVerificationScore(
                        minicheck_probability=0.10 if is_green else 0.95,
                        nli_entailment_prob=0.05 if is_green else 0.90,
                        nli_contradiction_prob=0.85 if is_green else 0.05,
                        nli_neutral_prob=0.10 if is_green else 0.05,
                        retrieval_similarity=0.80,
                        custom_deberta_supported_prob=0.05 if is_green else 0.92,
                        custom_deberta_contradiction_prob=0.90 if is_green else 0.04,
                        custom_deberta_unsupported_prob=0.05 if is_green else 0.04,
                        custom_deberta_predicted_label=(
                            "CONTRADICTED" if is_green else "SUPPORTED_PROXY"
                        ),
                    ),
                )
                verified.append(vc)
            return VerificationReport(
                document_name=document_name,
                total_claims_processed=len(verified),
                claims=verified,
                metrics=VerificationMetrics(
                    total_claims=2,
                    factual_claims=2,
                    non_factual_claims=0,
                    supported_count=1,
                    contradicted_count=1,
                    unsupported_count=0,
                    unverifiable_count=0,
                    non_factual_count=0,
                    has_factual_claims=True,
                    denominator_description="mock",
                ),
                execution_time_seconds=0.01,
            )

    direct_sig = evaluate_direct_context_records(
        [rec1, rec2],
        minicheck_verifier=MockMiniCheck(),
        nli_verifier=MockNLI(),
        custom_deberta_verifier=MockCustomDeberta(),
    )
    ret_sig = evaluate_retrieval_pipeline_for_source(
        [rec1, rec2],
        chunker=SlidingWindowChunker(chunk_size_words=50, overlap_words=10, min_chunk_words=3),
        passage_retriever=MockPassageRetriever(),
        claim_verifier=MockClaimVerifier(),
    )

    cached = []
    for r in (rec1, rec2):
        row = dict(r)
        row["direct"] = direct_sig[r["example_id"]]
        row["retrieval"] = ret_sig[r["example_id"]]
        cached.append(row)

    for regime in ("direct", "retrieval"):
        for arm in ARM_NAMES:
            p1, _ = predict_arm_for_record(cached[0], regime=regime, arm_name=arm)
            p2, _ = predict_arm_for_record(cached[1], regime=regime, arm_name=arm)
            if arm == "arm0b_minicheck_only":
                assert (p1, p2) == ("UNSUPPORTED", "SUPPORTED_PROXY")
            else:
                assert (p1, p2) == ("CONTRADICTED", "SUPPORTED_PROXY")

    # Test response-level ResponseScorer alignment and statistical tests
    resp_eval = compute_response_level_evaluation(
        cached, ["CONTRADICTED", "SUPPORTED_PROXY"]
    )
    assert resp_eval["num_responses"] == 1
    assert resp_eval["confirmed_hallucination_rate_pct"]["mae"] == pytest.approx(0.0)


def test_statistical_tests_mcnemar_and_clustered_bootstrap():
    y_true = ["CONTRADICTED", "UNSUPPORTED", "SUPPORTED_PROXY", "SUPPORTED_PROXY"] * 5
    pred_base = ["UNSUPPORTED", "UNSUPPORTED", "SUPPORTED_PROXY", "SUPPORTED_PROXY"] * 5
    pred_cand = ["CONTRADICTED", "UNSUPPORTED", "SUPPORTED_PROXY", "SUPPORTED_PROXY"] * 5

    res = exact_mcnemar_test(y_true, pred_base, pred_cand)
    # Candidate fixes all 5 CONTRADICTED examples that baseline got wrong (b=0, c=5)
    assert res["three_class_exact"]["baseline_only_correct"] == 0
    assert res["three_class_exact"]["candidate_only_correct"] == 5
    # Two-sided exact binomial p = 2 * (0.5^5) = 0.0625
    assert res["three_class_exact"]["exact_two_sided_p_value"] == pytest.approx(0.0625)

    recs = [
        _make_record(
            example_id=f"id_{i}",
            source_id=f"src_{i // 4}",
            response_id=f"resp_{i // 2}",
            task_type="QA" if i < 12 else "Summary",
            primary_label=y_true[i],
            sentence_text=f"s{i}",
            context_text="ctx",
        )
        for i in range(len(y_true))
    ]
    boot = source_clustered_bootstrap_diff(
        recs,
        pred_base,
        pred_cand,
        metric_keys=("macro_f1_penalized", "contradicted_f1_penalized"),
        n_boot=200,
        seed=42,
    )
    assert boot["macro_f1_penalized"]["delta_point"] > 0.0
    assert boot["contradicted_f1_penalized"]["candidate_point"] == pytest.approx(1.0)
    assert boot["macro_f1_penalized"]["ci95_low"] <= boot["macro_f1_penalized"]["ci95_high"]


def test_step6_discrete_arms_omit_continuous_scores_and_auroc_none() -> None:
    from ml_engine.custom_model.validation_ablation import (
        ARM_NAMES,
        compute_arm_metrics,
        predict_arm_for_record,
    )

    rec = {
        "example_id": "ex1",
        "response_id": "resp1",
        "source_id": "src1",
        "task_type": "Summary",
        "model": "gpt-4-0613",
        "primary_label": "SUPPORTED_PROXY",
        "sentence_text": "The company launched in 2020.",
        "has_implicit_true": False,
        "has_due_to_null": False,
        "direct": {
            "mc_prob": 0.9,
            "nli_label": "entailment",
            "nli_entail": 0.85,
            "nli_contra": 0.05,
            "nli_neutral": 0.10,
            "deberta_predicted_label": "SUPPORTED_PROXY",
            "deberta_supported_prob": 0.8,
            "deberta_contradicted_prob": 0.1,
            "deberta_unsupported_prob": 0.1,
            "exceeds_512_tokens": False,
        },
    }
    native_scorer_arms = {"arm0a_nli_only", "arm0b_minicheck_only", "arm2_custom_deberta_only"}
    for arm in ARM_NAMES:
        pred, score = predict_arm_for_record(rec, regime="direct", arm_name=arm)
        assert pred in {"SUPPORTED_PROXY", "CONTRADICTED", "UNSUPPORTED", "UNVERIFIABLE"}
        if arm in native_scorer_arms:
            assert score is not None
        else:
            assert score is None

    rec2 = {**rec, "example_id": "ex2", "primary_label": "CONTRADICTED"}
    m = compute_arm_metrics(
        [rec, rec2],
        ["SUPPORTED_PROXY", "CONTRADICTED"],
        continuous_scores=[None, None],
    )
    assert m["binary_hallucination_penalized"]["auroc"] is None
    assert m["binary_hallucination_penalized"]["pr_auc"] is None


def test_step6_mcnemar_underflow_precision_holm_bonferroni_and_direction() -> None:
    from ml_engine.custom_model.validation_ablation import (
        apply_holm_bonferroni,
        exact_mcnemar_test,
    )

    # Construct b=400 (baseline only correct), c=10 (candidate only correct)
    n = 410
    y_true = ["SUPPORTED_PROXY"] * n
    y_base = ["SUPPORTED_PROXY"] * 400 + ["CONTRADICTED"] * 10
    y_cand = ["CONTRADICTED"] * 400 + ["SUPPORTED_PROXY"] * 10
    mcn = exact_mcnemar_test(y_true, y_base, y_cand)["three_class_exact"]
    assert mcn["baseline_only_correct"] == 400
    assert mcn["candidate_only_correct"] == 10
    assert mcn["net_gain_cand_minus_base"] == -390
    assert mcn["direction"] == "baseline_better"
    assert mcn["exact_two_sided_p_value"] > 0.0
    assert "e-" in mcn["p_value_sci"]
    assert mcn["log10_p_value"] < -50.0
    assert "anti-conservative" in mcn["clustering_caveat"]

    # Extreme b=1200, c=0 where float64 underflows below 1e-308 -> finite log10_p_value preserved
    y_true_ext = ["SUPPORTED_PROXY"] * 1200
    y_base_ext = ["SUPPORTED_PROXY"] * 1200
    y_cand_ext = ["CONTRADICTED"] * 1200
    mcn_ext = exact_mcnemar_test(y_true_ext, y_base_ext, y_cand_ext)["three_class_exact"]
    assert mcn_ext["log10_p_value"] < -300.0
    assert "e-" in mcn_ext["p_value_sci"]

    # Step-down Holm-Bonferroni check
    adj = apply_holm_bonferroni([0.001, 0.04, 0.03, 0.2])
    assert adj == pytest.approx([0.004, 0.09, 0.09, 0.2])


def test_step6_clustered_paired_comparison_weighted_bootstrap_and_response_bootstrap() -> None:
    from ml_engine.custom_model.validation_ablation import (
        clustered_paired_comparison,
        response_level_bootstrap_diff,
        source_clustered_bootstrap_diff,
    )

    records = []
    base_preds = []
    cand_preds = []
    models = ["gpt-4-0613", "llama-2-7b-chat", "mistral-7B-instruct"]
    labels = ["SUPPORTED_PROXY", "CONTRADICTED", "UNSUPPORTED"]
    for s_idx in range(6):
        src = f"src_{s_idx}"
        mdl = models[s_idx % len(models)]
        for r_off in range(2):
            resp = f"{src}_r{r_off}"
            for c_idx in range(3):
                true_lbl = labels[(s_idx + c_idx) % 3]
                records.append(
                    {
                        "example_id": f"{resp}_c{c_idx}",
                        "response_id": resp,
                        "source_id": src,
                        "task_type": "QA",
                        "model": mdl,
                        "primary_label": true_lbl,
                    }
                )
                # Baseline mostly right on src 0..1, candidate consistently better on src 2..5
                if s_idx < 2:
                    base_preds.append(true_lbl)
                    cand_preds.append("UNVERIFIABLE" if c_idx == 0 else true_lbl)
                else:
                    base_preds.append("SUPPORTED_PROXY")
                    cand_preds.append(true_lbl)

    weights = {f"src_{i}": (3.0 if i < 2 else 0.5) for i in range(6)}
    src_cmp = clustered_paired_comparison(
        records,
        base_preds,
        cand_preds,
        cluster_key="source_id",
        source_weights=weights,
        n_perm=200,
        seed=7,
    )
    c3 = src_cmp["three_class_accuracy"]
    assert c3["n_clusters"] == 6
    assert c3["positive_clusters"] == 4
    assert c3["negative_clusters"] == 2
    assert c3["mean_cluster_delta_unweighted"] != pytest.approx(c3["mean_cluster_delta_weighted"])
    assert 0.0 <= c3["paired_sign_flip_p_value"] <= 1.0
    assert 0.0 <= c3["exact_sign_test_p_value"] <= 1.0

    resp_cmp = clustered_paired_comparison(
        records,
        base_preds,
        cand_preds,
        cluster_key="response_id",
        n_perm=200,
        seed=7,
    )
    assert resp_cmp["three_class_accuracy"]["n_clusters"] == 12
    assert resp_cmp["three_class_accuracy"]["wilcoxon_signed_rank_p_value"] is not None

    boot_unw = source_clustered_bootstrap_diff(
        records, base_preds, cand_preds, source_weights=None, n_boot=60, seed=7
    )
    boot_wt = source_clustered_bootstrap_diff(
        records, base_preds, cand_preds, source_weights=weights, n_boot=60, seed=7
    )
    assert "support_weighted_f1_penalized" in boot_unw
    assert boot_unw["accuracy_penalized"]["delta_point"] != pytest.approx(
        boot_wt["accuracy_penalized"]["delta_point"]
    )

    resp_boot = response_level_bootstrap_diff(
        records, base_preds, cand_preds, n_boot=60, seed=7
    )
    assert "confirmed_hall_mae_pct_delta" in resp_boot
    assert "gfi_spearman_rho_delta" in resp_boot


def test_step6_response_level_confirmed_vs_unverifiable_and_generator_llm_ranking() -> None:
    from ml_engine.custom_model.validation_ablation import compute_response_level_evaluation

    records = [
        {"example_id": "1", "response_id": "r1", "source_id": "s1", "task_type": "QA", "model": "gpt-4-0613", "primary_label": "SUPPORTED_PROXY", "sentence_text": "Claim one."},
        {"example_id": "2", "response_id": "r1", "source_id": "s1", "task_type": "QA", "model": "gpt-4-0613", "primary_label": "SUPPORTED_PROXY", "sentence_text": "Claim two."},
        {"example_id": "3", "response_id": "r2", "source_id": "s2", "task_type": "QA", "model": "llama-2-7b-chat", "primary_label": "CONTRADICTED", "sentence_text": "Claim three."},
        {"example_id": "4", "response_id": "r2", "source_id": "s2", "task_type": "QA", "model": "llama-2-7b-chat", "primary_label": "UNSUPPORTED", "sentence_text": "Claim four."},
    ]
    preds = ["UNVERIFIABLE", "SUPPORTED_PROXY", "CONTRADICTED", "UNSUPPORTED"]
    re_out = compute_response_level_evaluation(records, preds)
    assert re_out["confirmed_hallucination_rate_pct"]["mean_pred_confirmed_pct"] == pytest.approx(50.0)
    assert re_out["unverifiable_rate_pct"]["mean_pred_pct"] == pytest.approx(25.0)
    assert re_out["non_supported_rate_100_minus_gfi_pct"]["mean_pred_pct"] == pytest.approx(75.0)
    assert set(re_out["by_generator_model"]["per_model"].keys()) == {"gpt-4-0613", "llama-2-7b-chat"}


def test_step6_zero_support_slice_and_weighted_ht_metrics_and_two_phase_provenance() -> None:
    from ml_engine.custom_model.validation_ablation import (
        compute_arm_metrics,
        compute_weighted_three_class_metrics,
        select_pilot_sources,
    )

    records = [
        {"example_id": "1", "response_id": "r1", "source_id": "s1", "task_type": "Summary", "model": "m1", "primary_label": "SUPPORTED_PROXY"},
        {"example_id": "2", "response_id": "r2", "source_id": "s2", "task_type": "QA", "model": "m1", "primary_label": "CONTRADICTED"},
        {"example_id": "3", "response_id": "r3", "source_id": "s3", "task_type": "Data2txt", "model": "m1", "primary_label": "SUPPORTED_PROXY"},
    ]
    preds = ["SUPPORTED_PROXY", "CONTRADICTED", "SUPPORTED_PROXY"]
    m = compute_arm_metrics(records, preds)
    p3 = m["penalized_3class"]
    assert p3["present_classes_count"] == 2
    assert p3["macro_f1_present_classes"] == pytest.approx(1.0)
    assert p3["macro_f1"] == pytest.approx(2.0 / 3.0)
    assert p3["support_weighted_f1"] == pytest.approx(1.0)

    w = compute_weighted_three_class_metrics(
        records,
        ["SUPPORTED_PROXY", "SUPPORTED_PROXY", "SUPPORTED_PROXY"],
        source_weights={"s1": 1.0, "s2": 2.0, "s3": 1.0},
    )
    # 2/4 weight on s1+s3 (correct), 2/4 weight on s2 (incorrect) -> accuracy 0.5 unrounded
    assert w["weighted_accuracy_penalized"] == pytest.approx(0.5, abs=1e-6)

    train_idx = {
        "source_ids": set(),
        "context_hashes": set(),
        "exact_pair_keys": set(),
        "sentence_strings": set(),
    }
    val_records = []
    for task in ("Summary", "QA", "Data2txt"):
        for k in range(3):
            sid = f"{task}_{k}"
            val_records.append(
                {
                    "example_id": f"{sid}_e1",
                    "response_id": f"{sid}_r1",
                    "source_id": sid,
                    "task_type": task,
                    "model": "gpt-4",
                    "primary_label": "CONTRADICTED" if k == 0 else "SUPPORTED_PROXY",
                    "context_text": f"ctx {sid}",
                    "sentence_text": f"sent {sid}",
                }
            )
    phase1 = select_pilot_sources(
        val_records,
        train_idx,
        sources_per_task=1,
        seed=42,
        min_total_contradicted=0,
        min_total_unsupported=0,
        min_contradicted_per_task=0,
        min_unsupported_per_task=0,
    )
    phase2 = select_pilot_sources(
        val_records,
        train_idx,
        sources_per_task=2,
        seed=42,
        min_total_contradicted=0,
        min_total_unsupported=0,
        min_contradicted_per_task=0,
        min_unsupported_per_task=0,
        locked_source_ids=phase1["selected_source_ids"],
        locked_manifest_path="dummy_phase1_manifest.json",
        locked_selection_method=phase1["selection_method"],
    )
    assert phase2["selection_method"] == "two_phase_locked_prefix_plus_uniform_followup"
    assert phase2["locked_manifest_path"] == "dummy_phase1_manifest.json"
    assert len(phase2["stratum_sampling_summary"]) > 0
