"""
Phase 8 Step 7: Deterministic Unit Tests for Calibration & Hybrid Verification.

Tests verify:
1. Consistent source-grouped 5x4 fold assignments across regimes with zero source leakage.
2. validate_fold_class_support raises ValueError on inadequate support; summarize_source_class_support reports absent classes safely.
3. Logit reconstruction and MulticlassCalibrator (C0, C1, C2) / BinaryPlattCalibrator fitting and serialization.
4. fast_eval_fold_predictions numerical parity with compute_arm_metrics and compute_response_level_evaluation.
5. score_objective respects pre-registered alpha=0.005, MAE_max ceiling, recall floors, and deterministic tie-breakers.
6. Semantic guards (NON_FACTUAL, UNVERIFIABLE) and explicit Stage 1 H3_sec_deb exclusion metadata.
7. End-to-end synthetic Stage 1 nested CV, Stage 2 bundle freezing, and held-out Val-Lock evaluation.
8. Safety guards blocking ragtruth_test paths and unapproved --stage2-run execution.

ZERO neural inference is executed by these tests.
"""

from __future__ import annotations

import copy
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml_engine.custom_model.calibration import (
    CALIBRATION_CLASSES,
    BinaryPlattCalibrator,
    MulticlassCalibrator,
    assert_disjoint_sources,
    assign_consistent_source_folds,
    compute_binary_calibration_metrics,
    compute_multiclass_calibration_metrics,
    extract_deberta_logits,
    extract_deberta_probs,
    extract_minicheck_prob,
    extract_nli_probs,
    fit_binary_platt_calibrator,
    fit_multiclass_calibrator,
    reconstruct_logits_from_probs,
    summarize_source_class_support,
    validate_fold_class_support,
)
from ml_engine.custom_model.hybrid_strategies import (
    INT_TO_PRED_LABEL,
    PRE_REGISTERED_OBJECTIVE_PARAMS,
    STAGE1_STRATEGY_ORDER,
    STRATEGY_REGISTRY,
    apply_frozen_bundle_to_records,
    extract_feature_arrays,
    fast_eval_fold_predictions,
    freeze_stage2_bundle,
    predict_h3_sec_deb,
    run_nested_source_cv_stage1,
    score_objective,
    select_best_candidate,
)
from ml_engine.custom_model.validation_ablation import (
    compute_arm_metrics,
    compute_response_level_evaluation,
)

# experiments/10_calibration_and_hybrid_study starts with a number, import via importlib
study_module = importlib.import_module("experiments.10_calibration_and_hybrid_study")


# ---------------------------------------------------------------------------
# Synthetic Test Fixtures (Zero Neural Inference)
# ---------------------------------------------------------------------------

def _make_synthetic_record(
    example_id: str,
    response_id: str,
    source_id: str,
    task_type: str,
    label: str,
    is_factual: bool = True,
    mc_prob: float = 0.8,
    deb_probs: Tuple[float, float, float] = (0.1, 0.8, 0.1),
    nli_scores: Tuple[float, float, float] = (0.1, 0.8, 0.1),
    sim_score: float = 0.75,
    empty_ev: bool = False,
    conflict_triggered: bool = False,
) -> Dict[str, Any]:
    c, s, u = deb_probs
    nc, ne, nn = nli_scores
    return {
        "example_id": example_id,
        "response_id": response_id,
        "source_id": source_id,
        "task_type": task_type,
        "model": "gpt-4",
        "split": "val",
        "primary_label": label,
        "sentence_text": f"Claim sentence for {example_id}",
        "context_text": f"Context text for {source_id}",
        "is_factual": is_factual,
        "direct": {
            "mc_label": 1 if mc_prob >= 0.5 else 0,
            "mc_prob": mc_prob,
            "nli_contra": nc,
            "nli_entail": ne,
            "nli_neutral": nn,
            "deberta_contradiction_prob": c,
            "deberta_supported_prob": s,
            "deberta_unsupported_prob": u,
            "deberta_predicted_label": (
                "CONTRADICTED" if c > s and c > u else ("SUPPORTED_PROXY" if s > u else "UNSUPPORTED")
            ),
        },
        "retrieval": {
            "mc_label": 1 if mc_prob >= 0.5 else 0,
            "mc_prob": mc_prob,
            "nli_contra": nc,
            "nli_entail": ne,
            "nli_neutral": nn,
            "deberta_contradiction_prob": c,
            "deberta_supported_prob": s,
            "deberta_unsupported_prob": u,
            "deberta_predicted_label": (
                "CONTRADICTED" if c > s and c > u else ("SUPPORTED_PROXY" if s > u else "UNSUPPORTED")
            ),
            "best_similarity_score": sim_score,
            "relevant_passages_count": 0 if empty_ev else 3,
            "baseline_decision_rule": "Rule 2: Empty evidence" if empty_ev else "Rule 3: Top-1",
            "conflict_triggered": conflict_triggered,
            "sec_nli_max_contra": 0.85 if conflict_triggered else 0.1,
            "sec_nli_max_entail": 0.75 if conflict_triggered else 0.2,
        },
    }


def _make_synthetic_corpus(n_sources: int = 15) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    tasks = ["Summary", "QA", "Data2txt"]
    for s_idx in range(n_sources):
        sid = f"src_{s_idx:03d}"
        task = tasks[s_idx % len(tasks)]
        for r_idx in range(2):
            rid = f"resp_{sid}_{r_idx}"
            # 4 claims per response: mix of supported, contradicted, unsupported, mixed
            recs_spec = [
                ("SUPPORTED_PROXY", (0.05, 0.90, 0.05), 0.92),
                ("SUPPORTED_PROXY", (0.10, 0.80, 0.10), 0.85),
                ("UNSUPPORTED", (0.15, 0.25, 0.60), 0.30),
                ("CONTRADICTED", (0.75, 0.10, 0.15), 0.15),
            ]
            for c_idx, (lbl, dp, mc) in enumerate(recs_spec):
                eid = f"{rid}_c{c_idx}"
                records.append(
                    _make_synthetic_record(
                        example_id=eid,
                        response_id=rid,
                        source_id=sid,
                        task_type=task,
                        label=lbl,
                        mc_prob=mc,
                        deb_probs=dp,
                        nli_scores=(dp[0], dp[1], dp[2]),
                    )
                )
    return records


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

def test_consistent_source_fold_assignment_zero_leakage():
    corpus = _make_synthetic_corpus(n_sources=15)
    plan = assign_consistent_source_folds(corpus, n_outer_folds=5, n_inner_folds=4, seed=42)

    assert plan["n_outer_folds"] == 5
    assert plan["n_inner_folds"] == 4
    assert plan["total_sources"] == 15

    # Check outer disjointness
    for outer in plan["outer_folds"]:
        tr_s = set(outer["train_sources"])
        ev_s = set(outer["eval_sources"])
        assert len(tr_s.intersection(ev_s)) == 0
        assert len(tr_s) + len(ev_s) == 15
        assert_disjoint_sources(tr_s, ev_s)

        # Check inner disjointness
        for inner in outer["inner_folds"]:
            in_tr = set(inner["train_sources"])
            in_ev = set(inner["eval_sources"])
            assert len(in_tr.intersection(in_ev)) == 0
            assert_disjoint_sources(in_tr, in_ev)

    # Identical fold assignment when applied twice or across direct/retrieval records
    plan2 = assign_consistent_source_folds(corpus, n_outer_folds=5, n_inner_folds=4, seed=42)
    assert plan["source_to_outer_fold"] == plan2["source_to_outer_fold"]


def test_validate_fold_class_support_guards():
    corpus = _make_synthetic_corpus(n_sources=15)
    # Adequate support succeeds
    support = validate_fold_class_support(
        corpus, fold_name="test_adequate", min_examples_per_class=5, min_sources_per_class=2
    )
    assert support["n_3class_examples"] == len(corpus)

    # Inadequate support (e.g. absent CONTRADICTED) raises ValueError
    no_contra = [r for r in corpus if r["primary_label"] != "CONTRADICTED"]
    with pytest.raises(ValueError, match="CONTRADICTED has 0 examples"):
        validate_fold_class_support(
            no_contra, fold_name="test_no_contra", min_examples_per_class=5, min_sources_per_class=2
        )

    # summarize_source_class_support does NOT raise on absent classes
    summary = summarize_source_class_support(no_contra)
    assert "CONTRADICTED" in summary["absent_classes"]
    assert summary["counts_by_class"]["CONTRADICTED"] == 0


def test_logit_reconstruction_and_multiclass_calibrators():
    probs = np.array(
        [
            [0.1, 0.7, 0.2],
            [0.6, 0.2, 0.2],
            [0.05, 0.05, 0.9],
        ],
        dtype=np.float64,
    )
    recon_logits = reconstruct_logits_from_probs(probs)
    assert recon_logits.shape == (3, 3)
    np.testing.assert_allclose(np.mean(recon_logits, axis=-1), np.zeros(3), atol=1e-10)

    # C0 preserves probabilities exactly
    c0 = MulticlassCalibrator("C0_identity", [1.0], [0.0, 0.0, 0.0])
    c0_p = c0.predict_probs(probs)
    np.testing.assert_allclose(c0_p, probs, atol=1e-6)

    # C1 preserves argmax
    c1 = MulticlassCalibrator("C1_temp_scaling", [1.5], [0.0, 0.0, 0.0])
    c1_p = c1.predict_probs(probs)
    assert np.all(np.argmax(c1_p, axis=1) == np.argmax(probs, axis=1))

    # C2 prior bias + temp fitting and serialization roundtrip
    rng = np.random.default_rng(42)
    synth_z = rng.normal(size=(60, 3))
    synth_y = rng.choice([0, 1, 2], size=60)
    # Ensure all 3 classes present
    synth_y[:3] = [0, 1, 2]

    c2 = fit_multiclass_calibrator(synth_z, synth_y, method="C2_prior_bias_temp", l2_reg=0.01)
    assert c2.method == "C2_prior_bias_temp"
    assert len(c2.temperatures) == 1
    assert len(c2.bias) == 3

    d = c2.to_dict()
    c2_loaded = MulticlassCalibrator.from_dict(d)
    np.testing.assert_allclose(
        c2.transform_logits(synth_z), c2_loaded.transform_logits(synth_z), atol=1e-12
    )

    # Multiclass calibration metrics
    m = compute_multiclass_calibration_metrics(c2.transform_logits(synth_z), synth_y)
    assert "nll_3class" in m
    assert "cw_ece" in m
    assert "top1_ece" in m
    assert "binary_hallucination" in m


def test_binary_platt_calibrator():
    p_raw = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9], dtype=np.float64)
    y_bin = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)

    cal = fit_binary_platt_calibrator(p_raw, y_bin, l2_reg=1e-4)
    assert cal.method == "platt_logit"
    p_cal = cal.predict_probs(p_raw)
    assert len(p_cal) == 6
    assert np.all(p_cal >= 0.0) and np.all(p_cal <= 1.0)

    # Serialization roundtrip
    d = cal.to_dict()
    cal2 = BinaryPlattCalibrator.from_dict(d)
    np.testing.assert_allclose(p_cal, cal2.predict_probs(p_raw), atol=1e-12)

    metrics = compute_binary_calibration_metrics(p_cal, y_bin)
    assert "nll" in metrics
    assert "brier" in metrics
    assert "ece_uniform" in metrics


def test_fast_eval_parity_with_arm_and_response_evaluation():
    corpus = _make_synthetic_corpus(n_sources=9)
    arrays = extract_feature_arrays(corpus, "direct")

    # Generate random prediction codes
    rng = np.random.default_rng(123)
    codes = rng.choice([0, 1, 2, 3], size=len(corpus))
    pred_labels = [INT_TO_PRED_LABEL[c] for c in codes]

    fast_m = fast_eval_fold_predictions(arrays, codes)
    arm_m = compute_arm_metrics(corpus, pred_labels)
    resp_m = compute_response_level_evaluation(corpus, pred_labels)

    # Parity check
    pen3 = arm_m["penalized_3class"]
    assert pytest.approx(fast_m["macro_f1"], abs=1e-5) == pen3["macro_f1"]
    assert pytest.approx(fast_m["support_weighted_f1"], abs=1e-5) == pen3["support_weighted_f1"]
    assert pytest.approx(fast_m["accuracy"], abs=1e-5) == pen3["accuracy"]
    assert pytest.approx(fast_m["coverage"], abs=1e-5) == arm_m["coverage"]
    assert pytest.approx(fast_m["mae_conf_pp"], abs=1e-5) == resp_m["confirmed_hallucination_rate_pct"]["mae"]
    for cls in CALIBRATION_CLASSES:
        assert pytest.approx(fast_m["recalls"][cls], abs=1e-5) == pen3["per_class"][cls]["recall"]


def test_pre_registered_objective_and_tie_breaking():
    # OBJ_CONSTRAINED: alpha = 0.005, MAE_max direct = 10.0 pp
    metrics_good = {
        "macro_f1": 0.80,
        "support_weighted_f1": 0.82,
        "accuracy": 0.85,
        "coverage": 0.98,
        "mae_conf_pp": 8.0,  # <= 10.0 -> no penalty
        "recalls": {"CONTRADICTED": 0.35, "SUPPORTED_PROXY": 0.85, "UNSUPPORTED": 0.45},
    }
    p_cov, p_rec, key_good = score_objective(metrics_good, "direct", "OBJ_CONSTRAINED")
    assert p_cov is True
    assert p_rec is True
    assert pytest.approx(key_good[0], abs=1e-6) == 0.80

    metrics_penalized = {
        "macro_f1": 0.80,
        "support_weighted_f1": 0.82,
        "accuracy": 0.85,
        "coverage": 0.98,
        "mae_conf_pp": 14.0,  # 4 pp above ceiling -> penalty = 0.005 * 4 = 0.02
        "recalls": {"CONTRADICTED": 0.35, "SUPPORTED_PROXY": 0.85, "UNSUPPORTED": 0.45},
    }
    _, _, key_pen = score_objective(metrics_penalized, "direct", "OBJ_CONSTRAINED")
    assert pytest.approx(key_pen[0], abs=1e-6) == 0.80 - 0.02  # 0.78

    # OBJ_MACRO_F1_ONLY ignores MAE penalty
    _, _, key_unconstrained = score_objective(metrics_penalized, "direct", "OBJ_MACRO_F1_ONLY")
    assert pytest.approx(key_unconstrained[0], abs=1e-6) == 0.80

    # Tie-breaking: identical primary utility breaks on higher support_weighted_f1 then accuracy
    cand1 = {
        "params": {"p": 1},
        "metrics": dict(metrics_good, support_weighted_f1=0.81, accuracy=0.80),
        "complexity_penalty": 0.0,
    }
    cand2 = {
        "params": {"p": 2},
        "metrics": dict(metrics_good, support_weighted_f1=0.84, accuracy=0.80),
        "complexity_penalty": 0.0,
    }
    best = select_best_candidate([cand1, cand2], "direct", "OBJ_CONSTRAINED")
    assert best["params"]["p"] == 2


def test_semantic_guards_and_h3_exclusion():
    # Non-factual short-circuits to NON_FACTUAL
    rec_nf = _make_synthetic_record(
        "ex_nf", "r_nf", "s_nf", "QA", "NON_FACTUAL", is_factual=False
    )
    arr_nf = extract_feature_arrays([rec_nf], "direct")
    codes_nf = fast_eval_fold_predictions(arr_nf, np.array([4]))
    assert arr_nf["non_factual"][0] == 1.0

    # Retrieval empty evidence produces UNVERIFIABLE
    rec_empty = _make_synthetic_record(
        "ex_em", "r_em", "s_em", "Summary", "SUPPORTED_PROXY", empty_ev=True
    )
    arr_empty = extract_feature_arrays([rec_empty], "retrieval")
    assert arr_empty["empty_ev"][0] == 1.0

    # H3 exclusion metadata
    h3_meta = STRATEGY_REGISTRY["H3_sec_deb"]
    assert h3_meta["stage1_cache_available"] is False
    assert "requires_stage2_secondary_chunk_inference" in h3_meta["exclusion_reason"]


def test_end_to_end_synthetic_stage1_nested_cv_and_freezing(tmp_path: Path):
    corpus = _make_synthetic_corpus(n_sources=15)
    # Stage 1 nested CV
    results = run_nested_source_cv_stage1(
        corpus,
        n_outer_folds=3,
        n_inner_folds=2,
        seed=42,
        n_boot=20,
    )
    assert results["schema_version"] == "phase8_step7_stage1_v1"
    for reg in ("direct", "retrieval"):
        r_res = results["regimes"][reg]
        assert "pillar1_calibration" in r_res
        assert "OBJ_CONSTRAINED" in r_res["objectives"]
        assert "OBJ_MACRO_F1_ONLY" in r_res["objectives"]
        for strat in STAGE1_STRATEGY_ORDER:
            assert strat in r_res["objectives"]["OBJ_CONSTRAINED"]["strategies"]

    # Stage 2 bundle freezing strictly on Val-Dev
    frozen_bundle = freeze_stage2_bundle(corpus, n_outer_folds=3, seed=42)
    assert frozen_bundle["schema_version"] == "phase8_step7_frozen_bundle_v1"
    assert len(frozen_bundle["val_dev_source_ids"]) == 15

    # Evaluate frozen bundle on held-out synthetic Val-Lock corpus
    val_lock_corpus = _make_synthetic_corpus(n_sources=6)
    for r in val_lock_corpus:
        r["source_id"] = f"lock_{r['source_id']}"
        r["example_id"] = f"lock_{r['example_id']}"
        r["response_id"] = f"lock_{r['response_id']}"

    stage2_res = apply_frozen_bundle_to_records(
        val_lock_corpus, frozen_bundle, n_boot=20, seed=42
    )
    assert stage2_res["schema_version"] == "phase8_step7_stage2_vallock_v1"
    assert stage2_res["val_lock_sources_count"] == 6

    # Source overlap between Val-Dev and Val-Lock must raise AssertionError
    with pytest.raises(AssertionError, match="Source leakage detected"):
        apply_frozen_bundle_to_records(corpus, frozen_bundle, n_boot=20, seed=42)


def test_safety_guards_and_split_protection(monkeypatch: pytest.MonkeyPatch):
    forbidden_test_path = Path("backend/data/processed/ragtruth_test.jsonl")
    with pytest.raises(ValueError, match="ACCESS VIOLATION: ragtruth_test.jsonl is strictly forbidden"):
        study_module.assert_safe_non_test_path(forbidden_test_path)

    # Unapproved --stage2-run must fail fast via sys.exit(1)
    monkeypatch.setattr(sys, "argv", ["10_calibration_and_hybrid_study.py", "--stage2-run"])
    with pytest.raises(SystemExit) as exc_info:
        study_module.main()
    assert exc_info.value.code == 1


def test_predict_h3_sec_deb_and_vallock_disjointness():
    rec = _make_synthetic_record("ex1", "r1", "s1", "QA", "CONTRADICTED", sim_score=0.7)
    # Primary chunk thinks supported, secondary chunk sees CONTRADICTED >= 0.30
    pred = predict_h3_sec_deb(
        rec,
        sec_deberta_cal_probs=[[0.45, 0.40, 0.15]],
        primary_deberta_cal_probs=[0.05, 0.85, 0.10],
        tau_contra=0.30,
        tau_sup=0.50,
        tau_floor=0.25,
    )
    assert pred == "CONTRADICTED"


def test_stage2_vallock_selection_content_leakage_and_adequacy():
    import hashlib
    full_pool: List[Dict[str, Any]] = []
    tasks = ["Summary", "QA", "Data2txt"]
    for i in range(48):
        sid = str(1000 + i)
        task = tasks[i % 3]
        for j, (lbl, dp, mc) in enumerate([
            ("SUPPORTED_PROXY", (0.05, 0.90, 0.05), 0.9),
            ("CONTRADICTED", (0.75, 0.10, 0.15), 0.2),
            ("UNSUPPORTED", (0.10, 0.20, 0.70), 0.2),
            ("UNSUPPORTED", (0.10, 0.20, 0.70), 0.2),
        ]):
            r = _make_synthetic_record(
                f"ex_{sid}_{j}",
                f"resp_{sid}_{j // 2}",
                sid,
                task,
                lbl,
                mc_prob=mc,
                deb_probs=dp,
            )
            # Give specific context text to source 1020 that will match train
            if sid == "1020":
                r["context_text"] = "EXACT_LEAKED_TRAIN_CONTEXT_CONTENT"
            else:
                r["context_text"] = f"Clean context for {sid}"
            full_pool.append(r)

    val_dev_sids = [str(1000 + i) for i in range(15)]
    leaked_ctx_hash = hashlib.sha256("EXACT_LEAKED_TRAIN_CONTEXT_CONTENT".encode("utf-8")).hexdigest()
    mock_train_index = {
        "context_hashes": {leaked_ctx_hash},
        "exact_pair_keys": set(),
        "sentence_strings": set(),
        "source_ids": {"train_src_999"},
    }

    manifest = study_module.select_stage2_vallock_sources(
        val_records=full_pool,
        val_dev_source_ids=val_dev_sids,
        train_index=mock_train_index,
        sources_per_task=10,
        seed=43,
    )

    # 1. Exactly 30 sources selected
    assert len(manifest["selected_source_ids"]) == 30
    assert len(set(manifest["selected_source_ids"])) == 30

    # 2. Val-Dev exclusion verified
    assert set(manifest["selected_source_ids"]).isdisjoint(set(val_dev_sids))
    assert manifest["val_dev_exclusion_verified"] is True
    assert set(manifest["val_dev_excluded_source_ids"]) == set(val_dev_sids)

    # 3. Exact context overlap exclusion verified
    assert "1020" not in manifest["selected_source_ids"]
    assert "1020" in manifest["train_content_leakage_excluded_sources"]

    # 4. Adequacy preserved
    assert manifest["final_adequacy"]["is_adequate"] is True
    assert len(manifest["final_adequacy"]["failures"]) == 0

    # 5. Deterministic selection check (identical result across runs)
    manifest2 = study_module.select_stage2_vallock_sources(
        val_records=full_pool,
        val_dev_source_ids=val_dev_sids,
        train_index=mock_train_index,
        sources_per_task=10,
        seed=43,
    )
    assert manifest["selected_source_ids"] == manifest2["selected_source_ids"]


def test_stage2_vallock_selection_fails_when_inadequate():
    # Synthetic pool where Summary sources have zero minority labels
    full_pool: List[Dict[str, Any]] = []
    tasks = ["Summary", "QA", "Data2txt"]
    for i in range(45):
        sid = str(2000 + i)
        task = tasks[i % 3]
        for j in range(4):
            # Summary sources have zero contradicted/unsupported
            lbl = "SUPPORTED_PROXY" if task == "Summary" else ("CONTRADICTED" if j == 0 else "SUPPORTED_PROXY")
            r = _make_synthetic_record(
                f"ex_{sid}_{j}",
                f"resp_{sid}_{j // 2}",
                sid,
                task,
                lbl,
            )
            r["context_text"] = f"Context for {sid}"
            full_pool.append(r)

    val_dev_sids = [str(2000 + i) for i in range(15)]
    with pytest.raises(ValueError, match="Stage 2 Val-Lock selection failed adequacy constraints"):
        study_module.select_stage2_vallock_sources(
            val_records=full_pool,
            val_dev_source_ids=val_dev_sids,
            train_index=None,
            sources_per_task=10,
            seed=43,
        )


def test_stage2_full_unapproved_inference_guard_and_safety(tmp_path: Path):
    manifest_path = tmp_path / "stage2_manifest.json"
    val_data_path = tmp_path / "ragtruth_val.jsonl"
    val_dev_manifest_path = tmp_path / "val_dev_manifest.json"
    cache_path = tmp_path / "stage2_cache.jsonl"
    frozen_bundle_path = tmp_path / "frozen_bundle.json"
    results_path = tmp_path / "results.json"
    report_path = tmp_path / "report.md"
    profile_path = tmp_path / "profile.json"

    manifest_data = {
        "selected_source_ids": ["src_lock_1", "src_lock_2"],
        "total_sources": 2,
    }
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")
    val_dev_manifest_path.write_text(json.dumps({"selected_source_ids": ["src_dev_1", "src_dev_2"]}), encoding="utf-8")

    val_records = [
        {"example_id": "ex1", "response_id": "r1", "source_id": "src_lock_1", "task_type": "QA", "model": "gpt-4", "split": "val", "primary_label": "SUPPORTED_PROXY", "sentence_text": "text1", "context_text": "ctx1"},
        {"example_id": "ex2", "response_id": "r2", "source_id": "src_lock_2", "task_type": "QA", "model": "gpt-4", "split": "val", "primary_label": "CONTRADICTED", "sentence_text": "text2", "context_text": "ctx2"},
    ]
    with val_data_path.open("w", encoding="utf-8") as h:
        for r in val_records:
            h.write(json.dumps(r) + "\n")

    # Access violation when ragtruth_test is passed
    with pytest.raises(ValueError, match="ACCESS VIOLATION: ragtruth_test.jsonl is strictly forbidden"):
        study_module.run_stage2_full(val_data_path=Path("data/ragtruth_test.jsonl"))

    # Unapproved inference must fail fast
    with pytest.raises(RuntimeError, match="SAFETY GUARD: Inference is required"):
        study_module.run_stage2_full(
            manifest_path=manifest_path,
            val_data_path=val_data_path,
            val_dev_manifest_path=val_dev_manifest_path,
            cache_path=cache_path,
            frozen_bundle_path=frozen_bundle_path,
            results_path=results_path,
            report_path=report_path,
            profile_path=profile_path,
            approve_inference=False,
            require_30_sources=False,
        )


def test_stage2_full_manifest_and_disjointness_validation(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    val_dev_path = tmp_path / "val_dev.json"
    val_data_path = tmp_path / "val.jsonl"

    # 1. Manifest with duplicate sources
    manifest_path.write_text(json.dumps({"selected_source_ids": ["s1", "s1", "s2"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate source IDs in manifest"):
        study_module.run_stage2_full(
            manifest_path=manifest_path,
            val_data_path=val_data_path,
            require_30_sources=False,
        )

    # 2. Manifest not 30 sources when require_30_sources=True
    manifest_path.write_text(json.dumps({"selected_source_ids": ["s1", "s2"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="Expected exactly 30 Val-Lock sources"):
        study_module.run_stage2_full(
            manifest_path=manifest_path,
            val_data_path=val_data_path,
            require_30_sources=True,
        )

    # 3. Source overlap with Val-Dev
    val_dev_path.write_text(json.dumps({"selected_source_ids": [f"dev_{i}" for i in range(29)] + ["shared_src"]}), encoding="utf-8")
    manifest_path.write_text(json.dumps({"selected_source_ids": ["shared_src", "lock_2"]}), encoding="utf-8")
    with pytest.raises(AssertionError, match="Source leakage detected"):
        study_module.run_stage2_full(
            manifest_path=manifest_path,
            val_dev_manifest_path=val_dev_path,
            val_data_path=val_data_path,
            require_30_sources=False,
        )


def test_stage2_full_mocked_execution_resumption_and_reports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Setup synthetic Val-Dev (15 sources) and Val-Lock (6 sources)
    val_dev_corpus = _make_synthetic_corpus(n_sources=15)
    for r in val_dev_corpus:
        r["source_id"] = f"dev_{r['source_id']}"
        r["example_id"] = f"dev_{r['example_id']}"
        r["response_id"] = f"dev_{r['response_id']}"

    val_lock_corpus = _make_synthetic_corpus(n_sources=6)
    for r in val_lock_corpus:
        r["source_id"] = f"lock_{r['source_id']}"
        r["example_id"] = f"lock_{r['example_id']}"
        r["response_id"] = f"lock_{r['response_id']}"

    # Generate frozen bundle from Val-Dev strictly
    frozen_bundle = freeze_stage2_bundle(val_dev_corpus, n_outer_folds=3, seed=42)
    bundle_path = tmp_path / "10_stage1_frozen_bundle.json"
    bundle_path.write_text(json.dumps(frozen_bundle, indent=2), encoding="utf-8")

    # Manifests
    val_dev_manifest_path = tmp_path / "val_dev_manifest.json"
    val_dev_manifest_path.write_text(
        json.dumps({"selected_source_ids": sorted(list({r["source_id"] for r in val_dev_corpus}))}),
        encoding="utf-8",
    )
    val_lock_sids = sorted(list({r["source_id"] for r in val_lock_corpus}))
    stage2_manifest_path = tmp_path / "10_stage2_vallock_manifest.json"
    stage2_manifest_path.write_text(
        json.dumps({
            "selected_source_ids": val_lock_sids,
            "total_sources": len(val_lock_sids),
            "total_sentences": len(val_lock_corpus),
        }),
        encoding="utf-8",
    )

    # Combined validation dataset file
    val_data_path = tmp_path / "ragtruth_val.jsonl"
    with val_data_path.open("w", encoding="utf-8") as h:
        for r in val_dev_corpus + val_lock_corpus:
            h.write(json.dumps(r) + "\n")

    cache_path = tmp_path / "10_stage2_vallock_raw_cache.jsonl"
    results_path = tmp_path / "10_stage2_confirmation_results.json"
    report_path = tmp_path / "10_stage2_confirmation_report.md"
    profile_path = tmp_path / "10_stage2_runtime_profile.json"

    # Mock direct and retrieval inference
    inference_call_count = {"direct": 0, "retrieval": 0}

    def _mock_direct(records, **kwargs):
        inference_call_count["direct"] += 1
        out = {}
        for rec in records:
            out[str(rec["example_id"])] = {
                "mc_label": 1,
                "mc_prob": 0.85,
                "nli_entail": 0.1,
                "nli_contra": 0.05,
                "nli_neutral": 0.85,
                "deberta_supported_prob": 0.85,
                "deberta_contradiction_prob": 0.10,
                "deberta_unsupported_prob": 0.05,
                "deberta_predicted_label": "SUPPORTED_PROXY",
                "pair_token_count": 120,
                "exceeds_512_tokens": False,
            }
        return out

    def _mock_retrieval(records, **kwargs):
        inference_call_count["retrieval"] += 1
        out = {}
        for rec in records:
            out[str(rec["example_id"])] = {
                "num_chunks": 2,
                "best_similarity_score": 0.80,
                "retrieved_chunk_text_word_count": 50,
                "retrieved_chunk_preview": "preview",
                "relevant_passages_count": 1,
                "baseline_classification": "SUPPORTED_PROXY",
                "baseline_decision_rule": "EVIDENCE_ENTAILMENT",
                "baseline_reasoning_trace": "trace",
                "mc_label": 1,
                "mc_prob": 0.82,
                "nli_entail": 0.15,
                "nli_contra": 0.05,
                "nli_neutral": 0.80,
                "secondary_nli_scores": [],
                "sec_nli_max_entail": 0.15,
                "sec_nli_max_contra": 0.05,
                "conflict_triggered": False,
                "deberta_supported_prob": 0.82,
                "deberta_contradiction_prob": 0.10,
                "deberta_unsupported_prob": 0.08,
                "deberta_predicted_label": "SUPPORTED_PROXY",
                "pair_token_count": 80,
                "exceeds_512_tokens": False,
            }
        return out

    monkeypatch.setattr(study_module, "evaluate_direct_context_records", _mock_direct)
    monkeypatch.setattr(study_module, "evaluate_retrieval_pipeline_for_source", _mock_retrieval)

    # 1. First execution: runs inference and writes all artifacts
    res1 = study_module.run_stage2_full(
        manifest_path=stage2_manifest_path,
        val_data_path=val_data_path,
        cache_path=cache_path,
        frozen_bundle_path=bundle_path,
        val_dev_manifest_path=val_dev_manifest_path,
        results_path=results_path,
        report_path=report_path,
        profile_path=profile_path,
        approve_inference=True,
        n_boot=10,
        seed=42,
        claim_verifier=object(),
        require_30_sources=False,
    )

    assert res1["schema_version"] == "phase8_step7_stage2_vallock_v1"
    assert res1["val_lock_sources_count"] == 6
    assert res1["runtime_profile"]["missing_records_inferred"] == len(val_lock_corpus)
    assert inference_call_count["direct"] == 6
    assert inference_call_count["retrieval"] == 6
    assert cache_path.exists()
    assert results_path.exists()
    assert profile_path.exists()
    assert report_path.exists()

    report_text = report_path.read_text(encoding="utf-8")
    assert "# VeriLLM Phase 8 Step 7: Stage 2 Val-Lock Confirmation Study Report" in report_text
    assert "Pillar 1 — Probability Calibration Confirmation on Held-Out Val-Lock" in report_text
    assert "Pillar 2 & 3 — Frozen Strategy Confirmation on Val-Lock" in report_text
    assert "Pillar 4 — Runtime Profile & Resource Consumption" in report_text

    # 2. Second execution (Cache Resumption): zero inference calls executed
    inference_call_count["direct"] = 0
    inference_call_count["retrieval"] = 0

    res2 = study_module.run_stage2_full(
        manifest_path=stage2_manifest_path,
        val_data_path=val_data_path,
        cache_path=cache_path,
        frozen_bundle_path=bundle_path,
        val_dev_manifest_path=val_dev_manifest_path,
        results_path=results_path,
        report_path=report_path,
        profile_path=profile_path,
        approve_inference=True,
        n_boot=10,
        seed=42,
        claim_verifier=object(),
        require_30_sources=False,
    )

    assert inference_call_count["direct"] == 0
    assert inference_call_count["retrieval"] == 0
    assert res2["runtime_profile"]["missing_records_inferred"] == 0
    assert res2["runtime_profile"]["inference_elapsed_seconds"] == 0.0

    # 3. Analyze-only execution on the generated cache
    res_ana = study_module.run_stage2_analyze_only(
        cache_path=cache_path,
        frozen_bundle_path=bundle_path,
        manifest_path=stage2_manifest_path,
        val_data_path=val_data_path,
        val_dev_manifest_path=val_dev_manifest_path,
        results_path=results_path,
        report_path=report_path,
        profile_path=profile_path,
        n_boot=10,
        seed=42,
        require_30_sources=False,
    )
    assert res_ana["val_lock_sources_count"] == 6
    assert res_ana["runtime_profile"]["missing_records_inferred"] == 0


def test_stage2_cli_dispatch(monkeypatch: pytest.MonkeyPatch):
    dispatched = {"called": False}

    def _mock_run_stage2_full(**kwargs):
        dispatched["called"] = True
        assert kwargs["approve_inference"] is True
        return {}

    monkeypatch.setattr(study_module, "run_stage2_full", _mock_run_stage2_full)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "10_calibration_and_hybrid_study.py",
            "--stage2-run",
            "--approve-stage2-inference",
        ],
    )
    study_module.main()
    assert dispatched["called"] is True


def test_stage2_analyze_only_validation_rules(tmp_path: Path):
    val_dev_path = tmp_path / "val_dev_manifest.json"
    val_dev_path.write_text(json.dumps({"selected_source_ids": ["dev_1", "dev_2"]}), encoding="utf-8")

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"selected_source_ids": ["lock_1", "lock_2"]}), encoding="utf-8")

    bundle_data = {
        "val_dev_source_ids": ["dev_1", "dev_2"],
        "regimes": {
            "direct": {"calibrators": {}, "objectives": {}},
            "retrieval": {"calibrators": {}, "objectives": {}},
        },
    }
    bundle_path = tmp_path / "frozen_bundle.json"
    bundle_path.write_text(json.dumps(bundle_data), encoding="utf-8")

    val_records = [
        _make_synthetic_record("l_1", "r_1", "lock_1", "QA", "SUPPORTED_PROXY"),
        _make_synthetic_record("l_2", "r_2", "lock_1", "QA", "CONTRADICTED"),
        _make_synthetic_record("l_3", "r_3", "lock_2", "QA", "UNSUPPORTED"),
    ]
    val_data_path = tmp_path / "val.jsonl"
    with val_data_path.open("w", encoding="utf-8") as h:
        for r in val_records:
            h.write(json.dumps(r) + "\n")

    def _make_cache_entry(r: Dict[str, Any], direct_valid: bool = True, ret_valid: bool = True, src_override: Optional[str] = None) -> Dict[str, Any]:
        return {
            "example_id": str(r["example_id"]),
            "source_id": src_override or str(r["source_id"]),
            "response_id": str(r["response_id"]),
            "task_type": r["task_type"],
            "model": r["model"],
            "split": r["split"],
            "primary_label": r["primary_label"],
            "sentence_text": r["sentence_text"],
            "direct": {"mc_prob": 0.8, "mc_label": 1, "deberta_supported_prob": 0.8, "deberta_contradiction_prob": 0.1, "deberta_unsupported_prob": 0.1, "deberta_predicted_label": "SUPPORTED_PROXY"} if direct_valid else {},
            "retrieval": {"mc_prob": 0.75, "mc_label": 1, "deberta_supported_prob": 0.75, "deberta_contradiction_prob": 0.15, "deberta_unsupported_prob": 0.1, "deberta_predicted_label": "SUPPORTED_PROXY"} if ret_valid else {},
        }

    # 1. Missing records in cache
    cache_incomplete = tmp_path / "cache_incomplete.jsonl"
    with cache_incomplete.open("w", encoding="utf-8") as h:
        h.write(json.dumps(_make_cache_entry(val_records[0])) + "\n")
        h.write(json.dumps(_make_cache_entry(val_records[1])) + "\n")
    with pytest.raises(ValueError, match="Stage 2 Val-Lock cache is incomplete: 1 expected records are missing"):
        study_module.run_stage2_analyze_only(
            cache_path=cache_incomplete,
            frozen_bundle_path=bundle_path,
            manifest_path=manifest_path,
            val_data_path=val_data_path,
            val_dev_manifest_path=val_dev_path,
            require_30_sources=False,
        )

    # 2. Unexpected IDs in cache
    cache_unexpected = tmp_path / "cache_unexpected.jsonl"
    with cache_unexpected.open("w", encoding="utf-8") as h:
        for r in val_records:
            h.write(json.dumps(_make_cache_entry(r)) + "\n")
        unexp_rec = _make_synthetic_record("unexp_999", "r_x", "lock_1", "QA", "SUPPORTED_PROXY")
        h.write(json.dumps(_make_cache_entry(unexp_rec)) + "\n")
    with pytest.raises(ValueError, match="Stage 2 Val-Lock cache contains 1 unexpected records not in manifest"):
        study_module.run_stage2_analyze_only(
            cache_path=cache_unexpected,
            frozen_bundle_path=bundle_path,
            manifest_path=manifest_path,
            val_data_path=val_data_path,
            val_dev_manifest_path=val_dev_path,
            require_30_sources=False,
        )

    # 3. Duplicate example_id in cache
    cache_duplicate = tmp_path / "cache_duplicate.jsonl"
    with cache_duplicate.open("w", encoding="utf-8") as h:
        h.write(json.dumps(_make_cache_entry(val_records[0])) + "\n")
        h.write(json.dumps(_make_cache_entry(val_records[0])) + "\n")
        h.write(json.dumps(_make_cache_entry(val_records[1])) + "\n")
        h.write(json.dumps(_make_cache_entry(val_records[2])) + "\n")
    with pytest.raises(ValueError, match="Duplicate example_id 'l_1' detected"):
        study_module.run_stage2_analyze_only(
            cache_path=cache_duplicate,
            frozen_bundle_path=bundle_path,
            manifest_path=manifest_path,
            val_data_path=val_data_path,
            val_dev_manifest_path=val_dev_path,
            require_30_sources=False,
        )

    # 4. Missing direct regime
    cache_no_direct = tmp_path / "cache_no_direct.jsonl"
    with cache_no_direct.open("w", encoding="utf-8") as h:
        h.write(json.dumps(_make_cache_entry(val_records[0], direct_valid=False)) + "\n")
        h.write(json.dumps(_make_cache_entry(val_records[1])) + "\n")
        h.write(json.dumps(_make_cache_entry(val_records[2])) + "\n")
    with pytest.raises(ValueError, match="missing valid 'direct' regime"):
        study_module.run_stage2_analyze_only(
            cache_path=cache_no_direct,
            frozen_bundle_path=bundle_path,
            manifest_path=manifest_path,
            val_data_path=val_data_path,
            val_dev_manifest_path=val_dev_path,
            require_30_sources=False,
        )

    # 5. Missing retrieval regime
    cache_no_ret = tmp_path / "cache_no_ret.jsonl"
    with cache_no_ret.open("w", encoding="utf-8") as h:
        h.write(json.dumps(_make_cache_entry(val_records[0], ret_valid=False)) + "\n")
        h.write(json.dumps(_make_cache_entry(val_records[1])) + "\n")
        h.write(json.dumps(_make_cache_entry(val_records[2])) + "\n")
    with pytest.raises(ValueError, match="missing valid 'retrieval' regime"):
        study_module.run_stage2_analyze_only(
            cache_path=cache_no_ret,
            frozen_bundle_path=bundle_path,
            manifest_path=manifest_path,
            val_data_path=val_data_path,
            val_dev_manifest_path=val_dev_path,
            require_30_sources=False,
        )

    # 6. Source mismatch in cache entry
    cache_mismatch = tmp_path / "cache_mismatch.jsonl"
    with cache_mismatch.open("w", encoding="utf-8") as h:
        h.write(json.dumps(_make_cache_entry(val_records[0], src_override="lock_2")) + "\n")
        h.write(json.dumps(_make_cache_entry(val_records[1])) + "\n")
        h.write(json.dumps(_make_cache_entry(val_records[2])) + "\n")
    with pytest.raises(ValueError, match="Source ID mismatch for 'l_1'"):
        study_module.run_stage2_analyze_only(
            cache_path=cache_mismatch,
            frozen_bundle_path=bundle_path,
            manifest_path=manifest_path,
            val_data_path=val_data_path,
            val_dev_manifest_path=val_dev_path,
            require_30_sources=False,
        )

    # 7. Disjointness violation between Val-Dev and Val-Lock manifests in analyze-only
    leaked_manifest_path = tmp_path / "leaked_manifest.json"
    leaked_manifest_path.write_text(json.dumps({"selected_source_ids": ["dev_1", "lock_2"]}), encoding="utf-8")
    with pytest.raises(AssertionError, match="Source leakage detected"):
        study_module.run_stage2_analyze_only(
            cache_path=cache_incomplete,
            frozen_bundle_path=bundle_path,
            manifest_path=leaked_manifest_path,
            val_data_path=val_data_path,
            val_dev_manifest_path=val_dev_path,
            require_30_sources=False,
        )


def test_cache_integrity_corrupt_middle_and_interrupted_trailing_line(tmp_path: Path):
    rec1 = {"example_id": "ex1", "source_id": "s1", "direct": {"m": 1}, "retrieval": {"m": 2}}
    rec2 = {"example_id": "ex2", "source_id": "s1", "direct": {"m": 1}, "retrieval": {"m": 2}}
    rec3 = {"example_id": "ex3", "source_id": "s1", "direct": {"m": 1}, "retrieval": {"m": 2}}

    # Case 1: Corrupt line in the middle must fail fast and NOT discard or skip
    mid_corrupt_path = tmp_path / "mid_corrupt.jsonl"
    with mid_corrupt_path.open("wb") as h:
        h.write((json.dumps(rec1) + "\n").encode("utf-8"))
        h.write(b'{"example_id": "ex2", "source_id": "BROKEN\n')
        h.write((json.dumps(rec3) + "\n").encode("utf-8"))

    with pytest.raises(ValueError, match="Malformed cache record at line 2 .* Cache corruption detected in non-trailing line"):
        study_module.load_raw_cache_by_id(mid_corrupt_path, allow_repair_trailing_partial=True)

    with pytest.raises(ValueError, match="Malformed cache record at line 2"):
        study_module.load_raw_cache_by_id(mid_corrupt_path, allow_repair_trailing_partial=False)

    # Case 2: Interrupted trailing line without allow_repair must fail fast
    interrupted_path = tmp_path / "interrupted.jsonl"
    valid_bytes = (json.dumps(rec1) + "\n" + json.dumps(rec2) + "\n").encode("utf-8")
    partial_bytes = b'{"example_id": "ex3", "source_id": "s1", "direct": {"m":'
    interrupted_path.write_bytes(valid_bytes + partial_bytes)

    with pytest.raises(ValueError, match="Malformed cache record at line 3"):
        study_module.load_raw_cache_by_id(interrupted_path, allow_repair_trailing_partial=False)

    # Case 3: Interrupted trailing line with allow_repair_trailing_partial=True safely recovers and truncates
    loaded = study_module.load_raw_cache_by_id(interrupted_path, allow_repair_trailing_partial=True)
    assert len(loaded) == 2
    assert "ex1" in loaded and "ex2" in loaded
    assert "ex3" not in loaded

    # File on disk is now safely truncated to valid bytes
    assert interrupted_path.stat().st_size == len(valid_bytes)

    # Re-reading without repair flag now succeeds cleanly
    re_loaded = study_module.load_raw_cache_by_id(interrupted_path, allow_repair_trailing_partial=False)
    assert len(re_loaded) == 2


def test_atomic_writes_and_cleanup(tmp_path: Path):
    target_json = tmp_path / "subdir" / "target.json"
    target_text = tmp_path / "subdir" / "target.txt"

    # 1. Successful atomic writes
    study_module.atomic_write_json(target_json, {"hello": "world", "status": 200})
    assert target_json.exists()
    assert json.loads(target_json.read_text(encoding="utf-8")) == {"hello": "world", "status": 200}

    study_module.atomic_write_text(target_text, "deterministic content")
    assert target_text.exists()
    assert target_text.read_text(encoding="utf-8") == "deterministic content"

    # Verify no temp files left behind
    assert len(list(target_json.parent.glob("*.tmp_*"))) == 0

    # 2. Failed atomic write cleans up temp file and leaves existing file intact
    with pytest.raises(TypeError):
        study_module.atomic_write_json(target_json, {"unserializable": object()})

    # Content of target_json was untouched
    assert json.loads(target_json.read_text(encoding="utf-8")) == {"hello": "world", "status": 200}
    # No dangling temp files
    assert len(list(target_json.parent.glob("*.tmp_*"))) == 0


def test_peak_memory_profiling(monkeypatch: pytest.MonkeyPatch):
    # 1. Real runtime environment profiling
    prof = study_module.get_peak_memory_profile()
    assert isinstance(prof, dict)
    assert "peak_memory_mb" in prof
    assert "peak_memory_metric" in prof
    assert "current_rss_mb" in prof
    assert prof["peak_memory_mb"] > 0.0
    assert prof["current_rss_mb"] > 0.0
    assert prof["peak_memory_metric"] in (
        "peak_working_set_mb",
        "peak_ru_maxrss_mb",
        "current_rss_fallback_mb",
    )

    # 2. Test fallback behavior when peak_wset is absent
    class MockMemInfo:
        rss = 104857600  # 100 MB
        # No peak_wset attribute

    class MockProcess:
        def memory_info(self):
            return MockMemInfo()

    monkeypatch.setattr(study_module.psutil, "Process", lambda: MockProcess())

    prof_fb = study_module.get_peak_memory_profile()
    assert prof_fb["current_rss_mb"] == 100.0
    assert prof_fb["peak_memory_mb"] == 100.0
    assert prof_fb["peak_memory_metric"] in ("peak_ru_maxrss_mb", "current_rss_fallback_mb")



