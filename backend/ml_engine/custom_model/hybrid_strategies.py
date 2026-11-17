"""
Phase 8 Step 7: Calibrated Hybrid Verification Strategies & Pre-Registered Nested CV Optimizer.

Implements regime-specific hybrid verification rules, fast vectorized objective evaluation,
5x4 source-grouped nested cross-validation (Stage 1 on Val-Dev cache), predetermined Stage 2
freezing without peeking at Val-Lock, and immutable Stage 2 bundle evaluation.

H3_sec_deb is explicitly marked excluded from Stage 1 cache-only evaluation because
09_val_pilot_raw_cache.jsonl does not store per-chunk DeBERTa scores for secondary passages.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ml_engine.custom_model.calibration import (
    CALIBRATION_CLASSES,
    LABEL_TO_INDEX,
    BinaryPlattCalibrator,
    MulticlassCalibrator,
    assert_disjoint_sources,
    assign_consistent_source_folds,
    compute_binary_calibration_metrics,
    compute_multiclass_calibration_metrics,
    extract_deberta_logits,
    extract_minicheck_prob,
    extract_nli_probs,
    fit_binary_platt_calibrator,
    fit_multiclass_calibrator,
    summarize_source_class_support,
    validate_fold_class_support,
)
from ml_engine.custom_model.validation_ablation import (
    CANONICAL_LABELS,
    MIXED_LABEL,
    apply_holm_bonferroni,
    clustered_paired_comparison,
    compute_arm_metrics,
    compute_response_level_evaluation,
    exact_mcnemar_test,
    predict_arm_for_record,
    response_level_bootstrap_diff,
    source_clustered_bootstrap_diff,
)

# ---------------------------------------------------------------------------
# Pre-Registered Optimization Constants & Candidate Grids
# ---------------------------------------------------------------------------

PRE_REGISTERED_OBJECTIVE_PARAMS: Dict[str, Any] = {
    "alpha": 0.005,
    "macro_f1_scale": "0_to_1",
    "mae_scale": "percentage_points_0_to_100",
    "utility_formula": "macro_f1 - alpha * max(0.0, mae_conf_pp - mae_max_pp)",
    "mae_max_pp": {
        "direct": 10.0,
        "retrieval": 18.0,
    },
    "coverage_floor": {
        "direct": 0.95,
        "retrieval": 0.90,
    },
    "recall_floors": {
        "CONTRADICTED": 0.20,
        "SUPPORTED_PROXY": 0.70,
        "UNSUPPORTED": 0.30,
    },
    "tie_breakers": [
        "1. Higher support_weighted_f1",
        "2. Higher 3-class accuracy",
        "3. Lower confirmed-hallucination MAE (pp)",
        "4. Higher coverage",
        "5. Stronger regularization / smaller threshold drift from 0.50",
    ],
}

PRE_REGISTERED_GRIDS: Dict[str, Any] = {
    "c2_lambda": [0.0, 0.001, 0.01, 0.05, 0.25, 1.0],
    "h1_mc": {
        "tau_mc": [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
        "tau_floor": [0.0],  # retrieval optionally checks [0.20, 0.25, 0.30]
        "retrieval_tau_floor": [0.20, 0.25, 0.30],
    },
    "h2_deberta": {
        "tau_contra": [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
        "tau_sup": [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
        "direct_tau_floor": [0.0],
        "retrieval_tau_floor": [0.20, 0.25, 0.30],
    },
    "h2b_conflict": {
        "rule4_conflict_enabled": [True, False],
        "sec_contra_thresh": 0.70,
        "sec_entail_thresh": 0.50,
    },
    "h4_cascade": {
        "margin_thresh": [0.15, 0.30, 0.50],
        "tau_mc_sup": [0.50, 0.60, 0.70],
        "tau_mc_hall": [0.65, 0.75, 0.85],
    },
    "h5_logreg": {
        "C": [0.05, 0.1, 0.5, 1.0, 5.0],
        "retrieval_tau_floor": [0.20, 0.25, 0.30],
    },
    "h6_abstain": {
        "tau_abstain": [0.40, 0.45, 0.50, 0.55],
        "retrieval_sim_gate": 0.35,
    },
}

STRATEGY_REGISTRY: Dict[str, Dict[str, Any]] = {
    "H0_baseline": {
        "label": "H0: Frozen Production Baseline (MiniCheck + NLI)",
        "stage1_cache_available": True,
        "trainable_params": False,
        "description": "Exact cached production decision strategy baseline (arm1_baseline).",
    },
    "H1_mc_cal": {
        "label": "H1: Platt-Calibrated MiniCheck Binary Support vs Unsupported",
        "stage1_cache_available": True,
        "trainable_params": True,
        "description": "1D Platt-calibrated MiniCheck support probability with tuned tau_mc and evidence floor.",
    },
    "H2_deberta_cal_c1": {
        "label": "H2 (C1): Temperature-Scaled DeBERTa 3-Way Thresholds",
        "stage1_cache_available": True,
        "trainable_params": True,
        "description": "C1 scalar temperature scaling of Custom DeBERTa with tuned (tau_contra, tau_sup, tau_floor).",
    },
    "H2_deberta_cal_c2": {
        "label": "H2 (C2): Regularized Prior-Bias + Temp Scaled DeBERTa",
        "stage1_cache_available": True,
        "trainable_params": True,
        "description": "C2 L2-regularized prior-bias + temperature calibration with inner-CV lambda and thresholds.",
    },
    "H2b_deberta_cal_plus_nli_rule4": {
        "label": "H2b: C2 Calibrated DeBERTa + Cached Secondary NLI Conflict Guard",
        "stage1_cache_available": True,
        "trainable_params": True,
        "description": "C2 calibrated DeBERTa augmented with cached Rule-4 secondary NLI conflict guard.",
    },
    "H3_sec_deb": {
        "label": "H3: Secondary-Passage DeBERTa Conflict & Support Recovery",
        "stage1_cache_available": False,
        "exclusion_reason": (
            "requires_stage2_secondary_chunk_inference: 09_val_pilot_raw_cache.jsonl stores "
            "DeBERTa scores only for the top-1 retrieved passage plus NLI scores for secondary "
            "passages; separate DeBERTa scores/logits for top-3 secondary chunks are absent."
        ),
        "trainable_params": True,
        "description": "Evaluates DeBERTa across top-k retrieved chunks; deferred to Stage 2 inference.",
    },
    "H4_mc_deberta_cascade": {
        "label": "H4: Calibrated MiniCheck -> C2 DeBERTa Uncertainty Cascade",
        "stage1_cache_available": True,
        "trainable_params": True,
        "description": "Confident Platt-MiniCheck fast-path when DeBERTa contra < tau_contra; uncertain claims route to C2 DeBERTa.",
    },
    "H5_meta_calibrated_logreg": {
        "label": "H5: Leakage-Free OOF-Calibrated Meta Logistic Regression",
        "stage1_cache_available": True,
        "trainable_params": True,
        "description": "Multiclass L2 LogReg over inner-OOF calibrated DeBERTa, Platt MiniCheck, NLI, and retrieval signals.",
    },
    "H6_selective_abstention": {
        "label": "H6: Selective UNVERIFIABLE Confidence/Evidence Abstention",
        "stage1_cache_available": True,
        "trainable_params": True,
        "description": "Applies max-probability confidence and similarity abstention gates on top of C2 DeBERTa thresholds.",
    },
}

STAGE1_STRATEGY_ORDER: Tuple[str, ...] = (
    "H0_baseline",
    "H1_mc_cal",
    "H2_deberta_cal_c1",
    "H2_deberta_cal_c2",
    "H2b_deberta_cal_plus_nli_rule4",
    "H4_mc_deberta_cascade",
    "H5_meta_calibrated_logreg",
    "H6_selective_abstention",
)


# ---------------------------------------------------------------------------
# Signal Preparation & Semantic Guard Helpers
# ---------------------------------------------------------------------------

def _is_non_factual_record(record: Dict[str, Any]) -> bool:
    return record.get("is_factual") is False or record.get("primary_label") == "NON_FACTUAL"


def _has_empty_evidence_retrieval(record: Dict[str, Any], regime: str) -> bool:
    if regime != "retrieval":
        return False
    sig = record["retrieval"]
    if sig.get("relevant_passages_count") == 0:
        return True
    if str(sig.get("baseline_decision_rule", "")).startswith("Rule 2"):
        return True
    return False


def _has_conflict_triggered(record: Dict[str, Any], regime: str) -> bool:
    if regime != "retrieval":
        return False
    sig = record["retrieval"]
    if bool(sig.get("conflict_triggered", False)):
        return True
    sec_c = float(sig.get("sec_nli_max_contra", 0.0) or 0.0)
    sec_e = float(sig.get("sec_nli_max_entail", 0.0) or 0.0)
    return (
        sec_c >= float(PRE_REGISTERED_GRIDS["h2b_conflict"]["sec_contra_thresh"])
        and sec_e >= float(PRE_REGISTERED_GRIDS["h2b_conflict"]["sec_entail_thresh"])
    )


def extract_feature_arrays(
    records: Sequence[Dict[str, Any]],
    regime: str,
) -> Dict[str, np.ndarray]:
    """Pre-extract numeric arrays for fast vectorized fold fitting and search."""
    n = len(records)
    logits = np.zeros((n, 3), dtype=np.float64)
    nli = np.zeros((n, 3), dtype=np.float64)
    mc_raw = np.full(n, 0.5, dtype=np.float64)
    best_sim = np.ones(n, dtype=np.float64)
    sec_contra = np.zeros(n, dtype=np.float64)
    sec_entail = np.zeros(n, dtype=np.float64)
    empty_ev = np.zeros(n, dtype=bool)
    conflict = np.zeros(n, dtype=bool)
    non_factual = np.zeros(n, dtype=bool)
    y_idx = np.full(n, -1, dtype=np.int64)
    is_3class = np.zeros(n, dtype=bool)
    is_true_hall_incl_mixed = np.zeros(n, dtype=np.float64)

    resp_to_idx: Dict[str, int] = {}
    resp_ids = np.zeros(n, dtype=np.int64)

    for i, rec in enumerate(records):
        sig = rec[regime]
        logits[i] = extract_deberta_logits(rec, regime)
        nli[i] = extract_nli_probs(rec, regime)
        mc = extract_minicheck_prob(rec, regime)
        mc_raw[i] = 0.5 if mc is None else float(mc)
        if regime == "retrieval":
            best_sim[i] = float(sig.get("best_similarity_score", 0.0) or 0.0)
            sec_contra[i] = float(sig.get("sec_nli_max_contra", 0.0) or 0.0)
            sec_entail[i] = float(sig.get("sec_nli_max_entail", 0.0) or 0.0)
            empty_ev[i] = _has_empty_evidence_retrieval(rec, regime)
            conflict[i] = _has_conflict_triggered(rec, regime)
        non_factual[i] = _is_non_factual_record(rec)

        lbl = rec["primary_label"]
        if lbl in LABEL_TO_INDEX:
            y_idx[i] = LABEL_TO_INDEX[lbl]
            is_3class[i] = True
        if lbl in ("CONTRADICTED", "UNSUPPORTED", MIXED_LABEL):
            is_true_hall_incl_mixed[i] = 1.0

        rid = str(rec["response_id"])
        if rid not in resp_to_idx:
            resp_to_idx[rid] = len(resp_to_idx)
        resp_ids[i] = resp_to_idx[rid]

    n_resp = max(1, len(resp_to_idx))
    resp_claim_counts = np.bincount(resp_ids, minlength=n_resp).astype(np.float64)
    resp_true_hall_counts = np.bincount(
        resp_ids, weights=is_true_hall_incl_mixed, minlength=n_resp
    )
    resp_true_hall_pct = np.where(
        resp_claim_counts > 0, (resp_true_hall_counts / resp_claim_counts) * 100.0, 0.0
    )

    return {
        "logits": logits,
        "nli": nli,
        "mc_raw": mc_raw,
        "best_sim": best_sim,
        "sec_contra": sec_contra,
        "sec_entail": sec_entail,
        "empty_ev": empty_ev,
        "conflict": conflict,
        "non_factual": non_factual,
        "y_idx": y_idx,
        "is_3class": is_3class,
        "resp_ids": resp_ids,
        "n_resp": np.asarray([n_resp], dtype=np.int64),
        "resp_claim_counts": resp_claim_counts,
        "resp_true_hall_pct": resp_true_hall_pct,
    }


# ---------------------------------------------------------------------------
# Fast Vectorized Evaluation & Pre-Registered Objective Scoring
# ---------------------------------------------------------------------------

# Internal prediction integer codes:
# 0: CONTRADICTED, 1: SUPPORTED_PROXY, 2: UNSUPPORTED, 3: UNVERIFIABLE, 4: NON_FACTUAL
INT_TO_PRED_LABEL: Tuple[str, ...] = (
    "CONTRADICTED",
    "SUPPORTED_PROXY",
    "UNSUPPORTED",
    "UNVERIFIABLE",
    "NON_FACTUAL",
)


def fast_eval_fold_predictions(
    arrays: Dict[str, np.ndarray],
    pred_codes: np.ndarray,
) -> Dict[str, Any]:
    """Fast NumPy equivalent of penalized 3-class + response confirmed-hallucination MAE.

    Parity with ``compute_arm_metrics`` and ``compute_response_level_evaluation`` is
    verified in unit tests.
    """
    mask3 = arrays["is_3class"]
    y3 = arrays["y_idx"][mask3]
    p3 = pred_codes[mask3]
    n3 = int(len(y3))
    if n3 == 0:
        return {
            "macro_f1": 0.0,
            "support_weighted_f1": 0.0,
            "accuracy": 0.0,
            "coverage": 0.0,
            "mae_conf_pp": 100.0,
            "recalls": {cls: 0.0 for cls in CALIBRATION_CLASSES},
        }

    n_covered = int(np.sum(p3 != 3))
    coverage = n_covered / n3
    accuracy = float(np.sum(p3 == y3)) / n3

    f1s: List[float] = []
    supports: List[int] = []
    recalls: Dict[str, float] = {}
    for cls_idx, cls_name in enumerate(CALIBRATION_CLASSES):
        tp = float(np.sum((y3 == cls_idx) & (p3 == cls_idx)))
        fp = float(np.sum((y3 != cls_idx) & (p3 == cls_idx)))
        fn = float(np.sum((y3 == cls_idx) & (p3 != cls_idx)))
        sup = int(np.sum(y3 == cls_idx))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
        supports.append(sup)
        recalls[cls_name] = rec

    macro_f1 = float(np.mean(f1s))
    support_weighted_f1 = float(
        sum(f * s for f, s in zip(f1s, supports)) / max(1, sum(supports))
    )

    n_resp = int(arrays["n_resp"][0])
    resp_ids = arrays["resp_ids"]
    is_pred_hall = ((pred_codes == 0) | (pred_codes == 2)).astype(np.float64)
    is_factual_pred = (pred_codes != 4).astype(np.float64)

    pred_hall_counts = np.bincount(resp_ids, weights=is_pred_hall, minlength=n_resp)
    denom_counts = np.bincount(resp_ids, weights=is_factual_pred, minlength=n_resp)
    denom_safe = np.where(denom_counts > 0, denom_counts, arrays["resp_claim_counts"])
    pred_hall_pct = np.where(denom_safe > 0, (pred_hall_counts / denom_safe) * 100.0, 0.0)
    mae_conf_pp = float(np.mean(np.abs(pred_hall_pct - arrays["resp_true_hall_pct"])))

    return {
        "macro_f1": macro_f1,
        "support_weighted_f1": support_weighted_f1,
        "accuracy": accuracy,
        "coverage": coverage,
        "mae_conf_pp": mae_conf_pp,
        "recalls": recalls,
    }


def score_objective(
    metrics: Dict[str, Any],
    regime: str,
    objective_name: str,
    complexity_penalty: float = 0.0,
) -> Tuple[bool, bool, Tuple[float, float, float, float, float, float]]:
    """Evaluate a candidate against OBJ_CONSTRAINED or OBJ_MACRO_F1_ONLY with deterministic tie-breakers.

    Returns ``(passes_hard_coverage, passes_all_recall_floors, sort_key)``. Higher ``sort_key`` wins.
    Note: ``macro_f1`` is in [0.0, 1.0], ``mae_conf_pp`` is in [0.0, 100.0] pp, and
    ``alpha = 0.005`` subtracts 0.005 (0.50 Macro-F1 points) per 1.0 pp of MAE above ceiling.
    """
    cov_floor = float(PRE_REGISTERED_OBJECTIVE_PARAMS["coverage_floor"][regime])
    passes_cov = metrics["coverage"] >= cov_floor - 1e-12

    recall_floors = PRE_REGISTERED_OBJECTIVE_PARAMS["recall_floors"]
    passes_recall = all(
        metrics["recalls"][cls] >= floor - 1e-12 for cls, floor in recall_floors.items()
    )

    macro_f1 = float(metrics["macro_f1"])
    mae_pp = float(metrics["mae_conf_pp"])
    if objective_name == "OBJ_CONSTRAINED":
        alpha = float(PRE_REGISTERED_OBJECTIVE_PARAMS["alpha"])
        mae_max = float(PRE_REGISTERED_OBJECTIVE_PARAMS["mae_max_pp"][regime])
        primary = macro_f1 - alpha * max(0.0, mae_pp - mae_max)
    elif objective_name == "OBJ_MACRO_F1_ONLY":
        primary = macro_f1
    else:
        raise ValueError(f"Unknown objective_name: {objective_name!r}")

    key = (
        round(primary, 12),
        round(float(metrics["support_weighted_f1"]), 12),
        round(float(metrics["accuracy"]), 12),
        round(-mae_pp, 12),
        round(float(metrics["coverage"]), 12),
        round(-float(complexity_penalty), 12),
    )
    return passes_cov, passes_recall, key


def select_best_candidate(
    candidates: Sequence[Dict[str, Any]],
    regime: str,
    objective_name: str,
) -> Dict[str, Any]:
    """Select the winning candidate respecting hard coverage floors and recall-floor fallback."""
    if not candidates:
        raise ValueError("Empty candidate list passed to select_best_candidate")

    evaluated: List[Tuple[bool, bool, Tuple[float, ...], Dict[str, Any]]] = []
    for cand in candidates:
        passes_cov, passes_rec, key = score_objective(
            cand["metrics"],
            regime=regime,
            objective_name=objective_name,
            complexity_penalty=float(cand.get("complexity_penalty", 0.0)),
        )
        evaluated.append((passes_cov, passes_rec, key, cand))

    if objective_name == "OBJ_CONSTRAINED":
        tier1 = [x for x in evaluated if x[0] and x[1]]
        tier2 = [x for x in evaluated if x[0]]
        pool = tier1 if tier1 else (tier2 if tier2 else evaluated)
        constraint_status = (
            "all_constraints_met"
            if tier1
            else ("coverage_met_recall_relaxed" if tier2 else "unconstrained_fallback")
        )
    else:
        tier_cov = [x for x in evaluated if x[0]]
        pool = tier_cov if tier_cov else evaluated
        constraint_status = "coverage_met" if tier_cov else "unconstrained_fallback"

    best = max(pool, key=lambda x: x[2])
    out = copy.deepcopy(best[3])
    out["constraint_status"] = constraint_status
    out["objective_value"] = best[2][0]
    return out


# ---------------------------------------------------------------------------
# Strategy Prediction Vectorizers & Single-Record Evaluators
# ---------------------------------------------------------------------------

def _predict_h1_codes(
    arrays: Dict[str, np.ndarray],
    mc_cal_probs: np.ndarray,
    regime: str,
    tau_mc: float,
    tau_floor: float,
) -> np.ndarray:
    n = len(mc_cal_probs)
    preds = np.where(mc_cal_probs >= tau_mc, 1, 2).astype(np.int64)
    if regime == "retrieval":
        abstain = arrays["empty_ev"] | (arrays["best_sim"] < tau_floor)
        preds = np.where(abstain, 3, preds)
    preds = np.where(arrays["non_factual"], 4, preds)
    if len(preds) != n:
        raise RuntimeError("Length mismatch in _predict_h1_codes")
    return preds


def _predict_h2_codes(
    arrays: Dict[str, np.ndarray],
    cal_probs: np.ndarray,
    regime: str,
    tau_contra: float,
    tau_sup: float,
    tau_floor: float,
    apply_conflict_guard: bool = False,
) -> np.ndarray:
    p_c = cal_probs[:, 0]
    p_s = cal_probs[:, 1]
    preds = np.where(
        p_c >= tau_contra,
        0,
        np.where(p_s >= tau_sup, 1, 2),
    ).astype(np.int64)
    if apply_conflict_guard and regime == "retrieval":
        # Rule 4 analog: secondary NLI conflict prevents false SUPPORTED_PROXY
        preds = np.where(arrays["conflict"] & (preds == 1), 0, preds)
    if regime == "retrieval":
        abstain = arrays["empty_ev"] | (arrays["best_sim"] < tau_floor)
        preds = np.where(abstain, 3, preds)
    return np.where(arrays["non_factual"], 4, preds)


def _predict_h4_codes(
    arrays: Dict[str, np.ndarray],
    cal_probs: np.ndarray,
    mc_cal_probs: np.ndarray,
    regime: str,
    tau_contra: float,
    tau_sup: float,
    tau_floor: float,
    margin_thresh: float,
    tau_mc_sup: float,
    tau_mc_hall: float,
) -> np.ndarray:
    base = _predict_h2_codes(arrays, cal_probs, regime, tau_contra, tau_sup, tau_floor)
    p_c = cal_probs[:, 0]
    mc_margin = np.abs(mc_cal_probs - 0.5)

    # Confident MiniCheck fast path only when confident & DeBERTa does not strongly contradict
    fast_sup = (mc_margin >= margin_thresh) & (mc_cal_probs >= tau_mc_sup) & (p_c < tau_contra)
    fast_hall = (mc_margin >= margin_thresh) & ((1.0 - mc_cal_probs) >= tau_mc_hall)
    fast_hall_code = np.where(p_c >= tau_contra, 0, 2)

    preds = np.where(fast_sup, 1, np.where(fast_hall, fast_hall_code, base))
    if regime == "retrieval":
        abstain = arrays["empty_ev"] | (arrays["best_sim"] < tau_floor)
        preds = np.where(abstain, 3, preds)
    return np.where(arrays["non_factual"], 4, preds)


def _predict_h6_codes(
    arrays: Dict[str, np.ndarray],
    cal_probs: np.ndarray,
    regime: str,
    tau_contra: float,
    tau_sup: float,
    tau_floor: float,
    tau_abstain: float,
) -> np.ndarray:
    preds = _predict_h2_codes(arrays, cal_probs, regime, tau_contra, tau_sup, tau_floor)
    max_p = np.max(cal_probs, axis=1)
    abstain_conf = max_p < tau_abstain
    if regime == "retrieval":
        sim_gate = arrays["best_sim"] < float(PRE_REGISTERED_GRIDS["h6_abstain"]["retrieval_sim_gate"])
        abstain_conf = abstain_conf | sim_gate
    preds = np.where(abstain_conf, 3, preds)
    return np.where(arrays["non_factual"], 4, preds)


def build_h5_feature_matrix(
    arrays: Dict[str, np.ndarray],
    cal_probs: np.ndarray,
    mc_cal_probs: np.ndarray,
    regime: str,
) -> np.ndarray:
    """Construct standardized meta-learner features from calibrated probabilities and NLI/retrieval signals."""
    log_cal = np.log(np.clip(cal_probs, 1e-6, 1.0))
    mc_clip = np.clip(mc_cal_probs, 1e-6, 1.0 - 1e-6)
    logit_mc = np.log(mc_clip / (1.0 - mc_clip))[:, np.newaxis]
    cols = [log_cal, logit_mc, arrays["nli"]]
    if regime == "retrieval":
        cols.append(arrays["best_sim"][:, np.newaxis])
        cols.append(arrays["conflict"].astype(np.float64)[:, np.newaxis])
        cols.append(arrays["sec_contra"][:, np.newaxis])
        cols.append(arrays["sec_entail"][:, np.newaxis])
    return np.hstack(cols)


def _predict_h5_codes(
    arrays: Dict[str, np.ndarray],
    features: np.ndarray,
    scaler: StandardScaler,
    clf: LogisticRegression,
    regime: str,
    tau_floor: float,
) -> Tuple[np.ndarray, np.ndarray]:
    x_scaled = scaler.transform(features)
    probs = clf.predict_proba(x_scaled)
    # Map clf.classes_ (always subset/full {0,1,2}) defensively
    full_probs = np.zeros((len(features), 3), dtype=np.float64)
    for col_i, cls_val in enumerate(clf.classes_):
        full_probs[:, int(cls_val)] = probs[:, col_i]
    preds = np.argmax(full_probs, axis=1).astype(np.int64)
    if regime == "retrieval":
        abstain = arrays["empty_ev"] | (arrays["best_sim"] < tau_floor)
        preds = np.where(abstain, 3, preds)
    preds = np.where(arrays["non_factual"], 4, preds)
    return preds, full_probs


def predict_h3_sec_deb(
    record: Dict[str, Any],
    sec_deberta_cal_probs: Sequence[Sequence[float]],
    primary_deberta_cal_probs: Sequence[float],
    tau_contra: float = 0.30,
    tau_sup: float = 0.50,
    tau_floor: float = 0.25,
) -> str:
    """Optional Stage 2 secondary-passage DeBERTa evaluator (never used in Stage 1 cache-only).

    Recovers CONTRADICTED / SUPPORTED_PROXY across retrieved secondary chunks when raw top-3
    chunk DeBERTa probabilities have been explicitly computed in Stage 2.
    """
    if _is_non_factual_record(record):
        return "NON_FACTUAL"
    if _has_empty_evidence_retrieval(record, "retrieval"):
        return "UNVERIFIABLE"
    best_sim = float(record["retrieval"].get("best_similarity_score", 0.0) or 0.0)
    if best_sim < tau_floor:
        return "UNVERIFIABLE"

    p1 = np.asarray(primary_deberta_cal_probs, dtype=np.float64)
    all_probs = [p1] + [np.asarray(p, dtype=np.float64) for p in sec_deberta_cal_probs]
    max_c = max(float(p[0]) for p in all_probs)
    max_s = max(float(p[1]) for p in all_probs)
    if max_c >= tau_contra:
        return "CONTRADICTED"
    if max_s >= tau_sup:
        return "SUPPORTED_PROXY"
    return "UNSUPPORTED"


# ---------------------------------------------------------------------------
# Regime Calibration & Nested 5x4 Source-Grouped Stage 1 Study
# ---------------------------------------------------------------------------

def _split_records_by_sources(
    records: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, List[int]], Dict[str, List[Dict[str, Any]]]]:
    idx_by_src: Dict[str, List[int]] = defaultdict(list)
    rec_by_src: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for i, r in enumerate(records):
        sid = str(r["source_id"])
        idx_by_src[sid].append(i)
        rec_by_src[sid].append(r)
    return idx_by_src, rec_by_src


def _fit_fold_calibrators(
    train_arrays: Dict[str, np.ndarray],
    train_recs: Sequence[Dict[str, Any]],
    fold_name: str,
    c2_lambda_grid: Sequence[float],
    source_folds_inner: Optional[Sequence[Dict[str, Any]]] = None,
    rec_by_src: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    min_examples: int = 3,
) -> Dict[str, Any]:
    """Fit Platt MiniCheck, C1 temp scaling, C2 lambda candidates (selecting best lambda on inner OOF NLL if provided)."""
    support = validate_fold_class_support(
        train_recs,
        fold_name=fold_name,
        min_examples_per_class=min_examples,
        min_sources_per_class=2,
    )
    mask3 = train_arrays["is_3class"]
    logits3 = train_arrays["logits"][mask3]
    y3 = train_arrays["y_idx"][mask3]
    mc3 = train_arrays["mc_raw"][mask3]
    y_sup_bin = (y3 == 1).astype(np.float64)

    cal_mc = fit_binary_platt_calibrator(mc3, y_sup_bin, l2_reg=1e-4, fold_summary=support)
    cal_c1 = fit_multiclass_calibrator(logits3, y3, method="C1_temp_scaling", fold_summary=support)

    c2_lambda_nlls: Dict[float, float] = {}
    if source_folds_inner and rec_by_src is not None:
        # Tune C2 L2 lambda strictly on inner source-grouped OOF NLL
        for lam in c2_lambda_grid:
            oof_probs: List[np.ndarray] = []
            oof_y: List[np.ndarray] = []
            for in_meta in source_folds_inner:
                in_train_recs = [r for s in in_meta["train_sources"] for r in rec_by_src[s]]
                in_eval_recs = [r for s in in_meta["eval_sources"] for r in rec_by_src[s]]
                arr_tr = extract_feature_arrays(in_train_recs, train_recs[0]["_active_regime"])
                arr_ev = extract_feature_arrays(in_eval_recs, train_recs[0]["_active_regime"])
                m_tr = arr_tr["is_3class"]
                m_ev = arr_ev["is_3class"]
                c_in = fit_multiclass_calibrator(
                    arr_tr["logits"][m_tr],
                    arr_tr["y_idx"][m_tr],
                    method="C2_prior_bias_temp",
                    l2_reg=float(lam),
                )
                if np.any(m_ev):
                    oof_probs.append(c_in.transform_logits(arr_ev["logits"][m_ev]))
                    oof_y.append(arr_ev["y_idx"][m_ev])
            all_p = np.vstack(oof_probs)
            all_y = np.concatenate(oof_y)
            c2_lambda_nlls[float(lam)] = float(
                compute_multiclass_calibration_metrics(all_p, all_y)["nll_3class"]
            )
        best_lam = min(sorted(c2_lambda_nlls.keys()), key=lambda k: c2_lambda_nlls[k])
    else:
        best_lam = float(c2_lambda_grid[0])

    cal_c2 = fit_multiclass_calibrator(
        logits3,
        y3,
        method="C2_prior_bias_temp",
        l2_reg=float(best_lam),
        fold_summary=support,
    )
    return {
        "support": support,
        "cal_mc": cal_mc,
        "cal_c1": cal_c1,
        "cal_c2": cal_c2,
        "selected_c2_lambda": float(best_lam),
        "c2_lambda_inner_oof_nll": c2_lambda_nlls,
    }


def _tag_regime(records: Sequence[Dict[str, Any]], regime: str) -> List[Dict[str, Any]]:
    tagged: List[Dict[str, Any]] = []
    for r in records:
        rc = dict(r)
        rc["_active_regime"] = regime
        tagged.append(rc)
    return tagged


def _tune_strategy_on_inner_oof(
    strategy_name: str,
    regime: str,
    inner_oof_recs: Sequence[Dict[str, Any]],
    inner_oof_c1_probs: np.ndarray,
    inner_oof_c2_probs: np.ndarray,
    inner_oof_mc_probs: np.ndarray,
    inner_oof_h5_features: np.ndarray,
    objective_name: str,
    inner_folds: Sequence[Dict[str, Any]],
    rec_by_src: Dict[str, List[Dict[str, Any]]],
    outer_cal_c2_probs: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Select hyperparameters for one strategy strictly on outer-train inner-OOF predictions."""
    del outer_cal_c2_probs
    arrays = extract_feature_arrays(inner_oof_recs, regime)
    tau_floors = (
        PRE_REGISTERED_GRIDS["h2_deberta"]["direct_tau_floor"]
        if regime == "direct"
        else PRE_REGISTERED_GRIDS["h2_deberta"]["retrieval_tau_floor"]
    )
    candidates: List[Dict[str, Any]] = []

    if strategy_name == "H1_mc_cal":
        h1_floors = (
            [0.0] if regime == "direct" else PRE_REGISTERED_GRIDS["h1_mc"]["retrieval_tau_floor"]
        )
        for tf in h1_floors:
            for tmc in PRE_REGISTERED_GRIDS["h1_mc"]["tau_mc"]:
                codes = _predict_h1_codes(arrays, inner_oof_mc_probs, regime, tmc, tf)
                m = fast_eval_fold_predictions(arrays, codes)
                candidates.append(
                    {
                        "params": {"tau_mc": float(tmc), "tau_floor": float(tf)},
                        "metrics": m,
                        "complexity_penalty": abs(float(tmc) - 0.50),
                    }
                )
        return select_best_candidate(candidates, regime, objective_name)

    if strategy_name in ("H2_deberta_cal_c1", "H2_deberta_cal_c2"):
        probs = inner_oof_c1_probs if strategy_name == "H2_deberta_cal_c1" else inner_oof_c2_probs
        for tf in tau_floors:
            for tc in PRE_REGISTERED_GRIDS["h2_deberta"]["tau_contra"]:
                for ts in PRE_REGISTERED_GRIDS["h2_deberta"]["tau_sup"]:
                    codes = _predict_h2_codes(arrays, probs, regime, tc, ts, tf)
                    m = fast_eval_fold_predictions(arrays, codes)
                    candidates.append(
                        {
                            "params": {
                                "tau_contra": float(tc),
                                "tau_sup": float(ts),
                                "tau_floor": float(tf),
                            },
                            "metrics": m,
                            "complexity_penalty": abs(float(tc) - 0.30) + abs(float(ts) - 0.50),
                        }
                    )
        return select_best_candidate(candidates, regime, objective_name)

    if strategy_name == "H2b_deberta_cal_plus_nli_rule4":
        flags = (
            [False]
            if regime == "direct"
            else PRE_REGISTERED_GRIDS["h2b_conflict"]["rule4_conflict_enabled"]
        )
        for flag in flags:
            for tf in tau_floors:
                for tc in PRE_REGISTERED_GRIDS["h2_deberta"]["tau_contra"]:
                    for ts in PRE_REGISTERED_GRIDS["h2_deberta"]["tau_sup"]:
                        codes = _predict_h2_codes(
                            arrays,
                            inner_oof_c2_probs,
                            regime,
                            tc,
                            ts,
                            tf,
                            apply_conflict_guard=bool(flag),
                        )
                        m = fast_eval_fold_predictions(arrays, codes)
                        candidates.append(
                            {
                                "params": {
                                    "tau_contra": float(tc),
                                    "tau_sup": float(ts),
                                    "tau_floor": float(tf),
                                    "rule4_conflict_enabled": bool(flag),
                                },
                                "metrics": m,
                                "complexity_penalty": abs(float(tc) - 0.30)
                                + abs(float(ts) - 0.50)
                                + (0.01 if flag else 0.0),
                            }
                        )
        return select_best_candidate(candidates, regime, objective_name)

    if strategy_name == "H4_mc_deberta_cascade":
        # Anchor base thresholds via constrained H2_c2 first so cascade search stays compact and interpretable
        h2_base = _tune_strategy_on_inner_oof(
            "H2_deberta_cal_c2",
            regime,
            inner_oof_recs,
            inner_oof_c1_probs,
            inner_oof_c2_probs,
            inner_oof_mc_probs,
            inner_oof_h5_features,
            objective_name,
            inner_folds,
            rec_by_src,
        )
        tc = float(h2_base["params"]["tau_contra"])
        ts = float(h2_base["params"]["tau_sup"])
        tf = float(h2_base["params"]["tau_floor"])
        for mg in PRE_REGISTERED_GRIDS["h4_cascade"]["margin_thresh"]:
            for t_sup in PRE_REGISTERED_GRIDS["h4_cascade"]["tau_mc_sup"]:
                for t_hall in PRE_REGISTERED_GRIDS["h4_cascade"]["tau_mc_hall"]:
                    codes = _predict_h4_codes(
                        arrays,
                        inner_oof_c2_probs,
                        inner_oof_mc_probs,
                        regime,
                        tc,
                        ts,
                        tf,
                        mg,
                        t_sup,
                        t_hall,
                    )
                    m = fast_eval_fold_predictions(arrays, codes)
                    candidates.append(
                        {
                            "params": {
                                "tau_contra": tc,
                                "tau_sup": ts,
                                "tau_floor": tf,
                                "margin_thresh": float(mg),
                                "tau_mc_sup": float(t_sup),
                                "tau_mc_hall": float(t_hall),
                            },
                            "metrics": m,
                            "complexity_penalty": abs(float(mg) - 0.30),
                        }
                    )
        return select_best_candidate(candidates, regime, objective_name)

    if strategy_name == "H5_meta_calibrated_logreg":
        # Evaluate each (C, tau_floor) by fitting LogReg across inner folds on precomputed OOF features
        by_src_idx: Dict[str, List[int]] = defaultdict(list)
        for idx, rec in enumerate(inner_oof_recs):
            by_src_idx[str(rec["source_id"])].append(idx)

        h5_floors = (
            [0.0] if regime == "direct" else PRE_REGISTERED_GRIDS["h5_logreg"]["retrieval_tau_floor"]
        )
        for c_val in PRE_REGISTERED_GRIDS["h5_logreg"]["C"]:
            # Generate inner-CV predictions for LogReg at C=c_val
            oof_pred_base = np.zeros(len(inner_oof_recs), dtype=np.int64)
            for in_meta in inner_folds:
                tr_idx = [i for s in in_meta["train_sources"] for i in by_src_idx[s]]
                ev_idx = [i for s in in_meta["eval_sources"] for i in by_src_idx[s]]
                tr_mask = arrays["is_3class"][tr_idx]
                tr_idx_3 = [tr_idx[k] for k in range(len(tr_idx)) if tr_mask[k]]

                scaler = StandardScaler()
                x_tr = scaler.fit_transform(inner_oof_h5_features[tr_idx_3])
                y_tr = arrays["y_idx"][tr_idx_3]
                clf = LogisticRegression(
                    C=float(c_val),
                    max_iter=1000,
                    solver="lbfgs",
                    random_state=42,
                )
                clf.fit(x_tr, y_tr)
                ev_arr = {
                    k: (v[ev_idx] if isinstance(v, np.ndarray) and len(v) == len(inner_oof_recs) else v)
                    for k, v in arrays.items()
                }
                p_ev, _ = _predict_h5_codes(
                    ev_arr,
                    inner_oof_h5_features[ev_idx],
                    scaler,
                    clf,
                    regime,
                    tau_floor=0.0,
                )
                oof_pred_base[ev_idx] = p_ev

            for tf in h5_floors:
                codes = oof_pred_base.copy()
                if regime == "retrieval":
                    abstain = arrays["empty_ev"] | (arrays["best_sim"] < tf)
                    codes = np.where(abstain, 3, codes)
                codes = np.where(arrays["non_factual"], 4, codes)
                m = fast_eval_fold_predictions(arrays, codes)
                candidates.append(
                    {
                        "params": {"C": float(c_val), "tau_floor": float(tf)},
                        "metrics": m,
                        "complexity_penalty": 1.0 / max(float(c_val), 1e-6),
                    }
                )
        return select_best_candidate(candidates, regime, objective_name)

    if strategy_name == "H6_selective_abstention":
        h2_base = _tune_strategy_on_inner_oof(
            "H2_deberta_cal_c2",
            regime,
            inner_oof_recs,
            inner_oof_c1_probs,
            inner_oof_c2_probs,
            inner_oof_mc_probs,
            inner_oof_h5_features,
            objective_name,
            inner_folds,
            rec_by_src,
        )
        tc = float(h2_base["params"]["tau_contra"])
        ts = float(h2_base["params"]["tau_sup"])
        tf = float(h2_base["params"]["tau_floor"])
        for ta in PRE_REGISTERED_GRIDS["h6_abstain"]["tau_abstain"]:
            codes = _predict_h6_codes(
                arrays, inner_oof_c2_probs, regime, tc, ts, tf, tau_abstain=float(ta)
            )
            m = fast_eval_fold_predictions(arrays, codes)
            candidates.append(
                {
                    "params": {
                        "tau_contra": tc,
                        "tau_sup": ts,
                        "tau_floor": tf,
                        "tau_abstain": float(ta),
                        "retrieval_sim_gate": float(
                            PRE_REGISTERED_GRIDS["h6_abstain"]["retrieval_sim_gate"]
                        ),
                    },
                    "metrics": m,
                    "complexity_penalty": float(ta),
                }
            )
        return select_best_candidate(candidates, regime, objective_name)

    raise ValueError(f"Unsupported strategy in Stage 1 tuning: {strategy_name!r}")


def _evaluate_params_on_eval_arrays(
    strategy_name: str,
    regime: str,
    params: Dict[str, Any],
    eval_arrays: Dict[str, np.ndarray],
    eval_c1_probs: np.ndarray,
    eval_c2_probs: np.ndarray,
    eval_mc_probs: np.ndarray,
    eval_h5_features: Optional[np.ndarray] = None,
    h5_scaler: Optional[StandardScaler] = None,
    h5_clf: Optional[LogisticRegression] = None,
) -> Tuple[List[str], List[Optional[float]]]:
    """Apply fixed parameters/calibrators to held-out evaluation records and return labels + continuous hallucination scores."""
    n = len(eval_c2_probs)
    if strategy_name == "H1_mc_cal":
        codes = _predict_h1_codes(
            eval_arrays,
            eval_mc_probs,
            regime,
            float(params["tau_mc"]),
            float(params["tau_floor"]),
        )
        cont = [round(float(1.0 - eval_mc_probs[i]), 6) for i in range(n)]
    elif strategy_name in ("H2_deberta_cal_c1", "H2_deberta_cal_c2"):
        probs = eval_c1_probs if strategy_name == "H2_deberta_cal_c1" else eval_c2_probs
        codes = _predict_h2_codes(
            eval_arrays,
            probs,
            regime,
            float(params["tau_contra"]),
            float(params["tau_sup"]),
            float(params["tau_floor"]),
        )
        cont = [round(float(1.0 - probs[i, 1]), 6) for i in range(n)]
    elif strategy_name == "H2b_deberta_cal_plus_nli_rule4":
        codes = _predict_h2_codes(
            eval_arrays,
            eval_c2_probs,
            regime,
            float(params["tau_contra"]),
            float(params["tau_sup"]),
            float(params["tau_floor"]),
            apply_conflict_guard=bool(params["rule4_conflict_enabled"]),
        )
        cont = [round(float(1.0 - eval_c2_probs[i, 1]), 6) for i in range(n)]
    elif strategy_name == "H4_mc_deberta_cascade":
        codes = _predict_h4_codes(
            eval_arrays,
            eval_c2_probs,
            eval_mc_probs,
            regime,
            float(params["tau_contra"]),
            float(params["tau_sup"]),
            float(params["tau_floor"]),
            float(params["margin_thresh"]),
            float(params["tau_mc_sup"]),
            float(params["tau_mc_hall"]),
        )
        cont = [round(float(1.0 - eval_c2_probs[i, 1]), 6) for i in range(n)]
    elif strategy_name == "H5_meta_calibrated_logreg":
        if eval_h5_features is None or h5_scaler is None or h5_clf is None:
            raise ValueError("H5 evaluation requires eval_h5_features, h5_scaler, and h5_clf")
        codes, h5_probs = _predict_h5_codes(
            eval_arrays,
            eval_h5_features,
            h5_scaler,
            h5_clf,
            regime,
            float(params["tau_floor"]),
        )
        cont = [round(float(1.0 - h5_probs[i, 1]), 6) for i in range(n)]
    elif strategy_name == "H6_selective_abstention":
        codes = _predict_h6_codes(
            eval_arrays,
            eval_c2_probs,
            regime,
            float(params["tau_contra"]),
            float(params["tau_sup"]),
            float(params["tau_floor"]),
            float(params["tau_abstain"]),
        )
        cont = [round(float(1.0 - eval_c2_probs[i, 1]), 6) for i in range(n)]
    else:
        raise ValueError(f"Unsupported strategy: {strategy_name!r}")

    preds = [INT_TO_PRED_LABEL[int(c)] for c in codes]
    return preds, cont


def _fit_h5_model_on_features(
    features: np.ndarray,
    arrays: Dict[str, np.ndarray],
    c_val: float,
) -> Tuple[StandardScaler, LogisticRegression]:
    mask3 = arrays["is_3class"]
    scaler = StandardScaler()
    x_tr = scaler.fit_transform(features[mask3])
    y_tr = arrays["y_idx"][mask3]
    clf = LogisticRegression(
        C=float(c_val),
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
    )
    clf.fit(x_tr, y_tr)
    return scaler, clf


def run_nested_source_cv_stage1(
    records: Sequence[Dict[str, Any]],
    n_outer_folds: int = 5,
    n_inner_folds: int = 4,
    seed: int = 42,
    n_boot: int = 500,
) -> Dict[str, Any]:
    """Run complete leakage-free Stage 1 evaluation on cached Val-Dev records across both regimes.

    - Assigns consistent outer (5) and inner (4) source-grouped folds across ``direct`` and ``retrieval``.
    - Evaluates Pillar 1 calibration (Raw C0, C1 temp scaling, C2 prior-bias+temp, MiniCheck raw vs Platt).
    - Evaluates all 8 Stage-1 strategies under both ``OBJ_CONSTRAINED`` and ``OBJ_MACRO_F1_ONLY``.
    - Excludes ``H3_sec_deb`` explicitly with cache unavailability justification.
    - Computes Holm-Bonferroni corrected paired source-clustered significance tests vs ``H0_baseline``.
    """
    fold_plan = assign_consistent_source_folds(
        records, n_outer_folds=n_outer_folds, n_inner_folds=n_inner_folds, seed=seed
    )
    _, rec_by_src = _split_records_by_sources(records)
    rec_index_map = {str(r["example_id"]): i for i, r in enumerate(records)}

    regimes_out: Dict[str, Any] = {}
    for regime in ("direct", "retrieval"):
        tagged_records = _tag_regime(records, regime)
        _, tagged_by_src = _split_records_by_sources(tagged_records)
        all_arrays = extract_feature_arrays(tagged_records, regime)
        mask3_all = all_arrays["is_3class"]

        # Outer-OOF probability collectors for Pillar 1
        n_recs = len(records)
        oof_raw_probs = np.zeros((n_recs, 3), dtype=np.float64)
        oof_c1_probs = np.zeros((n_recs, 3), dtype=np.float64)
        oof_c2_probs = np.zeros((n_recs, 3), dtype=np.float64)
        oof_c2_fixed_probs: Dict[float, np.ndarray] = {
            float(lam): np.zeros((n_recs, 3), dtype=np.float64)
            for lam in PRE_REGISTERED_GRIDS["c2_lambda"]
        }
        oof_mc_raw = np.zeros(n_recs, dtype=np.float64)
        oof_mc_cal = np.zeros(n_recs, dtype=np.float64)

        # Predictions per objective and strategy
        preds_by_obj: Dict[str, Dict[str, List[str]]] = {
            "OBJ_CONSTRAINED": {s: ["UNVERIFIABLE"] * n_recs for s in STAGE1_STRATEGY_ORDER},
            "OBJ_MACRO_F1_ONLY": {s: ["UNVERIFIABLE"] * n_recs for s in STAGE1_STRATEGY_ORDER},
        }
        cont_by_obj: Dict[str, Dict[str, List[Optional[float]]]] = {
            "OBJ_CONSTRAINED": {s: [None] * n_recs for s in STAGE1_STRATEGY_ORDER},
            "OBJ_MACRO_F1_ONLY": {s: [None] * n_recs for s in STAGE1_STRATEGY_ORDER},
        }

        # Populate H0 baseline identically for both objectives
        for i, rec in enumerate(records):
            p0, c0 = predict_arm_for_record(rec, regime, "arm1_baseline")
            for obj_name in ("OBJ_CONSTRAINED", "OBJ_MACRO_F1_ONLY"):
                preds_by_obj[obj_name]["H0_baseline"][i] = p0
                cont_by_obj[obj_name]["H0_baseline"][i] = c0

        outer_fold_diagnostics: List[Dict[str, Any]] = []
        for outer_meta in fold_plan["outer_folds"]:
            tr_sources = outer_meta["train_sources"]
            ev_sources = outer_meta["eval_sources"]
            assert_disjoint_sources(
                set(tr_sources), set(ev_sources), context=f"{regime}_outer_{outer_meta['outer_fold']}"
            )

            tr_recs = [r for s in tr_sources for r in tagged_by_src[s]]
            ev_recs = [r for s in ev_sources for r in tagged_by_src[s]]
            tr_arr = extract_feature_arrays(tr_recs, regime)
            ev_arr = extract_feature_arrays(ev_recs, regime)

            cal_bundle = _fit_fold_calibrators(
                tr_arr,
                tr_recs,
                fold_name=f"{regime}_outer_{outer_meta['outer_fold']}",
                c2_lambda_grid=PRE_REGISTERED_GRIDS["c2_lambda"],
                source_folds_inner=outer_meta["inner_folds"],
                rec_by_src=tagged_by_src,
                min_examples=5,
            )

            # Collect outer OOF raw + calibrated probs
            raw_ev_p = MulticlassCalibrator("C0_identity", [1.0], [0.0, 0.0, 0.0]).transform_logits(
                ev_arr["logits"]
            )
            c1_ev_p = cal_bundle["cal_c1"].transform_logits(ev_arr["logits"])
            c2_ev_p = cal_bundle["cal_c2"].transform_logits(ev_arr["logits"])
            mc_ev_p = cal_bundle["cal_mc"].predict_probs(ev_arr["mc_raw"])

            # Fixed-lambda C2 OOF diagnostics across lambda grid
            tr_m3 = tr_arr["is_3class"]
            c2_fixed_ev: Dict[float, np.ndarray] = {}
            for lam in PRE_REGISTERED_GRIDS["c2_lambda"]:
                c_fix = fit_multiclass_calibrator(
                    tr_arr["logits"][tr_m3],
                    tr_arr["y_idx"][tr_m3],
                    method="C2_prior_bias_temp",
                    l2_reg=float(lam),
                )
                c2_fixed_ev[float(lam)] = c_fix.transform_logits(ev_arr["logits"])

            for local_i, rec in enumerate(ev_recs):
                g_i = rec_index_map[str(rec["example_id"])]
                oof_raw_probs[g_i] = raw_ev_p[local_i]
                oof_c1_probs[g_i] = c1_ev_p[local_i]
                oof_c2_probs[g_i] = c2_ev_p[local_i]
                oof_mc_raw[g_i] = ev_arr["mc_raw"][local_i]
                oof_mc_cal[g_i] = mc_ev_p[local_i]
                for lam, arr_p in c2_fixed_ev.items():
                    oof_c2_fixed_probs[lam][g_i] = arr_p[local_i]

            # Build strictly leakage-free inner-OOF calibrated probabilities across outer train sources
            n_tr = len(tr_recs)
            tr_idx_map = {str(r["example_id"]): k for k, r in enumerate(tr_recs)}
            inner_oof_c1 = np.zeros((n_tr, 3), dtype=np.float64)
            inner_oof_c2 = np.zeros((n_tr, 3), dtype=np.float64)
            inner_oof_mc = np.zeros(n_tr, dtype=np.float64)
            for in_meta in outer_meta["inner_folds"]:
                in_tr_recs = [r for s in in_meta["train_sources"] for r in tagged_by_src[s]]
                in_ev_recs = [r for s in in_meta["eval_sources"] for r in tagged_by_src[s]]
                in_tr_arr = extract_feature_arrays(in_tr_recs, regime)
                in_ev_arr = extract_feature_arrays(in_ev_recs, regime)
                in_m3 = in_tr_arr["is_3class"]
                in_support = validate_fold_class_support(
                    in_tr_recs,
                    fold_name=f"{regime}_outer_{outer_meta['outer_fold']}_inner_{in_meta['inner_fold']}",
                    min_examples_per_class=3,
                    min_sources_per_class=2,
                )
                in_c1 = fit_multiclass_calibrator(
                    in_tr_arr["logits"][in_m3],
                    in_tr_arr["y_idx"][in_m3],
                    method="C1_temp_scaling",
                    fold_summary=in_support,
                )
                in_c2 = fit_multiclass_calibrator(
                    in_tr_arr["logits"][in_m3],
                    in_tr_arr["y_idx"][in_m3],
                    method="C2_prior_bias_temp",
                    l2_reg=cal_bundle["selected_c2_lambda"],
                    fold_summary=in_support,
                )
                in_mc = fit_binary_platt_calibrator(
                    in_tr_arr["mc_raw"][in_m3],
                    (in_tr_arr["y_idx"][in_m3] == 1).astype(np.float64),
                    fold_summary=in_support,
                )
                p1 = in_c1.transform_logits(in_ev_arr["logits"])
                p2 = in_c2.transform_logits(in_ev_arr["logits"])
                pm = in_mc.predict_probs(in_ev_arr["mc_raw"])
                for li, r in enumerate(in_ev_recs):
                    tk = tr_idx_map[str(r["example_id"])]
                    inner_oof_c1[tk] = p1[li]
                    inner_oof_c2[tk] = p2[li]
                    inner_oof_mc[tk] = pm[li]

            inner_h5_features = build_h5_feature_matrix(
                tr_arr, inner_oof_c2, inner_oof_mc, regime
            )
            ev_h5_features = build_h5_feature_matrix(ev_arr, c2_ev_p, mc_ev_p, regime)

            fold_selected: Dict[str, Any] = {}
            for obj_name in ("OBJ_CONSTRAINED", "OBJ_MACRO_F1_ONLY"):
                fold_selected[obj_name] = {}
                for strat in STAGE1_STRATEGY_ORDER:
                    if strat == "H0_baseline":
                        continue
                    best_inner = _tune_strategy_on_inner_oof(
                        strat,
                        regime,
                        tr_recs,
                        inner_oof_c1,
                        inner_oof_c2,
                        inner_oof_mc,
                        inner_h5_features,
                        obj_name,
                        outer_meta["inner_folds"],
                        tagged_by_src,
                    )
                    h5_scaler = None
                    h5_clf = None
                    if strat == "H5_meta_calibrated_logreg":
                        h5_scaler, h5_clf = _fit_h5_model_on_features(
                            inner_h5_features, tr_arr, float(best_inner["params"]["C"])
                        )
                    ev_preds, ev_cont = _evaluate_params_on_eval_arrays(
                        strat,
                        regime,
                        best_inner["params"],
                        ev_arr,
                        c1_ev_p,
                        c2_ev_p,
                        mc_ev_p,
                        eval_h5_features=ev_h5_features,
                        h5_scaler=h5_scaler,
                        h5_clf=h5_clf,
                    )
                    for li, r in enumerate(ev_recs):
                        g_i = rec_index_map[str(r["example_id"])]
                        preds_by_obj[obj_name][strat][g_i] = ev_preds[li]
                        cont_by_obj[obj_name][strat][g_i] = ev_cont[li]

                    fold_selected[obj_name][strat] = {
                        "params": best_inner["params"],
                        "constraint_status": best_inner["constraint_status"],
                        "inner_oof_utility": round(float(best_inner["objective_value"]), 6),
                    }

            outer_fold_diagnostics.append(
                {
                    "outer_fold": outer_meta["outer_fold"],
                    "train_sources_count": len(tr_sources),
                    "eval_sources_count": len(ev_sources),
                    "selected_c2_lambda": cal_bundle["selected_c2_lambda"],
                    "c2_lambda_inner_oof_nll": {
                        str(k): round(v, 6)
                        for k, v in cal_bundle["c2_lambda_inner_oof_nll"].items()
                    },
                    "c1_temp": cal_bundle["cal_c1"].temperatures[0],
                    "c2_temp": cal_bundle["cal_c2"].temperatures[0],
                    "c2_bias": [round(b, 4) for b in cal_bundle["cal_c2"].bias],
                    "mc_platt": {
                        "scale_a": cal_bundle["cal_mc"].scale_a,
                        "bias_b": cal_bundle["cal_mc"].bias_b,
                    },
                    "eval_support": outer_meta["eval_support"],
                    "selected_hyperparams": fold_selected,
                }
            )

        # Pillar 1: Calibration Metrics over full OOF 3-class examples
        y3_all = all_arrays["y_idx"][mask3_all]
        y_sup_all = (y3_all == 1).astype(np.float64)
        pillar1_cal = {
            "deberta_C0_raw_oof": compute_multiclass_calibration_metrics(
                oof_raw_probs[mask3_all], y3_all
            ),
            "deberta_C1_temp_oof": compute_multiclass_calibration_metrics(
                oof_c1_probs[mask3_all], y3_all
            ),
            "deberta_C2_nested_oof": compute_multiclass_calibration_metrics(
                oof_c2_probs[mask3_all], y3_all
            ),
            "deberta_C2_lambda_sweep_oof_nll": {
                str(lam): compute_multiclass_calibration_metrics(p[mask3_all], y3_all)["nll_3class"]
                for lam, p in sorted(oof_c2_fixed_probs.items())
            },
            "minicheck_raw_support_oof": compute_binary_calibration_metrics(
                oof_mc_raw[mask3_all], y_sup_all
            ),
            "minicheck_platt_support_oof": compute_binary_calibration_metrics(
                oof_mc_cal[mask3_all], y_sup_all
            ),
        }

        # Pillars 2 & 3 + Statistical comparisons per objective
        objectives_out: Dict[str, Any] = {}
        for obj_name in ("OBJ_CONSTRAINED", "OBJ_MACRO_F1_ONLY"):
            strat_results: Dict[str, Any] = {}
            base_preds = preds_by_obj[obj_name]["H0_baseline"]

            raw_p_3way: List[float] = []
            raw_p_bin: List[float] = []
            comp_names: List[str] = []
            comp_payloads: Dict[str, Any] = {}

            for strat in STAGE1_STRATEGY_ORDER:
                preds = preds_by_obj[obj_name][strat]
                cont = cont_by_obj[obj_name][strat]
                arm_m = compute_arm_metrics(records, preds, continuous_scores=cont)
                resp_m = compute_response_level_evaluation(records, preds)
                strat_results[strat] = {
                    "strategy_meta": STRATEGY_REGISTRY[strat],
                    "oof_claim_metrics": arm_m,
                    "oof_response_metrics": resp_m,
                }

                if strat != "H0_baseline":
                    src_paired = clustered_paired_comparison(
                        records, base_preds, preds, cluster_key="source_id", n_perm=5000, seed=seed
                    )
                    y_true_lbls = [str(r["primary_label"]) for r in records]
                    mcnemar_res = exact_mcnemar_test(y_true_lbls, base_preds, preds)
                    boot_src = source_clustered_bootstrap_diff(
                        records,
                        base_preds,
                        preds,
                        n_boot=n_boot,
                        seed=seed,
                    )
                    boot_resp = response_level_bootstrap_diff(
                        records,
                        base_preds,
                        preds,
                        n_boot=min(n_boot, 400),
                        seed=seed,
                    )
                    raw_p_3way.append(
                        float(src_paired["three_class_accuracy"]["paired_sign_flip_p_value"])
                    )
                    raw_p_bin.append(
                        float(src_paired["binary_hallucination_accuracy"]["paired_sign_flip_p_value"])
                    )
                    comp_names.append(strat)
                    comp_payloads[strat] = {
                        "source_clustered_paired": src_paired,
                        "mcnemar_three_class": mcnemar_res["three_class_exact"],
                        "mcnemar_binary_hall": mcnemar_res["binary_hallucination_strict"],
                        "source_clustered_bootstrap_diff": boot_src,
                        "response_clustered_bootstrap_diff": boot_resp,
                    }

            adj_3way = apply_holm_bonferroni(raw_p_3way)
            adj_bin = apply_holm_bonferroni(raw_p_bin)
            for idx, s_name in enumerate(comp_names):
                comp_payloads[s_name]["holm_adjusted_p_3class_source_perm"] = adj_3way[idx]
                comp_payloads[s_name]["holm_adjusted_p_binary_source_perm"] = adj_bin[idx]

            # Rank candidate strategies on outer-OOF according to the active objective
            ranking_cands: List[Dict[str, Any]] = []
            for strat in STAGE1_STRATEGY_ORDER:
                cm = strat_results[strat]["oof_claim_metrics"]
                rm = strat_results[strat]["oof_response_metrics"]
                p3 = cm["penalized_3class"]
                fast_like = {
                    "macro_f1": float(p3["macro_f1"]),
                    "support_weighted_f1": float(p3["support_weighted_f1"]),
                    "accuracy": float(p3["accuracy"]),
                    "coverage": float(cm["coverage"]),
                    "mae_conf_pp": float(rm["confirmed_hallucination_rate_pct"]["mae"]),
                    "recalls": {
                        cls: float(p3["per_class"][cls]["recall"]) for cls in CALIBRATION_CLASSES
                    },
                }
                _, _, sort_k = score_objective(fast_like, regime, obj_name, 0.0)
                passes_cov, passes_rec, _ = score_objective(fast_like, regime, obj_name, 0.0)
                ranking_cands.append(
                    {
                        "strategy": strat,
                        "objective_utility": round(float(sort_k[0]), 6),
                        "passes_coverage_floor": passes_cov,
                        "passes_recall_floors": passes_rec,
                        "macro_f1": round(fast_like["macro_f1"], 6),
                        "support_weighted_f1": round(fast_like["support_weighted_f1"], 6),
                        "accuracy": round(fast_like["accuracy"], 6),
                        "coverage": round(fast_like["coverage"], 6),
                        "mae_conf_pp": round(fast_like["mae_conf_pp"], 4),
                        "sort_key": sort_k,
                    }
                )
            ranking_cands_sorted = sorted(
                ranking_cands,
                key=lambda x: (
                    x["passes_coverage_floor"],
                    x["passes_recall_floors"] if obj_name == "OBJ_CONSTRAINED" else True,
                    x["sort_key"],
                ),
                reverse=True,
            )
            winner_info = {k: v for k, v in ranking_cands_sorted[0].items() if k != "sort_key"}
            objectives_out[obj_name] = {
                "winner": winner_info,
                "leaderboard": [
                    {k: v for k, v in item.items() if k != "sort_key"}
                    for item in ranking_cands_sorted
                ],
                "strategies": strat_results,
                "paired_comparisons_vs_H0": comp_payloads,
            }

        regimes_out[regime] = {
            "regime": regime,
            "pillar1_calibration": pillar1_cal,
            "outer_fold_diagnostics": outer_fold_diagnostics,
            "objectives": objectives_out,
        }

    return {
        "schema_version": "phase8_step7_stage1_v1",
        "pre_registered_objective_params": PRE_REGISTERED_OBJECTIVE_PARAMS,
        "pre_registered_grids": PRE_REGISTERED_GRIDS,
        "strategy_registry": STRATEGY_REGISTRY,
        "excluded_strategies": {
            "H3_sec_deb": STRATEGY_REGISTRY["H3_sec_deb"]["exclusion_reason"],
        },
        "source_fold_plan_summary": {
            "n_outer_folds": fold_plan["n_outer_folds"],
            "n_inner_folds": fold_plan["n_inner_folds"],
            "total_sources": fold_plan["total_sources"],
            "source_to_outer_fold": fold_plan["source_to_outer_fold"],
        },
        "regimes": regimes_out,
    }


# ---------------------------------------------------------------------------
# Predetermined Stage 1 -> Stage 2 Freezing & Val-Lock Evaluation
# ---------------------------------------------------------------------------

def freeze_stage2_bundle(
    val_dev_records: Sequence[Dict[str, Any]],
    n_outer_folds: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """Deterministically fit and freeze calibrators + thresholds strictly from all 30 Val-Dev sources.

    Procedure (never touches Val-Lock):
    1. Use 5-fold source-grouped Val-Dev CV (the same outer fold assignment as Stage 1) to generate
       strictly out-of-fold Val-Dev predictions for calibrator families (C1, C2 lambda sweep, Platt MC).
    2. Select global best C2 lambda by lowest 5-fold OOF 3-class NLL.
    3. Refit final C1, C2, and Platt MC calibrators ONCE on all 30 Val-Dev sources (after checking
       ``validate_fold_class_support``).
    4. Tune frozen strategy thresholds (H1, H2_c1, H2_c2, H2b, H4, H5_C, H6) strictly over the
       5-fold Val-Dev OOF calibrated predictions for both ``OBJ_CONSTRAINED`` and ``OBJ_MACRO_F1_ONLY``.
    5. Refit H5 LogisticRegression weights ONCE over the 5-fold OOF calibrated Val-Dev feature matrix.
    """
    fold_plan = assign_consistent_source_folds(
        val_dev_records, n_outer_folds=n_outer_folds, n_inner_folds=3, seed=seed
    )
    val_dev_sources = sorted({str(r["source_id"]) for r in val_dev_records})
    rec_index_map = {str(r["example_id"]): i for i, r in enumerate(val_dev_records)}
    dummy_inner_folds = [
        {"train_sources": f["train_sources"], "eval_sources": f["eval_sources"]}
        for f in fold_plan["outer_folds"]
    ]

    bundle_regimes: Dict[str, Any] = {}
    for regime in ("direct", "retrieval"):
        tagged = _tag_regime(val_dev_records, regime)
        _, tagged_by_src = _split_records_by_sources(tagged)
        all_arr = extract_feature_arrays(tagged, regime)
        full_support = validate_fold_class_support(
            tagged,
            fold_name=f"{regime}_valdev_full_30_refit",
            min_examples_per_class=10,
            min_sources_per_class=3,
        )

        # 5-fold Val-Dev OOF calibration probabilities for threshold freezing
        n_recs = len(tagged)
        oof_c1 = np.zeros((n_recs, 3), dtype=np.float64)
        oof_c2_by_lam: Dict[float, np.ndarray] = {
            float(lam): np.zeros((n_recs, 3), dtype=np.float64)
            for lam in PRE_REGISTERED_GRIDS["c2_lambda"]
        }
        oof_mc = np.zeros(n_recs, dtype=np.float64)

        for f_meta in fold_plan["outer_folds"]:
            tr_recs = [r for s in f_meta["train_sources"] for r in tagged_by_src[s]]
            ev_recs = [r for s in f_meta["eval_sources"] for r in tagged_by_src[s]]
            tr_arr = extract_feature_arrays(tr_recs, regime)
            ev_arr = extract_feature_arrays(ev_recs, regime)
            m3 = tr_arr["is_3class"]

            f_c1 = fit_multiclass_calibrator(tr_arr["logits"][m3], tr_arr["y_idx"][m3], "C1_temp_scaling")
            f_mc = fit_binary_platt_calibrator(
                tr_arr["mc_raw"][m3], (tr_arr["y_idx"][m3] == 1).astype(np.float64)
            )
            p1 = f_c1.transform_logits(ev_arr["logits"])
            pm = f_mc.predict_probs(ev_arr["mc_raw"])
            p2_map = {
                float(lam): fit_multiclass_calibrator(
                    tr_arr["logits"][m3],
                    tr_arr["y_idx"][m3],
                    "C2_prior_bias_temp",
                    l2_reg=float(lam),
                ).transform_logits(ev_arr["logits"])
                for lam in PRE_REGISTERED_GRIDS["c2_lambda"]
            }
            for li, r in enumerate(ev_recs):
                gi = rec_index_map[str(r["example_id"])]
                oof_c1[gi] = p1[li]
                oof_mc[gi] = pm[li]
                for lam, arr_p in p2_map.items():
                    oof_c2_by_lam[lam][gi] = arr_p[li]

        m3_all = all_arr["is_3class"]
        y3_all = all_arr["y_idx"][m3_all]
        lam_nlls = {
            float(lam): float(
                compute_multiclass_calibration_metrics(p[m3_all], y3_all)["nll_3class"]
            )
            for lam, p in oof_c2_by_lam.items()
        }
        best_lam = min(sorted(lam_nlls.keys()), key=lambda k: lam_nlls[k])
        oof_c2_best = oof_c2_by_lam[best_lam]
        oof_h5_feats = build_h5_feature_matrix(all_arr, oof_c2_best, oof_mc, regime)

        # Refit final calibrators once on ALL 30 Val-Dev sources
        final_cal_c1 = fit_multiclass_calibrator(
            all_arr["logits"][m3_all],
            y3_all,
            method="C1_temp_scaling",
            fold_summary=full_support,
        )
        final_cal_c2 = fit_multiclass_calibrator(
            all_arr["logits"][m3_all],
            y3_all,
            method="C2_prior_bias_temp",
            l2_reg=float(best_lam),
            fold_summary=full_support,
        )
        final_cal_mc = fit_binary_platt_calibrator(
            all_arr["mc_raw"][m3_all],
            (y3_all == 1).astype(np.float64),
            fold_summary=full_support,
        )

        # Freeze thresholds per objective over Val-Dev OOF calibrated scores
        frozen_by_obj: Dict[str, Any] = {}
        for obj_name in ("OBJ_CONSTRAINED", "OBJ_MACRO_F1_ONLY"):
            strat_frozen: Dict[str, Any] = {}
            for strat in STAGE1_STRATEGY_ORDER:
                if strat == "H0_baseline":
                    strat_frozen[strat] = {"params": {}, "val_dev_oof_utility": None}
                    continue
                best = _tune_strategy_on_inner_oof(
                    strat,
                    regime,
                    tagged,
                    oof_c1,
                    oof_c2_best,
                    oof_mc,
                    oof_h5_feats,
                    obj_name,
                    dummy_inner_folds,
                    tagged_by_src,
                )
                entry: Dict[str, Any] = {
                    "params": best["params"],
                    "constraint_status": best["constraint_status"],
                    "val_dev_oof_utility": round(float(best["objective_value"]), 6),
                }
                if strat == "H5_meta_calibrated_logreg":
                    scaler, clf = _fit_h5_model_on_features(
                        oof_h5_feats, all_arr, float(best["params"]["C"])
                    )
                    entry["model_state"] = {
                        "scaler_mean": scaler.mean_.tolist(),
                        "scaler_scale": scaler.scale_.tolist(),
                        "classes": clf.classes_.tolist(),
                        "coef": clf.coef_.tolist(),
                        "intercept": clf.intercept_.tolist(),
                    }
                strat_frozen[strat] = entry

            # Record primary Val-Dev OOF champion strategy for this objective
            winners = []
            for strat in STAGE1_STRATEGY_ORDER:
                if strat == "H0_baseline":
                    # Compute H0 Val-Dev metrics for reference comparison
                    p0 = [predict_arm_for_record(r, regime, "arm1_baseline")[0] for r in tagged]
                    codes0 = np.asarray([INT_TO_PRED_LABEL.index(x) for x in p0], dtype=np.int64)
                    m0 = fast_eval_fold_predictions(all_arr, codes0)
                    p_cov0, p_rec0, k0 = score_objective(m0, regime, obj_name, 0.0)
                    winners.append(((p_cov0, p_rec0 if obj_name == "OBJ_CONSTRAINED" else True, k0), strat))
                else:
                    # Score frozen strategy on Val-Dev OOF predictions
                    h5_s, h5_c = None, None
                    if strat == "H5_meta_calibrated_logreg":
                        h5_s, h5_c = _restore_h5_model(strat_frozen[strat]["model_state"])
                    preds_s, _ = _evaluate_params_on_eval_arrays(
                        strat,
                        regime,
                        strat_frozen[strat]["params"],
                        all_arr,
                        oof_c1,
                        oof_c2_best,
                        oof_mc,
                        eval_h5_features=oof_h5_feats,
                        h5_scaler=h5_s,
                        h5_clf=h5_c,
                    )
                    codes_s = np.asarray(
                        [INT_TO_PRED_LABEL.index(x) for x in preds_s], dtype=np.int64
                    )
                    ms = fast_eval_fold_predictions(all_arr, codes_s)
                    p_cov, p_rec, ks = score_objective(ms, regime, obj_name, 0.0)
                    winners.append(((p_cov, p_rec if obj_name == "OBJ_CONSTRAINED" else True, ks), strat))
            champion = max(winners, key=lambda x: x[0])[1]
            frozen_by_obj[obj_name] = {
                "frozen_winner_strategy": champion,
                "strategies": strat_frozen,
            }

        bundle_regimes[regime] = {
            "val_dev_support": full_support,
            "selected_c2_lambda": float(best_lam),
            "val_dev_oof_c2_lambda_nll": {str(k): round(v, 6) for k, v in lam_nlls.items()},
            "calibrators": {
                "C1_temp_scaling": final_cal_c1.to_dict(),
                "C2_prior_bias_temp": final_cal_c2.to_dict(),
                "Platt_minicheck": final_cal_mc.to_dict(),
            },
            "objectives": frozen_by_obj,
        }

    return {
        "schema_version": "phase8_step7_frozen_bundle_v1",
        "freezing_policy": (
            "Calibrators selected by 5-fold Val-Dev OOF NLL and refitted once on all 30 Val-Dev "
            "sources; strategy thresholds selected on Val-Dev OOF calibrated probs; zero Val-Lock peeking."
        ),
        "val_dev_source_ids": val_dev_sources,
        "regimes": bundle_regimes,
    }


def _restore_h5_model(model_state: Dict[str, Any]) -> Tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(model_state["scaler_mean"], dtype=np.float64)
    scaler.scale_ = np.asarray(model_state["scaler_scale"], dtype=np.float64)
    scaler.var_ = np.square(scaler.scale_)
    scaler.n_features_in_ = len(scaler.mean_)

    clf = LogisticRegression()
    clf.classes_ = np.asarray(model_state["classes"], dtype=np.int64)
    clf.coef_ = np.asarray(model_state["coef"], dtype=np.float64)
    clf.intercept_ = np.asarray(model_state["intercept"], dtype=np.float64)
    clf.n_features_in_ = scaler.n_features_in_
    return scaler, clf


def apply_frozen_bundle_to_records(
    val_lock_records: Sequence[Dict[str, Any]],
    frozen_bundle: Dict[str, Any],
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Apply an immutable Stage 1 frozen bundle once to held-out Stage 2 Val-Lock records.

    Asserts strict source disjointness between ``val_dev_source_ids`` and ``val_lock_records``.
    Does NOT refit or retune any calibrators or thresholds on ``val_lock_records``.
    """
    val_dev_sources = {str(s) for s in frozen_bundle["val_dev_source_ids"]}
    val_lock_sources = {str(r["source_id"]) for r in val_lock_records}
    assert_disjoint_sources(val_dev_sources, val_lock_sources, context="Stage2_ValDev_vs_ValLock")

    regimes_out: Dict[str, Any] = {}
    for regime in ("direct", "retrieval"):
        r_bundle = frozen_bundle["regimes"][regime]
        tagged = _tag_regime(val_lock_records, regime)
        arr = extract_feature_arrays(tagged, regime)
        mask3 = arr["is_3class"]
        y3 = arr["y_idx"][mask3]
        y_sup_bin = (y3 == 1).astype(np.float64)

        cal_c0 = MulticlassCalibrator("C0_identity", [1.0], [0.0, 0.0, 0.0])
        cal_c1 = MulticlassCalibrator.from_dict(r_bundle["calibrators"]["C1_temp_scaling"])
        cal_c2 = MulticlassCalibrator.from_dict(r_bundle["calibrators"]["C2_prior_bias_temp"])
        cal_mc = BinaryPlattCalibrator.from_dict(r_bundle["calibrators"]["Platt_minicheck"])

        p_raw = cal_c0.transform_logits(arr["logits"])
        p_c1 = cal_c1.transform_logits(arr["logits"])
        p_c2 = cal_c2.transform_logits(arr["logits"])
        p_mc = cal_mc.predict_probs(arr["mc_raw"])
        h5_feats = build_h5_feature_matrix(arr, p_c2, p_mc, regime)

        pillar1 = {
            "deberta_C0_raw_vallock": compute_multiclass_calibration_metrics(p_raw[mask3], y3),
            "deberta_C1_temp_vallock": compute_multiclass_calibration_metrics(p_c1[mask3], y3),
            "deberta_C2_frozen_vallock": compute_multiclass_calibration_metrics(p_c2[mask3], y3),
            "minicheck_raw_vallock": compute_binary_calibration_metrics(
                arr["mc_raw"][mask3], y_sup_bin
            ),
            "minicheck_platt_vallock": compute_binary_calibration_metrics(p_mc[mask3], y_sup_bin),
        }

        objectives_out: Dict[str, Any] = {}
        for obj_name in ("OBJ_CONSTRAINED", "OBJ_MACRO_F1_ONLY"):
            obj_bundle = r_bundle["objectives"][obj_name]
            base_preds = [
                predict_arm_for_record(r, regime, "arm1_baseline")[0] for r in val_lock_records
            ]
            base_cont = [
                predict_arm_for_record(r, regime, "arm1_baseline")[1] for r in val_lock_records
            ]
            strat_results: Dict[str, Any] = {}
            raw_p_3way: List[float] = []
            raw_p_bin: List[float] = []
            comp_names: List[str] = []
            comp_payloads: Dict[str, Any] = {}

            for strat in STAGE1_STRATEGY_ORDER:
                if strat == "H0_baseline":
                    preds, cont = base_preds, base_cont
                else:
                    s_info = obj_bundle["strategies"][strat]
                    h5_s, h5_c = (None, None)
                    if strat == "H5_meta_calibrated_logreg":
                        h5_s, h5_c = _restore_h5_model(s_info["model_state"])
                    preds, cont = _evaluate_params_on_eval_arrays(
                        strat,
                        regime,
                        s_info["params"],
                        arr,
                        p_c1,
                        p_c2,
                        p_mc,
                        eval_h5_features=h5_feats,
                        h5_scaler=h5_s,
                        h5_clf=h5_c,
                    )

                arm_m = compute_arm_metrics(val_lock_records, preds, continuous_scores=cont)
                resp_m = compute_response_level_evaluation(val_lock_records, preds)
                strat_results[strat] = {
                    "frozen_params": obj_bundle["strategies"][strat]["params"],
                    "vallock_claim_metrics": arm_m,
                    "vallock_response_metrics": resp_m,
                }
                if strat != "H0_baseline":
                    src_paired = clustered_paired_comparison(
                        val_lock_records,
                        base_preds,
                        preds,
                        cluster_key="source_id",
                        n_perm=5000,
                        seed=seed,
                    )
                    y_true_lbls = [str(r["primary_label"]) for r in val_lock_records]
                    mcnemar_res = exact_mcnemar_test(y_true_lbls, base_preds, preds)
                    boot_src = source_clustered_bootstrap_diff(
                        val_lock_records,
                        base_preds,
                        preds,
                        n_boot=n_boot,
                        seed=seed,
                    )
                    boot_resp = response_level_bootstrap_diff(
                        val_lock_records,
                        base_preds,
                        preds,
                        n_boot=min(n_boot, 400),
                        seed=seed,
                    )
                    raw_p_3way.append(
                        float(src_paired["three_class_accuracy"]["paired_sign_flip_p_value"])
                    )
                    raw_p_bin.append(
                        float(src_paired["binary_hallucination_accuracy"]["paired_sign_flip_p_value"])
                    )
                    comp_names.append(strat)
                    comp_payloads[strat] = {
                        "source_clustered_paired": src_paired,
                        "mcnemar_three_class": mcnemar_res["three_class_exact"],
                        "mcnemar_binary_hall": mcnemar_res["binary_hallucination_strict"],
                        "source_clustered_bootstrap_diff": boot_src,
                        "response_clustered_bootstrap_diff": boot_resp,
                    }

            adj_3 = apply_holm_bonferroni(raw_p_3way)
            adj_b = apply_holm_bonferroni(raw_p_bin)
            for idx, s_name in enumerate(comp_names):
                comp_payloads[s_name]["holm_adjusted_p_3class_source_perm"] = adj_3[idx]
                comp_payloads[s_name]["holm_adjusted_p_binary_source_perm"] = adj_b[idx]

            objectives_out[obj_name] = {
                "stage1_frozen_winner": obj_bundle["frozen_winner_strategy"],
                "strategies": strat_results,
                "paired_comparisons_vs_H0": comp_payloads,
            }

        regimes_out[regime] = {
            "regime": regime,
            "val_lock_support": summarize_source_class_support(val_lock_records),
            "pillar1_calibration": pillar1,
            "objectives": objectives_out,
        }

    return {
        "schema_version": "phase8_step7_stage2_vallock_v1",
        "val_dev_sources_count": len(val_dev_sources),
        "val_lock_sources_count": len(val_lock_sources),
        "regimes": regimes_out,
    }

