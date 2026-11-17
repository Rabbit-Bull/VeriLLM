"""
Phase 8 Step 7: Offline Probability Calibration & Source-Grouped Cross-Validation Utilities.

Provides regime-specific post-hoc probability calibration (temperature scaling,
L2-regularized prior-bias + temperature scaling, vector scaling, binary Platt scaling),
raw-probability logit reconstruction, rare-class fold support guards, and proper
scoring / ECE / Classwise-ECE metrics strictly separated at the ``source_id`` level.

No production decision logic, model weights, or ragtruth_test.jsonl paths are modified
or accessed by this module.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.optimize import minimize

CALIBRATION_CLASSES: Tuple[str, str, str] = (
    "CONTRADICTED",
    "SUPPORTED_PROXY",
    "UNSUPPORTED",
)
LABEL_TO_INDEX: Dict[str, int] = {lbl: idx for idx, lbl in enumerate(CALIBRATION_CLASSES)}
INDEX_TO_LABEL: Dict[int, str] = {idx: lbl for idx, lbl in enumerate(CALIBRATION_CLASSES)}


def _stable_softmax(logits: np.ndarray) -> np.ndarray:
    """Compute numerically stable softmax over the last axis."""
    arr = np.asarray(logits, dtype=np.float64)
    shifted = arr - np.max(arr, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    denom = np.sum(exp, axis=-1, keepdims=True)
    probs = exp / np.maximum(denom, 1e-30)
    return np.clip(probs, 1e-12, 1.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    arr = np.clip(np.asarray(x, dtype=np.float64), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-arr))


def reconstruct_logits_from_probs(
    probs: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Reconstruct centered log-probabilities equivalent to logits up to an additive constant.

    Because ``softmax(z)`` is shift-invariant, ``log(softmax(z)) = z - logsumexp(z)``, so
    affine transforms of centered log-probs equal affine transforms of raw logits.
    """
    arr = np.asarray(probs, dtype=np.float64)
    if arr.ndim not in (1, 2) or arr.shape[-1] != 3:
        raise ValueError(f"Expected probs shape (3,) or (N, 3), got {arr.shape}")
    clipped = np.clip(arr, eps, 1.0)
    renorm = clipped / np.sum(clipped, axis=-1, keepdims=True)
    log_p = np.log(renorm)
    return log_p - np.mean(log_p, axis=-1, keepdims=True)


def extract_deberta_probs(record: Dict[str, Any], regime: str) -> np.ndarray:
    """Extract [CONTRADICTED, SUPPORTED_PROXY, UNSUPPORTED] raw DeBERTa probabilities."""
    sig = record[regime]
    return np.asarray(
        [
            float(sig["deberta_contradiction_prob"]),
            float(sig["deberta_supported_prob"]),
            float(sig["deberta_unsupported_prob"]),
        ],
        dtype=np.float64,
    )


def extract_deberta_logits(record: Dict[str, Any], regime: str) -> np.ndarray:
    """Return raw DeBERTa logits if explicitly cached, else reconstruct from cached softmax probs."""
    sig = record[regime]
    if sig.get("deberta_raw_logits") is not None:
        raw = np.asarray(sig["deberta_raw_logits"], dtype=np.float64)
        if raw.shape != (3,):
            raise ValueError(f"Expected deberta_raw_logits shape (3,), got {raw.shape}")
        return raw - np.mean(raw)
    return reconstruct_logits_from_probs(extract_deberta_probs(record, regime))


def extract_nli_probs(record: Dict[str, Any], regime: str) -> np.ndarray:
    """Extract [CONTRADICTED, ENTAILMENT, NEUTRAL] probabilities from cached NLI scores."""
    sig = record[regime]
    return np.asarray(
        [
            float(sig["nli_contra"]),
            float(sig["nli_entail"]),
            float(sig["nli_neutral"]),
        ],
        dtype=np.float64,
    )


def extract_minicheck_prob(record: Dict[str, Any], regime: str) -> Optional[float]:
    """Extract MiniCheck support probability from cached record."""
    val = record[regime].get("mc_prob")
    return None if val is None else float(val)


def assert_disjoint_sources(
    train_sources: Set[str],
    eval_sources: Set[str],
    context: str = "fold",
) -> None:
    """Hard guarantee that train and evaluation splits share zero source_id clusters."""
    overlap = sorted(set(train_sources).intersection(set(eval_sources)))
    if overlap:
        raise AssertionError(
            f"Source leakage detected in {context}: {len(overlap)} overlapping source_ids ({overlap[:5]})"
        )


def validate_fold_class_support(
    records: Sequence[Dict[str, Any]],
    fold_name: str,
    min_examples_per_class: int = 5,
    min_sources_per_class: int = 2,
) -> Dict[str, Any]:
    """Validate canonical 3-class example and source counts on a training fold.

    Stops cleanly with an informative ``ValueError`` rather than silently fitting
    calibrators on empty or single-source rare classes.
    """
    class_counts: Dict[str, int] = {cls: 0 for cls in CALIBRATION_CLASSES}
    class_sources: Dict[str, Set[str]] = {cls: set() for cls in CALIBRATION_CLASSES}
    all_sources: Set[str] = set()

    for rec in records:
        sid = str(rec["source_id"])
        all_sources.add(sid)
        lbl = rec["primary_label"]
        if lbl in LABEL_TO_INDEX:
            class_counts[lbl] += 1
            class_sources[lbl].add(sid)

    n_3class = sum(class_counts.values())
    if n_3class == 0:
        raise ValueError(f"[{fold_name}] Zero canonical 3-class examples available for fitting.")

    failures: List[str] = []
    for cls in CALIBRATION_CLASSES:
        cnt = class_counts[cls]
        n_src = len(class_sources[cls])
        if cnt < min_examples_per_class:
            failures.append(
                f"{cls} has {cnt} examples (< required {min_examples_per_class})"
            )
        if n_src < min_sources_per_class:
            failures.append(
                f"{cls} spans {n_src} sources (< required {min_sources_per_class})"
            )

    summary = {
        "fold_name": fold_name,
        "total_sources": len(all_sources),
        "n_3class_examples": n_3class,
        "counts_by_class": class_counts,
        "sources_by_class": {cls: len(sids) for cls, sids in class_sources.items()},
    }
    if failures:
        raise ValueError(
            f"Invalid training fold [{fold_name}] for calibration/fitting: {'; '.join(failures)}. "
            f"Summary={summary}"
        )
    return summary


def summarize_source_class_support(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Report non-failing class and contributing source counts (safe for evaluation folds)."""
    counts = {cls: 0 for cls in CALIBRATION_CLASSES}
    sources = {cls: set() for cls in CALIBRATION_CLASSES}
    all_sources: Set[str] = set()
    mixed_count = 0
    for rec in records:
        sid = str(rec["source_id"])
        all_sources.add(sid)
        lbl = rec["primary_label"]
        if lbl in LABEL_TO_INDEX:
            counts[lbl] += 1
            sources[lbl].add(sid)
        else:
            mixed_count += 1
    return {
        "total_sources": len(all_sources),
        "n_3class_examples": sum(counts.values()),
        "mixed_diagnostic_examples": mixed_count,
        "counts_by_class": counts,
        "sources_by_class": {cls: len(sids) for cls, sids in sources.items()},
        "absent_classes": [cls for cls, c in counts.items() if c == 0],
    }


def _stratified_source_round_robin(
    records: Sequence[Dict[str, Any]],
    n_folds: int,
    seed: int,
) -> Dict[str, int]:
    """Deterministically assign each unique source_id to one fold stratified by (task_type, has_contra)."""
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")

    by_src: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_src[str(rec["source_id"])].append(rec)

    if len(by_src) < n_folds:
        raise ValueError(
            f"Cannot form {n_folds} source-grouped folds from only {len(by_src)} unique sources."
        )

    strata: Dict[Tuple[str, str], List[Tuple[str, int, int]]] = defaultdict(list)
    for sid, recs in by_src.items():
        task = str(recs[0].get("task_type", "Unknown"))
        n_c = sum(1 for r in recs if r["primary_label"] == "CONTRADICTED")
        n_u = sum(1 for r in recs if r["primary_label"] == "UNSUPPORTED")
        stratum_key = (task, "CONTRA_PRESENT" if n_c > 0 else "NO_CONTRA")
        strata[stratum_key].append((sid, n_c, n_u))

    rng = np.random.default_rng(seed)
    ordered_sids: List[str] = []
    # Prioritize CONTRA_PRESENT strata first so minority labels spread evenly across folds.
    for key in sorted(strata.keys(), key=lambda k: (0 if k[1] == "CONTRA_PRESENT" else 1, k[0])):
        bucket = sorted(strata[key], key=lambda item: (-item[1], -item[2], item[0]))
        # Permute ties deterministically while preserving coarse minority balance
        tie_groups: Dict[Tuple[int, int], List[str]] = defaultdict(list)
        for sid, nc, nu in bucket:
            tie_groups[(nc, nu)].append(sid)
        for nc_nu in sorted(tie_groups.keys(), reverse=True):
            grp = sorted(tie_groups[nc_nu])
            perm_idx = rng.permutation(len(grp)).tolist()
            ordered_sids.extend(grp[i] for i in perm_idx)

    source_to_fold: Dict[str, int] = {}
    for idx, sid in enumerate(ordered_sids):
        source_to_fold[sid] = idx % n_folds
    return source_to_fold


def assign_consistent_source_folds(
    records: Sequence[Dict[str, Any]],
    n_outer_folds: int = 5,
    n_inner_folds: int = 4,
    seed: int = 42,
) -> Dict[str, Any]:
    """Assign each ``source_id`` to exactly one outer and nested inner fold identically across regimes."""
    source_to_outer = _stratified_source_round_robin(records, n_folds=n_outer_folds, seed=seed)
    by_src: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_src[str(rec["source_id"])].append(rec)

    outer_folds_meta: List[Dict[str, Any]] = []
    for f_id in range(n_outer_folds):
        eval_sources = sorted(sid for sid, fold in source_to_outer.items() if fold == f_id)
        train_sources = sorted(sid for sid, fold in source_to_outer.items() if fold != f_id)
        assert_disjoint_sources(set(train_sources), set(eval_sources), context=f"outer_fold_{f_id}")

        train_recs = [r for sid in train_sources for r in by_src[sid]]
        eval_recs = [r for sid in eval_sources for r in by_src[sid]]
        validate_fold_class_support(
            train_recs,
            fold_name=f"outer_fold_{f_id}_train",
            min_examples_per_class=5,
            min_sources_per_class=2,
        )
        eval_support = summarize_source_class_support(eval_recs)

        src_to_inner = _stratified_source_round_robin(
            train_recs, n_folds=n_inner_folds, seed=seed + 100 + f_id
        )
        inner_splits: List[Dict[str, Any]] = []
        for in_id in range(n_inner_folds):
            in_eval = sorted(sid for sid, fold in src_to_inner.items() if fold == in_id)
            in_train = sorted(sid for sid, fold in src_to_inner.items() if fold != in_id)
            assert_disjoint_sources(
                set(in_train),
                set(in_eval),
                context=f"outer_{f_id}_inner_{in_id}",
            )
            in_train_recs = [r for sid in in_train for r in by_src[sid]]
            validate_fold_class_support(
                in_train_recs,
                fold_name=f"outer_{f_id}_inner_{in_id}_train",
                min_examples_per_class=3,
                min_sources_per_class=2,
            )
            in_eval_recs = [r for sid in in_eval for r in by_src[sid]]
            inner_splits.append(
                {
                    "inner_fold": in_id,
                    "train_sources": in_train,
                    "eval_sources": in_eval,
                    "train_support": summarize_source_class_support(in_train_recs),
                    "eval_support": summarize_source_class_support(in_eval_recs),
                }
            )

        outer_folds_meta.append(
            {
                "outer_fold": f_id,
                "train_sources": train_sources,
                "eval_sources": eval_sources,
                "train_support": summarize_source_class_support(train_recs),
                "eval_support": eval_support,
                "source_to_inner_fold": src_to_inner,
                "inner_folds": inner_splits,
            }
        )

    return {
        "n_outer_folds": n_outer_folds,
        "n_inner_folds": n_inner_folds,
        "seed": seed,
        "total_sources": len(by_src),
        "source_to_outer_fold": source_to_outer,
        "outer_folds": outer_folds_meta,
    }


@dataclass
class MulticlassCalibrator:
    """Serializable post-hoc calibrator mapping (N, 3) centered log-probs/logits -> (N, 3) probabilities."""

    method: str
    temperatures: List[float]  # len 1 (scalar/prior_bias) or len 3 (vector)
    bias: List[float]          # len 3 (zero for C0/C1)
    l2_reg: float = 0.0
    fit_summary: Dict[str, Any] = field(default_factory=dict)

    def transform_logits(self, logits: np.ndarray) -> np.ndarray:
        arr = np.asarray(logits, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
            squeeze = True
        else:
            squeeze = False

        if self.method == "C0_identity":
            out = _stable_softmax(arr)
        elif self.method in ("C1_temp_scaling", "C2_prior_bias_temp"):
            temp = max(float(self.temperatures[0]), 1e-4)
            bias_vec = np.asarray(self.bias, dtype=np.float64)[np.newaxis, :]
            out = _stable_softmax(arr / temp + bias_vec)
        elif self.method == "C3_vector_scaling":
            scale_vec = 1.0 / np.maximum(np.asarray(self.temperatures, dtype=np.float64), 1e-4)
            bias_vec = np.asarray(self.bias, dtype=np.float64)[np.newaxis, :]
            out = _stable_softmax(arr * scale_vec[np.newaxis, :] + bias_vec)
        else:
            raise ValueError(f"Unknown multiclass calibration method: {self.method!r}")

        return out[0] if squeeze else out

    def predict_probs(self, probs_or_logits: np.ndarray, are_logits: bool = False) -> np.ndarray:
        logits = (
            np.asarray(probs_or_logits, dtype=np.float64)
            if are_logits
            else reconstruct_logits_from_probs(probs_or_logits)
        )
        return self.transform_logits(logits)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MulticlassCalibrator":
        return cls(**payload)


@dataclass
class BinaryPlattCalibrator:
    """Serializable 1D regularized logistic (Platt) calibrator on logit(p)."""

    method: str
    scale_a: float
    bias_b: float
    l2_reg: float = 1e-4
    fit_summary: Dict[str, Any] = field(default_factory=dict)

    def predict_probs(self, raw_probs: np.ndarray) -> np.ndarray:
        arr = np.asarray(raw_probs, dtype=np.float64)
        if self.method == "identity":
            return np.clip(arr, 1e-12, 1.0 - 1e-12)
        p_clip = np.clip(arr, 1e-6, 1.0 - 1e-6)
        raw_logit = np.log(p_clip / (1.0 - p_clip))
        return np.clip(_sigmoid(self.scale_a * raw_logit + self.bias_b), 1e-12, 1.0 - 1e-12)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "BinaryPlattCalibrator":
        return cls(**payload)


def fit_multiclass_calibrator(
    logits: np.ndarray,
    y_idx: np.ndarray,
    method: str,
    l2_reg: float = 0.0,
    fold_summary: Optional[Dict[str, Any]] = None,
) -> MulticlassCalibrator:
    """Fit a post-hoc multiclass calibrator strictly on training-fold logits."""
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y_idx, dtype=np.int64)
    if z.ndim != 2 or z.shape[1] != 3:
        raise ValueError(f"Expected logits shape (N, 3), got {z.shape}")
    if len(y) != len(z) or len(y) == 0:
        raise ValueError("Mismatched or empty labels/logits for calibration.")

    present = set(np.unique(y).tolist())
    if present != {0, 1, 2}:
        raise ValueError(
            f"All 3 canonical classes {{0,1,2}} must be present when fitting calibrator; got {sorted(present)}"
        )

    fit_info = dict(fold_summary or {})
    if method == "C0_identity":
        probs0 = _stable_softmax(z)
        nll0 = float(-np.mean(np.log(probs0[np.arange(len(y)), y])))
        fit_info.update({"train_nll": round(nll0, 6), "converged": True})
        return MulticlassCalibrator(
            method="C0_identity",
            temperatures=[1.0],
            bias=[0.0, 0.0, 0.0],
            l2_reg=0.0,
            fit_summary=fit_info,
        )

    if method == "C1_temp_scaling":
        def _loss_t(params: np.ndarray) -> float:
            temp = math.exp(float(params[0]))
            probs = _stable_softmax(z / temp)
            return float(-np.mean(np.log(probs[np.arange(len(y)), y])))

        res = minimize(
            _loss_t,
            x0=np.array([0.0], dtype=np.float64),
            bounds=[(-2.5, 2.5)],
            method="L-BFGS-B",
        )
        if not np.all(np.isfinite(res.x)):
            raise ValueError(f"C1_temp_scaling failed with non-finite parameters: {res.message}")
        temp = float(math.exp(float(res.x[0])))
        fit_info.update({"train_nll": round(float(res.fun), 6), "converged": bool(res.success)})
        return MulticlassCalibrator(
            method="C1_temp_scaling",
            temperatures=[round(temp, 8)],
            bias=[0.0, 0.0, 0.0],
            l2_reg=0.0,
            fit_summary=fit_info,
        )

    if method == "C2_prior_bias_temp":
        def _loss_tb(params: np.ndarray) -> float:
            temp = math.exp(float(params[0]))
            b_free = params[1:3]
            b_vec = np.array([b_free[0], b_free[1], -(b_free[0] + b_free[1])], dtype=np.float64)
            probs = _stable_softmax(z / temp + b_vec[np.newaxis, :])
            nll = -np.mean(np.log(probs[np.arange(len(y)), y]))
            reg = float(l2_reg) * float(np.sum(b_vec * b_vec))
            return float(nll + reg)

        res = minimize(
            _loss_tb,
            x0=np.zeros(3, dtype=np.float64),
            bounds=[(-2.5, 2.5), (-5.0, 5.0), (-5.0, 5.0)],
            method="L-BFGS-B",
        )
        if not np.all(np.isfinite(res.x)):
            raise ValueError(f"C2_prior_bias_temp failed with non-finite parameters: {res.message}")
        temp = float(math.exp(float(res.x[0])))
        b0, b1 = float(res.x[1]), float(res.x[2])
        b_list = [round(b0, 8), round(b1, 8), round(-(b0 + b1), 8)]
        fit_info.update({"train_nll": round(float(res.fun), 6), "converged": bool(res.success)})
        return MulticlassCalibrator(
            method="C2_prior_bias_temp",
            temperatures=[round(temp, 8)],
            bias=b_list,
            l2_reg=float(l2_reg),
            fit_summary=fit_info,
        )

    if method == "C3_vector_scaling":
        def _loss_vec(params: np.ndarray) -> float:
            scales = np.exp(params[0:3])
            b_free = params[3:5]
            b_vec = np.array([b_free[0], b_free[1], -(b_free[0] + b_free[1])], dtype=np.float64)
            probs = _stable_softmax(z * scales[np.newaxis, :] + b_vec[np.newaxis, :])
            nll = -np.mean(np.log(probs[np.arange(len(y)), y]))
            reg = float(l2_reg) * float(np.sum(b_vec * b_vec))
            return float(nll + reg)

        res = minimize(
            _loss_vec,
            x0=np.zeros(5, dtype=np.float64),
            bounds=[(-2.5, 2.5)] * 3 + [(-5.0, 5.0)] * 2,
            method="L-BFGS-B",
        )
        if not np.all(np.isfinite(res.x)):
            raise ValueError(f"C3_vector_scaling failed with non-finite parameters: {res.message}")
        temps = [round(float(1.0 / math.exp(float(v))), 8) for v in res.x[0:3]]
        b0, b1 = float(res.x[3]), float(res.x[4])
        b_list = [round(b0, 8), round(b1, 8), round(-(b0 + b1), 8)]
        fit_info.update({"train_nll": round(float(res.fun), 6), "converged": bool(res.success)})
        return MulticlassCalibrator(
            method="C3_vector_scaling",
            temperatures=temps,
            bias=b_list,
            l2_reg=float(l2_reg),
            fit_summary=fit_info,
        )

    raise ValueError(f"Unsupported multiclass calibration method: {method!r}")


def fit_binary_platt_calibrator(
    raw_probs: np.ndarray,
    y_bin: np.ndarray,
    l2_reg: float = 1e-4,
    fold_summary: Optional[Dict[str, Any]] = None,
) -> BinaryPlattCalibrator:
    """Fit 1D logistic calibration on logit(p) minimizing binary cross-entropy + L2 penalty."""
    p = np.clip(np.asarray(raw_probs, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    y = np.asarray(y_bin, dtype=np.float64)
    if len(np.unique(y)) < 2:
        raise ValueError("Both binary classes {0, 1} must be present for Platt scaling.")
    raw_logit = np.log(p / (1.0 - p))

    def _loss(params: np.ndarray) -> float:
        a, b = float(params[0]), float(params[1])
        pred = _sigmoid(a * raw_logit + b)
        bce = -np.mean(y * np.log(np.clip(pred, 1e-12, 1.0)) + (1.0 - y) * np.log(np.clip(1.0 - pred, 1e-12, 1.0)))
        return float(bce + l2_reg * ((a - 1.0) ** 2 + b * b))

    res = minimize(
        _loss,
        x0=np.array([1.0, 0.0], dtype=np.float64),
        bounds=[(0.05, 10.0), (-6.0, 6.0)],
        method="L-BFGS-B",
    )
    if not np.all(np.isfinite(res.x)):
        raise ValueError(f"Platt scaling failed: {res.message}")
    fit_info = dict(fold_summary or {})
    fit_info.update({"train_bce": round(float(res.fun), 6), "converged": bool(res.success)})
    return BinaryPlattCalibrator(
        method="platt_logit",
        scale_a=round(float(res.x[0]), 8),
        bias_b=round(float(res.x[1]), 8),
        l2_reg=float(l2_reg),
        fit_summary=fit_info,
    )


def compute_binary_calibration_metrics(
    p_pos: np.ndarray,
    y_bin: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Compute binary NLL, Brier score, uniform ECE, and equal-mass ECE."""
    p = np.clip(np.asarray(p_pos, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    y = np.asarray(y_bin, dtype=np.float64)
    n = len(y)
    if n == 0:
        return {"n": 0, "nll": None, "brier": None, "ece_uniform": None, "ece_equal_mass": None}

    nll = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
    brier = float(np.mean((p - y) ** 2))

    # Uniform bins [0, 1]
    ece_unif = 0.0
    mce_unif = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for idx in range(n_bins):
        lo, hi = edges[idx], edges[idx + 1]
        mask = (p >= lo) & (p <= hi if idx == n_bins - 1 else p < hi)
        if np.any(mask):
            gap = abs(float(np.mean(y[mask])) - float(np.mean(p[mask])))
            ece_unif += (float(np.sum(mask)) / n) * gap
            mce_unif = max(mce_unif, gap)

    # Equal-mass quantile bins
    order = np.argsort(p)
    chunks = np.array_split(order, min(n_bins, n))
    ece_eq = 0.0
    for chunk in chunks:
        if len(chunk) > 0:
            gap = abs(float(np.mean(y[chunk])) - float(np.mean(p[chunk])))
            ece_eq += (len(chunk) / n) * gap

    return {
        "n": int(n),
        "positive_rate": round(float(np.mean(y)), 6),
        "mean_pred": round(float(np.mean(p)), 6),
        "nll": round(nll, 6),
        "brier": round(brier, 6),
        "ece_uniform": round(ece_unif, 6),
        "mce_uniform": round(mce_unif, 6),
        "ece_equal_mass": round(ece_eq, 6),
    }


def compute_multiclass_calibration_metrics(
    probs: np.ndarray,
    y_idx: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Compute 3-class NLL, multiclass Brier, Top-1 ECE, Classwise ECE (CW-ECE), and binary hallucination ECE."""
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    p = p / np.sum(p, axis=1, keepdims=True)
    y = np.asarray(y_idx, dtype=np.int64)
    n = len(y)
    if n == 0:
        return {"n": 0, "nll_3class": None}

    one_hot = np.zeros_like(p)
    one_hot[np.arange(n), y] = 1.0

    nll = float(-np.mean(np.log(p[np.arange(n), y])))
    brier_3cls = float(np.mean(np.sum((p - one_hot) ** 2, axis=1)))

    # Top-1 confidence calibration
    top_conf = np.max(p, axis=1)
    top_pred = np.argmax(p, axis=1)
    top_corr = (top_pred == y).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    top1_ece = 0.0
    top1_mce = 0.0
    reliability_bins: List[Dict[str, Any]] = []
    for idx in range(n_bins):
        lo, hi = edges[idx], edges[idx + 1]
        mask = (top_conf >= lo) & (top_conf <= hi if idx == n_bins - 1 else top_conf < hi)
        cnt = int(np.sum(mask))
        if cnt > 0:
            acc_b = float(np.mean(top_corr[mask]))
            conf_b = float(np.mean(top_conf[mask]))
            gap = abs(acc_b - conf_b)
            top1_ece += (cnt / n) * gap
            top1_mce = max(top1_mce, gap)
            reliability_bins.append(
                {
                    "bin": [round(float(lo), 2), round(float(hi), 2)],
                    "count": cnt,
                    "mean_conf": round(conf_b, 4),
                    "accuracy": round(acc_b, 4),
                }
            )

    # Classwise ECE (CW-ECE): mean across all 3 canonical classes of binned calibration error
    per_class_ece: Dict[str, float] = {}
    for cls_idx, cls_name in enumerate(CALIBRATION_CLASSES):
        p_c = p[:, cls_idx]
        y_c = (y == cls_idx).astype(np.float64)
        cls_ece = 0.0
        for idx in range(n_bins):
            lo, hi = edges[idx], edges[idx + 1]
            mask = (p_c >= lo) & (p_c <= hi if idx == n_bins - 1 else p_c < hi)
            if np.any(mask):
                gap = abs(float(np.mean(y_c[mask])) - float(np.mean(p_c[mask])))
                cls_ece += (float(np.sum(mask)) / n) * gap
        per_class_ece[cls_name] = round(cls_ece, 6)

    cw_ece = float(np.mean(list(per_class_ece.values())))
    emp_priors = {
        cls_name: round(float(np.mean(y == cls_idx)), 6)
        for cls_idx, cls_name in enumerate(CALIBRATION_CLASSES)
    }
    mean_pred = {
        cls_name: round(float(np.mean(p[:, cls_idx])), 6)
        for cls_idx, cls_name in enumerate(CALIBRATION_CLASSES)
    }

    # Binary hallucination score = 1 - P(SUPPORTED_PROXY) vs y_bin = (y != SUPPORTED_PROXY)
    p_hall = np.clip(1.0 - p[:, 1], 1e-12, 1.0 - 1e-12)
    y_hall = (y != 1).astype(np.float64)
    bin_cal = compute_binary_calibration_metrics(p_hall, y_hall, n_bins=n_bins)

    return {
        "n": int(n),
        "class_counts": {
            cls_name: int(np.sum(y == cls_idx))
            for cls_idx, cls_name in enumerate(CALIBRATION_CLASSES)
        },
        "nll_3class": round(nll, 6),
        "brier_3class": round(brier_3cls, 6),
        "top1_ece": round(top1_ece, 6),
        "top1_mce": round(top1_mce, 6),
        "cw_ece": round(cw_ece, 6),
        "per_class_ece": per_class_ece,
        "empirical_priors": emp_priors,
        "mean_predicted_probs": mean_pred,
        "argmax_accuracy": round(float(np.mean(top_corr)), 6),
        "binary_hallucination": bin_cal,
        "top1_reliability_bins": reliability_bins,
    }

