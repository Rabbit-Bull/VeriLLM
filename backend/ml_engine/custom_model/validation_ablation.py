"""
VeriLLM Phase 8 Step 4: Validation Ablation Study Utilities.

Provides strict validation-only dataset loading, train-validation duplicate
auditing, deterministic source-grouped pilot selection, dual-regime (Direct
Context vs. Chunked Retrieval) evaluation runners, non-averaging comparison arms,
explicit UNVERIFIABLE selective/penalized metrics, MIXED_HALLUCINATION
diagnostics, response-level scoring alignment, and paired statistical tests.

STRICT INVARIANTS:
- NEVER read, inspect, or evaluate ragtruth_test.jsonl.
- NEVER modify production verification, decision strategy, scoring, or training files.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ml_engine.claim_extractor.models import ExtractedClaim
from ml_engine.claim_verifier.decision_strategy import VerificationDecisionStrategy
from ml_engine.claim_verifier.models import VerificationCategory, VerifiedClaim
from ml_engine.evidence_retriever.chunker import SlidingWindowChunker
from ml_engine.evidence_retriever.models import (
    ClaimEvidenceMatch,
    DocumentChunk,
    RetrievedPassage,
)
from ml_engine.scorer.response_scorer import ResponseScorer

CANONICAL_LABELS: Tuple[str, ...] = ("CONTRADICTED", "SUPPORTED_PROXY", "UNSUPPORTED")
EXTENDED_PRED_LABELS: Tuple[str, ...] = (
    "CONTRADICTED",
    "SUPPORTED_PROXY",
    "UNSUPPORTED",
    "UNVERIFIABLE",
)
MIXED_LABEL = "MIXED_HALLUCINATION"
TASK_TYPES: Tuple[str, ...] = ("Summary", "QA", "Data2txt")
MAX_CONTEXT_TOKENS = 512
MIN_PILOT_CONTRADICTED = 25
MIN_PILOT_UNSUPPORTED = 40
MIN_PILOT_CONTRADICTED_PER_TASK = 3
MIN_PILOT_UNSUPPORTED_PER_TASK = 5

_CATEGORY_TO_RAGTRUTH: Dict[str, str] = {
    VerificationCategory.SUPPORTED.value: "SUPPORTED_PROXY",
    VerificationCategory.CONTRADICTED.value: "CONTRADICTED",
    VerificationCategory.UNSUPPORTED.value: "UNSUPPORTED",
    VerificationCategory.UNVERIFIABLE.value: "UNVERIFIABLE",
    "SUPPORTED": "SUPPORTED_PROXY",
    "SUPPORTED_PROXY": "SUPPORTED_PROXY",
    "CONTRADICTED": "CONTRADICTED",
    "UNSUPPORTED": "UNSUPPORTED",
    "UNVERIFIABLE": "UNVERIFIABLE",
}

_RAGTRUTH_TO_CATEGORY: Dict[str, VerificationCategory] = {
    "SUPPORTED_PROXY": VerificationCategory.SUPPORTED,
    "SUPPORTED": VerificationCategory.SUPPORTED,
    "CONTRADICTED": VerificationCategory.CONTRADICTED,
    "UNSUPPORTED": VerificationCategory.UNSUPPORTED,
    "UNVERIFIABLE": VerificationCategory.UNVERIFIABLE,
}

_FORBIDDEN_TEST_PATH_RE = re.compile(r"(^|[\\/._-])test([\\/._-]|$)", re.IGNORECASE)

ARM_NAMES: Tuple[str, ...] = (
    "arm0a_nli_only",
    "arm0b_minicheck_only",
    "arm1_baseline",
    "arm2_custom_deberta_only",
    "arm3a_mc_plus_custom_deberta",
    "arm3b_baseline_support_deberta_disambig",
    "arm3c_selective_agreement_gate",
)

ARM_DESCRIPTIONS: Dict[str, str] = {
    "arm0a_nli_only": (
        "Single-scorer NLI cross-encoder (cross-encoder/nli-deberta-v3-xsmall); "
        "argmax 3-class prediction and continuous score (1 - p_entail)."
    ),
    "arm0b_minicheck_only": (
        "Single-scorer MiniCheck (roberta-large); binary cutoff (>=0.50 SUPPORTED_PROXY vs UNSUPPORTED) "
        "and continuous score (1 - p_support)."
    ),
    "arm1_baseline": (
        "Production discrete rule engine (VerificationDecisionStrategy) over MiniCheck + NLI-xsmall "
        "(plus retrieval similarity floor & Rule 4 secondary NLI conflicts in Retrieval); no unified continuous score."
    ),
    "arm2_custom_deberta_only": (
        "Single-scorer fine-tuned DeBERTa argmax over uncalibrated softmax logits from class-weighted training "
        "(not calibrated truth probabilities); continuous score (1 - raw_softmax_support)."
    ),
    "arm3a_mc_plus_custom_deberta": (
        "Experimental discrete threshold gate plugging uncalibrated class-weighted DeBERTa logits into fixed NLI "
        "thresholds (0.40/0.50/0.80) alongside MiniCheck; in Retrieval also retains NLI-xsmall "
        "(cross-encoder/nli-deberta-v3-xsmall) for Rule 4 secondary passage conflicts. "
        "Not validated for production decisions; no unified continuous score."
    ),
    "arm3b_baseline_support_deberta_disambig": (
        "Discrete rule gate: keeps arm1_baseline SUPPORTED_PROXY/UNVERIFIABLE decisions and uses DeBERTa argmax "
        "to disambiguate CONTRADICTED vs UNSUPPORTED; no unified continuous score."
    ),
    "arm3c_selective_agreement_gate": (
        "Discrete selective agreement gate: abstains to UNVERIFIABLE when arm1_baseline and DeBERTa disagree on "
        "supported vs hallucinated; uses DeBERTa error type when both agree hallucinated; no unified continuous score."
    ),
}


def map_category_to_ragtruth(label: str) -> str:
    """Normalize VerificationCategory or string label to validation ablation schema."""
    if isinstance(label, VerificationCategory):
        label = label.value
    if label not in _CATEGORY_TO_RAGTRUTH:
        raise ValueError(f"Unsupported verification label: {label!r}")
    return _CATEGORY_TO_RAGTRUTH[label]


def assert_validation_only_path(path: Path) -> Path:
    """Refuse any path that points to a test split artifact or missing file."""
    resolved = Path(path).resolve()
    name_lower = resolved.name.lower()
    if "ragtruth_test" in name_lower or _FORBIDDEN_TEST_PATH_RE.search(name_lower):
        raise ValueError(
            f"Refusing to access forbidden test-like dataset path: {resolved}"
        )
    if not resolved.exists():
        raise FileNotFoundError(f"Validation file not found: {resolved}")
    return resolved


def assert_train_only_path(path: Path) -> Path:
    """Refuse any path that points to a test split artifact when indexing train overlap."""
    resolved = Path(path).resolve()
    name_lower = resolved.name.lower()
    if "ragtruth_test" in name_lower or _FORBIDDEN_TEST_PATH_RE.search(name_lower):
        raise ValueError(
            f"Refusing to access forbidden test-like dataset path: {resolved}"
        )
    if not resolved.exists():
        raise FileNotFoundError(f"Train file not found: {resolved}")
    return resolved


def compute_context_hash(context_text: str) -> str:
    """Compute a collision-resistant SHA-256 hash of exact context_text."""
    return hashlib.sha256(context_text.encode("utf-8")).hexdigest()


def compute_pair_key(context_text: str, sentence_text: str) -> str:
    """Compute a length-delimited SHA-256 key for exact (context_text, sentence_text) pairs."""
    ctx_bytes = context_text.encode("utf-8")
    sent_bytes = sentence_text.encode("utf-8")
    hasher = hashlib.sha256()
    hasher.update(f"{len(ctx_bytes)}:".encode("ascii"))
    hasher.update(ctx_bytes)
    hasher.update(f"|{len(sent_bytes)}:".encode("ascii"))
    hasher.update(sent_bytes)
    return hasher.hexdigest()


def load_validation_records(val_path: Path) -> List[Dict[str, Any]]:
    """
    Load processed RAGTruth validation JSONL records with strict split enforcement.

    Raises ValueError if any record does not declare split == "val".
    """
    safe_path = assert_validation_only_path(val_path)
    records: List[Dict[str, Any]] = []
    with safe_path.open("r", encoding="utf-8") as handle:
        for line_num, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            item = json.loads(stripped)
            if item.get("split") != "val":
                raise ValueError(
                    f"Split isolation violation at {safe_path}:{line_num}: "
                    f"expected split='val', got {item.get('split')!r}"
                )
            records.append(item)
    if not records:
        raise ValueError(f"Validation dataset is empty: {safe_path}")
    return records


def build_train_overlap_index(train_path: Path) -> Dict[str, Any]:
    """
    Index source_ids, exact context hashes, exact (context, sentence) pair keys,
    and exact sentence strings from ragtruth_train.jsonl ONLY (never touches test).
    """
    safe_train = assert_train_only_path(train_path)
    source_ids: Set[str] = set()
    context_hashes: Set[str] = set()
    exact_pair_keys: Set[str] = set()
    sentence_strings: Set[str] = set()
    total_train_records = 0

    with safe_train.open("r", encoding="utf-8") as handle:
        for line_num, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            item = json.loads(stripped)
            if item.get("split") != "train":
                raise ValueError(
                    f"Split isolation violation at {safe_train}:{line_num}: "
                    f"expected split='train', got {item.get('split')!r}"
                )
            total_train_records += 1
            source_ids.add(str(item["source_id"]))
            ctx = item.get("context_text", "")
            sent = item.get("sentence_text", "")
            context_hashes.add(compute_context_hash(ctx))
            exact_pair_keys.add(compute_pair_key(ctx, sent))
            sentence_strings.add(sent)

    return {
        "train_path": str(safe_train),
        "total_train_records": total_train_records,
        "source_ids": source_ids,
        "context_hashes": context_hashes,
        "exact_pair_keys": exact_pair_keys,
        "sentence_strings": sentence_strings,
    }


def audit_duplicate_and_leakage(
    records: Sequence[Dict[str, Any]],
    train_index: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Audit source disjointness and separate exact (context, sentence) pair duplicates
    against train from repeated sentence strings.
    """
    val_source_ids = {str(r["source_id"]) for r in records}
    pair_key_counts = Counter(
        compute_pair_key(r.get("context_text", ""), r.get("sentence_text", ""))
        for r in records
    )
    sentence_counts = Counter(r.get("sentence_text", "") for r in records)

    within_val_extra_pair_dups = sum(max(0, c - 1) for c in pair_key_counts.values())
    within_val_repeated_sentence_extra = sum(
        max(0, c - 1) for c in sentence_counts.values()
    )

    if train_index is None:
        return {
            "train_index_checked": False,
            "source_id_overlap_count": None,
            "exact_context_match_with_train_count": None,
            "exact_pair_duplicate_with_train_count": 0,
            "exact_pair_duplicate_example_ids": [],
            "sentence_string_in_train_total": None,
            "sentence_string_only_overlap_count": None,
            "within_sample_extra_pair_duplicates": within_val_extra_pair_dups,
            "within_sample_repeated_sentence_extra": within_val_repeated_sentence_extra,
        }

    source_overlap = sorted(val_source_ids.intersection(train_index["source_ids"]))
    if source_overlap:
        raise ValueError(
            f"Source leakage detected between train and validation: {source_overlap[:10]}"
        )

    exact_ctx_ids: List[str] = []
    exact_pair_ids: List[str] = []
    sent_in_train_ids: List[str] = []
    sent_only_ids: List[str] = []

    for r in records:
        ex_id = str(r["example_id"])
        ctx = r.get("context_text", "")
        sent = r.get("sentence_text", "")
        in_ctx = compute_context_hash(ctx) in train_index["context_hashes"]
        in_pair = compute_pair_key(ctx, sent) in train_index["exact_pair_keys"]
        in_sent = sent in train_index["sentence_strings"]

        if in_ctx:
            exact_ctx_ids.append(ex_id)
        if in_pair:
            exact_pair_ids.append(ex_id)
        if in_sent:
            sent_in_train_ids.append(ex_id)
            if not in_pair:
                sent_only_ids.append(ex_id)

    return {
        "train_index_checked": True,
        "source_id_overlap_count": len(source_overlap),
        "exact_context_match_with_train_count": len(exact_ctx_ids),
        "exact_context_match_source_ids": sorted(
            {str(r["source_id"]) for r in records if r["example_id"] in set(exact_ctx_ids)}
        ),
        "exact_pair_duplicate_with_train_count": len(exact_pair_ids),
        "exact_pair_duplicate_example_ids": exact_pair_ids,
        "sentence_string_in_train_total": len(sent_in_train_ids),
        "sentence_string_only_overlap_count": len(sent_only_ids),
        "within_sample_extra_pair_duplicates": within_val_extra_pair_dups,
        "within_sample_repeated_sentence_extra": within_val_repeated_sentence_extra,
    }


def _assign_source_stratum(counts: Dict[str, int]) -> str:
    """Categorize a source by presence of minority hallucination classes."""
    has_contra = counts.get("CONTRADICTED", 0) > 0
    has_unsup = counts.get("UNSUPPORTED", 0) > 0
    if has_contra and has_unsup:
        return "A_CONTRA_AND_UNSUP"
    if has_contra:
        return "B_CONTRA_ONLY"
    if has_unsup:
        return "C_UNSUP_ONLY"
    return "D_SUPPORTED_OR_MIXED"


def summarize_sources(
    val_records: Sequence[Dict[str, Any]],
    train_index: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Aggregate sentence, response, label, and duplicate counts per validation source_id."""
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in val_records:
        grouped[str(rec["source_id"])].append(rec)

    summaries: Dict[str, Dict[str, Any]] = {}
    for sid, recs in grouped.items():
        task_types = {r["task_type"] for r in recs}
        if len(task_types) != 1:
            raise ValueError(f"Source {sid} spans multiple tasks: {task_types}")
        counts = Counter(r["primary_label"] for r in recs)
        responses = {r["response_id"] for r in recs}
        exact_train_dups = 0
        if train_index is not None:
            exact_train_dups = sum(
                1
                for r in recs
                if compute_pair_key(r.get("context_text", ""), r.get("sentence_text", ""))
                in train_index["exact_pair_keys"]
            )
        label_counts = {
            "SUPPORTED_PROXY": int(counts.get("SUPPORTED_PROXY", 0)),
            "CONTRADICTED": int(counts.get("CONTRADICTED", 0)),
            "UNSUPPORTED": int(counts.get("UNSUPPORTED", 0)),
            MIXED_LABEL: int(counts.get(MIXED_LABEL, 0)),
        }
        summaries[sid] = {
            "source_id": sid,
            "task_type": next(iter(task_types)),
            "num_responses": len(responses),
            "num_sentences": len(recs),
            "counts": label_counts,
            "stratum": _assign_source_stratum(label_counts),
            "exact_train_pair_dups": exact_train_dups,
        }
    return summaries


def evaluate_minority_adequacy(
    records: Sequence[Dict[str, Any]],
    min_total_contradicted: int = MIN_PILOT_CONTRADICTED,
    min_total_unsupported: int = MIN_PILOT_UNSUPPORTED,
    min_contradicted_per_task: int = MIN_PILOT_CONTRADICTED_PER_TASK,
    min_unsupported_per_task: int = MIN_PILOT_UNSUPPORTED_PER_TASK,
) -> Dict[str, Any]:
    """
    Check whether a validation subset contains sufficient minority-class examples
    overall and across Summary, QA, and Data2txt.
    """
    overall = Counter(r["primary_label"] for r in records)
    by_task: Dict[str, Dict[str, int]] = {}
    for task in TASK_TYPES:
        task_recs = [r for r in records if r["task_type"] == task]
        tc = Counter(r["primary_label"] for r in task_recs)
        by_task[task] = {
            "sentences": len(task_recs),
            "SUPPORTED_PROXY": int(tc.get("SUPPORTED_PROXY", 0)),
            "CONTRADICTED": int(tc.get("CONTRADICTED", 0)),
            "UNSUPPORTED": int(tc.get("UNSUPPORTED", 0)),
            MIXED_LABEL: int(tc.get(MIXED_LABEL, 0)),
        }

    failures: List[str] = []
    if overall.get("CONTRADICTED", 0) < min_total_contradicted:
        failures.append(
            f"total CONTRADICTED={overall.get('CONTRADICTED', 0)} < {min_total_contradicted}"
        )
    if overall.get("UNSUPPORTED", 0) < min_total_unsupported:
        failures.append(
            f"total UNSUPPORTED={overall.get('UNSUPPORTED', 0)} < {min_total_unsupported}"
        )
    for task in TASK_TYPES:
        if by_task[task]["CONTRADICTED"] < min_contradicted_per_task:
            failures.append(
                f"{task} CONTRADICTED={by_task[task]['CONTRADICTED']} < {min_contradicted_per_task}"
            )
        if by_task[task]["UNSUPPORTED"] < min_unsupported_per_task:
            failures.append(
                f"{task} UNSUPPORTED={by_task[task]['UNSUPPORTED']} < {min_unsupported_per_task}"
            )

    return {
        "is_adequate": len(failures) == 0,
        "failures": failures,
        "thresholds": {
            "min_total_contradicted": min_total_contradicted,
            "min_total_unsupported": min_total_unsupported,
            "min_contradicted_per_task": min_contradicted_per_task,
            "min_unsupported_per_task": min_unsupported_per_task,
        },
        "counts_by_label": {
            "SUPPORTED_PROXY": int(overall.get("SUPPORTED_PROXY", 0)),
            "CONTRADICTED": int(overall.get("CONTRADICTED", 0)),
            "UNSUPPORTED": int(overall.get("UNSUPPORTED", 0)),
            MIXED_LABEL: int(overall.get(MIXED_LABEL, 0)),
        },
        "counts_by_task_and_label": by_task,
    }


def _natural_sort_key(sid: str) -> Tuple[int, Any]:
    return (0, int(sid)) if sid.isdigit() else (1, sid)


def select_pilot_sources(
    val_records: Sequence[Dict[str, Any]],
    train_index: Optional[Dict[str, Any]] = None,
    sources_per_task: int = 5,
    seed: int = 42,
    min_total_contradicted: int = MIN_PILOT_CONTRADICTED,
    min_total_unsupported: int = MIN_PILOT_UNSUPPORTED,
    min_contradicted_per_task: int = MIN_PILOT_CONTRADICTED_PER_TASK,
    min_unsupported_per_task: int = MIN_PILOT_UNSUPPORTED_PER_TASK,
    locked_source_ids: Optional[Iterable[str]] = None,
    locked_manifest_path: Optional[str] = None,
    locked_selection_method: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Deterministically select a source-grouped validation pilot balanced across tasks.

    Stage 1 tries a uniform task-stratified sample (seed=42).
    If minority-class thresholds fail, Stage 2 deterministically allocates sources from
    minority-bearing source strata per task (iterating min contradicted sources per task)
    while preserving complete sources and complete responses.
    When locked_source_ids are provided from an earlier manifest, provenance reflects the
    two-phase locked-prefix plus follow-up sampling procedure.
    """
    source_summaries = summarize_sources(val_records, train_index=train_index)
    locked_list = [str(s) for s in (locked_source_ids or [])]
    locked_set = set(locked_list)
    unknown_locked = locked_set - set(source_summaries.keys())
    if unknown_locked:
        raise ValueError(f"Unknown locked_source_ids not in val: {sorted(unknown_locked)}")

    recs_by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in val_records:
        recs_by_source[str(rec["source_id"])].append(rec)

    sources_by_task: Dict[str, List[str]] = {}
    for task in TASK_TYPES:
        candidates = sorted(
            (sid for sid, meta in source_summaries.items() if meta["task_type"] == task),
            key=_natural_sort_key,
        )
        if len(candidates) < sources_per_task:
            raise ValueError(
                f"Task {task} has {len(candidates)} validation sources, "
                f"cannot select {sources_per_task}."
            )
        sources_by_task[task] = candidates

    # Stage 1: Uniform task-stratified deterministic sample
    rng_stage1 = random.Random(seed)
    stage1_selected: List[str] = []
    for task in TASK_TYPES:
        locked_for_task = sorted(
            (s for s in sources_by_task[task] if s in locked_set),
            key=_natural_sort_key,
        )
        if len(locked_for_task) > sources_per_task:
            raise ValueError(
                f"Locked sources for {task} ({len(locked_for_task)}) exceed "
                f"sources_per_task={sources_per_task}."
            )
        remaining_pool = [s for s in sources_by_task[task] if s not in locked_set]
        shuffled_pool = list(remaining_pool)
        rng_stage1.shuffle(shuffled_pool)
        need = sources_per_task - len(locked_for_task)
        stage1_selected.extend(locked_for_task + shuffled_pool[:need])

    stage1_records = [r for sid in stage1_selected for r in recs_by_source[sid]]
    stage1_adequacy = evaluate_minority_adequacy(
        stage1_records,
        min_total_contradicted=min_total_contradicted,
        min_total_unsupported=min_total_unsupported,
        min_contradicted_per_task=min_contradicted_per_task,
        min_unsupported_per_task=min_unsupported_per_task,
    )

    final_selected = stage1_selected
    final_records = stage1_records
    final_adequacy = stage1_adequacy
    if locked_set:
        selection_method = "two_phase_locked_prefix_plus_uniform_followup"
        prefix_desc = (
            f"prior manifest ({locked_selection_method})"
            if locked_selection_method
            else "locked prior manifest"
        )
        selection_rationale = (
            f"Two-phase selection: preserved {len(locked_set)} locked source(s) from {prefix_desc} "
            f"and sampled remaining sources uniformly without replacement per task (seed={seed}, "
            f"{sources_per_task} sources/task total). Combined set satisfied all minority adequacy checks."
        )
    else:
        selection_method = "stage1_uniform_task_stratified"
        selection_rationale = (
            f"Uniform task-stratified selection (seed={seed}, {sources_per_task} sources/task) "
            "satisfied all minority-class adequacy thresholds."
        )
    stage2_contra_quota_used: Optional[int] = None

    if not stage1_adequacy["is_adequate"]:
        # Stage 2: Deterministic Two-Stratum Allocation per task (no model peeking).
        # Pre-shuffle CONTRA_PRESENT vs NO_CONTRA pools deterministically once with seed
        # so neither A-vs-B nor C-vs-D is artificially favored and clean D sources stay represented.
        rng_stage2 = random.Random(seed)
        contra_pools: Dict[str, List[str]] = {}
        non_contra_pools: Dict[str, List[str]] = {}
        for task in TASK_TYPES:
            c_pool: List[str] = []
            nc_pool: List[str] = []
            for sid in sources_by_task[task]:
                if sid in locked_set:
                    continue
                if source_summaries[sid]["counts"]["CONTRADICTED"] > 0:
                    c_pool.append(sid)
                else:
                    nc_pool.append(sid)
            rng_stage2.shuffle(c_pool)
            rng_stage2.shuffle(nc_pool)
            contra_pools[task] = c_pool
            non_contra_pools[task] = nc_pool

        candidate_selected: Optional[List[str]] = None
        candidate_adequacy: Optional[Dict[str, Any]] = None
        for k_contra in range(1, sources_per_task + 1):
            trial_selected: List[str] = []
            for task in TASK_TYPES:
                locked_for_task = sorted(
                    (s for s in sources_by_task[task] if s in locked_set),
                    key=_natural_sort_key,
                )
                need = sources_per_task - len(locked_for_task)
                already_contra = sum(
                    1
                    for s in locked_for_task
                    if source_summaries[s]["counts"]["CONTRADICTED"] > 0
                )
                contra_need = max(0, k_contra - already_contra)
                picked_contra = contra_pools[task][: min(contra_need, need)]
                rem_need = need - len(picked_contra)
                picked_set = set(locked_for_task) | set(picked_contra)
                fallback_extra = [
                    s for s in contra_pools[task][len(picked_contra) :] if s not in picked_set
                ]
                rem_candidates = [
                    s for s in non_contra_pools[task] + fallback_extra if s not in picked_set
                ]
                trial_selected.extend(locked_for_task + picked_contra + rem_candidates[:rem_need])

            trial_records = [r for sid in trial_selected for r in recs_by_source[sid]]
            trial_adequacy = evaluate_minority_adequacy(
                trial_records,
                min_total_contradicted=min_total_contradicted,
                min_total_unsupported=min_total_unsupported,
                min_contradicted_per_task=min_contradicted_per_task,
                min_unsupported_per_task=min_unsupported_per_task,
            )
            candidate_selected = trial_selected
            candidate_adequacy = trial_adequacy
            stage2_contra_quota_used = k_contra
            if trial_adequacy["is_adequate"]:
                break

        if candidate_selected is not None and candidate_adequacy is not None:
            final_selected = candidate_selected
            final_records = [r for sid in final_selected for r in recs_by_source[sid]]
            final_adequacy = candidate_adequacy
            selection_method = "stage2_minority_stratified_source_allocation"
            selection_rationale = (
                f"Stage 1 uniform selection failed minority adequacy checks "
                f"({'; '.join(stage1_adequacy['failures'])}). Revised deterministically "
                f"(seed={seed}) by reserving k_contra={stage2_contra_quota_used} "
                "random source(s) per task from the CONTRA_PRESENT stratum and filling "
                "remaining slots uniformly from NO_CONTRA sources while keeping all "
                "responses and sentences per source intact."
            )

    def _design_stratum(meta_obj: Dict[str, Any]) -> str:
        return "CONTRA_PRESENT" if meta_obj["counts"]["CONTRADICTED"] > 0 else "NO_CONTRA"

    # Compute full-val design-stratum counts and exact Horvitz-Thompson sample weights
    val_stratum_counts = Counter(
        (m["task_type"], _design_stratum(m)) for m in source_summaries.values()
    )
    sample_stratum_counts = Counter(
        (source_summaries[sid]["task_type"], _design_stratum(source_summaries[sid]))
        for sid in final_selected
    )
    val_total_sources = len(source_summaries)
    sample_total_sources = len(final_selected)

    selected_sources_meta: List[Dict[str, Any]] = []
    for sid in final_selected:
        meta = dict(source_summaries[sid])
        d_strat = _design_stratum(meta)
        key = (meta["task_type"], d_strat)
        n_pop = val_stratum_counts[key]
        n_samp = sample_stratum_counts[key]
        # Horvitz-Thompson relative weight normalized so mean source weight ~= 1.0
        rel_weight = (n_pop / val_total_sources) / (n_samp / sample_total_sources)
        meta["design_stratum"] = d_strat
        meta["stratum_population_sources"] = int(n_pop)
        meta["stratum_sample_sources"] = int(n_samp)
        meta["relative_stratum_weight"] = round(float(rel_weight), 6)
        selected_sources_meta.append(meta)

    full_val_adequacy = evaluate_minority_adequacy(
        val_records,
        min_total_contradicted=0,
        min_total_unsupported=0,
        min_contradicted_per_task=0,
        min_unsupported_per_task=0,
    )
    dup_audit = audit_duplicate_and_leakage(final_records, train_index=train_index)
    min_stratum_sample = min(sample_stratum_counts.values()) if sample_stratum_counts else 0
    small_strata = sorted(
        f"{t}/{st} (n_samp={cnt}, n_pop={val_stratum_counts[(t, st)]})"
        for (t, st), cnt in sample_stratum_counts.items()
        if cnt < 5
    )

    return {
        "seed": seed,
        "sources_per_task": sources_per_task,
        "total_sources": len(final_selected),
        "total_responses": len({r["response_id"] for r in final_records}),
        "total_sentences": len(final_records),
        "selection_method": selection_method,
        "selection_rationale": selection_rationale,
        "locked_manifest_path": locked_manifest_path,
        "locked_selection_method": locked_selection_method,
        "locked_source_ids": locked_list,
        "stage2_contra_quota_per_task": stage2_contra_quota_used,
        "stratum_sampling_summary": {
            "min_sample_sources_in_any_stratum": int(min_stratum_sample),
            "small_strata_below_5": small_strata,
            "stratum_counts": [
                {
                    "task_type": t,
                    "design_stratum": st,
                    "population_sources": int(val_stratum_counts[(t, st)]),
                    "sample_sources": int(sample_stratum_counts[(t, st)]),
                    "relative_stratum_weight": round(
                        (val_stratum_counts[(t, st)] / val_total_sources)
                        / (sample_stratum_counts[(t, st)] / sample_total_sources),
                        6,
                    ),
                }
                for (t, st) in sorted(sample_stratum_counts.keys())
            ],
        },
        "stage1_uniform_attempt": {
            "source_ids": stage1_selected,
            "total_sentences": len(stage1_records),
            "adequacy": stage1_adequacy,
        },
        "final_adequacy": final_adequacy,
        "full_validation_distribution": {
            "total_sources": len(source_summaries),
            "total_sentences": len(val_records),
            "counts_by_label": full_val_adequacy["counts_by_label"],
        },
        "duplicate_and_leakage_audit": dup_audit,
        "selected_source_ids": final_selected,
        "selected_sources": selected_sources_meta,
    }


def evaluate_direct_context_records(
    records: Sequence[Dict[str, Any]],
    minicheck_verifier: Any,
    nli_verifier: Any,
    custom_deberta_verifier: Any,
    token_length_fn: Optional[Callable[[Sequence[Tuple[str, str]]], List[int]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate records under Condition 1 (Direct Source Context) in batch.

    Returns mapping example_id -> raw signal dict for 'direct' regime.
    """
    if not records:
        return {}

    docs: List[str] = []
    claims: List[str] = []
    pairs: List[Tuple[str, str]] = []
    non_empty_indices: List[int] = []
    deberta_pairs: List[Tuple[str, str]] = []

    for idx, rec in enumerate(records):
        ctx = str(rec.get("context_text", ""))
        sent = str(rec.get("sentence_text", ""))
        docs.append(ctx)
        claims.append(sent)
        pairs.append((ctx, sent))
        if ctx.strip() and sent.strip():
            non_empty_indices.append(idx)
            deberta_pairs.append((ctx, sent))

    mc_scores = minicheck_verifier.score_pairs(docs, claims)
    nli_scores = nli_verifier.predict_pairs(pairs)
    deberta_preds_map: Dict[int, Any] = {}
    if deberta_pairs:
        deberta_preds = custom_deberta_verifier.predict_pairs(deberta_pairs)
        for orig_idx, pred in zip(non_empty_indices, deberta_preds):
            deberta_preds_map[orig_idx] = pred

    token_lengths: Optional[List[int]] = None
    if token_length_fn is not None:
        token_lengths = token_length_fn(pairs)

    out: Dict[str, Dict[str, Any]] = {}
    for idx, rec in enumerate(records):
        ex_id = str(rec["example_id"])
        mc_label, mc_prob = mc_scores[idx]
        nli_e, nli_c, nli_n = nli_scores[idx]
        deb = deberta_preds_map.get(idx)
        tok_len = int(token_lengths[idx]) if token_lengths is not None else None
        out[ex_id] = {
            "mc_label": int(mc_label),
            "mc_prob": float(mc_prob),
            "nli_entail": float(nli_e),
            "nli_contra": float(nli_c),
            "nli_neutral": float(nli_n),
            "deberta_supported_prob": float(deb.supported_proxy_prob) if deb else None,
            "deberta_contradiction_prob": float(deb.contradicted_prob) if deb else None,
            "deberta_unsupported_prob": float(deb.unsupported_prob) if deb else None,
            "deberta_predicted_label": str(deb.predicted_label) if deb else None,
            "pair_token_count": tok_len,
            "exceeds_512_tokens": (tok_len > MAX_CONTEXT_TOKENS) if tok_len is not None else None,
        }
    return out


def _build_extracted_claim_for_record(rec: Dict[str, Any]) -> ExtractedClaim:
    sent = str(rec["sentence_text"])
    return ExtractedClaim(
        claim_id=str(rec["example_id"]),
        text=sent,
        source_sentence=sent,
        start_char=int(rec.get("sentence_start", 0) or 0),
        end_char=int(rec.get("sentence_end", len(sent)) or len(sent)),
        claim_start_char=None,
        claim_end_char=None,
        offset_type="EXACT",
        is_factual=True,
    )


def evaluate_retrieval_pipeline_for_source(
    source_records: Sequence[Dict[str, Any]],
    chunker: SlidingWindowChunker,
    passage_retriever: Any,
    claim_verifier: Any,
    token_length_fn: Optional[Callable[[Sequence[Tuple[str, str]]], List[int]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate all validation sentences sharing a single source_id through Condition 2
    (SlidingWindowChunker -> PassageRetriever -> ClaimVerifier(use_custom_model=True)).
    """
    if not source_records:
        return {}

    source_id = str(source_records[0]["source_id"])
    context_text = str(source_records[0]["context_text"])
    doc_name = f"val_source_{source_id}.txt"

    chunks = chunker.chunk_plain_text(
        context_text,
        document_name=doc_name,
        page_number=1,
    )
    if not chunks and context_text.strip():
        chunks = [
            DocumentChunk(
                chunk_id=f"{doc_name}_p01_c000",
                document_name=doc_name,
                page_number=1,
                chunk_index=0,
                text=context_text.strip(),
                start_char=0,
                end_char=len(context_text.strip()),
                word_count=len(context_text.strip().split()),
            )
        ]

    claims = [_build_extracted_claim_for_record(rec) for rec in source_records]

    if chunks:
        passage_retriever.index_chunks(chunks)
        evidence_matches = passage_retriever.retrieve_for_claims(claims)
    else:
        evidence_matches = [
            ClaimEvidenceMatch(
                claim_id=c.claim_id,
                claim_text=c.text,
                retrieved_passages=[],
                best_passage=None,
                best_similarity_score=0.0,
            )
            for c in claims
        ]

    report = claim_verifier.verify_claims(
        claims=claims,
        evidence_matches=evidence_matches,
        document_name=doc_name,
    )

    match_by_id = {m.claim_id: m for m in evidence_matches}
    sec_nli_by_id: Dict[str, List[Tuple[float, float, float]]] = {}
    # Capture secondary NLI conflict tuples cleanly so candidate Arm 3a mirrors production Rule 4
    for m, vc in zip(evidence_matches, report.claims):
        rel_passages = [
            p
            for p in m.retrieved_passages[:3]
            if p.chunk.text.strip() and p.similarity_score >= 0.45
        ]
        if len(rel_passages) >= 2 and getattr(claim_verifier, "nli", None) is not None:
            sec_pairs = [(p.chunk.text, vc.claim_text) for p in rel_passages]
            sec_scores = claim_verifier.nli.predict_pairs(sec_pairs)
            sec_nli_by_id[vc.claim_id] = [
                (float(e), float(c), float(n)) for e, c, n in sec_scores
            ]

    top1_pairs = [
        (
            vc.primary_evidence.chunk.text if vc.primary_evidence else "",
            vc.claim_text,
        )
        for vc in report.claims
    ]
    top1_tokens: Optional[List[int]] = None
    if token_length_fn is not None and top1_pairs:
        top1_tokens = token_length_fn(top1_pairs)

    out: Dict[str, Dict[str, Any]] = {}
    for idx, vc in enumerate(report.claims):
        match = match_by_id.get(vc.claim_id)
        best_sim = float(match.best_similarity_score) if match else 0.0
        best_text = vc.primary_evidence.chunk.text if vc.primary_evidence else ""
        sec_scores = sec_nli_by_id.get(vc.claim_id, [])
        max_sec_e = max((s[0] for s in sec_scores), default=None)
        max_sec_c = max((s[1] for s in sec_scores), default=None)
        conflict_thresh = getattr(
            getattr(claim_verifier, "decision_strategy", None),
            "conflict_threshold",
            0.70,
        )
        conflict_triggered = (
            max_sec_e is not None
            and max_sec_c is not None
            and max_sec_e > conflict_thresh
            and max_sec_c > conflict_thresh
        )
        mc_prob = vc.scores.minicheck_probability
        mc_label = 1 if (mc_prob is not None and mc_prob >= 0.50) else 0

        out[vc.claim_id] = {
            "num_chunks": len(chunks),
            "best_similarity_score": best_sim,
            "retrieved_chunk_text_word_count": len(best_text.split()) if best_text else 0,
            "retrieved_chunk_preview": best_text[:200] if best_text else "",
            "relevant_passages_count": sum(
                1
                for p in (match.retrieved_passages[:3] if match else [])
                if p.chunk.text.strip() and p.similarity_score >= 0.45
            ),
            "baseline_classification": map_category_to_ragtruth(vc.classification),
            "baseline_decision_rule": vc.decision_rule,
            "baseline_reasoning_trace": vc.reasoning_trace,
            "mc_label": mc_label if mc_prob is not None else None,
            "mc_prob": float(mc_prob) if mc_prob is not None else None,
            "nli_entail": (
                float(vc.scores.nli_entailment_prob)
                if vc.scores.nli_entailment_prob is not None
                else None
            ),
            "nli_contra": (
                float(vc.scores.nli_contradiction_prob)
                if vc.scores.nli_contradiction_prob is not None
                else None
            ),
            "nli_neutral": (
                float(vc.scores.nli_neutral_prob)
                if vc.scores.nli_neutral_prob is not None
                else None
            ),
            "secondary_nli_scores": [list(s) for s in sec_scores],
            "sec_nli_max_entail": float(max_sec_e) if max_sec_e is not None else None,
            "sec_nli_max_contra": float(max_sec_c) if max_sec_c is not None else None,
            "conflict_triggered": bool(conflict_triggered),
            "deberta_supported_prob": vc.scores.custom_deberta_supported_prob,
            "deberta_contradiction_prob": vc.scores.custom_deberta_contradiction_prob,
            "deberta_unsupported_prob": vc.scores.custom_deberta_unsupported_prob,
            "deberta_predicted_label": vc.scores.custom_deberta_predicted_label,
            "pair_token_count": int(top1_tokens[idx]) if top1_tokens is not None else None,
            "exceeds_512_tokens": (
                (top1_tokens[idx] > MAX_CONTEXT_TOKENS) if top1_tokens is not None else None
            ),
        }
    return out


_DIRECT_STRATEGY = VerificationDecisionStrategy()


def _synthetic_passage(similarity_score: float, text: str = "evidence") -> RetrievedPassage:
    chunk = DocumentChunk(
        chunk_id="synthetic_chunk",
        document_name="synthetic",
        page_number=1,
        chunk_index=0,
        text=text or "evidence",
        start_char=0,
        end_char=len(text or "evidence"),
        word_count=len((text or "evidence").split()),
    )
    return RetrievedPassage(chunk=chunk, similarity_score=similarity_score, rank=1)


def _evaluate_strategy_on_signal(
    claim_id: str,
    sentence_text: str,
    signal: Dict[str, Any],
    regime: str,
    use_deberta_as_nli: bool = False,
) -> str:
    """Run VerificationDecisionStrategy deterministically on cached signals."""
    sim = 1.0 if regime == "direct" else float(signal.get("best_similarity_score", 0.0))
    primary = _synthetic_passage(sim) if (regime == "direct" or signal.get("retrieved_chunk_text_word_count", 0) > 0) else None
    all_ev = [primary] if primary is not None else []

    mc_prob = signal.get("mc_prob")
    mc_tuple = (
        (1 if mc_prob >= 0.50 else 0, float(mc_prob))
        if mc_prob is not None
        else None
    )

    if use_deberta_as_nli:
        d_s = signal.get("deberta_supported_prob")
        d_c = signal.get("deberta_contradiction_prob")
        d_u = signal.get("deberta_unsupported_prob")
        nli_tuple = (
            (float(d_s), float(d_c), float(d_u))
            if (d_s is not None and d_c is not None and d_u is not None)
            else None
        )
    else:
        e = signal.get("nli_entail")
        c = signal.get("nli_contra")
        n = signal.get("nli_neutral")
        nli_tuple = (
            (float(e), float(c), float(n))
            if (e is not None and c is not None and n is not None)
            else None
        )

    sec_scores = None
    if regime == "retrieval":
        raw_sec = signal.get("secondary_nli_scores")
        if raw_sec:
            sec_scores = [(float(s[0]), float(s[1]), float(s[2])) for s in raw_sec]

    vc = _DIRECT_STRATEGY.evaluate_claim(
        claim_id=claim_id,
        claim_text=sentence_text,
        source_sentence=sentence_text,
        start_char=0,
        end_char=len(sentence_text),
        claim_start_char=None,
        claim_end_char=None,
        offset_type="EXACT",
        is_factual=True,
        primary_evidence=primary,
        all_evidence=all_ev,
        minicheck_score=mc_tuple,
        nli_scores=nli_tuple,
        secondary_nli_scores=sec_scores,
    )
    return map_category_to_ragtruth(vc.classification)


def _argmax_three_way(p_contra: float, p_support: float, p_unsup: float) -> str:
    scores = [
        (p_contra, "CONTRADICTED"),
        (p_support, "SUPPORTED_PROXY"),
        (p_unsup, "UNSUPPORTED"),
    ]
    return max(scores, key=lambda item: item[0])[1]


def predict_arm_for_record(
    record: Dict[str, Any],
    regime: str,
    arm_name: str,
) -> Tuple[str, Optional[float]]:
    """
    Compute discrete prediction (in EXTENDED_PRED_LABELS) and continuous hallucination
    score (higher => more likely hallucinated) for one cached record, regime, and arm.

    Only pure single-scorer arms return continuous scores (with Custom DeBERTa flagged as
    uncalibrated normalized logits from class-weighted training). Multi-model discrete rule
    and hybrid gates return None for continuous score rather than mixing uncalibrated
    cross-model score distributions. No probability averaging is performed in any arm.
    """
    if regime not in ("direct", "retrieval"):
        raise ValueError(f"Unsupported regime: {regime!r}")
    if arm_name not in ARM_NAMES:
        raise ValueError(f"Unknown arm_name: {arm_name!r}")

    signal = record[regime]
    ex_id = str(record["example_id"])
    sent = str(record["sentence_text"])

    mc_prob = signal.get("mc_prob")
    nli_e = signal.get("nli_entail")
    nli_c = signal.get("nli_contra")
    nli_n = signal.get("nli_neutral")
    deb_label = signal.get("deberta_predicted_label")
    deb_sup = signal.get("deberta_supported_prob")

    if arm_name == "arm0a_nli_only":
        if nli_e is None or nli_c is None or nli_n is None:
            return "UNVERIFIABLE", None
        pred = _argmax_three_way(float(nli_c), float(nli_e), float(nli_n))
        return pred, float(1.0 - float(nli_e))

    if arm_name == "arm0b_minicheck_only":
        if mc_prob is None:
            return "UNVERIFIABLE", None
        pred = "SUPPORTED_PROXY" if float(mc_prob) >= 0.50 else "UNSUPPORTED"
        return pred, float(1.0 - float(mc_prob))

    if arm_name == "arm1_baseline":
        if regime == "retrieval" and signal.get("baseline_classification") is not None:
            pred = map_category_to_ragtruth(signal["baseline_classification"])
        else:
            pred = _evaluate_strategy_on_signal(
                ex_id, sent, signal, regime=regime, use_deberta_as_nli=False
            )
        return pred, None

    if arm_name == "arm2_custom_deberta_only":
        if deb_label is None or deb_sup is None:
            return "UNVERIFIABLE", None
        # Raw uncalibrated logit softmax score from class-weighted training
        return str(deb_label), float(1.0 - float(deb_sup))

    if arm_name == "arm3a_mc_plus_custom_deberta":
        pred = _evaluate_strategy_on_signal(
            ex_id, sent, signal, regime=regime, use_deberta_as_nli=True
        )
        return pred, None

    # Baseline prediction shared by arm3b and arm3c
    baseline_pred, _ = predict_arm_for_record(
        record, regime=regime, arm_name="arm1_baseline"
    )

    if arm_name == "arm3b_baseline_support_deberta_disambig":
        if baseline_pred in ("UNVERIFIABLE", "SUPPORTED_PROXY"):
            return baseline_pred, None
        if deb_label == "CONTRADICTED":
            return "CONTRADICTED", None
        return "UNSUPPORTED", None

    # arm3c_selective_agreement_gate: Abstain to UNVERIFIABLE when baseline and Custom DeBERTa disagree
    if baseline_pred == "UNVERIFIABLE" or deb_label is None:
        return "UNVERIFIABLE", None
    baseline_is_hall = baseline_pred in ("CONTRADICTED", "UNSUPPORTED")
    deberta_is_hall = deb_label in ("CONTRADICTED", "UNSUPPORTED")
    if baseline_is_hall != deberta_is_hall:
        return "UNVERIFIABLE", None
    if not baseline_is_hall:
        return "SUPPORTED_PROXY", None
    # Both agree hallucinated: use Custom DeBERTa's error-type distinction (already CONTRADICTED or UNSUPPORTED)
    return str(deb_label), None


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def _prf_from_counts(tp: float, fp: float, fn: float) -> Dict[str, Any]:
    prec = _safe_div(tp, tp + fp)
    rec = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * prec * rec, prec + rec)
    return {
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "f1": round(f1, 6),
        "raw_f1": float(f1),
        "tp": int(tp) if isinstance(tp, int) else round(float(tp), 6),
        "fp": int(fp) if isinstance(fp, int) else round(float(fp), 6),
        "fn": int(fn) if isinstance(fn, int) else round(float(fn), 6),
        "support": int(tp + fn) if isinstance(tp, int) and isinstance(fn, int) else round(float(tp + fn), 6),
        "predicted": int(tp + fp) if isinstance(tp, int) and isinstance(fp, int) else round(float(tp + fp), 6),
    }


def _compute_three_class_block(
    y_true: Sequence[str],
    y_pred: Sequence[str],
) -> Dict[str, Any]:
    """
    Compute 3-class P/R/F1, Macro-F1, Support-Weighted F1, Accuracy, and confusion matrix.

    Note: When y_pred contains 'UNVERIFIABLE', any true-c example predicted as
    UNVERIFIABLE contributes 1 False Negative to class c (reducing recall & F1) and
    counts as an error in accuracy, never inflating precision or dropping from support.
    When a slice lacks support in one or more classes, `macro_f1_present_classes`
    averages F1 strictly over classes with support > 0 alongside the fixed 3-class mean.
    """
    total = len(y_true)
    per_class: Dict[str, Dict[str, Any]] = {}
    for cls in CANONICAL_LABELS:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == cls and yp == cls)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != cls and yp == cls)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == cls and yp != cls)
        metrics_cls = _prf_from_counts(tp, fp, fn)
        metrics_cls["has_support"] = metrics_cls["support"] > 0
        per_class[cls] = metrics_cls

    present_classes = [cls for cls in CANONICAL_LABELS if per_class[cls]["support"] > 0]
    macro_f1 = (
        sum(per_class[cls]["raw_f1"] for cls in CANONICAL_LABELS) / len(CANONICAL_LABELS)
        if total > 0
        else 0.0
    )
    macro_f1_present = (
        sum(per_class[cls]["raw_f1"] for cls in present_classes) / len(present_classes)
        if present_classes
        else 0.0
    )
    support_weighted_f1 = (
        sum(
            (per_class[cls]["support"] / total) * per_class[cls]["raw_f1"]
            for cls in present_classes
        )
        if total > 0 and present_classes
        else 0.0
    )
    for cls in CANONICAL_LABELS:
        per_class[cls].pop("raw_f1", None)

    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    accuracy = _safe_div(correct, total)

    confusion_3x4: Dict[str, Dict[str, int]] = {}
    for t_cls in CANONICAL_LABELS:
        confusion_3x4[t_cls] = {
            p_cls: sum(
                1 for yt, yp in zip(y_true, y_pred) if yt == t_cls and yp == p_cls
            )
            for p_cls in EXTENDED_PRED_LABELS
        }

    return {
        "n": int(total),
        "correct": int(correct),
        "accuracy": round(accuracy, 6),
        "macro_f1": round(macro_f1, 6),
        "macro_f1_present_classes": round(macro_f1_present, 6),
        "present_classes_count": len(present_classes),
        "support_weighted_f1": round(support_weighted_f1, 6),
        "per_class": per_class,
        "confusion_3x4": confusion_3x4,
    }


def compute_weighted_three_class_metrics(
    records: Sequence[Dict[str, Any]],
    predictions: Sequence[str],
    source_weights: Dict[str, float],
) -> Dict[str, Any]:
    """
    Compute Horvitz-Thompson stratum-reweighted 3-class Macro-F1, Support-Weighted F1,
    penalized Accuracy, per-class F1s, and binary strict F1 without premature rounding.
    """
    std_idx = [i for i, r in enumerate(records) if r["primary_label"] in CANONICAL_LABELS]
    if not std_idx:
        return {
            "weighted_accuracy_penalized": 0.0,
            "weighted_macro_f1_penalized": 0.0,
            "weighted_support_weighted_f1_penalized": 0.0,
            "weighted_binary_f1_penalized": 0.0,
            "weighted_per_class_f1": {cls: 0.0 for cls in CANONICAL_LABELS},
        }

    weights = [float(source_weights[str(records[i]["source_id"])]) for i in std_idx]
    total_w = sum(weights)
    correct_w = sum(
        w
        for w, i in zip(weights, std_idx)
        if records[i]["primary_label"] == predictions[i]
    )

    raw_f1s: Dict[str, float] = {}
    class_supports_w: Dict[str, float] = {}
    for cls in CANONICAL_LABELS:
        tp_w = sum(
            w
            for w, i in zip(weights, std_idx)
            if records[i]["primary_label"] == cls and predictions[i] == cls
        )
        fp_w = sum(
            w
            for w, i in zip(weights, std_idx)
            if records[i]["primary_label"] != cls and predictions[i] == cls
        )
        fn_w = sum(
            w
            for w, i in zip(weights, std_idx)
            if records[i]["primary_label"] == cls and predictions[i] != cls
        )
        class_supports_w[cls] = tp_w + fn_w
        prec = _safe_div(tp_w, tp_w + fp_w)
        rec = _safe_div(tp_w, tp_w + fn_w)
        raw_f1s[cls] = _safe_div(2.0 * prec * rec, prec + rec)

    weighted_macro_f1 = sum(raw_f1s.values()) / len(CANONICAL_LABELS)
    weighted_support_f1 = (
        sum((class_supports_w[cls] / total_w) * raw_f1s[cls] for cls in CANONICAL_LABELS)
        if total_w > 0
        else 0.0
    )

    # Binary hallucination strict weighted F1
    tp_b = fp_b = fn_b = 0.0
    for w, i in zip(weights, std_idx):
        tb = 1 if records[i]["primary_label"] in ("CONTRADICTED", "UNSUPPORTED") else 0
        yp = predictions[i]
        if yp == "UNVERIFIABLE":
            if tb == 1:
                fn_b += w
            else:
                fp_b += w
        else:
            pb = 1 if yp in ("CONTRADICTED", "UNSUPPORTED") else 0
            if tb == 1 and pb == 1:
                tp_b += w
            elif tb == 0 and pb == 1:
                fp_b += w
            elif tb == 1 and pb == 0:
                fn_b += w
    b_prec = _safe_div(tp_b, tp_b + fp_b)
    b_rec = _safe_div(tp_b, tp_b + fn_b)
    b_f1 = _safe_div(2.0 * b_prec * b_rec, b_prec + b_rec)

    return {
        "weighted_accuracy_penalized": round(_safe_div(correct_w, total_w), 6),
        "weighted_macro_f1_penalized": round(weighted_macro_f1, 6),
        "weighted_support_weighted_f1_penalized": round(weighted_support_f1, 6),
        "weighted_binary_f1_penalized": round(b_f1, 6),
        "weighted_per_class_f1": {cls: round(val, 6) for cls, val in raw_f1s.items()},
    }


def _compute_binary_hallucination_block(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    cont_scores: Optional[Sequence[Optional[float]]] = None,
) -> Dict[str, Any]:
    """
    Compute binary hallucination detection (CONTRADICTED + UNSUPPORTED = 1, SUPPORTED_PROXY = 0).

    Reports:
    - penalized_strict: UNVERIFIABLE always counts as wrong (false negative if true=1, false positive if true=0).
    - abstention_as_flagged: UNVERIFIABLE treated as positive hallucination flag.
    - AUROC & PR-AUC when continuous scores exist across all examples and both classes occur.
    """
    total = len(y_true)
    true_bin = [1 if yt in ("CONTRADICTED", "UNSUPPORTED") else 0 for yt in y_true]

    tp_s = fp_s = fn_s = tn_s = 0
    tp_f = fp_f = fn_f = tn_f = 0

    for tb, yp in zip(true_bin, y_pred):
        if yp == "UNVERIFIABLE":
            # Strict penalized: always counted as error
            if tb == 1:
                fn_s += 1
            else:
                fp_s += 1
            # Flagged view: abstention flags for review
            if tb == 1:
                tp_f += 1
            else:
                fp_f += 1
        else:
            pb = 1 if yp in ("CONTRADICTED", "UNSUPPORTED") else 0
            if tb == 1 and pb == 1:
                tp_s += 1
                tp_f += 1
            elif tb == 0 and pb == 1:
                fp_s += 1
                fp_f += 1
            elif tb == 1 and pb == 0:
                fn_s += 1
                fn_f += 1
            else:
                tn_s += 1
                tn_f += 1

    strict_prf = _prf_from_counts(tp_s, fp_s, fn_s)
    strict_prf.pop("raw_f1", None)
    flagged_prf = _prf_from_counts(tp_f, fp_f, fn_f)
    flagged_prf.pop("raw_f1", None)

    auroc: Optional[float] = None
    pr_auc: Optional[float] = None
    if cont_scores is not None and total > 0 and all(s is not None for s in cont_scores):
        if len(set(true_bin)) == 2:
            y_score = [float(s) for s in cont_scores]
            auroc = round(float(roc_auc_score(true_bin, y_score)), 6)
            pr_auc = round(float(average_precision_score(true_bin, y_score)), 6)

    return {
        "n": int(total),
        "positive_hallucinated_support": int(sum(true_bin)),
        "penalized_strict": {
            **strict_prf,
            "accuracy": round(_safe_div(tp_s + tn_s, total), 6),
        },
        "abstention_as_flagged": {
            **flagged_prf,
            "accuracy": round(_safe_div(tp_f + tn_f, total), 6),
        },
        "auroc": auroc,
        "pr_auc": pr_auc,
    }


def compute_arm_metrics(
    records: Sequence[Dict[str, Any]],
    predictions: Sequence[str],
    continuous_scores: Optional[Sequence[Optional[float]]] = None,
) -> Dict[str, Any]:
    """
    Compute complete 3-class, UNVERIFIABLE abstention/coverage/selective/penalized,
    binary hallucination, and MIXED_HALLUCINATION diagnostic metrics.
    """
    if len(records) != len(predictions):
        raise ValueError("records and predictions must have equal length")

    std_indices = [
        i for i, r in enumerate(records) if r["primary_label"] in CANONICAL_LABELS
    ]
    mixed_indices = [
        i for i, r in enumerate(records) if r["primary_label"] == MIXED_LABEL
    ]

    y_true = [records[i]["primary_label"] for i in std_indices]
    y_pred = [predictions[i] for i in std_indices]
    std_scores = (
        [continuous_scores[i] for i in std_indices]
        if continuous_scores is not None
        else None
    )

    n_total = len(std_indices)
    unverifiable_indices = [k for k, yp in enumerate(y_pred) if yp == "UNVERIFIABLE"]
    covered_indices = [k for k, yp in enumerate(y_pred) if yp != "UNVERIFIABLE"]
    n_unverifiable = len(unverifiable_indices)
    n_covered = len(covered_indices)

    penalized = _compute_three_class_block(y_true, y_pred)

    y_true_cov = [y_true[k] for k in covered_indices]
    y_pred_cov = [y_pred[k] for k in covered_indices]
    selective = _compute_three_class_block(y_true_cov, y_pred_cov)
    selective["covered_of_total"] = f"{n_covered}/{n_total}"

    abstention_true_breakdown = {
        cls: sum(1 for k in unverifiable_indices if y_true[k] == cls)
        for cls in CANONICAL_LABELS
    }

    binary_penalized = _compute_binary_hallucination_block(
        y_true, y_pred, cont_scores=std_scores
    )
    binary_selective = _compute_binary_hallucination_block(
        y_true_cov,
        y_pred_cov,
        cont_scores=(
            [std_scores[k] for k in covered_indices] if std_scores is not None else None
        ),
    )

    # Separate MIXED_HALLUCINATION diagnostic (excluded from 3-class metrics; abstentions separated)
    mixed_preds = Counter(predictions[i] for i in mixed_indices)
    n_mixed = len(mixed_indices)
    mixed_flagged_confirmed = sum(
        1
        for i in mixed_indices
        if predictions[i] in ("CONTRADICTED", "UNSUPPORTED")
    )
    mixed_unverifiable = int(mixed_preds.get("UNVERIFIABLE", 0))
    mixed_falsely_supported = int(mixed_preds.get("SUPPORTED_PROXY", 0))

    return {
        "n_total_3class": int(n_total),
        "n_covered": int(n_covered),
        "n_unverifiable": int(n_unverifiable),
        "abstention_rate": round(_safe_div(n_unverifiable, n_total), 6),
        "coverage": round(_safe_div(n_covered, n_total), 6),
        "abstention_true_label_breakdown": abstention_true_breakdown,
        "penalized_3class": penalized,
        "selective_3class": selective,
        "binary_hallucination_penalized": binary_penalized,
        "binary_hallucination_selective": binary_selective,
        "mixed_hallucination_diagnostic": {
            "n_mixed": int(n_mixed),
            "predicted_counts": {lbl: int(mixed_preds.get(lbl, 0)) for lbl in EXTENDED_PRED_LABELS},
            "flagged_confirmed_hallucination_count": int(mixed_flagged_confirmed),
            "flagged_confirmed_hallucination_rate": round(
                _safe_div(mixed_flagged_confirmed, n_mixed), 6
            ),
            "unverifiable_abstained_count": mixed_unverifiable,
            "unverifiable_abstained_rate": round(_safe_div(mixed_unverifiable, n_mixed), 6),
            "falsely_supported_count": mixed_falsely_supported,
            "falsely_supported_rate": round(
                _safe_div(mixed_falsely_supported, n_mixed), 6
            ),
        },
    }


def _safe_pearson_and_spearman(xs: Sequence[float], ys: Sequence[float]) -> Dict[str, Optional[float]]:
    if len(xs) < 2:
        return {"pearson_r": None, "spearman_rho": None}
    arr_x = np.asarray(xs, dtype=float)
    arr_y = np.asarray(ys, dtype=float)
    if np.std(arr_x) == 0.0 or np.std(arr_y) == 0.0:
        return {"pearson_r": None, "spearman_rho": None}
    pearson = float(np.corrcoef(arr_x, arr_y)[0, 1])

    def _ranks(vals: np.ndarray) -> np.ndarray:
        order = np.argsort(vals, kind="mergesort")
        ranks = np.empty(len(vals), dtype=float)
        i = 0
        while i < len(vals):
            j = i
            while j + 1 < len(vals) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg_rank = 0.5 * (i + j) + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg_rank
            i = j + 1
        return ranks

    rx = _ranks(arr_x)
    ry = _ranks(arr_y)
    if np.std(rx) == 0.0 or np.std(ry) == 0.0:
        spearman = None
    else:
        spearman = round(float(np.corrcoef(rx, ry)[0, 1]), 6)
    return {"pearson_r": round(pearson, 6), "spearman_rho": spearman}


def compute_response_level_evaluation(
    records: Sequence[Dict[str, Any]],
    predictions: Sequence[str],
    scorer: Optional[ResponseScorer] = None,
) -> Dict[str, Any]:
    """
    Group sentence predictions by complete response_id, compute ResponseScorer metrics,
    and compare predicted vs RAGTruth-annotated confirmed hallucination, abstention,
    support (GFI), and severity (HSI) rates plus generator-LLM aggregate rankings.

    Explicitly separates confirmed hallucination rate ((CONTRADICTED + UNSUPPORTED) / N)
    and UNVERIFIABLE rate (UNVERIFIABLE / N) from (100 - GFI).
    """
    scorer = scorer or ResponseScorer()
    by_resp: Dict[str, List[Tuple[Dict[str, Any], str]]] = defaultdict(list)
    for rec, pred in zip(records, predictions):
        by_resp[str(rec["response_id"])].append((rec, pred))

    pred_conf_hall_rates: List[float] = []
    pred_unverifiable_rates: List[float] = []
    pred_non_supported_rates: List[float] = []
    pred_sup_rates: List[float] = []
    pred_contra_rates: List[float] = []
    resp_coverages: List[float] = []

    true_hall_rates_incl_mixed: List[float] = []
    true_hall_rates_excl_mixed: List[float] = []
    true_sup_rates: List[float] = []
    true_contra_rates: List[float] = []

    total_claims = 0
    total_covered_claims = 0
    total_mixed_claims = 0
    total_true_hall_incl_mixed = 0
    total_pred_conf_hall = 0
    total_pred_unverifiable = 0

    by_task_data: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {"true_hall": [], "pred_conf_hall": [], "true_sup": [], "pred_sup": []}
    )
    by_model_rows: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "num_responses": 0,
            "num_claims": 0,
            "true_hall_claims_incl_mixed": 0,
            "pred_conf_hall_claims": 0,
            "pred_unverifiable_claims": 0,
            "pred_sup_claims": 0,
            "resp_true_hall_rates": [],
            "resp_pred_conf_hall_rates": [],
            "resp_pred_unverifiable_rates": [],
            "resp_pred_gfi_rates": [],
            "resp_coverages": [],
        }
    )

    for resp_id in sorted(by_resp.keys()):
        items = by_resp[resp_id]
        n = len(items)
        task_type = str(items[0][0].get("task_type", "Unknown"))
        model_id = str(items[0][0].get("model", ""))

        verified_claims: List[VerifiedClaim] = []
        true_hall_incl = 0
        true_hall_excl = 0
        true_sup = 0
        true_contra = 0
        resp_mixed = 0

        for rec, pred in items:
            lbl = rec["primary_label"]
            if lbl == MIXED_LABEL:
                resp_mixed += 1
            if lbl in ("CONTRADICTED", "UNSUPPORTED", MIXED_LABEL):
                true_hall_incl += 1
            if lbl in ("CONTRADICTED", "UNSUPPORTED"):
                true_hall_excl += 1
            if lbl == "SUPPORTED_PROXY":
                true_sup += 1
            if lbl in ("CONTRADICTED", MIXED_LABEL):
                true_contra += 1

            sent = str(rec["sentence_text"])
            vc = _DIRECT_STRATEGY.evaluate_claim(
                claim_id=str(rec["example_id"]),
                claim_text=sent,
                source_sentence=sent,
                start_char=0,
                end_char=len(sent),
                claim_start_char=None,
                claim_end_char=None,
                offset_type="EXACT",
                is_factual=True,
                primary_evidence=_synthetic_passage(1.0),
                all_evidence=[_synthetic_passage(1.0)],
                minicheck_score=(1, 0.99),
                nli_scores=(0.99, 0.005, 0.005),
                secondary_nli_scores=None,
            )
            target_cat = _RAGTRUTH_TO_CATEGORY[pred]
            verified_claims.append(
                vc.model_copy(update={"classification": target_cat})
            )

        resp_score = scorer.score_response(
            verified_claims,
            response_id=resp_id,
            model_id=model_id,
        )
        cnt = resp_score.counts
        denom = cnt.factual_claims if cnt.factual_claims > 0 else n
        gfi = (cnt.supported_count / denom) * 100.0
        conf_hall = ((cnt.contradicted_count + cnt.unsupported_count) / denom) * 100.0
        unver_rate = (cnt.unverifiable_count / denom) * 100.0
        non_sup = 100.0 - gfi
        hsi = (cnt.contradicted_count / denom) * 100.0
        resp_cov = ((denom - cnt.unverifiable_count) / denom) * 100.0

        t_hall_incl = (true_hall_incl / n) * 100.0
        t_hall_excl = (true_hall_excl / n) * 100.0
        t_sup = (true_sup / n) * 100.0
        t_contra = (true_contra / n) * 100.0

        pred_conf_hall_rates.append(conf_hall)
        pred_unverifiable_rates.append(unver_rate)
        pred_non_supported_rates.append(non_sup)
        pred_sup_rates.append(gfi)
        pred_contra_rates.append(hsi)
        resp_coverages.append(resp_cov)

        true_hall_rates_incl_mixed.append(t_hall_incl)
        true_hall_rates_excl_mixed.append(t_hall_excl)
        true_sup_rates.append(t_sup)
        true_contra_rates.append(t_contra)

        total_claims += n
        total_covered_claims += denom - cnt.unverifiable_count
        total_mixed_claims += resp_mixed
        total_true_hall_incl_mixed += true_hall_incl
        total_pred_conf_hall += cnt.contradicted_count + cnt.unsupported_count
        total_pred_unverifiable += cnt.unverifiable_count

        by_task_data[task_type]["true_hall"].append(t_hall_incl)
        by_task_data[task_type]["pred_conf_hall"].append(conf_hall)
        by_task_data[task_type]["true_sup"].append(t_sup)
        by_task_data[task_type]["pred_sup"].append(gfi)

        m_entry = by_model_rows[model_id]
        m_entry["num_responses"] += 1
        m_entry["num_claims"] += n
        m_entry["true_hall_claims_incl_mixed"] += true_hall_incl
        m_entry["pred_conf_hall_claims"] += cnt.contradicted_count + cnt.unsupported_count
        m_entry["pred_unverifiable_claims"] += cnt.unverifiable_count
        m_entry["pred_sup_claims"] += cnt.supported_count
        m_entry["resp_true_hall_rates"].append(t_hall_incl)
        m_entry["resp_pred_conf_hall_rates"].append(conf_hall)
        m_entry["resp_pred_unverifiable_rates"].append(unver_rate)
        m_entry["resp_pred_gfi_rates"].append(gfi)
        m_entry["resp_coverages"].append(resp_cov)

    conf_hall_corr = _safe_pearson_and_spearman(true_hall_rates_incl_mixed, pred_conf_hall_rates)
    conf_hall_excl_mixed_corr = _safe_pearson_and_spearman(
        true_hall_rates_excl_mixed, pred_conf_hall_rates
    )
    non_sup_corr = _safe_pearson_and_spearman(true_hall_rates_incl_mixed, pred_non_supported_rates)
    sup_corr = _safe_pearson_and_spearman(true_sup_rates, pred_sup_rates)
    contra_corr = _safe_pearson_and_spearman(true_contra_rates, pred_contra_rates)

    conf_hall_diffs = [
        abs(p - t) for p, t in zip(pred_conf_hall_rates, true_hall_rates_incl_mixed)
    ]
    conf_hall_excl_diffs = [
        abs(p - t) for p, t in zip(pred_conf_hall_rates, true_hall_rates_excl_mixed)
    ]
    non_sup_diffs = [
        abs(p - t) for p, t in zip(pred_non_supported_rates, true_hall_rates_incl_mixed)
    ]
    sup_diffs = [abs(p - t) for p, t in zip(pred_sup_rates, true_sup_rates)]

    by_task_out: Dict[str, Any] = {}
    for t_name, td in sorted(by_task_data.items()):
        t_diffs = [abs(p - t) for p, t in zip(td["pred_conf_hall"], td["true_hall"])]
        by_task_out[t_name] = {
            "num_responses": len(td["true_hall"]),
            "confirmed_hallucination_rate_pct": {
                **_safe_pearson_and_spearman(td["true_hall"], td["pred_conf_hall"]),
                "mae": round(float(np.mean(t_diffs)), 6) if t_diffs else None,
            },
            "support_rate_gfi_pct": _safe_pearson_and_spearman(td["true_sup"], td["pred_sup"]),
        }

    per_model_summary: Dict[str, Dict[str, Any]] = {}
    m_true_macro: List[float] = []
    m_pred_macro: List[float] = []
    m_true_pooled: List[float] = []
    m_pred_pooled: List[float] = []

    for mod_id in sorted(by_model_rows.keys()):
        md = by_model_rows[mod_id]
        mc = md["num_claims"]
        t_macro = float(np.mean(md["resp_true_hall_rates"]))
        p_macro = float(np.mean(md["resp_pred_conf_hall_rates"]))
        t_pool = (md["true_hall_claims_incl_mixed"] / mc) * 100.0 if mc > 0 else 0.0
        p_pool = (md["pred_conf_hall_claims"] / mc) * 100.0 if mc > 0 else 0.0
        u_pool = (md["pred_unverifiable_claims"] / mc) * 100.0 if mc > 0 else 0.0
        gfi_macro = float(np.mean(md["resp_pred_gfi_rates"]))
        cov_macro = float(np.mean(md["resp_coverages"]))

        m_true_macro.append(t_macro)
        m_pred_macro.append(p_macro)
        m_true_pooled.append(t_pool)
        m_pred_pooled.append(p_pool)

        per_model_summary[mod_id] = {
            "num_responses": int(md["num_responses"]),
            "num_claims": int(mc),
            "true_hall_rate_response_mean_pct": round(t_macro, 4),
            "true_hall_rate_pooled_pct": round(t_pool, 4),
            "pred_confirmed_hall_rate_response_mean_pct": round(p_macro, 4),
            "pred_confirmed_hall_rate_pooled_pct": round(p_pool, 4),
            "pred_unverifiable_rate_pooled_pct": round(u_pool, 4),
            "pred_gfi_response_mean_pct": round(gfi_macro, 4),
            "response_coverage_pct": round(cov_macro, 4),
        }

    model_macro_corr = _safe_pearson_and_spearman(m_true_macro, m_pred_macro)
    model_pooled_corr = _safe_pearson_and_spearman(m_true_pooled, m_pred_pooled)
    model_macro_mae = (
        round(float(np.mean([abs(p - t) for p, t in zip(m_pred_macro, m_true_macro)])), 4)
        if m_true_macro
        else None
    )
    model_pooled_mae = (
        round(float(np.mean([abs(p - t) for p, t in zip(m_pred_pooled, m_true_pooled)])), 4)
        if m_true_pooled
        else None
    )

    return {
        "num_responses": len(by_resp),
        "total_claims": int(total_claims),
        "mixed_claims_in_true_hall": int(total_mixed_claims),
        "response_coverage": {
            "mean_response_coverage_pct": round(float(np.mean(resp_coverages)), 4)
            if resp_coverages
            else None,
            "pooled_claim_coverage_pct": round(_safe_div(total_covered_claims, total_claims) * 100.0, 4),
        },
        "confirmed_hallucination_rate_pct": {
            **conf_hall_corr,
            "mae": round(float(np.mean(conf_hall_diffs)), 6) if conf_hall_diffs else None,
            "rmse": round(float(np.sqrt(np.mean(np.square(conf_hall_diffs)))), 6)
            if conf_hall_diffs
            else None,
            "mean_true_incl_mixed_pct": round(float(np.mean(true_hall_rates_incl_mixed)), 4)
            if true_hall_rates_incl_mixed
            else None,
            "mean_true_excl_mixed_pct": round(float(np.mean(true_hall_rates_excl_mixed)), 4)
            if true_hall_rates_excl_mixed
            else None,
            "mean_pred_confirmed_pct": round(float(np.mean(pred_conf_hall_rates)), 4)
            if pred_conf_hall_rates
            else None,
            "pooled_true_incl_mixed_pct": round(
                _safe_div(total_true_hall_incl_mixed, total_claims) * 100.0, 4
            ),
            "pooled_pred_confirmed_pct": round(
                _safe_div(total_pred_conf_hall, total_claims) * 100.0, 4
            ),
            "excl_mixed_true_sensitivity": {
                **conf_hall_excl_mixed_corr,
                "mae": round(float(np.mean(conf_hall_excl_diffs)), 6)
                if conf_hall_excl_diffs
                else None,
            },
        },
        "unverifiable_rate_pct": {
            "mean_pred_pct": round(float(np.mean(pred_unverifiable_rates)), 4)
            if pred_unverifiable_rates
            else None,
            "pooled_pred_pct": round(_safe_div(total_pred_unverifiable, total_claims) * 100.0, 4),
        },
        "non_supported_rate_100_minus_gfi_pct": {
            "note": "Secondary composite: (CONTRADICTED + UNSUPPORTED + UNVERIFIABLE) / N; do not interpret UNVERIFIABLE as confirmed hallucination.",
            **non_sup_corr,
            "mae_vs_true_hall": round(float(np.mean(non_sup_diffs)), 6) if non_sup_diffs else None,
            "mean_pred_pct": round(float(np.mean(pred_non_supported_rates)), 4)
            if pred_non_supported_rates
            else None,
        },
        "support_rate_gfi_pct": {
            **sup_corr,
            "mae": round(float(np.mean(sup_diffs)), 6) if sup_diffs else None,
            "mean_true_pct": round(float(np.mean(true_sup_rates)), 4) if true_sup_rates else None,
            "mean_pred_pct": round(float(np.mean(pred_sup_rates)), 4) if pred_sup_rates else None,
        },
        "contradiction_rate_hsi_pct": contra_corr,
        "by_task": by_task_out,
        "by_generator_model": {
            "num_models": len(per_model_summary),
            "ranking_response_mean_hall": {
                **model_macro_corr,
                "mae": model_macro_mae,
            },
            "ranking_pooled_hall": {
                **model_pooled_corr,
                "mae": model_pooled_mae,
            },
            "per_model": per_model_summary,
        },
    }


def _exact_two_sided_binomial_log10_and_p(k: int, n: int) -> Tuple[float, float]:
    """
    Compute exact two-sided binomial p-value and log10(p) for symmetric p=0.5 in log-space.

    Avoids IEEE-754 underflow and rounding truncation for arbitrarily small tail probabilities.
    """
    if n <= 0:
        return 1.0, 0.0
    m = min(k, n - k)
    if m * 2 == n:
        return 1.0, 0.0

    log_half_n = n * math.log(0.5)
    log_fact_n = math.lgamma(n + 1)
    log_terms = [
        log_fact_n - math.lgamma(i + 1) - math.lgamma(n - i + 1) + log_half_n
        for i in range(m + 1)
    ]
    max_lt = max(log_terms)
    log_tail = max_lt + math.log(sum(math.exp(lt - max_lt) for lt in log_terms))
    log_p = min(0.0, math.log(2.0) + log_tail)
    log10_p = log_p / math.log(10.0)
    p_val = 1.0 if log_p >= 0.0 else (math.exp(log_p) if log_p > -745.0 else 0.0)
    return float(p_val), float(log10_p)


def _format_p_sci(p_val: float, log10_p: Optional[float] = None) -> str:
    if p_val >= 1e-4:
        return f"{p_val:.6g}"
    if p_val > 0.0:
        return f"{p_val:.3e}"
    if log10_p is not None and math.isfinite(log10_p):
        exp = math.floor(log10_p)
        mant = 10.0 ** (log10_p - exp)
        return f"{mant:.3f}e{exp:+d}"
    return "0.0"


def apply_holm_bonferroni(p_values: Sequence[float]) -> List[float]:
    """Compute step-down Holm-Bonferroni adjusted p-values preserving input order."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda pair: float(pair[1]))
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p_val) in enumerate(indexed):
        factor = m - rank
        val = min(1.0, float(p_val) * factor)
        running_max = max(running_max, val)
        adjusted[orig_idx] = running_max
    return adjusted


def exact_mcnemar_test(
    y_true: Sequence[str],
    pred_baseline: Sequence[str],
    pred_candidate: Sequence[str],
) -> Dict[str, Any]:
    """
    Run legacy sentence-level exact two-sided McNemar's test comparing baseline vs candidate
    on 3-class exact correctness and binary hallucination correctness.

    Limitation: Sentences nested within source_id/response_id are correlated, so sentence-level
    McNemar is anti-conservative; source-clustered comparisons should be treated as primary.
    """
    if not (len(y_true) == len(pred_baseline) == len(pred_candidate)):
        raise ValueError("Input sequences must share length")

    def _run(correct_a: List[bool], correct_b: List[bool]) -> Dict[str, Any]:
        both_correct = sum(1 for a, b in zip(correct_a, correct_b) if a and b)
        both_wrong = sum(1 for a, b in zip(correct_a, correct_b) if not a and not b)
        b_base_only = sum(1 for a, b in zip(correct_a, correct_b) if a and not b)
        c_cand_only = sum(1 for a, b in zip(correct_a, correct_b) if not a and b)
        n_disc = b_base_only + c_cand_only
        net_gain = c_cand_only - b_base_only
        direction = (
            "candidate_better"
            if net_gain > 0
            else ("baseline_better" if net_gain < 0 else "tie")
        )
        p_val, log10_p = _exact_two_sided_binomial_log10_and_p(
            min(b_base_only, c_cand_only), n_disc
        )
        return {
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "baseline_only_correct": b_base_only,
            "candidate_only_correct": c_cand_only,
            "discordant_total": n_disc,
            "net_gain_cand_minus_base": net_gain,
            "direction": direction,
            "exact_two_sided_p_value": p_val,
            "p_value_sci": _format_p_sci(p_val, log10_p),
            "log10_p_value": round(log10_p, 4),
            "clustering_caveat": (
                "Sentence-level exact McNemar treats nested claims as i.i.d. and is "
                "anti-conservative under within-source/within-response correlation."
            ),
        }

    std_idx = [i for i, yt in enumerate(y_true) if yt in CANONICAL_LABELS]
    three_a = [pred_baseline[i] == y_true[i] for i in std_idx]
    three_b = [pred_candidate[i] == y_true[i] for i in std_idx]

    def _bin_correct(yt: str, yp: str) -> bool:
        if yp == "UNVERIFIABLE":
            return False
        return (yt in ("CONTRADICTED", "UNSUPPORTED")) == (
            yp in ("CONTRADICTED", "UNSUPPORTED")
        )

    bin_a = [_bin_correct(y_true[i], pred_baseline[i]) for i in std_idx]
    bin_b = [_bin_correct(y_true[i], pred_candidate[i]) for i in std_idx]

    return {
        "n_3class": len(std_idx),
        "three_class_exact": _run(three_a, three_b),
        "binary_hallucination_strict": _run(bin_a, bin_b),
    }


def _wilcoxon_signed_rank_p(diffs: Sequence[float]) -> Optional[float]:
    """Two-sided Wilcoxon signed-rank normal-approximation p-value with tie correction."""
    nz = [d for d in diffs if abs(d) > 1e-12]
    n = len(nz)
    if n < 2:
        return 1.0 if n == 0 else None
    abs_vals = np.abs(np.asarray(nz, dtype=float))
    order = np.argsort(abs_vals, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    tie_adj = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs_vals[order[j + 1]] == abs_vals[order[i]]:
            j += 1
        avg_r = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_r
        t = j - i + 1
        if t > 1:
            tie_adj += t * t * t - t
        i = j + 1

    w_pos = sum(r for d, r in zip(nz, ranks) if d > 0)
    mu = n * (n + 1) / 4.0
    var = (n * (n + 1) * (2 * n + 1) - 0.5 * tie_adj) / 24.0
    if var <= 0.0:
        return 1.0
    z = (abs(w_pos - mu) - 0.5) / math.sqrt(var)
    z = max(0.0, z)
    return float(math.erfc(z / math.sqrt(2.0)))


def clustered_paired_comparison(
    records: Sequence[Dict[str, Any]],
    pred_baseline: Sequence[str],
    pred_candidate: Sequence[str],
    cluster_key: str = "source_id",
    source_weights: Optional[Dict[str, float]] = None,
    n_perm: int = 20000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Paired cluster-level comparison (primary: source_id; supplementary: response_id).

    Computes within-cluster correctness differences (candidate - baseline) for both 3-class
    exact accuracy and binary hallucination strict accuracy, evaluated via exact paired sign
    test, stratified sign-flip randomization test, and Wilcoxon signed-rank test.
    """
    if not (len(records) == len(pred_baseline) == len(pred_candidate)):
        raise ValueError("records and prediction lists must share length")

    std_indices = [
        i for i, r in enumerate(records) if r["primary_label"] in CANONICAL_LABELS
    ]
    clusters: Dict[str, List[int]] = defaultdict(list)
    cluster_tasks: Dict[str, str] = {}
    cluster_source_id: Dict[str, str] = {}
    for i in std_indices:
        cid = str(records[i][cluster_key])
        clusters[cid].append(i)
        cluster_tasks[cid] = str(records[i]["task_type"])
        cluster_source_id[cid] = str(records[i]["source_id"])

    def _bin_correct(yt: str, yp: str) -> bool:
        if yp == "UNVERIFIABLE":
            return False
        return (yt in ("CONTRADICTED", "UNSUPPORTED")) == (
            yp in ("CONTRADICTED", "UNSUPPORTED")
        )

    def _evaluate_endpoint(use_binary: bool) -> Dict[str, Any]:
        deltas: List[float] = []
        weights: List[float] = []
        by_task_deltas: Dict[str, List[Tuple[float, float]]] = defaultdict(list)

        for cid in sorted(clusters.keys()):
            idxs = clusters[cid]
            if not use_binary:
                base_acc = sum(
                    1 for i in idxs if pred_baseline[i] == records[i]["primary_label"]
                ) / len(idxs)
                cand_acc = sum(
                    1 for i in idxs if pred_candidate[i] == records[i]["primary_label"]
                ) / len(idxs)
            else:
                base_acc = sum(
                    1
                    for i in idxs
                    if _bin_correct(records[i]["primary_label"], pred_baseline[i])
                ) / len(idxs)
                cand_acc = sum(
                    1
                    for i in idxs
                    if _bin_correct(records[i]["primary_label"], pred_candidate[i])
                ) / len(idxs)
            d = cand_acc - base_acc
            w = (
                float(source_weights[cluster_source_id[cid]])
                if source_weights is not None and cluster_source_id[cid] in source_weights
                else 1.0
            )
            deltas.append(d)
            weights.append(w)
            by_task_deltas[cluster_tasks[cid]].append((d, w))

        n_pos = sum(1 for d in deltas if d > 1e-12)
        n_neg = sum(1 for d in deltas if d < -1e-12)
        n_zero = len(deltas) - n_pos - n_neg
        sign_p, sign_log10 = _exact_two_sided_binomial_log10_and_p(
            min(n_pos, n_neg), n_pos + n_neg
        )

        obs_unw = float(np.mean(deltas)) if deltas else 0.0
        sum_w = sum(weights)
        obs_w = (
            sum(d * w for d, w in zip(deltas, weights)) / sum_w
            if sum_w > 0
            else 0.0
        )

        # Stratified within-task paired sign-flip randomization test
        rng = np.random.default_rng(seed)
        arr_d = np.asarray(deltas, dtype=float)
        if len(arr_d) > 0 and np.any(np.abs(arr_d) > 1e-12):
            flips = rng.choice([-1.0, 1.0], size=(n_perm, len(arr_d)))
            perm_means = (flips * arr_d).mean(axis=1)
            exceed_unw = int(np.sum(np.abs(perm_means) >= abs(obs_unw) - 1e-12))
            perm_p_unw = float((1 + exceed_unw) / (n_perm + 1))

            arr_w = np.asarray(weights, dtype=float)
            perm_w_means = (flips * (arr_d * arr_w)).sum(axis=1) / max(float(arr_w.sum()), 1e-12)
            exceed_w = int(np.sum(np.abs(perm_w_means) >= abs(obs_w) - 1e-12))
            perm_p_w = float((1 + exceed_w) / (n_perm + 1))
        else:
            perm_p_unw = 1.0
            perm_p_w = 1.0

        wilcoxon_p = _wilcoxon_signed_rank_p(deltas)
        direction = (
            "candidate_better"
            if obs_unw > 1e-12
            else ("baseline_better" if obs_unw < -1e-12 else "tie")
        )

        return {
            "n_clusters": len(deltas),
            "positive_clusters": n_pos,
            "negative_clusters": n_neg,
            "tied_clusters": n_zero,
            "mean_cluster_delta_unweighted": round(obs_unw, 6),
            "mean_cluster_delta_weighted": round(obs_w, 6) if source_weights else None,
            "median_cluster_delta": round(float(np.median(deltas)), 6) if deltas else 0.0,
            "direction": direction,
            "exact_sign_test_p_value": sign_p,
            "exact_sign_test_p_sci": _format_p_sci(sign_p, sign_log10),
            "paired_sign_flip_p_value": perm_p_unw,
            "paired_sign_flip_weighted_p_value": perm_p_w if source_weights else None,
            "wilcoxon_signed_rank_p_value": wilcoxon_p,
            "n_perm": int(n_perm),
        }

    return {
        "cluster_unit": cluster_key,
        "three_class_accuracy": _evaluate_endpoint(use_binary=False),
        "binary_hallucination_accuracy": _evaluate_endpoint(use_binary=True),
    }


def _extract_metric_value(metrics: Dict[str, Any], metric_key: str) -> float:
    if metric_key == "macro_f1_penalized":
        return float(metrics["penalized_3class"]["macro_f1"])
    if metric_key == "support_weighted_f1_penalized":
        return float(metrics["penalized_3class"]["support_weighted_f1"])
    if metric_key == "accuracy_penalized":
        return float(metrics["penalized_3class"]["accuracy"])
    if metric_key == "contradicted_f1_penalized":
        return float(metrics["penalized_3class"]["per_class"]["CONTRADICTED"]["f1"])
    if metric_key == "unsupported_f1_penalized":
        return float(metrics["penalized_3class"]["per_class"]["UNSUPPORTED"]["f1"])
    if metric_key == "binary_f1_penalized":
        return float(metrics["binary_hallucination_penalized"]["penalized_strict"]["f1"])
    raise ValueError(f"Unsupported metric_key: {metric_key!r}")


def _extract_weighted_metric_value(metrics: Dict[str, Any], metric_key: str) -> float:
    if metric_key == "macro_f1_penalized":
        return float(metrics["weighted_macro_f1_penalized"])
    if metric_key == "support_weighted_f1_penalized":
        return float(metrics["weighted_support_weighted_f1_penalized"])
    if metric_key == "accuracy_penalized":
        return float(metrics["weighted_accuracy_penalized"])
    if metric_key == "contradicted_f1_penalized":
        return float(metrics["weighted_per_class_f1"]["CONTRADICTED"])
    if metric_key == "unsupported_f1_penalized":
        return float(metrics["weighted_per_class_f1"]["UNSUPPORTED"])
    if metric_key == "binary_f1_penalized":
        return float(metrics["weighted_binary_f1_penalized"])
    raise ValueError(f"Unsupported metric_key: {metric_key!r}")


def _cluster_sufficient_stats(
    records: Sequence[Dict[str, Any]],
    preds: Sequence[str],
    indices: Sequence[int],
) -> Tuple[int, int, Dict[str, Tuple[int, int, int]], Tuple[int, int, int]]:
    """Precompute exact confusion counts for one source cluster for fast exact bootstrap."""
    n = len(indices)
    correct = sum(1 for i in indices if records[i]["primary_label"] == preds[i])
    per_cls: Dict[str, Tuple[int, int, int]] = {}
    for cls in CANONICAL_LABELS:
        tp = sum(1 for i in indices if records[i]["primary_label"] == cls and preds[i] == cls)
        fp = sum(1 for i in indices if records[i]["primary_label"] != cls and preds[i] == cls)
        fn = sum(1 for i in indices if records[i]["primary_label"] == cls and preds[i] != cls)
        per_cls[cls] = (tp, fp, fn)

    tp_b = fp_b = fn_b = 0
    for i in indices:
        tb = 1 if records[i]["primary_label"] in ("CONTRADICTED", "UNSUPPORTED") else 0
        yp = preds[i]
        if yp == "UNVERIFIABLE":
            if tb == 1:
                fn_b += 1
            else:
                fp_b += 1
        else:
            pb = 1 if yp in ("CONTRADICTED", "UNSUPPORTED") else 0
            if tb == 1 and pb == 1:
                tp_b += 1
            elif tb == 0 and pb == 1:
                fp_b += 1
            elif tb == 1 and pb == 0:
                fn_b += 1
    return n, correct, per_cls, (tp_b, fp_b, fn_b)


def _metrics_from_summed_stats(
    n_sum: float,
    correct_sum: float,
    cls_sums: Dict[str, List[float]],
    bin_sums: List[float],
) -> Dict[str, float]:
    acc = _safe_div(correct_sum, n_sum)
    f1s: Dict[str, float] = {}
    supports: Dict[str, float] = {}
    for cls in CANONICAL_LABELS:
        tp, fp, fn = cls_sums[cls]
        prec = _safe_div(tp, tp + fp)
        rec = _safe_div(tp, tp + fn)
        f1s[cls] = _safe_div(2.0 * prec * rec, prec + rec)
        supports[cls] = tp + fn
    macro_f1 = sum(f1s.values()) / len(CANONICAL_LABELS) if n_sum > 0 else 0.0
    sup_wf1 = (
        sum((supports[cls] / n_sum) * f1s[cls] for cls in CANONICAL_LABELS)
        if n_sum > 0
        else 0.0
    )
    btp, bfp, bfn = bin_sums
    bprec = _safe_div(btp, btp + bfp)
    brec = _safe_div(btp, btp + bfn)
    bf1 = _safe_div(2.0 * bprec * brec, bprec + brec)
    return {
        "macro_f1_penalized": macro_f1,
        "support_weighted_f1_penalized": sup_wf1,
        "accuracy_penalized": acc,
        "contradicted_f1_penalized": f1s["CONTRADICTED"],
        "unsupported_f1_penalized": f1s["UNSUPPORTED"],
        "binary_f1_penalized": bf1,
    }


def source_clustered_bootstrap_diff(
    records: Sequence[Dict[str, Any]],
    pred_baseline: Sequence[str],
    pred_candidate: Sequence[str],
    metric_keys: Sequence[str] = (
        "macro_f1_penalized",
        "support_weighted_f1_penalized",
        "accuracy_penalized",
        "contradicted_f1_penalized",
        "unsupported_f1_penalized",
        "binary_f1_penalized",
    ),
    source_weights: Optional[Dict[str, float]] = None,
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute task-stratified source_id-clustered paired bootstrap 95% CIs on
    (candidate_metric - baseline_metric).

    When source_weights is provided, applies Horvitz-Thompson stratum weights per source
    cluster; otherwise computes unweighted pilot bootstrap intervals.
    """
    if not (len(records) == len(pred_baseline) == len(pred_candidate)):
        raise ValueError("records and prediction lists must match in length")

    std_indices = [
        i for i, r in enumerate(records) if r["primary_label"] in CANONICAL_LABELS
    ]

    if source_weights is None:
        base_metrics = compute_arm_metrics(
            [records[i] for i in std_indices],
            [pred_baseline[i] for i in std_indices],
        )
        cand_metrics = compute_arm_metrics(
            [records[i] for i in std_indices],
            [pred_candidate[i] for i in std_indices],
        )
        point_base = {k: _extract_metric_value(base_metrics, k) for k in metric_keys}
        point_cand = {k: _extract_metric_value(cand_metrics, k) for k in metric_keys}
    else:
        std_recs = [records[i] for i in std_indices]
        base_w = compute_weighted_three_class_metrics(
            std_recs, [pred_baseline[i] for i in std_indices], source_weights
        )
        cand_w = compute_weighted_three_class_metrics(
            std_recs, [pred_candidate[i] for i in std_indices], source_weights
        )
        point_base = {k: _extract_weighted_metric_value(base_w, k) for k in metric_keys}
        point_cand = {k: _extract_weighted_metric_value(cand_w, k) for k in metric_keys}

    point_deltas = {k: point_cand[k] - point_base[k] for k in metric_keys}

    # Group standard 3-class indices by (task_type, source_id)
    by_task_sources: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    for i in std_indices:
        rec = records[i]
        by_task_sources[str(rec["task_type"])][str(rec["source_id"])].append(i)

    task_clusters: Dict[str, List[Tuple[float, Tuple[Any, Any]]]] = {}
    for task, src_map in sorted(by_task_sources.items()):
        c_list: List[Tuple[float, Tuple[Any, Any]]] = []
        for sid, idxs in sorted(src_map.items()):
            w = float(source_weights[sid]) if source_weights is not None else 1.0
            b_stat = _cluster_sufficient_stats(records, pred_baseline, idxs)
            c_stat = _cluster_sufficient_stats(records, pred_candidate, idxs)
            c_list.append((w, (b_stat, c_stat)))
        task_clusters[task] = c_list

    rng = random.Random(seed)
    boot_deltas: Dict[str, List[float]] = {k: [] for k in metric_keys}

    for _ in range(n_boot):
        bn = 0.0
        bcorr = 0.0
        bcls = {cls: [0.0, 0.0, 0.0] for cls in CANONICAL_LABELS}
        bbin = [0.0, 0.0, 0.0]
        cn = 0.0
        ccorr = 0.0
        ccls = {cls: [0.0, 0.0, 0.0] for cls in CANONICAL_LABELS}
        cbin = [0.0, 0.0, 0.0]

        for task, src_clusters in task_clusters.items():
            chosen = rng.choices(src_clusters, k=len(src_clusters))
            for w, (bst, cst) in chosen:
                n_i, corr_b, cls_b, bin_b = bst
                _, corr_c, cls_c, bin_c = cst
                bn += w * n_i
                bcorr += w * corr_b
                cn += w * n_i
                ccorr += w * corr_c
                for cls in CANONICAL_LABELS:
                    tp, fp, fn = cls_b[cls]
                    bcls[cls][0] += w * tp
                    bcls[cls][1] += w * fp
                    bcls[cls][2] += w * fn
                    tp2, fp2, fn2 = cls_c[cls]
                    ccls[cls][0] += w * tp2
                    ccls[cls][1] += w * fp2
                    ccls[cls][2] += w * fn2
                for j in range(3):
                    bbin[j] += w * bin_b[j]
                    cbin[j] += w * bin_c[j]

        mb = _metrics_from_summed_stats(bn, bcorr, bcls, bbin)
        mc = _metrics_from_summed_stats(cn, ccorr, ccls, cbin)
        for k in metric_keys:
            boot_deltas[k].append(mc[k] - mb[k])

    summary: Dict[str, Dict[str, Any]] = {}
    for k in metric_keys:
        arr = np.asarray(boot_deltas[k], dtype=float)
        low = float(np.percentile(arr, 2.5))
        high = float(np.percentile(arr, 97.5))
        summary[k] = {
            "baseline_point": round(float(point_base[k]), 6),
            "candidate_point": round(float(point_cand[k]), 6),
            "delta_point": round(float(point_deltas[k]), 6),
            "boot_mean_delta": round(float(np.mean(arr)), 6),
            "boot_std_delta": round(float(np.std(arr)), 6),
            "ci95_low": round(low, 6),
            "ci95_high": round(high, 6),
            "n_boot": int(n_boot),
            "excludes_zero": bool(low > 0.0 or high < 0.0),
        }
    return summary


def response_level_bootstrap_diff(
    records: Sequence[Dict[str, Any]],
    pred_baseline: Sequence[str],
    pred_candidate: Sequence[str],
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """
    Task-stratified source_id-clustered paired bootstrap 95% CIs for response-level
    confirmed hallucination rate MAE and GFI Spearman rho differences (candidate - baseline).
    """
    by_resp: Dict[str, List[Tuple[Dict[str, Any], str, str]]] = defaultdict(list)
    for r, pb, pc in zip(records, pred_baseline, pred_candidate):
        by_resp[str(r["response_id"])].append((r, pb, pc))

    resp_by_task_src: Dict[str, Dict[str, List[Tuple[float, float, float, float, float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    all_rows: List[Tuple[float, float, float, float, float, float]] = []

    for resp_id in sorted(by_resp.keys()):
        items = by_resp[resp_id]
        n = len(items)
        sid = str(items[0][0]["source_id"])
        task = str(items[0][0]["task_type"])
        t_hall = sum(
            1 for r, _, _ in items if r["primary_label"] in ("CONTRADICTED", "UNSUPPORTED", MIXED_LABEL)
        ) / n * 100.0
        t_sup = sum(1 for r, _, _ in items if r["primary_label"] == "SUPPORTED_PROXY") / n * 100.0
        b_hall = sum(1 for _, pb, _ in items if pb in ("CONTRADICTED", "UNSUPPORTED")) / n * 100.0
        c_hall = sum(1 for _, _, pc in items if pc in ("CONTRADICTED", "UNSUPPORTED")) / n * 100.0
        b_sup = sum(1 for _, pb, _ in items if pb == "SUPPORTED_PROXY") / n * 100.0
        c_sup = sum(1 for _, _, pc in items if pc == "SUPPORTED_PROXY") / n * 100.0
        row = (t_hall, b_hall, c_hall, t_sup, b_sup, c_sup)
        resp_by_task_src[task][sid].append(row)
        all_rows.append(row)

    def _calc(rows: Sequence[Tuple[float, float, float, float, float, float]]) -> Tuple[float, Optional[float], Optional[float], float, float]:
        th = [x[0] for x in rows]
        bh = [x[1] for x in rows]
        ch = [x[2] for x in rows]
        ts = [x[3] for x in rows]
        bs = [x[4] for x in rows]
        cs = [x[5] for x in rows]
        b_mae = float(np.mean([abs(b - t) for b, t in zip(bh, th)]))
        c_mae = float(np.mean([abs(c - t) for c, t in zip(ch, th)]))
        b_rho = _safe_pearson_and_spearman(ts, bs)["spearman_rho"]
        c_rho = _safe_pearson_and_spearman(ts, cs)["spearman_rho"]
        rho_diff = (c_rho - b_rho) if (c_rho is not None and b_rho is not None) else None
        return c_mae - b_mae, rho_diff, (c_rho if c_rho is not None else None), b_mae, c_mae

    pt_mae_diff, pt_rho_diff, _, pt_b_mae, pt_c_mae = _calc(all_rows)

    task_src_lists = {
        task: list(src_map.values()) for task, src_map in sorted(resp_by_task_src.items())
    }
    rng = random.Random(seed)
    mae_diffs: List[float] = []
    rho_diffs: List[float] = []
    for _ in range(n_boot):
        b_rows: List[Tuple[float, float, float, float, float, float]] = []
        for task, clusters in task_src_lists.items():
            for cl in rng.choices(clusters, k=len(clusters)):
                b_rows.extend(cl)
        md, rd, _, _, _ = _calc(b_rows)
        mae_diffs.append(md)
        if rd is not None:
            rho_diffs.append(rd)

    def _summarize(arr_vals: List[float], pt_val: Optional[float], b_pt: Optional[float] = None, c_pt: Optional[float] = None) -> Dict[str, Any]:
        if not arr_vals:
            return {"delta_point": pt_val, "ci95_low": None, "ci95_high": None, "excludes_zero": False}
        arr = np.asarray(arr_vals, dtype=float)
        low = float(np.percentile(arr, 2.5))
        high = float(np.percentile(arr, 97.5))
        return {
            "baseline_point": round(b_pt, 4) if b_pt is not None else None,
            "candidate_point": round(c_pt, 4) if c_pt is not None else None,
            "delta_point": round(pt_val, 4) if pt_val is not None else None,
            "ci95_low": round(low, 4),
            "ci95_high": round(high, 4),
            "n_boot": len(arr_vals),
            "excludes_zero": bool(low > 0.0 or high < 0.0),
        }

    return {
        "confirmed_hall_mae_pct_delta": _summarize(mae_diffs, pt_mae_diff, pt_b_mae, pt_c_mae),
        "gfi_spearman_rho_delta": _summarize(rho_diffs, pt_rho_diff),
    }
