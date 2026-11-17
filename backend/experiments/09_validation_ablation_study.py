"""
VeriLLM Phase 8 Step 4: Validation Ablation Study Experiment Runner.

Evaluates the existing MiniCheck + NLI pipeline vs. CustomDebertaVerifier alone
vs. candidate non-averaging signal compositions strictly on ragtruth_val.jsonl.

IMPORTANT:
- NEVER opens, reads, or evaluates ragtruth_test.jsonl.
- Supports --sources-per-task (default 5 -> 15 sources total pilot) with
  reusable per-example raw signal caching so future approved expansions
  reuse already-evaluated sources cleanly.
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import hashlib
import json
import logging
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import sklearn
import torch
import transformers

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml_engine.claim_verifier.verifier import ClaimVerifier
from ml_engine.custom_model.validation_ablation import (
    ARM_DESCRIPTIONS,
    ARM_NAMES,
    CANONICAL_LABELS,
    MIXED_LABEL,
    TASK_TYPES,
    apply_holm_bonferroni,
    audit_duplicate_and_leakage,
    build_train_overlap_index,
    clustered_paired_comparison,
    compute_arm_metrics,
    compute_pair_key,
    compute_response_level_evaluation,
    compute_weighted_three_class_metrics,
    evaluate_direct_context_records,
    evaluate_minority_adequacy,
    evaluate_retrieval_pipeline_for_source,
    exact_mcnemar_test,
    load_validation_records,
    predict_arm_for_record,
    response_level_bootstrap_diff,
    select_pilot_sources,
    source_clustered_bootstrap_diff,
    summarize_sources,
)
from ml_engine.evidence_retriever.chunker import SlidingWindowChunker
from ml_engine.evidence_retriever.passage_retriever import PassageRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("verillm.val_ablation")

VAL_DATA_PATH = (
    BACKEND_DIR / "data" / "processed" / "ragtruth_val.jsonl"
)
TRAIN_DATA_PATH = (
    BACKEND_DIR / "data" / "processed" / "ragtruth_train.jsonl"
)
RESULTS_DIR = BACKEND_DIR / "experiments" / "results"
DEFAULT_CACHE_PATH = RESULTS_DIR / "09_val_pilot_raw_cache.jsonl"


def _make_token_length_fn(tokenizer: Any) -> Callable[[Sequence[Tuple[str, str]]], List[int]]:
    """Measure untruncated (context, claim) DeBERTa pair token length cleanly."""
    prev_max = getattr(tokenizer, "model_max_length", 512)

    def _measure(pairs: Sequence[Tuple[str, str]]) -> List[int]:
        lengths: List[int] = []
        # Temporarily raise model_max_length to avoid spurious HF length warnings
        tokenizer.model_max_length = 1_000_000
        try:
            for ctx, claim in pairs:
                if not ctx.strip() and not claim.strip():
                    lengths.append(0)
                    continue
                encoded = tokenizer(
                    ctx,
                    claim,
                    truncation=False,
                    add_special_tokens=True,
                )
                lengths.append(len(encoded["input_ids"]))
        finally:
            tokenizer.model_max_length = prev_max
        return lengths

    return _measure


def _load_raw_cache(cache_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load cached per-example signals keyed by example_id."""
    cache: Dict[str, Dict[str, Any]] = {}
    if not cache_path.exists():
        return cache
    with cache_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            s = line.strip()
            if not s:
                continue
            item = json.loads(s)
            ex_id = str(item["example_id"])
            if "direct" in item and "retrieval" in item:
                cache[ex_id] = item
    return cache


def _append_cache_records(cache_path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def run_warmup_latency_benchmark(
    sample_records: Sequence[Dict[str, Any]],
    claim_verifier: ClaimVerifier,
    chunker: SlidingWindowChunker,
    passage_retriever: PassageRetriever,
    load_timings: Dict[str, float],
    total_target_sentences: int,
) -> Dict[str, Any]:
    """
    Measure actual per-model CPU inference latency on a small warm-up slice after
    one cold-start warm-up pass, and project full pilot + 30-source runtimes.
    """
    custom_deberta = claim_verifier._get_custom_deberta_verifier()
    if custom_deberta is None:
        raise RuntimeError("CustomDebertaVerifier failed to initialize.")

    n_bench = len(sample_records)
    if n_bench < 2:
        raise ValueError("Need at least 2 sample records for latency benchmark.")

    warm_rec = sample_records[0]
    warm_doc = str(warm_rec["context_text"])
    warm_claim = str(warm_rec["sentence_text"])

    # Unmeasured warm-up pass to initialize PyTorch CPU kernels / caches
    claim_verifier.minicheck.score_pairs([warm_doc], [warm_claim])
    claim_verifier.nli.predict_pairs([(warm_doc, warm_claim)])
    custom_deberta.predict_pairs([(warm_doc, warm_claim)])
    warm_chunks = chunker.chunk_plain_text(warm_doc, document_name="warm.txt", page_number=1)
    if warm_chunks:
        passage_retriever.index_chunks(warm_chunks)

    docs = [str(r["context_text"]) for r in sample_records]
    claims = [str(r["sentence_text"]) for r in sample_records]
    pairs = list(zip(docs, claims))

    t0 = time.perf_counter()
    claim_verifier.minicheck.score_pairs(docs, claims)
    mc_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    claim_verifier.nli.predict_pairs(pairs)
    nli_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    custom_deberta.predict_pairs(pairs)
    deb_sec = time.perf_counter() - t0

    # Benchmark chunking + PassageRetriever across unique sources in benchmark batch
    by_src: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in sample_records:
        by_src[str(r["source_id"])].append(r)

    t0 = time.perf_counter()
    from ml_engine.custom_model.validation_ablation import _build_extracted_claim_for_record

    for sid, recs in by_src.items():
        ch = chunker.chunk_plain_text(
            str(recs[0]["context_text"]), document_name=f"bench_{sid}.txt", page_number=1
        )
        ex_claims = [_build_extracted_claim_for_record(r) for r in recs]
        if ch:
            passage_retriever.index_chunks(ch)
            passage_retriever.retrieve_for_claims(ex_claims)
    ret_sec = time.perf_counter() - t0

    mc_ms = (mc_sec / n_bench) * 1000.0
    nli_ms = (nli_sec / n_bench) * 1000.0
    deb_ms = (deb_sec / n_bench) * 1000.0
    ret_ms = (ret_sec / n_bench) * 1000.0

    # End-to-end runs MiniCheck + NLI + CustomDeberta TWICE (Direct + Retrieval top-1), plus Retrieval + sec NLI
    est_ms_per_sentence_dual_regime = 2.0 * (mc_ms + nli_ms + deb_ms) + ret_ms
    est_pilot_minutes = (est_ms_per_sentence_dual_regime * total_target_sentences) / 60_000.0
    est_30src_minutes = est_pilot_minutes * 2.0

    bench = {
        "benchmark_sample_sentences": n_bench,
        "benchmark_sample_sources": len(by_src),
        "cold_load_seconds": {k: round(v, 3) for k, v in load_timings.items()},
        "per_sentence_ms": {
            "minicheck": round(mc_ms, 2),
            "nli_xsmall": round(nli_ms, 2),
            "custom_deberta": round(deb_ms, 2),
            "chunk_and_retrieve": round(ret_ms, 2),
            "dual_regime_combined_estimate": round(est_ms_per_sentence_dual_regime, 2),
        },
        "projected_runtime_minutes": {
            "pilot_current_target": round(est_pilot_minutes, 2),
            "expanded_30_sources_est": round(est_30src_minutes, 2),
        },
    }
    return bench


def _apply_holm_family(
    arms_report: Dict[str, Any],
    candidate_arms: Sequence[str],
) -> None:
    """Apply Holm-Bonferroni correction per family across candidate arms within one regime."""
    if not candidate_arms:
        return

    families: List[Tuple[str, Callable[[Dict[str, Any]], Optional[float]], Callable[[Dict[str, Any], float], None]]] = [
        (
            "source_sign_flip_unweighted_3cls",
            lambda vs: vs["source_clustered_primary"]["three_class_accuracy"]["paired_sign_flip_p_value"],
            lambda vs, adj: vs["source_clustered_primary"]["three_class_accuracy"].__setitem__("holm_bonferroni_p_value", adj),
        ),
        (
            "source_sign_flip_weighted_3cls",
            lambda vs: vs["source_clustered_primary"]["three_class_accuracy"].get("paired_sign_flip_weighted_p_value"),
            lambda vs, adj: vs["source_clustered_primary"]["three_class_accuracy"].__setitem__("holm_bonferroni_weighted_p_value", adj),
        ),
        (
            "source_sign_test_3cls",
            lambda vs: vs["source_clustered_primary"]["three_class_accuracy"]["exact_sign_test_p_value"],
            lambda vs, adj: vs["source_clustered_primary"]["three_class_accuracy"].__setitem__("holm_bonferroni_sign_test_p_value", adj),
        ),
        (
            "source_sign_flip_unweighted_bin",
            lambda vs: vs["source_clustered_primary"]["binary_hallucination_accuracy"]["paired_sign_flip_p_value"],
            lambda vs, adj: vs["source_clustered_primary"]["binary_hallucination_accuracy"].__setitem__("holm_bonferroni_p_value", adj),
        ),
        (
            "response_sign_flip_3cls",
            lambda vs: vs["response_clustered_supplementary"]["three_class_accuracy"]["paired_sign_flip_p_value"],
            lambda vs, adj: vs["response_clustered_supplementary"]["three_class_accuracy"].__setitem__("holm_bonferroni_p_value", adj),
        ),
        (
            "response_wilcoxon_3cls",
            lambda vs: vs["response_clustered_supplementary"]["three_class_accuracy"].get("wilcoxon_signed_rank_p_value"),
            lambda vs, adj: vs["response_clustered_supplementary"]["three_class_accuracy"].__setitem__("holm_bonferroni_wilcoxon_p_value", adj),
        ),
        (
            "mcnemar_legacy_3cls",
            lambda vs: vs["mcnemar_sentence_legacy"]["three_class_exact"]["exact_two_sided_p_value"],
            lambda vs, adj: vs["mcnemar_sentence_legacy"]["three_class_exact"].__setitem__("holm_bonferroni_p_value", adj),
        ),
        (
            "mcnemar_legacy_bin",
            lambda vs: vs["mcnemar_sentence_legacy"]["binary_hallucination_strict"]["exact_two_sided_p_value"],
            lambda vs, adj: vs["mcnemar_sentence_legacy"]["binary_hallucination_strict"].__setitem__("holm_bonferroni_p_value", adj),
        ),
    ]

    for _, getter, setter in families:
        idxs: List[str] = []
        raws: List[float] = []
        for arm in candidate_arms:
            val = getter(arms_report[arm]["vs_baseline"])
            if val is not None:
                idxs.append(arm)
                raws.append(float(val))
        if not raws:
            continue
        adjs = apply_holm_bonferroni(raws)
        for arm, adj in zip(idxs, adjs):
            setter(arms_report[arm]["vs_baseline"], adj)


def _evaluate_regime(
    cached_records: Sequence[Dict[str, Any]],
    regime: str,
    source_weights: Dict[str, float],
    exact_dup_ids: Set[str],
    sent_overlap_ids: Set[str],
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    """Compute all arm tables, slices, sensitivity, response correlation, LLM ranking, and clustered tests."""
    arm_preds: Dict[str, List[str]] = {}
    arm_scores: Dict[str, List[Optional[float]]] = {}
    for arm in ARM_NAMES:
        preds: List[str] = []
        scores: List[Optional[float]] = []
        for rec in cached_records:
            p, s = predict_arm_for_record(rec, regime=regime, arm_name=arm)
            preds.append(p)
            scores.append(s)
        arm_preds[arm] = preds
        arm_scores[arm] = scores

    y_true_all = [r["primary_label"] for r in cached_records]

    arms_report: Dict[str, Any] = {}
    candidate_arms: List[str] = []
    for arm in ARM_NAMES:
        preds = arm_preds[arm]
        scores = arm_scores[arm]
        base_metrics = compute_arm_metrics(cached_records, preds, continuous_scores=scores)
        weighted = compute_weighted_three_class_metrics(
            cached_records, preds, source_weights=source_weights
        )

        # Task slices
        by_task: Dict[str, Any] = {}
        for task in TASK_TYPES:
            idx = [i for i, r in enumerate(cached_records) if r["task_type"] == task]
            by_task[task] = compute_arm_metrics(
                [cached_records[i] for i in idx],
                [preds[i] for i in idx],
                continuous_scores=[scores[i] for i in idx],
            )

        # Length slices measured by Direct Context DeBERTa token count (<=512 vs >512)
        short_idx = [
            i
            for i, r in enumerate(cached_records)
            if r["direct"].get("pair_token_count") is not None
            and not r["direct"].get("exceeds_512_tokens")
        ]
        long_idx = [
            i
            for i, r in enumerate(cached_records)
            if r["direct"].get("exceeds_512_tokens") is True
        ]
        by_length = {
            "le_512_direct_tokens": compute_arm_metrics(
                [cached_records[i] for i in short_idx],
                [preds[i] for i in short_idx],
                continuous_scores=[scores[i] for i in short_idx],
            ),
            "gt_512_direct_tokens": compute_arm_metrics(
                [cached_records[i] for i in long_idx],
                [preds[i] for i in long_idx],
                continuous_scores=[scores[i] for i in long_idx],
            ),
        }

        # Special annotation flags (implicit_true, due_to_null)
        imp_idx = [i for i, r in enumerate(cached_records) if r.get("has_implicit_true")]
        null_idx = [i for i, r in enumerate(cached_records) if r.get("has_due_to_null")]
        by_flags = {
            "has_implicit_true": compute_arm_metrics(
                [cached_records[i] for i in imp_idx],
                [preds[i] for i in imp_idx],
                continuous_scores=[scores[i] for i in imp_idx],
            ),
            "has_due_to_null": compute_arm_metrics(
                [cached_records[i] for i in null_idx],
                [preds[i] for i in null_idx],
                continuous_scores=[scores[i] for i in null_idx],
            ),
        }

        # Sensitivity removing exact train-val (context, sentence) duplicates
        no_exact_dup_idx = [
            i for i, r in enumerate(cached_records) if r["example_id"] not in exact_dup_ids
        ]
        no_sent_ovlp_idx = [
            i for i, r in enumerate(cached_records) if r["example_id"] not in sent_overlap_ids
        ]
        sensitivity = {
            "excluding_exact_train_pair_duplicates": {
                "excluded_count": len(cached_records) - len(no_exact_dup_idx),
                "metrics": compute_arm_metrics(
                    [cached_records[i] for i in no_exact_dup_idx],
                    [preds[i] for i in no_exact_dup_idx],
                    continuous_scores=[scores[i] for i in no_exact_dup_idx],
                ),
            },
            "excluding_sentence_string_in_train": {
                "excluded_count": len(cached_records) - len(no_sent_ovlp_idx),
                "metrics": compute_arm_metrics(
                    [cached_records[i] for i in no_sent_ovlp_idx],
                    [preds[i] for i in no_sent_ovlp_idx],
                    continuous_scores=[scores[i] for i in no_sent_ovlp_idx],
                ),
            },
        }

        response_eval = compute_response_level_evaluation(cached_records, preds)

        stats_vs_baseline: Optional[Dict[str, Any]] = None
        if arm != "arm1_baseline":
            candidate_arms.append(arm)
            src_cmp = clustered_paired_comparison(
                cached_records,
                arm_preds["arm1_baseline"],
                preds,
                cluster_key="source_id",
                source_weights=source_weights,
                n_perm=20000,
                seed=seed,
            )
            resp_cmp = clustered_paired_comparison(
                cached_records,
                arm_preds["arm1_baseline"],
                preds,
                cluster_key="response_id",
                source_weights=None,
                n_perm=20000,
                seed=seed,
            )
            boot_unweighted = source_clustered_bootstrap_diff(
                cached_records,
                arm_preds["arm1_baseline"],
                preds,
                source_weights=None,
                n_boot=n_boot,
                seed=seed,
            )
            boot_weighted = source_clustered_bootstrap_diff(
                cached_records,
                arm_preds["arm1_baseline"],
                preds,
                source_weights=source_weights,
                n_boot=n_boot,
                seed=seed,
            )
            resp_boot = response_level_bootstrap_diff(
                cached_records,
                arm_preds["arm1_baseline"],
                preds,
                n_boot=n_boot,
                seed=seed,
            )
            mcn = exact_mcnemar_test(y_true_all, arm_preds["arm1_baseline"], preds)
            stats_vs_baseline = {
                "source_clustered_primary": src_cmp,
                "source_clustered_bootstrap_unweighted": boot_unweighted,
                "source_clustered_bootstrap_stratum_weighted": boot_weighted,
                "response_level_bootstrap": resp_boot,
                "response_clustered_supplementary": resp_cmp,
                "mcnemar_sentence_legacy": mcn,
            }

        arms_report[arm] = {
            "description": ARM_DESCRIPTIONS[arm],
            "overall": base_metrics,
            "prevalence_reweighted": weighted,
            "by_task": by_task,
            "by_direct_token_length": by_length,
            "by_annotation_flags": by_flags,
            "duplicate_sensitivity": sensitivity,
            "response_level": response_eval,
            "vs_baseline": stats_vs_baseline,
        }

    _apply_holm_family(arms_report, candidate_arms)

    # Disagreement matrix between arm1_baseline and arm2_custom_deberta_only (3-class examples)
    disagreement_cells: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "true_counts": Counter()}
    )
    for rec, b_pred, d_pred in zip(
        cached_records,
        arm_preds["arm1_baseline"],
        arm_preds["arm2_custom_deberta_only"],
    ):
        if rec["primary_label"] not in CANONICAL_LABELS:
            continue
        key = f"baseline={b_pred}__deberta={d_pred}"
        disagreement_cells[key]["count"] += 1
        disagreement_cells[key]["true_counts"][rec["primary_label"]] += 1

    serialized_disagreement = {
        k: {
            "count": v["count"],
            "true_counts": dict(v["true_counts"]),
        }
        for k, v in sorted(disagreement_cells.items(), key=lambda kv: -kv[1]["count"])
    }

    # Regime-specific context/retrieval diagnostics
    direct_tok = [
        r["direct"]["pair_token_count"]
        for r in cached_records
        if r["direct"].get("pair_token_count") is not None
    ]
    ret_tok = [
        r["retrieval"]["pair_token_count"]
        for r in cached_records
        if r["retrieval"].get("pair_token_count") is not None
    ]
    ret_sims = [
        float(r["retrieval"]["best_similarity_score"]) for r in cached_records
    ]
    diagnostics = {
        "direct_pairs_total": len(cached_records),
        "direct_exceeds_512_tokens_count": sum(1 for t in direct_tok if t > 512),
        "direct_exceeds_512_tokens_pct": round(
            (sum(1 for t in direct_tok if t > 512) / len(direct_tok) * 100.0)
            if direct_tok
            else 0.0,
            2,
        ),
        "direct_token_length_median": float(np.median(direct_tok)) if direct_tok else None,
        "direct_token_length_p95": float(np.percentile(direct_tok, 95)) if direct_tok else None,
        "retrieval_top1_exceeds_512_tokens_count": sum(1 for t in ret_tok if t > 512),
        "retrieval_mean_similarity": round(float(np.mean(ret_sims)), 4) if ret_sims else None,
        "retrieval_median_similarity": round(float(np.median(ret_sims)), 4) if ret_sims else None,
        "retrieval_below_floor_0_25_count": sum(1 for s in ret_sims if s < 0.25),
        "retrieval_secondary_conflict_triggered_count": sum(
            1 for r in cached_records if r["retrieval"].get("conflict_triggered")
        ),
    }

    return {
        "regime": regime,
        "diagnostics": diagnostics,
        "baseline_vs_deberta_agreement": serialized_disagreement,
        "arms": arms_report,
    }


def _collect_git_metadata() -> Dict[str, Any]:
    """Capture current repository commit and cleanliness without modifying git state."""
    repo_root = BACKEND_DIR.parent
    commit = None
    dirty_count = None
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        status_out = subprocess.check_output(
            ["git", "status", "--short"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        dirty_lines = [ln for ln in status_out.splitlines() if ln.strip()]
        dirty_count = len(dirty_lines)
    except Exception:
        pass
    return {
        "commit": commit,
        "is_dirty": (dirty_count > 0) if dirty_count is not None else None,
        "dirty_entries_count": dirty_count,
    }


def _collect_checkpoint_hashes() -> Dict[str, Optional[str]]:
    """Compute SHA-256 hashes of local Custom DeBERTa artifact files (read-only)."""
    ckpt_dir = BACKEND_DIR / "models" / "verillm_deberta"
    files = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")
    out: Dict[str, Optional[str]] = {}
    for fname in files:
        fpath = ckpt_dir / fname
        if not fpath.exists():
            out[fname] = None
            continue
        h = hashlib.sha256()
        with fpath.open("rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
        out[fname] = h.hexdigest()
    return out


def _fmt_p(val: Optional[float], sci: Optional[str] = None) -> str:
    if sci:
        return sci
    if val is None:
        return "N/A"
    if val == 0.0:
        return "0 (<1e-308)"
    if val < 1e-4:
        return f"{val:.4e}"
    return f"{val:.4g}"


def _fmt_ci_pp(block: Dict[str, Any]) -> str:
    return (
        f"{block['delta_point']*100:+.2f}pp "
        f"[{block['ci95_low']*100:+.2f}, {block['ci95_high']*100:+.2f}]"
    )


def _fmt_slice_macro(metric_block: Dict[str, Any]) -> str:
    p3 = metric_block["penalized_3class"]
    if p3.get("n", metric_block.get("n_total_3class", 0)) == 0:
        return "N/A (n=0)"
    present_cnt = p3.get("present_classes_count", 3)
    if present_cnt < 3:
        pres = p3.get("macro_f1_present_classes")
        pres_s = f"{pres*100:.2f}%" if pres is not None else "N/A"
        return f"{p3['macro_f1']*100:.2f}% (pres:{pres_s}, {present_cnt}/3 cls)"
    return f"{p3['macro_f1']*100:.2f}%"


def _render_markdown_report(results: Dict[str, Any]) -> str:
    manifest = results["manifest"]
    meta = results["metadata"]
    bench = results.get("warmup_benchmark")
    lines: List[str] = []
    title_tag = "Corrected Cache-Only Analysis" if meta.get("cache_only") else "Validation Ablation Report"
    lines.append(
        f"# Phase 8 — {title_tag} ({manifest['total_sources']} Validation Sources)"
    )
    lines.append("")
    lines.append(f"- **Generated UTC**: `{meta['generated_utc']}` (cache_only=`{meta.get('cache_only', False)}`)")
    lines.append(
        "- **Dataset**: `ragtruth_val.jsonl` ONLY (`ragtruth_test.jsonl` untouched)"
    )
    git_meta = meta.get("git", {})
    vers = meta.get("versions", {})
    lines.append(
        f"- **Code & Runtime Provenance**: git=`{git_meta.get('commit')}` "
        f"(dirty=`{git_meta.get('is_dirty')}`), device=`{meta.get('device')}`, "
        f"python=`{vers.get('python')}`, torch=`{vers.get('torch')}`, "
        f"transformers=`{vers.get('transformers')}`, sklearn=`{vers.get('sklearn')}`, numpy=`{vers.get('numpy')}`"
    )
    models_meta = meta.get("model_identifiers", {})
    hashes_meta = meta.get("custom_deberta_checkpoint_sha256", {})
    short_hashes = {k: (v[:16] if v else None) for k, v in hashes_meta.items()}
    lines.append(
        f"- **Models Evaluated (from cache)**: MiniCheck=`{models_meta.get('minicheck')}`, "
        f"NLI=`{models_meta.get('nli')}`, Retriever=`{models_meta.get('retriever')}`, "
        f"CustomDeBERTa=`{models_meta.get('custom_deberta')}` (sha256[:16]=`{json.dumps(short_hashes)}`)"
    )
    lines.append(
        f"- **Manifest & Selection Provenance**: manifest=`{meta.get('manifest_path')}`, "
        f"locked_manifest=`{meta.get('locked_manifest_path')}`, "
        f"method=`{manifest['selection_method']}`, seed=`{manifest['seed']}`, "
        f"bootstrap_seed=`{meta['seed']}`, n_boot=`{meta['n_boot']}`, resampling_unit=`{meta.get('bootstrap_resampling_unit')}`"
    )
    lines.append(f"- **Selection Rationale**: {manifest['selection_rationale']}")
    lines.append(
        f"- **Pilot Scope**: `{manifest['total_sources']}` sources, "
        f"`{manifest['total_responses']}` complete responses, "
        f"`{manifest['total_sentences']}` total rows "
        f"(`{manifest['final_adequacy']['counts_by_label']['SUPPORTED_PROXY'] + manifest['final_adequacy']['counts_by_label']['CONTRADICTED'] + manifest['final_adequacy']['counts_by_label']['UNSUPPORTED']}` 3-class + "
        f"`{manifest['final_adequacy']['counts_by_label'][MIXED_LABEL]}` `{MIXED_LABEL}` diagnostic)"
    )
    lines.append("")
    lines.append(
        "> **Methodological Scope & Caveats**: (1) **Exploratory inference only**: `30` validation sources provide "
        "exploratory precision rather than definitive population-level confidence intervals, and some selection strata "
        "contain only `2–4` sources (high leverage up to `2.40x`), making stratum-weighted bootstrap CIs unstable. "
        "(2) **Primary vs. supplementary clustering**: `source_id` (`K=30`) is the primary independent unit; response-clustered "
        "(`R=177`) checks are supplementary (responses share a source), and sentence-level McNemar (`n=1,250`) is legacy and "
        "anti-conservative because it ignores intra-source/intra-response correlation. "
        "(3) **Uncalibrated DeBERTa outputs & `arm3a` warning**: CustomDeBERTa's class-weighted softmax outputs are "
        "**uncalibrated normalized logits**, not calibrated probabilities; fixed legacy NLI thresholds inside `arm3a_mc_plus_custom_deberta` "
        "are unvalidated on these outputs, so `arm3a` must NOT be used to justify a production decision prior to out-of-fold calibration. "
        "(4) **Retrieval `NLI-xsmall` dependency**: in Condition 2 (`retrieval`), `ClaimVerifier`'s Rule 4 secondary-evidence conflict "
        "resolution relies on `NLI-xsmall` (`cross-encoder/nli-deberta-v3-xsmall`), not a pure NLI-free retriever."
    )
    lines.append("")

    lines.append("## 1. Pilot Class Distribution, Stratum Weights & Adequacy Audit")
    lines.append("")
    adeq = manifest["final_adequacy"]
    lines.append(
        "| Slice | Total | SUPPORTED_PROXY | CONTRADICTED | UNSUPPORTED | MIXED_HALLUCINATION |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    full_c = manifest["full_validation_distribution"]["counts_by_label"]
    lines.append(
        f"| Full `ragtruth_val.jsonl` ({manifest['full_validation_distribution']['total_sources']} src) | "
        f"{manifest['full_validation_distribution']['total_sentences']} | "
        f"{full_c['SUPPORTED_PROXY']} | {full_c['CONTRADICTED']} | {full_c['UNSUPPORTED']} | {full_c[MIXED_LABEL]} |"
    )
    pc = adeq["counts_by_label"]
    lines.append(
        f"| **Selected Pilot** ({manifest['total_sources']} src) | {manifest['total_sentences']} | "
        f"{pc['SUPPORTED_PROXY']} | {pc['CONTRADICTED']} | {pc['UNSUPPORTED']} | {pc[MIXED_LABEL]} |"
    )
    for task in TASK_TYPES:
        tc = adeq["counts_by_task_and_label"][task]
        lines.append(
            f"| Pilot `{task}` | {tc['sentences']} | {tc['SUPPORTED_PROXY']} | "
            f"{tc['CONTRADICTED']} | {tc['UNSUPPORTED']} | {tc[MIXED_LABEL]} |"
        )
    lines.append("")

    stratum_summary = manifest.get("stratum_sampling_summary") or {}
    stratum_rows = (
        stratum_summary.get("stratum_counts", [])
        if isinstance(stratum_summary, dict)
        else []
    )
    if stratum_rows:
        lines.append("### Stratum Sampling Summary & Horvitz-Thompson Weights")
        lines.append("")
        lines.append("| Task | Stratum | Val Sources | Pilot Sources | Rel. Stratum Weight | Small-Cell Flag |")
        lines.append("|---|---|---:|---:|---:|---|")
        for row in stratum_rows:
            n_sel = int(row["sample_sources"])
            flag = "SMALL STRATUM (<5)" if n_sel < 5 else "ok"
            lines.append(
                f"| `{row['task_type']}` | `{row['design_stratum']}` | {row['population_sources']} | "
                f"{n_sel} | {row['relative_stratum_weight']:.4f} | {flag} |"
            )
        lines.append("")

    dup = manifest["duplicate_and_leakage_audit"]
    lines.append("## 2. Train-Validation Leakage & Duplicate Separation")
    lines.append("")
    lines.append(f"- `source_id` overlap with `ragtruth_train.jsonl`: **{dup['source_id_overlap_count']}**")
    lines.append(
        f"- Exact `(context_text, sentence_text)` duplicates in train: **{dup['exact_pair_duplicate_with_train_count']}** "
        f"(IDs: `{dup['exact_pair_duplicate_example_ids']}`)"
    )
    lines.append(
        f"- Repeated `sentence_text` strings appearing in train (total / string-only diff context): "
        f"**{dup['sentence_string_in_train_total']}** / **{dup['sentence_string_only_overlap_count']}**"
    )
    lines.append(
        f"- Within-pilot extra pair duplicates / repeated sentence strings: "
        f"**{dup['within_sample_extra_pair_duplicates']}** / **{dup['within_sample_repeated_sentence_extra']}**"
    )
    lines.append("")

    if bench:
        lines.append("## 3. Recorded Warm-Up Latency Benchmark & 30-Source Runtime")
        lines.append("")
        ps = bench["per_sentence_ms"]
        combined_ms = float(ps.get("dual_regime_combined_estimate", 0.0))
        scaled_total_min = round((combined_ms * manifest["total_sentences"]) / 60_000.0, 2)
        wall_sec = meta.get("inference_wall_seconds") or 0.0
        wall_min = round(float(wall_sec) / 60.0, 2)
        lines.append(
            f"- Cold model load seconds (recorded during initial Phase 8 run): `{bench['cold_load_seconds']}`"
        )
        lines.append(
            f"- Measured on warm-up slice (`{bench['benchmark_sample_sentences']}` sentences, "
            f"`{bench['benchmark_sample_sources']}` sources): "
            f"MiniCheck={ps['minicheck']} ms/pair, "
            f"NLI-xsmall={ps['nli_xsmall']} ms/pair, "
            f"CustomDeBERTa={ps['custom_deberta']} ms/pair, "
            f"Chunk+Retrieve={ps['chunk_and_retrieve']} ms/pair, "
            f"Dual-regime estimate={combined_ms} ms/sentence"
        )
        lines.append(
            f"- Scaled dual-regime estimate for `{manifest['total_sentences']}` sentences (`{manifest['total_sources']}` sources): "
            f"**{scaled_total_min} min** (actual tracked inference wall time across 30 sources: **{wall_sec}s = {wall_min} min**; "
            f"stale 15-source benchmark entry was `{bench['projected_runtime_minutes']['pilot_current_target']} min`)."
        )
        lines.append("")

    for regime_key, title in (
        ("direct", "4. Condition 1 — Direct Source Context Ablation"),
        ("retrieval", "5. Condition 2 — Chunked Retrieval End-to-End Ablation"),
    ):
        reg = results["regimes"][regime_key]
        lines.append(f"## {title}")
        lines.append("")
        diag = reg["diagnostics"]
        lines.append(
            f"- Diagnostics: Direct pairs >512 tokens = `{diag['direct_exceeds_512_tokens_count']}` "
            f"({diag['direct_exceeds_512_tokens_pct']}%), "
            f"Retrieval top-1 >512 tokens = `{diag['retrieval_top1_exceeds_512_tokens_count']}`, "
            f"Retrieval mean sim = `{diag['retrieval_mean_similarity']}`, "
            f"<0.25 floor = `{diag['retrieval_below_floor_0_25_count']}`, "
            f"Rule 4 `NLI-xsmall` secondary-conflict triggers = `{diag['retrieval_secondary_conflict_triggered_count']}`"
        )
        if regime_key == "retrieval":
            lines.append(
                "- **Retrieval Caveat**: Rule 4 secondary-evidence conflict resolution inside `ClaimVerifier` invokes `NLI-xsmall` "
                "(`cross-encoder/nli-deberta-v3-xsmall`); candidate arms (`arm3a` uses uncalibrated DeBERTa logits with legacy NLI thresholds) "
                "do not represent a calibrated or NLI-free retrieval pipeline."
            )
        lines.append("")

        lines.append(f"### A. Claim-Level 3-Class & Binary Detection ({regime_key})")
        lines.append("")
        lines.append(
            "| Arm | Coverage | Pen. Macro-F1 | Sup-Weighted F1 | Sel. Macro-F1 | Pen. Acc | CONTRA F1 | SUP_P F1 | UNSUP F1 | Bin-F1 (Strict) | AUROC | PR-AUC |"
        )
        lines.append(
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        )
        for arm in ARM_NAMES:
            a = reg["arms"][arm]
            ov = a["overall"]
            p3 = ov["penalized_3class"]
            s3 = ov["selective_3class"]
            pc_map = p3["per_class"]
            bp = ov["binary_hallucination_penalized"]
            auroc_s = f"{bp['auroc']:.3f}" if bp["auroc"] is not None else "N/A (discrete)"
            prauc_s = f"{bp['pr_auc']:.3f}" if bp["pr_auc"] is not None else "N/A (discrete)"
            sw_f1 = p3.get("support_weighted_f1", 0.0)
            lines.append(
                f"| `{arm}` | {ov['coverage']*100:.1f}% ({ov['n_covered']}/{ov['n_total_3class']}) | "
                f"{p3['macro_f1']*100:.2f}% | {sw_f1*100:.2f}% | {s3['macro_f1']*100:.2f}% | "
                f"{p3['accuracy']*100:.2f}% | "
                f"{pc_map['CONTRADICTED']['f1']*100:.2f}% | "
                f"{pc_map['SUPPORTED_PROXY']['f1']*100:.2f}% | "
                f"{pc_map['UNSUPPORTED']['f1']*100:.2f}% | "
                f"{bp['penalized_strict']['f1']*100:.2f}% | "
                f"{auroc_s} | {prauc_s} |"
            )
        lines.append("")

        lines.append(f"### B. Per-Task, HT-Reweighted & Length/Duplicate Slices ({regime_key})")
        lines.append("")
        lines.append(
            "| Arm | Summary Macro-F1 (Acc) | QA Macro-F1 (Acc) | Data2txt Macro-F1 (Acc) | HT-Weighted Macro-F1 (Sup-W F1) | <=512 tok Macro-F1 | >512 tok Macro-F1 | No Exact Train-Dup Macro-F1 |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for arm in ARM_NAMES:
            a = reg["arms"][arm]
            bt = a["by_task"]
            rw = a["prevalence_reweighted"]
            bl = a["by_direct_token_length"]
            nd = a["duplicate_sensitivity"]["excluding_exact_train_pair_duplicates"]["metrics"]
            rw_macro = rw.get("weighted_macro_f1_penalized", 0.0)
            rw_swf1 = rw.get("weighted_support_weighted_f1_penalized", 0.0)
            lines.append(
                f"| `{arm}` | "
                f"{bt['Summary']['penalized_3class']['macro_f1']*100:.2f}% ({bt['Summary']['penalized_3class']['accuracy']*100:.1f}%) | "
                f"{bt['QA']['penalized_3class']['macro_f1']*100:.2f}% ({bt['QA']['penalized_3class']['accuracy']*100:.1f}%) | "
                f"{bt['Data2txt']['penalized_3class']['macro_f1']*100:.2f}% ({bt['Data2txt']['penalized_3class']['accuracy']*100:.1f}%) | "
                f"{rw_macro*100:.2f}% ({rw_swf1*100:.2f}%) | "
                f"{_fmt_slice_macro(bl['le_512_direct_tokens'])} | "
                f"{_fmt_slice_macro(bl['gt_512_direct_tokens'])} | "
                f"{_fmt_slice_macro(nd)} |"
            )
        lines.append("")

        lines.append(f"### C. Per-Class P / R / F1 & 3x4 Confusion Matrices ({regime_key})")
        lines.append("")
        lines.append("| Arm | Class | P | R | F1 | Support | Pred: CONTRA | Pred: SUP_P | Pred: UNSUP | Pred: UNVER |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for arm in ARM_NAMES:
            p3 = reg["arms"][arm]["overall"]["penalized_3class"]
            for cls in CANONICAL_LABELS:
                pc = p3["per_class"][cls]
                cm = p3["confusion_3x4"][cls]
                lines.append(
                    f"| `{arm}` | `{cls}` | "
                    f"{pc['precision']*100:.2f}% | {pc['recall']*100:.2f}% | {pc['f1']*100:.2f}% | {pc['support']} | "
                    f"{cm['CONTRADICTED']} | {cm['SUPPORTED_PROXY']} | {cm['UNSUPPORTED']} | {cm['UNVERIFIABLE']} |"
                )
        lines.append("")

        lines.append(f"### D. Primary Paired Comparisons vs `arm1_baseline` — Source-Clustered (`K=30`) ({regime_key})")
        lines.append("")
        lines.append(
            "Family-wise **Holm–Bonferroni** adjustment (`m=6` candidate arms) applied per test family within this regime."
        )
        lines.append("")
        lines.append(
            "| Candidate Arm | Unw. Δ 3-Cls Acc | Clusters (+/-/=) | Unw. Sign-Flip Raw p | Unw. Sign-Flip Holm p | Exact Sign Raw p (Holm) | Wt. Δ 3-Cls Acc | Wt. Sign-Flip Raw p | Wt. Sign-Flip Holm p | Bin Acc Δ (Unw. Holm p) |"
        )
        lines.append("|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|")
        for arm in ARM_NAMES:
            if arm == "arm1_baseline":
                continue
            vs = reg["arms"][arm]["vs_baseline"]
            sp3 = vs["source_clustered_primary"]["three_class_accuracy"]
            spb = vs["source_clustered_primary"]["binary_hallucination_accuracy"]
            lines.append(
                f"| `{arm}` | "
                f"{sp3['mean_cluster_delta_unweighted']*100:+.2f}pp | "
                f"{sp3['positive_clusters']}/{sp3['negative_clusters']}/{sp3['tied_clusters']} | "
                f"{_fmt_p(sp3['paired_sign_flip_p_value'])} | "
                f"{_fmt_p(sp3.get('holm_bonferroni_p_value'))} | "
                f"{_fmt_p(sp3['exact_sign_test_p_value'], sp3.get('exact_sign_test_p_sci'))} ({_fmt_p(sp3.get('holm_bonferroni_sign_test_p_value'))}) | "
                f"{(sp3.get('mean_cluster_delta_weighted') or 0.0)*100:+.2f}pp | "
                f"{_fmt_p(sp3.get('paired_sign_flip_weighted_p_value'))} | "
                f"{_fmt_p(sp3.get('holm_bonferroni_weighted_p_value'))} | "
                f"{spb['mean_cluster_delta_unweighted']*100:+.2f}pp ({_fmt_p(spb.get('holm_bonferroni_p_value'))}) |"
            )
        lines.append("")

        lines.append(f"#### Source-Clustered Bootstrap 95% CIs — Unweighted (`K=30`, Primary Exploratory) ({regime_key})")
        lines.append("")
        lines.append(
            "| Candidate Arm | Δ Macro-F1 (95% CI) | Δ Sup-Weighted F1 (95% CI) | Δ Pen. Acc (95% CI) | Δ CONTRA F1 (95% CI) | Δ UNSUP F1 (95% CI) | Δ Bin-F1 (95% CI) |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for arm in ARM_NAMES:
            if arm == "arm1_baseline":
                continue
            bu = reg["arms"][arm]["vs_baseline"]["source_clustered_bootstrap_unweighted"]
            lines.append(
                f"| `{arm}` | "
                f"{_fmt_ci_pp(bu['macro_f1_penalized'])} | "
                f"{_fmt_ci_pp(bu['support_weighted_f1_penalized'])} | "
                f"{_fmt_ci_pp(bu['accuracy_penalized'])} | "
                f"{_fmt_ci_pp(bu['contradicted_f1_penalized'])} | "
                f"{_fmt_ci_pp(bu['unsupported_f1_penalized'])} | "
                f"{_fmt_ci_pp(bu['binary_f1_penalized'])} |"
            )
        lines.append("")

        lines.append(f"#### Source-Clustered Bootstrap 95% CIs — Stratum-Weighted (Exploratory & Small-Stratum Unstable) ({regime_key})")
        lines.append("")
        lines.append(
            "> **Small-Stratum Instability Note**: Reweighting by validation stratum populations amplifies small strata "
            "(`Data2txt/NO_CONTRA` `n=2`, `QA/NO_CONTRA` `n=3`, `Summary/CONTRA_PRESENT` `n=4`, max relative weight `2.40`), "
            "widening bootstrap variance. Do not treat these 30-source weighted intervals as definitive population CIs."
        )
        lines.append("")
        lines.append(
            "| Candidate Arm | Δ Macro-F1 (95% CI) | Δ Sup-Weighted F1 (95% CI) | Δ Pen. Acc (95% CI) | Δ CONTRA F1 (95% CI) | Δ UNSUP F1 (95% CI) | Δ Bin-F1 (95% CI) |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for arm in ARM_NAMES:
            if arm == "arm1_baseline":
                continue
            bw = reg["arms"][arm]["vs_baseline"]["source_clustered_bootstrap_stratum_weighted"]
            lines.append(
                f"| `{arm}` | "
                f"{_fmt_ci_pp(bw['macro_f1_penalized'])} | "
                f"{_fmt_ci_pp(bw['support_weighted_f1_penalized'])} | "
                f"{_fmt_ci_pp(bw['accuracy_penalized'])} | "
                f"{_fmt_ci_pp(bw['contradicted_f1_penalized'])} | "
                f"{_fmt_ci_pp(bw['unsupported_f1_penalized'])} | "
                f"{_fmt_ci_pp(bw['binary_f1_penalized'])} |"
            )
        lines.append("")

        lines.append(f"### E. Supplementary Response-Clustered (`R=177`) & Legacy Sentence McNemar ({regime_key})")
        lines.append("")
        lines.append(
            "Response-clustered (`R=177`) tests are supplementary because responses sharing a `source_id` are correlated. "
            "Sentence-level McNemar (`n=1,250`) assumes independent sentences and is **anti-conservative** (kept only for legacy comparison)."
        )
        lines.append("")
        lines.append(
            "| Candidate Arm | Resp Δ 3-Cls Acc | Resp Sign-Flip Holm p | Resp Wilcoxon Holm p | Resp Boot Δ Conf-MAE (95% CI) | Resp Boot Δ GFI-ρ (95% CI) | Legacy McNemar 3-Cls (b/c, net) | Legacy McNemar 3-Cls Raw p (Holm p) | Legacy McNemar Bin Raw p (Holm p) |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for arm in ARM_NAMES:
            if arm == "arm1_baseline":
                continue
            vs = reg["arms"][arm]["vs_baseline"]
            rc3 = vs["response_clustered_supplementary"]["three_class_accuracy"]
            rb = vs["response_level_bootstrap"]
            mae_b = rb["confirmed_hall_mae_pct_delta"]
            rho_b = rb["gfi_spearman_rho_delta"]
            m3 = vs["mcnemar_sentence_legacy"]["three_class_exact"]
            mb = vs["mcnemar_sentence_legacy"]["binary_hallucination_strict"]
            lines.append(
                f"| `{arm}` | "
                f"{rc3['mean_cluster_delta_unweighted']*100:+.2f}pp | "
                f"{_fmt_p(rc3.get('holm_bonferroni_p_value'))} | "
                f"{_fmt_p(rc3.get('holm_bonferroni_wilcoxon_p_value'))} | "
                f"{mae_b['delta_point']:+.2f}pp [{mae_b['ci95_low']:+.2f}, {mae_b['ci95_high']:+.2f}] | "
                f"{rho_b['delta_point']:+.3f} [{rho_b['ci95_low']:+.3f}, {rho_b['ci95_high']:+.3f}] | "
                f"{m3['baseline_only_correct']}/{m3['candidate_only_correct']} ({m3['net_gain_cand_minus_base']:+d}, {m3['direction']}) | "
                f"{_fmt_p(m3['exact_two_sided_p_value'], m3.get('p_value_sci'))} ({_fmt_p(m3.get('holm_bonferroni_p_value'))}) | "
                f"{_fmt_p(mb['exact_two_sided_p_value'], mb.get('p_value_sci'))} ({_fmt_p(mb.get('holm_bonferroni_p_value'))}) |"
            )
        lines.append("")

        lines.append(f"### F. Response-Level & Generator-LLM (`6` LLMs) Evaluation ({regime_key})")
        lines.append("")
        lines.append(
            "Confirmed hallucination rate `(CONTRADICTED + UNSUPPORTED) / N` is reported separately from abstained `UNVERIFIABLE / N`; "
            "their sum is `(100 - GFI)%` (`non_supported_rate`)."
        )
        lines.append("")
        lines.append(
            "| Arm | Resp Cov | True Conf Hall % | Pred Conf Hall % | Conf MAE (pp) | Conf Spearman ρ | Pred (100-GFI) % | Pred UNVER % | Excl-Mixed MAE (ρ) | 6-LLM Mean-Hall ρ | 6-LLM Pooled ρ |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for arm in ARM_NAMES:
            re_arm = reg["arms"][arm]["response_level"]
            ch = re_arm["confirmed_hallucination_rate_pct"]
            ns = re_arm["non_supported_rate_100_minus_gfi_pct"]
            uv = re_arm["unverifiable_rate_pct"]
            rc = re_arm["response_coverage"]
            em = ch["excl_mixed_true_sensitivity"]
            gl = re_arm["by_generator_model"]
            rho_ch = f"{ch['spearman_rho']:.3f}" if ch["spearman_rho"] is not None else "N/A"
            em_s = (
                f"{em['mae']:.2f} ({em['spearman_rho']:.3f})"
                if em.get("mae") is not None and em.get("spearman_rho") is not None
                else "N/A"
            )
            rho_m = (
                f"{gl['ranking_response_mean_hall']['spearman_rho']:.3f}"
                if gl["ranking_response_mean_hall"]["spearman_rho"] is not None
                else "N/A"
            )
            rho_p = (
                f"{gl['ranking_pooled_hall']['spearman_rho']:.3f}"
                if gl["ranking_pooled_hall"]["spearman_rho"] is not None
                else "N/A"
            )
            lines.append(
                f"| `{arm}` | {rc['mean_response_coverage_pct']:.1f}% | {ch['mean_true_incl_mixed_pct']:.2f}% | "
                f"{ch['mean_pred_confirmed_pct']:.2f}% | "
                f"{ch['mae']:.2f} | {rho_ch} | {ns['mean_pred_pct']:.2f}% | {uv['mean_pred_pct']:.2f}% | "
                f"{em_s} | {rho_m} | {rho_p} |"
            )
        lines.append("")

        lines.append(f"#### Generator-LLM Breakdown (`arm1_baseline` vs `arm2_custom_deberta_only`) ({regime_key})")
        lines.append("")
        lines.append(
            "| Generator Model | Resp (Sent) | True Hall Mean % | Baseline Conf Mean % (GFI %) | Baseline UNVER Pooled % | DeBERTa Conf Mean % (GFI %) | DeBERTa UNVER Pooled % |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        base_pm = reg["arms"]["arm1_baseline"]["response_level"]["by_generator_model"]["per_model"]
        deb_pm = reg["arms"]["arm2_custom_deberta_only"]["response_level"]["by_generator_model"]["per_model"]
        for model_name in sorted(base_pm.keys()):
            bm = base_pm[model_name]
            dm = deb_pm[model_name]
            lines.append(
                f"| `{model_name}` | {bm['num_responses']} ({bm['num_claims']}) | "
                f"{bm['true_hall_rate_response_mean_pct']:.2f}% | "
                f"{bm['pred_confirmed_hall_rate_response_mean_pct']:.2f}% ({bm['pred_gfi_response_mean_pct']:.1f}%) | "
                f"{bm['pred_unverifiable_rate_pooled_pct']:.2f}% | "
                f"{dm['pred_confirmed_hall_rate_response_mean_pct']:.2f}% ({dm['pred_gfi_response_mean_pct']:.1f}%) | "
                f"{dm['pred_unverifiable_rate_pooled_pct']:.2f}% |"
            )
        lines.append("")

        lines.append(f"### G. MIXED_HALLUCINATION Separate Diagnostic ({regime_key})")
        lines.append("")
        lines.append(
            "| Arm | n_mixed | Confirmed Flagged (`CONTRA+UNSUP`) | Abstained (`UNVERIFIABLE`) | Falsely `SUPPORTED_PROXY` |"
        )
        lines.append("|---|---:|---:|---:|---:|")
        for arm in ARM_NAMES:
            mx = reg["arms"][arm]["overall"]["mixed_hallucination_diagnostic"]
            lines.append(
                f"| `{arm}` | {mx['n_mixed']} | "
                f"{mx['flagged_confirmed_hallucination_rate']*100:.1f}% ({mx['flagged_confirmed_hallucination_count']}) | "
                f"{mx['unverifiable_abstained_rate']*100:.1f}% ({mx['unverifiable_abstained_count']}) | "
                f"{mx['falsely_supported_rate']*100:.1f}% ({mx['falsely_supported_count']}) |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run VeriLLM Phase 8 Step 4 validation ablation study."
    )
    parser.add_argument(
        "--sources-per-task",
        type=int,
        default=5,
        help="Validation sources per task (default 5 -> 15-source pilot).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic source-selection and bootstrap seed.",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=2000,
        help="Source-clustered bootstrap iterations (default 2000).",
    )
    parser.add_argument(
        "--warmup-samples",
        type=int,
        default=16,
        help="Warm-up latency benchmark sample size before full run.",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Analyze using existing raw signal cache without running PyTorch inference.",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Generate and print the deterministic pilot manifest and duplicate audit, then exit.",
    )
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Run model loading and warm-up latency benchmark only, then exit.",
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Skip extra warm-up latency benchmark pass when resuming cached batches.",
    )
    parser.add_argument(
        "--max-sources",
        type=int,
        default=None,
        help="Process at most N uncached sources in this invocation, saving progress to cache.",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help="Path to raw signal cache JSONL.",
    )
    parser.add_argument(
        "--lock-manifest",
        type=Path,
        default=None,
        help="Optional prior manifest JSON whose selected_source_ids remain locked when expanding.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Optional existing manifest JSON to load read-only instead of overwriting the default manifest file.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Optional output filename prefix inside results/ (e.g. 09_val_pilot_30sources_corrected) so prior reports are preserved.",
    )
    args = parser.parse_args()

    logger.info("Loading validation split strictly from %s", VAL_DATA_PATH)
    val_records = load_validation_records(VAL_DATA_PATH)
    logger.info("Indexing train split overlap from %s (test split untouched)", TRAIN_DATA_PATH)
    train_index = build_train_overlap_index(TRAIN_DATA_PATH)

    locked_ids: Optional[List[str]] = None
    locked_manifest_path_str: Optional[str] = None
    locked_selection_method: Optional[str] = None
    if args.lock_manifest is not None:
        prior = json.loads(args.lock_manifest.read_text(encoding="utf-8"))
        locked_ids = [str(s) for s in prior.get("selected_source_ids", [])]
        locked_manifest_path_str = str(args.lock_manifest)
        locked_selection_method = prior.get("selection_method")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.manifest_path is not None:
        manifest = json.loads(args.manifest_path.read_text(encoding="utf-8"))
        val_src_set = {str(r["source_id"]) for r in val_records}
        missing_srcs = [s for s in manifest.get("selected_source_ids", []) if str(s) not in val_src_set]
        if missing_srcs:
            raise ValueError(f"Manifest {args.manifest_path} references unknown validation sources: {missing_srcs}")
        if locked_ids is not None:
            # Verify locked prefix and record two-phase metadata in memory only (never overwrite --manifest-path)
            sel_set = set(str(s) for s in manifest.get("selected_source_ids", []))
            missing_locked = [s for s in locked_ids if s not in sel_set]
            if missing_locked:
                raise ValueError(f"Locked manifest sources missing from {args.manifest_path}: {missing_locked}")
            manifest["locked_manifest_path"] = locked_manifest_path_str
            manifest["locked_selection_method"] = locked_selection_method
            manifest["locked_source_ids"] = locked_ids
        if "stratum_sampling_summary" not in manifest or locked_ids is not None:
            recomputed = select_pilot_sources(
                val_records=val_records,
                train_index=train_index,
                sources_per_task=int(manifest.get("sources_per_task", args.sources_per_task)),
                seed=int(manifest.get("seed", args.seed)),
                locked_source_ids=locked_ids,
                locked_manifest_path=locked_manifest_path_str,
                locked_selection_method=locked_selection_method,
            )
            if recomputed["selected_source_ids"] == manifest["selected_source_ids"]:
                manifest["stratum_sampling_summary"] = recomputed["stratum_sampling_summary"]
                if locked_ids is not None:
                    manifest["selection_method"] = recomputed["selection_method"]
                    manifest["selection_rationale"] = recomputed["selection_rationale"]
        manifest_path_used = str(args.manifest_path)
        logger.info(
            "Loaded existing manifest read-only from %s (%d sources, %d responses, %d sentences, method=%s)",
            manifest_path_used,
            manifest["total_sources"],
            manifest["total_responses"],
            manifest["total_sentences"],
            manifest["selection_method"],
        )
    else:
        manifest = select_pilot_sources(
            val_records=val_records,
            train_index=train_index,
            sources_per_task=args.sources_per_task,
            seed=args.seed,
            locked_source_ids=locked_ids,
            locked_manifest_path=locked_manifest_path_str,
            locked_selection_method=locked_selection_method,
        )
        manifest_path = RESULTS_DIR / f"09_val_pilot_{manifest['total_sources']}sources_manifest.json"
        if args.cache_only and manifest_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing manifest {manifest_path} in --cache-only mode; pass --manifest-path."
            )
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        manifest_path_used = str(manifest_path)
        logger.info(
            "Manifest saved to %s (%d sources, %d responses, %d sentences, method=%s)",
            manifest_path,
            manifest["total_sources"],
            manifest["total_responses"],
            manifest["total_sentences"],
            manifest["selection_method"],
        )
    if args.manifest_only:
        print(json.dumps(manifest, indent=2))
        return

    selected_set = set(manifest["selected_source_ids"])
    pilot_records = [r for r in val_records if str(r["source_id"]) in selected_set]
    pilot_records.sort(key=lambda r: (str(r["task_type"]), str(r["source_id"]), str(r["example_id"])))

    cache = _load_raw_cache(args.cache_path)
    missing_records = [r for r in pilot_records if str(r["example_id"]) not in cache]
    logger.info(
        "Cache status: %d/%d pilot sentences cached; %d remaining.",
        len(pilot_records) - len(missing_records),
        len(pilot_records),
        len(missing_records),
    )

    warmup_benchmark: Optional[Dict[str, Any]] = None
    inference_wall_seconds = 0.0

    if missing_records:
        if args.cache_only:
            raise RuntimeError(
                f"--cache-only requested but {len(missing_records)} records are uncached."
            )

        load_timings: Dict[str, float] = {}
        t_start = time.perf_counter()
        t0 = time.perf_counter()
        claim_verifier = ClaimVerifier(use_custom_model=True)
        load_timings["claim_verifier_minicheck_plus_nli"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        custom_deberta = claim_verifier._get_custom_deberta_verifier()
        load_timings["custom_deberta"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        chunker = SlidingWindowChunker(chunk_size_words=150, overlap_words=35, min_chunk_words=20)
        passage_retriever = PassageRetriever(model_name="all-MiniLM-L6-v2", top_k=3)
        load_timings["passage_retriever"] = time.perf_counter() - t0
        load_timings["total_model_init"] = time.perf_counter() - t_start

        if custom_deberta is None:
            raise RuntimeError("CustomDebertaVerifier could not be loaded.")

        if not args.skip_warmup or args.benchmark_only:
            bench_n = min(max(4, args.warmup_samples), len(missing_records))
            # Take balanced slice across tasks for warm-up benchmark
            bench_slice: List[Dict[str, Any]] = []
            for task in TASK_TYPES:
                t_recs = [r for r in missing_records if r["task_type"] == task]
                bench_slice.extend(t_recs[: max(1, bench_n // 3)])
            if len(bench_slice) < bench_n:
                seen_b = {r["example_id"] for r in bench_slice}
                for r in missing_records:
                    if r["example_id"] not in seen_b:
                        bench_slice.append(r)
                        if len(bench_slice) >= bench_n:
                            break

            warmup_benchmark = run_warmup_latency_benchmark(
                sample_records=bench_slice,
                claim_verifier=claim_verifier,
                chunker=chunker,
                passage_retriever=passage_retriever,
                load_timings=load_timings,
                total_target_sentences=len(missing_records),
            )
            logger.info("Warm-up latency benchmark: %s", json.dumps(warmup_benchmark))
            bench_path = RESULTS_DIR / "09_val_warmup_latency_benchmark.json"
            bench_path.write_text(json.dumps(warmup_benchmark, indent=2), encoding="utf-8")
            if args.benchmark_only:
                print(json.dumps(warmup_benchmark, indent=2))
                return

        token_len_fn = _make_token_length_fn(custom_deberta.tokenizer)
        missing_by_src: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in missing_records:
            missing_by_src[str(r["source_id"])].append(r)

        inf_t0 = time.perf_counter()
        total_srcs = len(missing_by_src)
        processed_srcs = 0
        for sid in manifest["selected_source_ids"]:
            src_recs = missing_by_src.get(sid)
            if not src_recs:
                continue
            if args.max_sources is not None and processed_srcs >= args.max_sources:
                logger.info(
                    "Reached --max-sources=%d limit cleanly (%d/%d uncached sources processed).",
                    args.max_sources,
                    processed_srcs,
                    total_srcs,
                )
                break
            processed_srcs += 1
            s_t0 = time.perf_counter()
            d_t0 = time.perf_counter()
            direct_map = evaluate_direct_context_records(
                src_recs,
                minicheck_verifier=claim_verifier.minicheck,
                nli_verifier=claim_verifier.nli,
                custom_deberta_verifier=custom_deberta,
                token_length_fn=token_len_fn,
            )
            direct_sec = time.perf_counter() - d_t0
            r_t0 = time.perf_counter()
            ret_map = evaluate_retrieval_pipeline_for_source(
                src_recs,
                chunker=chunker,
                passage_retriever=passage_retriever,
                claim_verifier=claim_verifier,
                token_length_fn=token_len_fn,
            )
            ret_sec = time.perf_counter() - r_t0
            to_cache: List[Dict[str, Any]] = []
            for r in src_recs:
                ex_id = str(r["example_id"])
                pair_k = compute_pair_key(r.get("context_text", ""), r.get("sentence_text", ""))
                cached_row = {
                    "example_id": ex_id,
                    "response_id": str(r["response_id"]),
                    "source_id": str(r["source_id"]),
                    "task_type": r["task_type"],
                    "model": r["model"],
                    "split": r["split"],
                    "primary_label": r["primary_label"],
                    "has_hallucination": bool(r.get("has_hallucination")),
                    "has_implicit_true": bool(r.get("has_implicit_true")),
                    "has_due_to_null": bool(r.get("has_due_to_null")),
                    "sentence_text": r["sentence_text"],
                    "context_word_count": len(str(r.get("context_text", "")).split()),
                    "is_exact_train_pair_dup": pair_k in train_index["exact_pair_keys"],
                    "is_sentence_string_in_train": str(r.get("sentence_text", "")) in train_index["sentence_strings"],
                    "direct": direct_map[ex_id],
                    "retrieval": ret_map[ex_id],
                }
                cache[ex_id] = cached_row
                to_cache.append(cached_row)
            _append_cache_records(args.cache_path, to_cache)
            timings_path = RESULTS_DIR / "09_val_pilot_source_timings.json"
            src_timings: Dict[str, Dict[str, Any]] = {}
            if timings_path.exists():
                src_timings = json.loads(timings_path.read_text(encoding="utf-8"))
            src_timings[str(sid)] = {
                "task_type": src_recs[0]["task_type"],
                "num_sentences": len(src_recs),
                "direct_seconds": round(direct_sec, 3),
                "retrieval_seconds": round(ret_sec, 3),
                "total_seconds": round(direct_sec + ret_sec, 3),
            }
            timings_path.write_text(json.dumps(src_timings, indent=2), encoding="utf-8")
            logger.info(
                "Source %d/%d (%s, %s, %d sentences) finished in %.2fs (direct=%.2fs, retrieval=%.2fs)",
                processed_srcs,
                total_srcs,
                sid,
                src_recs[0]["task_type"],
                len(src_recs),
                time.perf_counter() - s_t0,
                direct_sec,
                ret_sec,
            )
        inference_wall_seconds = round(time.perf_counter() - inf_t0, 2)

    missing_after = [r for r in pilot_records if str(r["example_id"]) not in cache]
    if missing_after:
        logger.info(
            "Partial batch complete: %d/%d pilot sentences now cached (%d remaining across %d sources). Skipping full report generation until all sources finish.",
            len(pilot_records) - len(missing_after),
            len(pilot_records),
            len(missing_after),
            len({str(r["source_id"]) for r in missing_after}),
        )
        return

    bench_path = RESULTS_DIR / "09_val_warmup_latency_benchmark.json"
    if warmup_benchmark is None and bench_path.exists():
        warmup_benchmark = json.loads(bench_path.read_text(encoding="utf-8"))

    timings_path = RESULTS_DIR / "09_val_pilot_source_timings.json"
    per_source_timings: Dict[str, Any] = (
        json.loads(timings_path.read_text(encoding="utf-8"))
        if timings_path.exists()
        else {}
    )
    tracked_wall_sum = round(
        sum(
            float(per_source_timings[sid]["total_seconds"])
            for sid in manifest["selected_source_ids"]
            if sid in per_source_timings
        ),
        2,
    )
    if tracked_wall_sum > 0:
        inference_wall_seconds = tracked_wall_sum

    cached_pilot = [cache[str(r["example_id"])] for r in pilot_records]

    source_weights = {
        str(m["source_id"]): float(m["relative_stratum_weight"])
        for m in manifest["selected_sources"]
    }
    dup_audit = manifest["duplicate_and_leakage_audit"]
    exact_dup_ids = set(dup_audit.get("exact_pair_duplicate_example_ids", []))
    sent_overlap_ids = {
        str(r["example_id"]) for r in cached_pilot if r.get("is_sentence_string_in_train")
    }

    regimes_out = {
        "direct": _evaluate_regime(
            cached_pilot,
            regime="direct",
            source_weights=source_weights,
            exact_dup_ids=exact_dup_ids,
            sent_overlap_ids=sent_overlap_ids,
            n_boot=args.n_boot,
            seed=args.seed,
        ),
        "retrieval": _evaluate_regime(
            cached_pilot,
            regime="retrieval",
            source_weights=source_weights,
            exact_dup_ids=exact_dup_ids,
            sent_overlap_ids=sent_overlap_ids,
            n_boot=args.n_boot,
            seed=args.seed,
        ),
    }

    n_src = manifest["total_sources"]
    output_prefix = args.output_prefix or f"09_val_pilot_{n_src}sources"
    results_json_path = RESULTS_DIR / f"{output_prefix}_ablation_results.json"
    results_md_path = RESULTS_DIR / f"{output_prefix}_ablation_report.md"

    if args.cache_only and args.output_prefix is None and (results_json_path.exists() or results_md_path.exists()):
        raise FileExistsError(
            f"Refusing to overwrite existing results ({results_json_path.name} / {results_md_path.name}) "
            "in --cache-only mode without --output-prefix."
        )

    weight_vals = list(source_weights.values()) if source_weights else []
    results = {
        "metadata": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "cache_only": bool(args.cache_only),
            "val_path": str(VAL_DATA_PATH),
            "cache_path": str(args.cache_path),
            "manifest_path": manifest_path_used,
            "locked_manifest_path": locked_manifest_path_str or manifest.get("locked_manifest_path"),
            "locked_source_ids": manifest.get("locked_source_ids"),
            "selection_method": manifest.get("selection_method"),
            "output_prefix": output_prefix,
            "sources_per_task": args.sources_per_task,
            "seed": args.seed,
            "n_boot": args.n_boot,
            "inference_wall_seconds": inference_wall_seconds,
            "test_split_accessed": False,
            "cli_args": sys.argv[1:],
            "git": _collect_git_metadata(),
            "versions": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "sklearn": sklearn.__version__,
                "numpy": np.__version__,
            },
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "model_identifiers": {
                "minicheck": "bespokelabs/MiniCheck (roberta-large)",
                "nli": "cross-encoder/nli-deberta-v3-xsmall",
                "retriever": "sentence-transformers/all-MiniLM-L6-v2",
                "custom_deberta": str(BACKEND_DIR / "models" / "verillm_deberta"),
            },
            "custom_deberta_checkpoint_sha256": _collect_checkpoint_hashes(),
            "bootstrap_resampling_unit": "source_id",
            "sampling_weights_summary": {
                "source_count": len(weight_vals),
                "min_weight": min(weight_vals) if weight_vals else None,
                "max_weight": max(weight_vals) if weight_vals else None,
                "weights_by_stratum": manifest.get("stratum_sampling_summary", []),
            },
            "uncalibrated_outputs_note": (
                "CustomDeBERTa outputs are class-weighted uncalibrated normalized logits; "
                "arm3a applies unvalidated legacy NLI thresholds and must not justify a production decision."
            ),
        },
        "manifest": manifest,
        "warmup_benchmark": warmup_benchmark,
        "per_source_timings": per_source_timings,
        "regimes": regimes_out,
    }

    results_json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    results_md_path.write_text(_render_markdown_report(results), encoding="utf-8")

    logger.info("Saved full ablation results JSON -> %s", results_json_path)
    logger.info("Saved ablation Markdown report -> %s", results_md_path)


if __name__ == "__main__":
    main()

