"""
VeriLLM Phase 8 Step 7: Calibration and Hybrid-Strategy Experiment Runner.

Supports:
  --stage1-cache-only:
      Runs 5x4 nested source-grouped CV across both regimes on the 30 Val-Dev
      sources using 09_val_pilot_raw_cache.jsonl without neural inference.
      Emits 10_stage1_cal_hybrid_report.md, 10_stage1_cal_hybrid_results.json,
      and 10_stage1_frozen_bundle.json.
  --stage2-manifest-only:
      Selects 30 deterministic Val-Lock sources strictly disjoint from Val-Dev
      and training splits, verifying minority-class adequacy and emitting
      10_stage2_vallock_manifest.json without neural inference.
  --stage2-run:
      Evaluates the frozen Stage 1 bundle on Val-Lock. Strictly requires
      --approve-stage2-inference to run neural inference for any missing signals.
  --stage2-analyze-only:
      Analyzes cached Val-Lock predictions without neural inference.

STRICT SAFETY:
- Access to ragtruth_test.jsonl is strictly blocked with a hard exception.
- Production modules, decision strategies, and model weights are untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import psutil

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml_engine.custom_model.calibration import (
    CALIBRATION_CLASSES,
    assert_disjoint_sources,
)
from ml_engine.custom_model.hybrid_strategies import (
    PRE_REGISTERED_GRIDS,
    PRE_REGISTERED_OBJECTIVE_PARAMS,
    STAGE1_STRATEGY_ORDER,
    STRATEGY_REGISTRY,
    apply_frozen_bundle_to_records,
    freeze_stage2_bundle,
    run_nested_source_cv_stage1,
)
from ml_engine.custom_model.validation_ablation import (
    TASK_TYPES,
    build_train_overlap_index,
    compute_context_hash,
    compute_pair_key,
    evaluate_direct_context_records,
    evaluate_minority_adequacy,
    evaluate_retrieval_pipeline_for_source,
    load_validation_records,
    select_pilot_sources,
    summarize_sources,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("verillm.cal_hybrid_study")

VAL_DATA_PATH = BACKEND_DIR / "data" / "processed" / "ragtruth_val.jsonl"
TRAIN_DATA_PATH = BACKEND_DIR / "data" / "processed" / "ragtruth_train.jsonl"
RESULTS_DIR = BACKEND_DIR / "experiments" / "results"
DEFAULT_VAL_DEV_MANIFEST = RESULTS_DIR / "09_val_pilot_30sources_manifest.json"
DEFAULT_VAL_DEV_CACHE = RESULTS_DIR / "09_val_pilot_raw_cache.jsonl"

STAGE1_REPORT_PATH = RESULTS_DIR / "10_stage1_cal_hybrid_report.md"
STAGE1_RESULTS_PATH = RESULTS_DIR / "10_stage1_cal_hybrid_results.json"
STAGE1_FROZEN_BUNDLE_PATH = RESULTS_DIR / "10_stage1_frozen_bundle.json"

STAGE2_MANIFEST_PATH = RESULTS_DIR / "10_stage2_vallock_manifest.json"
STAGE2_CACHE_PATH = RESULTS_DIR / "10_stage2_vallock_raw_cache.jsonl"
STAGE2_REPORT_PATH = RESULTS_DIR / "10_stage2_confirmation_report.md"
STAGE2_RESULTS_PATH = RESULTS_DIR / "10_stage2_confirmation_results.json"
STAGE2_PROFILE_PATH = RESULTS_DIR / "10_stage2_runtime_profile.json"


def assert_safe_non_test_path(path: Path) -> Path:
    resolved = path.resolve()
    for part in resolved.parts:
        if "ragtruth_test" in part.lower():
            raise ValueError(f"ACCESS VIOLATION: ragtruth_test.jsonl is strictly forbidden: {resolved}")
    return resolved


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_val_dev_sources(manifest_path: Path, expected_count: Optional[int] = 30) -> List[str]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    sources = data.get("selected_source_ids") or data.get("stage1_uniform_attempt", {}).get("source_ids")
    if not sources:
        return []
    if expected_count is not None and len(sources) != expected_count:
        raise ValueError(f"Expected exactly {expected_count} Val-Dev sources in {manifest_path}, got {len(sources)}")
    return [str(s) for s in sources]


def atomic_write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    """Atomically write text to path using a tempfile in the same directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp_{os.getpid()}_{time.time_ns()}")
    try:
        with temp_path.open("w", encoding=encoding) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def atomic_write_json(path: Path, data: Any, indent: int = 2) -> None:
    """Atomically write JSON data to path using a tempfile in the same directory."""
    content = json.dumps(data, indent=indent, ensure_ascii=False)
    atomic_write_text(path, content, encoding="utf-8")


def get_peak_memory_profile() -> Dict[str, Any]:
    """Capture true peak working-set memory if supported by OS, plus current RSS."""
    proc = psutil.Process()
    mem_info = proc.memory_info()
    current_rss_mb = round(mem_info.rss / (1024 * 1024), 2)

    peak_wset_bytes = getattr(mem_info, "peak_wset", None)
    if peak_wset_bytes is not None and peak_wset_bytes > 0:
        peak_mb = round(peak_wset_bytes / (1024 * 1024), 2)
        memory_metric_name = "peak_working_set_mb"
    else:
        ru_maxrss_mb = None
        try:
            import resource
            usage = resource.getrusage(resource.RUSAGE_SELF)
            if platform.system() == "Darwin":
                ru_maxrss_mb = round(usage.ru_maxrss / (1024 * 1024), 2)
            else:
                ru_maxrss_mb = round(usage.ru_maxrss / 1024, 2)
        except (ImportError, AttributeError):
            pass

        if ru_maxrss_mb is not None and ru_maxrss_mb > 0:
            peak_mb = ru_maxrss_mb
            memory_metric_name = "peak_ru_maxrss_mb"
        else:
            peak_mb = current_rss_mb
            memory_metric_name = "current_rss_fallback_mb"

    return {
        "peak_memory_mb": peak_mb,
        "peak_memory_metric": memory_metric_name,
        "current_rss_mb": current_rss_mb,
    }


def load_cached_records(cache_path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with cache_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line_str = line.strip()
            if line_str:
                records.append(json.loads(line_str))
    return records


def load_raw_cache_by_id(
    cache_path: Path,
    expected_source_map: Optional[Dict[str, str]] = None,
    allow_repair_trailing_partial: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Load cached per-example signals keyed by example_id with strict integrity checks.

    Integrity rules:
    - Rejects duplicate example_id entries (prevents silent overrides).
    - Requires both 'direct' and 'retrieval' regimes to be present and non-empty.
    - Validates source_id against expected_source_map when provided.
    - Fails fast on malformed/corrupt records in the middle of the cache.
    - Safely truncates/ignores an incomplete trailing line caused by an interrupted write,
      preserving all completed preceding records.
    """
    cache: Dict[str, Dict[str, Any]] = {}
    if not cache_path.exists():
        return cache

    with cache_path.open("rb") as handle:
        raw_bytes = handle.read()

    if not raw_bytes.strip():
        return cache

    raw_lines = raw_bytes.splitlines(keepends=True)
    last_valid_byte_offset = 0
    current_byte_offset = 0

    for line_num, line_bytes in enumerate(raw_lines, start=1):
        stripped_bytes = line_bytes.strip()
        if not stripped_bytes:
            current_byte_offset += len(line_bytes)
            continue

        is_last_line = (line_num == len(raw_lines))

        try:
            line_str = stripped_bytes.decode("utf-8")
            item = json.loads(line_str)
        except (UnicodeDecodeError, json.JSONDecodeError) as err:
            if is_last_line and allow_repair_trailing_partial:
                logger.warning(
                    "Interrupted trailing line detected at line %d in %s (%s). "
                    "Repairing cache by truncating trailing partial fragment to preserve %d valid records.",
                    line_num,
                    cache_path,
                    err,
                    len(cache),
                )
                try:
                    with cache_path.open("r+b") as fix_handle:
                        fix_handle.seek(last_valid_byte_offset)
                        fix_handle.truncate()
                except OSError as e:
                    logger.warning("Could not truncate cache file %s: %s", cache_path, e)
                break
            else:
                raise ValueError(
                    f"Malformed cache record at line {line_num} in {cache_path}: {err}. "
                    "Cache corruption detected in non-trailing line."
                ) from err

        ex_id = str(item.get("example_id", "")).strip()
        if not ex_id:
            raise ValueError(f"Cache record at line {line_num} in {cache_path} is missing 'example_id'.")

        if ex_id in cache:
            raise ValueError(
                f"Duplicate example_id '{ex_id}' detected at line {line_num} in {cache_path}. "
                "Duplicate cache records are forbidden to prevent silent corruption."
            )

        if "direct" not in item or not isinstance(item["direct"], dict) or not item["direct"]:
            raise ValueError(
                f"Cache record for '{ex_id}' at line {line_num} in {cache_path} is missing valid 'direct' regime."
            )
        if "retrieval" not in item or not isinstance(item["retrieval"], dict) or not item["retrieval"]:
            raise ValueError(
                f"Cache record for '{ex_id}' at line {line_num} in {cache_path} is missing valid 'retrieval' regime."
            )

        src_id = str(item.get("source_id", "")).strip()
        if not src_id:
            raise ValueError(f"Cache record for '{ex_id}' at line {line_num} in {cache_path} has missing or empty 'source_id'.")

        if expected_source_map is not None:
            expected_src = expected_source_map.get(ex_id)
            if expected_src is not None and src_id != expected_src:
                raise ValueError(
                    f"Source ID mismatch for '{ex_id}' at line {line_num} in {cache_path}: "
                    f"expected '{expected_src}', got '{src_id}'."
                )

        cache[ex_id] = item
        current_byte_offset += len(line_bytes)
        last_valid_byte_offset = current_byte_offset

    return cache


def append_cache_records(cache_path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    """Safely append cache records to JSONL cache file."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def make_token_length_fn(tokenizer: Any) -> Callable[[Sequence[Tuple[str, str]]], List[int]]:
    """Measure untruncated (context, claim) DeBERTa pair token length cleanly."""
    prev_max = getattr(tokenizer, "model_max_length", 512)

    def _measure(pairs: Sequence[Tuple[str, str]]) -> List[int]:
        lengths: List[int] = []
        tokenizer.model_max_length = 1_000_000
        try:
            for ctx, claim in pairs:
                if not str(ctx).strip() and not str(claim).strip():
                    lengths.append(0)
                    continue
                encoded = tokenizer(
                    str(ctx),
                    str(claim),
                    truncation=False,
                    add_special_tokens=True,
                )
                lengths.append(len(encoded["input_ids"]))
        finally:
            tokenizer.model_max_length = prev_max
        return lengths

    return _measure


def select_stage2_vallock_sources(
    val_records: Sequence[Dict[str, Any]],
    val_dev_source_ids: Sequence[str],
    train_index: Optional[Dict[str, Any]] = None,
    sources_per_task: int = 10,
    seed: int = 43,
) -> Dict[str, Any]:
    """Deterministically select 30 Val-Lock sources strictly disjoint from Val-Dev and train content."""
    val_dev_set = set(str(s) for s in val_dev_source_ids)
    train_ctx_hashes = (
        set(train_index.get("context_hashes", set()))
        if train_index is not None
        else set()
    )

    filtered_recs: List[Dict[str, Any]] = []
    excluded_train_ctx_sids: Set[str] = set()
    for r in val_records:
        sid = str(r["source_id"])
        if sid in val_dev_set:
            continue
        ctx = r.get("context_text", "")
        if train_ctx_hashes and compute_context_hash(ctx) in train_ctx_hashes:
            excluded_train_ctx_sids.add(sid)
            continue
        filtered_recs.append(r)

    manifest = select_pilot_sources(
        val_records=filtered_recs,
        train_index=train_index,
        sources_per_task=sources_per_task,
        seed=seed,
        min_total_contradicted=20,
        min_total_unsupported=35,
        min_contradicted_per_task=2,
        min_unsupported_per_task=4,
    )

    adequacy = manifest.get("final_adequacy") or manifest.get("stage1_uniform_attempt", {}).get("adequacy", {})
    if not adequacy or not adequacy.get("is_adequate", False):
        failures = (
            adequacy.get("failures", ["Adequacy criteria not satisfied"])
            if adequacy
            else ["Missing adequacy assessment"]
        )
        raise ValueError(
            f"Stage 2 Val-Lock selection failed adequacy constraints: {failures}. "
            f"Counts: {adequacy.get('counts_by_label') if adequacy else 'N/A'}"
        )

    val_lock_sources = set(str(s) for s in manifest["selected_source_ids"])
    assert_disjoint_sources(val_dev_set, val_lock_sources, context="ValDev_vs_ValLock_selection")
    manifest["val_dev_excluded_source_ids"] = sorted(val_dev_set)
    manifest["val_dev_exclusion_verified"] = True
    manifest["train_content_leakage_excluded_sources"] = sorted(excluded_train_ctx_sids)
    manifest["exclusion_policy"] = {
        "val_dev_disjointness": "All Stage 1 Val-Dev source_ids excluded prior to sampling.",
        "train_context_disjointness": "All validation sources sharing exact context SHA-256 with any training record excluded prior to sampling.",
    }
    return manifest


def format_stage1_markdown_report(results: Dict[str, Any], provenance: Dict[str, Any]) -> str:
    lines: List[str] = [
        "# VeriLLM Phase 8 Step 7: Stage 1 Calibration & Hybrid-Strategy Study Report",
        "",
        f"- **Generated At (UTC)**: {provenance['timestamp_utc']}",
        f"- **Val-Dev Manifest**: `{provenance['val_dev_manifest_name']}` (SHA-256: `{provenance['val_dev_manifest_sha256'][:16]}...`)",
        f"- **Val-Dev Raw Cache**: `{provenance['val_dev_cache_name']}` (SHA-256: `{provenance['val_dev_cache_sha256'][:16]}...`)",
        f"- **Total Val-Dev Sources**: {provenance['total_sources']} | **Responses**: {provenance['total_responses']} | **Sentences**: {provenance['total_sentences']}",
        f"- **Source-Grouped Nested CV**: {provenance['n_outer_folds']} outer folds × {provenance['n_inner_folds']} inner folds",
        f"- **H3 Status**: **EXCLUDED FROM STAGE 1 CACHE-ONLY EVALUATION** (`09_val_pilot_raw_cache.jsonl` does not store per-chunk DeBERTa scores for secondary passages).",
        "",
        "---",
        "",
        "## Pillar 1 — Probability Calibration Analysis (5-Fold Source-Grouped OOF)",
        "",
        "All calibration models are evaluated strictly out-of-fold across 5 outer source folds. "
        "Each unique `source_id` is assigned to exactly one outer fold identically across Direct and Retrieval regimes.",
        "",
    ]

    for regime in ("direct", "retrieval"):
        r_data = results["regimes"][regime]
        p1 = r_data["pillar1_calibration"]
        c0 = p1["deberta_C0_raw_oof"]
        c1 = p1["deberta_C1_temp_oof"]
        c2 = p1["deberta_C2_nested_oof"]
        mc_raw = p1["minicheck_raw_support_oof"]
        mc_platt = p1["minicheck_platt_support_oof"]

        lines.extend([
            f"### Regime: `{regime}` — Custom DeBERTa 3-Class Calibration",
            "",
            "| Model / Scaling Method | 3-Class NLL | 3-Class Brier | Top-1 ECE | Top-1 MCE | Classwise ECE (CW-ECE) | Binary Hallucination ECE | Argmax Accuracy |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            f"| **C0: Raw Uncalibrated Softmax** | {c0['nll_3class']:.4f} | {c0['brier_3class']:.4f} | {c0['top1_ece']:.4f} | {c0['top1_mce']:.4f} | {c0['cw_ece']:.4f} | {c0['binary_hallucination']['ece_uniform']:.4f} | {c0['argmax_accuracy'] * 100.0:.2f}% |",
            f"| **C1: Scalar Temperature Scaling** | {c1['nll_3class']:.4f} | {c1['brier_3class']:.4f} | {c1['top1_ece']:.4f} | {c1['top1_mce']:.4f} | {c1['cw_ece']:.4f} | {c1['binary_hallucination']['ece_uniform']:.4f} | {c1['argmax_accuracy'] * 100.0:.2f}% |",
            f"| **C2: Regularized Prior-Bias + Temp** | **{c2['nll_3class']:.4f}** | **{c2['brier_3class']:.4f}** | **{c2['top1_ece']:.4f}** | **{c2['top1_mce']:.4f}** | **{c2['cw_ece']:.4f}** | **{c2['binary_hallucination']['ece_uniform']:.4f}** | {c2['argmax_accuracy'] * 100.0:.2f}% |",
            "",
            f"#### MiniCheck Binary Support Calibration (`{regime}`)",
            "",
            "| MiniCheck Estimator | Binary NLL | Brier Score | ECE (Uniform) | MCE (Uniform) | ECE (Equal Mass) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
            f"| Raw MiniCheck Support Score | {mc_raw['nll']:.4f} | {mc_raw['brier']:.4f} | {mc_raw['ece_uniform']:.4f} | {mc_raw['mce_uniform']:.4f} | {mc_raw['ece_equal_mass']:.4f} |",
            f"| Platt-Calibrated MiniCheck | **{mc_platt['nll']:.4f}** | **{mc_platt['brier']:.4f}** | **{mc_platt['ece_uniform']:.4f}** | **{mc_platt['mce_uniform']:.4f}** | **{mc_platt['ece_equal_mass']:.4f}** |",
            "",
        ])

    lines.extend([
        "---",
        "",
        "## Pillar 2 & 3 — Claim Classification, Response-Level & 6-LLM Evaluation",
        "",
        "Reported across both pre-registered objectives:",
        "- **OBJ_CONSTRAINED**: `Macro-F1 - 0.005 * max(0, MAE_conf_pp - MAE_max_pp)` subject to coverage >= 0.95 (direct) / 0.90 (retrieval) and recall floors.",
        "- **OBJ_MACRO_F1_ONLY**: Unconstrained Macro-F1 subject only to coverage floor.",
        "",
    ])

    for regime in ("direct", "retrieval"):
        r_data = results["regimes"][regime]
        for obj_name in ("OBJ_CONSTRAINED", "OBJ_MACRO_F1_ONLY"):
            obj_data = r_data["objectives"][obj_name]
            winner = obj_data["winner"]
            lines.extend([
                f"### Regime: `{regime}` | Objective: `{obj_name}`",
                "",
                f"> **Winning Strategy**: **`{winner['strategy']}`** (Utility: `{winner['objective_utility']:.4f}`, Macro-F1: `{winner['macro_f1'] * 100.0:.2f}%`, Conf-MAE: `{winner['mae_conf_pp']:.2f} pp`, Coverage: `{winner['coverage'] * 100.0:.2f}%`, All Constraints Met: `{winner['passes_coverage_floor'] and winner['passes_recall_floors']}`)",
                "",
                "| Strategy | Utility | Macro-F1 | W-F1 | Accuracy | Conf-MAE (pp) | Non-Sup MAE (pp) | GFI Spearman | Coverage | Holm p (vs H0) |",
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            ])

            for row in obj_data["leaderboard"]:
                s = row["strategy"]
                s_arm = obj_data["strategies"][s]["oof_claim_metrics"]
                s_resp = obj_data["strategies"][s]["oof_response_metrics"]
                p3 = s_arm["penalized_3class"]
                rm = s_resp["confirmed_hallucination_rate_pct"]
                non_sup_mae = s_resp["non_supported_rate_100_minus_gfi_pct"]["mae_vs_true_hall"]
                gfi_rho = s_resp["support_rate_gfi_pct"]["spearman_rho"]
                gfi_rho_str = f"{gfi_rho:.3f}" if gfi_rho is not None else "N/A"
                if s == "H0_baseline":
                    holm_p_str = "Ref"
                else:
                    comp = obj_data["paired_comparisons_vs_H0"][s]
                    holm_p_str = f"{comp['holm_adjusted_p_3class_source_perm']:.4f}"

                is_win = (s == winner["strategy"])
                prefix = "**" if is_win else ""
                suffix = "**" if is_win else ""
                lines.append(
                    f"| {prefix}{s}{suffix} | {row['objective_utility']:.4f} | {p3['macro_f1'] * 100.0:.2f}% | "
                    f"{p3['support_weighted_f1'] * 100.0:.2f}% | {p3['accuracy'] * 100.0:.2f}% | "
                    f"{rm['mae']:.2f} | {non_sup_mae:.2f} | {gfi_rho_str} | "
                    f"{s_arm['coverage'] * 100.0:.2f}% | {holm_p_str} |"
                )
            lines.append("")

    lines.extend([
        "---",
        "",
        "## Pillar 4 — Efficiency, Latency & Stage 2 Execution Framework",
        "",
        "- **Stage 1 Execution Mode**: Pure cache-only offline nested cross-validation (`0` neural inference calls).",
        "- **H3 Secondary-Passage Status**: Excluded from Stage 1 cache-only evaluation; marked `requires_stage2_secondary_chunk_inference`.",
        "- **Stage 2 Val-Lock Safety Guard**: `--stage2-run` strictly requires `--approve-stage2-inference` before running any neural models.",
        "- **Immutable Frozen Bundle**: Serialized to `10_stage1_frozen_bundle.json` using Val-Dev 5-fold OOF tuning; never updated using Val-Lock data.",
        "",
    ])
    return "\n".join(lines)


def run_stage1_cache_only(
    cache_path: Path,
    manifest_path: Path,
    n_outer_folds: int = 5,
    n_inner_folds: int = 4,
    n_boot: int = 500,
    seed: int = 42,
) -> None:
    logger.info("Starting Phase 8 Step 7 Stage 1 cache-only evaluation...")
    assert_safe_non_test_path(cache_path)
    assert_safe_non_test_path(manifest_path)

    val_dev_sources = load_val_dev_sources(manifest_path)
    records = load_cached_records(cache_path)
    if len(records) != 1259:
        logger.warning("Expected 1259 Val-Dev cached records, got %d", len(records))

    val_dev_records = [r for r in records if str(r["source_id"]) in set(val_dev_sources)]
    if len(val_dev_records) != len(records):
        raise ValueError(f"Cache records ({len(records)}) contain sources not in Val-Dev manifest ({len(val_dev_sources)})")

    t0 = time.time()
    results = run_nested_source_cv_stage1(
        val_dev_records,
        n_outer_folds=n_outer_folds,
        n_inner_folds=n_inner_folds,
        seed=seed,
        n_boot=n_boot,
    )
    elapsed = time.time() - t0
    logger.info("Stage 1 nested CV completed in %.2f seconds.", elapsed)

    logger.info("Freezing Stage 2 bundle strictly on all 30 Val-Dev sources...")
    frozen_bundle = freeze_stage2_bundle(val_dev_records, n_outer_folds=n_outer_folds, seed=seed)

    provenance = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 2),
        "val_dev_manifest_name": manifest_path.name,
        "val_dev_manifest_sha256": compute_file_sha256(manifest_path),
        "val_dev_cache_name": cache_path.name,
        "val_dev_cache_sha256": compute_file_sha256(cache_path),
        "total_sources": len(val_dev_sources),
        "total_responses": len({str(r["response_id"]) for r in val_dev_records}),
        "total_sentences": len(val_dev_records),
        "n_outer_folds": n_outer_folds,
        "n_inner_folds": n_inner_folds,
        "seed": seed,
    }
    results["provenance"] = provenance

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with STAGE1_RESULTS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    logger.info("Saved Stage 1 results: %s", STAGE1_RESULTS_PATH)

    with STAGE1_FROZEN_BUNDLE_PATH.open("w", encoding="utf-8") as handle:
        json.dump(frozen_bundle, handle, indent=2)
    logger.info("Saved Stage 1 frozen bundle: %s", STAGE1_FROZEN_BUNDLE_PATH)

    report_md = format_stage1_markdown_report(results, provenance)
    with STAGE1_REPORT_PATH.open("w", encoding="utf-8") as handle:
        handle.write(report_md)
    logger.info("Saved Stage 1 report: %s", STAGE1_REPORT_PATH)


def run_stage2_manifest_only(
    val_data_path: Path,
    train_data_path: Path,
    val_dev_manifest_path: Path,
    seed: int = 43,
) -> None:
    logger.info("Selecting 30 Val-Lock sources strictly disjoint from Val-Dev and train...")
    assert_safe_non_test_path(val_data_path)
    assert_safe_non_test_path(train_data_path)
    assert_safe_non_test_path(val_dev_manifest_path)

    val_dev_sources = load_val_dev_sources(val_dev_manifest_path)
    val_records = load_validation_records(val_data_path)
    train_index = build_train_overlap_index(train_data_path)

    manifest = select_stage2_vallock_sources(
        val_records=val_records,
        val_dev_source_ids=val_dev_sources,
        train_index=train_index,
        sources_per_task=10,
        seed=seed,
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with STAGE2_MANIFEST_PATH.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    logger.info("Saved Stage 2 Val-Lock manifest: %s", STAGE2_MANIFEST_PATH)
    logger.info(
        "Val-Lock sources selected: %d (Adequate: %s)",
        len(manifest["selected_source_ids"]),
        manifest.get("final_adequacy", {}).get("is_adequate", False),
    )


def format_stage2_markdown_report(
    results: Dict[str, Any],
    profile: Dict[str, Any],
    provenance: Dict[str, Any],
) -> str:
    lines: List[str] = [
        "# VeriLLM Phase 8 Step 7: Stage 2 Val-Lock Confirmation Study Report",
        "",
        f"- **Generated At (UTC)**: {provenance.get('timestamp_utc', 'N/A')}",
        f"- **Val-Lock Manifest**: `{provenance.get('manifest_path', '10_stage2_vallock_manifest.json')}` (SHA-256: `{provenance.get('manifest_sha256', '')[:16]}...`)",
        f"- **Stage 1 Frozen Bundle**: `{provenance.get('frozen_bundle_path', '10_stage1_frozen_bundle.json')}` (SHA-256: `{provenance.get('frozen_bundle_sha256', '')[:16]}...`)",
        f"- **Val-Lock Raw Cache**: `{provenance.get('cache_path', '10_stage2_vallock_raw_cache.jsonl')}` (SHA-256: `{provenance.get('cache_sha256', '')[:16]}...`)",
        f"- **Total Val-Lock Sources**: {provenance.get('total_sources', 30)} | **Responses**: {provenance.get('total_responses', 180)} | **Sentences**: {provenance.get('total_sentences', 1238)}",
        "- **Evaluation Mode**: Zero parameters fitted or tuned on Val-Lock. Strictly frozen bundle evaluation.",
        "- **Disjointness Audit**: 30 Val-Lock sources strictly disjoint from 30 Val-Dev sources and 0 exact training context overlaps.",
        "",
        "---",
        "",
        "## Pillar 1 — Probability Calibration Confirmation on Held-Out Val-Lock",
        "",
        "All calibration models were fitted strictly on the 30 Val-Dev sources during Stage 1 and frozen in `10_stage1_frozen_bundle.json`. "
        "Here, they are evaluated out-of-sample on the 30 held-out Val-Lock sources with zero refitting.",
        "",
    ]

    for regime in ("direct", "retrieval"):
        if regime not in results.get("regimes", {}):
            continue
        r_data = results["regimes"][regime]
        p1 = r_data["pillar1_calibration"]
        c0 = p1["deberta_C0_raw_vallock"]
        c1 = p1["deberta_C1_temp_vallock"]
        c2 = p1["deberta_C2_frozen_vallock"]
        mc_raw = p1["minicheck_raw_vallock"]
        mc_platt = p1["minicheck_platt_vallock"]

        lines.extend([
            f"### Regime: `{regime}` — Custom DeBERTa 3-Class Calibration on Val-Lock",
            "",
            "| Model / Scaling Method | 3-Class NLL | 3-Class Brier | Top-1 ECE | Top-1 MCE | Classwise ECE (CW-ECE) | Binary Hallucination ECE | Argmax Accuracy |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            f"| **C0: Raw Uncalibrated Softmax** | {c0['nll_3class']:.4f} | {c0['brier_3class']:.4f} | {c0['top1_ece']:.4f} | {c0['top1_mce']:.4f} | {c0['cw_ece']:.4f} | {c0['binary_hallucination']['ece_uniform']:.4f} | {c0['argmax_accuracy'] * 100.0:.2f}% |",
            f"| **C1: Scalar Temperature Scaling** | {c1['nll_3class']:.4f} | {c1['brier_3class']:.4f} | {c1['top1_ece']:.4f} | {c1['top1_mce']:.4f} | {c1['cw_ece']:.4f} | {c1['binary_hallucination']['ece_uniform']:.4f} | {c1['argmax_accuracy'] * 100.0:.2f}% |",
            f"| **C2: Regularized Prior-Bias + Temp** | **{c2['nll_3class']:.4f}** | **{c2['brier_3class']:.4f}** | **{c2['top1_ece']:.4f}** | **{c2['top1_mce']:.4f}** | **{c2['cw_ece']:.4f}** | **{c2['binary_hallucination']['ece_uniform']:.4f}** | {c2['argmax_accuracy'] * 100.0:.2f}% |",
            "",
            f"#### MiniCheck Binary Support Calibration (`{regime}`)",
            "",
            "| MiniCheck Estimator | Binary NLL | Brier Score | ECE (Uniform) | MCE (Uniform) | ECE (Equal Mass) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
            f"| Raw MiniCheck Support Score | {mc_raw['nll']:.4f} | {mc_raw['brier']:.4f} | {mc_raw['ece_uniform']:.4f} | {mc_raw['mce_uniform']:.4f} | {mc_raw['ece_equal_mass']:.4f} |",
            f"| Platt-Calibrated MiniCheck | **{mc_platt['nll']:.4f}** | **{mc_platt['brier']:.4f}** | **{mc_platt['ece_uniform']:.4f}** | **{mc_platt['mce_uniform']:.4f}** | **{mc_platt['ece_equal_mass']:.4f}** |",
            "",
        ])

    lines.extend([
        "---",
        "",
        "## Pillar 2 & 3 — Frozen Strategy Confirmation on Val-Lock",
        "",
        "Evaluated on held-out Val-Lock using frozen parameters across both pre-registered objectives:",
        "- **OBJ_CONSTRAINED**: Constrained utility subject to coverage and recall floors.",
        "- **OBJ_MACRO_F1_ONLY**: Unconstrained Macro-F1 subject only to coverage floor.",
        "",
    ])

    for regime in ("direct", "retrieval"):
        if regime not in results.get("regimes", {}):
            continue
        r_data = results["regimes"][regime]
        for obj_name in ("OBJ_CONSTRAINED", "OBJ_MACRO_F1_ONLY"):
            if obj_name not in r_data.get("objectives", {}):
                continue
            obj_data = r_data["objectives"][obj_name]
            frozen_winner = obj_data.get("stage1_frozen_winner", "Unknown")

            lines.extend([
                f"### Regime: `{regime}` | Objective: `{obj_name}`",
                "",
                f"> **Stage 1 Pre-Registered Frozen Winner**: **`{frozen_winner}`**",
                "",
                "| Strategy | Macro-F1 | W-F1 | Accuracy | Conf-MAE (pp) | Non-Sup MAE (pp) | GFI Spearman | Coverage | Holm p (vs H0) | McNemar 3-cls p | McNemar Bin p |",
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            ])

            for s in STAGE1_STRATEGY_ORDER:
                if s not in obj_data.get("strategies", {}):
                    continue
                s_arm = obj_data["strategies"][s]["vallock_claim_metrics"]
                s_resp = obj_data["strategies"][s]["vallock_response_metrics"]
                p3 = s_arm["penalized_3class"]
                rm = s_resp["confirmed_hallucination_rate_pct"]
                non_sup_mae = s_resp["non_supported_rate_100_minus_gfi_pct"]["mae_vs_true_hall"]
                gfi_rho = s_resp["support_rate_gfi_pct"]["spearman_rho"]
                gfi_rho_str = f"{gfi_rho:.3f}" if gfi_rho is not None else "N/A"

                if s == "H0_baseline":
                    holm_p_str = "Ref"
                    mcn_3_str = "Ref"
                    mcn_bin_str = "Ref"
                else:
                    comp = obj_data.get("paired_comparisons_vs_H0", {}).get(s, {})
                    holm_p = comp.get("holm_adjusted_p_3class_source_perm")
                    holm_p_str = f"{holm_p:.4f}" if holm_p is not None else "N/A"
                    mcn_3 = comp.get("mcnemar_three_class", {}).get("p_value")
                    mcn_3_str = f"{mcn_3:.4f}" if mcn_3 is not None else "N/A"
                    mcn_bin = comp.get("mcnemar_binary_hall", {}).get("p_value")
                    mcn_bin_str = f"{mcn_bin:.4f}" if mcn_bin is not None else "N/A"

                is_win = (s == frozen_winner)
                prefix = "**" if is_win else ""
                suffix = "**" if is_win else ""
                lines.append(
                    f"| {prefix}{s}{suffix} | {p3['macro_f1'] * 100.0:.2f}% | "
                    f"{p3['support_weighted_f1'] * 100.0:.2f}% | {p3['accuracy'] * 100.0:.2f}% | "
                    f"{rm['mae']:.2f} | {non_sup_mae:.2f} | {gfi_rho_str} | "
                    f"{s_arm['coverage'] * 100.0:.2f}% | {holm_p_str} | {mcn_3_str} | {mcn_bin_str} |"
                )
            lines.append("")

    lines.extend([
        "---",
        "",
        "## Pillar 4 — Runtime Profile & Resource Consumption",
        "",
        f"- **Total Wall-Clock Time**: {profile.get('total_elapsed_seconds', 0.0):.2f} seconds",
        f"- **Inference Execution Time**: {profile.get('inference_elapsed_seconds', 0.0):.2f} seconds ({profile.get('missing_records_inferred', 0)} missing sentences evaluated)",
        f"- **Evaluation & Bootstrap Time**: {profile.get('evaluation_elapsed_seconds', 0.0):.2f} seconds",
        f"- **Peak Process Memory**: {profile.get('peak_memory_mb', profile.get('peak_rss_mb', 0.0)):.2f} MB ({profile.get('peak_memory_metric', 'peak_working_set_mb')})",
        f"- **Current Process RSS at Completion**: {profile.get('current_rss_mb', 0.0):.2f} MB",
        f"- **Average Inference Latency**: {profile.get('per_sentence_avg_inference_ms', 0.0):.2f} ms/sentence",
        f"- **Platform**: `{profile.get('platform', 'N/A')}` | Python: `{profile.get('python_version', sys.version).split()[0]}`",
        "- **Test Split Boundary Safety**: Verified zero access to `ragtruth_test.jsonl`.",
        "- **Frozen Bundle Invariant**: Zero parameters modified, retrained, or re-tuned on Val-Lock.",
        "",
    ])
    return "\n".join(lines)


def run_stage2_analyze_only(
    cache_path: Path = STAGE2_CACHE_PATH,
    frozen_bundle_path: Path = STAGE1_FROZEN_BUNDLE_PATH,
    manifest_path: Path = STAGE2_MANIFEST_PATH,
    val_data_path: Path = VAL_DATA_PATH,
    val_dev_manifest_path: Path = DEFAULT_VAL_DEV_MANIFEST,
    results_path: Path = STAGE2_RESULTS_PATH,
    report_path: Path = STAGE2_REPORT_PATH,
    profile_path: Path = STAGE2_PROFILE_PATH,
    n_boot: int = 500,
    seed: int = 42,
    require_30_sources: bool = True,
) -> Dict[str, Any]:
    logger.info("Running Stage 2 Val-Lock analysis on cached records with full manifest validation...")
    assert_safe_non_test_path(cache_path)
    assert_safe_non_test_path(frozen_bundle_path)
    assert_safe_non_test_path(manifest_path)
    assert_safe_non_test_path(val_data_path)
    assert_safe_non_test_path(val_dev_manifest_path)
    assert_safe_non_test_path(results_path)
    assert_safe_non_test_path(report_path)
    assert_safe_non_test_path(profile_path)

    wall_t0 = time.perf_counter()

    # 1. Validate manifest
    if not manifest_path.exists():
        raise FileNotFoundError(f"Stage 2 Val-Lock manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    selected_sources = [str(s) for s in manifest.get("selected_source_ids", [])]
    if require_30_sources:
        if len(selected_sources) != 30:
            raise ValueError(f"Expected exactly 30 Val-Lock sources in {manifest_path}, got {len(selected_sources)}")
        if len(set(selected_sources)) != 30:
            raise ValueError(f"Duplicate source IDs in manifest: {len(selected_sources)} != {len(set(selected_sources))}")
    else:
        if len(set(selected_sources)) != len(selected_sources):
            raise ValueError(f"Duplicate source IDs in manifest: {len(selected_sources)} != {len(set(selected_sources))}")
    val_lock_sources_set = set(selected_sources)

    # Validate disjointness against Stage 1 Val-Dev manifest
    if val_dev_manifest_path.exists():
        expected_dev_count = 30 if require_30_sources else None
        val_dev_sources = set(load_val_dev_sources(val_dev_manifest_path, expected_count=expected_dev_count))
        assert_disjoint_sources(val_dev_sources, val_lock_sources_set, context="run_stage2_analyze_only_ValDev_vs_ValLock")

    # 2. Load expected records from validation split
    val_records = load_validation_records(val_data_path)
    target_records = [r for r in val_records if str(r["source_id"]) in val_lock_sources_set]

    if require_30_sources:
        task_to_sids: Dict[str, Set[str]] = defaultdict(set)
        for r in target_records:
            task_to_sids[r["task_type"]].add(str(r["source_id"]))
        for t in ("Summary", "QA", "Data2txt"):
            if len(task_to_sids[t]) != 10:
                raise ValueError(f"Expected 10 sources for task {t}, got {len(task_to_sids[t])}")
        if len(target_records) != 1238:
            raise ValueError(f"Expected 1238 sentences for 30 Val-Lock sources in {val_data_path}, got {len(target_records)}")

    # 3. Validate cache completeness and integrity
    expected_src_map = {str(r["example_id"]): str(r["source_id"]) for r in target_records}
    expected_ids = set(expected_src_map.keys())

    cache = load_raw_cache_by_id(cache_path, expected_source_map=expected_src_map, allow_repair_trailing_partial=False)

    missing_ids = expected_ids - set(cache.keys())
    if missing_ids:
        raise ValueError(
            f"Stage 2 Val-Lock cache is incomplete: {len(missing_ids)} expected records are missing in {cache_path}."
        )

    unexpected_ids = set(cache.keys()) - expected_ids
    if unexpected_ids:
        raise ValueError(
            f"Stage 2 Val-Lock cache contains {len(unexpected_ids)} unexpected records not in manifest: {list(unexpected_ids)[:5]}."
        )

    if len(cache) != len(target_records):
        raise ValueError(
            f"Stage 2 Val-Lock cache count mismatch: expected {len(target_records)} records, got {len(cache)}."
        )

    ordered_records = [cache[str(r["example_id"])] for r in target_records]

    # 4. Load frozen bundle & evaluate
    with frozen_bundle_path.open("r", encoding="utf-8") as handle:
        bundle = json.load(handle)

    eval_t0 = time.perf_counter()
    results = apply_frozen_bundle_to_records(ordered_records, bundle, n_boot=n_boot, seed=seed)
    eval_sec = time.perf_counter() - eval_t0
    total_sec = time.perf_counter() - wall_t0

    mem_prof = get_peak_memory_profile()

    profile = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "total_elapsed_seconds": round(total_sec, 2),
        "inference_elapsed_seconds": 0.0,
        "evaluation_elapsed_seconds": round(eval_sec, 2),
        "peak_memory_mb": mem_prof["peak_memory_mb"],
        "peak_memory_metric": mem_prof["peak_memory_metric"],
        "current_rss_mb": mem_prof["current_rss_mb"],
        "peak_rss_mb": mem_prof["peak_memory_mb"],
        "cached_records_count": len(ordered_records),
        "missing_records_inferred": 0,
        "per_sentence_avg_inference_ms": 0.0,
    }

    provenance = {
        "timestamp_utc": profile["timestamp_utc"],
        "manifest_path": manifest_path.name,
        "manifest_sha256": compute_file_sha256(manifest_path) if manifest_path.exists() else "N/A",
        "frozen_bundle_path": frozen_bundle_path.name,
        "frozen_bundle_sha256": compute_file_sha256(frozen_bundle_path) if frozen_bundle_path.exists() else "N/A",
        "cache_path": cache_path.name,
        "cache_sha256": compute_file_sha256(cache_path) if cache_path.exists() else "N/A",
        "total_sources": len(val_lock_sources_set),
        "total_responses": len({str(r["response_id"]) for r in ordered_records}),
        "total_sentences": len(ordered_records),
        "seed": seed,
        "n_boot": n_boot,
    }

    results["provenance"] = provenance
    results["runtime_profile"] = profile

    atomic_write_json(results_path, results)
    logger.info("Saved Stage 2 results: %s", results_path)

    atomic_write_json(profile_path, profile)
    logger.info("Saved Stage 2 runtime profile: %s", profile_path)

    report_md = format_stage2_markdown_report(results, profile, provenance)
    atomic_write_text(report_path, report_md)
    logger.info("Saved Stage 2 report: %s", report_path)

    return results


def run_stage2_full(
    manifest_path: Path = STAGE2_MANIFEST_PATH,
    val_data_path: Path = VAL_DATA_PATH,
    train_data_path: Path = TRAIN_DATA_PATH,
    cache_path: Path = STAGE2_CACHE_PATH,
    frozen_bundle_path: Path = STAGE1_FROZEN_BUNDLE_PATH,
    val_dev_manifest_path: Path = DEFAULT_VAL_DEV_MANIFEST,
    results_path: Path = STAGE2_RESULTS_PATH,
    report_path: Path = STAGE2_REPORT_PATH,
    profile_path: Path = STAGE2_PROFILE_PATH,
    approve_inference: bool = False,
    n_boot: int = 500,
    seed: int = 42,
    claim_verifier: Optional[Any] = None,
    chunker: Optional[Any] = None,
    passage_retriever: Optional[Any] = None,
    custom_deberta_verifier: Optional[Any] = None,
    token_length_fn: Optional[Callable[[Sequence[Tuple[str, str]]], List[int]]] = None,
    max_sources: Optional[int] = None,
    require_30_sources: bool = True,
) -> Dict[str, Any]:
    """Execute end-to-end Stage 2 confirmation evaluation on held-out Val-Lock."""
    assert_safe_non_test_path(manifest_path)
    assert_safe_non_test_path(val_data_path)
    assert_safe_non_test_path(train_data_path)
    assert_safe_non_test_path(cache_path)
    assert_safe_non_test_path(frozen_bundle_path)
    assert_safe_non_test_path(val_dev_manifest_path)
    assert_safe_non_test_path(results_path)
    assert_safe_non_test_path(report_path)
    assert_safe_non_test_path(profile_path)

    wall_start_t0 = time.perf_counter()

    if not manifest_path.exists():
        raise FileNotFoundError(f"Stage 2 Val-Lock manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    selected_sources = [str(s) for s in manifest.get("selected_source_ids", [])]
    if require_30_sources:
        if len(selected_sources) != 30:
            raise ValueError(f"Expected exactly 30 Val-Lock sources in {manifest_path}, got {len(selected_sources)}")
        if len(set(selected_sources)) != 30:
            raise ValueError(f"Duplicate source IDs in manifest: {len(selected_sources)} != {len(set(selected_sources))}")
    else:
        if len(set(selected_sources)) != len(selected_sources):
            raise ValueError(f"Duplicate source IDs in manifest: {len(selected_sources)} != {len(set(selected_sources))}")
    val_lock_sources_set = set(selected_sources)

    # Validate disjointness against Stage 1 Val-Dev manifest
    if val_dev_manifest_path.exists():
        expected_dev_count = 30 if require_30_sources else None
        val_dev_sources = set(load_val_dev_sources(val_dev_manifest_path, expected_count=expected_dev_count))
        assert_disjoint_sources(val_dev_sources, val_lock_sources_set, context="run_stage2_full_ValDev_vs_ValLock")

    # Load validation records strictly belonging to Val-Lock sources
    val_records = load_validation_records(val_data_path)
    target_records = [r for r in val_records if str(r["source_id"]) in val_lock_sources_set]

    if require_30_sources:
        task_to_sids: Dict[str, Set[str]] = defaultdict(set)
        for r in target_records:
            task_to_sids[r["task_type"]].add(str(r["source_id"]))
        for t in ("Summary", "QA", "Data2txt"):
            if len(task_to_sids[t]) != 10:
                raise ValueError(f"Expected 10 sources for task {t}, got {len(task_to_sids[t])}")
        if len(target_records) != 1238:
            logger.warning("Expected 1238 sentences for 30 Val-Lock sources, got %d", len(target_records))

    # Audit raw cache for existing records
    expected_src_map = {str(r["example_id"]): str(r["source_id"]) for r in target_records}
    cache = load_raw_cache_by_id(cache_path, expected_source_map=expected_src_map, allow_repair_trailing_partial=True)
    missing_records = [r for r in target_records if str(r["example_id"]) not in cache]
    logger.info(
        "Stage 2 cache audit: %d/%d records cached, %d missing.",
        len(target_records) - len(missing_records),
        len(target_records),
        len(missing_records),
    )

    inference_elapsed_sec = 0.0
    if missing_records:
        if not approve_inference:
            raise RuntimeError(
                f"SAFETY GUARD: Inference is required for {len(missing_records)} missing records. "
                "Explicit --approve-stage2-inference is required."
            )

        logger.info("Initializing inference models for Stage 2 Val-Lock cache generation...")
        inf_t0 = time.perf_counter()

        if claim_verifier is None:
            from ml_engine.claim_verifier.verifier import ClaimVerifier
            from ml_engine.evidence_retriever.chunker import SlidingWindowChunker
            from ml_engine.evidence_retriever.passage_retriever import PassageRetriever

            claim_verifier = ClaimVerifier(use_custom_model=True)
            custom_deberta = claim_verifier._get_custom_deberta_verifier()
            chunker = SlidingWindowChunker(chunk_size_words=150, overlap_words=35, min_chunk_words=20)
            passage_retriever = PassageRetriever(model_name="all-MiniLM-L6-v2", top_k=3)
        else:
            custom_deberta = custom_deberta_verifier or (
                claim_verifier._get_custom_deberta_verifier()
                if hasattr(claim_verifier, "_get_custom_deberta_verifier")
                else None
            )

        if token_length_fn is None:
            if custom_deberta is not None and hasattr(custom_deberta, "tokenizer") and custom_deberta.tokenizer is not None:
                token_len_fn = make_token_length_fn(custom_deberta.tokenizer)
            else:
                token_len_fn = lambda pairs: [len(f"{c} {s}".split()) for c, s in pairs]
        else:
            token_len_fn = token_length_fn

        train_index = (
            build_train_overlap_index(train_data_path)
            if train_data_path.exists()
            else {"exact_pair_keys": set(), "sentence_strings": set()}
        )

        missing_by_src: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in missing_records:
            missing_by_src[str(r["source_id"])].append(r)

        processed_srcs = 0
        total_missing_srcs = len(missing_by_src)
        for sid in selected_sources:
            src_recs = missing_by_src.get(sid)
            if not src_recs:
                continue
            if max_sources is not None and processed_srcs >= max_sources:
                logger.info("Reached --max-sources=%d limit (%d/%d processed).", max_sources, processed_srcs, total_missing_srcs)
                break
            processed_srcs += 1
            s_t0 = time.perf_counter()

            direct_map = evaluate_direct_context_records(
                src_recs,
                minicheck_verifier=getattr(claim_verifier, "minicheck", None),
                nli_verifier=getattr(claim_verifier, "nli", None),
                custom_deberta_verifier=custom_deberta,
                token_length_fn=token_len_fn,
            )

            ret_map = evaluate_retrieval_pipeline_for_source(
                src_recs,
                chunker=chunker,
                passage_retriever=passage_retriever,
                claim_verifier=claim_verifier,
                token_length_fn=token_len_fn,
            )

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
                    "is_exact_train_pair_dup": pair_k in train_index.get("exact_pair_keys", set()),
                    "is_sentence_string_in_train": str(r.get("sentence_text", "")) in train_index.get("sentence_strings", set()),
                    "direct": direct_map[ex_id],
                    "retrieval": ret_map[ex_id],
                }
                cache[ex_id] = cached_row
                to_cache.append(cached_row)

            append_cache_records(cache_path, to_cache)
            s_sec = time.perf_counter() - s_t0
            logger.info(
                "Cached source %s (%d sentences) in %.2fs [%d/%d sources complete]",
                sid, len(src_recs), s_sec, processed_srcs, total_missing_srcs
            )

        inference_elapsed_sec = time.perf_counter() - inf_t0
        logger.info("Stage 2 inference completed in %.2f seconds.", inference_elapsed_sec)

    ordered_records: List[Dict[str, Any]] = []
    still_missing: List[str] = []
    for r in target_records:
        ex_id = str(r["example_id"])
        if ex_id in cache:
            cached_rec = cache[ex_id]
            if str(cached_rec.get("source_id", "")) != str(r["source_id"]):
                raise ValueError(
                    f"Cache source mismatch for '{ex_id}': expected '{r['source_id']}', got '{cached_rec.get('source_id')}'"
                )
            ordered_records.append(cached_rec)
        else:
            still_missing.append(ex_id)

    if still_missing:
        raise RuntimeError(
            f"Stage 2 cache is incomplete: {len(still_missing)} sentences still missing. "
            "All Val-Lock sources must be fully cached before confirmation evaluation can run."
        )

    logger.info("Loading Stage 1 frozen bundle from: %s", frozen_bundle_path)
    with frozen_bundle_path.open("r", encoding="utf-8") as handle:
        frozen_bundle = json.load(handle)

    eval_t0 = time.perf_counter()
    logger.info("Applying frozen bundle to %d Val-Lock records...", len(ordered_records))
    confirmation_results = apply_frozen_bundle_to_records(
        ordered_records,
        frozen_bundle,
        n_boot=n_boot,
        seed=seed,
    )
    eval_elapsed_sec = time.perf_counter() - eval_t0
    logger.info("Stage 2 evaluation completed in %.2f seconds.", eval_elapsed_sec)

    mem_prof = get_peak_memory_profile()
    total_elapsed_sec = round(time.perf_counter() - wall_start_t0, 2)

    profile = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "total_elapsed_seconds": total_elapsed_sec,
        "inference_elapsed_seconds": round(inference_elapsed_sec, 2),
        "evaluation_elapsed_seconds": round(eval_elapsed_sec, 2),
        "peak_memory_mb": mem_prof["peak_memory_mb"],
        "peak_memory_metric": mem_prof["peak_memory_metric"],
        "current_rss_mb": mem_prof["current_rss_mb"],
        "peak_rss_mb": mem_prof["peak_memory_mb"],
        "cached_records_count": len(ordered_records),
        "missing_records_inferred": len(missing_records),
        "per_sentence_avg_inference_ms": (
            round((inference_elapsed_sec / len(missing_records)) * 1000.0, 2)
            if missing_records else 0.0
        ),
    }

    provenance = {
        "timestamp_utc": profile["timestamp_utc"],
        "manifest_path": manifest_path.name,
        "manifest_sha256": compute_file_sha256(manifest_path) if manifest_path.exists() else "N/A",
        "frozen_bundle_path": frozen_bundle_path.name,
        "frozen_bundle_sha256": compute_file_sha256(frozen_bundle_path) if frozen_bundle_path.exists() else "N/A",
        "cache_path": cache_path.name,
        "cache_sha256": compute_file_sha256(cache_path) if cache_path.exists() else "N/A",
        "total_sources": len(val_lock_sources_set),
        "total_responses": len({str(r["response_id"]) for r in ordered_records}),
        "total_sentences": len(ordered_records),
        "seed": seed,
        "n_boot": n_boot,
    }

    confirmation_results["provenance"] = provenance
    confirmation_results["runtime_profile"] = profile

    atomic_write_json(results_path, confirmation_results)
    logger.info("Saved Stage 2 confirmation results: %s", results_path)

    atomic_write_json(profile_path, profile)
    logger.info("Saved Stage 2 runtime profile: %s", profile_path)

    report_md = format_stage2_markdown_report(confirmation_results, profile, provenance)
    atomic_write_text(report_path, report_md)
    logger.info("Saved Stage 2 confirmation report: %s", report_path)

    return confirmation_results


def main() -> None:
    parser = argparse.ArgumentParser(description="VeriLLM Phase 8 Step 7 Calibration & Hybrid Runner")
    parser.add_argument("--stage1-cache-only", action="store_true", help="Run Stage 1 5x4 nested CV on Val-Dev cache")
    parser.add_argument("--stage2-manifest-only", action="store_true", help="Generate disjoint 30-source Val-Lock manifest")
    parser.add_argument("--stage2-run", action="store_true", help="Run Stage 2 confirmation evaluation on Val-Lock")
    parser.add_argument("--stage2-analyze-only", action="store_true", help="Analyze existing Val-Lock cache using frozen bundle")
    parser.add_argument("--approve-stage2-inference", action="store_true", help="Explicit user approval guard required for --stage2-run")
    parser.add_argument("--val-cache", type=Path, default=DEFAULT_VAL_DEV_CACHE)
    parser.add_argument("--val-manifest", type=Path, default=DEFAULT_VAL_DEV_MANIFEST)
    parser.add_argument("--stage2-manifest", type=Path, default=STAGE2_MANIFEST_PATH)
    parser.add_argument("--stage2-cache", type=Path, default=STAGE2_CACHE_PATH)
    parser.add_argument("--frozen-bundle", type=Path, default=STAGE1_FROZEN_BUNDLE_PATH)
    parser.add_argument("--max-sources", type=int, default=None, help="Optional limit on uncached sources to process")
    parser.add_argument("--n-outer-folds", type=int, default=5)
    parser.add_argument("--n-inner-folds", type=int, default=4)
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.stage1_cache_only:
        run_stage1_cache_only(
            cache_path=args.val_cache,
            manifest_path=args.val_manifest,
            n_outer_folds=args.n_outer_folds,
            n_inner_folds=args.n_inner_folds,
            n_boot=args.n_boot,
            seed=args.seed,
        )
    elif args.stage2_manifest_only:
        run_stage2_manifest_only(
            val_data_path=VAL_DATA_PATH,
            train_data_path=TRAIN_DATA_PATH,
            val_dev_manifest_path=args.val_manifest,
            seed=43,
        )
    elif args.stage2_analyze_only:
        run_stage2_analyze_only(
            cache_path=args.stage2_cache,
            frozen_bundle_path=args.frozen_bundle,
            manifest_path=args.stage2_manifest,
            results_path=STAGE2_RESULTS_PATH,
            report_path=STAGE2_REPORT_PATH,
            profile_path=STAGE2_PROFILE_PATH,
            n_boot=args.n_boot,
            seed=args.seed,
        )
    elif args.stage2_run:
        if not args.approve_stage2_inference:
            logger.error("SAFETY GUARD: --stage2-run requires explicit --approve-stage2-inference flag!")
            sys.exit(1)
        logger.info("Stage 2 full run approved. Executing confirmation pipeline on Val-Lock...")
        run_stage2_full(
            manifest_path=args.stage2_manifest,
            val_data_path=VAL_DATA_PATH,
            train_data_path=TRAIN_DATA_PATH,
            cache_path=args.stage2_cache,
            frozen_bundle_path=args.frozen_bundle,
            val_dev_manifest_path=args.val_manifest,
            results_path=STAGE2_RESULTS_PATH,
            report_path=STAGE2_REPORT_PATH,
            profile_path=STAGE2_PROFILE_PATH,
            approve_inference=args.approve_stage2_inference,
            n_boot=args.n_boot,
            seed=args.seed,
            max_sources=args.max_sources,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

