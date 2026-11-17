"""
VeriLLM Phase 7: RAGTruth Dataset Preprocessor.
Performs sentence-level alignment, label mapping (SUPPORTED_PROXY, CONTRADICTED, UNSUPPORTED, MIXED_HALLUCINATION),
source-grouped train/val/test splitting, deduplication, length analysis, and quality filtering.
"""

import json
import random
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Any, Tuple, Optional
import spacy

CURRENT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = CURRENT_DIR.parent.parent
RAW_DATA_DIR = BACKEND_DIR / "data" / "ragtruth"
PROCESSED_DATA_DIR = BACKEND_DIR / "data" / "processed"


def format_source_context(task_type_or_dict: Any, source_info: Any = None) -> str:
    """
    Format source_info into a clean text context depending on task_type.
    Accepts either:
      - format_source_context(source_dict) where source_dict contains 'task_type' and 'source_info'
      - format_source_context(task_type, source_info)
    """
    if source_info is None and isinstance(task_type_or_dict, dict):
        source_dict = task_type_or_dict
        task_type = source_dict.get("task_type", "")
        source_info = source_dict.get("source_info", "")
    else:
        task_type = str(task_type_or_dict)

    if isinstance(source_info, str):
        return f"Document Context:\n{source_info.strip()}"
    elif isinstance(source_info, dict):
        if task_type == "QA":
            q = source_info.get("question", "").strip()
            p = source_info.get("passages", "") or source_info.get("context", "")
            return f"Question: {q}\n\nPassage Context:\n{str(p).strip()}"
        elif task_type == "Data2txt":
            lines = ["Structured Data Table / Entities:"]
            for k, v in source_info.items():
                if isinstance(v, dict):
                    v_str = ", ".join(f"{sub_k}: {sub_v}" for sub_k, sub_v in v.items())
                else:
                    v_str = str(v)
                lines.append(f"- {k}: {v_str}")
            return "\n".join(lines)
        else:
            return "\n".join(f"- {k}: {v}" for k, v in source_info.items())
    elif isinstance(source_info, list):
        lines = ["Structured Data Table / Entities:"]
        for item in source_info:
            lines.append(f"- {item}")
        return "\n".join(lines)
    return str(source_info)


def determine_sentence_label(
    sent_start: int,
    sent_end: int,
    sent_text: str,
    labels_list: List[Dict[str, Any]],
    conflict_types: Optional[set] = None,
    baseless_types: Optional[set] = None
) -> Tuple[str, List[Dict[str, Any]], bool, bool, bool, bool]:
    """
    Determine the primary label and overlapping spans for a candidate sentence.

    Args:
        sent_start: Character start index of sentence in the response.
        sent_end: Character end index of sentence in the response.
        sent_text: Full text of candidate sentence.
        labels_list: List of hallucination annotation dictionaries for the response.
        conflict_types: Optional set of conflict annotation label types.
        baseless_types: Optional set of baseless info annotation label types.

    Returns:
        Tuple of:
            - primary_label: "SUPPORTED_PROXY", "CONTRADICTED", "UNSUPPORTED", or "MIXED_HALLUCINATION"
            - overlapping_spans: List of matched hallucination span dicts with offsets and overlap ratios
            - has_conflict: bool indicating whether any conflict span overlapped
            - has_baseless: bool indicating whether any baseless info span overlapped
            - has_implicit_true: bool indicating if any overlapping span was flagged implicit_true
            - has_due_to_null: bool indicating if any overlapping span was flagged due_to_null
    """
    conflict_types = conflict_types or {"Evident Conflict", "Subtle Conflict"}
    baseless_types = baseless_types or {"Evident Baseless Info", "Subtle Baseless Info"}

    overlapping_spans = []
    has_conflict = False
    has_baseless = False
    has_implicit_true = False
    has_due_to_null = False
    max_overlap_ratio = 0.0

    for lbl in labels_list:
        l_start = lbl.get("start", 0)
        l_end = lbl.get("end", 0)

        # Check character overlap: max(sent_start, l_start) < min(sent_end, l_end)
        if max(sent_start, l_start) < min(sent_end, l_end):
            overlap_len = min(sent_end, l_end) - max(sent_start, l_start)
            overlap_ratio = round(overlap_len / max(len(sent_text), 1), 4)
            if overlap_ratio > max_overlap_ratio:
                max_overlap_ratio = overlap_ratio

            l_type = lbl.get("label_type", "")
            if l_type in conflict_types:
                has_conflict = True
            if l_type in baseless_types:
                has_baseless = True
            if lbl.get("implicit_true"):
                has_implicit_true = True
            if lbl.get("due_to_null"):
                has_due_to_null = True

            overlapping_spans.append({
                "text": lbl.get("text", ""),
                "label_type": l_type,
                "start": l_start,
                "end": l_end,
                "overlap_ratio": overlap_ratio,
                "implicit_true": lbl.get("implicit_true", False),
                "due_to_null": lbl.get("due_to_null", False)
            })

    # Policy assignment:
    # 1. No overlapping hallucination spans -> SUPPORTED_PROXY
    # 2. Both conflict and baseless spans in same sentence -> MIXED_HALLUCINATION
    # 3. Only conflict spans -> CONTRADICTED
    # 4. Only baseless spans -> UNSUPPORTED
    if not overlapping_spans:
        primary_label = "SUPPORTED_PROXY"
    elif has_conflict and has_baseless:
        primary_label = "MIXED_HALLUCINATION"
    elif has_conflict:
        primary_label = "CONTRADICTED"
    else:
        primary_label = "UNSUPPORTED"

    return primary_label, overlapping_spans, has_conflict, has_baseless, has_implicit_true, has_due_to_null


class RAGTruthPreprocessor:
    """
    Parses, segments, labels, and partitions the RAGTruth dataset.
    """

    def __init__(self, random_seed: int = 42, val_ratio: float = 0.15):
        self.random_seed = random_seed
        self.val_ratio = val_ratio
        self.nlp = spacy.blank("en")
        self.nlp.add_pipe("sentencizer")

    def process(self) -> Dict[str, Any]:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        results_dir = BACKEND_DIR / "experiments" / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        source_file = RAW_DATA_DIR / "source_info.jsonl"
        response_file = RAW_DATA_DIR / "response.jsonl"

        # -------------------------------------------------------------
        # Step 1: Load Sources
        # -------------------------------------------------------------
        print("Loading source contexts from source_info.jsonl...")
        sources: Dict[str, Dict[str, Any]] = {}
        with open(source_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                sources[str(item["source_id"])] = item

        # -------------------------------------------------------------
        # Step 2: Stratified Train / Validation Split on Official Train Sources
        # -------------------------------------------------------------
        print("Partitioning train sources into train and validation sets (source-grouped)...")
        train_sources_by_task = defaultdict(list)
        test_source_ids = set()

        # Identify which sources belong to train vs test in raw response.jsonl
        with open(response_file, "r", encoding="utf-8") as f:
            for line in f:
                r_item = json.loads(line)
                sid = str(r_item["source_id"])
                sp = r_item.get("split", "train")
                if sp == "test":
                    test_source_ids.add(sid)
                else:
                    t_type = sources.get(sid, {}).get("task_type", "unknown")
                    if sid not in train_sources_by_task[t_type]:
                        train_sources_by_task[t_type].append(sid)

        # Stratified random split by task_type
        rng = random.Random(self.random_seed)
        val_source_ids = set()
        train_source_ids = set()

        for t_type, sids in train_sources_by_task.items():
            sids_sorted = sorted(sids)  # Deterministic order
            rng.shuffle(sids_sorted)
            num_val = int(len(sids_sorted) * self.val_ratio)
            val_sids = set(sids_sorted[:num_val])
            train_sids = set(sids_sorted[num_val:])
            val_source_ids.update(val_sids)
            train_source_ids.update(train_sids)

        print(f"  Source split counts: Train={len(train_source_ids)}, Val={len(val_source_ids)}, Test={len(test_source_ids)}")

        # Source leakage assertion
        assert len(train_source_ids.intersection(val_source_ids)) == 0, "Train-Val source overlap detected!"
        assert len(train_source_ids.intersection(test_source_ids)) == 0, "Train-Test source overlap detected!"
        assert len(val_source_ids.intersection(test_source_ids)) == 0, "Val-Test source overlap detected!"

        # -------------------------------------------------------------
        # Step 3: Parse and Segment Responses into Sentence-Level Examples
        # -------------------------------------------------------------
        print("Parsing responses and aligning sentence-level spans...")
        split_examples = {"train": [], "val": [], "test": []}
        quality_counts = Counter()
        exclusions_count = 0
        duplicate_sentence_counter = Counter()

        label_counts = {
            "train": Counter(),
            "val": Counter(),
            "test": Counter(),
            "total": Counter()
        }

        conflict_types = {"Evident Conflict", "Subtle Conflict"}
        baseless_types = {"Evident Baseless Info", "Subtle Baseless Info"}

        token_lengths = {
            "context_words": [],
            "sentence_words": [],
            "combined_words": []
        }

        with open(response_file, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                r_item = json.loads(line)
                resp_id = str(r_item["id"])
                sid = str(r_item["source_id"])
                model_name = r_item.get("model", "unknown")
                raw_split = r_item.get("split", "train")
                quality = r_item.get("quality", "good")
                quality_counts[quality] += 1

                # Quality filtering
                if quality != "good":
                    exclusions_count += 1
                    continue

                # Determine target partition
                if raw_split == "test":
                    target_split = "test"
                elif sid in val_source_ids:
                    target_split = "val"
                else:
                    target_split = "train"

                src_meta = sources.get(sid, {})
                task_type = src_meta.get("task_type", "unknown")
                formatted_context = format_source_context(task_type, src_meta.get("source_info", ""))

                resp_text = r_item.get("response", "").strip()
                labels = r_item.get("labels", [])

                # Segment into sentences
                doc = self.nlp(resp_text)
                for sent_idx, sent in enumerate(doc.sents):
                    s_text = sent.text.strip()
                    if not s_text or len(s_text) < 3:
                        continue

                    s_start = sent.start_char
                    s_end = sent.end_char

                    # Determine label and overlapping spans
                    (
                        primary_label,
                        overlapping_spans,
                        has_conflict,
                        has_baseless,
                        has_implicit_true,
                        has_due_to_null
                    ) = determine_sentence_label(
                        sent_start=s_start,
                        sent_end=s_end,
                        sent_text=s_text,
                        labels_list=labels,
                        conflict_types=conflict_types,
                        baseless_types=baseless_types
                    )
                    max_overlap_ratio = max((sp["overlap_ratio"] for sp in overlapping_spans), default=0.0)

                    label_counts[target_split][primary_label] += 1
                    label_counts["total"][primary_label] += 1

                    # Track lengths (in words as proxy for tokens)
                    c_words = len(formatted_context.split())
                    s_words = len(s_text.split())
                    tot_words = c_words + s_words
                    token_lengths["context_words"].append(c_words)
                    token_lengths["sentence_words"].append(s_words)
                    token_lengths["combined_words"].append(tot_words)

                    duplicate_sentence_counter[s_text] += 1

                    example = {
                        "example_id": f"{resp_id}_s{sent_idx+1:02d}",
                        "response_id": resp_id,
                        "source_id": sid,
                        "task_type": task_type,
                        "model": model_name,
                        "split": target_split,
                        "primary_label": primary_label,
                        "sentence_text": s_text,
                        "sentence_start": s_start,
                        "sentence_end": s_end,
                        "context_text": formatted_context,
                        "has_hallucination": primary_label != "SUPPORTED_PROXY",
                        "max_overlap_ratio": max_overlap_ratio,
                        "has_implicit_true": has_implicit_true,
                        "has_due_to_null": has_due_to_null,
                        "overlapping_spans": overlapping_spans
                    }
                    split_examples[target_split].append(example)

        # -------------------------------------------------------------
        # Step 4: Write Out Partitioned Datasets
        # -------------------------------------------------------------
        print("Writing processed JSONL datasets to disk...")
        for sp_name, ex_list in split_examples.items():
            out_file = PROCESSED_DATA_DIR / f"ragtruth_{sp_name}.jsonl"
            with open(out_file, "w", encoding="utf-8") as out_f:
                for ex in ex_list:
                    out_f.write(json.dumps(ex) + "\n")
            print(f"  -> Saved {len(ex_list):>6} examples to {out_file.name} ({out_file.stat().st_size / (1024*1024):.2f} MB)")

        # -------------------------------------------------------------
        # Step 5: Compute Statistical Distributions
        # -------------------------------------------------------------
        def calc_percentiles(vals):
            vals_sorted = sorted(vals)
            n = len(vals_sorted)
            if n == 0:
                return {}
            return {
                "min": vals_sorted[0],
                "p50_median": vals_sorted[int(n * 0.50)],
                "p90": vals_sorted[int(n * 0.90)],
                "p95": vals_sorted[int(n * 0.95)],
                "p99": vals_sorted[int(n * 0.99)],
                "max": vals_sorted[-1]
            }

        length_stats = {
            "context_word_percentiles": calc_percentiles(token_lengths["context_words"]),
            "sentence_word_percentiles": calc_percentiles(token_lengths["sentence_words"]),
            "combined_word_percentiles": calc_percentiles(token_lengths["combined_words"]),
            "exceeds_512_words_count": sum(1 for w in token_lengths["combined_words"] if w > 512),
            "exceeds_512_words_pct": round(sum(1 for w in token_lengths["combined_words"] if w > 512) / len(token_lengths["combined_words"]) * 100, 2)
        }

        # Select inspectable samples for each label
        inspectable_samples = {}
        for sp in ["train", "val", "test"]:
            for ex in split_examples[sp]:
                lbl = ex["primary_label"]
                if lbl not in inspectable_samples:
                    inspectable_samples[lbl] = {
                        "example_id": ex["example_id"],
                        "task_type": ex["task_type"],
                        "model": ex["model"],
                        "label": lbl,
                        "sentence": ex["sentence_text"],
                        "context_snippet": ex["context_text"][:200] + "...",
                        "overlapping_spans": ex["overlapping_spans"]
                    }
                if len(inspectable_samples) == 4:
                    break

        summary_report = {
            "total_raw_responses": len(quality_counts),
            "quality_distribution": dict(quality_counts),
            "excluded_responses_count": exclusions_count,
            "total_generated_examples": sum(len(v) for v in split_examples.values()),
            "partition_counts": {k: len(v) for k, v in split_examples.items()},
            "source_counts_by_partition": {
                "train_sources": len(train_source_ids),
                "val_sources": len(val_source_ids),
                "test_sources": len(test_source_ids)
            },
            "class_distributions": {
                split_k: dict(label_counts[split_k])
                for split_k in ["train", "val", "test", "total"]
            },
            "length_statistics": length_stats,
            "sample_examples_by_class": inspectable_samples
        }

        # Export JSON and Markdown Reports
        out_json = results_dir / "06_ragtruth_preprocessing_stats.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(summary_report, f, indent=2)

        out_md = results_dir / "06_ragtruth_preprocessing_report.md"
        with open(out_md, "w", encoding="utf-8") as f:
            f.write("# VeriLLM Phase 7: RAGTruth Dataset Preprocessing Report\n\n")
            f.write("## 1. Quality Filtering and Exclusions\n\n")
            f.write(f"- Total Raw Responses: `{sum(quality_counts.values())}`\n")
            f.write(f"- Good Quality: `{quality_counts['good']}` ({quality_counts['good']/sum(quality_counts.values())*100:.1f}%)\n")
            f.write(f"- Truncated (Excluded): `{quality_counts['truncated']}`\n")
            f.write(f"- Incorrect Refusals (Excluded): `{quality_counts['incorrect_refusal']}`\n\n")
            
            f.write("## 2. Partition and Class Distributions\n\n")
            f.write("| Partition | Sources | Total Sentences | SUPPORTED_PROXY | CONTRADICTED | UNSUPPORTED | MIXED_HALLUCINATION |\n")
            f.write("|:---|:---:|:---:|:---:|:---:|:---:|:---:|\n")
            for sp in ["train", "val", "test"]:
                counts = label_counts[sp]
                n_src = len(train_source_ids) if sp == "train" else (len(val_source_ids) if sp == "val" else len(test_source_ids))
                tot = len(split_examples[sp])
                f.write(f"| **`{sp.upper()}`** | {n_src} | {tot} | {counts['SUPPORTED_PROXY']} ({counts['SUPPORTED_PROXY']/tot*100:.1f}%) | {counts['CONTRADICTED']} ({counts['CONTRADICTED']/tot*100:.1f}%) | {counts['UNSUPPORTED']} ({counts['UNSUPPORTED']/tot*100:.1f}%) | {counts['MIXED_HALLUCINATION']} ({counts['MIXED_HALLUCINATION']/tot*100:.1f}%) |\n")
            
            tot_all = sum(len(v) for v in split_examples.values())
            c_all = label_counts["total"]
            f.write(f"| **TOTAL** | {len(sources)} | {tot_all} | {c_all['SUPPORTED_PROXY']} ({c_all['SUPPORTED_PROXY']/tot_all*100:.1f}%) | {c_all['CONTRADICTED']} ({c_all['CONTRADICTED']/tot_all*100:.1f}%) | {c_all['UNSUPPORTED']} ({c_all['UNSUPPORTED']/tot_all*100:.1f}%) | {c_all['MIXED_HALLUCINATION']} ({c_all['MIXED_HALLUCINATION']/tot_all*100:.1f}%) |\n\n")

            f.write("## 3. Sequence Length Statistics\n\n")
            f.write(f"- **Sentence Length**: Median: `{length_stats['sentence_word_percentiles']['p50_median']}` words, 95th: `{length_stats['sentence_word_percentiles']['p95']}` words, Max: `{length_stats['sentence_word_percentiles']['max']}` words\n")
            f.write(f"- **Context Length**: Median: `{length_stats['context_word_percentiles']['p50_median']}` words, 95th: `{length_stats['context_word_percentiles']['p95']}` words, Max: `{length_stats['context_word_percentiles']['max']}` words\n")
            f.write(f"- **Combined Pair Length**: Median: `{length_stats['combined_word_percentiles']['p50_median']}` words, 95th: `{length_stats['combined_word_percentiles']['p95']}` words, 99th: `{length_stats['combined_word_percentiles']['p99']}` words\n")
            f.write(f"- **Pairs exceeding 512 words**: `{length_stats['exceeds_512_words_count']}` ({length_stats['exceeds_512_words_pct']}%)\n\n")

            f.write("## 4. Sample Examples Across Classes\n\n")
            for c_name, sample in inspectable_samples.items():
                f.write(f"### Class: `{c_name}`\n")
                f.write(f"- **Example ID**: `{sample['example_id']}` (Model: `{sample['model']}`, Task: `{sample['task_type']}`)\n")
                f.write(f"- **Sentence**: *\"{sample['sentence']}\"*\n")
                if sample['overlapping_spans']:
                    f.write("- **Annotations**:\n")
                    for sp in sample['overlapping_spans']:
                        f.write(f"  - Span: *\"{sp['text']}\"* | Type: `{sp['label_type']}` | Overlap: `{sp['overlap_ratio']*100:.1f}%`\n")
                f.write("\n")

        print(f"[Done] Preprocessing complete. Report saved to: {out_md}")
        return summary_report


if __name__ == "__main__":
    preprocessor = RAGTruthPreprocessor()
    preprocessor.process()
