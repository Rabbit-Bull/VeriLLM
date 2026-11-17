"""
VeriLLM Phase 7: RAGTruth Dataset In-Depth Inspection Script.
Analyzes the official RAGTruth dataset structure, splits, models, tasks,
hallucination label types, token/span lengths, and split leakage.
"""

import json
from pathlib import Path
from collections import Counter, defaultdict

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "ragtruth"


def inspect():
    source_file = DATA_DIR / "source_info.jsonl"
    response_file = DATA_DIR / "response.jsonl"

    print("=" * 80)
    print(" VeriLLM Phase 7: RAGTruth Dataset In-Depth Inspection")
    print("=" * 80)

    # 1. Inspect Sources
    sources = {}
    task_counter = Counter()
    source_type_counter = Counter()

    with open(source_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            s_id = item["source_id"]
            sources[s_id] = item
            task_counter[item.get("task_type", "unknown")] += 1
            source_type_counter[item.get("source", "unknown")] += 1

    print(f"\n[1] Source Info (Total sources: {len(sources)}):")
    print("  Task Types:")
    for t, count in task_counter.most_common():
        print(f"    - {t:<15}: {count:>5} ({count/len(sources)*100:.1f}%)")
    print("  Source Domains:")
    for s, count in source_type_counter.most_common(10):
        print(f"    - {s:<15}: {count:>5} ({count/len(sources)*100:.1f}%)")

    # Sample source
    sample_sid = next(iter(sources))
    sample_src = sources[sample_sid]
    print(f"\n  Sample Source ID: {sample_sid}")
    print(f"  Task: {sample_src.get('task_type')}")
    print(f"  Source Type: {type(sample_src.get('source_info'))}")
    if isinstance(sample_src.get("source_info"), str):
        print(f"  Source Info Snippet: {sample_src['source_info'][:150]}...")
    elif isinstance(sample_src.get("source_info"), dict):
        print(f"  Source Info Keys: {list(sample_src['source_info'].keys())}")

    # 2. Inspect Responses
    responses = []
    split_counter = Counter()
    model_counter = Counter()
    label_type_counter = Counter()
    responses_with_hallucination = Counter()
    sources_by_split = defaultdict(set)

    hallucination_spans_count = 0
    due_to_null_count = 0
    implicit_true_count = 0

    with open(response_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            responses.append(item)
            split = item.get("split", "unknown")
            split_counter[split] += 1
            model_counter[item.get("model", "unknown")] += 1
            sources_by_split[split].add(item.get("source_id"))

            labels = item.get("labels", [])
            has_hallucination = len(labels) > 0
            responses_with_hallucination[split] += int(has_hallucination)

            for lbl in labels:
                hallucination_spans_count += 1
                l_type = lbl.get("label_type", "unknown")
                label_type_counter[l_type] += 1
                if lbl.get("due_to_null"):
                    due_to_null_count += 1
                if lbl.get("implicit_true"):
                    implicit_true_count += 1

    print(f"\n[2] Response Info (Total responses: {len(responses)}):")
    print("  Splits:")
    for sp, count in split_counter.most_common():
        halluc_resp = responses_with_hallucination[sp]
        print(f"    - {sp:<10}: {count:>6} responses | {halluc_resp:>5} with hallucinations ({halluc_resp/count*100:.1f}%) | {len(sources_by_split[sp]):>5} unique sources")

    print("\n  Models:")
    for m, count in model_counter.most_common():
        print(f"    - {m:<25}: {count:>5} ({count/len(responses)*100:.1f}%)")

    print(f"\n[3] Hallucination Annotations (Total spans: {hallucination_spans_count}):")
    print("  Label Types:")
    for lt, count in label_type_counter.most_common():
        print(f"    - {lt:<25}: {count:>5} ({count/hallucination_spans_count*100:.1f}%)")
    print(f"  Flag due_to_null   : {due_to_null_count:>5} ({due_to_null_count/hallucination_spans_count*100:.1f}%)")
    print(f"  Flag implicit_true : {implicit_true_count:>5} ({implicit_true_count/hallucination_spans_count*100:.1f}%)")

    # 4. Leakage Check
    train_sources = sources_by_split.get("train", set())
    test_sources = sources_by_split.get("test", set())
    overlap = train_sources.intersection(test_sources)
    print(f"\n[4] Data Leakage / Source Overlap Check:")
    print(f"  Train sources count : {len(train_sources)}")
    print(f"  Test sources count  : {len(test_sources)}")
    print(f"  Overlap count       : {len(overlap)} sources")
    if len(overlap) == 0:
        print("  -> STRICT SOURCE ISOLATION: Zero source context overlap between train and test splits!")
    else:
        print(f"  -> WARNING: {len(overlap)} sources are shared between train and test.")

    # 5. Sample Response with Labels
    print("\n[5] Sample Hallucinated Response:")
    for r in responses:
        if len(r.get("labels", [])) >= 2:
            s_item = sources.get(r["source_id"], {})
            print(f"  Response ID : {r['id']}")
            print(f"  Model       : {r['model']}")
            print(f"  Task        : {s_item.get('task_type')}")
            print(f"  Text        : \"{r.get('response', '')[:200]}...\"")
            print("  Labels:")
            for l in r["labels"]:
                print(f"    * span [{l.get('start')}:{l.get('end')}]: \"{l.get('text')}\" -> {l.get('label_type')} (implicit_true={l.get('implicit_true')})")
            break


if __name__ == "__main__":
    inspect()

