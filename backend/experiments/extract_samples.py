"""Extract representative sample examples for all 4 classes from ragtruth_val.jsonl."""
import json
from pathlib import Path

val_path = Path("backend/data/processed/ragtruth_val.jsonl")
found = {}

with open(val_path, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        lbl = item["primary_label"]
        if lbl not in found:
            # truncate context for display if too long
            ctx_display = item["context_text"]
            if len(ctx_display) > 300:
                ctx_display = ctx_display[:300] + "..."
            
            sample_entry = {
                "example_id": item["example_id"],
                "source_id": item["source_id"],
                "task_type": item["task_type"],
                "model_name": item.get("model", "unknown"),
                "primary_label": item["primary_label"],
                "sentence_text": item["sentence_text"],
                "context_snippet": ctx_display,
                "overlapping_spans_count": len(item.get("overlapping_spans", [])),
                "overlapping_spans": item.get("overlapping_spans", []),
                "has_conflict": item.get("has_conflict", False),
                "has_baseless": item.get("has_baseless", False),
                "has_implicit_true": item.get("has_implicit_true", False),
                "has_due_to_null": item.get("has_due_to_null", False)
            }
            found[lbl] = sample_entry
        if len(found) == 4:
            break

out_path = Path("backend/experiments/results/sample_4classes.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(found, f, indent=2)

print(f"Extracted {len(found)} samples to {out_path}.")
for k, v in found.items():
    print(f"\n--- Class: {k} ---")
    print(f"ID: {v['example_id']} | Task: {v['task_type']} | Model: {v['model_name']}")
    print(f"Sentence: \"{v['sentence_text']}\"")
    print(f"Spans: {v['overlapping_spans']}")
