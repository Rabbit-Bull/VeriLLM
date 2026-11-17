"""
Investigate the MIXED sentence-count discrepancy:
Earlier preprocessing report: 560 MIXED sentences.
Audit report: 449 MIXED sentences in train.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
PROC_DIR = ROOT / "backend" / "data" / "processed"

splits = [
    ("Train", PROC_DIR / "ragtruth_train.jsonl"),
    ("Val", PROC_DIR / "ragtruth_val.jsonl"),
    ("Test", PROC_DIR / "ragtruth_test.jsonl")
]

total_mixed = 0
for name, p in splits:
    cnt = 0
    total_sents = 0
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            total_sents += 1
            if json.loads(line).get("primary_label") == "MIXED_HALLUCINATION":
                cnt += 1
    total_mixed += cnt
    print(f"{name:5s} partition: {cnt:5d} MIXED sentences out of {total_sents:6d} sentences ({cnt/total_sents*100:.2f}%)")

print("-" * 50)
print(f"Total across all 3 splits: {total_mixed} MIXED sentences")
print(f"Train + Val total: {449 + 74} MIXED sentences")

