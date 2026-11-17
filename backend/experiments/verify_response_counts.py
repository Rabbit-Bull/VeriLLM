"""
Verify response counts in raw response.jsonl and in processed jsonl files.
"""
import json
from collections import Counter
from pathlib import Path

raw_resp = Path("backend/data/ragtruth/response.jsonl")

# 1. Raw response counts
raw_counts = Counter()
raw_good_counts = Counter()
raw_excluded = Counter()

with open(raw_resp, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        sp = item.get("split", "unknown")
        q = item.get("quality", "unknown")
        raw_counts[sp] += 1
        if q == "good":
            raw_good_counts[sp] += 1
        else:
            raw_excluded[sp] += 1

print("1. Raw response.jsonl by split:")
for sp in ["train", "test"]:
    print(f"  {sp}: Total = {raw_counts[sp]}, Good = {raw_good_counts[sp]}, Excluded = {raw_excluded[sp]}")

# 2. Processed response IDs per file
for sp in ["train", "val", "test"]:
    proc_file = Path(f"backend/data/processed/ragtruth_{sp}.jsonl")
    resp_ids = set()
    source_ids = set()
    sentences = 0
    with open(proc_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            resp_ids.add(str(item["response_id"]))
            source_ids.add(str(item["source_id"]))
            sentences += 1
    print(f"\n2. Processed ragtruth_{sp}.jsonl:")
    print(f"  Sources       : {len(source_ids)}")
    print(f"  Responses     : {len(resp_ids)}")
    print(f"  Sentences     : {sentences}")

print("\n3. Sum of processed responses across train + val + test:")
tot_proc_resp = 0
for sp in ["train", "val", "test"]:
    proc_file = Path(f"backend/data/processed/ragtruth_{sp}.jsonl")
    s = set(json.loads(line)["response_id"] for line in open(proc_file, "r", encoding="utf-8"))
    tot_proc_resp += len(s)
print(f"  Total processed responses: {tot_proc_resp}")
print(f"  Total raw good responses : {sum(raw_good_counts.values())}")
print(f"  Match exactly            : {tot_proc_resp == sum(raw_good_counts.values())}")

