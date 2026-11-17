"""
Investigate the test split discrepancy:
Official test reported: 2,700 responses.
Processed test reported: 2,734 responses.
Examine source_ids, response_ids, split field, and quality filtering.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

DATA_DIR = Path("backend/data/ragtruth")
PROCESSED_DIR = Path("backend/data/processed")

raw_response_file = DATA_DIR / "response.jsonl"
raw_source_file = DATA_DIR / "source_info.jsonl"
processed_test_file = PROCESSED_DIR / "ragtruth_test.jsonl"

print("=" * 80)
print("1. RAW RESPONSE.JSONL SPLIT FIELD ANALYSIS")
print("=" * 80)

raw_by_split = Counter()
raw_quality_by_split = defaultdict(Counter)
raw_test_response_ids = set()
raw_test_source_ids = set()
raw_train_source_ids = set()

all_response_ids = set()
duplicate_response_ids = []

with open(raw_response_file, "r", encoding="utf-8") as f:
    for line_idx, line in enumerate(f):
        item = json.loads(line)
        r_id = str(item.get("id", item.get("response_id", line_idx)))
        if r_id in all_response_ids:
            duplicate_response_ids.append(r_id)
        all_response_ids.add(r_id)

        sp = item.get("split", "unknown")
        q = item.get("quality", "unknown")
        sid = str(item.get("source_id"))
        
        raw_by_split[sp] += 1
        raw_quality_by_split[sp][q] += 1
        
        if sp == "test":
            raw_test_response_ids.add(r_id)
            raw_test_source_ids.add(sid)
        elif sp == "train":
            raw_train_source_ids.add(sid)

print(f"Total raw lines in response.jsonl: {len(all_response_ids)}")
print(f"Duplicate response IDs: {len(duplicate_response_ids)}")
print(f"Raw counts by 'split' field:")
for sp, cnt in raw_by_split.items():
    print(f"  split='{sp}': {cnt} responses")
    for q, q_cnt in raw_quality_by_split[sp].items():
        print(f"    - quality='{q}': {q_cnt}")

print(f"\nUnique sources with split='test': {len(raw_test_source_ids)}")
print(f"Unique sources with split='train': {len(raw_train_source_ids)}")
overlap = raw_test_source_ids.intersection(raw_train_source_ids)
print(f"Source overlap between train and test: {len(overlap)}")

print("\n" + "=" * 80)
print("2. PROCESSED DATASET TEST SPLIT ANALYSIS")
print("=" * 80)

proc_test_example_count = 0
proc_test_response_ids = set()
proc_test_source_ids = set()

with open(processed_test_file, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        proc_test_example_count += 1
        proc_test_response_ids.add(str(item["response_id"]))
        proc_test_source_ids.add(str(item["source_id"]))

print(f"Total sentences in ragtruth_test.jsonl: {proc_test_example_count}")
print(f"Unique response_ids in ragtruth_test.jsonl: {len(proc_test_response_ids)}")
print(f"Unique source_ids in ragtruth_test.jsonl: {len(proc_test_source_ids)}")

# Check difference between raw test response_ids and processed test response_ids
extra_in_proc = proc_test_response_ids - raw_test_response_ids
missing_from_proc = raw_test_response_ids - proc_test_response_ids

print(f"\nResponse IDs in processed test but NOT in raw split='test': {len(extra_in_proc)}")
print(f"Response IDs in raw split='test' but NOT in processed test: {len(missing_from_proc)}")

if extra_in_proc:
    print(f"Sample extra IDs in processed test: {list(extra_in_proc)[:5]}")
    # Where did they come from?
    extra_sources = set()
    with open(raw_response_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            r_id = str(item.get("id", item.get("response_id")))
            if r_id in extra_in_proc:
                extra_sources.add(str(item.get("source_id")))
                print(f"  Extra ID {r_id}: split in raw={item.get('split')}, source_id={item.get('source_id')}, quality={item.get('quality')}")
                if len(extra_sources) >= 5:
                    break

print("\n" + "=" * 80)
print("3. EXAMINING DATASET_PREPROCESSOR.PY LOGIC")
print("=" * 80)
# Look at how dataset_preprocessor decided target_split

print("\n" + "=" * 80)
print("4. SEARCHING FOR '2734' IN CODEBASE AND REPORTS")
print("=" * 80)
for p in Path(".").rglob("*"):
    if p.is_file() and ".venv" not in p.parts and ".git" not in p.parts:
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                for line_no, line in enumerate(f, 1):
                    if "2734" in line or "2,734" in line:
                        print(f"{p}:{line_no} -> {line.strip()}")
        except Exception:
            pass
