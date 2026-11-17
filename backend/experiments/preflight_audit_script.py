"""
VeriLLM Phase 7: Pre-Flight Audit Script.
Executes comprehensive verification across datasets, model architectures,
label mappings, class weights, context tokenization, and notebook consistency.
"""
import os
import sys
import json
from pathlib import Path
from collections import Counter

# Set offline for local inspection
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND = ROOT / "backend"
DATA_DIR = BACKEND / "data" / "ragtruth"
PROC_DIR = BACKEND / "data" / "processed"

print("=" * 80)
print("AUDIT 1: DATASET ISOLATION & LABEL METRICS")
print("=" * 80)

raw_resp_file = DATA_DIR / "response.jsonl"
train_file = PROC_DIR / "ragtruth_train.jsonl"
val_file = PROC_DIR / "ragtruth_val.jsonl"
test_file = PROC_DIR / "ragtruth_test.jsonl"

raw_train_sources = set()
raw_test_sources = set()
raw_test_responses = set()
raw_test_good = set()

with open(raw_resp_file, 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        sid = str(item['source_id'])
        rid = str(item['id'])
        sp = item.get('split')
        q = item.get('quality')
        if sp == 'test':
            raw_test_sources.add(sid)
            raw_test_responses.add(rid)
            if q == 'good':
                raw_test_good.add(rid)
        elif sp == 'train':
            raw_train_sources.add(sid)

def load_proc(fpath):
    sources = set()
    responses = set()
    labels = Counter()
    total_sents = 0
    with open(fpath, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            sources.add(str(item['source_id']))
            responses.add(str(item['response_id']))
            labels[item['primary_label']] += 1
            total_sents += 1
    return sources, responses, labels, total_sents

train_src, train_resp, train_lbl, train_sents = load_proc(train_file)
val_src, val_resp, val_lbl, val_sents = load_proc(val_file)
test_src, test_resp, test_lbl, test_sents = load_proc(test_file)

print(f"Sources: Train={len(train_src)}, Val={len(val_src)}, Test={len(test_src)}")
print(f"Disjointness:")
print(f"  Train & Val  : {len(train_src & val_src) == 0} (overlap: {len(train_src & val_src)})")
print(f"  Train & Test : {len(train_src & test_src) == 0} (overlap: {len(train_src & test_src)})")
print(f"  Val & Test   : {len(val_src & test_src) == 0} (overlap: {len(val_src & test_src)})")
print(f"All 450 official test sources in test_file: {test_src == raw_test_sources}")
print(f"All train+val sources in raw train sources : {(train_src | val_src) == raw_train_sources}")

print(f"\nResponses in processed test_file: {len(test_resp)}")
print(f"Matches raw test quality-good exactly (2,675): {test_resp == raw_test_good}")
print(f"Test response overlap with train/val: {len(test_resp & (train_resp | val_resp))}")

print(f"\nLabel Counts:")
print(f"  Train: {dict(train_lbl)}")
print(f"  Val  : {dict(val_lbl)}")
print(f"  Test : {dict(test_lbl)}")

# Training class weights calculation
# 3 classes: SUPPORTED_PROXY, CONTRADICTED, UNSUPPORTED (MIXED excluded)
train_3c = {k: train_lbl[k] for k in ['SUPPORTED_PROXY', 'CONTRADICTED', 'UNSUPPORTED']}
n_train_3c = sum(train_3c.values())
mixed_train = train_lbl['MIXED_HALLUCINATION']
print(f"\nActual Sentences in 3-Class Training: {n_train_3c} (Mixed Excluded: {mixed_train})")

K = 3
raw_weights = {k: n_train_3c / (K * v) for k, v in train_3c.items()}
sum_raw = sum(raw_weights.values())
norm_weights = {k: (v / sum_raw) * K for k, v in raw_weights.items()}

print("Class Weights Calculated Strictly From Train Partition:")
for k in ['SUPPORTED_PROXY', 'CONTRADICTED', 'UNSUPPORTED']:
    print(f"  {k:<16}: count={train_3c[k]:>6} ({train_3c[k]/n_train_3c*100:.2f}%) | raw_w={raw_weights[k]:.4f} | norm_w={norm_weights[k]:.4f}")

print("\n" + "=" * 80)
print("AUDIT 2: MODEL CHECKPOINT, CONFIG & LABEL MAPPING")
print("=" * 80)

from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = "cross-encoder/nli-deberta-v3-xsmall"
config = AutoConfig.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)

print(f"Model ID          : {MODEL_ID}")
print(f"Architecture      : {config.architectures}")
print(f"Num Labels in Ckpt: {config.num_labels}")
print(f"Checkpoint id2label: {config.id2label}")
print(f"Checkpoint label2id: {config.label2id}")
print(f"Tokenizer Class   : {tokenizer.__class__.__name__}")
print(f"Vocab Size        : {tokenizer.vocab_size}")

# Check classifier weights in pretrained model
head_weights = None
head_bias = None
for name, param in model.named_parameters():
    if "classifier" in name or "pooler" in name:
        print(f"  Head Parameter  : {name:<35} shape={list(param.shape)}")
        if "classifier.weight" in name:
            head_weights = param.clone()
        if "classifier.bias" in name:
            head_bias = param.clone()

print(f"\nPretrained Classifier Weights loaded: {head_weights is not None} (weight norm={head_weights.norm().item():.4f})")

print("\n" + "=" * 80)
print("AUDIT 3: TOKENIZER PAIRED INPUT & CANDIDATE SENTENCE TRUNCATION")
print("=" * 80)

sample_ctx = "The Apollo 11 mission was launched on July 16, 1969, carrying Neil Armstrong, Buzz Aldrin, and Michael Collins."
sample_claim = "Neil Armstrong walked on the Moon on July 20, 1969."

tokens_pair = tokenizer(sample_ctx, sample_claim, add_special_tokens=True)
input_ids = tokens_pair["input_ids"]
decoded = tokenizer.convert_ids_to_tokens(input_ids)
print(f"Decoded token sequence (first 15 tokens): {decoded[:15]}")
print(f"Decoded token sequence (last 10 tokens) : {decoded[-10:]}")
print(f"Special Tokens: {tokenizer.special_tokens_map}")
print(f"CLS token ID: {tokenizer.cls_token_id} ('{tokenizer.cls_token}')")
print(f"SEP token ID: {tokenizer.sep_token_id} ('{tokenizer.sep_token}')")

# Verify token-type IDs or separators
sep_indices = [i for i, tid in enumerate(input_ids) if tid == tokenizer.sep_token_id]
print(f"Separator positions: {sep_indices}")

print("\n" + "=" * 80)
print("AUDIT 4: COLAB NOTEBOOK & TRAIN.PY IMPLEMENTATION CHECK")
print("=" * 80)

nb_file = BACKEND / "ml_engine" / "custom_model" / "colab_ragtruth_training.ipynb"
with open(nb_file, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Colab Notebook Path: {nb_file}")
print(f"Total cells        : {len(nb['cells'])}")
has_gpu_check = any("nvidia-smi" in "".join(c["source"]) for c in nb["cells"])
has_pip_install = any("pip install" in "".join(c["source"]) for c in nb["cells"])
has_drive_mount = any("drive.mount" in "".join(c["source"]) for c in nb["cells"])
has_save_model = any("save_pretrained" in "".join(c["source"]) for c in nb["cells"])
has_zip_archive = any("zip -r" in "".join(c["source"]) for c in nb["cells"])
has_offline_flag = any("HF_HUB_OFFLINE" in "".join(c["source"]) for c in nb["cells"])

print(f"  - Checks GPU (!nvidia-smi)          : {has_gpu_check}")
print(f"  - Installs requirements             : {has_pip_install}")
print(f"  - Google Drive mounting option      : {has_drive_mount}")
print(f"  - Saves best model                  : {has_save_model}")
print(f"  - Archives to zip for download      : {has_zip_archive}")
print(f"  - Has unwanted HF_HUB_OFFLINE in nb : {has_offline_flag} (Must be False for Colab)")

