"""
Inspect and compare microsoft/deberta-v3-small vs cross-encoder/nli-deberta-v3-xsmall.
Report exact parameter counts, tokenizer, architecture, layer counts, and input formats.
"""
import os
from pathlib import Path

# Ensure offline execution first
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

print("=" * 80)
print("COMPARING CANDIDATE MODEL ARCHITECTURES")
print("=" * 80)

# Check cache directory for available checkpoints
cache_hub = Path(os.path.expanduser("~/.cache/huggingface/hub"))
print(f"HuggingFace Hub Cache Directory: {cache_hub}")
if cache_hub.exists():
    for d in sorted(cache_hub.iterdir()):
        if d.is_dir() and "models--" in d.name:
            print(f"  - {d.name}")

# 1. Model 1: cross-encoder/nli-deberta-v3-xsmall
m1_name = "cross-encoder/nli-deberta-v3-xsmall"
print(f"\n[Model 1]: {m1_name}")
try:
    c1 = AutoConfig.from_pretrained(m1_name)
    tok1 = AutoTokenizer.from_pretrained(m1_name)
    mod1 = AutoModelForSequenceClassification.from_pretrained(m1_name)
    p1_total = sum(p.numel() for p in mod1.parameters())
    p1_train = sum(p.numel() for p in mod1.parameters() if p.requires_grad)
    print(f"  Architecture       : {c1.architectures}")
    print(f"  Model Type         : {c1.model_type}")
    print(f"  Hidden Size        : {c1.hidden_size}")
    print(f"  Num Hidden Layers  : {c1.num_hidden_layers}")
    print(f"  Num Attention Heads: {c1.num_attention_heads}")
    print(f"  Total Parameters   : {p1_total:,} ({p1_total/1e6:.2f}M)")
    print(f"  Classifier Head    : {c1.num_labels} labels: {c1.id2label}")
    print(f"  Tokenizer Class    : {tok1.__class__.__name__}")
    print(f"  Vocab Size         : {tok1.vocab_size}")
    print(f"  Pretrained Task    : Fine-tuned on MNLI / SNLI (NLI Cross-Encoder)")
except Exception as e:
    print(f"  Error loading {m1_name}: {e}")

# 2. Model 2: microsoft/deberta-v3-small
m2_name = "microsoft/deberta-v3-small"
print(f"\n[Model 2]: {m2_name}")
try:
    c2 = AutoConfig.from_pretrained(m2_name)
    tok2 = AutoTokenizer.from_pretrained(m2_name)
    mod2 = AutoModelForSequenceClassification.from_pretrained(m2_name)
    p2_total = sum(p.numel() for p in mod2.parameters())
    print(f"  Architecture       : {c2.architectures}")
    print(f"  Model Type         : {c2.model_type}")
    print(f"  Hidden Size        : {c2.hidden_size}")
    print(f"  Num Hidden Layers  : {c2.num_hidden_layers}")
    print(f"  Num Attention Heads: {c2.num_attention_heads}")
    print(f"  Total Parameters   : {p2_total:,} ({p2_total/1e6:.2f}M)")
    print(f"  Classifier Head    : {c2.num_labels} labels")
    print(f"  Tokenizer Class    : {tok2.__class__.__name__}")
    print(f"  Vocab Size         : {tok2.vocab_size}")
    print(f"  Pretrained Task    : Raw Pretrained Foundation Model (RTD MLM only)")
except Exception as e:
    print(f"  Status for {m2_name}: {e}")

