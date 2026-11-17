"""
Test weight alignment between pretrained cross-encoder/nli-deberta-v3-xsmall
and VeriLLM labels.
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from transformers import AutoModelForSequenceClassification, AutoConfig

MODEL_ID = 'cross-encoder/nli-deberta-v3-xsmall'
m_orig = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
w_orig = m_orig.classifier.weight.clone()
b_orig = m_orig.classifier.bias.clone()

# Pretrained mappings:
# {0: 'contradiction', 1: 'entailment', 2: 'neutral'}
# VeriLLM semantic mapping:
# 0 -> CONTRADICTED  (contradiction)
# 1 -> SUPPORTED_PROXY (entailment)
# 2 -> UNSUPPORTED   (neutral)

id2label_aligned = {0: 'CONTRADICTED', 1: 'SUPPORTED_PROXY', 2: 'UNSUPPORTED'}
label2id_aligned = {'CONTRADICTED': 0, 'SUPPORTED_PROXY': 1, 'UNSUPPORTED': 2}

config = AutoConfig.from_pretrained(MODEL_ID, id2label=id2label_aligned, label2id=label2id_aligned)
m_aligned = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, config=config)

w_aligned = m_aligned.classifier.weight
b_aligned = m_aligned.classifier.bias

print("Weights match exactly:", (w_orig == w_aligned).all().item())
print("Bias match exactly   :", (b_orig == b_aligned).all().item())
print("Index 0 in orig:", m_orig.config.id2label[0], "--> Aligned:", m_aligned.config.id2label[0])
print("Index 1 in orig:", m_orig.config.id2label[1], "--> Aligned:", m_aligned.config.id2label[1])
print("Index 2 in orig:", m_orig.config.id2label[2], "--> Aligned:", m_aligned.config.id2label[2])

