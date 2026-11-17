# VeriLLM — Final Test Evaluation Report

## 1. Model Information

- Model: cross-encoder/nli-deberta-v3-xsmall
- Checkpoint: Original Epoch 1
- Parameters: 70,831,107
- Dataset: RAGTruth
- Evaluation level: Sentence-level
- Classes: Contradicted, Supported Proxy, Unsupported

## 2. Overall Performance

| Metric | Result |
|---|---:|
| Accuracy | 92.86% |
| Macro-F1 | 69.99% |
| Weighted F1 | 92.76% |
| Validation Macro-F1 | 70.99% |
| Test examples | 17,906 |
| Excluded MIXED examples | 37 |
| Evaluation time | 3.27 minutes |

## 3. Class-wise Performance

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| CONTRADICTED | 66.74% | 57.32% | 61.67% | 553 |\n| SUPPORTED_PROXY | 95.92% | 96.49% | 96.20% | 16,388 |\n| UNSUPPORTED | 52.59% | 51.61% | 52.09% | 965 |\n
## 4. Test Dataset Distribution

| Class | Examples |
|---|---:|
| SUPPORTED_PROXY | 16,388 |
| CONTRADICTED | 553 |
| UNSUPPORTED | 965 |
| MIXED_HALLUCINATION (excluded) | 37 |
| Total raw examples | 17,943 |
| Three-class evaluation examples | 17,906 |

## 5. Confusion Matrix

Rows represent actual labels; columns represent predicted labels.

| Actual / Predicted | CONTRADICTED | SUPPORTED_PROXY | UNSUPPORTED |
|---|---:|---:|---:|
| CONTRADICTED | 317 | 221 | 15 |\n| SUPPORTED_PROXY | 142 | 15812 | 434 |\n| UNSUPPORTED | 16 | 451 | 498 |\n
## 6. Fine-tuning History

| Checkpoint | Validation Macro-F1 |
|---|---:|
| Epoch 1 (original) | 70.99% |
| Epoch 2 | 67.97% |
| Epoch 3 | 70.79% |

The original Epoch 1 checkpoint remained the best validation checkpoint.
The later fine-tuning epochs did not surpass its validation Macro-F1.

## 7. Interpretation and Limitations

- SUPPORTED_PROXY identifies examples without an annotated hallucination span. It is not a guarantee of factual truth.
- CONTRADICTED and UNSUPPORTED are distinct annotation categories.
- The 37 MIXED_HALLUCINATION examples were excluded rather than forced into one of the three classes.
- The test distribution is imbalanced, so Macro-F1 is more informative than accuracy alone.
- These results apply to the RAGTruth sentence-level task and should not be interpreted as universal hallucination-detection accuracy.

## 8. Research Conclusion

The preserved Epoch 1 checkpoint achieved a test Macro-F1 of
{macro_f1*100:.2f}% and accuracy of {accuracy*100:.2f}% on
17,906 three-class sentence examples.

The results demonstrate strong performance on SUPPORTED_PROXY examples,
while CONTRADICTED and UNSUPPORTED classification remain areas for improvement.

This evaluation is a benchmark for the current VeriLLM model component,
not proof of general factual correctness across arbitrary LLM outputs.
