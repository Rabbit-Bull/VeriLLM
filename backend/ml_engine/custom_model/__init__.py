"""
VeriLLM Custom Model Package.
Implements fine-tuning and evaluation pipelines for hallucination detection on RAGTruth,
featuring dataset preprocessing, baseline benchmarking, and model training.
"""

from pathlib import Path

CUSTOM_MODEL_DIR = Path(__file__).resolve().parent

from ml_engine.custom_model.dataset_preprocessor import (
    format_source_context,
    determine_sentence_label,
    RAGTruthPreprocessor
)
from ml_engine.custom_model.train import (
    RAGTruthDataset,
    compute_class_weights
)

__all__ = [
    "CUSTOM_MODEL_DIR",
    "format_source_context",
    "determine_sentence_label",
    "RAGTruthPreprocessor",
    "RAGTruthDataset",
    "compute_class_weights",
]

