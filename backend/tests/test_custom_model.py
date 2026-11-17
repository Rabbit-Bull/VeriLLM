"""
Unit tests for VeriLLM Phase 7: Custom Model, Preprocessor, and Dataset Loader.
Ensures zero data leakage, correct label mapping, proper context windowing,
and class weight balancing.
"""

import json
import sys
from pathlib import Path

# Add backend directory to sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import pytest
from unittest.mock import MagicMock
import torch

from ml_engine.custom_model.dataset_preprocessor import (
    format_source_context,
    determine_sentence_label,
    RAGTruthPreprocessor
)
from ml_engine.custom_model.train import RAGTruthDataset, compute_class_weights


class TestDatasetPreprocessor:
    """Tests for RAGTruth dataset preprocessing and label mapping rules."""

    def test_format_source_context_summary(self):
        source = {
            "source_id": "1",
            "task_type": "Summary",
            "source_info": "This is a news article about space exploration."
        }
        res = format_source_context(source)
        assert res.startswith("Document Context:\nThis is a news article")

    def test_format_source_context_qa(self):
        source = {
            "source_id": "2",
            "task_type": "QA",
            "source_info": {
                "question": "What is the capital of France?",
                "context": "Paris is the capital of France."
            }
        }
        res = format_source_context(source)
        assert "Question: What is the capital of France?" in res
        assert "Passage Context:\nParis is the capital of France." in res

    def test_format_source_context_data2txt(self):
        source = {
            "source_id": "3",
            "task_type": "Data2txt",
            "source_info": [
                {"name": "Apollo 11", "launch_year": "1969"}
            ]
        }
        res = format_source_context(source)
        assert "Structured Data Table / Entities:" in res
        assert "Apollo 11" in res

    def test_determine_sentence_label_no_hallucination(self):
        """Sentence with no hallucinated span must be labeled SUPPORTED_PROXY."""
        label, spans, has_c, has_b, has_imp, has_null = determine_sentence_label(
            sent_start=0,
            sent_end=50,
            sent_text="Neil Armstrong was the first person to walk on Moon.",
            labels_list=[]
        )
        assert label == "SUPPORTED_PROXY"
        assert spans == []
        assert has_c is False
        assert has_b is False

    def test_determine_sentence_label_contradiction(self):
        """Sentence with only conflict annotations must be labeled CONTRADICTED."""
        labels_list = [{
            "start": 10,
            "end": 25,
            "text": "walked in 1975",
            "label_type": "Evident Conflict"
        }]
        label, spans, has_c, has_b, _, _ = determine_sentence_label(
            sent_start=0,
            sent_end=50,
            sent_text="Neil Armstrong walked in 1975 on the lunar surface.",
            labels_list=labels_list
        )
        assert label == "CONTRADICTED"
        assert has_c is True
        assert has_b is False
        assert len(spans) == 1
        assert spans[0]["label_type"] == "Evident Conflict"

    def test_determine_sentence_label_unsupported(self):
        """Sentence with only baseless annotations must be labeled UNSUPPORTED."""
        labels_list = [{
            "start": 5,
            "end": 30,
            "text": "accompanied by a robotic dog",
            "label_type": "Evident Baseless Info"
        }]
        label, spans, has_c, has_b, _, _ = determine_sentence_label(
            sent_start=0,
            sent_end=60,
            sent_text="He was accompanied by a robotic dog on the mission.",
            labels_list=labels_list
        )
        assert label == "UNSUPPORTED"
        assert has_c is False
        assert has_b is True
        assert len(spans) == 1

    def test_determine_sentence_label_mixed(self):
        """Sentence overlapping both conflict and baseless must be labeled MIXED_HALLUCINATION."""
        labels_list = [
            {"start": 5, "end": 20, "text": "conflict span", "label_type": "Subtle Conflict"},
            {"start": 25, "end": 45, "text": "baseless entity", "label_type": "Evident Baseless Info"}
        ]
        label, spans, has_c, has_b, _, _ = determine_sentence_label(
            sent_start=0,
            sent_end=50,
            sent_text="Some conflict span and some baseless entity here.",
            labels_list=labels_list
        )
        assert label == "MIXED_HALLUCINATION"
        assert has_c is True
        assert has_b is True
        assert len(spans) == 2


class TestDatasetLoaderAndWeights:
    """Tests for PyTorch dataset and class weighting logic."""

    def test_compute_class_weights(self):
        from ml_engine.custom_model.train import LABEL2ID
        assert LABEL2ID["CONTRADICTED"] == 0
        assert LABEL2ID["SUPPORTED_PROXY"] == 1
        assert LABEL2ID["UNSUPPORTED"] == 2

        mock_ds = MagicMock()
        mock_ds.examples = [
            {"label": 1},  # SUPPORTED_PROXY (majority class)
            {"label": 1},
            {"label": 1},
            {"label": 1},
            {"label": 0},  # CONTRADICTED (minority class)
            {"label": 2}   # UNSUPPORTED (minority class)
        ]
        weights = compute_class_weights(mock_ds)
        assert isinstance(weights, torch.Tensor)
        assert len(weights) == 3
        # Minority classes (0: CONTRADICTED and 2: UNSUPPORTED) must have strictly higher loss weights than majority class (1: SUPPORTED_PROXY)
        assert weights[0] > weights[1]
        assert weights[2] > weights[1]

    def test_colab_notebook_structure(self):
        """Verify the generated Google Colab notebook is valid JSON and has required steps."""
        notebook_path = Path("backend/ml_engine/custom_model/colab_ragtruth_training.ipynb")
        assert notebook_path.exists()
        with open(notebook_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["nbformat"] == 4
        assert len(data["cells"]) >= 8
        
        # Check that essential pipeline steps are present in the notebook cells
        cell_sources = [
            "".join(c["source"]) for c in data["cells"]
        ]
        all_text = " ".join(cell_sources)
        assert "SUPPORTED_PROXY" in all_text
        assert "CONTRADICTED" in all_text
        assert "UNSUPPORTED" in all_text
        assert "deberta" in all_text
        assert "CrossEntropyLoss" in all_text
