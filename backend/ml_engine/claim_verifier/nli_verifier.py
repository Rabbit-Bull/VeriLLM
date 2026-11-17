"""
VeriLLM NLI Cross-Encoder Verifier.
Uses cross-encoder/nli-deberta-v3-xsmall (70.8M parameters, 384 dimensions)
to produce calibrated probabilities across standard 3-way NLI labels:
0: contradiction, 1: entailment, 2: neutral.
"""

import os
from typing import List, Tuple, Optional, Any
import numpy as np

# Force offline mode if already cached locally
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from sentence_transformers import CrossEncoder


class NLIVerifier:
    """
    Cross-Encoder NLI model for fine-grained factual entailment, contradiction, and neutrality detection.
    """
    _instance: Optional["NLIVerifier"] = None
    _model: Optional[Any] = None

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-xsmall"):
        self.model_name = model_name
        self._ensure_loaded()

    def _ensure_loaded(self):
        """Lazy-loads the CrossEncoder model if not already initialized."""
        if NLIVerifier._model is None:
            NLIVerifier._model = CrossEncoder(
                self.model_name,
                num_labels=3
            )
        self.model = NLIVerifier._model

    def predict_pairs(
        self, 
        premise_hypothesis_pairs: List[Tuple[str, str]]
    ) -> List[Tuple[float, float, float]]:
        """
        Evaluates a batch of (premise, hypothesis) pairs.

        Args:
            premise_hypothesis_pairs: List of (evidence_text, claim_text) tuples.

        Returns:
            List of (entailment_prob, contradiction_prob, neutral_prob) tuples in range [0.0, 1.0].
        """
        if not premise_hypothesis_pairs:
            return []

        # Predict raw logits
        logits = self.model.predict(
            premise_hypothesis_pairs,
            batch_size=16,
            show_progress_bar=False
        )

        # Ensure 2D array
        if len(premise_hypothesis_pairs) == 1 and len(logits.shape) == 1:
            logits = np.expand_dims(logits, axis=0)

        # Numerically stable softmax:
        # Index 0: contradiction, Index 1: entailment, Index 2: neutral
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted_logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        results: List[Tuple[float, float, float]] = []
        for row in probs:
            p_contra = float(row[0])
            p_entail = float(row[1])
            p_neutral = float(row[2])
            results.append((p_entail, p_contra, p_neutral))

        return results

