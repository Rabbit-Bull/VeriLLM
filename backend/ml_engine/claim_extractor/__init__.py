"""
VeriLLM Claim Extraction Package.
Exports the main ClaimExtractor, models, and helper functions.
"""

from .models import ExtractedClaim, ExtractionResult
from .extractor import ClaimExtractor
from .sentence_splitter import split_sentences_with_spans, SentenceSpan
from .proposition_decomposer import decompose_sentence
from .claim_filter import classify_proposition

__all__ = [
    "ClaimExtractor",
    "ExtractedClaim",
    "ExtractionResult",
    "split_sentences_with_spans",
    "SentenceSpan",
    "decompose_sentence",
    "classify_proposition"
]

