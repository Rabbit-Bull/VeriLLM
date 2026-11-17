"""
VeriLLM Evidence and Document-Processing Package.
Exports PDF parser, text cleaner, sliding-window chunker, and dense passage retriever.
"""

from .models import DocumentChunk, RetrievedPassage, ClaimEvidenceMatch, IngestedDocument
from .text_cleaner import clean_text
from .pdf_parser import parse_pdf, ExtractedPage, ParsedPDFDocument
from .chunker import SlidingWindowChunker
from .passage_retriever import PassageRetriever, get_embedding_model

__all__ = [
    "DocumentChunk",
    "RetrievedPassage",
    "ClaimEvidenceMatch",
    "IngestedDocument",
    "clean_text",
    "parse_pdf",
    "ExtractedPage",
    "ParsedPDFDocument",
    "SlidingWindowChunker",
    "PassageRetriever",
    "get_embedding_model"
]

