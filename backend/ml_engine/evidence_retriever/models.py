"""
VeriLLM Evidence Retrieval Data Models.
Defines strongly-typed Pydantic schemas for document chunks, retrieved passages, and claim-evidence associations.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """
    Represents an atomic, semantically coherent passage chunk extracted from a document.
    Maintains full provenance (document name, page number, sequence index).
    """
    chunk_id: str = Field(
        ...,
        description="Unique identifier for the chunk (e.g. 'doc_apollo11_p01_c001')."
    )
    document_name: str = Field(
        ...,
        description="Source document filename or identifier."
    )
    page_number: Optional[int] = Field(
        default=None,
        description="1-indexed page number in the source document, where applicable."
    )
    chunk_index: int = Field(
        ...,
        description="Sequential index of the chunk within the document."
    )
    text: str = Field(
        ...,
        description="Cleaned text content of the chunk."
    )
    start_char: int = Field(
        default=0,
        description="Character start offset in the source document/page."
    )
    end_char: int = Field(
        default=0,
        description="Character end offset in the source document/page."
    )
    word_count: int = Field(
        default=0,
        description="Number of words in this chunk."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional chunk metadata (e.g. heading, section, overlap stats)."
    )


class RetrievedPassage(BaseModel):
    """
    Represents a chunk retrieved for a specific query, paired with its semantic relevance score.
    """
    chunk: DocumentChunk = Field(..., description="The underlying document passage.")
    similarity_score: float = Field(
        ...,
        description="Cosine similarity score between query embedding and chunk embedding [-1.0, 1.0]."
    )
    rank: int = Field(
        ...,
        description="1-indexed rank of this passage among retrieved candidates."
    )


class ClaimEvidenceMatch(BaseModel):
    """
    Associates an extracted claim with its top-k retrieved evidence passages.
    """
    claim_id: str = Field(..., description="Unique claim identifier.")
    claim_text: str = Field(..., description="The factual proposition being verified.")
    retrieved_passages: List[RetrievedPassage] = Field(
        default_factory=list,
        description="Top-k relevant passages ordered by descending similarity."
    )
    best_passage: Optional[RetrievedPassage] = Field(
        default=None,
        description="Top-1 highest-scoring retrieved passage."
    )
    best_similarity_score: float = Field(
        default=0.0,
        description="Cosine similarity score of the top-1 passage."
    )


class IngestedDocument(BaseModel):
    """
    Container for an ingested document, its extracted pages, and segmented chunks.
    """
    document_id: str = Field(..., description="Unique document ID (hash or sanitized filename).")
    document_name: str = Field(..., description="Original filename or document title.")
    document_type: str = Field(..., description="'PDF' or 'PLAIN_TEXT'.")
    total_pages: int = Field(default=1, description="Total number of pages processed.")
    total_chunks: int = Field(default=0, description="Total number of chunks produced.")
    chunks: List[DocumentChunk] = Field(default_factory=list, description="All ordered chunks.")
    is_scanned: bool = Field(
        default=False,
        description="True if document had zero extractable text (scanned / image-only)."
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata.")

