"""
VeriLLM Dense Passage Retriever Module.
Uses Sentence Transformers ('all-MiniLM-L6-v2') to index document passages
and retrieve the most semantically relevant evidence for atomic claims.
"""

import os
from typing import List, Optional, Union, Any
import numpy as np

# Ensure offline loading for cached embedding models
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from sentence_transformers import SentenceTransformer
from .models import DocumentChunk, RetrievedPassage, ClaimEvidenceMatch

# Singleton holder for the embedding model
_EMBEDDING_MODEL = None


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Lazily loads the SentenceTransformer model once on CPU."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        _EMBEDDING_MODEL = SentenceTransformer(model_name, device="cpu")
    return _EMBEDDING_MODEL


class PassageRetriever:
    """
    Indexes document chunks in dense vector space and retrieves top-k relevant passages for claims.
    
    IMPORTANT SCIENTIFIC DISTINCTION:
    The similarity score produced by this retriever represents semantic / topical relatedness.
    It does NOT prove factual entailment. A retrieved passage might share 90% semantic similarity
    with a claim while explicitly contradicting it. Entailment vs contradiction is evaluated
    by the verification stage in Phase 5.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        top_k: int = 3
    ):
        self.model_name = model_name
        self.top_k = top_k
        self.model = get_embedding_model(model_name)
        self.chunks: List[DocumentChunk] = []
        self.chunk_embeddings: Optional[np.ndarray] = None

    def index_chunks(self, chunks: List[DocumentChunk], batch_size: int = 32):
        """
        Encodes document chunks into normalized dense vectors and builds an in-memory index.

        Args:
            chunks: List of DocumentChunk objects to index.
            batch_size: Batch size for CPU embedding computation.
        """
        self.chunks = list(chunks)
        if not self.chunks:
            self.chunk_embeddings = None
            return

        texts = [c.text for c in self.chunks]
        # normalize_embeddings=True ensures L2 norm = 1, so dot product == cosine similarity
        self.chunk_embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

    def clear_index(self) -> None:
        """Release indexed chunks and dense vector embeddings from memory."""
        self.chunks = []
        self.chunk_embeddings = None

    def retrieve_for_claim(self, claim_text: str, top_k: Optional[int] = None) -> List[RetrievedPassage]:
        """
        Retrieves the top-k most semantically similar passages for a single claim.

        Args:
            claim_text: Factual proposition to match.
            top_k: Number of candidate passages to retrieve (defaults to self.top_k).

        Returns:
            List of RetrievedPassage objects ordered by descending cosine similarity.
        """
        k = top_k or self.top_k
        if not self.chunks or self.chunk_embeddings is None:
            return []

        # Encode claim text
        claim_emb = self.model.encode(
            claim_text,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        # Compute cosine similarity via dot product against normalized chunk vectors
        # Shape: (num_chunks,)
        similarities = np.dot(self.chunk_embeddings, claim_emb)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:k]

        results: List[RetrievedPassage] = []
        for rank, idx in enumerate(top_indices, start=1):
            score = float(similarities[idx])
            results.append(RetrievedPassage(
                chunk=self.chunks[idx],
                similarity_score=round(score, 4),
                rank=rank
            ))

        return results

    def retrieve_for_claims(
        self, 
        claims: List[Any], 
        top_k: Optional[int] = None
    ) -> List[ClaimEvidenceMatch]:
        """
        Batch retrieves relevant evidence for multiple claims efficiently using vectorized matrix math.

        Args:
            claims: List of ExtractedClaim objects or string propositions.
            top_k: Number of passages per claim.

        Returns:
            List of ClaimEvidenceMatch objects.
        """
        k = top_k or self.top_k
        if not claims or not self.chunks or self.chunk_embeddings is None:
            return []

        # Extract text strings from claims
        claim_texts = []
        claim_ids = []
        for idx, c in enumerate(claims, start=1):
            if isinstance(c, dict):
                c_id = c.get("claim_id", f"claim_{idx:03d}")
                c_text = c.get("text") or c.get("claim_text") or str(c)
                claim_ids.append(c_id)
                claim_texts.append(c_text)
            elif hasattr(c, "text"):
                claim_texts.append(c.text)
                claim_ids.append(getattr(c, "claim_id", f"claim_{idx:03d}"))
            elif hasattr(c, "claim_text"):
                claim_texts.append(c.claim_text)
                claim_ids.append(getattr(c, "claim_id", f"claim_{idx:03d}"))
            else:
                claim_texts.append(str(c))
                claim_ids.append(f"claim_{idx:03d}")

        # Batch encode all claims
        claim_embeddings = self.model.encode(
            claim_texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        # Matrix multiplication: (num_claims, num_chunks)
        sim_matrix = np.dot(claim_embeddings, self.chunk_embeddings.T)

        matches: List[ClaimEvidenceMatch] = []

        for row_idx, claim_id in enumerate(claim_ids):
            scores = sim_matrix[row_idx]
            top_indices = np.argsort(scores)[::-1][:k]

            passages: List[RetrievedPassage] = []
            for rank, chunk_idx in enumerate(top_indices, start=1):
                score = float(scores[chunk_idx])
                passages.append(RetrievedPassage(
                    chunk=self.chunks[chunk_idx],
                    similarity_score=round(score, 4),
                    rank=rank
                ))

            best = passages[0] if passages else None
            matches.append(ClaimEvidenceMatch(
                claim_id=claim_id,
                claim_text=claim_texts[row_idx],
                retrieved_passages=passages,
                best_passage=best,
                best_similarity_score=best.similarity_score if best else 0.0
            ))

        return matches
