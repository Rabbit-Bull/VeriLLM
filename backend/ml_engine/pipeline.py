"""
VeriLLM Unified End-to-End Pipeline Module.
Orchestrates document parsing, sliding-window chunking, atomic claim extraction,
dense passage retrieval, calibrated DeBERTa-v3 verification, and response-level
hallucination quantification into a single production entrypoint.
"""

from __future__ import annotations

import io
import logging
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field

from ml_engine.claim_extractor.extractor import ClaimExtractor
from ml_engine.claim_extractor.models import ExtractedClaim, ExtractionResult
from ml_engine.claim_verifier.calibrated_verifier import (
    CalibratedDebertaVerifier,
    RegimeName,
)
from ml_engine.claim_verifier.models import (
    VerificationCategory,
    VerificationReport,
    VerifiedClaim,
)
from ml_engine.evidence_retriever.chunker import SlidingWindowChunker
from ml_engine.evidence_retriever.models import (
    ClaimEvidenceMatch,
    DocumentChunk,
    RetrievedPassage,
)
from ml_engine.evidence_retriever.passage_retriever import PassageRetriever
from ml_engine.evidence_retriever.pdf_parser import ParsedPDFDocument, parse_pdf
from ml_engine.evidence_retriever.text_cleaner import clean_text
from ml_engine.scorer.models import ResponseScore
from ml_engine.scorer.response_scorer import ResponseScorer

logger = logging.getLogger("verillm.pipeline")

# Supported plain-text extensions for disk files
SUPPORTED_TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".json",
    ".csv",
    ".log",
    ".tsv",
    ".text",
    ".rst",
}

# DeBERTa token limit constants for Direct mode detection
DEBERTA_MAX_INPUT_TOKENS = 512
DIRECT_MODE_SAFE_WORD_LIMIT = 350


class PipelineResult(BaseModel):
    """Unified result container returned by VeriLLMPipeline.

    Provides structured claim-level verification results, retrieval provenance,
    decision explanations, response-level hallucination metrics, and execution metadata.
    """

    document_name: str = Field(..., description="Name of the reference document.")
    regime: str = Field(
        ..., description="Verification regime executed ('direct' or 'retrieval')."
    )
    prompt: Optional[str] = Field(
        default=None, description="Optional user prompt or query."
    )
    response: str = Field(..., description="Raw evaluated LLM response text.")
    claims: List[VerifiedClaim] = Field(
        default_factory=list, description="Claim-level verification results."
    )
    verification_report: VerificationReport = Field(
        ..., description="Detailed verification report from CalibratedDebertaVerifier."
    )
    score: ResponseScore = Field(
        ..., description="Response-level factuality and hallucination quantification."
    )
    execution_time_seconds: float = Field(
        default=0.0,
        description="Total end-to-end pipeline execution time in seconds.",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Operational warnings (e.g. document truncation, low evidence, empty text).",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Pipeline processing metadata (document stats, timings, calibrated probabilities).",
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the pipeline result to a standard dictionary."""
        return self.model_dump()


class VeriLLMPipeline:
    """Unified factual verification pipeline for VeriLLM.

    Coordinates document parsing, sliding-window chunking, atomic claim extraction,
    dense passage retrieval (in retrieval mode), calibrated DeBERTa-v3 NLI verification,
    and response-level hallucination scoring into a single production interface.
    """

    def __init__(
        self,
        bundle_path: Optional[Union[str, Path]] = None,
        default_regime: str = "retrieval",
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[Any] = None,
        batch_size: int = 16,
        calibrated_verifier: Optional[CalibratedDebertaVerifier] = None,
        claim_extractor: Optional[ClaimExtractor] = None,
        passage_retriever: Optional[PassageRetriever] = None,
        response_scorer: Optional[ResponseScorer] = None,
        chunk_size_words: int = 150,
        overlap_words: int = 35,
        retrieval_top_k: int = 3,
    ) -> None:
        """Initialize the unified VeriLLM verification pipeline.

        Args:
            bundle_path: Path to frozen calibration bundle (10_stage1_frozen_bundle.json).
            default_regime: Default verification regime ('direct' or 'retrieval').
            model_path: Path to fine-tuned DeBERTa model checkpoint directory.
            device: Execution device ('cpu' or torch.device).
            batch_size: Batch size for neural inference.
            calibrated_verifier: Pre-instantiated CalibratedDebertaVerifier instance (allows mock injection).
            claim_extractor: Pre-instantiated ClaimExtractor instance.
            passage_retriever: Pre-instantiated PassageRetriever instance.
            response_scorer: Pre-instantiated ResponseScorer instance.
            chunk_size_words: Target word length for passage chunks.
            overlap_words: Number of overlapping words between consecutive chunks.
            retrieval_top_k: Number of candidate passages retrieved per claim in retrieval mode.
        """
        regime_norm = default_regime.lower().strip()
        if regime_norm not in ("direct", "retrieval"):
            raise ValueError(
                f"Invalid default_regime: '{default_regime}'. Expected 'direct' or 'retrieval'."
            )
        self.default_regime = regime_norm
        self.retrieval_top_k = retrieval_top_k

        # Chunker
        self.chunker = SlidingWindowChunker(
            chunk_size_words=chunk_size_words,
            overlap_words=overlap_words,
        )

        # Claim Extractor
        self.claim_extractor = claim_extractor or ClaimExtractor(
            resolve_coreference=True,
            decompose_compound=True,
            include_non_factual=True,
        )

        # Passage Retriever
        self.passage_retriever = passage_retriever or PassageRetriever(
            top_k=retrieval_top_k
        )

        # Calibrated Verifier (Lazy model loading preserved)
        self.calibrated_verifier = calibrated_verifier or CalibratedDebertaVerifier(
            bundle_path=bundle_path,
            default_regime=regime_norm,  # type: ignore[arg-type]
            model_path=model_path,
            device=device,
            batch_size=batch_size,
        )

        # Response Scorer
        self.response_scorer = response_scorer or ResponseScorer()

    def _parse_reference_document(
        self,
        document: Union[str, Path, bytes, io.BytesIO],
        document_name: Optional[str] = None,
    ) -> Tuple[str, str, List[DocumentChunk], Dict[str, Any]]:
        """Parses and chunks the reference document across all supported input formats.

        Returns:
            Tuple of (doc_name, full_cleaned_text, document_chunks, doc_metadata)
        """
        if document is None:
            raise ValueError("Reference document cannot be None.")

        doc_meta: Dict[str, Any] = {}
        chunks: List[DocumentChunk] = []

        if isinstance(document, (str, Path)):
            # If string is empty or whitespace, treat immediately as empty raw text
            if isinstance(document, str) and not document.strip():
                doc_name = document_name or "document.txt"
                return doc_name, "", [], {
                    "source_type": "raw_text",
                    "total_words": 0,
                    "char_count": 0,
                    "is_scanned": False,
                }

            doc_path = Path(document) if isinstance(document, str) else document
            is_file_on_disk = False
            try:
                is_file_on_disk = doc_path.exists()
            except Exception:
                is_file_on_disk = False

            if is_file_on_disk:
                if doc_path.is_dir():
                    raise ValueError(f"Reference document path is a directory, not a file: {doc_path}")

                ext = doc_path.suffix.lower()
                if ext == ".pdf":
                    parsed = parse_pdf(doc_path, filename=document_name or doc_path.name)
                    doc_name = parsed.filename
                    if parsed.is_scanned:
                        raise ValueError(
                            f"Scanned PDF detected in '{doc_name}': The document contains no extractable "
                            "machine-readable text. OCR is not supported; please provide a text-searchable "
                            "PDF or plain text document."
                        )
                    full_text = parsed.full_text
                    chunks = self.chunker.chunk_pdf_pages(parsed.pages, document_name=doc_name)
                    doc_meta = {
                        "source_type": "pdf_file",
                        "total_pages": parsed.total_pages,
                        "total_words": parsed.total_words,
                        "char_count": len(full_text),
                        "is_scanned": False,
                    }
                elif ext in SUPPORTED_TEXT_EXTENSIONS:
                    raw_text = doc_path.read_text(encoding="utf-8", errors="replace")
                    doc_name = document_name or doc_path.name
                    full_text = clean_text(raw_text)
                    chunks = self.chunker.chunk_plain_text(full_text, document_name=doc_name)
                    doc_meta = {
                        "source_type": "text_file",
                        "total_words": len(full_text.split()),
                        "char_count": len(full_text),
                        "is_scanned": False,
                    }
                else:
                    raise ValueError(
                        f"Unsupported file format '{doc_path.suffix}' for document '{doc_path.name}'. "
                        f"Supported formats are PDF (.pdf) and plain text ({', '.join(sorted(SUPPORTED_TEXT_EXTENSIONS))})."
                    )
            else:
                # Path does not exist on disk
                if isinstance(document, Path):
                    raise FileNotFoundError(f"Reference document file not found at: {document}")

                doc_str = str(document)
                if doc_str.lower().endswith(".pdf"):
                    raise FileNotFoundError(f"Reference PDF file not found at: {document}")

                # If string ends with a file extension or contains path separators without newlines, treat as missing file
                if "\n" not in doc_str and (doc_str.lower().endswith(tuple(SUPPORTED_TEXT_EXTENSIONS)) or "/" in doc_str or "\\" in doc_str):
                    raise FileNotFoundError(f"Reference document file not found at: {document}")

                # Otherwise, treat string as in-memory raw text
                doc_name = document_name or "document.txt"
                full_text = clean_text(doc_str)
                chunks = self.chunker.chunk_plain_text(full_text, document_name=doc_name)
                doc_meta = {
                    "source_type": "raw_text",
                    "total_words": len(full_text.split()),
                    "char_count": len(full_text),
                    "is_scanned": False,
                }

        elif isinstance(document, bytes):
            # Check for PDF magic header
            if document.startswith(b"%PDF"):
                doc_name = document_name or "document.pdf"
                parsed = parse_pdf(document, filename=doc_name)
                if parsed.is_scanned:
                    raise ValueError(
                        f"Scanned PDF detected in '{doc_name}': The document contains no extractable "
                        "machine-readable text. OCR is not supported; please provide a text-searchable "
                        "PDF or plain text document."
                    )
                full_text = parsed.full_text
                chunks = self.chunker.chunk_pdf_pages(parsed.pages, document_name=doc_name)
                doc_meta = {
                    "source_type": "pdf_bytes",
                    "total_pages": parsed.total_pages,
                    "total_words": parsed.total_words,
                    "char_count": len(full_text),
                    "is_scanned": False,
                }
            elif b"\x00" in document:
                # Binary non-PDF data
                raise ValueError(
                    "Binary data detected in document bytes. Supported formats are PDF and UTF-8 encoded text."
                )
            else:
                doc_name = document_name or "document.txt"
                try:
                    raw_text = document.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ValueError(
                        f"Document bytes could not be decoded as UTF-8 text: {exc}"
                    ) from exc
                full_text = clean_text(raw_text)
                chunks = self.chunker.chunk_plain_text(full_text, document_name=doc_name)
                doc_meta = {
                    "source_type": "text_bytes",
                    "total_words": len(full_text.split()),
                    "char_count": len(full_text),
                    "is_scanned": False,
                }

        elif isinstance(document, io.BytesIO):
            doc_name = document_name or "document.pdf"
            parsed = parse_pdf(document, filename=doc_name)
            if parsed.is_scanned:
                raise ValueError(
                    f"Scanned PDF detected in '{doc_name}': The document contains no extractable "
                    "machine-readable text. OCR is not supported; please provide a text-searchable "
                    "PDF or plain text document."
                )
            full_text = parsed.full_text
            chunks = self.chunker.chunk_pdf_pages(parsed.pages, document_name=doc_name)
            doc_meta = {
                "source_type": "bytes_io",
                "total_pages": parsed.total_pages,
                "total_words": parsed.total_words,
                "char_count": len(full_text),
                "is_scanned": False,
            }

        else:
            raise ValueError(
                f"Unsupported document input type: '{type(document).__name__}'. "
                "Expected str (text or path), Path, bytes, or io.BytesIO."
            )

        return doc_name, full_text, chunks, doc_meta

    def verify(
        self,
        document: Union[str, Path, bytes, io.BytesIO],
        response: str,
        prompt: Optional[str] = None,
        regime: Optional[str] = None,
        document_name: Optional[str] = None,
        response_id: str = "resp_001",
    ) -> PipelineResult:
        """Execute end-to-end factual verification on an LLM response.

        Args:
            document: Reference document (file path, raw text, bytes, or BytesIO).
            response: Candidate response text generated by an LLM.
            prompt: Optional user query or instruction that prompted the response.
            regime: Verification regime to execute ('direct' or 'retrieval'). Defaults to self.default_regime.
            document_name: Optional custom filename/title for provenance reporting.
            response_id: Identifier for provenance tracking in the scorer.

        Returns:
            PipelineResult containing verified claims, metrics, factuality indices, and provenance.
        """
        start_time = time.perf_counter()
        warnings: List[str] = []

        # 1. Validate response
        if response is None:
            raise ValueError("Response text cannot be None.")
        if not isinstance(response, str):
            raise ValueError(f"Response must be a string, got {type(response).__name__}.")

        # 2. Validate regime
        active_regime = (regime or self.default_regime).lower().strip()
        if active_regime not in ("direct", "retrieval"):
            raise ValueError(
                f"Invalid verification regime: '{regime}'. Expected 'direct' or 'retrieval'."
            )

        # 3. Ingest and parse reference document
        doc_name, full_text, chunks, doc_meta = self._parse_reference_document(
            document, document_name=document_name
        )
        is_empty_document = not full_text or not full_text.strip()
        if is_empty_document:
            warnings.append(
                f"Reference document '{doc_name}' is empty or contains only whitespace. "
                "All empirical factual claims are marked UNVERIFIABLE."
            )

        # 4. Extract atomic claims from candidate response
        is_empty_response = not response or not response.strip()
        if is_empty_response:
            extracted_claims: List[ExtractedClaim] = []
            extraction_stats: Dict[str, Any] = {
                "total_sentences": 0,
                "factual_claims_count": 0,
                "non_factual_count": 0,
            }
            warnings.append("Candidate LLM response is empty or contains only whitespace.")
        else:
            extraction_result = self.claim_extractor.extract_claims(response.strip())
            extracted_claims = extraction_result.claims
            extraction_stats = {
                "total_sentences": extraction_result.total_sentences,
                "factual_claims_count": extraction_result.factual_claims_count,
                "non_factual_count": extraction_result.non_factual_count,
            }
            if extraction_result.factual_claims_count == 0 and len(extracted_claims) > 0:
                warnings.append(
                    "Response contains no empirical factual claims (only opinions, greetings, "
                    "or conversational remarks). Factuality metrics are undefined."
                )

        # 5. Long-document Direct mode detection & warning
        direct_context_truncated = False
        if active_regime == "direct" and not is_empty_document:
            word_count = len(full_text.split())
            approx_tokens = int(word_count * 1.35)

            # Check tokenizer directly if available
            tokenizer = None
            if (
                hasattr(self.calibrated_verifier, "_custom_deberta")
                and self.calibrated_verifier._custom_deberta is not None
            ):
                tokenizer = getattr(
                    self.calibrated_verifier._custom_deberta, "tokenizer", None
                )

            if tokenizer is not None and hasattr(tokenizer, "encode"):
                try:
                    sample_claim = extracted_claims[0].text if extracted_claims else ""
                    pair_tokens = len(
                        tokenizer.encode(full_text, sample_claim, truncation=False)
                    )
                    direct_context_truncated = pair_tokens > DEBERTA_MAX_INPUT_TOKENS
                    approx_tokens = pair_tokens
                except Exception:
                    direct_context_truncated = approx_tokens > DEBERTA_MAX_INPUT_TOKENS or word_count > DIRECT_MODE_SAFE_WORD_LIMIT
            else:
                direct_context_truncated = approx_tokens > DEBERTA_MAX_INPUT_TOKENS or word_count > DIRECT_MODE_SAFE_WORD_LIMIT

            if direct_context_truncated:
                trunc_warning = (
                    f"Direct mode document length (~{approx_tokens} tokens, {word_count} words) "
                    f"exceeds DeBERTa's {DEBERTA_MAX_INPUT_TOKENS}-token limit. Context will be "
                    "truncated during cross-encoder inference, meaning propositions later in the document "
                    "will be omitted from neural inference. Retrieval mode ('retrieval') is strongly "
                    "recommended for long documents."
                )
                warnings.append(trunc_warning)
                logger.warning(trunc_warning)

        # 6. Formulate evidence matches according to regime
        evidence_matches: List[ClaimEvidenceMatch] = []

        if active_regime == "direct":
            if is_empty_document:
                # Context is absent/empty -> create empty evidence matches
                for claim in extracted_claims:
                    evidence_matches.append(
                        ClaimEvidenceMatch(
                            claim_id=claim.claim_id,
                            claim_text=claim.text,
                            retrieved_passages=[],
                            best_passage=None,
                            best_similarity_score=0.0,
                        )
                    )
            else:
                # Direct context: construct a single document chunk representing the full reference text
                clean_doc_id = re.sub(r"[^\w\-]", "_", doc_name.lower())
                direct_chunk = DocumentChunk(
                    chunk_id=f"{clean_doc_id}_direct",
                    document_name=doc_name,
                    page_number=1,
                    chunk_index=0,
                    text=full_text.strip(),
                    start_char=0,
                    end_char=len(full_text.strip()),
                    word_count=len(full_text.strip().split()),
                )
                direct_passage = RetrievedPassage(
                    chunk=direct_chunk,
                    similarity_score=1.0,
                    rank=1,
                )
                for claim in extracted_claims:
                    evidence_matches.append(
                        ClaimEvidenceMatch(
                            claim_id=claim.claim_id,
                            claim_text=claim.text,
                            retrieved_passages=[direct_passage],
                            best_passage=direct_passage,
                            best_similarity_score=1.0,
                        )
                    )

        elif active_regime == "retrieval":
            if is_empty_document or not chunks:
                # No chunks to retrieve from -> create empty evidence matches
                for claim in extracted_claims:
                    evidence_matches.append(
                        ClaimEvidenceMatch(
                            claim_id=claim.claim_id,
                            claim_text=claim.text,
                            retrieved_passages=[],
                            best_passage=None,
                            best_similarity_score=0.0,
                        )
                    )
            else:
                # Index passages and retrieve top-k for each claim
                self.passage_retriever.index_chunks(chunks)
                evidence_matches = self.passage_retriever.retrieve_for_claims(
                    extracted_claims, top_k=self.retrieval_top_k
                )
                # Release chunk embeddings from memory to prevent memory accumulation
                self.passage_retriever.clear_index()

        # 7. Execute calibrated claim verification
        report = self.calibrated_verifier.verify_claims(
            claims=extracted_claims,
            evidence_matches=evidence_matches,
            regime=active_regime,  # type: ignore[arg-type]
            document_name=doc_name,
        )

        # Annotate claims if direct context was truncated
        if direct_context_truncated:
            for vc in report.claims:
                if vc.is_factual and vc.classification != VerificationCategory.UNVERIFIABLE:
                    vc.reasoning_trace += (
                        " [Warning: Direct context exceeded 512 tokens and was truncated. "
                        "Earlier/later context may be omitted; consider retrieval mode.]"
                    )

        # 8. Compute response-level factuality metrics and indices
        elapsed_seconds = round(time.perf_counter() - start_time, 4)
        score = self.response_scorer.score_response(
            report_or_claims=report,
            response_id=response_id,
            document_name=doc_name,
            execution_time_seconds=elapsed_seconds,
        )

        # 9. Extract calibrated probabilities for convenient structured lookup
        calibrated_probs_map: Dict[str, Dict[str, float]] = {}
        for vc in report.claims:
            m_cal = re.search(r"\[Calibrated:\s*([^\]]+)\]", vc.reasoning_trace)
            if m_cal:
                pairs = m_cal.group(1).split("|")[0].split(",")
                prob_dict: Dict[str, float] = {}
                for p in pairs:
                    parts = p.strip().split("=")
                    if len(parts) == 2:
                        k_name, v_str = parts[0].strip(), parts[1].strip()
                        v_num_str = re.sub(r"\s*\(.*?\)", "", v_str).strip()
                        try:
                            prob_dict[k_name] = float(v_num_str)
                        except ValueError:
                            pass
                if prob_dict:
                    calibrated_probs_map[vc.claim_id] = prob_dict

        pipeline_metadata: Dict[str, Any] = {
            "document": doc_meta,
            "document_empty": is_empty_document,
            "response_empty": is_empty_response,
            "direct_context_truncated": direct_context_truncated,
            "recommended_regime": (
                "retrieval"
                if (direct_context_truncated or len(chunks) > 1)
                else active_regime
            ),
            "warnings": list(warnings),
            "extraction_stats": extraction_stats,
            "indexed_chunks_count": len(chunks),
            "verification_summary": {
                "total_claims": report.total_claims_processed,
                "factual_claims": report.metrics.factual_claims,
                "non_factual_claims": report.metrics.non_factual_claims,
                "supported": report.metrics.supported_count,
                "contradicted": report.metrics.contradicted_count,
                "unsupported": report.metrics.unsupported_count,
                "unverifiable": report.metrics.unverifiable_count,
            },
            "calibrated_probabilities": calibrated_probs_map,
            "regime": active_regime,
            "prompt_provided": prompt is not None,
        }

        return PipelineResult(
            document_name=doc_name,
            regime=active_regime,
            prompt=prompt,
            response=response,
            claims=report.claims,
            verification_report=report,
            score=score,
            execution_time_seconds=elapsed_seconds,
            warnings=warnings,
            metadata=pipeline_metadata,
        )

    # Convenience alias
    def verify_response(
        self,
        document: Union[str, Path, bytes, io.BytesIO],
        response: str,
        prompt: Optional[str] = None,
        regime: Optional[str] = None,
        document_name: Optional[str] = None,
        response_id: str = "resp_001",
    ) -> PipelineResult:
        """Alias for verify()."""
        return self.verify(
            document=document,
            response=response,
            prompt=prompt,
            regime=regime,
            document_name=document_name,
            response_id=response_id,
        )
