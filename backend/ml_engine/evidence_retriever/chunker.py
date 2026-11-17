"""
VeriLLM Document Chunker Module.
Implements a sliding-window chunking strategy with sentence boundary awareness and token/word overlap.
Preserves page numbers, character offsets, and document provenance for all chunks.
"""

import re
from typing import List, Optional
from .models import DocumentChunk
from .pdf_parser import ExtractedPage


class SlidingWindowChunker:
    """
    Splits text or multi-page documents into overlapping passages.
    Prioritizes sentence boundaries to avoid slicing propositions in half.
    """

    def __init__(
        self,
        chunk_size_words: int = 150,
        overlap_words: int = 35,
        min_chunk_words: int = 20
    ):
        """
        Args:
            chunk_size_words: Target word length for each passage chunk.
            overlap_words: Number of overlapping words between consecutive chunks.
            min_chunk_words: Minimum word length to retain a trailing chunk.
        """
        assert chunk_size_words > overlap_words, "chunk_size_words must be greater than overlap_words"
        self.chunk_size_words = chunk_size_words
        self.overlap_words = overlap_words
        self.min_chunk_words = min_chunk_words

    def chunk_plain_text(
        self, 
        text: str, 
        document_name: str = "document.txt",
        page_number: Optional[int] = 1
    ) -> List[DocumentChunk]:
        """
        Segments a single plain text document into sliding window chunks.
        """
        if not text or not text.strip():
            return []

        clean_doc_id = re.sub(r"[^\w\-]", "_", document_name.lower())
        chunks: List[DocumentChunk] = []

        # Split into sentences to respect semantic boundaries
        sentence_regex = re.compile(r'(?<=[.?!])\s+(?=[A-Z0-9"\'(])|\n\n+')
        sentences = [s.strip() for s in sentence_regex.split(text) if s.strip()]

        if not sentences:
            sentences = [text.strip()]

        current_sentences: List[str] = []
        current_word_count = 0
        chunk_seq = 1

        for sent in sentences:
            sent_words = len(sent.split())

            # If adding this sentence exceeds chunk size and we already have enough words
            if current_word_count + sent_words > self.chunk_size_words and current_word_count >= self.min_chunk_words:
                chunk_text = " ".join(current_sentences).strip()
                chunk_id = f"{clean_doc_id}_p{page_number or 1:02d}_c{chunk_seq:03d}"
                
                # Approximate char offsets in text
                start_char = text.find(current_sentences[0]) if current_sentences else 0
                end_char = start_char + len(chunk_text)

                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    document_name=document_name,
                    page_number=page_number,
                    chunk_index=chunk_seq - 1,
                    text=chunk_text,
                    start_char=max(0, start_char),
                    end_char=end_char,
                    word_count=len(chunk_text.split()),
                    metadata={"sentence_count": len(current_sentences)}
                ))
                chunk_seq += 1

                # Retain overlapping sentences for the next window
                overlap_accum: List[str] = []
                overlap_words_count = 0
                for prev_sent in reversed(current_sentences):
                    prev_cnt = len(prev_sent.split())
                    if overlap_words_count + prev_cnt <= self.overlap_words:
                        overlap_accum.insert(0, prev_sent)
                        overlap_words_count += prev_cnt
                    else:
                        break

                current_sentences = overlap_accum
                current_word_count = overlap_words_count

            current_sentences.append(sent)
            current_word_count += sent_words

        # Handle remaining trailing sentences
        if current_sentences:
            chunk_text = " ".join(current_sentences).strip()
            if len(chunk_text.split()) >= self.min_chunk_words or not chunks:
                chunk_id = f"{clean_doc_id}_p{page_number or 1:02d}_c{chunk_seq:03d}"
                start_char = text.find(current_sentences[0]) if current_sentences else 0
                end_char = start_char + len(chunk_text)

                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    document_name=document_name,
                    page_number=page_number,
                    chunk_index=chunk_seq - 1,
                    text=chunk_text,
                    start_char=max(0, start_char),
                    end_char=end_char,
                    word_count=len(chunk_text.split()),
                    metadata={"sentence_count": len(current_sentences)}
                ))

        return chunks

    def chunk_pdf_pages(
        self, 
        pages: List[ExtractedPage], 
        document_name: str = "document.pdf"
    ) -> List[DocumentChunk]:
        """
        Segments a multi-page PDF document into chunks while strictly preserving page numbers.
        Each page is chunked independently to guarantee exact page-number attribution.
        """
        all_chunks: List[DocumentChunk] = []
        global_idx = 0

        for page in pages:
            if not page.text or not page.text.strip():
                continue

            page_chunks = self.chunk_plain_text(
                text=page.text,
                document_name=document_name,
                page_number=page.page_number
            )

            # Re-index chunk_index sequentially across the whole document
            for c in page_chunks:
                c.chunk_index = global_idx
                global_idx += 1
                all_chunks.append(c)

        return all_chunks

