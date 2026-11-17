"""
VeriLLM Phase 4 Unit Tests: Evidence and Document-Processing Engine.
Tests PDF parsing, page numbering, text cleaning, sliding-window chunking,
dense passage retrieval, and edge cases (empty PDFs, irrelevant queries).
"""

import sys
import io
from pathlib import Path
import pytest
import pypdf

# Ensure backend root is on sys.path
CURRENT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = CURRENT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml_engine.evidence_retriever import (
    clean_text,
    parse_pdf,
    SlidingWindowChunker,
    PassageRetriever,
    DocumentChunk,
    RetrievedPassage
)


# Helper function to generate an in-memory multi-page PDF using pypdf
def create_test_pdf() -> bytes:
    """Generates a valid 3-page PDF document in memory using pypdf."""
    writer = pypdf.PdfWriter()

    font = pypdf.generic.DictionaryObject({
        pypdf.generic.NameObject("/Type"): pypdf.generic.NameObject("/Font"),
        pypdf.generic.NameObject("/Subtype"): pypdf.generic.NameObject("/Type1"),
        pypdf.generic.NameObject("/BaseFont"): pypdf.generic.NameObject("/Helvetica")
    })
    font_ref = writer._add_object(font)

    pages_text = [
        # Page 1: Launch and Descent
        (
            "Apollo 11 was launched by a Saturn V rocket from Kennedy Space Center on July 16, 1969. "
            "The spacecraft had three parts: a Command Module with a cabin for the three astronauts, "
            "a Service Module, and a Lunar Module. Commander Neil Armstrong and Lunar Module Pilot "
            "Buzz Aldrin landed the Lunar Module Eagle on the Moon on July 20, 1969."
        ),
        # Page 2: Lunar Surface Operations
        (
            "Neil Armstrong became the first person to walk on the Moon on July 21, 1969. "
            "Buzz Aldrin joined him 19 minutes later. They spent over two hours on the surface, "
            "taking photographs and collecting 47.5 pounds of lunar rock and soil samples. "
            "Meanwhile, Michael Collins flew the Command Module Columbia alone in lunar orbit."
        ),
        # Page 3: Return to Earth
        (
            "After more than 21 hours on the lunar surface, Eagle lifted off to rejoin Columbia. "
            "The astronauts returned safely to Earth and splashed down in the Pacific Ocean on July 24, 1969. "
            "They were recovered by the USS Hornet and spent 21 days in quarantine."
        )
    ]

    for p_text in pages_text:
        page = writer.add_blank_page(width=612, height=792)
        stream_data = f"BT /F1 12 Tf 50 700 Td ({p_text}) Tj ET".encode("latin-1")
        stream = pypdf.generic.DecodedStreamObject()
        stream.set_data(stream_data)
        stream_ref = writer._add_object(stream)

        page[pypdf.generic.NameObject("/Contents")] = stream_ref
        page[pypdf.generic.NameObject("/Resources")] = pypdf.generic.DictionaryObject({
            pypdf.generic.NameObject("/Font"): pypdf.generic.DictionaryObject({
                pypdf.generic.NameObject("/F1"): font_ref
            })
        })

    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


# Helper function to generate an empty/scanned PDF (pages with zero text)
def create_empty_scanned_pdf() -> bytes:
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.add_blank_page(width=612, height=792)
    stream = io.BytesIO()
    writer.write(stream)
    return stream.getvalue()


def test_text_cleaner_dehyphenation_and_unicode():
    """Verifies that hyphenated line wraps are merged and unicode whitespace is normalized."""
    raw = "The space-\ncraft was estab-\r\nlishend in 1969.   Page 1 of 5   It collected 47.5\xa0pounds."
    cleaned = clean_text(raw)

    assert "spacecraft" in cleaned
    assert "established" in cleaned or "establishend" in cleaned
    assert "Page 1 of 5" not in cleaned
    assert "\xa0" not in cleaned
    assert "47.5 pounds" in cleaned


def test_pdf_parsing_page_preservation():
    """Verifies PDF extraction, page count, and page text attribution."""
    pdf_bytes = create_test_pdf()
    doc = parse_pdf(pdf_bytes, filename="apollo11.pdf")

    assert doc.total_pages == 3
    assert doc.filename == "apollo11.pdf"
    assert doc.is_scanned is False

    # Page 1 checks
    assert doc.pages[0].page_number == 1
    assert "Saturn V rocket" in doc.pages[0].text
    assert "July 20, 1969" in doc.pages[0].text

    # Page 2 checks
    assert doc.pages[1].page_number == 2
    assert "47.5 pounds" in doc.pages[1].text
    assert "Michael Collins" in doc.pages[1].text

    # Page 3 checks
    assert doc.pages[2].page_number == 3
    assert "Pacific Ocean" in doc.pages[2].text
    assert "USS Hornet" in doc.pages[2].text


def test_empty_scanned_pdf_detection():
    """Verifies that PDFs with no machine-readable text are flagged as is_scanned=True."""
    empty_bytes = create_empty_scanned_pdf()
    doc = parse_pdf(empty_bytes, filename="scanned.pdf")

    assert doc.total_pages == 2
    assert doc.is_scanned is True
    assert doc.total_words == 0


def test_corrupted_pdf_handling():
    """Verifies that corrupted non-PDF byte streams raise a clear ValueError."""
    corrupted_bytes = b"NOT_A_REAL_PDF_HEADER_123456789"
    with pytest.raises(ValueError) as excinfo:
        parse_pdf(corrupted_bytes, filename="bad.pdf")
    assert "Corrupted or invalid PDF" in str(excinfo.value)


def test_sliding_window_chunking_with_overlap():
    """Verifies sliding window chunking word limits and overlap preservation."""
    chunker = SlidingWindowChunker(chunk_size_words=30, overlap_words=10, min_chunk_words=10)
    text = (
        "Sentence one discusses the launch of the mission from the launchpad. "
        "Sentence two describes the flight trajectory into Earth orbit. "
        "Sentence three covers the translunar injection burn. "
        "Sentence four describes entering lunar orbit. "
        "Sentence five details the descent of the lunar module Eagle."
    )
    chunks = chunker.chunk_plain_text(text, document_name="mission.txt", page_number=1)

    assert len(chunks) >= 2
    for c in chunks:
        assert c.document_name == "mission.txt"
        assert c.page_number == 1
        assert c.word_count >= 10
        assert c.chunk_id.startswith("mission_txt_p01_c")

    # Check that adjacent chunks share overlapping words
    words_c1 = set(chunks[0].text.split())
    words_c2 = set(chunks[1].text.split())
    common_words = words_c1.intersection(words_c2)
    assert len(common_words) >= 3


def test_dense_passage_retrieval():
    """Verifies that dense passage retrieval retrieves the correct page/chunk for specific claims."""
    pdf_bytes = create_test_pdf()
    doc = parse_pdf(pdf_bytes, filename="apollo11.pdf")
    
    chunker = SlidingWindowChunker(chunk_size_words=60, overlap_words=15)
    chunks = chunker.chunk_pdf_pages(doc.pages, document_name="apollo11.pdf")
    assert len(chunks) >= 3

    retriever = PassageRetriever(top_k=2)
    retriever.index_chunks(chunks)

    # Query 1: Targeting Page 2 fact (lunar rocks collected)
    claim_1 = "Armstrong and Aldrin collected 47.5 pounds of lunar rock."
    passages_1 = retriever.retrieve_for_claim(claim_1, top_k=2)

    assert len(passages_1) == 2
    top_1 = passages_1[0]
    assert top_1.chunk.page_number == 2
    assert "47.5 pounds" in top_1.chunk.text
    assert top_1.similarity_score > 0.5

    # Query 2: Targeting Page 3 fact (Pacific Ocean splashdown)
    claim_2 = "The astronauts splashed down safely in the Pacific Ocean."
    passages_2 = retriever.retrieve_for_claim(claim_2, top_k=2)

    top_2 = passages_2[0]
    assert top_2.chunk.page_number == 3
    assert "Pacific Ocean" in top_2.chunk.text
    assert top_2.similarity_score > 0.40

    # Query 3: Completely irrelevant query
    irrelevant_claim = "Chocolate chip cookies require flour, sugar, and butter."
    passages_irrelevant = retriever.retrieve_for_claim(irrelevant_claim, top_k=1)
    
    # Cosine score should be substantially lower than relevant queries
    assert passages_irrelevant[0].similarity_score < top_1.similarity_score
    assert passages_irrelevant[0].similarity_score < top_2.similarity_score
    assert passages_irrelevant[0].similarity_score < 0.25


def test_batch_claim_evidence_matching():
    """Verifies batch retrieval for multiple extracted claims simultaneously."""
    pdf_bytes = create_test_pdf()
    doc = parse_pdf(pdf_bytes, filename="apollo11.pdf")
    chunker = SlidingWindowChunker(chunk_size_words=60, overlap_words=15)
    chunks = chunker.chunk_pdf_pages(doc.pages, document_name="apollo11.pdf")

    retriever = PassageRetriever(top_k=2)
    retriever.index_chunks(chunks)

    claims = [
        {"claim_id": "c_001", "text": "A Saturn V rocket launched the Apollo 11 spacecraft."},
        {"claim_id": "c_002", "text": "Michael Collins stayed in lunar orbit inside Columbia."}
    ]

    matches = retriever.retrieve_for_claims(claims, top_k=2)

    assert len(matches) == 2
    # Match 1 should point to Page 1
    assert matches[0].claim_id == "c_001"
    assert matches[0].best_passage.chunk.page_number == 1
    assert "Saturn V" in matches[0].best_passage.chunk.text

    # Match 2: retrieved passages should contain the Michael Collins lunar orbit chunk (Page 2)
    assert matches[1].claim_id == "c_002"
    retrieved_pages = [p.chunk.page_number for p in matches[1].retrieved_passages]
    assert 2 in retrieved_pages
    assert any("Michael Collins" in p.chunk.text for p in matches[1].retrieved_passages)

