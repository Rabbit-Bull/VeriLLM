"""
VeriLLM PDF Parser Module.
Extracts clean, page-indexed text from PDF documents using pypdf.
Handles empty pages, scanned PDFs, encrypted files, and corrupted streams gracefully.
"""

from pathlib import Path
from typing import List, Dict, Any, Union, Optional
import io
import pypdf
from .text_cleaner import clean_text


class ExtractedPage:
    """Represents text extracted from a single PDF page."""
    def __init__(self, page_number: int, text: str, raw_text: str):
        self.page_number = page_number  # 1-indexed
        self.text = text
        self.raw_text = raw_text
        self.char_count = len(text)
        self.word_count = len(text.split())

    def __repr__(self):
        return f"ExtractedPage(page={self.page_number}, words={self.word_count})"


class ParsedPDFDocument:
    """Represents a fully parsed PDF with all page-level contents."""
    def __init__(
        self,
        filename: str,
        pages: List[ExtractedPage],
        is_scanned: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.filename = filename
        self.pages = pages
        self.total_pages = len(pages)
        self.is_scanned = is_scanned
        self.metadata = metadata or {}
        self.total_words = sum(p.word_count for p in pages)

    @property
    def full_text(self) -> str:
        """Returns concatenated text across all pages separated by double newlines."""
        return "\n\n".join(p.text for p in self.pages if p.text)


def parse_pdf(
    source: Union[str, Path, bytes, io.BytesIO], 
    filename: Optional[str] = None
) -> ParsedPDFDocument:
    """
    Parses a PDF file from a disk path or in-memory byte stream.

    Args:
        source: File path (str or Path), bytes, or io.BytesIO stream.
        filename: Optional human-readable filename for provenance tracking.

    Returns:
        ParsedPDFDocument containing ordered ExtractedPage objects and document metadata.

    Raises:
        FileNotFoundError: If a given disk path does not exist.
        PermissionError: If the PDF is password-protected and cannot be decrypted.
        ValueError: If the file is not a valid PDF or is corrupted.
    """
    stream = None
    doc_name = filename or "document.pdf"

    if isinstance(source, (str, Path)):
        file_path = Path(source)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found at: {file_path}")
        doc_name = filename or file_path.name
        stream = open(file_path, "rb")
    elif isinstance(source, bytes):
        stream = io.BytesIO(source)
    elif isinstance(source, io.BytesIO):
        stream = source
    else:
        raise ValueError(f"Unsupported source type: {type(source)}")

    try:
        reader = pypdf.PdfReader(stream)

        # Handle encryption / password protection
        if reader.is_encrypted:
            try:
                decrypt_success = reader.decrypt("")
                if decrypt_success == 0:
                    raise PermissionError(
                        f"PDF '{doc_name}' is password-protected. "
                        "Encrypted PDFs must be unlocked before analysis."
                    )
            except Exception as e:
                raise PermissionError(f"Failed to decrypt password-protected PDF: {e}")

        total_pages = len(reader.pages)
        extracted_pages: List[ExtractedPage] = []
        has_any_text = False

        for idx, page in enumerate(reader.pages):
            page_num = idx + 1
            try:
                raw_page_text = page.extract_text() or ""
            except Exception:
                raw_page_text = ""

            cleaned = clean_text(raw_page_text)
            if cleaned:
                has_any_text = True

            extracted_pages.append(ExtractedPage(
                page_number=page_num,
                text=cleaned,
                raw_text=raw_page_text
            ))

        # Check if PDF is image-only / scanned
        is_scanned = (total_pages > 0 and not has_any_text)

        # Extract basic PDF metadata if available
        meta_dict = {}
        if reader.metadata:
            for k, v in reader.metadata.items():
                clean_key = str(k).lstrip("/").lower()
                meta_dict[clean_key] = str(v)

        return ParsedPDFDocument(
            filename=doc_name,
            pages=extracted_pages,
            is_scanned=is_scanned,
            metadata=meta_dict
        )

    except (pypdf.errors.PdfReadError, pypdf.errors.EmptyFileError) as e:
        raise ValueError(f"Corrupted or invalid PDF file '{doc_name}': {e}")
    finally:
        # Close file stream if we opened it from a disk path
        if isinstance(source, (str, Path)) and stream and not stream.closed:
            stream.close()

