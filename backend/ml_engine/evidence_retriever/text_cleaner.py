"""
VeriLLM Text Cleaner Module.
Cleans raw extracted text from PDFs and text files without destroying meaningful numbers, dates, or symbols.
"""

import re
from typing import Optional

# Regex to match words broken by a hyphen at the end of a line (e.g. "estab-\nlished" -> "established")
HYPHENATED_LINE_BREAK_REGEX = re.compile(r"(\b[a-zA-Z]{2,})-\s*[\r\n]+\s*([a-zA-Z]{2,}\b)")

# Regex to clean recurring standalone page numbers (e.g. "Page 1 of 10", "--- 3 ---", "- 4 -", "Page 1")
PAGE_NUMBER_FOOTER_REGEX = re.compile(
    r"(?i)(?:^|\n)\s*[-—–]*\s*(?:page\s+)?\d+(?:\s*(?:of|\/)\s*\d+)?\s*[-—–]*\s*(?:\n|$)|(?:\bpage\s+\d+(?:\s*(?:of|\/)\s*\d+)?\b)",
    re.MULTILINE
)

# Unicode characters to normalize
UNICODE_REPLACEMENTS = {
    "\xa0": " ",       # Non-breaking space
    "\u200b": "",      # Zero-width space
    "\u200e": "",      # Left-to-right mark
    "\u200f": "",      # Right-to-left mark
    "\u2018": "'",     # Left single quote
    "\u2019": "'",     # Right single quote
    "\u201c": '"',     # Left double quote
    "\u201d": '"',     # Right double quote
    "\u2013": "-",     # En dash
    "\u2014": "--",    # Em dash
    "\r\n": "\n",      # Normalize Windows line endings
    "\r": "\n"
}


def clean_text(text: str) -> str:
    """
    Cleans raw document text while preserving semantic content, numbers, and dates.

    Args:
        text: Raw text extracted from PDF or plain text document.

    Returns:
        Cleaned, normalized text string.
    """
    if not text:
        return ""

    cleaned = text

    # 1. Normalize unicode characters
    for orig, repl in UNICODE_REPLACEMENTS.items():
        cleaned = cleaned.replace(orig, repl)

    # 2. De-hyphenate words broken across line wraps
    cleaned = HYPHENATED_LINE_BREAK_REGEX.sub(r"\1\2", cleaned)

    # 3. Strip standalone page number footers
    cleaned = PAGE_NUMBER_FOOTER_REGEX.sub("", cleaned)

    # 4. Normalize multiple blank lines (max 2 consecutive newlines for paragraph breaks)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    # 5. Normalize multiple horizontal spaces and tabs into a single space
    cleaned = re.sub(r"[ \t]+", " ", cleaned)

    # 6. Normalize space before punctuation
    cleaned = re.sub(r"\s+([,\.!?;:])", r"\1", cleaned)

    return cleaned.strip()
