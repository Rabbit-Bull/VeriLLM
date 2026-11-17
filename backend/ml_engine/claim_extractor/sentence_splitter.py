"""
VeriLLM Sentence Splitter Module.
Segments raw response text into clean sentence units while preserving exact character offsets.
"""

import re
from typing import List, Tuple

# Common English abbreviations that should not trigger sentence boundaries
ABBREVIATIONS = {
    "dr.", "mr.", "mrs.", "ms.", "prof.", "sr.", "jr.", "st.",
    "u.s.", "u.k.", "e.u.", "u.n.", "e.g.", "i.e.", "etc.",
    "vs.", "approx.", "fig.", "dept.", "jan.", "feb.", "mar.",
    "apr.", "jun.", "jul.", "aug.", "sep.", "sept.", "oct.",
    "nov.", "dec.", "a.m.", "p.m.", "b.c.", "a.d."
}

# Regex to detect bullet point markers at start of a line/sentence
BULLET_REGEX = re.compile(r"^(\s*[-*•]\s+|\s*\d+[\.\)]\s+)")


class SentenceSpan:
    """Represents a segmented sentence with character offsets."""
    def __init__(self, text: str, start_char: int, end_char: int):
        self.text = text.strip()
        self.start_char = start_char
        self.end_char = end_char

    def __repr__(self):
        return f"SentenceSpan({self.text!r}, {self.start_char}:{self.end_char})"


def split_sentences_with_spans(text: str) -> List[SentenceSpan]:
    """
    Splits text into sentences while calculating exact start and end offsets.
    Handles abbreviations, decimals, bullet points, and newlines.
    """
    if not text or not text.strip():
        return []

    # First, normalize Windows line endings while preserving length
    # Note: text is processed with original indices
    spans: List[SentenceSpan] = []

    # Split paragraphs by newlines first
    paragraph_pattern = re.compile(r"[^\r\n]+", re.MULTILINE)
    
    for para_match in paragraph_pattern.finditer(text):
        para_text = para_match.group()
        para_start = para_match.start()

        # Check if the paragraph is a bullet list item
        bullet_match = BULLET_REGEX.match(para_text)
        prefix_len = 0
        cleaned_para = para_text
        if bullet_match:
            prefix_len = len(bullet_match.group())
            cleaned_para = para_text[prefix_len:]

        # Segment the paragraph text into sentences
        # Sentence ending regex: period, exclamation, question mark followed by space or end of string
        # Lookbehind prevents splitting inside decimals (e.g. 3.14) or initials
        sent_end_pattern = re.compile(r'([.?!])+(?=\s+[A-Z0-9"\'(]|\s*$)')
        
        last_idx = 0
        current_len = len(cleaned_para)

        for match in sent_end_pattern.finditer(cleaned_para):
            end_pos = match.end()
            cand_sent = cleaned_para[last_idx:end_pos].strip()

            if not cand_sent:
                last_idx = end_pos
                continue

            # Check if this period is part of a known abbreviation
            tokens = cand_sent.split()
            last_word = tokens[-1].lower() if tokens else ""

            # If last word is an abbreviation and we haven't reached end of paragraph, do not split
            if last_word in ABBREVIATIONS and end_pos < current_len:
                continue

            # Check for initials like "J. K. Rowling"
            if re.match(r"^[A-Za-z]\.$", last_word) and end_pos < current_len:
                continue

            # Valid sentence boundary found
            # Calculate absolute character offsets in the original text
            raw_substr = cleaned_para[last_idx:end_pos]
            leading_spaces = len(raw_substr) - len(raw_substr.lstrip())
            trailing_spaces = len(raw_substr) - len(raw_substr.rstrip())

            abs_start = para_start + prefix_len + last_idx + leading_spaces
            abs_end = para_start + prefix_len + end_pos - trailing_spaces

            if cand_sent:
                spans.append(SentenceSpan(
                    text=cand_sent,
                    start_char=abs_start,
                    end_char=abs_end
                ))
            last_idx = end_pos

        # Remaining trailing text in paragraph
        if last_idx < current_len:
            trailing_sent = cleaned_para[last_idx:].strip()
            if trailing_sent:
                raw_substr = cleaned_para[last_idx:]
                leading_spaces = len(raw_substr) - len(raw_substr.lstrip())
                trailing_spaces = len(raw_substr) - len(raw_substr.rstrip())
                abs_start = para_start + prefix_len + last_idx + leading_spaces
                abs_end = para_start + prefix_len + current_len - trailing_spaces

                spans.append(SentenceSpan(
                    text=trailing_sent,
                    start_char=abs_start,
                    end_char=abs_end
                ))

    return spans

