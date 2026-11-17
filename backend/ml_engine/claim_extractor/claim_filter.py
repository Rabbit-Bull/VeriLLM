"""
VeriLLM Claim Filter & Classifier Module.
Distinguishes verifiable factual assertions from subjective opinions, questions, and conversational filler.
"""

import re
from typing import Tuple, Dict, Any, List

# Rhetorical question pattern containing embedded factual assertions
RHETORICAL_QUESTION_REGEX = re.compile(
    r"^(?:did you know|did you realize|have you heard|are you aware|can you believe|do you know)(?:\s+that)?\s+(.+?)\??$",
    re.IGNORECASE
)

# Subjective markers indicating opinion or unverifiable sentiment
OPINION_MARKERS = [
    r"\bin my (?:humble )?opinion\b",
    r"\bi (?:strongly )?(?:believe|think|feel|reckon|guess)\b",
    r"\barguably\b",
    r"\bperhaps the most\b",
    r"\bthe greatest of all time\b",
    r"\bpersonally\b",
    r"\bit seems to me\b",
    r"\bit is my view\b",
    r"\bbeautiful|wonderful|amazing|terrible|horrible|gorgeous\b",
]

# Conversational / AI meta-text markers
CONVERSATIONAL_MARKERS = [
    r"^hello\b",
    r"^hi there\b",
    r"^greetings\b",
    r"\bhope this helps\b",
    r"\blet me know if you (?:have|need)\b",
    r"\bas an ai\b",
    r"\bsure,? (?:here is|i can help)\b",
    r"\bfeel free to ask\b",
]

# Imperative / instruction markers
IMPERATIVE_MARKERS = [
    r"^(?:please )?note that\b",
    r"^(?:please )?keep in mind\b",
    r"^(?:please )?remember to\b",
    r"^(?:refer to|check out|see)\b",
]

# Negation tokens
NEGATION_REGEX = re.compile(r"\b(not|n't|never|no|neither|nor|none|nowhere|hardly|scarcely)\b", re.IGNORECASE)

# Numerical / quantitative pattern
NUMBERS_REGEX = re.compile(r"(\b\d+(?:[\.,]\d+)?\b|\b(?:first|second|third|fourth|fifth|one|two|three|four|five|six|seven|eight|nine|ten|hundred|thousand|million|billion)\b)", re.IGNORECASE)


def classify_proposition(text: str) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Classifies a proposition text into:
    - is_factual: True / False
    - claim_type: FACTUAL_PROPOSITION, NUMERICAL_ASSERTION, SUBJECTIVE_OPINION, QUESTION, CONVERSATIONAL, IMPERATIVE
    - metadata: dict of attributes (has_negation, has_numbers, opinion_detected)
    """
    clean_text = text.strip()
    lower_text = clean_text.lower()

    has_negation = bool(NEGATION_REGEX.search(clean_text))
    has_numbers = bool(NUMBERS_REGEX.search(clean_text))

    # 1. Check for rhetorical questions with embedded factual assertions
    rhet_match = RHETORICAL_QUESTION_REGEX.match(clean_text)
    if rhet_match:
        embedded_clause = rhet_match.group(1).strip()
        # Verify that the embedded clause contains a verbal predicate
        has_verb = bool(re.search(r"\b(is|are|was|were|has|have|had|landed|houses|built|contains|occurred|took|won|discovered|orbits|stands)\b", embedded_clause, re.IGNORECASE))
        if has_verb:
            claim_type = "NUMERICAL_ASSERTION" if has_numbers else "FACTUAL_PROPOSITION"
            return True, claim_type, {
                "has_negation": has_negation,
                "has_numbers": has_numbers,
                "filter_reason": "Rhetorical question containing embedded factual assertion"
            }

    # 2. Genuine Questions
    if clean_text.endswith("?") or lower_text.startswith(("what ", "who ", "why ", "where ", "when ", "how ", "is it ", "are there ", "can you ")):
        return False, "QUESTION", {
            "has_negation": has_negation,
            "has_numbers": has_numbers,
            "filter_reason": "Interrogative sentence"
        }

    # 2. Conversational greetings / meta-text
    for pattern in CONVERSATIONAL_MARKERS:
        if re.search(pattern, lower_text):
            return False, "CONVERSATIONAL", {
                "has_negation": has_negation,
                "has_numbers": has_numbers,
                "filter_reason": "Conversational framing / AI meta-text"
            }

    # 3. Imperative instructions
    for pattern in IMPERATIVE_MARKERS:
        if re.search(pattern, lower_text):
            return False, "IMPERATIVE", {
                "has_negation": has_negation,
                "has_numbers": has_numbers,
                "filter_reason": "Imperative instruction"
            }

    # 4. Subjective opinions
    for pattern in OPINION_MARKERS:
        if re.search(pattern, lower_text):
            return False, "SUBJECTIVE_OPINION", {
                "has_negation": has_negation,
                "has_numbers": has_numbers,
                "filter_reason": "Subjective opinion or hedging language"
            }

    # 5. Numerical / statistical factual claim vs general factual claim
    if has_numbers:
        claim_type = "NUMERICAL_ASSERTION"
    else:
        claim_type = "FACTUAL_PROPOSITION"

    return True, claim_type, {
        "has_negation": has_negation,
        "has_numbers": has_numbers,
        "filter_reason": "Empirical proposition"
    }

