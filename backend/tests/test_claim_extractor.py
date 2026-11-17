"""
VeriLLM Phase 3 Unit Tests: Claim Extraction Engine.
Covers atomic claim extraction, compound sentences, dates/numbers, negations,
lists, opinions vs facts, questions, and character offset traceability.
"""

import sys
from pathlib import Path
import pytest

# Ensure backend root is on sys.path
CURRENT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = CURRENT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml_engine.claim_extractor import ClaimExtractor, ExtractedClaim, ExtractionResult


@pytest.fixture(scope="module")
def extractor():
    """Provides a reusable instance of ClaimExtractor."""
    return ClaimExtractor(resolve_coreference=True, decompose_compound=True, include_non_factual=True)


def test_master_prompt_eiffel_tower_example(extractor):
    """
    Verifies the exact example from PROJECT_MASTER_PROMPT.md:
    'The Eiffel Tower is in Paris. It was completed in 1889. It is 500 metres tall.'
    Should extract 3 atomic claims with 'It' resolved to 'The Eiffel Tower'.
    """
    text = "The Eiffel Tower is in Paris. It was completed in 1889. It is 500 metres tall."
    result = extractor.extract_claims(text)

    assert result.total_sentences == 3
    assert len(result.claims) == 3
    assert result.factual_claims_count == 3

    claim_texts = [c.text for c in result.claims]
    assert "The Eiffel Tower is in Paris." in claim_texts[0]
    # Check that pronoun 'It' was resolved to 'The Eiffel Tower'
    assert "The Eiffel Tower was completed in 1889." in claim_texts[1]
    assert "The Eiffel Tower is 500 metres tall." in claim_texts[2]

    # Verify numbers flag on claims 2 and 3
    assert result.claims[0].has_numbers is False
    assert result.claims[1].has_numbers is True
    assert result.claims[2].has_numbers is True


def test_compound_sentence_decomposition(extractor):
    """
    Verifies that coordinating conjunctions join multiple atomic claims in one sentence.
    """
    text = "Marie Curie was born in Poland in 1867 and was the first woman to win a Nobel Prize."
    result = extractor.extract_claims(text)

    # Should decompose into at least 2 propositions sharing the subject 'Marie Curie'
    assert len(result.claims) >= 2
    claim_texts = [c.text for c in result.claims]
    assert any("Marie Curie" in t and "born in Poland" in t for t in claim_texts)
    assert any("Marie Curie" in t and "Nobel Prize" in t for t in claim_texts)


def test_dates_numbers_and_entities(extractor):
    """
    Verifies detection of numerical values, dates, and named entities.
    """
    text = "Apollo 11 landed on the Moon on July 20, 1969, and collected 47.5 pounds of samples."
    result = extractor.extract_claims(text)

    factual = result.factual_claims
    assert len(factual) >= 1
    assert any(c.has_numbers for c in factual)
    
    # Check that entities like DATE or CARDINAL were captured
    all_entities = [e["text"] for c in factual for e in c.named_entities]
    assert any("1969" in e or "July 20" in e for e in all_entities)


def test_negation_detection(extractor):
    """
    Verifies that explicit negation tokens are tagged accurately.
    """
    text = "The Eiffel Tower is not made of wood. Mars has never supported liquid oceans."
    result = extractor.extract_claims(text)

    assert len(result.claims) == 2
    assert result.claims[0].has_negation is True
    assert result.claims[1].has_negation is True


def test_markdown_bullet_lists(extractor):
    """
    Verifies handling of markdown bullet points and numbered lists.
    """
    text = (
        "Here are three space facts:\n"
        "1. Earth orbits the Sun once every 365 days.\n"
        "2. Mars has two small moons named Phobos and Deimos.\n"
        "3. Jupiter is the largest planet in our solar system."
    )
    result = extractor.extract_claims(text)

    # Bullet items should be extracted as clean claims without list markers
    factual = result.factual_claims
    assert len(factual) >= 3
    assert any("Earth orbits the Sun" in c.text for c in factual)
    assert any("Mars has two small moons" in c.text for c in factual)
    assert any("Jupiter is the largest planet" in c.text for c in factual)


def test_questions_and_conversational_meta_text(extractor):
    """
    Verifies that questions and conversational greetings are categorized as non-factual.
    """
    text = "Hello! Did you know about the Moon? Apollo 11 landed on July 20, 1969. Let me know if you have questions!"
    result = extractor.extract_claims(text)

    assert result.total_sentences >= 4
    
    # 'Hello!' -> CONVERSATIONAL
    # 'Did you know...?' -> QUESTION
    # 'Apollo 11 landed...' -> FACTUAL
    # 'Let me know...' -> CONVERSATIONAL
    claim_types = [c.claim_type for c in result.claims]
    assert "CONVERSATIONAL" in claim_types
    assert "QUESTION" in claim_types
    assert "NUMERICAL_ASSERTION" in claim_types or "FACTUAL_PROPOSITION" in claim_types
    assert result.factual_claims_count >= 1
    assert result.non_factual_count >= 2


def test_opinions_vs_factual_assertions(extractor):
    """
    Verifies that subjective opinions are tagged as SUBJECTIVE_OPINION rather than factual claims.
    """
    text = "In my opinion, Paris is the most beautiful city in the world. The Eiffel Tower was built in 1889."
    result = extractor.extract_claims(text)

    assert len(result.claims) == 2
    assert result.claims[0].is_factual is False
    assert result.claims[0].claim_type == "SUBJECTIVE_OPINION"

    assert result.claims[1].is_factual is True
    assert "The Eiffel Tower was built in 1889." in result.claims[1].text


def test_character_offset_traceability(extractor):
    """
    Verifies that start_char and end_char accurately map back to the original text.
    """
    text = "The Eiffel Tower is in Paris. It was completed in 1889. It is 500 metres tall."
    result = extractor.extract_claims(text)

    for claim in result.claims:
        extracted_from_slice = text[claim.start_char:claim.end_char]
        # The slice must match or contain the source sentence
        assert claim.source_sentence in extracted_from_slice or extracted_from_slice in claim.source_sentence
        assert claim.start_char >= 0
        assert claim.end_char <= len(text)
        assert claim.start_char < claim.end_char


def test_empty_and_whitespace_input(extractor):
    """
    Verifies robust handling of empty, whitespace-only, or None inputs.
    """
    res1 = extractor.extract_claims("")
    assert res1.claims == []
    assert res1.factual_claims_count == 0

    res2 = extractor.extract_claims("   \n\t  ")
    assert res2.claims == []

    res3 = extractor.extract_claims(None)
    assert res3.claims == []


def test_rhetorical_question_embedded_fact(extractor):
    """
    Verifies that conversational rhetorical questions embedding hard facts
    are unwrapped into factual assertions, while genuine questions remain non-factual.
    """
    text = (
        "Did you know the Louvre houses over 380,000 objects? "
        "Where is the Mona Lisa displayed? "
        "Have you heard that Apollo 11 landed on the Moon in 1969?"
    )
    result = extractor.extract_claims(text)

    # Claim 1: Unwrapped from 'Did you know the Louvre houses over 380,000 objects?'
    c1 = result.claims[0]
    assert c1.is_factual is True
    assert c1.text == "The Louvre houses over 380,000 objects."
    assert c1.claim_type == "NUMERICAL_ASSERTION"
    assert c1.offset_type == "EMBEDDED_CLAUSE"
    # Verify exact claim slice matches the original embedded clause in raw text
    assert text[c1.claim_start_char:c1.claim_end_char] == "the Louvre houses over 380,000 objects"

    # Claim 2: Genuine open question
    c2 = result.claims[1]
    assert c2.is_factual is False
    assert c2.claim_type == "QUESTION"
    assert "Where is the Mona Lisa" in c2.text

    # Claim 3: Unwrapped from 'Have you heard that Apollo 11 landed on the Moon in 1969?'
    c3 = result.claims[2]
    assert c3.is_factual is True
    assert "Apollo 11 landed on the Moon in 1969" in c3.text
    assert c3.offset_type == "EMBEDDED_CLAUSE"


def test_claim_level_offsets_exact_vs_clause(extractor):
    """
    Verifies claim-level character offsets for simple vs compound sentences.
    Ensures no offsets are invented and fidelity is maintained.
    """
    text = "The Eiffel Tower is in Paris. Marie Curie was born in Poland in 1867 and was the first woman to win a Nobel Prize."
    result = extractor.extract_claims(text)

    # Claim 1: Simple sentence verbatim -> EXACT
    c1 = result.claims[0]
    assert c1.offset_type == "EXACT"
    assert c1.claim_start_char == 0
    assert c1.claim_end_char == 29
    assert text[c1.claim_start_char:c1.claim_end_char] == "The Eiffel Tower is in Paris."

    # Claim 2: First clause of compound sentence -> EXACT
    c2 = result.claims[1]
    assert c2.offset_type == "EXACT"
    assert text[c2.claim_start_char:c2.claim_end_char] == "Marie Curie was born in Poland in 1867"

    # Claim 3: Second clause of compound sentence (with carried subject) -> CLAUSE_SPAN
    c3 = result.claims[2]
    assert c3.offset_type == "CLAUSE_SPAN"
    # The localized slice inside the raw text must match the original verbal predicate
    assert text[c3.claim_start_char:c3.claim_end_char] == "was the first woman to win a Nobel Prize."
    assert "Marie Curie was the first woman to win a Nobel Prize." in c3.text


def test_pleonastic_it_not_resolved(extractor):
    """
    Verifies that pleonastic / dummy 'it' (e.g. 'It is important to note...')
    is NOT replaced with an unrelated previous subject.
    """
    text = "The Eiffel Tower is in Paris. It is important to note that tickets must be booked early."
    result = extractor.extract_claims(text)

    assert len(result.claims) == 2
    # Claim 1 has subject 'The Eiffel Tower'
    assert "The Eiffel Tower" in result.claims[0].text

    # Claim 2 MUST NOT say 'The Eiffel Tower is important to note...'
    c2 = result.claims[1]
    assert not c2.text.startswith("The Eiffel Tower is important")
    assert "It is important to note" in c2.text


def test_complex_negation_neither_nor(extractor):
    """
    Verifies detection of complex negation patterns like 'neither ... nor'.
    """
    text = "Mars has neither liquid oceans nor a breathable atmosphere."
    result = extractor.extract_claims(text)

    assert len(result.claims) == 1
    assert result.claims[0].has_negation is True


def test_numerical_and_percentage_assertions(extractor):
    """
    Verifies identification of percentages and currency numbers.
    """
    text = "Company revenue grew by 25.5% in 2023, totaling $1.5 billion."
    result = extractor.extract_claims(text)

    assert len(result.claims) == 1
    assert result.claims[0].is_factual is True
    assert result.claims[0].claim_type == "NUMERICAL_ASSERTION"
    assert result.claims[0].has_numbers is True


