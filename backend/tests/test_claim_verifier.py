"""
VeriLLM Phase 5 Unit Tests: Claim Verification Pipeline.
Verifies MiniCheck integration, 3-way NLI cross-encoder, 4-way classification,
conflict resolution, empty evidence, and factuality metric calculations.
"""

import os
import sys
from pathlib import Path
import pytest

# Ensure backend root is on sys.path
CURRENT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = CURRENT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Ensure offline mode is active for test stability
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from ml_engine.claim_extractor.models import ExtractedClaim
from ml_engine.evidence_retriever.models import (
    DocumentChunk,
    RetrievedPassage,
    ClaimEvidenceMatch
)
from ml_engine.claim_verifier.models import VerificationCategory
from ml_engine.claim_verifier.decision_strategy import VerificationDecisionStrategy
from ml_engine.claim_verifier.verifier import ClaimVerifier


APOLLO_REFERENCE_TEXT = (
    "The Apollo 11 mission was launched by a Saturn V rocket from Kennedy Space Center on Merritt Island, "
    "Florida, on July 16, 1969. Commander Neil Armstrong and Lunar Module Pilot Buzz Aldrin landed Eagle "
    "in the Sea of Tranquility on July 20, 1969. Neil Armstrong became the first human to step onto the lunar "
    "surface on July 21, 1969, and Buzz Aldrin joined him 19 minutes later. They collected 47.5 pounds of lunar rock. "
    "Meanwhile, Command Module Pilot Michael Collins flew the Command Module Columbia alone in lunar orbit. "
    "The astronauts safely returned to Earth and splashed down in the Pacific Ocean on July 24, 1969."
)

SAMPLE_CHUNK = DocumentChunk(
    chunk_id="apollo11_p02_c001",
    document_name="apollo11_report.pdf",
    page_number=2,
    chunk_index=1,
    text=APOLLO_REFERENCE_TEXT,
    start_char=0,
    end_char=len(APOLLO_REFERENCE_TEXT),
    word_count=len(APOLLO_REFERENCE_TEXT.split())
)


@pytest.fixture(scope="module")
def verifier():
    """Initializes the shared ClaimVerifier pipeline once for the test module."""
    return ClaimVerifier(
        minicheck_model="roberta-large",
        nli_model="cross-encoder/nli-deberta-v3-xsmall"
    )


def test_supported_claim_verification(verifier):
    """Verifies that an entailed factual claim is classified as SUPPORTED."""
    claim = ExtractedClaim(
        claim_id="claim_001",
        text="Neil Armstrong and Buzz Aldrin collected 47.5 pounds of lunar rock.",
        source_sentence="Neil Armstrong and Buzz Aldrin collected 47.5 pounds of lunar rock.",
        start_char=0,
        end_char=68,
        claim_start_char=0,
        claim_end_char=68,
        offset_type="EXACT",
        is_factual=True
    )
    passage = RetrievedPassage(chunk=SAMPLE_CHUNK, similarity_score=0.78, rank=1)
    match = ClaimEvidenceMatch(
        claim_id="claim_001",
        claim_text=claim.text,
        retrieved_passages=[passage],
        best_passage=passage,
        best_similarity_score=0.78
    )

    report = verifier.verify_claims([claim], [match], document_name="apollo11_report.pdf")

    assert len(report.claims) == 1
    vc = report.claims[0]
    assert vc.classification == VerificationCategory.SUPPORTED
    assert vc.scores.minicheck_prediction == 1
    assert vc.scores.minicheck_probability > 0.80
    assert vc.scores.nli_contradiction_prob < 0.20
    assert "apollo11_report.pdf" in vc.reasoning_trace
    assert "Page 2" in vc.reasoning_trace


def test_contradicted_claim_verification(verifier):
    """Verifies that a directly refuted claim is classified as CONTRADICTED."""
    claim = ExtractedClaim(
        claim_id="claim_002",
        text="Michael Collins walked on the Moon with Armstrong.",
        source_sentence="Michael Collins walked on the Moon with Armstrong.",
        start_char=0,
        end_char=50,
        claim_start_char=0,
        claim_end_char=50,
        offset_type="EXACT",
        is_factual=True
    )
    passage = RetrievedPassage(chunk=SAMPLE_CHUNK, similarity_score=0.68, rank=1)
    match = ClaimEvidenceMatch(
        claim_id="claim_002",
        claim_text=claim.text,
        retrieved_passages=[passage],
        best_passage=passage,
        best_similarity_score=0.68
    )

    report = verifier.verify_claims([claim], [match], document_name="apollo11_report.pdf")

    assert len(report.claims) == 1
    vc = report.claims[0]
    assert vc.classification == VerificationCategory.CONTRADICTED
    assert vc.scores.minicheck_prediction == 0
    assert vc.scores.nli_contradiction_prob > 0.85
    assert vc.decision_rule == "EVIDENCE_CONTRADICTION"
    assert "directly refutes" in vc.reasoning_trace


def test_unsupported_claim_mars_hallucination(verifier):
    """Verifies that an unmentioned claim is classified as UNSUPPORTED, not CONTRADICTED."""
    claim = ExtractedClaim(
        claim_id="claim_003",
        text="Buzz Aldrin also planted an olive branch on Mars.",
        source_sentence="Buzz Aldrin also planted an olive branch on Mars.",
        start_char=0,
        end_char=49,
        claim_start_char=0,
        claim_end_char=49,
        offset_type="EXACT",
        is_factual=True
    )
    # Retrieved chunk mentions Buzz Aldrin, giving moderate similarity, but no Mars information
    passage = RetrievedPassage(chunk=SAMPLE_CHUNK, similarity_score=0.52, rank=1)
    match = ClaimEvidenceMatch(
        claim_id="claim_003",
        claim_text=claim.text,
        retrieved_passages=[passage],
        best_passage=passage,
        best_similarity_score=0.52
    )

    report = verifier.verify_claims([claim], [match], document_name="apollo11_report.pdf")

    assert len(report.claims) == 1
    vc = report.claims[0]
    # Critical test: MiniCheck is 0, but NLI shows neutral -> Must be UNSUPPORTED
    assert vc.classification == VerificationCategory.UNSUPPORTED
    assert vc.scores.minicheck_prediction == 0
    assert vc.scores.nli_neutral_prob > 0.70
    assert vc.decision_rule == "EVIDENCE_UNMENTIONED_NEUTRAL"
    assert "does not contain factual assertions confirming or refuting" in vc.reasoning_trace


def test_unverifiable_empty_evidence(verifier):
    """Verifies that a claim with zero retrieved evidence is classified as UNVERIFIABLE."""
    claim = ExtractedClaim(
        claim_id="claim_004",
        text="The Saturn V rocket had five F-1 engines.",
        source_sentence="The Saturn V rocket had five F-1 engines.",
        start_char=0,
        end_char=41,
        is_factual=True
    )
    # Empty match: no passages found
    match = ClaimEvidenceMatch(
        claim_id="claim_004",
        claim_text=claim.text,
        retrieved_passages=[],
        best_passage=None,
        best_similarity_score=0.0
    )

    report = verifier.verify_claims([claim], [match])

    assert len(report.claims) == 1
    vc = report.claims[0]
    assert vc.classification == VerificationCategory.UNVERIFIABLE
    assert vc.decision_rule == "EMPTY_EVIDENCE"
    assert vc.scores.minicheck_prediction is None  # Never wasted CPU inference on empty evidence


def test_unverifiable_low_similarity_floor(verifier):
    """Verifies that a claim whose evidence is below the similarity floor is UNVERIFIABLE."""
    claim = ExtractedClaim(
        claim_id="claim_005",
        text="A chocolate chip cookie recipe requires butter and vanilla extract.",
        source_sentence="A chocolate chip cookie recipe requires butter and vanilla extract.",
        start_char=0,
        end_char=68,
        is_factual=True
    )
    # Distractor chunk with low cosine similarity (<0.25)
    passage = RetrievedPassage(chunk=SAMPLE_CHUNK, similarity_score=0.14, rank=1)
    match = ClaimEvidenceMatch(
        claim_id="claim_005",
        claim_text=claim.text,
        retrieved_passages=[passage],
        best_passage=passage,
        best_similarity_score=0.14
    )

    report = verifier.verify_claims([claim], [match])

    assert len(report.claims) == 1
    vc = report.claims[0]
    assert vc.classification == VerificationCategory.UNVERIFIABLE
    assert vc.decision_rule == "EVIDENCE_SIMILARITY_BELOW_FLOOR"
    assert "low semantic similarity" in vc.reasoning_trace


def test_unverifiable_non_factual_claim(verifier):
    """Verifies that non-factual assertions (opinions, greetings) are classified as NON_FACTUAL."""
    claim = ExtractedClaim(
        claim_id="claim_006",
        text="In my opinion, Apollo 11 was the greatest achievement in human history.",
        source_sentence="In my opinion, Apollo 11 was the greatest achievement in human history.",
        start_char=0,
        end_char=71,
        is_factual=False  # Tagged as non-factual by ClaimExtractor
    )
    passage = RetrievedPassage(chunk=SAMPLE_CHUNK, similarity_score=0.65, rank=1)
    match = ClaimEvidenceMatch(
        claim_id="claim_006",
        claim_text=claim.text,
        retrieved_passages=[passage],
        best_passage=passage,
        best_similarity_score=0.65
    )

    report = verifier.verify_claims([claim], [match])

    assert len(report.claims) == 1
    vc = report.claims[0]
    assert vc.classification == VerificationCategory.NON_FACTUAL
    assert vc.is_factual is False
    assert vc.decision_rule == "NON_FACTUAL_ASSERTION"
    assert "subjective, opinion-based" in vc.reasoning_trace


def test_conflicting_evidence_passages():
    """Verifies that mutually contradictory passages trigger UNVERIFIABLE."""
    strategy = VerificationDecisionStrategy(conflict_threshold=0.70)
    passage = RetrievedPassage(chunk=SAMPLE_CHUNK, similarity_score=0.60, rank=1)

    # Simulate two passages: one entails (p_entail=0.85), one contradicts (p_contra=0.88)
    verified = strategy.evaluate_claim(
        claim_id="claim_007",
        claim_text="The spacecraft landed at 20:17 UTC.",
        source_sentence="The spacecraft landed at 20:17 UTC.",
        start_char=0,
        end_char=35,
        claim_start_char=0,
        claim_end_char=35,
        offset_type="EXACT",
        is_factual=True,
        primary_evidence=passage,
        all_evidence=[passage],
        minicheck_score=(0, 0.50),
        nli_scores=(0.85, 0.05, 0.10),
        secondary_nli_scores=[(0.85, 0.05, 0.10), (0.05, 0.88, 0.07)]  # Strong conflict!
    )

    assert verified.classification == VerificationCategory.UNVERIFIABLE
    assert verified.decision_rule == "CONFLICTING_EVIDENCE_PASSAGES"
    assert "directly conflicting assertions" in verified.reasoning_trace


def test_factuality_metrics_calculation(verifier):
    """Verifies that factuality metrics compute exact percentages and respect denominators."""
    c1 = ExtractedClaim(claim_id="c1", text="Neil Armstrong and Buzz Aldrin collected 47.5 pounds of lunar rock.", source_sentence="S1", start_char=0, end_char=68, is_factual=True)
    c2 = ExtractedClaim(claim_id="c2", text="The Apollo 11 mission was launched by a Saturn V rocket.", source_sentence="S2", start_char=69, end_char=125, is_factual=True)
    c3 = ExtractedClaim(claim_id="c3", text="Michael Collins walked on the Moon with Armstrong.", source_sentence="S3", start_char=126, end_char=176, is_factual=True)
    c4 = ExtractedClaim(claim_id="c4", text="Buzz Aldrin also planted an olive branch on Mars.", source_sentence="S4", start_char=177, end_char=226, is_factual=True)
    c5 = ExtractedClaim(claim_id="c5", text="In my opinion, space exploration is fascinating.", source_sentence="S5", start_char=227, end_char=275, is_factual=False)

    p_support = RetrievedPassage(chunk=SAMPLE_CHUNK, similarity_score=0.75, rank=1)
    
    # Custom matches matching the factual propositions
    matches = [
        ClaimEvidenceMatch(claim_id="c1", claim_text=c1.text, retrieved_passages=[p_support], best_passage=p_support, best_similarity_score=0.75),
        ClaimEvidenceMatch(claim_id="c2", claim_text=c2.text, retrieved_passages=[p_support], best_passage=p_support, best_similarity_score=0.75),
        ClaimEvidenceMatch(claim_id="c3", claim_text=c3.text, retrieved_passages=[p_support], best_passage=p_support, best_similarity_score=0.75),
        ClaimEvidenceMatch(claim_id="c4", claim_text=c4.text, retrieved_passages=[p_support], best_passage=p_support, best_similarity_score=0.55),
        ClaimEvidenceMatch(claim_id="c5", claim_text=c5.text, retrieved_passages=[], best_passage=None, best_similarity_score=0.0)
    ]

    report = verifier.verify_claims([c1, c2, c3, c4, c5], matches, document_name="apollo.pdf")
    m = report.metrics

    assert m.total_claims == 5
    assert m.factual_claims == 4  # c5 excluded from factual denominator
    assert m.non_factual_claims == 1
    assert m.supported_count == 2
    assert m.contradicted_count == 1
    assert m.unsupported_count == 1
    assert m.unverifiable_count == 0
    assert m.non_factual_count == 1

    # Over 4 factual claims:
    # 2 supported = 50.0%
    # 1 contradicted = 25.0%
    # 1 unsupported = 25.0%
    # 0 unverifiable factual = 0.0%
    # Uncertainty rate = (unsupported + unverifiable) / 4 = 25.0%
    assert m.evidence_support_rate == 50.0
    assert m.confirmed_contradiction_rate == 25.0
    assert m.unsupported_rate == 25.0
    assert m.unverifiable_rate == 0.0
    assert m.uncertainty_rate == 25.0
    # Invariant: rates sum to exactly 100.0%
    assert round(m.evidence_support_rate + m.confirmed_contradiction_rate + m.unsupported_rate + m.unverifiable_rate, 2) == 100.0
    assert round(m.evidence_support_rate + m.confirmed_contradiction_rate + m.uncertainty_rate, 2) == 100.0
    assert "evaluated over 4 factual claims" in m.denominator_description


def test_response_with_only_subjective_claims(verifier):
    """Verifies that a response with ONLY subjective non-factual claims reports undefined (None) metrics."""
    c1 = ExtractedClaim(claim_id="op_1", text="In my opinion, Apollo 11 was inspiring.", source_sentence="S1", start_char=0, end_char=40, is_factual=False)
    c2 = ExtractedClaim(claim_id="op_2", text="I think space travel is cool.", source_sentence="S2", start_char=41, end_char=70, is_factual=False)

    report = verifier.verify_claims([c1, c2], [])
    m = report.metrics

    assert m.total_claims == 2
    assert m.factual_claims == 0
    assert m.non_factual_claims == 2
    assert m.has_factual_claims is False
    assert m.supported_count == 0
    assert m.contradicted_count == 0
    assert m.unsupported_count == 0
    assert m.unverifiable_count == 0
    assert m.non_factual_count == 2

    # All rates must be None, NOT misleadingly 0%
    assert m.evidence_support_rate is None
    assert m.confirmed_contradiction_rate is None
    assert m.unsupported_rate is None
    assert m.unverifiable_rate is None
    assert m.uncertainty_rate is None
    assert "No empirical factual claims to evaluate" in m.denominator_description
    assert "N/A" in m.denominator_description


def test_response_with_zero_claims(verifier):
    """Verifies that an empty response with zero claims returns None metrics gracefully."""
    report = verifier.verify_claims([], [])
    m = report.metrics

    assert m.total_claims == 0
    assert m.factual_claims == 0
    assert m.non_factual_claims == 0
    assert m.has_factual_claims is False
    assert m.evidence_support_rate is None
    assert m.confirmed_contradiction_rate is None
    assert m.unsupported_rate is None
    assert m.unverifiable_rate is None
    assert m.uncertainty_rate is None
    assert "No claims to evaluate" in m.denominator_description


def test_response_with_all_four_factual_categories(verifier):
    """
    Verifies metric accounting when factual claims exist in ALL four categories:
    SUPPORTED, CONTRADICTED, UNSUPPORTED, UNVERIFIABLE.
    """
    c_supp = ExtractedClaim(claim_id="f1", text="Neil Armstrong and Buzz Aldrin collected 47.5 pounds of lunar rock.", source_sentence="S1", start_char=0, end_char=68, is_factual=True)
    c_contra = ExtractedClaim(claim_id="f2", text="Michael Collins walked on the Moon with Armstrong.", source_sentence="S2", start_char=69, end_char=119, is_factual=True)
    c_unsup = ExtractedClaim(claim_id="f3", text="Buzz Aldrin also planted an olive branch on Mars.", source_sentence="S3", start_char=120, end_char=169, is_factual=True)
    c_unver = ExtractedClaim(claim_id="f4", text="The Apollo spacecraft carried a secret golden key.", source_sentence="S4", start_char=170, end_char=220, is_factual=True)

    p_support = RetrievedPassage(chunk=SAMPLE_CHUNK, similarity_score=0.75, rank=1)
    p_low = RetrievedPassage(chunk=SAMPLE_CHUNK, similarity_score=0.10, rank=1)  # Below floor -> UNVERIFIABLE

    matches = [
        ClaimEvidenceMatch(claim_id="f1", claim_text=c_supp.text, retrieved_passages=[p_support], best_passage=p_support, best_similarity_score=0.75),
        ClaimEvidenceMatch(claim_id="f2", claim_text=c_contra.text, retrieved_passages=[p_support], best_passage=p_support, best_similarity_score=0.75),
        ClaimEvidenceMatch(claim_id="f3", claim_text=c_unsup.text, retrieved_passages=[p_support], best_passage=p_support, best_similarity_score=0.55),
        ClaimEvidenceMatch(claim_id="f4", claim_text=c_unver.text, retrieved_passages=[p_low], best_passage=p_low, best_similarity_score=0.10)
    ]

    report = verifier.verify_claims([c_supp, c_contra, c_unsup, c_unver], matches)
    m = report.metrics

    assert m.total_claims == 4
    assert m.factual_claims == 4
    assert m.non_factual_claims == 0
    assert m.has_factual_claims is True

    # 1 in each of the 4 categories
    assert m.supported_count == 1
    assert m.contradicted_count == 1
    assert m.unsupported_count == 1
    assert m.unverifiable_count == 1

    # Exact 25.0% for each
    assert m.evidence_support_rate == 25.0
    assert m.confirmed_contradiction_rate == 25.0
    assert m.unsupported_rate == 25.0
    assert m.unverifiable_rate == 25.0

    # Invariants
    # 1. Four categories sum to 100.0%
    assert round(m.evidence_support_rate + m.confirmed_contradiction_rate + m.unsupported_rate + m.unverifiable_rate, 2) == 100.0
    # 2. Combined uncertainty = unsupported + unverifiable = 50.0%
    assert m.uncertainty_rate == 50.0
    # 3. Support + Contradiction + Uncertainty = 100.0%
    assert round(m.evidence_support_rate + m.confirmed_contradiction_rate + m.uncertainty_rate, 2) == 100.0


def test_response_with_factual_and_non_factual_claims_invariants(verifier):
    """
    Verifies that non-factual claims do not contaminate factual rates,
    and mathematical invariants hold strictly:
    support_rate + contradiction_rate + unsupported_rate + unverifiable_rate == 100.0%
    """
    f1 = ExtractedClaim(claim_id="f1", text="The Apollo 11 mission was launched by a Saturn V rocket.", source_sentence="S1", start_char=0, end_char=56, is_factual=True)
    f2 = ExtractedClaim(claim_id="f2", text="Michael Collins walked on the Moon with Armstrong.", source_sentence="S2", start_char=57, end_char=107, is_factual=True)
    nf1 = ExtractedClaim(claim_id="nf1", text="Hello and welcome to my space presentation!", source_sentence="S3", start_char=108, end_char=151, is_factual=False)
    nf2 = ExtractedClaim(claim_id="nf2", text="In my opinion, NASA is amazing.", source_sentence="S4", start_char=152, end_char=183, is_factual=False)

    p_support = RetrievedPassage(chunk=SAMPLE_CHUNK, similarity_score=0.75, rank=1)

    matches = [
        ClaimEvidenceMatch(claim_id="f1", claim_text=f1.text, retrieved_passages=[p_support], best_passage=p_support, best_similarity_score=0.75),
        ClaimEvidenceMatch(claim_id="f2", claim_text=f2.text, retrieved_passages=[p_support], best_passage=p_support, best_similarity_score=0.75),
        ClaimEvidenceMatch(claim_id="nf1", claim_text=nf1.text, retrieved_passages=[], best_passage=None, best_similarity_score=0.0),
        ClaimEvidenceMatch(claim_id="nf2", claim_text=nf2.text, retrieved_passages=[], best_passage=None, best_similarity_score=0.0)
    ]

    report = verifier.verify_claims([f1, f2, nf1, nf2], matches)
    m = report.metrics

    assert m.total_claims == 4
    assert m.factual_claims == 2
    assert m.non_factual_claims == 2
    assert m.non_factual_count == 2
    assert m.supported_count == 1
    assert m.contradicted_count == 1
    assert m.unsupported_count == 0
    assert m.unverifiable_count == 0

    # Over 2 factual claims: 1 supported (50%), 1 contradicted (50%)
    assert m.evidence_support_rate == 50.0
    assert m.confirmed_contradiction_rate == 50.0
    assert m.unsupported_rate == 0.0
    assert m.unverifiable_rate == 0.0
    assert m.uncertainty_rate == 0.0

    # Invariants
    assert round(m.evidence_support_rate + m.confirmed_contradiction_rate + m.unsupported_rate + m.unverifiable_rate, 2) == 100.0
    assert round(m.evidence_support_rate + m.confirmed_contradiction_rate + m.uncertainty_rate, 2) == 100.0

