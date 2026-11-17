"""
VeriLLM Phase 3 Demonstration Script: Claim Extraction Engine.
Demonstrates atomic proposition decomposition, coreference resolution, and character offset traceability.
"""

import sys
import json
from pathlib import Path

# Ensure backend root is on sys.path
CURRENT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = CURRENT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml_engine.claim_extractor import ClaimExtractor

DEMO_RESPONSES = [
    {
        "title": "Example 1: Master Prompt Benchmark (Eiffel Tower)",
        "text": "The Eiffel Tower is in Paris. It was completed in 1889. It is 500 metres tall."
    },
    {
        "title": "Example 2: Compound Predicates & Conjunctions (Marie Curie)",
        "text": "Marie Curie was born in Poland in 1867 and was the first woman to win a Nobel Prize."
    },
    {
        "title": "Example 3: Structured Markdown List with Numbers & Dates",
        "text": (
            "Here are key facts about Apollo 11:\n"
            "1. Apollo 11 landed on the Moon on July 20, 1969.\n"
            "2. Neil Armstrong and Buzz Aldrin collected 47.5 pounds of lunar rock.\n"
            "3. Michael Collins did not walk on the Moon."
        )
    },
    {
        "title": "Example 4: Conversational Framing, Opinions, and Inquiries",
        "text": (
            "Hello! In my opinion, Paris is the most beautiful city in Europe. "
            "The Louvre Museum was established in 1793. "
            "Did you know it houses over 380,000 objects? Hope this helps!"
        )
    }
]


def run_demo():
    print("=" * 80)
    print(" VeriLLM Phase 3: Atomic Claim Extraction Demonstration")
    print("=" * 80)

    extractor = ClaimExtractor(
        resolve_coreference=True,
        decompose_compound=True,
        include_non_factual=True
    )

    all_demo_results = []

    for demo in DEMO_RESPONSES:
        print(f"\n>>> {demo['title']}")
        print(f"Original Text:\n  \"{demo['text']}\"\n")

        result = extractor.extract_claims(demo["text"])
        print(f"Total Sentences: {result.total_sentences} | Factual Claims: {result.factual_claims_count} | Non-Factual: {result.non_factual_count}")
        print("-" * 80)
        print(f"{'ID':<10} | {'Factual?':<9} | {'Type':<21} | Claim Text")
        print("-" * 80)

        serialized_claims = []
        for claim in result.claims:
            fact_icon = "YES" if claim.is_factual else "NO"
            print(f"{claim.claim_id:<10} | {fact_icon:<9} | {claim.claim_type:<21} | {claim.text}")
            if claim.named_entities:
                ents_str = ", ".join([f"{e['text']} ({e['label']})" for e in claim.named_entities])
                print(f"{'':<10}   Entities: {ents_str}")
            if claim.has_negation:
                print(f"{'':<10}   [Negation detected]")
            print(f"{'':<10}   Offsets: sentence [{claim.start_char}:{claim.end_char}] | claim-level [{claim.claim_start_char}:{claim.claim_end_char}] ({claim.offset_type})")
            if claim.claim_start_char is not None and claim.claim_end_char is not None and claim.claim_start_char < len(demo["text"]):
                exact_slice = demo["text"][claim.claim_start_char:claim.claim_end_char]
                print(f"{'':<10}   Raw slice in response: \"{exact_slice}\"")
            print(f"{'':<10}   Source sentence: \"{claim.source_sentence}\"")
            print()

            serialized_claims.append(claim.model_dump())

        all_demo_results.append({
            "title": demo["title"],
            "raw_text": demo["text"],
            "total_sentences": result.total_sentences,
            "factual_count": result.factual_claims_count,
            "non_factual_count": result.non_factual_count,
            "claims": serialized_claims
        })

    # Save demo outputs to results directory
    results_dir = CURRENT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / "02_claim_extraction_demo_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_demo_results, f, indent=2)

    print("=" * 80)
    print(f"Demonstration complete. Structured results saved to:\n  {out_file}")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()

