"""
Phase 9 Step 9.5: Live End-to-End Validation Script.
Executes Cases A through F using the real VeriLLMPipeline with the fine-tuned
DeBERTa checkpoint and frozen Stage 1 calibration bundle.
Logs comprehensive diagnostics to verify system integrity before backend closure.
"""

import json
import os
import sys
import time
from pathlib import Path

# Ensure backend root is on sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Force offline execution
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from ml_engine.pipeline import VeriLLMPipeline, PipelineResult
from ml_engine.claim_verifier.models import VerificationCategory


def run_e2e_validation():
    print("=" * 80)
    print(" VeriLLM Phase 9 — Step 9.5: Live End-to-End Pipeline Validation")
    print("=" * 80)

    # Initialize live pipeline (uses real DeBERTa model and frozen bundle)
    print("\n[Init] Initializing production VeriLLMPipeline with local model & frozen bundle...")
    t0 = time.perf_counter()
    pipeline = VeriLLMPipeline(default_regime="retrieval")
    init_duration = time.perf_counter() - t0
    print(f"  -> Pipeline initialized in {init_duration:.3f}s (lazy loading active).")

    results_log = {}

    # -------------------------------------------------------------------------
    # Case A: Text document + factual response (Direct mode)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Case A: Text document + factual response (Direct mode)")
    print("-" * 70)
    doc_a = (
        "The Apollo 11 mission was launched on a Saturn V rocket from Kennedy Space Center on July 16, 1969. "
        "Commander Neil Armstrong and Lunar Module Pilot Buzz Aldrin landed the Apollo Lunar Module Eagle on the Moon on July 20, 1969."
    )
    resp_a = (
        "Apollo 11 was launched on a Saturn V rocket. "
        "The astronauts had a pizza party on the Moon."
    )
    res_a = pipeline.verify(document=doc_a, response=resp_a, regime="direct", document_name="apollo_mission.txt")
    print(f"Document: {res_a.document_name} ({res_a.metadata['document']['total_words']} words)")
    print(f"Regime: {res_a.regime} | Total Claims: {len(res_a.claims)} | Duration: {res_a.execution_time_seconds:.3f}s")
    for c in res_a.claims:
        print(f"  - [{c.classification.value}] {c.claim_text}")
        print(f"    Rule: {c.decision_rule}")
        print(f"    Trace: {c.reasoning_trace[:120]}...")
    print(f"Score Metrics: ESR={res_a.score.rates.evidence_support_rate}%, CCR={res_a.score.rates.confirmed_contradiction_rate}%, HSI={res_a.score.indices.hallucination_severity_index}%")
    assert len(res_a.claims) >= 2
    assert res_a.score.has_factual_claims is True

    results_log["Case_A"] = {
        "status": "PASS",
        "claims_count": len(res_a.claims),
        "classifications": [c.classification.value for c in res_a.claims],
        "esr": res_a.score.rates.evidence_support_rate,
        "ccr": res_a.score.rates.confirmed_contradiction_rate,
        "duration": res_a.execution_time_seconds,
    }

    # -------------------------------------------------------------------------
    # Case B: Text document + contradicted claim (Direct mode)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Case B: Text document + contradicted claim (Direct mode)")
    print("-" * 70)
    doc_b = "The speed of light in vacuum is approximately 299,792 kilometers per second."
    resp_b = "The speed of light in a vacuum is 100 kilometers per hour."
    res_b = pipeline.verify(document=doc_b, response=resp_b, regime="direct", document_name="physics_constants.txt")
    print(f"Document: {res_b.document_name}")
    print(f"Regime: {res_b.regime} | Total Claims: {len(res_b.claims)} | Duration: {res_b.execution_time_seconds:.3f}s")
    for c in res_b.claims:
        print(f"  - [{c.classification.value}] {c.claim_text}")
        print(f"    Rule: {c.decision_rule}")
        print(f"    Trace: {c.reasoning_trace[:140]}...")
    print(f"Score Metrics: CCR={res_b.score.rates.confirmed_contradiction_rate}%, HSI={res_b.score.indices.hallucination_severity_index}%")
    assert len(res_b.claims) >= 1

    results_log["Case_B"] = {
        "status": "PASS",
        "claims_count": len(res_b.claims),
        "classifications": [c.classification.value for c in res_b.claims],
        "hsi": res_b.score.indices.hallucination_severity_index,
        "duration": res_b.execution_time_seconds,
    }

    # -------------------------------------------------------------------------
    # Case C: Retrieval mode with multi-passage document
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Case C: Retrieval mode with multi-passage document")
    print("-" * 70)
    doc_c = (
        "The James Webb Space Telescope (JWST) is an infrared space observatory launched on December 25, 2021. "
        "It orbits around the Sun-Earth L2 Lagrange point 1.5 million kilometers away from Earth. "
        "JWST features a primary mirror composed of 18 hexagonal beryllium segments coated in gold. "
        "Its primary science goals include studying the first stars and galaxies that formed after the Big Bang.\n\n"
        "The Hubble Space Telescope was launched into low Earth orbit in April 1990 by Space Shuttle Discovery. "
        "Hubble has made more than 1.5 million observations during its lifetime and orbits at approximately 540 kilometers altitude. "
        "It observes primarily in the ultraviolet, visible, and near-infrared spectra with exceptional angular resolution.\n\n"
        "The Chandra X-ray Observatory is a NASA Flagship-class telescope launched on July 23, 1999, aboard Space Shuttle Columbia. "
        "It is designed to detect X-ray emissions from extremely hot and energetic regions of the universe. "
        "Chandra orbits the Earth in a highly elliptical orbit extending more than one-third of the distance to the Moon."
    )
    resp_c = (
        "The James Webb Space Telescope was launched in December 2021. "
        "Chandra is a telescope designed to observe ultraviolet light."
    )
    res_c = pipeline.verify(document=doc_c, response=resp_c, regime="retrieval", document_name="great_observatories.txt")
    print(f"Document: {res_c.document_name} | Indexed Chunks: {res_c.metadata['indexed_chunks_count']}")
    print(f"Regime: {res_c.regime} | Total Claims: {len(res_c.claims)} | Duration: {res_c.execution_time_seconds:.3f}s")
    for c in res_c.claims:
        print(f"  - [{c.classification.value}] {c.claim_text}")
        if c.primary_evidence:
            print(f"    Top Match: sim={c.primary_evidence.similarity_score:.4f} in {c.primary_evidence.chunk.chunk_id}")
            print(f"    Evidence Text: {c.primary_evidence.chunk.text[:90]}...")
        print(f"    Trace: {c.reasoning_trace[:120]}...")
    print(f"Direct Context Truncated: {res_c.metadata['direct_context_truncated']} (must be False)")
    print(f"Warnings: {res_c.warnings}")
    assert res_c.metadata["direct_context_truncated"] is False
    assert res_c.metadata["indexed_chunks_count"] >= 2
    assert not any("512-token limit" in w for w in res_c.warnings)

    results_log["Case_C"] = {
        "status": "PASS",
        "chunks_indexed": res_c.metadata["indexed_chunks_count"],
        "classifications": [c.classification.value for c in res_c.claims],
        "direct_truncated": res_c.metadata["direct_context_truncated"],
        "duration": res_c.execution_time_seconds,
    }

    # -------------------------------------------------------------------------
    # Case D: Non-factual response
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Case D: Non-factual response (greetings and subjective opinions)")
    print("-" * 70)
    doc_d = "The solar system consists of eight planets and numerous dwarf planets orbiting the Sun."
    resp_d = "Hello! In my opinion, the cosmos is beautiful. Personally, I feel great about astronomy. Hope this helps!"
    res_d = pipeline.verify(document=doc_d, response=resp_d, regime="retrieval", document_name="solar_system.txt")
    print(f"Total Claims: {len(res_d.claims)} | Factual: {res_d.score.counts.factual_claims} | Non-factual: {res_d.score.counts.non_factual_count}")
    print(f"Has Factual Claims: {res_d.score.has_factual_claims} (must be False)")
    print(f"ESR: {res_d.score.rates.evidence_support_rate} (must be None)")
    print(f"Warnings: {res_d.warnings}")
    assert res_d.score.has_factual_claims is False
    assert res_d.score.counts.factual_claims == 0
    assert res_d.score.counts.non_factual_count > 0
    assert any("no empirical factual claims" in w for w in res_d.warnings)

    results_log["Case_D"] = {
        "status": "PASS",
        "has_factual_claims": res_d.score.has_factual_claims,
        "non_factual_count": res_d.score.counts.non_factual_count,
        "warning_present": any("no empirical factual claims" in w for w in res_d.warnings),
        "duration": res_d.execution_time_seconds,
    }

    # -------------------------------------------------------------------------
    # Case E: Insufficient or missing evidence (Retrieval mode abstention)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Case E: Insufficient / missing evidence (Retrieval floor abstention)")
    print("-" * 70)
    doc_e = "Traditional French bread requires wheat flour, clean water, baker's yeast, and salt."
    resp_e = "Quantum computers leverage superconducting qubits operating at cryogenic temperatures."
    res_e = pipeline.verify(document=doc_e, response=resp_e, regime="retrieval", document_name="bread_recipe.txt")
    print(f"Document: {res_e.document_name} | Response: \"{resp_e}\"")
    for c in res_e.claims:
        print(f"  - [{c.classification.value}] {c.claim_text}")
        if c.primary_evidence:
            print(f"    Similarity: {c.primary_evidence.similarity_score:.4f} (Floor is 0.20)")
        print(f"    Rule: {c.decision_rule}")
        print(f"    Trace: {c.reasoning_trace[:120]}...")
    print(f"HSI: {res_e.score.indices.hallucination_severity_index}% (Must be 0.0% - UNVERIFIABLE is not hallucination)")
    assert any(c.classification == VerificationCategory.UNVERIFIABLE for c in res_e.claims)
    assert res_e.score.indices.hallucination_severity_index == 0.0

    results_log["Case_E"] = {
        "status": "PASS",
        "classifications": [c.classification.value for c in res_e.claims],
        "hsi": res_e.score.indices.hallucination_severity_index,
        "unverifiable_count": res_e.score.counts.unverifiable_count,
        "duration": res_e.execution_time_seconds,
    }

    # -------------------------------------------------------------------------
    # Case F: Direct mode with long document (Truncation detection)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Case F: Direct mode with long document (Truncation detection)")
    print("-" * 70)
    sentences_f = [
        f"Astronomical catalog item {i} describes galaxy cluster Abell-{i:03d} located at redshift z={0.05 * i:.3f}."
        for i in range(1, 45)
    ]
    long_doc_f = " ".join(sentences_f)
    resp_f = "Galaxy cluster Abell-001 is located at redshift z=0.050."
    res_f = pipeline.verify(document=long_doc_f, response=resp_f, regime="direct", document_name="galaxy_catalog.txt")
    print(f"Document Word Count: {len(long_doc_f.split())} words")
    print(f"Direct Context Truncated: {res_f.metadata['direct_context_truncated']} (must be True)")
    print(f"Recommended Regime: {res_f.metadata['recommended_regime']} (must be 'retrieval')")
    print(f"Warnings:")
    for w in res_f.warnings:
        print(f"  * {w}")
    assert res_f.metadata["direct_context_truncated"] is True
    assert res_f.metadata["recommended_regime"] == "retrieval"
    assert any("512-token limit" in w for w in res_f.warnings)

    results_log["Case_F"] = {
        "status": "PASS",
        "direct_truncated": res_f.metadata["direct_context_truncated"],
        "recommended_regime": res_f.metadata["recommended_regime"],
        "warnings_count": len(res_f.warnings),
        "duration": res_f.execution_time_seconds,
    }

    print("\n" + "=" * 80)
    print(" All Six End-to-End Test Cases Completed Successfully!")
    print("=" * 80)
    print(json.dumps(results_log, indent=2))
    return results_log


if __name__ == "__main__":
    run_e2e_validation()

