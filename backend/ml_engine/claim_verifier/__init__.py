"""
VeriLLM Claim Verification Engine.
Provides 4-way factuality verification (SUPPORTED, CONTRADICTED, UNSUPPORTED, UNVERIFIABLE)
by synthesizing MiniCheck entailment, 3-way NLI cross-encoder scores, and retrieval provenance.
"""

from .models import (
    VerificationCategory,
    ModelVerificationScore,
    VerifiedClaim,
    VerificationMetrics,
    VerificationReport
)
from .minicheck_verifier import MiniCheckVerifier
from .nli_verifier import NLIVerifier
from .custom_deberta_verifier import CustomDebertaPrediction, CustomDebertaVerifier
from .decision_strategy import VerificationDecisionStrategy
from .verifier import ClaimVerifier
from .calibrated_verifier import (
    CalibratedDebertaVerifier,
    CalibratedRegimeConfig,
    load_frozen_calibration_bundle,
)

__all__ = [
    "VerificationCategory",
    "ModelVerificationScore",
    "VerifiedClaim",
    "VerificationMetrics",
    "VerificationReport",
    "MiniCheckVerifier",
    "NLIVerifier",
    "CustomDebertaPrediction",
    "CustomDebertaVerifier",
    "VerificationDecisionStrategy",
    "ClaimVerifier",
    "CalibratedDebertaVerifier",
    "CalibratedRegimeConfig",
    "load_frozen_calibration_bundle",
]

