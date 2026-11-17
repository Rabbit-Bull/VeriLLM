"""
VeriLLM Production Calibrated DeBERTa Verifier.

Implements the production-ready factual verification engine using the fine-tuned
RAGTruth DeBERTa backbone (verillm_deberta) and frozen C2 prior-bias + temperature
calibration parameters loaded strictly from 10_stage1_frozen_bundle.json.

Supports two operational regimes:
  1. direct: For full context / direct documents with frozen Direct calibration and thresholds.
  2. retrieval: For chunked evidence with frozen Retrieval calibration and similarity-floor gating.

Zero refitting, zero retraining, zero network dependencies.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np

from ml_engine.claim_extractor.models import ExtractedClaim
from ml_engine.evidence_retriever.models import ClaimEvidenceMatch, RetrievedPassage
from ml_engine.claim_verifier.models import (
    ModelVerificationScore,
    VerificationCategory,
    VerificationMetrics,
    VerificationReport,
    VerifiedClaim,
)
from ml_engine.claim_verifier.custom_deberta_verifier import (
    CustomDebertaPrediction,
    CustomDebertaVerifier,
)
from ml_engine.custom_model.calibration import (
    MulticlassCalibrator,
    reconstruct_logits_from_probs,
)
from ml_engine.custom_model.hybrid_strategies import _predict_h2_codes

logger = logging.getLogger(__name__)

DEFAULT_FROZEN_BUNDLE_PATH: Path = (
    Path(__file__).resolve().parent.parent.parent
    / "experiments"
    / "results"
    / "10_stage1_frozen_bundle.json"
)

RegimeName = Literal["direct", "retrieval"]
SUPPORTED_REGIMES: Tuple[str, ...] = ("direct", "retrieval")


@dataclass(frozen=True)
class CalibratedRegimeConfig:
    """Frozen calibration and decision parameters for a specific operational regime."""

    regime: str
    calibrator: MulticlassCalibrator
    tau_contra: float
    tau_sup: float
    tau_floor: float
    temperature: float
    prior_bias: Tuple[float, float, float]
    l2_reg: float


def load_frozen_calibration_bundle(
    bundle_path: Optional[Union[str, Path]] = None,
) -> Dict[str, CalibratedRegimeConfig]:
    """Load and strictly validate the frozen Stage 1 calibration bundle from disk.

    Args:
        bundle_path: Optional custom path to 10_stage1_frozen_bundle.json.
                     Defaults to backend/experiments/results/10_stage1_frozen_bundle.json.

    Returns:
        Dictionary mapping regime names ('direct', 'retrieval') to CalibratedRegimeConfig objects.

    Raises:
        FileNotFoundError: If the bundle file does not exist.
        ValueError: If the bundle structure, parameters, or values are invalid or malformed.
    """
    resolved_path = (
        DEFAULT_FROZEN_BUNDLE_PATH
        if bundle_path is None
        else Path(bundle_path).expanduser().resolve()
    )

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Frozen calibration bundle not found at: {resolved_path}. "
            "Ensure Stage 1 calibration artifact 10_stage1_frozen_bundle.json is present."
        )
    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"Frozen calibration bundle path is not a file: {resolved_path}"
        )

    try:
        with open(resolved_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except Exception as exc:
        raise ValueError(
            f"Failed to parse frozen calibration bundle as valid JSON from {resolved_path}: {exc}"
        ) from exc

    if not isinstance(raw_data, dict):
        raise ValueError(f"Expected dictionary at root of frozen bundle, got {type(raw_data).__name__}")

    if "regimes" not in raw_data or not isinstance(raw_data["regimes"], dict):
        raise ValueError("Malformed frozen bundle: missing required top-level 'regimes' dictionary.")

    configs: Dict[str, CalibratedRegimeConfig] = {}

    for regime in SUPPORTED_REGIMES:
        if regime not in raw_data["regimes"]:
            raise ValueError(
                f"Malformed frozen bundle: missing configuration for required regime '{regime}'."
            )

        r_data = raw_data["regimes"][regime]
        if not isinstance(r_data, dict):
            raise ValueError(f"Malformed frozen bundle: regime '{regime}' must be a dictionary.")

        # Validate calibrators section
        calibrators = r_data.get("calibrators")
        if not isinstance(calibrators, dict) or "C2_prior_bias_temp" not in calibrators:
            raise ValueError(
                f"Malformed frozen bundle: regime '{regime}' missing 'calibrators.C2_prior_bias_temp'."
            )

        c2_payload = calibrators["C2_prior_bias_temp"]
        if not isinstance(c2_payload, dict):
            raise ValueError(
                f"Malformed frozen bundle: 'C2_prior_bias_temp' in regime '{regime}' must be a dict."
            )

        # Validate temperatures and bias
        temps = c2_payload.get("temperatures")
        bias = c2_payload.get("bias")
        if not isinstance(temps, (list, tuple)) or len(temps) < 1:
            raise ValueError(
                f"Malformed frozen bundle: invalid temperatures in regime '{regime}': {temps!r}"
            )
        temp_val = float(temps[0])
        if not math.isfinite(temp_val) or temp_val <= 0.0:
            raise ValueError(
                f"Malformed frozen bundle: temperature must be positive finite float, got {temp_val} in '{regime}'."
            )

        if not isinstance(bias, (list, tuple)) or len(bias) != 3:
            raise ValueError(
                f"Malformed frozen bundle: bias must be a sequence of 3 floats, got {bias!r} in '{regime}'."
            )
        bias_floats = [float(b) for b in bias]
        if not all(math.isfinite(b) for b in bias_floats):
            raise ValueError(
                f"Malformed frozen bundle: non-finite bias values in regime '{regime}': {bias_floats}"
            )

        l2_reg_val = float(c2_payload.get("l2_reg", 0.0))

        # Instantiate MulticlassCalibrator via existing from_dict method
        try:
            calibrator_obj = MulticlassCalibrator.from_dict(c2_payload)
        except Exception as exc:
            raise ValueError(
                f"Failed to instantiate MulticlassCalibrator for regime '{regime}': {exc}"
            ) from exc

        # Validate strategy thresholds section
        objectives = r_data.get("objectives")
        if not isinstance(objectives, dict):
            raise ValueError(f"Malformed frozen bundle: regime '{regime}' missing 'objectives' dict.")

        obj_constrained = objectives.get("OBJ_CONSTRAINED")
        if not isinstance(obj_constrained, dict):
            raise ValueError(
                f"Malformed frozen bundle: regime '{regime}' missing 'objectives.OBJ_CONSTRAINED'."
            )

        strategies = obj_constrained.get("strategies")
        if not isinstance(strategies, dict) or "H2_deberta_cal_c2" not in strategies:
            raise ValueError(
                f"Malformed frozen bundle: missing strategy 'H2_deberta_cal_c2' in regime '{regime}'."
            )

        h2_data = strategies["H2_deberta_cal_c2"]
        params = h2_data.get("params") if isinstance(h2_data, dict) else None
        if not isinstance(params, dict):
            raise ValueError(
                f"Malformed frozen bundle: missing 'params' for H2_deberta_cal_c2 in regime '{regime}'."
            )

        for param_name in ("tau_contra", "tau_sup", "tau_floor"):
            if param_name not in params:
                raise ValueError(
                    f"Malformed frozen bundle: missing '{param_name}' in H2 params for regime '{regime}'."
                )

        tau_contra = float(params["tau_contra"])
        tau_sup = float(params["tau_sup"])
        tau_floor = float(params["tau_floor"])

        for t_name, t_val in (("tau_contra", tau_contra), ("tau_sup", tau_sup), ("tau_floor", tau_floor)):
            if not math.isfinite(t_val) or not (0.0 <= t_val <= 1.0):
                raise ValueError(
                    f"Malformed frozen bundle: threshold '{t_name}' must be in [0.0, 1.0], got {t_val} in '{regime}'."
                )

        configs[regime] = CalibratedRegimeConfig(
            regime=regime,
            calibrator=calibrator_obj,
            tau_contra=tau_contra,
            tau_sup=tau_sup,
            tau_floor=tau_floor,
            temperature=temp_val,
            prior_bias=(bias_floats[0], bias_floats[1], bias_floats[2]),
            l2_reg=l2_reg_val,
        )

    logger.info(
        "Successfully loaded frozen calibration bundle from %s. Regimes: %s",
        resolved_path,
        list(configs.keys()),
    )
    return configs


class CalibratedDebertaVerifier:
    """Production verifier applying fine-tuned DeBERTa-v3 with frozen C2 calibration and H2 rules.

    This component serves as the primary factual verification engine for VeriLLM Phase 9.
    It takes extracted claims and evidence matches, performs batched neural inference with
    verillm_deberta, reconstructs centered logits, applies frozen C2 calibration, and synthesizes
    transparent classifications using pre-registered H2 thresholding rules.
    """

    def __init__(
        self,
        bundle_path: Optional[Union[str, Path]] = None,
        default_regime: RegimeName = "direct",
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[Any] = None,
        batch_size: int = 16,
        deberta_verifier: Optional[CustomDebertaVerifier] = None,
    ) -> None:
        """Initialize the CalibratedDebertaVerifier.

        Args:
            bundle_path: Path to 10_stage1_frozen_bundle.json (defaults to experiments/results).
            default_regime: Default operational regime ('direct' or 'retrieval').
            model_path: Optional custom path to verillm_deberta checkpoint directory.
            device: Optional torch.device or string ('cpu', 'cuda:0').
            batch_size: Batch size for CustomDebertaVerifier.predict_pairs.
            deberta_verifier: Optional pre-instantiated or mocked CustomDebertaVerifier.
        """
        if default_regime not in SUPPORTED_REGIMES:
            raise ValueError(
                f"Unsupported default regime '{default_regime}'. Expected one of: {SUPPORTED_REGIMES}"
            )

        self._bundle_path = bundle_path
        self._configs = load_frozen_calibration_bundle(bundle_path)
        self._default_regime: RegimeName = default_regime
        self._model_path = model_path
        self._device = device
        self._batch_size = max(1, int(batch_size))
        self._custom_deberta: Optional[CustomDebertaVerifier] = deberta_verifier

    @property
    def configs(self) -> Dict[str, CalibratedRegimeConfig]:
        """Dictionary of loaded regime configurations."""
        return self._configs

    @property
    def default_regime(self) -> RegimeName:
        """Default regime applied when not explicitly overridden in verify_claims."""
        return self._default_regime

    @property
    def custom_deberta(self) -> CustomDebertaVerifier:
        """Lazily initialized CustomDebertaVerifier singleton wrapper."""
        if self._custom_deberta is None:
            logger.info("Lazily loading CustomDebertaVerifier singleton...")
            self._custom_deberta = CustomDebertaVerifier(
                model_path=self._model_path,
                device=self._device,
                batch_size=self._batch_size,
            )
        return self._custom_deberta

    def get_regime_config(self, regime: Optional[str] = None) -> CalibratedRegimeConfig:
        """Resolve and return the CalibratedRegimeConfig for a given regime."""
        target_regime = self._default_regime if regime is None else regime.lower()
        if target_regime not in self._configs:
            raise ValueError(
                f"Invalid regime '{regime}'. Expected one of: {list(self._configs.keys())}"
            )
        return self._configs[target_regime]

    def verify_claims(
        self,
        claims: List[ExtractedClaim],
        evidence_matches: List[ClaimEvidenceMatch],
        regime: Optional[str] = None,
        document_name: Optional[str] = None,
    ) -> VerificationReport:
        """Verify extracted claims against evidence using frozen C2 calibration and H2 rules.

        Args:
            claims: List of ExtractedClaim objects to verify.
            evidence_matches: List of ClaimEvidenceMatch objects corresponding to the claims.
            regime: Operational regime ('direct' or 'retrieval'). Defaults to self.default_regime.
            document_name: Optional document provenance label.

        Returns:
            VerificationReport containing verified claims, scores, rules, and aggregated metrics.
        """
        start_time = time.time()
        active_config = self.get_regime_config(regime)
        active_regime = active_config.regime

        # Handle empty claims edge case
        if not claims:
            empty_metrics = VerificationMetrics(
                total_claims=0,
                factual_claims=0,
                non_factual_claims=0,
                supported_count=0,
                contradicted_count=0,
                unsupported_count=0,
                unverifiable_count=0,
                non_factual_count=0,
                has_factual_claims=False,
                evidence_support_rate=None,
                confirmed_contradiction_rate=None,
                unsupported_rate=None,
                unverifiable_rate=None,
                uncertainty_rate=None,
                denominator_description="No claims to evaluate. Factuality metrics are undefined (N/A).",
            )
            return VerificationReport(
                document_name=document_name,
                total_claims_processed=0,
                claims=[],
                metrics=empty_metrics,
                execution_time_seconds=0.0,
            )

        # Index evidence by claim_id
        evidence_by_id: Dict[str, ClaimEvidenceMatch] = {
            m.claim_id: m for m in evidence_matches
        }

        # Step 1: Collect pairs requiring neural inference
        # In Direct mode: evaluate factual claims that have non-empty context.
        # In Retrieval mode: evaluate factual claims that have evidence with similarity >= tau_floor.
        eval_indices: List[int] = []
        eval_pairs: List[Tuple[str, str]] = []

        for idx, claim in enumerate(claims):
            if not claim.is_factual:
                continue

            match = evidence_by_id.get(claim.claim_id)
            primary_evidence = match.best_passage if match else None

            if primary_evidence is None or not primary_evidence.chunk.text.strip():
                continue

            # In Retrieval mode, enforce the evidence similarity floor gate
            if active_regime == "retrieval":
                sim = primary_evidence.similarity_score
                if sim is not None and sim < active_config.tau_floor:
                    continue

            eval_indices.append(idx)
            eval_pairs.append((primary_evidence.chunk.text, claim.text))

        # Step 2: Batched neural inference
        raw_predictions_by_idx: Dict[int, CustomDebertaPrediction] = {}
        cal_probs_by_idx: Dict[int, np.ndarray] = {}

        if eval_pairs:
            predictions = self.custom_deberta.predict_pairs(
                eval_pairs, batch_size=self._batch_size
            )
            if len(predictions) != len(eval_indices):
                raise ValueError(
                    f"CustomDebertaVerifier returned {len(predictions)} predictions "
                    f"for {len(eval_indices)} input pairs."
                )

            # Build raw probabilities matrix (N_eval, 3) in [CONTRADICTED, SUPPORTED_PROXY, UNSUPPORTED] order
            raw_probs_matrix = np.empty((len(predictions), 3), dtype=np.float64)
            for p_idx, pred in enumerate(predictions):
                c_prob = float(pred.contradicted_prob)
                s_prob = float(pred.supported_proxy_prob)
                u_prob = float(pred.unsupported_prob)

                if not (math.isfinite(c_prob) and math.isfinite(s_prob) and math.isfinite(u_prob)):
                    raise ValueError(f"Non-finite probability returned by verifier: {pred}")

                raw_probs_matrix[p_idx, 0] = c_prob
                raw_probs_matrix[p_idx, 1] = s_prob
                raw_probs_matrix[p_idx, 2] = u_prob

            # Step 3: Reconstruct centered logits and apply frozen C2 calibration
            logits = reconstruct_logits_from_probs(raw_probs_matrix)
            cal_probs = active_config.calibrator.transform_logits(logits)

            for i, original_idx in enumerate(eval_indices):
                raw_predictions_by_idx[original_idx] = predictions[i]
                cal_probs_by_idx[original_idx] = cal_probs[i]

        # Step 4: Construct arrays required for _predict_h2_codes
        n_claims = len(claims)
        empty_ev = np.zeros(n_claims, dtype=bool)
        best_sim = np.zeros(n_claims, dtype=np.float64)
        non_factual = np.zeros(n_claims, dtype=bool)
        all_cal_probs = np.zeros((n_claims, 3), dtype=np.float64)
        all_cal_probs[:, 2] = 1.0  # Default to unsupported neutral for non-evaluated claims

        for idx, claim in enumerate(claims):
            match = evidence_by_id.get(claim.claim_id)
            primary_evidence = match.best_passage if match else None

            if not claim.is_factual:
                non_factual[idx] = True

            if primary_evidence is None or not primary_evidence.chunk.text.strip():
                empty_ev[idx] = True
            else:
                sim = primary_evidence.similarity_score
                best_sim[idx] = 1.0 if sim is None else float(sim)

            if idx in cal_probs_by_idx:
                all_cal_probs[idx] = cal_probs_by_idx[idx]

        h2_arrays = {
            "empty_ev": empty_ev,
            "best_sim": best_sim,
            "non_factual": non_factual,
            "conflict": np.zeros(n_claims, dtype=bool),
        }

        # Step 5: Vectorized H2 decision codes
        predicted_codes = _predict_h2_codes(
            arrays=h2_arrays,
            cal_probs=all_cal_probs,
            regime=active_regime,
            tau_contra=active_config.tau_contra,
            tau_sup=active_config.tau_sup,
            tau_floor=active_config.tau_floor,
            apply_conflict_guard=False,
        )
        # Empty evidence produces UNVERIFIABLE across all regimes
        predicted_codes = np.where(empty_ev, 3, predicted_codes)

        # Step 6: Assemble VerifiedClaim objects
        verified_claims: List[VerifiedClaim] = []

        for idx, claim in enumerate(claims):
            code = int(predicted_codes[idx])
            match = evidence_by_id.get(claim.claim_id)
            primary_evidence = match.best_passage if match else None
            all_evidence = match.retrieved_passages if match else []

            raw_pred = raw_predictions_by_idx.get(idx)
            cal_prob_vec = cal_probs_by_idx.get(idx)

            sim_val = primary_evidence.similarity_score if primary_evidence else None
            doc_str = (
                primary_evidence.chunk.document_name
                if (primary_evidence and primary_evidence.chunk)
                else (document_name or "N/A")
            )
            page_str = (
                f"Page {primary_evidence.chunk.page_number}"
                if (primary_evidence and primary_evidence.chunk and primary_evidence.chunk.page_number)
                else "N/A"
            )

            scores = ModelVerificationScore(
                retrieval_similarity=sim_val,
                custom_deberta_supported_prob=raw_pred.supported_proxy_prob if raw_pred else None,
                custom_deberta_contradiction_prob=raw_pred.contradicted_prob if raw_pred else None,
                custom_deberta_unsupported_prob=raw_pred.unsupported_prob if raw_pred else None,
                custom_deberta_predicted_label=raw_pred.predicted_label if raw_pred else None,
            )

            # Code mappings:
            # 0: CONTRADICTED
            # 1: SUPPORTED_PROXY -> SUPPORTED
            # 2: UNSUPPORTED
            # 3: UNVERIFIABLE
            # 4: NON_FACTUAL
            if code == 4:
                classification = VerificationCategory.NON_FACTUAL
                decision_rule = "NON_FACTUAL_ASSERTION"
                reasoning_trace = (
                    "Statement is subjective, opinion-based, or conversational meta-text "
                    "and does not assert an empirical factual proposition."
                )
            elif code == 3:
                classification = VerificationCategory.UNVERIFIABLE
                if empty_ev[idx]:
                    decision_rule = "EMPTY_EVIDENCE"
                    reasoning_trace = (
                        "No reference evidence was provided or retrieved to verify this claim."
                    )
                else:
                    decision_rule = "EVIDENCE_SIMILARITY_BELOW_FLOOR"
                    reasoning_trace = (
                        f"Top retrieved passage semantic similarity ({best_sim[idx]:.4f}) "
                        f"is below the required evidence floor ({active_config.tau_floor:.2f}). "
                        "The reference document lacks sufficient context regarding this proposition."
                    )
            elif code == 0:
                classification = VerificationCategory.CONTRADICTED
                decision_rule = "CALIBRATED_DEBERTA_C2_CONTRADICTION"
                p_c = float(cal_prob_vec[0]) if cal_prob_vec is not None else 0.0
                p_s = float(cal_prob_vec[1]) if cal_prob_vec is not None else 0.0
                reasoning_trace = (
                    f"Evidence in '{doc_str}' ({page_str}) refutes the claim. "
                    f"Calibrated DeBERTa contradiction probability ({p_c:.4f}) meets or exceeds "
                    f"threshold ({active_config.tau_contra:.2f}). "
                    f"[Calibrated: contra={p_c:.4f}, sup={p_s:.4f} | "
                    f"Raw softmax: contra={scores.custom_deberta_contradiction_prob:.4f}, "
                    f"sup={scores.custom_deberta_supported_prob:.4f}]"
                )
            elif code == 1:
                classification = VerificationCategory.SUPPORTED
                decision_rule = "CALIBRATED_DEBERTA_C2_SUPPORT"
                p_c = float(cal_prob_vec[0]) if cal_prob_vec is not None else 0.0
                p_s = float(cal_prob_vec[1]) if cal_prob_vec is not None else 0.0
                reasoning_trace = (
                    f"Evidence in '{doc_str}' ({page_str}) entails the claim. "
                    f"Calibrated DeBERTa support probability ({p_s:.4f}) meets or exceeds "
                    f"threshold ({active_config.tau_sup:.2f}) with contradiction ({p_c:.4f}) "
                    f"below threshold ({active_config.tau_contra:.2f}). "
                    f"[Calibrated: sup={p_s:.4f}, contra={p_c:.4f} | "
                    f"Raw softmax: sup={scores.custom_deberta_supported_prob:.4f}, "
                    f"contra={scores.custom_deberta_contradiction_prob:.4f}]"
                )
            elif code == 2:
                classification = VerificationCategory.UNSUPPORTED
                decision_rule = "CALIBRATED_DEBERTA_C2_UNSUPPORTED"
                p_c = float(cal_prob_vec[0]) if cal_prob_vec is not None else 0.0
                p_s = float(cal_prob_vec[1]) if cal_prob_vec is not None else 0.0
                p_u = float(cal_prob_vec[2]) if cal_prob_vec is not None else 0.0
                reasoning_trace = (
                    f"Evidence in '{doc_str}' ({page_str}) discusses related context but does not "
                    f"contain factual assertions confirming or refuting the claim. "
                    f"[Calibrated: unsup={p_u:.4f}, sup={p_s:.4f} (< {active_config.tau_sup:.2f}), "
                    f"contra={p_c:.4f} (< {active_config.tau_contra:.2f})]"
                )
            else:
                raise ValueError(f"Unexpected H2 decision code: {code}")

            verified_claims.append(
                VerifiedClaim(
                    claim_id=claim.claim_id,
                    claim_text=claim.text,
                    source_sentence=claim.source_sentence,
                    start_char=claim.start_char,
                    end_char=claim.end_char,
                    claim_start_char=claim.claim_start_char,
                    claim_end_char=claim.claim_end_char,
                    offset_type=claim.offset_type,
                    is_factual=claim.is_factual,
                    classification=classification,
                    scores=scores,
                    primary_evidence=primary_evidence,
                    all_evidence=all_evidence,
                    decision_rule=decision_rule,
                    reasoning_trace=reasoning_trace,
                )
            )

        # Step 7: Calculate factuality and uncertainty metrics
        total_claims = len(verified_claims)
        factual_claims = sum(1 for c in verified_claims if c.is_factual)
        non_factual_claims = total_claims - factual_claims

        supported_count = sum(
            1 for c in verified_claims
            if c.is_factual and c.classification == VerificationCategory.SUPPORTED
        )
        contradicted_count = sum(
            1 for c in verified_claims
            if c.is_factual and c.classification == VerificationCategory.CONTRADICTED
        )
        unsupported_count = sum(
            1 for c in verified_claims
            if c.is_factual and c.classification == VerificationCategory.UNSUPPORTED
        )
        unverifiable_count = sum(
            1 for c in verified_claims
            if c.is_factual and c.classification == VerificationCategory.UNVERIFIABLE
        )
        non_factual_count = sum(
            1 for c in verified_claims
            if not c.is_factual or c.classification == VerificationCategory.NON_FACTUAL
        )

        if factual_claims > 0:
            support_rate = round((supported_count / factual_claims) * 100.0, 2)
            contradiction_rate = round((contradicted_count / factual_claims) * 100.0, 2)
            unsupported_rate = round((unsupported_count / factual_claims) * 100.0, 2)
            unverifiable_rate = round((unverifiable_count / factual_claims) * 100.0, 2)
            uncertainty_rate = round(
                ((unsupported_count + unverifiable_count) / factual_claims) * 100.0, 2
            )
            has_factual = True
            denominator_desc = (
                f"Metrics evaluated over {factual_claims} empirical factual claims "
                f"({non_factual_claims} non-factual statements excluded from factuality denominator)."
            )
        else:
            support_rate = None
            contradiction_rate = None
            unsupported_rate = None
            unverifiable_rate = None
            uncertainty_rate = None
            has_factual = False
            denominator_desc = (
                f"No empirical factual claims to evaluate ({non_factual_claims} non-factual statements excluded). "
                "Factuality metrics are undefined (N/A)."
            )

        metrics = VerificationMetrics(
            total_claims=total_claims,
            factual_claims=factual_claims,
            non_factual_claims=non_factual_claims,
            supported_count=supported_count,
            contradicted_count=contradicted_count,
            unsupported_count=unsupported_count,
            unverifiable_count=unverifiable_count,
            non_factual_count=non_factual_count,
            has_factual_claims=has_factual,
            evidence_support_rate=support_rate,
            confirmed_contradiction_rate=contradiction_rate,
            unsupported_rate=unsupported_rate,
            unverifiable_rate=unverifiable_rate,
            uncertainty_rate=uncertainty_rate,
            denominator_description=denominator_desc,
        )

        duration = time.time() - start_time

        return VerificationReport(
            document_name=document_name,
            total_claims_processed=total_claims,
            claims=verified_claims,
            metrics=metrics,
            execution_time_seconds=round(duration, 3),
        )

    def verify_claim(
        self,
        claim: ExtractedClaim,
        evidence_match: Optional[ClaimEvidenceMatch] = None,
        regime: Optional[str] = None,
        document_name: Optional[str] = None,
    ) -> VerifiedClaim:
        """Convenience method to verify a single claim proposition."""
        matches = [evidence_match] if evidence_match is not None else []
        report = self.verify_claims([claim], matches, regime=regime, document_name=document_name)
        return report.claims[0]
