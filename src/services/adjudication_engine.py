"""
AssureX Claim Engine — Multi-Model Adjudication & Decision Orchestrator

Integrates and arbitrates:
1. Deterministic Business Rule Engine (coverage window, exclusions, serials)
2. Python Tabular ML Model (XGBoost)
3. Google Teachable Machine Computer Vision Model (MobileNetV2 on Claim Summary Card)

Implements the multi-tiered decision matrix, confidence arbitration,
and automated reason generation.
"""

import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

from .rule_engine import RuleEngine
from .card_service import CardService
from ..ml.predictor import TabularPredictor
from ..ml.teachable_machine import TeachableMachinePredictor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
THRESHOLDS_PATH = os.path.join(BASE_DIR, "config", "thresholds.json")


def load_thresholds() -> Dict[str, Any]:
    """Loads confidence thresholds from configuration."""
    if os.path.exists(THRESHOLDS_PATH):
        try:
            with open(THRESHOLDS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "strong_match": {"max_confidence_difference": 0.05, "min_confidence": 0.70},
        "acceptable_match": {"max_confidence_difference": 0.15, "min_confidence": 0.60},
        "weak_match": {"max_confidence_difference": 0.25, "min_confidence": 0.50},
    }


class AdjudicationEngine:
    """Orchestrates multi-model adjudication and generates auditable claim outcomes."""

    def __init__(self):
        self.tabular_predictor = TabularPredictor()
        self.tm_predictor = TeachableMachinePredictor()
        self.thresholds = load_thresholds()

    def arbitrate(
        self,
        claim_data: Dict[str, Any],
        tab_eval: Dict[str, Any],
        tm_eval: Dict[str, Any],
        card_image_path: str
    ) -> Dict[str, Any]:
        """
        Combines rule evaluation, tabular model output, and TM vision output into final decision.
        """
        rule_eval = RuleEngine.evaluate_claim(claim_data)
        tab_class = tab_eval["predicted_class"]
        tab_conf = tab_eval["top_confidence"]
        tm_class = tm_eval["predicted_class"]
        tm_conf = tm_eval["top_confidence"]

        conf_diff = round(abs(tab_conf - tm_conf), 4)
        models_agreed = (tab_class == tm_class)

        # Match Category Classification
        if tab_conf < 0.50 or tm_conf < 0.50:
            match_category = "Uncertain"
        elif not models_agreed:
            match_category = "Disagreement"
        elif conf_diff < 0.05:
            match_category = "Strong Match"
        elif conf_diff <= 0.15:
            match_category = "Acceptable Match"
        elif conf_diff <= 0.25:
            match_category = "Weak Match"
        else:
            match_category = "Weak Match (>25% gap)"

        # Arbitration & Decision Matrix
        reasons: List[str] = []

        # Scenario A: Hard Business Rule Failure (Exclusions, Expirations, Serials, Duplicates)
        # Per SRS Step 12: if AI models are borderline (both < 0.60) or strongly disagree (gap > 0.40),
        # route to Manual Review instead of outright Auto-Rejection.
        if not rule_eval["passed"]:
            both_low_conf = tab_conf < 0.60 and tm_conf < 0.60
            strong_disagreement = (not models_agreed) and conf_diff > 0.40
            if both_low_conf or strong_disagreement:
                adjudication_status = "Manual Review Required"
                adjudication_stage = "Manual_Review"
                reasons.extend(rule_eval["hard_failures"])
                reasons.append(
                    f"AI models show borderline or conflicting confidence ({tab_conf*100:.1f}% vs {tm_conf*100:.1f}%) — "
                    f"escalated for human adjudicator review despite rule violations."
                )
                final_confidence = round((tab_conf + tm_conf) / 2.0, 4)
                reason_summary = (
                    f"Routed to manual review: {len(rule_eval['hard_failures'])} rule violation(s) "
                    f"with low/conflicting AI confidence."
                )
            else:
                adjudication_status = "Auto-Rejected"
                adjudication_stage = "Automated"
                reasons.extend(rule_eval["hard_failures"])
                final_confidence = 0.99
                reason_summary = f"Claim automatically rejected due to {len(rule_eval['hard_failures'])} policy rule violation(s)."

        # Scenario B: Both AI Models Agree on Likely Invalid
        # Per SRS Step 12: low confidence (< 0.50) must escalate even if both agree.
        elif tab_class == "Likely Invalid" and tm_class == "Likely Invalid":
            if tab_conf < 0.50 or tm_conf < 0.50:
                adjudication_status = "Manual Review Required"
                adjudication_stage = "Manual_Review"
                reasons.append(
                    f"Both models classify as Likely Invalid, but confidence is below threshold "
                    f"({tab_conf*100:.1f}% tabular, {tm_conf*100:.1f}% vision). Human review required."
                )
                final_confidence = round((tab_conf + tm_conf) / 2.0, 4)
                reason_summary = "Escalated: both models lean invalid but confidence insufficient for automated rejection."
            else:
                adjudication_status = "Auto-Rejected"
                adjudication_stage = "Automated"
                reasons.append("Both structured data classifier (XGBoost) and visual card classifier (MobileNetV2) identified invalid claim patterns.")
                final_confidence = round((tab_conf + tm_conf) / 2.0, 4)
                reason_summary = "Automated rejection confirmed by consensus of dual AI models."

        # Scenario C: Both Models Agree on Likely Valid AND All Rules Passed
        elif tab_class == "Likely Valid" and tm_class == "Likely Valid" and rule_eval["passed"] and len(rule_eval["review_flags"]) == 0:
            min_conf = self.thresholds.get("strong_match", {}).get("min_confidence", 0.70)
            if tab_conf >= min_conf and tm_conf >= min_conf:
                adjudication_status = "Auto-Approved"
                adjudication_stage = "Automated"
                reasons.append("All policy rules satisfied. Dual AI models confirmed valid claim with high confidence.")
                final_confidence = round(0.55 * tab_conf + 0.45 * tm_conf, 4)
                reason_summary = "Claim qualified for instant automated approval."
            else:
                adjudication_status = "Manual Review Required"
                adjudication_stage = "Manual_Review"
                reasons.append(f"AI models agree on validity, but top confidence did not meet the auto-approval threshold ({min_conf*100:.0f}%).")
                final_confidence = round((tab_conf + tm_conf) / 2.0, 4)
                reason_summary = "Routed to manual review queue due to moderate confidence score."

        # Scenario D: Model Disagreement
        elif not models_agreed:
            adjudication_status = "Manual Review Required"
            adjudication_stage = "Manual_Review"
            reasons.append(f"AI Disagreement: Tabular model predicted '{tab_class}' ({tab_conf*100:.1f}%), while Vision model predicted '{tm_class}' ({tm_conf*100:.1f}%).")
            final_confidence = round(abs(tab_conf - tm_conf), 4)
            reason_summary = "Escalated to human reviewer due to dual-model classification disagreement."

        # Scenario E: Review Flags from Rule Engine (Grace period, missing documents)
        elif len(rule_eval["review_flags"]) > 0:
            adjudication_status = "Manual Review Required"
            adjudication_stage = "Manual_Review"
            reasons.extend(rule_eval["review_flags"])
            final_confidence = round(tab_conf, 4)
            reason_summary = f"Claim flagged for manual adjuster review ({len(rule_eval['review_flags'])} policy flag(s))."

        # Scenario F: Catch-all Manual Review
        else:
            adjudication_status = "Manual Review Required"
            adjudication_stage = "Manual_Review"
            reasons.append("Claim conditions require human adjudicator determination.")
            final_confidence = round(tab_conf, 4)
            reason_summary = "Routed to human review queue."

        return {
            "adjudication_status": adjudication_status,
            "adjudication_stage": adjudication_stage,
            "final_confidence": final_confidence,
            "decision_reason_summary": reason_summary,
            "decision_reasons": reasons,
            "card_image_path": card_image_path,
            "rule_evaluation": rule_eval,
            "tabular_prediction": tab_eval,
            "tm_prediction": tm_eval,
            "confidence_difference": conf_diff,
            "models_agreed": models_agreed,
            "match_category": match_category,
            "adjudication_timestamp": datetime.utcnow().isoformat(),
        }

    def adjudicate(self, claim_data: Dict[str, Any], card_image_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Runs complete adjudication pipeline on a claim.
        Returns all evaluation details, model predictions, decision status, and itemized reasons.
        """
        if not card_image_path or not os.path.exists(card_image_path):
            card_image_path = CardService.generate_and_save(claim_data)

        tab_eval = self.tabular_predictor.predict(claim_data)
        tm_eval = self.tm_predictor.predict(card_image_path)

        return self.arbitrate(claim_data, tab_eval, tm_eval, card_image_path)
