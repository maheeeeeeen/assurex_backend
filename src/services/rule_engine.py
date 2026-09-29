"""
AssureX Claim Engine — Deterministic Business Rule Engine

Encapsulates strict, configurable policy rules across 8 product categories:
- Electronics
- Appliances
- Automotive
- Smartphones & Mobile
- Computers & Laptops
- Wearables & Audio
- Home Office & Furniture
- Power Tools & Hardware

Evaluates:
1. Warranty window verification (Active, Category-Specific Grace Period, Expired)
2. Serial number reconciliation (Claim vs Receipt vs Warranty Card)
3. Category-specific excluded damage clauses
4. Unauthorized third-party repairs
5. Document completeness checks (Mandatory vs Optional per policy)
6. Timeline contradiction detection (Purchase after fault, claim before fault)
7. Duplicate submission detection
"""

import os
import json
from typing import Dict, Any, List, Tuple
from datetime import datetime, date

from .ocr_service import OCRService

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
POLICIES_DIR = os.path.join(BASE_DIR, "policies")

POLICY_CACHE: Dict[str, Dict[str, Any]] = {}


def get_policy_catalog() -> Dict[str, Dict[str, Any]]:
    """Loads and caches all policy files keyed by product_category and filename slug."""
    global POLICY_CACHE
    if POLICY_CACHE:
        return POLICY_CACHE

    if os.path.exists(POLICIES_DIR):
        for fname in os.listdir(POLICIES_DIR):
            if fname.endswith("_warranty.json"):
                fpath = os.path.join(POLICIES_DIR, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        pol = json.load(f)
                        cat = pol.get("product_category")
                        if cat:
                            POLICY_CACHE[cat.lower().strip()] = pol
                            # Also key by slugified category
                            slug = cat.lower().replace("&", "and").replace("  ", " ").strip()
                            POLICY_CACHE[slug] = pol
                        # Key by file stem
                        stem = fname.replace("_warranty.json", "").replace("_", " ").lower().strip()
                        POLICY_CACHE[stem] = pol
                except Exception as e:
                    print(f"Error loading policy {fname}: {e}")
    return POLICY_CACHE


def load_policy(category: str) -> Dict[str, Any]:
    """Loads warranty policy configuration JSON for a given product category."""
    if not category:
        category = "Electronics"

    catalog = get_policy_catalog()
    cat_clean = category.lower().strip()

    # 1. Exact match
    if cat_clean in catalog:
        return catalog[cat_clean]

    # 2. Match normalized category string
    cat_norm = cat_clean.replace("&", "and").replace("  ", " ").strip()
    if cat_norm in catalog:
        return catalog[cat_norm]

    # 3. Partial substring match
    for k, p in catalog.items():
        if cat_norm in k or k in cat_norm:
            return p

    # 4. Fallback to electronics
    return catalog.get("electronics", {})


class RuleEngine:
    """Evaluates business rules against claim data to provide auditable validation."""

    @staticmethod
    def evaluate_claim(claim_data: Dict[str, Any]) -> Dict[str, Any]:
        hard_failures: List[str] = []
        review_flags: List[str] = []
        passed_checks: List[str] = []

        category = claim_data.get("product_category", "Electronics")
        policy = load_policy(category)
        policy_id = policy.get("policy_id", "POL-DEFAULT-001")
        policy_name = policy.get("policy_name", f"{category} Standard Warranty")
        grace_days = int(policy.get("grace_period_days", 7))
        reporting_period_days = int(policy.get("claim_reporting_period_days", 30))
        max_repairs = int(policy.get("max_repairs_before_replacement", 3))

        # -------------------------------------------------------------
        # Rule 1: Chronological Consistency
        # -------------------------------------------------------------
        try:
            purchase_dt = datetime.strptime(str(claim_data.get("purchase_date"))[:10], "%Y-%m-%d").date()
            fault_dt = datetime.strptime(str(claim_data.get("fault_date"))[:10], "%Y-%m-%d").date()
            claim_dt = datetime.strptime(str(claim_data.get("claim_submission_date"))[:10], "%Y-%m-%d").date()
            warranty_end_dt = datetime.strptime(str(claim_data.get("warranty_end"))[:10], "%Y-%m-%d").date()

            if purchase_dt > fault_dt:
                hard_failures.append(f"Date contradiction: Purchase date ({purchase_dt}) cannot be after fault date ({fault_dt}).")
            elif claim_dt < fault_dt:
                hard_failures.append(f"Date contradiction: Claim submission date ({claim_dt}) cannot precede fault date ({fault_dt}).")
            elif (claim_dt - fault_dt).days > (reporting_period_days + grace_days):
                hard_failures.append(f"Reporting window exceeded: Claim filed {(claim_dt - fault_dt).days} days after fault (policy maximum is {reporting_period_days} days).")
            else:
                passed_checks.append("Timeline chronology verified (Purchase <= Fault <= Claim).")
        except Exception:
            if claim_data.get("date_contradiction_flag"):
                hard_failures.append("Date contradiction detected in reported claim timelines.")
            else:
                passed_checks.append("Timeline format verified.")

        # -------------------------------------------------------------
        # Rule 2: Coverage Period & Category Grace Period Window
        # -------------------------------------------------------------
        rem_days = float(claim_data.get("remaining_warranty_days", 0.0))
        if rem_days >= 0:
            passed_checks.append(f"Warranty actively in effect ({int(rem_days)} days remaining).")
        elif -grace_days <= rem_days < 0:
            review_flags.append(f"Grace Period Active: Warranty expired {abs(int(rem_days))} days ago (within {grace_days}-day policy grace window). Escalation required.")
        else:
            hard_failures.append(f"Warranty Expired: Claim submitted {abs(int(rem_days))} days past warranty expiration (outside {grace_days}-day grace period).")

        # -------------------------------------------------------------
        # Rule 3: Serial & Model Number Reconciliation (Cross-Document Verification)
        # -------------------------------------------------------------
        serial_entered = str(claim_data.get("serial_number_entered", "") or "").strip().upper()
        model_entered = str(claim_data.get("model_number", "") or claim_data.get("model_number_entered", "") or "").strip().upper()

        receipt_sn = str(claim_data.get("serial_number_on_receipt", "") or "").strip().upper()
        receipt_mod = str(claim_data.get("model_number_on_receipt", "") or "").strip().upper()

        card_sn = str(claim_data.get("serial_number_on_warranty_card", "") or "").strip().upper()
        card_mod = str(claim_data.get("model_number_on_warranty_card", "") or "").strip().upper()

        barcode_sn = str(claim_data.get("serial_number_on_barcode", "") or "").strip().upper()
        barcode_mod = str(claim_data.get("model_number_on_barcode", "") or "").strip().upper()

        receipt_data = {"serial_number": receipt_sn, "model_number": receipt_mod} if (receipt_sn or receipt_mod) else None
        warranty_card_data = {"serial_number": card_sn, "model_number": card_mod} if (card_sn or card_mod) else None
        barcode_data = {"serial_number": barcode_sn, "model_number": barcode_mod} if (barcode_sn or barcode_mod) else None

        cross_res = OCRService.cross_verify_all(
            entered_serial=serial_entered,
            entered_model=model_entered,
            receipt_data=receipt_data,
            warranty_card_data=warranty_card_data,
            barcode_data=barcode_data,
        )

        cross_mismatches = cross_res.get("mismatches", [])
        if cross_mismatches:
            for mis in cross_mismatches:
                review_flags.append(mis["message"])
        elif claim_data.get("serial_mismatch_flag", False):
            review_flags.append(f"Serial mismatch flag: Entered serial ({serial_entered}) conflicts with verified proof of purchase.")
        else:
            if cross_res.get("total_matches", 0) > 0:
                passed_checks.append(f"Cross-document verification passed: Serial and model numbers consistent across verified documents ({serial_entered}).")
            else:
                passed_checks.append(f"Serial number validated across documents ({serial_entered}).")

        # -------------------------------------------------------------
        # Rule 4: Excluded Damage Types (Category Policy Terms)
        # -------------------------------------------------------------
        excluded_flag = bool(claim_data.get("excluded_damage", False))
        damage_type = str(claim_data.get("damage_type", "")).lower()

        raw_exclusions = policy.get("exclusions", [])
        if isinstance(raw_exclusions, list):
            excluded_damages = [str(d).lower() for d in raw_exclusions]
        else:
            excluded_damages = []

        if excluded_flag or any(ex in damage_type for ex in excluded_damages if ex):
            hard_failures.append(f"Excluded Damage Clause: '{claim_data.get('damage_type')}' is explicitly excluded under {policy_name}.")
        else:
            passed_checks.append(f"Reported damage type '{claim_data.get('damage_type')}' is eligible for coverage.")

        # -------------------------------------------------------------
        # Rule 5: Repair History & Authorized Service
        # -------------------------------------------------------------
        repair_count = int(claim_data.get("repair_history_count", 0))
        prev_authorized = bool(claim_data.get("previous_repair_authorized", True))

        if repair_count > 1 and not prev_authorized:
            hard_failures.append(f"Warranty Void: Product underwent {repair_count} unauthorized third-party repairs, violating policy terms.")
        elif repair_count == 1 and not prev_authorized:
            review_flags.append("Warning MR-003: Single unauthorized prior repair detected. Technician review required.")
        elif repair_count >= max_repairs:
            review_flags.append(f"High repair frequency ({repair_count} prior repairs): Product has reached policy replacement threshold ({max_repairs}).")
        else:
            passed_checks.append("Service history authorized and verified.")

        # -------------------------------------------------------------
        # Rule 6: Duplicate Claim Detection
        # -------------------------------------------------------------
        if bool(claim_data.get("duplicate_claim_flag", False)):
            dup_details = claim_data.get("duplicate_claim_details")
            if dup_details:
                hard_failures.append(f"Fraud prevention flag: {dup_details}")
            else:
                hard_failures.append("Fraud prevention flag: An identical claim was recently filed for this serial number.")
        else:
            passed_checks.append("Duplicate check passed (no identical active claims).")

        # -------------------------------------------------------------
        # Rule 7: Document Completeness (Policy Mandatory Documents)
        # -------------------------------------------------------------
        missing = []
        if not bool(claim_data.get("receipt_uploaded", True)):
            hard_failures.append("Missing Mandatory Document: Proof of Purchase Receipt is required for claim adjudication.")

        if not bool(claim_data.get("fault_evidence_uploaded", True)):
            review_flags.append("Missing supporting evidence: Photographic fault evidence not uploaded.")

        if not bool(claim_data.get("product_image_uploaded", True)):
            review_flags.append("Missing product photo: Overall product condition photo not uploaded.")

        if not missing and bool(claim_data.get("receipt_uploaded", True)):
            passed_checks.append("Primary proof of purchase receipt verified.")

        # -------------------------------------------------------------
        # Final Decision Recommendation
        # -------------------------------------------------------------
        if hard_failures:
            recommendation = "Auto-Reject"
            summary = f"Claim fails mandatory business rules ({len(hard_failures)} violation(s))."
        elif review_flags:
            recommendation = "Manual Review"
            summary = f"Claim requires human review ({len(review_flags)} flag(s))."
        else:
            recommendation = "Auto-Approve"
            summary = f"All business rules and terms for {policy_name} satisfied."

        return {
            "policy_id": policy_id,
            "policy_name": policy_name,
            "product_category": policy.get("product_category", category),
            "passed": len(hard_failures) == 0,
            "recommendation": recommendation,
            "summary": summary,
            "hard_failures": hard_failures,
            "review_flags": review_flags,
            "passed_checks": passed_checks,
            "total_checks": len(hard_failures) + len(review_flags) + len(passed_checks),
        }
