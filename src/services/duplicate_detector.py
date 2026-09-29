"""
AssureX Claim Engine — Semantic Duplicate Claim Detection Service

Provides multi-factor duplicate claim identification:
1. Cryptographic Document Fingerprint (SHA-256 Hash Matching)
2. Matching Invoice / Receipt Numbers Across Prior Claims
3. Semantic Fault Description Text Similarity for Same Product / Hardware Serial
   (Fuzzy matching via SequenceMatcher + token overlap ratio)
"""

import re
from difflib import SequenceMatcher
from typing import Dict, Any, Optional, List
from sqlmodel import Session, select
from src.models import Claim


class DuplicateDetector:
    SIMILARITY_THRESHOLD = 0.70  # 70% threshold for semantic duplicate flag

    SYNONYMS = {
        "display": "screen", "monitor": "screen", "panel": "screen",
        "hot": "overheat", "overheating": "overheat", "overheated": "overheat", "overheats": "overheat",
        "flickered": "flicker", "flickering": "flicker", "flickers": "flicker",
        "dark": "black", "blank": "black",
        "laptop": "device", "machine": "device", "unit": "device", "computer": "device",
        "broken": "damaged", "crack": "damaged", "cracked": "damaged",
        "shut": "power", "shutdown": "power", "died": "power", "dead": "power"
    }

    STOP_WORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "with", "is", "was", "are", "were", "of", "it", "my", "this", "that",
        "has", "had", "have", "not", "be", "been", "from", "by", "as", "at",
        "after", "when", "while", "during", "then", "so", "became", "went"
    }

    @classmethod
    def stem_token(cls, token: str) -> str:
        """Lightweight morphological normalization and synonym canonicalization."""
        token = cls.SYNONYMS.get(token, token)
        for suffix in ("ing", "ed", "es", "s", "ly"):
            if token.endswith(suffix) and len(token) > len(suffix) + 3:
                token = token[:-len(suffix)]
                break
        return cls.SYNONYMS.get(token, token)

    @staticmethod
    def clean_text(text: Optional[str]) -> str:
        """Normalizes text by removing punctuation, extra spaces, and lowercasing."""
        if not text:
            return ""
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        return re.sub(r"\s+", " ", cleaned).strip()

    @classmethod
    def compute_similarity(cls, text1: str, text2: str) -> float:
        """
        Calculates hybrid similarity between two fault descriptions using:
        1. Character-level fuzzy SequenceMatcher ratio
        2. Word-level token Jaccard and Dice coefficient
        3. Stemmed content keyword token-set alignment
        Returns a score in range [0.0, 1.0].
        """
        c1 = cls.clean_text(text1)
        c2 = cls.clean_text(text2)

        if not c1 or not c2:
            return 0.0
        if c1 == c2:
            return 1.0

        # Character sequence ratio
        seq_ratio = SequenceMatcher(None, c1, c2).ratio()

        # Stemmed tokens without stopwords
        raw_t1 = [cls.stem_token(w) for w in c1.split()]
        raw_t2 = [cls.stem_token(w) for w in c2.split()]
        k1 = set(raw_t1) - cls.STOP_WORDS
        k2 = set(raw_t2) - cls.STOP_WORDS

        if not k1 or not k2:
            return round(float(seq_ratio), 4)

        inter = k1 & k2
        union = k1 | k2
        jaccard = len(inter) / len(union) if union else 0.0
        dice = (2.0 * len(inter)) / (len(k1) + len(k2)) if (k1 or k2) else 0.0

        # Token set ratio
        t_inter = " ".join(sorted(inter))
        t1_sorted = " ".join(sorted(k1))
        t2_sorted = " ".join(sorted(k2))
        r1 = SequenceMatcher(None, t_inter, t1_sorted).ratio() if t_inter else 0.0
        r2 = SequenceMatcher(None, t_inter, t2_sorted).ratio() if t_inter else 0.0
        r3 = SequenceMatcher(None, t1_sorted, t2_sorted).ratio()
        token_set = max(r1, r2, r3) if t_inter else r3

        # Combine measures: TokenSet & Dice capture semantic overlap, seq_ratio captures exact phrasing
        hybrid = max(seq_ratio, token_set, dice, (token_set * 0.7 + jaccard * 0.3))
        return round(float(hybrid), 4)

    @classmethod
    def check_duplicate(
        cls,
        session: Session,
        product_id: Optional[str] = None,
        serial_number: Optional[str] = None,
        fault_description: Optional[str] = None,
        invoice_number: Optional[str] = None,
        receipt_hash: Optional[str] = None,
        current_claim_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Scans database for duplicate indicators:
        - Document hash reuse
        - Invoice number match
        - Semantic fault description similarity (for same product/serial)
        """
        norm_serial = serial_number.strip().upper() if serial_number else ""
        norm_invoice = invoice_number.strip().upper() if invoice_number else ""

        # Factor 1: Cryptographic Document Hash
        if receipt_hash and receipt_hash.strip():
            query = select(Claim).where(Claim.receipt_hash == receipt_hash.strip())
            if current_claim_id:
                query = query.where(Claim.claim_id != current_claim_id)
            existing_hash = session.exec(query).first()
            if existing_hash:
                return {
                    "is_duplicate": True,
                    "duplicate_type": "document_hash",
                    "matched_claim_id": existing_hash.claim_id,
                    "similarity_score": 1.0,
                    "details": f"Identical receipt document hash ({receipt_hash.strip()}) reused from prior claim {existing_hash.claim_id}.",
                }

        # Factor 2: Invoice Number Matching Across Prior Claims
        if norm_invoice:
            query = select(Claim).where(Claim.invoice_number.is_not(None))
            if current_claim_id:
                query = query.where(Claim.claim_id != current_claim_id)
            existing_inv_claims = session.exec(query).all()
            for c in existing_inv_claims:
                if c.invoice_number and c.invoice_number.strip().upper() == norm_invoice:
                    return {
                        "is_duplicate": True,
                        "duplicate_type": "invoice_number",
                        "matched_claim_id": c.claim_id,
                        "similarity_score": 1.0,
                        "details": f"Identical invoice number '{invoice_number.strip()}' was previously filed in claim {c.claim_id}.",
                    }

        # Factor 3: Semantic Similarity of Fault Description on Same Product / Serial
        clean_desc = cls.clean_text(fault_description)
        if clean_desc and (product_id or norm_serial):
            conditions = []
            if product_id:
                conditions.append(Claim.product_id == product_id)
            if norm_serial:
                conditions.append(Claim.serial_number_entered == serial_number.strip())

            if len(conditions) == 1:
                where_clause = conditions[0]
            else:
                from sqlmodel import or_
                where_clause = or_(*conditions)

            query = select(Claim).where(where_clause)
            if current_claim_id:
                query = query.where(Claim.claim_id != current_claim_id)

            prior_claims = session.exec(query).all()

            for pc in prior_claims:
                if not pc.fault_description:
                    continue
                sim = cls.compute_similarity(fault_description, pc.fault_description)
                if sim >= cls.SIMILARITY_THRESHOLD:
                    pct = round(sim * 100, 1)
                    return {
                        "is_duplicate": True,
                        "duplicate_type": "semantic_description",
                        "matched_claim_id": pc.claim_id,
                        "similarity_score": sim,
                        "details": (
                            f"Semantic similarity ({pct}%) in fault description matches prior claim "
                            f"{pc.claim_id} for the same product/serial."
                        ),
                    }

        # Factor 4: Repeated Hardware Serial Number Claim
        # Flag if the same serial number has any prior claim, regardless of description similarity.
        # Prevents filing multiple claims on the same physical device with different wordings.
        if norm_serial:
            serial_query = select(Claim).where(
                Claim.serial_number_entered == serial_number.strip()
            )
            if current_claim_id:
                serial_query = serial_query.where(Claim.claim_id != current_claim_id)
            prior_serial_claims = session.exec(serial_query).all()
            # Only flag if there's a prior claim that isn't already rejected
            active_serial_claims = [
                c for c in prior_serial_claims
                if c.adjudication_status not in ("Auto-Rejected", "Rejected")
            ]
            if active_serial_claims:
                prior = active_serial_claims[0]
                return {
                    "is_duplicate": True,
                    "duplicate_type": "repeated_serial_number",
                    "matched_claim_id": prior.claim_id,
                    "similarity_score": 1.0,
                    "details": (
                        f"Hardware serial '{serial_number.strip()}' was already submitted in claim "
                        f"{prior.claim_id} (status: {prior.adjudication_status}). "
                        f"Multiple claims on the same device serial are not permitted."
                    ),
                }

        return {
            "is_duplicate": False,
            "duplicate_type": None,
            "matched_claim_id": None,
            "similarity_score": 0.0,
            "details": None,
        }
