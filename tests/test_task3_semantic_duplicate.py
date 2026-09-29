"""
AssureX Claim Engine — Task 3 Semantic Duplicate Claim Detection Test Suite
Verifies:
1. Unit tests for DuplicateDetector hybrid similarity calculation.
2. Initial unique claim submits with duplicate_claim_flag = False.
3. Pre-check API endpoint (/api/claims/check-duplicate) identifies semantic duplicates.
4. Second claim with paraphrased fault description triggers duplicate_claim_flag = True,
   hard fraud failure (Auto-Rejected), and audit logging.
5. Distinct fault description on the same product passes duplicate check.
6. Duplicate invoice number triggers duplicate claim detection.
7. Duplicate receipt file hash triggers duplicate claim detection.
"""

import uuid
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from src.main import app
from src.database_setup import engine
from src.models import Product, Warranty, User, Claim, ClaimAuditLog
from src.auth.service import create_access_token, hash_password
from src.services.duplicate_detector import DuplicateDetector

client = TestClient(app)


def setup_duplicate_fixtures():
    with Session(engine) as s:
        # Create test customer
        cust = s.exec(select(User).where(User.username == "test_dup_user")).first()
        if not cust:
            cust = User(
                username="test_dup_user",
                email="dupuser@test.com",
                hashed_password=hash_password("Pass123!"),
                role="customer",
                full_name="Duplicate Test User",
            )
            s.add(cust)
            s.commit()
            s.refresh(cust)

        # Create active test product
        p_id = f"PROD-DUP-{uuid.uuid4().hex[:6].upper()}"
        prod = Product(
            product_id=p_id,
            user_id=cust.id,
            name="QuantumBook Pro 16",
            category="Laptop",
            brand="AssureTech",
            model_number="QB-16-PRO",
            serial_number=f"SN-DUP-{uuid.uuid4().hex[:6].upper()}",
            purchase_date="2025-01-10",
            purchase_price=1899.99,
            retailer="TechSuperstore",
        )
        s.add(prod)
        s.commit()
        s.refresh(prod)

        # Active warranty
        war = Warranty(
            warranty_id=f"WAR-DUP-{uuid.uuid4().hex[:6].upper()}",
            product_id=prod.product_id,
            user_id=cust.id,
            start_date="2025-01-10",
            end_date=(datetime.now() + timedelta(days=365)).strftime("%Y-%m-%d"),
            warranty_type="Manufacturer Standard",
            provider="AssureCare Direct",
            status="Active",
        )
        s.add(war)
        s.commit()

        token = create_access_token({"sub": cust.username, "role": cust.role, "user_id": cust.id})
        return token, prod.product_id, prod.serial_number


def test_similarity_algorithm_unit():
    """Verify hybrid similarity matches paraphrases while ignoring unrelated text."""
    desc_orig = "Screen went completely black with flickering lines and device overheats"
    desc_paraphrase = "The display turned dark and screen flickered while machine became extremely hot"
    desc_unrelated = "Volume rocker button physically jammed and audio jack produces static noise"

    # Paraphrase should exceed 70% threshold
    score_para = DuplicateDetector.compute_similarity(desc_orig, desc_paraphrase)
    assert score_para >= 0.70, f"Expected similarity >= 0.70, got {score_para}"

    # Unrelated should be well below 50%
    score_unrel = DuplicateDetector.compute_similarity(desc_orig, desc_unrelated)
    assert score_unrel < 0.50, f"Expected unrelated similarity < 0.50, got {score_unrel}"


def test_claim_lifecycle_semantic_duplicate_detection():
    token, product_id, serial_number = setup_duplicate_fixtures()
    headers = {"Authorization": f"Bearer {token}"}

    # 1. Submit initial clean claim
    claim1_payload = {
        "product_id": product_id,
        "product_name": "QuantumBook Pro 16",
        "product_category": "Laptop",
        "brand": "AssureTech",
        "model_number": "QB-16-PRO",
        "serial_number_entered": serial_number,
        "purchase_price": 1899.99,
        "retailer": "TechSuperstore",
        "purchase_date": "2025-01-10",
        "warranty_provider": "AssureCare Direct",
        "warranty_type": "Manufacturer Standard",
        "warranty_start": "2025-01-10",
        "warranty_end": (datetime.now() + timedelta(days=365)).strftime("%Y-%m-%d"),
        "fault_date": "2025-03-01",
        "fault_type": "Display Failure",
        "damage_type": "Internal Component Failure",
        "fault_description": "Screen went completely black with flickering lines and device overheats",
        "repair_history_count": 0,
        "previous_repair_authorized": True,
        "receipt_uploaded": True,
        "warranty_card_uploaded": True,
        "product_image_uploaded": True,
        "receipt_hash": f"hash_first_{uuid.uuid4().hex[:12]}",
        "invoice_number": f"INV-DUP-TEST-{uuid.uuid4().hex[:6].upper()}",
    }

    res1 = client.post("/api/claims/submit", json=claim1_payload, headers=headers)
    assert res1.status_code == 201, f"Initial claim failed: {res1.text}"
    claim1_data = res1.json()
    assert claim1_data["duplicate_claim_flag"] is False
    claim1_id = claim1_data["claim_id"]

    # 2. Test pre-check API endpoint for semantic duplicate
    paraphrased_desc = "The display turned dark and screen flickered while machine became extremely hot"
    precheck_res = client.post(
        "/api/claims/check-duplicate",
        json={
            "product_id": product_id,
            "serial_number": serial_number,
            "fault_description": paraphrased_desc,
        },
        headers=headers,
    )
    assert precheck_res.status_code == 200
    precheck_data = precheck_res.json()
    assert precheck_data["is_duplicate"] is True
    assert precheck_data["matched_claim_id"] == claim1_id
    assert precheck_data["similarity_score"] >= 0.70

    # 3. Submit second claim with paraphrased description (must be auto-rejected as duplicate)
    claim2_payload = dict(claim1_payload)
    claim2_payload["fault_description"] = paraphrased_desc
    claim2_payload["receipt_hash"] = f"hash_second_{uuid.uuid4().hex[:12]}"
    claim2_payload["invoice_number"] = f"INV-DUP-TEST-{uuid.uuid4().hex[:6].upper()}"

    res2 = client.post("/api/claims/submit", json=claim2_payload, headers=headers)
    assert res2.status_code == 201
    claim2_data = res2.json()
    assert claim2_data["duplicate_claim_flag"] is True
    assert "semantic similarity" in claim2_data["duplicate_claim_details"].lower()
    assert claim1_id in claim2_data["duplicate_claim_details"]
    assert claim2_data["adjudication_status"] == "Auto-Rejected"

    # Verify audit log
    with Session(engine) as s:
        audit = s.exec(
            select(ClaimAuditLog)
            .where(ClaimAuditLog.claim_id == claim2_data["claim_id"])
            .where(ClaimAuditLog.action == "DUPLICATE_CLAIM_DETECTED")
        ).first()
        assert audit is not None
        assert claim1_id in audit.details

    # 4. Submit distinct fault on same product (must pass duplicate check)
    claim3_payload = dict(claim1_payload)
    claim3_payload["fault_description"] = "Volume rocker button physically jammed and audio jack produces static noise"
    claim3_payload["receipt_hash"] = f"hash_third_{uuid.uuid4().hex[:12]}"
    claim3_payload["invoice_number"] = f"INV-DUP-TEST-{uuid.uuid4().hex[:6].upper()}"

    res3 = client.post("/api/claims/submit", json=claim3_payload, headers=headers)
    assert res3.status_code == 201
    claim3_data = res3.json()
    assert claim3_data["duplicate_claim_flag"] is False


def test_duplicate_invoice_number_detection():
    token, product_id, serial_number = setup_duplicate_fixtures()
    headers = {"Authorization": f"Bearer {token}"}
    shared_invoice = f"INV-SHARED-{uuid.uuid4().hex[:6].upper()}"

    base_payload = {
        "product_id": product_id,
        "product_name": "QuantumBook Pro 16",
        "product_category": "Laptop",
        "brand": "AssureTech",
        "model_number": "QB-16-PRO",
        "serial_number_entered": serial_number,
        "purchase_price": 1899.99,
        "retailer": "TechSuperstore",
        "purchase_date": "2025-01-10",
        "warranty_provider": "AssureCare Direct",
        "warranty_type": "Manufacturer Standard",
        "warranty_start": "2025-01-10",
        "warranty_end": (datetime.now() + timedelta(days=365)).strftime("%Y-%m-%d"),
        "fault_date": "2025-03-01",
        "fault_type": "Trackpad Failure",
        "damage_type": "Internal Component Failure",
        "fault_description": "Trackpad click response completely dead after normal usage",
        "repair_history_count": 0,
        "previous_repair_authorized": True,
        "receipt_uploaded": True,
        "warranty_card_uploaded": True,
        "product_image_uploaded": True,
        "receipt_hash": f"hash_inv1_{uuid.uuid4().hex[:12]}",
        "invoice_number": shared_invoice,
    }

    res1 = client.post("/api/claims/submit", json=base_payload, headers=headers)
    assert res1.status_code == 201
    c1_id = res1.json()["claim_id"]

    # Submit second claim with identical invoice number
    payload2 = dict(base_payload)
    payload2["fault_description"] = "Microphone stopped picking up voice input during meetings"
    payload2["receipt_hash"] = f"hash_inv2_{uuid.uuid4().hex[:12]}"

    res2 = client.post("/api/claims/submit", json=payload2, headers=headers)
    assert res2.status_code == 201
    c2 = res2.json()
    assert c2["duplicate_claim_flag"] is True
    assert shared_invoice in c2["duplicate_claim_details"]
    assert c1_id in c2["duplicate_claim_details"]
    assert c2["adjudication_status"] == "Auto-Rejected"


def test_duplicate_file_hash_detection():
    token, product_id, serial_number = setup_duplicate_fixtures()
    headers = {"Authorization": f"Bearer {token}"}
    shared_hash = f"hash_shared_{uuid.uuid4().hex[:16]}"

    base_payload = {
        "product_id": product_id,
        "product_name": "QuantumBook Pro 16",
        "product_category": "Laptop",
        "brand": "AssureTech",
        "model_number": "QB-16-PRO",
        "serial_number_entered": serial_number,
        "purchase_price": 1899.99,
        "retailer": "TechSuperstore",
        "purchase_date": "2025-01-10",
        "warranty_provider": "AssureCare Direct",
        "warranty_type": "Manufacturer Standard",
        "warranty_start": "2025-01-10",
        "warranty_end": (datetime.now() + timedelta(days=365)).strftime("%Y-%m-%d"),
        "fault_date": "2025-03-01",
        "fault_type": "Battery Failure",
        "damage_type": "Internal Component Failure",
        "fault_description": "Battery completely discharges within fifteen minutes of unplugging",
        "repair_history_count": 0,
        "previous_repair_authorized": True,
        "receipt_uploaded": True,
        "warranty_card_uploaded": True,
        "product_image_uploaded": True,
        "receipt_hash": shared_hash,
        "invoice_number": f"INV-HASH1-{uuid.uuid4().hex[:6].upper()}",
    }

    res1 = client.post("/api/claims/submit", json=base_payload, headers=headers)
    assert res1.status_code == 201
    c1_id = res1.json()["claim_id"]

    # Submit second claim with identical file hash
    payload2 = dict(base_payload)
    payload2["fault_description"] = "Keyboard spacebar key stuck down permanently"
    payload2["invoice_number"] = f"INV-HASH2-{uuid.uuid4().hex[:6].upper()}"

    res2 = client.post("/api/claims/submit", json=payload2, headers=headers)
    assert res2.status_code == 201
    c2 = res2.json()
    assert c2["duplicate_claim_flag"] is True
    assert shared_hash in c2["duplicate_claim_details"]
    assert c1_id in c2["duplicate_claim_details"]
    assert c2["adjudication_status"] == "Auto-Rejected"


if __name__ == "__main__":
    test_similarity_algorithm_unit()
    print("[PASS] 1/4 Unit similarity calculation")
    test_claim_lifecycle_semantic_duplicate_detection()
    print("[PASS] 2/4 Semantic duplicate lifecycle & pre-check")
    test_duplicate_invoice_number_detection()
    print("[PASS] 3/4 Duplicate invoice detection")
    test_duplicate_file_hash_detection()
    print("[PASS] 4/4 Duplicate file hash detection")
    print("\n>>> ALL TASK 3 SEMANTIC DUPLICATE TESTS PASSED 100%! <<<")

