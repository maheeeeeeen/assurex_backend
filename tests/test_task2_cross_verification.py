"""
AssureX Claim Engine — Task 2 Verification Tests
Tests Cross-Document Serial & Model Number Verification across:
1. OCR extraction of serial & model numbers from receipts, warranty cards, and barcodes.
2. OCRService cross-verification logic (exact/partial match, mismatches, inter-document contradictions).
3. Real-time POST /api/claims/cross-verify endpoint.
4. End-to-end claim submission with cross-document mismatch triggering review flags and manual review routing.
5. End-to-end claim submission with consistent cross-documents passing Rule 3.
"""

import io
import json
from fastapi.testclient import TestClient
from sqlmodel import Session, select
from PIL import Image, ImageDraw

from src.main import app
from src.database_setup import engine
from src.models import User, Product, Warranty, Claim
from src.auth.service import hash_password
from src.services.ocr_service import OCRService

client = TestClient(app)


def get_token(username="customer_mike", password="Customer@12345", role="customer"):
    with Session(engine) as session:
        user = session.exec(select(User).where(User.username == username)).first()
        if not user:
            user = User(
                username=username,
                email=f"{username}@example.com",
                hashed_password=hash_password(password),
                role=role,
                full_name=f"Test {username}",
            )
            session.add(user)
            session.commit()
            session.refresh(user)

    res = client.post("/api/auth/login", json={"username": username, "password": password})
    assert res.status_code == 200, f"Login failed for {username}"
    return res.json()["access_token"], user.id


def test_ocr_model_and_serial_parsing():
    """Validates regex extraction of model numbers and serial numbers."""
    text_receipt = (
        "BEST BUY STORE #104\n"
        "Date: 2025-03-10\n"
        "Product: Sony WH-1000XM5\n"
        "Model: WH1000XM5-BLK\n"
        "SN: SN-SONY-99120\n"
        "Total: $399.99"
    )
    parsed = OCRService._parse_document_text(text_receipt, doc_type="receipt")
    assert parsed["serial_number"] == "SN-SONY-99120"
    assert parsed["model_number"] == "WH1000XM5-BLK"

    text_card = (
        "OFFICIAL MANUFACTURER WARRANTY CARD\n"
        "Model Number: WH1000XM5-BLK\n"
        "Serial No: SN-SONY-99120\n"
        "Coverage: 24 Months"
    )
    parsed_card = OCRService._parse_document_text(text_card, doc_type="warranty_card")
    assert parsed_card["serial_number"] == "SN-SONY-99120"
    assert parsed_card["model_number"] == "WH1000XM5-BLK"
    print("[Pass] Regex parsing for model and serial numbers verified.")


def test_ocr_cross_verify_all_logic():
    """Tests multi-source cross verification combinations."""
    # 1. Exact Match across all documents
    recon = OCRService.cross_verify_all(
        entered_serial="SN-SONY-99120",
        entered_model="WH1000XM5-BLK",
        receipt_data={"serial_number": "SN-SONY-99120", "model_number": "WH1000XM5-BLK"},
        warranty_card_data={"serial_number": "SN-SONY-99120", "model_number": "WH1000XM5-BLK"},
        barcode_data={"serial_number": "SN-SONY-99120", "model_number": "WH1000XM5-BLK"},
    )
    assert recon["has_mismatch"] is False
    assert recon["has_serial_mismatch"] is False
    assert recon["has_model_mismatch"] is False
    assert recon["total_matches"] >= 6
    assert recon["status"] == "Verified Match"

    # 2. Entered Serial mismatch vs Receipt
    recon_sn_mismatch = OCRService.cross_verify_all(
        entered_serial="SN-DIFFERENT-1111",
        entered_model="WH1000XM5-BLK",
        receipt_data={"serial_number": "SN-SONY-99120", "model_number": "WH1000XM5-BLK"},
    )
    assert recon_sn_mismatch["has_mismatch"] is True
    assert recon_sn_mismatch["has_serial_mismatch"] is True
    assert recon_sn_mismatch["has_model_mismatch"] is False
    assert any("Serial mismatch" in m["message"] for m in recon_sn_mismatch["mismatches"])

    # 3. Inter-document model mismatch (Receipt vs Warranty Card)
    recon_mod_mismatch = OCRService.cross_verify_all(
        entered_serial="SN-SONY-99120",
        entered_model="WH1000XM5-BLK",
        receipt_data={"serial_number": "SN-SONY-99120", "model_number": "WH1000XM5-BLK"},
        warranty_card_data={"serial_number": "SN-SONY-99120", "model_number": "WH1000XM4-SILVER"},
    )
    assert recon_mod_mismatch["has_mismatch"] is True
    assert recon_mod_mismatch["has_model_mismatch"] is True
    assert any("Cross-document model mismatch" in m["message"] for m in recon_mod_mismatch["mismatches"])
    print("[Pass] OCRService cross_verify_all logic verified across match and mismatch cases.")


def test_cross_verify_api_endpoint():
    """Tests POST /api/claims/cross-verify endpoint."""
    token, _ = get_token("customer_mike")
    headers = {"Authorization": f"Bearer {token}"}

    payload = {
        "entered_serial": "SN-SONY-99120",
        "entered_model": "WH1000XM5-BLK",
        "receipt_serial": "SN-SONY-99120",
        "receipt_model": "WH1000XM5-BLK",
        "warranty_card_serial": "SN-WRONG-8888",
        "warranty_card_model": "WH1000XM5-BLK",
    }
    res = client.post("/api/claims/cross-verify", json=payload, headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert data["has_mismatch"] is True
    assert data["has_serial_mismatch"] is True
    assert data["total_mismatches"] >= 1
    assert any("Warranty Card" in m["message"] for m in data["mismatches"])
    print("[Pass] POST /api/claims/cross-verify endpoint verified.")


def test_claim_submission_with_mismatch_triggers_manual_review():
    """Submitting a claim with cross-document mismatch routes to Manual Review Required."""
    token, user_id = get_token("customer_cross_user")
    headers = {"Authorization": f"Bearer {token}"}

    # Setup valid product and warranty
    with Session(engine) as session:
        product = session.exec(select(Product).where(Product.product_id == "PRD-TASK2-01")).first()
        if not product:
            product = Product(
                product_id="PRD-TASK2-01",
                user_id=user_id,
                name="Sony Wireless Headphones",
                category="Electronics",
                brand="Sony",
                model_number="WH1000XM5",
                serial_number="SN-TASK2-TRUE",
                purchase_price=399.99,
                retailer="Best Buy",
                purchase_date="2025-03-01",
                status="Active",
            )
            session.add(product)
            session.commit()
            session.refresh(product)
        elif product.user_id != user_id:
            product.user_id = user_id
            session.add(product)
            session.commit()

        warranty = session.exec(select(Warranty).where(Warranty.product_id == "PRD-TASK2-01")).first()
        if not warranty:
            warranty = Warranty(
                warranty_id="WAR-TASK2-01",
                product_id="PRD-TASK2-01",
                user_id=user_id,
                coverage_type="Manufacturer Standard",
                start_date="2025-03-01",
                end_date="2027-03-01",
                status="Active",
                provider="Sony Care",
            )
            session.add(warranty)
            session.commit()

    claim_payload = {
        "product_id": "PRD-TASK2-01",
        "product_name": "Sony Wireless Headphones",
        "product_category": "Electronics",
        "brand": "Sony",
        "model_number": "WH1000XM5",
        "serial_number_entered": "SN-TASK2-TRUE",
        "purchase_price": 399.99,
        "retailer": "Best Buy",
        "purchase_date": "2025-03-01",
        "warranty_start": "2025-03-01",
        "warranty_end": "2027-03-01",
        "warranty_provider": "Sony Care",
        "warranty_type": "Manufacturer Standard",
        "fault_date": "2026-05-10",
        "fault_type": "Audio Distortion",
        "damage_type": "Internal Component Failure",
        "fault_description": "Crackling sound in right ear cup during playback.",
        "receipt_uploaded": True,
        "receipt_path": "uploads/receipts/test_receipt.jpg",
        "serial_number_on_receipt": "SN-TASK2-TRUE",
        "model_number_on_receipt": "WH1000XM5",
        "warranty_card_uploaded": True,
        "warranty_card_path": "uploads/evidence/warranty_cards/test_card.pdf",
        # Deliberate mismatch on warranty card serial number!
        "serial_number_on_warranty_card": "SN-CONTRADICTION-9999",
        "model_number_on_warranty_card": "WH1000XM5",
        "product_image_uploaded": True,
        "fault_evidence_uploaded": True,
    }

    res = client.post("/api/claims/submit", json=claim_payload, headers=headers)
    assert res.status_code == 201
    claim = res.json()
    cid = claim["claim_id"]

    # Must flag mismatch
    assert claim["serial_mismatch_flag"] is True
    assert claim["adjudication_status"] in ["Manual Review Required", "Auto-Rejected"]
    assert claim["cross_verification_json"] is not None

    cross_data = json.loads(claim["cross_verification_json"])
    assert cross_data["has_serial_mismatch"] is True

    # Check dossier retrieval contains cross-verification details and audit log
    dossier_res = client.get(f"/api/claims/{cid}", headers=headers)
    assert dossier_res.status_code == 200
    dossier = dossier_res.json()
    assert "cross_verification" in dossier
    assert dossier["cross_verification"]["has_serial_mismatch"] is True

    # Rule evaluation must show the mismatch review flag
    rule_eval = dossier["rule_evaluation"]
    assert any("Serial mismatch" in flag for flag in rule_eval.get("review_flags", []))
    print("[Pass] Claim with cross-document mismatch routed to review with discrepancy flags.")


def test_claim_submission_clean_match():
    """Consistent document serials and models pass Rule 3 validation."""
    token, user_id = get_token("customer_cross_user")
    headers = {"Authorization": f"Bearer {token}"}

    claim_payload = {
        "product_id": "PRD-TASK2-01",
        "product_name": "Sony Wireless Headphones",
        "product_category": "Electronics",
        "brand": "Sony",
        "model_number": "WH1000XM5",
        "serial_number_entered": "SN-TASK2-TRUE",
        "purchase_price": 399.99,
        "retailer": "Best Buy",
        "purchase_date": "2025-03-01",
        "warranty_start": "2025-03-01",
        "warranty_end": "2027-03-01",
        "warranty_provider": "Sony Care",
        "warranty_type": "Manufacturer Standard",
        "fault_date": "2026-05-10",
        "fault_type": "Audio Distortion",
        "damage_type": "Internal Component Failure",
        "fault_description": "Crackling sound in right ear cup during playback.",
        "receipt_uploaded": True,
        "receipt_path": "uploads/receipts/test_receipt.jpg",
        "serial_number_on_receipt": "SN-TASK2-TRUE",
        "model_number_on_receipt": "WH1000XM5",
        "warranty_card_uploaded": True,
        "warranty_card_path": "uploads/evidence/warranty_cards/test_card.pdf",
        "serial_number_on_warranty_card": "SN-TASK2-TRUE",
        "model_number_on_warranty_card": "WH1000XM5",
        "product_image_uploaded": True,
        "fault_evidence_uploaded": True,
    }

    res = client.post("/api/claims/submit", json=claim_payload, headers=headers)
    assert res.status_code == 201
    claim = res.json()
    assert claim["serial_mismatch_flag"] is False

    cross_data = json.loads(claim["cross_verification_json"])
    assert cross_data["has_mismatch"] is False

    dossier = client.get(f"/api/claims/{claim['claim_id']}", headers=headers).json()
    rule_eval = dossier["rule_evaluation"]
    assert any("Cross-document verification passed" in chk for chk in rule_eval.get("passed_checks", []))
    print("[Pass] Clean claim verified with Cross-document verification passed in passed_checks.")


if __name__ == "__main__":
    test_ocr_model_and_serial_parsing()
    test_ocr_cross_verify_all_logic()
    test_cross_verify_api_endpoint()
    test_claim_submission_with_mismatch_triggers_manual_review()
    test_claim_submission_clean_match()
    print("\n>>> ALL TASK 2 TESTS COMPLETED WITH 100% SUCCESS! <<<")
