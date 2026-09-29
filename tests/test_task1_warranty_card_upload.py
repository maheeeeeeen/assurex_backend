"""
AssureX Claim Engine — Task 1 Warranty Card Upload Test Suite

Validates that:
- Warranty card file uploads (PNG, PDF) are supported via POST /api/claims/upload-media with media_type="warranty_card"
- SHA-256 fingerprint is computed and returned
- File is saved to storage and served statically
- Claim submitted with warranty card persists warranty_card_uploaded, warranty_card_path, and warranty_card_hash
- Dossier endpoint GET /api/claims/{claim_id} returns warranty card metadata for display and download
"""

import io
import os
import hashlib
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from src.main import app
from src.database_setup import engine
from src.models import User, Product, Warranty
from src.auth.service import hash_password

client = TestClient(app)


def test_warranty_card_upload_and_persistence():
    # 1. Authenticate as customer
    with Session(engine) as session:
        user = session.exec(select(User).where(User.username == "customer_mike")).first()
        if not user:
            user = User(
                username="customer_mike",
                email="mike@example.com",
                hashed_password=hash_password("Customer@12345"),
                role="customer",
                full_name="Mike Customer",
            )
            session.add(user)
            session.commit()
            session.refresh(user)

        # Ensure an active product and warranty exist for this customer
        product = session.exec(select(Product).where(Product.user_id == user.id)).first()
        if not product:
            product = Product(
                product_id="PRD-WAR-TEST01",
                user_id=user.id,
                name="Galaxy Tab S9",
                category="Smartphones and Mobile",
                brand="Samsung",
                model_number="SM-X710",
                serial_number="SN-SAMS-7100",
                purchase_price=799.99,
                retailer="Best Buy",
                purchase_date="2025-01-15",
                status="Active",
            )
            session.add(product)
            session.commit()
            session.refresh(product)

        warranty = session.exec(select(Warranty).where(Warranty.product_id == product.product_id)).first()
        if not warranty:
            warranty = Warranty(
                warranty_id="WAR-TEST01",
                product_id=product.product_id,
                coverage_type="Standard",
                start_date="2025-01-15",
                end_date="2027-01-15",
                status="Active",
                provider="Manufacturer",
            )
            session.add(warranty)
            session.commit()
            session.refresh(warranty)

        prod_id = product.product_id
        user_uid = user.id

    # Login
    login_res = client.post(
        "/api/auth/login",
        json={"username": "customer_mike", "password": "Customer@12345"},
    )
    assert login_res.status_code == 200, f"Login failed: {login_res.text}"
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Upload a warranty card (simulated PDF)
    fake_pdf_content = b"%PDF-1.4\n%Official AssureX Warranty Certificate\nSerial: SN-SAMS-7100\n%%EOF"
    expected_hash = hashlib.sha256(fake_pdf_content).hexdigest()

    upload_res = client.post(
        "/api/claims/upload-media",
        data={"media_type": "warranty_card"},
        files={"file": ("warranty_certificate.pdf", io.BytesIO(fake_pdf_content), "application/pdf")},
        headers=headers,
    )
    assert upload_res.status_code == 200, f"Upload failed: {upload_res.text}"
    upload_data = upload_res.json()

    assert upload_data["file_hash"] == expected_hash, f"Hash mismatch: {upload_data['file_hash']} vs {expected_hash}"
    assert "warranty_cards" in upload_data["file_path"].replace("\\", "/")
    assert upload_data["media_type"] == "warranty_card"
    print(f"[Pass] Warranty card PDF uploaded successfully: {upload_data['file_path']} (SHA: {upload_data['file_hash']})")

    # 3. Check static file accessibility
    clean_path = upload_data["file_path"].replace("\\", "/")
    static_url = f"/{clean_path}" if not clean_path.startswith("/") else clean_path
    get_file_res = client.get(static_url)
    assert get_file_res.status_code == 200, f"Static fetch failed for {static_url}"
    assert get_file_res.content == fake_pdf_content
    print(f"[Pass] Static serving verified for uploaded warranty card ({len(get_file_res.content)} bytes).")

    # 4. Submit claim with warranty card attached
    claim_payload = {
        "product_id": prod_id,
        "product_name": "Galaxy Tab S9",
        "product_category": "Smartphones and Mobile",
        "brand": "Samsung",
        "model_number": "SM-X710",
        "serial_number_entered": "SN-SAMS-7100",
        "purchase_price": 799.99,
        "retailer": "Best Buy",
        "purchase_date": "2025-01-15",
        "warranty_start": "2025-01-15",
        "warranty_end": "2027-01-15",
        "warranty_provider": "Manufacturer",
        "warranty_type": "Standard",
        "fault_date": "2025-08-10",
        "fault_type": "Screen Failure",
        "damage_type": "Internal Component Failure",
        "fault_description": "Display flickers and shows vertical white lines during normal usage.",
        "repair_history_count": 0,
        "previous_repair_authorized": False,
        "receipt_uploaded": True,
        "warranty_card_uploaded": True,
        "warranty_card_path": upload_data["file_path"],
        "warranty_card_hash": upload_data["file_hash"],
        "product_image_uploaded": True,
        "fault_evidence_uploaded": True,
    }

    sub_res = client.post("/api/claims/submit", json=claim_payload, headers=headers)
    assert sub_res.status_code in (200, 201), f"Claim submission failed: {sub_res.text}"
    claim_result = sub_res.json()
    new_cid = claim_result["claim_id"]
    print(f"[Pass] Claim submitted with warranty card: {new_cid}")

    # 5. Fetch claim dossier and verify warranty card fields
    dossier_res = client.get(f"/api/claims/{new_cid}", headers=headers)
    assert dossier_res.status_code == 200, f"Dossier retrieval failed: {dossier_res.text}"
    dossier = dossier_res.json()
    claim_record = dossier["claim"]

    assert claim_record["warranty_card_uploaded"] is True
    assert claim_record["warranty_card_path"] == upload_data["file_path"]
    assert claim_record["warranty_card_hash"] == upload_data["file_hash"]
    print(f"[Pass] Claim dossier confirmed warranty card persistence: Path={claim_record['warranty_card_path']}, Hash={claim_record['warranty_card_hash']}")

    print("\n>>> ALL TASK 1 TESTS PASSED SUCCESSFULLY! <<<")


if __name__ == "__main__":
    test_warranty_card_upload_and_persistence()
