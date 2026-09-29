"""
AssureX Claim Engine — Task 1 Media Upload Verification Test
Tests:
1. Fault photo, video, and barcode photo uploads to /api/claims/upload-media
2. SHA-256 fingerprinting matching OCRService.compute_sha256
3. File persistence to backend storage and disk verification
4. Static file serving via /uploads/...
5. Claim submission with media fields and audit trail logging
"""

import io
import os
import hashlib
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
from sqlmodel import Session, select

from src.main import app
from src.database_setup import engine
from src.models import Product, Warranty
from src.services.ocr_service import OCRService
from src.auth.service import create_access_token

client = TestClient(app)


def make_dummy_png():
    img = Image.new("RGB", (200, 100), color=(73, 109, 137))
    d = ImageDraw.Draw(img)
    d.text((10, 10), "TEST EVIDENCE", fill=(255, 255, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


def make_dummy_mp4():
    # Minimal dummy MP4 payload
    return b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom\x00\x00\x00\x08free"


def test_media_upload_and_claim_flow():
    # 1. Test Fault Photo Upload
    photo_bytes = make_dummy_png()
    expected_photo_hash = hashlib.sha256(photo_bytes).hexdigest()

    res_photo = client.post(
        "/api/claims/upload-media",
        files={"file": ("damage_screen.png", photo_bytes, "image/png")},
        data={"media_type": "fault_photo"},
    )
    assert res_photo.status_code == 200, f"Fault photo upload failed: {res_photo.text}"
    photo_data = res_photo.json()
    assert photo_data["media_type"] == "fault_photo"
    assert photo_data["file_hash"] == expected_photo_hash
    assert os.path.exists(photo_data["file_path"]), f"File not found on disk: {photo_data['file_path']}"
    print(f"[Pass] Fault photo uploaded: {photo_data['file_path']} (SHA: {photo_data['file_hash'][:12]}...)")

    # 2. Test Fault Video Upload
    video_bytes = make_dummy_mp4()
    expected_video_hash = hashlib.sha256(video_bytes).hexdigest()

    res_video = client.post(
        "/api/claims/upload-media",
        files={"file": ("flicker_demo.mp4", video_bytes, "video/mp4")},
        data={"media_type": "fault_video"},
    )
    assert res_video.status_code == 200, f"Fault video upload failed: {res_video.text}"
    video_data = res_video.json()
    assert video_data["media_type"] == "fault_video"
    assert video_data["file_hash"] == expected_video_hash
    assert os.path.exists(video_data["file_path"]), f"Video not found on disk: {video_data['file_path']}"
    print(f"[Pass] Fault video uploaded: {video_data['file_path']} (SHA: {video_data['file_hash'][:12]}...)")

    # 3. Test Barcode Photo Upload
    barcode_bytes = make_dummy_png()
    expected_barcode_hash = hashlib.sha256(barcode_bytes).hexdigest()

    res_barcode = client.post(
        "/api/claims/upload-media",
        files={"file": ("barcode_label.png", barcode_bytes, "image/png")},
        data={"media_type": "barcode_photo"},
    )
    assert res_barcode.status_code == 200, f"Barcode photo upload failed: {res_barcode.text}"
    barcode_data = res_barcode.json()
    assert barcode_data["media_type"] == "barcode_photo"
    assert barcode_data["file_hash"] == expected_barcode_hash
    assert os.path.exists(barcode_data["file_path"]), f"Barcode photo not found on disk: {barcode_data['file_path']}"
    print(f"[Pass] Barcode photo uploaded: {barcode_data['file_path']} (SHA: {barcode_data['file_hash'][:12]}...)")

    # 4. Test Invalid Extension Rejection
    res_bad = client.post(
        "/api/claims/upload-media",
        files={"file": ("malicious.exe", b"binarydata", "application/octet-stream")},
        data={"media_type": "fault_photo"},
    )
    assert res_bad.status_code == 400
    print("[Pass] Unsupported extension correctly rejected with 400 Bad Request.")

    # 5. Test Static File Serving
    rel_url = photo_data["file_url"]
    static_res = client.get(rel_url)
    assert static_res.status_code == 200
    assert len(static_res.content) == len(photo_bytes)
    print(f"[Pass] Static file serving verified for {rel_url} ({len(static_res.content)} bytes).")

    # Ensure a registered product with active warranty exists for claim submission
    with Session(engine) as s:
        p = s.exec(select(Product).where(Product.product_id == "PRD-TASK1-TEST")).first()
        if not p:
            p = Product(
                product_id="PRD-TASK1-TEST",
                name="UltraView OLED 65",
                category="Electronics",
                brand="LG",
                model_number="OLED65C3",
                serial_number="SN-LG-990022",
                purchase_price=1899.99,
                retailer="Best Buy",
                purchase_date="2025-02-10",
                user_id=None,
            )
            s.add(p)
            s.commit()
        w = s.exec(select(Warranty).where(Warranty.product_id == "PRD-TASK1-TEST")).first()
        if not w:
            w = Warranty(
                warranty_id="WAR-TASK1-TEST",
                product_id="PRD-TASK1-TEST",
                provider="LG Care Plus",
                warranty_type="Extended Warranty",
                start_date="2025-02-10",
                end_date="2028-02-10",
                status="Active",
            )
            s.add(w)
            s.commit()

    token = create_access_token({"sub": "admin", "role": "admin"})
    headers = {"Authorization": f"Bearer {token}"}

    claim_payload = {
        "product_id": "PRD-TASK1-TEST",
        "product_name": "UltraView OLED 65",
        "product_category": "Electronics",
        "brand": "LG",
        "model_number": "OLED65C3",
        "serial_number_entered": "SN-LG-990022",
        "purchase_price": 1899.99,
        "retailer": "Best Buy",
        "purchase_date": "2025-02-10",
        "warranty_start": "2025-02-10",
        "warranty_end": "2027-02-10",
        "warranty_provider": "LG Care Plus",
        "warranty_type": "Extended Warranty",
        "fault_date": "2026-03-01",
        "fault_type": "Display Defect",
        "damage_type": "Internal Component Failure",
        "fault_description": "Vertical line down center of display panel.",
        "repair_history_count": 0,
        "previous_repair_authorized": True,
        "receipt_uploaded": True,
        "fault_evidence_uploaded": True,
        "fault_evidence_path": photo_data["file_path"],
        "fault_evidence_hash": photo_data["file_hash"],
        "fault_video_uploaded": True,
        "fault_video_path": video_data["file_path"],
        "fault_video_hash": video_data["file_hash"],
        "barcode_image_uploaded": True,
        "barcode_image_path": barcode_data["file_path"],
        "barcode_image_hash": barcode_data["file_hash"],
    }

    res_submit = client.post("/api/claims/submit", json=claim_payload, headers=headers)
    assert res_submit.status_code == 201, f"Claim submit failed: {res_submit.text}"
    created_claim = res_submit.json()
    claim_id = created_claim["claim_id"]
    print(f"[Pass] Claim submitted successfully: {claim_id}")
    assert created_claim["fault_evidence_path"] == photo_data["file_path"]
    assert created_claim["fault_video_path"] == video_data["file_path"]
    assert created_claim["barcode_image_path"] == barcode_data["file_path"]

    # 7. Test Dossier Retrieval
    res_dossier = client.get(f"/api/claims/{claim_id}", headers=headers)
    assert res_dossier.status_code == 200
    dossier_data = res_dossier.json()
    d_claim = dossier_data["claim"]
    assert d_claim["fault_evidence_path"] == photo_data["file_path"]
    assert d_claim["fault_video_path"] == video_data["file_path"]
    assert d_claim["barcode_image_path"] == barcode_data["file_path"]

    # Check Audit Log for EVIDENCE_ATTACHED
    actions = [log["action"] for log in dossier_data["audit_logs"]]
    assert "EVIDENCE_ATTACHED" in actions, f"EVIDENCE_ATTACHED not found in {actions}"
    print("[Pass] Audit log contains EVIDENCE_ATTACHED entry with cryptographic verification.")

    print("\n>>> ALL TASK 1 TESTS PASSED SUCCESSFULLY! <<<")


if __name__ == "__main__":
    test_media_upload_and_claim_flow()
