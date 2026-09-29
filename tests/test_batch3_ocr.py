"""
AssureX Claim Engine — Batch 3 Verification Test
Tests OCR file processing, SHA-256 fingerprinting, duplicate detection, and serial reconciliation.
"""

import io
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from src.main import app
from src.services.ocr_service import OCRService

client = TestClient(app)


def create_test_image(text="OFFICIAL SALES RECEIPT"):
    img = Image.new("RGB", (400, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), text, fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


def test_ocr_process_endpoint():
    # 1. Create test receipt image
    image_bytes = create_test_image("Store: Best Buy\nDate: 2025-01-15\nSN: SN-TEST-7788\nTotal: $899.99")
    files = {"file": ("test_receipt.png", image_bytes, "image/png")}

    # 2. Call /api/claims/ocr-process
    res = client.post("/api/claims/ocr-process", files=files)
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    data = res.json()

    print("[Test] OCR response keys:", list(data.keys()))
    assert "file_hash" in data
    assert len(data["file_hash"]) == 64
    assert "ocr_confidence" in data
    assert "is_duplicate_file" in data
    assert data["is_duplicate_file"] is False
    print(f"[Test] Successfully extracted: Merchant={data.get('merchant')}, SN={data.get('serial_number')}, Hash={data['file_hash'][:16]}...")

    # 3. Test Serial Reconciliation logic
    recon_match = OCRService.compare_serials("SN-TEST-7788", "SN-TEST-7788")
    assert recon_match["match"] is True
    assert recon_match["status"] == "Exact Match"

    recon_mismatch = OCRService.compare_serials("SN-ENTERED-1234", "SN-TEST-7788")
    assert recon_mismatch["match"] is False
    assert recon_mismatch["status"] == "Mismatch"
    print("[Test] Serial reconciliation logic verified: Match and Mismatch states accurate.")

    print("ALL BATCH 3 TESTS COMPLETED WITH 100% SUCCESS!")


if __name__ == "__main__":
    test_ocr_process_endpoint()
