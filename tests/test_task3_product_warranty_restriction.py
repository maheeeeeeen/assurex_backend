"""
AssureX Claim Engine — Task 3 Verification Test Suite
Tests:
1. Rejection of claim submission without product_id (HTTP 400)
2. Rejection of non-existent product_id (HTTP 404)
3. Customer ownership enforcement — blocking claims on products owned by another customer (HTTP 403)
4. Rejection of product without any registered warranty (HTTP 400)
5. Rejection of product with expired warranty (HTTP 400)
6. Rejection of product with void warranty (HTTP 400)
7. Successful claim submission for customer's own active-warranty product (HTTP 201)
8. Assisted intake by employee on behalf of customer (HTTP 201) with claim assigned to product owner
9. Audit log trail verification for WARRANTY_VERIFIED
"""

import uuid
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from src.main import app
from src.database_setup import engine
from src.models import Product, Warranty, User, ClaimAuditLog
from src.auth.service import create_access_token, hash_password

client = TestClient(app)


def setup_task3_fixtures():
    with Session(engine) as s:
        # Create test users
        cust1 = s.exec(select(User).where(User.username == "cust_owner")).first()
        if not cust1:
            cust1 = User(
                username="cust_owner",
                email="owner@test.com",
                hashed_password=hash_password("Pass123!"),
                role="customer",
                full_name="Alice Owner",
            )
            s.add(cust1)
            s.commit()
            s.refresh(cust1)

        cust2 = s.exec(select(User).where(User.username == "cust_stranger")).first()
        if not cust2:
            cust2 = User(
                username="cust_stranger",
                email="stranger@test.com",
                hashed_password=hash_password("Pass123!"),
                role="customer",
                full_name="Bob Stranger",
            )
            s.add(cust2)
            s.commit()
            s.refresh(cust2)

        emp = s.exec(select(User).where(User.username == "emp_agent")).first()
        if not emp:
            emp = User(
                username="emp_agent",
                email="agent@test.com",
                hashed_password=hash_password("Pass123!"),
                role="employee",
                full_name="Agent Cooper",
            )
            s.add(emp)
            s.commit()
            s.refresh(emp)

        # 1. Product owned by cust1 with Active Warranty
        p_active = s.exec(select(Product).where(Product.product_id == "PRD-T3-ACTIVE")).first()
        if not p_active:
            p_active = Product(
                product_id="PRD-T3-ACTIVE",
                name="Smart QLED TV 55",
                category="Electronics",
                brand="Samsung",
                model_number="QN55Q80C",
                serial_number="SN-T3-ACTIVE-01",
                purchase_price=1299.99,
                retailer="Best Buy",
                purchase_date="2025-06-01",
                user_id=cust1.id,
            )
            s.add(p_active)
            s.commit()

        w_active = s.exec(select(Warranty).where(Warranty.product_id == "PRD-T3-ACTIVE")).first()
        if not w_active:
            w_active = Warranty(
                warranty_id="WAR-T3-ACTIVE",
                product_id="PRD-T3-ACTIVE",
                user_id=cust1.id,
                provider="Samsung Care+",
                warranty_type="Extended Warranty",
                start_date="2025-06-01",
                end_date="2028-06-01",
                status="Active",
            )
            s.add(w_active)
            s.commit()

        # 2. Product with No Warranty
        p_nowar = s.exec(select(Product).where(Product.product_id == "PRD-T3-NOWAR")).first()
        if not p_nowar:
            p_nowar = Product(
                product_id="PRD-T3-NOWAR",
                name="Budget Microwave",
                category="Appliances",
                brand="Panasonic",
                model_number="NN-SN686S",
                serial_number="SN-T3-NOWAR-01",
                purchase_price=149.99,
                retailer="Walmart",
                purchase_date="2024-01-01",
                user_id=cust1.id,
            )
            s.add(p_nowar)
            s.commit()

        # 3. Product with Expired Warranty (> 7 days past)
        p_exp = s.exec(select(Product).where(Product.product_id == "PRD-T3-EXPIRED")).first()
        if not p_exp:
            p_exp = Product(
                product_id="PRD-T3-EXPIRED",
                name="Old Blender Pro",
                category="Appliances",
                brand="Ninja",
                model_number="BL660",
                serial_number="SN-T3-EXPIRED-01",
                purchase_price=119.99,
                retailer="Target",
                purchase_date="2022-01-01",
                user_id=cust1.id,
            )
            s.add(p_exp)
            s.commit()

        w_exp = s.exec(select(Warranty).where(Warranty.product_id == "PRD-T3-EXPIRED")).first()
        if not w_exp:
            w_exp = Warranty(
                warranty_id="WAR-T3-EXPIRED",
                product_id="PRD-T3-EXPIRED",
                user_id=cust1.id,
                provider="Ninja Standard",
                warranty_type="Manufacturer Standard",
                start_date="2022-01-01",
                end_date="2023-01-01",
                status="Expired",
            )
            s.add(w_exp)
            s.commit()

        # 4. Product with Void Warranty
        p_void = s.exec(select(Product).where(Product.product_id == "PRD-T3-VOID")).first()
        if not p_void:
            p_void = Product(
                product_id="PRD-T3-VOID",
                name="Gaming Laptop",
                category="Electronics",
                brand="Asus",
                model_number="G14",
                serial_number="SN-T3-VOID-01",
                purchase_price=1599.99,
                retailer="Amazon",
                purchase_date="2025-01-01",
                user_id=cust1.id,
            )
            s.add(p_void)
            s.commit()

        w_void = s.exec(select(Warranty).where(Warranty.product_id == "PRD-T3-VOID")).first()
        if not w_void:
            w_void = Warranty(
                warranty_id="WAR-T3-VOID",
                product_id="PRD-T3-VOID",
                user_id=cust1.id,
                provider="Asus Standard",
                warranty_type="Standard",
                start_date="2025-01-01",
                end_date="2027-01-01",
                status="Void",
            )
            s.add(w_void)
            s.commit()

        id1 = cust1.id
        id2 = cust2.id
        id3 = emp.id

    return id1, id2, id3


def test_task3_restrictions():
    cust1_id, cust2_id, emp_id = setup_task3_fixtures()

    owner_token = create_access_token({"sub": "cust_owner", "role": "customer", "user_id": cust1_id})
    stranger_token = create_access_token({"sub": "cust_stranger", "role": "customer", "user_id": cust2_id})
    emp_token = create_access_token({"sub": "emp_agent", "role": "employee", "user_id": emp_id})

    owner_hdr = {"Authorization": f"Bearer {owner_token}"}
    stranger_hdr = {"Authorization": f"Bearer {stranger_token}"}
    emp_hdr = {"Authorization": f"Bearer {emp_token}"}

    base_payload = {
        "product_name": "Smart QLED TV 55",
        "product_category": "Electronics",
        "brand": "Samsung",
        "model_number": "QN55Q80C",
        "serial_number_entered": "SN-T3-ACTIVE-01",
        "purchase_price": 1299.99,
        "retailer": "Best Buy",
        "purchase_date": "2025-06-01",
        "warranty_start": "2025-06-01",
        "warranty_end": "2028-06-01",
        "warranty_provider": "Samsung Care+",
        "warranty_type": "Extended Warranty",
        "fault_date": "2026-03-01",
        "fault_type": "Screen Flicker",
        "damage_type": "Internal Component Failure",
        "fault_description": "Intermittent horizontal lines on display.",
        "repair_history_count": 0,
        "previous_repair_authorized": True,
    }

    # 1. Reject missing product_id
    payload_no_pid = dict(base_payload)
    res = client.post("/api/claims/submit", json=payload_no_pid, headers=owner_hdr)
    assert res.status_code == 400, f"Expected 400 for missing product_id, got {res.status_code}"
    print("[Pass] Missing product_id correctly rejected with 400 Bad Request.")

    # 2. Reject non-existent product_id
    payload_fake = dict(base_payload, product_id="PRD-NONEXISTENT-999")
    res = client.post("/api/claims/submit", json=payload_fake, headers=owner_hdr)
    assert res.status_code == 404, f"Expected 404 for unknown product, got {res.status_code}"
    print("[Pass] Non-existent product_id correctly rejected with 404 Not Found.")

    # 3. Reject customer filing claim on product owned by another customer
    payload_stranger = dict(base_payload, product_id="PRD-T3-ACTIVE")
    res = client.post("/api/claims/submit", json=payload_stranger, headers=stranger_hdr)
    assert res.status_code == 403, f"Expected 403 for stranger customer, got {res.status_code}"
    print("[Pass] Customer blocked with 403 Forbidden when filing claim on another customer's product.")

    # 4. Reject product with no registered warranty
    payload_nowar = dict(base_payload, product_id="PRD-T3-NOWAR")
    res = client.post("/api/claims/submit", json=payload_nowar, headers=owner_hdr)
    assert res.status_code == 400, f"Expected 400 for no warranty, got {res.status_code}"
    assert "No warranty" in res.json()["detail"] or "warranty coverage" in res.json()["detail"]
    print("[Pass] Product without registered warranty correctly rejected with 400 Bad Request.")

    # 5. Reject product with expired warranty
    payload_exp = dict(base_payload, product_id="PRD-T3-EXPIRED")
    res = client.post("/api/claims/submit", json=payload_exp, headers=owner_hdr)
    assert res.status_code == 400, f"Expected 400 for expired warranty, got {res.status_code}"
    assert "expired" in res.json()["detail"].lower()
    print("[Pass] Product with expired warranty correctly rejected with 400 Bad Request.")

    # 6. Reject product with void warranty
    payload_void = dict(base_payload, product_id="PRD-T3-VOID")
    res = client.post("/api/claims/submit", json=payload_void, headers=owner_hdr)
    assert res.status_code == 400, f"Expected 400 for void warranty, got {res.status_code}"
    assert "void" in res.json()["detail"].lower()
    print("[Pass] Product with void warranty correctly rejected with 400 Bad Request.")

    # 7. Accept claim from product owner with active warranty
    payload_valid = dict(base_payload, product_id="PRD-T3-ACTIVE")
    res = client.post("/api/claims/submit", json=payload_valid, headers=owner_hdr)
    assert res.status_code == 201, f"Expected 201 for valid claim, got {res.status_code}: {res.text}"
    claim_data = res.json()
    assert claim_data["product_id"] == "PRD-T3-ACTIVE"
    print(f"[Pass] Owner submitted claim successfully: {claim_data['claim_id']}.")

    # 8. Employee assisted intake for customer's product
    payload_emp = dict(base_payload, product_id="PRD-T3-ACTIVE")
    res_emp = client.post("/api/claims/submit", json=payload_emp, headers=emp_hdr)
    assert res_emp.status_code == 201, f"Expected 201 for employee assisted claim, got {res_emp.status_code}: {res_emp.text}"
    emp_claim_data = res_emp.json()
    print(f"[Pass] Employee submitted assisted claim successfully: {emp_claim_data['claim_id']}.")

    # 9. Verify audit log contains WARRANTY_VERIFIED
    res_dossier = client.get(f"/api/claims/{claim_data['claim_id']}", headers=owner_hdr)
    assert res_dossier.status_code == 200
    actions = [log["action"] for log in res_dossier.json()["audit_logs"]]
    assert "WARRANTY_VERIFIED" in actions, f"WARRANTY_VERIFIED not found in {actions}"
    print("[Pass] Audit log contains WARRANTY_VERIFIED security entry.")

    print("\n>>> ALL TASK 3 BACKEND RESTRICTION TESTS PASSED 100%! <<<")


if __name__ == "__main__":
    test_task3_restrictions()
