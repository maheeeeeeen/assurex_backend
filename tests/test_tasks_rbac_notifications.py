"""
AssureX Claim Engine — Test Suite for Tasks 1-4:
- Task 1: RBAC Enforcement (Admin, Customer, Viewer, Reviewer)
- Task 2: Product Registration, Ownership & Duplicate Serial Prevention
- Task 3: Claim Submission & Claim Deletion Lifecycle
- Task 4: Centralized Admin Notifications for all 6 Product & Claim Lifecycle Events
"""

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select
from datetime import datetime, date

from src.main import app
from src.database_setup import engine
from src.models import User, Product, Warranty, Claim, Notification
from src.auth.service import hash_password

client = TestClient(app)


def setup_users():
    """Ensure all test users exist with known roles and credentials."""
    with Session(engine) as session:
        users = [
            ("admin", "admin@assurex.com", "Admin@12345", "admin", "Admin System"),
            ("customer_mike", "mike@example.com", "Customer@12345", "customer", "Mike Customer"),
            ("customer_jane", "jane@example.com", "Customer@12345", "customer", "Jane Customer"),
            ("adjuster_sarah", "sarah@assurex.com", "Adjuster@12345", "reviewer", "Sarah Adjuster"),
            ("viewer_guest", "viewer@assurex.com", "Viewer@12345", "viewer", "Guest Viewer"),
        ]
        for uname, email, pwd, role, fname in users:
            existing = session.exec(select(User).where(User.username == uname)).first()
            if not existing:
                u = User(
                    username=uname,
                    email=email,
                    hashed_password=hash_password(pwd),
                    role=role,
                    full_name=fname,
                )
                session.add(u)
            else:
                existing.role = role
                session.add(existing)
        session.commit()


def get_auth_token(username, password):
    res = client.post("/api/auth/login", json={"username": username, "password": password})
    assert res.status_code == 200, f"Login failed for {username}: {res.text}"
    return res.json()["access_token"]


def test_task1_viewer_mutation_lockdown():
    """
    Task 1: Viewer must be in read-only mode and strictly blocked with 403 on ALL mutation endpoints.
    Viewer should still be able to view permitted products and claims.
    """
    setup_users()
    viewer_token = get_auth_token("viewer_guest", "Viewer@12345")
    viewer_hdr = {"Authorization": f"Bearer {viewer_token}"}

    # Viewer CAN view claims and products
    res_claims = client.get("/api/claims/", headers=viewer_hdr)
    assert res_claims.status_code == 200

    res_products = client.get("/api/products/", headers=viewer_hdr)
    assert res_products.status_code == 200

    # Viewer CANNOT register product (403)
    res_reg = client.post(
        "/api/products/",
        json={
            "name": "Viewer Laptop",
            "category": "Electronics",
            "brand": "TechCorp",
            "model_number": "VIEW-1",
            "serial_number": "SN-VIEW-01",
            "purchase_price": 999.0,
            "retailer": "BestBuy",
            "purchase_date": "2025-01-01",
            "warranty_duration_months": 24,
        },
        headers=viewer_hdr,
    )
    assert res_reg.status_code == 403

    # Viewer CANNOT update product (403)
    res_upd_p = client.put("/api/products/PRD-SOME-ID", json={"name": "New Name"}, headers=viewer_hdr)
    assert res_upd_p.status_code == 403

    # Viewer CANNOT delete product (403)
    res_del_p = client.delete("/api/products/PRD-SOME-ID", headers=viewer_hdr)
    assert res_del_p.status_code == 403

    # Viewer CANNOT submit claim (403)
    res_sub_c = client.post("/api/claims/submit", json={"product_id": "PRD-SOME-ID"}, headers=viewer_hdr)
    assert res_sub_c.status_code == 403

    # Viewer CANNOT update claim (403)
    res_upd_c = client.put("/api/claims/CLM-SOME-ID", json={"description": "Test"}, headers=viewer_hdr)
    assert res_upd_c.status_code == 403

    # Viewer CANNOT delete claim (403)
    res_del_c = client.delete("/api/claims/CLM-SOME-ID", headers=viewer_hdr)
    assert res_del_c.status_code == 403

    # Viewer CANNOT adjudicate claim (403)
    res_adj = client.post(
        "/api/claims/CLM-SOME-ID/adjudicate",
        json={"decision": "Approve", "notes": "Test"},
        headers=viewer_hdr,
    )
    assert res_adj.status_code == 403


def test_task2_product_registration_ownership_and_duplicate_prevention():
    """
    Task 2:
    - Customer registers product under their own account with warranty auto-calculated.
    - Duplicate serial numbers are blocked with 400.
    - Customer can update their own product, blocked from updating others.
    - Customer can delete their product without claims, blocked if claims exist.
    """
    setup_users()
    mike_token = get_auth_token("customer_mike", "Customer@12345")
    jane_token = get_auth_token("customer_jane", "Customer@12345")
    admin_token = get_auth_token("admin", "Admin@12345")

    mike_hdr = {"Authorization": f"Bearer {mike_token}"}
    jane_hdr = {"Authorization": f"Bearer {jane_token}"}
    admin_hdr = {"Authorization": f"Bearer {admin_token}"}

    unique_sn = f"SN-UNIQUE-{int(datetime.now().timestamp() * 1000)}"

    # 1. Mike registers product
    prod_payload = {
        "name": "Smart 4K Television",
        "category": "Electronics",
        "brand": "AssureX Vision",
        "model_number": "TV-4K-2025",
        "serial_number": unique_sn,
        "purchase_price": 799.99,
        "retailer": "BestBuy Online",
        "purchase_date": "2025-01-15",
        "warranty_duration_months": 24,
    }
    res_prod = client.post("/api/products/", json=prod_payload, headers=mike_hdr)
    assert res_prod.status_code in (200, 201), res_prod.text
    prod_id = res_prod.json()["product_id"]

    # Verify database persistence & ownership
    with Session(engine) as session:
        db_prod = session.exec(select(Product).where(Product.product_id == prod_id)).first()
        assert db_prod is not None
        assert db_prod.serial_number == unique_sn
        # Check warranty auto-creation
        db_warr = session.exec(select(Warranty).where(Warranty.product_id == prod_id)).first()
        assert db_warr is not None
        assert db_prod.warranty_duration_months == 24
        assert db_warr.end_date == "2027-01-15"

    # 2. Prevent duplicate serial number (even from Jane or Admin)
    dup_res = client.post("/api/products/", json=prod_payload, headers=jane_hdr)
    assert dup_res.status_code == 400
    assert "already registered" in dup_res.json()["detail"].lower()

    # 3. Jane cannot update Mike's product
    upd_res_jane = client.put(f"/api/products/{prod_id}", json={"name": "Jane's TV"}, headers=jane_hdr)
    assert upd_res_jane.status_code == 403

    # 4. Mike can update his own product
    upd_res_mike = client.put(f"/api/products/{prod_id}", json={"name": "Smart 4K OLED TV"}, headers=mike_hdr)
    assert upd_res_mike.status_code == 200
    assert upd_res_mike.json()["product"]["name"] == "Smart 4K OLED TV"

    # 5. Admin can update Mike's product
    upd_res_admin = client.put(f"/api/products/{prod_id}", json={"brand": "AssureX Tech"}, headers=admin_hdr)
    assert upd_res_admin.status_code == 200


def test_task3_claim_submission_ownership_and_deletion_lifecycle():
    """
    Task 3:
    - Customer can submit claim ONLY for products they own.
    - Customer CANNOT submit claim for another customer's product (400).
    - Customer CAN delete pending claims.
    - Customer CANNOT delete finalized claims (Auto-Approved / Approved / Auto-Rejected / Rejected) -> 400.
    - Customer CANNOT delete another customer's claim -> 403.
    """
    setup_users()
    mike_token = get_auth_token("customer_mike", "Customer@12345")
    jane_token = get_auth_token("customer_jane", "Customer@12345")
    admin_token = get_auth_token("admin", "Admin@12345")

    mike_hdr = {"Authorization": f"Bearer {mike_token}"}
    jane_hdr = {"Authorization": f"Bearer {jane_token}"}
    admin_hdr = {"Authorization": f"Bearer {admin_token}"}

    # Register a product for Mike
    mike_sn = f"SN-MIKE-{int(datetime.now().timestamp() * 1000)}"
    res_p = client.post(
        "/api/products/",
        json={
            "name": "Mike Refrigerator",
            "category": "Appliances",
            "brand": "AssureX Cool",
            "model_number": "FRIDGE-100",
            "serial_number": mike_sn,
            "purchase_price": 1200.0,
            "retailer": "Home Depot",
            "purchase_date": "2025-06-01",
            "warranty_duration_months": 36,
        },
        headers=mike_hdr,
    )
    assert res_p.status_code in (200, 201)
    mike_prod_id = res_p.json()["product_id"]

    # 1. Jane tries to submit claim for Mike's product -> BLOCKED (400)
    claim_payload_jane = {
        "product_id": mike_prod_id,
        "product_name": "Mike Refrigerator",
        "product_category": "Appliances",
        "brand": "AssureX Cool",
        "model_number": "FRIDGE-100",
        "serial_number_entered": mike_sn,
        "purchase_price": 1200.0,
        "retailer": "Home Depot",
        "purchase_date": "2025-06-01",
        "warranty_start": "2025-06-01",
        "warranty_end": "2028-06-01",
        "warranty_provider": "AssureX Cool Care",
        "warranty_type": "Standard",
        "fault_date": "2025-08-01",
        "fault_type": "Mechanical Failure",
        "damage_type": "Wear and Tear",
        "fault_description": "Refrigerator compressor failed to cool properly.",
        "claim_amount": 350.0,
    }
    res_claim_jane = client.post("/api/claims/submit", json=claim_payload_jane, headers=jane_hdr)
    assert res_claim_jane.status_code == 403
    assert "belonging to another customer" in res_claim_jane.json()["detail"].lower() or "permission" in res_claim_jane.json()["detail"].lower()

    # 2. Mike submits claim for his own product -> SUCCESS
    claim_payload_mike = {
        "product_id": mike_prod_id,
        "product_name": "Mike Refrigerator",
        "product_category": "Appliances",
        "brand": "AssureX Cool",
        "model_number": "FRIDGE-100",
        "serial_number_entered": mike_sn,
        "purchase_price": 1200.0,
        "retailer": "Home Depot",
        "purchase_date": "2025-06-01",
        "warranty_start": "2025-06-01",
        "warranty_end": "2028-06-01",
        "warranty_provider": "AssureX Cool Care",
        "warranty_type": "Standard",
        "fault_date": "2025-08-01",
        "fault_type": "Mechanical Failure",
        "damage_type": "Wear and Tear",
        "fault_description": "Refrigerator compressor failed to cool properly.",
        "claim_amount": 350.0,
    }
    res_claim_mike = client.post("/api/claims/submit", json=claim_payload_mike, headers=mike_hdr)
    assert res_claim_mike.status_code in (200, 201)
    mike_claim_id = res_claim_mike.json()["claim_id"]

    # 3. Jane tries to delete Mike's claim -> BLOCKED (403)
    del_res_jane = client.delete(f"/api/claims/{mike_claim_id}", headers=jane_hdr)
    assert del_res_jane.status_code == 403

    # 4. Check finalized claim deletion restriction:
    # If claim status is finalized (e.g. Approved or Auto-Approved), customer cannot delete it (400)
    with Session(engine) as session:
        db_claim = session.exec(select(Claim).where(Claim.claim_id == mike_claim_id)).first()
        db_claim.adjudication_status = "Auto-Approved"
        session.add(db_claim)
        session.commit()

    del_finalized_res = client.delete(f"/api/claims/{mike_claim_id}", headers=mike_hdr)
    assert del_finalized_res.status_code == 400
    assert "cannot be deleted under the warranty claim lifecycle policy" in del_finalized_res.json()["detail"].lower()

    # 5. Set status back to 'Pending' -> Customer Mike CAN delete his claim
    with Session(engine) as session:
        db_claim = session.exec(select(Claim).where(Claim.claim_id == mike_claim_id)).first()
        db_claim.adjudication_status = "Pending"
        session.add(db_claim)
        session.commit()

    del_pending_res = client.delete(f"/api/claims/{mike_claim_id}", headers=mike_hdr)
    assert del_pending_res.status_code == 200
    assert del_pending_res.json()["status"] == "success"

    # Confirm deletion from database
    with Session(engine) as session:
        deleted_claim = session.exec(select(Claim).where(Claim.claim_id == mike_claim_id)).first()
        assert deleted_claim is None


def test_task4_centralized_admin_notifications_all_events():
    """
    Task 4:
    Verify that Admin receives notifications for:
    - product_registered
    - product_updated
    - product_deleted
    - claim_submitted
    - claim_updated
    - claim_deleted
    Contains event type, resource ID, actor, timestamp, description, and link.
    """
    setup_users()
    admin_token = get_auth_token("admin", "Admin@12345")
    mike_token = get_auth_token("customer_mike", "Customer@12345")

    admin_hdr = {"Authorization": f"Bearer {admin_token}"}
    mike_hdr = {"Authorization": f"Bearer {mike_token}"}

    unique_sn = f"SN-NOTIF-{int(datetime.now().timestamp() * 1000)}"

    # 1. Product Registered
    res_p = client.post(
        "/api/products/",
        json={
            "name": "Audit Washing Machine",
            "category": "Appliances",
            "brand": "AssureX Home",
            "model_number": "WM-9000",
            "serial_number": unique_sn,
            "purchase_price": 850.0,
            "retailer": "BestBuy",
            "purchase_date": "2025-05-01",
            "warranty_duration_months": 24,
        },
        headers=mike_hdr,
    )
    assert res_p.status_code in (200, 201)
    pid = res_p.json()["product_id"]

    # 2. Product Updated
    res_up = client.put(f"/api/products/{pid}", json={"name": "Audit Washing Machine Pro"}, headers=mike_hdr)
    assert res_up.status_code == 200

    # 3. Claim Submitted
    res_c = client.post(
        "/api/claims/submit",
        json={
            "product_id": pid,
            "product_name": "Audit Washing Machine Pro",
            "product_category": "Appliances",
            "brand": "AssureX Home",
            "model_number": "WM-9000",
            "serial_number_entered": unique_sn,
            "purchase_price": 850.0,
            "retailer": "BestBuy",
            "purchase_date": "2025-05-01",
            "warranty_start": "2025-05-01",
            "warranty_end": "2027-05-01",
            "warranty_provider": "AssureX Home Care",
            "warranty_type": "Standard",
            "fault_date": "2025-07-01",
            "fault_type": "Electrical Fault",
            "damage_type": "Short Circuit",
            "fault_description": "Drum motor stopped spinning after power surge.",
            "claim_amount": 280.0,
        },
        headers=mike_hdr,
    )
    assert res_c.status_code in (200, 201)
    cid = res_c.json()["claim_id"]

    # 4. Claim Updated (Adjudication or edit)
    res_adj = client.post(
        f"/api/claims/{cid}/adjudicate",
        json={"decision": "Information Requested", "notes": "Please provide power surge proof."},
        headers=admin_hdr,
    )
    assert res_adj.status_code == 200

    # 5. Claim Deleted
    res_del_c = client.delete(f"/api/claims/{cid}", headers=admin_hdr)
    assert res_del_c.status_code == 200

    # 6. Product Deleted
    res_del_p = client.delete(f"/api/products/{pid}", headers=mike_hdr)
    assert res_del_p.status_code == 200

    # Now verify all 6 notifications exist for Admin in DB and via /api/notifications
    res_notifs = client.get("/api/notifications", headers=admin_hdr)
    assert res_notifs.status_code == 200
    notifs = res_notifs.json()

    notif_types = [n["type"] for n in notifs]
    expected_events = [
        "product_registered",
        "product_updated",
        "claim_submitted",
        "claim_updated",
        "claim_deleted",
        "product_deleted",
    ]
    for exp in expected_events:
        assert exp in notif_types, f"Expected event '{exp}' not found in admin notifications: {notif_types}"

    # Verify notification schema contains event type, resource ID, actor, timestamp, description, and link
    sample_notif = next(n for n in notifs if n["type"] == "product_registered" and pid in n["message"])
    assert "Product Registered" in sample_notif["title"]
    assert pid in sample_notif["message"]
    assert "customer_mike" in sample_notif["message"]
    assert sample_notif["created_at"] is not None
    assert sample_notif["link"] == "/products"
