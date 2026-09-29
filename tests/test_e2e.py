"""
AssureX Claim Engine — Automated End-to-End Integration Tests
Validates authentication, claim submission, card generation, dual-AI adjudication, and dossier retrieval.
"""

from fastapi.testclient import TestClient
from sqlmodel import Session, select
from src.main import app
from src.database_setup import engine
from src.models import User, Product, Warranty
from src.auth.service import hash_password

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_auth_login():
    response = client.post("/api/auth/login", json={"username": "admin", "password": "Admin@12345"})
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["user"]["role"] == "admin"

def test_claim_stats():
    # Login
    auth_res = client.post("/api/auth/login", json={"username": "admin", "password": "Admin@12345"})
    token = auth_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    res = client.get("/api/claims/stats/summary", headers=headers)
    assert res.status_code == 200
    stats = res.json()
    assert "total_claims" in stats
    assert stats["total_claims"] > 0
    assert "dual_model_agreement_rate" in stats

def test_full_claim_submission_and_adjudication():
    # 0. Ensure customer and registered product with active warranty exist
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

        product = session.exec(select(Product).where(Product.product_id == "PRD-E2E-WM01")).first()
        if not product:
            product = Product(
                product_id="PRD-E2E-WM01",
                user_id=user.id,
                name="LG Smart Washing Machine",
                category="Appliances",
                brand="LG",
                model_number="WM4000HBA",
                serial_number="SN-LG-98214",
                purchase_price=899.99,
                retailer="Home Depot",
                purchase_date="2025-02-10",
                status="Active",
            )
            session.add(product)
            session.commit()
            session.refresh(product)
        elif product.user_id != user.id:
            product.user_id = user.id
            session.add(product)
            session.commit()

        warranty = session.exec(select(Warranty).where(Warranty.product_id == "PRD-E2E-WM01")).first()
        if not warranty:
            warranty = Warranty(
                warranty_id="WAR-E2E-WM01",
                product_id="PRD-E2E-WM01",
                user_id=user.id,
                coverage_type="Extended Warranty",
                start_date="2025-02-10",
                end_date="2027-02-10",
                status="Active",
                provider="LG Extended Care",
            )
            session.add(warranty)
            session.commit()
            session.refresh(warranty)

    # 1. Login as customer
    auth_res = client.post("/api/auth/login", json={"username": "customer_mike", "password": "Customer@12345"})
    token = auth_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Submit realistic claim
    claim_data = {
        "product_id": "PRD-E2E-WM01",
        "product_name": "LG Smart Washing Machine",
        "product_category": "Appliances",
        "brand": "LG",
        "model_number": "WM4000HBA",
        "serial_number_entered": "SN-LG-98214",
        "purchase_price": 899.99,
        "retailer": "Home Depot",
        "purchase_date": "2025-02-10",
        "warranty_start": "2025-02-10",
        "warranty_end": "2027-02-10",
        "warranty_provider": "LG Extended Care",
        "warranty_type": "Extended Warranty",
        "fault_date": "2026-04-12",
        "fault_type": "Motor Failure",
        "damage_type": "Mechanical Breakdown",
        "fault_description": "Motor stopped spinning during wash cycle, error code dC on display.",
        "repair_history_count": 0,
        "previous_repair_authorized": True,
        "receipt_uploaded": True,
        "warranty_card_uploaded": True,
        "product_image_uploaded": True,
        "fault_evidence_uploaded": True,
        "repair_report_uploaded": False,
    }

    sub_res = client.post("/api/claims/submit", json=claim_data, headers=headers)
    assert sub_res.status_code in (200, 201)
    claim = sub_res.json()
    cid = claim["claim_id"]

    assert claim["card_image_path"] is not None
    assert claim["tabular_prediction"] in ["Likely Valid", "Likely Invalid", "Manual Review Required"]
    assert claim["tm_prediction"] in ["Likely Valid", "Likely Invalid", "Manual Review Required"]
    assert claim["final_confidence"] > 0.0

    # 3. Retrieve claim dossier as adjuster
    adj_auth = client.post("/api/auth/login", json={"username": "adjuster_sarah", "password": "Adjuster@12345"})
    adj_token = adj_auth.json()["access_token"]
    adj_headers = {"Authorization": f"Bearer {adj_token}"}

    dossier_res = client.get(f"/api/claims/{cid}", headers=adj_headers)
    assert dossier_res.status_code == 200
    dossier = dossier_res.json()
    assert dossier["claim"]["claim_id"] == cid
    assert len(dossier["audit_logs"]) >= 3
    assert "passed" in dossier["rule_evaluation"]

    # 4. Reviewer action override
    act_res = client.post(
        f"/api/claims/{cid}/adjudicate",
        json={"decision": "Approve", "notes": "Approved after manual inspection of motor failure code."},
        headers=adj_headers,
    )
    assert act_res.status_code == 200
    assert act_res.json()["claim"]["adjudication_status"] == "Approved"

if __name__ == "__main__":
    test_health_check()
    test_auth_login()
    test_claim_stats()
    test_full_claim_submission_and_adjudication()
    print("ALL INTEGRATION TESTS PASSED CLEANLY!")
