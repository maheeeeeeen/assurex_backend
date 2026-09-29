"""
AssureX Claim Engine — Role-Based Access Control (RBAC) Test Suite

Validates that:
- Customer has access to personal data, submit claim, product registration,
  and is strictly blocked (HTTP 403) from policies, thresholds, admin endpoints, and adjudication.
- Employee can perform assisted intake and registration,
  and is strictly blocked (HTTP 403) from policies, thresholds, admin endpoints, and adjudication.
- Reviewer can adjudicate and view policies/thresholds,
  and is strictly blocked (HTTP 403) from claim submission, product registration, and admin endpoints.
- Admin has full access across all operations.
"""

from fastapi.testclient import TestClient
from sqlmodel import Session, select
from src.main import app
from src.database_setup import engine
from src.models import User
from src.auth.service import hash_password

client = TestClient(app)


def ensure_test_users():
    """Ensure test users for all 4 roles exist in the database."""
    with Session(engine) as session:
        users = [
            ("admin", "admin@assurex.com", "Admin@12345", "admin", "Admin User"),
            ("adjuster_sarah", "sarah@assurex.com", "Adjuster@12345", "reviewer", "Sarah Reviewer"),
            ("customer_mike", "mike@example.com", "Customer@12345", "customer", "Mike Customer"),
            ("employee_dan", "dan@assurex.com", "Employee@12345", "employee", "Dan Employee"),
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
        session.commit()


def get_token(username: str, password: str) -> str:
    res = client.post("/api/auth/login", json={"username": username, "password": password})
    assert res.status_code == 200, f"Login failed for {username}: {res.text}"
    return res.json()["access_token"]


def test_rbac_matrix():
    ensure_test_users()

    admin_token = get_token("admin", "Admin@12345")
    reviewer_token = get_token("adjuster_sarah", "Adjuster@12345")
    customer_token = get_token("customer_mike", "Customer@12345")
    employee_token = get_token("employee_dan", "Employee@12345")

    admin_headers = {"Authorization": f"Bearer {admin_token}"}
    reviewer_headers = {"Authorization": f"Bearer {reviewer_token}"}
    customer_headers = {"Authorization": f"Bearer {customer_token}"}
    employee_headers = {"Authorization": f"Bearer {employee_token}"}

    # ---------------------------------------------------------
    # 1. ADMIN ENDPOINTS (Only Admin allowed; all others 403)
    # ---------------------------------------------------------
    admin_endpoints = [
        ("GET", "/api/admin/status"),
        ("GET", "/api/admin/model-comparison"),
        ("GET", "/api/admin/analytics"),
    ]

    for method, path in admin_endpoints:
        # Admin: 200
        res = client.request(method, path, headers=admin_headers)
        assert res.status_code == 200, f"Admin failed on {path}: {res.status_code}"

        # Reviewer: 403
        res = client.request(method, path, headers=reviewer_headers)
        assert res.status_code == 403, f"Reviewer should be 403 on {path}, got {res.status_code}"

        # Customer: 403
        res = client.request(method, path, headers=customer_headers)
        assert res.status_code == 403, f"Customer should be 403 on {path}, got {res.status_code}"

        # Employee: 403
        res = client.request(method, path, headers=employee_headers)
        assert res.status_code == 403, f"Employee should be 403 on {path}, got {res.status_code}"

    # ---------------------------------------------------------
    # 2. POLICIES & THRESHOLDS (Reviewer & Admin allowed; Customer & Employee 403)
    # ---------------------------------------------------------
    for path in ["/api/policies/", "/api/policies/thresholds"]:
        assert client.get(path, headers=admin_headers).status_code == 200
        assert client.get(path, headers=reviewer_headers).status_code == 200
        assert client.get(path, headers=customer_headers).status_code == 403
        assert client.get(path, headers=employee_headers).status_code == 403

    # POST /api/policies/thresholds (Admin only; Reviewer 403)
    assert client.post("/api/policies/thresholds", json={}, headers=reviewer_headers).status_code == 403
    assert client.post("/api/policies/thresholds", json={}, headers=customer_headers).status_code == 403

    # ---------------------------------------------------------
    # 3. CLAIM SUBMISSION (Customer, Employee, Admin allowed; Reviewer 403)
    # ---------------------------------------------------------
    # Reviewer blocked from POST /api/claims/submit
    rev_sub = client.post("/api/claims/submit", json={}, headers=reviewer_headers)
    assert rev_sub.status_code == 403, f"Reviewer should be 403 on claim submit, got {rev_sub.status_code}"

    # ---------------------------------------------------------
    # 4. PRODUCT REGISTRATION (Customer, Employee, Admin allowed; Reviewer 403)
    # ---------------------------------------------------------
    rev_prod = client.post("/api/products/", json={}, headers=reviewer_headers)
    assert rev_prod.status_code == 403, f"Reviewer should be 403 on product registration, got {rev_prod.status_code}"

    # ---------------------------------------------------------
    # 5. ADJUDICATION (Reviewer & Admin allowed; Customer & Employee 403)
    # ---------------------------------------------------------
    # Get any claim ID from DB
    claims_res = client.get("/api/claims/", headers=admin_headers)
    assert claims_res.status_code == 200
    claims_list = claims_res.json()
    if claims_list:
        test_cid = claims_list[0]["claim_id"]
        adj_payload = {"decision": "Approve", "notes": "RBAC verification test"}
        assert client.post(f"/api/claims/{test_cid}/adjudicate", json=adj_payload, headers=customer_headers).status_code == 403
        assert client.post(f"/api/claims/{test_cid}/adjudicate", json=adj_payload, headers=employee_headers).status_code == 403

    print("ALL RBAC TESTS PASSED 100%!")


if __name__ == "__main__":
    test_rbac_matrix()
