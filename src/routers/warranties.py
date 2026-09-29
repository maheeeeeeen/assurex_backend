"""
AssureX Claim Engine — Warranties Router
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlmodel import Session, select

from src.database_setup import get_session
from src.models import Warranty, Product, User
from src.auth.service import get_current_user

router = APIRouter()


@router.get("/")
def list_warranties(
    status: Optional[str] = None,
    category: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 150,
    session: Session = Depends(get_session),
    current_user: Optional[User] = Depends(get_current_user)
):
    """
    List registered warranties enriched with associated product details,
    dynamically calculated remaining days, and expiry threshold alerts.
    Scoped by user role: Customers only see their own warranties.
    """
    query = select(Warranty)
    if current_user and current_user.role == "customer":
        query = query.where(Warranty.user_id == current_user.id)

    warranties = session.exec(query.order_by(Warranty.id.desc()).limit(limit)).all()
    today = datetime.utcnow().date()

    # Preload product map for O(1) enrichment
    product_ids = [w.product_id for w in warranties if w.product_id]
    products = session.exec(select(Product).where(Product.product_id.in_(product_ids))).all() if product_ids else []
    product_map = {p.product_id: p for p in products}

    results = []
    for w in warranties:
        prod = product_map.get(w.product_id)

        try:
            end_date = datetime.strptime(w.end_date[:10], "%Y-%m-%d").date()
            remaining_days = (end_date - today).days

            if remaining_days < 0:
                calc_status = "Expired"
            elif 0 <= remaining_days <= 30:
                calc_status = "Expiring Soon (< 30 days)"
            elif 30 < remaining_days <= 60:
                calc_status = "Nearing Expiry (< 60 days)"
            else:
                calc_status = "Active"
        except Exception:
            remaining_days = 0
            calc_status = w.status

        # Apply Category Filter
        if category and category != "all" and prod and prod.category != category:
            continue

        # Apply Status Filter
        if status and status != "all":
            if status == "active" and remaining_days < 0:
                continue
            elif status == "expiring_soon" and not (0 <= remaining_days <= 30):
                continue
            elif status == "nearing_expiry" and not (0 <= remaining_days <= 60):
                continue
            elif status == "expired" and remaining_days >= 0:
                continue

        # Apply Search Filter
        if search:
            s = search.lower()
            prod_name = prod.name.lower() if prod else ""
            prod_sn = prod.serial_number.lower() if prod else ""
            prod_brand = prod.brand.lower() if prod else ""
            if s not in w.warranty_id.lower() and s not in prod_name and s not in prod_sn and s not in prod_brand:
                continue

        results.append({
            "id": w.id,
            "warranty_id": w.warranty_id,
            "product_id": w.product_id,
            "user_id": w.user_id,
            "provider": w.provider,
            "warranty_type": w.warranty_type,
            "start_date": w.start_date,
            "end_date": w.end_date,
            "terms": w.terms,
            "status": w.status,
            "remaining_days": remaining_days,
            "calculated_status": calc_status,
            "is_expiring_soon": 0 <= remaining_days <= 30,
            "is_nearing_expiry": 0 <= remaining_days <= 60,
            "product": {
                "name": prod.name if prod else "Unknown Product",
                "category": prod.category if prod else "General",
                "brand": prod.brand if prod else "General",
                "model_number": prod.model_number if prod else "",
                "serial_number": prod.serial_number if prod else "",
                "retailer": prod.retailer if prod else "",
                "purchase_price": prod.purchase_price if prod else 0.0,
                "purchase_date": prod.purchase_date if prod else "",
            } if prod else None,
        })

    return results


@router.get("/check/{product_id}")
def check_warranty(
    product_id: str,
    session: Session = Depends(get_session),
    current_user: Optional[User] = Depends(get_current_user)
):
    """Evaluate live warranty status for a product."""
    warranty = session.exec(select(Warranty).where(Warranty.product_id == product_id)).first()
    if not warranty:
        raise HTTPException(status_code=404, detail="Warranty not found for product")

    # RBAC check: Customers cannot inspect warranties belonging to other users
    if current_user and current_user.role == "customer":
        if warranty.user_id and warranty.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden: You do not have permission to inspect this warranty"
            )

    try:
        end_date = datetime.strptime(warranty.end_date[:10], "%Y-%m-%d").date()
        today = datetime.utcnow().date()
        remaining_days = (end_date - today).days

        if remaining_days >= 0:
            status_desc = "Active"
        elif -7 <= remaining_days < 0:
            status_desc = "Grace Period (7 Days)"
        else:
            status_desc = "Expired"
    except Exception:
        remaining_days = 0
        status_desc = warranty.status

    return {
        "warranty": warranty,
        "remaining_days": remaining_days,
        "calculated_status": status_desc,
        "is_claim_eligible": remaining_days >= -7,
    }
