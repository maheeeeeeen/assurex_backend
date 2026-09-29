"""
AssureX Claim Engine — Products Router
"""

import uuid
from datetime import datetime, timedelta
from typing import List, Optional
from pydantic import BaseModel, Field as PydanticField
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlmodel import Session, select

from src.database_setup import get_session
from src.models import Product, Warranty, ClaimAuditLog, User, Claim
from src.auth.service import get_current_user, require_role
from src.services.notification_service import notify_admins

router = APIRouter()


class ProductCreate(BaseModel):
    name: str = PydanticField(min_length=2, description="Product Name")
    category: str = PydanticField(description="Product Category")
    brand: str = PydanticField(min_length=1, description="Brand Name")
    model_number: str = PydanticField(min_length=1, description="Model Number")
    serial_number: str = PydanticField(min_length=3, description="Unique Serial Number")
    purchase_price: float = PydanticField(gt=0, description="Purchase Price in USD")
    retailer: str = PydanticField(min_length=1, description="Retailer / Store Name")
    purchase_date: str = PydanticField(description="Purchase Date (YYYY-MM-DD)")
    warranty_duration_months: int = PydanticField(default=24, ge=1, le=120, description="Duration in months")
    warranty_provider: Optional[str] = "Manufacturer Standard"
    warranty_type: Optional[str] = "Standard"
    user_id: Optional[int] = None


class ProductUpdate(BaseModel):
    name: Optional[str] = None
    category: Optional[str] = None
    brand: Optional[str] = None
    model_number: Optional[str] = None
    serial_number: Optional[str] = None
    purchase_price: Optional[float] = None
    retailer: Optional[str] = None
    purchase_date: Optional[str] = None
    warranty_duration_months: Optional[int] = None


@router.get("/")
def list_products(
    category: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 100,
    session: Session = Depends(get_session),
    current_user: Optional[User] = Depends(get_current_user)
):
    """List products with optional search and category filters, scoped by user role."""
    query = select(Product)

    # Scoped access: Customers only see their own registered products
    if current_user and current_user.role == "customer":
        user_warranties = session.exec(select(Warranty.product_id).where(Warranty.user_id == current_user.id)).all()
        owned_pids = set(user_warranties)
        if owned_pids:
            query = query.where((Product.user_id == current_user.id) | (Product.product_id.in_(list(owned_pids))))
        else:
            query = query.where(Product.user_id == current_user.id)

    if category and category != "all":
        query = query.where(Product.category == category)
    if search:
        search_filter = f"%{search}%"
        query = query.where(
            (Product.name.like(search_filter)) |
            (Product.serial_number.like(search_filter)) |
            (Product.model_number.like(search_filter)) |
            (Product.brand.like(search_filter))
        )
    query = query.order_by(Product.id.desc()).limit(limit)
    products = session.exec(query).all()
    return products


@router.post("/", status_code=status.HTTP_201_CREATED)
def register_product(
    prod_in: ProductCreate,
    session: Session = Depends(get_session),
    current_user: User = Depends(require_role(["customer", "employee", "admin"]))
):
    """
    Registers a new product with auto-generated unique Product ID (PRD-YYYY-XXXX)
    and automatically attaches a corresponding warranty policy record.
    """
    clean_serial = prod_in.serial_number.strip().upper()

    # Verify unique serial number
    existing_sn = session.exec(select(Product).where(Product.serial_number == clean_serial)).first()
    if existing_sn:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Serial number '{clean_serial}' is already registered for product '{existing_sn.name}' (ID: {existing_sn.product_id})"
        )

    # Generate unique Product ID
    short_uuid = uuid.uuid4().hex[:6].upper()
    product_id = f"PRD-2026-{short_uuid}"

    # Parse purchase date and compute warranty coverage period
    try:
        start_dt = datetime.strptime(prod_in.purchase_date[:10], "%Y-%m-%d").date()
    except Exception:
        start_dt = datetime.utcnow().date()

    end_dt = start_dt + timedelta(days=int(prod_in.warranty_duration_months * 30.44))
    today = datetime.utcnow().date()
    warranty_status = "Active" if end_dt >= today else "Expired"

    # Determine owner user ID
    owner_user_id = current_user.id if current_user.role == "customer" else (prod_in.user_id or current_user.id)

    # Create Product record
    product = Product(
        product_id=product_id,
        name=prod_in.name.strip(),
        category=prod_in.category.strip(),
        brand=prod_in.brand.strip(),
        model_number=prod_in.model_number.strip(),
        serial_number=clean_serial,
        purchase_price=float(prod_in.purchase_price),
        retailer=prod_in.retailer.strip(),
        purchase_date=start_dt.strftime("%Y-%m-%d"),
        warranty_duration_months=prod_in.warranty_duration_months,
        warranty_status=warranty_status,
        user_id=owner_user_id,
    )
    session.add(product)

    # Create corresponding Warranty record
    warranty_id = f"WAR-2026-{short_uuid}"
    provider_name = prod_in.warranty_provider or f"{prod_in.brand.strip()} Care"
    warranty = Warranty(
        warranty_id=warranty_id,
        product_id=product_id,
        user_id=owner_user_id,
        provider=provider_name,
        warranty_type=prod_in.warranty_type or "Standard",
        start_date=start_dt.strftime("%Y-%m-%d"),
        end_date=end_dt.strftime("%Y-%m-%d"),
        terms=f"Standard coverage for manufacturing defects and component failures for {prod_in.warranty_duration_months} months from purchase.",
        status=warranty_status,
    )
    session.add(warranty)

    # Immutable Audit Log
    actor_name = current_user.username if current_user else "Customer_Portal"
    audit = ClaimAuditLog(
        claim_id=f"PRODUCT-{product_id}",
        actor=actor_name,
        action="PRODUCT_REGISTERED",
        details=f"Product '{product.name}' (SN: {product.serial_number}) registered with attached warranty '{warranty_id}' expiring {warranty.end_date}."
    )
    session.add(audit)

    session.commit()
    session.refresh(product)
    session.refresh(warranty)

    # Centralized Admin Notification
    try:
        notify_admins(
            session=session,
            event_type="product_registered",
            identifier=product.product_id,
            actor_username=current_user.username,
            description=f"Product '{product.name}' (SN: {product.serial_number}) registered by {current_user.username}.",
            link="/products"
        )
        session.commit()
    except Exception as e:
        print(f"[Warning] Failed to dispatch admin notification: {e}")

    return {
        "message": "Product and warranty successfully registered",
        "product_id": product.product_id,
        "warranty_id": warranty.warranty_id,
        "product": product.model_dump(),
        "warranty": warranty.model_dump(),
    }


@router.get("/{product_id}")
def get_product(
    product_id: str,
    session: Session = Depends(get_session),
    current_user: Optional[User] = Depends(get_current_user)
):
    """Retrieve product details and attached warranty terms."""
    product = session.exec(select(Product).where(Product.product_id == product_id)).first()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    warranty = session.exec(select(Warranty).where(Warranty.product_id == product_id)).first()

    # RBAC check: Customers cannot access products belonging to other users
    if current_user and current_user.role == "customer":
        is_owner = (product.user_id == current_user.id) or (warranty and warranty.user_id == current_user.id)
        if not is_owner:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden: You do not have permission to view this product"
            )

    return {
        "product": product,
        "warranty": warranty,
    }


@router.put("/{product_id}")
def update_product(
    product_id: str,
    prod_update: ProductUpdate,
    session: Session = Depends(get_session),
    current_user: User = Depends(require_role(["customer", "admin"]))
):
    """
    Updates product details with ownership verification, serial collision checks,
    and automatic warranty expiry recalculation. Dispatches admin notification on success.
    """
    product = session.exec(select(Product).where(Product.product_id == product_id)).first()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    # Ownership guard for customer role
    if current_user.role == "customer" and product.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: You do not have permission to modify this product."
        )

    # Check serial number uniqueness if changed
    if prod_update.serial_number is not None:
        clean_sn = prod_update.serial_number.strip().upper()
        if clean_sn != product.serial_number:
            existing_sn = session.exec(
                select(Product).where(Product.serial_number == clean_sn, Product.product_id != product_id)
            ).first()
            if existing_sn:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Serial number '{clean_sn}' is already registered for product '{existing_sn.name}' (ID: {existing_sn.product_id})."
                )
            product.serial_number = clean_sn

    # Update basic fields
    if prod_update.name is not None:
        product.name = prod_update.name.strip()
    if prod_update.category is not None:
        product.category = prod_update.category.strip()
    if prod_update.brand is not None:
        product.brand = prod_update.brand.strip()
    if prod_update.model_number is not None:
        product.model_number = prod_update.model_number.strip()
    if prod_update.purchase_price is not None:
        product.purchase_price = float(prod_update.purchase_price)
    if prod_update.retailer is not None:
        product.retailer = prod_update.retailer.strip()

    # Check warranty & date recalculation
    warranty = session.exec(select(Warranty).where(Warranty.product_id == product_id)).first()
    date_changed = False
    if prod_update.purchase_date is not None:
        product.purchase_date = prod_update.purchase_date.strip()
        date_changed = True
    if prod_update.warranty_duration_months is not None:
        product.warranty_duration_months = prod_update.warranty_duration_months
        date_changed = True

    if date_changed and warranty:
        try:
            start_dt = datetime.strptime(product.purchase_date[:10], "%Y-%m-%d").date()
        except Exception:
            start_dt = datetime.utcnow().date()
        end_dt = start_dt + timedelta(days=int(product.warranty_duration_months * 30.44))
        today = datetime.utcnow().date()
        warranty_status = "Active" if end_dt >= today else "Expired"
        product.warranty_status = warranty_status
        warranty.start_date = start_dt.strftime("%Y-%m-%d")
        warranty.end_date = end_dt.strftime("%Y-%m-%d")
        warranty.status = warranty_status
        session.add(warranty)

    session.add(product)

    # Audit log
    audit = ClaimAuditLog(
        claim_id=f"PRODUCT-{product_id}",
        actor=current_user.username,
        action="PRODUCT_UPDATED",
        details=f"Product '{product.name}' (ID: {product_id}, SN: {product.serial_number}) updated by {current_user.username}."
    )
    session.add(audit)

    # Centralized Admin Notification
    try:
        notify_admins(
            session=session,
            event_type="product_updated",
            identifier=product_id,
            actor_username=current_user.username,
            description=f"Product '{product.name}' (SN: {product.serial_number}) was updated by {current_user.username}.",
            link="/products"
        )
    except Exception as e:
        print(f"[Warning] Failed to dispatch admin notification: {e}")

    session.commit()
    session.refresh(product)
    return {"message": "Product updated successfully", "product": product.model_dump(), "product_id": product.product_id}


@router.delete("/{product_id}")
def delete_product(
    product_id: str,
    session: Session = Depends(get_session),
    current_user: User = Depends(require_role(["customer", "admin"]))
):
    """
    Deletes a registered product and associated warranty.
    Enforces ownership and prevents deletion if active warranty claims exist.
    Dispatches admin notification upon successful deletion.
    """
    product = session.exec(select(Product).where(Product.product_id == product_id)).first()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    # Ownership check for customer
    if current_user.role == "customer" and product.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: You do not have permission to delete this product."
        )

    # Prevent deletion if claims are filed against this product
    active_claim = session.exec(select(Claim).where(Claim.product_id == product_id)).first()
    if active_claim:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot delete product '{product_id}' because claim '{active_claim.claim_id}' exists against it."
        )

    prod_name = product.name
    prod_sn = product.serial_number

    # Delete attached warranties
    warranties = session.exec(select(Warranty).where(Warranty.product_id == product_id)).all()
    for w in warranties:
        session.delete(w)

    session.delete(product)

    # Audit log
    audit = ClaimAuditLog(
        claim_id=f"PRODUCT-{product_id}",
        actor=current_user.username,
        action="PRODUCT_DELETED",
        details=f"Product '{prod_name}' (ID: {product_id}, SN: {prod_sn}) deleted by {current_user.username}."
    )
    session.add(audit)

    # Centralized Admin Notification
    try:
        notify_admins(
            session=session,
            event_type="product_deleted",
            identifier=product_id,
            actor_username=current_user.username,
            description=f"Product '{prod_name}' (SN: {prod_sn}) was deleted by {current_user.username}.",
            link="/products"
        )
    except Exception as e:
        print(f"[Warning] Failed to dispatch admin notification: {e}")

    session.commit()
    return {"status": "success", "message": f"Product '{product_id}' deleted successfully", "product_id": product_id}
