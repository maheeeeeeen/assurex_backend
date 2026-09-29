"""
Notifications Router — Handles user alerts and system notifications.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select
from datetime import datetime
import json

from src.database_setup import get_session
from src.models import User, Notification, Warranty, Product
from src.routers.auth import get_current_user

router = APIRouter()

def generate_warranty_expiry_notifications(session: Session, user: User):
    """
    Dynamically generates warranty expiry notifications for products expiring within 30 days.
    (FR ix: The system MUST issue a notification 30 days prior to the expiration of a registered product warranty)
    """
    today = datetime.utcnow().date()
    
    # Get all active warranties for user
    warranties = session.exec(
        select(Warranty).where(Warranty.user_id == user.id, Warranty.status == "Active")
    ).all()
    
    for w in warranties:
        try:
            end_dt = datetime.strptime(w.end_date, "%Y-%m-%d").date()
            days_left = (end_dt - today).days
            
            if 0 <= days_left <= 30:
                # Check if we already notified for this warranty recently
                # We can store the warranty_id in the link or just check by title
                notif_title = f"Warranty Expiring: {w.warranty_id}"
                existing = session.exec(
                    select(Notification)
                    .where(Notification.user_id == user.id)
                    .where(Notification.title == notif_title)
                ).first()
                
                if not existing:
                    product = session.exec(select(Product).where(Product.product_id == w.product_id)).first()
                    p_name = product.name if product else "Product"
                    
                    new_notif = Notification(
                        user_id=user.id,
                        title=notif_title,
                        message=f"Your warranty for {p_name} is expiring in {days_left} days on {w.end_date}.",
                        type="warranty_expiry",
                        link="/warranties"
                    )
                    session.add(new_notif)
        except Exception as e:
            print(f"Error parsing date for warranty {w.warranty_id}: {e}")
            
    session.commit()

@router.get("")
def get_notifications(
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Fetch all notifications for the current user."""
    # First, generate any dynamic alerts (like warranty expiry)
    generate_warranty_expiry_notifications(session, current_user)
    
    # Then fetch all notifications, newest first
    notifications = session.exec(
        select(Notification)
        .where(Notification.user_id == current_user.id)
        .order_by(Notification.created_at.desc())
    ).all()
    
    return [n.model_dump() for n in notifications]

@router.put("/{notification_id}/read")
def mark_notification_read(
    notification_id: int,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Mark a specific notification as read."""
    notif = session.exec(
        select(Notification)
        .where(Notification.id == notification_id, Notification.user_id == current_user.id)
    ).first()
    
    if not notif:
        raise HTTPException(status_code=404, detail="Notification not found")
        
    notif.is_read = True
    session.add(notif)
    session.commit()
    session.refresh(notif)
    return notif.model_dump()

@router.post("/mark-all-read")
def mark_all_notifications_read(
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Mark all notifications for the user as read."""
    notifications = session.exec(
        select(Notification)
        .where(Notification.user_id == current_user.id, Notification.is_read == False)
    ).all()
    
    for n in notifications:
        n.is_read = True
        session.add(n)
        
    session.commit()
    return {"status": "success", "marked_count": len(notifications)}
