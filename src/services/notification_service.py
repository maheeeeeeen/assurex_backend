"""
AssureX Claim Engine — Notification Service
Handles system notifications and centralized admin alerts for:
- Product registration, updates, deletion
- Claim submission, updates, deletion
"""

import logging
from datetime import datetime, timezone
from typing import Optional, List
from sqlmodel import Session, select
from src.models import Notification, User

logger = logging.getLogger("assurex.notifications")


def notify_admins(
    session: Session,
    event_type: str,
    identifier: str,
    actor_username: str,
    description: str,
    link: Optional[str] = None
) -> List[Notification]:
    """
    Creates and persists notifications for all active Administrator accounts.
    Ensures event type, resource identifier, actor, timestamp, and concise description
    are present. Safe to call: handles errors gracefully without interrupting primary operations.
    """
    created_notifications = []
    try:
        admin_users = session.exec(select(User).where(User.role == "admin")).all()
        if not admin_users:
            logger.info("No admin accounts found to receive notification.")
            return []

        # Canonical timestamp
        now_iso = datetime.now(timezone.utc).isoformat()

        # Format title and concise message
        event_display = event_type.replace("_", " ").title()
        title = f"Admin Alert: {event_display} ({identifier})"
        message = f"[{actor_username}] {description} | Resource ID: {identifier} | Timestamp: {now_iso}"

        for admin in admin_users:
            # Dedup check within last 30 seconds for identical event & resource
            existing = session.exec(
                select(Notification)
                .where(
                    Notification.user_id == admin.id,
                    Notification.type == event_type,
                    Notification.title == title
                )
                .order_by(Notification.id.desc())
            ).first()

            if existing:
                # Avoid duplicate notification if identical event exists
                continue

            notif = Notification(
                user_id=admin.id,
                title=title,
                message=message,
                type=event_type,
                link=link or ("/products" if "product" in event_type else f"/claims/{identifier}"),
                is_read=False,
                created_at=now_iso
            )
            session.add(notif)
            created_notifications.append(notif)

        logger.info(f"Generated {len(created_notifications)} admin notification(s) for event '{event_type}' ({identifier}).")
    except Exception as exc:
        logger.error(f"Error creating admin notifications for event '{event_type}': {exc}", exc_info=True)

    return created_notifications
