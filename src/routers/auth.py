"""
AssureX Claim Engine — Authentication Router
"""

import uuid
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select

from src.database_setup import get_session
from src.models import User, ClaimAuditLog
from src.auth.service import hash_password, verify_password, create_access_token, get_current_user
from src.schemas.auth import UserRegister, UserLogin, UserUpdate, UserResponse, TokenResponse

router = APIRouter()


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(user_in: UserRegister, session: Session = Depends(get_session)):
    """Register a new user account with RBAC (customer, employee, reviewer, admin)."""
    uname = user_in.username or user_in.email.split("@")[0]
    # Check if username exists
    existing_user = session.exec(select(User).where(User.username == uname)).first()
    if existing_user and user_in.username:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered",
        )
    elif existing_user:
        uname = f"{uname}_{int(datetime.utcnow().timestamp())}"

    # Check if email exists
    existing_email = session.exec(select(User).where(User.email == user_in.email)).first()
    if existing_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email address already registered",
        )

    valid_roles = ["customer", "employee", "reviewer", "admin", "viewer"]
    role = user_in.role if user_in.role in valid_roles else "customer"

    # Generate unique user code
    user_code = f"USR-{uuid.uuid4().hex[:6].upper()}"

    db_user = User(
        user_code=user_code,
        username=uname,
        email=user_in.email,
        hashed_password=hash_password(user_in.password),
        full_name=user_in.full_name,
        role=role,
        phone=user_in.phone,
    )
    session.add(db_user)

    # Immutable audit trail entry
    audit = ClaimAuditLog(
        claim_id=f"USER-{uname}",
        actor=uname,
        action="USER_REGISTERED",
        details=f"Account created for {user_in.full_name} ({uname}) with role '{role}' and ID {user_code}.",
    )
    session.add(audit)

    session.commit()
    session.refresh(db_user)

    token = create_access_token({"sub": db_user.username, "role": db_user.role})
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": db_user,
    }


@router.post("/login", response_model=TokenResponse)
def login(login_data: UserLogin, session: Session = Depends(get_session)):
    """Authenticate and obtain JWT access token."""
    identifier = login_data.username or login_data.email
    if not identifier:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username or email is required",
        )

    user = session.exec(
        select(User).where((User.username == identifier) | (User.email == identifier))
    ).first()

    if not user or not verify_password(login_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token({"sub": user.username, "role": user.role})
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": user,
    }


@router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    """Retrieve profile of the currently logged-in user."""
    if not current_user.user_code:
        current_user.user_code = f"USR-{current_user.id:04d}"
    return current_user


@router.get("/profile", response_model=UserResponse)
def get_profile(current_user: User = Depends(get_current_user)):
    """Alias for /me."""
    if not current_user.user_code:
        current_user.user_code = f"USR-{current_user.id:04d}"
    return current_user


@router.put("/profile", response_model=UserResponse)
def update_profile(
    profile_in: UserUpdate,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    """Update user profile information with audit trail."""
    # Ensure user_code is populated
    if not current_user.user_code:
        current_user.user_code = f"USR-{current_user.id:04d}"

    updated_fields = []

    if profile_in.full_name and profile_in.full_name.strip():
        current_user.full_name = profile_in.full_name.strip()
        updated_fields.append("full_name")

    if profile_in.phone is not None:
        current_user.phone = profile_in.phone.strip()
        updated_fields.append("phone")

    if profile_in.email and profile_in.email.strip() and profile_in.email != current_user.email:
        new_email = profile_in.email.strip()
        existing = session.exec(select(User).where(User.email == new_email)).first()
        if existing and existing.id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email address is already in use by another account",
            )
        current_user.email = new_email
        updated_fields.append("email")

    if updated_fields:
        audit = ClaimAuditLog(
            claim_id=f"USER-{current_user.username}",
            actor=current_user.username,
            action="USER_PROFILE_UPDATED",
            details=f"User {current_user.username} updated profile fields: {', '.join(updated_fields)}.",
        )
        session.add(audit)
        session.add(current_user)
        session.commit()
        session.refresh(current_user)

    return current_user
