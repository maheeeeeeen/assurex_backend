"""
AssureX Claim Engine — Authentication Schemas
"""

from typing import Optional
from pydantic import BaseModel


class UserRegister(BaseModel):
    username: Optional[str] = None
    email: str
    password: str
    full_name: str
    phone: Optional[str] = None
    role: Optional[str] = "customer"  # customer, employee, reviewer, admin, viewer


class UserLogin(BaseModel):
    username: Optional[str] = None
    email: Optional[str] = None
    password: str


class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None


class UserResponse(BaseModel):
    id: int
    user_code: Optional[str] = None
    username: str
    email: str
    full_name: str
    role: str
    phone: Optional[str] = None
    created_at: str

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse
