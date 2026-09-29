"""
AssureX Claim Engine — SQLModel Database Entities

Defines the relational schema for:
- Users (Customer, Reviewer, Admin with RBAC)
- Products (Catalog with serial tracking)
- Warranties (Coverage policies, timelines, providers)
- Claims (Multi-stage adjudication dossier, OCR data, Dual-AI predictions, audit logs)
- ClaimAuditLog (Immutable compliance audit trail)
"""

from typing import Optional, List
from datetime import datetime
from sqlmodel import SQLModel, Field, Relationship


class User(SQLModel, table=True):
    __tablename__ = "users"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_code: Optional[str] = Field(default=None, unique=True, index=True)
    username: str = Field(unique=True, index=True)
    email: str = Field(unique=True, index=True)
    hashed_password: str
    role: str = Field(default="customer", index=True)  # customer, employee, reviewer, admin, viewer
    full_name: str
    phone: Optional[str] = Field(default=None)
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class Product(SQLModel, table=True):
    __tablename__ = "products"

    id: Optional[int] = Field(default=None, primary_key=True)
    product_id: str = Field(unique=True, index=True)
    name: str = Field(index=True)
    category: str = Field(index=True)  # Electronics, Appliances, Automotive
    brand: str = Field(index=True)
    model_number: str
    serial_number: str = Field(unique=True, index=True)
    purchase_price: float
    retailer: str
    purchase_date: str
    warranty_duration_months: int = Field(default=24)
    warranty_status: Optional[str] = Field(default="Active", index=True)
    user_id: Optional[int] = Field(default=None, index=True)
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class Warranty(SQLModel, table=True):
    __tablename__ = "warranties"

    id: Optional[int] = Field(default=None, primary_key=True)
    warranty_id: str = Field(unique=True, index=True)
    product_id: str = Field(index=True)
    user_id: Optional[int] = Field(default=None, index=True)
    provider: str
    warranty_type: str = Field(default="Standard")  # Standard, Extended, Premium
    start_date: str
    end_date: str
    terms: Optional[str] = None
    status: str = Field(default="Active", index=True)  # Active, Expired, Void


class Claim(SQLModel, table=True):
    __tablename__ = "claims"

    id: Optional[int] = Field(default=None, primary_key=True)
    claim_id: str = Field(unique=True, index=True)
    user_id: Optional[int] = Field(default=None, index=True)

    # Product details
    product_id: str = Field(index=True)
    product_name: str
    product_category: str = Field(index=True)
    brand: str
    model_number: str
    serial_number_entered: str = Field(index=True)
    serial_number_on_receipt: Optional[str] = None
    model_number_on_receipt: Optional[str] = None
    serial_number_on_warranty_card: Optional[str] = None
    model_number_on_warranty_card: Optional[str] = None
    serial_number_on_barcode: Optional[str] = None
    model_number_on_barcode: Optional[str] = None
    cross_verification_json: Optional[str] = None

    # Purchase & Warranty terms
    purchase_date: str
    purchase_price: float
    retailer: str
    warranty_start: str
    warranty_end: str
    warranty_provider: str
    warranty_type: str

    # Fault details
    fault_date: str
    claim_submission_date: str
    fault_type: str
    fault_description: str
    damage_type: str

    # Calculated metrics
    product_age_months: float = Field(default=0.0)
    remaining_warranty_days: float = Field(default=0.0)
    repair_history_count: int = Field(default=0)
    previous_repair_authorized: bool = Field(default=True)

    # Document upload flags
    receipt_uploaded: bool = Field(default=True)
    warranty_card_uploaded: bool = Field(default=True)
    product_image_uploaded: bool = Field(default=True)
    fault_evidence_uploaded: bool = Field(default=True)
    repair_report_uploaded: bool = Field(default=False)
    missing_doc_count: int = Field(default=0)

    # Stored file paths & cryptographic hashes
    receipt_path: Optional[str] = None
    receipt_hash: Optional[str] = Field(default=None, index=True)
    warranty_card_path: Optional[str] = None
    warranty_card_hash: Optional[str] = Field(default=None, index=True)
    product_image_path: Optional[str] = None
    product_image_hash: Optional[str] = Field(default=None, index=True)
    fault_evidence_path: Optional[str] = None
    fault_evidence_hash: Optional[str] = Field(default=None, index=True)
    fault_video_path: Optional[str] = None
    fault_video_hash: Optional[str] = Field(default=None, index=True)
    barcode_image_path: Optional[str] = None
    barcode_image_hash: Optional[str] = Field(default=None, index=True)
    card_image_path: Optional[str] = None  # 1200x1680 High-DPI Claim Summary Card

    # Anomaly flags
    serial_mismatch_flag: bool = Field(default=False)
    date_contradiction_flag: bool = Field(default=False)
    excluded_damage: bool = Field(default=False)
    duplicate_claim_flag: bool = Field(default=False)
    duplicate_claim_details: Optional[str] = None
    invoice_number: Optional[str] = Field(default=None, index=True)

    # Intelligence & Analysis JSONs
    ocr_extracted_json: Optional[str] = None
    rule_evaluation_json: Optional[str] = None

    # Dual ML Predictions
    tabular_prediction: Optional[str] = None
    tabular_confidence: Optional[float] = None
    tabular_probabilities_json: Optional[str] = None

    tm_prediction: Optional[str] = None
    tm_confidence: Optional[float] = None
    tm_probabilities_json: Optional[str] = None

    confidence_difference: Optional[float] = None
    models_agreed: Optional[bool] = None
    match_category: Optional[str] = None

    # Final Adjudication
    adjudication_status: str = Field(default="Pending", index=True)
    # Options: Auto-Approved, Auto-Rejected, Manual Review Required, Approved, Rejected, Information Requested
    adjudication_stage: str = Field(default="Automated")  # Automated, Manual_Review, Final
    final_confidence: float = Field(default=0.0)
    decision_reason_summary: str = Field(default="")
    decision_reasons_json: Optional[str] = None

    # Human Reviewer Audit
    reviewed_by: Optional[str] = None
    reviewer_notes: Optional[str] = None
    adjudication_timestamp: Optional[str] = None

    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class ClaimAuditLog(SQLModel, table=True):
    __tablename__ = "claim_audit_logs"

    id: Optional[int] = Field(default=None, primary_key=True)
    claim_id: str = Field(index=True)
    actor: str
    action: str  # SUBMITTED, OCR_EXTRACTED, RULES_RUN, ML_PREDICTED, ADJUDICATED, REVIEWER_ACTION
    details: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class Notification(SQLModel, table=True):
    __tablename__ = "notifications"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(index=True)
    title: str
    message: str
    type: str = Field(default="system", index=True)  # claim_submitted, claim_status_changed, info_requested, claim_approved, claim_rejected, warranty_expiry
    link: Optional[str] = None
    is_read: bool = Field(default=False, index=True)
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

