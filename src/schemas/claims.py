"""
AssureX Claim Engine — Warranty Claim Schemas
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class ClaimSubmitRequest(BaseModel):
    product_id: Optional[str] = None
    product_name: str
    product_category: str  # Electronics, Appliances, Automotive
    brand: str
    model_number: str
    serial_number_entered: str

    purchase_date: str
    purchase_price: float
    retailer: str
    warranty_start: str
    warranty_end: str
    warranty_provider: str
    warranty_type: str = "Standard"

    fault_date: str
    fault_type: str
    fault_description: str
    damage_type: str

    repair_history_count: int = 0
    previous_repair_authorized: bool = True

    # Upload indicator flags & cryptographic document hashes
    receipt_uploaded: bool = False
    receipt_path: Optional[str] = None
    receipt_hash: Optional[str] = None
    serial_number_on_receipt: Optional[str] = None
    model_number_on_receipt: Optional[str] = None
    warranty_card_uploaded: bool = False
    warranty_card_path: Optional[str] = None
    warranty_card_hash: Optional[str] = None
    serial_number_on_warranty_card: Optional[str] = None
    model_number_on_warranty_card: Optional[str] = None
    product_image_uploaded: bool = False
    product_image_path: Optional[str] = None
    product_image_hash: Optional[str] = None
    fault_evidence_uploaded: bool = False
    fault_evidence_path: Optional[str] = None
    fault_evidence_hash: Optional[str] = None
    fault_video_uploaded: bool = False
    fault_video_path: Optional[str] = None
    fault_video_hash: Optional[str] = None
    barcode_image_uploaded: bool = False
    barcode_image_path: Optional[str] = None
    barcode_image_hash: Optional[str] = None
    serial_number_on_barcode: Optional[str] = None
    model_number_on_barcode: Optional[str] = None
    repair_report_uploaded: bool = False
    invoice_number: Optional[str] = None
    ocr_extracted_json: Optional[str] = None
    cross_verification_json: Optional[str] = None


class ClaimUpdateRequest(BaseModel):
    fault_date: Optional[str] = None
    fault_type: Optional[str] = None
    fault_description: Optional[str] = None
    damage_type: Optional[str] = None
    repair_history_count: Optional[int] = None
    previous_repair_authorized: Optional[bool] = None


class MediaUploadResponse(BaseModel):
    media_type: str
    filename: str
    file_path: str
    file_url: str
    file_hash: str
    file_size: int
    content_type: str
    serial_number: Optional[str] = None
    model_number: Optional[str] = None
    detected_serials: List[str] = []
    detected_models: List[str] = []
    ocr_confidence: Optional[float] = None


class OCRProcessResponse(BaseModel):
    merchant: Optional[str] = None
    retailer: Optional[str] = None
    purchase_date: Optional[str] = None
    serial_number: Optional[str] = None
    detected_serials: List[str] = []
    model_number: Optional[str] = None
    detected_models: List[str] = []
    invoice_number: Optional[str] = None
    purchase_amount: Optional[float] = None
    ocr_confidence: float = 0.85
    ocr_engine: str = "Tesseract_OCR"
    file_hash: str
    file_path: str
    is_duplicate_file: bool = False
    duplicate_claim_id: Optional[str] = None
    is_duplicate_invoice: bool = False
    duplicate_invoice_claim_id: Optional[str] = None
    raw_text: Optional[str] = None


class DuplicateCheckRequest(BaseModel):
    product_id: Optional[str] = None
    serial_number_entered: Optional[str] = None
    fault_description: Optional[str] = None
    invoice_number: Optional[str] = None
    receipt_hash: Optional[str] = None
    current_claim_id: Optional[str] = None


class DuplicateCheckResponse(BaseModel):
    is_duplicate: bool
    duplicate_type: Optional[str] = None
    matched_claim_id: Optional[str] = None
    similarity_score: float = 0.0
    details: Optional[str] = None


class CrossVerificationRequest(BaseModel):
    entered_serial: str
    entered_model: str
    receipt_serial: Optional[str] = None
    receipt_model: Optional[str] = None
    warranty_card_serial: Optional[str] = None
    warranty_card_model: Optional[str] = None
    barcode_serial: Optional[str] = None
    barcode_model: Optional[str] = None


class CrossVerificationResponse(BaseModel):
    has_mismatch: bool
    has_serial_mismatch: bool
    has_model_mismatch: bool
    total_mismatches: int
    total_matches: int
    mismatches: List[Dict[str, Any]] = []
    matches: List[Dict[str, Any]] = []
    sources: Dict[str, Any] = {}
    status: str


class ClaimAdjudicationAction(BaseModel):
    decision: Optional[str] = None
    action: Optional[str] = None  # Approve, Reject, Request Information
    notes: Optional[str] = None
    reviewer_notes: Optional[str] = None


class ClaimResponse(BaseModel):
    id: int
    claim_id: str
    user_id: Optional[int] = None
    product_id: str
    product_name: str
    product_category: str
    brand: str
    model_number: str
    serial_number_entered: str
    purchase_date: str
    purchase_price: float
    retailer: str
    warranty_start: str
    warranty_end: str
    warranty_provider: str
    warranty_type: str
    fault_date: str
    claim_submission_date: str
    fault_type: str
    fault_description: str
    damage_type: str
    product_age_months: float
    remaining_warranty_days: float
    repair_history_count: int
    previous_repair_authorized: bool

    # Document & Anomaly Flags
    receipt_uploaded: bool
    warranty_card_uploaded: bool
    product_image_uploaded: bool
    fault_evidence_uploaded: bool
    serial_mismatch_flag: bool
    date_contradiction_flag: bool
    excluded_damage: bool
    duplicate_claim_flag: bool
    duplicate_claim_details: Optional[str] = None
    invoice_number: Optional[str] = None

    # Paths & Hashes
    card_image_path: Optional[str] = None
    receipt_path: Optional[str] = None
    receipt_hash: Optional[str] = None
    serial_number_on_receipt: Optional[str] = None
    model_number_on_receipt: Optional[str] = None
    warranty_card_path: Optional[str] = None
    warranty_card_hash: Optional[str] = None
    serial_number_on_warranty_card: Optional[str] = None
    model_number_on_warranty_card: Optional[str] = None
    product_image_path: Optional[str] = None
    product_image_hash: Optional[str] = None
    fault_evidence_path: Optional[str] = None
    fault_evidence_hash: Optional[str] = None
    fault_video_path: Optional[str] = None
    fault_video_hash: Optional[str] = None
    barcode_image_path: Optional[str] = None
    barcode_image_hash: Optional[str] = None
    serial_number_on_barcode: Optional[str] = None
    model_number_on_barcode: Optional[str] = None
    cross_verification_json: Optional[str] = None

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

    # Decision
    adjudication_status: str
    adjudication_stage: str
    final_confidence: float
    decision_reason_summary: str
    decision_reasons_json: Optional[str] = None

    reviewed_by: Optional[str] = None
    reviewer_notes: Optional[str] = None
    adjudication_timestamp: Optional[str] = None
    created_at: str

    class Config:
        from_attributes = True


class ClaimStatsResponse(BaseModel):
    total_claims: int
    auto_approved_count: int
    auto_approved_pct: float
    auto_rejected_count: int
    auto_rejected_pct: float
    manual_review_count: int
    manual_review_pct: float
    resolved_count: int
    dual_model_agreement_rate: float
    category_counts: Dict[str, int]
    status_counts: Dict[str, int]
