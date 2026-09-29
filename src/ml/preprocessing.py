"""
AssureX Claim Engine — Tabular ML Preprocessing Pipeline

Preprocesses claim records into numerical vectors suitable for scikit-learn, XGBoost, and LightGBM.
Handles:
- Missing value imputation
- Feature scaling for numerical attributes
- One-hot encoding for categorical attributes
- Boolean feature conversion
- Class label encoding / decoding
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Target class mappings
LABEL_TO_INT = {
    "Likely Valid": 0,
    "Likely Invalid": 1,
    "Manual Review Required": 2,
}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}

NUMERICAL_FEATURES = [
    "purchase_price",
    "product_age_months",
    "remaining_warranty_days",
    "repair_history_count",
    "missing_doc_count",
]

BOOLEAN_FEATURES = [
    "previous_repair_authorized",
    "receipt_uploaded",
    "warranty_card_uploaded",
    "product_image_uploaded",
    "fault_evidence_uploaded",
    "repair_report_uploaded",
]

CATEGORICAL_FEATURES = [
    "product_category",
    "brand",
    "warranty_type",
    "fault_type",
    "damage_type",
]

ALL_INPUT_FEATURES = NUMERICAL_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all required columns exist with proper data types and default values."""
    df_clean = df.copy()

    # Fill numerical defaults
    for col in NUMERICAL_FEATURES:
        if col not in df_clean.columns:
            df_clean[col] = 0.0
        else:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce").fillna(0.0)

    # Fill boolean defaults
    for col in BOOLEAN_FEATURES:
        if col not in df_clean.columns:
            df_clean[col] = 0.0
        else:
            df_clean[col] = df_clean[col].astype(bool).astype(float)

    # Fill categorical defaults
    for col in CATEGORICAL_FEATURES:
        if col not in df_clean.columns:
            df_clean[col] = "Unknown"
        else:
            df_clean[col] = df_clean[col].fillna("Unknown").astype(str)

    return df_clean[ALL_INPUT_FEATURES]


def build_preprocessor() -> ColumnTransformer:
    """Build scikit-learn ColumnTransformer for scaling and encoding."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("bool", "passthrough", BOOLEAN_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def extract_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Extract human-readable feature names from the fitted preprocessor."""
    feature_names = []
    # 1. Numericals
    feature_names.extend(NUMERICAL_FEATURES)
    # 2. Booleans
    feature_names.extend(BOOLEAN_FEATURES)
    # 3. Categoricals
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES)
    feature_names.extend(list(cat_names))
    return feature_names
