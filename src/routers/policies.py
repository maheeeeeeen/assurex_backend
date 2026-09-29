"""
AssureX Claim Engine — Policies & Thresholds Router
"""

import os
import json
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from src.models import User
from src.auth.service import require_role

router = APIRouter()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
POLICIES_DIR = os.path.join(BASE_DIR, "policies")
THRESHOLDS_PATH = os.path.join(BASE_DIR, "config", "thresholds.json")


@router.get("/")
def get_all_policies(
    current_user: User = Depends(require_role(["reviewer", "admin"]))
):
    """Returns active warranty policy configurations for all product categories dynamically."""
    policies = {}
    if os.path.exists(POLICIES_DIR):
        for fname in sorted(os.listdir(POLICIES_DIR)):
            if fname.endswith("_warranty.json"):
                fpath = os.path.join(POLICIES_DIR, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        cat = data.get("product_category", fname.replace("_warranty.json", "").replace("_", " ").title())
                        policies[cat] = data
                except Exception as e:
                    print(f"Error loading {fname}: {e}")
    return policies


@router.get("/thresholds")
def get_thresholds(
    current_user: User = Depends(require_role(["reviewer", "admin"]))
):
    """Returns model agreement and confidence thresholds."""
    if os.path.exists(THRESHOLDS_PATH):
        try:
            with open(THRESHOLDS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


@router.post("/thresholds")
def update_thresholds(
    thresholds_data: Dict[str, Any],
    current_user: User = Depends(require_role(["admin"]))
):
    """Admin-only endpoint to update model agreement and confidence thresholds."""
    try:
        os.makedirs(os.path.dirname(THRESHOLDS_PATH), exist_ok=True)
        with open(THRESHOLDS_PATH, "w", encoding="utf-8") as f:
            json.dump(thresholds_data, f, indent=2)
        return {"message": "Thresholds successfully updated", "thresholds": thresholds_data}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update thresholds: {str(e)}"
        )
