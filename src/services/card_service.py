"""
AssureX Claim Engine — Dynamic Claim Summary Card Service

Generates the official 1200x1680 High-DPI Claim Summary Card for live claim submissions.
Maintains strict compliance with SRS competition guidelines:
- Strictly objective claim facts and verifiable document evidence
- Zero model predictions, confidence scores, or adjudication status on the card canvas
"""

import os
import sys
from typing import Dict, Any
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CARDS_UPLOAD_DIR = os.path.join(BASE_DIR, "uploads", "cards")
os.makedirs(CARDS_UPLOAD_DIR, exist_ok=True)

# Import the card generator engine
try:
    from dataset_generator.generate_cards import render_claim_card
except ImportError:
    try:
        from backend.dataset_generator.generate_cards import render_claim_card
    except ImportError:
        # Fallback path if root is on sys.path
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from dataset_generator.generate_cards import render_claim_card


class CardService:
    """Service to render and manage high-DPI Claim Summary Cards."""

    @staticmethod
    def generate_and_save(claim_dict: Dict[str, Any], variant: int = 0) -> str:
        """
        Renders the 1200x1680 card and saves it to disk.
        Returns the absolute filepath on disk.
        """
        claim_id = claim_dict.get("claim_id", "CLM-UNKNOWN")
        filename = f"{claim_id}_card.png"
        filepath = os.path.join(CARDS_UPLOAD_DIR, filename)

        # Render using the 1200x1680 high-DPI engine
        img: Image.Image = render_claim_card(claim_dict, variant=variant)
        img.save(filepath, "PNG", optimize=True)

        return filepath

    @staticmethod
    def get_card_url(claim_id: str) -> str:
        """Returns the public API URL path for the card."""
        return f"/uploads/cards/{claim_id}_card.png"
