"""
AssureX Claim Engine — High-Resolution Claim Summary Card Generator (Phase 1 & Phase 4)

Renders professional, standardized, high-resolution (1200x1680) Claim Summary Cards.
Designed for pixel-perfect clarity, zero text overlap, and complete visual legibility.

CRITICAL DESIGN RULE (SRS Deliverable 5 & Steps 7, 20):
The Claim Summary Card MUST NOT contain:
- Python model prediction
- Teachable Machine prediction
- Confidence scores
- Final decision (Likely Valid / Likely Invalid / Manual Review)

It contains ONLY objective factual data:
- Claim ID, Product Info, Serial Numbers
- Dates (Purchase, Warranty, Fault, Submission)
- Reported Incident & Damage Type
- Document Verification Status
- Objective Discrepancies (Serial mismatch, date contradictions, exclusion flags)
"""

import os
import sys
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
ROOT_DIR = BASE_DIR
CARDS_DIR = os.path.join(ROOT_DIR, "sample_claims", "cards")

CLASS_FOLDER_MAP = {
    "Likely Valid": "likely_valid",
    "Likely Invalid": "likely_invalid",
    "Manual Review Required": "manual_review",
}

# Typography cache
FONT_CACHE = {}


def get_font(size: int, bold: bool = False):
    """Load high-DPI TrueType font with memoization."""
    key = (size, bold)
    if key in FONT_CACHE:
        return FONT_CACHE[key]

    font_paths = [
        r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf",
    ]
    for path in font_paths:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, size)
                FONT_CACHE[key] = font
                return font
            except Exception:
                continue

    font = ImageFont.load_default()
    FONT_CACHE[key] = font
    return font


def render_claim_card(claim, variant=0, width=1200, height=1680):
    """
    Renders an individual high-resolution claim summary card as a PIL Image.
    variant: 0 (Executive Navy) or 1 (Modern Technical Teal)
    Resolution: 1200 x 1680 (2x High-DPI Retina clarity)
    """
    # Color scheme
    if variant == 0:
        header_bg = (15, 23, 42)        # Deep Navy / Slate
        accent_color = (37, 99, 235)     # Royal Blue
        section_hdr_bg = (241, 245, 249) # Cool Gray 100
        pill_bg = (224, 242, 254)       # Light Sky
        pill_fg = (3, 105, 161)
    else:
        header_bg = (19, 78, 74)        # Deep Teal
        accent_color = (13, 148, 136)    # Teal
        section_hdr_bg = (240, 253, 250) # Light Teal Tint
        pill_bg = (204, 251, 241)       # Pale Teal
        pill_fg = (15, 118, 110)

    card_bg = (255, 255, 255)
    card_border = (203, 213, 225)
    text_primary = (15, 23, 42)
    text_secondary = (71, 85, 105)
    text_muted = (148, 163, 184)
    line_divider = (226, 232, 240)

    # Status badge palettes
    green_bg = (220, 252, 231)
    green_fg = (22, 101, 52)
    green_border = (134, 239, 172)

    red_bg = (254, 226, 226)
    red_fg = (153, 27, 27)
    red_border = (252, 165, 165)

    amber_bg = (254, 243, 199)
    amber_fg = (146, 64, 14)
    amber_border = (252, 211, 77)

    # Create canvas
    img = Image.new("RGB", (width, height), card_bg)
    draw = ImageDraw.Draw(img)

    # High-DPI Fonts
    f_title = get_font(30, bold=True)
    f_sub = get_font(18, bold=False)
    f_sec_hdr = get_font(21, bold=True)
    f_body = get_font(19, bold=False)
    f_body_bold = get_font(19, bold=True)
    f_large_bold = get_font(22, bold=True)
    f_badge = get_font(18, bold=True)
    f_small = get_font(15, bold=False)
    f_small_bold = get_font(15, bold=True)

    # Outer border (Crisp double line)
    draw.rectangle([(12, 12), (width - 13, height - 13)], outline=card_border, width=3)
    draw.rectangle([(16, 16), (width - 17, height - 17)], outline=(241, 245, 249), width=2)

    # Top Header Banner
    draw.rectangle([(20, 20), (width - 20, 140)], fill=header_bg)
    draw.text((50, 42), "ASSUREX CLAIM SUMMARY DOSSIER", fill=(255, 255, 255), font=f_title)
    draw.text((50, 88), "OBJECTIVE VERIFICATION & FACT SHEET — NO MODEL PREDICTIONS", fill=(148, 163, 184), font=f_sub)

    # Claim ID Tag (Top Right)
    claim_id = str(claim.get("claim_id", "CLM-00000"))
    draw.rounded_rectangle([(width - 270, 42), (width - 50, 112)], radius=8, fill=(255, 255, 255))
    draw.text((width - 245, 58), claim_id, fill=header_bg, font=f_large_bold)

    y = 165
    left_x = 50
    right_x = width - 50

    def draw_section_header(title, cur_y):
        """Draw a full-width structured section header."""
        draw.rounded_rectangle([(left_x, cur_y), (right_x, cur_y + 44)], radius=4, fill=section_hdr_bg)
        draw.rounded_rectangle([(left_x, cur_y), (left_x + 8, cur_y + 44)], radius=2, fill=accent_color)
        draw.text((left_x + 24, cur_y + 9), title, fill=text_primary, font=f_sec_hdr)
        return cur_y + 60

    # =========================================================================
    # SECTION 1: PRODUCT & PURCHASE IDENTIFICATION
    # =========================================================================
    y = draw_section_header("1. PRODUCT & PURCHASE IDENTIFICATION", y)

    brand = str(claim.get("brand", "N/A"))
    p_name = str(claim.get("product_name", "N/A"))
    category = str(claim.get("product_category", "N/A"))
    model = str(claim.get("model_number", "N/A"))
    serial_entered = str(claim.get("serial_number_entered", "N/A"))
    purchase_date = str(claim.get("purchase_date", "N/A"))
    purchase_price = f"${float(claim.get('purchase_price', 0.0)):,.2f}"
    retailer = str(claim.get("retailer", "N/A"))

    # Dedicated Full-Width Row for Product Name
    draw.text((left_x + 20, y), "Product Name:", fill=text_secondary, font=f_body)
    draw.text((left_x + 230, y), f"{brand} {p_name}", fill=text_primary, font=f_large_bold)
    y += 38

    # 2-Column Grid
    c1_lbl = left_x + 20
    c1_val = left_x + 230
    c2_lbl = 650
    c2_val = 860

    # Row 2: Category | Brand
    draw.text((c1_lbl, y), "Product Category:", fill=text_secondary, font=f_body)
    draw.text((c1_val, y), category, fill=text_primary, font=f_body_bold)
    draw.text((c2_lbl, y), "Manufacturer / Brand:", fill=text_secondary, font=f_body)
    draw.text((c2_val, y), brand, fill=text_primary, font=f_body_bold)
    y += 34

    # Row 3: Model # | Serial Entered
    draw.text((c1_lbl, y), "Model Number:", fill=text_secondary, font=f_body)
    draw.text((c1_val, y), model, fill=text_primary, font=f_body)
    draw.text((c2_lbl, y), "Entered Serial #:", fill=text_secondary, font=f_body)
    draw.text((c2_val, y), serial_entered, fill=text_primary, font=f_body_bold)
    y += 34

    # Row 4: Purchase Date | Retailer & Price
    draw.text((c1_lbl, y), "Purchase Date:", fill=text_secondary, font=f_body)
    draw.text((c1_val, y), purchase_date, fill=text_primary, font=f_body)
    draw.text((c2_lbl, y), "Authorized Retailer:", fill=text_secondary, font=f_body)
    ret_str = f"{retailer} ({purchase_price})"
    if len(ret_str) > 28:
        ret_str = ret_str[:25] + "..."
    draw.text((c2_val, y), ret_str, fill=text_primary, font=f_body)
    y += 52

    # =========================================================================
    # SECTION 2: WARRANTY TERMS & COVERAGE PERIOD
    # =========================================================================
    y = draw_section_header("2. WARRANTY TERMS & COVERAGE PERIOD", y)

    w_type = str(claim.get("warranty_type", "Standard"))
    provider = str(claim.get("warranty_provider", "Manufacturer"))
    w_start = str(claim.get("warranty_start", "N/A"))
    w_end = str(claim.get("warranty_end", "N/A"))
    remaining_days = int(claim.get("remaining_warranty_days", 0))

    # Row 1: Coverage Type | Warranty Provider
    draw.text((c1_lbl, y), "Coverage Plan:", fill=text_secondary, font=f_body)
    draw.text((c1_val, y), f"{w_type} Warranty", fill=text_primary, font=f_body_bold)
    draw.text((c2_lbl, y), "Warranty Provider:", fill=text_secondary, font=f_body)
    draw.text((c2_val, y), provider, fill=text_primary, font=f_body_bold)
    y += 34

    # Row 2: Policy Period | Days Count
    draw.text((c1_lbl, y), "Policy Duration:", fill=text_secondary, font=f_body)
    draw.text((c1_val, y), f"{w_start}  to  {w_end}", fill=text_primary, font=f_body)
    draw.text((c2_lbl, y), "Days Remaining:", fill=text_secondary, font=f_body)
    days_label = f"{remaining_days} days active" if remaining_days >= 0 else f"{-remaining_days} days overdue"
    draw.text((c2_val, y), days_label, fill=text_primary, font=f_body_bold)
    y += 36

    # Row 3: Warranty Status Badge
    draw.text((c1_lbl, y), "Warranty Status:", fill=text_secondary, font=f_body)
    if remaining_days >= 0:
        status_text = f"ACTIVE ({remaining_days} days remaining)"
        s_bg, s_fg, s_bd = green_bg, green_fg, green_border
    elif -7 <= remaining_days < 0:
        status_text = f"GRACE PERIOD ({-remaining_days} days past expiry)"
        s_bg, s_fg, s_bd = amber_bg, amber_fg, amber_border
    else:
        status_text = f"EXPIRED ({-remaining_days} days past coverage)"
        s_bg, s_fg, s_bd = red_bg, red_fg, red_border

    bbox = f_badge.getbbox(status_text)
    badge_w = (bbox[2] - bbox[0]) + 36
    draw.rounded_rectangle([(c1_val, y - 4), (c1_val + badge_w, y + 30)], radius=6, fill=s_bg, outline=s_bd, width=1)
    draw.text((c1_val + 18, y + 2), status_text, fill=s_fg, font=f_badge)
    y += 54

    # =========================================================================
    # SECTION 3: INCIDENT CLASSIFICATION & DAMAGE
    # =========================================================================
    y = draw_section_header("3. INCIDENT CLASSIFICATION & DAMAGE", y)

    fault_date = str(claim.get("fault_date", "N/A"))
    sub_date = str(claim.get("claim_submission_date", "N/A"))
    fault_type = str(claim.get("fault_type", "N/A"))
    damage_type = str(claim.get("damage_type", "N/A"))
    fault_desc = str(claim.get("fault_description", "No narrative provided."))

    # Row: Fault Date | Submission Date
    draw.text((c1_lbl, y), "Fault Incident Date:", fill=text_secondary, font=f_body)
    draw.text((c1_val, y), fault_date, fill=text_primary, font=f_body_bold)
    draw.text((c2_lbl, y), "Claim Filed Date:", fill=text_secondary, font=f_body)
    draw.text((c2_val, y), sub_date, fill=text_primary, font=f_body)
    y += 34

    # Row: Incident Classification | Damage Type
    draw.text((c1_lbl, y), "Reported Incident:", fill=text_secondary, font=f_body)
    draw.text((c1_val, y), fault_type, fill=text_primary, font=f_body)
    draw.text((c2_lbl, y), "Damage Type:", fill=text_secondary, font=f_body)
    draw.text((c2_val, y), damage_type, fill=text_primary, font=f_body_bold)
    y += 42

    # Narrative Container with Clean Background
    draw.rounded_rectangle([(left_x + 10, y), (right_x - 10, y + 80)], radius=6, fill=(248, 250, 252), outline=line_divider, width=1)
    draw.text((left_x + 28, y + 10), "CUSTOMER REPORTED INCIDENT NARRATIVE", fill=text_muted, font=f_small_bold)
    if len(fault_desc) > 96:
        fault_desc = fault_desc[:93] + "..."
    draw.text((left_x + 28, y + 36), f'"{fault_desc}"', fill=text_primary, font=f_body)
    y += 105

    # =========================================================================
    # SECTION 4: REPAIR HISTORY & DOCUMENTATION AUDIT
    # =========================================================================
    y = draw_section_header("4. REPAIR HISTORY & DOCUMENTATION AUDIT", y)

    repairs = int(claim.get("repair_history_count", 0))
    auth_repair = bool(claim.get("previous_repair_authorized", True))

    rep_summary = f"{repairs} prior service repair(s)"
    if repairs > 0:
        rep_summary += f" — {'Authorized Service Center' if auth_repair else 'UNAUTHORIZED Repair Facility'}"
    else:
        rep_summary += " — Original Factory State"

    draw.text((c1_lbl, y), "Prior Service History:", fill=text_secondary, font=f_body)
    draw.text((c1_val, y), rep_summary, fill=text_primary, font=f_body_bold)
    y += 38

    draw.text((c1_lbl, y), "Submitted Documents Checklist:", fill=text_secondary, font=f_body)
    y += 32

    # 4 Evenly Spaced Document Cards
    doc_specs = [
        ("Purchase Receipt", bool(claim.get("receipt_uploaded", True))),
        ("Warranty Card", bool(claim.get("warranty_card_uploaded", True))),
        ("Fault Photo/Video", bool(claim.get("fault_evidence_uploaded", True))),
        ("Product Overview", bool(claim.get("product_image_uploaded", True))),
    ]

    card_w = 255
    spacing = 26
    start_bx = left_x + 10

    for i, (name, present) in enumerate(doc_specs):
        bx = start_bx + i * (card_w + spacing)
        bg = green_bg if present else red_bg
        fg = green_fg if present else red_fg
        bd = green_border if present else red_border
        status_tag = "[YES] VERIFIED" if present else "[NO] MISSING"

        draw.rounded_rectangle([(bx, y), (bx + card_w, y + 54)], radius=6, fill=bg, outline=bd, width=1)
        draw.text((bx + 16, y + 8), name, fill=fg, font=f_small_bold)
        draw.text((bx + 16, y + 28), status_tag, fill=fg, font=f_badge)

    y += 78

    # =========================================================================
    # SECTION 5: SYSTEM DISCREPANCY & AUDIT FLAGS
    # =========================================================================
    y = draw_section_header("5. SYSTEM DISCREPANCY & AUDIT FLAGS", y)

    sn_mismatch = bool(claim.get("serial_mismatch_flag", False))
    date_contradiction = bool(claim.get("date_contradiction_flag", False))
    if not date_contradiction:
        try:
            p_dt = str(claim.get("purchase_date", ""))[:10]
            f_dt = str(claim.get("fault_date", ""))[:10]
            if p_dt and f_dt and p_dt > f_dt:
                date_contradiction = True
        except Exception:
            pass
    excluded = bool(claim.get("excluded_damage", False))
    duplicate = bool(claim.get("duplicate_claim_flag", False))

    flags = [
        ("Serial Cross-Match Check", "MATCH CONFIRMED" if not sn_mismatch else "MISMATCH DETECTED", not sn_mismatch),
        ("Date Chronology Verification", "CHRONOLOGICAL VALID" if not date_contradiction else "CONTRADICTION DETECTED", not date_contradiction),
        ("Policy Exclusion Screening", "NO EXCLUSIONS MATCHED" if not excluded else "EXCLUSION DETECTED", not excluded),
        ("Duplicate Submission Screening", "UNIQUE SUBMISSION" if not duplicate else "POTENTIAL DUPLICATE", not duplicate),
    ]

    for label, val_text, is_valid in flags:
        bg = green_bg if is_valid else red_bg
        fg = green_fg if is_valid else red_fg
        bd = green_border if is_valid else red_border

        draw.text((c1_lbl, y), f"•  {label}:", fill=text_secondary, font=f_body)

        # Dynamic badge with generous padding
        bbox = f_badge.getbbox(val_text)
        bw = (bbox[2] - bbox[0]) + 48
        draw.rounded_rectangle([(420, y - 4), (420 + bw, y + 30)], radius=6, fill=bg, outline=bd, width=1)
        draw.text((444, y + 2), val_text, fill=fg, font=f_badge)
        y += 42

    # =========================================================================
    # FOOTER: Machine-Learning / Teachable Machine Verification Metadata
    # =========================================================================
    footer_y = height - 90
    draw.line([(left_x, footer_y), (right_x, footer_y)], fill=line_divider, width=2)
    draw.text((left_x + 10, footer_y + 16), "AssureX Automated Evidence Dossier — Machine Learning Verification Format", fill=text_muted, font=f_small_bold)
    draw.text((left_x + 10, footer_y + 40), "CONFIDENTIAL DOCUMENT RECORD — FOR VERIFICATION & COMPUTER VISION CLASSIFICATION ONLY", fill=text_muted, font=f_small)

    # Simulated Barcode Security Lines (Right side of footer)
    barcode_x = right_x - 220
    for i in range(36):
        bx = barcode_x + (i * 6)
        bw = 3 if (i % 4 == 0 or i % 7 == 0) else 1
        draw.line([(bx, footer_y + 12), (bx, footer_y + 60)], fill=(71, 85, 105), width=bw)

    return img


def generate_single_card_task(args):
    """Worker task for multiprocessing with adaptive palette optimization."""
    row_dict, split, variant, save_dir = args
    claim_id = row_dict["claim_id"]
    cls_name = row_dict["class_label"]
    folder_cls = CLASS_FOLDER_MAP.get(cls_name, "manual_review")

    target_dir = os.path.join(save_dir, split, folder_cls)
    os.makedirs(target_dir, exist_ok=True)

    filename = f"{claim_id}_var{variant}.png"
    filepath = os.path.join(target_dir, filename)

    img = render_claim_card(row_dict, variant=variant)
    # Adaptive 256-color palette maintains full 1200x1680 Retina clarity with 60% disk savings
    img_opt = img.convert("P", palette=Image.ADAPTIVE, colors=256)
    img_opt.save(filepath, "PNG", optimize=True)

    rel_path = os.path.relpath(filepath, ROOT_DIR).replace("\\", "/")
    return {
        "claim_id": claim_id,
        "split": split,
        "variant": variant,
        "image_path": rel_path,
        "class_label": cls_name,
    }


def batch_generate_cards(max_workers=None):
    """
    Batch generate high-resolution card images across splits.
    Train: 7,000 x 2 = 14,000
    Val: 1,500 x 1 = 1,500
    Test: 1,500 x 1 = 1,500
    Total: 17,000
    """
    train_path = os.path.join(DATA_DIR, "claims_train.csv")
    val_path = os.path.join(DATA_DIR, "claims_val.csv")
    test_path = os.path.join(DATA_DIR, "claims_test.csv")

    if not all(os.path.exists(p) for p in [train_path, val_path, test_path]):
        raise FileNotFoundError("Split CSVs not found! Run generate_claims.py first.")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    tasks = []
    for _, row in train_df.iterrows():
        tasks.append((row.to_dict(), "train", 0, CARDS_DIR))
        tasks.append((row.to_dict(), "train", 1, CARDS_DIR))

    for _, row in val_df.iterrows():
        tasks.append((row.to_dict(), "val", 0, CARDS_DIR))

    for _, row in test_df.iterrows():
        tasks.append((row.to_dict(), "test", 0, CARDS_DIR))

    total_tasks = len(tasks)
    print(f"Generating {total_tasks} High-Resolution Claim Summary Cards (1200x1680)...")

    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), 8)

    mapping_records = []
    completed = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(generate_single_card_task, t) for t in tasks]
        for f in as_completed(futures):
            res = f.result()
            mapping_records.append(res)
            completed += 1
            if completed % 500 == 0 or completed == total_tasks:
                print(f"Rendered {completed}/{total_tasks} high-resolution cards ({completed/total_tasks*100:.1f}%)...")

    mapping_df = pd.DataFrame(mapping_records)
    mapping_path = os.path.join(DATA_DIR, "card_image_mapping.csv")
    mapping_df.to_csv(mapping_path, index=False)
    print(f"\nSaved image mapping to {mapping_path} ({len(mapping_df)} records)")


if __name__ == "__main__":
    batch_generate_cards()
