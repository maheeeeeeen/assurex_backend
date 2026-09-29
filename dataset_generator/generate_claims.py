"""
AssureX Claim Engine — Scaled Synthetic Claim Dataset Generator (10,000 Records)

Generates 10,000 realistic, unique warranty claim records:
- Perfectly balanced across 3 classes (~3,334 Valid, ~3,333 Invalid, ~3,333 Manual Review)
- Zero feature duplicates: Enforces SHA-256 normalized feature hashing with collision rejection
- Diverse product categories, brands, fault modes, and edge cases
- Stratified 70/15/15 split (7,000 train / 1,500 val / 1,500 test) with 0 Claim ID or feature leakage
"""

import hashlib
import json
import os
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Expanded realistic product catalog across 6 categories
PRODUCT_CATALOG = {
    "Electronics": {
        "brands": ["Sony", "Samsung", "LG", "Panasonic", "TCL", "Hisense", "Vizio", "Philips"],
        "items": [
            ("OLED 4K Smart TV 65-inch", 1499.0, 24),
            ("QLED HDR Television 55-inch", 899.0, 24),
            ("Home Theater AV Receiver 7.2", 650.0, 24),
            ("Soundbar with Wireless Subwoofer", 380.0, 24),
            ("Mirrorless Full-Frame Camera", 1750.0, 24),
            ("Compact Digital Vlogging Camera", 550.0, 24),
            ("Smart Projector 4K Ultra", 1250.0, 24),
            ("Blu-ray 4K Disc Player", 220.0, 12),
        ],
        "covered_faults": [
            "Manufacturing Defect",
            "Electrical Failure",
            "Control Board Failure",
            "Display Artifacts",
            "Audio Processing Malfunction",
            "HDMI Port Failure",
            "Power Supply Failure",
        ],
        "excluded_damages": [
            "Accidental Drop Damage",
            "Liquid Spill Damage",
            "Screen Crack from Impact",
            "Power Surge Damage",
            "Unauthorized Modification",
            "Cosmetic Scratches and Dents",
        ],
        "retailers": ["BestBuy", "Amazon Electronics", "B&H Photo", "Walmart", "Direct Manufacturer Store", "Target", "Costco Wholesale"],
    },
    "Appliances": {
        "brands": ["Whirlpool", "Bosch", "LG", "Samsung", "GE Appliances", "KitchenAid", "Maytag", "Electrolux"],
        "items": [
            ("French Door Refrigerator 28 cu. ft.", 1850.0, 36),
            ("Front-Load Smart Washer 4.5 cu. ft.", 950.0, 36),
            ("Smart Electric Dryer 7.4 cu. ft.", 900.0, 36),
            ("Quiet Built-In Dishwasher 24-inch", 820.0, 36),
            ("Slide-In Induction Range Oven", 1450.0, 36),
            ("Over-the-Range Convection Microwave", 380.0, 24),
            ("Smart Inverter Air Conditioner 12000 BTU", 580.0, 24),
            ("HEPA Air Purifier Whole-Home", 320.0, 24),
        ],
        "covered_faults": [
            "Manufacturing Defect",
            "Motor Failure",
            "Compressor Failure",
            "Thermostat Malfunction",
            "Internal Wiring Defect",
            "Water Valve Leakage",
            "Heating Element Failure",
            "Control Panel Malfunction",
        ],
        "excluded_damages": [
            "Accidental Physical Damage",
            "Voltage Surge Damage",
            "Pest Infestation Damage",
            "Commercial Use of Domestic Unit",
            "Damage from Improper Installation",
            "Cosmetic Wear and Tear",
            "Lime Scale Mineral Clogging",
        ],
        "retailers": ["Home Depot", "Lowe's", "Sears Outlet", "Costco", "Direct Factory Outlet", "Menards", "Appliance Direct"],
    },
    "Automotive": {
        "brands": ["Bosch Auto", "Denso", "Brembo", "ACDelco", "Monroe", "Continental Auto", "K&N", "Magnaflow"],
        "items": [
            ("Ceramic Performance Brake Pad Set", 140.0, 12),
            ("High-Output Heavy-Duty Alternator", 380.0, 24),
            ("Electronic Starter Motor Assembly", 290.0, 24),
            ("Multi-Port Fuel Injector Rail Kit", 460.0, 24),
            ("Adaptive Shock Absorber Set", 420.0, 24),
            ("ABS Wheel Speed Sensor System", 115.0, 12),
            ("Direct-Fit Catalytic Converter", 680.0, 24),
            ("High-Capacity AGM Car Battery", 240.0, 36),
        ],
        "covered_faults": [
            "Manufacturing Defect",
            "Premature Bearing Wear",
            "Material Structural Failure",
            "Internal Coil Short Circuit",
            "Seal and Gasket Blowout",
            "Sensor Circuit Discontinuity",
        ],
        "excluded_damages": [
            "Accidental Collision Damage",
            "Racing and Track Abuse",
            "Damage from Incompatible Fluids",
            "Improper DIY Installation",
            "Neglected Fluid Maintenance",
            "Environmental Road Salt Corrosion",
        ],
        "retailers": ["AutoZone", "Advance Auto Parts", "O'Reilly Auto", "NAPA Auto Parts", "RockAuto", "CarParts.com"],
    },
    "Smartphones & Mobile": {
        "brands": ["Apple", "Samsung", "Google", "OnePlus", "Motorola", "Xiaomi"],
        "items": [
            ("Pro Max Flagship Smartphone 256GB", 1199.0, 24),
            ("Foldable Dual-Screen Smartphone", 1699.0, 24),
            ("5G Mid-Range Smartphone 128GB", 499.0, 24),
            ("Compact Travel Smartphone 128GB", 399.0, 24),
            ("High-Performance Tablet 12.9-inch", 999.0, 24),
            ("Rugged Outdoor Smartphone IP68", 550.0, 24),
        ],
        "covered_faults": [
            "Manufacturing Defect",
            "OLED Display Flickering",
            "Cellular Modem Signal Drop",
            "Microphone Component Failure",
            "Battery Premature Swelling",
            "Motherboard Power Controller Crash",
        ],
        "excluded_damages": [
            "Shattered Screen from Drop",
            "Submersion Water Damage",
            "Unauthorized Screen Replacement",
            "Jailbreak or Unofficial ROM Brick",
            "Cracked Rear Glass Chassis",
        ],
        "retailers": ["Verizon Wireless", "AT&T Store", "T-Mobile", "Apple Store", "BestBuy Mobile", "Amazon"],
    },
    "Computers & Laptops": {
        "brands": ["Dell", "HP", "Lenovo", "Apple", "Asus", "Acer", "MSI"],
        "items": [
            ("Ultra-Thin 14-inch Laptop 16GB RAM", 1050.0, 24),
            ("Workstation Laptop 32GB GPU", 1950.0, 36),
            ("Gaming Laptop 16-inch 165Hz", 1450.0, 24),
            ("Mini Business Desktop PC", 720.0, 36),
            ("Ultrawide Curved Monitor 34-inch", 580.0, 36),
            ("Network Attached Storage 4-Bay", 490.0, 24),
        ],
        "covered_faults": [
            "Manufacturing Defect",
            "Motherboard Capacitor Failure",
            "Cooling Fan Bearing Seizure",
            "RAM Controller Failure",
            "Backlight Inverter Dropout",
            "Power Brick Voltage Instability",
        ],
        "excluded_damages": [
            "Liquid Spill on Keyboard",
            "Hinge Fracture from Drop",
            "Overclocking Thermal Damage",
            "Unauthorized Component Soldering",
            "Chassis Impact Deformation",
        ],
        "retailers": ["Newegg", "Micro Center", "Amazon Computers", "Dell Direct", "BestBuy", "CDW"],
    },
    "Wearables & Audio": {
        "brands": ["Sony", "Bose", "Sennheiser", "Garmin", "Apple", "Fitbit", "JBL"],
        "items": [
            ("Active Noise Cancelling Wireless Over-Ear", 349.0, 24),
            ("True Wireless Noise-Cancelling Earbuds", 229.0, 24),
            ("GPS Multisport Smartwatch", 449.0, 24),
            ("Fitness Activity Tracker Band", 129.0, 12),
            ("Portable Bluetooth Rugged Speaker", 169.0, 24),
            ("Audiophile Open-Back Studio Headphones", 499.0, 24),
        ],
        "covered_faults": [
            "Manufacturing Defect",
            "Driver Voice Coil Failure",
            "Bluetooth Radio Intermittent Sync",
            "Heart Rate Sensor Inaccuracy",
            "Charging Pin Contact Failure",
            "Active Noise Cancellation Distortion",
        ],
        "excluded_damages": [
            "Crushed Ear-Cup Impact",
            "Saltwater Corrosion",
            "Torn Headband Fabric",
            "Cable Snag Damage",
            "Dog Chew Damage",
        ],
        "retailers": ["Amazon Audio", "BestBuy", "Crutchfield", "Target", "Direct Brand Store", "REI Co-op"],
    },
}

FAULT_DESCRIPTIONS = {
    "Manufacturing Defect": [
        "Internal solder joint separated during normal operation, halting power delivery.",
        "PCB trace delamination caused intermittent shutoffs under standard operating load.",
        "Factory assembly misalignment resulted in mechanical binding and device stall.",
        "Primary controller IC failed to initialize following unprompted device reboot.",
    ],
    "Electrical Failure": [
        "Internal low-voltage rail shorted to ground without external power surge.",
        "Capacitor ruptured on main logic board under standard ambient operating temperature.",
        "No current flowing through secondary transformer windings despite active AC input.",
        "Switch-mode power controller failed, rendering the unit completely unpowered.",
    ],
    "Motor Failure": [
        "Brushless drive motor stator winding open-circuit detected; rotor does not turn.",
        "Motor bearings degraded prematurely, producing high-pitched screech and thermal lockout.",
        "Drive shaft seized under rated load conditions with zero external obstruction.",
    ],
    "Compressor Failure": [
        "Hermetic compressor motor locked rotor amps exceeded, tripping thermal protector.",
        "Internal valve reed cracked, preventing pressure differential and cooling cycle.",
        "Compressor clutch coil burned out with normal electrical supply.",
    ],
    "Thermostat Malfunction": [
        "Internal thermistor reading drifted 40 degrees off calibration, causing system runaway.",
        "Bimetallic thermostat contact welded shut, preventing temperature regulation.",
        "Digital thermal sensor communicates checksum errors continuously on I2C bus.",
    ],
    "Display Artifacts": [
        "Permanent vertical green line appeared across pixel column without physical impact.",
        "Panel backlight inverter failed on the left quadrant, causing severe dimming.",
        "OLED sub-pixels degraded prematurely with permanent persistent retention.",
    ],
    "Component Failure": [
        "Integrated sensor array stopped communicating over internal peripheral bus.",
        "Cooling fan PWM tachometer line disconnected, triggering emergency shutdown.",
        "Micro-switch contact resistance spiked to mega-ohms, preventing actuation.",
    ],
}


def make_feature_hash(record_dict):
    """
    Computes a deterministic SHA-256 hash across all normalized features.
    Guarantees absolute zero duplicates or collisions across the dataset.
    """
    keys_to_hash = [
        "product_category", "brand", "model_number", "purchase_date",
        "purchase_price", "retailer", "warranty_start", "warranty_end",
        "fault_date", "claim_submission_date", "fault_type", "damage_type",
        "serial_number_entered", "serial_number_on_receipt",
        "repair_history_count", "previous_repair_authorized",
        "receipt_uploaded", "warranty_card_uploaded", "product_image_uploaded",
        "fault_evidence_uploaded", "repair_report_uploaded",
        "serial_mismatch_flag", "date_contradiction_flag",
        "excluded_damage", "duplicate_claim_flag"
    ]
    norm_str = "|".join(str(record_dict.get(k, "")) for k in keys_to_hash)
    return hashlib.sha256(norm_str.encode("utf-8")).hexdigest()


def generate_single_claim(claim_idx, category, target_class):
    """
    Generates a single realistic claim record for the specified target class.
    """
    claim_id = f"CLM-{claim_idx:05d}"
    product_id = f"PRD-{random.randint(10000, 99999)}"

    cat_info = PRODUCT_CATALOG[category]
    brand = random.choice(cat_info["brands"])
    item_tuple = random.choice(cat_info["items"])
    item_name, base_price, default_warranty_months = item_tuple

    # Price jitter (+/- 12%)
    price_factor = 1.0 + random.uniform(-0.12, 0.12)
    purchase_price = round(base_price * price_factor, 2)
    retailer = random.choice(cat_info["retailers"])
    model_number = f"{brand[:3].upper()}-{category[:3].upper()}-{random.randint(100, 999)}"

    # Generate dates
    today = datetime(2026, 3, 20)
    # Purchase between 6 months and 4 years ago
    days_ago_purchase = random.randint(180, 1400)
    purchase_date = today - timedelta(days=days_ago_purchase)
    warranty_start = purchase_date

    # Warranty duration: standard vs extended
    warranty_type = random.choice(["Standard", "Standard", "Extended 1-Year", "Extended 2-Year"])
    extra_months = 0
    if warranty_type == "Extended 1-Year":
        extra_months = 12
    elif warranty_type == "Extended 2-Year":
        extra_months = 24
    total_warranty_months = default_warranty_months + extra_months
    warranty_end = warranty_start + timedelta(days=int(total_warranty_months * 30.4375))
    warranty_provider = f"{brand} Care Protection"

    # Serials
    base_sn = f"SN{brand[:2].upper()}{random.randint(100000, 999999)}"
    serial_entered = base_sn
    serial_receipt = base_sn
    serial_warranty_card = base_sn

    # Flags default
    receipt_uploaded = True
    warranty_card_uploaded = True
    product_image_uploaded = True
    fault_evidence_uploaded = True
    repair_report_uploaded = False
    previous_repair_authorized = True
    repair_history_count = 0
    serial_mismatch_flag = False
    date_contradiction_flag = False
    excluded_damage = False
    duplicate_claim_flag = False

    # -------------------------------------------------------------
    # Scenario Generation by Class
    # -------------------------------------------------------------
    if target_class == "Likely Valid":
        # Fault occurred strictly INSIDE warranty period (at least 20 days before expiry)
        total_warranty_days = max(40, (warranty_end - warranty_start).days)
        fault_offset = random.randint(20, max(21, total_warranty_days - 25))
        fault_date = warranty_start + timedelta(days=fault_offset)
        claim_submission_date = fault_date + timedelta(days=random.randint(1, 14))

        fault_type = random.choice(cat_info["covered_faults"])
        damage_type = "Manufacturing"
        warranty_card_uploaded = random.random() > 0.08

        # 0 or 1 authorized previous repair
        repair_history_count = random.choice([0, 0, 0, 1])
        if repair_history_count > 0:
            previous_repair_authorized = True
            repair_report_uploaded = True

    elif target_class == "Likely Invalid":
        # Dispositive hard-fail reasons
        fail_mode = random.choice([
            "expired_warranty",
            "excluded_damage",
            "no_receipt",
            "unauthorized_repair_multi",
            "duplicate_claim",
        ])

        if fail_mode == "expired_warranty":
            # Occurred 20 to 240 days past warranty expiration
            days_past = random.randint(20, 240)
            fault_date = warranty_end + timedelta(days=days_past)
            claim_submission_date = fault_date + timedelta(days=random.randint(2, 20))
            fault_type = random.choice(cat_info["covered_faults"])
            damage_type = "Manufacturing"

        elif fail_mode == "excluded_damage":
            total_warranty_days = max(40, (warranty_end - warranty_start).days)
            fault_date = warranty_start + timedelta(days=random.randint(20, max(21, total_warranty_days - 20)))
            claim_submission_date = fault_date + timedelta(days=random.randint(1, 14))
            damage_type = random.choice(cat_info["excluded_damages"])
            fault_type = damage_type
            excluded_damage = True

        elif fail_mode == "no_receipt":
            total_warranty_days = max(40, (warranty_end - warranty_start).days)
            fault_date = warranty_start + timedelta(days=random.randint(20, max(21, total_warranty_days - 20)))
            claim_submission_date = fault_date + timedelta(days=random.randint(1, 14))
            fault_type = random.choice(cat_info["covered_faults"])
            damage_type = "Manufacturing"
            receipt_uploaded = False  # Hard fail: mandatory proof of purchase missing

        elif fail_mode == "unauthorized_repair_multi":
            total_warranty_days = max(40, (warranty_end - warranty_start).days)
            fault_date = warranty_start + timedelta(days=random.randint(20, max(21, total_warranty_days - 20)))
            claim_submission_date = fault_date + timedelta(days=random.randint(1, 14))
            fault_type = random.choice(cat_info["covered_faults"])
            damage_type = "Unauthorized Modification"
            repair_history_count = random.randint(2, 4)
            previous_repair_authorized = False
            excluded_damage = True

        else:  # duplicate_claim
            total_warranty_days = max(40, (warranty_end - warranty_start).days)
            fault_date = warranty_start + timedelta(days=random.randint(20, max(21, total_warranty_days - 20)))
            claim_submission_date = fault_date + timedelta(days=random.randint(1, 14))
            fault_type = random.choice(cat_info["covered_faults"])
            damage_type = "Manufacturing"
            duplicate_claim_flag = True

    else:  # Manual Review Required
        # Ambiguous, borderline, or warning edge-cases
        review_mode = random.choice([
            "grace_period_borderline",
            "serial_mismatch_typo",
            "date_contradiction_entry_error",
            "unauthorized_repair_single",
            "high_repair_count_warning",
            "missing_secondary_evidence",
        ])

        if review_mode == "grace_period_borderline":
            # Fault occurred 1 to 14 days after expiration (within discretionary grace period)
            fault_date = warranty_end + timedelta(days=random.randint(1, 14))
            claim_submission_date = fault_date + timedelta(days=random.randint(1, 7))
            fault_type = random.choice(cat_info["covered_faults"])
            damage_type = "Manufacturing"

        elif review_mode == "serial_mismatch_typo":
            total_warranty_days = max(40, (warranty_end - warranty_start).days)
            fault_date = warranty_start + timedelta(days=random.randint(20, max(21, total_warranty_days - 20)))
            claim_submission_date = fault_date + timedelta(days=random.randint(1, 10))
            fault_type = random.choice(cat_info["covered_faults"])
            damage_type = "Manufacturing"
            # Receipt has typo in serial (last 2 digits swapped/altered)
            serial_receipt = base_sn[:-2] + str(random.randint(10, 99))
            serial_mismatch_flag = True

        elif review_mode == "date_contradiction_entry_error":
            # Fault date mistakenly set before purchase date or claim date before fault date
            fault_date = purchase_date - timedelta(days=random.randint(5, 30))
            claim_submission_date = purchase_date + timedelta(days=random.randint(5, 25))
            fault_type = random.choice(cat_info["covered_faults"])
            damage_type = "Manufacturing"
            date_contradiction_flag = True

        elif review_mode == "unauthorized_repair_single":
            total_warranty_days = max(40, (warranty_end - warranty_start).days)
            fault_date = warranty_start + timedelta(days=random.randint(20, max(21, total_warranty_days - 20)))
            claim_submission_date = fault_date + timedelta(days=random.randint(1, 10))
            fault_type = random.choice(cat_info["covered_faults"])
            damage_type = "Component Failure"
            repair_history_count = 1
            previous_repair_authorized = False  # Warning Rule MR-003

        elif review_mode == "high_repair_count_warning":
            total_warranty_days = max(40, (warranty_end - warranty_start).days)
            fault_date = warranty_start + timedelta(days=random.randint(20, max(21, total_warranty_days - 20)))
            claim_submission_date = fault_date + timedelta(days=random.randint(1, 10))
            fault_type = random.choice(cat_info["covered_faults"])
            damage_type = "Manufacturing"
            repair_history_count = 3  # Near replacement limit
            previous_repair_authorized = True
            repair_report_uploaded = True

        else:  # missing_secondary_evidence
            total_warranty_days = max(40, (warranty_end - warranty_start).days)
            fault_date = warranty_start + timedelta(days=random.randint(20, max(21, total_warranty_days - 20)))
            claim_submission_date = fault_date + timedelta(days=random.randint(1, 10))
            fault_type = random.choice(cat_info["covered_faults"])
            damage_type = "Electrical Failure"
            fault_evidence_uploaded = False  # Missing photo evidence, but receipt is present

    # Pick fault description
    desc_list = FAULT_DESCRIPTIONS.get(fault_type, FAULT_DESCRIPTIONS.get(damage_type, [
        "Unspecified system degradation observed during normal operating routine."
    ]))
    fault_description = random.choice(desc_list)

    # Derived metrics
    product_age_months = max(0, int((fault_date - purchase_date).days / 30.4375))
    remaining_warranty_days = (warranty_end - fault_date).days

    all_possible_docs = [
        receipt_uploaded,
        warranty_card_uploaded,
        product_image_uploaded,
        fault_evidence_uploaded,
        repair_report_uploaded if repair_history_count > 0 else True,
    ]
    missing_doc_count = sum(1 for d in all_possible_docs if not d)

    record = {
        "claim_id": claim_id,
        "product_id": product_id,
        "product_name": item_name,
        "product_category": category,
        "brand": brand,
        "model_number": model_number,
        "serial_number_entered": serial_entered,
        "serial_number_on_receipt": serial_receipt,
        "serial_number_on_warranty_card": serial_warranty_card,
        "purchase_date": purchase_date.strftime("%Y-%m-%d"),
        "purchase_price": purchase_price,
        "retailer": retailer,
        "warranty_start": warranty_start.strftime("%Y-%m-%d"),
        "warranty_end": warranty_end.strftime("%Y-%m-%d"),
        "warranty_provider": warranty_provider,
        "warranty_type": warranty_type,
        "fault_date": fault_date.strftime("%Y-%m-%d"),
        "claim_submission_date": claim_submission_date.strftime("%Y-%m-%d"),
        "fault_type": fault_type,
        "fault_description": fault_description,
        "damage_type": damage_type,
        "product_age_months": int(product_age_months),
        "remaining_warranty_days": int(remaining_warranty_days),
        "repair_history_count": int(repair_history_count),
        "previous_repair_authorized": bool(previous_repair_authorized),
        "receipt_uploaded": bool(receipt_uploaded),
        "warranty_card_uploaded": bool(warranty_card_uploaded),
        "product_image_uploaded": bool(product_image_uploaded),
        "fault_evidence_uploaded": bool(fault_evidence_uploaded),
        "repair_report_uploaded": bool(repair_report_uploaded),
        "missing_doc_count": int(missing_doc_count),
        "serial_mismatch_flag": bool(serial_mismatch_flag),
        "date_contradiction_flag": bool(date_contradiction_flag),
        "excluded_damage": bool(excluded_damage),
        "duplicate_claim_flag": bool(duplicate_claim_flag),
        "class_label": target_class,
    }
    return record


def generate_dataset(num_records=10000):
    """
    Generate 10,000 unique synthetic claim records:
    - Balanced ~3,334 Valid, ~3,333 Invalid, ~3,333 Manual Review
    - Guaranteed zero duplicates via SHA-256 hash set with collision rejection
    - Stratified 70/15/15 split (7,000 / 1,500 / 1,500)
    """
    print(f"=== ASSUREX DATASET GENERATOR: Scaling to {num_records} Unique Records ===")

    category_list = list(PRODUCT_CATALOG.keys())
    # Balanced classes
    target_classes = (
        ["Likely Valid"] * (num_records // 3 + (1 if num_records % 3 > 0 else 0)) +
        ["Likely Invalid"] * (num_records // 3 + (1 if num_records % 3 > 1 else 0)) +
        ["Manual Review Required"] * (num_records // 3)
    )
    random.shuffle(target_classes)

    records = []
    seen_hashes = set()
    collisions_avoided = 0

    print("Generating records with strict hash-based collision rejection...")
    for idx in range(1, num_records + 1):
        target_cls = target_classes[idx - 1]
        cat = random.choice(category_list)

        while True:
            rec = generate_single_claim(idx, cat, target_cls)
            h = make_feature_hash(rec)
            if h not in seen_hashes:
                seen_hashes.add(h)
                records.append(rec)
                break
            else:
                collisions_avoided += 1

        if idx % 2000 == 0 or idx == num_records:
            print(f"  Generated {idx}/{num_records} records (Collisions rejected: {collisions_avoided})...")

    df = pd.DataFrame(records)
    print(f"\nGeneration complete. Total records: {len(df)}, Unique signatures: {len(seen_hashes)}")
    print(f"Total collision attempts safely rejected: {collisions_avoided}")

    # Check duplicates across full feature set
    feature_cols = [
        "product_category", "brand", "model_number", "purchase_price", "retailer",
        "warranty_type", "fault_type", "damage_type", "product_age_months",
        "remaining_warranty_days", "repair_history_count", "previous_repair_authorized",
        "receipt_uploaded", "warranty_card_uploaded", "product_image_uploaded",
        "fault_evidence_uploaded", "serial_mismatch_flag", "date_contradiction_flag",
        "excluded_damage", "duplicate_claim_flag"
    ]
    dups = df.duplicated(subset=feature_cols).sum()
    print(f"Empirical feature duplicates in dataset: {dups}")
    assert dups == 0, f"Error: Found {dups} duplicate records!"

    # Distribution checks
    print("\n--- Class Distribution ---")
    print(df["class_label"].value_counts())
    print("\n--- Category Distribution ---")
    print(df["product_category"].value_counts())

    # Stratified 70/15/15 split
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=df[["class_label", "product_category"]]
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temp_df[["class_label", "product_category"]]
    )

    # Save to disk
    full_path = os.path.join(DATA_DIR, "claims_full.csv")
    train_path = os.path.join(DATA_DIR, "claims_train.csv")
    val_path = os.path.join(DATA_DIR, "claims_val.csv")
    test_path = os.path.join(DATA_DIR, "claims_test.csv")

    df.to_csv(full_path, index=False)
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\nSaved Full Dataset: {full_path} ({len(df)} rows)")
    print(f"Saved Train Split: {train_path} ({len(train_df)} rows, {len(train_df)/len(df)*100:.1f}%)")
    print(f"Saved Val Split: {val_path} ({len(val_df)} rows, {len(val_df)/len(df)*100:.1f}%)")
    print(f"Saved Test Split: {test_path} ({len(test_df)} rows, {len(test_df)/len(df)*100:.1f}%)")

    # Generate summary JSON
    summary = {
        "total_records": len(df),
        "seed": RANDOM_SEED,
        "splits": {
            "train": {"count": len(train_df), "percentage": 70.0},
            "validation": {"count": len(val_df), "percentage": 15.0},
            "test": {"count": len(test_df), "percentage": 15.0},
        },
        "class_counts": df["class_label"].value_counts().to_dict(),
        "category_counts": df["product_category"].value_counts().to_dict(),
        "anomaly_statistics": {
            "serial_mismatches": int(df["serial_mismatch_flag"].sum()),
            "date_contradictions": int(df["date_contradiction_flag"].sum()),
            "excluded_damages": int(df["excluded_damage"].sum()),
            "missing_receipts": int((~df["receipt_uploaded"]).sum()),
            "unauthorized_repairs": int((~df["previous_repair_authorized"]).sum()),
            "duplicate_claims": int(df["duplicate_claim_flag"].sum()),
        },
        "uniqueness_check": {
            "unique_signatures": len(seen_hashes),
            "collisions_rejected": collisions_avoided,
            "duplicate_rows": int(dups),
        },
        "generated_at": datetime.now().isoformat(),
    }

    summary_path = os.path.join(DATA_DIR, "dataset_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved Dataset Summary to: {summary_path}")
    return df, train_df, val_df, test_df


if __name__ == "__main__":
    generate_dataset(10000)
