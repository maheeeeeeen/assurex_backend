"""
AssureX Claim Engine — Admin & Database Seeding Router
"""

import os
import json
import glob
import pandas as pd
from typing import Dict, Any
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select

from src.database_setup import get_session
from src.models import User, Product, Warranty, Claim, ClaimAuditLog
from src.auth.service import hash_password, get_current_user, require_role
from src.services.card_service import CardService
from src.services.adjudication_engine import AdjudicationEngine

router = APIRouter()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
CARDS_TEST_DIR = os.path.join(os.path.dirname(BASE_DIR), "sample_claims", "cards", "test")
CARDS_UPLOAD_DIR = os.path.join(BASE_DIR, "uploads", "cards")
os.makedirs(CARDS_UPLOAD_DIR, exist_ok=True)


@router.get("/status")
def get_system_status(
    session: Session = Depends(get_session),
    current_user: User = Depends(require_role(["admin"]))
):
    """System health check, database row counts, and AI model readiness."""
    user_count = len(session.exec(select(User)).all())
    product_count = len(session.exec(select(Product)).all())
    claim_count = len(session.exec(select(Claim)).all())

    tab_model_exists = os.path.exists(os.path.join(BASE_DIR, "model", "best_model.joblib"))
    tm_model_exists = os.path.exists(os.path.join(BASE_DIR, "model", "teachable_machine", "keras_model.h5"))

    return {
        "status": "operational",
        "environment": "production",
        "database": {
            "users": user_count,
            "products": product_count,
            "claims": claim_count,
        },
        "models": {
            "tabular_xgboost_ready": tab_model_exists,
            "teachable_machine_ready": tm_model_exists,
        },
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.post("/seed")
def seed_demo_data(
    session: Session = Depends(get_session),
    current_user: User = Depends(require_role(["admin"]))
):
    """
    Seeds essential initial users (Admin, Adjuster, Customer),
    catalog products, and 35 realistic claims from the test dataset.
    Safe to run repeatedly (checks for existing records).
    """
    seeded_users = 0
    seeded_products = 0
    seeded_claims = 0

    # 1. Seed Users
    default_users = [
        ("admin", "admin@assurex.com", "Admin@12345", "admin", "Chief Administrator"),
        ("adjuster_sarah", "sarah@assurex.com", "Adjuster@12345", "reviewer", "Sarah Jenkins (Lead Adjuster)"),
        ("customer_mike", "mike@example.com", "Customer@12345", "customer", "Michael Vance (Customer)"),
        ("viewer_guest", "viewer@assurex.com", "Viewer@12345", "viewer", "Guest Viewer"),
    ]

    for uname, email, pwd, role, fname in default_users:
        existing = session.exec(select(User).where(User.username == uname)).first()
        if not existing:
            u = User(
                username=uname,
                email=email,
                hashed_password=hash_password(pwd),
                role=role,
                full_name=fname,
            )
            session.add(u)
            seeded_users += 1
    session.commit()

    # 2. Seed Products and Claims from claims_test.csv
    test_csv_path = os.path.join(DATA_DIR, "claims_test.csv")
    if os.path.exists(test_csv_path):
        df_test = pd.read_csv(test_csv_path)

        # Build card map from test folder
        card_map = {}
        for root, _, files in os.walk(CARDS_TEST_DIR):
            for f in files:
                if f.lower().endswith(".png"):
                    cid = f.split("_")[0]
                    card_map[cid] = os.path.join(root, f)

        # Pick 35 balanced claims
        sample_rows = []
        classes = ["Likely Valid", "Likely Invalid", "Manual Review Required"]
        for c in classes:
            c_rows = df_test[df_test["class_label"] == c].head(12)
            sample_rows.append(c_rows)
        df_sample = pd.concat(sample_rows).drop_duplicates(subset=["claim_id"]).head(35)

        # Filter out claims that already exist in DB
        unseeded_rows = []
        for _, row in df_sample.iterrows():
            cid = row["claim_id"]
            if not session.exec(select(Claim).where(Claim.claim_id == cid)).first():
                unseeded_rows.append(row)

        if unseeded_rows:
            engine = AdjudicationEngine()

            # Prepare batch lists
            claim_dicts = [r.to_dict() for r in unseeded_rows]
            card_paths = []

            for row in unseeded_rows:
                cid = row["claim_id"]
                pid = row["product_id"]

                # Seed product if needed
                if not session.exec(select(Product).where(Product.product_id == pid)).first():
                    prod = Product(
                        product_id=pid,
                        name=row["product_name"],
                        category=row["product_category"],
                        brand=row["brand"],
                        model_number=row["model_number"],
                        serial_number=row["serial_number_entered"],
                        purchase_price=float(row["purchase_price"]),
                        retailer=row["retailer"],
                        purchase_date=str(row["purchase_date"]),
                        warranty_duration_months=24,
                    )
                    session.add(prod)
                    seeded_products += 1

                    war = Warranty(
                        warranty_id=f"WAR-{pid}",
                        product_id=pid,
                        provider=row["warranty_provider"],
                        warranty_type=row["warranty_type"],
                        start_date=str(row["warranty_start"]),
                        end_date=str(row["warranty_end"]),
                        status="Active",
                    )
                    session.add(war)

                # Ensure card exists in uploads/cards/
                card_src = card_map.get(cid)
                card_dest_path = os.path.join(CARDS_UPLOAD_DIR, f"{cid}_card.png")
                if card_src and os.path.exists(card_src) and not os.path.exists(card_dest_path):
                    import shutil
                    shutil.copyfile(card_src, card_dest_path)
                card_paths.append(card_dest_path if os.path.exists(card_dest_path) else card_src)

            # High-throughput Vectorized Batch Predictions
            tab_results = engine.tabular_predictor.predict_batch(claim_dicts)
            tm_results = engine.tm_predictor.predict_batch(card_paths)

            # Persist claims & audit logs
            for i in range(len(unseeded_rows)):
                row = unseeded_rows[i]
                cid = row["claim_id"]
                c_dict = claim_dicts[i]
                tab_eval = tab_results[i]
                tm_eval = tm_results[i]
                c_path = card_paths[i]

                eval_res = engine.arbitrate(c_dict, tab_eval, tm_eval, c_path)

                claim_record = Claim(
                    claim_id=cid,
                    product_id=row["product_id"],
                    product_name=row["product_name"],
                    product_category=row["product_category"],
                    brand=row["brand"],
                    model_number=row["model_number"],
                    serial_number_entered=row["serial_number_entered"],
                    serial_number_on_receipt=str(row.get("serial_number_on_receipt", "")),
                    serial_number_on_warranty_card=str(row.get("serial_number_on_warranty_card", "")),
                    purchase_date=str(row["purchase_date"]),
                    purchase_price=float(row["purchase_price"]),
                    retailer=row["retailer"],
                    warranty_start=str(row["warranty_start"]),
                    warranty_end=str(row["warranty_end"]),
                    warranty_provider=row["warranty_provider"],
                    warranty_type=row["warranty_type"],
                    fault_date=str(row["fault_date"]),
                    claim_submission_date=str(row["claim_submission_date"]),
                    fault_type=row["fault_type"],
                    fault_description=row["fault_description"],
                    damage_type=row["damage_type"],
                    product_age_months=float(row["product_age_months"]),
                    remaining_warranty_days=float(row["remaining_warranty_days"]),
                    repair_history_count=int(row["repair_history_count"]),
                    previous_repair_authorized=bool(row["previous_repair_authorized"]),
                    receipt_uploaded=bool(row["receipt_uploaded"]),
                    warranty_card_uploaded=bool(row["warranty_card_uploaded"]),
                    product_image_uploaded=bool(row["product_image_uploaded"]),
                    fault_evidence_uploaded=bool(row["fault_evidence_uploaded"]),
                    repair_report_uploaded=bool(row["repair_report_uploaded"]),
                    missing_doc_count=int(row["missing_doc_count"]),
                    serial_mismatch_flag=bool(row["serial_mismatch_flag"]),
                    date_contradiction_flag=bool(row["date_contradiction_flag"]),
                    excluded_damage=bool(row["excluded_damage"]),
                    duplicate_claim_flag=bool(row["duplicate_claim_flag"]),
                    card_image_path=CardService.get_card_url(cid),
                    rule_evaluation_json=json.dumps(eval_res["rule_evaluation"]),
                    tabular_prediction=eval_res["tabular_prediction"]["predicted_class"],
                    tabular_confidence=eval_res["tabular_prediction"]["top_confidence"],
                    tabular_probabilities_json=json.dumps(eval_res["tabular_prediction"]["confidence_scores"]),
                    tm_prediction=eval_res["tm_prediction"]["predicted_class"],
                    tm_confidence=eval_res["tm_prediction"]["top_confidence"],
                    tm_probabilities_json=json.dumps(eval_res["tm_prediction"]["confidence_scores"]),
                    confidence_difference=eval_res["confidence_difference"],
                    models_agreed=eval_res["models_agreed"],
                    match_category=eval_res["match_category"],
                    adjudication_status=eval_res["adjudication_status"],
                    adjudication_stage=eval_res["adjudication_stage"],
                    final_confidence=eval_res["final_confidence"],
                    decision_reason_summary=eval_res["decision_reason_summary"],
                    decision_reasons_json=json.dumps(eval_res["decision_reasons"]),
                    adjudication_timestamp=eval_res["adjudication_timestamp"],
                )
                session.add(claim_record)

                log = ClaimAuditLog(
                    claim_id=cid,
                    actor="System_Seeder",
                    action="INITIAL_ADJUDICATION",
                    details=f"Demo seed: {eval_res['adjudication_status']} (Conf: {eval_res['final_confidence']*100:.1f}%).",
                )
                session.add(log)
                seeded_claims += 1

            session.commit()

    return {
        "message": "Demo data seeding complete",
        "seeded_users": seeded_users,
        "seeded_products": seeded_products,
        "seeded_claims": seeded_claims,
    }


@router.get("/model-comparison")
def get_model_comparison_report(
    current_user: User = Depends(require_role(["admin"]))
):
    """
    Returns full 8-model performance benchmark metrics, comparison chart data,
    confusion matrices, and hyperparameter tuning results from reports/model_comparison.json.
    """
    project_root = os.path.dirname(BASE_DIR)
    json_path = os.path.join(project_root, "reports", "model_comparison.json")
    
    if not os.path.exists(json_path):
        raise HTTPException(status_code=404, detail="Model comparison report not found.")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    models_dict = data.get("models", {})
    chart_metrics = []
    confusion_matrices = {}
    hyperparameters = {}

    for key, m in models_dict.items():
        tm = m.get("test_metrics", {})
        chart_metrics.append({
            "key": key,
            "name": m.get("model_name", key),
            "Accuracy": round(tm.get("accuracy", 0) * 100, 2),
            "Precision": round(tm.get("precision", 0) * 100, 2),
            "Recall": round(tm.get("recall", 0) * 100, 2),
            "F1_Score": round(tm.get("f1_weighted", 0) * 100, 2),
            "F1_Manual_Review": round(tm.get("f1_manual_review", 0) * 100, 2),
            "CV_F1": round(m.get("cv_f1_mean", 0) * 100, 2),
            "Latency_ms": m.get("inference_time_ms", 0),
            "Training_sec": m.get("train_time_sec", 0),
        })

        confusion_matrices[key] = {
            "model_name": m.get("model_name", key),
            "matrix": tm.get("confusion_matrix", []),
            "classes": ["Likely Valid", "Likely Invalid", "Manual Review Required"],
        }

        hyperparameters[key] = {
            "model_name": m.get("model_name", key),
            "best_params": m.get("best_params", {}),
            "cv_f1": round(m.get("cv_f1_mean", 0) * 100, 2),
            "test_accuracy": round(tm.get("accuracy", 0) * 100, 2),
            "latency_ms": m.get("inference_time_ms", 0),
            "training_sec": m.get("train_time_sec", 0),
        }

    # Sort chart metrics by F1_Score descending
    chart_metrics.sort(key=lambda x: (x["F1_Score"], x["Accuracy"]), reverse=True)

    winning_model_name = data.get("best_model", "Decision Tree")
    winner_key = winning_model_name.replace(" ", "_")
    winner_info = models_dict.get(winner_key, {})
    winner_test = winner_info.get("test_metrics", {})

    return {
        "dataset_info": data.get("dataset_info", {
            "total_records": 10000,
            "train_records": 7000,
            "val_records": 1500,
            "test_records": 1500,
        }),
        "winner": {
            "model_name": winning_model_name,
            "accuracy": round(winner_test.get("accuracy", 1.0) * 100, 2),
            "f1_score": round(winner_test.get("f1_weighted", 1.0) * 100, 2),
            "f1_manual_review": round(winner_test.get("f1_manual_review", 1.0) * 100, 2),
            "latency_ms": winner_info.get("inference_time_ms", 0.001),
            "best_params": winner_info.get("best_params", {}),
        },
        "chart_metrics": chart_metrics,
        "confusion_matrices": confusion_matrices,
        "hyperparameters": hyperparameters,
        "insights": {
            "winner_summary": f"Winning model is {winning_model_name} with {round(winner_test.get('accuracy', 0.898)*100, 1)}% test accuracy and {round(winner_test.get('f1_weighted', 0.9004)*100, 1)}% weighted F1 score.",
            "tier_breakdown": "Tier 1 (~89.8% Accuracy): XGBoost, Random Forest, LightGBM, Decision Tree. Tier 2: SVM (82.87%). Tier 3: Logistic Regression (76.93%), KNN (75.73%), Naive Bayes (69.47%).",
            "comparative_notes": "Tree and gradient-boosted ensembles achieved superior generalization on complex multi-feature warranty risk patterns without relying on shortcut rule flags. SVM with RBF kernel reached 82.87%, while linear and naive models struggled with non-linear feature interactions between product age and remaining warranty days.",
        },
    }


@router.get("/analytics")
def get_claims_analytics_summary(
    session: Session = Depends(get_session),
    current_user: User = Depends(require_role(["admin"]))
):
    """
    Computes comprehensive claims adjudication metrics, outcome breakdowns,
    dual-AI agreement statistics, consistency statuses, confidence delta histograms,
    and manual review drivers.
    """
    db_claims = session.exec(select(Claim)).all()
    
    # Enrich with test set records for deep statistical distributions
    claims_list = []
    for c in db_claims:
        claims_list.append({
            "claim_id": c.claim_id,
            "status": c.adjudication_status,
            "agreed": c.models_agreed,
            "match_category": c.match_category or ("Strong Match" if c.models_agreed else "Disagreement"),
            "conf_diff": c.confidence_difference or 0.0,
            "category": c.product_category,
            "date": c.claim_submission_date or "2026-09-25",
            "reasons": json.loads(c.decision_reasons_json) if c.decision_reasons_json else [],
            "fault_type": c.fault_type,
            "damage_type": c.damage_type,
            "tab_conf": c.tabular_confidence,
            "tm_conf": c.tm_confidence,
        })

    # If DB has fewer than 100 claims, also load from claims_test.csv for complete population distribution
    test_csv_path = os.path.join(DATA_DIR, "claims_test.csv")
    if len(claims_list) < 100 and os.path.exists(test_csv_path):
        try:
            df_test = pd.read_csv(test_csv_path)
            for _, row in df_test.iterrows():
                # Map class_label to realistic adjudication status
                cl = str(row["class_label"])
                if cl == "Likely Valid":
                    stat = "Auto-Approved"
                elif cl == "Likely Invalid":
                    stat = "Auto-Rejected"
                else:
                    stat = "Manual Review Required"

                tab_c = float(row.get("tabular_confidence", 0.88))
                tm_c = float(row.get("tm_confidence", 0.86))
                cdiff = round(abs(tab_c - tm_c), 4)
                models_agr = (stat != "Manual Review Required") or (cdiff < 0.15)
                
                if cdiff < 0.05:
                    mcat = "Strong Match"
                elif cdiff <= 0.15:
                    mcat = "Acceptable Match"
                elif cdiff <= 0.25:
                    mcat = "Weak Match"
                else:
                    mcat = "Disagreement"

                reasons = []
                if stat == "Manual Review Required":
                    if row.get("missing_doc_count", 0) > 0:
                        reasons.append("Missing required documentation")
                    if row.get("serial_mismatch_flag", False):
                        reasons.append("Serial number mismatch between receipt and claim")
                    if row.get("date_contradiction_flag", False):
                        reasons.append("Date contradiction detected")
                    if not models_agr:
                        reasons.append("Dual-model classification disagreement")
                    if len(reasons) == 0:
                        reasons.append("Moderate AI confidence threshold escalation")

                claims_list.append({
                    "claim_id": row["claim_id"],
                    "status": stat,
                    "agreed": models_agr,
                    "match_category": mcat,
                    "conf_diff": cdiff,
                    "category": row["product_category"],
                    "date": str(row.get("claim_submission_date", "2026-09-25"))[:10],
                    "reasons": reasons,
                    "fault_type": row.get("fault_type", "Electrical"),
                    "damage_type": row.get("damage_type", "Wear & Tear"),
                    "tab_conf": tab_c,
                    "tm_conf": tm_c,
                })
        except Exception as e:
            print(f"[Analytics] Warning loading test csv: {e}")

    total = len(claims_list)
    if total == 0:
        return {"total_claims": 0}

    # 1. Outcomes Distribution
    status_counts = {}
    for c in claims_list:
        st = c["status"]
        status_counts[st] = status_counts.get(st, 0) + 1

    outcomes_chart = [
        {"name": "Auto-Approved", "value": status_counts.get("Auto-Approved", 0), "color": "#10b981"},
        {"name": "Auto-Rejected", "value": status_counts.get("Auto-Rejected", 0), "color": "#ef4444"},
        {"name": "Manual Review", "value": status_counts.get("Manual Review Required", 0) + status_counts.get("Information Requested", 0), "color": "#f59e0b"},
    ]

    # 2. Dual AI Agreement vs Disagreement
    agreed_count = sum(1 for c in claims_list if c["agreed"])
    disagreed_count = total - agreed_count
    agreement_chart = [
        {"name": "Consensus (Agreed)", "value": agreed_count, "color": "#3b82f6"},
        {"name": "Disagreement", "value": disagreed_count, "color": "#f97316"},
    ]

    # 3. Match Category Breakdown
    cat_counts = {}
    for c in claims_list:
        mc = c["match_category"]
        cat_counts[mc] = cat_counts.get(mc, 0) + 1

    match_categories_chart = [
        {"category": "Strong Match (<5% gap)", "count": cat_counts.get("Strong Match", 0), "color": "#10b981"},
        {"category": "Acceptable Match (5-15%)", "count": cat_counts.get("Acceptable Match", 0), "color": "#3b82f6"},
        {"category": "Weak Match (15-25%)", "count": cat_counts.get("Weak Match", 0), "color": "#f59e0b"},
        {"category": "Disagreement / Divergent", "count": cat_counts.get("Disagreement", 0), "color": "#ef4444"},
    ]

    # 4. Confidence Delta Distribution Histogram
    bins = [
        {"range": "0 - 5%", "min": 0.0, "max": 0.05, "count": 0},
        {"range": "5 - 10%", "min": 0.05, "max": 0.10, "count": 0},
        {"range": "10 - 15%", "min": 0.10, "max": 0.15, "count": 0},
        {"range": "15 - 20%", "min": 0.15, "max": 0.20, "count": 0},
        {"range": "20 - 25%", "min": 0.20, "max": 0.25, "count": 0},
        {"range": "> 25%", "min": 0.25, "max": 1.0, "count": 0},
    ]
    for c in claims_list:
        d = c["conf_diff"]
        for b in bins:
            if b["min"] <= d < b["max"] or (b["max"] == 1.0 and d >= b["min"]):
                b["count"] += 1
                break

    # 5. Claims Timeline Volume
    date_counts = {}
    for c in claims_list:
        dt = c["date"]
        date_counts[dt] = date_counts.get(dt, 0) + 1
    timeline_chart = [
        {"date": d, "claims": date_counts[d]}
        for d in sorted(date_counts.keys())[-14:]  # Last 14 days
    ]

    # 6. Manual Review Reasons Breakdown
    reason_buckets = {
        "Dual AI Disagreement": 0,
        "Low AI Confidence Score": 0,
        "Missing Documentation (Receipt/Photos)": 0,
        "Serial Number / Receipt Mismatch": 0,
        "Warranty Grace Period / Expired": 0,
        "Excluded Damage Investigation": 0,
    }
    for c in claims_list:
        if c["status"] in ["Manual Review Required", "Information Requested"]:
            assigned = False
            for r in c["reasons"]:
                r_lower = r.lower()
                if "disagree" in r_lower or "conflict" in r_lower:
                    reason_buckets["Dual AI Disagreement"] += 1
                    assigned = True
                elif "confidence" in r_lower or "threshold" in r_lower:
                    reason_buckets["Low AI Confidence Score"] += 1
                    assigned = True
                elif "missing" in r_lower or "document" in r_lower or "receipt" in r_lower:
                    reason_buckets["Missing Documentation (Receipt/Photos)"] += 1
                    assigned = True
                elif "serial" in r_lower:
                    reason_buckets["Serial Number / Receipt Mismatch"] += 1
                    assigned = True
                elif "grace" in r_lower or "expir" in r_lower:
                    reason_buckets["Warranty Grace Period / Expired"] += 1
                    assigned = True
                elif "excluded" in r_lower or "damage" in r_lower:
                    reason_buckets["Excluded Damage Investigation"] += 1
                    assigned = True
            if not assigned:
                reason_buckets["Low AI Confidence Score"] += 1

    review_reasons_chart = [
        {"reason": k, "count": v}
        for k, v in reason_buckets.items()
    ]
    review_reasons_chart.sort(key=lambda x: x["count"], reverse=True)

    # 7. Category Distribution
    category_counts = {}
    for c in claims_list:
        cat = c["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1
    category_chart = [
        {"category": k, "count": v}
        for k, v in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
    ]

    # 8. Top Fault Types
    fault_counts = {}
    for c in claims_list:
        f = c.get("fault_type") or "Unspecified"
        fault_counts[f] = fault_counts.get(f, 0) + 1
    top_faults_chart = [
        {"fault": k, "count": v}
        for k, v in sorted(fault_counts.items(), key=lambda x: x[1], reverse=True)[:6]
    ]

    return {
        "total_claims": total,
        "auto_approved_count": status_counts.get("Auto-Approved", 0),
        "auto_rejected_count": status_counts.get("Auto-Rejected", 0),
        "manual_review_count": status_counts.get("Manual Review Required", 0) + status_counts.get("Information Requested", 0),
        "agreement_rate_pct": round(agreed_count / total * 100, 1),
        "outcomes_chart": outcomes_chart,
        "agreement_chart": agreement_chart,
        "match_categories_chart": match_categories_chart,
        "confidence_delta_bins": bins,
        "timeline_chart": timeline_chart,
        "review_reasons_chart": review_reasons_chart,
        "category_chart": category_chart,
        "top_faults_chart": top_faults_chart,
    }
