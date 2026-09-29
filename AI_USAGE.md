# AI Tool Usage Declaration — AssureX Claim Engine

> As required by SRS §1.8, all AI tools used during development are declared below.
> Each entry documents: tool name, purpose, affected files, team modifications, and testing performed.

---

## 2026-09-26 — Google Antigravity (Gemini)
- **Purpose:** Scaffold project structure — FastAPI backend, React frontend, config files, warranty policies
- **Prompt/assistance type:** "Scaffold a FastAPI project with SQLModel/SQLite and React+Vite frontend with Bootstrap"
- **Files/modules affected:** All initial project files (main.py, App.jsx, requirements.txt, package.json, policies/*.json, config/*.json)
- **Modifications made by team:** Reviewed all generated files, verified structure matches SRS requirements, adjusted config values
- **Testing performed:** Backend `/health` endpoint returns 200, frontend dev server starts without errors
- **Verified by:** Team Lead

## 2026-09-26 — Google Antigravity (Gemini)
- **Purpose:** Synthetic claim dataset and Claim Summary Card image generation (Phase 1)
- **Prompt/assistance type:** "Generate synthetic warranty claim records with realistic anomalies and render Claim Summary Cards without predictions using Pillow" (initially 2,500 records; later regenerated to 10,000 in Session 7)
- **Files/modules affected:**
  - `backend/dataset_generator/generate_claims.py`
  - `backend/dataset_generator/generate_cards.py`
  - `backend/data/claims_*.csv`
  - `backend/data/dataset_summary.json`
  - `backend/data/card_image_mapping.csv`
  - `sample_claims/demo_cards/*`
- **Modifications made by team:** Verified stratified split distributions (70/15/15), tuned anomaly ratios (serial mismatches, date contradictions, excluded damages), audited card layout to guarantee strict absence of model predictions/confidence scores per SRS.
- **Testing performed:** Validated column consistency across train/val/test splits, checked Pillow rendering output, confirmed card PNG files generated successfully with zero errors.
- **Verified by:** Team Lead

## 2026-09-26 — Google Antigravity (Gemini)
- **Purpose:** High-Resolution Card Engine Overhaul & Tabular ML 8-Model Benchmark (Phase 2)
- **Prompt/assistance type:** "Overhaul card generator to 1200x1680 high-DPI with zero overlap; train and compare 8 ML classifiers with 5-fold CV, confusion matrices, and feature importance"
- **Files/modules affected:**
  - `backend/dataset_generator/generate_cards.py`
  - `sample_claims/demo_cards/*`
  - `backend/src/ml/preprocessing.py`
  - `backend/src/ml/train_models.py`
  - `backend/src/ml/predictor.py`
  - `backend/model/best_model.joblib`
  - `reports/confusion_matrices/*`
  - `reports/feature_importance.png`
  - `reports/model_comparison.json`
  - `reports/model_comparison.md`
  - `reports/model_comparison_chart.png`
- **Modifications made by team:** Redesigned card layout grid with full-width product row, dynamic badge widths, and high-DPI TrueType fonts. Tuned 8 classifier hyper-parameters, audited 5-fold cross-validation results, inspected confusion matrix heatmaps, verified live inference on edge-case claims.
- **Testing performed:** Verified high-res card renders (zero overlaps verified via image inspection). Executed 5-fold CV across 8 models, verified `TabularPredictor` live inference on Valid, Invalid, and Manual Review test cases. (Note: metrics were later updated when the dataset was regenerated to 10,000 records in Session 7.)
- **Verified by:** Team Lead

## 2026-09-26 — Google Antigravity (Gemini)
- **Purpose:** Teachable Machine Image Classifier Training & Dual-Model Comparison Benchmark (Phase 3)
- **Prompt/assistance type:** "Train MobileNetV2 image model matching Google Teachable Machine specification, export keras_model.h5 + labels.txt, build inference engine, and run 35-claim benchmark for Deliverable 6"
- **Files/modules affected:**
  - `backend/src/ml/train_teachable_machine.py`
  - `backend/src/ml/teachable_machine.py`
  - `backend/src/ml/predictor.py`
  - `backend/src/ml/compare_models.py`
  - `backend/model/teachable_machine/keras_model.h5`
  - `backend/model/teachable_machine/labels.txt`
  - `backend/model/teachable_machine/model_metadata.json`
  - `reports/model_comparison_30_claims.md`
  - `reports/model_comparison_30_claims.json`
- **Modifications made by team:** Optimized pipeline for CPU execution using memory-cached transfer learning; built vectorized batch prediction interfaces for both tabular and vision inference engines; implemented 5-category match taxonomy for Deliverable 6.
- **Testing performed:** Validated exported Keras model against Teachable Machine format (224x224 RGB, [-1, 1] normalization); verified 95.47% validation accuracy on 375 validation cards; executed live inference across demo cards and 35 unseen test claims; verified JSON and Markdown benchmark reports.

## 2026-09-26 — Google Antigravity (Gemini)
- **Purpose:** Full Application Development & End-to-End Integration (Phase 4)
- **Prompt/assistance type:** "Develop complete backend services (Rule Engine, OCR, Card Service, Adjudication Engine, RBAC Auth, REST Routers), SQLite seeding, and modern React dashboard with live claim feed, intake wizard, and full claim dossier"
- **Files/modules affected:**
  - `backend/src/models/entities.py`
  - `backend/src/auth/service.py`
  - `backend/src/schemas/*.py`
  - `backend/src/services/*.py` (rule_engine, ocr_service, card_service, adjudication_engine)
  - `backend/src/routers/*.py` (auth, claims, products, warranties, policies, admin)
  - `backend/tests/test_e2e.py`
  - `frontend/src/index.css`
  - `frontend/src/api/*.js`
  - `frontend/src/pages/*.jsx` (Dashboard, SubmitClaim, ClaimsList, ClaimDetail, ReviewQueue, Admin, Products, Login, Register)
  - `frontend/src/components/Navbar.jsx`
  - `frontend/src/App.jsx`
- **Modifications made by team:** Built multi-model arbitration logic uniting deterministic rule engine + XGBoost tabular classifier + MobileNetV2 vision AI; engineered real-time 1200x1680 card generation without predictions; created enterprise dark insurtech UI design system; wired human adjuster override workflow with immutable audit logging.
- **Testing performed:** Executed automated integration test suite (`backend/tests/test_e2e.py`) testing all REST endpoints, auth flows, live claim submission, and reviewer overrides (100% pass); verified frontend production build (`npm run build`).
- **Verified by:** Team Lead

## 2026-09-28 — Google Antigravity (Gemini)
- **Purpose:** Issue 6, 7, and 8 Implementations
- **Prompt/assistance type:** "Implement 30-day prior notifications, CSV/HTML exports, and bundle demo claims"
- **Files/modules affected:**
  - `backend/src/routers/notifications.py`
  - `backend/src/routers/claims.py`
  - `frontend/src/components/NotificationDropdown.jsx`
  - `frontend/src/pages/ClaimsList.jsx`
  - `frontend/src/pages/ClaimDetail.jsx`
  - `sample_claims/`
- **Modifications made by team:** Connected frontend export buttons to API routes generating dynamic CSV and HTML templates. Extracted test claims for demo bundle.
- **Testing performed:** Tested CSV and HTML download behavior in browser UI. Verified file contents.
- **Verified by:** Team Lead

## 2026-09-27 — Google Antigravity (Gemini)
- **Purpose:** 10K dataset regeneration, 8 warranty policies, model re-training, and admin analytics dashboard
- **Prompt/assistance type:** "Regenerate dataset to 10,000 records across 6 categories, add 5 new warranty policy files, re-train all 8 classifiers, and build admin analytics tabs"
- **Files/modules affected:**
  - `backend/dataset_generator/generate_claims.py`, `generate_cards.py`
  - `backend/data/claims_*.csv`, `dataset_summary.json`, `card_image_mapping.csv`
  - `backend/policies/*.json` (5 new policy files added)
  - `backend/src/ml/train_models.py`
  - `backend/model/*.joblib` (all 8 model artifacts re-serialized)
  - `reports/model_comparison.md`, `model_comparison.json`, `model_comparison_chart.png`
  - `frontend/src/components/admin/ClaimsAnalyticsTab.jsx`, `ModelPerformanceTab.jsx`, `OperationsTab.jsx`
  - `frontend/src/pages/Admin.jsx`
- **Modifications made by team:** Verified 10,000-record dataset shape and stratification. Audited 5 new policy files against SRS requirements. Reviewed re-trained model metrics. Tested admin dashboard tab rendering.
- **Testing performed:** Verified CSV line counts (10,001 including header), confirmed 6-category class balance, re-ran 5-fold CV, verified admin tabs render correctly.
- **Verified by:** Team Lead

## 2026-09-27 — Google Antigravity (Gemini)
- **Purpose:** Employee role, profile management, product registration, RBAC enforcement, claim form redesign, media uploads, product validation, warranty card upload, and cross-document verification
- **Prompt/assistance type:** Multiple prompts for implementing RBAC, product registration, media uploads, ownership validation, warranty card handling, and OCR cross-verification
- **Files/modules affected:**
  - `backend/src/routers/auth.py` (profile GET/PUT)
  - `backend/src/routers/products.py` (product registration)
  - `backend/src/routers/claims.py` (media upload, warranty card upload, cross-verify, product validation)
  - `backend/src/services/ocr_service.py` (cross_verify_all)
  - `frontend/src/components/ProfileModal.jsx`
  - `frontend/src/pages/Products.jsx`, `SubmitClaim.jsx`, `ClaimDetail.jsx`
  - `frontend/src/routes/ProtectedRoute.jsx`
  - `frontend/src/context/AuthContext.jsx`
  - `backend/tests/test_rbac.py`, `test_task1_media_upload.py`, `test_task1_warranty_card_upload.py`, `test_task2_cross_verification.py`, `test_task3_product_warranty_restriction.py`
- **Modifications made by team:** Reviewed RBAC guard placement on all routes, verified product ownership scoping, tested OCR cross-verification logic across receipt/warranty card/barcode, confirmed SHA-256 hash persistence.
- **Testing performed:** Ran all new test files (`test_rbac.py`, `test_task1_media_upload.py`, `test_task1_warranty_card_upload.py`, `test_task2_cross_verification.py`, `test_task3_product_warranty_restriction.py`). All passing.
- **Verified by:** Team Lead

## 2026-09-28 — Google Antigravity (Gemini)
- **Purpose:** Semantic duplicate claim detection, TM Keras 3 re-integration, and bug fixes (Issues 1–5)
- **Prompt/assistance type:** Multiple prompts for duplicate detection, Keras 3 compatibility, and 5 bug fix issues
- **Files/modules affected:**
  - `backend/src/services/duplicate_detector.py`
  - `backend/src/ml/teachable_machine.py` (Keras 3 architecture reconstruction, alpha=1.0 restore)
  - `backend/src/services/adjudication_engine.py` (borderline claim routing)
  - `backend/src/services/rule_engine.py` (date contradiction fix)
  - `backend/dataset_generator/generate_cards.py` (date flag fix)
  - `backend/src/routers/claims.py` (duplicate check endpoint)
  - `frontend/src/pages/SubmitClaim.jsx` (duplicate check UI)
  - `backend/tests/test_task3_semantic_duplicate.py`, `test_batch3_ocr.py`
- **Modifications made by team:** Reviewed 4-factor duplicate detection logic, verified Keras 3 weight transfer from H5, confirmed borderline routing thresholds match SRS Step 12, tested all 5 bug fix patches.
- **Testing performed:** Ran `test_task3_semantic_duplicate.py`, `test_batch3_ocr.py`, `test_e2e.py`. All passing. Verified TM model loads and predicts correctly on demo cards.
- **Verified by:** Team Lead

---

*Entries will be added for every AI-assisted development session.*
