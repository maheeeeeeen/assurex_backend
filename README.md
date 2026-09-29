# AssureX Claim Engine

> AI-Powered Warranty Claim Validation Application

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)


## Project Blog

Read the complete blog about AssureX Claim Engine:

[Read the AssureX Claim Engine Blog](https://medium.com/@asrafaheem124/building-the-assurex-claim-engine-a-dual-model-ai-architecture-for-warranty-validation-390c54faf0a3?sharedUserId=asrafaheem124)
## Overview

The AssureX Claim Engine is a full-stack web application that automates warranty claim validation using dual AI models and configurable business rules. It processes claim information through a **Python tabular classifier (XGBoost)** and a **Google Teachable Machine image classifier (MobileNetV2)**, compares their predictions, applies deterministic warranty rule checks, and produces a final decision: **Auto-Approved**, **Auto-Rejected**, or **Manual Review Required**.

## Architecture

- **Backend:** FastAPI + Uvicorn (JSON API)
- **Frontend:** React (Vite) + Bootstrap 5 + React Router
- **Database:** SQLite via SQLModel ORM
- **Auth:** JWT (python-jose + passlib bcrypt), role-based access (customer, employee, reviewer, admin, viewer)
- **ML (tabular):** scikit-learn, XGBoost, LightGBM — 8 models trained and compared
- **ML (image):** Google Teachable Machine (MobileNetV2, Keras H5 export)
- **OCR:** pytesseract (Tesseract) with heuristic fallback when Tesseract is unavailable
- **Card rendering:** Pillow (1200×1680 High-DPI PNG)

## Project Structure

```
assurex-claim-engine/
├── README.md
├── AI_USAGE.md
├── LICENSE
├── devlog.md
├── project_analysis.md
├── backend/
│   ├── requirements.txt
│   ├── src/
│   │   ├── main.py              # FastAPI app, CORS, lifespan startup
│   │   ├── database_setup.py    # SQLModel engine, migrations, session dependency
│   │   ├── routers/             # API route handlers
│   │   ├── models/              # SQLModel table definitions
│   │   ├── schemas/             # Pydantic request/response models
│   │   ├── services/            # Business logic: rule engine, OCR, adjudication, cards, duplicates
│   │   ├── auth/                # JWT auth, password hashing, role guards
│   │   └── ml/                  # Training scripts, inference services, preprocessing
│   ├── data/                    # Generated datasets (CSV) and metadata
│   ├── model/                   # Serialized model artifacts (.joblib)
│   │   └── teachable_machine/   # Exported TM model (keras_model.h5, labels.txt)
│   ├── policies/                # Warranty policy JSON files (8 categories)
│   ├── dataset_generator/       # Scripts to generate claims and render cards
│   ├── database/                # SQLite database file (auto-created on first run)
│   ├── config/                  # thresholds.json, settings.json
│   ├── tests/                   # pytest test suite
│   └── uploads/                 # Uploaded documents and generated card images
├── frontend/
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       ├── api/                 # Axios instance and endpoint wrappers
│       ├── context/             # AuthContext (JWT storage, user role)
│       ├── pages/               # One file per route
│       ├── components/          # Reusable UI components
│       └── routes/              # Protected/role-based route wrappers
├── sample_claims/               # 11 demo claims for testing (CSV + card images)
├── reports/                     # Model comparison reports, charts, confusion matrices
└── screenshots/                 # Application screenshots
```

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- (Optional) [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) — OCR falls back to a heuristic parser if Tesseract is not installed

### Backend Setup

```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
uvicorn src.main:app --reload --port 8000
```

On first startup the database is created automatically and seeded with demo accounts, products, warranties, and benchmark claims.

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

The frontend runs at `http://localhost:5173` and the API at `http://localhost:8000`.

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `CORS_ORIGINS` | `http://localhost:5173` | Comma-separated allowed origins for CORS |

### API Documentation

Once the backend is running, visit `http://localhost:8000/docs` for interactive Swagger documentation.

## Demo Credentials

These accounts are created automatically by the seed script on first startup:

| Role | Username | Email | Password |
|---|---|---|---|
| Admin | `admin` | admin@assurex.com | `Admin@12345` |
| Reviewer | `adjuster_sarah` | sarah@assurex.com | `Adjuster@12345` |
| Customer | `customer_mike` | mike@example.com | `Customer@12345` |
| Viewer | `viewer_guest` | viewer@assurex.com | `Viewer@12345` |

## Dataset

- **10,000** synthetic warranty claim records across 6 product categories
- **3 classes:** Likely Valid (3,334) · Likely Invalid (3,333) · Manual Review Required (3,333)
- **Stratified split:** 70% train (7,000) / 15% validation (1,500) / 15% test (1,500)
- **Categories:** Electronics · Appliances · Automotive · Smartphones & Mobile · Computers & Laptops · Wearables & Audio
- 1200×1680 High-DPI Claim Summary Card images generated per record (multiple visual variants for training)

## Models

### Python Tabular Classifiers (8 compared)

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. XGBoost
5. LightGBM
6. SVM
7. KNN
8. Naive Bayes

Best model selected by weighted F1-score across 5-fold stratified cross-validation. See [`reports/model_comparison.md`](reports/model_comparison.md).

### Google Teachable Machine

MobileNetV2-based image classifier trained on 1200×1680 Claim Summary Cards. Exported as Keras H5 model. Validation accuracy: 95.47%.

## Warranty Policies

Eight configurable JSON policy files in `backend/policies/`:

| Policy File | Product Category |
|---|---|
| `electronics_warranty.json` | Electronics |
| `appliances_warranty.json` | Appliances |
| `automotive_warranty.json` | Automotive |
| `smartphones_and_mobile_warranty.json` | Smartphones & Mobile |
| `computers_and_laptops_warranty.json` | Computers & Laptops |
| `wearables_and_audio_warranty.json` | Wearables & Audio |
| `home_office_and_furniture_warranty.json` | Home Office & Furniture |
| `power_tools_and_hardware_warranty.json` | Power Tools & Hardware |

## Testing

```bash
cd backend
python -m pytest tests/ -v
```

## Assumptions and Known Limitations

- Tesseract OCR is optional; when absent, a heuristic text parser extracts fields from raw byte streams. Full OCR accuracy requires Tesseract to be installed and on PATH.
- The SQLite database is file-based and created on first startup. No external database server is required.
- The Teachable Machine model was trained locally using the same MobileNetV2 architecture as Google Teachable Machine; weights are loaded from the exported H5 file.
- Demo data is seeded once. Re-seeding is available via the Admin panel or the `/api/admin/seed` endpoint.

## Deployment

Deployed at: *[URL will be added after deployment]*

## Blog

Technical blog: *[URL will be added after publication]*

## Demo Video

*[URL will be added]*

## Teachable Machine Project

*[Link will be added]*

## License

MIT — see [LICENSE](LICENSE).
