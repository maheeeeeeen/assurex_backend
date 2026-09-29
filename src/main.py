"""
AssureX Claim Engine — FastAPI Application Entry Point

This is the main FastAPI application file:
1. Creates the FastAPI app instance with metadata for docs
2. Configures CORS to allow the React frontend
3. Sets up the SQLite database and creates tables on startup
4. Auto-seeds initial demo users, catalog, and claims on first run
5. Mounts static files for generated Claim Summary Cards and uploads
6. Registers all REST API routers under /api
"""

import os
import sys

# Ensure backend directory is in sys.path for robust imports from any cwd
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlmodel import Session, select

from src.database_setup import create_db_and_tables, engine
from src.models import User
from src.routers import auth, products, warranties, claims, policies, admin, notifications

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
CARDS_DIR = os.path.join(UPLOADS_DIR, "cards")
os.makedirs(CARDS_DIR, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup tasks: initialize database tables and seed demo data if empty."""
    create_db_and_tables()

    # Auto-seed if database is brand new
    with Session(engine) as session:
        first_user = session.exec(select(User)).first()
        if not first_user:
            print("[Startup] Initializing default demo data & user accounts...")
            from src.routers.admin import seed_demo_data
            seed_demo_data(session)
            print("[Startup] Demo data seeding complete!")

    yield


app = FastAPI(
    title="AssureX Claim Engine",
    description="AI-Powered Warranty Claim Adjudication Platform",
    version="1.0.0",
    lifespan=lifespan,
)

# --- CORS Configuration ---
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
    "http://localhost:5175",
    "http://127.0.0.1:5175",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

extra_origins = os.getenv("CORS_ORIGINS", "")
if extra_origins:
    ALLOWED_ORIGINS.extend(extra_origins.split(","))

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Health & Info Endpoints ---
@app.get("/health", tags=["System"])
async def health_check():
    """Returns application health status."""
    return {
        "status": "healthy",
        "application": "AssureX Claim Engine",
        "version": "1.0.0",
    }


@app.get("/api", tags=["System"])
async def api_info():
    """Returns basic API information and available endpoints."""
    return {
        "application": "AssureX Claim Engine",
        "version": "1.0.0",
        "description": "AI-Powered Warranty Claim Adjudication API",
        "documentation": "/docs",
        "health": "/health",
    }


# --- Static Files (Uploads & Generated Claim Cards) ---
app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")


# --- API Routers ---
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(products.router, prefix="/api/products", tags=["Products"])
app.include_router(warranties.router, prefix="/api/warranties", tags=["Warranties"])
app.include_router(claims.router, prefix="/api/claims", tags=["Claims"])
app.include_router(policies.router, prefix="/api/policies", tags=["Policies"])
app.include_router(admin.router, prefix="/api/admin", tags=["Admin"])
app.include_router(notifications.router, prefix="/api/notifications", tags=["Notifications"])


# --- Production Static Frontend Bundle (if built) ---
frontend_dist = os.path.join(BASE_DIR, "dist")
if os.path.exists(frontend_dist):
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
