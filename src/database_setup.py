"""
Database Setup — SQLModel + SQLite Configuration

This module:
1. Configures the SQLite database engine with a file-based DB in backend/database/
2. Creates the sessionmaker for dependency injection into routes
3. Provides create_db_and_tables() called on app startup
4. Exposes get_session() as a FastAPI dependency for DB access
5. Uses SQLModel's create_engine which wraps SQLAlchemy under the hood
"""

from sqlmodel import SQLModel, Session, create_engine
import os

# Database file location
DATABASE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "database")
os.makedirs(DATABASE_DIR, exist_ok=True)
DATABASE_URL = f"sqlite:///{os.path.join(DATABASE_DIR, 'assurex.db')}"

# Create engine — connect_args needed for SQLite to allow multi-thread access
engine = create_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False},
)


def create_db_and_tables():
    """Create all tables defined by SQLModel classes. Safe to call multiple times."""
    import src.models  # Ensure models are registered in SQLModel metadata
    SQLModel.metadata.create_all(engine)
    _migrate_sqlite_columns()


def _migrate_sqlite_columns():
    """Safely adds missing columns to existing SQLite tables if not present."""
    import sqlite3
    db_path = os.path.join(DATABASE_DIR, "assurex.db")
    if not os.path.exists(db_path):
        return
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        existing_cols = {col[1] for col in cur.execute("PRAGMA table_info(claims)").fetchall()}
        if existing_cols:
            new_cols = [
                ("receipt_hash", "TEXT"),
                ("warranty_card_hash", "TEXT"),
                ("fault_evidence_hash", "TEXT"),
                ("product_image_hash", "TEXT"),
                ("fault_video_path", "TEXT"),
                ("fault_video_hash", "TEXT"),
                ("barcode_image_path", "TEXT"),
                ("barcode_image_hash", "TEXT"),
                ("model_number_on_receipt", "TEXT"),
                ("model_number_on_warranty_card", "TEXT"),
                ("serial_number_on_barcode", "TEXT"),
                ("model_number_on_barcode", "TEXT"),
                ("cross_verification_json", "TEXT"),
                ("invoice_number", "TEXT"),
                ("duplicate_claim_details", "TEXT"),
            ]
            for col_name, col_type in new_cols:
                if col_name not in existing_cols:
                    cur.execute(f"ALTER TABLE claims ADD COLUMN {col_name} {col_type}")
            conn.commit()

        # Check products table
        prod_cols = {col[1] for col in cur.execute("PRAGMA table_info(products)").fetchall()}
        if prod_cols:
            if "user_id" not in prod_cols:
                cur.execute("ALTER TABLE products ADD COLUMN user_id INTEGER")
            if "warranty_status" not in prod_cols:
                cur.execute("ALTER TABLE products ADD COLUMN warranty_status TEXT DEFAULT 'Active'")
            conn.commit()

        conn.close()
    except Exception as e:
        print(f"[database_setup] Migration notice: {e}")


def get_session():
    """
    FastAPI dependency that provides a database session.
    Usage in routes: session: Session = Depends(get_session)
    Automatically closes the session when the request is done.
    """
    with Session(engine) as session:
        yield session


# Run migration automatically on module load so SQLite schema stays synchronized
_migrate_sqlite_columns()

