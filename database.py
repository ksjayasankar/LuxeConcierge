# luxury_leather_chatbot/database.py

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from typing import Optional # *** ADD THIS IMPORT ***

# --- Direct import from config ---
try:
    from config import DB_URL
except ImportError as e:
    print(f"ERROR in database.py: Cannot import config. ({e})")
    raise

db = SQLAlchemy()
_engine = None
# Use the simpler hint if still on Python < 3.9, otherwise keep the more specific one
# _SessionLocal: Optional[sessionmaker] = None # For Python < 3.9
_SessionLocal: Optional[sessionmaker[Session]] = None # For Python >= 3.9

def init_db(app):
    """Initializes the database connection for the Flask app."""
    global _engine, _SessionLocal

    app.config["SQLALCHEMY_DATABASE_URI"] = DB_URL
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    db.init_app(app)

    with app.app_context():
        current_engine = getattr(db, 'engine', None)
        if current_engine:
             _engine = current_engine
             # print(f"SQLAlchemy engine obtained via db.engine: {_engine}") # Less verbose
        else:
            # print("Warning: db.engine not found directly, creating engine manually.") # Less verbose
            _engine = create_engine(DB_URL)
            if not _engine:
                 print("CRITICAL: Manual engine creation failed.")
                 raise RuntimeError("Failed to obtain SQLAlchemy engine.")
            # print(f"Engine created manually: {_engine}") # Less verbose

    if _engine:
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
        print(f"Database initialized successfully. Session factory created.")
    else:
        print("CRITICAL: Engine is None after initialization attempt.")
        raise RuntimeError("Database engine initialization failed.")

    return _engine

def get_engine():
    """Returns the SQLAlchemy engine. Raises error if not initialized."""
    if _engine is None:
        print("ERROR: get_engine() called but engine is not initialized. Call init_db() first.")
        raise RuntimeError("Database engine not initialized.")
    return _engine

def get_session() -> Session: # Add return type hint
    """Provides a new database session. Raises error if not initialized."""
    if _SessionLocal is None:
        print("ERROR: get_session() called but session factory is not initialized. Call init_db() first.")
        raise RuntimeError("Database session factory not initialized.")
    # Create a new session instance
    session = _SessionLocal()
    return session