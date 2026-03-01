"""
database/db.py
--------------
Database engine and session management.
Reads DATABASE_URL from settings (env var / .env file).

SQLite     → default for local dev (zero config)
PostgreSQL → recommended for production; set DATABASE_URL env var:
               DATABASE_URL=postgresql://user:pass@host:5432/gpu_orch
"""

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from database.models import Base
from utils.settings import get_settings


def _build_engine():
    settings = get_settings()
    url = settings.database_url

    if settings.is_sqlite():
        # SQLite: disable same-thread check (needed for FastAPI's multi-threaded access)
        engine = create_engine(
            url,
            connect_args={"check_same_thread": False},
            echo=False,
        )
        # Enable WAL mode for better concurrent read performance
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_conn, _):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.close()
    else:
        # PostgreSQL / MySQL: use connection pooling
        engine = create_engine(
            url,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_pre_ping=True,   # test connections before handing out
            echo=False,
        )
    return engine


engine = _build_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Create all tables (idempotent – safe to call on every startup)."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Session:
    """
    FastAPI dependency: yields a scoped DB session.
    Guaranteed to close even if the request raises an exception.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
