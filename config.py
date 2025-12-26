# config.py
import os


def _normalize_database_url(url: str) -> str:
    """
    Render typically provides DATABASE_URL like:
      postgres://...
      postgresql://...

    We are using psycopg v3, so SQLAlchemy must use:
      postgresql+psycopg://...

    This prevents SQLAlchemy from trying to import psycopg2.
    """
    if not url:
        return url

    url = url.strip()

    # Render/Heroku legacy scheme
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url[len("postgres://") :]

    # Common postgres scheme
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url[len("postgresql://") :]

    # If user already set a SQLAlchemy dialect (keep it)
    return url


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-change-me")

    # Local default (SQLite) for dev
    default_sqlite = "sqlite:///instance/enforcedspeed.db"

    raw_db_url = os.environ.get("DATABASE_URL", "").strip()
    SQLALCHEMY_DATABASE_URI = _normalize_database_url(raw_db_url) or default_sqlite

    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Good defaults for managed Postgres
    SQLALCHEMY_ENGINE_OPTIONS = {
        "pool_pre_ping": True,
    }
