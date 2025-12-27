# config.py
import os
from pathlib import Path


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

    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url[len("postgres://") :]

    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url[len("postgresql://") :]

    return url


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-change-me")

    # ----- Local default (SQLite) for dev -----
    BASEDIR = Path(__file__).resolve().parent
    INSTANCE_DIR = BASEDIR / "instance"
    INSTANCE_DIR.mkdir(parents=True, exist_ok=True)

    SQLITE_PATH = (INSTANCE_DIR / "enforcedspeed.db").resolve()
    default_sqlite = f"sqlite:///{SQLITE_PATH.as_posix()}"

    # ----- Production (Render Postgres) -----
    raw_db_url = os.environ.get("DATABASE_URL", "").strip()
    SQLALCHEMY_DATABASE_URI = _normalize_database_url(raw_db_url) or default_sqlite

    SQLALCHEMY_TRACK_MODIFICATIONS = False

    SQLALCHEMY_ENGINE_OPTIONS = {
        "pool_pre_ping": True,
    }
