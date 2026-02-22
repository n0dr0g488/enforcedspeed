# config.py
import os
from pathlib import Path


def _normalize_database_url(url: str) -> str:
    """
    Render typically provides DATABASE_URL like:
      postgres://...
      postgresql://...

    We use a pure-Python Postgres driver (pg8000) for maximum compatibility, so SQLAlchemy must use:
      postgresql+pg8000://...

    This avoids native libpq DLL issues on some Windows environments.
    """
    if not url:
        return url

    url = url.strip()

    # Normalize old local runner scheme
    if url.startswith("postgresql+psycopg://"):
        return "postgresql+pg8000://" + url[len("postgresql+psycopg://"):]

    if url.startswith("postgres://"):
        return "postgresql+pg8000://" + url[len("postgres://") :]

    if url.startswith("postgresql://"):
        return "postgresql+pg8000://" + url[len("postgresql://") :]

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

    # Photo uploads (quarantine + async OCR). Keep this tight to control costs.
    MAX_CONTENT_LENGTH = int(os.environ.get("MAX_CONTENT_LENGTH", str(12 * 1024 * 1024)))

    SQLALCHEMY_ENGINE_OPTIONS = {
        "pool_pre_ping": True,
    }

    # Fail fast if Postgres is unreachable (prevents "browser spins forever")
    if SQLALCHEMY_DATABASE_URI.startswith("postgresql+pg8000://"):
        # pg8000 uses a pure-Python connection; keep a short timeout to avoid hanging.
        SQLALCHEMY_ENGINE_OPTIONS["connect_args"] = {"timeout": 3}

    # Google Maps Platform
    # Public key (safe to expose in templates). Restrict by HTTP referrer and API.
    GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", "").strip()

    # Server key for Roads API calls (keep private; do not expose to clients).
    # Recommended restrictions: API restriction (Roads API) and server-side usage controls.
    GOOGLE_MAPS_SERVER_KEY = os.environ.get("GOOGLE_MAPS_SERVER_KEY", "").strip()

    # Public base URL (scheme + host) used to build externally reachable asset URLs.
    # Google Static Maps custom marker icons must be fetchable by Google from the public internet.
    # Example: https://enforcedspeed.com
    PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "").strip().rstrip("/")

    # Base URL for Static Maps custom pin icons. Prefer a dedicated static asset host
    # (e.g. Cloudflare R2 custom domain) so Google can fetch icons reliably.
    # Example: https://static.enforcedspeed.com/pins
    STATIC_PIN_BASE_URL = os.environ.get("STATIC_PIN_BASE_URL", "").strip().rstrip("/")

    # Public base URL for user profile photos served from Cloudflare R2 (public bucket / custom domain).
    # Example: https://static.enforcedspeed.com/profile_photos
    PROFILE_PHOTO_BASE_URL = os.environ.get("PROFILE_PHOTO_BASE_URL", "").strip().rstrip("/")

    # Optional: if you want profile photos in a separate bucket/prefix.
    # Defaults are chosen to work with the existing public `enforcedspeed-static` bucket.
    R2_PROFILE_BUCKET = os.environ.get("R2_PROFILE_BUCKET", "").strip()
    R2_PROFILE_PREFIX = os.environ.get("R2_PROFILE_PREFIX", "profile_photos/").strip()
