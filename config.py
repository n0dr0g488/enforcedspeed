# config.py
import os


class Config:
    """
    App configuration:
    - Local default: SQLite file at instance/enforcedspeed.db
    - Production (Render): DATABASE_URL provided by Render (Postgres)
    """

    # Secret key (Render should set SECRET_KEY in env)
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-me")

    # Database
    # Render typically provides DATABASE_URL (often starting with postgres://)
    database_url = os.getenv("DATABASE_URL", "").strip()

    if database_url:
        # SQLAlchemy expects postgresql:// not postgres://
        if database_url.startswith("postgres://"):
            database_url = database_url.replace("postgres://", "postgresql://", 1)
        SQLALCHEMY_DATABASE_URI = database_url
    else:
        # Local SQLite in /instance (Flask-friendly)
        base_dir = os.path.abspath(os.path.dirname(__file__))
        instance_dir = os.path.join(base_dir, "instance")
        os.makedirs(instance_dir, exist_ok=True)
        SQLALCHEMY_DATABASE_URI = "sqlite:///" + os.path.join(instance_dir, "enforcedspeed.db")

    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Useful defaults
    WTF_CSRF_ENABLED = True
