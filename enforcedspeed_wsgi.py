"""WSGI entrypoint for Render/Gunicorn.

We intentionally avoid generic module names like `wsgi` to prevent conflicts with
third-party packages that may be importable as `wsgi` on the Python path.
"""

from app import create_app

app = create_app()
