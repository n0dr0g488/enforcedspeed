"""WSGI entrypoint for Render/Gunicorn.

We intentionally avoid generic module names like `wsgi` to prevent conflicts with
third-party packages that may be importable as `wsgi` on the Python path.
"""

import traceback

try:
    from app import create_app  # noqa: E402
    app = create_app()
except Exception:
    # Ensure the real import error shows up in Render logs (Gunicorn may otherwise
    # collapse import failures into "Failed to find application object").
    traceback.print_exc()
    raise
