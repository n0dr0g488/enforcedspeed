"""RQ worker entrypoint for Render background worker.

Start command suggestion (Render dashboard):
  python worker.py

This listens on the 'default' queue.
"""

import os
import sys
from urllib.parse import urlparse

# Load local environment variables from .env (if present) for local dev.
try:
    from dotenv import load_dotenv

    load_dotenv(override=True)
except Exception:
    pass


from redis import Redis
from rq import Worker, Queue
try:
    # SimpleWorker avoids os.fork(), which doesn't exist on Windows.
    from rq.worker import SimpleWorker  # type: ignore
except Exception:
    SimpleWorker = None  # type: ignore


def main() -> None:
    redis_url = os.environ.get("REDIS_URL", "").strip()
    if not redis_url:
        print("[WORKER] REDIS_URL is not set. Worker will not start.", file=sys.stderr)
        print("[WORKER] Tip: start your Redis container and ensure REDIS_URL is present in .env / env vars.", file=sys.stderr)
        sys.exit(2)

    def _safe_target(url: str) -> str:
        try:
            p = urlparse(url)
            host = p.hostname or "?"
            port = p.port or "?"
            return f"{host}:{port}"
        except Exception:
            return "(unparsed)"

    conn = Redis.from_url(redis_url)

    # Fail fast with a clear message if Redis isn't reachable.
    try:
        conn.ping()
    except Exception as e:
        print(f"[WORKER] Redis not reachable at {_safe_target(redis_url)}. Worker will not start.", file=sys.stderr)
        print("[WORKER] Tip: ensure Docker Desktop is running and the Redis container is up (port 6379).", file=sys.stderr)
        print(f"[WORKER] Underlying error: {e}", file=sys.stderr)
        sys.exit(2)

    queues = [Queue("default", connection=conn)]
    worker_cls = Worker
    if os.name == "nt" and SimpleWorker is not None:
        worker_cls = SimpleWorker  # type: ignore

    worker = worker_cls(queues, connection=conn)  # type: ignore
    worker.work()


if __name__ == "__main__":
    main()
