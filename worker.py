"""RQ worker entrypoint for Render background worker.

Start command suggestion (Render dashboard):
  python worker.py

This listens on the 'default' queue.
"""

import os

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
        raise RuntimeError("REDIS_URL is not set")

    conn = Redis.from_url(redis_url)

    queues = [Queue("default", connection=conn)]
    worker_cls = Worker
    if os.name == "nt" and SimpleWorker is not None:
        worker_cls = SimpleWorker  # type: ignore

    worker = worker_cls(queues, connection=conn)  # type: ignore
    worker.work()


if __name__ == "__main__":
    main()
