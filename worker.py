"""RQ worker entrypoint for Render background worker.

Start command suggestion (Render dashboard):
  python worker.py

This listens on the 'default' queue.
"""

import os

from redis import Redis
from rq import Worker, Queue, Connection


def main() -> None:
    redis_url = os.environ.get("REDIS_URL", "").strip()
    if not redis_url:
        raise RuntimeError("REDIS_URL is not set")

    conn = Redis.from_url(redis_url)
    with Connection(conn):
        Worker([Queue("default")]).work()


if __name__ == "__main__":
    main()
