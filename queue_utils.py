from __future__ import annotations

import os

from redis import Redis
from rq import Queue


def get_queue() -> Queue:
    # Render Key Value exposes REDIS_URL (internal URL recommended).
    redis_url = os.environ.get("REDIS_URL", "").strip()
    if not redis_url:
        raise RuntimeError("REDIS_URL is not set")

    conn = Redis.from_url(redis_url)
    return Queue("default", connection=conn)
