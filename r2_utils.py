from __future__ import annotations

import os
from typing import Optional

import boto3


def _r2_client():
    endpoint = os.environ.get("R2_ENDPOINT", "").strip()
    access_key = os.environ.get("R2_ACCESS_KEY_ID", "").strip()
    secret_key = os.environ.get("R2_SECRET_ACCESS_KEY", "").strip()

    if not endpoint or not access_key or not secret_key:
        return None

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
    )


def put_bytes(bucket: str, key: str, data: bytes, content_type: str = "image/jpeg") -> bool:
    c = _r2_client()
    if c is None:
        return False
    c.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)
    return True


def get_bytes(bucket: str, key: str) -> Optional[bytes]:
    c = _r2_client()
    if c is None:
        return None
    obj = c.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()


def delete_object(bucket: str, key: str) -> None:
    c = _r2_client()
    if c is None:
        return
    c.delete_object(Bucket=bucket, Key=key)
