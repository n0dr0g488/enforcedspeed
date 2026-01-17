from __future__ import annotations

import io
import json
import os
import re
from datetime import datetime, timezone
from typing import Optional, Tuple

from google.cloud import vision
from google.oauth2.service_account import Credentials

from models import db, SpeedReport
from r2_utils import get_bytes, delete_object


TICKET_KEYWORDS = {
    "citation", "traffic", "violation", "offense", "statute", "court",
    "speed", "mph", "limit", "posted", "alleged",
}


def _vision_client() -> vision.ImageAnnotatorClient:
    raw = os.environ.get("GOOGLE_CREDENTIALS_JSON", "").strip()
    if not raw:
        raise RuntimeError("GOOGLE_CREDENTIALS_JSON is not set")

    # Expect a JSON string (including braces) in the env var.
    info = json.loads(raw)
    creds = Credentials.from_service_account_info(info)
    return vision.ImageAnnotatorClient(credentials=creds)


def _extract_speeds(text: str) -> Tuple[Optional[int], Optional[int], float]:
    """Return (posted_speed, ticketed_speed, confidence) from OCR text.

    Heuristic v1:
    - Look for a number (10..120) on lines containing 'limit' or 'posted' => posted
    - Look for a number (10..150) on lines containing 'speed' or 'alleged' (but not 'limit') => ticketed
    - Ticket-likeness gate: require at least one ticket keyword present.
    """
    if not text:
        return None, None, 0.0

    t = text.lower()
    found_kw = any(k in t for k in TICKET_KEYWORDS)
    if not found_kw:
        return None, None, 0.0

    posted = None
    ticketed = None

    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    num_re = re.compile(r"\b(\d{2,3})\b")

    for ln in lines:
        if ("limit" in ln) or ("posted" in ln):
            for m in num_re.findall(ln):
                n = int(m)
                if 10 <= n <= 120:
                    posted = n
                    break
        if posted is not None:
            break

    for ln in lines:
        if ("speed" in ln or "alleged" in ln or "mph" in ln) and ("limit" not in ln) and ("posted" not in ln):
            for m in num_re.findall(ln):
                n = int(m)
                if 10 <= n <= 150:
                    ticketed = n
                    break
        if ticketed is not None:
            break

    # Fallback: if we saw keywords but didn't find structured lines, pick two plausible speeds.
    if posted is None or ticketed is None:
        nums = [int(x) for x in num_re.findall(t) if 10 <= int(x) <= 150]
        nums = sorted(set(nums))
        if len(nums) >= 2:
            # Prefer a smaller posted and larger ticketed.
            if posted is None:
                posted = min(nums)
            if ticketed is None:
                ticketed = max(nums)

    if posted is None or ticketed is None:
        return posted, ticketed, 0.3

    # Basic sanity: ticketed should be >= posted.
    if ticketed < posted:
        return posted, ticketed, 0.3

    return posted, ticketed, 0.7


def verify_ticket_from_r2(report_id: int, bucket: str, key: str) -> None:
    """RQ job: download image from R2, OCR it, verify against submitted speeds, update DB, delete object."""
    # Import app lazily to avoid circular imports.
    from app import create_app

    app = create_app()
    with app.app_context():
        report = SpeedReport.query.get(report_id)
        if report is None:
            # Nothing to update, but still delete the object.
            delete_object(bucket, key)
            return

        report.verify_attempts = (report.verify_attempts or 0) + 1

        img_bytes = get_bytes(bucket, key)
        # Always delete quarantine object regardless of success.
        delete_object(bucket, key)

        if not img_bytes:
            report.verification_status = "unverified"
            report.verify_reason = "no_image"
            db.session.commit()
            return

        client = _vision_client()
        image = vision.Image(content=img_bytes)
        resp = client.text_detection(image=image)
        if resp.error.message:
            report.verification_status = "unverified"
            report.verify_reason = "ocr_error"
            db.session.commit()
            return

        full_text = ""
        if resp.text_annotations:
            full_text = resp.text_annotations[0].description or ""

        ocr_posted, ocr_ticketed, conf = _extract_speeds(full_text)
        report.ocr_posted_speed = ocr_posted
        report.ocr_ticketed_speed = ocr_ticketed
        report.ocr_confidence = float(conf)

        if ocr_posted is None or ocr_ticketed is None:
            report.verification_status = "unverified"
            report.verify_reason = "no_speeds"
            db.session.commit()
            return

        # Verified means OCR matches the user's submitted numbers.
        if (ocr_posted == report.posted_speed) and (ocr_ticketed == report.ticketed_speed) and conf >= 0.6:
            report.verification_status = "verified"
            report.verified_at = datetime.now(timezone.utc)
            report.verify_reason = None
        else:
            report.verification_status = "unverified"
            report.verify_reason = "mismatch"

        db.session.commit()
