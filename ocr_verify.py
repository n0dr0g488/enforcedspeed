"""Ticket photo verification using Google Gemini Vision (v583).

Replaces the previous Google Vision OCR + custom scoring approach.
Gemini independently determines:
  1. Whether the image appears to be a genuine traffic/speeding citation
  2. The posted speed limit shown on the document
  3. The ticketed/clocked speed shown on the document
  4. A confidence score (0.0-1.0)

Verification logic:
  - is_ticket must be true
  - Gemini-extracted speeds must match user-submitted speeds within +/-2 mph
  - ai_confidence must be >= 0.60 (below this we treat as unverified)

PII note: image bytes are sent to Google Gemini API and immediately discarded.
No response text beyond the structured JSON fields is stored.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Optional, Tuple

from models import db, SpeedReport
from r2_utils import get_bytes, delete_object


# Confidence threshold below which we do not mark as verified.
_CONFIDENCE_THRESHOLD = 0.60

# Gemini model
_GEMINI_MODEL = "models/gemini-2.5-flash"

# Tolerance: AI can misread a digit
_SPEED_TOLERANCE = 2


def _gemini_verify_image(
    img_bytes: bytes,
    submitted_posted: Optional[int],
    submitted_ticketed: Optional[int],
) -> Tuple[bool, Optional[int], Optional[int], float, str]:
    """Send image to Gemini and extract verification result.

    Returns:
        (is_ticket, posted_speed, ticketed_speed, confidence, reason)
    """
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    client = genai.Client(api_key=api_key)

    # Encode image as inline bytes part
    # Detect mime type from magic bytes
    mime = "image/jpeg"
    if img_bytes[:4] == b'\x89PNG':
        mime = "image/png"
    elif img_bytes[:2] == b'%P':
        mime = "application/pdf"

    prompt = """You are analyzing an image to verify a speeding ticket report.

Examine this image carefully and respond with ONLY a JSON object — no other text, no markdown, no explanation.

The JSON must have exactly these fields:
{
  "is_ticket": true or false,
  "posted_speed": integer or null,
  "ticketed_speed": integer or null,
  "confidence": float between 0.0 and 1.0,
  "reason": "short internal reason string"
}

Rules:
- "is_ticket" must be true only if the image clearly appears to be a traffic citation, speeding ticket, or law enforcement speed enforcement document. A photo with random numbers is NOT a ticket.
- "posted_speed" is the speed LIMIT shown on the document (the legal limit, typically 25-80 mph)
- "ticketed_speed" is the speed the vehicle was allegedly traveling (typically higher than posted_speed)
- "confidence" reflects how clearly readable and unambiguous the speeds are (1.0 = crystal clear, 0.0 = unreadable)
- "reason" is a short internal string like "clear_ticket", "no_speeds_visible", "not_a_ticket", "redacted_speeds", etc.
- If the image is a photo of a ticket but speeds are redacted/blocked, set is_ticket=true, speeds=null, confidence=0.2
- If you cannot find speed values, set them to null
- Never guess speeds — only extract what is clearly visible"""

    image_part = types.Part.from_bytes(data=img_bytes, mime_type=mime)
    response = client.models.generate_content(
        model=_GEMINI_MODEL,
        contents=[prompt, image_part],
    )

    raw = (response.text or "").strip()

    # Strip markdown code fences if present
    raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'\s*```$', '', raw, flags=re.MULTILINE)
    raw = raw.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
        else:
            raise ValueError(f"Could not parse Gemini response as JSON: {raw[:200]}")

    is_ticket = bool(data.get("is_ticket", False))
    posted = data.get("posted_speed")
    ticketed = data.get("ticketed_speed")
    confidence = float(data.get("confidence", 0.0))
    reason = str(data.get("reason", "unknown"))[:120]

    # Sanitize speeds
    if posted is not None:
        try:
            posted = int(posted)
            if not (15 <= posted <= 90):
                posted = None
        except (TypeError, ValueError):
            posted = None

    if ticketed is not None:
        try:
            ticketed = int(ticketed)
            if not (15 <= ticketed <= 150):
                ticketed = None
        except (TypeError, ValueError):
            ticketed = None

    confidence = max(0.0, min(1.0, confidence))

    return is_ticket, posted, ticketed, confidence, reason


def verify_ticket_from_r2(report_id: int, bucket: str, key: str) -> None:
    """RQ job: download image from R2, verify with Gemini AI, update DB, delete object."""
    from app import create_app

    app = create_app()

    def _update(fn) -> bool:
        with app.app_context():
            r = SpeedReport.query.get(report_id)
            if r is None:
                return False
            fn(r)
            db.session.commit()
            return True

    # Mark as processing
    if not _update(lambda r: (
        setattr(r, "verify_attempts", (r.verify_attempts or 0) + 1),
        setattr(r, "ocr_status", "processing"),
        setattr(r, "ocr_error", None),
    )):
        try:
            delete_object(bucket, key)
        except Exception:
            pass
        return

    img_bytes = b""
    try:
        img_bytes = get_bytes(bucket, key)
    finally:
        try:
            delete_object(bucket, key)
        except Exception:
            pass

    if not img_bytes:
        _update(lambda r: (
            setattr(r, "verification_status", "unverified"),
            setattr(r, "ocr_status", "not_verified"),
            setattr(r, "ocr_error", "no_image"),
            setattr(r, "verify_reason", "no_image"),
        ))
        return

    def _final_update(r: SpeedReport) -> None:
        # Normalize swapped submitted speeds
        try:
            if (r.posted_speed is not None and r.ticketed_speed is not None
                    and int(r.ticketed_speed) < int(r.posted_speed)):
                r.posted_speed, r.ticketed_speed = int(r.ticketed_speed), int(r.posted_speed)
                r.overage = max(0, int(r.ticketed_speed) - int(r.posted_speed))
        except Exception:
            pass

        try:
            is_ticket, ai_posted, ai_ticketed, confidence, reason = _gemini_verify_image(
                img_bytes,
                submitted_posted=r.posted_speed,
                submitted_ticketed=r.ticketed_speed,
            )
        except Exception as e:
            r.verification_status = "unverified"
            r.ocr_status = "not_verified"
            r.ocr_error = f"{type(e).__name__}: {str(e)[:180]}"
            r.verify_reason = "ai_exception"
            r.ai_model = _GEMINI_MODEL
            return

        # Store AI results
        r.ocr_posted_speed = ai_posted
        r.ocr_ticketed_speed = ai_ticketed
        r.ocr_confidence = confidence
        r.ai_confidence = confidence
        r.ai_model = _GEMINI_MODEL

        # Gate 1: must be a ticket
        if not is_ticket:
            r.verification_status = "unverified"
            r.ocr_status = "not_verified"
            r.ocr_error = "not_a_ticket"
            r.verify_reason = reason
            return

        # Gate 2: confidence threshold
        if confidence < _CONFIDENCE_THRESHOLD:
            r.verification_status = "unverified"
            r.ocr_status = "not_verified"
            r.ocr_error = "low_confidence"
            r.verify_reason = f"{reason} (conf={confidence:.2f})"
            return

        # Gate 3: speeds must be present
        if ai_posted is None or ai_ticketed is None:
            r.verification_status = "unverified"
            r.ocr_status = "not_verified"
            r.ocr_error = "speeds_not_found"
            r.verify_reason = reason
            return

        # Gate 4: speeds must match submission within tolerance
        posted_ok = r.posted_speed is not None and abs(ai_posted - int(r.posted_speed)) <= _SPEED_TOLERANCE
        ticketed_ok = r.ticketed_speed is not None and abs(ai_ticketed - int(r.ticketed_speed)) <= _SPEED_TOLERANCE

        if posted_ok and ticketed_ok:
            r.verification_status = "verified"
            r.ocr_status = "verified"
            r.ocr_error = None
            r.verified_at = datetime.now(timezone.utc)
            r.verify_reason = f"ai_matched (conf={confidence:.2f})"
            return

        # Speeds extracted but don't match submission
        r.verification_status = "unverified"
        r.ocr_status = "not_verified"
        r.ocr_error = "speed_mismatch"
        r.verify_reason = f"submitted={r.posted_speed}/{r.ticketed_speed} ai={ai_posted}/{ai_ticketed}"

    _update(_final_update)
