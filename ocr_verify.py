from __future__ import annotations

import io
import json
import os
import re
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from google.cloud import vision
from google.oauth2.service_account import Credentials

from models import db, SpeedReport
from r2_utils import get_bytes, delete_object


TICKET_KEYWORDS = {
    # Common ticket words (broad on purpose; we score them rather than hard-gating too aggressively).
    "citation", "traffic", "violation", "offense", "statute", "code", "court",
    "speed", "mph", "limit", "posted", "alleged", "clocked", "radar", "lidar",
    "officer", "agency", "defendant", "fine", "appearance",
}


# NOTE: We intentionally keep verification deterministic + auditable.
# "Verified (Automated)" means we extracted a plausible (limit, speed) pair and it matches
# the user-submitted values within tolerance. This is designed to remain robust even when
# users upload heavily redacted ticket photos (PII blocked out).


def _vision_client() -> vision.ImageAnnotatorClient:
    """Create a Vision client.

    Supports local/dev auth patterns:
    1) GOOGLE_CREDENTIALS_JSON: either
       - a filesystem path to a service-account JSON file (recommended), OR
       - the full service-account JSON as a JSON string.
    2) GOOGLE_APPLICATION_CREDENTIALS: path to a service-account JSON file (standard).

    If (2) is set, the Google library will pick it up automatically.
    """

    raw = os.environ.get("GOOGLE_CREDENTIALS_JSON", "").strip()
    if raw:
        # If the env var is actually a path, load the file.
        if raw.lower().endswith('.json') and os.path.exists(raw):
            with open(raw, 'r', encoding='utf-8') as f:
                info = json.load(f)
        else:
            info = json.loads(raw)

        creds = Credentials.from_service_account_info(info)
        return vision.ImageAnnotatorClient(credentials=creds)

    # Fall back to ADC / GOOGLE_APPLICATION_CREDENTIALS.
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        return vision.ImageAnnotatorClient()

    raise RuntimeError(
        "Neither GOOGLE_CREDENTIALS_JSON nor GOOGLE_APPLICATION_CREDENTIALS is set"
    )



class _Candidate:
    __slots__ = ("value", "weight")

    def __init__(self, value: int, weight: float):
        self.value = int(value)
        self.weight = float(weight)


def _best_speed_pair(
    text: str,
    submitted_posted: Optional[int],
    submitted_ticketed: Optional[int],
) -> Tuple[Optional[int], Optional[int], float, str]:
    """Return (ocr_posted, ocr_ticketed, score, reason) from OCR text.

    Deterministic + redaction-tolerant:
    - We try to extract *many* candidate limits/speeds, then choose the best pair.
    - Primary success path is matching the user-submitted speeds within tolerance.
      (Users often redact PII, but still leave limit/speed visible.)

    "score" is an internal 0..1 score used for tuning/logging (not user-visible).
    "reason" is an internal string used for debugging/tuning.
    """
    if not text:
        return None, None, 0.0, "no_text"

    t = text.lower()
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]

    # Ticket-likeness: count keyword hits (bonus only; never a hard gate).
    kw_hits = sum(1 for k in TICKET_KEYWORDS if k in t)

    # Extract 2-3 digit numbers.
    num_re = re.compile(r"\b(\d{2,3})\b")

    # Marker lists (broad on purpose).
    posted_markers = ("speed limit", "spd limit", "spd lmt", "spd lmt.", "limit", "posted", "pstd", "lmt")
    speed_markers = ("vehicle speed", "veh speed", "speed", "alleged", "clocked", "measured", "mph", "radar", "lidar")

    def _nums_in_line(ln: str) -> List[int]:
        out: List[int] = []
        for m in num_re.findall(ln):
            try:
                out.append(int(m))
            except Exception:
                continue
        return out

    # Candidate collection
    limit_cands: List[_Candidate] = []
    speed_cands: List[_Candidate] = []

    # Line-driven candidates (higher weight)
    for ln in lines:
        nums = _nums_in_line(ln)
        if not nums:
            continue

        has_posted = any(m in ln for m in posted_markers)
        has_speed = any(m in ln for m in speed_markers)

        # If line clearly indicates limit, add those numbers as limit candidates.
        if has_posted:
            for n in nums:
                if 15 <= n <= 85:
                    limit_cands.append(_Candidate(n, 0.55))

        # If line indicates speed (and not also a limit line), add as speed candidates.
        if has_speed and not has_posted:
            for n in nums:
                if 15 <= n <= 120:
                    speed_cands.append(_Candidate(n, 0.55))

    # Global fallback candidates (lower weight)
    all_nums = [int(x) for x in num_re.findall(t)]
    for n in all_nums:
        if 15 <= n <= 85:
            limit_cands.append(_Candidate(n, 0.20))
        if 15 <= n <= 120:
            speed_cands.append(_Candidate(n, 0.20))

    # De-duplicate by keeping max weight per value
    def _dedupe(cands: List[_Candidate]) -> List[_Candidate]:
        best = {}
        for c in cands:
            best[c.value] = max(best.get(c.value, 0.0), c.weight)
        return [_Candidate(v, w) for v, w in best.items()]

    limit_cands = _dedupe(limit_cands)
    speed_cands = _dedupe(speed_cands)

    if not limit_cands or not speed_cands:
        return None, None, 0.0, "missing_candidates"

    # Pair scoring
    tol = 2
    best_pair: Tuple[Optional[int], Optional[int]] = (None, None)
    best_score = -1.0
    best_reason = "no_pair"

    for lc in limit_cands:
        for sc in speed_cands:
            limit = lc.value
            speed = sc.value

            # Basic plausibility
            if not (15 <= limit <= 85 and 15 <= speed <= 120):
                continue

            score = 0.0
            score += lc.weight + sc.weight

            # Prefer realistic relationship
            delta = speed - limit
            if delta >= 0:
                score += 0.20
            else:
                score -= 0.40

            if 1 <= delta <= 60:
                score += 0.15
            elif delta > 60:
                score -= 0.10

            # Strongest signal: match to the user-submitted speeds.
            posted_ok = False
            ticketed_ok = False
            if submitted_posted is not None:
                posted_ok = abs(limit - int(submitted_posted)) <= tol
                if posted_ok:
                    score += 0.35
                else:
                    # Small penalty for being far away from submitted.
                    score -= min(0.20, abs(limit - int(submitted_posted)) / 100.0)

            if submitted_ticketed is not None:
                ticketed_ok = abs(speed - int(submitted_ticketed)) <= tol
                if ticketed_ok:
                    score += 0.35
                else:
                    score -= min(0.20, abs(speed - int(submitted_ticketed)) / 100.0)

            # Ticket-likeness is bonus only (redaction tolerant).
            score += min(0.10, 0.02 * kw_hits)

            # Clamp-ish to avoid runaway scores
            score = max(-1.0, min(1.5, score))

            if score > best_score:
                best_score = score
                best_pair = (limit, speed)
                if submitted_posted is not None and submitted_ticketed is not None and posted_ok and ticketed_ok:
                    best_reason = "matched_submission"
                elif delta >= 0 and 1 <= delta <= 60:
                    best_reason = "internal_consistency"
                else:
                    best_reason = "weak_pair"

    if best_pair == (None, None):
        return None, None, 0.0, "no_pair"

    # Normalize best_score to 0..1 for logging
    norm = (best_score + 1.0) / 2.5  # maps [-1,1.5] -> [0,1]
    norm = max(0.0, min(1.0, norm))
    return int(best_pair[0]), int(best_pair[1]), float(norm), best_reason


def verify_ticket_from_r2(report_id: int, bucket: str, key: str) -> None:
    """RQ job: download image from R2, OCR it, verify against submitted speeds, update DB, delete object."""
    # Import app lazily to avoid circular imports.
    from app import create_app

    app = create_app()

    def _update_report(update_fn) -> bool:
        """Run an update against the SpeedReport inside an app context."""
        with app.app_context():
            r = SpeedReport.query.get(report_id)
            if r is None:
                return False
            update_fn(r)
            db.session.commit()
            return True

    # Mark as processing + increment attempts.
    if not _update_report(
        lambda r: (
            setattr(r, "verify_attempts", (r.verify_attempts or 0) + 1),
            setattr(r, "ocr_status", "processing"),
            setattr(r, "ocr_error", None),
        )
    ):
        # Nothing to update, but still delete the object.
        delete_object(bucket, key)
        return

    img_bytes = b""
    try:
        # Download bytes from R2
        img_bytes = get_bytes(bucket, key)
    finally:
        # Always delete quarantine object regardless of success.
        try:
            delete_object(bucket, key)
        except Exception:
            # Don't fail the job just because cleanup failed
            pass

    if not img_bytes:
        # User-facing behavior: if OCR can't run / no bytes, treat as "Not Verified".
        _update_report(
            lambda r: (
                setattr(r, "verification_status", "unverified"),
                setattr(r, "verify_reason", "no_image"),
                setattr(r, "ocr_status", "not_verified"),
                setattr(r, "ocr_error", "no_image"),
            )
        )
        return

    try:
        client = _vision_client()
        image = vision.Image(content=img_bytes)
        resp = client.text_detection(image=image)
    except Exception as e:
        # Any Vision/credentials/billing/network error => "Not Verified" (we keep details internally).
        _update_report(
            lambda r: (
                setattr(r, "verification_status", "unverified"),
                setattr(r, "verify_reason", "ocr_exception"),
                setattr(r, "ocr_status", "not_verified"),
                setattr(r, "ocr_error", f"{type(e).__name__}: {str(e)[:180]}"),
            )
        )
        return

    # Vision API can return a response-level error even if no exception was raised.
    if getattr(resp, "error", None) and getattr(resp.error, "message", ""):
        _update_report(
            lambda r: (
                setattr(r, "verification_status", "unverified"),
                setattr(r, "verify_reason", "ocr_error"),
                setattr(r, "ocr_status", "not_verified"),
                setattr(r, "ocr_error", str(resp.error.message)[:180]),
            )
        )
        return

    full_text = ""
    if resp.text_annotations:
        full_text = resp.text_annotations[0].description or ""

    # Final: choose best pair using the submitted values, then set verified/unverified.
    def _final_update(r: SpeedReport) -> None:
        # Choose the best pair *using the submitted values* (primary success path).
        best_posted, best_ticketed, score, reason = _best_speed_pair(
            full_text,
            submitted_posted=r.posted_speed,
            submitted_ticketed=r.ticketed_speed,
        )

        r.ocr_posted_speed = best_posted
        r.ocr_ticketed_speed = best_ticketed
        r.ocr_confidence = float(score)

        if best_posted is None or best_ticketed is None:
            r.verification_status = "unverified"
            r.ocr_status = "not_verified"
            r.ocr_error = "no_speeds"
            r.verify_reason = "no_speeds"
            return

        # Tolerances: OCR can misread digits or latch onto nearby numbers.
        tol = 2
        posted_ok = abs(int(best_posted) - int(r.posted_speed)) <= tol
        ticketed_ok = abs(int(best_ticketed) - int(r.ticketed_speed)) <= tol

        # Verified (Automated) = extracted plausible pair that matches submission within tolerance.
        if posted_ok and ticketed_ok:
            r.verification_status = "verified"
            r.ocr_status = "verified"
            r.ocr_error = None
            r.verified_at = datetime.now(timezone.utc)
            r.verify_reason = "matched_submission"
            return

        # Otherwise, it remains Not Verified (users may upload nature photos, unreadable images, etc.).
        r.verification_status = "unverified"
        r.ocr_status = "not_verified"
        r.ocr_error = "not_verified"
        # Internal detail only.
        r.verify_reason = reason or "mismatch"

    _update_report(_final_update)
