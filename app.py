# app.py
from __future__ import annotations

import os
import requests

def _get_county_filter_from_request():
    """Parse county filter from query params.

    Accepts either ?county_geoid=<GEOID> or legacy ?county=<GEOID>.
    Returns (geoid, label). Label is best-effort and may be empty.

    NOTE: County GEOIDs are 5 digits (statefp+countyfp). We validate to avoid
    accidentally treating the visible label text as a GEOID.
    """
    geoid = (request.args.get('county_geoid') or request.args.get('county') or '').strip()
    label = ''

    # Validate county GEOID (5 digits). If it isn't, ignore it.
    if geoid and not re.fullmatch(r"\d{5}", geoid):
        geoid = ''

    if geoid:
        try:
            row = db.session.execute(
                text('SELECT namelsad, name, stusps FROM counties WHERE geoid = :g LIMIT 1'),
                {'g': geoid},
            ).mappings().first()
            if row:
                nm = row.get('namelsad') or row.get('name') or ''
                st = row.get('stusps') or ''
                label = f"{nm}, {st}".strip(', ')
        except Exception:
            # If the counties table/columns aren't available, don't poison the session.
            try:
                db.session.rollback()
            except Exception:
                pass
            label = ''

    return geoid, label


# Load local environment variables from .env (if present).
# This avoids having to re-type secrets each run during local dev.
try:
    from dotenv import load_dotenv

    load_dotenv(override=True)
except Exception:
    pass

import io
import uuid
import re
import math
from functools import wraps
import json
import urllib.parse
import urllib.request
import urllib.parse
from datetime import datetime, timedelta, timezone
from statistics import median
from bisect import bisect_right
from typing import Dict, List
from state_bounds import get_bounds_for_state

import jwt
from jwt import ExpiredSignatureError, InvalidTokenError

from flask import Flask, render_template, redirect, url_for, jsonify, request, flash, abort, Response
from flask_login import LoginManager, login_user, logout_user, current_user, login_required
from sqlalchemy import text, func, case, tuple_, or_, and_, bindparam
from sqlalchemy.orm import joinedload
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

from PIL import Image

# Optional: accept iPhone HEIC/HEIF and normalize to JPEG.
try:
    import pillow_heif  # type: ignore

    pillow_heif.register_heif_opener()
except Exception:
    pillow_heif = None  # type: ignore

# Optional: accept PDF uploads by rendering page 1 to an image.
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None  # type: ignore


from config import Config
from forms import (
    SpeedReportForm,
    SubmitTicketForm,
    RegisterForm,
    LoginForm,
    ChangePasswordForm,
    ForgotPasswordForm,
    ResetPasswordForm,
    LikeForm,
    CommentForm,
    DeleteCommentForm,
)
from models import db, SpeedReport, User, Like, Comment, FollowedCounty, normalize_road, state_code_from_value


# -----------------------------------------------------------------------------
# DB schema patching (migration safety net)
#
# Your local DB can easily drift behind the latest model definitions (especially
# when switching ZIP versions or recreating containers). When that happens, the
# app will crash with “column does not exist”. We keep this patcher intentionally
# conservative: it only *adds* missing columns with safe types/defaults.
# It never drops/renames columns.
# -----------------------------------------------------------------------------

from sqlalchemy import text as sa_text


def _col_exists(conn, table: str, column: str, schema: str = "public") -> bool:
    q = sa_text(
        """
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = :schema
          AND table_name = :table
          AND column_name = :column
        LIMIT 1
        """
    )
    return conn.execute(q, {"schema": schema, "table": table, "column": column}).first() is not None


def _ensure_col(conn, table: str, column: str, ddl: str, schema: str = "public") -> None:
    if not _col_exists(conn, table, column, schema=schema):
        conn.execute(sa_text(f"ALTER TABLE {schema}.{table} ADD COLUMN {ddl}"))


def ensure_schema_patches() -> None:
    """Best-effort schema patches so the app can start on older DBs."""
    # db must already be initialized + have an engine
    with db.engine.begin() as conn:
        # --- speed_reports: columns frequently added over time ---
        # Core identity
        _ensure_col(conn, "speed_reports", "route_class", "route_class TEXT")
        _ensure_col(conn, "speed_reports", "overage", "overage INTEGER")

        # County fields (single-page submit)
        _ensure_col(conn, "speed_reports", "county_geoid", "county_geoid TEXT")
        _ensure_col(conn, "speed_reports", "county_name", "county_name TEXT")
        _ensure_col(conn, "speed_reports", "county_state", "county_state TEXT")

        # Location fields
        _ensure_col(conn, "speed_reports", "raw_lat", "raw_lat DOUBLE PRECISION")
        _ensure_col(conn, "speed_reports", "raw_lng", "raw_lng DOUBLE PRECISION")
        _ensure_col(conn, "speed_reports", "lat", "lat DOUBLE PRECISION")
        _ensure_col(conn, "speed_reports", "lng", "lng DOUBLE PRECISION")
        _ensure_col(conn, "speed_reports", "location_hint", "location_hint TEXT")
        _ensure_col(conn, "speed_reports", "location_source", "location_source TEXT")
        _ensure_col(conn, "speed_reports", "location_accuracy_m", "location_accuracy_m INTEGER")

        # Photo/OCR
        _ensure_col(conn, "speed_reports", "photo_key", "photo_key TEXT")
        _ensure_col(conn, "speed_reports", "ocr_status", "ocr_status TEXT")
        _ensure_col(conn, "speed_reports", "verification_status", "verification_status TEXT")
        _ensure_col(conn, "speed_reports", "ocr_posted_speed", "ocr_posted_speed INTEGER")
        _ensure_col(conn, "speed_reports", "ocr_ticketed_speed", "ocr_ticketed_speed INTEGER")
        _ensure_col(conn, "speed_reports", "ocr_confidence", "ocr_confidence DOUBLE PRECISION")

        # Verification / moderation
        _ensure_col(conn, "speed_reports", "verified_at", "verified_at TIMESTAMP")
        _ensure_col(conn, "speed_reports", "verified_by", "verified_by INTEGER")
        _ensure_col(conn, "speed_reports", "verify_attempts", "verify_attempts INTEGER DEFAULT 0")
        _ensure_col(conn, "speed_reports", "verify_reason", "verify_reason TEXT")

        _ensure_col(conn, "speed_reports", "is_deleted", "is_deleted BOOLEAN DEFAULT FALSE")
        _ensure_col(conn, "speed_reports", "deleted_at", "deleted_at TIMESTAMP")
        _ensure_col(conn, "speed_reports", "deleted_by", "deleted_by INTEGER")

        # created_at exists in most versions, but add if missing
        _ensure_col(conn, "speed_reports", "created_at", "created_at TIMESTAMP DEFAULT NOW()")

        # Optional: backfill overage for existing rows (safe + idempotent)
        # Only runs if overage exists; the UPDATE is harmless if already correct.
        if _col_exists(conn, "speed_reports", "overage"):
            conn.execute(
                sa_text(
                    """
                    UPDATE speed_reports
                    SET overage = (ticketed_speed - posted_speed)
                    WHERE overage IS NULL
                      AND ticketed_speed IS NOT NULL
                      AND posted_speed IS NOT NULL;
                    """
                )
            )


        # --- users: account integrity fields ---
        _ensure_col(conn, "users", "phone", "phone TEXT")
        _ensure_col(conn, "users", "birthdate", "birthdate DATE")

from queue_utils import get_queue
from r2_utils import put_bytes


STATE_OPTIONS = [
    "AL - Alabama", "AK - Alaska", "AZ - Arizona", "AR - Arkansas", "CA - California",
    "CO - Colorado", "CT - Connecticut", "DE - Delaware", "FL - Florida", "GA - Georgia",
    "HI - Hawaii", "ID - Idaho", "IL - Illinois", "IN - Indiana", "IA - Iowa",
    "KS - Kansas", "KY - Kentucky", "LA - Louisiana", "ME - Maine", "MD - Maryland",
    "MA - Massachusetts", "MI - Michigan", "MN - Minnesota", "MS - Mississippi", "MO - Missouri",
    "MT - Montana", "NE - Nebraska", "NV - Nevada", "NH - New Hampshire", "NJ - New Jersey",
    "NM - New Mexico", "NY - New York", "NC - North Carolina", "ND - North Dakota", "OH - Ohio",
    "OK - Oklahoma", "OR - Oregon", "PA - Pennsylvania", "RI - Rhode Island", "SC - South Carolina",
    "SD - South Dakota", "TN - Tennessee", "TX - Texas", "UT - Utah", "VT - Vermont",
    "VA - Virginia", "WA - Washington", "WV - West Virginia", "WI - Wisconsin", "WY - Wyoming",
    "DC - District of Columbia",
]

# Convenience map for canonical state storage ("MO" -> "MO - Missouri").
STATE_BY_ABBR = {s[:2].upper(): s for s in STATE_OPTIONS}

# Template-friendly [("MO", "Missouri"), ...] pairs.
STATE_PAIRS = []
for _s in STATE_OPTIONS:
    try:
        _abbr, _name = _s.split(" - ", 1)
        STATE_PAIRS.append((_abbr.strip().upper(), _name.strip()))
    except Exception:
        # Fallback: best-effort two-letter code + full string
        STATE_PAIRS.append((_s[:2].upper(), _s))


def normalize_state_group(value: str) -> str:
    if not value:
        return ""
    v = value.strip()
    m = re.match(r"^\s*([A-Za-z]{2})\b", v)
    if m:
        return m.group(1).upper()
    name = v.lower()
    name = re.sub(r"[^a-z\s]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name.title()


def base_road_key(road_key: str) -> str:
    if not road_key:
        return ""
    return re.sub(r"-(nb|sb|eb|wb)\b", "", road_key)


def format_road_bucket(road_key: str) -> str:
    """
    Display formatter only. Does NOT change stored buckets.
    Supports:
      i95 -> I-95
      us50 -> US-50
      route95 -> Route 95
      va288 -> VA-288
      i95-nb -> I-95 NB
    """
    if not road_key:
        return ""

    k = road_key.strip().lower()

    dir_suffix = ""
    m_dir = re.search(r"-(nb|sb|eb|wb)$", k)
    if m_dir:
        dir_suffix = " " + m_dir.group(1).upper()
        k = k[: -(len(m_dir.group(0)))]  # remove "-nb"

    m = re.fullmatch(r"i(\d{1,3})", k)
    if m:
        return f"I-{m.group(1)}{dir_suffix}"

    m = re.fullmatch(r"us(\d{1,3})", k)
    if m:
        return f"US-{m.group(1)}{dir_suffix}"

    m = re.fullmatch(r"route(\d{1,4})", k)
    if m:
        return f"Route {m.group(1)}{dir_suffix}"

    # State route keys like va288, tx121, ca1, etc.
    m = re.fullmatch(r"([a-z]{2})(\d{1,4})", k)
    if m:
        return f"{m.group(1).upper()}-{m.group(2)}{dir_suffix}"

    return road_key.upper()


def static_map_url(lat: float | None, lng: float | None, *, zoom: int = 14, width: int = 640, height: int = 360) -> str:
    """Build a Google Static Maps URL for a single pin (used as feed preview).

    NOTE: This uses GOOGLE_MAPS_API_KEY (browser key). Restrict it by HTTP referrer in Google Cloud Console.
    """
    try:
        from flask import current_app
        key = (current_app.config.get("GOOGLE_MAPS_API_KEY") or "").strip()
    except Exception:
        key = (os.environ.get("GOOGLE_MAPS_API_KEY") or "").strip()

    if not key or lat is None or lng is None:
        return ""

    center = f"{lat:.6f},{lng:.6f}"
    params = {
        "center": center,
        "zoom": str(int(zoom)),
        "size": f"{int(width)}x{int(height)}",
        "scale": "2",
        "maptype": "roadmap",
        "markers": center,
        "key": key,
    }
    return "https://maps.googleapis.com/maps/api/staticmap?" + urllib.parse.urlencode(params)




# ---- County static map helper (feed card) ----
# We want feed/map previews to be stable and uniform: show the whole county outline,
# not a zoomed-in street view that varies by pin placement.
# This uses the same GOOGLE_MAPS_API_KEY as static_map_url().

def _encode_polyline(points):
    """Encode a list of (lat, lng) into a Google Encoded Polyline string."""
    result = []
    prev_lat = 0
    prev_lng = 0

    def _enc(value):
        value = ~(value << 1) if value < 0 else (value << 1)
        while value >= 0x20:
            result.append(chr((0x20 | (value & 0x1F)) + 63))
            value >>= 5
        result.append(chr(value + 63))

    for lat, lng in points:
        ilat = int(round(lat * 1e5))
        ilng = int(round(lng * 1e5))
        _enc(ilat - prev_lat)
        _enc(ilng - prev_lng)
        prev_lat, prev_lng = ilat, ilng

    return ''.join(result)


def _county_bbox_and_outline_polyline(geoid: str):
    """Return (minlat, minlng, maxlat, maxlng, polyline) for the county geom.

    Notes
    - Static Maps path data is carried in the URL, so we must simplify/limit the
      outline somewhat.
    - We intentionally draw ONLY the exterior ring (no holes) for the feed
      preview. This keeps the URL smaller and avoids complex multi-path shapes.
    - To keep zoom framing consistent with what we actually draw, we compute the
      bbox from the exterior ring we choose (not from the full raw geometry).
    """
    try:
        # Gentler simplification than earlier versions (0.01 was too aggressive).
        # ~0.0025 degrees is roughly ~275m in latitude; the point/URL cap below
        # provides an additional safety net for huge/complex counties.
        row = db.session.execute(
            text(
                """
                SELECT ST_AsGeoJSON(ST_SimplifyPreserveTopology(geom, :tol)) AS geom_json
                FROM counties
                WHERE geoid = :geoid
                LIMIT 1
                """
            ),
            {"geoid": geoid, "tol": 0.0025},
        ).mappings().first()
    except Exception:
        return None

    if not row:
        return None

    geom_json = row.get('geom_json')
    if not geom_json:
        return None

    try:
        import json
        geom = json.loads(geom_json)
    except Exception:
        geom = None

    if not geom or 'coordinates' not in geom:
        return None

    # Choose a reasonable exterior ring: pick the longest exterior ring.
    rings = []
    gtype = (geom.get('type') or '').lower()
    coords = geom.get('coordinates')

    def add_polygon(poly):
        # poly: [ [ [lng,lat], ... ] , holes...]
        if not poly or not poly[0]:
            return
        ring = poly[0]
        rings.append(ring)

    try:
        if gtype == 'polygon':
            add_polygon(coords)
        elif gtype == 'multipolygon':
            for poly in coords:
                add_polygon(poly)
    except Exception:
        rings = []

    if not rings:
        return None

    ring = max(rings, key=lambda r: len(r) if r else 0)

    # Compute bbox from the chosen exterior ring (so zoom framing matches the outline).
    try:
        lats = [float(p[1]) for p in ring if p and len(p) >= 2]
        lngs = [float(p[0]) for p in ring if p and len(p) >= 2]
        if not lats or not lngs:
            return None
        min_lat = min(lats)
        max_lat = max(lats)
        min_lng = min(lngs)
        max_lng = max(lngs)
    except Exception:
        return None

    # Downsample to keep URL length reasonable (Static Maps has URL length limits).
    # Use a higher point budget than before and auto-tighten only if the encoded
    # polyline becomes too long.
    if not ring:
        return (float(min_lat), float(min_lng), float(max_lat), float(max_lng), None)

    max_poly_chars = 4200  # conservative; leaves room for other URL params
    target_points = 420
    step = max(1, int(len(ring) / target_points))

    def build_pts(step_n: int):
        pts_local = []
        for i in range(0, len(ring), max(1, int(step_n))):
            lng, lat = ring[i]
            pts_local.append((float(lat), float(lng)))
        if pts_local and pts_local[0] != pts_local[-1]:
            pts_local.append(pts_local[0])
        return pts_local

    poly = None
    for _ in range(6):
        pts = build_pts(step)
        poly_try = _encode_polyline(pts) if len(pts) >= 3 else None
        if not poly_try:
            poly = None
            break
        if len(poly_try) <= max_poly_chars:
            poly = poly_try
            break
        # Too long: sample fewer points.
        step = int(step * 1.6) + 1

    return (float(min_lat), float(min_lng), float(max_lat), float(max_lng), poly)


def _mercator_y(lat: float) -> float:
    """Mercator projection helper for zoom calculations."""
    # Google Maps clamps usable latitude to ~85.0511.
    lat = max(min(lat, 85.05112878), -85.05112878)
    rad = math.radians(lat)
    return math.log(math.tan((rad / 2.0) + (math.pi / 4.0)))


def _zoom_for_bbox(
    min_lat: float,
    min_lng: float,
    max_lat: float,
    max_lng: float,
    *,
    width_px: int,
    height_px: int,
    padding: float = 1.03,
    max_zoom: int = 16,
) -> int:
    """Approximate a Google Maps zoom level that fits the bbox into a given pixel size.

    We use a small padding factor so the county outline fills more of the preview
    while still avoiding edge clipping.
    """
    # Longitude span (handle weird negatives; we don't expect antimeridian counties).
    lng_diff = float(max_lng) - float(min_lng)
    if lng_diff <= 0:
        lng_diff = abs(lng_diff)
    lng_diff = max(lng_diff, 1e-6)

    y_min = _mercator_y(float(min_lat))
    y_max = _mercator_y(float(max_lat))
    y_diff = max(abs(y_max - y_min), 1e-6)

    # World is 256px at zoom 0.
    zoom_lng = math.log2((float(width_px) * 360.0) / (lng_diff * 256.0 * float(padding)))
    zoom_lat = math.log2((float(height_px) * 2.0 * math.pi) / (y_diff * 256.0 * float(padding)))

    z = int(math.floor(min(zoom_lng, zoom_lat)))
    return max(0, min(z, int(max_zoom)))



def _zoom_for_bbox_pxpad(
    min_lat: float,
    min_lng: float,
    max_lat: float,
    max_lng: float,
    *,
    width_px: int,
    height_px: int,
    pad_px: int = 24,
    max_zoom: int = 18,
) -> int:
    """Approximate zoom similar to Google Maps JS `fitBounds` with padding.

    We emulate `map.fitBounds(bounds, {top/right/bottom/left: pad_px})` by shrinking
    the available viewport before computing zoom. This produces a noticeably tighter
    framing than the older multiplicative padding factor and matches the interactive
    county view more closely.
    """
    w = max(1, int(width_px) - (2 * int(pad_px)))
    h = max(1, int(height_px) - (2 * int(pad_px)))
    return _zoom_for_bbox(
        float(min_lat),
        float(min_lng),
        float(max_lat),
        float(max_lng),
        width_px=w,
        height_px=h,
        padding=1.0,
        max_zoom=int(max_zoom),
    )


def _bbox_fits_zoom_pxpad(
    min_lat: float,
    min_lng: float,
    max_lat: float,
    max_lng: float,
    *,
    zoom: int,
    width_px: int,
    height_px: int,
    pad_px: int = 24,
) -> bool:
    """Return True if the bbox fits in the viewport at the given zoom (fitBounds-style padding)."""
    w = max(1, int(width_px) - (2 * int(pad_px)))
    h = max(1, int(height_px) - (2 * int(pad_px)))

    lng_diff = float(max_lng) - float(min_lng)
    if lng_diff <= 0:
        lng_diff = abs(lng_diff)
    lng_diff = max(lng_diff, 1e-9)

    y_min = _mercator_y(float(min_lat))
    y_max = _mercator_y(float(max_lat))
    y_diff = max(abs(y_max - y_min), 1e-9)

    world_px = 256.0 * (2.0 ** float(int(zoom)))
    span_x = world_px * (lng_diff / 360.0)
    span_y = world_px * (y_diff / (2.0 * math.pi))
    return (span_x <= float(w)) and (span_y <= float(h))

# --- Static Maps polyline simplification (to avoid g.co/staticmaperror on very detailed county outlines) ---
# We keep outlines but simplify them only when a generated Static Maps URL would exceed safe length.
_SIMPLIFIED_POLY_CACHE: dict[tuple[str, int], str] = {}

def _decode_google_polyline(poly: str):
    # Decode an encoded polyline string into a list of (lat, lng) tuples.
    if not poly:
        return []
    index = 0
    lat = 0
    lng = 0
    coords = []
    length = len(poly)
    while index < length:
        shift = 0
        result = 0
        while True:
            b = ord(poly[index]) - 63
            index += 1
            result |= (b & 0x1f) << shift
            shift += 5
            if b < 0x20:
                break
        dlat = ~(result >> 1) if (result & 1) else (result >> 1)
        lat += dlat

        shift = 0
        result = 0
        while True:
            b = ord(poly[index]) - 63
            index += 1
            result |= (b & 0x1f) << shift
            shift += 5
            if b < 0x20:
                break
        dlng = ~(result >> 1) if (result & 1) else (result >> 1)
        lng += dlng

        coords.append((lat / 1e5, lng / 1e5))
    return coords

def _encode_google_polyline(coords):
    # Encode a list of (lat, lng) tuples into a polyline string.
    def _enc_value(v):
        v = ~(v << 1) if v < 0 else (v << 1)
        out = []
        while v >= 0x20:
            out.append(chr((0x20 | (v & 0x1f)) + 63))
            v >>= 5
        out.append(chr(v + 63))
        return ''.join(out)

    last_lat = 0
    last_lng = 0
    out = []
    for lat, lng in coords:
        ilat = int(round(lat * 1e5))
        ilng = int(round(lng * 1e5))
        out.append(_enc_value(ilat - last_lat))
        out.append(_enc_value(ilng - last_lng))
        last_lat, last_lng = ilat, ilng
    return ''.join(out)

def _dp_simplify(coords, eps):
    # Douglas-Peucker simplification for lat/lng coords (approx planar).
    if len(coords) <= 2:
        return coords

    lat0 = coords[0][0]
    cos0 = math.cos(math.radians(lat0)) or 1.0

    def dist_point_to_segment(p, a, b):
        (py, px) = (p[0], p[1] * cos0)
        (ay, ax) = (a[0], a[1] * cos0)
        (by, bx) = (b[0], b[1] * cos0)
        vx, vy = (bx - ax), (by - ay)
        wx, wy = (px - ax), (py - ay)
        c1 = vx * wx + vy * wy
        if c1 <= 0:
            return math.hypot(px - ax, py - ay)
        c2 = vx * vx + vy * vy
        if c2 <= c1:
            return math.hypot(px - bx, py - by)
        t = c1 / c2
        projx = ax + t * vx
        projy = ay + t * vy
        return math.hypot(px - projx, py - projy)

    keep = [False] * len(coords)
    keep[0] = True
    keep[-1] = True
    stack = [(0, len(coords) - 1)]
    while stack:
        start, end = stack.pop()
        max_dist = 0.0
        idx = None
        a = coords[start]
        b = coords[end]
        for i in range(start + 1, end):
            d = dist_point_to_segment(coords[i], a, b)
            if d > max_dist:
                max_dist = d
                idx = i
        if idx is not None and max_dist > eps:
            keep[idx] = True
            stack.append((start, idx))
            stack.append((idx, end))

    return [c for c, k in zip(coords, keep) if k]

def _simplify_polyline_to_maxlen(geoid: str, poly: str, max_len: int):
    # Return a simplified polyline string whose encoded length is <= max_len when possible.
    cache_key = (geoid, max_len)
    if cache_key in _SIMPLIFIED_POLY_CACHE:
        return _SIMPLIFIED_POLY_CACHE[cache_key]

    coords = _decode_google_polyline(poly)
    if len(coords) < 50:
        _SIMPLIFIED_POLY_CACHE[cache_key] = poly
        return poly

    eps = 0.0005
    best = poly
    for _ in range(10):
        simp = _dp_simplify(coords, eps)
        if simp and coords[0] == coords[-1] and simp[0] != simp[-1]:
            simp = list(simp) + [simp[0]]
        enc = _encode_google_polyline(simp)
        if len(enc) < len(best):
            best = enc
        if len(enc) <= max_len or len(simp) < 80:
            best = enc
            break
        eps *= 2.0

    _SIMPLIFIED_POLY_CACHE[cache_key] = best
    return best


def county_static_map_url(county_geoid: str | None, pin_lat: float | None = None, pin_lng: float | None = None, *, width: int = 640, height: int = 340, center_on_pin: bool = False, use_custom_icon: bool = True) -> str:
    """Static map URL showing the county boundary (black outline + subtle fill) and an optional red pin.

    If center_on_pin=True (and a pin is provided), the map center is set to the pin location.
    This makes it possible to place UI overlays (like the speed pill) deterministically relative to the marker.

    v292+ approach (static, premium):
      - Integer zoom with a small padding factor (consistent framing)
      - Request full-size retina images (scale=2)
      - Let the browser downscale (no fractional-size upscale hack)
    """
    try:
        from flask import current_app
        # Use ONLY the server key for Static Maps. Do not fall back to the browser key.
        key = (current_app.config.get('GOOGLE_MAPS_SERVER_KEY') or '').strip()
    except Exception:
        key = (os.environ.get('GOOGLE_MAPS_SERVER_KEY') or '').strip()

    geoid = (county_geoid or '').strip()
    if not key or not geoid:
        return ''

    info = _county_bbox_and_outline_polyline(geoid)
    if not info:
        return ''

    min_lat, min_lng, max_lat, max_lng, poly = info

    # Compute center (Mercator midpoint) + a continuous fit zoom (Web Mercator math).
    # Static Maps `zoom` is integer-only: we use the best integer zoom with a small padding factor.
    pad_px = 24
    pad_pct = 0.12  # ~12% extra span for consistent breathing room

    # BBox in Mercator space
    y_min = _mercator_y(float(min_lat))
    y_max = _mercator_y(float(max_lat))

    # Default center: Mercator midpoint (closer to Google fitBounds than simple lat/lng midpoint).
    y_center = (y_min + y_max) / 2.0
    center_lat = math.degrees(2.0 * math.atan(math.exp(y_center)) - (math.pi / 2.0))
    center_lng = (float(min_lng) + float(max_lng)) / 2.0

    # Optional: for feed pin-based tickets, re-center on the pin.
    if center_on_pin and pin_lat is not None and pin_lng is not None:
        center_lat = float(pin_lat)
        center_lng = float(pin_lng)

    # Available viewport inside the image after padding.
    avail_w = max(1, int(width) - (2 * int(pad_px)))
    avail_h = max(1, int(height) - (2 * int(pad_px)))

    lng_min = float(min_lng)
    lng_max = float(max_lng)
    lng_diff = max(abs(lng_max - lng_min), 1e-9)

    # Keep zoom consistent (based on the county bbox), even when centering on a user pin.
    # This avoids zooming out when the pin is near the edge of the county.
    y_diff = max(abs(y_max - y_min), 1e-9)

    # Apply padding percentage.
    lng_diff *= (1.0 + pad_pct)
    y_diff *= (1.0 + pad_pct)

    zoom_lng = math.log2((float(avail_w) * 360.0) / (lng_diff * 256.0))
    zoom_lat = math.log2((float(avail_h) * 2.0 * math.pi) / (y_diff * 256.0))
    z_star = min(zoom_lng, zoom_lat)
    z_star = max(0.0, min(float(z_star), 18.0))

    zoom_int = int(math.floor(z_star))

    def _public_base_url() -> str:
        """Return a public-facing base URL (scheme+host) suitable for Google fetching assets.

        Render + proxies can produce internal hosts in some request contexts. For Static Maps custom
        marker icons, Google must be able to reach the icon URL from the public internet.

        Priority:
          1) PUBLIC_BASE_URL (env/config)
          2) forwarded proto/host
          3) if we're on an onrender.com host, fall back to enforcedspeed.com
        """
        try:
            from flask import current_app
            cfg = (current_app.config.get("PUBLIC_BASE_URL") or "").strip().rstrip("/")
        except Exception:
            cfg = (os.environ.get("PUBLIC_BASE_URL") or "").strip().rstrip("/")

        if cfg and cfg.startswith("http"):
            return cfg

        proto = (request.headers.get("X-Forwarded-Proto") or request.scheme or "https").split(",")[0].strip()
        host = (request.headers.get("X-Forwarded-Host") or request.host).split(",")[0].strip()

        # If we're on Render's default hostname, prefer the canonical custom domain.
        # (Google fetching a marker icon from an internal/onrender host is unreliable.)
        if host.endswith("onrender.com") or ".onrender.com" in host:
            host = "enforcedspeed.com"

        # Google Static Maps requires icon URLs to be publicly reachable; force https.
        if proto == "http":
            proto = "https"
        return f"{proto}://{host}"

    base_url = _public_base_url()

    def _static_pin_base_url() -> str:
        """Public URL base for Static Maps marker icon assets.

        Custom marker icons are fetched by Google. Serving icon assets from the main app domain can
        be unreliable if Cloudflare security/bot mitigation challenges non-browser clients.

        Priority:
          1) STATIC_PIN_BASE_URL (env or Flask config) — e.g. https://static.enforcedspeed.com/pins
          2) fallback to app-served pins under the public base URL
        """
        try:
            from flask import current_app
            cfg = (current_app.config.get("STATIC_PIN_BASE_URL") or "").strip()
        except Exception:
            cfg = ""

        envv = (os.environ.get("STATIC_PIN_BASE_URL") or "").strip()
        val = (envv or cfg).rstrip("/")
        if val and val.startswith("http"):
            return val
        return f"{base_url}/static/img/pins"

    pin_base = _static_pin_base_url()
    icon_url = f"{pin_base}/pin_inside_deepred_static.png"

    def _qs(items: list[tuple[str, str]]) -> str:
        """Query-string builder tuned for Google Static Maps.

        Critical: do NOT encode the marker/path directive syntax (e.g. 'icon:' / 'enc:'), and do NOT
        turn 'icon:' into 'icon%3A'. We preserve ":/,|%" so Google parses marker/path directives.
        """
        return "&".join(
            f"{urllib.parse.quote(k, safe='')}={urllib.parse.quote(v, safe=':/,|%')}"
            for k, v in items
        )

    items: list[tuple[str, str]] = [
        ('size', f"{int(width)}x{int(height)}"),
        ('scale', '2'),
        ('maptype', 'roadmap'),
        ('center', f"{center_lat:.6f},{center_lng:.6f}"),
        ('zoom', str(int(zoom_int))),
        # De-clutter slightly so roads/labels stand out.
        ('style', 'feature:poi|visibility:off'),
        ('style', 'feature:transit|visibility:off'),
        ('key', key),
    ]

    # County outline (black + subtle fill).
    # IMPORTANT: Outline is required. However, Google Static Maps can return g.co/staticmaperror
    # if the request URL becomes too large (county polylines can be huge). To prevent this while
    # still keeping the outline, we simplify the encoded polyline when it's above a safe budget.
    poly_use = poly
    if poly_use:
        # Budget for the encoded polyline itself (not the entire URL). Keeping this conservative
        # preserves room for styles + markers while avoiding Static Maps URL limits.
        POLY_MAXLEN = 3200
        if len(poly_use) > POLY_MAXLEN:
            poly_use = _simplify_polyline_to_maxlen(geoid, poly_use, POLY_MAXLEN)
        items.append(('path', f"fillcolor:0x00000017|color:0x000000ff|weight:1|enc:{poly_use}"))

    # Optional pin
    if pin_lat is not None and pin_lng is not None:
        center = f"{pin_lat:.6f},{pin_lng:.6f}"
        if use_custom_icon:
            # Custom icon markers can be rejected if Google cannot fetch the icon URL quickly/reliably.
            encoded_icon = urllib.parse.quote(icon_url, safe="")
            items.append(('markers', f"icon:{encoded_icon}|{center}"))
        else:
            # Safe fallback (no custom icon) to avoid g.co/staticmaperror in proxy flows.
            items.append(('markers', center))

    return 'https://maps.googleapis.com/maps/api/staticmap?' + _qs(items)

def us_static_map_url(*, width: int = 640, height: int = 640) -> str:
    """Static map URL of the United States (default preview before a county is selected)."""
    try:
        from flask import current_app
        # Use ONLY the server key for Static Maps. Do not fall back to the browser key.
        key = (current_app.config.get('GOOGLE_MAPS_SERVER_KEY') or '').strip()
    except Exception:
        key = (os.environ.get('GOOGLE_MAPS_SERVER_KEY') or '').strip()

    if not key:
        return ''

    w = max(120, min(1024, int(width)))
    h = max(120, min(1024, int(height)))

    params = {
        'size': f"{w}x{h}",
        'scale': '2',
        'maptype': 'roadmap',
        # Rough visual center of the continental US
        'center': '39.8283,-98.5795',
        'zoom': '4',
        # Keep roads/labels clear; just reduce non-essential clutter
        'style': [
            'feature:poi|visibility:off',
            'feature:transit|visibility:off',
        ],
        'key': key,
    }
    return 'https://maps.googleapis.com/maps/api/staticmap?' + urllib.parse.urlencode(params, doseq=True)


def column_exists(table_name: str, column_name: str) -> bool:
    engine = db.engine
    dialect = engine.dialect.name

    if dialect == "sqlite":
        rows = db.session.execute(text(f"PRAGMA table_info({table_name})")).fetchall()
        return any(r[1] == column_name for r in rows)

    q = text(
        """
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = :t
          AND column_name = :c
        LIMIT 1
        """
    )
    return db.session.execute(q, {"t": table_name, "c": column_name}).first() is not None


def ensure_schema():
    """Best-effort schema bootstrap for local/dev DBs.

    IMPORTANT:
    - We do NOT drop or recreate tables here.
    - We only add missing columns/indexes that the current models expect.
    - This keeps older local DBs working after model evolution, without Alembic.
    """
    try:
        from sqlalchemy import inspect as sa_inspect
        from sqlalchemy import text as sa_text
        from sqlalchemy.sql.sqltypes import Integer, String, Text, Float, Boolean, DateTime
    except Exception as e:  # pragma: no cover
        print(f"[SCHEMA INIT ERROR] unable to import sqlalchemy helpers: {e}")
        return

    def _sql_type(coltype) -> str:
        # Keep this conservative and Postgres-friendly.
        if isinstance(coltype, Integer):
            return "INTEGER"
        if isinstance(coltype, Float):
            return "DOUBLE PRECISION"
        if isinstance(coltype, Boolean):
            return "BOOLEAN"
        if isinstance(coltype, DateTime):
            return "TIMESTAMP"
        if isinstance(coltype, Text):
            return "TEXT"
        if isinstance(coltype, String):
            # String(length=None) => use TEXT to avoid arbitrary limits
            return f"VARCHAR({coltype.length})" if getattr(coltype, "length", None) else "TEXT"
        # Fallback
        return "TEXT"

    # Columns where we want a strong default to avoid NULL semantics breaking queries.
    _overrides = {
        # speed_reports
        "is_deleted": "BOOLEAN NOT NULL DEFAULT FALSE",
        "verify_attempts": "INTEGER NOT NULL DEFAULT 0",
        "verification_status": "TEXT NOT NULL DEFAULT 'pending'",
        "ocr_status": "TEXT NOT NULL DEFAULT 'pending'",
        # likes/comments soft defaults can be added here if needed later
    }

    def _ensure_table_columns(model):
        table = model.__table__
        table_name = table.name

        insp = sa_inspect(db.engine)
        if not insp.has_table(table_name):
            # Create missing table(s)
            db.create_all()
            return

        existing = {c["name"] for c in insp.get_columns(table_name)}
        missing = []
        for col in table.columns:
            if col.name not in existing:
                missing.append(col)

        if not missing:
            return

        for col in missing:
            # Quote column names that could be reserved (route_class etc are fine but safe anyway).
            col_name = col.name
            col_sql = _overrides.get(col_name, _sql_type(col.type))

            # Allow NULL by default unless we have an override specifying NOT NULL.
            # If override includes NOT NULL, it *must* include DEFAULT so existing rows pass.
            if "NOT NULL" not in col_sql:
                col_sql = f"{col_sql} NULL"

            ddl = f'ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS "{col_name}" {col_sql};'
            db.session.execute(sa_text(ddl))

        db.session.commit()
        print(f"[SCHEMA] Added missing columns to {table_name}: {[c.name for c in missing]}")
    try:
        # Ensure all base tables exist first
        db.create_all()

        # Apply additive patches for evolving models
        _ensure_table_columns(User)
        _ensure_table_columns(SpeedReport)

        # Optional tables (these may not exist in some older zips)
        try:
            _ensure_table_columns(Like)
        except Exception:
            pass
        try:
            _ensure_table_columns(Comment)
        except Exception:
            pass

        # --- Data backfills / defaults (safe, additive) ---
        # Some older local DBs may have the column but NULL values, which can break filters/queries.
        try:
            db.session.execute(sa_text("UPDATE speed_reports SET is_deleted = FALSE WHERE is_deleted IS NULL;"))
            db.session.execute(sa_text("UPDATE speed_reports SET verify_attempts = 0 WHERE verify_attempts IS NULL;"))
            db.session.execute(sa_text("UPDATE speed_reports SET ocr_status = 'pending' WHERE ocr_status IS NULL;"))
            db.session.execute(sa_text("UPDATE speed_reports SET verification_status = 'pending' WHERE verification_status IS NULL;"))
            db.session.execute(sa_text(
                """
                UPDATE speed_reports
                SET overage = (ticketed_speed - posted_speed)
                WHERE overage IS NULL
                  AND ticketed_speed IS NOT NULL
                  AND posted_speed IS NOT NULL;
                """
            ))
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            print(f"[SCHEMA] warning: could not apply value backfills: {e}")

        # --- Enforce critical NOT NULL defaults (best-effort) ---
        # Some DBs may have NOT NULL constraints without DEFAULTs, causing inserts to fail
        # even when the app intends these fields to be 0/'pending'.
        try:
            db.session.execute(sa_text("ALTER TABLE speed_reports ALTER COLUMN verify_attempts SET DEFAULT 0;"))
            db.session.execute(sa_text("UPDATE speed_reports SET verify_attempts = 0 WHERE verify_attempts IS NULL;"))
            db.session.execute(sa_text("ALTER TABLE speed_reports ALTER COLUMN verify_attempts SET NOT NULL;"))

            db.session.execute(sa_text("ALTER TABLE speed_reports ALTER COLUMN is_deleted SET DEFAULT FALSE;"))
            db.session.execute(sa_text("UPDATE speed_reports SET is_deleted = FALSE WHERE is_deleted IS NULL;"))
            db.session.execute(sa_text("ALTER TABLE speed_reports ALTER COLUMN is_deleted SET NOT NULL;"))

            db.session.execute(sa_text("ALTER TABLE speed_reports ALTER COLUMN ocr_status SET DEFAULT 'pending';"))
            db.session.execute(sa_text("UPDATE speed_reports SET ocr_status = 'pending' WHERE ocr_status IS NULL;"))
            db.session.execute(sa_text("ALTER TABLE speed_reports ALTER COLUMN ocr_status SET NOT NULL;"))

            db.session.execute(sa_text("ALTER TABLE speed_reports ALTER COLUMN verification_status SET DEFAULT 'pending';"))
            db.session.execute(sa_text("UPDATE speed_reports SET verification_status = 'pending' WHERE verification_status IS NULL;"))
            db.session.execute(sa_text("ALTER TABLE speed_reports ALTER COLUMN verification_status SET NOT NULL;"))

            db.session.commit()
        except Exception as e:
            db.session.rollback()
            print(f"[SCHEMA] warning: could not enforce defaults/constraints: {e}")

        # Backfill road_key in Python (matches normalize_road() behavior).
        try:
            missing = (
                db.session.query(SpeedReport.id, SpeedReport.road_name, SpeedReport.state)
                .filter((SpeedReport.road_key.is_(None)) | (SpeedReport.road_key == ""))
                .limit(50000)
                .all()
            )
            if missing:
                updates = []
                for rid, road_name, state_val in missing:
                    rk = normalize_road(road_name or "", state_val or "")
                    updates.append({"id": rid, "road_key": (rk or None)})
                db.session.bulk_update_mappings(SpeedReport, updates)
                db.session.commit()
                print(f"[SCHEMA] Backfilled road_key for {len(updates)} speed_reports rows")
        except Exception as e:
            db.session.rollback()
            print(f"[SCHEMA] warning: could not backfill road_key: {e}")

        # --- Indexes (safe, additive) ---
        try:
            db.session.execute(sa_text("CREATE INDEX IF NOT EXISTS ix_speed_reports_state_road_key ON speed_reports(state, road_key);"))
            db.session.execute(sa_text("CREATE INDEX IF NOT EXISTS ix_speed_reports_state_posted_speed ON speed_reports(state, posted_speed);"))
            db.session.execute(sa_text("CREATE INDEX IF NOT EXISTS ix_speed_reports_state_ticketed_speed ON speed_reports(state, ticketed_speed);"))
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            print(f"[SCHEMA] warning: could not create indexes: {e}")

    except Exception as e:
        db.session.rollback()
        print(f"[SCHEMA INIT ERROR] failed to ensure schema: {e}")
def create_app() -> Flask:
    app = Flask(__name__)
    # Ensure correct scheme/host when behind Render's proxy (affects external URLs).
    try:
        from werkzeug.middleware.proxy_fix import ProxyFix
        app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
        app.config["PREFERRED_URL_SCHEME"] = "https"
    except Exception:
        pass
    app.config.from_object(Config)
    db.init_app(app)

    login_manager = LoginManager()
    login_manager.login_view = "login"
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(user_id: str):
        # SQLAlchemy 2.x: Query.get() is legacy; use Session.get()
        try:
            return db.session.get(User, int(user_id))
        except Exception:
            return None

    @app.context_processor
    def inject_helpers():
        return {"format_road_bucket": format_road_bucket,
                "static_map_url": static_map_url,
                "county_static_map_url": county_static_map_url}


    # --- Health check (Render) ---
    # Keep this endpoint fast and DB-independent so deploys never hang on health checks.
    @app.get("/healthz")
    def healthz():
        return "ok", 200


    # --- County GIS (PostGIS) ---
    # ZIP 1 foundation: create table + indexes. Import happens via scripts (see scripts/import_counties_*).
    def ensure_counties_schema() -> None:
        """Ensure the counties table supports both autocomplete and (optional) geometry features.

        Why this exists:
          - The import script creates/uses a `center` point column.
          - Earlier app code used a `centroid` column.
          - We want the app to work with either, without re-importing.

        Columns we support (all optional except geoid/stusps/name_norm for autocomplete):
          - geom     : MULTIPOLYGON (SRID 4326) for boundary outline + inside-county validation
          - center   : POINT (SRID 4326) stable point-on-surface used by our import script
          - centroid : POINT (SRID 4326) legacy name used by older app code
        """
        try:
            with db.engine.begin() as conn:
                conn.exec_driver_sql('CREATE EXTENSION IF NOT EXISTS postgis;')

                # Base table (kept permissive so existing installs don't break).
                conn.exec_driver_sql('''
CREATE TABLE IF NOT EXISTS counties (
    geoid TEXT PRIMARY KEY,
    name TEXT,
    namelsad TEXT,
    statefp TEXT,
    stusps TEXT,
    state_name TEXT,
    name_norm TEXT,
    geom geometry(MULTIPOLYGON, 4326),
    center geometry(POINT, 4326),
    centroid geometry(POINT, 4326)
);
''')

                # Additive patches (table may already exist with a slightly different schema).
                conn.exec_driver_sql('ALTER TABLE counties ADD COLUMN IF NOT EXISTS namelsad TEXT;')
                conn.exec_driver_sql('ALTER TABLE counties ADD COLUMN IF NOT EXISTS statefp TEXT;')
                conn.exec_driver_sql('ALTER TABLE counties ADD COLUMN IF NOT EXISTS state_name TEXT;')
                conn.exec_driver_sql('ALTER TABLE counties ADD COLUMN IF NOT EXISTS name_norm TEXT;')

                conn.exec_driver_sql('ALTER TABLE counties ADD COLUMN IF NOT EXISTS geom geometry(MULTIPOLYGON, 4326);')
                conn.exec_driver_sql('ALTER TABLE counties ADD COLUMN IF NOT EXISTS center geometry(POINT, 4326);')
                conn.exec_driver_sql('ALTER TABLE counties ADD COLUMN IF NOT EXISTS centroid geometry(POINT, 4326);')

                # Keep center/centroid in sync when one exists and the other is NULL.
                conn.exec_driver_sql('UPDATE counties SET centroid = center WHERE centroid IS NULL AND center IS NOT NULL;')
                conn.exec_driver_sql('UPDATE counties SET center = centroid WHERE center IS NULL AND centroid IS NOT NULL;')

                # Indexes (safe, additive).
                conn.exec_driver_sql('CREATE INDEX IF NOT EXISTS ix_counties_stusps ON counties(stusps);')
                conn.exec_driver_sql('CREATE INDEX IF NOT EXISTS ix_counties_name_norm ON counties(name_norm);')
                conn.exec_driver_sql('CREATE INDEX IF NOT EXISTS ix_counties_geom_gist ON counties USING GIST(geom);')
                conn.exec_driver_sql('CREATE INDEX IF NOT EXISTS ix_counties_center_gist ON counties USING GIST(center);')
                conn.exec_driver_sql('CREATE INDEX IF NOT EXISTS ix_counties_centroid_gist ON counties USING GIST(centroid);')
        except Exception as e:
            try:
                print(f'[COUNTY INIT ERROR] failed to ensure counties schema: {e}')
            except Exception:
                pass

    
    # --- State GIS (derived from counties, cached in PostGIS) ---
    # We are county-based, but we want a state-level highlight view that behaves like county:
    # outline + subtle fill + fit-to-frame zoom.
    #
    # IMPORTANT: We do NOT union counties on every request. We build cached state geometries once
    # into a `states` table and then read them cheaply at runtime.
    def ensure_states_schema() -> None:
        try:
            with db.engine.begin() as conn:
                conn.exec_driver_sql('CREATE EXTENSION IF NOT EXISTS postgis;')

                conn.exec_driver_sql('''
CREATE TABLE IF NOT EXISTS states (
    stusps TEXT PRIMARY KEY,
    state_name TEXT,
    geom geometry(MULTIPOLYGON, 4326)
);
''')
                conn.exec_driver_sql('ALTER TABLE states ADD COLUMN IF NOT EXISTS state_name TEXT;')
                conn.exec_driver_sql('ALTER TABLE states ADD COLUMN IF NOT EXISTS geom geometry(MULTIPOLYGON, 4326);')
                conn.exec_driver_sql('CREATE INDEX IF NOT EXISTS ix_states_geom_gist ON states USING GIST(geom);')

                # Build once (only when empty). Protect with an advisory lock to avoid races
                # when both web + worker boot at the same time.
                rows = conn.exec_driver_sql('SELECT COUNT(1) FROM states;').fetchone()
                n = int(rows[0] or 0) if rows else 0
                if n > 0:
                    return

                # Try to acquire lock quickly; if we can't, another process is building.
                lock = conn.exec_driver_sql('SELECT pg_try_advisory_lock(987654321);').fetchone()
                got_lock = bool(lock and lock[0])
                if not got_lock:
                    return

                try:
                    # Re-check now that we hold the lock.
                    rows2 = conn.exec_driver_sql('SELECT COUNT(1) FROM states;').fetchone()
                    n2 = int(rows2[0] or 0) if rows2 else 0
                    if n2 > 0:
                        return

                    # Populate cached state geometries from counties (one-time).
                    # We take the union of county MULTIPOLYGONs per state and coerce to MULTIPOLYGON.
                    conn.exec_driver_sql('''
INSERT INTO states (stusps, state_name, geom)
SELECT
    UPPER(TRIM(stusps)) AS stusps,
    MAX(NULLIF(TRIM(state_name), '')) AS state_name,
    ST_Multi(ST_Union(geom)) AS geom
FROM counties
WHERE stusps IS NOT NULL AND TRIM(stusps) <> '' AND geom IS NOT NULL
GROUP BY UPPER(TRIM(stusps));
''')
                finally:
                    try:
                        conn.exec_driver_sql('SELECT pg_advisory_unlock(987654321);')
                    except Exception:
                        pass
        except Exception as e:
            try:
                print(f'[STATE INIT ERROR] failed to ensure states schema: {e}')
            except Exception:
                pass


# Try to ensure counties schema at boot (non-fatal). (non-fatal).


    # --- Mobile/API auth (JWT) ---
    def _jwt_secret() -> str:
        secret = (app.config.get("SECRET_KEY") or "").strip()
        # Secret must exist for JWT auth to work; fall back to a fixed string only for dev.
        return secret or "dev-insecure-secret"

    def _jwt_encode(user: User) -> str:
        now = datetime.now(timezone.utc)
        payload = {
            "sub": str(user.id),
            "username": user.username,
            "email": user.email,
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(days=30)).timestamp()),
        }
        token = jwt.encode(payload, _jwt_secret(), algorithm="HS256")
        # PyJWT may return bytes in older versions; normalize to str.
        return token.decode("utf-8") if isinstance(token, (bytes, bytearray)) else str(token)

    def _jwt_decode(token: str) -> dict | None:
        if not token:
            return None
        try:
            return jwt.decode(token, _jwt_secret(), algorithms=["HS256"], leeway=30)
        except ExpiredSignatureError:
            return None
        except InvalidTokenError:
            return None
        except Exception:
            return None

    def _api_user_from_request() -> User | None:
        """Return the authenticated user for API routes.

        Supports BOTH:
          - Web session auth (Flask-Login current_user)
          - Mobile bearer token auth (Authorization: Bearer <jwt>)
        """
        try:
            if getattr(current_user, "is_authenticated", False):
                return current_user
        except Exception:
            pass

        auth = (request.headers.get("Authorization") or "").strip()
        if not auth:
            return None
        if not auth.lower().startswith("bearer "):
            return None
        token = auth.split(" ", 1)[1].strip()
        data = _jwt_decode(token)
        if not data:
            return None
        sub = data.get("sub")
        try:
            uid = int(sub)
        except Exception:
            return None
        return User.query.get(uid)

    def api_login_required(fn):
        @wraps(fn)
        def _wrap(*args, **kwargs):
            u = _api_user_from_request()
            if not u:
                return jsonify({"error": "auth_required"}), 401
            # attach for handlers if desired
            request._api_user = u  # type: ignore
            return fn(*args, **kwargs)
        return _wrap


    def get_google_maps_static_maps_key() -> str:
        """Return a key suitable for Google Static Maps requests.

        Priority:
          1) GOOGLE_MAPS_SERVER_KEY (server key; preferred for server-side proxying)
          2) GOOGLE_MAPS_API_KEY (browser key; used by the website's direct <img> loads)
        """
        key = (app.config.get("GOOGLE_MAPS_SERVER_KEY") or os.environ.get("GOOGLE_MAPS_SERVER_KEY") or "").strip()
        if key:
            return key
        key = (app.config.get("GOOGLE_MAPS_API_KEY") or os.environ.get("GOOGLE_MAPS_API_KEY") or "").strip()
        return key

    def snap_to_nearest_road(lat: float, lng: float):
        """Snap a single point to the nearest road using Google Roads API.

        Requires Config.GOOGLE_MAPS_SERVER_KEY. Returns (snapped_lat, snapped_lng) or (None, None).
        """
        key = app.config.get("GOOGLE_MAPS_SERVER_KEY") or ""
        if not key:
            return None, None

        try:
            q = urllib.parse.urlencode({"points": f"{lat},{lng}", "key": key})
            url = f"https://roads.googleapis.com/v1/nearestRoads?{q}"
            req = urllib.request.Request(url, headers={"User-Agent": "EnforcedSpeed/1.0"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                payload = resp.read().decode("utf-8")
            data = json.loads(payload)
            pts = data.get("snappedPoints") or []
            if not pts:
                return None, None
            loc = (pts[0] or {}).get("location") or {}
            slat = loc.get("latitude")
            slng = loc.get("longitude")
            if slat is None or slng is None:
                return None, None
            return float(slat), float(slng)
        except Exception:
            return None, None


    def nearest_roads_multi(points: list[tuple[float, float]]):
        """Batch Roads API nearestRoads call for multiple points.

        Returns a list of snappedPoints with `originalIndex` preserved.
        """
        try:
            if not points:
                return []

            # Roads API supports multiple points as a '|' separated list.
            # Keep this bounded to avoid URL length/timeouts.
            pts = [(float(a), float(b)) for a, b in points[:100]]

            pts_str = "|".join([f"{a:.6f},{b:.6f}" for a, b in pts])
            q = urllib.parse.urlencode({"points": pts_str, "key": app.config.get("GOOGLE_MAPS_SERVER_KEY") or ""})
            url = f"https://roads.googleapis.com/v1/nearestRoads?{q}"
            req = urllib.request.Request(url, headers={"User-Agent": "EnforcedSpeed/1.0"})
            with urllib.request.urlopen(req, timeout=4) as resp:
                payload = resp.read().decode("utf-8")
            data = json.loads(payload)
            return data.get("snappedPoints") or []
        except Exception:
            return []


    # --- Highway-biased snapping (optional) ---
    _HIGHWAY_PENALTY_WORDS = (
        # Things that often show up on frontage/side roads or interchanges.
        'ramp', 'service', 'frontage', 'access', 'collector', 'local', 'drive', 'dr', 'loop', 'spur', 'bypass',
        'parkway', 'connector', 'circle'
    )

    def _offset_latlng_m(lat: float, lng: float, north_m: float, east_m: float):
        """Return a lat/lng offset by meters (rough)."""
        try:
            dlat = north_m / 111320.0
            dlng = east_m / (111320.0 * math.cos(math.radians(lat)) + 1e-9)
            return lat + dlat, lng + dlng
        except Exception:
            return lat, lng

    def _score_highway_candidate(road_label: str) -> int:
        s = (road_label or '').strip().lower()
        if not s:
            return 0

        score = 0

        # Strongest: Interstates
        if s.startswith('interstate ') or ' interstate ' in s:
            score += 160
        if re.search(r'(?<!\w)i\s*-?\s*\d{1,3}(?!\w)', s):
            score += 140

        # US routes
        if 'us route' in s or re.search(r'(?<!\w)us\s*-?\s*\d{1,4}(?!\w)', s):
            score += 110

        # State routes often appear like "nc-33", "wa 520", etc.
        if 'state route' in s or re.search(r'(?<!\w)[a-z]{2}\s*-?\s*\d{1,4}(?!\w)', s):
            score += 85

        # Generic highway keywords / route markers
        if 'highway' in s or re.search(r'(?<!\w)hwy(?!\w)', s):
            score += 70
        if 'route' in s:
            score += 25

        # Route numbers alone (fallback)
        if re.search(r'(?<!\w)\d{1,4}(?!\w)', s) and (
            'rd' in s or 'road' in s or 'route' in s or 'hwy' in s or '-' in s
        ):
            score += 10

        # Penalize common side-road artifacts.
        for w in _HIGHWAY_PENALTY_WORDS:
            if w in s:
                score -= 35

        # Heavy penalty for obvious local streets if we have no route-ish markers
        if score < 50 and not re.search(r'(?<!\w)(i\s*-?\s*\d|us\s*-?\s*\d|[a-z]{2}\s*-?\s*\d|highway|hwy|route)(?!\w)', s):
            score -= 40

        return score

    def _route_kind(road_label: str) -> str:
        """Classify a road label into: interstate | numbered | local."""
        s = (road_label or '').strip().lower()
        if not s:
            return 'local'

        # Interstates
        if s.startswith('interstate ') or ' interstate ' in s:
            return 'interstate'
        if re.search(r'(?<!\w)i\s*-?\s*\d{1,3}(?!\w)', s):
            return 'interstate'

        # Helper: avoid treating "1st/2nd/3rd" street names as numbered routes.
        if re.search(r'(?<!\w)\d{1,2}(st|nd|rd|th)(?!\w)', s):
            return 'local'

        # Numbered routes (US routes, state routes, highways/parkways with numbers)
        if 'us route' in s or 'u.s. route' in s:
            return 'numbered'
        if re.search(r'(?<!\w)us\s*-?\s*\d{1,4}(?!\w)', s):
            return 'numbered'
        if 'state route' in s or re.search(r'(?<!\w)[a-z]{2}\s*-?\s*\d{1,4}(?!\w)', s):
            return 'numbered'
        if ('highway' in s or re.search(r'(?<!\w)hwy(?!\w)', s) or 'route' in s or 'parkway' in s) and re.search(r'(?<!\w)\d{1,4}(?!\w)', s):
            return 'numbered'

        return 'local'

    def _score_route_class_candidate(road_label: str, route_class: str, rstage_m: int, max_r_m: int) -> int:
        """Score a candidate *within* a selected route_class.

        route_class: interstate | numbered | local
        Strict: if the candidate doesn't match the selected class (or is excluded), return a very negative score.
        """
        rc = (route_class or '').strip().lower()
        kind = _route_kind(road_label)
        if rc == 'interstate':
            if kind != 'interstate':
                return -10**9
            base = _score_highway_candidate(road_label)
        elif rc in ('numbered', 'us_route', 'other'):
            # Back-compat: accept older values too.
            if kind != 'numbered':
                return -10**9
            base = _score_highway_candidate(road_label)
        elif rc == 'local':
            # Explicitly exclude Interstate + numbered routes.
            if kind in ('interstate', 'numbered'):
                return -10**9
            base = 0
            s = (road_label or '').strip().lower()
            # Lightly penalize obvious ramps/frontage for local too.
            for w in _HIGHWAY_PENALTY_WORDS:
                if w in s:
                    base -= 15
        else:
            # Unknown selection: don't accept anything.
            return -10**9

        # Prefer closer candidates within the chosen class.
        try:
            rstage = int(rstage_m)
        except Exception:
            rstage = max_r_m
        dist_bonus = int(max(0, int(max_r_m) - rstage) / 50)
        return int(base) + dist_bonus

    def _snap_and_get_road(lat: float, lng: float):
        slat, slng = snap_to_nearest_road(lat, lng)
        if slat is None or slng is None:
            return None
        road = _reverse_geocode_route(float(slat), float(slng))
        if not road:
            return None
        return {
            'snapped_lat': float(slat),
            'snapped_lng': float(slng),
            'road_name': road,
        }

    def _confirm_location_route_class(raw_lat: float, raw_lng: float, route_class: str, max_radius_m: int = 2500):
        """Highway-biased snapping.

        Strategy:
        - Generate a candidate cloud around the pin.
        - Run ONE Roads nearestRoads call for all candidates.
        - Reverse-geocode a bounded set of unique snapped points.
        - Pick the best highway-like match by score.

        When route_class is:
        - interstate: only accept Interstates
        - numbered: only accept numbered routes/highways (US routes, state routes, parkways with numbers)
        - local: reject interstate + numbered and choose a named street/road

        Returns None when no match is found within the selected class.
        """
        try:
            max_r = int(max_radius_m)
        except Exception:
            max_r = 2500
        max_r = max(200, min(6000, max_r))

        # Radii stages: close-first, then broader.
        radii = sorted(set([0, 150, 500, 1500, max_r]))

        dirs = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1),
        ]

        candidates = [(raw_lat, raw_lng)]
        meta = [0]
        for r in radii:
            if r == 0:
                continue
            for dn, de in dirs:
                scale = (1 / math.sqrt(2)) if (dn != 0 and de != 0) else 1.0
                c_lat, c_lng = _offset_latlng_m(raw_lat, raw_lng, dn * r * scale, de * r * scale)
                candidates.append((c_lat, c_lng))
                meta.append(r)

        snapped = nearest_roads_multi(candidates)
        if not snapped:
            return None

        # original_index -> (slat, slng, radius_stage)
        idx_to: dict[int, tuple[float, float, int]] = {}
        for sp in snapped:
            oi = sp.get('originalIndex')
            loc = (sp.get('location') or {}) if isinstance(sp, dict) else {}
            if oi is None:
                continue
            try:
                oi_i = int(oi)
            except Exception:
                continue
            if oi_i < 0 or oi_i >= len(candidates):
                continue
            slat = loc.get('latitude')
            slng = loc.get('longitude')
            if slat is None or slng is None:
                continue
            idx_to[oi_i] = (float(slat), float(slng), int(meta[oi_i]))

        # Deduplicate snapped points (rounded), keep smallest radius stage
        uniq: dict[tuple[float, float], dict[str, float | int]] = {}
        for (slat, slng, rstage) in idx_to.values():
            key = (round(slat, 5), round(slng, 5))
            cur = uniq.get(key)
            if cur is None or rstage < int(cur['rstage']):
                uniq[key] = {'slat': slat, 'slng': slng, 'rstage': rstage}

        stages_present = sorted({int(u['rstage']) for u in uniq.values()})
        stage_order = [s for s in radii if s in stages_present]
        for s in stages_present:
            if s not in stage_order:
                stage_order.append(s)

        best = None
        best_score = -10**9

        rc = (route_class or '').strip().lower()
        if rc == 'interstate':
            ACCEPT_THRESHOLD = 120
        elif rc in ('numbered', 'us_route', 'other'):
            # Back-compat: accept older values too.
            ACCEPT_THRESHOLD = 90
        elif rc == 'local':
            ACCEPT_THRESHOLD = 10
        else:
            ACCEPT_THRESHOLD = 999999
        remaining_budget = 18  # reverse-geocode budget

        for rstage in stage_order:
            stage_pts = [u for u in uniq.values() if int(u['rstage']) == rstage]
            for u in stage_pts:
                if remaining_budget <= 0:
                    break
                road = _reverse_geocode_route(float(u['slat']), float(u['slng']))
                remaining_budget -= 1
                if not road:
                    continue
                score = _score_route_class_candidate(road, rc, int(u['rstage']), max_r)
                if score > best_score:
                    best_score = score
                    best = {
                        'snapped_lat': float(u['slat']),
                        'snapped_lng': float(u['slng']),
                        'road_name': road,
                    }

                # If we hit a very strong match, stop early.
                if score >= 180:
                    break
            if best_score >= 120:
                break

        if not best or best_score < ACCEPT_THRESHOLD:
            return None

        return best

    # Backward-compatible wrapper for older clients that still send prefer_highway Yes/No.
    def _confirm_location_prefer_highway(raw_lat: float, raw_lng: float, max_radius_m: int = 2500):
        # Try major route types first; do NOT fall back to local streets.
        return _confirm_location_route_class(raw_lat, raw_lng, route_class='interstate', max_radius_m=max_radius_m) or \
               _confirm_location_route_class(raw_lat, raw_lng, route_class='us_route', max_radius_m=max_radius_m)

    with app.app_context():
        try:
            # Single, centralized schema bootstrap (additive, no destructive changes).
            ensure_schema()
        except Exception as e:
            # Don't start a server that can't reach the database; fail fast with a clear error.
            print(f"[DB INIT ERROR] {e}")
            db.session.rollback()
            raise

    @app.route("/register", methods=["GET", "POST"])
    def register():
        if current_user.is_authenticated:
            return redirect(_safe_next_url('home'))

        form = RegisterForm()
        if form.validate_on_submit():
            email = form.email.data.strip().lower()
            # Preserve casing as-entered, but treat usernames as case-insensitive for uniqueness.
            username = form.username.data.strip()

            # Birthday (validated in form: must be 18+)
            try:
                import datetime as _dt
                dob = _dt.date(int(form.birth_year.data), int(form.birth_month.data), int(form.birth_day.data))
            except Exception:
                flash("Please select a valid birth date.", "error")
                return render_template("register.html", form=form)
            if User.query.filter_by(email=email).first():
                flash("That email is already registered. Please log in.", "error")
                return redirect(url_for("login", next=request.args.get('next')))

            if User.query.filter(func.lower(User.username) == username.lower()).first():
                flash("That username is taken. Try another.", "error")
                return render_template("register.html", form=form)
            user = User(email=email, username=username, birthdate=dob)
            user.set_password(form.password.data)
            db.session.add(user)
            db.session.commit()

            login_user(user)
            return redirect(_safe_next_url('profile', username=user.username))

        return render_template("register.html", form=form)

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if current_user.is_authenticated:
            return redirect(_safe_next_url('home'))

        form = LoginForm()
        if form.validate_on_submit():
            identifier = form.email.data.strip()
            ident_lc = identifier.lower()

            # Allow login via username OR email (username first, then email).
            user = User.query.filter(func.lower(User.username) == ident_lc).first()
            if not user:
                user = User.query.filter_by(email=ident_lc).first()

            if not user or not user.check_password(form.password.data):
                flash("Invalid username/email or password.", "error")
                return render_template("login.html", form=form)

            login_user(user)
            return redirect(_safe_next_url('home'))

        return render_template("login.html", form=form)

    @app.get("/logout")
    def logout():
        logout_user()
        return redirect(url_for("home"))


    def _password_reset_serializer() -> URLSafeTimedSerializer:
        return URLSafeTimedSerializer(app.config['SECRET_KEY'])

    def _send_password_reset_email(user: User) -> None:
        api_key = os.environ.get('SENDGRID_API_KEY', '').strip()
        from_email = os.environ.get('SENDGRID_FROM_EMAIL', 'noreply@enforcedspeed.com').strip()
        if not api_key:
            # No API key configured; skip sending (do not error to user).
            return

        s = _password_reset_serializer()
        token = s.dumps({'uid': user.id}, salt='password-reset')
        reset_url = url_for('reset_password', token=token, _external=True)

        subject = 'EnforcedSpeed password reset'
        content = (
            "You requested a password reset for your EnforcedSpeed account.\n\n"
            f"Reset your password here (link expires in 60 minutes):\n{reset_url}\n\n"
            "If you did not request this, you can ignore this email."
        )

        message = Mail(from_email=from_email, to_emails=user.email, subject=subject, plain_text_content=content)
        try:
            SendGridAPIClient(api_key).send(message)
        except Exception:
            # Intentionally swallow errors to avoid breaking UX / leaking info.
            return

    def _can_issue_reset_email(user: User) -> bool:
        """Max 5 reset emails per hour per account."""
        now = datetime.utcnow()
        window = timedelta(hours=1)

        if not getattr(user, 'reset_req_window_start', None) or not getattr(user, 'reset_req_count', None):
            user.reset_req_window_start = now
            user.reset_req_count = 1
            db.session.commit()
            return True

        start = user.reset_req_window_start
        count = int(user.reset_req_count or 0)

        if start is None or (now - start) >= window:
            user.reset_req_window_start = now
            user.reset_req_count = 1
            db.session.commit()
            return True

        if count >= 5:
            return False

        user.reset_req_count = count + 1
        db.session.commit()
        return True

    @app.route('/change-password', methods=['GET', 'POST'])
    @login_required
    def change_password():
        form = ChangePasswordForm()
        if form.validate_on_submit():
            if not current_user.check_password(form.current_password.data):
                flash('Current password is incorrect.', 'error')
                return render_template('change_password.html', form=form)

            current_user.set_password(form.new_password.data)
            db.session.commit()
            flash('Password updated.', 'success')
            return redirect(url_for('profile', username=current_user.username))

        return render_template('change_password.html', form=form)

    @app.route('/forgot-password', methods=['GET', 'POST'])
    def forgot_password():
        if current_user.is_authenticated:
            return redirect(url_for('home'))

        form = ForgotPasswordForm()
        if form.validate_on_submit():
            identifier = form.identifier.data.strip()
            ident_lc = identifier.lower()

            user = User.query.filter(func.lower(User.username) == ident_lc).first()
            if not user:
                user = User.query.filter_by(email=ident_lc).first()

            # Always show the same message to avoid revealing whether an account exists.
            flash('If an account exists for that username/email, a reset link has been sent.', 'success')

            if user and _can_issue_reset_email(user):
                _send_password_reset_email(user)

            return redirect(url_for('login'))

        return render_template('forgot_password.html', form=form)

    @app.route('/reset-password/<token>', methods=['GET', 'POST'])
    def reset_password(token: str):
        if current_user.is_authenticated:
            return redirect(url_for('home'))

        s = _password_reset_serializer()
        try:
            data = s.loads(token, salt='password-reset', max_age=60 * 60)
        except SignatureExpired:
            flash('That reset link has expired. Please request a new one.', 'error')
            return redirect(url_for('forgot_password'))
        except BadSignature:
            flash('Invalid reset link. Please request a new one.', 'error')
            return redirect(url_for('forgot_password'))

        uid = (data or {}).get('uid')
        user = User.query.get(int(uid)) if uid is not None else None
        if not user:
            flash('Invalid reset link. Please request a new one.', 'error')
            return redirect(url_for('forgot_password'))

        form = ResetPasswordForm()
        if form.validate_on_submit():
            user.set_password(form.new_password.data)
            db.session.commit()
            flash('Password updated. You can log in now.', 'success')
            return redirect(url_for('login'))

        return render_template('reset_password.html', form=form, token=token)

    @app.get("/u/<username>")
    def profile(username: str):
        ident_lc = username.strip().lower()
        u = User.query.filter(func.lower(User.username) == ident_lc).first_or_404()

        reports = (
            SpeedReport.query.filter(SpeedReport.is_deleted.is_(False)).filter(SpeedReport.user_id == u.id)
            .order_by(SpeedReport.created_at.desc())
            .limit(50)
            .all()
        )

        return render_template(
            "profile.html",
            profile_user=u,
            reports=reports,
            ticket_count=len(reports),
        )

    def strictness_rows(
        limit: int,
        exclude_anonymous: bool,
        state_filter: str | None = None,
        road_filter: str | None = None,
        county_geoid: str | None = None,
        speed_limit_list: List[str] | None = None,
        over_list: List[str] | None = None,
        photo_only: bool = False,
        verify: str = "any",
        date: str = "any",
                pin_only: bool = False,
        deleted_mode: str = "hide",
    ) -> Dict[str, List[Dict]]:
        """Return county-based statistics (Lower/Higher median overage).

        Metric: MEDIAN overage (ticketed - posted), considering only tickets where ticketed_speed > posted_speed.
        Lower median overage = more strict.

        Supports the shared filter rail inputs so the Statistics page behaves like other pages.
        """

        speed_limit_list = speed_limit_list or []
        over_list = over_list or []

        expr = (SpeedReport.ticketed_speed - SpeedReport.posted_speed)

        q = (
            db.session.query(
                SpeedReport.state.label("state"),
                SpeedReport.county_geoid.label("county_geoid"),
                expr.label("overage"),
                SpeedReport.created_at.label("created_at"),
                SpeedReport.photo_key.label("photo_key"),
                SpeedReport.ocr_status.label("ocr_status"),
                SpeedReport.verification_status.label("verification_status"),
                SpeedReport.user_id.label("user_id"),
            )
            .filter(SpeedReport.ticketed_speed > SpeedReport.posted_speed)
            .filter(SpeedReport.county_geoid.isnot(None))
        )

        # Deleted visibility (admin can request include/only; non-admin callers should pass "hide")
        if deleted_mode == "only":
            q = q.filter(SpeedReport.is_deleted.is_(True))
        elif deleted_mode == "include":
            pass
        else:
            q = q.filter(SpeedReport.is_deleted.is_(False))

        # Hide anonymous
        if exclude_anonymous:
            q = q.filter(SpeedReport.user_id.isnot(None))

        # State filter (2-letter prefix)
        if state_filter:
            q = q.filter(SpeedReport.state.ilike(f"{state_filter}%"))

        # County filter (GEOID)
        if county_geoid:
            q = q.filter(SpeedReport.county_geoid == county_geoid)

        # Map pin filter (only tickets with a user-placed pin)
        if pin_only:
            q = q.filter(SpeedReport.location_source.in_(("user_pin", "user_pin_snapped")))

        # Date filter
        if date not in ("", "any"):
            try:
                days = int(date)
            except Exception:
                days = 0
            if days > 0:
                q = q.filter(SpeedReport.created_at >= (datetime.utcnow() - timedelta(days=days)))

        # Optional road filter (legacy param support)
        if road_filter:
            try:
                road_key = normalize_road(road_filter, (state_filter or ""))
            except Exception:
                road_key = ""
            conds = [SpeedReport.road_name.ilike(f"%{road_filter}%")]
            if road_key:
                conds.append(SpeedReport.road_key == road_key)
            q = q.filter(or_(*conds))

        # Speed limit filter (posted_speed) — multi-select buckets
        if speed_limit_list:
            sl_conds = []
            for v in speed_limit_list:
                if v == "lte35":
                    sl_conds.append(SpeedReport.posted_speed <= 35)
                elif v == "40-55":
                    sl_conds.append(and_(SpeedReport.posted_speed >= 40, SpeedReport.posted_speed <= 55))
                elif v == "gte60":
                    sl_conds.append(SpeedReport.posted_speed >= 60)
                elif v == "25-35":
                    sl_conds.append(and_(SpeedReport.posted_speed >= 25, SpeedReport.posted_speed <= 35))
                elif v == "40-50":
                    sl_conds.append(and_(SpeedReport.posted_speed >= 40, SpeedReport.posted_speed <= 50))
                elif v == "55":
                    sl_conds.append(SpeedReport.posted_speed == 55)
                elif v == "65":
                    sl_conds.append(SpeedReport.posted_speed == 65)
                elif v == "70+":
                    sl_conds.append(SpeedReport.posted_speed >= 70)
            if sl_conds:
                q = q.filter(or_(*sl_conds))

        # Overage filter (ticketed_speed - posted_speed) — multi-select buckets
        if over_list:
            over_conds = []
            for v in over_list:
                if v == "1-10":
                    over_conds.append(and_(expr >= 1, expr <= 10))
                elif v == "11-20":
                    over_conds.append(and_(expr >= 11, expr <= 20))
                elif v == "21+":
                    over_conds.append(expr >= 21)
                elif v == "5-9":
                    over_conds.append(and_(expr >= 5, expr <= 9))
                elif v == "10-14":
                    over_conds.append(and_(expr >= 10, expr <= 14))
                elif v == "15-19":
                    over_conds.append(and_(expr >= 15, expr <= 19))
                elif v == "20+":
                    over_conds.append(expr >= 20)
            if over_conds:
                q = q.filter(or_(*over_conds))

        # Evidence: photo only
        if photo_only:
            q = q.filter(SpeedReport.photo_key.isnot(None))

        # Evidence: verification status (implies photo exists)
        if verify == "verified":
            q = q.filter(
                SpeedReport.photo_key.isnot(None),
                or_(SpeedReport.ocr_status == "verified", SpeedReport.verification_status == "verified"),
            )
        elif verify == "not_verified":
            q = q.filter(
                SpeedReport.photo_key.isnot(None),
                ~or_(SpeedReport.ocr_status == "verified", SpeedReport.verification_status == "verified"),
            )

        rows = q.all()

        groups: Dict[tuple, Dict] = {}
        for r in rows:
            geoid = (r.county_geoid or "").strip()
            if not geoid:
                continue
            key = (normalize_state_group(r.state), geoid)
            g = groups.get(key)
            if g is None:
                g = {
                    "state": key[0],
                    "county_geoid": geoid,
                    "overages": [],
                    "tickets": 0,
                }
                groups[key] = g

            try:
                g["overages"].append(int(r.overage))
            except Exception:
                continue
            g["tickets"] += 1

        geoids = sorted({g["county_geoid"] for g in groups.values() if g.get("county_geoid")})

        # Resolve county + full state names from counties table (best-effort)
        county_meta = {}
        if geoids:
            try:
                stmt = (
                    text(
                        """
                        SELECT geoid,
                               COALESCE(namelsad, name) AS county_label,
                               COALESCE(state_name, '') AS state_name,
                               COALESCE(stusps, '') AS stusps
                        FROM counties
                        WHERE geoid IN :geoids
                        """
                    ).bindparams(bindparam("geoids", expanding=True))
                )
                meta_rows = db.session.execute(stmt, {"geoids": geoids}).mappings().all()
                for rr in meta_rows:
                    county_meta[rr.get("geoid")] = {
                        "county_name": (rr.get("county_label") or "").strip(),
                        "state_name": (rr.get("state_name") or "").strip(),
                        "stusps": (rr.get("stusps") or "").strip().upper(),
                    }
            except Exception:
                try:
                    db.session.rollback()
                except Exception:
                    pass

        state_name_by_abbr = {abbr: name for abbr, name in STATE_PAIRS}

        results = []
        for key, g in groups.items():
            overages = g.get("overages") or []
            if not overages:
                continue

            med = float(median(sorted(overages)))

            geoid = g.get("county_geoid")
            meta = county_meta.get(geoid, {})

            st_code = g.get("state") or ""
            st_name = (meta.get("state_name") or "").strip() or state_name_by_abbr.get(st_code, st_code)
            county_name = (meta.get("county_name") or "").strip() or "Unknown County"

            results.append(
                {
                    "state": st_code,
                    "state_name": st_name,
                    "county_geoid": geoid,
                    "county_name": county_name,
                    "median_overage": med,
                    "tickets": int(g.get("tickets") or 0),
                }
            )

        results.sort(key=lambda d: (d["median_overage"], -d["tickets"], d["state"], d["county_name"]))
        most_strict = results[:limit]

        least_sorted = sorted(results, key=lambda d: (-d["median_overage"], -d["tickets"], d["state"], d["county_name"]))
        least_strict = least_sorted[:limit]

        return {"most_strict": most_strict, "least_strict": least_strict}

    @app.get("/strictness", endpoint="strictness")
    def strictness():
        filter_county_geoid, filter_county_label = _get_county_filter_from_request()
        """Web rankings page (server-rendered) used by base.html nav."""
        hide_anon_requested = (request.args.get("hide_anon") == "1")
        if hide_anon_requested and not current_user.is_authenticated:
            flash("You must be logged in to hide anonymous posts.", "info")

        hide_anon = bool(current_user.is_authenticated and hide_anon_requested)

        is_admin = is_admin_user(current_user)
        deleted_mode = (request.args.get("deleted") or "hide").strip().lower()
        if not is_admin:
            deleted_mode = "hide"
        if deleted_mode not in ("hide", "include", "only"):
            deleted_mode = "hide"

        filter_state = (request.args.get("state") or "").strip().upper()
        filter_road = (request.args.get("road") or "").strip()

        filter_speed_limit_list = [v.strip() for v in request.args.getlist("speed_limit") if (v or "").strip()]
        filter_speed_limit_list = [v for v in filter_speed_limit_list if v != "any"]
        filter_speed_limit_list = list(dict.fromkeys(filter_speed_limit_list))

        filter_over_list = [v.strip() for v in request.args.getlist("overage") if (v or "").strip()]
        filter_over_list = [v for v in filter_over_list if v not in ("all", "any")]
        legacy_over = (request.args.get("over") or "").strip()
        if legacy_over and legacy_over not in ("all", ""):
            filter_over_list = list(dict.fromkeys(filter_over_list + [legacy_over]))

        filter_photo_only = (request.args.get("photo_only") == "1")
        filter_verify = (request.args.get("verify") or "any").strip()
        filter_pin = (request.args.get("pin") == "1")

        filter_date = (request.args.get("date") or "any").strip()

        if request.args.get("verified_photo") == "1":
            filter_photo_only = True
            filter_verify = "verified"

        filter_sort = (request.args.get("sort") or "new").strip()

        # Back-compat: older UI used 'most_strict'/'least_strict' labels for overage sorting
        if filter_sort == "most_strict":
            filter_sort = "least_over"
        elif filter_sort == "least_strict":
            filter_sort = "most_over"

        # State dropdown options (2-letter codes)
        state_filter_options = []
        try:
            rows = db.session.query(SpeedReport.state).distinct().all()
        except Exception:
            try:
                db.session.rollback()
            except Exception:
                pass
            rows = []
        seen = set()
        for (sv,) in rows:
            if not sv:
                continue
            code = sv.split(" - ")[0].strip().upper()
            if len(code) != 2 or not code.isalpha():
                continue
            if code in seen:
                continue
            seen.add(code)
            state_filter_options.append({"code": code})
        state_filter_options.sort(key=lambda d: d["code"])

        speed_limit_buckets = [
            {"value": "any", "label": "Any"},
            {"value": "lte35", "label": "≤ 35 mph"},
            {"value": "40-55", "label": "40–55 mph"},
            {"value": "gte60", "label": "≥ 60 mph"},
        ]

        over_buckets = [
            {"value": "all", "label": "Any"},
            {"value": "1-10", "label": "1–10 mph over"},
            {"value": "11-20", "label": "11–20 mph over"},
            {"value": "21+", "label": "≥ 21 mph over"},
        ]

        date_options = [
            {"value": "any", "label": "Any"},
            {"value": "7", "label": "Last 7 days"},
            {"value": "30", "label": "Last 30 days"},
            {"value": "90", "label": "Last 90 days"},
            {"value": "365", "label": "Last year"},
        ]

        verify_options = [
            {"value": "any", "label": "Any"},
            {"value": "verified", "label": "Auto-Extracted"},
            {"value": "not_verified", "label": "Not auto-extracted"},
        ]

        filters_active = bool(
            filter_state
            or filter_county_geoid
            or filter_road
            or bool(filter_speed_limit_list)
            or bool(filter_over_list)
            or filter_pin
            or filter_photo_only
            or (filter_verify not in ("", "any"))
            or (filter_date not in ("", "any"))
            or (filter_sort not in ("", "new"))
            or (deleted_mode not in ("", "hide"))
            or hide_anon
        )

        # Road suggestions for datalist (same pattern as home)
        road_suggestions: List[str] = []
        try:
            rq = db.session.query(SpeedReport.road_name).filter(SpeedReport.road_name.isnot(None))
            if filter_state:
                rq = rq.filter(SpeedReport.state.ilike(f"{filter_state}%"))
            rows2 = rq.distinct().order_by(SpeedReport.road_name.asc()).limit(40).all()
            road_suggestions = [rn for (rn,) in rows2 if rn]
        except Exception:
            try:
                db.session.rollback()
            except Exception:
                pass
            road_suggestions = []

        data = strictness_rows(
            limit=20,
            exclude_anonymous=hide_anon,
            state_filter=(filter_state or None),
            road_filter=(filter_road or None),
            county_geoid=(filter_county_geoid or None),
            speed_limit_list=filter_speed_limit_list,
            over_list=filter_over_list,
            photo_only=filter_photo_only,
            verify=(filter_verify or "any"),
            date=(filter_date or "any"),
            pin_only=filter_pin,
            deleted_mode=deleted_mode,
        )

        return render_template(
            "strictness.html",
            most_strict=data.get("most_strict", []),
            least_strict=data.get("least_strict", []),
            is_admin=is_admin,
            deleted_mode=deleted_mode,
            hide_anon=hide_anon,
            filters_active=filters_active,
            filter_state=filter_state,
            filter_road=filter_road,
            filter_county_geoid=filter_county_geoid,
            filter_county_label=filter_county_label,
            filter_speed_limit_list=filter_speed_limit_list,
            filter_over_list=filter_over_list,
            filter_photo_only=filter_photo_only,
            filter_verify=filter_verify,
            filter_pin=filter_pin,
            filter_date=filter_date,
            filter_sort=filter_sort,
            state_filter_options=state_filter_options,
            speed_limit_buckets=speed_limit_buckets,
            over_buckets=over_buckets,
            date_options=date_options,
            verify_options=verify_options,
            road_suggestions=road_suggestions,
        )


    @app.get("/statistics")
    def statistics_redirect():
        """Alias for the Statistics page (redirects to /strictness)."""
        qs = request.query_string.decode('utf-8') if request.query_string else ''
        return redirect('/strictness' + (('?' + qs) if qs else ''))



    @app.get("/bucket", endpoint="bucket_tickets")
    def bucket_tickets():
        """Simple per-(state, road) ticket list used by strictness + feed links."""
        state_raw = (request.args.get("state") or "").strip()
        road_raw = (request.args.get("road") or "").strip()

        if not state_raw or not road_raw:
            flash("Missing road selection.", "error")
            return redirect(url_for("home"))

        state_code = state_raw.split(" - ")[0].strip().upper()
        if len(state_code) != 2 or not state_code.isalpha():
            # fall back to raw prefix matching
            state_code = state_raw.strip()

        # Normalize road filter if possible
        road_key = road_raw
        try:
            rk = normalize_road(road_raw, state_code if len(state_code) == 2 else "")
            if rk:
                road_key = rk
        except Exception:
            road_key = road_raw

        q = SpeedReport.query.options(joinedload(SpeedReport.user))

        # state prefix match (keeps compatibility with "MO - Missouri" stored values)
        if len(state_code) == 2:
            q = q.filter(SpeedReport.state.ilike(f"{state_code}%"))
        else:
            q = q.filter(SpeedReport.state.ilike(f"{state_code}%"))

        # road match: exact key OR forgiving partial name
        conds = []
        if road_key:
            conds.append(SpeedReport.road_key == road_key)
        conds.append(SpeedReport.road_name.ilike(f"%{road_raw}%"))
        q = q.filter(or_(*conds))

        q = q.filter(SpeedReport.is_deleted.is_(False)).order_by(SpeedReport.created_at.desc())

        reports = q.limit(50).all()

        total_tickets = len(reports)
        member_count = sum(1 for r in reports if getattr(r, "user_id", None))
        anon_count = total_tickets - member_count
        member_pct = int(round((member_count / total_tickets) * 100)) if total_tickets else 0
        anon_pct = int(round((anon_count / total_tickets) * 100)) if total_tickets else 0

        rows = []
        for r in reports:
            try:
                username = r.user.username if getattr(r, "user", None) else None
            except Exception:
                username = None
            rows.append(
                {
                    "id": r.id,
                    "posted_speed": r.posted_speed,
                    "ticketed_speed": r.ticketed_speed,
                    "username": username,
                    "created_at": r.created_at,
                }
            )

        return render_template(
            "bucket_tickets.html",
            state=(state_code if len(state_code) == 2 else state_raw),
            road_key=road_key,
            rows=rows,
            total_tickets=total_tickets,
            member_pct=member_pct,
            anon_pct=anon_pct,
        )


    @app.get("/result/<int:report_id>", endpoint="result")
    def result(report_id: int):
        """Minimal report detail page (compat for legacy profile links)."""
        r = SpeedReport.query.options(joinedload(SpeedReport.user)).get_or_404(report_id)

        try:
            username = r.user.username if r.user else None
        except Exception:
            username = None

        state_code = (r.state or "").split(" - ")[0].strip().upper()
        if len(state_code) != 2:
            state_code = (r.state or "").strip()

        road_label = r.road_key or r.road_name or ""

        return render_template(
            "report_detail.html",
            report=r,
            username=username,
            state_code=state_code,
            road_label=road_label,
        )


    @app.get("/submit-ticket", endpoint="submit_ticket")
    def submit_ticket():
        """Legacy endpoint alias (older landing pages)."""
        return redirect(url_for("submit"))

    @app.get("/")
    def home():
        filter_county_geoid, filter_county_label = _get_county_filter_from_request()
        """
        Soft-gated public feed (newest first).
        Anonymous posts are visible by default.
        Logged-in users can optionally hide anonymous posts via ?hide_anon=1.
        """
        hide_anon_requested = (request.args.get("hide_anon") == "1")
        if hide_anon_requested and not current_user.is_authenticated:
            flash("You must be logged in to hide anonymous posts.", "info")

        hide_anon = bool(current_user.is_authenticated and hide_anon_requested)

        is_admin = is_admin_user(current_user)
        deleted_mode = (request.args.get("deleted") or "hide").strip().lower()
        if not is_admin:
            deleted_mode = "hide"
        if deleted_mode not in ("hide", "include", "only"):
            deleted_mode = "hide"
        show_deleted = bool(is_admin and deleted_mode in ("include", "only"))
        deleted_only = bool(is_admin and deleted_mode == "only")

        page = request.args.get("page", 1, type=int)
        per_page = 20

        # --- Feed filters (GET) ---
        filter_state = (request.args.get("state") or "").strip().upper()
        filter_road = (request.args.get("road") or "").strip()

        # Speed limit buckets (posted_speed)
        # UI supports multi-select via repeated speed_limit= values.
        # Back-compat: older UI used a single ?speed_limit=<bucket>.
        filter_speed_limit_list = [v.strip() for v in request.args.getlist("speed_limit") if (v or "").strip()]
        # Normalize: ignore 'any' and de-dup while preserving order.
        filter_speed_limit_list = [v for v in filter_speed_limit_list if v != "any"]
        filter_speed_limit_list = list(dict.fromkeys(filter_speed_limit_list))

        # Overage buckets (ticketed_speed - posted_speed)
        # New UI supports multi-select via repeated overage= values.
        # Back-compat: old UI used a single ?over=<bucket>.
        filter_over_list = [v.strip() for v in request.args.getlist("overage") if (v or "").strip()]
        legacy_over = (request.args.get("over") or "").strip()
        if legacy_over and legacy_over not in ("all", ""):
            filter_over_list = list(dict.fromkeys(filter_over_list + [legacy_over]))

        # Evidence filters
        filter_photo_only = (request.args.get("photo_only") == "1")
        filter_verify = (request.args.get("verify") or "any").strip()
        filter_pin = (request.args.get("pin") == "1")

        # Date filter (created_at)
        filter_date = (request.args.get("date") or "any").strip()

        # Back-compat: old ?verified_photo=1 means 'verified photos only'
        if request.args.get("verified_photo") == "1":
            filter_photo_only = True
            filter_verify = "verified"

        # Sort
        filter_sort = (request.args.get("sort") or "new").strip()

        # Back-compat: older UI used 'most_strict'/'least_strict' labels for overage sorting
        if filter_sort == "most_strict":
            filter_sort = "least_over"
        elif filter_sort == "least_strict":
            filter_sort = "most_over"

        # State dropdown options (2-letter codes)
        state_filter_options = []
        try:
            rows = db.session.query(SpeedReport.state).distinct().all()
        except Exception:
            try:
                db.session.rollback()
            except Exception:
                pass
            rows = []
        seen = set()
        for (sv,) in rows:
            if not sv:
                continue
            code = sv.split(' - ')[0].strip().upper()
            if len(code) != 2 or not code.isalpha():
                continue
            if code in seen:
                continue
            seen.add(code)
            state_filter_options.append({"code": code})
        state_filter_options.sort(key=lambda d: d["code"])

        speed_limit_buckets = [
            {"value": "any", "label": "Any"},
            {"value": "25-35", "label": "25–35 mph"},
            {"value": "40-50", "label": "40–50 mph"},
            {"value": "55", "label": "55 mph"},
            {"value": "65", "label": "65 mph"},
            {"value": "70+", "label": "70+ mph"},
        ]

        over_buckets = [
            {"value": "all", "label": "Any"},
            {"value": "5-9", "label": "5–9 mph"},
            {"value": "10-14", "label": "10–14 mph"},
            {"value": "15-19", "label": "15–19 mph"},
            {"value": "20+", "label": "20+ mph"},
        ]

        date_options = [
            {"value": "any", "label": "Any"},
            {"value": "7", "label": "Last 7 days"},
            {"value": "30", "label": "Last 30 days"},
            {"value": "90", "label": "Last 90 days"},
            {"value": "365", "label": "Last year"},
        ]

        verify_options = [
            {"value": "any", "label": "Any"},
            {"value": "verified", "label": "Auto-Extracted"},
            {"value": "not_verified", "label": "Not auto-extracted"},
        ]

        filters_active = bool(
            filter_state
            or filter_county_geoid
            or filter_road
            or bool(filter_speed_limit_list)
            or bool(filter_over_list)
            or filter_pin
            or filter_photo_only
            or (filter_verify not in ("", "any"))
            or (filter_date not in ("", "any"))
            or (filter_sort not in ("", "new"))
            or (deleted_mode not in ("", "hide"))
            or hide_anon
        )

        # Base query
        q = SpeedReport.query.options(joinedload(SpeedReport.user))

        # Deleted visibility (admin: ?deleted=hide|include|only)
        if deleted_only:
            q = q.filter(SpeedReport.is_deleted.is_(True))
        elif deleted_only:
            q = q.filter(SpeedReport.is_deleted.is_(True))
        elif not show_deleted:
            q = q.filter(SpeedReport.is_deleted.is_(False))

        # Hide anonymous (members only)
        if hide_anon:
            q = q.filter(SpeedReport.user_id.isnot(None))

        # State filter (2-letter code prefix)
        if filter_state:
            q = q.filter(SpeedReport.state.ilike(f"{filter_state}%"))

        # County filter (GEOID)
        # If we're focused on a county in viewport mode, do NOT force the query to only that county.
        # We'll color pins instead (inside vs outside).
        if filter_county_geoid:
            q = q.filter(SpeedReport.county_geoid == filter_county_geoid)

        # Viewport filter

        # Map pin filter (only tickets with a user-placed pin)
        if filter_pin:
            q = q.filter(SpeedReport.location_source.in_(("user_pin", "user_pin_snapped")))

        # Date filter
        if filter_date not in ("", "any"):
            try:
                days = int(filter_date)
            except Exception:
                days = 0
            if days > 0:
                q = q.filter(SpeedReport.created_at >= (datetime.utcnow() - timedelta(days=days)))

        # Road filter (forgiving: match normalized bucket OR partial raw text)
        if filter_road:
            try:
                road_key = normalize_road(filter_road, filter_state)
            except Exception:
                road_key = ""
            conds = [SpeedReport.road_name.ilike(f"%{filter_road}%")]
            if road_key:
                conds.append(SpeedReport.road_key == road_key)
            q = q.filter(or_(*conds))

        # Speed limit filter (posted_speed) — multi-select buckets.
        if filter_speed_limit_list:
            sl_conds = []
            for v in filter_speed_limit_list:
                if v == "lte35":
                    sl_conds.append(SpeedReport.posted_speed <= 35)
                elif v == "40-55":
                    sl_conds.append(and_(SpeedReport.posted_speed >= 40, SpeedReport.posted_speed <= 55))
                elif v == "gte60":
                    sl_conds.append(SpeedReport.posted_speed >= 60)
                elif v == "25-35":
                    sl_conds.append(and_(SpeedReport.posted_speed >= 25, SpeedReport.posted_speed <= 35))
                elif v == "40-50":
                    sl_conds.append(and_(SpeedReport.posted_speed >= 40, SpeedReport.posted_speed <= 50))
                elif v == "55":
                    sl_conds.append(SpeedReport.posted_speed == 55)
                elif v == "65":
                    sl_conds.append(SpeedReport.posted_speed == 65)
                elif v == "70+":
                    sl_conds.append(SpeedReport.posted_speed >= 70)
            if sl_conds:
                q = q.filter(or_(*sl_conds))

        # Overage filter (ticketed_speed - posted_speed)
        expr = (SpeedReport.ticketed_speed - SpeedReport.posted_speed)
        if filter_over_list:
            over_conds = []
            for v in filter_over_list:
                if v == "1-10":
                    over_conds.append(and_(expr >= 1, expr <= 10))
                elif v == "11-20":
                    over_conds.append(and_(expr >= 11, expr <= 20))
                elif v == "21+":
                    over_conds.append(expr >= 21)
                elif v == "5-9":
                    over_conds.append(and_(expr >= 5, expr <= 9))
                elif v == "10-14":
                    over_conds.append(and_(expr >= 10, expr <= 14))
                elif v == "15-19":
                    over_conds.append(and_(expr >= 15, expr <= 19))
                elif v == "20+":
                    over_conds.append(expr >= 20)
            if over_conds:
                q = q.filter(or_(*over_conds))

        # Evidence: photo only
        if filter_photo_only:
            q = q.filter(SpeedReport.photo_key.isnot(None))

        # Evidence: verification status (implies photo exists)
        if filter_verify == "verified":
            q = q.filter(
                SpeedReport.photo_key.isnot(None),
                or_(SpeedReport.ocr_status == "verified", SpeedReport.verification_status == "verified"),
            )
        elif filter_verify == "not_verified":
            q = q.filter(
                SpeedReport.photo_key.isnot(None),
                ~or_(SpeedReport.ocr_status == "verified", SpeedReport.verification_status == "verified"),
            )

        # Sort
        if filter_sort == "old":
            q = q.order_by(SpeedReport.created_at.asc())
        elif filter_sort == "most_over":
            q = q.order_by(expr.desc(), SpeedReport.created_at.desc())
        elif filter_sort == "least_over":
            # Back-compat with older UI
            q = q.order_by(expr.asc(), SpeedReport.created_at.desc())
        elif filter_sort in ("most_strict", "least_strict"):
            # Strictness proxy for the feed: the 90% enforced overage threshold for this (state, road_key).
            # Lower threshold = more strict.
            p90_over = func.percentile_cont(0.1).within_group(expr.asc())
            subq = (
                db.session.query(
                    SpeedReport.state.label("st"),
                    SpeedReport.road_key.label("rk"),
                    p90_over.label("p90_over"),
                )
                .group_by(SpeedReport.state, SpeedReport.road_key)
                .subquery()
            )
            q = q.join(subq, and_(SpeedReport.state == subq.c.st, SpeedReport.road_key == subq.c.rk))
            if filter_sort == "most_strict":
                q = q.order_by(subq.c.p90_over.asc(), SpeedReport.created_at.desc())
            else:
                q = q.order_by(subq.c.p90_over.desc(), SpeedReport.created_at.desc())
        else:
            q = q.order_by(SpeedReport.created_at.desc())

        pagination = q.paginate(page=page, per_page=per_page, error_out=False)
        reports = pagination.items

        # Road suggestions for predictive input (datalist).
        # Keep it light: show top distinct road_names for the selected state (or all).
        road_suggestions: List[str] = []
        try:
            rq = db.session.query(SpeedReport.road_name).filter(SpeedReport.road_name.isnot(None))
            if filter_state:
                rq = rq.filter(SpeedReport.state.ilike(f"{filter_state}%"))
            rows = rq.distinct().order_by(SpeedReport.road_name.asc()).limit(40).all()
            road_suggestions = [rn for (rn,) in rows if rn]
        except Exception:
            try:
                db.session.rollback()
            except Exception:
                pass
            road_suggestions = []

        # Compute per-post min/max percentiles using only group min/max (fast, stable meaning).
        keys_a = {(r.state, r.road_key) for r in reports if r.road_key}  # Road + State group
        keys_b = {(r.state, r.posted_speed) for r in reports}  # State + Posted Speed group

        expr = (SpeedReport.ticketed_speed - SpeedReport.posted_speed)

        # Road + State: true percentile rank within the (state, road_key) distribution.
        # Percent = % of tickets in the same group with overage <= this ticket's overage.
        a_dist: Dict[tuple[str, str], List[float]] = {}
        b_dist: Dict[tuple[str, int], List[float]] = {}
        if keys_a:
            rows = (
                db.session.query(
                    SpeedReport.state,
                    SpeedReport.road_key,
                    expr.label("overage"),
                )
                .filter(SpeedReport.is_deleted.is_(False))
                .filter(tuple_(SpeedReport.state, SpeedReport.road_key).in_(list(keys_a)))
                .all()
            )
            for st, rk, overage in rows:
                a_dist.setdefault((st, rk), []).append(float(overage))
            for k, vals in a_dist.items():
                vals.sort()

        if keys_b:
            rows = (
                db.session.query(
                    SpeedReport.state,
                    SpeedReport.posted_speed,
                    expr.label("overage"),
                )
                .filter(SpeedReport.is_deleted.is_(False))
                .filter(tuple_(SpeedReport.state, SpeedReport.posted_speed).in_(list(keys_b)))
                .all()
            )
            for st, ps, overage in rows:
                b_dist.setdefault((st, ps), []).append(float(overage))
            for k, vals in b_dist.items():
                vals.sort()

        def _pctl(sorted_vals, p):
            """EnforcedSpeed threshold helper.

            Returns the smallest value v such that at least p (e.g., 0.90)
            of the distribution is at or above v.

            Implementation (Option B): pick index k = floor((1-p)*n) + 1
            from the ascending-sorted list.
            """
            if not sorted_vals:
                return 0.0
            n = len(sorted_vals)
            # k is 1-indexed into ascending list
            k = int(math.floor((1.0 - p) * n)) + 1
            if k < 1:
                k = 1
            if k > n:
                k = n
            return float(sorted_vals[k - 1])

        a_p90 = {k: _pctl(vals, 0.90) for k, vals in a_dist.items()}
        b_p90 = {k: _pctl(vals, 0.90) for k, vals in b_dist.items()}

        pct_a = {}
        pct_b = {}
        enforced_a = {}
        enforced_b = {}
        for r in reports:
            v = float(r.ticketed_speed - r.posted_speed)

            vals = a_dist.get((r.state, r.road_key))
            if not vals:
                pct_a[r.id] = 100.0
            else:
                # bisect_right gives count of values <= v
                pct_a[r.id] = round((bisect_right(vals, v) / len(vals)) * 100.0, 1)

            vals_b = b_dist.get((r.state, r.posted_speed))
            if not vals_b:
                pct_b[r.id] = 100.0
            else:
                pct_b[r.id] = round((bisect_right(vals_b, v) / len(vals_b)) * 100.0, 1)

            enforced_a[r.id] = a_p90.get((r.state, r.road_key), v)
            enforced_b[r.id] = b_p90.get((r.state, r.posted_speed), v)

        a_counts = {k: len(vals) for k, vals in a_dist.items()}

        # Per-user ticket counts (for feed display: Username (X))
        user_ticket_counts: Dict[int, int] = {}
        user_ids = {r.user_id for r in reports if r.user_id}
        if user_ids:
            rows = (
                db.session.query(SpeedReport.user_id, func.count(SpeedReport.id))
                .filter(SpeedReport.is_deleted.is_(False))
                .filter(SpeedReport.user_id.in_(list(user_ids)))
                .group_by(SpeedReport.user_id)
                .all()
            )
            user_ticket_counts = {uid: int(c) for uid, c in rows if uid is not None}


        report_ids = [r.id for r in reports]

        # Likes: counts + whether current user liked each report
        like_counts: Dict[int, int] = {}
        user_liked: set[int] = set()
        if report_ids:
            rows = (
                db.session.query(Like.report_id, func.count(Like.id))
                .filter(Like.report_id.in_(report_ids))
                .group_by(Like.report_id)
                .all()
            )
            like_counts = {rid: int(c) for rid, c in rows}

            if current_user.is_authenticated:
                rows = (
                    db.session.query(Like.report_id)
                    .filter(Like.report_id.in_(report_ids), Like.user_id == current_user.id)
                    .all()
                )
                user_liked = {rid for (rid,) in rows}

                # Ticket post counts per user (for feed display: username (N))
                user_ids = [uid for uid in {r.user_id for r in reports} if uid]
                if user_ids:
                    rows = (
                        db.session.query(SpeedReport.user_id, func.count(SpeedReport.id))
                        .filter(SpeedReport.user_id.in_(user_ids))
                        .group_by(SpeedReport.user_id)
                        .all()
                    )
                    user_ticket_posts_counts = {int(uid): int(c) for uid, c in rows}

        # Comments: show oldest -> newest so threads read naturally.
        # IMPORTANT: legacy data may contain invalid/cyclic parent pointers from before true nesting.
        # If we build threads naively, a cycle can hang template rendering (browser spins forever).
        # We therefore sanitize parent pointers per-report before building the thread maps.
        comments_by_report: Dict[int, List[Comment]] = {rid: [] for rid in report_ids}
        comment_threads_by_report: Dict[int, Dict[str, object]] = {rid: {"top": [], "replies": {}} for rid in report_ids}

        if report_ids:
            rows = (
                Comment.query.options(joinedload(Comment.user))
                .filter(Comment.report_id.in_(report_ids))
                .order_by(Comment.created_at.asc())
                .all()
            )

            # Group by report so we can sanitize parent pointers within each report.
            by_report: Dict[int, List[Comment]] = {}
            for c in rows:
                by_report.setdefault(c.report_id, []).append(c)

            def _safe_parent_id(comment: Comment, id_map: Dict[int, Comment]) -> int | None:
                """Return a safe parent_id (or None) by rejecting self-loops, missing parents, and cycles."""
                pid = comment.parent_id
                if not pid:
                    return None
                if pid == comment.id:
                    return None

                # Walk upward to detect cycles / broken pointers.
                seen: set[int] = {comment.id}
                cur = pid
                steps = 0
                while cur:
                    steps += 1
                    if steps > 50:
                        # Hard safety cap; treat as top-level if chain is suspiciously deep.
                        return None
                    if cur in seen:
                        return None
                    seen.add(cur)
                    parent = id_map.get(cur)
                    if parent is None:
                        return None
                    if parent.parent_id == parent.id:
                        return None
                    cur = parent.parent_id
                return pid

            for rid, comments in by_report.items():
                id_map = {c.id: c for c in comments}
                for c in comments:
                    comments_by_report.setdefault(rid, []).append(c)
                    safe_pid = _safe_parent_id(c, id_map)
                    if safe_pid:
                        comment_threads_by_report.setdefault(rid, {"top": [], "replies": {}})
                        comment_threads_by_report[rid]["replies"].setdefault(safe_pid, []).append(c)
                    else:
                        comment_threads_by_report.setdefault(rid, {"top": [], "replies": {}})
                        comment_threads_by_report[rid]["top"].append(c)

        like_form = LikeForm()
        comment_form = CommentForm()
        delete_comment_form = DeleteCommentForm()

        # Home mini-map: only plot tickets that already have coordinates.
        map_pins = []
        for r in reports:
            try:
                if r.lat is None or r.lng is None:
                    continue
                map_pins.append(
                    {
                        "id": int(r.id),
                        "lat": float(r.lat),
                        "lng": float(r.lng),
                        "state": (r.state or ""),
                        "road": (r.road_key or r.road_name or ""),
                        "url": url_for("bucket_tickets", state=r.state, road=(r.road_key or ""))
                        if (r.state and (r.road_key or ""))
                        else None,
                    }
                )
            except Exception:
                continue

        # Counts for the under-username metadata line (only for states/counties visible on this page).
        state_ticket_counts: dict[str, int] = {}
        county_ticket_counts: dict[str, int] = {}
        try:
            states_in_page = sorted({(r.state or '').strip() for r in reports if (r.state or '').strip()})
            counties_in_page = sorted({(r.county_geoid or '').strip() for r in reports if (r.county_geoid or '').strip()})

            if states_in_page:
                rows = (
                    db.session.query(SpeedReport.state, func.count(SpeedReport.id))
                    .filter(SpeedReport.is_deleted == False)
                    .filter(SpeedReport.state.in_(states_in_page))
                    .group_by(SpeedReport.state)
                    .all()
                )
                state_ticket_counts = {str(s): int(c) for (s, c) in rows if s is not None}

            if counties_in_page:
                rows = (
                    db.session.query(SpeedReport.county_geoid, func.count(SpeedReport.id))
                    .filter(SpeedReport.is_deleted == False)
                    .filter(SpeedReport.county_geoid.in_(counties_in_page))
                    .group_by(SpeedReport.county_geoid)
                    .all()
                )
                county_ticket_counts = {str(g): int(c) for (g, c) in rows if g is not None}
        except Exception:
            # Never break the home page for counts.
            state_ticket_counts = {}
            county_ticket_counts = {}

        
        # Right-rail: Trending counties + followed counties (top 5). Keep this best-effort and never break home.
        trending_counties = []
        followed_counties = []
        followed_geoids = set()

        try:
            now_utc = datetime.utcnow()
            t7 = now_utc - timedelta(days=7)
            t30 = now_utc - timedelta(days=30)
            t365 = now_utc - timedelta(days=365)

            # Aggregate per-county counts and "last seen" timestamp once.
            rows = (
                db.session.query(
                    SpeedReport.county_geoid.label("geoid"),
                    func.sum(case((SpeedReport.created_at >= t7, 1), else_=0)).label("c7"),
                    func.sum(case((SpeedReport.created_at >= t30, 1), else_=0)).label("c30"),
                    func.sum(case((SpeedReport.created_at >= t365, 1), else_=0)).label("c365"),
                    func.max(SpeedReport.created_at).label("last_ts"),
                )
                .filter(SpeedReport.is_deleted == False)
                .filter(SpeedReport.county_geoid.isnot(None))
                .group_by(SpeedReport.county_geoid)
                .all()
            )

            # Compute acceleration score (7d vs monthly baseline), with a small-data guard.
            scored = []
            recent = []
            agg_counts = {}
            for r in rows:
                geoid = str(r.geoid) if r.geoid is not None else ""
                c7 = int(r.c7 or 0)
                c30 = int(r.c30 or 0)
                c365 = int(r.c365 or 0)
                last_ts = r.last_ts
                recent.append((last_ts, geoid, c7, c30, c365))
                agg_counts[geoid] = (c7, c30, c365)

                # Guard: require some volume so 1 ticket doesn't "trend".
                if c30 < 5:
                    continue

                baseline = (c30 / 4.285)  # expected weekly volume from 30d
                score = (c7 - baseline) / math.sqrt(baseline + 1.0)
                scored.append((score, c7, geoid, c7, c30, c365))

            scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
            chosen = []
            chosen_geoids = set()
            for s in scored:
                if len(chosen) >= 5:
                    break
                geoid = s[2]
                chosen.append((geoid, s[3], s[4], s[5]))
                chosen_geoids.add(geoid)

            # Fallback: if not enough trending data, fill remaining slots with most recent counties.
            if len(chosen) < 5:
                recent.sort(key=lambda x: (x[0] is not None, x[0]), reverse=True)
                for last_ts, geoid, c7, c30, c365 in recent:
                    if len(chosen) >= 5:
                        break
                    if not geoid or geoid in chosen_geoids:
                        continue
                    chosen.append((geoid, c7, c30, c365))
                    chosen_geoids.add(geoid)

            # Fallback 2: if we *still* don't have enough, fill with top counties overall
            # (so the panel never looks empty).
            if len(chosen) < 5:
                top_rows = (
                    db.session.query(
                        SpeedReport.county_geoid.label("geoid"),
                        func.count(SpeedReport.id).label("ct"),
                        func.max(SpeedReport.created_at).label("last_ts"),
                    )
                    .filter(SpeedReport.is_deleted == False)
                    .filter(SpeedReport.county_geoid.isnot(None))
                    .group_by(SpeedReport.county_geoid)
                    .order_by(func.count(SpeedReport.id).desc(), func.max(SpeedReport.created_at).desc())
                    .limit(10)
                    .all()
                )
                for r2 in top_rows:
                    if len(chosen) >= 5:
                        break
                    geoid2 = str(r2.geoid) if r2.geoid is not None else ""
                    if not geoid2 or geoid2 in chosen_geoids:
                        continue
                    c7b, c30b, c365b = agg_counts.get(geoid2, (0, 0, 0))
                    chosen.append((geoid2, c7b, c30b, c365b))
                    chosen_geoids.add(geoid2)

            # Compute median overage (last 365d) for the counties we will display.
            # Keep this best-effort and never break the home page.
            median_overage_by_geoid: dict[str, str] = {}
            try:
                geoids_needed = set(chosen_geoids)
                if current_user.is_authenticated and followed_geoids:
                    geoids_needed.update(followed_geoids)

                if geoids_needed:
                    # Prefer Postgres percentile_cont; otherwise fall back to a small Python median per geoid.
                    dialect = getattr(db.engine.dialect, "name", "") or ""
                    if "postgres" in dialect:
                        q = text(
                            "SELECT county_geoid AS geoid, "
                            "percentile_cont(0.5) WITHIN GROUP (ORDER BY overage) AS med "
                            "FROM speed_reports "
                            "WHERE is_deleted = false "
                            "AND county_geoid = ANY(:geoids) "
                            "AND created_at >= :t365 "
                            "AND overage IS NOT NULL "
                            "GROUP BY county_geoid"
                        )
                        rows_med = db.session.execute(q, {"geoids": list(geoids_needed), "t365": t365}).mappings().all()
                        for rmed in rows_med:
                            g = str(rmed.get("geoid") or "")
                            med = rmed.get("med")
                            if not g or med is None:
                                continue
                            try:
                                v = int(round(float(med)))
                                median_overage_by_geoid[g] = f"+{v}mph"
                            except Exception:
                                continue
                    else:
                        # Lightweight fallback: only for the small set of displayed counties.
                        vals = (
                            db.session.query(SpeedReport.county_geoid, SpeedReport.overage)
                            .filter(SpeedReport.is_deleted == False)
                            .filter(SpeedReport.county_geoid.in_(list(geoids_needed)))
                            .filter(SpeedReport.created_at >= t365)
                            .filter(SpeedReport.overage.isnot(None))
                            .all()
                        )
                        by_g: dict[str, list[int]] = {}
                        for g, v in vals:
                            if g is None or v is None:
                                continue
                            by_g.setdefault(str(g), []).append(int(v))
                        for g, arr in by_g.items():
                            if not arr:
                                continue
                            arr.sort()
                            n = len(arr)
                            if n % 2 == 1:
                                medv = arr[n // 2]
                            else:
                                medv = int(round((arr[n // 2 - 1] + arr[n // 2]) / 2.0))
                            median_overage_by_geoid[g] = f"+{int(medv)}mph"
            except Exception:
                median_overage_by_geoid = {}

            # Attach labels for display.
            for geoid, c7, c30, c365 in chosen:
                label = geoid
                st = ""
                try:
                    row = db.session.execute(
                        text("SELECT namelsad, stusps FROM counties WHERE geoid = :g LIMIT 1"),
                        {"g": geoid},
                    ).mappings().first()
                    if row:
                        label = row.get("namelsad") or label
                        st = row.get("stusps") or ""
                except Exception:
                    pass

                trending_counties.append(
                    {"geoid": geoid, "label": label, "st": st, "c7": c7, "c30": c30, "c365": c365, "ovr": median_overage_by_geoid.get(geoid, "—")}
                )

            # Followed counties for the current user.
            if current_user.is_authenticated:
                followed_rows = (
                    db.session.query(FollowedCounty.county_geoid)
                    .filter(FollowedCounty.user_id == current_user.id)
                    .order_by(FollowedCounty.created_at.desc())
                    .limit(50)
                    .all()
                )
                followed_geoids = {str(r[0]) for r in followed_rows if r and r[0]}

                if followed_geoids:
                    # Re-use aggregates above (rows) by building a map.
                    agg = {str(r.geoid): (int(r.c7 or 0), int(r.c30 or 0), int(r.c365 or 0), r.last_ts) for r in rows if r.geoid}
                    # Sort followed by most recent activity.
                    order_geoids = sorted(
                        list(followed_geoids),
                        key=lambda g: (agg.get(g, (0, 0, 0, None))[3] is not None, agg.get(g, (0, 0, 0, None))[3]),
                        reverse=True,
                    )[:5]

                    for geoid in order_geoids:
                        c7, c30, c365, _ = agg.get(geoid, (0, 0, 0, None))
                        label = geoid
                        st = ""
                        try:
                            row = db.session.execute(
                                text("SELECT namelsad, stusps FROM counties WHERE geoid = :g LIMIT 1"),
                                {"g": geoid},
                            ).mappings().first()
                            if row:
                                label = row.get("namelsad") or label
                                st = row.get("stusps") or ""
                        except Exception:
                            pass
                        followed_counties.append(
                            {"geoid": geoid, "label": label, "st": st, "c7": c7, "c30": c30, "c365": c365, "ovr": median_overage_by_geoid.get(geoid, "—")}
                        )
        except Exception:
            trending_counties = []
            followed_counties = []
            followed_geoids = set()
        return render_template(

            "home_feed.html",
            reports=reports,
            pagination=pagination,
            hide_anon=hide_anon,
            pct_a=pct_a,
            pct_b=pct_b,
            enforced_a=enforced_a,
            enforced_b=enforced_b,
            a_counts=a_counts,
            user_ticket_counts=user_ticket_counts,
            state_ticket_counts=state_ticket_counts,
            county_ticket_counts=county_ticket_counts,
            like_counts=like_counts,
            user_liked=user_liked,
            comments_by_report=comments_by_report,
            comment_threads_by_report=comment_threads_by_report,
            delete_comment_form=delete_comment_form,
            like_form=like_form,
            comment_form=comment_form,
            state_filter_options=state_filter_options,
            filter_state=filter_state,
            filter_road=filter_road,
            filter_county_geoid=filter_county_geoid,
            filter_county_label=filter_county_label,
            speed_limit_buckets=speed_limit_buckets,
            filter_speed_limit_list=filter_speed_limit_list,
            over_buckets=over_buckets,
            filter_over_list=filter_over_list,
            filter_photo_only=filter_photo_only,
            verify_options=verify_options,
            filter_verify=filter_verify,
            filter_pin=filter_pin,
            date_options=date_options,
            filter_date=filter_date,
            road_suggestions=road_suggestions,
            filter_sort=filter_sort,
            filters_active=filters_active,
            maps_api_key=app.config.get("GOOGLE_MAPS_API_KEY") or "",
            map_pins=map_pins,
            trending_counties=trending_counties,
            followed_counties=followed_counties,
            followed_geoids=followed_geoids,
            is_admin=is_admin,
            show_deleted=show_deleted,
            deleted_mode=deleted_mode,
            deleted_only=deleted_only,
        )


    @app.get("/map")
    def map_view():
        """Map view of tickets that have lat/lng."""
        filter_county_geoid, filter_county_label = _get_county_filter_from_request()
        hide_anon_requested = (request.args.get("hide_anon") == "1")
        if hide_anon_requested and not current_user.is_authenticated:
            flash("You must be logged in to hide anonymous posts.", "info")

        hide_anon = bool(current_user.is_authenticated and hide_anon_requested)

        is_admin = is_admin_user(current_user)
        deleted_mode = (request.args.get("deleted") or "hide").strip().lower()
        if not is_admin:
            deleted_mode = "hide"
        if deleted_mode not in ("hide", "include", "only"):
            deleted_mode = "hide"
        show_deleted = bool(is_admin and deleted_mode in ("include", "only"))
        deleted_only = bool(is_admin and deleted_mode == "only")

        # --- Filter parsing matches /home (so links are interchangeable) ---
        filter_state = (request.args.get("state") or "").strip().upper()
        filter_road = (request.args.get("road") or "").strip()

        filter_speed_limit_list = [v.strip() for v in request.args.getlist("speed_limit") if (v or "").strip()]
        filter_speed_limit_list = [v for v in filter_speed_limit_list if v != "any"]
        filter_speed_limit_list = list(dict.fromkeys(filter_speed_limit_list))

        filter_over_list = [v.strip() for v in request.args.getlist("overage") if (v or "").strip()]
        legacy_over = (request.args.get("over") or "").strip()
        if legacy_over and legacy_over not in ("all", ""):
            filter_over_list = list(dict.fromkeys(filter_over_list + [legacy_over]))

        filter_photo_only = (request.args.get("photo_only") == "1")
        filter_verify = (request.args.get("verify") or "any").strip()
        filter_pin = (request.args.get("pin") == "1")
        filter_date = (request.args.get("date") or "any").strip()
        filter_sort = (request.args.get("sort") or "new").strip()

        # Back-compat: older UI used 'most_strict'/'least_strict' labels for overage sorting
        if filter_sort == "most_strict":
            filter_sort = "least_over"
        elif filter_sort == "least_strict":
            filter_sort = "most_over"

        if request.args.get("verified_photo") == "1":
            filter_photo_only = True
            filter_verify = "verified"

        # Options for filter rail (mirrors home).
        state_filter_options = []
        try:
            rows = db.session.query(SpeedReport.state).distinct().all()
        except Exception:
            try:
                db.session.rollback()
            except Exception:
                pass
            rows = []
        seen = set()
        for (sv,) in rows:
            if not sv:
                continue
            code = sv.split(" - ")[0].strip().upper()
            if len(code) != 2 or not code.isalpha():
                continue
            if code in seen:
                continue
            seen.add(code)
            state_filter_options.append({"code": code})
        state_filter_options.sort(key=lambda d: d["code"])

        speed_limit_buckets = [
            {"value": "any", "label": "Any"},
            {"value": "25-35", "label": "25–35 mph"},
            {"value": "40-50", "label": "40–50 mph"},
            {"value": "55", "label": "55 mph"},
            {"value": "65", "label": "65 mph"},
            {"value": "70+", "label": "70+ mph"},
        ]

        over_buckets = [
            {"value": "all", "label": "Any"},
            {"value": "5-9", "label": "5–9 mph"},
            {"value": "10-14", "label": "10–14 mph"},
            {"value": "15-19", "label": "15–19 mph"},
            {"value": "20+", "label": "20+ mph"},
        ]

        date_options = [
            {"value": "any", "label": "Any"},
            {"value": "7", "label": "Last 7 days"},
            {"value": "30", "label": "Last 30 days"},
            {"value": "90", "label": "Last 90 days"},
            {"value": "365", "label": "Last year"},
        ]

        verify_options = [
            {"value": "any", "label": "Any"},
            {"value": "verified", "label": "Auto-Extracted"},
            {"value": "not_verified", "label": "Not auto-extracted"},
        ]

        filters_active = bool(
            filter_state
            or filter_county_geoid
            or filter_road
            or bool(filter_speed_limit_list)
            or bool(filter_over_list)
            or filter_pin
            or filter_photo_only
            or (filter_verify not in ("", "any"))
            or (filter_date not in ("", "any"))
            or (deleted_mode not in ("", "hide"))
            or hide_anon
        )

        return render_template(
            "map_page.html",
            hide_anon=hide_anon,
            is_admin=is_admin,
            deleted_mode=deleted_mode,
            state_filter_options=state_filter_options,
            filter_state=filter_state,
            filter_road=filter_road,
            filter_county_geoid=filter_county_geoid,
            filter_county_label=filter_county_label,
            speed_limit_buckets=speed_limit_buckets,
            filter_speed_limit_list=filter_speed_limit_list,
            over_buckets=over_buckets,
            filter_over_list=filter_over_list,
            filter_photo_only=filter_photo_only,
            verify_options=verify_options,
            filter_verify=filter_verify,
            filter_pin=filter_pin,
            date_options=date_options,
            filter_date=filter_date,
            filter_sort=filter_sort,
            filters_active=filters_active,
            maps_api_key=app.config.get("GOOGLE_MAPS_API_KEY") or "",
        )


    def _admin_email_set() -> set[str]:
        raw = (os.environ.get("ADMIN_EMAILS") or os.environ.get("ADMIN_EMAIL") or "").strip()
        if not raw:
            return set()
        parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
        return set(parts)

    def is_admin_user(user) -> bool:
        try:
            if not user or not getattr(user, "is_authenticated", False):
                return False
            admin_emails = _admin_email_set()
            if not admin_emails:
                return False
            email = (getattr(user, "email", "") or "").strip().lower()
            return bool(email and email in admin_emails)
        except Exception:
            return False

    def _require_admin():
        if not is_admin_user(current_user):
            abort(403)


    def _safe_next_url(fallback_endpoint: str = "home", **fallback_kwargs) -> str:
        """Return a safe relative next url from query/form, else fall back to an endpoint."""
        nxt = (request.args.get("next") or request.form.get("next") or "").strip()
        if nxt.startswith("/") and not nxt.startswith("//"):
            return nxt
        return url_for(fallback_endpoint, **fallback_kwargs)


    @app.post("/admin/report/<int:report_id>/delete")
    @login_required
    def admin_delete_report(report_id: int):
        _require_admin()
        r = SpeedReport.query.get_or_404(report_id)
        if not getattr(r, "is_deleted", False):
            r.is_deleted = True
            r.deleted_at = datetime.utcnow()
            r.deleted_by = int(current_user.id)
            db.session.commit()
            flash("Ticket soft-deleted.", "info")
        return redirect(_safe_next_url())


    @app.post("/admin/report/<int:report_id>/restore")
    @login_required
    def admin_restore_report(report_id: int):
        _require_admin()
        r = SpeedReport.query.get_or_404(report_id)
        if getattr(r, "is_deleted", False):
            r.is_deleted = False
            r.deleted_at = None
            r.deleted_by = None
            db.session.commit()
            flash("Ticket restored.", "info")
        return redirect(_safe_next_url())


    @app.post("/like/<int:report_id>")
    def toggle_like(report_id: int):
        # Members-only
        if not current_user.is_authenticated:
            flash("Members only.", "info")
            return redirect(_safe_next_url())

        # Validate CSRF via FlaskForm
        form = LikeForm()
        if not form.validate_on_submit():
            flash("Something went wrong. Please try again.", "error")
            return redirect(_safe_next_url())

        report = SpeedReport.query.get_or_404(report_id)
        existing = Like.query.filter_by(report_id=report.id, user_id=current_user.id).first()
        if existing:
            db.session.delete(existing)
        else:
            db.session.add(Like(report_id=report.id, user_id=current_user.id))

        db.session.commit()
        return redirect(_safe_next_url())


    @app.post("/comment/<int:report_id>")
    def add_comment(report_id: int):
        # Members-only
        if not current_user.is_authenticated:
            flash("Members only.", "info")
            return redirect(_safe_next_url())

        form = CommentForm()
        if not form.validate_on_submit():
            # Show the first validation error if present
            err = None
            if form.body.errors:
                err = form.body.errors[0]
            flash(err or "Please enter a valid comment.", "error")
            return redirect(_safe_next_url())

        body = (form.body.data or "").strip()
        if not body:
            flash("Comment can’t be empty.", "error")
            return redirect(_safe_next_url())

        # Lightweight anti-abuse: max 5 comments per 60 seconds per user
        window_start = datetime.utcnow() - timedelta(seconds=60)
        recent_count = (
            Comment.query.filter(Comment.user_id == current_user.id, Comment.created_at >= window_start)
            .count()
        )
        if recent_count >= 5:
            flash("Please slow down for a moment.", "info")
            return redirect(_safe_next_url())

        report = SpeedReport.query.get_or_404(report_id)

        parent_id = None
        raw_parent = (request.form.get("parent_id") or "").strip()
        if raw_parent:
            try:
                parent_id = int(raw_parent)
            except ValueError:
                parent_id = None

        if parent_id:
            parent = Comment.query.get(parent_id)
            if (not parent) or (parent.report_id != report.id):
                flash("That comment no longer exists.", "error")
                return redirect(_safe_next_url())
            # Allow replying to replies (nested threads). We only validate that the parent
            # belongs to the same report above.

        c = Comment(report_id=report.id, user_id=current_user.id, body=body, parent_id=parent_id)
        db.session.add(c)
        db.session.commit()

        # After posting, return to the normal home view (comment/reply boxes closed by default).
        return redirect(_safe_next_url())


    @app.post("/comment/<int:comment_id>/delete")
    def delete_comment(comment_id: int):
        # Members-only
        if not current_user.is_authenticated:
            flash("Members only.", "info")
            return redirect(_safe_next_url())

        c = Comment.query.get_or_404(comment_id)
        if c.user_id != current_user.id:
            flash("You can only delete your own comments.", "error")
            return redirect(_safe_next_url())

        # Delete descendants (supports nested replies)
        to_delete = [c.id]
        all_ids = []
        while to_delete:
            cur_id = to_delete.pop()
            all_ids.append(cur_id)
            child_ids = [cid for (cid,) in db.session.query(Comment.id).filter(Comment.parent_id == cur_id).all()]
            to_delete.extend(child_ids)

        # Delete children first, then the root (reverse topological order)
        for cid in reversed(all_ids):
            obj = Comment.query.get(cid)
            if obj is not None:
                db.session.delete(obj)
        db.session.commit()

        flash("Comment deleted.", "info")
        return redirect(_safe_next_url())


    # Register /submit and related routes on the app instance

    @app.get("/submit", endpoint="submit")
    @login_required
    def submit_get():
        """Single-page submit. County required. Pin optional."""
        form = SubmitTicketForm()

        maps_api_key = app.config.get("GOOGLE_MAPS_API_KEY") or ""

        # Default: no state selected unless explicitly prefilled via ?st=XX (optional).
        default_st = (request.args.get("st") or "").strip().upper()
        if default_st and default_st not in STATE_BY_ABBR:
            default_st = ""
        if default_st:
            form.state.data = default_st
        return render_template(
            "submit_ticket.html",
            form=form,
            states=STATE_PAIRS,
            default_st=default_st,
            maps_api_key=maps_api_key,
        )


    @app.post("/submit")
    @login_required
    def submit_post():
        form = SubmitTicketForm()

        maps_api_key = app.config.get("GOOGLE_MAPS_API_KEY") or ""

        # Preserve what the user picked (or blank if none yet) for re-render on errors.
        default_st = (form.state.data or "").strip().upper()
        if default_st and default_st not in STATE_BY_ABBR:
            default_st = ""

        if not form.validate_on_submit():
            flash("Please fix the highlighted fields.", "error")
            return render_template("submit_ticket.html", form=form, states=STATE_PAIRS, default_st=default_st, maps_api_key=maps_api_key), 400

        st = (form.state.data or "").strip().upper()
        if st not in STATE_BY_ABBR:
            flash("Select a valid state.", "error")
            return render_template("submit_ticket.html", form=form, states=STATE_PAIRS, default_st=default_st, maps_api_key=maps_api_key), 400

        county_geoid = (form.county_geoid.data or "").strip()
        if not county_geoid:
            flash("County is required.", "error")
            return render_template("submit_ticket.html", form=form, states=STATE_PAIRS, default_st=default_st, maps_api_key=maps_api_key), 400

        # Enforce 2 tickets per calendar month (non-admin)
        if current_user.is_authenticated and not is_admin_user(current_user):
            from datetime import datetime
            now = datetime.utcnow()
            month_start = datetime(now.year, now.month, 1)
            submitted_this_month = (
                db.session.query(SpeedReport)
                .filter(SpeedReport.user_id == current_user.id)
                .filter(SpeedReport.created_at >= month_start)
                .count()
            )
            if submitted_this_month >= 2:
                flash("Limit reached: 2 tickets per month.", "error")
                return render_template("submit_ticket.html", form=form, states=STATE_PAIRS, default_st=default_st, maps_api_key=maps_api_key), 400

        # Fetch canonical county record (also enforces county belongs to state)
        row = db.session.execute(
            text(
                """
                SELECT geoid, name, namelsad, stusps
                FROM counties
                WHERE geoid = :geoid AND stusps = :st
                LIMIT 1
                """
            ),
            {"geoid": county_geoid, "st": st},
        ).mappings().first()

        if not row:
            flash("Selected county is invalid for that state.", "error")
            return render_template("submit_ticket.html", form=form, states=STATE_PAIRS, default_st=default_st, maps_api_key=maps_api_key), 400

        county_name = row["namelsad"] or row["name"]

        # Optional pin
        lat = (getattr(form, 'lat', None).data or "") if getattr(form, 'lat', None) is not None else ""
        lng = (getattr(form, 'lng', None).data or "") if getattr(form, 'lng', None) is not None else ""
        lat = (lat or "").strip()
        lng = (lng or "").strip()
        # Backward compatibility: accept legacy field names if present
        if not lat and getattr(form, 'latitude', None) is not None:
            lat = (form.latitude.data or "").strip()
        if not lng and getattr(form, 'longitude', None) is not None:
            lng = (form.longitude.data or "").strip()
        lat_f = lng_f = None
        if lat or lng:
            try:
                lat_f = float(lat)
                lng_f = float(lng)
            except Exception:
                flash("Invalid pin coordinates.", "error")
                return render_template("submit_ticket.html", form=form, states=STATE_PAIRS, default_st=default_st, maps_api_key=maps_api_key), 400

            inside = db.session.execute(
                text(
                    """
                    SELECT CASE
                        WHEN geom IS NULL THEN NULL
                        ELSE ST_Covers(
                            geom,
                            ST_SetSRID(ST_MakePoint(:lng, :lat), 4326)
                        )
                    END AS inside
                    FROM counties
                    WHERE geoid = :geoid
                    LIMIT 1
                    """
                ),
                {"geoid": county_geoid, "lat": lat_f, "lng": lng_f},
            ).scalar()

            if inside is False:
                flash("Pin must be inside the selected county.", "error")
                return render_template("submit_ticket.html", form=form, states=STATE_PAIRS, default_st=default_st, maps_api_key=maps_api_key), 400

        # Save report
        posted = int(form.posted_speed.data)
        ticketed = int(form.ticketed_speed.data)
        # If user entered them reversed (e.g., 84 as posted and 65 as ticketed), normalize.
        if ticketed < posted:
            posted, ticketed = ticketed, posted
        overage = max(0, ticketed - posted)

        state_full = STATE_BY_ABBR.get(st, st)

        report = SpeedReport(
            user_id=current_user.id,
            state=state_full,
            route_class=None,
            road_name=form.road_name.data.strip(),
            posted_speed=posted,
            ticketed_speed=ticketed,
            overage=overage,
            caption=(form.caption.data or "").strip() or None,
            raw_lat=lat_f,
            raw_lng=lng_f,
            lat=lat_f,
            lng=lng_f,
            location_hint=None,
            location_source=("user_pin" if lat_f is not None else "none"),
            location_accuracy_m=None,
            # Hard defaults to satisfy NOT NULL constraints on evolved schemas
            # (older DBs may have NOT NULL without DEFAULT).
            ocr_status='none',
            verification_status='none',
            verify_attempts=0,
            is_deleted=False,
            county_geoid=county_geoid,
            county_name=county_name,
            county_state=st,
        )

        # Keep grouping/filter key in sync with the shared normalization logic.
        try:
            report.refresh_road_key()
        except Exception:
            pass

        # Photo (optional)
        if form.photo.data:
            # Mark that a photo was submitted immediately, even before upload completes.
            report.photo_key = f"pending:{uuid.uuid4().hex}"
            report.ocr_status = "processing"
            report.verification_status = "submitted"
        else:
            # No photo submitted
            report.ocr_status = "none"
            report.verification_status = "none"

        db.session.add(report)
        db.session.commit()

        # Kick off photo upload + OCR/verification if a photo was included
        if form.photo.data:
            try:
                # Sanitize + standardize the uploaded image (strip metadata, normalize format)
                img = Image.open(form.photo.data.stream)
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                # Resize down to a safe max edge (keeps OCR costs predictable)
                max_edge = 2000
                w, h = img.size
                if max(w, h) > max_edge:
                    scale = max_edge / float(max(w, h))
                    img = img.resize((int(w * scale), int(h * scale)))

                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85, optimize=True)
                buf.seek(0)

                # Bucket name: prefer dedicated tickets bucket; fall back to legacy quarantine bucket name
                bucket = (
                    os.environ.get("R2_TICKETS_BUCKET")
                    or os.environ.get("R2_QUARANTINE_BUCKET")
                    or os.environ.get("R2_BUCKET")
                )
                if not bucket:
                    raise RuntimeError("R2_TICKETS_BUCKET (or legacy R2_QUARANTINE_BUCKET / R2_BUCKET) is not set")

                photo_key = f"tickets/{report.id}/{uuid.uuid4().hex}.jpg"
                put_bytes(bucket, photo_key, buf.read(), content_type="image/jpeg")

                # Persist the real key and enqueue OCR/verification
                report.photo_key = photo_key
                report.ocr_status = "processing"
                report.verification_status = "submitted"
                report.ocr_error = None
                report.verify_reason = None
                db.session.commit()

                q = get_queue()
                job = q.enqueue(
                    "ocr_verify.verify_ticket_from_r2",
                    report.id,
                    bucket,
                    photo_key,
                    job_timeout=120,
                )
                report.ocr_job_id = job.id
                db.session.commit()

            except Exception:
                app.logger.exception("Photo/OCR enqueue failed")
                report.ocr_status = "failed"
                # Keep verification in 'pending' so UI can show status pill clearly
                report.verification_status = "pending"
                # Preserve the fact a photo was submitted even if upload/enqueue fails.
                if report.photo_key and str(report.photo_key).startswith("pending:"):
                    report.photo_key = f"failed:{str(report.photo_key).split(':',1)[1]}"
                else:
                    report.photo_key = f"failed:{uuid.uuid4().hex}"
                report.ocr_error = "enqueue_failed"
                report.verify_reason = "upload_failed"
                db.session.commit()
                flash("Ticket saved, but photo processing failed.", "error")

        flash("Ticket submitted!", "success")
        return redirect(url_for("home"))

    @app.post("/api/confirm_location")
    def api_confirm_location():
        """Snap pin -> road name (server-side) for consistent web/mobile behavior."""
        try:
            payload = request.get_json(silent=True) or {}
        except Exception:
            payload = {}

        try:
            raw_lat = float(payload.get("lat"))
            raw_lng = float(payload.get("lng"))
        except Exception:
            return jsonify({"ok": False, "error": "bad_lat_lng"}), 400

        # New: route_class selection (web + mobile parity)
        route_class = payload.get('route_class')
        if route_class is None:
            route_class = payload.get('routeClass')
        route_class = (str(route_class).strip().lower() if route_class is not None else '')

        # Backward compatibility: older clients send prefer_highway Yes/No.
        prefer_raw = payload.get('prefer_highway')
        if prefer_raw is None:
            prefer_raw = payload.get('preferHighway')
        prefer_s = str(prefer_raw).strip().lower() if prefer_raw is not None else ''
        prefer_highway = prefer_raw is True or prefer_s in ('1', 'true', 'yes', 'y', 'on')

        snapped_lat = raw_lat
        snapped_lng = raw_lng
        road = None
        snapped_source = 'nearest'

        # Normalize older route_class values (us_route/other) to the new buckets.
        if route_class == 'us_route':
            route_class = 'numbered'
        elif route_class == 'other':
            route_class = 'local'

        if route_class in ('interstate', 'numbered', 'local'):
            best = _confirm_location_route_class(raw_lat, raw_lng, route_class=route_class, max_radius_m=2500)
            if not best:
                err = {
                    'interstate': ('no_interstate_found', 'Could not find an Interstate near that pin. Zoom out and drop closer to the Interstate.'),
                    'numbered': ('no_numbered_found', 'Could not find a numbered route/highway near that pin. Zoom out and drop closer to the route.'),
                    'local': ('no_local_found', 'Could not find a street/road (non-interstate, non-numbered route) near that pin. Move the pin, or pick a different road type.'),
                }
                code, msg = err.get(route_class, ('no_match', 'Could not match that road type near the pin.'))
                return jsonify({"ok": False, "error": code, "message": msg}), 400

            snapped_lat = float(best['snapped_lat'])
            snapped_lng = float(best['snapped_lng'])
            road = best.get('road_name')
            snapped_source = f'route_class:{route_class}'

        elif prefer_highway:
            # Old behavior (Yes/No): try to find a major route-ish road.
            best = _confirm_location_prefer_highway(raw_lat, raw_lng, max_radius_m=2500)
            if not best:
                return jsonify({
                    "ok": False,
                    "error": "no_highway_found",
                    "message": "Could not find a major route near that pin. Zoom out and drop closer to the highway/route.",
                }), 400

            snapped_lat = float(best['snapped_lat'])
            snapped_lng = float(best['snapped_lng'])
            road = best.get('road_name')
            snapped_source = 'prefer_highway'

        if not road:
            slat, slng = snap_to_nearest_road(raw_lat, raw_lng)
            snapped_lat = float(slat) if slat is not None else raw_lat
            snapped_lng = float(slng) if slng is not None else raw_lng
            road = _reverse_geocode_route(snapped_lat, snapped_lng)

        road = road or "Map pin"

        return jsonify({
            "ok": True,
            "raw_lat": raw_lat,
            "raw_lng": raw_lng,
            "snapped_lat": snapped_lat,
            "snapped_lng": snapped_lng,
            "road_name": road,
            "snapped_source": snapped_source,
            "route_class": route_class,
            "prefer_highway": bool(prefer_highway),
        })

    @app.post("/api/update_ticket_pin")
    @login_required
    def api_update_ticket_pin():
        """Owner-only: update an existing ticket's pin (raw + snapped lat/lng)."""
        try:
            payload = request.get_json(silent=True) or {}
        except Exception:
            payload = {}

        try:
            report_id = int(payload.get("report_id"))
        except Exception:
            return jsonify({"ok": False, "error": "bad_report_id"}), 400

        try:
            lat = float(payload.get("lat"))
            lng = float(payload.get("lng"))
        except Exception:
            return jsonify({"ok": False, "error": "bad_lat_lng"}), 400

        r = SpeedReport.query.get(report_id)
        if not r:
            return jsonify({"ok": False, "error": "not_found"}), 404

        # Only the ticket owner can refine their own pin.
        if not r.user_id or int(r.user_id) != int(current_user.id):
            return jsonify({"ok": False, "error": "forbidden"}), 403

        r.raw_lat = lat
        r.raw_lng = lng

        slat, slng = snap_to_nearest_road(lat, lng)
        if slat is None or slng is None:
            r.lat = lat
            r.lng = lng
            r.location_source = "user_pin"
            snapped = False
        else:
            r.lat = float(slat)
            r.lng = float(slng)
            r.location_source = "user_pin_snapped"
            snapped = True

        db.session.commit()
        return jsonify({"ok": True, "snapped": snapped, "lat": r.lat, "lng": r.lng})

    

    # --- County autocomplete API ---
    @app.get("/api/counties")
    def api_counties():
        """Autocomplete counties for a given state.

        Query params:
          - st: 2-letter state code (preferred)
          - q:  user search text (prefix match on normalized county name)

        Returns:
          { ok: true, counties: [{geoid, name, namelsad, stusps, state_name}, ...] }
        """
        st = (request.args.get("st") or request.args.get("state") or "").strip()
        q = (request.args.get("q") or request.args.get("query") or "").strip()

        # Normalize state: accept "VA", "Virginia", "VA — Virginia"
        st_up = (st or "").strip().upper()
        if not st_up:
            return jsonify({"ok": True, "counties": []})

        # Extract 2-letter code if embedded
        m = re.search(r"\b([A-Za-z]{2})\b", st_up)
        if m:
            cand = m.group(1).upper()
            if cand in STATE_BY_ABBR:
                st_up = cand

        # If still not a 2-letter code, try exact name lookup
        if st_up not in STATE_BY_ABBR:
            st_low = (st or "").strip().lower()
            st_low = re.sub(r"[^a-z\s]", " ", st_low)
            st_low = re.sub(r"\s+", " ", st_low).strip()
            found = None
            for code, name in STATE_PAIRS:
                if name.lower() == st_low:
                    found = code
                    break
            if found:
                st_up = found
            else:
                return jsonify({"ok": True, "counties": []})

        # Normalize query to match name_norm built by import script
        qn = (q or "").lower()
        qn = re.sub(r"[^a-z0-9\s]", " ", qn)
        qn = re.sub(r"\s+", " ", qn).strip()
        if len(qn) < 2:
            return jsonify({"ok": True, "counties": []})

        try:
            rows = db.session.execute(
                text(
                    """
                    SELECT geoid, name, namelsad, stusps, state_name
                    FROM counties
                    WHERE stusps = :st
                      AND name_norm LIKE :pat
                    ORDER BY length(name_norm) ASC, namelsad ASC
                    LIMIT 25
                    """
                ),
                {"st": st_up, "pat": qn + "%"},
            ).mappings().all()

            # Fallback: contains match if prefix yields nothing
            if not rows:
                rows = db.session.execute(
                    text(
                        """
                        SELECT geoid, name, namelsad, stusps, state_name
                        FROM counties
                        WHERE stusps = :st
                          AND name_norm LIKE :pat
                        ORDER BY length(name_norm) ASC, namelsad ASC
                        LIMIT 25
                        """
                    ),
                    {"st": st_up, "pat": "%" + qn + "%"},
                ).mappings().all()

            return jsonify({"ok": True, "counties": [dict(r) for r in rows]})
        except Exception as e:
            try:
                db.session.rollback()
            except Exception:
                pass
            try:
                print(f"[COUNTY API ERROR] {e}")
            except Exception:
                pass
            return jsonify({"ok": False, "error": "counties_unavailable", "counties": []}), 200

    @app.get("/api/username_available")
    def api_username_available():
        u = (request.args.get("u") or "").strip()
        if not u or len(u) < 3:
            return jsonify({"ok": True, "available": False, "message": "Enter at least 3 characters."})

        # mirror RegisterForm rules
        if len(u) > 20:
            return jsonify({"ok": True, "available": False, "message": "Keep it 20 characters or fewer."})

        import re as _re
        if not _re.match(r"^[A-Za-z0-9_]+$", u):
            return jsonify({"ok": True, "available": False, "message": "Only letters, numbers, and underscores."})

        taken = User.query.filter(func.lower(User.username) == u.lower()).first() is not None
        if taken:
            return jsonify({"ok": True, "available": False, "message": "That username is taken."})
        return jsonify({"ok": True, "available": True, "message": "Username is available."})



    @app.get("/api/county/<geoid>")
    def api_county(geoid: str):
        """Lookup a single county by GEOID."""
        geoid = (geoid or "").strip()
        if not geoid:
            return jsonify({"ok": False, "error": "bad_geoid"}), 400
        try:
            row = db.session.execute(
                text(
                    """
                    SELECT geoid, name, namelsad, stusps, state_name
                    FROM counties
                    WHERE geoid = :geoid
                    LIMIT 1
                    """
                ),
                {"geoid": geoid},
            ).mappings().first()
            if not row:
                return jsonify({"ok": False, "error": "not_found"}), 404
            return jsonify({"ok": True, "county": dict(row)})
        except Exception as e:
            try:
                print(f"[COUNTY API ERROR] {e}")
            except Exception:
                pass
            return jsonify({"ok": False, "error": "counties_unavailable"}), 200


    @app.get("/api/county_geom/<geoid>")
    def api_county_geom(geoid: str):
        """Return GeoJSON geometry for a county (for the web submit pin picker)."""

        geoid = (geoid or "").strip()
        if not geoid:
            return jsonify({"ok": False, "error": "bad_geoid"}), 400

        try:
            row = db.session.execute(
                text(
                    """
                    SELECT
                        geoid,
                        name,
                        namelsad,
                        stusps,
                        state_name,
                        ST_AsGeoJSON(geom) AS geom_json,
                        ST_Y(COALESCE(center, centroid, ST_PointOnSurface(geom))) AS centroid_lat,
                        ST_X(COALESCE(center, centroid, ST_PointOnSurface(geom))) AS centroid_lng
                    FROM counties
                    WHERE geoid = :geoid
                    LIMIT 1
                    """
                ),
                {"geoid": geoid},
            ).mappings().first()

            if not row or not row.get("geom_json"):
                return jsonify({"ok": False, "error": "not_found"}), 404

            try:
                geom = json.loads(row["geom_json"])
            except Exception:
                geom = None
            if not geom:
                return jsonify({"ok": False, "error": "geom_unavailable"}), 200

            feature = {
                "type": "Feature",
                "properties": {
                    "geoid": row.get("geoid"),
                    "name": row.get("name"),
                    "namelsad": row.get("namelsad"),
                    "stusps": row.get("stusps"),
                    "state_name": row.get("state_name"),
                },
                "geometry": geom,
            }

            return jsonify(
                {
                    "ok": True,
                    "geoid": row.get("geoid"),
                    "centroid": {
                        "lat": float(row.get("centroid_lat")) if row.get("centroid_lat") is not None else None,
                        "lng": float(row.get("centroid_lng")) if row.get("centroid_lng") is not None else None,
                    },
                    "geojson": {"type": "FeatureCollection", "features": [feature]},
                }
            )
        except Exception as e:
            try:
                print(f"[COUNTY GEOM API ERROR] {e}")
            except Exception:
                pass
            return jsonify({"ok": False, "error": "counties_unavailable"}), 200
    @app.get("/api/map_pins")
    def api_map_pins():
        """Return map pins filtered like the feed, but only tickets with lat/lng."""
        hide_anon_requested = (request.args.get("hide_anon") == "1")
        hide_anon = bool(current_user.is_authenticated and hide_anon_requested)

        is_admin = is_admin_user(current_user)
        deleted_mode = (request.args.get("deleted") or "hide").strip().lower()
        if not is_admin:
            deleted_mode = "hide"
        if deleted_mode not in ("hide", "include", "only"):
            deleted_mode = "hide"
        show_deleted = bool(is_admin and deleted_mode in ("include", "only"))
        deleted_only = bool(is_admin and deleted_mode == "only")

        # --- same filter semantics as /home ---
        filter_state = (request.args.get("state") or "").strip().upper()
        filter_road = (request.args.get("road") or "").strip()
        filter_county_geoid, _ = _get_county_filter_from_request()

        # Focus params (used for inside/outside coloring without forcing a filter)
        focus_state = (request.args.get("focus_state") or "").strip().upper()
        focus_county_geoid = (request.args.get("focus_county_geoid") or "").strip()
        selected_ticket_id = (request.args.get("ticket_id") or "").strip()

        # Optional viewport bounds (when present, filter pins to the visible map)
        def _f(name: str):
            v = (request.args.get(name) or "").strip()
            if not v:
                return None
            try:
                return float(v)
            except Exception:
                return None

        north = _f("north")
        south = _f("south")
        east = _f("east")
        west = _f("west")
        viewport_mode = (north is not None and south is not None and east is not None and west is not None)


        filter_speed_limit_list = [v.strip() for v in request.args.getlist("speed_limit") if (v or "").strip()]
        filter_speed_limit_list = [v for v in filter_speed_limit_list if v != "any"]
        filter_speed_limit_list = list(dict.fromkeys(filter_speed_limit_list))

        filter_over_list = [v.strip() for v in request.args.getlist("overage") if (v or "").strip()]
        legacy_over = (request.args.get("over") or "").strip()
        if legacy_over and legacy_over not in ("all", ""):
            filter_over_list = list(dict.fromkeys(filter_over_list + [legacy_over]))

        filter_photo_only = (request.args.get("photo_only") == "1")
        filter_verify = (request.args.get("verify") or "any").strip()
        filter_pin = (request.args.get("pin") == "1")
        filter_date = (request.args.get("date") or "any").strip()
        filter_sort = (request.args.get("sort") or "new").strip()

        # Back-compat: older UI used 'most_strict'/'least_strict' labels for overage sorting
        if filter_sort == "most_strict":
            filter_sort = "least_over"
        elif filter_sort == "least_strict":
            filter_sort = "most_over"
        if request.args.get("verified_photo") == "1":
            filter_photo_only = True
            filter_verify = "verified"

        q = SpeedReport.query.options(joinedload(SpeedReport.user))

        # Deleted visibility (admin: ?deleted=hide|include|only)
        if deleted_only:
            q = q.filter(SpeedReport.is_deleted.is_(True))
        elif deleted_only:
            q = q.filter(SpeedReport.is_deleted.is_(True))
        elif not show_deleted:
            q = q.filter(SpeedReport.is_deleted.is_(False))

        # Only pinned tickets
        q = q.filter(SpeedReport.lat.isnot(None)).filter(SpeedReport.lng.isnot(None))

        if hide_anon:
            q = q.filter(SpeedReport.user_id.isnot(None))

        if filter_state:
            q = q.filter(SpeedReport.state.ilike(f"{filter_state}%"))

        # County filter (GEOID)
        if filter_county_geoid:
            q = q.filter(SpeedReport.county_geoid == filter_county_geoid)

        # Map pin filter (only tickets with a user-placed pin)
        if filter_pin:
            q = q.filter(SpeedReport.location_source.in_(("user_pin", "user_pin_snapped")))

        if filter_date not in ("", "any"):
            try:
                days = int(filter_date)
            except Exception:
                days = 0
            if days > 0:
                q = q.filter(SpeedReport.created_at >= (datetime.utcnow() - timedelta(days=days)))

        if filter_road:
            try:
                road_key = normalize_road(filter_road, filter_state)
            except Exception:
                road_key = ""
            conds = [SpeedReport.road_name.ilike(f"%{filter_road}%")]
            if road_key:
                conds.append(SpeedReport.road_key == road_key)
            q = q.filter(or_(*conds))

        if filter_speed_limit_list:
            sl_conds = []
            for v in filter_speed_limit_list:
                if v == "lte35":
                    sl_conds.append(SpeedReport.posted_speed <= 35)
                elif v == "40-55":
                    sl_conds.append(and_(SpeedReport.posted_speed >= 40, SpeedReport.posted_speed <= 55))
                elif v == "gte60":
                    sl_conds.append(SpeedReport.posted_speed >= 60)
                elif v == "25-35":
                    sl_conds.append(and_(SpeedReport.posted_speed >= 25, SpeedReport.posted_speed <= 35))
                elif v == "40-50":
                    sl_conds.append(and_(SpeedReport.posted_speed >= 40, SpeedReport.posted_speed <= 50))
                elif v == "55":
                    sl_conds.append(SpeedReport.posted_speed == 55)
                elif v == "65":
                    sl_conds.append(SpeedReport.posted_speed == 65)
                elif v == "70+":
                    sl_conds.append(SpeedReport.posted_speed >= 70)
            if sl_conds:
                q = q.filter(or_(*sl_conds))

        expr = (SpeedReport.ticketed_speed - SpeedReport.posted_speed)
        if filter_over_list:
            over_conds = []
            for v in filter_over_list:
                if v == "1-10":
                    over_conds.append(and_(expr >= 1, expr <= 10))
                elif v == "11-20":
                    over_conds.append(and_(expr >= 11, expr <= 20))
                elif v == "21+":
                    over_conds.append(expr >= 21)
                elif v == "5-9":
                    over_conds.append(and_(expr >= 5, expr <= 9))
                elif v == "10-14":
                    over_conds.append(and_(expr >= 10, expr <= 14))
                elif v == "15-19":
                    over_conds.append(and_(expr >= 15, expr <= 19))
                elif v == "20+":
                    over_conds.append(expr >= 20)
            if over_conds:
                q = q.filter(or_(*over_conds))

        if filter_photo_only:
            q = q.filter(SpeedReport.photo_key.isnot(None))

        if filter_verify == "verified":
            q = q.filter(SpeedReport.ocr_status == "verified")
        elif filter_verify == "not_verified":
            q = q.filter(or_(SpeedReport.ocr_status.is_(None), SpeedReport.ocr_status != "verified"))

        # Hard cap pins returned to keep the page snappy.
        # Ordering (keep semantics aligned with the feed where possible).
        if filter_sort == "old":
            q = q.order_by(SpeedReport.created_at.asc())
        elif filter_sort == "most_over":
            q = q.order_by((SpeedReport.ticketed_speed - SpeedReport.posted_speed).desc(), SpeedReport.created_at.desc())
        elif filter_sort in ("least_over", "least_strict"):
            q = q.order_by((SpeedReport.ticketed_speed - SpeedReport.posted_speed).asc(), SpeedReport.created_at.desc())
        else:
            q = q.order_by(SpeedReport.created_at.desc())

        rows = q.limit(1500).all()

        # Determine focus for inside/outside coloring.
        focus_type = None
        focus_value = None
        if focus_county_geoid:
            focus_type = "county"
            focus_value = focus_county_geoid
        elif focus_state and len(focus_state) == 2:
            focus_type = "state"
            focus_value = focus_state
        elif filter_county_geoid:
            focus_type = "county"
            focus_value = filter_county_geoid
        elif filter_state and len(filter_state) == 2:
            focus_type = "state"
            focus_value = filter_state

        selected_id_norm = None
        try:
            if selected_ticket_id:
                selected_id_norm = str(int(selected_ticket_id))
        except Exception:
            selected_id_norm = None

        pins = []
        for r in rows:
            inside_focus = True
            try:
                if focus_type == "county" and focus_value:
                    inside_focus = (r.county_geoid == focus_value)
                elif focus_type == "state" and focus_value:
                    inside_focus = (normalize_state_group(r.state) == focus_value)
            except Exception:
                inside_focus = True

            pins.append(
                {
                    "id": r.id,
                    "lat": r.lat,
                    "lng": r.lng,
                    "state": normalize_state_group(r.state),
                    "road": r.road_name,
                    "posted": r.posted_speed,
                    "ticketed": r.ticketed_speed,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "username": (r.user.username if getattr(r, "user", None) else None),
                    "inside_focus": bool(inside_focus),
                }
            )

        return jsonify({"ok": True, "count": len(pins), "pins": pins, "selected_id": selected_id_norm})



    @app.get("/api/state_geom/<stusps>")
    def api_state_geom(stusps: str):
        """Return GeoJSON geometry for a state (cached from counties)."""
        st = (stusps or "").strip().upper()
        if len(st) != 2 or not st.isalpha():
            return jsonify({"ok": False, "error": "bad_state"}), 400

        try:
            # Slight simplification for state outlines to keep payloads reasonable.
            row = db.session.execute(
                text(
                    """
                    SELECT
                        stusps,
                        state_name,
                        ST_AsGeoJSON(ST_SimplifyPreserveTopology(geom, :tol)) AS geom_json,
                        ST_Y(ST_PointOnSurface(geom)) AS centroid_lat,
                        ST_X(ST_PointOnSurface(geom)) AS centroid_lng
                    FROM states
                    WHERE stusps = :st
                    LIMIT 1
                    """
                ),
                {"st": st, "tol": 0.005},
            ).mappings().first()

            if not row or not row.get("geom_json"):
                return jsonify({"ok": False, "error": "not_found"}), 404

            try:
                geom = json.loads(row["geom_json"])
            except Exception:
                geom = None
            if not geom:
                return jsonify({"ok": False, "error": "geom_unavailable"}), 200

            feature = {
                "type": "Feature",
                "properties": {
                    "stusps": row.get("stusps"),
                    "state_name": row.get("state_name"),
                },
                "geometry": geom,
            }

            return jsonify(
                {
                    "ok": True,
                    "stusps": row.get("stusps"),
                    "centroid": {
                        "lat": float(row.get("centroid_lat")) if row.get("centroid_lat") is not None else None,
                        "lng": float(row.get("centroid_lng")) if row.get("centroid_lng") is not None else None,
                    },
                    "geojson": {"type": "FeatureCollection", "features": [feature]},
                }
            )
        except Exception as e:
            try:
                print(f"[STATE GEOM API ERROR] {e}")
            except Exception:
                pass
            return jsonify({"ok": False, "error": "states_unavailable"}), 200



    @app.get("/api/search_suggest")
    def search_suggest():
        q = (request.args.get("q") or "").strip()
        if not q:
            return jsonify([])

        q_clean = q[1:] if q.startswith("@") else q
        q_like = f"%{q_clean}%"
        q_lower = q_clean.lower()
        q_upper = q_clean.upper()

        suggestions = []
        seen = set()

        # Users (partial, case-insensitive)
        user_rows = (
            User.query
            .filter(db.func.lower(User.username).like(db.func.lower(q_like)))
            .order_by(User.username.asc())
            .limit(5)
            .all()
        )
        for u in user_rows:
            url = url_for("profile", username=u.username)
            key = ("user", u.username)
            if key in seen:
                continue
            suggestions.append({"type": "user", "label": f"@{u.username}", "url": url})
            seen.add(key)

        # States (based on states present in DB)
        state_rows = db.session.query(SpeedReport.state).distinct().all()
        state_groups = sorted({normalize_state_group(r[0]) for r in state_rows if r and r[0]})
        for st in state_groups:
            if q_upper in st or q_lower in st.lower():
                url = url_for("strictness", state=st)
                key = ("state", st)
                if key in seen:
                    continue
                suggestions.append({"type": "state", "label": st, "url": url})
                seen.add(key)

        # Roads (state + road bucket)
        road_rows = (
            db.session.query(SpeedReport.state, SpeedReport.road_key)
            .distinct()
            .limit(80)
            .all()
        )
        for st_raw, road_key in road_rows:
            st = normalize_state_group(st_raw)
            label = f"{st} • {format_road_bucket(road_key)}"
            if (q_lower not in label.lower()):
                continue
            url = url_for("bucket_tickets", state=st, road=road_key)
            key = ("road", st, road_key)
            if key in seen:
                continue
            suggestions.append({"type": "road", "label": label, "url": url})
            seen.add(key)

        # Keep top 5, users first, then states, then roads (current order)
        return jsonify(suggestions[:5])


    @app.get("/api/tickets")
    def api_tickets():
        try:
            limit = int(request.args.get("limit", "50"))
        except Exception:
            limit = 50
        limit = max(1, min(limit, 200))

        try:
            offset = int(request.args.get("offset", "0"))
        except Exception:
            offset = 0
        offset = max(0, offset)

        state = (request.args.get("state") or "").strip().upper()
        if state and len(state) != 2:
            # Ignore invalid state filters (backward-compatible)
            state = ""

        q = SpeedReport.query.options(joinedload(SpeedReport.user))

        # Hide soft-deleted tickets from public API feeds
        q = q.filter(SpeedReport.is_deleted.is_(False))
        if state:
            # State is stored as raw user input (often like "KS" or "KS - Kansas").
            # Filter by the first 2 letters to support both formats (backward-compatible).
            q = q.filter(func.upper(func.substr(func.trim(SpeedReport.state), 1, 2)) == state)

        reports = (
            q.order_by(SpeedReport.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

        # Compute per-page EnforcedSpeed threshold (P90 overage) per (state, road_key).
        # This keeps mobile/web parity for the "90% occur ≥ X mph" sentence.
        a_p90 = {}
        if reports:
            keys = {(r.state, r.road_key) for r in reports if getattr(r, 'state', None) and getattr(r, 'road_key', None)}
            if keys:
                expr = (SpeedReport.ticketed_speed - SpeedReport.posted_speed)
                rows = (
                    db.session.query(
                        SpeedReport.state,
                        SpeedReport.road_key,
                        expr.label('overage'),
                    )
                    .filter(SpeedReport.is_deleted.is_(False))
                    .filter(tuple_(SpeedReport.state, SpeedReport.road_key).in_(list(keys)))
                    .all()
                )
                a_dist = {}
                for st, rk, overage in rows:
                    if st is None or rk is None or overage is None:
                        continue
                    a_dist.setdefault((st, rk), []).append(float(overage))
                for vals in a_dist.values():
                    vals.sort()

                def _pctl(sorted_vals, p):
                    """Return smallest v such that at least p of distribution is at or above v."""
                    if not sorted_vals:
                        return 0.0
                    n = len(sorted_vals)
                    k = int(math.floor((1.0 - p) * n)) + 1
                    if k < 1:
                        k = 1
                    if k > n:
                        k = n
                    return float(sorted_vals[k - 1])

                a_p90 = {k: _pctl(vals, 0.90) for k, vals in a_dist.items()}

        report_ids = [r.id for r in reports]
        like_counts: Dict[int, int] = {}
        comment_counts: Dict[int, int] = {}
        user_liked: set[int] = set()
        user_ticket_posts_counts: Dict[int, int] = {}

        if report_ids:
            rows = (
                db.session.query(Like.report_id, func.count(Like.id))
                .filter(Like.report_id.in_(report_ids))
                .group_by(Like.report_id)
                .all()
            )
            like_counts = {rid: int(c) for rid, c in rows}

            rows = (
                db.session.query(Comment.report_id, func.count(Comment.id))
                .filter(Comment.report_id.in_(report_ids))
                .group_by(Comment.report_id)
                .all()
            )
            comment_counts = {rid: int(c) for rid, c in rows}

            # Ticket post counts per user (for feed display: username (N))
            user_ids = [uid for uid in {r.user_id for r in reports} if uid]
            if user_ids:
                rows = (
                    db.session.query(SpeedReport.user_id, func.count(SpeedReport.id))
                    .filter(SpeedReport.user_id.in_(user_ids))
                    .filter(SpeedReport.is_deleted.is_(False))
                    .group_by(SpeedReport.user_id)
                    .all()
                )
                user_ticket_posts_counts = {int(uid): int(c) for uid, c in rows}

            api_u = _api_user_from_request()
            if api_u:
                rows = (
                    db.session.query(Like.report_id)
                    .filter(Like.user_id == api_u.id, Like.report_id.in_(report_ids))
                    .all()
                )
                user_liked = {rid for (rid,) in rows}

        items = []
        for r in reports:
            items.append({
                "id": r.id,
                "state": r.state,
                "road": r.road_name,
                "posted_speed": r.posted_speed,
                "cited_speed": r.ticketed_speed,
                "overage": (r.ticketed_speed - r.posted_speed)
                    if (r.ticketed_speed is not None and r.posted_speed is not None)
                    else None,
                "percentile": 90,
                "enforced_speed_mph": (
                    int(round((r.posted_speed or 0) + a_p90.get((r.state, r.road_key), float((r.ticketed_speed - r.posted_speed) if (r.ticketed_speed is not None and r.posted_speed is not None) else 0))))
                    if (r.ticketed_speed is not None and r.posted_speed is not None)
                    else None
                ),
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "lat": r.lat,
                "lng": r.lng,
                "static_map_url": (
                    url_for("api_staticmap", lat=r.lat, lng=r.lng, zoom=11, w=640, h=340, _external=True)
                    if (r.lat is not None and r.lng is not None and get_google_maps_static_maps_key())
                    else None
                ),
                "verification_status": r.verification_status,
                "ocr_status": r.ocr_status,

                "username": (r.user.username if r.user else None),
                "user_id": (r.user.id if r.user else None),
                "user_ticket_posts_count": user_ticket_posts_counts.get(r.user_id, 0) if r.user_id else 0,
                "is_anonymous": (False if r.user else True),

                "likes_count": like_counts.get(r.id, 0),
                "comments_count": comment_counts.get(r.id, 0),
                "liked_by_me": (r.id in user_liked),

                "photo_submitted": (r.photo_key is not None),
                "photo_verified": (True if (r.photo_key is not None and (r.ocr_status == "verified" or r.verification_status == "verified")) else False),
            })

        return jsonify({
            "items": items,
            "offset": offset,
            "limit": limit,
            "next_offset": offset + len(items),
        })

    @app.post("/api/tickets")
    def api_create_ticket():
        """Create a ticket from web/mobile clients.

        Accepts:
        • application/json (no file)
        • multipart/form-data (fields + optional photo/PDF)

        Anonymous create is allowed for now (matches current web submit behavior).
        """
        ctype = (request.content_type or "").lower()
        is_multipart = "multipart/form-data" in ctype
        file_obj = None

        if is_multipart:
            data = dict(request.form or {})
            file_obj = request.files.get("photo")
        else:
            data = request.get_json(silent=True) or {}

        def first(*keys):
            for k in keys:
                if k in data and data.get(k) is not None:
                    return data.get(k)
            return None

        state_in = str(first('state', 'state_code') or '').strip()
        road_in = str(first('road', 'road_name') or '').strip()

        location_hint = str(first('location_hint', 'locationHint', 'city_name', 'cityName', 'city') or '').strip() or None

        posted_in = first('posted_speed', 'postedMph', 'posted_mph', 'posted')
        cited_in = first('cited_speed', 'ticketed_speed', 'ticketedMph', 'ticketed_mph', 'cited')

        raw_lat_in = first('raw_lat', 'rawLat', 'rawLatitude', 'raw_latitude')
        raw_lng_in = first('raw_lng', 'rawLng', 'rawLongitude', 'raw_longitude')

        lat_in = first('lat', 'latitude')
        lng_in = first('lng', 'longitude', 'lon')

        # ---- Validate + normalize ----
        if not state_in:
            return jsonify({'error': 'state is required'}), 400

        # Accept "GA" or "GA - Georgia" etc. Store the canonical "XX - Name" value.
        code = state_code_from_value(state_in)
        if not code:
            return jsonify({'error': 'state must be a 2-letter code (e.g., GA)'}), 400

        code = code.upper()
        code_to_label = {s[:2].upper(): s for s in STATE_OPTIONS}
        state_value = code_to_label.get(code)
        if not state_value:
            return jsonify({'error': f'Unsupported state code: {code}'}), 400

        try:
            posted_speed = int(str(posted_in).strip())
        except Exception:
            return jsonify({'error': 'posted_speed must be an integer'}), 400

        try:
            cited_speed = int(str(cited_in).strip())
        except Exception:
            return jsonify({'error': 'cited_speed must be an integer'}), 400

        if posted_speed <= 0 or posted_speed > 120:
            return jsonify({'error': 'posted_speed must be between 1 and 120'}), 400
        if cited_speed <= 0 or cited_speed > 200:
            return jsonify({'error': 'cited_speed must be between 1 and 200'}), 400

        if cited_speed <= posted_speed:
            return jsonify({'error': 'cited_speed must be greater than posted_speed'}), 400

        # Optional, but if provided should be plausible.
        def to_float(v):
            try:
                if v is None:
                    return None
                s = str(v).strip()
                if not s:
                    return None
                return float(s)
            except Exception:
                return None

        raw_lat = to_float(raw_lat_in)
        raw_lng = to_float(raw_lng_in)

        lat = to_float(lat_in)
        lng = to_float(lng_in)
        if lat is not None and (lat < -90 or lat > 90):
            return jsonify({'error': 'lat must be between -90 and 90'}), 400
        if lng is not None and (lng < -180 or lng > 180):
            return jsonify({'error': 'lng must be between -180 and 180'}), 400

        # Road is preferred, but allow pin-only submissions (road OR lat/lng required).
        road_in = _normalize_road_label(road_in) or road_in

        if not road_in or len(road_in) < 2:
            if lat is None or lng is None:
                return jsonify({'error': 'road is required unless a map pin (lat/lng) is provided'}), 400
            road_in = "Map pin"

        api_u = _api_user_from_request()


        report = SpeedReport(
            state=state_value,
            road_name=road_in,
            posted_speed=posted_speed,
            ticketed_speed=cited_speed,
            overage=max(0, cited_speed - posted_speed),
            caption=None,
            user_id=(api_u.id if api_u else None),
            location_hint=location_hint,
        )
        try:
            report.refresh_road_key()
        except Exception:
            pass

        if lat is not None and lng is not None:
            report.raw_lat = raw_lat if raw_lat is not None else lat
            report.raw_lng = raw_lng if raw_lng is not None else lng
            report.lat = lat
            report.lng = lng
            report.location_source = 'user_pin_snapped' if (raw_lat is not None or raw_lng is not None) else 'user_pin'

        db.session.add(report)
        db.session.commit()

        # Optional file upload (image/PDF) -> quarantine in R2 + async OCR verification.
        if file_obj and getattr(file_obj, "filename", ""):
            try:
                def _normalize_upload_to_jpeg_bytes_api(raw_bytes: bytes, filename: str | None, content_type: str | None) -> bytes:
                    fname = (filename or "").lower()
                    ctype2 = (content_type or "").lower()
                    is_pdf = fname.endswith(".pdf") or ("application/pdf" in ctype2)

                    if is_pdf:
                        if fitz is None:
                            raise ValueError("pdf_support_missing")
                        doc = fitz.open(stream=raw_bytes, filetype="pdf")
                        if doc.page_count < 1:
                            raise ValueError("empty_pdf")
                        page = doc.load_page(0)
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                        raw_bytes = pix.tobytes("png")

                    im = Image.open(io.BytesIO(raw_bytes))
                    im = im.convert("RGB")
                    im.thumbnail((2000, 2000))
                    out = io.BytesIO()
                    im.save(out, format="JPEG", quality=80, optimize=True)
                    return out.getvalue()

                # Mark submitted immediately so the feed can show status even if upload fails.
                report.photo_key = report.photo_key or f"pending:{uuid.uuid4().hex}"
                report.verification_status = "submitted"
                report.ocr_status = "pending"
                report.ocr_error = None
                report.verify_reason = None
                db.session.commit()

                raw = file_obj.read()
                try:
                    jpg_bytes = _normalize_upload_to_jpeg_bytes_api(raw, getattr(file_obj, "filename", None), getattr(file_obj, "content_type", None))
                except Exception:
                    # Upload exists but could not be normalized for OCR (e.g., unsupported or unreadable).
                    report.verification_status = "unverified"
                    report.ocr_status = "not_verified"
                    report.ocr_error = "unsupported_or_unreadable_upload"
                    report.verify_reason = "unsupported_format"
                    try:
                        if report.photo_key and str(report.photo_key).startswith("pending:"):
                            report.photo_key = str(report.photo_key).replace("pending:", "failed:", 1)
                        elif not report.photo_key:
                            report.photo_key = f"failed:{uuid.uuid4().hex}"
                    except Exception:
                        pass
                    db.session.commit()
                    jpg_bytes = None

                if jpg_bytes:
                    missing = [k for k in ("R2_ENDPOINT", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY") if not (os.environ.get(k) or "").strip()]
                    if missing:
                        report.verification_status = "unverified"
                        report.ocr_status = "not_verified"
                        report.ocr_error = "missing_r2_env: " + ",".join(missing)
                        report.verify_reason = "upload_failed"
                        try:
                            if report.photo_key and str(report.photo_key).startswith('pending'):
                                report.photo_key = f"failed:{__import__('uuid').uuid4().hex}"
                        except Exception:
                            pass
                        db.session.commit()
                    else:
                        bucket = os.environ.get("R2_QUARANTINE_BUCKET", "enforcedspeed-ticket-quarantine").strip() or "enforcedspeed-ticket-quarantine"
                        prefix = os.environ.get("R2_PREFIX", "tickets/").strip()
                        if prefix and not prefix.endswith("/"):
                            prefix = prefix + "/"
                        key = f"{prefix}{report.id}/{uuid.uuid4().hex}.jpg"

                        ok = put_bytes(bucket=bucket, key=key, data=jpg_bytes, content_type="image/jpeg")
                        if not ok:
                            report.verification_status = "unverified"
                            report.ocr_status = "not_verified"
                            report.ocr_error = "upload_failed"
                            report.verify_reason = "upload_failed"
                            try:
                                if getattr(report, 'photo_key', None) and str(report.photo_key).startswith('pending:'):
                                    report.photo_key = str(report.photo_key).replace('pending:', 'failed:')
                                elif not getattr(report, 'photo_key', None):
                                    report.photo_key = f"failed:{__import__('uuid').uuid4().hex}"
                            except Exception:
                                pass
                            db.session.commit()
                        else:
                            report.photo_key = key
                            report.photo_uploaded_at = datetime.utcnow()

                            q = get_queue()
                            job = q.enqueue(
                                "ocr_verify.verify_ticket_from_r2",
                                report.id,
                                bucket,
                                key,
                                job_timeout=120,
                            )
                            report.ocr_job_id = job.id
                            db.session.commit()
            except Exception:
                try:
                    app.logger.exception("API photo upload/OCR enqueue failed for report_id=%s", report.id)
                except Exception:
                    pass
                report.verification_status = "unverified"
                report.ocr_status = "not_verified"
                report.ocr_error = "upload_failed"
                report.photo_key = (report.photo_key.replace('pending:', 'failed:') if (getattr(report, 'photo_key', None) and str(report.photo_key).startswith('pending:')) else 'failed')

                report.verify_reason = "upload_failed"
                db.session.commit()

        return jsonify({
            'id': report.id,
            'state': report.state,
            'road': report.road_name,
            'posted_speed': report.posted_speed,
            'cited_speed': report.ticketed_speed,
            'created_at': report.created_at.isoformat() if report.created_at else None,
            'lat': report.lat,
            'lng': report.lng,
            'photo_submitted': bool(report.photo_key),
            'verification_status': getattr(report, 'verification_status', None),
            'ocr_status': getattr(report, 'ocr_status', None),
            'static_map_url': (
                url_for('api_staticmap', lat=report.lat, lng=report.lng, zoom=11, w=640, h=340, _external=True)
                if (report.lat is not None and report.lng is not None and get_google_maps_static_maps_key())
                else None
            ),
        }), 201


    @app.get("/api/strictness")
    def api_strictness():
        """Return Most Strict / Least Strict road rankings as JSON for native clients.

        Query params:
          - limit: optional int (default 10, max 50)
          - state: optional 2-letter (e.g., GA). Filters to that state when provided.
          - verified_only: optional "1" to include only verified-photo tickets in calculations.
          - date: optional ("7","30","90","365") or ("7d","30d","90d","365d") to limit by recency.
        """
        try:
            limit = int(request.args.get("limit", "10"))
        except Exception:
            limit = 10
        limit = max(1, min(limit, 50))

        state = (request.args.get("state") or "").strip().upper()
        if state and (len(state) != 2 or not state.isalpha()):
            state = ""

        verified_only = (request.args.get("verified_only") == "1")

        date_raw = (request.args.get("date") or "").strip().lower()
        date_map = {
            "7d": "7",
            "30d": "30",
            "90d": "90",
            "365d": "365",
        }
        date = date_map.get(date_raw, date_raw)
        if date == "":
            date = "any"

        data = strictness_rows(
            limit=limit,
            exclude_anonymous=False,
            state_filter=(state or None),
            photo_only=False,
            verify=("verified" if verified_only else "any"),
            date=date,
            deleted_mode="hide",
        )

        return jsonify({
            "most_strict": data.get("most_strict", []),
            "least_strict": data.get("least_strict", []),
            "limit": limit,
            "state": (state or None),
            "verified_only": bool(verified_only),
            "date": (date if date != "any" else None),
            "generated_at": datetime.utcnow().isoformat(),
        })


    @app.get("/api/staticmap")
    def api_staticmap():
        """Proxy a Google Static Maps image (single pin) for mobile clients.

        Keeps the Google Maps API key server-side and avoids relying on HTTP referrer restrictions.
        Query params:
          - lat, lng: required floats
          - zoom: optional int (default 11)
          - w, h: optional ints (default 640x340)
        """
        try:
            lat = float(request.args.get("lat", ""))
            lng = float(request.args.get("lng", ""))
        except Exception:
            abort(400)

        try:
            zoom = int(request.args.get("zoom", "11"))
        except Exception:
            zoom = 11

        try:
            w = int(request.args.get("w", "640"))
            h = int(request.args.get("h", "340"))
        except Exception:
            w, h = 640, 340

        zoom = max(1, min(20, zoom))
        w = max(120, min(1024, w))
        h = max(120, min(1024, h))

        key = get_google_maps_static_maps_key()
        if not key:
            abort(404)

        center = f"{lat:.6f},{lng:.6f}"

        # Use the same public base URL logic as county_static_map_url so Google can fetch icons.
        try:
            from flask import current_app
            cfg_base = (current_app.config.get("PUBLIC_BASE_URL") or "").strip().rstrip("/")
        except Exception:
            cfg_base = (os.environ.get("PUBLIC_BASE_URL") or "").strip().rstrip("/")

        if cfg_base and cfg_base.startswith("http"):
            base_url = cfg_base
        else:
            proto = (request.headers.get("X-Forwarded-Proto") or request.scheme or "https").split(",")[0].strip()
            host = (request.headers.get("X-Forwarded-Host") or request.host).split(",")[0].strip()
            if host.endswith("onrender.com") or ".onrender.com" in host:
                host = "enforcedspeed.com"
            if proto == "http":
                proto = "https"
            base_url = f"{proto}://{host}"

        # Prefer a dedicated public static asset domain (e.g. Cloudflare R2 custom domain)
        # so Google can fetch the icon reliably.
        pin_base = ((os.environ.get("STATIC_PIN_BASE_URL") or "").strip() or (getattr(current_app, "config", {}).get("STATIC_PIN_BASE_URL") if 'current_app' in locals() else "") or "").strip().rstrip("/")
        if not (pin_base and pin_base.startswith("http")):
            pin_base = f"{base_url}/static/img/pins"
        icon_url = f"{pin_base}/pin_inside_deepred_static.png"
        enc_icon = urllib.parse.quote(icon_url, safe="")

        params = {
            "center": center,
            "zoom": str(int(zoom)),
            "size": f"{int(w)}x{int(h)}",
            "scale": "2",
            "maptype": "roadmap",
            "markers": f"icon:{enc_icon}|{center}",
            "key": key,
        }
        upstream = "https://maps.googleapis.com/maps/api/staticmap?" + urllib.parse.urlencode(params, safe=':/,|%')

        try:
            req = urllib.request.Request(upstream, headers={"User-Agent": "EnforcedSpeedMobile/1.0"})
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = resp.read()
                content_type = resp.headers.get("Content-Type") or "image/png"
        except Exception:
            abort(502)

        out = Response(data, mimetype=content_type)
        out.headers["Cache-Control"] = "public, max-age=86400"
        return out


    @app.get("/api/county_staticmap")
    def api_county_staticmap():
        """Proxy a Google Static Maps image that includes a county boundary overlay and optional pin.

        Keeps the Google Maps API key server-side.

        Query params:
          - geoid: optional (county GEOID). If omitted, returns a default US preview map.
          - pin_lat, pin_lng: optional floats (only used when geoid is provided)
          - center_on_pin: optional (1/0). When 1 and a pin is provided, center the map on the pin.
          - w, h: optional ints (default 640x640)
        """
        geoid = (request.args.get("geoid") or "").strip()

        try:
            w = int(request.args.get("w", "640"))
            h = int(request.args.get("h", "640"))
        except Exception:
            w, h = 640, 640

        w = max(120, min(1024, w))
        h = max(120, min(1024, h))

        upstream = ""
        if geoid:
            pin_lat = pin_lng = None
            if (request.args.get("pin_lat") or '').strip() and (request.args.get("pin_lng") or '').strip():
                try:
                    pin_lat = float(request.args.get("pin_lat", ""))
                    pin_lng = float(request.args.get("pin_lng", ""))
                except Exception:
                    abort(400)

            center_on_pin = (request.args.get("center_on_pin") == "1")

            upstream = county_static_map_url(
                geoid,
                pin_lat,
                pin_lng,
                width=w,
                height=h,
                center_on_pin=center_on_pin,
                use_custom_icon=True
            )
        else:
            upstream = us_static_map_url(width=w, height=h)


        # Debug helper: return the upstream Google Static Maps URL (and marker/icon details)
        # without proxying the image. Use only for troubleshooting.
        if request.args.get("debug") == "1":
            try:
                parsed = urllib.parse.urlparse(upstream)
                qs = urllib.parse.parse_qs(parsed.query)
                markers_raw = (qs.get("markers") or [None])[0]
                icon_url = None
                if markers_raw:
                    # markers usually looks like: icon:https://...%7Clat,lng
                    markers_decoded = urllib.parse.unquote(markers_raw)
                    if markers_decoded.startswith("icon:"):
                        icon_url = markers_decoded.split("|", 1)[0][len("icon:"):]
                icon_fetch = None
                if icon_url:
                    try:
                        req_icon = urllib.request.Request(icon_url, method="GET", headers={"User-Agent": "EnforcedSpeedWeb/1.0"})
                        with urllib.request.urlopen(req_icon, timeout=10) as r:
                            icon_fetch = {
                                "status": getattr(r, "status", None) or r.getcode(),
                                "final_url": getattr(r, "geturl", lambda: None)(),
                                "content_type": (r.headers.get("Content-Type") or ""),
                                "content_length": int(r.headers.get("Content-Length") or 0) if (r.headers.get("Content-Length") or "").isdigit() else None,
                            }
                    except Exception as e:
                        icon_fetch = {"error": str(e)}
                return jsonify({
                    "upstream": upstream,
                    "upstream_len": len(upstream),
                    "has_geoid": bool(geoid),
                    "pin_lat": request.args.get("pin_lat"),
                    "pin_lng": request.args.get("pin_lng"),
                    "markers_raw": markers_raw,
                    "icon_url": icon_url,
                    "icon_fetch": icon_fetch,
                })
            except Exception as e:
                return jsonify({"error": str(e), "upstream": upstream}), 200
        if not upstream:
            abort(404)

        # --- v370: server-side caching ---
        # Cache by the final upstream URL so identical requests across users/browsers are fast
        # and we don't repeatedly hit Google Static Maps.
        cache_ttl_sec = 7 * 24 * 60 * 60  # 7 days
        cache_key = None
        try:
            import hashlib
            cache_key = "es:staticmap:" + hashlib.sha256(upstream.encode("utf-8")).hexdigest()
        except Exception:
            cache_key = None

        def _cache_get(k: str):
            """Return (content_type, bytes) or (None, None)."""
            if not k:
                return (None, None)
            # Prefer Redis (shared across instances); fall back to local /tmp file cache.
            rurl = (os.environ.get("REDIS_URL") or "").strip()
            if rurl:
                try:
                    import redis
                    r = redis.Redis.from_url(rurl, socket_timeout=2, socket_connect_timeout=2)
                    blob = r.get(k)
                    if blob:
                        # Stored as: b"<content-type>\n" + image bytes
                        i = blob.find(b"\n")
                        if i > 0:
                            ct = blob[:i].decode("utf-8", errors="ignore") or "image/png"
                            return (ct, blob[i + 1 :])
                except Exception:
                    pass

            try:
                cache_dir = "/tmp/es_staticmap_cache"
                os.makedirs(cache_dir, exist_ok=True)
                p = os.path.join(cache_dir, k.replace(":", "_") + ".bin")
                if os.path.exists(p):
                    with open(p, "rb") as f:
                        blob = f.read()
                    i = blob.find(b"\n")
                    if i > 0:
                        ct = blob[:i].decode("utf-8", errors="ignore") or "image/png"
                        return (ct, blob[i + 1 :])
            except Exception:
                pass

            return (None, None)

        def _cache_set(k: str, content_type: str, data: bytes):
            if not k:
                return
            blob = (content_type or "image/png").encode("utf-8") + b"\n" + (data or b"")
            rurl = (os.environ.get("REDIS_URL") or "").strip()
            if rurl:
                try:
                    import redis
                    r = redis.Redis.from_url(rurl, socket_timeout=2, socket_connect_timeout=2)
                    r.setex(k, cache_ttl_sec, blob)
                    return
                except Exception:
                    pass
            try:
                cache_dir = "/tmp/es_staticmap_cache"
                os.makedirs(cache_dir, exist_ok=True)
                p = os.path.join(cache_dir, k.replace(":", "_") + ".bin")
                with open(p, "wb") as f:
                    f.write(blob)
            except Exception:
                pass

        ct_cached, data_cached = _cache_get(cache_key)
        if data_cached:
            out = Response(data_cached, mimetype=(ct_cached or "image/png"))
            out.headers["Cache-Control"] = "public, max-age=86400"
            out.headers["X-ES-Cache"] = "HIT"
            return out

        try:
            req = urllib.request.Request(upstream, headers={"User-Agent": "EnforcedSpeedWeb/1.0"})
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = resp.read()
                content_type = resp.headers.get("Content-Type") or "image/png"
        except Exception:
            abort(502)

        _cache_set(cache_key, content_type, data)

        out = Response(data, mimetype=content_type)
        out.headers["Cache-Control"] = "public, max-age=86400"
        out.headers["X-ES-Cache"] = "MISS"
        return out


    # --- API auth endpoints (JWT) ---
    def _user_public(u: User) -> dict:
        return {
            "id": u.id,
            "email": u.email,
            "username": u.username,
            "created_at": u.created_at.isoformat() if getattr(u, "created_at", None) else None,
        }

    @app.get("/api/home_minimap_staticmap")
    def api_home_minimap_staticmap():
        """Proxy a Google Static Maps image for the home right-rail mini-map.

        The mini-map shows up to 5 *unique counties* (trending-first), plotted at the county centroid/center.
        Falls back to the most recent counties / overall top counties so the map never looks empty.

        Click behavior is handled client-side by wrapping the <img> in a link to /map.
        """
        debug = (request.args.get("debug") or "").strip() in ("1", "true", "yes")

        google_key = (os.getenv("GOOGLE_MAPS_SERVER_KEY") or "").strip()
        if not google_key:
            return ("Missing GOOGLE_MAPS_SERVER_KEY", 503)

        # Fixed CONUS bounds (approx). Use 'visible=' so Google chooses an appropriate zoom.
        # Padded slightly so WA/OR (and Maine) don't feel clipped.
        visible_sw = "23.0,-127.5"
        visible_ne = "50.5,-65.0"

        try:
            now_utc = datetime.utcnow()
            t7 = now_utc - timedelta(days=7)
            t30 = now_utc - timedelta(days=30)
            t365 = now_utc - timedelta(days=365)

            rows = (
                db.session.query(
                    SpeedReport.county_geoid.label("geoid"),
                    func.sum(case((SpeedReport.created_at >= t7, 1), else_=0)).label("c7"),
                    func.sum(case((SpeedReport.created_at >= t30, 1), else_=0)).label("c30"),
                    func.sum(case((SpeedReport.created_at >= t365, 1), else_=0)).label("c365"),
                    func.max(SpeedReport.created_at).label("last_ts"),
                )
                .filter(SpeedReport.is_deleted == False)
                .filter(SpeedReport.county_geoid.isnot(None))
                .group_by(SpeedReport.county_geoid)
                .all()
            )

            scored = []
            recent = []
            for r in rows:
                geoid = str(r.geoid) if r.geoid is not None else ""
                c7 = int(r.c7 or 0)
                c30 = int(r.c30 or 0)
                c365 = int(r.c365 or 0)
                last_ts = r.last_ts
                recent.append((last_ts, geoid))

                # Guard: require some volume so 1 ticket doesn't "trend".
                if c30 < 5:
                    continue
                baseline = (c30 / 4.285)
                score = (c7 - baseline) / math.sqrt(baseline + 1.0)
                scored.append((score, c7, geoid))

            scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

            # Choose up to 5 unique geoids, trending-first then recent.
            chosen_geoids = []
            chosen_set = set()

            for score, c7, geoid in scored:
                if len(chosen_geoids) >= 5:
                    break
                if not geoid or geoid in chosen_set:
                    continue
                chosen_geoids.append(geoid)
                chosen_set.add(geoid)

            if len(chosen_geoids) < 5:
                recent.sort(key=lambda x: (x[0] is not None, x[0]), reverse=True)
                for last_ts, geoid in recent:
                    if len(chosen_geoids) >= 5:
                        break
                    if not geoid or geoid in chosen_set:
                        continue
                    chosen_geoids.append(geoid)
                    chosen_set.add(geoid)

            # Final fallback: fill remaining slots with highest-volume counties overall.
            if len(chosen_geoids) < 5:
                top_rows = (
                    db.session.query(
                        SpeedReport.county_geoid.label("geoid"),
                        func.count(SpeedReport.id).label("cnt"),
                        func.max(SpeedReport.created_at).label("last_ts"),
                    )
                    .filter(SpeedReport.is_deleted == False)
                    .filter(SpeedReport.county_geoid.isnot(None))
                    .group_by(SpeedReport.county_geoid)
                    .order_by(func.count(SpeedReport.id).desc(), func.max(SpeedReport.created_at).desc())
                    .limit(200)
                    .all()
                )
                for r in top_rows:
                    if len(chosen_geoids) >= 5:
                        break
                    geoid = str(r.geoid) if r.geoid is not None else ""
                    if not geoid or geoid in chosen_set:
                        continue
                    chosen_geoids.append(geoid)
                    chosen_set.add(geoid)

            # Pull county center/centroid coordinates.
            county_pts = []
            if chosen_geoids:
                # NOTE: use an expanding bind for compatibility across DB drivers.
                from sqlalchemy import bindparam

                # Determine which county coordinate columns exist (Render schema differs across versions).
                col_rows = db.session.execute(
                    text(
                        """
                        SELECT column_name
                        FROM information_schema.columns
                        WHERE table_schema = 'public' AND table_name = 'counties'
                        """
                    )
                ).all()
                cols = {r[0] for r in col_rows}

                # Pick the best available lat/lng expressions.
                # Different environments have used:
                # - center_lat/center_lng (numeric)
                # - centroid_lat/centroid_lng (numeric)
                # - center (PostGIS point)
                # - centroid (PostGIS point)
                # - geom (PostGIS polygon)
                if 'center_lat' in cols and 'center_lng' in cols:
                    lat_expr = "center_lat"
                    lng_expr = "center_lng"
                elif 'centroid_lat' in cols and 'centroid_lng' in cols:
                    lat_expr = "centroid_lat"
                    lng_expr = "centroid_lng"
                elif 'center' in cols:
                    lat_expr = "ST_Y(center::geometry)"
                    lng_expr = "ST_X(center::geometry)"
                elif 'centroid' in cols:
                    lat_expr = "ST_Y(centroid::geometry)"
                    lng_expr = "ST_X(centroid::geometry)"
                elif 'geom' in cols:
                    lat_expr = "ST_Y(ST_Centroid(geom)::geometry)"
                    lng_expr = "ST_X(ST_Centroid(geom)::geometry)"
                else:
                    raise RuntimeError("Counties table missing center/centroid/geom columns for minimap")

                stmt = text(
                    f"""
                    SELECT geoid,
                           {lat_expr} AS lat,
                           {lng_expr} AS lng
                    FROM counties
                    WHERE geoid IN :geoids
                      AND ({lat_expr} IS NOT NULL)
                      AND ({lng_expr} IS NOT NULL)
                    """
                ).bindparams(bindparam("geoids", expanding=True))

                county_rows = db.session.execute(stmt, {"geoids": chosen_geoids}).mappings().all()

                coord_map = {str(r["geoid"]): (float(r["lat"]), float(r["lng"])) for r in county_rows if r.get("geoid")}

                # Preserve chosen order; one marker per county.
                for geoid in chosen_geoids:
                    if geoid in coord_map:
                        county_pts.append(coord_map[geoid])

            # Build a small static map using the same custom pins as the rest of the site.
            params = [
                ("size", "640x346"),
                ("scale", "2"),
                ("maptype", "roadmap"),
                ("style", "feature:poi|visibility:off"),
                ("style", "feature:transit|visibility:off"),
                ("visible", f"{visible_sw}|{visible_ne}"),
                ("key", google_key),
            ]

            # Prefer Cloudflare R2 pin host so Google Static Maps can fetch the icon reliably.
            # Base URL should look like: https://static.enforcedspeed.com/pins
            pin_base = (os.getenv('STATIC_PIN_BASE_URL') or '').strip().rstrip('/')
            if pin_base:
                pin_icon_url = f"{pin_base}/pin_inside_deepred_static.png"
            else:
                # Fallback to our own origin (works locally)
                pin_icon_url = url_for('static', filename='img/pins/pin_inside_deepred_static.png', _external=True)
            for lat, lng in county_pts[:5]:
                # IMPORTANT: Google requires the icon URL to be URL-encoded, but the marker directive
                # itself must remain as literal 'icon:' (not 'icon%3A').
                enc_icon = urllib.parse.quote(pin_icon_url, safe="")
                params.append(("markers", f"icon:{enc_icon}|{lat:.6f},{lng:.6f}"))

            # Build URL manually so we don't accidentally encode 'icon:' into 'icon%3A'.
            def _qs(items: list[tuple[str, str]]) -> str:
                return "&".join(
                    f"{urllib.parse.quote(k, safe='')}={urllib.parse.quote(v, safe=':/,|%')}"
                    for k, v in items
                )
            url = f"https://maps.googleapis.com/maps/api/staticmap?{_qs(params)}"

            # Fetch and return image
            resp = requests.get(url, timeout=12, headers={"User-Agent": "EnforcedSpeedWeb/1.0"})

            if debug:
                return jsonify(
                    {
                        "chosen_geoids": chosen_geoids,
                        "county_pts": county_pts,
                        "url": url,
                        "status_code": resp.status_code,
                        "content_type": resp.headers.get("Content-Type"),
                        "content_len": len(resp.content or b""),
                        "first_bytes": (resp.content or b"")[:50].decode("latin1", errors="replace"),
                    }
                ), 200

            if resp.status_code != 200:
                return (f"Upstream error {resp.status_code}", 502)

            out = Response(resp.content, mimetype=(resp.headers.get("Content-Type") or "image/png"))
            out.headers["Cache-Control"] = "public, max-age=300"
            return out

        except Exception as e:
            try:
                db.session.rollback()
            except Exception:
                pass
            return (f"Error generating minimap: {e}", 500)

    @app.post("/api/auth/register")
    def api_auth_register():
        data = request.get_json(silent=True) or {}
        email = (data.get("email") or "").strip().lower()
        username = (data.get("username") or "").strip()
        password = (data.get("password") or "")

        if not email or "@" not in email:
            return jsonify({"error": "valid_email_required"}), 400
        if not username or len(username) < 3 or len(username) > 20:
            return jsonify({"error": "username_must_be_3_20"}), 400
        if not password or len(password) < 8:
            return jsonify({"error": "password_min_8"}), 400

        if User.query.filter_by(email=email).first():
            return jsonify({"error": "email_taken"}), 409
        if User.query.filter(func.lower(User.username) == func.lower(username)).first():
            return jsonify({"error": "username_taken"}), 409

        u = User(email=email, username=username)
        u.set_password(password)
        db.session.add(u)
        db.session.commit()

        token = _jwt_encode(u)
        return jsonify({"token": token, "user": _user_public(u)}), 201


    @app.post("/api/auth/login")
    def api_auth_login():
        data = request.get_json(silent=True) or {}
        identifier = (data.get("identifier") or data.get("email") or data.get("username") or "").strip()
        password = (data.get("password") or "")

        if not identifier or not password:
            return jsonify({"error": "identifier_and_password_required"}), 400

        u = None
        if "@" in identifier:
            u = User.query.filter_by(email=identifier.lower()).first()
        if u is None:
            u = User.query.filter(func.lower(User.username) == func.lower(identifier)).first()

        if (not u) or (not u.check_password(password)):
            return jsonify({"error": "invalid_credentials"}), 401

        token = _jwt_encode(u)
        return jsonify({"token": token, "user": _user_public(u)}), 200


    @app.get("/api/me")
    @api_login_required
    def api_me():
        u = getattr(request, "_api_user", None)  # type: ignore
        if not u:
            return jsonify({"error": "auth_required"}), 401

        ticket_posts = SpeedReport.query.filter(SpeedReport.user_id == u.id, SpeedReport.is_deleted.is_(False)).count()
        likes = Like.query.filter(Like.user_id == u.id).count()
        comments = Comment.query.filter(Comment.user_id == u.id).count()

        return jsonify({
            "user": _user_public(u),
            "stats": {
                "ticket_posts": int(ticket_posts),
                "likes": int(likes),
                "comments": int(comments),
            },
        })


    @app.post("/api/reports/<int:report_id>/like")
    @api_login_required
    def api_toggle_like(report_id: int):
        u = getattr(request, "_api_user", None)  # type: ignore
        report = SpeedReport.query.get_or_404(report_id)
        if getattr(report, "is_deleted", False):
            abort(404)


        existing = Like.query.filter_by(report_id=report.id, user_id=u.id).first()
        if existing:
            db.session.delete(existing)
            liked = False
        else:
            db.session.add(Like(report_id=report.id, user_id=u.id))
            liked = True

        db.session.commit()
        likes_count = Like.query.filter_by(report_id=report.id).count()

        return jsonify({
            "report_id": report.id,
            "liked": bool(liked),
            "likes_count": int(likes_count),
        })


    @app.get("/api/reports/<int:report_id>/comments")
    def api_get_comments(report_id: int):
        try:
            limit = int(request.args.get("limit", "50"))
        except Exception:
            limit = 50
        limit = max(1, min(limit, 200))

        try:
            offset = int(request.args.get("offset", "0"))
        except Exception:
            offset = 0
        offset = max(0, offset)

        report = SpeedReport.query.get_or_404(report_id)
        if getattr(report, "is_deleted", False):
            abort(404)

        q = Comment.query.options(joinedload(Comment.user)).filter(Comment.report_id == report_id)
        rows = q.order_by(Comment.created_at.asc()).offset(offset).limit(limit).all()

        items = []
        for c in rows:
            items.append({
                "id": c.id,
                "report_id": c.report_id,
                "body": c.body,
                "created_at": c.created_at.isoformat() if c.created_at else None,
                "username": (c.user.username if c.user else None),
                "user_id": c.user_id,
            })

        total = Comment.query.filter(Comment.report_id == report_id).count()

        return jsonify({
            "items": items,
            "offset": offset,
            "limit": limit,
            "total": int(total),
            "next_offset": offset + len(items),
        })


    @app.post("/api/reports/<int:report_id>/comments")
    @api_login_required
    def api_add_comment(report_id: int):
        u = getattr(request, "_api_user", None)  # type: ignore
        data = request.get_json(silent=True) or {}
        body = (data.get("body") or "").strip()
        if not body:
            return jsonify({"error": "comment_empty"}), 400
        if len(body) > 280:
            return jsonify({"error": "comment_too_long"}), 400

        # Basic rate limit: max 5 comments / 60s / user (same as web)
        window_start = datetime.utcnow() - timedelta(seconds=60)
        recent_count = (
            Comment.query.filter(Comment.user_id == u.id, Comment.created_at >= window_start)
            .count()
        )
        if recent_count >= 5:
            return jsonify({"error": "rate_limited"}), 429

        report = SpeedReport.query.get_or_404(report_id)
        if getattr(report, "is_deleted", False):
            abort(404)

        c = Comment(report_id=report_id, user_id=u.id, body=body)
        db.session.add(c)
        db.session.commit()

        return jsonify({
            "id": c.id,
            "report_id": c.report_id,
            "body": c.body,
            "created_at": c.created_at.isoformat() if c.created_at else None,
            "username": u.username,
        }), 201


    # --- Ticket alias routes (mobile naming parity) ---
    # Keep /api/reports/* for backward compatibility; mobile uses /api/tickets/* going forward.

    @app.get("/api/tickets/<int:report_id>/comments")
    def api_get_ticket_comments(report_id: int):
        return api_get_comments(report_id)

    @app.post("/api/tickets/<int:report_id>/comments")
    @api_login_required
    def api_add_ticket_comment(report_id: int):
        return api_add_comment(report_id)

    @app.post("/api/tickets/<int:report_id>/like")
    @api_login_required
    def api_toggle_ticket_like(report_id: int):
        return api_toggle_like(report_id)

    @app.get("/api/tickets/<int:report_id>/insights")
    def api_ticket_insights(report_id: int):
        """Return the same percentile stats shown on the web thank-you page.

        Used by mobile after submit to show the 'thermometer' bars with identical math.
        """
        report = SpeedReport.query.get_or_404(report_id)
        if getattr(report, "is_deleted", False):
            abort(404)

        # strictness = ticketed - posted (mph)
        try:
            user_strictness = int(report.ticketed_speed) - int(report.posted_speed)
        except Exception:
            user_strictness = None

        state_group = normalize_state_group(report.state)

        # Pull all rows for the state (prefix match) and bucket them by road_key + posted_speed.
        rows = (
            db.session.query(
                SpeedReport.state,
                SpeedReport.road_key,
                SpeedReport.posted_speed,
                SpeedReport.ticketed_speed,
                SpeedReport.is_deleted,
            )
            .filter(SpeedReport.state.ilike(f"{state_group}%"))
            .all()
        )

        strict_state_road: List[float] = []
        strict_state_posted: List[float] = []

        for st, road_key, posted, ticketed, is_deleted in rows:
            if is_deleted:
                continue
            if normalize_state_group(st) != state_group:
                continue
            try:
                strictness = float(int(ticketed) - int(posted))
            except Exception:
                continue

            if road_key == report.road_key:
                strict_state_road.append(strictness)
            if int(posted) == int(report.posted_speed):
                strict_state_posted.append(strictness)

        stats_a = compute_distribution_stats(strict_state_road, float(user_strictness) if user_strictness is not None else None)
        stats_b = compute_distribution_stats(strict_state_posted, float(user_strictness) if user_strictness is not None else None)

        return jsonify({
            "ok": True,
            "state_group": state_group,
            "report": {
                "id": report.id,
                "state": report.state,
                "road_name": report.road_name,
                "posted_speed": report.posted_speed,
                "ticketed_speed": report.ticketed_speed,
                "photo_submitted": bool(report.photo_key),
            },
            "stats_a": stats_a,
            "stats_b": stats_b,
        })


    @app.get("/api/users/<int:user_id>")
    def api_user_public_profile(user_id: int):
        u = User.query.get_or_404(user_id)

        ticket_posts = SpeedReport.query.filter(SpeedReport.user_id == u.id, SpeedReport.is_deleted.is_(False)).count()
        likes = Like.query.filter(Like.user_id == u.id).count()
        comments = Comment.query.filter(Comment.user_id == u.id).count()

        return jsonify({
            "user": {
                "id": u.id,
                "username": u.username,
                "created_at": u.created_at.isoformat() if getattr(u, "created_at", None) else None,
            },
            "stats": {
                "ticket_posts": int(ticket_posts),
                "likes": int(likes),
                "comments": int(comments),
            },
        })


    @app.get("/search")
    def search():
        q = (request.args.get("q") or "").strip()
        if not q:
            return redirect(url_for("home"))

        q_clean = q[1:] if q.startswith("@") else q
        q_like = f"%{q_clean}%"
        q_lower = q_clean.lower()
        q_upper = q_clean.upper()

        # Users
        users = (
            User.query
            .filter(db.func.lower(User.username).like(db.func.lower(q_like)))
            .order_by(User.username.asc())
            .limit(25)
            .all()
        )

        # States (present in DB)
        state_rows = db.session.query(SpeedReport.state).distinct().all()
        all_states = sorted({normalize_state_group(r[0]) for r in state_rows if r and r[0]})
        states = [st for st in all_states if (q_upper in st or q_lower in st.lower())][:25]

        # Roads (state + road bucket)
        road_rows = (
            db.session.query(SpeedReport.state, SpeedReport.road_key)
            .distinct()
            .limit(500)
            .all()
        )
        roads = []
        seen_roads = set()
        for st_raw, road_key in road_rows:
            st = normalize_state_group(st_raw)
            label = f"{st} • {format_road_bucket(road_key)}"
            if q_lower not in label.lower():
                continue
            key = (st, road_key)
            if key in seen_roads:
                continue
            roads.append({"state": st, "road_key": road_key, "label": label})
            seen_roads.add(key)
            if len(roads) >= 50:
                break

        return render_template("search_results.html", q=q, users=users, states=states, roads=roads)

    @app.route("/privacy")
    def privacy():
        return render_template("privacy.html")

    @app.route("/terms")
    def terms():
        return render_template("terms.html")

    @app.route("/cookies")
    def cookies():
        return render_template("cookies.html")


    @app.get("/health")
    def health():
        return {"status": "ok"}

    # Schema bootstrap for dev/local databases (additive, no destructive changes).
    with app.app_context():
        try:
            ensure_counties_schema()
            # ensure_states_schema()  # Disabled at boot (can be expensive); built lazily when needed.
        except Exception as e:
            # Keep app running even if counties are unavailable; submit page will degrade gracefully.
            print(f"[COUNTY INIT ERROR] failed to ensure counties schema: {e}")


    return app

app = create_app()


if __name__ == "__main__":
    import os

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "1") == "1"

    app.run(host=host, port=port, debug=debug)