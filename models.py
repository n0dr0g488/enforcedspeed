# models.py
import re
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash


db = SQLAlchemy()


def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def normalize_text_basic(value: str) -> str:
    """
    Lowercase, strip punctuation, collapse whitespace.
    Used as a fallback for local roads / misc text.
    """
    if not value:
        return ""
    v = value.strip().lower()
    v = re.sub(r"[^a-z0-9\s]", " ", v)
    v = normalize_whitespace(v)
    return v


def state_code_from_value(state_value: str) -> str:
    """
    Extract 2-letter state code from:
      - "VA"
      - "VA - Virginia"
      - "Virginia"  (won't map reliably without a lookup; we use only 2-letter detection)
    """
    if not state_value:
        return ""
    m = re.match(r"^\s*([A-Za-z]{2})\b", state_value.strip())
    return m.group(1).upper() if m else ""


def normalize_road(value: str, state_value: str = "") -> str:
    """
    State-aware, strict road normalization (no guessing):

    - If user indicates Interstate (I / I- / interstate): -> i<number> (+ optional direction)
    - If user indicates US route (US / U.S. / united states route): -> us<number> (+ optional direction)
    - If user indicates State Route (SR / state route / route / hwy / highway): -> <statecode><number> if state known,
      otherwise -> route<number> (+ optional direction)
    - If user types JUST a number like "95": -> route95 (generic), NOT i95
    """
    if not value:
        return ""

    raw = value.strip().lower()

    # Normalize punctuation to spaces early, keep letters/numbers
    cleaned = re.sub(r"[^a-z0-9\s]", " ", raw)
    cleaned = normalize_whitespace(cleaned)

    state_code = state_code_from_value(state_value)
    state_prefix = state_code.lower() if state_code else ""

    # Detect direction anywhere in the string
    direction = None
    dir_patterns = [
        (r"\b(northbound|nb|n)\b", "nb"),
        (r"\b(southbound|sb|s)\b", "sb"),
        (r"\b(eastbound|eb|e)\b", "eb"),
        (r"\b(westbound|wb|w)\b", "wb"),
    ]
    for pat, code in dir_patterns:
        if re.search(pat, cleaned):
            direction = code
            break

    def build(prefix: str, num: str) -> str:
        key = f"{prefix}{int(num)}"
        if direction:
            key += f"-{direction}"
        return key

    def build_state(num: str) -> str:
        prefix = state_prefix if state_prefix else "route"
        return build(prefix, num)

    # Interstate patterns: i 95, i95, interstate 95, i-95
    if re.search(r"\binterstate\b", cleaned) or re.search(r"\bi\s*\d+\b", cleaned):
        m = re.search(r"(?:\binterstate\b|\bi\b)\s*(\d+)\b", cleaned)
        if m:
            return build("i", m.group(1))

    # US route patterns: us 95, u s 95, u.s. 95, united states route 95
    if (
        re.search(r"\bunited states route\b", cleaned)
        or re.search(r"\bus\s*\d+\b", cleaned)
        or re.search(r"\bu\s*s\s*\d+\b", cleaned)
    ):
        m = re.search(r"(?:\bunited states route\b|\bus\b|\bu\s*s\b)\s*(\d+)\b", cleaned)
        if m:
            return build("us", m.group(1))

    # State route / generic route patterns: sr 95, state route 95, route 95, hwy 95, highway 95
    if re.search(r"\b(state route|sr|route|hwy|highway)\b", cleaned):
        m = re.search(r"(?:\bstate route\b|\bsr\b|\broute\b|\bhwy\b|\bhighway\b)\s*(\d+)\b", cleaned)
        if m:
            return build_state(m.group(1))

    # If input is ONLY a number, do not guess. Treat as generic route<number>.
    m_only = re.fullmatch(r"\s*(\d+)\s*", cleaned)
    if m_only:
        return build("route", m_only.group(1))

    # Fallback: slugify-ish
    return normalize_text_basic(value)

class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    username = db.Column(db.String(20), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, index=True)

    # Password reset rate limiting (per account)
    reset_req_window_start = db.Column(db.DateTime, nullable=True)
    reset_req_count = db.Column(db.Integer, nullable=True)

    reports = db.relationship("SpeedReport", backref="user", lazy=True)

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    def __repr__(self) -> str:
        return f"<User id={self.id} username={self.username!r}>"

class SpeedReport(db.Model):
    __tablename__ = "speed_reports"

    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True, index=True)

    # Raw user input
    state = db.Column(db.String(100), nullable=False)

    # Road/location raw input
    road_name = db.Column(db.String(200), nullable=False)

    # Normalized key used for grouping/aggregation
    road_key = db.Column(db.String(200), nullable=False, index=True)

    posted_speed = db.Column(db.Integer, nullable=False, index=True)
    ticketed_speed = db.Column(db.Integer, nullable=False)

    notes = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, index=True)

    # OCR verification (async)
    # pending -> show 'Pending verification'
    # verified -> show 'Verified by OCR' (green check)
    # unverified -> show nothing
    verification_status = db.Column(db.String(20), nullable=False, default="unverified", index=True)
    verified_at = db.Column(db.DateTime, nullable=True)

    # Stored for internal tuning/debug only (never shown publicly)
    ocr_posted_speed = db.Column(db.Integer, nullable=True)
    ocr_ticketed_speed = db.Column(db.Integer, nullable=True)
    ocr_confidence = db.Column(db.Float, nullable=True)
    verify_attempts = db.Column(db.Integer, nullable=False, default=0)
    verify_reason = db.Column(db.String(50), nullable=True)

    # Photo + OCR job tracking (R2 quarantine + RQ)
    photo_key = db.Column(db.Text, nullable=True)
    photo_uploaded_at = db.Column(db.DateTime, nullable=True)
    ocr_status = db.Column(db.String(20), nullable=True)  # pending|processing|verified|unverified|failed
    ocr_job_id = db.Column(db.String(64), nullable=True)
    ocr_error = db.Column(db.Text, nullable=True)

    # Mapping (optional)
    location_hint = db.Column(db.String(200), nullable=True)

    # Raw user-selected coordinates (before snapping). Stored for tuning/analytics.
    raw_lat = db.Column(db.Float, nullable=True)
    raw_lng = db.Column(db.Float, nullable=True)

    # Final coordinates used for mapping (may be snapped to nearest road).
    lat = db.Column(db.Float, nullable=True, index=True)
    lng = db.Column(db.Float, nullable=True, index=True)

    # Where lat/lng came from: none|user_pin|user_pin_snapped
    lat_lng_source = db.Column(db.String(30), nullable=True)

    # Place ID from Places Autocomplete (routes can return a place_id)
    google_place_id = db.Column(db.String(128), nullable=True)


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure road_key stays consistent with latest normalization rules
        self.road_key = normalize_road(self.road_name, self.state)

    def refresh_road_key(self) -> None:
        self.road_key = normalize_road(self.road_name, self.state)

    def __repr__(self) -> str:
        return (
            f"<SpeedReport id={self.id} state={self.state!r} road_name={self.road_name!r} "
            f"road_key={self.road_key!r} posted={self.posted_speed} ticketed={self.ticketed_speed} "
        )


class Like(db.Model):
    __tablename__ = "likes"

    id = db.Column(db.Integer, primary_key=True)
    report_id = db.Column(db.Integer, db.ForeignKey("speed_reports.id"), nullable=False, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, index=True)

    __table_args__ = (
        db.UniqueConstraint("report_id", "user_id", name="uq_likes_report_user"),
    )


class Comment(db.Model):
    __tablename__ = "comments"

    id = db.Column(db.Integer, primary_key=True)
    report_id = db.Column(db.Integer, db.ForeignKey("speed_reports.id"), nullable=False, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)

    body = db.Column(db.String(280), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, index=True)

    # Reserved for threaded replies (v3)
    parent_id = db.Column(db.Integer, db.ForeignKey("comments.id"), nullable=True, index=True)

    user = db.relationship("User", lazy=True)

