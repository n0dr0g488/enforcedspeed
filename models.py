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
    phone = db.Column(db.String(25), nullable=True, index=True)
    birthdate = db.Column(db.Date, nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, index=True)

    # Password reset rate limiting (per account)
    reset_req_window_start = db.Column(db.DateTime, nullable=True)
    reset_req_count = db.Column(db.Integer, nullable=True)

    reports = db.relationship(
        "SpeedReport",
        foreign_keys="SpeedReport.user_id",
        backref=db.backref("user", lazy=True),
        lazy=True,
    )

    # Reports this user soft-deleted (admin moderation)
    deleted_reports = db.relationship(
        "SpeedReport",
        foreign_keys="SpeedReport.deleted_by",
        backref=db.backref("deleted_by_user", lazy=True),
        lazy=True,
    )

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    def __repr__(self) -> str:
        return f"<User id={self.id} username={self.username!r}>"

class SpeedReport(db.Model):
    __tablename__ = 'speed_reports'

    id = db.Column(db.Integer, primary_key=True)

    # Ownership
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)

    # Core inputs
    state = db.Column(db.String(100), nullable=False)
    route_class = db.Column(db.String(20), nullable=True)
    road_name = db.Column(db.String(200), nullable=False)
    # Normalized, stable road identity key (used for grouping/filtering).
    # Must match normalize_road() so URL filters and analytics group correctly.
    road_key = db.Column(db.String(120), nullable=True, index=True)

    posted_speed = db.Column(db.Integer, nullable=False)
    ticketed_speed = db.Column(db.Integer, nullable=False)
    overage = db.Column(db.Integer, nullable=True, index=True)

    caption = db.Column(db.String(500), nullable=True)

    # County-first (required by app; nullable for legacy rows)
    county_geoid = db.Column(db.String(10), nullable=True, index=True)
    county_name = db.Column(db.String(120), nullable=True)
    county_state = db.Column(db.String(2), nullable=True, index=True)

    # Location
    raw_lat = db.Column(db.Float, nullable=True)
    raw_lng = db.Column(db.Float, nullable=True)
    lat = db.Column(db.Float, nullable=True)
    lng = db.Column(db.Float, nullable=True)
    location_hint = db.Column(db.String(120), nullable=True)
    location_source = db.Column(db.String(40), nullable=True)
    location_accuracy_m = db.Column(db.Integer, nullable=True)

    # Evidence / OCR / verification
    photo_key = db.Column(db.String(255), nullable=True)
    ocr_status = db.Column(db.String(30), nullable=False, default='pending')  # pending|processing|done|failed
    # Internal error details from OCR pipeline (short code / message).
    ocr_error = db.Column(db.String(255), nullable=True)
    verification_status = db.Column(db.String(30), nullable=False, default='pending')  # pending|verified|rejected

    ocr_posted_speed = db.Column(db.Integer, nullable=True)
    ocr_ticketed_speed = db.Column(db.Integer, nullable=True)
    ocr_confidence = db.Column(db.Float, nullable=True)

    verified_at = db.Column(db.DateTime, nullable=True)
    verified_by = db.Column(db.Integer, nullable=True)
    verify_attempts = db.Column(db.Integer, nullable=False, default=0)
    verify_reason = db.Column(db.String(120), nullable=True)

    # Soft delete
    is_deleted = db.Column(db.Boolean, nullable=False, default=False)
    deleted_at = db.Column(db.DateTime, nullable=True)
    # Who soft-deleted this report (admin / moderator). This must be a ForeignKey so
    # SQLAlchemy can build the User.deleted_reports relationship reliably.
    deleted_by = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    comments = db.relationship('Comment', backref='report', lazy=True, cascade='all, delete-orphan')
    likes = db.relationship('Like', backref='report', lazy=True, cascade='all, delete-orphan')

    def refresh_road_key(self) -> None:
        """Compute and store road_key using the shared normalize_road() logic."""
        try:
            self.road_key = normalize_road(self.road_name or "", self.state or "")
        except Exception:
            # Never fail a write because of a normalization edge case.
            self.road_key = (self.road_name or "").strip().lower()[:120] or None

    def __repr__(self):
        return f"<SpeedReport {self.id} {self.county_state}:{self.county_name}>"


class Like(db.Model):
    """A single user 'like' on a specific SpeedReport.

    One-like-per-user-per-report is enforced with a unique constraint.
    """

    __tablename__ = "likes"

    id = db.Column(db.Integer, primary_key=True)
    report_id = db.Column(db.Integer, db.ForeignKey("speed_reports.id"), nullable=False, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    __table_args__ = (
        db.UniqueConstraint("report_id", "user_id", name="uq_like_report_user"),
    )

    user = db.relationship("User", backref=db.backref("likes", lazy=True, cascade="all, delete-orphan"))

    def __repr__(self):
        return f"<Like report={self.report_id} user={self.user_id}>"

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



class FollowedCounty(db.Model):
    __tablename__ = "user_follow_counties"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    county_geoid = db.Column(db.String(10), nullable=False, index=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, index=True)

    __table_args__ = (
        db.UniqueConstraint("user_id", "county_geoid", name="uq_follow_user_county"),
    )

    user = db.relationship("User", backref=db.backref("followed_counties", lazy=True, cascade="all, delete-orphan"))

    def __repr__(self):
        return f"<FollowedCounty user={self.user_id} geoid={self.county_geoid}>"
