# app.py
from __future__ import annotations

import os

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
from datetime import datetime, timedelta, timezone
from statistics import median
from bisect import bisect_right
from typing import Dict, List

import jwt
from jwt import ExpiredSignatureError, InvalidTokenError

from flask import Flask, render_template, redirect, url_for, jsonify, request, flash, abort, Response
from flask_login import LoginManager, login_user, logout_user, current_user, login_required
from sqlalchemy import text, func, tuple_, or_, and_
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
    RegisterForm,
    LoginForm,
    ChangePasswordForm,
    ForgotPasswordForm,
    ResetPasswordForm,
    LikeForm,
    CommentForm,
    DeleteCommentForm,
)
from models import db, SpeedReport, User, Like, Comment, normalize_road, state_code_from_value

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
        "markers": f"color:red|{center}",
        "key": key,
    }
    return "https://maps.googleapis.com/maps/api/staticmap?" + urllib.parse.urlencode(params)


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


def ensure_schema_patches() -> None:
    # Add nullable user_id to speed_reports for existing databases.
    if not column_exists("speed_reports", "user_id"):
        db.session.execute(text("ALTER TABLE speed_reports ADD COLUMN user_id INTEGER"))
        db.session.commit()

    # Add OCR verification fields to speed_reports for existing databases.
    if not column_exists("speed_reports", "verification_status"):
        db.session.execute(text("ALTER TABLE speed_reports ADD COLUMN verification_status VARCHAR(20) NOT NULL DEFAULT 'unverified'"))
        db.session.commit()

    if not column_exists("speed_reports", "verified_at"):
        db.session.execute(text("ALTER TABLE speed_reports ADD COLUMN verified_at TIMESTAMP"))
        db.session.commit()

    if not column_exists("speed_reports", "ocr_posted_speed"):
        db.session.execute(text("ALTER TABLE speed_reports ADD COLUMN ocr_posted_speed INTEGER"))
        db.session.commit()

    if not column_exists("speed_reports", "ocr_ticketed_speed"):
        db.session.execute(text("ALTER TABLE speed_reports ADD COLUMN ocr_ticketed_speed INTEGER"))
        db.session.commit()

    if not column_exists("speed_reports", "ocr_confidence"):
        db.session.execute(text("ALTER TABLE speed_reports ADD COLUMN ocr_confidence DOUBLE PRECISION"))
        db.session.commit()

    if not column_exists("speed_reports", "verify_attempts"):
        db.session.execute(text("ALTER TABLE speed_reports ADD COLUMN verify_attempts INTEGER NOT NULL DEFAULT 0"))
        db.session.commit()

    if not column_exists("speed_reports", "verify_reason"):
        db.session.execute(text("ALTER TABLE speed_reports ADD COLUMN verify_reason VARCHAR(50)"))
        db.session.commit()

    # Add password reset rate limit columns to users for existing databases.
    if not column_exists("users", "reset_req_window_start"):
        db.session.execute(text("ALTER TABLE users ADD COLUMN reset_req_window_start TIMESTAMP"))
        db.session.commit()

    if not column_exists("users", "reset_req_count"):
        db.session.execute(text("ALTER TABLE users ADD COLUMN reset_req_count INTEGER"))
        db.session.commit()


    # Soft delete (admin moderation) columns for speed_reports.
    if not column_exists("speed_reports", "is_deleted"):
        db.session.execute(text("ALTER TABLE speed_reports ADD COLUMN is_deleted BOOLEAN NOT NULL DEFAULT FALSE"))
        db.session.commit()

    if not column_exists("speed_reports", "deleted_at"):
        db.session.execute(text("ALTER TABLE speed_reports ADD COLUMN deleted_at TIMESTAMP"))
        db.session.commit()

    if not column_exists("speed_reports", "deleted_by"):
        db.session.execute(text("ALTER TABLE speed_reports ADD COLUMN deleted_by INTEGER"))
        db.session.commit()




    # Photo + OCR job tracking columns for speed_reports.
    if not column_exists("speed_reports", "photo_key"):
        db.session.execute(text("ALTER TABLE speed_reports ADD COLUMN photo_key TEXT"))
        db.session.commit()

    if not column_exists("speed_reports", "photo_uploaded_at"):
        db.session.execute(text("ALTER TABLE speed_reports ADD COLUMN photo_uploaded_at TIMESTAMP"))
        db.session.commit()

    if not column_exists("speed_reports", "ocr_status"):
        db.session.execute(text("ALTER TABLE speed_reports ADD COLUMN ocr_status VARCHAR(20)"))
        db.session.commit()

    if not column_exists("speed_reports", "ocr_job_id"):
        db.session.execute(text("ALTER TABLE speed_reports ADD COLUMN ocr_job_id VARCHAR(64)"))
        db.session.commit()

    if not column_exists("speed_reports", "ocr_error"):
        db.session.execute(text("ALTER TABLE speed_reports ADD COLUMN ocr_error TEXT"))
        db.session.commit()

    # Mapping (optional) columns for speed_reports.
    if not column_exists("speed_reports", "location_hint"):
        db.session.execute(text("ALTER TABLE speed_reports ADD COLUMN location_hint VARCHAR(200)"))
        db.session.commit()

    if not column_exists("speed_reports", "raw_lat"):
        db.session.execute(text("ALTER TABLE speed_reports ADD COLUMN raw_lat DOUBLE PRECISION"))
        db.session.commit()

    if not column_exists("speed_reports", "raw_lng"):
        db.session.execute(text("ALTER TABLE speed_reports ADD COLUMN raw_lng DOUBLE PRECISION"))
        db.session.commit()

    if not column_exists("speed_reports", "lat"):
        db.session.execute(text("ALTER TABLE speed_reports ADD COLUMN lat DOUBLE PRECISION"))
        db.session.commit()

    if not column_exists("speed_reports", "lng"):
        db.session.execute(text("ALTER TABLE speed_reports ADD COLUMN lng DOUBLE PRECISION"))
        db.session.commit()

    if not column_exists("speed_reports", "lat_lng_source"):
        db.session.execute(text("ALTER TABLE speed_reports ADD COLUMN lat_lng_source VARCHAR(30)"))
        db.session.commit()

    if not column_exists("speed_reports", "google_place_id"):
        db.session.execute(text("ALTER TABLE speed_reports ADD COLUMN google_place_id VARCHAR(128)"))
        db.session.commit()

    # Add parent_id to comments for threaded replies.
    if not column_exists("comments", "parent_id"):
        db.session.execute(text("ALTER TABLE comments ADD COLUMN parent_id INTEGER"))
        db.session.commit()



def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)
    db.init_app(app)

    login_manager = LoginManager()
    login_manager.login_view = "login"
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(user_id: str):
        return User.query.get(int(user_id))

    @app.context_processor
    def inject_helpers():
        return {"format_road_bucket": format_road_bucket, "static_map_url": static_map_url}




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

    with app.app_context():
        try:
            db.create_all()
            ensure_schema_patches()
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

            if User.query.filter_by(email=email).first():
                flash("That email is already registered. Please log in.", "error")
                return redirect(url_for("login", next=request.args.get('next')))

            if User.query.filter(func.lower(User.username) == username.lower()).first():
                flash("That username is taken. Try another.", "error")
                return render_template("register.html", form=form)

            user = User(email=email, username=username)
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
        speed_limit_list: List[str] | None = None,
        over_list: List[str] | None = None,
        photo_only: bool = False,
        verify: str = "any",
        date: str = "any",
        deleted_mode: str = "hide",
    ) -> Dict[str, List[Dict]]:
        """Return Most Strict and Least Strict rankings.

        Strictness is based on the MEDIAN overage (ticketed - posted), considering only tickets where
        ticketed_speed > posted_speed. Lower median overage = more strict.

        IMPORTANT: This function supports the shared filter rail inputs so the strictness page
        filters behave the same way as the other pages.
        """

        speed_limit_list = speed_limit_list or []
        over_list = over_list or []

        expr = (SpeedReport.ticketed_speed - SpeedReport.posted_speed)

        q = (
            db.session.query(
                SpeedReport.state.label("state"),
                SpeedReport.road_key.label("road_key"),
                SpeedReport.road_name.label("road_name"),
                SpeedReport.posted_speed.label("posted_speed"),
                SpeedReport.ticketed_speed.label("ticketed_speed"),
                expr.label("overage"),
                SpeedReport.created_at.label("created_at"),
                SpeedReport.photo_key.label("photo_key"),
                SpeedReport.ocr_status.label("ocr_status"),
                SpeedReport.verification_status.label("verification_status"),
                SpeedReport.user_id.label("user_id"),
            )
            .filter(SpeedReport.ticketed_speed > SpeedReport.posted_speed)
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

        # Date filter
        if date not in ("", "any"):
            try:
                days = int(date)
            except Exception:
                days = 0
            if days > 0:
                q = q.filter(SpeedReport.created_at >= (datetime.utcnow() - timedelta(days=days)))

        # Road filter (forgiving: match normalized bucket OR partial raw text)
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
                if v == "25-35":
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
                if v == "5-9":
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
            key = (normalize_state_group(r.state), r.road_key)
            g = groups.get(key)
            if g is None:
                g = {
                    "state": key[0],
                    "road": key[1],
                    "overages": [],
                    "tickets": 0,
                    "anon_tickets": 0,
                    "member_tickets": 0,
                }
                groups[key] = g

            g["overages"].append(int(r.overage))
            g["tickets"] += 1

            if r.user_id is None:
                g["anon_tickets"] += 1
            else:
                g["member_tickets"] += 1

        results: List[Dict] = []
        for g in groups.values():
            med = float(median(g["overages"])) if g["overages"] else 0.0
            results.append(
                {
                    "state": g["state"],
                    "road": g["road"],
                    "median_overage": round(med, 1),
                    "tickets": int(g["tickets"]),
                    "anon_pct": int(round((g["anon_tickets"] / g["tickets"]) * 100)) if g["tickets"] else 0,
                    "member_pct": (100 - int(round((g["anon_tickets"] / g["tickets"]) * 100))) if g["tickets"] else 0,
                }
            )

        # Primary sort key: median_overage (asc for strict, desc for least)
        # Tie-breakers: tickets (desc), then (state, road) alphabetical
        most_strict = sorted(
            results,
            key=lambda x: (x["median_overage"], -x["tickets"], x["state"], x["road"]),
        )[:limit]

        least_strict = sorted(
            results,
            key=lambda x: (-x["median_overage"], -x["tickets"], x["state"], x["road"]),
        )[:limit]

        return {"most_strict": most_strict, "least_strict": least_strict}

    @app.get("/")
    def home():
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

        # Date filter (created_at)
        filter_date = (request.args.get("date") or "any").strip()

        # Back-compat: old ?verified_photo=1 means 'verified photos only'
        if request.args.get("verified_photo") == "1":
            filter_photo_only = True
            filter_verify = "verified"

        # Sort
        filter_sort = (request.args.get("sort") or "new").strip()

        # State dropdown options (2-letter codes)
        state_filter_options = []
        rows = db.session.query(SpeedReport.state).distinct().all()
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
            {"value": "verified", "label": "Verified (Automated)"},
            {"value": "not_verified", "label": "Not verified"},
        ]

        filters_active = bool(
            filter_state
            or filter_road
            or bool(filter_speed_limit_list)
            or bool(filter_over_list)
            or filter_photo_only
            or (filter_verify not in ("", "any"))
            or (filter_date not in ("", "any"))
            or (filter_sort not in ("", "new"))
            or hide_anon
        )

        # Base query
        q = SpeedReport.query.options(joinedload(SpeedReport.user))

        # Deleted visibility (admin: ?deleted=hide|include|only)
        if deleted_only:
            q = q.filter(SpeedReport.is_deleted.is_(True))
        elif deleted_only:
            q = q.filter(SpeedReport.is_deleted.is_(True))
        elif deleted_only:
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
                if v == "25-35":
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
                if v == "5-9":
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
            road_suggestions = []

        # Compute per-post min/max percentiles using only group min/max (fast, stable meaning).
        keys_a = {(r.state, r.road_key) for r in reports}  # Road + State group
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
            speed_limit_buckets=speed_limit_buckets,
            filter_speed_limit_list=filter_speed_limit_list,
            over_buckets=over_buckets,
            filter_over_list=filter_over_list,
            filter_photo_only=filter_photo_only,
            verify_options=verify_options,
            filter_verify=filter_verify,
            date_options=date_options,
            filter_date=filter_date,
            road_suggestions=road_suggestions,
            filter_sort=filter_sort,
            filters_active=filters_active,
            maps_api_key=app.config.get("GOOGLE_MAPS_API_KEY") or "",
            map_pins=map_pins,
            is_admin=is_admin,
            show_deleted=show_deleted,
            deleted_mode=deleted_mode,
            deleted_only=deleted_only,
        )


    @app.get("/map")
    def map_view():
        """Map view of tickets that have lat/lng."""
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
        filter_date = (request.args.get("date") or "any").strip()
        filter_sort = (request.args.get("sort") or "new").strip()

        if request.args.get("verified_photo") == "1":
            filter_photo_only = True
            filter_verify = "verified"

        # Options for filter rail (mirrors home).
        state_filter_options = []
        rows = db.session.query(SpeedReport.state).distinct().all()
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
            {"value": "verified", "label": "Verified (Automated)"},
            {"value": "not_verified", "label": "Not verified"},
        ]

        filters_active = bool(
            filter_state
            or filter_road
            or bool(filter_speed_limit_list)
            or bool(filter_over_list)
            or filter_photo_only
            or (filter_verify not in ("", "any"))
            or (filter_date not in ("", "any"))
            or hide_anon
        )

        return render_template(
            "map_page.html",
            hide_anon=hide_anon,
            state_filter_options=state_filter_options,
            filter_state=filter_state,
            filter_road=filter_road,
            speed_limit_buckets=speed_limit_buckets,
            filter_speed_limit_list=filter_speed_limit_list,
            over_buckets=over_buckets,
            filter_over_list=filter_over_list,
            filter_photo_only=filter_photo_only,
            verify_options=verify_options,
            filter_verify=filter_verify,
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


    @app.get("/submit")
    def submit():
        form = SpeedReportForm()
        ticket_count = SpeedReport.query.filter(SpeedReport.is_deleted.is_(False)).count()

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

        strictness = strictness_rows(limit=5, exclude_anonymous=hide_anon)
        most_strict = strictness["most_strict"]
        least_strict = strictness["least_strict"]

        return render_template(
            "mvp_home.html",
            form=form,
            ticket_count=ticket_count,
            most_strict=most_strict,
            least_strict=least_strict,
            hide_anon=hide_anon,
            state_options=STATE_OPTIONS,
            maps_api_key=app.config.get("GOOGLE_MAPS_API_KEY") or "",
        )


    @app.post("/submit")
    def submit_ticket():
        form = SpeedReportForm()

        if not form.validate_on_submit():
            # Make failure obvious (common case: photo format rejected by WTForms).
            flash("Ticket not submitted. Please fix the highlighted errors below.", "warning")
            return render_template(
                "mvp_home.html",
                form=form,
                ticket_count=SpeedReport.query.filter(SpeedReport.is_deleted.is_(False)).count(),
                most_strict=strictness_rows(limit=5, exclude_anonymous=bool(current_user.is_authenticated and request.args.get("hide_anon") == "1"))["most_strict"],
                least_strict=strictness_rows(limit=5, exclude_anonymous=bool(current_user.is_authenticated and request.args.get("hide_anon") == "1"))["least_strict"],
                hide_anon=bool(current_user.is_authenticated and request.args.get("hide_anon") == "1"),
                state_options=STATE_OPTIONS,
                maps_api_key=app.config.get("GOOGLE_MAPS_API_KEY") or "",
            )


        report = SpeedReport(
            state=form.state.data,
            road_name=form.road_name.data,
            posted_speed=form.posted_speed.data,
            ticketed_speed=form.ticketed_speed.data,
            notes=None,
            user_id=(current_user.id if current_user.is_authenticated else None),
        )
        report.refresh_road_key()

        # Optional mapping fields (filled by client-side JS).
        try:
            report.location_hint = (getattr(form, "location_hint", None).data or "").strip() or None
        except Exception:
            report.location_hint = None

        def _to_float(val):
            try:
                if val is None:
                    return None
                s = str(val).strip()
                if not s:
                    return None
                return float(s)
            except Exception:
                return None

        raw_lat = _to_float(getattr(form, "raw_lat", None).data if getattr(form, "raw_lat", None) else None)
        raw_lng = _to_float(getattr(form, "raw_lng", None).data if getattr(form, "raw_lng", None) else None)
        lat = _to_float(getattr(form, "lat", None).data if getattr(form, "lat", None) else None)
        lng = _to_float(getattr(form, "lng", None).data if getattr(form, "lng", None) else None)

        report.raw_lat = raw_lat
        report.raw_lng = raw_lng
        report.lat = lat
        report.lng = lng
        if lat is not None and lng is not None:
            report.lat_lng_source = "user_pin"

            # If a server key is configured, snap to the nearest road before saving.
            slat, slng = snap_to_nearest_road(lat, lng)
            if slat is not None and slng is not None:
                # Preserve raw pin if missing.
                if report.raw_lat is None:
                    report.raw_lat = lat
                if report.raw_lng is None:
                    report.raw_lng = lng
                report.lat = slat
                report.lng = slng
                report.lat_lng_source = "user_pin_snapped"

        try:
            report.google_place_id = (getattr(form, "google_place_id", None).data or "").strip() or None
        except Exception:
            report.google_place_id = None

        db.session.add(report)
        db.session.commit()

        
        def _normalize_upload_to_jpeg_bytes(raw_bytes: bytes, filename: str | None, content_type: str | None) -> bytes:
            """Normalize a user upload (image or PDF) into a sanitized JPEG.

            - Strips metadata by re-encoding.
            - Converts to RGB.
            - Resizes to max 2000px.
            - If PDF: renders page 1 to an image first.
            """
            fname = (filename or "").lower()
            ctype = (content_type or "").lower()

            is_pdf = fname.endswith(".pdf") or ("application/pdf" in ctype)

            if is_pdf:
                if fitz is None:
                    raise ValueError("pdf_support_missing")
                doc = fitz.open(stream=raw_bytes, filetype="pdf")
                if doc.page_count < 1:
                    raise ValueError("empty_pdf")
                page = doc.load_page(0)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                # Render to PNG bytes, then load with Pillow to normalize like images.
                raw_bytes = pix.tobytes("png")

            im = Image.open(io.BytesIO(raw_bytes))
            im = im.convert("RGB")
            im.thumbnail((2000, 2000))
            out = io.BytesIO()
            im.save(out, format="JPEG", quality=80, optimize=True)
            return out.getvalue()

        # Optional photo upload -> quarantine in R2 + async OCR verification (fail closed).
        photo_fs = getattr(form, "photo", None)
        file_obj = photo_fs.data if photo_fs is not None else None
        if file_obj and getattr(file_obj, "filename", ""):
            try:
                # Mark pending first so the feed can show status immediately.
                report.verification_status = "pending"
                report.ocr_status = "pending"
                report.ocr_error = None
                report.verify_reason = None
                db.session.commit()

                # Load + normalize + strip metadata by re-encoding.
                raw = file_obj.read()
                try:
                    jpg_bytes = _normalize_upload_to_jpeg_bytes(
                        raw,
                        getattr(file_obj, "filename", None),
                        getattr(file_obj, "content_type", None),
                    )
                except Exception:
                    report.verification_status = "unverified"
                    report.ocr_status = "not_verified"
                    report.ocr_error = "unsupported_or_unreadable_upload"
                    report.verify_reason = "unsupported_format"
                    db.session.commit()
                    flash("Photo upload failed: please upload a clear image (JPG/PNG/WebP/HEIC) or a PDF.", "warning")
                    return redirect(url_for("result", report_id=report.id))


                # Ensure required R2 env vars exist (local dev often forgets these).
                missing = [k for k in ("R2_ENDPOINT", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY") if not (os.environ.get(k) or "").strip()]
                if missing:
                    report.verification_status = "unverified"
                    report.ocr_status = "not_verified"
                    report.ocr_error = "missing_r2_env: " + ",".join(missing)
                    report.verify_reason = "upload_failed"
                    db.session.commit()
                    flash("Photo upload failed: missing local R2 settings: " + ", ".join(missing), "warning")
                    return redirect(url_for("result", report_id=report.id))

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
                    db.session.commit()
                else:
                    # Persist where the photo lives so the worker (and UI) can find it later.
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
            except Exception as e:
                # Store a debuggable error string (truncated) instead of a generic placeholder.
                try:
                    app.logger.exception("Photo upload/OCR enqueue failed for report_id=%s", report.id)
                except Exception:
                    pass
                report.verification_status = "unverified"
                report.ocr_status = "not_verified"
                err = f"{type(e).__name__}: {e}"
                report.ocr_error = (err[:500] if err else "exception")
                report.verify_reason = "exception"
                db.session.commit()

        return redirect(url_for("result", report_id=report.id))

    def compute_distribution_stats(values: List[float], user_value: float | None) -> Dict:
        """Return distribution stats + a min-max percentile (0–100).

        Percentile definition used here:
          0%  -> lowest value in group
          100%-> highest value in group
        This matches the product meaning for 'strictness score' comparisons.
        """
        vals = [v for v in values if v is not None]
        n = len(vals)
        if n == 0:
            return {"n": 0, "min": None, "max": None, "median": None, "percentile": None}

        vals_sorted = sorted(vals)
        vmin = vals_sorted[0]
        vmax = vals_sorted[-1]
        med = float(median(vals_sorted))

        if user_value is None:
            pct = None
        else:
            if vmax == vmin:
                pct = 100.0
            else:
                pct = round(((user_value - vmin) / (vmax - vmin)) * 100, 1)
                # clamp
                pct = max(0.0, min(100.0, pct))

        return {"n": n, "min": vmin, "max": vmax, "median": round(med, 1), "percentile": pct}


    @app.get("/strictness")
    def strictness():
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

        # --- Filters (GET) — used to render the shared filter rail UI ---
        filter_state = (request.args.get("state") or "").strip().upper()
        filter_road = (request.args.get("road") or "").strip()

        # Speed limit buckets (posted_speed) — multi-select via repeated speed_limit= values
        filter_speed_limit_list = [v.strip() for v in request.args.getlist("speed_limit") if (v or "").strip()]
        filter_speed_limit_list = [v for v in filter_speed_limit_list if v != "any"]
        filter_speed_limit_list = list(dict.fromkeys(filter_speed_limit_list))

        # Overage buckets (ticketed_speed - posted_speed) — multi-select via repeated overage= values
        filter_over_list = [v.strip() for v in request.args.getlist("overage") if (v or "").strip()]
        legacy_over = (request.args.get("over") or "").strip()
        if legacy_over and legacy_over not in ("all", ""):
            filter_over_list = list(dict.fromkeys(filter_over_list + [legacy_over]))

        # Evidence filters (kept for UI consistency; strictness calculation is already limited to ticketed > posted)
        filter_photo_only = (request.args.get("photo_only") == "1")
        filter_verify = (request.args.get("verify") or "any").strip()

        # Date filter (created_at)
        filter_date = (request.args.get("date") or "any").strip()

        # Back-compat: old ?verified_photo=1 means 'verified photos only'
        if request.args.get("verified_photo") == "1":
            filter_photo_only = True
            filter_verify = "verified"

        # Sort (UI only on this page)
        filter_sort = (request.args.get("sort") or "new").strip()

        # State dropdown options (2-letter codes)
        state_filter_options = []
        rows = db.session.query(SpeedReport.state).distinct().all()
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

        filters_active = bool(
            filter_state
            or filter_road
            or bool(filter_speed_limit_list)
            or bool(filter_over_list)
            or filter_photo_only
            or (filter_verify not in ("", "any"))
            or (filter_date not in ("", "any"))
            or (filter_sort not in ("", "new"))
            or hide_anon
            or (is_admin and deleted_mode != "hide")
        )

        strictness = strictness_rows(
            limit=20,
            exclude_anonymous=hide_anon,
            state_filter=(filter_state or None),
            road_filter=(filter_road or None),
            speed_limit_list=filter_speed_limit_list,
            over_list=filter_over_list,
            photo_only=filter_photo_only,
            verify=filter_verify,
            date=filter_date,
            deleted_mode=deleted_mode,
        )
        most_strict = strictness["most_strict"]
        least_strict = strictness["least_strict"]

        return render_template(
            "strictness.html",
            most_strict=most_strict,
            least_strict=least_strict,
            hide_anon=hide_anon,
            # filter rail context
            filter_state=filter_state,
            filter_road=filter_road,
            filter_speed_limit_list=filter_speed_limit_list,
            filter_over_list=filter_over_list,
            filter_photo_only=filter_photo_only,
            filter_verify=filter_verify,
            filter_date=filter_date,
            filter_sort=filter_sort,
            state_filter_options=state_filter_options,
            speed_limit_buckets=speed_limit_buckets,
            over_buckets=over_buckets,
            date_options=date_options,
            filters_active=filters_active,
            is_admin=is_admin,
            deleted_mode=deleted_mode,
        )


    @app.get("/tickets")
    def bucket_tickets():
        """Show up to 50 tickets for a given (state, road) combo."""
        state = (request.args.get("state") or "").strip().upper()
        road_key = (request.args.get("road") or "").strip()

        if not state or not road_key:
            flash("Missing state or road.", "error")
            return redirect(url_for("home"))

        # Match stored state strings like "KS - Kansas" using the two-letter prefix
        rows = (
            db.session.query(
                SpeedReport.state,
                SpeedReport.road_key,
                SpeedReport.posted_speed,
                SpeedReport.ticketed_speed,
                SpeedReport.created_at,
                User.username.label("username"),
                SpeedReport.user_id,
            )
            .outerjoin(User, User.id == SpeedReport.user_id)
            .filter(SpeedReport.is_deleted.is_(False))
            .filter(SpeedReport.state.ilike(f"{state}%"))
            .filter(SpeedReport.road_key == road_key)
            .order_by(SpeedReport.ticketed_speed.desc())
            .limit(50)
            .all()
        )

        total_tickets = len(rows)
        member_count = sum(1 for r in rows if getattr(r, "username", None))
        anon_count = total_tickets - member_count
        if total_tickets > 0:
            member_pct = int(round((member_count / total_tickets) * 100))
            # Keep the two-part pill readable and always summing to 100
            member_pct = max(0, min(100, member_pct))
            anon_pct = 100 - member_pct
        else:
            member_pct = 0
            anon_pct = 0

        return render_template(
            "bucket_tickets.html",
            state=state,
            road_key=road_key,
            rows=rows,
            total_tickets=total_tickets,
            member_pct=member_pct,
            anon_pct=anon_pct,
        )

    @app.get("/result/<int:report_id>")
    def result(report_id: int):
        report = SpeedReport.query.get_or_404(report_id)
        state_group = normalize_state_group(report.state)

        # Strictness score is "how many mph over posted" the ticketed speed is.
        user_strictness = int(report.ticketed_speed) - int(report.posted_speed)

        rows = (
            db.session.query(
                SpeedReport.state,
                SpeedReport.road_key,
                SpeedReport.posted_speed,
                SpeedReport.ticketed_speed,
            )
            .filter(SpeedReport.state.ilike(f"{state_group}%"))
            .all()
        )

        strict_state_road: List[float] = []
        strict_state_posted: List[float] = []

        for st, road_key, posted, ticketed in rows:
            if normalize_state_group(st) != state_group:
                continue
            strictness = float(int(ticketed) - int(posted))
            if road_key == report.road_key:
                strict_state_road.append(strictness)
            if int(posted) == int(report.posted_speed):
                strict_state_posted.append(strictness)

        stats_a = compute_distribution_stats(strict_state_road, float(user_strictness))
        stats_b = compute_distribution_stats(strict_state_posted, float(user_strictness))

        return render_template(
            "thank_you.html",
            report=report,
            stats_a=stats_a,
            stats_b=stats_b,
            state_group=state_group,
            user_strictness=user_strictness,
        )

    @app.get("/api/road_preview")
    def road_preview():
        state_raw = request.args.get("state", "", type=str)
        road_raw = request.args.get("road", "", type=str)

        predicted_key = normalize_road(road_raw, state_raw) if road_raw.strip() else ""
        base_key = base_road_key(predicted_key) if predicted_key else ""
        state_code = state_code_from_value(state_raw)

        # ambiguity helper: number-only input -> routeNN
        ambiguity_options = []
        m_num = re.fullmatch(r"\s*(\d+)\s*", re.sub(r"[^0-9]", "", road_raw.strip()) if road_raw else "")
        if predicted_key.startswith("route") and m_num:
            n = m_num.group(1)
            if n:
                if n.isdigit():
                    ambiguity_options = [
                        f"I-{int(n)}",
                        f"US-{int(n)}",
                        f"{state_code}-{int(n)}" if state_code else f"State Route {int(n)}",
                    ]

        # suggestions: similar buckets seen in this state
        suggestions = []
        if state_raw and predicted_key:
            suggestion_rows = (
                db.session.query(SpeedReport.road_key)
                .filter(SpeedReport.state == state_raw)
                .distinct()
                .all()
            )
            suggestion_keys = [r[0] for r in suggestion_rows]
            matches = [k for k in suggestion_keys if base_road_key(k) == base_key]
            matches = sorted(set(matches))[:10]
            suggestions = [{"key": k, "label": format_road_bucket(k)} for k in matches]

        return jsonify(
            {
                "predicted_road_key": predicted_key,
                "predicted_road_label": format_road_bucket(predicted_key),
                "base_key": base_key,
                "suggestions": suggestions,
                "ambiguity_options": ambiguity_options,
            }
        )

    @app.get("/api/snap_road")
    def api_snap_road():
        """Snap a lat/lng to the nearest road (server-side).

        This endpoint is optional; if GOOGLE_MAPS_SERVER_KEY is not configured, it returns the input.
        """
        lat = request.args.get("lat", type=float)
        lng = request.args.get("lng", type=float)
        if lat is None or lng is None:
            return jsonify({"ok": False, "error": "missing_lat_lng"}), 400

        slat, slng = snap_to_nearest_road(lat, lng)
        if slat is None or slng is None:
            return jsonify({"ok": True, "snapped": False, "lat": lat, "lng": lng})
        return jsonify({"ok": True, "snapped": True, "lat": slat, "lng": slng})

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
            r.lat_lng_source = "user_pin"
            snapped = False
        else:
            r.lat = float(slat)
            r.lng = float(slng)
            r.lat_lng_source = "user_pin_snapped"
            snapped = True

        db.session.commit()
        return jsonify({"ok": True, "snapped": snapped, "lat": r.lat, "lng": r.lng})

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

        filter_speed_limit_list = [v.strip() for v in request.args.getlist("speed_limit") if (v or "").strip()]
        filter_speed_limit_list = [v for v in filter_speed_limit_list if v != "any"]
        filter_speed_limit_list = list(dict.fromkeys(filter_speed_limit_list))

        filter_over_list = [v.strip() for v in request.args.getlist("overage") if (v or "").strip()]
        legacy_over = (request.args.get("over") or "").strip()
        if legacy_over and legacy_over not in ("all", ""):
            filter_over_list = list(dict.fromkeys(filter_over_list + [legacy_over]))

        filter_photo_only = (request.args.get("photo_only") == "1")
        filter_verify = (request.args.get("verify") or "any").strip()
        filter_date = (request.args.get("date") or "any").strip()
        filter_sort = (request.args.get("sort") or "new").strip()
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
                if v == "25-35":
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
                if v == "5-9":
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

        pins = []
        for r in rows:
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
                }
            )

        return jsonify({"ok": True, "count": len(pins), "pins": pins})

    
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

        posted_in = first('posted_speed', 'postedMph', 'posted_mph', 'posted')
        cited_in = first('cited_speed', 'ticketed_speed', 'ticketedMph', 'ticketed_mph', 'cited')

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

        lat = to_float(lat_in)
        lng = to_float(lng_in)
        if lat is not None and (lat < -90 or lat > 90):
            return jsonify({'error': 'lat must be between -90 and 90'}), 400
        if lng is not None and (lng < -180 or lng > 180):
            return jsonify({'error': 'lng must be between -180 and 180'}), 400

        # Road is preferred, but allow pin-only submissions (road OR lat/lng required).
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
            notes=None,
            user_id=(api_u.id if api_u else None),
        )
        report.refresh_road_key()

        if lat is not None and lng is not None:
            report.raw_lat = lat
            report.raw_lng = lng
            report.lat = lat
            report.lng = lng
            report.lat_lng_source = 'user_pin'

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

                # Mark pending first so the feed can show status immediately.
                report.verification_status = "pending"
                report.ocr_status = "pending"
                report.ocr_error = None
                report.verify_reason = None
                db.session.commit()

                raw = file_obj.read()
                try:
                    jpg_bytes = _normalize_upload_to_jpeg_bytes_api(raw, getattr(file_obj, "filename", None), getattr(file_obj, "content_type", None))
                except Exception:
                    report.verification_status = "unverified"
                    report.ocr_status = "not_verified"
                    report.ocr_error = "unsupported_or_unreadable_upload"
                    report.verify_reason = "unsupported_format"
                    db.session.commit()
                    jpg_bytes = None

                if jpg_bytes:
                    missing = [k for k in ("R2_ENDPOINT", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY") if not (os.environ.get(k) or "").strip()]
                    if missing:
                        report.verification_status = "unverified"
                        report.ocr_status = "not_verified"
                        report.ocr_error = "missing_r2_env: " + ",".join(missing)
                        report.verify_reason = "upload_failed"
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
        params = {
            "center": center,
            "zoom": str(int(zoom)),
            "size": f"{int(w)}x{int(h)}",
            "scale": "2",
            "maptype": "roadmap",
            "markers": f"color:red|{center}",
            "key": key,
        }
        upstream = "https://maps.googleapis.com/maps/api/staticmap?" + urllib.parse.urlencode(params)

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


    # --- API auth endpoints (JWT) ---
    def _user_public(u: User) -> dict:
        return {
            "id": u.id,
            "email": u.email,
            "username": u.username,
            "created_at": u.created_at.isoformat() if getattr(u, "created_at", None) else None,
        }

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

        ticket_posts = SpeedReport.query.filter(SpeedReport.user_id == u.id).count()
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

        SpeedReport.query.get_or_404(report_id)

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

    @app.get("/api/users/<int:user_id>")
    def api_user_public_profile(user_id: int):
        u = User.query.get_or_404(user_id)

        ticket_posts = SpeedReport.query.filter(SpeedReport.user_id == u.id).count()
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

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


app = create_app()
@app.route("/privacy")
def privacy():
    return render_template("privacy.html", last_updated="2026-01-25", contact_email="support@enforcedspeed.com")
@app.route("/terms")
def terms():
    return render_template("terms.html", last_updated="2026-01-25", contact_email="support@enforcedspeed.com")


if __name__ == "__main__":
    import os

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "1") == "1"

    app.run(host=host, port=port, debug=debug)