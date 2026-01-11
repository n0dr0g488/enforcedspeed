# app.py
from __future__ import annotations

import os
import re
from datetime import datetime, timedelta
from statistics import median
from typing import Dict, List

from flask import Flask, render_template, redirect, url_for, jsonify, request, flash
from flask_login import LoginManager, login_user, logout_user, current_user, login_required
from sqlalchemy import text, func, tuple_
from sqlalchemy.orm import joinedload
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

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
)
from models import db, SpeedReport, User, Like, Comment, normalize_road, state_code_from_value


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

    # Add password reset rate limit columns to users for existing databases.
    if not column_exists("users", "reset_req_window_start"):
        db.session.execute(text("ALTER TABLE users ADD COLUMN reset_req_window_start TIMESTAMP"))
        db.session.commit()

    if not column_exists("users", "reset_req_count"):
        db.session.execute(text("ALTER TABLE users ADD COLUMN reset_req_count INTEGER"))
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
        return {"format_road_bucket": format_road_bucket}

    @app.before_request
    def ensure_tables_exist():
        try:
            db.session.execute(text("SELECT 1"))
            db.create_all()
            ensure_schema_patches()
        except Exception:
            db.session.rollback()

    @app.route("/register", methods=["GET", "POST"])
    def register():
        if current_user.is_authenticated:
            return redirect(url_for("home"))

        form = RegisterForm()
        if form.validate_on_submit():
            email = form.email.data.strip().lower()
            username = form.username.data.strip().lower()

            if User.query.filter_by(email=email).first():
                flash("That email is already registered. Please log in.", "error")
                return redirect(url_for("login"))

            if User.query.filter_by(username=username).first():
                flash("That username is taken. Try another.", "error")
                return render_template("register.html", form=form)

            user = User(email=email, username=username)
            user.set_password(form.password.data)
            db.session.add(user)
            db.session.commit()

            login_user(user)
            return redirect(url_for("profile", username=user.username))

        return render_template("register.html", form=form)

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if current_user.is_authenticated:
            return redirect(url_for("home"))

        form = LoginForm()
        if form.validate_on_submit():
            identifier = form.email.data.strip().lower()

            # Allow login via username OR email (username first, then email).
            user = User.query.filter_by(username=identifier).first()
            if not user:
                user = User.query.filter_by(email=identifier).first()

            if not user or not user.check_password(form.password.data):
                flash("Invalid username/email or password.", "error")
                return render_template("login.html", form=form)

            login_user(user)
            return redirect(url_for("home"))

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
            identifier = form.identifier.data.strip().lower()

            user = User.query.filter_by(username=identifier).first()
            if not user:
                user = User.query.filter_by(email=identifier).first()

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
        u = User.query.filter_by(username=username.strip().lower()).first_or_404()

        reports = (
            SpeedReport.query.filter(SpeedReport.user_id == u.id)
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

    def strictness_rows(limit: int, exclude_anonymous: bool, state_filter: str | None = None) -> Dict[str, List[Dict]]:
        """Return Most Strict and Least Strict rankings.

        Strictness is based on the MEDIAN overage (ticketed - posted), considering only tickets where
        ticketed_speed > posted_speed. Lower median overage = more strict.
        """
        q = (
            db.session.query(
                SpeedReport.state.label("state"),
                SpeedReport.road_key.label("road_key"),
                (SpeedReport.ticketed_speed - SpeedReport.posted_speed).label("overage"),
                SpeedReport.created_at.label("created_at"),
                User.username.label("username"),
                SpeedReport.user_id.label("user_id"),
            )
            .outerjoin(User, User.id == SpeedReport.user_id)
            .filter(SpeedReport.ticketed_speed > SpeedReport.posted_speed)
        )

        if state_filter:
            q = q.filter(SpeedReport.state.ilike(f"{state_filter}%"))

        if exclude_anonymous:
            q = q.filter(SpeedReport.user_id.isnot(None))

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

        page = request.args.get("page", 1, type=int)
        per_page = 20

        q = SpeedReport.query.options(joinedload(SpeedReport.user)).order_by(SpeedReport.created_at.desc())
        if hide_anon:
            q = q.filter(SpeedReport.user_id.isnot(None))

        pagination = q.paginate(page=page, per_page=per_page, error_out=False)
        reports = pagination.items

        # Compute per-post min/max percentiles using only group min/max (fast, stable meaning).
        keys_a = {(r.state, r.road_key) for r in reports}  # Road + State group
        keys_b = {(r.state, r.posted_speed) for r in reports}  # State + Posted Speed group

        expr = (SpeedReport.ticketed_speed - SpeedReport.posted_speed)

        a_minmax = {}
        if keys_a:
            rows = (
                db.session.query(
                    SpeedReport.state,
                    SpeedReport.road_key,
                    func.min(expr).label("minv"),
                    func.max(expr).label("maxv"),
                )
                .filter(tuple_(SpeedReport.state, SpeedReport.road_key).in_(list(keys_a)))
                .group_by(SpeedReport.state, SpeedReport.road_key)
                .all()
            )
            for st, rk, minv, maxv in rows:
                a_minmax[(st, rk)] = (float(minv), float(maxv))

        b_minmax = {}
        if keys_b:
            rows = (
                db.session.query(
                    SpeedReport.state,
                    SpeedReport.posted_speed,
                    func.min(expr).label("minv"),
                    func.max(expr).label("maxv"),
                )
                .filter(tuple_(SpeedReport.state, SpeedReport.posted_speed).in_(list(keys_b)))
                .group_by(SpeedReport.state, SpeedReport.posted_speed)
                .all()
            )
            for st, ps, minv, maxv in rows:
                b_minmax[(st, ps)] = (float(minv), float(maxv))

        def _pct(minv: float | None, maxv: float | None, v: float) -> float:
            if minv is None or maxv is None:
                return 100.0
            if maxv == minv:
                return 100.0
            pct = round(((v - minv) / (maxv - minv)) * 100.0, 1)
            return max(0.0, min(100.0, pct))

        pct_a = {}
        pct_b = {}
        for r in reports:
            v = float(r.ticketed_speed - r.posted_speed)
            minv, maxv = a_minmax.get((r.state, r.road_key), (None, None))
            pct_a[r.id] = _pct(minv, maxv, v)

            minv, maxv = b_minmax.get((r.state, r.posted_speed), (None, None))
            pct_b[r.id] = _pct(minv, maxv, v)

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

        # Comments: show newest first per report (limited for feed performance)
        comments_by_report: Dict[int, List[Comment]] = {rid: [] for rid in report_ids}
        if report_ids:
            rows = (
                Comment.query.options(joinedload(Comment.user))
                .filter(Comment.report_id.in_(report_ids))
                .order_by(Comment.created_at.asc())
                .all()
            )
            for c in rows:
                comments_by_report.setdefault(c.report_id, []).append(c)

        like_form = LikeForm()
        comment_form = CommentForm()

        return render_template(
            "home_feed.html",
            reports=reports,
            pagination=pagination,
            hide_anon=hide_anon,
            pct_a=pct_a,
            pct_b=pct_b,
            like_counts=like_counts,
            user_liked=user_liked,
            comments_by_report=comments_by_report,
            like_form=like_form,
            comment_form=comment_form,
        )


    def _safe_next_url() -> str:
        """Return a safe relative next url from form data."""
        nxt = (request.form.get("next") or "").strip()
        if nxt.startswith("/") and not nxt.startswith("//"):
            return nxt
        return url_for("home")


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
        c = Comment(report_id=report.id, user_id=current_user.id, body=body)
        db.session.add(c)
        db.session.commit()

        return redirect(_safe_next_url())


    @app.get("/submit")
    def submit():
        form = SpeedReportForm()
        ticket_count = SpeedReport.query.count()

        hide_anon_requested = (request.args.get("hide_anon") == "1")

        if hide_anon_requested and not current_user.is_authenticated:
            flash("You must be logged in to hide anonymous posts.", "info")

        hide_anon = bool(current_user.is_authenticated and hide_anon_requested)

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
        )


    @app.post("/submit")
    def submit_ticket():
        form = SpeedReportForm()

        if not form.validate_on_submit():
            return render_template(
                "mvp_home.html",
                form=form,
                ticket_count=SpeedReport.query.count(),
                most_strict=strictness_rows(limit=5, exclude_anonymous=bool(current_user.is_authenticated and request.args.get("hide_anon") == "1"))["most_strict"],
                least_strict=strictness_rows(limit=5, exclude_anonymous=bool(current_user.is_authenticated and request.args.get("hide_anon") == "1"))["least_strict"],
                hide_anon=bool(current_user.is_authenticated and request.args.get("hide_anon") == "1"),
                state_options=STATE_OPTIONS,
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

        db.session.add(report)
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
        strictness = strictness_rows(limit=20, exclude_anonymous=hide_anon, state_filter=(request.args.get('state') or '').strip().upper() or None)
        most_strict = strictness["most_strict"]
        least_strict = strictness["least_strict"]

        return render_template(
            "strictness.html",
            most_strict=most_strict,
            least_strict=least_strict,
            hide_anon=hide_anon,
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

if __name__ == "__main__":
    app.run(debug=True)
