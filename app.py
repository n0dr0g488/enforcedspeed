# app.py
from __future__ import annotations

import os
import re
from datetime import datetime, timedelta
from statistics import median
from typing import Dict, List

from flask import Flask, render_template, redirect, url_for, jsonify, request, flash
from flask_login import LoginManager, login_user, logout_user, current_user, login_required
from sqlalchemy import text, func
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

from config import Config
from forms import SpeedReportForm, RegisterForm, LoginForm, ChangePasswordForm, ForgotPasswordForm, ResetPasswordForm
from models import db, SpeedReport, User, normalize_road, state_code_from_value


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

    def most_enforced_rows(since: datetime | None) -> List[Dict]:
        q = (
            db.session.query(
                SpeedReport.state.label("state"),
                SpeedReport.road_key.label("road_key"),
                func.count(SpeedReport.id).label("tickets"),
                func.avg(SpeedReport.ticketed_speed - SpeedReport.posted_speed).label("avg_overage"),
            )
            .group_by(SpeedReport.state, SpeedReport.road_key)
        )

        if since is not None:
            q = q.filter(SpeedReport.created_at >= since)

        q = q.order_by(func.avg(SpeedReport.ticketed_speed - SpeedReport.posted_speed).desc()).limit(5)

        rows = q.all()
        return [
            {
                "state": normalize_state_group(r.state),
                "road": r.road_key,
                "avg_overage": round(float(r.avg_overage), 1) if r.avg_overage is not None else 0.0,
                "tickets": int(r.tickets),
            }
            for r in rows
        ]

    @app.get("/")
    def home():
        form = SpeedReportForm()
        ticket_count = SpeedReport.query.count()

        most_enforced_all = most_enforced_rows(since=None)
        most_enforced_24h = most_enforced_rows(since=datetime.utcnow() - timedelta(hours=24))

        return render_template(
            "mvp_home.html",
            form=form,
            ticket_count=ticket_count,
            most_enforced_all=most_enforced_all,
            most_enforced_24h=most_enforced_24h,
            state_options=STATE_OPTIONS,
        )

    @app.post("/")
    def submit_ticket():
        form = SpeedReportForm()

        if not form.validate_on_submit():
            return render_template(
                "mvp_home.html",
                form=form,
                ticket_count=SpeedReport.query.count(),
                most_enforced_all=most_enforced_rows(since=None),
                most_enforced_24h=most_enforced_rows(since=datetime.utcnow() - timedelta(hours=24)),
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

    def compute_distribution_stats(speeds: List[int], user_speed: int) -> Dict:
        speeds_sorted = sorted(speeds)
        n = len(speeds_sorted)
        if n == 0:
            return {"n": 0, "enforced_speed": None, "percentile": None}

        le_count = sum(1 for s in speeds_sorted if s <= user_speed)
        percentile = round((le_count / n) * 100, 1)

        enforced = round(median(speeds_sorted), 1)
        return {"n": n, "enforced_speed": enforced, "percentile": percentile}

        med = round(median(vals), 2)
        if user_value is None:
            pct = None
        else:
            le_count = sum(1 for v in vals if v <= user_value)
            pct = round((le_count / n) * 100, 1)

        return {"n": n, "median": med, "percentile": pct}

    @app.get("/result/<int:report_id>")
    def result(report_id: int):
        report = SpeedReport.query.get_or_404(report_id)
        state_group = normalize_state_group(report.state)

        rows = (
            db.session.query(
                SpeedReport.state,
                SpeedReport.road_key,
                SpeedReport.posted_speed,
                SpeedReport.ticketed_speed,
            )
            .filter(SpeedReport.posted_speed == report.posted_speed)
            .all()
        )

        speeds_state_posted: List[int] = []
        speeds_state_road_posted: List[int] = []

        for st, road_key, posted, ticketed in rows:
            if normalize_state_group(st) != state_group:
                continue

            speeds_state_posted.append(ticketed)

            if road_key == report.road_key:
                speeds_state_road_posted.append(ticketed)

        stats_a = compute_distribution_stats(speeds_state_road_posted, report.ticketed_speed)
        stats_b = compute_distribution_stats(speeds_state_posted, report.ticketed_speed)

        return render_template(
            "thank_you.html",
            report=report,
            stats_a=stats_a,
            stats_b=stats_b,
            state_group=state_group,
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

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
