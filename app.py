# app.py
from __future__ import annotations

import re
from statistics import mean, median
from typing import Dict, List, Tuple

from flask import Flask, render_template, redirect, url_for, jsonify, request
from sqlalchemy import text

from config import Config
from forms import SpeedReportForm
from models import db, SpeedReport, normalize_road


def normalize_state_group(value: str) -> str:
    """
    Grouping key for state while keeping your current UI behavior.
    Accepts:
      - 'VA'
      - 'Virginia'
      - 'VA - Virginia'
    Returns:
      - if starts with 2 letters -> 'VA'
      - else normalized name -> 'virginia'
    """
    if not value:
        return ""

    v = value.strip()
    m = re.match(r"^\s*([A-Za-z]{2})\b", v)
    if m:
        return m.group(1).upper()

    name = v.lower().strip()
    name = re.sub(r"[^a-z\s]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def base_road_key(road_key: str) -> str:
    """
    Collapses directional suffixes so:
      i95-nb -> i95
      us50-eb -> us50
      route95-sb -> route95
    Leaves unknown keys as-is.
    """
    if not road_key:
        return ""
    return re.sub(r"-(nb|sb|eb|wb)$", "", road_key)


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)

    db.init_app(app)

    @app.before_request
    def ensure_tables_exist():
        try:
            db.session.execute(text("SELECT 1"))
            db.create_all()
        except Exception:
            db.session.rollback()

    @app.get("/")
    def home():
        form = SpeedReportForm()
        return render_template("mvp_home.html", form=form)

    @app.post("/")
    def submit_ticket():
        form = SpeedReportForm()
        if not form.validate_on_submit():
            return render_template("mvp_home.html", form=form), 400

        report = SpeedReport(
            state=form.state.data,
            road_name=form.road_name.data,
            posted_speed=form.posted_speed.data,
            ticketed_speed=form.ticketed_speed.data,
            notes=form.notes.data,
        )
        db.session.add(report)
        db.session.commit()

        return redirect(url_for("result", report_id=report.id))

    def compute_distribution_stats(speeds: List[int], user_speed: int) -> Dict:
        n = len(speeds)
        speeds_sorted = sorted(speeds)

        le_count = sum(1 for s in speeds_sorted if s <= user_speed)
        percentile = round((le_count / n) * 100, 1) if n else None

        return {
            "n": n,
            "min": speeds_sorted[0],
            "max": speeds_sorted[-1],
            "mean": round(mean(speeds_sorted), 1),
            "median": round(median(speeds_sorted), 1),
            "user_speed": user_speed,
            "percentile": percentile,
            "enforced_speed": round(median(speeds_sorted), 1),
            "speeds_sorted": speeds_sorted,
        }

    @app.get("/result/<int:report_id>")
    def result(report_id: int):
        report = SpeedReport.query.get_or_404(report_id)
        state_group = normalize_state_group(report.state)

        # Fetch all rows once for the relevant posted speed, then split in Python (MVP-simple)
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
    def api_road_preview():
        """
        Returns:
          - predicted_road_key for current input
          - base_key (directionless)
          - suggestions of existing buckets for the selected state that share the same base
        """
        state_raw = request.args.get("state", "", type=str)
        road_raw = request.args.get("road", "", type=str)

        state_group = normalize_state_group(state_raw)
        predicted_key = normalize_road(road_raw) if road_raw.strip() else ""
        base_key = base_road_key(predicted_key) if predicted_key else ""

        # If no state or no road, return minimal
        if not state_group or not predicted_key:
            return jsonify(
                {
                    "state_group": state_group,
                    "predicted_road_key": predicted_key,
                    "base_key": base_key,
                    "suggestions": [],
                }
            )

        # Pull roads for that state (MVP approach: scan all, filter in Python)
        rows = db.session.query(SpeedReport.state, SpeedReport.road_key, SpeedReport.road_name).all()

        # Count by road_key within state, and keep a "best label" (most recent road_name seen)
        counts: Dict[str, int] = {}
        label: Dict[str, str] = {}

        for st, road_key, road_name in rows:
            if normalize_state_group(st) != state_group:
                continue
            counts[road_key] = counts.get(road_key, 0) + 1
            # Use the latest seen as a simple label; good enough for MVP
            label[road_key] = road_name

        # Build suggestions: same base route as predicted
        suggestions: List[Dict] = []
        for rk, ct in counts.items():
            if base_road_key(rk) != base_key:
                continue
            suggestions.append(
                {
                    "road_key": rk,
                    "label": label.get(rk, rk),
                    "count": ct,
                }
            )

        # Sort by count desc
        suggestions.sort(key=lambda x: x["count"], reverse=True)

        # Limit and ensure predicted key appears (even if count is 0/new)
        suggestions = suggestions[:8]
        if predicted_key and all(s["road_key"] != predicted_key for s in suggestions):
            suggestions.insert(
                0,
                {
                    "road_key": predicted_key,
                    "label": road_raw.strip() or predicted_key,
                    "count": counts.get(predicted_key, 0),
                },
            )
            suggestions = suggestions[:8]

        return jsonify(
            {
                "state_group": state_group,
                "predicted_road_key": predicted_key,
                "base_key": base_key,
                "suggestions": suggestions,
            }
        )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
