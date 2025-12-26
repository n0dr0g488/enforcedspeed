# models.py
import re
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


def normalize_whitespace(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_text_basic(value: str) -> str:
    """
    Lowercase, strip punctuation, collapse whitespace.
    Useful as a fallback.
    """
    if not value:
        return ""
    v = value.strip().lower()
    v = re.sub(r"[^a-z0-9\s]", " ", v)
    v = normalize_whitespace(v)
    return v


def normalize_road(value: str) -> str:
    """
    STRICT road normalization (no guessing):
    - If user indicates Interstate (I / I- / interstate): -> i<number> (+ optional direction)
    - If user indicates US route (US / U.S. / united states route): -> us<number> (+ optional direction)
    - If user indicates State Route (SR / state route / route / hwy / highway): -> route<number> (+ optional direction)
    - If user types JUST a number like "95": -> route95 (generic), NOT i95

    Direction handling:
      - "NB/N" -> nb, "SB/S" -> sb, "EB/E" -> eb, "WB/W" -> wb
      - If present, appended as "-nb", etc.
    """
    if not value:
        return ""

    raw = value.strip().lower()

    # Normalize punctuation to spaces early, keep letters/numbers
    cleaned = re.sub(r"[^a-z0-9\s]", " ", raw)
    cleaned = normalize_whitespace(cleaned)

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

    # Helper to build key
    def build(prefix: str, num: str) -> str:
        key = f"{prefix}{int(num)}"
        if direction:
            key += f"-{direction}"
        return key

    # Interstate patterns: i 95, i95, interstate 95
    if re.search(r"\binterstate\b", cleaned) or re.search(r"\bi\s*\d+\b", cleaned):
        m = re.search(r"(?:\binterstate\b|\bi\b)\s*(\d+)\b", cleaned)
        if m:
            return build("i", m.group(1))

    # US route patterns: us 95, u s 95, u.s. 95, united states route 95
    if re.search(r"\bunited states route\b", cleaned) or re.search(r"\bus\s*\d+\b", cleaned) or re.search(r"\bu\s*s\s*\d+\b", cleaned):
        m = re.search(r"(?:\bunited states route\b|\bus\b|\bu\s*s\b)\s*(\d+)\b", cleaned)
        if m:
            return build("us", m.group(1))

    # State route / generic route patterns: sr 95, state route 95, route 95, hwy 95, highway 95
    if re.search(r"\b(state route|sr|route|hwy|highway)\b", cleaned):
        m = re.search(r"(?:\bstate route\b|\bsr\b|\broute\b|\bhwy\b|\bhighway\b)\s*(\d+)\b", cleaned)
        if m:
            return build("route", m.group(1))

    # If the input is ONLY a number (or basically just a number), DO NOT GUESS.
    # Treat it as generic route<number>
    m_only = re.fullmatch(r"\s*(\d+)\s*", cleaned)
    if m_only:
        return build("route", m_only.group(1))

    # Otherwise fallback: basic normalization of the whole phrase
    # (this still groups identical-ish free text like "i 70 exit 10" vs "i-70 exit 10"
    # once punctuation is removed)
    return normalize_text_basic(value)


class SpeedReport(db.Model):
    __tablename__ = "speed_reports"

    id = db.Column(db.Integer, primary_key=True)

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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.road_key = normalize_road(self.road_name)

    def refresh_road_key(self) -> None:
        self.road_key = normalize_road(self.road_name)

    def __repr__(self) -> str:
        return (
            f"<SpeedReport id={self.id} state={self.state!r} road_name={self.road_name!r} "
            f"road_key={self.road_key!r} posted={self.posted_speed} ticketed={self.ticketed_speed}>"
        )
