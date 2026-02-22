# forms.py
import re
import datetime
from flask_wtf import FlaskForm
from flask_wtf.file import FileAllowed, FileField
from wtforms import StringField, IntegerField, TextAreaField, SubmitField, PasswordField, HiddenField, SelectField, BooleanField
from wtforms.validators import DataRequired, Length, NumberRange, Optional, ValidationError, Email, EqualTo, Regexp

US_STATE_CODES = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA","KS","KY","LA",
    "ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK",
    "OR","PA","RI","SC","SD","TN","TX","UT","VT","VA","WA","WV","WI","WY","DC"
}
US_STATE_NAMES = {
    "alabama","alaska","arizona","arkansas","california","colorado","connecticut","delaware",
    "florida","georgia","hawaii","idaho","illinois","indiana","iowa","kansas","kentucky","louisiana",
    "maine","maryland","massachusetts","michigan","minnesota","mississippi","missouri","montana",
    "nebraska","nevada","new hampshire","new jersey","new mexico","new york","north carolina",
    "north dakota","ohio","oklahoma","oregon","pennsylvania","rhode island","south carolina",
    "south dakota","tennessee","texas","utah","vermont","virginia","washington","west virginia",
    "wisconsin","wyoming","district of columbia"
}


def normalize_state_input(value: str) -> str:
    """
    Accept:
      - 'VA'
      - 'Virginia'
      - 'VA - Virginia'
    Reject partials like 'virg'.
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


class SpeedReportForm(FlaskForm):
    state = StringField(
        "State",
        validators=[
            DataRequired(message="Please enter a state (e.g., VA or Virginia)."),
            Length(max=100, message="Keep it under 100 characters."),
        ],
        render_kw={"placeholder": "e.g., VA or Virginia"},
    )

    road_name = StringField(
        "Road / Location",
        validators=[
            DataRequired(message="Please enter the road/location."),
            Length(max=200, message="Keep it under 200 characters."),
        ],
        render_kw={"placeholder": "e.g., I-95, US-50, VA-288, Main St"},
    )

    location_hint = StringField(
        "Nearest exit / city (optional)",
        validators=[Optional(), Length(max=200, message="Keep it under 200 characters.")],
        render_kw={"placeholder": "Ex: Exit 118 / near Richmond / US-64 interchange"},
    )

    # Map refine (optional). These are filled by client-side JS.
    raw_lat = HiddenField()
    raw_lng = HiddenField()
    lat = HiddenField()
    lng = HiddenField()
    google_place_id = HiddenField()

    posted_speed = IntegerField(
        "Posted speed limit",
        validators=[
            DataRequired(message="Please enter the posted speed."),
            NumberRange(min=0, max=120, message="Enter a reasonable speed (0–120)."),
        ],
        render_kw={"placeholder": "e.g., 65"},
    )

    ticketed_speed = IntegerField(
        "Ticketed speed",
        validators=[
            DataRequired(message="Please enter the ticketed speed."),
            NumberRange(min=0, max=150, message="Enter a reasonable speed (0–150)."),
        ],
        render_kw={"placeholder": "e.g., 79"},
    )

    photo = FileField(
        "Photo (optional)",
        validators=[
            Optional(),
            # Keep validation aligned with the backend normalizer (which re-encodes to JPEG).
            # Many phones produce HEIC/HEIF; we support phone images + PDFs by normalizing to JPEG server-side.
            FileAllowed(["jpg", "jpeg", "png", "webp", "heic", "heif", "pdf"], "Upload a JPG/JPEG/PNG/WebP/HEIC/HEIF image or a PDF."),
        ],
    )

    notes = TextAreaField(
        "Notes (optional)",
        validators=[Optional(), Length(max=2000, message="Keep notes under 2000 characters.")],
        render_kw={"placeholder": "Optional context (weather, time, enforcement details, etc.)"},
    )

    submit = SubmitField("Submit Ticket")

    def validate_state(self, field):
        norm = normalize_state_input(field.data)
        if norm in US_STATE_CODES:
            return
        if norm in US_STATE_NAMES:
            return
        raise ValidationError("Pick a real U.S. state (e.g., 'VA' or 'Virginia'). Don’t leave it partial like 'virg'.")

    def validate_ticketed_speed(self, field):
        """EnforcedSpeed only accepts tickets where ticketed speed is above posted speed."""
        if self.posted_speed.data is None or field.data is None:
            return
        if field.data <= self.posted_speed.data:
            raise ValidationError("Ticketed speed must be higher than the posted speed limit.")

class RegisterForm(FlaskForm):
    email = StringField("Email", validators=[DataRequired(), Email(), Length(max=255)])

    username = StringField(
        "Desired username",
        validators=[
            DataRequired(),
            # Keep usernames short and consistent across the site.
            Length(min=3, max=12, message="Username must be 3–12 characters."),
            Regexp(r"^[A-Za-z0-9_]+$", message="Username can only use letters, numbers, and underscores."),
        ],
        # No placeholder example (per UX standardization).
        render_kw={"maxlength": "12"},
    )

    # Birthday (FB-style dropdowns)
    birth_month = SelectField("Birthday", choices=[], validators=[DataRequired(message="Please select your birth month.")])
    birth_day = SelectField("", choices=[], validators=[DataRequired(message="Please select your birth day.")])
    birth_year = SelectField("", choices=[], validators=[DataRequired(message="Please select your birth year.")])

    password = PasswordField("New password", validators=[DataRequired(), Length(min=8, max=128)])
    confirm_password = PasswordField(
        "Confirm Password",
        validators=[DataRequired(), EqualTo("password", message="Passwords must match.")],
    )
    submit = SubmitField("Create Account")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Choices + defaults (today's date prefilled, like FB)
        today = datetime.date.today()
        month_choices = [(str(i), datetime.date(2000, i, 1).strftime("%b")) for i in range(1, 13)]
        self.birth_month.choices = month_choices

        self.birth_day.choices = [(str(i), str(i)) for i in range(1, 32)]

        # 120-year range, newest first
        this_year = today.year
        self.birth_year.choices = [(str(y), str(y)) for y in range(this_year, this_year - 121, -1)]

        # Prefill today's date so users understand the layout
        if not self.birth_month.data:
            self.birth_month.data = str(today.month)
        if not self.birth_day.data:
            self.birth_day.data = str(today.day)
        if not self.birth_year.data:
            self.birth_year.data = str(today.year)

    def validate_birth_year(self, field):
        # Validate age >= 18
        try:
            y = int(self.birth_year.data)
            m = int(self.birth_month.data)
            d = int(self.birth_day.data)
            dob = datetime.date(y, m, d)
        except Exception:
            raise ValidationError("Please select a valid birth date.")

        today = datetime.date.today()
        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        if age < 18:
            raise ValidationError("You must be at least 18 years old to create an account.")


class LoginForm(FlaskForm):
    email = StringField("Username or Email", validators=[DataRequired(), Length(max=255)])
    password = PasswordField("New password", validators=[DataRequired()])
    submit = SubmitField("Log In")


class ProfileCarForm(FlaskForm):
    """Optional vehicle info shown on the profile page (v423)."""

    car_make = StringField(
        "Car brand (optional)",
        validators=[Optional(), Length(max=50, message="Keep it under 50 characters.")],
        render_kw={"placeholder": "e.g., Toyota"},
    )

    car_model = StringField(
        "Car model (optional)",
        validators=[Optional(), Length(max=60, message="Keep it under 60 characters.")],
        render_kw={"placeholder": "e.g., Tacoma"},
    )

    car_year = IntegerField(
        "Year (optional)",
        validators=[Optional(), NumberRange(min=1900, max=2035, message="Enter a valid year.")],
        render_kw={"placeholder": "e.g., 2019"},
    )

    submit = SubmitField("Save")


class ChangePasswordForm(FlaskForm):
    current_password = PasswordField("Current Password", validators=[DataRequired()])
    new_password = PasswordField("New Password", validators=[DataRequired(), Length(min=8, max=128)])
    confirm_new_password = PasswordField("Confirm New Password", validators=[DataRequired(), EqualTo('new_password', message='Passwords must match.')])
    submit = SubmitField("Change Password")


class ForgotPasswordForm(FlaskForm):
    identifier = StringField("Username or Email", validators=[DataRequired(), Length(max=255)])
    submit = SubmitField("Send Reset Link")


class ResetPasswordForm(FlaskForm):
    new_password = PasswordField("New Password", validators=[DataRequired(), Length(min=8, max=128)])
    confirm_new_password = PasswordField("Confirm New Password", validators=[DataRequired(), EqualTo('new_password', message='Passwords must match.')])
    submit = SubmitField("Set New Password")


class LikeForm(FlaskForm):
    """CSRF-protected empty form for like toggles."""
    submit = SubmitField("Like")


class CommentForm(FlaskForm):
    """Members-only comments on a SpeedReport (feed post)."""
    parent_id = HiddenField()
    body = TextAreaField(
        "Comment",
        validators=[DataRequired(message="Comment can’t be empty."), Length(max=280, message="Keep comments under 280 characters.")],
        render_kw={"maxlength": 280, "rows": 2, "placeholder": "Write a comment…"},
    )
    submit = SubmitField("Post")


class DeleteCommentForm(FlaskForm):
    """CSRF-protected empty form for deleting a comment."""
    submit = SubmitField("Delete")


class FollowCountyForm(FlaskForm):
    """CSRF-protected form for follow/unfollow county toggles (map pill)."""
    county_geoid = HiddenField(validators=[DataRequired()])
    submit = SubmitField("Toggle")


# County-first single-page submit (v229+)
class SubmitTicketForm(FlaskForm):
    state = HiddenField("State", validators=[DataRequired()])  # STUSPS

    county_query = StringField("County", validators=[DataRequired()], render_kw={"autocomplete": "off"})
    county_geoid = HiddenField("County GEOID", validators=[DataRequired()])

    road_name = StringField("Road", validators=[DataRequired()])

    posted_speed = IntegerField("Posted Speed", validators=[DataRequired(), NumberRange(min=1, max=200)])
    ticketed_speed = IntegerField("Ticketed Speed", validators=[DataRequired(), NumberRange(min=1, max=200)])

    # Optional pin (validated server-side to be within the selected county polygon)
    # NOTE: Template + submit handler use lat/lng; keep latitude/longitude for backward compatibility.
    lat = HiddenField("Latitude")
    lng = HiddenField("Longitude")
    latitude = HiddenField("Latitude (legacy)")
    longitude = HiddenField("Longitude (legacy)")

    caption = TextAreaField("Caption", validators=[Length(max=500)])

    photo = FileField("Photo")

    submit = SubmitField("Submit Ticket")


class ProfileDefaultFiltersForm(FlaskForm):
    """Default Home/Statistics filters stored on the user profile (v424)."""

    form_name = HiddenField(default="defaults")

    default_state = SelectField("Default state", choices=[], validators=[Optional()])
    default_county_label = StringField(
        "Default county (optional)",
        validators=[Optional(), Length(max=80)],
        render_kw={"placeholder": "Start typing (requires a state)"},
    )
    default_county_geoid = HiddenField()

    default_date = SelectField("Default date", choices=[], validators=[Optional()])
    default_verify = SelectField("Default photo", choices=[], validators=[Optional()])
    default_pin = BooleanField("Map pin only", default=False)

    default_speed_limit = SelectField("Posted speed limit", choices=[], validators=[Optional()])
    default_overage = SelectField("Overage", choices=[], validators=[Optional()])
    default_sort = SelectField("Sort by", choices=[], validators=[Optional()])

    save_defaults = SubmitField("Save defaults")
    clear_defaults = SubmitField("Clear defaults")

    def set_choices(self):
        # State choices: All + all US state codes
        st = sorted(list(US_STATE_CODES))
        self.default_state.choices = [("", "All states")] + [(c, c) for c in st]

        self.default_date.choices = [
            ("any", "Any"),
            ("7", "Last 7 days"),
            ("30", "Last 30 days"),
            ("90", "Last 90 days"),
            ("365", "Last year"),
        ]

        self.default_verify.choices = [
            ("any", "Any"),
            ("verified", "Auto-Extracted"),
            ("not_verified", "Not auto-extracted"),
        ]

        self.default_speed_limit.choices = [
            ("any", "Any"),
            ("lte35", "≤ 35 mph"),
            ("40-55", "40–55 mph"),
            ("gte60", "≥ 60 mph"),
        ]

        self.default_overage.choices = [
            ("all", "Any"),
            ("1-10", "1–10 mph over"),
            ("11-20", "11–20 mph over"),
            ("21+", "≥ 21 mph over"),
        ]

        self.default_sort.choices = [
            ("new", "Newest"),
            ("old", "Oldest"),
            ("most_over", "Highest overage"),
            ("least_over", "Lowest overage"),
        ]
