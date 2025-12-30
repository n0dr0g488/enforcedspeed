# forms.py
import re
from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, TextAreaField, SubmitField, PasswordField
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
class RegisterForm(FlaskForm):
    email = StringField("Email", validators=[DataRequired(), Email(), Length(max=255)])
    username = StringField(
        "Username",
        validators=[
            DataRequired(),
            Length(min=3, max=20),
            Regexp(r"^[a-z0-9_]+$", message="Username can only use a-z, 0-9, and underscores."),
        ],
        render_kw={"placeholder": "e.g., fastticketguy"},
    )
    password = PasswordField("Password", validators=[DataRequired(), Length(min=8, max=128)])
    confirm_password = PasswordField(
        "Confirm Password",
        validators=[DataRequired(), EqualTo("password", message="Passwords must match.")],
    )
    submit = SubmitField("Create Account")


class LoginForm(FlaskForm):
    email = StringField("Email", validators=[DataRequired(), Email(), Length(max=255)])
    password = PasswordField("Password", validators=[DataRequired()])
    submit = SubmitField("Log In")
