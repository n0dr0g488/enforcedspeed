# EnforcedSpeed CHANGELOG

> Single consolidated changelog. Newest entries at the top.
> Older split changelogs (v171-430) live in `changelogs/`.

---

## v694 — 2026-04-25

**Pre-launch hardening: rate-limited remaining auth endpoints + cleanup**
- Added rate limits to previously-unprotected auth endpoints:
  - `/api/auth/forgot-password`: 3/hour, 1/minute (prevents email enumeration + SendGrid quota abuse)
  - `/api/auth/reset-password`: 20/hour, 5/minute (prevents reset token brute-forcing)
  - `/api/auth/refresh`: 60/hour, 10/minute (prevents token refresh spam from compromised clients)
- Removed unused `SpeedReportForm` import from `app.py`.
- Documented all required environment variables in `render.yaml` (Maps keys, Gemini key, SendGrid, R2, public URLs, admin emails). Values still set in Render dashboard — `render.yaml` now serves as a discoverability index for what env vars are required, useful for disaster recovery and onboarding.

## v693 — 2026-04-25

**HOTFIX: Rate limiter was killing the production server**
- Render's health check (`/healthz`) hits every minute from a fixed Render IP. The global rate limit default of "50 per minute" was tripping against that IP, causing health checks to return HTTP 429, which Render interpreted as instance failure → repeated restarts every hour for 12+ hours.
- **Fix:** Both `/healthz` and `/health` are now exempted from the rate limiter via `limiter.exempt()` (called conditionally so it's safe even if Flask-Limiter init fails).
- **Also:** Bumped global default from "500/hour, 50/min" to "1000/hour, 120/min". The tight per-endpoint limits on auth/write endpoints (login, register, submit, etc.) are still in place — the global default only catches truly abusive traffic patterns.
- **Lesson:** When introducing a global rate limit, always exempt health check endpoints first. Cloud providers hammer them on a fixed cadence and they look like one client to a per-IP limiter.

## v692 — 2026-04-18

**Snap-to-road feature fully removed (was already legacy)**
- The "Refine pin" UI on the home feed remains — users can still drag-edit their ticket pin location after submission.
- BUT the snap-to-nearest-road behavior is gone. The user's pin stays exactly where they drop it.
- Removed `snap_to_nearest_road()` function from `app.py` (was the last ~30 lines of Roads API code).
- `/api/update_ticket_pin` now saves coordinates verbatim — `location_source` is always `'user_pin'`.
- `/api/tickets` simplified: no longer marks anything as `'user_pin_snapped'` (raw_lat/raw_lng accepted for backward compat with old clients but treated identically to lat/lng).
- Updated `home_feed.html` modal text: removed "We'll snap it to the nearest road line." → now says "Drag the pin to the exact spot you'd like to mark."
- Updated success message: "Saved. Pin snapped to nearest road." → "Pin saved."
- **Google Roads API can now be DISABLED entirely** in Google Cloud Console — no code calls it anymore.
- Legacy `'user_pin_snapped'` values in existing DB rows are still matched by feed/map filter queries to preserve backward compat with historical data.

## v691 — 2026-04-18

**Dead code cleanup (Roads API + dead helpers)**
- Removed 350+ lines of dead Roads-API-related code:
  - `nearest_roads_multi()` — batch Roads API helper, never called by live clients
  - `_confirm_location_route_class()` — highway-biased snapping, ~120 lines
  - `_confirm_location_prefer_highway()` — back-compat wrapper, never called
  - `_score_highway_candidate()`, `_score_route_class_candidate()`, `_route_kind()` — scoring helpers
  - `_offset_latlng_m()`, `_HIGHWAY_PENALTY_WORDS`, `_snap_and_get_road()` — supporting code
  - `/api/confirm_location` endpoint — no live client calls (mobile defined `confirmLocation()` but never invoked it; only legacy `mvp_home.html` template referenced it, and that template isn't rendered)
- Removed `templates/mvp_home.html` — legacy template not rendered anywhere
- **Retained** the small (~30 line) `snap_to_nearest_road()` function — still used by live `/api/update_ticket_pin` endpoint when users drag to refine ticket pin location on the website's home feed
- Roads API is still required (only for pin refinement); set Google Cloud quotas accordingly

## v690 — 2026-04-18

**Pre-launch hardening: rate limits + account deletion + render config**
- Added Flask-Limiter (3.5+) with Redis backend (`REDIS_URL` auto-injected by Render)
- Per-IP rate limits on critical endpoints:
  - `/login`: 20/hour, 5/minute
  - `/register`: 10/hour, 3/minute
  - `/api/auth/login`: 20/hour, 5/minute
  - `/api/auth/register`: 10/hour, 3/minute
  - `/submit`: 30/hour, 10/minute
  - `/api/tickets`: 30/hour, 10/minute
  - `/api/auth/delete-account`: 5/hour
  - Global default: 500/hour, 50/minute (DDoS protection)
- New endpoint: `POST /api/auth/delete-account` — Apple/Google App Store compliance
  - Body: `{"confirm": "DELETE"}` required
  - Hard-deletes: user's tickets + their dependent likes/comments, comments by user, likes by user, follows, device tokens, profile photo (R2), car photos (R2), user row
  - Revokes the current JWT
  - Rate limited to 5/hour
- Updated `render.yaml`:
  - `plan: free` → `plan: starter`
  - Gunicorn: `--workers 2 --timeout 60 --access-logfile -` (was no workers flag)
  - Documented `REDIS_URL` auto-injection for Flask-Limiter

## v689 — 2026-04-18

**Policy updates: AI verification, mobile app coverage, assumption of risk**
- `templates/terms.html` — full rewrite of TOS:
  - Platform definition now covers website + iOS app + Android app
  - "Automated Processing" definition expanded to AI-based image analysis (was OCR)
  - New definition: "Enforced Speed" — explicitly NOT a recommended driving speed
  - Section 3A: AI verification flow (Google Gemini) with broad IP-extraction rights
  - **NEW Section 3B: ASSUMPTION OF RISK** — six all-caps clauses waiving liability for any driving decisions influenced by Platform Data; covers tickets, fines, accidents, property damage, etc.
  - **NEW Section 4A: PROFILE INFORMATION AND PUBLIC VISIBILITY** — covers profile data, public visibility consent, photo upload warranties, comment moderation rights, no expectation of privacy in public content
  - Section 5 (License Grant) broadened: AI extraction, derived insights, statistical aggregations, public display rights for username/profile photo/comments/tickets, content moderation rights
  - Section 9 (Disclaimers): added explicit warranty disclaimer for Enforced Speed metrics
  - Section 10 (Limitation of Liability): added explicit no-liability clause for traffic citations, accidents, injuries, property damage from driving decisions
  - Section 11 (Indemnification): user indemnifies for "driving decisions or behavior (whether or not influenced by Platform Data)"
- `templates/privacy.html` — updates:
  - Platform now covers mobile app
  - Section 3.1 expanded to cover profile info, vehicle details, comments, location data
  - Section 5 (Retention) clarifies what's retained vs not
  - Section 6 changed "Conduct Automated Processing" → "Conduct AI-based verification"
  - Section 7 added Google Cloud Vision (image safety screening)
  - Section 8 expanded: explicitly Google Gemini 2.5 Flash, AI prompted only for speed extraction (not PII)
- `templates/cookies.html` — updates:
  - Platform now covers mobile app
  - New paragraph: mobile app uses AsyncStorage and SecureStore (functionally equivalent to cookies)
- All three policies now dated April 18, 2026

---

For changes prior to v689, see split files in `changelogs/`.
