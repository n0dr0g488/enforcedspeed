#!/usr/bin/env bash
set -euo pipefail

# EnforcedSpeed — Import US Counties (cb_2024_us_county_500k) into Render Postgres/PostGIS
#
# Requires:
#   - Docker running
#
# Usage:
#   ./scripts/import_counties_render.sh ./cb_2024_us_county_500k.zip "postgresql://..."
#
COUNTIES_ZIP="${1:-}"
DB_URL="${2:-${DATABASE_URL:-}}"

if [[ -z "${COUNTIES_ZIP}" ]]; then
  echo "Missing arg1: path to cb_2024_us_county_500k.zip" >&2
  exit 1
fi
if [[ ! -f "${COUNTIES_ZIP}" ]]; then
  echo "Zip not found: ${COUNTIES_ZIP}" >&2
  exit 1
fi
if [[ -z "${DB_URL}" ]]; then
  echo "Missing DB url (arg2 or DATABASE_URL env)" >&2
  exit 1
fi

# Ensure sslmode=require
if [[ "${DB_URL}" != *"sslmode="* ]]; then
  if [[ "${DB_URL}" == *"?"* ]]; then
    DB_URL="${DB_URL}&sslmode=require"
  else
    DB_URL="${DB_URL}?sslmode=require"
  fi
fi

# Parse the URI into libpq keywords for ogr2ogr
read -r PG_CONN HOST PORT DB USER PASS < <(
python - <<'PY'
import sys
from urllib.parse import urlparse
u = urlparse(sys.argv[1])
user = u.username or ''
passwd = u.password or ''
db = (u.path or '').lstrip('/')
host = u.hostname or ''
port = str(u.port or 5432)
print(f"host={host} port={port} dbname={db} user={user} password={passwd} sslmode=require")
print(host, port, db, user, passwd)
PY
"${DB_URL}")

TMP_DIR="$(mktemp -d -t es_counties_XXXXXX)"
trap 'rm -rf "${TMP_DIR}"' EXIT

unzip -q "${COUNTIES_ZIP}" -d "${TMP_DIR}"
SHP="$(find "${TMP_DIR}" -type f -name "*.shp" | head -n 1)"
if [[ -z "${SHP}" ]]; then
  echo "No .shp found in zip" >&2
  exit 1
fi

REL="${SHP#${TMP_DIR}/}"

echo "[1/3] Importing shapefile -> counties_stage"
docker run --rm -v "${TMP_DIR}:/data" osgeo/gdal:alpine-small \
  ogr2ogr -f PostgreSQL "PG:${PG_CONN}" "/data/${REL}" \
  -nln counties_stage -overwrite -nlt MULTIPOLYGON -t_srs EPSG:4326 -lco GEOMETRY_NAME=geom

echo "[2/3] Upserting into counties (canonical)"
MERGE_SQL=$(cat <<'SQL'
BEGIN;
INSERT INTO counties (
  geoid, statefp, countyfp, stusps, state_name, name, namelsad, lsad, aland, awater, name_norm, geom, center
)
SELECT
  geoid,
  statefp,
  countyfp,
  stusps,
  state_name,
  name,
  namelsad,
  lsad,
  aland,
  awater,
  regexp_replace(regexp_replace(lower(namelsad), '[^a-z0-9\s]', ' ', 'g'), '\s+', ' ', 'g') AS name_norm,
  ST_Multi(geom) AS geom,
  ST_PointOnSurface(geom) AS center
FROM counties_stage
ON CONFLICT (geoid) DO UPDATE SET
  statefp     = EXCLUDED.statefp,
  countyfp    = EXCLUDED.countyfp,
  stusps      = EXCLUDED.stusps,
  state_name  = EXCLUDED.state_name,
  name        = EXCLUDED.name,
  namelsad    = EXCLUDED.namelsad,
  lsad        = EXCLUDED.lsad,
  aland       = EXCLUDED.aland,
  awater      = EXCLUDED.awater,
  name_norm   = EXCLUDED.name_norm,
  geom        = EXCLUDED.geom,
  center      = EXCLUDED.center;

DROP TABLE IF EXISTS counties_stage;
COMMIT;
SQL
)

docker run -i --rm postgres:18 bash -lc "export PGSSLMODE=require; export PGPASSWORD='${PASS}'; psql -h '${HOST}' -U '${USER}' -d '${DB}' -p '${PORT}' -v ON_ERROR_STOP=1" <<EOF
${MERGE_SQL}
EOF

echo "[3/3] Done. Verify with: SELECT COUNT(*) FROM counties;"
