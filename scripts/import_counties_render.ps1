# EnforcedSpeed — Import US Counties (cb_2024_us_county_500k) into Render Postgres/PostGIS
#
# Requires:
#   - Docker Desktop running
#
# Usage (PowerShell):
#   .\scripts\import_counties_render.ps1 -CountiesZip ".\cb_2024_us_county_500k.zip" -DatabaseUrl "<Render EXTERNAL DATABASE URL>"
#
# Notes:
#   - Uses Docker for BOTH:
#       - GDAL/ogr2ogr import (osgeo/gdal)
#       - psql merge step (postgres)
#   - This script creates a temporary staging table (counties_stage), then upserts into the app's
#     canonical table (counties) which the app expects.

param(
  [Parameter(Mandatory=$false)]
  [string]$CountiesZip,

  [Parameter(Mandatory=$false)]
  [string]$DatabaseUrl,

  [Parameter(Mandatory=$false)]
  [ValidateSet('auto','require','disable','prefer')]
  [string]$SslMode = 'auto'
)

$ErrorActionPreference = "Stop"

if (-not $CountiesZip -or $CountiesZip.Trim() -eq "") {
  $CountiesZip = Read-Host "Path to cb_2024_us_county_500k.zip"
}
if (-not (Test-Path $CountiesZip)) {
  throw "Counties zip not found: $CountiesZip"
}

if (-not $DatabaseUrl -or $DatabaseUrl.Trim() -eq "") {
  $DatabaseUrl = Read-Host "Paste Database URL (postgresql://...)"
}

# Decide SSL mode.
# IMPORTANT: We treat SslMode='auto' as a sentinel (default) and compute an effective mode.
# - Local DBs (localhost/127.0.0.1/host.docker.internal) should use sslmode=disable.
# - Render/remote should use sslmode=require.
$effectiveSsl = $SslMode
if (-not $effectiveSsl -or $effectiveSsl.Trim() -eq "" -or $effectiveSsl -eq 'auto') {
  if ($DatabaseUrl -match 'localhost|127\.0\.0\.1|host\.docker\.internal') { $effectiveSsl = 'disable' } else { $effectiveSsl = 'require' }
}

# Ensure sslmode=<mode> is present in URL (do NOT overwrite if already provided)
if ($DatabaseUrl -notmatch 'sslmode=') {
  if ($DatabaseUrl -match '\?') { $DatabaseUrl = "$DatabaseUrl&sslmode=$effectiveSsl" } else { $DatabaseUrl = "$DatabaseUrl?sslmode=$effectiveSsl" }
}

# Parse the URI into libpq keywords for ogr2ogr
# NOTE: PowerShell's type cast ([System.Uri]$x) can silently yield $null in some contexts.
# Use the .NET constructor to force a hard failure on invalid URLs.
if (-not $DatabaseUrl) {
  throw "DatabaseUrl is required"
}
try {
  $uri = [System.Uri]::new($DatabaseUrl)
} catch {
  throw "Invalid DatabaseUrl (expected postgresql://...): $DatabaseUrl"
}

$dbName = $uri.AbsolutePath.TrimStart('/')
$hostName = $uri.Host
$portNum = if ($uri.Port -gt 0) { $uri.Port } else { 5432 }
$userInfo = $uri.UserInfo.Split(':', 2)
$dbUser = $userInfo[0]
$dbPass = if ($userInfo.Length -gt 1) { $userInfo[1] } else { "" }

if (-not $dbName -or -not $hostName -or -not $dbUser) {
  throw "Could not parse DatabaseUrl into host/db/user"
}

$pgConn = "host=$hostName port=$portNum dbname=$dbName user=$dbUser password=$dbPass sslmode=$effectiveSsl"

# GDAL container image (osgeo/gdal migrated; use GHCR latest)
$GdalImage = "ghcr.io/osgeo/gdal:alpine-small-latest"

# Temp working directory
$tmpRoot = Join-Path $env:TEMP ("es_counties_" + [System.Guid]::NewGuid().ToString())
New-Item -ItemType Directory -Path $tmpRoot | Out-Null

try {
  Write-Host "[1/3] Extracting zip to: $tmpRoot"
  Expand-Archive -Force -Path $CountiesZip -DestinationPath $tmpRoot

  $shp = Get-ChildItem -Path $tmpRoot -Recurse -Filter *.shp | Select-Object -First 1
  if (-not $shp) {
    throw "No .shp found after extracting $CountiesZip"
  }

  Write-Host "[2/3] Importing shapefile to staging table counties_stage via GDAL..."

  # Mount the temp folder into /data (Linux path inside container)
  # NOTE: PowerShell needs an absolute path for docker -v
  $absTmp = (Resolve-Path $tmpRoot).Path
  $relShp = $shp.FullName.Substring($absTmp.Length).TrimStart([char[]]@('\','/')) -replace '\\','/'
  $containerShp = "/data/$relShp"

  docker run --rm -v "${absTmp}:/data" $GdalImage ogr2ogr -f PostgreSQL "PG:$pgConn" "$containerShp" -nln counties_stage -overwrite -nlt MULTIPOLYGON -t_srs EPSG:4326 -lco GEOMETRY_NAME=geom
  if ($LASTEXITCODE -ne 0) { throw "GDAL import failed (docker/gdal exit $LASTEXITCODE)" }

  Write-Host "[3/3] Upserting into canonical counties table + computing name_norm + center..."

  $mergeSql = @"
-- Ensure PostGIS + canonical table exist (script is self-contained)
CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS counties (
  geoid      text PRIMARY KEY,
  statefp    text,
  countyfp   text,
  stusps     text,
  state_name text,
  name       text,
  namelsad   text,
  lsad       text,
  aland      bigint,
  awater     bigint,
  name_norm  text,
  geom       geometry(MultiPolygon, 4326),
  center     geometry(Point, 4326)
);

CREATE INDEX IF NOT EXISTS counties_geom_gix ON counties USING GIST (geom);
CREATE INDEX IF NOT EXISTS counties_center_gix ON counties USING GIST (center);
CREATE INDEX IF NOT EXISTS counties_name_norm_idx ON counties (name_norm);

BEGIN;

-- Upsert into canonical table (created by app boot, but safe if already exists)
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
  regexp_replace(regexp_replace(lower(namelsad), '[^a-z0-9\\s]', ' ', 'g'), '\\s+', ' ', 'g') AS name_norm,
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
"@

  # Use postgres image as a psql client.
  # Pass password through env to avoid interactive prompt.
  # Write merge SQL to a temp file to avoid quoting/escaping issues on Windows/PowerShell.
  $mergeFile = Join-Path $tmpRoot "counties_merge.sql"
  Set-Content -Path $mergeFile -Value $mergeSql -Encoding UTF8

  # Use postgres image as a psql client. Mount the temp dir and run -f against the file.
  # Pass password through env to avoid interactive prompt.
  docker run --rm -e PGPASSWORD="$dbPass" -e PGSSLMODE="$effectiveSsl" -v "${tmpRoot}:/work" postgres:18 bash -lc "psql -v ON_ERROR_STOP=1 -h '$hostName' -U '$dbUser' -d '$dbName' -p $portNum -f /work/counties_merge.sql"
  if ($LASTEXITCODE -ne 0) { throw "psql merge failed (exit $LASTEXITCODE)" }

  Write-Host "DONE. Counties imported and ready."
} finally {
  try { Remove-Item -Recurse -Force $tmpRoot } catch {}
}
