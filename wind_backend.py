"""
Wind resource backend for the Turbine Yield Estimator.

Fetches real long-term average wind speed data from NASA POWER (free, public,
no API key required) and converts it into hub-height Weibull parameters that
the frontend can use for AEP calculations.

Data source: NASA POWER Climatology API
https://power.larc.nasa.gov/docs/services/api/temporal/climatology/

NASA POWER gives ~50km grid resolution wind data derived from the MERRA-2
reanalysis, 1981-present. It's much coarser than Global Wind Atlas (250m,
terrain-corrected) but it's free, keyless, and reliable for a v1 backend.
Swapping in GWA later is a drop-in replacement for the fetch_wind_climatology
function below, once you've sorted API access with them.

Run locally:
    pip install fastapi uvicorn httpx --break-system-packages
    uvicorn wind_backend:app --reload --port 8000

Then the frontend can call: http://localhost:8000/api/wind-resource?lat=56.6&lon=-4.9&hub_height=80
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import httpx
import math
import time

app = FastAPI(title="Wind Resource API")

# CORS restricted to the deployed frontend (plus localhost for development).
# The previous wildcard let any website on the internet call this API from
# their visitors' browsers, burning the free-tier quota. Note this only stops
# browser-based cross-origin use — direct curl/script access is unaffected
# (that would need auth or rate limiting, deliberately not added at this scale).
# If the frontend moves to a custom domain, add it here and redeploy.
ALLOWED_ORIGINS = [
    "https://gorgeous-swan-134cb1.netlify.app",
    "http://localhost:8000",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET"],
    allow_headers=["*"],
)

NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/climatology/point"

# Simple in-memory cache: wind data doesn't change, so no need to hit NASA's
# servers repeatedly for the same location. Keyed by lat/lon rounded to 0.25
# degrees (roughly the native grid resolution) plus hub height.
_cache: dict[str, dict] = {}
CACHE_TTL_SECONDS = 60 * 60 * 24 * 30  # 30 days — this data is climatological, it won't go stale fast
CACHE_MAX_ENTRIES = 5000  # simple insurance against unbounded growth on a long-lived process


def _cache_put(cache: dict, key: str, value: dict) -> None:
    """Insert with a crude size cap: evict the oldest ~20% when full. FIFO is
    fine here — entries are climatological and cheap to refetch."""
    if len(cache) >= CACHE_MAX_ENTRIES:
        for k in list(cache.keys())[: CACHE_MAX_ENTRIES // 5]:
            del cache[k]
    cache[key] = value


def _cache_key(lat: float, lon: float, hub_height: float, alpha: float) -> str:
    return f"{round(lat * 4) / 4}_{round(lon * 4) / 4}_{hub_height}_{alpha}"


def power_law_extrapolate(v_ref: float, h_ref: float, h_target: float, alpha: float) -> float:
    """
    Extrapolate wind speed from a reference height to hub height using the
    standard wind power law. alpha is user-supplied since, without site
    measurements at multiple heights, there's no way to derive a
    location-specific shear exponent — 0.14 is a common default for open
    onshore terrain, but forested or urban surroundings run higher (0.2-0.3+),
    and smooth offshore/flat terrain runs lower (~0.10-0.12).
    """
    return v_ref * (h_target / h_ref) ** alpha


def weibull_params_from_mean(v_mean: float, k: float = 2.0) -> tuple[float, float]:
    """
    Approximate Weibull scale (A) from mean wind speed for a given shape
    parameter k.
    """
    gamma_term = math.gamma(1 + 1 / k)
    A = v_mean / gamma_term
    return A, k


def estimate_k_from_monthly_variability(monthly_speeds: list[float]) -> float:
    """
    Rough estimate of the Weibull shape parameter k, used only when we don't
    have real per-sector data (i.e. no site centre / GWA rose set) — the
    NASA POWER climatology fallback previously just assumed k=2 (Rayleigh)
    everywhere, which is wrong: real k typically ranges ~1.5 (variable,
    storm-driven climates) to ~2.5+ (steady, consistently windy regimes).

    This uses the Justus & Mikhail (1976) empirical relation
        k ≈ (σ / mean)^-1.086
    which is normally applied to the standard deviation of the *actual*
    (hourly or sub-daily) wind speed distribution. We don't have that from
    a climatology endpoint — only 12 monthly means — so this substitutes
    the coefficient of variation of monthly means as a proxy for the true
    wind speed spread.

    IMPORTANT CAVEAT: monthly-mean variability only captures the seasonal
    cycle, not the much larger synoptic/diurnal variability that actually
    dominates a real Weibull k. This will UNDERSTATE the true spread in most
    climates (i.e. bias k slightly high / distribution slightly too narrow).
    It's a genuine improvement over a flat, location-blind k=2 default, but
    it is a heuristic proxy, not a rigorous derivation — treat it as better
    than nothing, not as good as real sub-daily data (which is what the GWA
    wind-rose path, used once a site centre is set, actually provides).
    """
    if not monthly_speeds or len(monthly_speeds) < 12:
        return 2.0
    mean = sum(monthly_speeds) / len(monthly_speeds)
    if mean <= 0:
        return 2.0
    variance = sum((v - mean) ** 2 for v in monthly_speeds) / len(monthly_speeds)
    stddev = math.sqrt(variance)
    cv = stddev / mean
    if cv <= 0:
        return 2.5
    k = cv ** (-1.086)
    return max(1.3, min(3.0, k))  # clamp to a physically plausible range


def isa_air_density(elevation_m: float) -> float:
    """
    Air density from elevation only, using the ICAO International Standard
    Atmosphere barometric formula (sea-level 15°C reference, 1.225 kg/m³).
    This ignores actual site temperature/pressure on the day, which a real
    IEC 61400-12 correction would use — elevation-only is the common
    simplified approximation for a screening tool.
    """
    return 1.225 * (1 - 2.25577e-5 * max(elevation_m, 0)) ** 5.25588


async def fetch_wind_climatology(lat: float, lon: float) -> tuple[float, float]:
    """
    Query NASA POWER for the long-term average wind speed at 50m above
    ground level for this location. Returns (annual_mean_ws, estimated_k).
    """
    params = {
        "parameters": "WS50M",
        "community": "RE",
        "longitude": lon,
        "latitude": lat,
        "format": "JSON",
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(NASA_POWER_URL, params=params)
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail="NASA POWER API request failed")
        data = resp.json()

    try:
        ws50m_block = data["properties"]["parameter"]["WS50M"]
        ws50m_annual = float(ws50m_block["ANN"])
    except (KeyError, TypeError):
        raise HTTPException(status_code=502, detail="Unexpected response shape from NASA POWER")

    month_keys = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    try:
        monthly = [float(ws50m_block[m]) for m in month_keys]
        k_estimate = estimate_k_from_monthly_variability(monthly)
    except (KeyError, TypeError, ValueError):
        k_estimate = 2.0  # fall back to the old assumption if monthly data is missing for any reason

    return ws50m_annual, k_estimate


@app.get("/api/wind-resource")
async def wind_resource(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    hub_height: float = Query(80, ge=10, le=300, description="Turbine hub height in metres"),
    alpha: float = Query(0.14, ge=0.05, le=0.4, description="Wind shear exponent (power law). No site data means this is an assumption, not a measurement."),
):
    """
    Returns a hub-height wind resource estimate for a given point:
    mean wind speed, Weibull A/k parameters, and the data vintage/source.
    """
    key = _cache_key(lat, lon, hub_height, alpha)
    cached = _cache.get(key)
    if cached and (time.time() - cached["fetched_at"]) < CACHE_TTL_SECONDS:
        return cached["result"]

    v_50m, k_estimate = await fetch_wind_climatology(lat, lon)
    v_hub = power_law_extrapolate(v_50m, h_ref=50, h_target=hub_height, alpha=alpha)
    A, k = weibull_params_from_mean(v_hub, k=k_estimate)

    result = {
        "lat": lat,
        "lon": lon,
        "hub_height_m": hub_height,
        "shear_exponent_used": alpha,
        "mean_wind_speed_50m": round(v_50m, 2),
        "mean_wind_speed_hub": round(v_hub, 2),
        "weibull_A": round(A, 2),
        "weibull_k": round(k, 3),
        "weibull_k_note": "Estimated from seasonal (monthly) variability via the Justus-Mikhail relation — "
                          "a heuristic proxy, not derived from real sub-daily wind speed distribution. "
                          "See METHODOLOGY.md for the caveat on this.",
        "source": "NASA POWER climatology (MERRA-2 reanalysis, 1981-present, ~50km grid)",
        "note": "Screening-grade estimate. Not terrain-corrected — a real ridge or valley "
                "at this exact point could differ meaningfully from this grid-cell average.",
    }
    _cache_put(_cache, key, {"result": result, "fetched_at": time.time()})
    return result


@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Site elevation + terrain speed-up: air density, terrain complexity flag,
# and a genuine (if simplified) orographic correction
# ---------------------------------------------------------------------------
# Uses Open-Elevation (free, keyless, https://open-elevation.com) to sample
# the turbine's own elevation plus a ring of 8 points ~2km away, in a single
# batched request. This gives us three things:
#
#   1. Air density at the turbine (ISA barometric formula) — corrects the
#      power curve for elevation, since manufacturer curves are certified at
#      standard sea-level density.
#
#   2. A terrain complexity flag — if elevation varies a lot nearby, real
#      orographic effects are likely present that the correction below only
#      partially captures.
#
#   3. A terrain speed-up/slowdown correction — this is the real fix for the
#      biggest known gap in this tool (see METHODOLOGY.md §8/§11). It is NOT
#      a WAsP-equivalent orographic flow model (that requires a full linearized
#      spectral solution over an actual terrain grid, a genuine commercial
#      product's worth of engineering). It IS a simplified version of the same
#      underlying physics WAsP's own orographic module is built on:
#
#      Jackson & Hunt (1975) linear flow theory gives the fractional wind
#      speed-up over an isolated 2D hill, at the surface near the crest, as
#      approximately:
#
#          ΔS/S ≈ 2H/L
#
#      where H is the hill height above its surroundings and L is the
#      hill's characteristic half-length. This is the standard textbook
#      approximation (see e.g. Troen & Petersen 1989, the original WAsP
#      reference) for how much a hill or ridge speeds up the wind at its
#      crest relative to the regional/undisturbed wind speed.
#
#      We approximate H as the turbine's own elevation minus the mean
#      elevation of an 8-point ring ~2km around it ("local prominence" — how
#      much higher this exact point sits than its surroundings), and L as
#      the 2km ring radius. A turbine sitting on a local high point gets a
#      positive speed-up; one sitting in a local dip gets a slowdown. The
#      result is clamped to ±25% to avoid over-claiming precision this
#      simplified, isotropic (direction-blind) approximation doesn't have.
#
#      KNOWN LIMITATIONS of this correction (documented in METHODOLOGY.md):
#      - Isotropic: applied equally regardless of wind direction, when real
#        speed-up is highly direction-dependent (only the upwind slope
#        matters for a given wind direction).
#      - Uses only 8 sampled points, not a full terrain grid — a real ridge
#        or valley shape narrower than the 2km sampling radius won't be
#        captured accurately.
#      - No roughness-change or displacement-height modelling, no
#        atmospheric stability dependence — WAsP's full model includes both.
#      - This is a genuine improvement over doing nothing, not a substitute
#        for a real WAsP/CFD terrain assessment before any investment decision.
OPEN_ELEVATION_URL = "https://api.open-elevation.com/api/v1/lookup"
_elevation_cache: dict[str, dict] = {}

TERRAIN_RING_RADIUS_M = 2000
MAX_SPEEDUP_FRACTION = 0.25  # clamp — this simplified model shouldn't claim more precision than this


def _offset_point(lat: float, lon: float, bearing_deg: float, distance_m: float) -> dict:
    """Offset a lat/lon point by a given bearing and distance (small-distance approximation)."""
    dlat = (distance_m * math.cos(math.radians(bearing_deg))) / 111320.0
    dlon = (distance_m * math.sin(math.radians(bearing_deg))) / (111320.0 * max(math.cos(math.radians(lat)), 0.01))
    return {"latitude": lat + dlat, "longitude": lon + dlon}


@app.get("/api/site-elevation")
async def site_elevation(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
):
    # Cache at 4 decimal places (~11 m). The previous 2-dp key (~0.7-1.1 km)
    # caused turbines within ~1 km of each other to SHARE one cache slot: whichever
    # turbine's request completed first had its terrain/elevation result served to
    # its neighbours, race-dependent across reloads. Found via a user bug report
    # where two turbines 537 m apart swapped elevation values between sessions,
    # moving gross yield ~2%. Elevation is point data — unlike the wind-resource
    # cache (deliberately coarse to match NASA POWER's ~50 km grid), it must not
    # be shared between distinct turbine positions.
    key = f"{round(lat * 10000) / 10000}_{round(lon * 10000) / 10000}"
    cached = _elevation_cache.get(key)
    if cached and (time.time() - cached["fetched_at"]) < CACHE_TTL_SECONDS:
        return cached["result"]

    # Centre point + an 8-point ring at ~2km for the terrain speed-up estimate,
    # plus a closer 4-point ring at ~1km (kept from before) for the complexity flag.
    ring_2km = [_offset_point(lat, lon, bearing, TERRAIN_RING_RADIUS_M) for bearing in range(0, 360, 45)]
    ring_1km = [_offset_point(lat, lon, bearing, 1000) for bearing in [0, 90, 180, 270]]
    points = [{"latitude": lat, "longitude": lon}] + ring_2km + ring_1km

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(OPEN_ELEVATION_URL, json={"locations": points})
            if resp.status_code != 200:
                raise HTTPException(status_code=502, detail="Open-Elevation API request failed")
            data = resp.json()
        elevations = [r["elevation"] for r in data["results"]]
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Elevation lookup failed: {exc}")

    centre_elevation = elevations[0]
    ring_2km_elevations = elevations[1:9]     # the 8-point 2km ring, in order
    ring_1km_elevations = elevations[9:13]    # the 4-point 1km ring

    # Terrain complexity: based on actual SLOPE (rise/run), not raw elevation range.
    # A 40m elevation change over 1km is only a 4% grade — completely ordinary rolling
    # terrain across most of the UK, not something that breaks the linear flow theory
    # this tool's terrain correction relies on. Real "complex terrain" in the sense
    # that matters (where linear theory like Jackson & Hunt starts to break down, and
    # where GWA's own RIX metric would flag caution) means genuinely steep slopes —
    # RIX itself is defined as the fraction of surrounding terrain exceeding 30% slope.
    # We use a somewhat more cautious 18% threshold here, checked across all 12 sampled
    # points (both rings) against their known distance from the turbine.
    TERRAIN_SLOPE_THRESHOLD = 0.18
    max_slope = 0.0
    for elev, dist in [(e, 2000) for e in ring_2km_elevations] + [(e, 1000) for e in ring_1km_elevations]:
        slope = abs(elev - centre_elevation) / dist
        if slope > max_slope:
            max_slope = slope
    elevation_range = max(ring_1km_elevations + [centre_elevation]) - min(ring_1km_elevations + [centre_elevation])

    # Terrain speed-up: local prominence (this point vs. its 2km surroundings),
    # via the simplified Jackson & Hunt 2H/L relation — see the module docstring
    # above for the full explanation and caveats.
    mean_surrounding_elevation = sum(ring_2km_elevations) / len(ring_2km_elevations)
    prominence_m = centre_elevation - mean_surrounding_elevation
    raw_speedup_fraction = (2 * prominence_m) / TERRAIN_RING_RADIUS_M
    speedup_fraction = max(-MAX_SPEEDUP_FRACTION, min(MAX_SPEEDUP_FRACTION, raw_speedup_fraction))

    air_density = isa_air_density(centre_elevation)

    result = {
        "lat": lat,
        "lon": lon,
        "elevation_m": round(centre_elevation, 1),
        "elevation_range_1km_m": round(elevation_range, 1),
        "max_slope_pct": round(max_slope * 100, 1),
        "terrain_complex": max_slope > TERRAIN_SLOPE_THRESHOLD,
        "prominence_m": round(prominence_m, 1),
        "terrain_speedup_fraction": round(speedup_fraction, 4),
        "terrain_speedup_clamped": abs(raw_speedup_fraction) > MAX_SPEEDUP_FRACTION,
        "air_density_kgm3": round(air_density, 4),
        "air_density_ratio": round(air_density / 1.225, 4),
        "source": "Open-Elevation (SRTM/derived DEM); terrain speed-up via simplified Jackson & Hunt (1975) linear flow theory",
        "note": "Terrain speed-up is a simplified, isotropic (direction-blind) approximation of the same "
                "physics WAsP's orographic model uses — a real improvement over no terrain correction at all, "
                "but not equivalent to a full WAsP/CFD terrain assessment. See METHODOLOGY.md §11.",
    }
    _cache_put(_elevation_cache, key, {"result": result, "fetched_at": time.time()})
    return result


# ---------------------------------------------------------------------------
# Wind rose: real Global Wind Atlas generalized wind climate (GWC) data
# ---------------------------------------------------------------------------
# GWA's GWC files contain sector-wise frequency of occurrence plus Weibull A/k
# per sector, at 5 reference roughness lengths (0.0, 0.03, 0.1, 0.4, 1.5m) and
# 5 heights (10/50/100/150/200m). This is the real methodology used by WAsP
# and every serious wind resource tool — NASA POWER's climatology endpoint
# (used elsewhere in this backend) can't produce this, since it only gives a
# single annual mean, not a directional distribution.
#
# The wind-stats library (pip install wind-stats) wraps GWA's API and returns
# this as an xarray Dataset. We interpolate to the requested hub height and a
# roughness length, since we don't have local land-cover data to pick one
# precisely — 0.03m ("open agricultural land, few buildings") is a common
# default for open onshore terrain.
try:
    from wind_stats import get_gwc_data
    WIND_STATS_AVAILABLE = True
except ImportError:
    WIND_STATS_AVAILABLE = False

_rose_cache: dict[str, dict] = {}


@app.get("/api/wind-rose")
def wind_rose(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    hub_height: float = Query(80, ge=10, le=200),
    roughness: float = Query(0.03, description="Surface roughness length in metres. 0.0=water, 0.03=open farmland, 0.1=scattered obstacles, 0.4=many obstacles/forest, 1.5=urban/city centre."),
):
    """
    Returns a 12-sector wind rose (frequency %, Weibull A, Weibull k per
    sector) from Global Wind Atlas GWC data, interpolated to hub height
    and the given roughness length. Note this is still not terrain-corrected
    at your exact point — GWC data represents a generalized regional climate,
    which is the input to microscale modelling (e.g. WAsP), not the output.
    """
    if not WIND_STATS_AVAILABLE:
        raise HTTPException(status_code=501, detail="wind-stats package not installed on this backend")

    # 4-dp key (~11 m): GWA GWC data has ~250 m effective resolution, so the
    # previous 0.25-degree (~28 km) rounding could serve one site centre's rose
    # to a different site tens of km away.
    key = f"{round(lat * 10000) / 10000}_{round(lon * 10000) / 10000}_{hub_height}_{roughness}"
    cached = _rose_cache.get(key)
    if cached and (time.time() - cached["fetched_at"]) < CACHE_TTL_SECONDS:
        return cached["result"]

    try:
        ds = get_gwc_data(lat, lon)
        interpolated = ds.interp(height=hub_height, roughness=roughness)
        sectors = ds["sector"].values.tolist()
        frequency = interpolated["frequency"].values.tolist()
        weibull_A = interpolated["A"].values.tolist()
        weibull_k = interpolated["k"].values.tolist()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Global Wind Atlas lookup failed: {exc}")

    result = {
        "lat": lat,
        "lon": lon,
        "hub_height_m": hub_height,
        "roughness_m": roughness,
        "sectors_deg": sectors,
        "frequency_pct": frequency,
        "weibull_A": weibull_A,
        "weibull_k": weibull_k,
        "source": "Global Wind Atlas v3 GWC (generalized wind climate, not terrain-corrected at this exact point)",
    }
    _cache_put(_rose_cache, key, {"result": result, "fetched_at": time.time()})
    return result
