# Geoapify geocoding and OSM boundary lookup, with on-disk caches.
# The KMZ source ships finished polygons and needs none of this.

import json
import os
import time

import geopandas as gpd
import osmnx as ox
import pandas as pd
import requests
from dotenv import load_dotenv

# osmnx 2.x keeps its exception types here and exports nothing at the top level
from osmnx._errors import InsufficientResponseError, ResponseStatusCodeError
from shapely import wkt

from maybemap.config import (
    CACHE_DIR,
    MAX_CANDIDATE_AREA_M2,
    OSM_FETCH_RETRIES,
    OSM_FETCH_RETRY_PAUSE_S,
    PROJECT_ROOT,
    SEARCH_RADIUS_M,
)
from maybemap.states import state_abbrev, state_full

# Anchored to the project root so the key is found regardless of CWD
load_dotenv(PROJECT_ROOT / ".env")

ox.settings.use_cache = True
ox.settings.cache_folder = str(CACHE_DIR / "osmnx")
ox.settings.log_console = False

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "MaybeMap/1.0"})

GEOCODE_CACHE_PATH = CACHE_DIR / "city_geocodes.json"
BOUNDARY_CACHE_PATH = CACHE_DIR / "osm_boundaries.json"


def _load_cache(path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    try:
        return json.loads(path.read_text() or "{}")
    except json.JSONDecodeError:
        return {}


GEOCODE_CACHE: dict[str, dict[str, float] | None] = _load_cache(GEOCODE_CACHE_PATH)
BOUNDARY_WKT: dict[str, str | None] = _load_cache(BOUNDARY_CACHE_PATH)


def _save_geocode_cache():
    GEOCODE_CACHE_PATH.write_text(json.dumps(GEOCODE_CACHE, indent=2))


def _save_boundary_cache():
    BOUNDARY_CACHE_PATH.write_text(json.dumps(BOUNDARY_WKT, indent=2))


def _require_api_key() -> str:
    # Checked here rather than at import so KMZ maps run without a key
    key = os.getenv("GEOAPIFY_KEY")
    if not key:
        raise RuntimeError("Missing GEOAPIFY_KEY in environment. Please set it in .env")
    return key


def geoapify_forward(text: str, *, retries: int = 3, backoff: float = 0.8):
    url = "https://api.geoapify.com/v1/geocode/search"
    params = {
        "text": text,
        "apiKey": _require_api_key(),
        "limit": 1,
        "filter": "countrycode:us",
    }
    for attempt in range(1, retries + 1):
        try:
            r = SESSION.get(url, params=params, timeout=15)
            if r.status_code == 429:
                time.sleep(backoff * attempt)
                continue
            r.raise_for_status()
            feats = r.json().get("features") or []
            if not feats:
                return None
            lon, lat = feats[0]["geometry"]["coordinates"]
            props = feats[0].get("properties") or {}
            return {
                "lon": float(lon),
                "lat": float(lat),
                "state_code": props.get("state_code"),
            }
        except requests.RequestException:
            if attempt == retries:
                return None
            time.sleep(backoff * attempt)
    return None


def geocode_city_state(city: str, state: str, progress_info: str = ""):
    # Keyed on the full state name: the xlsx says "Missouri" and the CSV says "MO",
    # and a cache that answers only one of them is a cache that never hits.
    full = state_full(state)
    key = f"{city}, {full}"
    if key in GEOCODE_CACHE:
        return GEOCODE_CACHE[key]

    print(f"Geocoding: {city}, {state} {progress_info}")
    res = geoapify_forward(f"{city}, {full}, USA")

    # A misspelled town name matches something plausible-looking in another state --
    # "Lamonte, Missouri" resolves to a street in Houston. One such point stretches
    # its region's enclosing circle across half the country, so the answer is
    # thrown away rather than mapped.
    if res and res.get("state_code") and res["state_code"] != state_abbrev(state):
        print(
            f"   Rejected: {city}, {state} geocoded into {res['state_code']}. "
            f"Check the spelling in the city data."
        )
        res = None

    GEOCODE_CACHE[key] = res
    _save_geocode_cache()

    if res:
        print(f"   Found coordinates: {res['lat']:.4f}, {res['lon']:.4f}")
    else:
        print(f"   Geocoding failed for {city}, {state}")

    return res


def get_cached_osm_polygon(city: str, state: str):
    key = f"{city}, {state_full(state)}".upper()
    w = BOUNDARY_WKT.get(key, None)
    if not w:
        return None
    geom = wkt.loads(w)
    g = gpd.GeoSeries([geom], crs=4326).to_crs(3857)
    if g.iloc[0].area > MAX_CANDIDATE_AREA_M2:  # discard too-large cached shapes
        return None
    return geom


def cache_osm_polygon(city: str, state: str, geom):
    key = f"{city}, {state_full(state)}".upper()
    BOUNDARY_WKT[key] = geom.wkt if geom is not None else None
    _save_boundary_cache()


def _with_retries(what: str, call):
    # Separates "the API answered, there is nothing there" from "the request
    # failed". The first is a real answer; the second, swallowed, hands back a
    # fallback circle indistinguishable from a genuine boundary.
    for attempt in range(1, OSM_FETCH_RETRIES + 1):
        try:
            return call()
        except InsufficientResponseError:
            return None
        except (ResponseStatusCodeError, requests.RequestException) as exc:
            if attempt == OSM_FETCH_RETRIES:
                raise RuntimeError(
                    f"{what} failed after {OSM_FETCH_RETRIES} attempts "
                    f"({type(exc).__name__}: {exc}). Boundaries fetched so far are "
                    f"saved in {BOUNDARY_CACHE_PATH.name}, so rerunning resumes from here."
                ) from exc
            pause = OSM_FETCH_RETRY_PAUSE_S * attempt
            print(
                f"   {what} failed ({type(exc).__name__}), "
                f"retrying in {pause:.0f}s [{attempt}/{OSM_FETCH_RETRIES - 1}]"
            )
            time.sleep(pause)
    raise AssertionError("unreachable")  # the loop always returns or raises


def _area_km2(geom_ll) -> float:
    return gpd.GeoSeries([geom_ll], crs=4326).to_crs(3857).area.iloc[0] / 1e6


# Nominatim ranks a municipality 16 or finer; a county is 12 and a state is 8.
# "Clinton, Missouri" matches Clinton County ahead of the city of Clinton, so the
# answer has to be checked rather than trusted.
MUNICIPAL_TYPES = frozenset({"city", "town", "village", "municipality", "borough", "hamlet"})
MUNICIPAL_PLACE_RANK = 16


def _is_municipal(row) -> bool:
    kind = row.get("addresstype")
    if isinstance(kind, str):
        return kind in MUNICIPAL_TYPES
    rank = row.get("place_rank")
    return rank is not None and rank >= MUNICIPAL_PLACE_RANK


def _boundary_from_nominatim(city: str, state: str, county: str = ""):
    # Asks "what is the boundary of this named place" and gets a ~4kB answer, rather
    # than downloading every boundary within SEARCH_RADIUS_M to keep one polygon.
    def _lookup(query):
        try:
            return ox.geocode_to_gdf(query)
        except TypeError:
            # What osmnx raises when the best match carries no (Multi)Polygon
            return None

    # The county comes first because Missouri is full of city/county name
    # collisions: "Clark, Missouri" matches Clark County ahead of the village.
    full = state_full(state)
    queries = [f"{city}, {county} County, {full}, USA"] if county else []
    queries.append(f"{city}, {full}, USA")

    for query in queries:
        gdf = _with_retries(f"Nominatim lookup for {query}", lambda q=query: _lookup(q))
        if gdf is None or gdf.empty:
            continue
        row = gdf.iloc[0]
        if not _is_municipal(row):
            print(f"   Nominatim matched {row.get('name', '?')!r} rather than a municipality")
            continue
        if _area_km2(row.geometry) * 1e6 > MAX_CANDIDATE_AREA_M2:
            continue
        return row.geometry
    return None


def _boundary_from_overpass(city: str, state: str, pt_3857):
    # Fallback for places Nominatim cannot name-match. Heavy: two wide-radius
    # queries that pull thousands of features to keep one, so it is a last resort.
    city_up = city.upper()
    pt_ll = gpd.GeoSeries([pt_3857], crs=3857).to_crs(4326).iloc[0]
    center = (pt_ll.y, pt_ll.x)  # (lat, lon)

    def _gather(tags):
        got = _with_retries(
            f"Overpass request for {city}, {state}",
            lambda: ox.features_from_point(center, tags=tags, dist=SEARCH_RADIUS_M),
        )
        return gpd.GeoDataFrame(geometry=[]) if got is None else got

    g1 = _gather({"boundary": "administrative", "admin_level": ["7", "8", "9", "10"]})
    g2 = _gather({"place": ["city", "town", "village", "municipality"]})
    if len(g1) == 0 and len(g2) == 0:
        return None

    g = pd.concat([g1, g2], ignore_index=True, sort=False)
    g = g[g.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    if g.empty:
        return None

    if "name" in g.columns and g["name"].notna().any():
        g["NAME_UP"] = g["name"].str.upper()
        exact = g[g["NAME_UP"] == city_up]
        if not exact.empty:
            g = exact

    g_3857 = g.to_crs(3857).copy()
    g_3857["__area"] = g_3857.geometry.area
    g_3857 = g_3857[g_3857["__area"] < MAX_CANDIDATE_AREA_M2]
    if g_3857.empty:
        return None

    dists = g_3857.geometry.centroid.distance(pt_3857)
    chosen_3857 = g_3857.loc[dists.idxmin(), "geometry"]
    return gpd.GeoSeries([chosen_3857], crs=3857).to_crs(4326).iloc[0]


def fetch_osm_city_polygon(
    city: str, state: str, pt_3857, progress_info: str = "", county: str = ""
):
    print(f"Fetching OSM boundary: {city}, {state} {progress_info}")

    geom_ll = _boundary_from_nominatim(city, state, county)
    source = "Nominatim"
    if geom_ll is None:
        try:
            geom_ll = _boundary_from_overpass(city, state, pt_3857)
        except RuntimeError as exc:
            # Overpass is only the fallback now, so one unreachable probe should not
            # end a run Nominatim is otherwise answering. A systemic outage still
            # trips the consecutive-fallback guard in the caller.
            print(f"   {exc}")
            return None
        source = "Overpass"
    if geom_ll is None:
        return None

    print(f"   Found OSM boundary polygon (area: {_area_km2(geom_ll):.2f} km², via {source})")
    return geom_ll
