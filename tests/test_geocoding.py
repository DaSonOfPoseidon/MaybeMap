# Tests for the OSM boundary lookup. Two things matter here: Nominatim answers
# first and Overpass is only the fallback, and the distinction between "this town
# has no boundary" and "the request failed" -- the first is a real answer, the
# second used to be swallowed into a fallback circle that looked just as genuine.

import geopandas as gpd
import pytest
import requests
from osmnx._errors import InsufficientResponseError, ResponseStatusCodeError
from shapely.geometry import Point

from maybemap import geocoding
from maybemap.config import OSM_FETCH_RETRIES

# Somewhere in mid-Missouri, in EPSG:3857
POINT = Point(-10300000, 4700000)


BOUNDARY = Point(-92.5, 38.9).buffer(0.05)


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(geocoding.time, "sleep", lambda _: None)


@pytest.fixture(autouse=True)
def nominatim_finds_nothing(monkeypatch):
    # Default for the Overpass tests below, which only run as the fallback
    def nothing(*args, **kwargs):
        raise InsufficientResponseError("Nominatim did not geocode query to a geometry")

    monkeypatch.setattr(geocoding.ox, "geocode_to_gdf", nothing)


def _nominatim_returns(monkeypatch, geom, addresstype="town", name="Nowhere"):
    frame = gpd.GeoDataFrame(
        {"name": [name], "addresstype": [addresstype]}, geometry=[geom], crs=4326
    )
    monkeypatch.setattr(geocoding.ox, "geocode_to_gdf", lambda *a, **k: frame)


def _fetch():
    return geocoding.fetch_osm_city_polygon("Nowhere", "Missouri", POINT)


def test_an_empty_response_is_a_real_answer(monkeypatch):
    calls = []

    def boom(*args, **kwargs):
        calls.append(1)
        raise InsufficientResponseError("No data elements in server response")

    monkeypatch.setattr(geocoding.ox, "features_from_point", boom)
    assert _fetch() is None
    # Both tag queries ran once each: no retry, because nothing failed
    assert len(calls) == 2


@pytest.mark.parametrize(
    "exc",
    [
        requests.ConnectionError("Connection refused"),
        requests.Timeout("timed out"),
        ResponseStatusCodeError("overpass-api.de returned HTTP status code 429"),
    ],
)
def test_every_kind_of_failure_is_retried_then_raised(monkeypatch, exc):
    calls = []

    def boom(*args, **kwargs):
        calls.append(1)
        raise exc

    monkeypatch.setattr(geocoding.ox, "features_from_point", boom)
    # Retried to the limit, then raised rather than mistaken for "no boundary"
    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        _fetch()
    assert len(calls) == OSM_FETCH_RETRIES


def test_a_blip_is_retried_rather_than_fatal(monkeypatch):
    calls = []

    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise requests.ConnectionError("Connection reset by peer")
        raise InsufficientResponseError("No data elements in server response")

    monkeypatch.setattr(geocoding.ox, "features_from_point", flaky)
    assert _fetch() is None
    # One failure, one retry that answered, then the second tag query
    assert len(calls) == 3


def test_a_polygon_that_comes_back_is_returned(monkeypatch):
    boundary = BOUNDARY
    frame = gpd.GeoDataFrame({"name": ["Nowhere"]}, geometry=[boundary], crs=4326)

    def answer(*args, **kwargs):
        return frame

    monkeypatch.setattr(geocoding.ox, "features_from_point", answer)
    # Round-tripping through EPSG:3857 to pick the nearest candidate costs a
    # little floating point, so compare by area rather than vertex for vertex
    assert _fetch().symmetric_difference(boundary).area < 1e-9


def test_a_long_run_of_misses_aborts(monkeypatch, spreadsheet, seeded_cache):
    # Overpass can answer 200 with an empty result while it is really timing out
    # or serving the wrong regional extract, which raises nothing at all. The only
    # remaining signal is that every city in a row comes back empty.
    from maybemap.sources import cities

    monkeypatch.setattr(cities, "OSM_MAX_CONSECUTIVE_FALLBACKS", 3)
    monkeypatch.setattr(cities, "fetch_osm_city_polygon", lambda *a, **k: None)
    monkeypatch.setattr(cities, "cache_osm_polygon", lambda *a, **k: None)
    monkeypatch.setattr(cities, "get_cached_osm_polygon", lambda *a, **k: geocoding.NOT_CACHED)

    points, _ = cities.load_city_points(spreadsheet)
    with pytest.raises(RuntimeError, match="3 cities in a row returned no OSM boundary"):
        cities.load_city_boundaries(points)


def test_scattered_misses_are_tolerated(monkeypatch, spreadsheet, seeded_cache):
    # Real coverage gaps are interleaved with hits, so the counter keeps resetting
    from maybemap.sources import cities

    seen = []

    def sometimes(city, state, pt, progress="", county=""):
        seen.append(city)
        return None if len(seen) % 2 else Point(-92.5, 38.9).buffer(0.05)

    monkeypatch.setattr(cities, "OSM_MAX_CONSECUTIVE_FALLBACKS", 3)
    monkeypatch.setattr(cities, "fetch_osm_city_polygon", sometimes)
    monkeypatch.setattr(cities, "cache_osm_polygon", lambda *a, **k: None)
    monkeypatch.setattr(cities, "get_cached_osm_polygon", lambda *a, **k: geocoding.NOT_CACHED)

    points, _ = cities.load_city_points(spreadsheet)
    boundaries = cities.load_city_boundaries(points)
    assert len(seen) == 4  # every city processed, nothing aborted
    assert boundaries.geometry.notna().sum() == 2


def test_nominatim_answers_first_and_overpass_is_never_called(monkeypatch):
    _nominatim_returns(monkeypatch, BOUNDARY)

    def unexpected(*args, **kwargs):
        raise AssertionError("Overpass must not be called when Nominatim answered")

    monkeypatch.setattr(geocoding.ox, "features_from_point", unexpected)
    assert _fetch().symmetric_difference(BOUNDARY).area < 1e-9


def test_a_non_polygon_answer_falls_through_to_overpass(monkeypatch):
    # osmnx raises TypeError when the best match carries no boundary geometry
    def no_polygon(*args, **kwargs):
        raise TypeError("Nominatim did not geocode query to a geometry of type (Multi)Polygon.")

    monkeypatch.setattr(geocoding.ox, "geocode_to_gdf", no_polygon)
    frame = gpd.GeoDataFrame({"name": ["Nowhere"]}, geometry=[BOUNDARY], crs=4326)
    monkeypatch.setattr(geocoding.ox, "features_from_point", lambda *a, **k: frame)
    assert _fetch().symmetric_difference(BOUNDARY).area < 1e-9


def test_a_county_match_is_rejected(monkeypatch):
    # "Clinton, Missouri" matches Clinton County ahead of the city of Clinton
    _nominatim_returns(monkeypatch, Point(-92.5, 38.9).buffer(3.0), "county", "Nowhere County")
    frame = gpd.GeoDataFrame({"name": ["Nowhere"]}, geometry=[BOUNDARY], crs=4326)
    monkeypatch.setattr(geocoding.ox, "features_from_point", lambda *a, **k: frame)
    # The oversized shape is discarded and Overpass supplies the answer instead
    assert _fetch().symmetric_difference(BOUNDARY).area < 1e-9


def test_a_failed_nominatim_lookup_stops_the_run(monkeypatch):
    # Not silently stampeding all 61 cities onto the heavy Overpass path
    def boom(*args, **kwargs):
        raise requests.ConnectionError("Connection refused")

    monkeypatch.setattr(geocoding.ox, "geocode_to_gdf", boom)
    with pytest.raises(RuntimeError, match="Nominatim lookup for Nowhere, Missouri, USA failed"):
        _fetch()


def test_a_municipality_is_accepted_by_place_rank_when_untyped(monkeypatch):
    # Older Nominatim responses carry no addresstype, only a rank
    frame = gpd.GeoDataFrame({"place_rank": [16]}, geometry=[BOUNDARY], crs=4326)
    monkeypatch.setattr(geocoding.ox, "geocode_to_gdf", lambda *a, **k: frame)
    assert _fetch().symmetric_difference(BOUNDARY).area < 1e-9


def test_an_overpass_outage_stops_the_run(monkeypatch):
    # Swallowed, the outage became None, and the caller cached None as "this town
    # has no boundary" -- a dot forever, from one bad minute of network
    def boom(*args, **kwargs):
        raise requests.ConnectionError("Connection refused")

    monkeypatch.setattr(geocoding.ox, "features_from_point", boom)
    with pytest.raises(RuntimeError, match="Overpass request for Nowhere, Missouri failed"):
        _fetch()


def test_the_county_disambiguates_a_shared_name(monkeypatch):
    # "Clark, Missouri" matches Clark County, but the spreadsheet knows it is the
    # village in Randolph County, and that query returns the right thing
    asked = []

    def by_query(query, **kwargs):
        asked.append(query)
        if "Randolph County" in query:
            return gpd.GeoDataFrame(
                {"name": ["Clark"], "addresstype": ["village"]}, geometry=[BOUNDARY], crs=4326
            )
        return gpd.GeoDataFrame(
            {"name": ["Clark County"], "addresstype": ["county"]},
            geometry=[Point(-92.5, 38.9).buffer(3.0)],
            crs=4326,
        )

    monkeypatch.setattr(geocoding.ox, "geocode_to_gdf", by_query)
    got = geocoding.fetch_osm_city_polygon("Clark", "Missouri", POINT, county="Randolph")
    assert got.symmetric_difference(BOUNDARY).area < 1e-9
    assert asked == ["Clark, Randolph County, Missouri, USA"]


def test_a_wrong_county_falls_back_to_the_plain_query(monkeypatch):
    asked = []

    def by_query(query, **kwargs):
        asked.append(query)
        if "County" in query:
            raise InsufficientResponseError("Nominatim geocoder returned 0 results")
        return gpd.GeoDataFrame(
            {"name": ["Nowhere"], "addresstype": ["town"]}, geometry=[BOUNDARY], crs=4326
        )

    monkeypatch.setattr(geocoding.ox, "geocode_to_gdf", by_query)
    got = geocoding.fetch_osm_city_polygon("Nowhere", "Missouri", POINT, county="Nonesuch")
    assert got.symmetric_difference(BOUNDARY).area < 1e-9
    assert asked == ["Nowhere, Nonesuch County, Missouri, USA", "Nowhere, Missouri, USA"]


def test_a_missing_boundary_is_left_empty(monkeypatch, spreadsheet, seeded_cache):
    # No stand-in circle: the map keeps that town as a dot, so a null geometry is
    # the honest answer
    from maybemap.sources import cities

    monkeypatch.setattr(cities, "get_cached_osm_polygon", lambda *a, **k: geocoding.NOT_CACHED)
    monkeypatch.setattr(cities, "fetch_osm_city_polygon", lambda *a, **k: None)
    monkeypatch.setattr(cities, "cache_osm_polygon", lambda *a, **k: None)
    monkeypatch.setattr(cities, "OSM_MAX_CONSECUTIVE_FALLBACKS", 99)
    points, _ = cities.load_city_points(spreadsheet)

    assert cities.load_city_boundaries(points).geometry.isna().all()


def test_a_geocode_landing_in_another_state_is_thrown_away(monkeypatch):
    # "Lamonte, Missouri" is a misspelling of La Monte, and Geoapify answers it with
    # a street in Houston. Mapped, that one point drags its region's enclosing
    # circle across half the country.
    monkeypatch.setattr(
        geocoding,
        "geoapify_forward",
        lambda text, **kw: {"lon": -95.46, "lat": 29.82, "state_code": "TX"},
    )
    monkeypatch.setattr(geocoding, "_save_geocode_cache", lambda: None)
    monkeypatch.setattr(geocoding, "GEOCODE_CACHE", {})

    assert geocoding.geocode_city_state("Lamonte", "MO") is None


def test_a_geocode_in_the_expected_state_is_kept(monkeypatch):
    monkeypatch.setattr(
        geocoding,
        "geoapify_forward",
        lambda text, **kw: {"lon": -93.43, "lat": 38.77, "state_code": "MO"},
    )
    monkeypatch.setattr(geocoding, "_save_geocode_cache", lambda: None)
    monkeypatch.setattr(geocoding, "GEOCODE_CACHE", {})

    # The spreadsheet may spell the state either way; both must match "MO"
    assert geocoding.geocode_city_state("La Monte", "MO")["lat"] == 38.77
    monkeypatch.setattr(geocoding, "GEOCODE_CACHE", {})
    assert geocoding.geocode_city_state("La Monte", "Missouri")["lat"] == 38.77


class _Response:
    def __init__(self, status, payload=None):
        self.status_code = status
        self._payload = payload or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


@pytest.mark.parametrize(
    "answer",
    [requests.ConnectionError("Connection refused"), _Response(429), _Response(503)],
    ids=["network", "rate-limited", "server-error"],
)
def test_a_failed_geocode_raises_and_is_never_cached(monkeypatch, answer):
    # A town the geocoder could not be asked about is not a town it could not find
    def get(*args, **kwargs):
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setenv("GEOAPIFY_KEY", "test")
    monkeypatch.setattr(geocoding.SESSION, "get", get)
    monkeypatch.setattr(geocoding, "_save_geocode_cache", lambda: None)
    with pytest.raises(RuntimeError, match="Geoapify"):
        geocoding.geocode_city_state("Nowhere", "MO")
    assert "Nowhere, Missouri" not in geocoding.GEOCODE_CACHE


def test_an_empty_geocode_answer_is_cached_as_not_found(monkeypatch):
    monkeypatch.setenv("GEOAPIFY_KEY", "test")
    monkeypatch.setattr(geocoding.SESSION, "get", lambda *a, **k: _Response(200, {"features": []}))
    monkeypatch.setattr(geocoding, "_save_geocode_cache", lambda: None)
    monkeypatch.delitem(geocoding.GEOCODE_CACHE, "Nowhere, Missouri", raising=False)
    assert geocoding.geocode_city_state("Nowhere", "MO") is None
    assert geocoding.GEOCODE_CACHE["Nowhere, Missouri"] is None
    del geocoding.GEOCODE_CACHE["Nowhere, Missouri"]


def test_the_cache_tells_unknown_from_known_empty(monkeypatch):
    monkeypatch.setitem(geocoding.BOUNDARY_WKT, "NOWHERE, MISSOURI", None)
    assert geocoding.get_cached_osm_polygon("Nowhere", "MO") is None
    assert geocoding.get_cached_osm_polygon("Elsewhere", "MO") is geocoding.NOT_CACHED


def test_a_known_missing_boundary_is_not_looked_up_again(monkeypatch, spreadsheet, seeded_cache):
    # Every town's answer is already "no boundary": the build must neither re-ask
    # Nominatim and Overpass for them nor read the run of known misses as an outage
    from maybemap.sources import cities

    asked = []
    monkeypatch.setattr(cities, "OSM_MAX_CONSECUTIVE_FALLBACKS", 2)
    monkeypatch.setattr(cities, "get_cached_osm_polygon", lambda *a, **k: None)
    monkeypatch.setattr(cities, "fetch_osm_city_polygon", lambda *a, **k: asked.append(a))
    points, _ = cities.load_city_points(spreadsheet)

    boundaries = cities.load_city_boundaries(points)
    assert asked == []
    assert boundaries.geometry.isna().all()
