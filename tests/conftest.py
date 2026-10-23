# Shared fixtures: a generated spreadsheet and a pre-seeded geocode cache, so the
# tests need neither a Geoapify key nor network access.

import pandas as pd
import pytest

from maybemap import geocoding
from maybemap.states import state_full

# City, State, Group, Customers, Service Mix, lat, lon. The last entry sits well
# outside Missouri, and between them the counts span every filter bucket including
# the open-ended top one.
FIXTURE_CITIES = [
    ("Kansas City", "MO", "Group 1: West", 8, "Mixed", 39.0997, -94.5786),
    ("Columbia", "MO", "Group 2: Central", 25, "Mixed", 38.9517, -92.3341),
    ("Saint Louis", "MO", "Group 3: East", 3, "Business Only", 38.6270, -90.1994),
    ("Chicago", "IL", "Group 4: Illinois", 1, "Residential Only", 41.8781, -87.6298),
]

# Same shape, but with no region -- what most of the CSV export looks like
FIXTURE_UNASSIGNED = [
    ("Springfield", "MO", "", 2, "Business Only", 37.2090, -93.2923),
    ("Joplin", "MO", "", 1, "Residential Only", 37.0842, -94.5133),
    ("Cape Girardeau", "MO", "", 12, "Mixed", 37.3059, -89.5181),
]


def _customer_split(count, mix):
    # Residential / business halves that add back up to the town's customer count.
    # Tasks 6 Mo is deliberately left out of the fixture: the export can carry some
    # of the optional columns and not others.
    if mix == "Business Only":
        return 0, count
    if mix == "Residential Only":
        return count, 0
    return count - count // 2, count // 2


@pytest.fixture
def spreadsheet(tmp_path):
    # The older export: a customer count and nothing else. It is the coverage for
    # every optional column being absent.
    path = tmp_path / "cities.xlsx"
    pd.DataFrame(
        [(c, s, g, n) for c, s, g, n, _, _, _ in FIXTURE_CITIES],
        columns=["City", "State", "Group", "Customers"],
    ).to_excel(path, index=False, engine="openpyxl")
    return path


@pytest.fixture
def city_csv(tmp_path):
    # Shaped like the real MBS export: caps names, a Region column rather than
    # Group, towns with no region at all, and the per-town service detail.
    path = tmp_path / "cities.csv"
    pd.DataFrame(
        [
            (c.upper(), "SOME", s, g, n, n, mix, *_customer_split(n, mix))
            for c, s, g, n, mix, _, _ in [*FIXTURE_CITIES, *FIXTURE_UNASSIGNED]
        ],
        columns=[
            "City",
            "County",
            "State",
            "Region",
            "Customers",
            "Active Addresses",
            "Service Mix",
            "Residential Customers",
            "Business Customers",
        ],
    ).to_csv(path, index=False)
    return path


@pytest.fixture
def seeded_cache(monkeypatch):
    # Keyed the way geocoding keys it -- on the spelled-out state name
    for city, state, _, _, _, lat, lon in [*FIXTURE_CITIES, *FIXTURE_UNASSIGNED]:
        key = f"{city}, {state_full(state)}"
        monkeypatch.setitem(geocoding.GEOCODE_CACHE, key, {"lat": lat, "lon": lon})


@pytest.fixture
def stub_city_boundaries(monkeypatch):
    # Keeps the boundary tests off the network and out of the real boundary cache,
    # which happens to hold the fixture cities and would otherwise hide a regression.
    from shapely.geometry import Point

    from maybemap.sources import cities

    fetched = []

    def fetch(city, state, pt, progress="", county=""):
        fetched.append(city)
        return Point(-92.5, 38.9).buffer(0.05)

    monkeypatch.setattr(cities, "get_cached_osm_polygon", lambda *a, **k: None)
    monkeypatch.setattr(cities, "cache_osm_polygon", lambda *a, **k: None)
    monkeypatch.setattr(cities, "fetch_osm_city_polygon", fetch)
    return fetched
