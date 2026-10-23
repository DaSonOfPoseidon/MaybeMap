# Shared fixtures: a generated spreadsheet and a pre-seeded geocode cache, so the
# overlay tests need neither a Geoapify key nor network access.

import pandas as pd
import pytest

from maybemap import config, geocoding
from maybemap.sources.kmz import load_kmz_zones
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
def zones():
    return load_kmz_zones(config.KMZ_PATH)


@pytest.fixture(autouse=True)
def no_boundary_lookups(monkeypatch):
    # Overlay tests must never touch the boundary cache or the network. Opt in with
    # the city_boundaries fixture below when the shapes are what is under test.
    from maybemap import overlay

    monkeypatch.setattr(overlay, "_city_boundaries", lambda tagged: _empty_boundaries())


def _empty_boundaries():
    import geopandas as gpd

    return gpd.GeoDataFrame(geometry=[], crs=3857)


@pytest.fixture
def city_boundaries(monkeypatch):
    # A square boundary around each city that has one. Kansas City and Saint Louis
    # get shapes; Columbia deliberately does not, so it stays a dot.
    import geopandas as gpd

    from maybemap import overlay

    def fake(tagged):
        rows = tagged[tagged["City"].isin({"Kansas City", "Saint Louis"})]
        if rows.empty:
            return _empty_boundaries()
        return gpd.GeoDataFrame(
            {
                "City": list(rows["City"]),
                "State": list(rows["State"]),
                "Group": list(rows["Group"]),
                "Zone": list(rows["Zone"]),
                "Color": list(rows["Color"]),
            },
            geometry=[g.buffer(20000) for g in rows.geometry],
            crs=3857,
        )

    monkeypatch.setattr(overlay, "_city_boundaries", fake)


@pytest.fixture
def stub_city_boundaries(monkeypatch):
    # Keeps the region-map tests off the network and out of the real boundary cache,
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


# City, Service, Street Address, Person Assigned, Task Status, Task ID, Subscriber,
# Region, Due Time, County. The Service column carries the full address for some
# rows and a phone number for others, which is what the ZIP is read out of.
FIXTURE_TASKS = [
    (
        "COLUMBIA",
        "507 Kentucky AVE COLUMBIA, MO  65201",
        "507 Kentucky AVE",
        "Quisenberry, Jim (jiqu0170)",
        "Active",
        "111111",
        "Arena Liquor",
        "Columbia Core",
        "7/23/2026 0:00",
        "BOONE",
    ),
    (
        "COLUMBIA",
        "    (573) 000-0000",
        "600 E Stadium",
        "",
        "Waiting",
        "222222",
        "Stadium Cafe",
        "Columbia Core",
        "8/25/2026 0:00",
        "BOONE",
    ),
    (
        "JEFFERSON CITY",
        "124 E DUNKLIN ST JEFFERSON CITY, MO  651011234",
        "124 E DUNKLIN ST",
        "Kelly, Nick (nike0170)",
        "Hold",
        "333333",
        "Salon Va Lise",
        "Jefferson City Area",
        "8/13/2026 0:00",
        "COLE",
    ),
    # Both of these fail to geocode, fall back to the same city center, and so are
    # the coverage for coincident pins being pushed apart
    (
        "KANSAS CITY",
        "    (816) 000-0000",
        "1 Nowhere LN",
        "",
        "Incomplete",
        "444444",
        "Ghost Property",
        "",
        "8/26/2026 0:00",
        "JACKSON",
    ),
    (
        "KANSAS CITY",
        "    (816) 000-0001",
        "2 Nowhere LN",
        "",
        "Active",
        "555555",
        "Second Ghost",
        "",
        "8/26/2026 0:00",
        "JACKSON",
    ),
    # Geocodes to a real Missouri point on the far side of the state, which is what
    # a highway address matching the wrong highway looks like
    (
        "SAINT LOUIS",
        "    (314) 000-0000",
        "65 S Highway 47",
        "",
        "Waiting",
        "666666",
        "Wrong Highway",
        "St Louis West",
        "8/26/2026 0:00",
        "SAINT LOUIS",
    ),
]

TASK_COLUMNS_OUT = [
    "City",
    "Service",
    "Street Address",
    "Person Assigned",
    "Task Status",
    "Task ID",
    "Subscriber Name",
    "Region",
    "Due Time",
    "County",
]

# Address query -> coordinates, keyed the way the loader builds the query. None is a
# cached failure, which is what sends a task to its city center.
FIXTURE_ADDRESSES = {
    "507 Kentucky AVE, Columbia, MO 65201": (38.9430, -92.3230),
    "600 E Stadium, Columbia, MO": (38.9370, -92.3290),
    "124 E DUNKLIN ST, Jefferson City, MO 65101": (38.5710, -92.1730),
    "1 Nowhere LN, Kansas City, MO": None,
    "2 Nowhere LN, Kansas City, MO": None,
    # Kansas City, four hundred kilometres from the town the task names
    "65 S Highway 47, Saint Louis, MO": (39.0997, -94.5786),
}


@pytest.fixture
def tasks_csv(tmp_path):
    path = tmp_path / "tasks.csv"
    pd.DataFrame(FIXTURE_TASKS, columns=TASK_COLUMNS_OUT).to_csv(path, index=False)
    return path


@pytest.fixture
def seeded_address_cache(monkeypatch):
    # Keeps the task tests off Geoapify: every query the fixture produces is already
    # answered, including the two that are answered with a failure.
    for query, coords in FIXTURE_ADDRESSES.items():
        value = None if coords is None else {"lat": coords[0], "lon": coords[1]}
        monkeypatch.setitem(geocoding.ADDRESS_CACHE, query, value)
