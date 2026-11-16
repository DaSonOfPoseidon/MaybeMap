# The payload the page draws from, built end to end against the fixture export and
# the generated KMZ, with geocoding, boundaries and OSRM all stubbed.

import json

import pytest

from maybemap import config, geocoding, payload
from maybemap.config import UNASSIGNED_GROUP_LABEL


def fake_matrix(points):
    n = len(points)
    meters = [[0 if i == j else 1000 * (i + j) for j in range(n)] for i in range(n)]
    return meters, [[m // 10 for m in row] for row in meters]


TABLE_CALLS = []


def fake_tech_table(origins, destinations):
    # Each cell encodes its origin and destination, so a swapped axis shows up
    TABLE_CALLS.append((list(origins), list(destinations)))
    meters = [[1000 * (i + 1) + j for j in range(len(destinations))] for i in range(len(origins))]
    return meters, [[m // 10 for m in row] for row in meters]


@pytest.fixture(autouse=True)
def _clear_table_calls():
    TABLE_CALLS.clear()


@pytest.fixture
def rosters(tmp_path, monkeypatch):
    # Joplin is in the export with no region; Seneca is not in the export at all
    monkeypatch.setitem(geocoding.GEOCODE_CACHE, "Seneca, Missouri", {"lat": 36.84, "lon": -94.61})
    techs = tmp_path / "techs.csv"
    # Eve has no town, only a region; Fay has neither and cannot be placed
    techs.write_text(
        "Tech,Town,State,Primary Region,Departments\n"
        'Ada,Columbia,MO,Group 2: Central,"FIELD, TC"\n'
        "Bo,Joplin,MO,,FIELD\n"
        "Cy,Seneca,MO,,FIELDBUS\n"
        "Dee,Columbia,MO,,FIELD\n"
        "Eve,,,Group 2: Central,TC\n"
        "Fay,,,,FIELD\n"
    )
    staging = tmp_path / "staging.csv"
    staging.write_text("Type,Town,State\nGenerator,Columbia,MO\nGenerator,Seneca,MO\n")
    return techs, staging


@pytest.fixture
def built(city_csv, seeded_cache, stub_city_boundaries, rosters, kmz_path):
    techs, staging = rosters
    return payload.build(
        city_path=city_csv,
        kmz_path=kmz_path,
        techs_path=techs,
        staging_path=staging,
        matrix=fake_matrix,
        tech_table=fake_tech_table,
    )


def test_zones_carry_their_kml_colors(built):
    props = {f["properties"]["label"]: f["properties"] for f in built["zones"]["features"]}
    assert props["MO West"]["color"] == "#0070ff"
    assert props["MO West"]["edge"] != "#0070ff"


def test_every_town_gets_a_dot_and_a_zone_when_inside_one(built):
    by_name = {c["name"]: c for c in built["cities"]}
    assert len(by_name) == 7
    assert by_name["Columbia"]["zone"] == "MO Central"
    # Outside every zone is None, not NaN -- NaN would break JSON.parse in the page
    assert by_name["Chicago"]["zone"] is None
    assert by_name["Springfield"]["color"] == config.UNASSIGNED_GROUP_COLOR


def test_only_regioned_towns_get_outlines(built, stub_city_boundaries):
    assert "Springfield" not in stub_city_boundaries
    assert len(built["outlines"]["features"]) == 4


def test_dispatch_set_is_regioned_towns_plus_tech_towns(built):
    places = {p["name"]: p for p in built["dispatch"]["places"]}
    regioned = {c["name"] for c in built["cities"] if c["region"] != UNASSIGNED_GROUP_LABEL}
    assert set(places) == regioned | {"Joplin", "Seneca"}
    techs = built["dispatch"]["techs"]
    assert [techs[t]["name"] for t in places["Columbia"]["techs"]] == ["Ada", "Dee", "Eve"]
    assert places["Seneca"]["city"] is None
    assert places["Joplin"]["city"] is not None


def test_matrix_matches_the_place_count(built):
    n = len(built["dispatch"]["places"])
    assert len(built["dispatch"]["meters"]) == n
    assert all(len(row) == n for row in built["dispatch"]["seconds"])


def test_tech_places_are_the_places_that_hold_a_tech(built):
    dispatch = built["dispatch"]
    held = [i for i, p in enumerate(dispatch["places"]) if p["techs"]]
    assert dispatch["techPlaces"] == held
    names = {dispatch["places"][i]["name"] for i in held}
    # Eve is placed at the Columbia hub, so it counts once; Fay is unplaced
    assert names == {"Columbia", "Joplin", "Seneca"}


def test_to_city_covers_exactly_the_cities_outside_the_dispatch_set(built):
    dispatch, cities = built["dispatch"], built["cities"]
    in_dispatch = {p["city"] for p in dispatch["places"] if p["city"] is not None}
    outside = [c["id"] for c in cities if c["id"] not in in_dispatch]
    to_city = dispatch["toCity"]
    assert to_city["ids"] == outside
    names = {cities[i]["name"] for i in outside}
    assert names == {"Springfield", "Cape Girardeau"}
    rows = len(dispatch["techPlaces"])
    for key in ("meters", "seconds"):
        assert len(to_city[key]) == rows
        assert all(len(row) == len(outside) for row in to_city[key])


def test_the_tech_table_is_asked_for_tech_places_then_outside_cities(built):
    ((origins, destinations),) = TABLE_CALLS
    dispatch, cities = built["dispatch"], built["cities"]
    places = dispatch["places"]
    assert origins == [(places[i]["lat"], places[i]["lon"]) for i in dispatch["techPlaces"]]
    assert destinations == [(cities[i]["lat"], cities[i]["lon"]) for i in dispatch["toCity"]["ids"]]
    # Rows are tech places and columns are cities, as the stub filled them in
    assert dispatch["toCity"]["meters"][1][0] == 2000


def test_no_techs_leaves_empty_tech_places_and_rows(
    city_csv, seeded_cache, stub_city_boundaries, rosters, kmz_path
):
    techs, staging = rosters
    techs.write_text("Tech,Town,State,Primary Region,Departments\n")
    staging.write_text("Type,Town,State\n")
    out = payload.build(
        city_path=city_csv,
        kmz_path=kmz_path,
        techs_path=techs,
        staging_path=staging,
        matrix=fake_matrix,
        tech_table=fake_tech_table,
    )
    assert out["dispatch"]["techPlaces"] == []
    assert out["dispatch"]["toCity"]["meters"] == []
    json.dumps(out, allow_nan=False)


def test_staging_points_at_dispatch_places(built):
    names = [built["dispatch"]["places"][i]["name"] for i in built["staging"][0]["places"]]
    assert built["staging"][0]["type"] == "Generator"
    assert names == ["Columbia", "Seneca"]


def test_payload_is_strict_json(built):
    json.dumps(built, allow_nan=False)


def test_staging_at_an_unknown_town_raises(
    city_csv, seeded_cache, stub_city_boundaries, rosters, kmz_path
):
    techs, staging = rosters
    staging.write_text("Type,Town,State\nOutage,Nowhere,MO\n")
    with pytest.raises(ValueError, match="Nowhere, MO is not a dispatch place"):
        payload.build(
            city_path=city_csv,
            kmz_path=kmz_path,
            techs_path=techs,
            staging_path=staging,
            matrix=fake_matrix,
            tech_table=fake_tech_table,
        )


def test_customer_buckets_cover_zero(built):
    # A town listed with no customers must land in a bucket, or no filter can hide it
    first = built["buckets"][0]
    assert first["lo"] == 0 and first["label"] == "0-5"


def test_techs_carry_their_basis_and_departments(built):
    techs = {t["name"]: t for t in built["dispatch"]["techs"]}
    assert techs["Ada"]["basis"] == "town" and techs["Ada"]["departments"] == ["FIELD", "TC"]
    # No town: placed at the hub of their primary region, and marked as such
    assert techs["Eve"]["basis"] == "region"
    assert built["dispatch"]["places"][techs["Eve"]["place"]]["name"] == "Columbia"
    assert "Fay" not in techs
    assert built["dispatch"]["unplaced"] == ["Fay"]
    assert built["departments"] == ["FIELD", "FIELDBUS", "TC"]


def test_a_region_hub_is_its_most_central_town():
    towns = [
        {
            "id": 0,
            "name": "West End",
            "state": "MO",
            "region": "R",
            "regioned": True,
            "lat": 38.0,
            "lon": -94.0,
        },
        {
            "id": 1,
            "name": "Middle",
            "state": "MO",
            "region": "R",
            "regioned": True,
            "lat": 38.0,
            "lon": -93.0,
        },
        {
            "id": 2,
            "name": "East End",
            "state": "MO",
            "region": "R",
            "regioned": True,
            "lat": 38.0,
            "lon": -92.0,
        },
        {
            "id": 3,
            "name": "Elsewhere",
            "state": "MO",
            "region": "S",
            "regioned": True,
            "lat": 40.0,
            "lon": -90.0,
        },
    ]
    assert payload.region_hubs(towns) == {"R": ("Middle", "MO"), "S": ("Elsewhere", "MO")}
