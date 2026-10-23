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


@pytest.fixture
def rosters(tmp_path, monkeypatch):
    # Joplin is in the export with no region; Seneca is not in the export at all
    monkeypatch.setitem(geocoding.GEOCODE_CACHE, "Seneca, Missouri", {"lat": 36.84, "lon": -94.61})
    techs = tmp_path / "techs.csv"
    techs.write_text(
        "Tech,Town,State\nAda,Columbia,MO\nBo,Joplin,MO\nCy,Seneca,MO\nDee,Columbia,MO\n"
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
    assert places["Columbia"]["techs"] == ["Ada", "Dee"]
    assert places["Seneca"]["city"] is None
    assert places["Joplin"]["city"] is not None


def test_matrix_matches_the_place_count(built):
    n = len(built["dispatch"]["places"])
    assert len(built["dispatch"]["meters"]) == n
    assert all(len(row) == n for row in built["dispatch"]["seconds"])


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
        )


def test_customer_buckets_cover_zero(built):
    # A town listed with no customers must land in a bucket, or no filter can hide it
    first = built["buckets"][0]
    assert first["lo"] == 0 and first["label"] == "0-5"
