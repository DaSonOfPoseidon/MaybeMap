# The payload the page draws from, built end to end against the fixture export and
# the generated KMZ, with geocoding, boundaries and OSRM all stubbed.

import json

import pandas as pd
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
def hubs(tmp_path):
    # No overrides by default: the path points at a file that does not exist
    return tmp_path / "region_hubs.csv"


@pytest.fixture
def built(city_csv, seeded_cache, stub_city_boundaries, rosters, hubs, kmz_path):
    techs, staging = rosters
    return payload.build(
        city_path=city_csv,
        kmz_path=kmz_path,
        techs_path=techs,
        staging_path=staging,
        hubs_path=hubs,
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
    city_csv, seeded_cache, stub_city_boundaries, rosters, hubs, kmz_path
):
    techs, staging = rosters
    techs.write_text("Tech,Town,State,Primary Region,Departments\n")
    staging.write_text("Type,Town,State\n")
    out = payload.build(
        city_path=city_csv,
        kmz_path=kmz_path,
        techs_path=techs,
        staging_path=staging,
        hubs_path=hubs,
        matrix=fake_matrix,
        tech_table=fake_tech_table,
    )
    assert out["dispatch"]["techPlaces"] == []
    assert out["dispatch"]["toCity"]["meters"] == []
    json.dumps(out, allow_nan=False)


def test_the_region_hubs_file_overrides_the_default_hub(
    city_csv, seeded_cache, stub_city_boundaries, rosters, hubs, kmz_path
):
    techs, staging = rosters
    hubs.write_text("Region,Hub,State\nGroup 2: Central,Kansas City,MO\n")
    out = payload.build(
        city_path=city_csv,
        kmz_path=kmz_path,
        techs_path=techs,
        staging_path=staging,
        hubs_path=hubs,
        matrix=fake_matrix,
        tech_table=fake_tech_table,
    )
    eve = next(t for t in out["dispatch"]["techs"] if t["name"] == "Eve")
    assert out["dispatch"]["places"][eve["place"]]["name"] == "Kansas City"


def test_staging_points_at_dispatch_places(built):
    names = [built["dispatch"]["places"][i]["name"] for i in built["staging"][0]["places"]]
    assert built["staging"][0]["type"] == "Generator"
    assert names == ["Columbia", "Seneca"]


def test_payload_is_strict_json(built):
    json.dumps(built, allow_nan=False)


def test_staging_at_an_unknown_town_raises(
    city_csv, seeded_cache, stub_city_boundaries, rosters, hubs, kmz_path
):
    techs, staging = rosters
    staging.write_text("Type,Town,State\nOutage,Nowhere,MO\n")
    with pytest.raises(ValueError, match="Nowhere, MO is not a dispatch place"):
        payload.build(
            city_path=city_csv,
            kmz_path=kmz_path,
            techs_path=techs,
            staging_path=staging,
            hubs_path=hubs,
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


def _town(i, name, region, customers, lat=38.0, lon=-93.0):
    return {
        "id": i,
        "name": name,
        "state": "MO",
        "region": region,
        "regioned": True,
        "customers": customers,
        "lat": lat,
        "lon": lon,
    }


def _techs(*rows):
    # (tech, town, state, region): the columns build_places reads
    return pd.DataFrame(
        [
            {
                "Tech": t,
                "Town": town,
                "State": state,
                "TechId": "",
                "Region": region,
                "Departments": [],
            }
            for t, town, state, region in rows
        ],
        columns=["Tech", "Town", "State", "TechId", "Region", "Departments"],
    )


def _placed_at(roster, i=0):
    return roster["places"][roster["techs"][i]["place"]]["name"]


def test_a_region_hub_is_its_town_with_the_most_customers():
    # Zz Middle is the most central by distance, but Zz East carries the customers
    towns = [
        _town(0, "Zz West", "R", 10, lon=-94.0),
        _town(1, "Zz Middle", "R", 20, lon=-93.0),
        _town(2, "Zz East", "R", 90, lon=-92.0),
        _town(3, "Zz Elsewhere", "S", 5, lat=40.0, lon=-90.0),
    ]
    found = payload.region_hubs(towns)
    assert found.hubs == {"R": ("Zz East", "MO"), "S": ("Zz Elsewhere", "MO")}
    assert found.defaulted == {"R", "S"} and found.unresolved == set()


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_a_tie_goes_to_the_town_named_in_the_region(order):
    # Zz Alpha sorts first, but the region is named for Zz Beta
    region = "Zz Beta-Zz Gamma"
    both = [_town(0, "Zz Alpha", region, 40), _town(1, "Zz Beta", region, 40)]
    hubs = payload.region_hubs([both[i] for i in order]).hubs
    assert hubs == {region: ("Zz Beta", "MO")}


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_a_tie_with_no_name_in_the_region_goes_to_the_first_name(order):
    both = [_town(0, "Zz Alpha", "Zz Region", 40), _town(1, "Zz Beta", "Zz Region", 40)]
    hubs = payload.region_hubs([both[i] for i in order]).hubs
    assert hubs == {"Zz Region": ("Zz Alpha", "MO")}


def test_a_town_name_inside_a_longer_word_is_not_named_in_the_region():
    # "Zzol" is a substring of "Zzolive" but not a word of it, so the tie goes to the
    # first name (Zz Alpha sorts before Zzol)
    region = "Zzolive-Zzlebanon"
    towns = [_town(0, "Zzol", region, 40), _town(1, "Zz Alpha", region, 40)]
    assert payload.region_hubs(towns).hubs == {region: ("Zz Alpha", "MO")}
    # The same town, spelled as a whole word of the region, does win the tie
    region = "Zzol-Zzlebanon"
    towns = [_town(0, "Zzol", region, 40), _town(1, "Zz Alpha", region, 40)]
    assert payload.region_hubs(towns).hubs == {region: ("Zzol", "MO")}


@pytest.mark.parametrize(
    "name, region",
    [
        ("Zz Lake Ozark", "Zz Lake Ozark Area"),  # a multi-word town, at the start
        ("Zz Lake Ozark", "Region of zz lake ozark"),  # at the end, other case
        ("Zz St. Paul", "Zz St. Paul-Other"),  # regex metacharacters in the name
    ],
)
def test_a_whole_word_town_name_in_the_region_breaks_the_tie(name, region):
    # Zz Aaa sorts first, so only a name match can give the tie to the other town
    towns = [_town(0, "Zz Aaa", region, 40), _town(1, name, region, 40)]
    assert payload.region_hubs(towns).hubs == {region: (name, "MO")}


def test_a_dot_in_a_town_name_is_not_a_wildcard():
    # "Zz St. Paul" as a pattern would also match "Zz St, Paul" if the dot were unescaped
    region = "Zz St, Paul"
    towns = [_town(0, "Zz Aaa", region, 40), _town(1, "Zz St. Paul", region, 40)]
    assert payload.region_hubs(towns).hubs == {region: ("Zz Aaa", "MO")}


def test_the_name_in_the_region_only_breaks_a_tie():
    # Zz Beta is named in the region, but Zz Alpha has more customers
    region = "Zz Beta-Zz Gamma"
    towns = [_town(0, "Zz Alpha", region, 41), _town(1, "Zz Beta", region, 40)]
    assert payload.region_hubs(towns).hubs == {region: ("Zz Alpha", "MO")}


def test_an_override_wins_over_the_counts():
    towns = [_town(0, "Zz Big", "R", 90), _town(1, "Zz Small", "R", 1)]
    found = payload.region_hubs(towns, {"R": ("Zz Small", "MO")})
    assert found.hubs == {"R": ("Zz Small", "MO")}
    assert found.defaulted == set()


def test_an_override_for_an_unknown_region_stops_the_build_and_names_it():
    towns = [_town(0, "Zz Big", "Zz R", 90), _town(1, "Zz Small", "Zz R", 1)]
    overrides = {"Zz Typo": ("Zz Big", "MO"), "Zz Other Typo": ("Zz Small", "MO")}
    with pytest.raises(ValueError, match=r"region_hubs\.csv") as err:
        payload.region_hubs(towns, overrides)
    # Every unknown region is named, not only the first
    assert "Zz Typo" in str(err.value) and "Zz Other Typo" in str(err.value)
    with pytest.raises(ValueError, match="Zz Typo"):
        payload.build_places(towns, _techs(("Zz Ada", "Zz Big", "MO", "Zz R")), overrides)


def test_an_override_region_is_compared_exactly_like_every_other_region():
    # Regions are matched as written (the files strip the edges, nothing else), so a
    # different case is a different region and is reported
    towns = [_town(0, "Zz Big", "Zz R", 90), _town(1, "Zz Small", "Zz R", 1)]
    with pytest.raises(ValueError, match="zz r"):
        payload.region_hubs(towns, {"zz r": ("Zz Small", "MO")})


def test_an_override_for_a_region_with_only_unregioned_towns_is_unknown():
    town = _town(0, "Zz Loose", "Unassigned", 9)
    town["regioned"] = False
    with pytest.raises(ValueError, match="Unassigned"):
        payload.region_hubs([town], {"Unassigned": ("Zz Loose", "MO")})


def test_an_override_resolves_a_region_with_no_counts():
    towns = [_town(0, "Zz One", "R", None), _town(1, "Zz Two", "R", None)]
    found = payload.region_hubs(towns, {"R": ("Zz Two", "MO")})
    assert found.hubs == {"R": ("Zz Two", "MO")} and found.unresolved == set()


def test_a_town_with_no_count_is_skipped_but_zero_is_a_count():
    towns = [
        _town(0, "Zz Aaa", "R", None),
        _town(1, "Zz Bbb", "R", 0),
        _town(2, "Zz Ccc", "S", None),
    ]
    found = payload.region_hubs(towns)
    assert found.hubs == {"R": ("Zz Bbb", "MO")}
    assert found.unresolved == {"S"}


def test_a_single_town_region_is_its_own_hub():
    assert payload.region_hubs([_town(0, "Zz Only", "R", 3)]).hubs == {"R": ("Zz Only", "MO")}


def test_towns_with_no_region_are_never_hubs():
    town = _town(0, "Zz Loose", "Unassigned", 99)
    town["regioned"] = False
    found = payload.region_hubs([town])
    assert found.hubs == {} and found.unresolved == set()


def test_a_tech_with_no_town_is_placed_at_the_hub_and_the_default_is_printed(capsys):
    towns = [_town(0, "Zz Small", "Zz R", 5), _town(1, "Zz Big", "Zz R", 50)]
    roster = payload.build_places(towns, _techs(("Zz Tech", "", "", "Zz R")))
    assert _placed_at(roster) == "Zz Big"
    assert roster["techs"][0]["basis"] == "region"
    out = capsys.readouterr().out
    assert "Zz R" in out and "Zz Big" in out and "region_hubs.csv" in out


def test_only_hubs_that_are_used_are_printed(capsys):
    towns = [_town(0, "Zz One", "Zz R1", 5), _town(1, "Zz Two", "Zz R2", 50)]
    payload.build_places(towns, _techs(("Zz Tech", "", "", "Zz R1")))
    out = capsys.readouterr().out
    assert "Zz R1" in out and "Zz R2" not in out


def test_an_override_hub_is_not_reported_as_a_default(capsys):
    towns = [_town(0, "Zz Small", "R", 5), _town(1, "Zz Big", "R", 50)]
    overrides = {"R": ("Zz Small", "MO")}
    roster = payload.build_places(towns, _techs(("Zz Tech", "", "", "R")), overrides)
    assert _placed_at(roster) == "Zz Small"
    assert "region_hubs.csv" not in capsys.readouterr().out


def test_a_region_with_no_counts_raises_only_when_a_tech_needs_it():
    dead = "Zz Dead Region"
    towns = [_town(0, "Zz One", dead, None), _town(1, "Zz Two", "Zz Live", 4)]
    ada = ("Zz Ada", "Zz Two", "MO", "Zz Live")
    # No tech needs the dead region's hub, so the build goes on
    assert len(payload.build_places(towns, _techs(ada))["techs"]) == 1
    needs = _techs(ada, ("Zz Bo", "", "", dead))
    with pytest.raises(ValueError, match=r"Zz Dead Region.*region_hubs\.csv"):
        payload.build_places(towns, needs)
    # An override row fixes it
    roster = payload.build_places(towns, needs, {dead: ("Zz One", "MO")})
    assert _placed_at(roster, 1) == "Zz One"


def test_an_override_hub_that_cannot_be_geocoded_names_the_hubs_file_and_region(monkeypatch):
    monkeypatch.setattr(payload, "geocode_city_state", lambda town, state: None)
    towns = [_town(0, "Zz Big", "Zz R", 90)]
    techs = _techs(("Zz Ada", "", "", "Zz R"))
    with pytest.raises(ValueError, match=r"Zz Nowhere.*Zz R.*region_hubs\.csv") as err:
        payload.build_places(towns, techs, {"Zz R": ("Zz Nowhere", "MO")})
    assert "tech_info.csv" not in str(err.value)


def test_a_techs_own_town_that_cannot_be_geocoded_still_names_the_tech_file(monkeypatch):
    monkeypatch.setattr(payload, "geocode_city_state", lambda town, state: None)
    towns = [_town(0, "Zz Big", "Zz R", 90)]
    # An override exists for the tech's region, but the tech has a town of their own
    techs = _techs(("Zz Ada", "Zz Nowhere", "MO", "Zz R"))
    with pytest.raises(ValueError, match=r"Zz Nowhere.*Zz Ada.*tech_info\.csv") as err:
        payload.build_places(towns, techs, {"Zz R": ("Zz Big", "MO")})
    assert "region_hubs.csv" not in str(err.value)


def test_a_tech_with_a_town_does_not_need_its_regions_hub():
    towns = [_town(0, "Zz One", "Zz Dead Region", None)]
    roster = payload.build_places(towns, _techs(("Zz Ada", "Zz One", "MO", "Zz Dead Region")))
    assert roster["techs"][0]["basis"] == "town"


def test_a_comma_separated_region_uses_the_first_and_says_so(capsys):
    towns = [_town(0, "Zz One", "Zz R1", 5), _town(1, "Zz Two", "Zz R2", 50)]
    roster = payload.build_places(towns, _techs(("Zz Ada", "", "", "Zz R1, Zz R2")))
    assert _placed_at(roster) == "Zz One"
    assert roster["unplaced"] == []
    line = next(x for x in capsys.readouterr().out.splitlines() if "Zz Ada" in x)
    assert "Zz R1" in line and "Zz R2" not in line


def test_a_comma_separated_region_whose_first_is_unknown_is_unplaced():
    towns = [_town(0, "Zz One", "Zz R1", 5)]
    roster = payload.build_places(towns, _techs(("Zz Ada", "", "", "Zz Nope, Zz R1")))
    assert roster["unplaced"] == ["Zz Ada"]


def test_a_region_whose_own_name_has_a_comma_matches_whole():
    towns = [_town(0, "Zz One", "Zz R1, East", 5), _town(1, "Zz Two", "Zz R1", 50)]
    roster = payload.build_places(towns, _techs(("Zz Ada", "", "", "Zz R1, East")))
    assert _placed_at(roster) == "Zz One"
