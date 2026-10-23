# Covers the task source: the columns the work-item export disagrees with the city
# export on, where a pin ends up when its address cannot be geocoded, and the one
# marker layer per status the interactive map filters with.

import folium
import pytest

from maybemap.config import TASK_STATUS_COLORS
from maybemap.interactive import build_map, export_geojson
from maybemap.sources.tasks import (
    _clean_assignee,
    _read_tasks,
    _zip_from_service,
    load_task_points,
    load_tasks,
)


@pytest.mark.parametrize(
    ("service", "expected"),
    [
        ("1073 B E Terra LN O FALLON, MO 63366", "63366"),
        # ZIP+4 arrives unhyphenated, and only the leading five digits are the ZIP
        ("403 E NORTH ST ELDON, MO  650261748", "65026"),
        # A subscriber ID trailing the address is not a ZIP
        ("2300 N Morley ST MOBERLY, MO  65270 - 92080002069", "65270"),
        ("    (573) 644-6468", ""),
        ("", ""),
    ],
)
def test_the_zip_is_read_out_of_the_service_column(service, expected):
    assert _zip_from_service(service) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Quisenberry, Jim (jiqu0170)", "Jim Quisenberry"),
        ("Kelly, Nick (nike0170)", "Nick Kelly"),
        ("", ""),
    ],
)
def test_the_assignee_loses_its_login_record(raw, expected):
    assert _clean_assignee(raw) == expected


def test_the_export_normalizes_onto_the_shared_column_names(tasks_csv):
    df = _read_tasks(tasks_csv)
    row = df.iloc[0]

    # The export shouts its place names and carries no State column at all
    assert (row.City, row.State) == ("Columbia", "MO")
    assert row.Status == "Active"
    assert row.Subscriber == "Arena Liquor"
    assert row.AssignedLabel == "Jim Quisenberry"
    assert row.DueLabel == "Jul 23, 2026"
    # The ZIP the Service column carried tightens the geocode query
    assert row.Query == "507 Kentucky AVE, Columbia, MO 65201"


def test_a_task_with_no_region_is_collected_rather_than_dropped(tasks_csv):
    df = _read_tasks(tasks_csv)
    assert set(df["Region"]) == {
        "Columbia Core",
        "Jefferson City Area",
        "St Louis West",
        "Unassigned",
    }
    assert len(df) == 6


def test_a_task_with_no_assignee_says_so(tasks_csv):
    df = _read_tasks(tasks_csv)
    assert df.loc[df["TaskID"] == "222222", "AssignedLabel"].iloc[0] == "Unassigned"


def test_an_address_that_cannot_be_geocoded_falls_back_to_its_city(
    tasks_csv, seeded_cache, seeded_address_cache
):
    points, _ = load_task_points(tasks_csv)
    approx = points[points["Approx"]]

    assert set(approx["TaskID"]) == {"444444", "555555", "666666"}
    assert all(approx["Placement"] == "City center (address not matched)")
    # And a real address is never labeled as a stand-in
    assert (points.loc[points["TaskID"] == "111111", "Placement"] == "").all()


def test_an_address_landing_far_from_its_town_is_rejected(
    tasks_csv, seeded_cache, seeded_address_cache
):
    # It geocoded successfully and it is in the right state, so nothing but the
    # distance to the town it names gives the bad match away
    points, _ = load_task_points(tasks_csv)
    row = points[points["TaskID"] == "666666"].iloc[0]

    assert row.Approx
    assert row.Placement == "City center (address not matched)"


def test_pins_sharing_a_point_are_pushed_apart(tasks_csv, seeded_cache, seeded_address_cache):
    # Both Kansas City tasks fall back to the same city center, where one marker
    # would sit invisibly underneath the other
    points, _ = load_task_points(tasks_csv)
    kc = points[points["City"] == "Kansas City"]

    first, second = kc.geometry.iloc[0], kc.geometry.iloc[1]
    assert first.distance(second) > 0
    # Far enough apart to click, close enough to still be the same place
    assert first.distance(second) < 1000


def test_each_city_is_named_once_however_many_tasks_it_holds(
    tasks_csv, seeded_cache, seeded_address_cache
):
    # Eleven tasks in one town want one label, not eleven
    points, _ = load_task_points(tasks_csv)
    labeled = points[points["Labeled"]]
    assert sorted(labeled["City"]) == [
        "Columbia",
        "Jefferson City",
        "Kansas City",
        "Saint Louis",
    ]


def test_status_drives_the_color_and_the_legend_order(
    tasks_csv, seeded_cache, seeded_address_cache
):
    data = load_tasks(tasks_csv)

    # The configured order, restricted to what the export actually contains
    assert data.order == ["Active", "Waiting", "Hold", "Incomplete"]
    assert data.colors["Hold"] == TASK_STATUS_COLORS["Hold"]
    assert list(data.points["Color"]) == [
        TASK_STATUS_COLORS[status] for status in data.points["Status"]
    ]


def _feature_groups(m):
    return [
        child.layer_name for child in m._children.values() if isinstance(child, folium.FeatureGroup)
    ]


def test_the_layer_control_gets_one_marker_layer_per_status(
    tasks_csv, seeded_cache, seeded_address_cache
):
    m = build_map(load_tasks(tasks_csv))
    assert _feature_groups(m) == ["Active (2)", "Waiting (2)", "Hold (1)", "Incomplete (1)"]


def test_a_pin_popup_names_the_subscriber_rather_than_the_town(
    tasks_csv, seeded_cache, seeded_address_cache
):
    m = build_map(load_tasks(tasks_csv))
    html = m.get_root().render()

    assert "Arena Liquor" in html
    assert "Jim Quisenberry" in html
    assert "Jul 23, 2026" in html


def test_the_geojson_falls_back_to_the_points_when_there_are_no_regions(
    tasks_csv, seeded_cache, seeded_address_cache, tmp_path
):
    # The task map is pins and nothing else, so an empty regions frame would export
    # an empty file rather than the map's own data
    import json

    data = load_tasks(tasks_csv)
    out = export_geojson(data, output_dir=tmp_path)
    features = json.loads(open(out).read())["features"]

    assert len(features) == len(data.points)
    assert features[0]["properties"]["Subscriber"] == "Arena Liquor"
