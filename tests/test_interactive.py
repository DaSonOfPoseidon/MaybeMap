# Tests for the interactive HTML export, focused on the hover hierarchy. Leaflet
# gives the hover to the last interactive layer containing the cursor, so these
# assert on the order layers are added in: areas, then region circles, then cities.

import re

import folium
import pandas as pd
from shapely.geometry import Point

from maybemap.interactive import (
    _add_city_markers,
    _bucket_label,
    _city_popup_html,
    _customer_bucket,
    _service_type,
    build_map,
)
from maybemap.overlay import attach_cities
from maybemap.sources.cities import load_city_regions


def _geojson_layers(m):
    # (position, layer name, tooltip text) per GeoJson, in the order they were added
    out = []
    for i, child in enumerate(m._children.values()):
        if not isinstance(child, folium.GeoJson):
            continue
        tips = [c.text for c in child._children.values() if isinstance(c, folium.map.Tooltip)]
        out.append((i, child.layer_name, tips[0] if tips else None))
    return out


def _split(layers, suffix):
    return [
        (i, name.removesuffix(f" – {suffix}"), tip)
        for i, name, tip in layers
        if name.endswith(suffix)
    ]


def test_every_circle_carries_its_region_name(zones, spreadsheet, seeded_cache):
    m = build_map(attach_cities(zones, spreadsheet))
    circles = _split(_geojson_layers(m), "circle")
    assert {(name, tip) for _, name, tip in circles} == {
        ("Group 1: West", "Group 1: West"),
        ("Group 2: Central", "Group 2: Central"),
        ("Group 3: East", "Group 3: East"),
    }


def test_areas_keep_their_own_tooltip(zones, spreadsheet, seeded_cache):
    m = build_map(attach_cities(zones, spreadsheet))
    areas = _split(_geojson_layers(m), "zone")
    assert {(name, tip) for _, name, tip in areas} == {
        ("MO West", "MO West"),
        ("MO East", "MO East"),
        ("MO Central", "MO Central"),
    }


def test_circles_are_added_after_the_areas(zones, spreadsheet, seeded_cache):
    # Otherwise the area polygon swallows the hover everywhere inside a circle
    layers = _geojson_layers(build_map(attach_cities(zones, spreadsheet)))
    last_area = max(i for i, *_ in _split(layers, "zone"))
    first_circle = min(i for i, *_ in _split(layers, "circle"))
    assert last_area < first_circle


def test_circles_are_added_largest_first(zones, spreadsheet, seeded_cache):
    # So the smallest circle lands on top and stays reachable when nested inside
    # a larger one, as Columbia Core is inside Jefferson City Area on the real map
    data = attach_cities(zones, spreadsheet)
    expected = list(data.circles.sort_values("r", ascending=False)["Group"])
    added = [name for _, name, _ in _split(_geojson_layers(build_map(data)), "circle")]
    assert added == expected


def test_city_markers_are_added_last(zones, spreadsheet, seeded_cache):
    m = build_map(attach_cities(zones, spreadsheet))
    children = list(m._children.values())
    cities = next(
        i
        for i, c in enumerate(children)
        if isinstance(c, folium.FeatureGroup) and c.layer_name == "Cities"
    )
    assert cities > max(i for i, *_ in _geojson_layers(m))


def test_a_map_without_circles_still_builds(zones):
    layers = _geojson_layers(build_map(zones))
    assert {tip for _, _, tip in layers} == {"MO West", "MO East", "MO Central"}


def _restack_order(html):
    # The layer variables the restack function lifts, bottom to top
    block = html[html.index("function restack_") :]
    return re.findall(r"(\w+)\.bringToFront\(\);", block)


def test_toggling_a_layer_back_on_does_not_steal_the_hover(zones, spreadsheet, seeded_cache):
    # Leaflet appends a re-added layer to the end of the draw order, so a rechecked
    # zone would otherwise land on top of the circles and take their tooltips
    m = build_map(attach_cities(zones, spreadsheet))
    html = m.get_root().render()
    assert "on('overlayadd', restack_" in html

    circles = [c for c in m._children.values() if getattr(c, "layer_name", "").endswith("circle")]
    cities = next(c for c in m._children.values() if getattr(c, "layer_name", "") == "Cities")
    assert _restack_order(html) == [c.get_name() for c in circles] + [cities.get_name()]


def test_the_areas_are_left_out_of_the_restack(zones, spreadsheet, seeded_cache):
    # They are the backdrop: everything else is lifted above them, not the reverse
    m = build_map(attach_cities(zones, spreadsheet))
    lifted = set(_restack_order(m.get_root().render()))
    areas = [c for c in m._children.values() if getattr(c, "layer_name", "").endswith("zone")]
    assert areas and not any(a.get_name() in lifted for a in areas)


def test_a_map_without_circles_or_cities_skips_the_restack(zones):
    assert "restack_" not in build_map(zones).get_root().render()


def test_zones_and_city_boundaries_are_both_drawn(
    zones, spreadsheet, seeded_cache, city_boundaries
):
    # The renderers used to draw one or the other; the zone map needs both
    m = build_map(attach_cities(zones, spreadsheet))
    assert {name for _, name, _ in _split(_geojson_layers(m), "zone")} == {
        "MO West",
        "MO East",
        "MO Central",
    }
    boundaries = next(
        c for c in m._children.values() if getattr(c, "layer_name", "") == "City boundaries"
    )
    assert len(boundaries._children) == 2  # Kansas City and Saint Louis


def test_a_city_boundary_names_its_city_on_hover(zones, spreadsheet, seeded_cache, city_boundaries):
    m = build_map(attach_cities(zones, spreadsheet))
    boundaries = next(
        c for c in m._children.values() if getattr(c, "layer_name", "") == "City boundaries"
    )
    tips = set()
    for child in boundaries._children.values():
        tips |= {c.text for c in child._children.values() if isinstance(c, folium.map.Tooltip)}
    assert tips == {"Kansas City, MO", "Saint Louis, MO"}


def test_city_boundaries_sit_above_the_circles_and_below_the_dots(
    zones, spreadsheet, seeded_cache, city_boundaries
):
    m = build_map(attach_cities(zones, spreadsheet))
    html = m.get_root().render()
    order = _restack_order(html)
    by_name = {c.get_name(): getattr(c, "layer_name", "") for c in m._children.values()}
    roles = [by_name.get(v, "circle") for v in order]
    assert roles[-2:] == ["City boundaries", "Cities"]
    assert all(r not in ("City boundaries", "Cities") for r in roles[:-2])


def test_only_cities_without_a_boundary_keep_a_dot(
    zones, spreadsheet, seeded_cache, city_boundaries
):
    m = build_map(attach_cities(zones, spreadsheet))
    dots = next(c for c in m._children.values() if getattr(c, "layer_name", "") == "Cities")
    assert len(dots._children) == 1  # only Columbia


class _Row:
    # Stands in for an itertuples row from the points frame
    def __init__(self, **kwargs):
        self.City, self.State, self.Group = "Columbia", "MO", "Central"
        self.geometry = Point(-92.33, 38.95)
        self.__dict__.update(kwargs)


def test_a_popup_shows_the_customer_count_when_the_export_carried_one():
    html = _city_popup_html(_Row(Customers=5), has_zone=False)
    assert "<strong>Customers:</strong> 5" in html


def test_a_popup_leaves_the_customer_row_out_when_there_is_no_count():
    # The xlsx has no such column, and the CSV leaves it blank for some towns
    assert "Customers:" not in _city_popup_html(_Row(), has_zone=False)
    assert "Customers:" not in _city_popup_html(_Row(Customers=pd.NA), has_zone=False)


def _filter_keys(filter_layers):
    return {(bucket, service): count for bucket, service, count, _ in filter_layers}


def test_every_city_with_a_count_is_filterable_on_the_region_map(
    city_csv, seeded_cache, stub_city_boundaries
):
    # The region map shows the whole export, so the filter box reaches all of it --
    # nothing is pinned to the map behind the checkboxes
    data = load_city_regions(city_csv)
    always_on, filter_layers = _add_city_markers(folium.Map(), data)

    assert not always_on._children
    assert sum(_filter_keys(filter_layers).values()) == len(data.points)


def test_the_filter_layers_are_keyed_on_bucket_and_service_together(
    city_csv, seeded_cache, stub_city_boundaries
):
    # One layer per combination present, which is what lets the box AND the two
    # dimensions by adding and removing whole layers
    _, filter_layers = _add_city_markers(folium.Map(), load_city_regions(city_csv))
    assert _filter_keys(filter_layers) == {
        (0, "Business"): 2,  # Saint Louis, Springfield
        (0, "Residential"): 2,  # Chicago, Joplin
        (1, "Mixed"): 1,  # Kansas City
        (2, "Mixed"): 1,  # Cape Girardeau
        (3, "Mixed"): 1,  # Columbia
    }


def test_the_filter_layers_stay_out_of_the_layer_control(
    city_csv, seeded_cache, stub_city_boundaries
):
    # They are driven by the filter box instead, which is why they are not listed
    _, filter_layers = _add_city_markers(folium.Map(), load_city_regions(city_csv))
    assert filter_layers and all(layer.control is False for *_, layer in filter_layers)


def test_the_filter_box_names_every_bucket_and_service_with_its_count(
    city_csv, seeded_cache, stub_city_boundaries
):
    html = build_map(load_city_regions(city_csv)).get_root().render()
    assert "Customers" in html and "Service type" in html
    for bucket, label, count in [(0, "1-5", 4), (1, "6-10", 1), (2, "11-20", 1), (3, "21+", 1)]:
        assert f'data-bucket="{bucket}"' in html
        assert f'{label}\n                    <span style="color: #888">({count})' in html
    for service, count in [("Business", 2), ("Residential", 2), ("Mixed", 3)]:
        assert f'data-service="{service}"' in html
        assert f'{service}\n                    <span style="color: #888">({count})' in html


def test_a_layer_is_shown_only_when_both_of_its_boxes_are_checked(
    city_csv, seeded_cache, stub_city_boundaries
):
    # Both dimensions have to agree, otherwise unchecking one service would bring
    # back towns whose customer bucket is switched off
    m = build_map(load_city_regions(city_csv))
    html = m.get_root().render()
    _, filter_layers = _add_city_markers(folium.Map(), load_city_regions(city_csv))

    assert "checked[entry.bucket] !== false" in html
    assert "&& checked[entry.service] !== false" in html
    for bucket, service, _, _ in filter_layers:
        assert f'bucket: "{bucket}", service: "{service}"' in html


def test_the_zone_map_has_no_filter_box(zones, spreadsheet, seeded_cache):
    # It covers the regions only, so it stays clean -- city_filters is off for it
    data = attach_cities(zones, spreadsheet)
    assert not data.city_filters
    html = build_map(data).get_root().render()
    assert "data-bucket" not in html and "data-service" not in html


def test_the_buckets_run_from_one_to_an_open_ended_top():
    assert [_bucket_label(i) for i in range(4)] == ["1-5", "6-10", "11-20", "21+"]
    assert _customer_bucket(1) == 0
    assert _customer_bucket(5) == 0  # the boundary belongs to the lower bucket
    assert _customer_bucket(6) == 1
    assert _customer_bucket(21) == 3
    assert _customer_bucket(26507) == 3  # Columbia, not its own category
    assert _customer_bucket(pd.NA) is None


def test_an_unrecognized_service_mix_leaves_the_town_out_of_that_dimension():
    assert _service_type("Business Only") == "Business"
    assert _service_type("Mixed") == "Mixed"
    assert _service_type("Something Else") is None
    assert _service_type(pd.NA) is None


def test_a_popup_shows_the_service_detail_the_export_carried():
    html = _city_popup_html(
        _Row(
            Customers=5,
            ActiveAddresses=4,
            Tasks6Mo=9,
            ServiceMix="Mixed",
            ResidentialCustomers=3,
            BusinessCustomers=2,
        ),
        has_zone=False,
    )
    assert "<strong>Active Addresses:</strong> 4" in html
    assert "<strong>Tasks (6 mo):</strong> 9" in html
    assert "<strong>Service:</strong> Mixed" in html
    assert "<strong>Residential / Business:</strong> 3 / 2" in html


def test_a_popup_leaves_out_the_service_rows_the_export_never_carried():
    # The xlsx has none of these columns, and the CSV can carry only some of them
    html = _city_popup_html(_Row(Customers=5, ServiceMix="Mixed"), has_zone=False)
    assert "<strong>Service:</strong> Mixed" in html
    assert "Active Addresses" not in html
    assert "Tasks (6 mo)" not in html


def test_a_popup_needs_both_halves_to_show_the_residential_business_split():
    html = _city_popup_html(_Row(ResidentialCustomers=3, BusinessCustomers=pd.NA), has_zone=False)
    assert "Residential / Business" not in html
