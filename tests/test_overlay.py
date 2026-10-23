# Tests for the city overlay, exercised against the shared spreadsheet and geocode
# cache fixtures in conftest.py, so no Geoapify key or network access is needed.

from maybemap.overlay import _city_boundaries, attach_cities
from maybemap.styling import darken


def test_cities_are_tagged_with_their_zone(zones, spreadsheet, seeded_cache):
    data = attach_cities(zones, spreadsheet)
    by_city = dict(zip(data.points["City"], data.points["Zone"], strict=True))
    assert by_city == {
        "Kansas City": "MO West",
        "Columbia": "MO Central",
        "Saint Louis": "MO East",
    }


def test_cities_outside_the_zones_are_dropped(zones, spreadsheet, seeded_cache):
    data = attach_cities(zones, spreadsheet)
    assert "Chicago" not in set(data.points["City"])
    assert data.draw_city_dots is True


def test_clipping_can_be_disabled(zones, spreadsheet, seeded_cache):
    data = attach_cities(zones, spreadsheet, clip_to_regions=False)
    assert "Chicago" in set(data.points["City"])


def test_zone_mode_darkens_the_dot_against_its_fill(zones, spreadsheet, seeded_cache):
    data = attach_cities(zones, spreadsheet, color_mode="zone")
    colors = dict(zip(data.points["City"], data.points["Color"], strict=True))
    # Same hue family as MO West's #0070ff, but darker so it reads on the fill
    assert colors["Kansas City"] != data.colors["MO West"]
    assert colors["Kansas City"] == darken(data.colors["MO West"])


def test_group_mode_colors_by_spreadsheet_group(zones, spreadsheet, seeded_cache):
    data = attach_cities(zones, spreadsheet, color_mode="group")
    colors = dict(zip(data.points["City"], data.points["Color"], strict=True))
    assert len(set(colors.values())) == 3  # one per surviving group
    # Groups join the regions in the legend
    assert "Group 1: West" in data.order
    assert "MO West" in data.order


def test_labels_stay_off_unless_requested(zones, spreadsheet, seeded_cache):
    assert attach_cities(zones, spreadsheet).label_cities is False
    assert attach_cities(zones, spreadsheet, label_cities=True).label_cities is True


def test_a_circle_is_drawn_per_surviving_region(zones, spreadsheet, seeded_cache):
    data = attach_cities(zones, spreadsheet)
    assert set(data.circles["Group"]) == {"Group 1: West", "Group 2: Central", "Group 3: East"}
    assert data.circles.crs.to_epsg() == 3857
    assert data.label_circles is True


def test_each_circle_encloses_its_cities(zones, spreadsheet, seeded_cache):
    data = attach_cities(zones, spreadsheet)
    circles = dict(zip(data.circles["Group"], data.circles.geometry, strict=True))
    for row in data.points.itertuples(index=False):
        assert circles[row.Group].contains(row.geometry)


def test_zone_mode_circles_take_the_darkened_zone_color(zones, spreadsheet, seeded_cache):
    data = attach_cities(zones, spreadsheet, color_mode="zone")
    # MO West's fill is what the Group 1 circle sits on, so it darkens against it
    assert data.colors["Group 1: West"] == darken(data.colors["MO West"])


def test_group_mode_circles_keep_the_group_color(zones, spreadsheet, seeded_cache):
    data = attach_cities(zones, spreadsheet, color_mode="group")
    dots = dict(zip(data.points["City"], data.points["Color"], strict=True))
    # The circle matches its own group's dots rather than darkening the zone
    assert data.colors["Group 1: West"] == dots["Kansas City"]


def test_circles_can_be_disabled(zones, spreadsheet, seeded_cache):
    data = attach_cities(zones, spreadsheet, region_circles=False)
    assert data.circles is None
    assert data.label_circles is False


def test_regions_outside_every_zone_get_no_circle(zones, spreadsheet, seeded_cache):
    # Chicago survives the disabled clip but has no zone to take a color from
    data = attach_cities(zones, spreadsheet, clip_to_regions=False)
    assert "Group 4: Illinois" not in set(data.circles["Group"])


def test_missing_spreadsheet_leaves_the_map_alone(zones, tmp_path):
    data = attach_cities(zones, tmp_path / "nope.xlsx")
    assert data.points is None
    assert data.draw_city_dots is False


def test_missing_api_key_leaves_the_map_alone(zones, spreadsheet, monkeypatch):
    # No seeded cache, so the loader has to call Geoapify -- which cannot run
    monkeypatch.delenv("GEOAPIFY_KEY", raising=False)
    data = attach_cities(zones, spreadsheet)
    assert data.points is None
    assert data.draw_city_dots is False


def test_cities_with_a_boundary_get_a_shape_and_lose_their_dot(
    zones, spreadsheet, seeded_cache, city_boundaries
):
    data = attach_cities(zones, spreadsheet)
    assert set(data.detail_polys["City"]) == {"Kansas City", "Saint Louis"}
    dotted = dict(zip(data.points["City"], data.points["HasBoundary"], strict=True))
    assert dotted == {"Kansas City": True, "Columbia": False, "Saint Louis": True}


def test_a_boundary_takes_the_color_of_the_dot_it_replaces(
    zones, spreadsheet, seeded_cache, city_boundaries
):
    data = attach_cities(zones, spreadsheet, color_mode="zone")
    polys = dict(zip(data.detail_polys["City"], data.detail_polys["Color"], strict=True))
    dots = dict(zip(data.points["City"], data.points["Color"], strict=True))
    assert polys["Kansas City"] == dots["Kansas City"] == darken(data.colors["MO West"])


def test_boundaries_can_be_disabled(zones, spreadsheet, seeded_cache, city_boundaries):
    data = attach_cities(zones, spreadsheet, city_boundaries=False)
    assert data.detail_polys is None
    assert not data.points["HasBoundary"].any()


def test_a_broken_boundary_lookup_leaves_every_city_a_dot(
    zones, spreadsheet, seeded_cache, monkeypatch, capsys
):
    from maybemap import overlay

    def boom(tagged):
        raise RuntimeError("8 cities in a row returned no OSM boundary")

    monkeypatch.setattr(overlay, "_city_boundaries", boom)
    data = attach_cities(zones, spreadsheet)
    assert data.detail_polys is None
    assert not data.points["HasBoundary"].any()
    assert "Falling back to city dots" in capsys.readouterr().out


def test_the_zone_map_still_draws_its_zones(zones, spreadsheet, seeded_cache, city_boundaries):
    # The Excel map hides its dissolved group blobs; the zone map must not hide its zones
    assert attach_cities(zones, spreadsheet).draw_regions is True


def test_a_boundary_carries_the_towns_service_detail(monkeypatch):
    # The zone map has no filter box, so its popups are the only way at the customer
    # data -- and on that map they are read off the boundary rows, not the dots
    import geopandas as gpd
    from shapely.geometry import Point

    from maybemap.sources import cities

    tagged = gpd.GeoDataFrame(
        {
            "City": ["Columbia"],
            "State": ["MO"],
            "Zone": ["MO Central"],
            "Color": ["#123456"],
            "Customers": [25],
            "ServiceMix": ["Mixed"],
        },
        geometry=[Point(0, 0)],
        crs=3857,
    )
    monkeypatch.setattr(
        cities,
        "load_city_boundaries",
        lambda t: gpd.GeoDataFrame(
            {"City": ["Columbia"], "State": ["MO"], "Group": ["Group 2: Central"]},
            geometry=[Point(0, 0).buffer(10)],
            crs=3857,
        ),
    )

    row = _city_boundaries(tagged).iloc[0]
    assert row.Zone == "MO Central"
    assert row.Customers == 25
    assert row.ServiceMix == "Mixed"
