# Tests for the KMZ zone loader, run against the real MO_Zones.kmz.

import pytest

from maybemap import config
from maybemap.colors import darken, kml_color_to_hex
from maybemap.sources.kmz import load_kmz_zones

EXPECTED_COLORS = {
    "MO West": "#0070ff",
    "MO East": "#ffaa00",
    "MO Central": "#8400a8",
}


@pytest.fixture(scope="module")
def zones():
    return load_kmz_zones(config.KMZ_PATH)


def _colors(zones):
    return dict(zip(zones["Label"], zones["Color"], strict=True))


def test_parses_every_zone(zones):
    assert len(zones) == 3
    assert set(zones["Label"]) == set(EXPECTED_COLORS)


def test_labels_come_from_the_description_table(zones):
    # The KMZ carries no ExtendedData, so these names only exist in the
    # <description> HTML table -- an empty label means that parsing broke
    assert all(label.strip() for label in zones["Label"])


def test_colors_decode_from_the_kml_polystyle(zones):
    assert _colors(zones) == EXPECTED_COLORS


def test_geometry_is_valid_and_geographic(zones):
    assert zones.crs.to_epsg() == 4326
    assert zones.geometry.is_valid.all()
    assert set(zones.geometry.geom_type) <= {"Polygon", "MultiPolygon"}


def test_zones_fall_inside_missouri(zones):
    minx, miny, maxx, maxy = zones.total_bounds
    assert -96.0 < minx < -89.0
    assert -96.0 < maxx < -89.0
    assert 35.5 < miny < 41.0
    assert 35.5 < maxy < 41.0


def test_zones_do_not_overlap(zones):
    geoms = list(zones.to_crs(3857).geometry)
    for i, a in enumerate(geoms):
        for b in geoms[i + 1 :]:
            assert a.intersection(b).area == pytest.approx(0.0, abs=1.0)


def test_color_override_beats_the_kml_style(monkeypatch):
    monkeypatch.setitem(config.ZONE_COLOR_OVERRIDES, "MO West", "#123456")
    colors = _colors(load_kmz_zones(config.KMZ_PATH))
    assert colors["MO West"] == "#123456"
    assert colors["MO East"] == EXPECTED_COLORS["MO East"]


@pytest.mark.parametrize(
    ("kml", "expected"),
    [("7fff7000", "#0070ff"), ("ff000000", "#000000"), ("ffffffff", "#ffffff")],
)
def test_kml_color_is_byte_reversed(kml, expected):
    assert kml_color_to_hex(kml)[0] == expected


def test_kml_color_alpha():
    assert kml_color_to_hex("7fff7000")[1] == pytest.approx(0.498, abs=0.001)


def test_rejects_malformed_kml_color():
    with pytest.raises(ValueError):
        kml_color_to_hex("abc")


def test_darken_scales_each_channel():
    assert darken("#ffffff", 0.5) == "#808080"
    assert darken("#0070ff", 0.0) == "#0070ff"
