# Covers the basemap tile budget. contextily assembles one array spanning the whole
# axes extent, so at a fixed zoom the tile count -- and the memory to render it --
# grows with the square of the extent. A city list that reached both coasts turned
# the region map's hardcoded zoom 9 into a 2190-tile mosaic that cost several GB.

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import mercantile  # noqa: E402
import pytest  # noqa: E402

from maybemap.config import MAX_BASEMAP_TILES  # noqa: E402
from maybemap.render import _resolve_basemap_zoom, _tile_count  # noqa: E402

# The extent the CSV grew to once it listed towns in California and Connecticut
COAST_TO_COAST = (-123.81, 26.14, -73.50, 43.01)
# What the map covered while it was still a Missouri story
MISSOURI = (-95.8, 35.9, -89.1, 40.6)


@pytest.fixture
def axes():
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)


def _extent(ax, bounds):
    west, south, east, north = bounds
    left, bottom = mercantile.xy(west, south)
    right, top = mercantile.xy(east, north)
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    return ax


def test_a_wide_extent_is_clamped_off_the_requested_zoom(axes):
    # The regression: zoom 9 across the full width of the country
    ax = _extent(axes, COAST_TO_COAST)
    assert _tile_count(COAST_TO_COAST, 9) > MAX_BASEMAP_TILES

    zoom = _resolve_basemap_zoom(ax, 9)

    assert zoom < 9
    assert _tile_count(COAST_TO_COAST, zoom) <= MAX_BASEMAP_TILES


def test_a_zoom_that_already_fits_is_left_alone(axes):
    # Missouri at zoom 9 is well inside the budget, so nothing should change
    ax = _extent(axes, MISSOURI)
    assert _tile_count(MISSOURI, 9) <= MAX_BASEMAP_TILES

    assert _resolve_basemap_zoom(ax, 9) == 9


@pytest.mark.parametrize("bounds", [COAST_TO_COAST, MISSOURI])
def test_auto_stays_inside_the_budget(axes, bounds):
    zoom = _resolve_basemap_zoom(_extent(axes, bounds), "auto")

    assert _tile_count(bounds, zoom) <= MAX_BASEMAP_TILES


@pytest.mark.parametrize("bounds", [COAST_TO_COAST, MISSOURI])
def test_auto_spends_the_whole_budget(axes, bounds):
    # Sharpest basemap the budget allows: one zoom further must overrun it
    zoom = _resolve_basemap_zoom(_extent(axes, bounds), "auto")

    assert _tile_count(bounds, zoom + 1) > MAX_BASEMAP_TILES


def test_a_clamp_says_so(axes, capsys):
    # A quietly downgraded basemap looks like a rendering bug months later
    _resolve_basemap_zoom(_extent(axes, COAST_TO_COAST), 9)

    assert "zoom 9" in capsys.readouterr().out


# --- PNG extent -------------------------------------------------------------
# The export lists towns in California and Connecticut that belong to no region.
# Letting them set the extent stretched the PNG across the whole country, which
# both wasted the tile budget on empty ocean and shrank Missouri to a smudge.

import geopandas as gpd  # noqa: E402

from maybemap.render import _extent_bounds  # noqa: E402
from maybemap.sources import MapData  # noqa: E402


def _map_data(points, *, regions=None, circles=None):
    # points as (City, State, lon, lat); regions/circles as (lon, lat, radius_m)
    gdf = gpd.GeoDataFrame(
        {"City": [p[0] for p in points], "State": [p[1] for p in points]},
        geometry=gpd.points_from_xy([p[2] for p in points], [p[3] for p in points]),
        crs=4326,
    ).to_crs(3857)

    def _blobs(rows):
        if not rows:
            return None
        g = gpd.GeoDataFrame(
            {"Label": [f"R{i}" for i in range(len(rows))]},
            geometry=gpd.points_from_xy([r[0] for r in rows], [r[1] for r in rows]),
            crs=4326,
        ).to_crs(3857)
        g["geometry"] = [geom.buffer(r[2]) for geom, r in zip(g.geometry, rows, strict=True)]
        return g

    return MapData(
        regions=_blobs(regions) if regions else gpd.GeoDataFrame(geometry=[], crs=3857),
        colors={},
        order=[],
        title="t",
        output_stem="t",
        points=gdf,
        circles=_blobs(circles),
    )


MISSOURI_CITY = ("Columbia", "MO", -92.33, 38.95)
CALIFORNIA_CITY = ("San Diego", "CA", -117.16, 32.72)


def _covers(bounds, lon, lat):
    x, y = mercantile.xy(lon, lat)
    return bounds[0] <= x <= bounds[2] and bounds[1] <= y <= bounds[3]


def test_towns_outside_the_chosen_states_do_not_widen_the_extent():
    data = _map_data([MISSOURI_CITY, CALIFORNIA_CITY])

    bounds = _extent_bounds(data, {"MO", "IL", "TN"})

    assert _covers(bounds, *MISSOURI_CITY[2:])
    assert not _covers(bounds, *CALIFORNIA_CITY[2:])


def test_no_state_list_covers_everything():
    data = _map_data([MISSOURI_CITY, CALIFORNIA_CITY])

    assert _extent_bounds(data, set()) is None


def test_regions_are_never_cropped_by_the_state_list():
    # A region reaching past its towns still has to fit, or the map crops its subject
    data = _map_data([MISSOURI_CITY], regions=[(-89.5, 37.0, 60000)])

    bounds = _extent_bounds(data, {"MO"})

    assert _covers(bounds, -89.5, 37.0)


def test_circles_are_never_cropped_by_the_state_list():
    data = _map_data([MISSOURI_CITY], circles=[(-94.6, 39.1, 80000)])

    bounds = _extent_bounds(data, {"MO"})

    assert _covers(bounds, -94.6, 39.1)


def test_a_state_list_matching_nothing_falls_back_to_the_full_extent():
    # A typo in the state list must not produce an empty map
    data = _map_data([MISSOURI_CITY, CALIFORNIA_CITY])

    assert _extent_bounds(data, {"ZZ"}) is None
