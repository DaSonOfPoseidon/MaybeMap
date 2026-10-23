# Covers basemap selection: the theme registry the renderers read, the optional CARTO
# key, and the zoom ceiling a provider's own limit imposes. Nothing here touches the
# network -- the assertions are about provider objects and the URLs they build.

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from maybemap import styling  # noqa: E402
from maybemap.render import MAX_TILE_ZOOM, _resolve_basemap_zoom  # noqa: E402
from maybemap.styling import basemap_names, basemap_theme  # noqa: E402

# What render.py and interactive.py read off a theme; a new theme missing one of
# these fails at draw time, well after the run has spent its geocoding budget
THEME_KEYS = {
    "base_src",
    "label_src",
    "label",
    "needs_key",
    "edgecolor",
    "textcolor",
    "leader_color",
    "halo_color",
}


@pytest.fixture
def no_carto_key(monkeypatch):
    monkeypatch.delenv("CARTO_API_KEY", raising=False)
    monkeypatch.setattr(styling, "_warned_about_key", set())


@pytest.fixture
def carto_key(monkeypatch):
    monkeypatch.setenv("CARTO_API_KEY", "TESTKEY123")
    monkeypatch.setattr(styling, "_warned_about_key", set())


@pytest.mark.parametrize("name", basemap_names())
def test_every_theme_carries_what_the_renderers_read(name, no_carto_key):
    theme = basemap_theme(name)
    assert THEME_KEYS <= set(theme)
    # Both layers have to be tile sources contextily and folium can build a URL from
    for layer in ("base_src", "label_src"):
        assert theme[layer].build_url(x=62, y=98, z=8).startswith("https://")


def test_an_unknown_basemap_names_the_valid_ones(no_carto_key):
    # Naming every registered theme, so adding one does not silently stop it being
    # offered to whoever mistyped
    with pytest.raises(ValueError) as raised:
        basemap_theme("moonscape")

    message = str(raised.value)
    assert "moonscape" in message
    assert all(name in message for name in basemap_names())


def test_the_esri_themes_need_no_key(no_carto_key):
    for name in ("gray", "darkgray"):
        theme = basemap_theme(name)
        assert theme["needs_key"] is False
        assert "arcgisonline" in theme["base_src"].build_url(x=62, y=98, z=8)
        # The labels layer is the transparent reference overlay, not the base again
        assert "Reference" in theme["label_src"].build_url(x=62, y=98, z=8)


def test_carto_without_a_key_still_renders_but_says_why(no_carto_key, capsys):
    theme = basemap_theme("voyager")
    out = capsys.readouterr().out

    assert "key=" not in theme["base_src"].build_url(x=62, y=98, z=8)
    assert "API KEY REQUIRED" in out
    assert "gray" in out  # points at the keyless way out

    # The PNG and the HTML both ask for the same theme; saying it twice is noise
    basemap_theme("voyager")
    assert capsys.readouterr().out == ""


def test_a_carto_key_reaches_every_variant(carto_key, capsys):
    # The variants disagree on path shape -- "rastertiles/voyager_nolabels" against a
    # bare "light_nolabels" -- so each one has to be checked, not just the first
    expected = {
        "voyager": "rastertiles/voyager_nolabels",
        "positron": "light_nolabels",
        "dark": "dark_nolabels",
    }
    for name, variant in expected.items():
        url = basemap_theme(name)["base_src"].build_url(x=62, y=98, z=8)
        assert f"{variant}/8/62/98.png?key=TESTKEY123" in url
        assert url.count("rastertiles") == (1 if "rastertiles" in variant else 0)

    # A key is not a reason to warn
    assert "API KEY REQUIRED" not in capsys.readouterr().out


def _axes_over_missouri(span_m: float):
    _, ax = plt.subplots()
    # Centered on mid-Missouri in EPSG:3857, which is what the renderers draw in
    cx, cy = -10_200_000, 4_700_000
    ax.set_xlim(cx - span_m, cx + span_m)
    ax.set_ylim(cy - span_m, cy + span_m)
    return ax


def test_the_zoom_never_passes_the_providers_own_limit(no_carto_key):
    # A block-sized extent has budget for a far deeper zoom than Esri Canvas serves,
    # and asking past a provider's limit answers 404 rather than a coarser tile
    ax = _axes_over_missouri(300)
    esri = basemap_theme("gray")["base_src"]

    assert esri["max_zoom"] == 16
    assert _resolve_basemap_zoom(ax, "auto", provider=esri) == 16
    # A provider that goes deeper is not held back by another one's ceiling
    carto = basemap_theme("voyager")["base_src"]
    assert _resolve_basemap_zoom(ax, "auto", provider=carto) > 16
    plt.close("all")


def test_the_tile_budget_still_wins_over_the_provider_limit(no_carto_key):
    # A statewide extent runs out of tile budget long before zoom 16
    ax = _axes_over_missouri(300_000)
    esri = basemap_theme("gray")["base_src"]

    zoom = _resolve_basemap_zoom(ax, "auto", provider=esri)
    assert zoom < 16
    plt.close("all")


def test_no_provider_keeps_todays_ceiling(no_carto_key):
    ax = _axes_over_missouri(300)
    assert _resolve_basemap_zoom(ax, "auto") == MAX_TILE_ZOOM
    plt.close("all")


def test_the_interactive_map_draws_the_same_tiles_as_the_png(zones, no_carto_key):
    # These used to be set separately: the task map's HTML asked for Positron while
    # its PNG drew Voyager
    import folium

    from maybemap.interactive import build_map

    m = build_map(zones, basemap="gray")
    tiles = [c.tiles for c in m._children.values() if isinstance(c, folium.TileLayer)]

    assert tiles and all("arcgisonline" in t for t in tiles)
    assert basemap_theme("gray")["base_src"].build_url(x=62, y=98, z=8).split("/tile/")[0] in (
        tiles[0].split("/tile/")[0]
    )


def test_the_label_halo_contrasts_with_the_label(no_carto_key):
    # White text wearing a white halo is a smudge, which is what every dark basemap
    # produced while the halo was hardcoded
    light, dark = basemap_theme("gray"), basemap_theme("darkgray")

    assert (light["textcolor"], light["halo_color"]) == ("black", "white")
    assert (dark["textcolor"], dark["halo_color"]) == ("white", "black")
    for theme in (light, dark):
        assert theme["textcolor"] != theme["halo_color"]
