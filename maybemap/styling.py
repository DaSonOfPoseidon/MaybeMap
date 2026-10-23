# Color and basemap helpers shared by the static and interactive renderers.

import os

import contextily as cx
import matplotlib.colors as mcolors
import xyzservices
from dotenv import load_dotenv

from maybemap.config import LABEL_LINE_COLOR, PROJECT_ROOT

# Anchored to the project root so the key is found regardless of CWD. geocoding.py
# loads the same file, but neither module may depend on the other being imported.
load_dotenv(PROJECT_ROOT / ".env")


def lighten(hex_color: str, amount: float = 0.6) -> str:
    r, g, b = mcolors.to_rgb(hex_color)
    r = r + (1 - r) * amount
    g = g + (1 - g) * amount
    b = b + (1 - b) * amount
    return mcolors.to_hex((r, g, b))


def darken(hex_color: str, amount: float = 0.35) -> str:
    r, g, b = mcolors.to_rgb(hex_color)
    scale = 1.0 - amount
    return mcolors.to_hex((r * scale, g * scale, b * scale))


def kml_color_to_hex(kml_color: str) -> tuple[str, float]:
    # KML packs color as aabbggrr, the reverse of the usual rrggbb ordering
    value = kml_color.strip().lstrip("#")
    if len(value) != 8:
        raise ValueError(f"Expected an 8-digit KML aabbggrr color, got {kml_color!r}")
    alpha = int(value[0:2], 16) / 255.0
    blue, green, red = value[2:4], value[4:6], value[6:8]
    return f"#{red}{green}{blue}".lower(), alpha


# CARTO stopped serving its raster basemaps anonymously in August 2026 -- without a
# key every tile comes back stamped "API KEY REQUIRED". A key restores them; the
# Esri themes below need none.


def carto_api_key() -> str:
    return os.getenv("CARTO_API_KEY", "").strip()


def _carto(provider) -> xyzservices.TileProvider:
    # The plain provider when there is no key -- watermarked, but it still renders and
    # the warning below says why. With a key, the same tiles come back clean.
    #
    # The key rides on the provider's own URL rather than a template of our own: the
    # variants do not share a path shape ("rastertiles/voyager_nolabels" against a
    # bare "light_nolabels"), so rebuilding the URL doubles the prefix on one of them.
    key = carto_api_key()
    if not key:
        return provider
    keyed = dict(provider)
    keyed["name"] = f"{provider.name}.Keyed"
    keyed["url"] = f"{provider.url}?key={{api_key}}"
    keyed["api_key"] = key
    return xyzservices.TileProvider(keyed)


def _esri_reference(variant: str) -> xyzservices.TileProvider:
    # Esri's labels-only overlays are transparent PNGs, and xyzservices ships the
    # Canvas base layers but not these. Same {z}/{y}/{x} order as its Esri entries.
    return xyzservices.TileProvider(
        name=f"Esri.{variant}",
        url="https://server.arcgisonline.com/ArcGIS/rest/services/{variant}/MapServer/tile/{z}/{y}/{x}",
        variant=f"Canvas/{variant}",
        attribution="Tiles (C) Esri -- Esri, DeLorme, NAVTEQ",
        max_zoom=16,
    )


# Light themes read black text over a pale basemap; dark themes invert that.
_LIGHT_TEXT = {
    "edgecolor": "black",
    "textcolor": "black",
    "leader_color": LABEL_LINE_COLOR,
    "halo_color": "white",
}
_DARK_TEXT = {
    "edgecolor": "white",
    "textcolor": "white",
    "leader_color": "white",
    "halo_color": "black",
}


def _themes() -> dict[str, dict]:
    # Built per call so a key set after import still takes effect, and so the tests
    # can monkeypatch the environment without reloading the module.
    return {
        # Esri's own description of this map is "a neutral background ... allowing
        # your data to come to the foreground", which is exactly the job here
        "gray": {
            "base_src": cx.providers.Esri.WorldGrayCanvas,
            "label_src": _esri_reference("World_Light_Gray_Reference"),
            "label": "Esri Light Gray",
            "needs_key": False,
            **_LIGHT_TEXT,
        },
        "darkgray": {
            "base_src": _esri_reference("World_Dark_Gray_Base"),
            "label_src": _esri_reference("World_Dark_Gray_Reference"),
            "label": "Esri Dark Gray",
            "needs_key": False,
            **_DARK_TEXT,
        },
        # These three bake their labels into the tiles, so they carry no separate
        # overlay and SHOW_LABELS has nothing to add on top of them
        "streets": {
            "base_src": cx.providers.Esri.WorldStreetMap,
            "label_src": cx.providers.Esri.WorldStreetMap,
            "label": "Esri World Street Map",
            "needs_key": False,
            "labels_in_base": True,
            **_LIGHT_TEXT,
        },
        "topo": {
            "base_src": cx.providers.Esri.WorldTopoMap,
            "label_src": cx.providers.Esri.WorldTopoMap,
            "label": "Esri World Topo",
            "needs_key": False,
            "labels_in_base": True,
            **_LIGHT_TEXT,
        },
        "natgeo": {
            "base_src": cx.providers.Esri.NatGeoWorldMap,
            "label_src": cx.providers.Esri.NatGeoWorldMap,
            "label": "Esri National Geographic",
            "needs_key": False,
            "labels_in_base": True,
            **_LIGHT_TEXT,
        },
        "voyager": {
            "base_src": _carto(cx.providers.CartoDB.VoyagerNoLabels),
            "label_src": _carto(cx.providers.CartoDB.VoyagerOnlyLabels),
            "label": "CARTO Voyager",
            "needs_key": True,
            **_LIGHT_TEXT,
        },
        "positron": {
            "base_src": _carto(cx.providers.CartoDB.PositronNoLabels),
            "label_src": _carto(cx.providers.CartoDB.PositronOnlyLabels),
            "label": "CARTO Positron",
            "needs_key": True,
            **_LIGHT_TEXT,
        },
        "dark": {
            "base_src": _carto(cx.providers.CartoDB.DarkMatterNoLabels),
            "label_src": _carto(cx.providers.CartoDB.DarkMatterOnlyLabels),
            "label": "CARTO Dark Matter",
            "needs_key": True,
            **_DARK_TEXT,
        },
    }


def basemap_names() -> list[str]:
    return list(_themes())


_warned_about_key: set[str] = set()


def basemap_theme(basemap: str) -> dict:
    # Returns the tile sources and text colors matching the requested basemap
    name = (basemap or "").lower()
    themes = _themes()
    if name not in themes:
        raise ValueError(f"Unknown basemap {basemap!r}. Choose one of: {', '.join(themes)}")

    theme = themes[name]
    # Themes that pair a label-free base with a separate overlay are the common case;
    # only the ones whose tiles already carry place names say otherwise.
    theme.setdefault("labels_in_base", False)
    # Rendering a watermarked map without saying so is what made this worth fixing.
    # Once per name per process: the PNG and the HTML both ask for the same theme.
    if theme["needs_key"] and not carto_api_key() and name not in _warned_about_key:
        _warned_about_key.add(name)
        keyless = [n for n, t in themes.items() if not t["needs_key"]]
        print(
            f"   Basemap {name!r} is a CARTO theme and CARTO_API_KEY is not set, so its "
            f'tiles will carry an "API KEY REQUIRED" watermark. Set the key in .env, or '
            f"use a keyless basemap: {', '.join(keyless)}."
        )
    return theme
