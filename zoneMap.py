# zoneMap.py - builds a PNG and interactive HTML map from a KMZ of zone polygons,
# overlaying cities from the spreadsheet when that file is available.

from maybemap.config import (
    CITY_DATA_PATH,
    EXPORT_INTERACTIVE_HTML,
    KMZ_PATH,
    ZONE_MAP_SHOW_CITIES,
)
from maybemap.interactive import export_geojson, export_html
from maybemap.overlay import attach_cities
from maybemap.render import render_png
from maybemap.sources.kmz import load_kmz_zones


def main():
    data = load_kmz_zones(KMZ_PATH)
    if ZONE_MAP_SHOW_CITIES:
        data = attach_cities(data, CITY_DATA_PATH)
    render_png(data)
    export_geojson(data)
    if EXPORT_INTERACTIVE_HTML:
        export_html(data)


if __name__ == "__main__":
    main()
