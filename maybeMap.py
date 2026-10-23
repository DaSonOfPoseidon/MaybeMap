# MaybeMap.py - Jackson Keithley
# Builds a PNG and interactive HTML map from a spreadsheet of City / State / Region rows.

from maybemap.config import CITY_DATA_PATH, EXPORT_INTERACTIVE_HTML
from maybemap.interactive import export_geojson, export_html
from maybemap.render import render_png
from maybemap.sources.cities import load_city_regions


def main():
    data = load_city_regions(CITY_DATA_PATH)
    render_png(data)
    export_geojson(data)
    if EXPORT_INTERACTIVE_HTML:
        export_html(data)


if __name__ == "__main__":
    main()
