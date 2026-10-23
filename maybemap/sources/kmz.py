# Loads zone polygons out of a KMZ produced by ArcGIS ("Export to KML").
#
# Parsed with the standard library rather than gpd.read_file() for two reasons:
# KMZ support depends on GDAL shipping the LIBKML driver, and the driver cannot
# see the zone name anyway -- these exports carry no <ExtendedData>, so the name
# only exists inside an HTML table in the <description> CDATA.

import re
import zipfile
from pathlib import Path

import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon

from maybemap.config import (
    KMZ_PATH,
    ZONE_BASEMAP_ZOOM,
    ZONE_COLOR_OVERRIDES,
    ZONE_EDGE_ALPHA,
    ZONE_EDGE_WIDTH,
    ZONE_FILL_ALPHA,
)
from maybemap.sources import MapData
from maybemap.styling import kml_color_to_hex

KML_NS = "http://www.opengis.net/kml/2.2"

# ArcGIS writes attributes as <td>FieldName</td><td>Value</td> rows
_DESC_ROW = r"<td>\s*{field}\s*</td>\s*<td>(.*?)</td>"

DEFAULT_NAME_FIELDS = ("Area", "Zone", "Name", "Region")


def _ns(tag: str) -> str:
    return f"{{{KML_NS}}}{tag}"


def _text(element, tag: str) -> str:
    child = element.find(_ns(tag))
    return (child.text or "").strip() if child is not None else ""


def _read_kml_bytes(path: Path) -> bytes:
    # A KMZ is a zip; ArcGIS also drops an .xsl stylesheet in beside the .kml
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.lower().endswith(".kml")]
        if not names:
            raise ValueError(f"No .kml member found inside {path}")
        preferred = next((n for n in names if n.lower().endswith("doc.kml")), names[0])
        return archive.read(preferred)


def _parse_styles(root) -> dict[str, tuple[str, float]]:
    styles = {}
    for style in root.iter(_ns("Style")):
        style_id = style.get("id")
        poly = style.find(_ns("PolyStyle"))
        if not style_id or poly is None:
            continue
        color = _text(poly, "color")
        if color:
            styles[style_id] = kml_color_to_hex(color)
    return styles


def _label_from_description(description: str, name_fields=DEFAULT_NAME_FIELDS) -> str:
    for field_name in name_fields:
        match = re.search(_DESC_ROW.format(field=re.escape(field_name)), description, re.S | re.I)
        if match:
            value = re.sub(r"<[^>]+>", "", match.group(1)).strip()
            if value:
                return value
    return ""


def _ring_coords(ring_parent) -> list[tuple[float, float]]:
    # <outerBoundaryIs>/<LinearRing>/<coordinates> holds "lon,lat,alt" triples
    ring = ring_parent.find(_ns("LinearRing"))
    if ring is None:
        return []
    raw = _text(ring, "coordinates")
    coords = []
    for chunk in raw.split():
        parts = chunk.split(",")
        if len(parts) >= 2:
            coords.append((float(parts[0]), float(parts[1])))
    return coords


def _parse_polygons(placemark) -> Polygon | MultiPolygon | None:
    polygons = []
    for poly in placemark.iter(_ns("Polygon")):
        outer = poly.find(_ns("outerBoundaryIs"))
        if outer is None:
            continue
        shell = _ring_coords(outer)
        if len(shell) < 4:
            continue
        holes = [
            coords
            for inner in poly.findall(_ns("innerBoundaryIs"))
            if len(coords := _ring_coords(inner)) >= 4
        ]
        polygons.append(Polygon(shell, holes))
    if not polygons:
        return None
    return polygons[0] if len(polygons) == 1 else MultiPolygon(polygons)


def load_kmz_zones(
    path: Path | str = KMZ_PATH,
    *,
    title: str = "Missouri Zones",
    output_stem: str | None = None,
    name_fields=DEFAULT_NAME_FIELDS,
) -> MapData:
    from xml.etree import ElementTree

    path = Path(path)
    root = ElementTree.fromstring(_read_kml_bytes(path))
    styles = _parse_styles(root)

    records, colors, order = [], {}, []
    for index, placemark in enumerate(root.iter(_ns("Placemark"))):
        geometry = _parse_polygons(placemark)
        if geometry is None:
            continue

        label = _text(placemark, "name")
        if not label:
            description = _text(placemark, "description")
            label = _label_from_description(description, name_fields)
        if not label:
            label = placemark.get("id") or f"Zone {index + 1}"

        style_id = _text(placemark, "styleUrl").lstrip("#")
        style_color = styles.get(style_id, ("#808080", 1.0))[0]
        color = ZONE_COLOR_OVERRIDES.get(label, style_color)

        records.append({"Label": label, "Color": color, "geometry": geometry})
        colors[label] = color
        order.append(label)

    if not records:
        raise ValueError(f"No polygon placemarks found in {path}")

    regions = gpd.GeoDataFrame(records, geometry="geometry", crs=4326).to_crs(3857)

    print(f"Loaded {len(regions)} zones from {path.name}: {', '.join(order)}")

    return MapData(
        regions=regions,
        colors=colors,
        order=order,
        title=title,
        output_stem=output_stem or path.stem.lower(),
        basemap_zoom=ZONE_BASEMAP_ZOOM,
        label_regions=True,
        fill_alpha=ZONE_FILL_ALPHA,
        edge_alpha=ZONE_EDGE_ALPHA,
        edge_width=ZONE_EDGE_WIDTH,
    )
