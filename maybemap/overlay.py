# Overlays city points from the spreadsheet onto a map built from another source.
# Everything here is best-effort: a missing spreadsheet or API key degrades to
# the plain zone map rather than failing the run.

from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.colors as mcolors
from shapely.geometry import Point

from maybemap.config import (
    CIRCLE_PAD_M,
    CITY_DATA_PATH,
    CITY_DOT_COLOR,
    ZONE_MAP_CITY_BOUNDARIES,
    ZONE_MAP_CITY_LABELS,
    ZONE_MAP_CLIP_CITIES_TO_ZONES,
    ZONE_MAP_MIN_BOUNDARY_KM2,
    ZONE_MAP_REGION_CIRCLE_LABELS,
    ZONE_MAP_REGION_CIRCLES,
)
from maybemap.labeling import minimum_enclosing_circle
from maybemap.sources import MapData
from maybemap.styling import darken


def _group_colors(ordered_groups: list[str]) -> dict[str, str]:
    cmap = matplotlib.colormaps["tab20"]
    return {grp: mcolors.to_hex(cmap(i % cmap.N)) for i, grp in enumerate(ordered_groups)}


def _region_circles(tagged, palette: dict[str, str]) -> tuple[gpd.GeoDataFrame, dict[str, str]]:
    # One minimum enclosing circle per spreadsheet region, using the same math as
    # the Excel map so the two outputs agree. A region already carrying its own
    # color (group mode) keeps it; otherwise the circle takes the darkened hue of
    # the zone its cities sit in.
    rows, colors = [], {}
    for grp, sub in tagged.groupby("Group", observed=True):
        zones = sub["Zone"].dropna().unique()
        if grp not in palette and len(zones) != 1:
            # Nothing to take a color from: the region straddles zones or sits
            # entirely outside them (only reachable with the clip turned off)
            continue

        mec = minimum_enclosing_circle([(g.x, g.y) for g in sub.geometry])
        if mec is None:
            continue
        cx, cy, radius = mec
        radius += CIRCLE_PAD_M
        rows.append(
            {"Group": grp, "geometry": Point(cx, cy).buffer(radius, quad_segs=256), "r": radius}
        )
        colors[grp] = palette.get(grp) or darken(palette[zones[0]])

    if not rows:
        return gpd.GeoDataFrame(geometry=[], crs=3857), {}
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=3857), colors


def _city_boundaries(tagged) -> gpd.GeoDataFrame:
    # Real OSM municipal shapes for the cities that have one. Colored per row from
    # the dot color they replace, so the two agree in either CITY_DOT_COLOR mode
    # without depending on a Group palette existing.
    from maybemap.sources.cities import OPTIONAL_COLUMNS, load_city_boundaries

    polys = load_city_boundaries(tagged)
    polys = polys[polys.geometry.notna()].copy()
    if polys.empty:
        return polys

    if ZONE_MAP_MIN_BOUNDARY_KM2 > 0:
        big_enough = polys.geometry.area / 1e6 >= ZONE_MAP_MIN_BOUNDARY_KM2
        polys = polys[big_enough].copy()

    # Whatever per-town detail the export carried rides along, so a boundary click
    # on the zone map opens the same popup its dot would have
    extra = [name for name, _ in OPTIONAL_COLUMNS.values() if name in tagged.columns]
    detail = tagged[["City", "State", "Zone", "Color", *extra]].drop_duplicates(
        subset=["City", "State"]
    )
    return polys.merge(detail, on=["City", "State"], how="left")


def attach_cities(
    data: MapData,
    path: Path | str = CITY_DATA_PATH,
    *,
    clip_to_regions: bool = ZONE_MAP_CLIP_CITIES_TO_ZONES,
    color_mode: str = CITY_DOT_COLOR,
    label_cities: bool = ZONE_MAP_CITY_LABELS,
    city_boundaries: bool = ZONE_MAP_CITY_BOUNDARIES,
    region_circles: bool = ZONE_MAP_REGION_CIRCLES,
    label_circles: bool = ZONE_MAP_REGION_CIRCLE_LABELS,
) -> MapData:
    from maybemap.sources.cities import load_city_points

    path = Path(path)
    if not path.exists():
        print(f"No city spreadsheet at {path.name} - drawing regions only")
        return data

    try:
        # A zone map is about the regions, so towns the export leaves unassigned
        # are dropped here rather than collected into a catch-all
        points, ordered_groups = load_city_points(path, unassigned_label=None)
    except RuntimeError as exc:
        # Raised when GEOAPIFY_KEY is absent and the geocode cache cannot answer
        print(f"Skipping city overlay: {exc}")
        return data

    if points.empty:
        print("Skipping city overlay: no cities geocoded successfully")
        return data

    # Tag each city with the region containing it
    tagged = gpd.sjoin(
        points,
        data.regions[["Label", "geometry"]],
        how="left",
        predicate="within",
    ).drop(columns="index_right")
    tagged = tagged.rename(columns={"Label": "Zone"})

    outside = int(tagged["Zone"].isna().sum())
    if clip_to_regions and outside:
        print(f"   Dropping {outside} cities that fall outside the mapped regions")
        tagged = tagged[tagged["Zone"].notna()].copy()

    if tagged.empty:
        print("Skipping city overlay: no cities fall inside the mapped regions")
        return data

    if color_mode == "group":
        colors = _group_colors(ordered_groups)
        tagged["Color"] = tagged["Group"].map(colors)
        # Surface the groups in the legend alongside the regions
        data.colors = {**data.colors, **colors}
        data.order = [*data.order, *ordered_groups]
    else:
        # Darkened against the region fill the dot sits on, which is the same hue
        tagged["Color"] = tagged["Zone"].map({k: darken(v) for k, v in data.colors.items()})

    tagged["HasBoundary"] = False
    if city_boundaries:
        try:
            polys = _city_boundaries(tagged)
        except RuntimeError as exc:
            # Best-effort like the rest of this module: a broken boundary lookup
            # costs the shapes, not the map
            print(f"   Falling back to city dots: {exc}")
            polys = gpd.GeoDataFrame(geometry=[], crs=3857)

        if not polys.empty:
            data.detail_polys = polys
            drawn = set(zip(polys["City"], polys["State"], strict=True))
            tagged["HasBoundary"] = [
                (c, st) in drawn for c, st in zip(tagged["City"], tagged["State"], strict=True)
            ]
            kept = int((~tagged["HasBoundary"]).sum())
            print(f"   Drew {len(polys)} city boundaries, {kept} cities kept as dots")

    data.points = tagged
    data.draw_city_dots = True
    data.label_cities = label_cities

    if region_circles:
        circles, circle_colors = _region_circles(tagged, data.colors)
        skipped = tagged["Group"].nunique() - len(circles)
        if skipped:
            print(f"   Skipping {skipped} region circles: no zone to take a color from")
        if not circles.empty:
            data.circles = circles
            # Both renderers look colors up by the circle's Group name
            data.colors = {**data.colors, **circle_colors}
            data.label_circles = label_circles
            print(f"   Drew {len(circles)} region circles")

    print(f"Overlaid {len(tagged)} cities from {path.name}")
    return data
