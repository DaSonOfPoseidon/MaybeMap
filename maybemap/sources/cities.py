# Builds region polygons from a spreadsheet of City / State / Region rows by
# geocoding each city and fetching its OSM municipal boundary. Reads either a CSV
# or an .xlsx -- the column names and the geometry work are the same either way.

import re
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.colors as mcolors
import pandas as pd
from shapely.geometry import Point
from shapely.ops import unary_union

from maybemap.config import (
    AREA_STYLE,
    BASEMAP_ZOOM,
    CIRCLE_PAD_M,
    CITY_DATA_PATH,
    CITY_EDGE_ALPHA,
    CITY_EDGE_WIDTH,
    CITY_FILL_ALPHA,
    MERGE_GAP_M,
    MIN_PART_M2,
    OSM_MAX_CONSECUTIVE_FALLBACKS,
    POINT_FALLBACK_BUFFER_M,
    REGION_MAP_CITY_FILTERS,
    SIMPLIFY_M,
    UNASSIGNED_GROUP_COLOR,
    UNASSIGNED_GROUP_LABEL,
)
from maybemap.geocoding import (
    cache_osm_polygon,
    fetch_osm_city_polygon,
    geocode_city_state,
    get_cached_osm_polygon,
)
from maybemap.labeling import minimum_enclosing_circle
from maybemap.sources import MapData
from maybemap.states import state_abbrev

# Optional per-town detail, keyed by the lower-cased column header the export uses.
# The frame names stay valid identifiers: the interactive export reads these back off
# itertuples rows, where a space would turn the attribute into a positional name.
OPTIONAL_COLUMNS = {
    "customers": ("Customers", "Int64"),
    "active addresses": ("ActiveAddresses", "Int64"),
    "tasks 6 mo": ("Tasks6Mo", "Int64"),
    "service mix": ("ServiceMix", "string"),
    "residential customers": ("ResidentialCustomers", "Int64"),
    "business customers": ("BusinessCustomers", "Int64"),
}


def _group_num(s: str) -> int:
    # The catch-all is not a region, so it sorts behind every real one and takes
    # the last palette slot rather than displacing a named region's color.
    if s == UNASSIGNED_GROUP_LABEL:
        return 10**9 + 1
    m = re.search(r"Group\s*(\d+)", s or "", re.IGNORECASE)
    return int(m.group(1)) if m else 10**9


def _drop_small_parts(geom, min_area):
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "Polygon":
        return geom if geom.area >= min_area else None
    if geom.geom_type == "MultiPolygon":
        parts = [p for p in geom.geoms if p.area >= min_area]
        if not parts:
            return None
        return unary_union(parts)
    return geom


def _title_case(series: pd.Series) -> pd.Series:
    # The CSV exports place names in caps. str.title() handles the apostrophes and
    # spaces these names actually contain -- O'FALLON, LAKE CITY, DE SOTO.
    return series.str.title()


def _load_sheet(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        # dtype=str keeps ZIP-like and numeric-looking place names intact
        return pd.read_csv(path, dtype=str)
    return pd.read_excel(path, engine="openpyxl")


def _read_frame(path: Path, *, unassigned_label: str | None) -> pd.DataFrame:
    # unassigned_label decides what happens to towns the export lists with no
    # region: a label collects them into a catch-all group, None drops them.
    df = _load_sheet(path)
    df.rename(columns={c: c.strip() for c in df.columns}, inplace=True)

    lower_index = {c.lower(): c for c in df.columns}
    city_col = lower_index.get("city")
    state_col = lower_index.get("state")
    group_col = (
        lower_index.get("group") or lower_index.get("region name") or lower_index.get("region")
    )
    # Optional, but it disambiguates cities that share a name with a county
    county_col = lower_index.get("county")
    # Optional, shown in the interactive popups and driving the filter box
    found = {
        name: lower_index[key] for key, (name, _) in OPTIONAL_COLUMNS.items() if key in lower_index
    }

    if not (city_col and state_col and group_col):
        raise ValueError(
            "City data must contain columns City, State, and Region (or Group / "
            f"Region Name). Found: {df.columns.tolist()}"
        )

    keep = {city_col: "City", state_col: "State", group_col: "Group"}
    if county_col:
        keep[county_col] = "County"
    keep.update({src: name for name, src in found.items()})
    df = df[list(keep)].rename(columns=keep)
    df = df.dropna(subset=["City", "State"]).copy()
    for c in ["City", "State", "Group"]:
        df[c] = df[c].fillna("").astype(str).str.strip()
    df["County"] = df["County"].fillna("").astype(str).str.strip() if county_col else ""

    # Both formats normalize to the same shape: title-case names and two-letter
    # states. classify_city_importance matches on the abbreviation, so the xlsx's
    # spelled-out "Missouri" never matched its own capital before this.
    df["City"] = _title_case(df["City"])
    df["County"] = _title_case(df["County"])
    df["State"] = df["State"].map(state_abbrev)

    # A column the export left out becomes all-NA rather than absent, so every
    # consumer can read it and skip the missing values the same way
    for name, dtype in OPTIONAL_COLUMNS.values():
        if name not in found:
            df[name] = pd.array([pd.NA] * len(df), dtype=dtype)
        elif dtype == "Int64":
            df[name] = pd.to_numeric(df[name], errors="coerce").astype("Int64")
        else:
            df[name] = df[name].astype("string").str.strip().replace("", pd.NA)

    blank = df["Group"] == ""
    if unassigned_label is None:
        if blank.any():
            print(f"Skipping {int(blank.sum())} towns with no region")
        df = df[~blank].copy()
    elif blank.any():
        print(f"Collecting {int(blank.sum())} towns with no region under {unassigned_label!r}")
        df.loc[blank, "Group"] = unassigned_label

    df["Full Location"] = df["City"] + ", " + df["State"]
    optional = ", ".join(f"{name}='{src}'" for name, src in found.items()) or "none"
    print(
        f"Loaded {len(df)} rows from {path.name} (columns mapped as "
        f"City='{city_col}', State='{state_col}', Group='{group_col}', "
        f"County='{county_col or 'none'}'; optional: {optional})"
    )
    return df


def load_city_points(
    path: Path | str = CITY_DATA_PATH,
    *,
    unassigned_label: str | None = UNASSIGNED_GROUP_LABEL,
) -> tuple[gpd.GeoDataFrame, list[str]]:
    # Geocodes the spreadsheet into city points. Shared by the region map and the
    # city overlay on zone maps -- neither needs OSM boundaries for this.
    path = Path(path)
    df = _read_frame(path, unassigned_label=unassigned_label)
    ordered_groups = sorted(df["Group"].unique().tolist(), key=_group_num)

    # Geocoding
    unique_places = df[["Full Location", "City", "State"]].drop_duplicates().reset_index(drop=True)
    total_places = len(unique_places)
    print(f"\nStarting geocoding process for {total_places} unique cities...")

    lat_list, lon_list = [], []
    for i, row in unique_places.iterrows():
        res = geocode_city_state(row["City"], row["State"], f"({i + 1}/{total_places})")
        lat_list.append(res["lat"] if res else None)
        lon_list.append(res["lon"] if res else None)

    print(f"Geocoding complete! Processed {total_places} cities.\n")
    unique_places["Latitude"] = lat_list
    unique_places["Longitude"] = lon_list
    df = df.merge(
        unique_places[["Full Location", "Latitude", "Longitude"]],
        on="Full Location",
        how="left",
    ).dropna(subset=["Latitude", "Longitude"])

    gdf_points_ll = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]), crs=4326
    )
    gdf_points = gdf_points_ll.to_crs(3857)
    gdf_points["Group"] = pd.Categorical(
        gdf_points["Group"], categories=ordered_groups, ordered=True
    )
    return gdf_points, ordered_groups


def load_city_boundaries(gdf_points, *, fallback_buffer_m: float | None = None):
    # One row per unique city: City, State, Group, geometry in EPSG:3857. Geometry is
    # None where OSM has no boundary, unless fallback_buffer_m asks for a circle around
    # the point instead -- the Excel map wants that, the zone overlay wants a dot.
    pt_lookup = {
        (r.City, r.State): r.geometry
        for r in gdf_points[["City", "State", "geometry"]]
        .drop_duplicates(subset=["City", "State"])
        .itertuples(index=False)
    }

    city_records = (
        gdf_points[["City", "State", "Group", "County"]].drop_duplicates().reset_index(drop=True)
    )
    total_cities = len(city_records)
    print(f"Starting OSM boundary fetching for {total_cities} cities...")

    records = []
    consecutive_fallbacks = total_fallbacks = 0
    for i, row in city_records.iterrows():
        city, state, group = row.City, row.State, row.Group
        progress = f"({i + 1}/{total_cities})"

        geom_ll = get_cached_osm_polygon(city, state)
        if geom_ll is None:
            geom_ll = fetch_osm_city_polygon(
                city, state, pt_lookup[(city, state)], progress, county=row.County
            )
            cache_osm_polygon(city, state, geom_ll)
        else:
            print(f"Using cached boundary: {city}, {state} {progress}")

        if geom_ll is None:
            # Overpass can answer 200 with an empty result when it is really
            # timing out or serving the wrong extract, which no exception catches.
            # A long unbroken run of misses is the only signal that reaches here.
            consecutive_fallbacks += 1
            total_fallbacks += 1
            if consecutive_fallbacks >= OSM_MAX_CONSECUTIVE_FALLBACKS:
                raise RuntimeError(
                    f"{consecutive_fallbacks} cities in a row returned no OSM boundary, "
                    f"the latest being {city}, {state}. That points at the lookup being "
                    f"broken rather than the data being missing, so stopping instead of "
                    f"drawing a map of {POINT_FALLBACK_BUFFER_M / 1000:.0f}km circles. "
                    f"Check the Overpass API, then rerun -- cached boundaries are kept."
                )
            if fallback_buffer_m is None:
                geom_3857 = None
            else:
                print(
                    f"   No OSM boundary found, using "
                    f"{fallback_buffer_m / 1000:.1f}km buffer around point"
                )
                geom_3857 = pt_lookup[(city, state)].buffer(fallback_buffer_m)
        else:
            consecutive_fallbacks = 0
            geom_3857 = gpd.GeoSeries([geom_ll], crs=4326).to_crs(3857).iloc[0]

        records.append({"City": city, "State": state, "Group": group, "geometry": geom_3857})

    # Say plainly how much of the map is real geometry rather than a stand-in circle
    stand_in = f"{fallback_buffer_m / 1000:.0f}km circles" if fallback_buffer_m else "no boundary"
    print(
        f"OSM boundary fetching complete! Processed {total_cities} cities, "
        f"{total_cities - total_fallbacks} with real boundaries and "
        f"{total_fallbacks} using {stand_in}.\n"
    )
    return gpd.GeoDataFrame(records, geometry="geometry", crs=3857)


def load_city_regions(
    path: Path | str = CITY_DATA_PATH,
    *,
    title: str = "Mapped Regions by Group",
    output_stem: str = "grouped_regions",
) -> MapData:
    path = Path(path)
    gdf_points, ordered_groups = load_city_points(path)

    # Only towns that belong to a real region get an OSM boundary. The catch-all
    # is most of the export and every one of those lookups is an Overpass round
    # trip, so those towns are drawn as dots instead.
    regioned = gdf_points[gdf_points["Group"] != UNASSIGNED_GROUP_LABEL].copy()
    gdf_city_polys = load_city_boundaries(regioned, fallback_buffer_m=POINT_FALLBACK_BUFFER_M)

    gdf_city_polys["Group"] = pd.Categorical(
        gdf_city_polys["Group"], categories=ordered_groups, ordered=True
    )

    gdf_polys = gdf_city_polys.dissolve(by="Group", as_index=False, aggfunc="first")
    if AREA_STYLE == "union_smooth":
        geoms = gdf_polys.geometry.buffer(MERGE_GAP_M).buffer(-MERGE_GAP_M)
        geoms = geoms.simplify(SIMPLIFY_M, preserve_topology=True)
        geoms = geoms.apply(lambda g: _drop_small_parts(g, MIN_PART_M2))
        gdf_polys["geometry"] = geoms
    elif AREA_STYLE == "convex_hull":
        gdf_polys["geometry"] = gdf_polys.geometry.convex_hull
    gdf_polys = gdf_polys[gdf_polys.geometry.notnull()].copy()
    gdf_polys["Group"] = pd.Categorical(gdf_polys["Group"], categories=ordered_groups, ordered=True)

    # Group visualization
    print("Calculating group boundaries and minimum enclosing circles...")
    circle_rows = []
    for grp, sub in gdf_points.groupby("Group", observed=False):
        if grp == UNASSIGNED_GROUP_LABEL:
            # Its towns are scattered across five states; the enclosing circle
            # would cover everything else on the map
            continue
        pts = [(geom.x, geom.y) for geom in sub.geometry]
        mec = minimum_enclosing_circle(pts)
        if mec is None:
            continue
        cx0, cy0, r = mec
        r = r + CIRCLE_PAD_M
        circle_rows.append(
            {"Group": grp, "geometry": Point(cx0, cy0).buffer(r, quad_segs=256), "r": r}
        )

    gdf_circles = gpd.GeoDataFrame(circle_rows, geometry="geometry", crs=3857)
    gdf_circles["Group"] = pd.Categorical(
        gdf_circles["Group"], categories=ordered_groups, ordered=True
    )

    cmap = matplotlib.colormaps["tab20"]
    colors = {grp: mcolors.to_hex(cmap(i % cmap.N)) for i, grp in enumerate(ordered_groups)}
    if UNASSIGNED_GROUP_LABEL in colors:
        colors[UNASSIGNED_GROUP_LABEL] = UNASSIGNED_GROUP_COLOR

    regions = gdf_polys.rename(columns={"Group": "Label"})[["Label", "geometry"]].copy()
    regions["Color"] = regions["Label"].map(colors)
    gdf_points["Color"] = gdf_points["Group"].map(colors)
    # The unassigned towns are context, not subject matter: they get a dot but no
    # label, so the map does not try to name several hundred of them at once.
    gdf_points["Labeled"] = gdf_points["Group"] != UNASSIGNED_GROUP_LABEL

    return MapData(
        regions=gpd.GeoDataFrame(regions, geometry="geometry", crs=3857),
        colors=colors,
        order=ordered_groups,
        title=title,
        output_stem=output_stem,
        points=gdf_points,
        detail_polys=gdf_city_polys,
        circles=gdf_circles,
        basemap_zoom=BASEMAP_ZOOM,
        draw_regions=False,
        label_regions=False,
        # The towns with no region have no boundary of their own to be seen by
        draw_city_dots=True,
        city_filters=REGION_MAP_CITY_FILTERS,
        fill_alpha=CITY_FILL_ALPHA,
        edge_alpha=CITY_EDGE_ALPHA,
        edge_width=CITY_EDGE_WIDTH,
    )
