# Builds a task map from the work-item export: one pin per open task, placed at its
# street address and colored by status. Unlike the city source this is not a map of
# places -- several tasks share a town, so the pins are the subject and the town name
# is context.

import re
from math import cos, pi, sin
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from maybemap.config import (
    TASK_COINCIDENT_OFFSET_M,
    TASK_DATA_PATH,
    TASK_DOT_SIZE,
    TASK_LABEL_CITIES,
    TASK_MAP_STATE,
    TASK_MARKER_RADIUS,
    TASK_MAX_ADDRESS_OFFSET_KM,
    TASK_STATUS_COLORS,
    TASK_STATUS_ORDER,
    UNASSIGNED_GROUP_COLOR,
    UNASSIGNED_GROUP_LABEL,
)
from maybemap.geocoding import geocode_address, geocode_city_state
from maybemap.sources import MapData, PointPopup

# Export header (lower-cased) -> frame column. The frame names stay valid
# identifiers because the renderers read these back off itertuples rows, where a
# space would turn the attribute into a positional name.
TASK_COLUMNS = {
    "city": "City",
    "task status": "Status",
    "street address": "Address",
    "subscriber name": "Subscriber",
    "task id": "TaskID",
    "person assigned": "Assigned",
    "region": "Region",
    "county": "County",
    "due time": "Due",
    "service": "Service",
}
REQUIRED = ["City", "Status", "Address"]

# What a pin click opens. Status is the color, so it leads.
TASK_POPUP = PointPopup(
    title_attr="Subscriber",
    subtitle_attr="Address",
    fields=[
        ("Status", "Status"),
        ("DueLabel", "Due"),
        ("AssignedLabel", "Assigned to"),
        ("Region", "Region"),
        ("CityState", "City"),
        ("County", "County"),
        ("TaskID", "Task ID"),
        ("Placement", "Placement"),
    ],
)


def _zip_from_service(value: str) -> str:
    # The Service column holds either a phone number or the full address, and where
    # it holds the address it carries the ZIP -- which is what tells Geoapify apart
    # the several "Main St"s in mid-Missouri. ZIP+4 arrives unhyphenated, so only
    # the leading five digits are kept.
    m = re.search(rf"{TASK_MAP_STATE}\s+(\d{{5}})", value or "")
    return m.group(1) if m else ""


def _clean_assignee(value: str) -> str:
    # "Quisenberry, Jim (jiqu0170)" is a login record, not a name a popup should show
    name = re.sub(r"\s*\([^)]*\)\s*$", "", value or "").strip()
    if "," in name:
        last, _, first = name.partition(",")
        name = f"{first.strip()} {last.strip()}".strip()
    return name


def _read_tasks(path: Path) -> pd.DataFrame:
    # dtype=str keeps subscriber and task IDs from being read as numbers
    df = pd.read_csv(path, dtype=str)
    df.rename(columns={c: c.strip() for c in df.columns}, inplace=True)

    lower_index = {c.lower(): c for c in df.columns}
    keep = {lower_index[key]: name for key, name in TASK_COLUMNS.items() if key in lower_index}
    df = df[list(keep)].rename(columns=keep)

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            f"Task data must contain columns City, Task Status and Street Address. "
            f"Missing: {missing}"
        )
    for name in TASK_COLUMNS.values():
        if name not in df.columns:
            df[name] = ""
        df[name] = df[name].fillna("").astype(str).str.strip()

    df = df[(df["City"] != "") & (df["Address"] != "")].copy()

    # The export shouts its place names and carries no State column at all: every
    # county in it is a Missouri county.
    df["City"] = df["City"].str.title()
    df["County"] = df["County"].str.title()
    df["State"] = TASK_MAP_STATE
    df["CityState"] = df["City"] + ", " + df["State"]
    df["Status"] = df["Status"].str.title()
    df["Region"] = df["Region"].replace("", UNASSIGNED_GROUP_LABEL)
    df["Zip"] = df["Service"].map(_zip_from_service)

    df["AssignedLabel"] = df["Assigned"].map(_clean_assignee).replace("", "Unassigned")
    due = pd.to_datetime(df["Due"], errors="coerce")
    df["DueLabel"] = due.dt.strftime("%b %d, %Y").fillna("")
    df["Due"] = due

    # One query per task, tightened by the ZIP when the export gave one
    df["Query"] = (
        df["Address"] + ", " + df["City"] + ", " + TASK_MAP_STATE + " " + df["Zip"]
    ).str.strip()

    print(f"Loaded {len(df)} tasks from {path.name}")
    return df


def _status_order(df: pd.DataFrame) -> list[str]:
    # The configured order first, then anything the export invented, so a new status
    # shows up on the map rather than being quietly dropped
    present = set(df["Status"])
    known = [s for s in TASK_STATUS_ORDER if s in present]
    return known + sorted(present - set(known))


def _separate_coincident(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # Tasks that fell back to their city center all land on the same point, where
    # one marker hides the rest. Pushing them onto a small ring around it keeps
    # every pin clickable without moving any of them meaningfully on the map.
    keys = [(round(g.x, 1), round(g.y, 1)) for g in gdf.geometry]
    counts = pd.Series(keys).value_counts()
    duplicated = {k for k, n in counts.items() if n > 1}
    if not duplicated:
        return gdf

    seen: dict[tuple[float, float], int] = {}
    moved = []
    for key, geom in zip(keys, gdf.geometry, strict=True):
        if key not in duplicated:
            moved.append(geom)
            continue
        i = seen.get(key, 0)
        seen[key] = i + 1
        angle = 2 * pi * i / counts[key]
        moved.append(
            Point(
                geom.x + TASK_COINCIDENT_OFFSET_M * cos(angle),
                geom.y + TASK_COINCIDENT_OFFSET_M * sin(angle),
            )
        )

    print(f"   Nudged {sum(counts[k] for k in duplicated)} pins sharing a point apart")
    return gdf.set_geometry(gpd.GeoSeries(moved, crs=gdf.crs, index=gdf.index))


# Web Mercator stretches distance by about a quarter at Missouri's latitude, which
# a kilometre threshold cannot afford; this equal-area CONUS projection does not.
DISTANCE_CRS = 5070


def _offset_km(res: dict, city: dict | None) -> float:
    if city is None:
        return 0.0
    pair = gpd.GeoSeries(
        [Point(res["lon"], res["lat"]), Point(city["lon"], city["lat"])], crs=4326
    ).to_crs(DISTANCE_CRS)
    return pair.iloc[0].distance(pair.iloc[1]) / 1000


def _within_town(res: dict, city: dict | None) -> bool:
    # A town with no geocode of its own has nothing to check the address against,
    # so the address is taken at its word
    return city is None or _offset_km(res, city) <= TASK_MAX_ADDRESS_OFFSET_KM


def load_task_points(path: Path | str = TASK_DATA_PATH) -> tuple[gpd.GeoDataFrame, list[str]]:
    path = Path(path)
    df = _read_tasks(path)

    queries = df[["Query", "City"]].drop_duplicates("Query").reset_index(drop=True)
    print(f"\nGeocoding {len(queries)} task addresses...")
    located = {}
    for i, row in queries.iterrows():
        res = geocode_address(row["Query"], TASK_MAP_STATE, f"({i + 1}/{len(queries)})")
        located[row["Query"]] = res

    # Every town is looked up too, both as the fallback position and as the check
    # on the addresses. These are the same lookups the city map already cached.
    city_points = {
        city: geocode_city_state(city, TASK_MAP_STATE) for city in sorted(set(queries["City"]))
    }

    # An address Geoapify cannot place -- or places implausibly far from the town it
    # names -- still belongs on the map: it falls back to its town, flagged so the
    # popup never passes a town center off as a rooftop.
    fell_back = set()
    for query, city in zip(queries["Query"], queries["City"], strict=True):
        res = located[query]
        if res is not None and _within_town(res, city_points[city]):
            continue
        if res is not None:
            print(
                f"   Rejected: {query} landed "
                f"{_offset_km(res, city_points[city]):.0f}km from {city} -- using the town center"
            )
        located[query] = city_points[city]
        fell_back.add(query)

    df["Latitude"] = df["Query"].map(lambda q: (located[q] or {}).get("lat"))
    df["Longitude"] = df["Query"].map(lambda q: (located[q] or {}).get("lon"))
    df["Approx"] = df["Query"].isin(fell_back)

    dropped = int(df["Latitude"].isna().sum())
    if dropped:
        print(f"   Dropping {dropped} tasks that could not be placed at all")
    df = df.dropna(subset=["Latitude", "Longitude"]).copy()

    df["Placement"] = df["Approx"].map({True: "City center (address not matched)", False: ""})

    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]), crs=4326
    ).to_crs(3857)
    gdf = _separate_coincident(gdf)

    order = _status_order(df)
    # Status is what the map is colored by, so it takes the Group slot every
    # renderer already reads
    gdf["Group"] = pd.Categorical(gdf["Status"], categories=order, ordered=True)
    gdf["Color"] = gdf["Status"].map(TASK_STATUS_COLORS).fillna(UNASSIGNED_GROUP_COLOR)
    # Eleven tasks in Jefferson City want one label, not eleven: only the first pin
    # in each town carries a name, and the shared placement code reads it off `City`.
    gdf["Labeled"] = ~gdf.duplicated(subset=["City"])

    approx = int(gdf["Approx"].sum())
    print(
        f"Placed {len(gdf)} tasks: {len(gdf) - approx} at their address, "
        f"{approx} at their city center.\n"
    )
    return gdf, order


def load_tasks(
    path: Path | str = TASK_DATA_PATH,
    *,
    title: str = "Open Tasks by Status",
    output_stem: str = "tasks",
) -> MapData:
    points, order = load_task_points(path)
    colors = {s: TASK_STATUS_COLORS.get(s, UNASSIGNED_GROUP_COLOR) for s in order}

    return MapData(
        # No zones and no regions: the pins are the whole map, on a plain basemap
        regions=gpd.GeoDataFrame({"Label": [], "Color": []}, geometry=[], crs=3857),
        colors=colors,
        order=order,
        title=title,
        output_stem=output_stem,
        points=points,
        draw_regions=False,
        draw_city_dots=True,
        label_cities=TASK_LABEL_CITIES,
        popup=TASK_POPUP,
        group_point_layers=True,
        point_size=TASK_DOT_SIZE,
        marker_radius=TASK_MARKER_RADIUS,
        legend_title="Task status",
    )
