# MaybeMap.py - Jackson Keithley

import os, json, time, re
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import requests
import pandas as pd
import geopandas as gpd
from adjustText import adjust_text 
from shapely.ops import unary_union
from shapely.geometry import LineString, Point, box
from shapely import wkt
from dotenv import load_dotenv 
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.transforms import Bbox

import contextily as cx
import osmnx as ox

load_dotenv()

# ======================================================================
# Settings
# ======================================================================
# Input Excel (must live beside this .py)
DATA_PATH = Path(__file__).parent / "Socket-Regions.xlsx"  # excel file

# Geocoding (Geoapify) — now pulled from environment
GEOAPIFY_KEY = os.getenv("GEOAPIFY_KEY")
if not GEOAPIFY_KEY:
    raise RuntimeError("Missing GEOAPIFY_KEY in environment. Please set it in .env")

# Basemap + output toggles
BASEMAP = "voyager"              # "voyager", "dark", or "positron"
BASEMAP_ZOOM = 9
SHOW_LABELS = False              # map labels, leave off
EXPORT_INTERACTIVE_HTML = True
INTERACTIVE_HTML_PATH = "grouped_regions_map.html"

# Outward label bias settings
OUTWARD_ONLY = False            # gate candidates to outward half-space
OUTWARD_MIN_COS = 0.0          # 0..1; 0 = any outward direction, 0.5 ≈ within 60° of straight-out
OUTWARD_MIN_DELTA_M = 1500     # label anchor must be at least this much farther from center than the pin
BBOX_INFLATE_PX = 6            # extra padding around text bboxes for safer no-overlap
LABEL_SEARCH_ANGLES = 32       # finer angular resolution
LABEL_SEARCH_RADII  = [1.0, 1.4, 1.9, 2.5, 3.3, 4.2, 5.5, 7.0, 9.0, 11.0]  # more rings to try
LABEL_PADDING_PX    = 6        # increases label's personal space (used in size pre-measure)

# AdjustText
ADJUSTTEXT_FORCE_TEXT  = 1.05    # repulsion strength between labels
ADJUSTTEXT_FORCE_POINTS = 1.0   # repulsion strength vs. pins
ADJUSTTEXT_EXPAND_TEXT  = (1.25, 1.3)  # extra space around labels (x, y)
ADJUSTTEXT_EXPAND_PTS   = (1.25, 1.3)   # extra space around pins (x, y)
ADJUSTTEXT_MAX_ITERS    = 240   # 120–240 typical; lower = faster, higher = cleaner
LABEL_OFFSET_M = 3000
SEED_BASE_STEP_M    = max(1200, LABEL_OFFSET_M)  # baseline seed distance
SEED_KNN_K          = 3                          # look at 3 nearest neighbors
SEED_BOOST_FACTOR   = 0.6                        # multiply by local crowding
LABEL_MIN_DIST_M    = 900                        # clamp: not closer than this to its pin
LABEL_MAX_DIST_M    = 9000                       # clamp: not farther than this from its pin

# OSM boundary search
SEARCH_RADIUS_M = 40000
MAX_CANDIDATE_AREA_M2 = 3_000_000_000  # drop shapes that are obviously county/state

# City fallback & MEC (circles)
POINT_FALLBACK_BUFFER_M = 5000
CIRCLE_PAD_M = 3000              # extra margin so circle clears all towns
CIRCLE_EDGE_WIDTH = 2.0
CIRCLE_ALPHA_FILL = 0.15
CIRCLE_ALPHA_EDGE = 0.9

# City limit styling
CITY_FILL_ALPHA = 0.32
CITY_EDGE_ALPHA = 0.75
CITY_EDGE_WIDTH = 0.7

# Pins
PIN_MARKER_SIZE = 80

# Dissolved group polygons (export only)
AREA_STYLE = "union_smooth"      # "union", "union_smooth", "convex_hull"
MERGE_GAP_M = 2000
SIMPLIFY_M = 600
MIN_PART_M2 = 1_000_000

# Labeling strategy: all cities with collision-aware leaders
LABEL_STRATEGY = "all_with_leaders"
LABEL_FONTSIZE = 9
LABEL_HALO = True
LABEL_LINE_COLOR = "black"       # (we’ll switch to white on dark tiles)
LABEL_LINE_ALPHA = 0.6
LABEL_LINE_WIDTH = 0.8

# Label placement search (fast, collision-aware)


# OSM / OSMnx politeness
ox.settings.use_cache = True
ox.settings.log_console = False
ox.settings.nominatim_pause = 1.1

# ======================================================================
# Lightweight caches
# ======================================================================
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "MaybeMap/1.0"})

GEOCODE_CACHE_PATH = Path("city_geocodes_cache.json")
GEOCODE_CACHE_PATH.touch(exist_ok=True)
try:
    GEOCODE_CACHE: Dict[str, Optional[Dict[str, float]]] = json.loads(GEOCODE_CACHE_PATH.read_text() or "{}")
except json.JSONDecodeError:
    GEOCODE_CACHE = {}

BOUNDARY_CACHE_PATH = Path("osm_boundaries_cache.json")
BOUNDARY_CACHE_PATH.touch(exist_ok=True)
try:
    BOUNDARY_WKT: Dict[str, Optional[str]] = json.loads(BOUNDARY_CACHE_PATH.read_text() or "{}")
except json.JSONDecodeError:
    BOUNDARY_WKT = {}

def _save_geocode_cache():
    GEOCODE_CACHE_PATH.write_text(json.dumps(GEOCODE_CACHE, indent=2))

def _save_boundary_cache():
    BOUNDARY_CACHE_PATH.write_text(json.dumps(BOUNDARY_WKT, indent=2))

# ======================================================================
# Helpers
# ======================================================================
def lighten(hex_color: str, amount: float = 0.6) -> str:
    r, g, b = mcolors.to_rgb(hex_color)
    r = r + (1 - r) * amount
    g = g + (1 - g) * amount
    b = b + (1 - b) * amount
    return mcolors.to_hex((r, g, b))

# Geoapify geocoding
def geoapify_forward(text: str, *, retries: int = 3, backoff: float = 0.8) -> Optional[Dict[str, float]]:
    url = "https://api.geoapify.com/v1/geocode/search"
    params = {"text": text, "apiKey": GEOAPIFY_KEY, "limit": 1, "filter": "countrycode:us"}
    for attempt in range(1, retries + 1):
        try:
            r = SESSION.get(url, params=params, timeout=15)
            if r.status_code == 429:
                time.sleep(backoff * attempt); continue
            r.raise_for_status()
            feats = (r.json().get("features") or [])
            if not feats: return None
            lon, lat = feats[0]["geometry"]["coordinates"]
            return {"lon": float(lon), "lat": float(lat)}
        except requests.RequestException:
            if attempt == retries: return None
            time.sleep(backoff * attempt)
    return None

def geocode_city_state(city: str, state: str) -> Optional[Dict[str, float]]:
    key = f"{city}, {state}"
    if key in GEOCODE_CACHE:
        return GEOCODE_CACHE[key]
    res = geoapify_forward(f"{city}, {state}, USA")
    GEOCODE_CACHE[key] = res
    _save_geocode_cache()
    return res

# OSM polygon cache helpers
def get_cached_osm_polygon(city: str, state: str):
    key = f"{city}, {state}".upper()
    w = BOUNDARY_WKT.get(key, None)
    if not w:
        return None
    geom = wkt.loads(w)
    g = gpd.GeoSeries([geom], crs=4326).to_crs(3857)
    if g.iloc[0].area > MAX_CANDIDATE_AREA_M2:  # discard too-large cached shapes
        return None
    return geom

def cache_osm_polygon(city: str, state: str, geom):
    key = f"{city}, {state}".upper()
    BOUNDARY_WKT[key] = (geom.wkt if geom is not None else None)
    _save_boundary_cache()

def fetch_osm_city_polygon(city: str, state: str):
    """Find a municipal polygon near the city geocode point, prefer proper scale."""
    city_up = city.upper()
    pt_3857 = pt_lookup[(city, state)]
    pt_ll = gpd.GeoSeries([pt_3857], crs=3857).to_crs(4326).iloc[0]
    center = (pt_ll.y, pt_ll.x)  # (lat, lon)

    def _gather(tags):
        try:
            return ox.features_from_point(center, tags=tags, dist=SEARCH_RADIUS_M)
        except Exception:
            return gpd.GeoDataFrame(geometry=[])

    g1 = _gather({"boundary": "administrative", "admin_level": ["7","8","9","10"]})
    g2 = _gather({"place": ["city", "town", "village", "municipality"]})
    if len(g1) == 0 and len(g2) == 0:
        return None

    g = pd.concat([g1, g2], ignore_index=True, sort=False)
    g = g[g.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    if g.empty:
        return None

    if "name" in g.columns and g["name"].notna().any():
        g["NAME_UP"] = g["name"].str.upper()
        exact = g[g["NAME_UP"] == city_up]
        if not exact.empty:
            g = exact

    g_3857 = g.to_crs(3857).copy()
    g_3857["__area"] = g_3857.geometry.area
    g_3857 = g_3857[g_3857["__area"] < MAX_CANDIDATE_AREA_M2]
    if g_3857.empty:
        return None

    dists = g_3857.geometry.centroid.distance(pt_3857)
    idx = dists.idxmin()
    chosen_3857 = g_3857.loc[idx, "geometry"]
    return gpd.GeoSeries([chosen_3857], crs=3857).to_crs(4326).iloc[0]

# Minimum Enclosing Circle helpers
from math import hypot, isclose

def _circle_from_two(p1, p2):
    (x1, y1), (x2, y2) = p1, p2
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    r = hypot(x1 - x2, y1 - y2) / 2.0
    return (cx, cy, r)

def _circle_from_three(p1, p2, p3):
    (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3
    d = 2 * (x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
    if isclose(d, 0.0, abs_tol=1e-9):
        return None
    ux = ((x1**2 + y1**2)*(y2 - y3) + (x2**2 + y2**2)*(y3 - y1) + (x3**2 + y3**2)*(y1 - y2)) / d
    uy = ((x1**2 + y1**2)*(x3 - x2) + (x2**2 + y2**2)*(x1 - x3) + (x3**2 + y3**2)*(x2 - x1)) / d
    r = hypot(ux - x1, uy - y1)
    return (ux, uy, r)

def _covers_all(cx, cy, r, pts, eps=1e-6):
    rr = r + eps
    for (x, y) in pts:
        if hypot(x - cx, y - cy) > rr:
            return False
    return True

def minimum_enclosing_circle(points):
    n = len(points)
    if n == 0: return None
    if n == 1:
        (x, y) = points[0]; return (x, y, 0.0)
    best = None
    for i in range(n):
        for j in range(i + 1, n):
            c = _circle_from_two(points[i], points[j])
            if _covers_all(*c, points):
                if best is None or c[2] < best[2]: best = c
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                c = _circle_from_three(points[i], points[j], points[k])
                if c is None: continue
                if _covers_all(*c, points):
                    if best is None or c[2] < best[2]: best = c
    if best is None:
        xs = [p[0] for p in points]; ys = [p[1] for p in points]
        cxm, cym = sum(xs)/n, sum(ys)/n
        r = max(hypot(x - cxm, y - cym) for (x, y) in points)
        best = (cxm, cym, r)
    return best

# ======================================================================
# Load data from Excel
# ======================================================================
# Expect columns: City, State, Group
DATA_PATH = Path(__file__).parent / "Socket-Regions.xlsx"

df = pd.read_excel(DATA_PATH, engine="openpyxl")

# normalize whitespace in headers
df.rename(columns={c: c.strip() for c in df.columns}, inplace=True)

# case-insensitive header lookup
lower_index = {c.lower(): c for c in df.columns}

city_col  = lower_index.get("city")
state_col = lower_index.get("state")
group_col = lower_index.get("group") or lower_index.get("region name")

if not (city_col and state_col and group_col):
    have = df.columns.tolist()
    raise ValueError(
        "Excel must contain columns City, State, and Group (or Region Name). "
        f"Found: {have}"
    )

# keep / rename to canonical names
df = df[[city_col, state_col, group_col]].rename(
    columns={city_col: "City", state_col: "State", group_col: "Group"}
)

# clean values
df = df.dropna(subset=["City", "State", "Group"]).copy()
for c in ["City", "State", "Group"]:
    df[c] = df[c].astype(str).str.strip()

# Build "Full Location"
df["Full Location"] = df["City"] + ", " + df["State"]

print(f"Loaded {len(df)} rows from {DATA_PATH.name} "
      f"(columns mapped as City='{city_col}', State='{state_col}', Group='{group_col}')")

# Group order (sort legend by the leading number in “Group N: …” if present)
def _group_num(s: str) -> int:
    m = re.search(r"Group\s*(\d+)", s or "", re.IGNORECASE)
    return int(m.group(1)) if m else 10**9

ordered_groups = sorted(df["Group"].unique().tolist(), key=_group_num)

# ======================================================================
# Geocode points (WGS84 & WebMercator)
# ======================================================================
unique_places = df[["Full Location", "City", "State"]].drop_duplicates().reset_index(drop=True)
lat_list, lon_list = [], []
for _, row in unique_places.iterrows():
    res = geocode_city_state(row["City"], row["State"])
    lat_list.append(res["lat"] if res else None); lon_list.append(res["lon"] if res else None)
unique_places["Latitude"] = lat_list; unique_places["Longitude"] = lon_list
df = df.merge(unique_places[["Full Location", "Latitude", "Longitude"]], on="Full Location", how="left").dropna(subset=["Latitude", "Longitude"])

gdf_points_ll = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]), crs=4326)
gdf_points = gdf_points_ll.to_crs(3857)
gdf_points["Group"] = pd.Categorical(gdf_points["Group"], categories=ordered_groups, ordered=True)

# lookup for city center points
pt_lookup = {
    (r.City, r.State): gdf_points.loc[(gdf_points["City"] == r.City) & (gdf_points["State"] == r.State), "geometry"].iloc[0]
    for r in df[["City","State"]].drop_duplicates().itertuples(index=False)
}

# ======================================================================
# City polygons (OSM or fallback buffer)
# ======================================================================
records = []
for _, row in df[["City","State","Group"]].drop_duplicates().iterrows():
    city, state, group = row.City, row.State, row.Group

    geom_ll = get_cached_osm_polygon(city, state)
    if geom_ll is None:
        geom_ll = fetch_osm_city_polygon(city, state)
        cache_osm_polygon(city, state, geom_ll)

    if geom_ll is None:
        pt = pt_lookup[(city, state)]
        geom_3857 = pt.buffer(POINT_FALLBACK_BUFFER_M)
    else:
        geom_3857 = gpd.GeoSeries([geom_ll], crs=4326).to_crs(3857).iloc[0]

    records.append({"City": city, "State": state, "Group": group, "geometry": geom_3857})

gdf_city_polys = gpd.GeoDataFrame(records, geometry="geometry", crs=3857)
gdf_city_polys["Group"] = pd.Categorical(gdf_city_polys["Group"], categories=ordered_groups, ordered=True)

# Dissolve city polygons per Group (export layer)
gdf_polys = gdf_city_polys.dissolve(by="Group", as_index=False, aggfunc="first")
def _drop_small_parts(geom, min_area):
    if geom is None or geom.is_empty: return None
    if geom.geom_type == "Polygon": return geom if geom.area >= min_area else None
    if geom.geom_type == "MultiPolygon":
        parts = [p for p in geom.geoms if p.area >= min_area]
        if not parts: return None
        return unary_union(parts)
    return geom
if AREA_STYLE == "union":
    pass
elif AREA_STYLE == "union_smooth":
    geoms = gdf_polys.geometry.buffer(MERGE_GAP_M).buffer(-MERGE_GAP_M)
    geoms = geoms.simplify(SIMPLIFY_M, preserve_topology=True)
    geoms = geoms.apply(lambda g: _drop_small_parts(g, MIN_PART_M2))
    gdf_polys["geometry"] = geoms
elif AREA_STYLE == "convex_hull":
    gdf_polys["geometry"] = gdf_polys.geometry.convex_hull
gdf_polys = gdf_polys[gdf_polys.geometry.notnull()].copy()
gdf_polys["Group"] = pd.Categorical(gdf_polys["Group"], categories=ordered_groups, ordered=True)

# ======================================================================
# Group circles (minimum enclosing circle per group)
# ======================================================================
circle_rows = []
for grp, sub in gdf_points.groupby("Group", observed=False):
    pts = [(geom.x, geom.y) for geom in sub.geometry]
    mec = minimum_enclosing_circle(pts)
    if mec is None: continue
    cx0, cy0, r = mec
    r = r + CIRCLE_PAD_M
    circle_geom = Point(cx0, cy0).buffer(r, resolution=256)
    circle_rows.append({"Group": grp, "geometry": circle_geom, "cx": cx0, "cy": cy0, "r": r})

gdf_circles = gpd.GeoDataFrame(circle_rows, geometry="geometry", crs=3857)
gdf_circles["Group"] = pd.Categorical(gdf_circles["Group"], categories=ordered_groups, ordered=True)

# ======================================================================
# Label placement: fast, collision-aware search in pixel space
# ======================================================================
def draw_labels_with_leaders(ax, gdf_pts, color_for, text_color, leader_color):
    """
    adjustText-based labeling with:
      • density-aware initial offsets (denser = farther seed),
      • collision resolution (text vs text & pins),
      • post-adjust clamping to keep labels readable,
      • leader lines from each pin to final label.
    """
    # --- group centers for outward seeding
    centers = {r.Group: Point(r.cx, r.cy) for r in gdf_circles.itertuples(index=False)}

    # --- precompute density: average distance to K nearest neighbors (within same group)
    # work in 3857 meters (already your CRS)
    pts_df = gdf_pts[["Group", "geometry", "City", "State"]].reset_index(drop=True)
    # build list per-group for quick neighbor lookups
    group_idx = {}
    for grp, sub in pts_df.groupby("Group", observed=False):
        coords = np.array([[p.x, p.y] for p in sub.geometry.to_list()])
        group_idx[grp] = (sub.index.to_numpy(), coords)

    # nearest-neighbor average distance for each point
    knn_dist = np.zeros(len(pts_df), dtype=float)
    for grp, (idxs, coords) in group_idx.items():
        n = coords.shape[0]
        if n <= 1:
            knn_dist[idxs] = SEED_BASE_STEP_M
            continue
        # pairwise distances (n is small; this is fine)
        dmat = np.sqrt(((coords[None, :, :] - coords[:, None, :])**2).sum(axis=2))
        np.fill_diagonal(dmat, np.inf)
        k = min(SEED_KNN_K, max(1, n-1))
        # average of k nearest distances
        knn_dist[idxs] = np.partition(dmat, kth=k-1, axis=1)[:, :k].mean(axis=1)

    # --- seed labels slightly outward, scaled by local density
    texts = []
    pins_xy = []
    xs, ys = [], []

    for i, row in enumerate(pts_df.itertuples(index=False)):
        px, py = row.geometry.x, row.geometry.y
        pins_xy.append((px, py))
        xs.append(px); ys.append(py)

        # outward unit from group center
        ctr = centers.get(row.Group)
        if ctr is not None:
            vx, vy = (px - ctr.x, py - ctr.y)
            nrm = np.hypot(vx, vy) or 1.0
            ux, uy = vx / nrm, vy / nrm
        else:
            ux, uy = 1.0, 0.0

        # density-aware step: baseline + boost from local crowding
        step = SEED_BASE_STEP_M + SEED_BOOST_FACTOR * knn_dist[i]
        lx = px + step * ux
        ly = py + step * uy

        t = ax.text(
            lx, ly, row.City,
            fontsize=LABEL_FONTSIZE, color=text_color,
            ha="center", va="center", zorder=8
        )
        if LABEL_HALO:
            t.set_path_effects([pe.Stroke(linewidth=2.5, foreground="white"), pe.Normal()])
        texts.append(t)

    # --- run adjustText to separate labels and keep them off the pins
    adjust_text(
        texts,
        x=xs, y=ys,
        ax=ax,
        only_move={"points": "xy", "text": "xy"},
        autoalign="xy",
        expand_points=ADJUSTTEXT_EXPAND_PTS,
        expand_text=ADJUSTTEXT_EXPAND_TEXT,
        force_points=ADJUSTTEXT_FORCE_POINTS,
        force_text=ADJUSTTEXT_FORCE_TEXT,
        lim=ADJUSTTEXT_MAX_ITERS,
        save_steps=False,
    )

    # --- post-adjust clamp: keep labels within [min, max] distance of their pin
    for (px, py), t in zip(pins_xy, texts):
        tx, ty = t.get_position()
        dx, dy = (tx - px), (ty - py)
        d = np.hypot(dx, dy)
        if d == 0:
            # nudge outward a touch to avoid zero-length leaders
            t.set_position((px + LABEL_MIN_DIST_M, py))
            continue
        # clamp
        if d < LABEL_MIN_DIST_M or d > LABEL_MAX_DIST_M:
            dd = np.clip(d, LABEL_MIN_DIST_M, LABEL_MAX_DIST_M)
            scale = dd / d
            nx, ny = (px + dx * scale, py + dy * scale)
            t.set_position((nx, ny))

    # --- draw leader lines after final positions
    for (px, py), t in zip(pins_xy, texts):
        tx, ty = t.get_position()
        ax.plot([px, tx], [py, ty],
                color=leader_color, linewidth=LABEL_LINE_WIDTH,
                alpha=LABEL_LINE_ALPHA, zorder=7)
        
# ======================================================================
# Plot
# ======================================================================
# Use constrained layout instead of tight_layout + bbox_inches='tight'
fig, ax = plt.subplots(figsize=(13, 10), constrained_layout=True)

# Basemap choice (unchanged)
if BASEMAP.lower() == "dark":
    base_src  = cx.providers.CartoDB.DarkMatterNoLabels
    label_src = cx.providers.CartoDB.DarkMatterOnlyLabels
    edgecolor = "white"; textcolor = "white"; leader_color = "white"
elif BASEMAP.lower() == "voyager":
    base_src  = cx.providers.CartoDB.VoyagerNoLabels
    label_src = cx.providers.CartoDB.VoyagerOnlyLabels
    edgecolor = "black"; textcolor = "black"; leader_color = LABEL_LINE_COLOR
else:
    base_src  = cx.providers.CartoDB.PositronNoLabels
    label_src = cx.providers.CartoDB.PositronOnlyLabels
    edgecolor = "black"; textcolor = "black"; leader_color = LABEL_LINE_COLOR

# Colors per ordered group (unchanged)
cmap = matplotlib.colormaps["tab20"]
color_for = {grp: mcolors.to_hex(cmap(i % cmap.N)) for i, grp in enumerate(ordered_groups)}

# 🔁 Draw your vector layers FIRST (so axes have final extent)
for grp, sub in gdf_circles.groupby("Group", observed=False):
    base = color_for[grp]
    pale = lighten(base, 0.70)
    sub.plot(ax=ax,
             facecolor=mcolors.to_rgba(pale, CIRCLE_ALPHA_FILL),
             edgecolor=mcolors.to_rgba(base, CIRCLE_ALPHA_EDGE),
             linewidth=CIRCLE_EDGE_WIDTH,
             zorder=4)

for grp, sub in gdf_city_polys.groupby("Group", observed=False):
    base = color_for[grp]
    sub.plot(ax=ax,
             facecolor=mcolors.to_rgba(base, CITY_FILL_ALPHA),
             edgecolor=mcolors.to_rgba(base, CITY_EDGE_ALPHA),
             linewidth=CITY_EDGE_WIDTH,
             zorder=5)

for r in gdf_points.itertuples(index=False):
    px, py = r.geometry.x, r.geometry.y
    c = color_for[r.Group]
    ax.scatter([px], [py], s=PIN_MARKER_SIZE, marker="v",
               facecolors=c, edgecolors="white", linewidths=0.9, zorder=6)

# Labels
if LABEL_STRATEGY == "all_with_leaders":
    draw_labels_with_leaders(ax, gdf_points, color_for, textcolor, leader_color)
else:
    lbl = gdf_points.sort_values("City").groupby("Group").head(1)
    for grp, row in lbl.groupby("Group"):
        p = row.geometry.iloc[0]
        ax.text(p.x + 2000, p.y + 2000, row.City.iloc[0],
                fontsize=LABEL_FONTSIZE, color=textcolor, zorder=8)

# ✅ NOW add the basemap; it will use the current extent set by your data
cx.add_basemap(ax, source=base_src, zoom=BASEMAP_ZOOM)
if SHOW_LABELS:
    cx.add_basemap(ax, source=label_src, zoom=BASEMAP_ZOOM, zorder=3)

# Legend outside, and leave room with subplots_adjust (no tight bbox needed)
from matplotlib.lines import Line2D
handles = [Line2D([0], [0], marker="o", linestyle="",
                  markerfacecolor=color_for[g], markeredgecolor="none",
                  markersize=8, label=g)
           for g in ordered_groups]
leg = ax.legend(handles=handles, title="Groups",
                loc="upper left", bbox_to_anchor=(1.02, 1.0))
plt.subplots_adjust(right=0.82)

ax.set_title(f"Mapped Regions by Group (all labels + leaders, {BASEMAP.title()} basemap)", fontsize=15)
ax.set_xlabel(""); ax.set_ylabel("")
ax.set_aspect('equal', adjustable='box')

# Save & show
# ⛔ remove bbox_inches='tight' to avoid pathological bboxes
try:
    plt.savefig("grouped_regions_map.png", dpi=220, bbox_inches="tight")
except Exception:
    # fallback to normal save if tight bbox goes haywire
    plt.savefig("grouped_regions_map.png", dpi=220)
print("Saved PNG: grouped_regions_map.png")
plt.show()

# ======================================================================
# Exports (GeoJSON + Folium HTML)
# ======================================================================
# Export dissolved polygons as GeoJSON (WGS84)
gdf_polys.to_crs(4326).to_file("grouped_regions.geojson", driver="GeoJSON")
print("Saved GeoJSON: grouped_regions.geojson")

# Interactive HTML (folium)
if EXPORT_INTERACTIVE_HTML:
    try:
        import folium

        # Re-use plotted data bounds in 3857 → convert to 4326 for Folium
        xmin, ymin, xmax, ymax = (gpd.GeoSeries(pd.concat([
            gdf_circles.geometry, gdf_city_polys.geometry, gdf_points.geometry
        ], ignore_index=True), crs=3857).total_bounds)

        bounds_4326 = gpd.GeoSeries([box(xmin, ymin, xmax, ymax)], crs=3857).to_crs(4326)
        bminx, bminy, bmaxx, bmaxy = bounds_4326.total_bounds
        center_lat = (bminy + bmaxy) / 2.0
        center_lon = (bminx + bmaxx) / 2.0

        pts_ll     = gdf_points.to_crs(4326)
        circles_ll = gdf_circles.to_crs(4326)
        cities_ll  = gdf_city_polys.to_crs(4326)

        m = folium.Map(
            location=[center_lat, center_lon],
            tiles="CartoDB Positron" if BASEMAP.lower() != "dark" else "CartoDB DarkMatter",
            zoom_start=7,
            control_scale=True
        )

        # Pale group circles
        for grp, sub in circles_ll.groupby("Group", observed=False):
            base = color_for[grp]
            pale = lighten(base, 0.70)
            folium.GeoJson(
                sub.__geo_interface__,
                style_function=lambda feat, base=base, pale=pale: {
                    "color": base, "weight": CIRCLE_EDGE_WIDTH,
                    "fillColor": pale, "fillOpacity": CIRCLE_ALPHA_FILL
                },
                name=f"{grp} – circle"
            ).add_to(m)

        # City polygons
        for grp, sub in cities_ll.groupby("Group", observed=False):
            base = color_for[grp]
            folium.GeoJson(
                sub.__geo_interface__,
                style_function=lambda feat, base=base: {
                    "color": base, "weight": CITY_EDGE_WIDTH,
                    "fillColor": base, "fillOpacity": CITY_FILL_ALPHA
                },
                name=f"{grp} – cities"
            ).add_to(m)

        # Pins
        for r in pts_ll.itertuples(index=False):
            folium.CircleMarker(
                location=[r.geometry.y, r.geometry.x],
                radius=5,
                color="white", weight=1,
                fill=True, fill_color=color_for[r.Group], fill_opacity=0.95,
                tooltip=f"{r.City}, {r.State} — {r.Group}"
            ).add_to(m)

        m.fit_bounds([[bminy, bminx], [bmaxy, bmaxx]])
        folium.LayerControl(collapsed=True).add_to(m)
        m.save(INTERACTIVE_HTML_PATH)
        print(f"Saved interactive HTML: {INTERACTIVE_HTML_PATH}")
    except Exception as e:
        print("Interactive HTML export skipped. Error:", e)
