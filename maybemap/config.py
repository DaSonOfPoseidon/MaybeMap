# Shared configuration for all map sources and renderers.

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data sources
# Either a .csv or a .xlsx; the loader dispatches on the suffix
CITY_DATA_PATH = PROJECT_ROOT / "serviced_cities.csv"
KMZ_PATH = PROJECT_ROOT / "MO_Zones.kmz"

# Towns the export lists with no region. The regions map collects them under this
# label so the full service footprint still shows; the zone overlay drops them.
UNASSIGNED_GROUP_LABEL = "Unassigned"
# Neutral gray, so the real regions keep the tab20 palette to themselves
UNASSIGNED_GROUP_COLOR = "#9e9e9e"

# Map appearance
# "gray"/"darkgray" (Esri Canvas), "streets"/"topo"/"natgeo" (Esri's labelled maps)
# need no key. "voyager", "positron" and "dark" are CARTO's, which since August 2026
# stamp every tile with "API KEY REQUIRED" unless CARTO_API_KEY is set in .env.
BASEMAP = "gray"
# Tile zoom for the region map. "auto" spends the whole tile budget below, which is
# the sharpest basemap that still fits in memory whatever extent the data turns out
# to cover. A fixed number is honored only while it fits that budget.
BASEMAP_ZOOM = "auto"
# Place names from the basemap provider, drawn over the regions. The "gray",
# "darkgray" and CARTO themes pair a label-free base with a separate labels overlay
# and need this on to show town names; "streets", "topo" and "natgeo" bake their
# labels into the tiles and ignore it.
SHOW_LABELS = True
EXPORT_INTERACTIVE_HTML = True

# Bottom-right filter box on the interactive map, by customer count and service
# type. The region map is the one that shows the whole export and needs it; the
# zone map covers the regions only and stays clean.
REGION_MAP_CITY_FILTERS = True

# Zone maps cover a whole state, so let the tile budget pick the zoom
ZONE_BASEMAP_ZOOM = "auto"

# contextily assembles one array covering the whole axes extent, so at a fixed zoom
# the tile count grows with the square of the extent -- and matplotlib expands that
# mosaic again to rasterize it. A city list reaching both coasts at zoom 9 came to
# 2190 tiles and several GB, enough to push the run into cgroup memory throttling.
# 256 tiles is a 4096px mosaic, already finer than this figure renders at dpi 220,
# and it peaks around 1 GB.
MAX_BASEMAP_TILES = 256

# Most of the export carries no region, and those towns are scattered nationwide --
# a single unassigned dot in California stretched the PNG across the whole country,
# spending the tile budget on empty ocean and shrinking Missouri to a smudge. Only
# towns in these states set the static extent; the rest are simply out of frame.
# Regions and their circles are always covered whatever this says, and an empty set
# hands the extent back to every town. The interactive HTML is unaffected -- it pans.
PNG_EXTENT_STATES = {"MO", "IL", "TN"}
# Breathing room around that extent, as a fraction of its span
PNG_EXTENT_MARGIN = 0.04

# Typography hierarchy
LABEL_FONTSIZE_PRIMARY = 11
LABEL_FONTSIZE_SECONDARY = 9
LABEL_FONTSIZE_TERTIARY = 8
LABEL_STRATEGY = "professional_cartographic"
LABEL_HALO = True
LABEL_LINE_COLOR = "black"
LABEL_LINE_ALPHA = 0.7
LABEL_LINE_WIDTH = 0.9

# Label placement - much more aggressive for regional scale
LABEL_EXCLUSION_RADIUS_M = 12000  # 12km exclusion zones
MAX_LABEL_DISTANCE_M = 80000  # 80km maximum leader lines for very dense areas

# OSM data fetching
SEARCH_RADIUS_M = 40000
MAX_CANDIDATE_AREA_M2 = 3_000_000_000
POINT_FALLBACK_BUFFER_M = 5000

# A failed Overpass request must never be mistaken for "this town has no boundary",
# which silently turns into a POINT_FALLBACK_BUFFER_M circle that looks real.
OSM_FETCH_RETRIES = 3  # attempts per query before the run gives up
OSM_FETCH_RETRY_PAUSE_S = 5.0  # base pause between attempts, scaled by attempt number
# Genuine coverage gaps are scattered; an unbroken run of them means the lookup is
# broken, not that Missouri stopped having city limits.
OSM_MAX_CONSECUTIVE_FALLBACKS = 8

# Visual styling
CITY_FILL_ALPHA = 0.32
CITY_EDGE_ALPHA = 0.75
CITY_EDGE_WIDTH = 0.7
CIRCLE_PAD_M = 3000
CIRCLE_EDGE_WIDTH = 2.0
CIRCLE_ALPHA_FILL = 0.15
CIRCLE_ALPHA_EDGE = 0.9

# Zone (KMZ) styling
ZONE_FILL_ALPHA = 0.45
ZONE_EDGE_ALPHA = 0.95
ZONE_EDGE_WIDTH = 1.4
ZONE_LABEL_FONTSIZE = 14

# Drop the color legend when zones are labeled directly on the polygons
SHOW_LEGEND = False

# Override any zone color read from the KML PolyStyle, keyed by zone label
ZONE_COLOR_OVERRIDES: dict[str, str] = {}

# Polygon processing
AREA_STYLE = "union_smooth"  # "union", "union_smooth", "convex_hull"
MERGE_GAP_M = 2000
SIMPLIFY_M = 600
MIN_PART_M2 = 1_000_000

# City overlay on zone maps (pulled from CITY_DATA_PATH when that file exists)
ZONE_MAP_SHOW_CITIES = True
ZONE_MAP_CITY_LABELS = False  # dots only; True also runs the full label placement system
ZONE_MAP_CLIP_CITIES_TO_ZONES = True  # drop cities that fall outside the zone polygons
CITY_DOT_COLOR = "zone"  # "zone" matches the zone fill, "group" uses the spreadsheet Group palette

# City dot sizing, by the classification in labeling.classify_city_importance
CITY_DOT_SIZE_PRIMARY = 90
CITY_DOT_SIZE_SECONDARY = 55
CITY_DOT_SIZE_TERTIARY = 32
CITY_DOT_EDGE_COLOR = "white"
CITY_DOT_EDGE_WIDTH = 1.2

# Minimum enclosing circle per spreadsheet region, drawn inside its zone
# Real OSM city shapes in place of dots; a city with no boundary keeps its dot
ZONE_MAP_CITY_BOUNDARIES = True
# 0 keeps every boundary however small it renders. Raise it to hold the dot for
# cities too small to see at map scale -- at the statewide extent a boundary needs
# roughly 20 km² to match the dot it replaces.
ZONE_MAP_MIN_BOUNDARY_KM2 = 0.0
# Boundaries sit on top of the zone fill, so they carry a heavier edge than the
# zones do to stay legible at statewide scale
CITY_BOUNDARY_FILL_ALPHA = 0.55
CITY_BOUNDARY_EDGE_WIDTH = 1.2

ZONE_MAP_REGION_CIRCLES = True
ZONE_MAP_REGION_CIRCLE_LABELS = True
ZONE_CIRCLE_LABEL_FONTSIZE = 9
# How far above the circle's top edge the region name sits, as a fraction of its radius
ZONE_CIRCLE_LABEL_OFFSET = 0.06

# Task map (one pin per open task, from the task export)
TASK_DATA_PATH = PROJECT_ROOT / "Tasks.csv"
# The task export carries no State column; every county in it is a Missouri county
TASK_MAP_STATE = "MO"
# Legend order, and the order the layer control lists the status layers in
TASK_STATUS_ORDER = ["Active", "Waiting", "Hold", "Incomplete"]
TASK_STATUS_COLORS = {
    "Active": "#2e7d32",  # green: work is moving
    "Waiting": "#f9a825",  # amber: queued
    "Hold": "#546e7a",  # slate: parked
    "Incomplete": "#c62828",  # red: fell over
}
# A task is a work item, not a place, so every pin carries the same weight --
# the count in a city reads off pin density rather than dot size.
TASK_DOT_SIZE = 60  # matplotlib scatter area
TASK_MARKER_RADIUS = 6  # folium CircleMarker radius
# Labels name each city once rather than each of its tasks
TASK_LABEL_CITIES = True
# How far apart to push pins that geocoded to the same point, so none hides another
TASK_COINCIDENT_OFFSET_M = 120
# An address that lands this far from the town it names is a bad match, not a rural
# address: "65 S Highway 47, Hawk Point" matched a Highway 47 outside St Louis, 155km
# away and still in Missouri, so the state guard never saw it. Such a task falls back
# to its town center like any other address the geocoder cannot place. The widest
# genuine offset in the export is a rural Columbia address at 18km.
TASK_MAX_ADDRESS_OFFSET_KM = 40
