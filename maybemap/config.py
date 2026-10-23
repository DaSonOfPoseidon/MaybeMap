# Shared configuration for the map build.

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Inputs you maintain, lookups the build can always redo, and the finished map
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Data sources
# Either a .csv or a .xlsx; the loader dispatches on the suffix
CITY_DATA_PATH = DATA_DIR / "serviced_cities.csv"
KMZ_PATH = DATA_DIR / "MO_Zones.kmz"
# Tech,Town,State -- one row per technician, at the town they dispatch from
DISPATCH_TECHS_PATH = DATA_DIR / "dispatch_techs.csv"
# Type,Town,State -- one row per staging site, e.g. Generator,Warrensburg,MO
DISPATCH_STAGING_PATH = DATA_DIR / "dispatch_staging.csv"

OUTPUT_PATH = OUTPUT_DIR / "maybe_map.html"
MAP_TITLE = "Missouri Service Map"

# Towns the export lists with no region. They still get a dot so the whole service
# footprint shows, but no boundary lookup and no place in the dispatch set.
UNASSIGNED_GROUP_LABEL = "Unassigned"
UNASSIGNED_GROUP_COLOR = "#9e9e9e"

# One color per region, assigned in region order. Twenty well-separated hues --
# tab20's light/dark pairs were too easy to confuse once regions sat side by side.
REGION_PALETTE = [
    "#1f77b4",
    "#e6550d",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#17becf",
    "#bcbd22",
    "#393b79",
    "#f5a623",
    "#00897b",
    "#ad1457",
    "#6d4c41",
    "#3f51b5",
    "#7cb342",
    "#c2185b",
    "#0097a7",
    "#ff7043",
    "#5e35b1",
]

# Customer-count filter buckets: inclusive (low, high) pairs, the last open-ended.
# They start at 0 so a town the export lists with no customers is still filterable;
# only a town with no count at all escapes the filter.
CUSTOMER_BUCKETS = [(0, 5), (6, 10), (11, 20), (21, None)]
# Service Mix values the filter knows; anything else is left out of that filter
SERVICE_TYPES = {
    "Business Only": "business",
    "Residential Only": "residential",
    "Mixed": "mixed",
}

# Override any zone color read from the KML PolyStyle, keyed by zone label
ZONE_COLOR_OVERRIDES: dict[str, str] = {}

# City outlines are simplified to this tolerance before they are embedded. Well
# under a pixel at statewide zoom, and it keeps the page a few hundred kB.
OUTLINE_SIMPLIFY_M = 50
# Coordinates are snapped to this grid in degrees (~1 m) for the same reason
COORD_PRECISION = 1e-5

# OSM data fetching
SEARCH_RADIUS_M = 40000
MAX_CANDIDATE_AREA_M2 = 3_000_000_000

# A failed Overpass request must never be mistaken for "this town has no boundary"
OSM_FETCH_RETRIES = 3  # attempts per query before the run gives up
OSM_FETCH_RETRY_PAUSE_S = 5.0  # base pause between attempts, scaled by attempt number
# Genuine coverage gaps are scattered; an unbroken run of them means the lookup is
# broken, not that Missouri stopped having city limits.
OSM_MAX_CONSECUTIVE_FALLBACKS = 8

# Drive times come from OSRM's table service, fetched at build time and cached per
# pair, so the page never depends on the public demo server being up.
OSRM_BASE_URL = "https://router.project-osrm.org"
# The demo server caps a table request at 100 coordinates; sources and destinations
# are chunked so each request carries at most twice this many.
OSRM_CHUNK = 50
OSRM_RETRIES = 3
OSRM_RETRY_PAUSE_S = 5.0
