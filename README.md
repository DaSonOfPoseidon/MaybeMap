# MaybeMap - Professional Regional Mapping Tool

A Python application that creates professional-quality regional maps for business presentations and internal reference. It reads three kinds of input:

- **CSV or Excel** city lists (City / State / Region) — geocoded and given OSM boundaries
- **KMZ** zone files (e.g. an ArcGIS export) — polygons used exactly as authored
- **CSV** task exports (one row per work item) — geocoded to the street address and pinned by status

## Overview

MaybeMap transforms either input into polished, executive-ready maps with:
- **Geographic-aware smart labeling** (labels placed toward empty space)
- **Numbered dense area system** (organized spiral placement for clustered cities)
- **Professional cartographic labeling** (zero overlapping text guaranteed)
- **Hierarchical typography** (major cities emphasized)
- **Interactive HTML maps** for digital exploration
- **High-resolution PNG exports** for print/presentations

## Quick Start

### Prerequisites
- Python 3.14 (dependencies ship PEP 758 syntax that older versions cannot parse)
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- A [Geoapify API key](https://www.geoapify.com/) — **only** for the city map; KMZ maps need no key

### Installation
```bash
uv sync --python 3.14
```

For the city map, also create a `.env` file:
```
GEOAPIFY_KEY=your_api_key_here
```

The basemap needs no key on the default setting — see [Basemaps](#basemaps).

### Running

Zone map from a KMZ — fast, no API key, no network beyond basemap tiles:
```bash
uv run python zoneMap.py
```

City map from a spreadsheet:
```bash
uv run python maybeMap.py
```

Task map from the work-item export:
```bash
uv run python taskMap.py
```

**The city map's first run takes a while** — it geocodes every town and fetches an OSM boundary for each one that belongs to a region. Subsequent runs are much faster due to caching. The task map geocodes one address per task on its first run, which is quick, and caches those too.

### Quality checks
```bash
uv run ruff check . && uv run black --check . && uv run pytest
```

## Data Format

Point `CITY_DATA_PATH` in `maybemap/config.py` at either a `.csv` or an `.xlsx`;
the loader dispatches on the suffix and normalizes both to the same shape. It must
contain these columns:
- **City**: City name (e.g., "Boonville" or "BOONVILLE" — case is normalized)
- **State**: State name or two-letter code — both work (e.g., "Missouri" or "MO")
- **Region** (or **Region Name** / **Group**): Regional grouping (e.g., "Boonville-Fayette")

- **County** (optional): Used to disambiguate boundary lookups. Missouri has many
  cities sharing a name with a county — "Clark, Missouri" matches Clark County
  ahead of the village in Randolph County — so the county is included in the
  lookup when present, falling back to the plain query if it finds nothing.

- **Customers** (optional): Per-town customer count, shown in the interactive popups
  and driving the customer filter on the region map.

- **Service Mix** (optional): `Business Only`, `Residential Only`, or `Mixed`. Shown in
  the popups and driving the service-type filter. Any other value leaves the town out
  of that filter rather than making a fourth category of it.

- **Active Addresses**, **Tasks 6 Mo**, **Residential Customers**, **Business Customers**
  (optional): Extra per-town detail, shown in the interactive popups. The residential /
  business split is only shown when both halves are present.

Column matching is case-insensitive, and any other extra columns are ignored.

Example:
```
City,County,State,Region,Customers,Active Addresses,Tasks 6 Mo,Service Mix,Residential Customers,Business Customers
BOONVILLE,COOPER,MO,Boonville-Fayette,3,3,7,Mixed,2,1
EDWARDSVILLE,MADISON,IL,Illinois Metro East,2,2,0,Residential Only,2,0
POCAHONTAS,RANDOLPH,AR,,1,1,0,Business Only,0,1
```

Groups named `Group 1`, `Group 2`, ... are ordered numerically; any other naming
keeps spreadsheet order.

### Towns with no region

The MBS export lists every serviced town, and most carry no region at all. On the
city map those are collected under a single `Unassigned` group (`UNASSIGNED_GROUP_LABEL`),
drawn as neutral gray dots so the full service footprint stays visible. They deliberately
get no OSM boundary lookup — that would be hundreds of extra Overpass round trips —
and no enclosing circle, since their towns are scattered across several states.

`zoneMap.py` drops them entirely; a zone map is about the regions.

### KMZ zone files

Drop the `.kmz` in the project root and point `KMZ_PATH` in `maybemap/config.py` at it. The loader reads each Placemark's polygon and its `PolyStyle` color, and takes the zone name from the first of these it finds:

1. a non-empty `<name>` element
2. an `Area` / `Zone` / `Name` / `Region` row in the `<description>` HTML table — ArcGIS exports put attributes here rather than in `<ExtendedData>`
3. the Placemark's `id`, as a last resort

`MO_Zones.kmz` yields three zones: MO West (blue), MO East (orange), MO Central (purple). Override any color with `ZONE_COLOR_OVERRIDES` in `maybemap/config.py`.

### Task exports

Point `TASK_DATA_PATH` in `maybemap/config.py` at the CSV. One row per open task, and unlike the city list it needs no `State` column — every row is read as `TASK_MAP_STATE`. Required columns:

- **City**, **Task Status**, **Street Address**

Everything else is optional and shows in the pin's popup when present: `Subscriber Name`, `Task ID`, `Person Assigned`, `Region`, `County`, `Due Time`, and `Service`. `Service` holds either a phone number or the full address; where it holds the address, the ZIP is read out of it and added to the geocode query.

Status drives the color, so `TASK_STATUS_COLORS` and `TASK_STATUS_ORDER` are what to edit to change the legend. A status the export invents that is not in `TASK_STATUS_ORDER` still appears on the map, in gray, after the known ones.

**Addresses that cannot be trusted fall back to the town center**, flagged as such in the popup. That covers an address the geocoder cannot place at all, and one it places more than `TASK_MAX_ADDRESS_OFFSET_KM` from the town the task names — `65 S Highway 47, Hawk Point` matched a Highway 47 outside St Louis, 155km away and still in Missouri, so the state check never saw it.

## Output Files

### 🖼️ Static Map (`grouped_regions_map.png`)
- High-resolution PNG (220 DPI)
- **Geographic-aware label placement** toward empty spaces
- **Numbered dense areas** with spiral arrangement and reference legend
- **Zero overlapping labels** guaranteed for professional appearance
- **Manual positioning overrides** for specific cities
- Perfect for presentations, reports, and printing

### 🌐 Interactive Map (`grouped_regions_map.html`)
- Click cities for detailed information popups, including the customer count and the
  residential / business breakdown
- Toggle layers on/off
- Filter every town by customer count and service type (see below)
- Fullscreen presentation mode
- Professional legend and controls
- Perfect for digital exploration and meetings

#### City filters

A box in the bottom-right corner carries two sets of checkboxes, each with the
number of towns behind it:

- **Customers** — `1-5`, `6-10`, `11-20`, `21+`. Most of the export is
  single-customer towns, so clearing `1-5` is usually enough to make the map
  readable.
- **Service type** — `Business`, `Residential`, `Mixed`.

The two apply together: a town shows only when its customer bucket **and** its
service type are both checked. Every town on the map is covered, whether or not it
belongs to a region. A town the export gave no customer count is never hidden, and
one with no service mix is only subject to the customer buckets.

The checkboxes deliberately do not appear in Leaflet's layer list, which already
carries a row per region and per circle; they get their own box instead. Edit
`CUSTOMER_BUCKETS` in `maybemap/interactive.py` to change the cut points — they are
inclusive `(low, high)` pairs and the last one is always open-ended, or set
`REGION_MAP_CITY_FILTERS = False` in `maybemap/config.py` to drop the box entirely.

The zone map never carries the box: it covers the regions only and stays clean, so
its customer data is reached by clicking a town instead.

### 📊 Data Export (`grouped_regions.geojson`)
- Geographic data in standard GeoJSON format
- Compatible with GIS software and web applications

### 🗺️ Zone Map (`mo_zones_map.png` / `.html` / `mo_zones.geojson`)
- Produced by `zoneMap.py` from a KMZ
- Each zone filled in its authored color and labeled at the polygon's interior point
- City dots overlaid from the spreadsheet when it is available (see below)
- Same interactive HTML treatment: layer toggles, tooltips, fullscreen

### 📍 Task Map (`tasks_map.png` / `.html` / `tasks.geojson`)
- Produced by `taskMap.py` from the work-item export
- One pin per open task at its street address, colored by status, on a plain basemap
- Uniform pins: a task is a work item, not a place, so the count in a town reads off pin density
- Each town is named once however many tasks it holds
- The interactive map carries one layer per status, so the layer control filters the backlog
- `tasks.geojson` exports the pins themselves, with every column the popup shows

## City Overlay on Zone Maps

`zoneMap.py` plots city dots from `CITY_DATA_PATH` on top of the zones. It is entirely optional — if the spreadsheet is missing, or no Geoapify key is configured and the geocode cache cannot answer, the run prints why and draws the zones alone.

Each city is tagged by spatial join with the zone containing it, and cities outside every zone (Illinois, for example) are dropped so they do not stretch the map.

Cities are drawn as their real OSM municipal boundary, in the colour the dot would have been. A city with no boundary in OSM keeps a dot instead, sized by the same Primary/Secondary/Tertiary classification the city map uses. The interactive version keeps the clickable popups on both, and they now also name the zone.

This means `zoneMap.py` reads `osm_boundaries_cache.json` and may call Nominatim for cities it has not seen — set `ZONE_MAP_CITY_BOUNDARIES = False` to keep it fully offline. A broken lookup is not fatal: it prints why and falls back to dots.

```python
ZONE_MAP_SHOW_CITIES = True           # set False for zones only
ZONE_MAP_CITY_LABELS = False          # True also runs the full label placement system
ZONE_MAP_CLIP_CITIES_TO_ZONES = True  # False keeps out-of-zone cities
CITY_DOT_COLOR = "zone"               # or "group" for the spreadsheet Group palette

ZONE_MAP_REGION_CIRCLES = True        # enclosing circle per spreadsheet region
ZONE_MAP_REGION_CIRCLE_LABELS = True  # name each circle on the PNG

ZONE_MAP_CITY_BOUNDARIES = True       # real OSM city shapes; False for dots only
ZONE_MAP_MIN_BOUNDARY_KM2 = 0.0       # keep the dot below this area (0 = never)
```

At the statewide extent the PNG is about 260 m per pixel, so a boundary needs roughly
20 km² to match the size of the dot it replaces — 23 of the 51 mapped cities are smaller
than that, and Hartsburg is barely two pixels across. Raise `ZONE_MAP_MIN_BOUNDARY_KM2`
to hold the dot for cities too small to see.

`CITY_DOT_COLOR = "zone"` darkens each dot against the zone fill it sits on, since they share a hue. `"group"` colors dots by their spreadsheet Group instead and adds those groups to the legend.

On the interactive map the hover tooltip resolves finest-first: anywhere inside a city's boundary names the city and holds for the whole shape, a city dot names the city, a region circle names the region, and anywhere else in a zone names the zone. Circles are drawn largest to smallest so a small circle nested inside a larger one — Columbia Core inside Jefferson City Area — stays reachable, and they are muted by the zone's own fill alpha so moving them above the zones does not change how the map looks.

## Features

### 🎯 Advanced Label Placement System
- **Geographic-aware positioning** - Labels placed toward empty spaces using 8-directional density analysis
- **Edge city intelligence** - Cities near map edges get inward-facing labels
- **Interior city optimization** - Labels directed toward least dense areas
- **Numbered dense areas** - Cities with 16+ neighbors get organized numbered placement
- **Spiral arrangement** - Numbered labels arranged in professional spiral pattern
- **Zero overlapping labels** - Guaranteed readable text placement

### 🏗️ Professional Cartography
- **Hierarchical typography** - Major cities in bold, larger fonts
- **Adaptive collision detection** - Relaxed separation for very dense areas
- **Manual positioning overrides** - Fine-tune specific city placements
- **Clean leader lines** - Professional connecting lines with optimal routing
- **Density-first processing** - Dense area cities get priority placement

### 🏙️ City Classification System
- **Primary Cities**: Major metropolitan areas (largest markers/fonts)
- **Secondary Cities**: State capitals and significant cities  
- **Tertiary Cities**: All other cities
- **Dense Area Cities**: Automatically numbered when 16+ neighbors detected

### 🔄 Smart Data Processing
- **Automatic geocoding** via Geoapify API
- **OSM boundary fetching** for accurate city shapes
- **Intelligent caching** for fast subsequent runs
- **Fallback systems** for missing data
- **Density-based sorting** for optimal placement order

## Basemaps

`BASEMAP` in `maybemap/config.py` picks the tiles every map draws on. The same setting
drives both the PNG and the interactive HTML, so the two can never disagree.

| Name | Tiles | Key needed | Place names |
|---|---|---|---|
| `gray` *(default)* | Esri Light Gray Canvas | no | separate overlay |
| `darkgray` | Esri Dark Gray Canvas | no | separate overlay |
| `streets` | Esri World Street Map | no | in the tiles |
| `topo` | Esri World Topo | no | in the tiles |
| `natgeo` | Esri National Geographic | no | in the tiles |
| `voyager` | CARTO Voyager | **yes** | separate overlay |
| `positron` | CARTO Positron | **yes** | separate overlay |
| `dark` | CARTO Dark Matter | **yes** | separate overlay |

Attribution is rendered automatically by contextily on the PNG and by folium on the
HTML — leave it in place.

### Place names

The Canvas and CARTO themes ship a label-free base paired with a transparent
labels-only overlay, and **`SHOW_LABELS` is what draws that overlay** — with it off,
those basemaps carry no town names at all. The `streets`, `topo` and `natgeo` themes
bake their labels into the tiles instead, so they ignore the setting rather than
printing every name twice.

At statewide extents the overlay's text is small, because it renders at the tile's
own resolution rather than the figure's. If you want prominent town names across a
whole state, a label-baked theme reads better than `gray` with the overlay on.

### The CARTO key

CARTO stopped serving its raster basemaps anonymously in August 2026. Without a key
every tile comes back stamped **"API KEY REQUIRED"** across the map; the run prints a
warning saying so rather than quietly producing a watermarked file. To use a CARTO
theme, put the key in `.env`:

```
CARTO_API_KEY=your_key_here
```

Note that CARTO's **free** tier is licensed for non-commercial use only — personal
projects, research, teaching, nonprofits. A business's internal tooling is outside
that, whether or not the maps are ever published, so the free key is not the right
answer for this repo's usual job. The default `gray` theme needs no key at all.

### On the Esri tiles

Esri serves the Canvas basemaps publicly under its Master License Agreement, with
attribution. Worth knowing: the item metadata says the standard layer "is not intended
to be used to export tiles for offline", and the export-licensed alternative is a
*vector* tile service requiring an ArcGIS key, which this raster pipeline cannot use.
Live tiles in the interactive HTML are squarely the intended use; the static PNG is
the grey area. Esri Canvas also stops at zoom 16 — the renderer caps its own zoom at
whatever the chosen provider serves, so a small extent renders coarser rather than
fetching 404s.

## Configuration

All settings live in `maybemap/config.py`:

```python
# Map styling
BASEMAP = "gray"                 # see Basemaps below; only the CARTO themes need a key
EXPORT_INTERACTIVE_HTML = True

# Static PNG extent and basemap resolution
PNG_EXTENT_STATES = {"MO", "IL", "TN"}  # only these states frame the PNG; set() = all
PNG_EXTENT_MARGIN = 0.04         # breathing room around that extent
BASEMAP_ZOOM = "auto"            # "auto" = sharpest zoom the tile budget allows
MAX_BASEMAP_TILES = 256          # hard ceiling on basemap memory

# Professional typography
LABEL_FONTSIZE_PRIMARY = 11      # Major cities
LABEL_FONTSIZE_SECONDARY = 9     # Medium cities
LABEL_FONTSIZE_TERTIARY = 8      # Small cities

# Advanced label placement
MAX_LABEL_DISTANCE_M = 80000     # Maximum leader line length (80km)
dense_area_threshold = 16        # Cities with 16+ neighbors get numbered

# Zone (KMZ) maps
ZONE_FILL_ALPHA = 0.45
ZONE_LABEL_FONTSIZE = 14
ZONE_COLOR_OVERRIDES = {}        # e.g. {"MO West": "#123456"}
SHOW_LEGEND = False              # Zones are labeled on the map itself
```

## File Structure

```
MaybeMap/
├── maybeMap.py                  # Entrypoint: CSV/Excel city map
├── zoneMap.py                   # Entrypoint: KMZ zone map
├── taskMap.py                   # Entrypoint: task export map
├── maybemap/
│   ├── config.py                # All tunable settings
│   ├── styling.py               # Colors, KML color decoding, basemap themes
│   ├── labeling.py              # Cartographic label placement for city maps
│   ├── geocoding.py             # Geoapify + OSM lookups and caches
│   ├── render.py                # Static PNG output
│   ├── interactive.py           # Folium HTML + GeoJSON output
│   └── sources/
│       ├── cities.py            # CSV or Excel -> MapData
│       ├── kmz.py               # KMZ -> MapData
│       └── tasks.py             # Task export CSV -> MapData
├── tests/                       # conftest fixtures + kmz, overlay, interactive tests
├── pyproject.toml               # Dependencies and tooling config
├── serviced_cities_v2.csv       # City input, CSV or xlsx (not tracked)
├── Tasks.csv                    # Task input (not tracked)
├── MO_Zones.kmz                 # KMZ input
├── .env                         # API key for the city map (create this)
├── city_geocodes_cache.json     # Geocoding cache (auto-generated)
├── osm_boundaries_cache.json    # Boundary cache (auto-generated)
├── address_geocodes_cache.json  # Task address cache (auto-generated)
├── grouped_regions_map.{png,html}   # Output: city map
├── mo_zones_map.{png,html}          # Output: zone map
└── tasks_map.{png,html}             # Output: task map
```

## CLI Progress Output

The application provides real-time progress feedback:

```
Starting geocoding process for 61 unique cities...
   Found coordinates: 38.6270, -90.1994
...
Geocoding complete! Processed 61 cities.

Starting OSM boundary fetching for 61 cities...
   Found OSM boundary polygon (area: 162.45 km²)
...
OSM boundary fetching complete! Processed 61 cities.

Generating professional cartographic map...
Applying professional cartographic labeling system...
   Map analysis: 58 cities analyzed
      East Alton: Found preferred position at 8000m after 1 attempts
   Numbered Cahokia Heights: #1 pin(-10040052, 4660327) -> label(-10018112, 4672313)
   Placing Columbia: pin(-10040995, 4642170) -> label(-10065995, 4617170)
   Professional labeling complete - no overlapping labels!
```

## Troubleshooting

### Common Issues

**"Missing GEOAPIFY_KEY" error**
- Create a `.env` file with your API key
- Get a free key at [geoapify.com](https://www.geoapify.com/)
- Only `maybeMap.py` needs this; `zoneMap.py` runs without a key

**`SyntaxError` on an unparenthesized `except`**
- The venv is on a Python older than 3.14; rebuild it with `uv sync --python 3.14`

**Slow first run**
- Normal behavior - building geocoding and boundary caches
- Subsequent runs will be much faster
- You can monitor progress in the CLI output

**Cities not found**
- Check city name spelling in the city data file
- Verify state abbreviations are correct (two letters)
- Some small towns may not be in the geocoding database

**Empty or missing boundaries**
- Script automatically falls back to circular buffers around points
- This is normal for smaller cities without detailed OSM data

## Use Cases

### Business Applications
- **Regional sales territories** mapping
- **Market analysis** visualization  
- **Operations planning** reference maps
- **Executive presentations** with professional appearance
- **Internal documentation** with interactive exploration

### Technical Applications
- **Geographic data visualization** from spreadsheets
- **Quick prototyping** of regional analysis
- **Data validation** through visual inspection

## API Dependencies

Only the city map calls out to these:

- **Geoapify**: City geocoding (latitude/longitude lookup)
- **Nominatim**: City boundary polygons, looked up by name (via OSMnx)
- **Overpass**: Fallback boundary search by proximity, for places Nominatim cannot
  name-match. Deliberately a last resort — each query pulls thousands of features to
  keep one polygon, which is what the Overpass usage policy asks you to avoid.
- **Esri / CARTO**: Base map tiles — see [Basemaps](#basemaps) for which need a key

## Development

See `CLAUDE.md` for detailed development guidance and architecture notes.

## License

Internal business tool - modify as needed for your organization's requirements.