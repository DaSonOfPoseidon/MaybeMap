# MaybeMap

Builds **one interactive HTML map** of the service territory. Zones, city limits, every
serviced town, and dispatch drive times are layers on the same map, and each one
switches on and off independently, so you can stack whatever you need at the time.

```bash
uv sync --python 3.14
uv run python buildMap.py        # -> output/maybe_map.html
```

The output is a single self-contained file. Open it in a browser, or send it to someone.
It needs internet access only for Leaflet and the basemap tiles. Drive times are baked in.

## Folders

```
buildMap.py              # entrypoint
maybemap/                # the build
  config.py              # every path and setting
  sources/               # readers: KMZ zones, city export, dispatch roster
  geocoding.py           # Geoapify + OSM boundary lookups (cached)
  routing.py             # OSRM drive-time matrix (cached)
  payload.py             # everything the page draws, as one JSON document
  page.py                # inlines payload + web/ into the single HTML file
  web/                   # the page: template.html, app.css, app.js (Leaflet 1.9.4)
tests/
data/     (not tracked)  # inputs you maintain
cache/    (not tracked)  # lookups the build can always redo
output/   (not tracked)  # maybe_map.html
```

`data/`, `cache/` and `output/` are gitignored. The data is business data, and the other
two are rebuilt by the build. Deleting `cache/` is safe; the next build just takes longer.

## Inputs (`data/`)

| File | What it is |
|---|---|
| `serviced_cities.csv` | The MBS cities query: `City, State, County, Region, Latitude, Longitude, Customers, Residential Customers, Business Customers, Service Mix, Active Addresses, Tasks 6 Mo`. One row per town. An `.xlsx` works too; point `CITY_DATA_PATH` at it. |
| `MO_Zones.kmz` | The zone polygons, exported from ArcGIS. Colors come from the KML styles. |
| `tech_info.csv` | The MBS techs query: `Tech, Town, State, Active, Tech ID, Primary Region, Departments`. Town may be blank (see below). Rows with `Active` = N are skipped. A plain `Tech, Town, State` roster also works. |
| `dispatch_staging.csv` | `Type, Town, State`: one row per staging site. Each Type becomes its own switch. |

Column names are matched case-insensitively. The roster CSVs are optional; without them
the dispatch layer routes town to town with no techs, and there is no staging section.

A town with `Latitude`/`Longitude` in the export is placed exactly there. A town without
them, or with impossible values, is geocoded instead. Geocoding a town the cache has not
seen needs a [Geoapify](https://www.geoapify.com/) key in `.env` (see `.env.example`).
Nothing else needs a key.

## Layers

Every section of the sidebar is a layer with its own switch. The **Territory**,
**Coverage** and **Dispatch** buttons in the top bar are shortcuts that flip those
switches. They are not separate views.

- **Zones**: the KMZ zones in their own colors. Each zone has its own switch, and there
  are switches for labels and fill strength.
- **City outlines**: real OSM city limits for every town that has a region. They are
  shaded in their zone's color while Zones is on, and in their region's color otherwise.
- **City dots**: a dot for every serviced town, in its region's color, sized by customer
  count. Towns with no region are gray.
- **Town filters**: customer count, service type and region. These apply to both dots
  and outlines.
- **Dispatch**: pins for every town that has a region, plus every tech's home town. Tech
  pins are orange. Route lines can show each town's closest towns, its nearest tech, or
  the selected town's three closest, all labeled with road miles and drive time.
- **Staging sites**: badges on the staging towns, one switch per type.
- **Basemap**: Light gray, Satellite or Streets, with optional place names.

Click any dot, outline or pin to open its card. The card shows customer detail, region and
zone, the 3 closest dispatch towns and 2 nearest techs by drive time, Google Maps links,
and **notes**. Notes are saved in that browser only; use **Export notes CSV** to back them
up or share them, and **Import** to load them elsewhere. Imports also accept the old
dispatch map's `Market, Notes` export.

Layer settings are remembered per browser, so the map reopens the way it was left.

## Which towns are in the dispatch set

A town is in the dispatch set if it has a Region in the export, or if it is a tech's
home town. Tech towns are always added, so a technician is never unroutable. The build
prints which tech towns had no region. To add any other town to dispatch, give it a
region in the export.

A tech with no Town is placed at the **hub of their Primary Region**: the region's most
central town, meaning the one with the least total straight-line distance to the
region's other towns. Their card says so, since drive times from there are an estimate
of where they start. A tech with neither a Town nor a known region can't be placed. The
build names them, and so does the dispatch section of the page.

The **Departments** buttons in the dispatch section choose which techs count. Tech pins,
labels and "nearest tech" results only consider techs in a selected department, so with
only TC on, "nearest tech" means the nearest TC tech.

## Drive times

Drive times come from OSRM's public `table` service, using the fastest route between town
centers with no live traffic. They are fetched **at build time** and cached per pair in
`cache/osrm_matrix.json`. A rebuild with the same towns makes no requests. A new town
only needs its own row and column fetched, sent in blocks of up to 50 places
(`OSRM_CHUNK`), so with around 90 towns that comes to about three requests. If OSRM cannot answer, the build stops
rather than write a map with missing drive times.

The route lines are straight connectors. They are not road shapes.

## City boundaries

Outlines come from Nominatim (with the county, to separate cities from same-named
counties), with an Overpass fallback. They are cached in `cache/osm_boundaries.json`.
Missing a few boundaries is normal, and those towns just keep their dot. If Overpass (the
fallback) is unreachable, the affected towns are drawn as dots for that build only. The
build names them in a warning and asks again next time, and each request waits at most
`OSM_REQUEST_TIMEOUT_S` (30 s). A long unbroken
run of misses stops the build, because it means the lookup is broken
(`OSM_MAX_CONSECUTIVE_FALLBACKS`).

## Quality checks

```bash
uv run ruff check . && uv run black --check . && uv run pytest
```

The tests stub every network call. Business data never enters the tests: the KMZ tests use a generated KMZ in the same ArcGIS format.
