# Reads the serviced-city export (City / State / Region rows, CSV or .xlsx),
# geocodes each town, and fetches OSM municipal boundaries for the ones that
# belong to a region.

import re
from pathlib import Path

import geopandas as gpd
import pandas as pd

from maybemap.config import (
    CITY_DATA_PATH,
    OSM_MAX_CONSECUTIVE_FALLBACKS,
    UNASSIGNED_GROUP_LABEL,
)
from maybemap.geocoding import (
    NOT_CACHED,
    FallbackUnavailable,
    cache_osm_polygon,
    fetch_osm_city_polygon,
    geocode_city_state,
    get_cached_osm_polygon,
)
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
    # Town-center coordinates, when the export carries them. A town without them
    # (or with an impossible value) is geocoded instead.
    "latitude": ("SourceLat", "Float64"),
    "longitude": ("SourceLon", "Float64"),
}
COORD_LIMITS = {"SourceLat": 90.0, "SourceLon": 180.0}


def _group_num(s: str) -> int:
    # The catch-all is not a region, so it sorts behind every real one and takes
    # the last palette slot rather than displacing a named region's color.
    if s == UNASSIGNED_GROUP_LABEL:
        return 10**9 + 1
    m = re.search(r"Group\s*(\d+)", s or "", re.IGNORECASE)
    return int(m.group(1)) if m else 10**9


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
    # states, so a town reads the same whichever export it came from.
    df["City"] = _title_case(df["City"])
    df["County"] = _title_case(df["County"])
    df["State"] = df["State"].map(state_abbrev)

    # A column the export left out becomes all-NA rather than absent, so every
    # consumer can read it and skip the missing values the same way
    for name, dtype in OPTIONAL_COLUMNS.values():
        if name not in found:
            df[name] = pd.array([pd.NA] * len(df), dtype=dtype)
        elif dtype in ("Int64", "Float64"):
            df[name] = pd.to_numeric(df[name], errors="coerce").astype(dtype)
        else:
            df[name] = df[name].astype("string").str.strip().replace("", pd.NA)

    for name, limit in COORD_LIMITS.items():
        df.loc[df[name].abs() > limit, name] = pd.NA

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
    # City points in EPSG:3857, from the export's own coordinates where it has them
    # and the geocoder otherwise. A town neither can place is dropped with the rest
    # of its rows.
    path = Path(path)
    df = _read_frame(path, unassigned_label=unassigned_label)
    ordered_groups = sorted(df["Group"].unique().tolist(), key=_group_num)

    unique_places = (
        df.groupby("Full Location", sort=False)
        .agg(
            City=("City", "first"),
            State=("State", "first"),
            SourceLat=("SourceLat", "first"),
            SourceLon=("SourceLon", "first"),
        )
        .reset_index()
    )
    has_coords = unique_places["SourceLat"].notna() & unique_places["SourceLon"].notna()
    total_places = len(unique_places)
    print(
        f"\n{int(has_coords.sum())} of {total_places} towns carry coordinates; "
        f"geocoding the other {int((~has_coords).sum())}..."
    )

    lat_list, lon_list = [], []
    for i, row in unique_places.iterrows():
        if has_coords[i]:
            lat_list.append(float(row["SourceLat"]))
            lon_list.append(float(row["SourceLon"]))
            continue
        res = geocode_city_state(row["City"], row["State"], f"({i + 1}/{total_places})")
        lat_list.append(res["lat"] if res else None)
        lon_list.append(res["lon"] if res else None)

    print(f"Located {total_places} towns.\n")
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


def load_city_boundaries(gdf_points):
    # One row per unique city: City, State, Group, geometry in EPSG:3857. Geometry is
    # None where OSM has no boundary, and the map keeps that town as a dot.
    pt_lookup = {
        (r.City, r.State): r.geometry
        for r in gdf_points[["City", "State", "geometry"]]
        .drop_duplicates(subset=["City", "State"])
        .itertuples(index=False)
    }

    city_records = (
        gdf_points[["City", "State", "Group", "County"]]
        .drop_duplicates(subset=["City", "State"])
        .reset_index(drop=True)
    )
    total_cities = len(city_records)
    print(f"Starting OSM boundary fetching for {total_cities} cities...")

    records = []
    consecutive_fallbacks = total_fallbacks = 0
    # Fresh misses wait here until something shows the lookup is working -- a fresh
    # hit, or the run finishing without tripping the outage check -- so the misses
    # that do trip it are never cached as permanent "no boundary" answers
    pending_misses = []
    # Towns whose only remaining lookup (Overpass) was unreachable this build
    unavailable = []
    for i, row in city_records.iterrows():
        city, state, group = row.City, row.State, row.Group
        progress = f"({i + 1}/{total_cities})"

        geom_ll = get_cached_osm_polygon(city, state)
        fetched = geom_ll is NOT_CACHED
        if fetched:
            try:
                geom_ll = fetch_osm_city_polygon(
                    city, state, pt_lookup[(city, state)], progress, county=row.County
                )
            except FallbackUnavailable as exc:
                # Not a miss and not cached: a dot this build, asked again next time
                print(f"   {exc}")
                unavailable.append(f"{city}, {state}")
                records.append({"City": city, "State": state, "Group": group, "geometry": None})
                continue
            if geom_ll is None:
                pending_misses.append((city, state))
            else:
                cache_osm_polygon(city, state, geom_ll)
                for missed in pending_misses:
                    cache_osm_polygon(*missed, None)
                pending_misses.clear()
        else:
            print(f"Using cached boundary: {city}, {state} {progress}")

        if geom_ll is None:
            total_fallbacks += 1
            # Overpass can answer 200 with an empty result when it is really
            # timing out or serving the wrong extract, which no exception catches.
            # A long unbroken run of fresh misses is the only signal that reaches
            # here; a miss answered from the cache is a known gap, not that signal.
            if fetched:
                consecutive_fallbacks += 1
            if consecutive_fallbacks >= OSM_MAX_CONSECUTIVE_FALLBACKS:
                raise RuntimeError(
                    f"{consecutive_fallbacks} cities in a row returned no OSM boundary, "
                    f"the latest being {city}, {state}. That points at the lookup being "
                    f"broken rather than the data being missing, so stopping rather than "
                    f"drawing a map of dots. Check Nominatim and the Overpass API, then "
                    f"rerun -- cached boundaries are kept."
                )
            geom_3857 = None
        else:
            # A cached hit says nothing about whether the lookup works right now
            if fetched:
                consecutive_fallbacks = 0
            geom_3857 = gpd.GeoSeries([geom_ll], crs=4326).to_crs(3857).iloc[0]

        records.append({"City": city, "State": state, "Group": group, "geometry": geom_3857})

    for missed in pending_misses:
        cache_osm_polygon(*missed, None)
    if unavailable:
        print(
            f"WARNING: Overpass was unreachable for {len(unavailable)} towns, drawn as dots "
            f"this build and not cached, so the next build tries again: "
            f"{', '.join(unavailable)}"
        )
    print(
        f"OSM boundary fetching complete! Processed {total_cities} cities, "
        f"{total_cities - total_fallbacks} with real boundaries and "
        f"{total_fallbacks} kept as dots.\n"
    )
    # Spelled out so an empty run still has a geometry column to build on
    frame = pd.DataFrame(records, columns=["City", "State", "Group", "geometry"])
    return gpd.GeoDataFrame(frame, geometry="geometry", crs=3857)
