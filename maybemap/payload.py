# Assembles everything the page draws into one JSON-serialisable dict. The page
# itself holds no data of its own: every layer reads from this.

from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import geopandas as gpd
import pandas as pd
import shapely
from shapely.geometry import mapping

from maybemap import routing
from maybemap.colors import darken
from maybemap.config import (
    CITY_DATA_PATH,
    COORD_PRECISION,
    CUSTOMER_BUCKETS,
    DISPATCH_STAGING_PATH,
    DISPATCH_TECHS_PATH,
    KMZ_PATH,
    MAP_TITLE,
    OUTLINE_SIMPLIFY_M,
    REGION_HUBS_PATH,
    REGION_PALETTE,
    SERVICE_TYPES,
    UNASSIGNED_GROUP_COLOR,
    UNASSIGNED_GROUP_LABEL,
)
from maybemap.geocoding import geocode_city_state
from maybemap.sources.cities import load_city_boundaries, load_city_points
from maybemap.sources.dispatch import load_region_hubs, load_staging, load_techs
from maybemap.sources.kmz import load_kmz_zones

# Per-town detail columns from the export, under the names the page reads
DETAIL_FIELDS = {
    "Customers": "customers",
    "ActiveAddresses": "active",
    "Tasks6Mo": "tasks",
    "ResidentialCustomers": "residential",
    "BusinessCustomers": "business",
}


def _value(v):
    # pandas NA and numpy scalars are not JSON; None and plain ints are
    if v is None or pd.isna(v):
        return None
    return v.item() if hasattr(v, "item") else v


def _snap(geom):
    return shapely.set_precision(geom, COORD_PRECISION)


def _feature(geom, properties: dict) -> dict:
    return {"type": "Feature", "properties": properties, "geometry": mapping(_snap(geom))}


def region_colors(ordered_groups: list[str]) -> dict[str, str]:
    real = [g for g in ordered_groups if g != UNASSIGNED_GROUP_LABEL]
    colors = {g: REGION_PALETTE[i % len(REGION_PALETTE)] for i, g in enumerate(real)}
    if UNASSIGNED_GROUP_LABEL in ordered_groups:
        colors[UNASSIGNED_GROUP_LABEL] = UNASSIGNED_GROUP_COLOR
    return colors


def build_zones(zones: gpd.GeoDataFrame) -> dict:
    features = []
    for row in zones.itertuples(index=False):
        anchor = row.geometry.representative_point()
        features.append(
            _feature(
                row.geometry,
                {
                    "label": row.Label,
                    "color": row.Color,
                    "edge": darken(row.Color, 0.25),
                    "anchor": [round(anchor.y, 5), round(anchor.x, 5)],
                },
            )
        )
    return {"type": "FeatureCollection", "features": features}


def tag_zones(points_ll: gpd.GeoDataFrame, zones: gpd.GeoDataFrame) -> pd.Series:
    # The zone each town sits in, or None outside every zone (Illinois, Arkansas...)
    joined = gpd.sjoin(
        points_ll[["geometry"]], zones[["Label", "geometry"]], how="left", predicate="within"
    )
    # A point on a shared border can match two zones; keep the first
    joined = joined[~joined.index.duplicated(keep="first")]
    return joined["Label"]


def build_cities(points: gpd.GeoDataFrame, zones: gpd.GeoDataFrame, colors: dict) -> list[dict]:
    # One entry per town. The export can list a town twice (two counties, say), and
    # the map has one dot per place, so the first row speaks for it.
    points = points.drop_duplicates(subset=["City", "State"]).reset_index(drop=True)
    points_ll = points.to_crs(4326)
    zone_of = tag_zones(points_ll, zones)

    cities = []
    for i, row in points_ll.iterrows():
        group = str(row["Group"])
        mix = _value(row["ServiceMix"])
        city = {
            "id": i,
            "name": row["City"],
            "state": row["State"],
            "county": row["County"] or None,
            "region": group,
            "regioned": group != UNASSIGNED_GROUP_LABEL,
            "zone": _value(zone_of.loc[i]),
            "color": colors[group],
            "lat": round(row.geometry.y, 5),
            "lon": round(row.geometry.x, 5),
            "mix": mix,
            "mixKey": SERVICE_TYPES.get(mix) if mix else None,
        }
        city.update({key: _value(row[col]) for col, key in DETAIL_FIELDS.items()})
        cities.append(city)
    return cities


def build_outlines(boundaries: gpd.GeoDataFrame, cities: list[dict]) -> dict:
    by_name = {(c["name"], c["state"]): c["id"] for c in cities}
    found = boundaries[boundaries.geometry.notna()].copy()
    found["geometry"] = found.geometry.simplify(OUTLINE_SIMPLIFY_M, preserve_topology=True)
    found = found.to_crs(4326)
    features = [
        _feature(row.geometry, {"city": by_name[(row.City, row.State)]})
        for row in found.itertuples(index=False)
        if (row.City, row.State) in by_name
    ]
    return {"type": "FeatureCollection", "features": features}


class RegionHubs(NamedTuple):
    hubs: dict[str, tuple[str, str]]  # region -> (town, state)
    defaulted: set[str]  # regions whose hub came from the counts, not an override
    unresolved: set[str]  # regions with no counted town and no override


def region_hubs(
    cities: list[dict], overrides: dict[str, tuple[str, str]] | None = None
) -> RegionHubs:
    # Each region's hub is its member town with the most customers. A town with no
    # count is skipped. Ties go to a town whose name appears in the region's name,
    # then to the name first in order. An override row wins over all of that. A
    # region with no counted town and no override has no hub, and is reported apart.
    overrides = overrides or {}
    by_region: dict[str, list[dict]] = {}
    for c in cities:
        if c["regioned"]:
            by_region.setdefault(c["region"], []).append(c)
    hubs = dict(overrides)
    defaulted, unresolved = set(), set()
    for region, towns in by_region.items():
        if region in overrides:
            continue
        counted = [t for t in towns if t.get("customers") is not None]
        if not counted:
            unresolved.add(region)
            continue
        top = max(t["customers"] for t in counted)
        tied = [t for t in counted if t["customers"] == top]
        named = [t for t in tied if t["name"].lower() in region.lower()]
        hub = min(named or tied, key=lambda t: t["name"])
        hubs[region] = (hub["name"], hub["state"])
        defaulted.add(region)
    return RegionHubs(hubs, defaulted, unresolved)


def _hub_region(region: str, found: RegionHubs) -> str | None:
    # The region a tech with no town is placed by. The whole Primary Region wins when
    # it is a region (names can hold a comma); otherwise a list takes its first entry.
    known = found.hubs.keys() | found.unresolved
    if region in known:
        return region
    first = region.split(",")[0].strip()
    return first if first in known else None


def build_places(
    cities: list[dict],
    techs: pd.DataFrame,
    overrides: dict[str, tuple[str, str]] | None = None,
) -> dict:
    # The dispatch set: every regioned town, plus each tech's home town whether or
    # not it carries a region, since a tech nobody can route to is no use. A tech
    # with no town is placed at the hub of their primary region instead (the first one
    # listed, when there are several); one with neither cannot be placed and is only
    # listed. A region with no possible hub stops the build only when such a tech needs it.
    places, index = [], {}

    def add(name, state, lat, lon, city_id):
        index[(name, state)] = len(places)
        places.append(
            {"name": name, "state": state, "lat": lat, "lon": lon, "city": city_id, "techs": []}
        )

    for c in cities:
        if c["regioned"]:
            add(c["name"], c["state"], c["lat"], c["lon"], c["id"])

    by_name = {(c["name"], c["state"]): c for c in cities}
    found = region_hubs(cities, overrides)
    out, added, unplaced = [], [], []
    hubs_used: dict[str, tuple[str, str]] = {}
    for row in techs.itertuples(index=False):
        if row.Town and row.State:
            key, basis = (row.Town, row.State), "town"
        elif (region := _hub_region(row.Region, found)) is not None:
            if region in found.unresolved:
                raise ValueError(
                    f"Tech {row.Tech} has no town, and region {region} has no town with a "
                    f"customer count to be its hub. Add a row for it to "
                    f"{REGION_HUBS_PATH.name}: Region,Hub,State."
                )
            key, basis = found.hubs[region], "region"
            if region != row.Region:
                print(f"Tech {row.Tech} lists several regions; placed by the first, {region}")
            if region in found.defaulted:
                hubs_used.setdefault(region, key)
        else:
            unplaced.append(row.Tech)
            continue
        if key not in index:
            city = by_name.get(key)
            if city is not None:
                add(*key, city["lat"], city["lon"], city["id"])
            else:
                res = geocode_city_state(*key)
                if not res:
                    raise ValueError(
                        f"Could not geocode {key[0]}, {key[1]} for tech {row.Tech}. "
                        f"Check the spelling in {DISPATCH_TECHS_PATH.name}."
                    )
                add(*key, round(res["lat"], 5), round(res["lon"], 5), None)
            added.append(f"{key[0]}, {key[1]}")
        places[index[key]]["techs"].append(len(out))
        out.append(
            {
                "name": row.Tech,
                "id": row.TechId or None,
                "region": row.Region or None,
                "departments": list(row.Departments),
                "place": index[key],
                "basis": basis,
            }
        )
    for region, (town, state) in hubs_used.items():
        print(
            f"Hub for region {region}: {town}, {state}, the town with the most customers. "
            f"Add a row to {REGION_HUBS_PATH.name} to change it."
        )
    return {"places": places, "techs": out, "added": added, "unplaced": unplaced}


def build_staging(staging: pd.DataFrame, places: list[dict]) -> list[dict]:
    index = {(p["name"], p["state"]): i for i, p in enumerate(places)}
    groups: dict[str, list[int]] = {}
    for row in staging.itertuples(index=False):
        key = (row.Town, row.State)
        if key not in index:
            raise ValueError(
                f"Staging site {row.Type} at {row.Town}, {row.State} is not a dispatch "
                f"place. Give the town a region in the city export, or fix the spelling "
                f"in {DISPATCH_STAGING_PATH.name}."
            )
        towns = groups.setdefault(row.Type, [])
        if index[key] not in towns:
            towns.append(index[key])
    return [{"type": t, "places": towns} for t, towns in groups.items()]


def _bucket_labels() -> list[str]:
    return [f"{lo}+" if hi is None else f"{lo}-{hi}" for lo, hi in CUSTOMER_BUCKETS]


def build(
    *,
    city_path: Path | str = CITY_DATA_PATH,
    kmz_path: Path | str = KMZ_PATH,
    techs_path: Path | str = DISPATCH_TECHS_PATH,
    staging_path: Path | str = DISPATCH_STAGING_PATH,
    hubs_path: Path | str = REGION_HUBS_PATH,
    matrix=routing.drive_matrix,
    tech_table=routing.drive_table,
) -> dict:
    zones = load_kmz_zones(kmz_path)
    points, ordered_groups = load_city_points(city_path)
    colors = region_colors(ordered_groups)
    cities = build_cities(points, zones, colors)

    regioned = points[points["Group"] != UNASSIGNED_GROUP_LABEL]
    regioned = regioned.drop_duplicates(subset=["City", "State"])
    outlines = build_outlines(load_city_boundaries(regioned), cities)

    roster = build_places(cities, load_techs(techs_path), load_region_hubs(hubs_path))
    places = roster["places"]
    if roster["added"]:
        print(f"Added tech towns with no region to the dispatch set: {', '.join(roster['added'])}")
    by_region = sum(t["basis"] == "region" for t in roster["techs"])
    if by_region:
        print(f"Placed {by_region} technicians with no town at their primary region's hub")
    if roster["unplaced"]:
        print(
            f"Could not place {len(roster['unplaced'])} technicians with neither a town nor a "
            f"known region: {', '.join(roster['unplaced'])}"
        )
    staging = build_staging(load_staging(staging_path), places)
    meters, seconds = matrix([(p["lat"], p["lon"]) for p in places])

    # Every card lists its closest techs by home town. A dispatch town already has its
    # column in the matrix; the other towns need the drive time from each tech place.
    tech_places = [i for i, p in enumerate(places) if p["techs"]]
    in_dispatch = {p["city"] for p in places if p["city"] is not None}
    outside = [c for c in cities if c["id"] not in in_dispatch]
    to_meters, to_seconds = tech_table(
        [(places[i]["lat"], places[i]["lon"]) for i in tech_places],
        [(c["lat"], c["lon"]) for c in outside],
    )

    print(
        f"Payload: {len(zones)} zones, {len(cities)} cities, {len(outlines['features'])} "
        f"outlines, {len(places)} dispatch places, {len(staging)} staging types"
    )
    return {
        "title": MAP_TITLE,
        "builtAt": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "zones": build_zones(zones),
        "regions": [{"name": g, "color": colors[g]} for g in ordered_groups],
        "cities": cities,
        "outlines": outlines,
        "buckets": [
            {"label": label, "lo": lo, "hi": hi}
            for label, (lo, hi) in zip(_bucket_labels(), CUSTOMER_BUCKETS, strict=True)
        ],
        "unassignedLabel": UNASSIGNED_GROUP_LABEL,
        "serviceTypes": [{"key": k, "label": k.title()} for k in SERVICE_TYPES.values()],
        "departments": sorted({d for t in roster["techs"] for d in t["departments"]}),
        "dispatch": {
            "places": places,
            "techs": roster["techs"],
            "unplaced": roster["unplaced"],
            "meters": meters,
            "seconds": seconds,
            "techPlaces": tech_places,
            "toCity": {
                "ids": [c["id"] for c in outside],
                "meters": to_meters,
                "seconds": to_seconds,
            },
        },
        "staging": staging,
    }
