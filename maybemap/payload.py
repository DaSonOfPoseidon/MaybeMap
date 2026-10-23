# Assembles everything the page draws into one JSON-serialisable dict. The page
# itself holds no data of its own: every layer reads from this.

from datetime import datetime
from pathlib import Path

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
    REGION_PALETTE,
    SERVICE_TYPES,
    UNASSIGNED_GROUP_COLOR,
    UNASSIGNED_GROUP_LABEL,
)
from maybemap.geocoding import geocode_city_state
from maybemap.sources.cities import load_city_boundaries, load_city_points
from maybemap.sources.dispatch import load_staging, load_techs
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


def build_places(cities: list[dict], techs: pd.DataFrame) -> tuple[list[dict], list[str]]:
    # The dispatch set: every regioned town, plus each tech's home town whether or
    # not it carries a region, since a tech nobody can route to is no use. Returns
    # the places and the tech towns that had to be added from outside the regions.
    places, index = [], {}
    for c in cities:
        if c["regioned"]:
            index[(c["name"], c["state"])] = len(places)
            places.append(
                {
                    "name": c["name"],
                    "state": c["state"],
                    "lat": c["lat"],
                    "lon": c["lon"],
                    "city": c["id"],
                    "techs": [],
                }
            )

    by_name = {(c["name"], c["state"]): c for c in cities}
    added = []
    for row in techs.itertuples(index=False):
        key = (row.Town, row.State)
        if key not in index:
            city = by_name.get(key)
            if city is not None:
                lat, lon, city_id = city["lat"], city["lon"], city["id"]
            else:
                res = geocode_city_state(row.Town, row.State)
                if not res:
                    raise ValueError(
                        f"Could not geocode {row.Town}, {row.State} for tech {row.Tech}. "
                        f"Check the spelling in {DISPATCH_TECHS_PATH.name}."
                    )
                lat, lon, city_id = round(res["lat"], 5), round(res["lon"], 5), None
            index[key] = len(places)
            places.append(
                {
                    "name": row.Town,
                    "state": row.State,
                    "lat": lat,
                    "lon": lon,
                    "city": city_id,
                    "techs": [],
                }
            )
            added.append(f"{row.Town}, {row.State}")
        places[index[key]]["techs"].append(row.Tech)
    return places, added


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
    matrix=routing.drive_matrix,
) -> dict:
    zones = load_kmz_zones(kmz_path)
    points, ordered_groups = load_city_points(city_path)
    colors = region_colors(ordered_groups)
    cities = build_cities(points, zones, colors)

    regioned = points[points["Group"] != UNASSIGNED_GROUP_LABEL]
    regioned = regioned.drop_duplicates(subset=["City", "State"])
    outlines = build_outlines(load_city_boundaries(regioned), cities)

    places, added = build_places(cities, load_techs(techs_path))
    if added:
        print(f"Added tech towns with no region to the dispatch set: {', '.join(added)}")
    staging = build_staging(load_staging(staging_path), places)
    meters, seconds = matrix([(p["lat"], p["lon"]) for p in places])

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
        "dispatch": {"places": places, "meters": meters, "seconds": seconds},
        "staging": staging,
    }
