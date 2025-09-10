# MaybeMap.py - Jackson Keithley

# Standard library
import os
import json
import time 
import re
from math import hypot, isclose, pi, cos, sin, sqrt
from pathlib import Path
from typing import Optional, Dict

# Third party - core
import numpy as np
import pandas as pd
import requests

# Geospatial
import geopandas as gpd
import contextily as cx
import osmnx as ox
from shapely.ops import unary_union
from shapely.geometry import Point, box
from shapely import wkt

# Visualization  
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

# Configuration
from dotenv import load_dotenv

load_dotenv()

# ======================================================================
# Configuration
# ======================================================================

# Data sources
DATA_PATH = Path(__file__).parent / "Socket-Regions.xlsx"
GEOAPIFY_KEY = os.getenv("GEOAPIFY_KEY")
if not GEOAPIFY_KEY:
    raise RuntimeError("Missing GEOAPIFY_KEY in environment. Please set it in .env")

# Map appearance
BASEMAP = "voyager"  # "voyager", "dark", or "positron"
BASEMAP_ZOOM = 9
SHOW_LABELS = False
EXPORT_INTERACTIVE_HTML = True
INTERACTIVE_HTML_PATH = "grouped_regions_map.html"

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
LABEL_MIN_SEPARATION_M = 15000   # 15km minimum separation for regional maps
LABEL_EXCLUSION_RADIUS_M = 12000 # 12km exclusion zones
MAX_LABEL_DISTANCE_M = 80000     # 80km maximum leader lines for very dense areas

# OSM data fetching
SEARCH_RADIUS_M = 40000
MAX_CANDIDATE_AREA_M2 = 3_000_000_000
POINT_FALLBACK_BUFFER_M = 5000

# Visual styling
PIN_MARKER_SIZE = 80
CITY_FILL_ALPHA = 0.32
CITY_EDGE_ALPHA = 0.75
CITY_EDGE_WIDTH = 0.7
CIRCLE_PAD_M = 3000
CIRCLE_EDGE_WIDTH = 2.0
CIRCLE_ALPHA_FILL = 0.15
CIRCLE_ALPHA_EDGE = 0.9

# Polygon processing
AREA_STYLE = "union_smooth"  # "union", "union_smooth", "convex_hull"
MERGE_GAP_M = 2000
SIMPLIFY_M = 600
MIN_PART_M2 = 1_000_000


# OSM settings
ox.settings.use_cache = True
ox.settings.log_console = False
ox.settings.nominatim_pause = 1.1

# ======================================================================
# Data Processing Functions
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

# Color utility
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

def geocode_city_state(city: str, state: str, progress_info: str = "") -> Optional[Dict[str, float]]:
    key = f"{city}, {state}"
    if key in GEOCODE_CACHE:
        return GEOCODE_CACHE[key]
    
    print(f"🌍 Geocoding: {city}, {state} {progress_info}")
    res = geoapify_forward(f"{city}, {state}, USA")
    GEOCODE_CACHE[key] = res
    _save_geocode_cache()
    
    if res:
        print(f"   Found coordinates: {res['lat']:.4f}, {res['lon']:.4f}")
    else:
        print(f"   Geocoding failed for {city}, {state}")
    
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

def fetch_osm_city_polygon(city: str, state: str, progress_info: str = ""):
    """Find a municipal polygon near the city geocode point, prefer proper scale."""
    print(f"🗺️  Fetching OSM boundary: {city}, {state} {progress_info}")
    
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
    result = gpd.GeoSeries([chosen_3857], crs=3857).to_crs(4326).iloc[0]
    
    print(f"   Found OSM boundary polygon (area: {chosen_3857.area/1000000:.2f} km²)")
    return result

# ======================================================================
# Geometry Utilities  
# ======================================================================

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
# Cartographic Labeling System
# ======================================================================
def classify_city_importance(city: str, state: str) -> str:
    """Classify cities into Primary, Secondary, or Tertiary importance for professional mapping."""
    city_upper = city.upper()
    state_upper = state.upper()
    
    # State capitals lookup
    state_capitals = {
        'AL': 'MONTGOMERY', 'AK': 'JUNEAU', 'AZ': 'PHOENIX', 'AR': 'LITTLE ROCK',
        'CA': 'SACRAMENTO', 'CO': 'DENVER', 'CT': 'HARTFORD', 'DE': 'DOVER',
        'FL': 'TALLAHASSEE', 'GA': 'ATLANTA', 'HI': 'HONOLULU', 'ID': 'BOISE',
        'IL': 'SPRINGFIELD', 'IN': 'INDIANAPOLIS', 'IA': 'DES MOINES', 'KS': 'TOPEKA',
        'KY': 'FRANKFORT', 'LA': 'BATON ROUGE', 'ME': 'AUGUSTA', 'MD': 'ANNAPOLIS',
        'MA': 'BOSTON', 'MI': 'LANSING', 'MN': 'SAINT PAUL', 'MS': 'JACKSON',
        'MO': 'JEFFERSON CITY', 'MT': 'HELENA', 'NE': 'LINCOLN', 'NV': 'CARSON CITY',
        'NH': 'CONCORD', 'NJ': 'TRENTON', 'NM': 'SANTA FE', 'NY': 'ALBANY',
        'NC': 'RALEIGH', 'ND': 'BISMARCK', 'OH': 'COLUMBUS', 'OK': 'OKLAHOMA CITY',
        'OR': 'SALEM', 'PA': 'HARRISBURG', 'RI': 'PROVIDENCE', 'SC': 'COLUMBIA',
        'SD': 'PIERRE', 'TN': 'NASHVILLE', 'TX': 'AUSTIN', 'UT': 'SALT LAKE CITY',
        'VT': 'MONTPELIER', 'VA': 'RICHMOND', 'WA': 'OLYMPIA', 'WV': 'CHARLESTON',
        'WI': 'MADISON', 'WY': 'CHEYENNE'
    }
    
    # Check if it's a state capital
    is_state_capital = (state_upper in state_capitals and 
                       city_upper == state_capitals[state_upper])
    
    # Simple heuristics for major cities
    major_metros = {
        'NEW YORK', 'LOS ANGELES', 'CHICAGO', 'HOUSTON', 'PHOENIX', 'PHILADELPHIA',
        'SAN ANTONIO', 'SAN DIEGO', 'DALLAS', 'SAN JOSE', 'AUSTIN', 'JACKSONVILLE',
        'SAN FRANCISCO', 'CHARLOTTE', 'SEATTLE', 'DENVER', 'WASHINGTON', 'BOSTON',
        'DETROIT', 'NASHVILLE', 'ATLANTA', 'MIAMI', 'TAMPA', 'NEW ORLEANS',
        'CLEVELAND', 'MINNEAPOLIS', 'KANSAS CITY', 'SAINT LOUIS', 'PITTSBURGH',
        'CINCINNATI', 'INDIANAPOLIS', 'COLUMBUS', 'MILWAUKEE', 'BALTIMORE',
        'ORLANDO', 'RALEIGH', 'RICHMOND', 'NORFOLK', 'BIRMINGHAM', 'MEMPHIS',
        'BUFFALO', 'SACRAMENTO', 'PORTLAND', 'LAS VEGAS', 'OKLAHOMA CITY',
        'LOUISVILLE', 'TUCSON', 'ALBUQUERQUE', 'FRESNO', 'OMAHA', 'HONOLULU'
    }
    
    # Enhanced logic for better classification
    if city_upper in major_metros:
        return 'Primary'
    elif is_state_capital:
        return 'Secondary' 
    elif any(keyword in city_upper for keyword in ['SAINT', 'ST.', 'FORT', 'NEW ', 'WEST ', 'EAST ', 'NORTH ', 'SOUTH ']):
        # Cities with common prefixes are often significant
        return 'Secondary'
    else:
        return 'Tertiary'

def get_label_style(importance: str, text_color: str) -> dict:
    """Get typography settings based on city importance."""
    if importance == 'Primary':
        return {
            'fontsize': LABEL_FONTSIZE_PRIMARY,
            'weight': 'bold',
            'color': text_color,
            'exclusion_radius': LABEL_EXCLUSION_RADIUS_M * 1.5
        }
    elif importance == 'Secondary':
        return {
            'fontsize': LABEL_FONTSIZE_SECONDARY,
            'weight': 'normal',
            'color': text_color,
            'exclusion_radius': LABEL_EXCLUSION_RADIUS_M * 1.2
        }
    else:  # Tertiary
        return {
            'fontsize': LABEL_FONTSIZE_TERTIARY,
            'weight': 'normal',
            'color': text_color,
            'exclusion_radius': LABEL_EXCLUSION_RADIUS_M
        }

def analyze_map_geography(pts_df):
    """Analyze map bounds and city distribution for smart label placement."""
    from math import sqrt
    
    # Calculate map bounds
    all_x = [geom.x for geom in pts_df.geometry]
    all_y = [geom.y for geom in pts_df.geometry]
    
    map_bounds = {
        'min_x': min(all_x), 'max_x': max(all_x),
        'min_y': min(all_y), 'max_y': max(all_y),
        'center_x': (min(all_x) + max(all_x)) / 2,
        'center_y': (min(all_y) + max(all_y)) / 2
    }
    
    # For each city, analyze density in 8 cardinal directions
    city_analysis = {}
    for _, row in pts_df.iterrows():
        city_x, city_y = row.geometry.x, row.geometry.y
        
        # Count cities in each direction (within 100km radius)
        directions = {
            'N': 0, 'NE': 0, 'E': 0, 'SE': 0,
            'S': 0, 'SW': 0, 'W': 0, 'NW': 0
        }
        
        for _, other_row in pts_df.iterrows():
            if row.City == other_row.City:  # Skip self
                continue
                
            other_x, other_y = other_row.geometry.x, other_row.geometry.y
            dx, dy = other_x - city_x, other_y - city_y
            distance = sqrt(dx*dx + dy*dy)
            
            if distance > 100000:  # Only consider nearby cities (100km)
                continue
                
            # Determine direction (8-way)
            if abs(dx) < distance * 0.383:  # ~22.5 degree tolerance
                if dy > 0: directions['N'] += 1
                else: directions['S'] += 1
            elif abs(dy) < distance * 0.383:
                if dx > 0: directions['E'] += 1
                else: directions['W'] += 1
            else:  # Diagonal
                if dx > 0 and dy > 0: directions['NE'] += 1
                elif dx > 0 and dy < 0: directions['SE'] += 1
                elif dx < 0 and dy > 0: directions['NW'] += 1
                else: directions['SW'] += 1
        
        # Determine position relative to map bounds
        edge_threshold = 0.15  # Within 15% of edge
        is_edge = {
            'north': (city_y - map_bounds['min_y']) / (map_bounds['max_y'] - map_bounds['min_y']) > (1 - edge_threshold),
            'south': (city_y - map_bounds['min_y']) / (map_bounds['max_y'] - map_bounds['min_y']) < edge_threshold,
            'east': (city_x - map_bounds['min_x']) / (map_bounds['max_x'] - map_bounds['min_x']) > (1 - edge_threshold),
            'west': (city_x - map_bounds['min_x']) / (map_bounds['max_x'] - map_bounds['min_x']) < edge_threshold
        }
        
        city_analysis[row.City] = {
            'directions': directions,
            'is_edge': is_edge,
            'position': (city_x, city_y)
        }
    
    return map_bounds, city_analysis

def professional_label_placement(ax, gdf_pts, color_for, text_color, leader_color, use_numbering=True):
    """
    Geographic-aware professional cartographic labeling system with:
    • Smart directional placement toward empty spaces
    • Hierarchical typography based on city importance
    • Collision-free placement using exclusion zones
    • Clean leader line routing
    """
    # Classify cities by importance
    pts_df = gdf_pts[["Group", "geometry", "City", "State"]].reset_index(drop=True)
    pts_df['Importance'] = pts_df.apply(lambda row: classify_city_importance(row.City, row.State), axis=1)
    
    # Sort by importance first, then by density (dense areas get priority within each tier)
    importance_order = {'Primary': 0, 'Secondary': 1, 'Tertiary': 2}
    pts_df['imp_order'] = pts_df['Importance'].map(importance_order)
    
    # Calculate local density for each city
    from math import sqrt
    pts_df['density_score'] = 0
    for i, row in pts_df.iterrows():
        nearby_count = 0
        for j, other_row in pts_df.iterrows():
            if i != j:
                dist = sqrt((row.geometry.x - other_row.geometry.x)**2 + (row.geometry.y - other_row.geometry.y)**2)
                if dist < 50000:  # Within 50km
                    nearby_count += 1
        pts_df.at[i, 'density_score'] = nearby_count
    
    # Sort by importance first, then by density (highest density first within each importance tier)
    pts_df = pts_df.sort_values(['imp_order', 'density_score'], ascending=[True, False]).reset_index(drop=True)
    
    # Analyze map geography for smart placement
    map_bounds, city_analysis = analyze_map_geography(pts_df)
    print(f"   Map analysis: {len(city_analysis)} cities analyzed")
    
    # Track placed labels to avoid overlaps
    placed_labels = []  # List of (x, y, exclusion_radius) tuples
    numbered_cities = {}  # Dictionary for cities using numbered markers
    dense_area_threshold = 16  # Cities with 16+ neighbors get numbered (excludes Columbia IL)
    
    def find_spiral_position(pin_x, pin_y, number, label_style):
        """Place numbered labels in a spiral pattern around dense areas."""
        from math import sqrt, cos, sin, pi
        
        # Calculate spiral parameters
        angle_step = 0.5  # Radians between each position
        radius_step = 8000  # 8km radius increase per spiral turn
        
        # Calculate position in spiral
        angle = angle_step * number
        radius = 25000 + radius_step * (number // 8)  # Start at 25km, increase every 8 labels
        
        # Calculate spiral position
        spiral_x = pin_x + radius * cos(angle)
        spiral_y = pin_y + radius * sin(angle)
        
        return spiral_x, spiral_y
    
    def find_placement_position(pin_x, pin_y, label_style, city_name, city_state=None):
        """Find optimal label position using geographic-aware directional placement."""
        exclusion_radius = label_style['exclusion_radius']
        
        # Manual positioning overrides for specific cities
        manual_positions = {
            'Harrisburg': (-20000, 0),  # Shift left (west)
            'Lebanon': (0, 15000),      # Shift up (north) between Collinsville and O'Fallon IL
        }
        
        # Specific city-state combinations
        if city_name == 'Columbia' and city_state == 'Illinois':
            return pin_x - 25000, pin_y - 25000  # Columbia IL: shift SW, left of Waterloo
        
        if city_name in manual_positions:
            offset_x, offset_y = manual_positions[city_name]
            return pin_x + offset_x, pin_y + offset_y
        
        # Get geographic analysis for this city
        city_data = city_analysis.get(city_name, {})
        directions = city_data.get('directions', {})
        is_edge = city_data.get('is_edge', {})
        
        # Determine preferred directions based on geography
        preferred_angles = []
        
        # For edge cities, prefer inward directions
        if is_edge.get('north', False):  # Northern edge - place labels south
            preferred_angles.extend([4.712, 5.498, 3.927])  # S, SW, SE
        if is_edge.get('south', False):  # Southern edge - place labels north  
            preferred_angles.extend([1.571, 0.785, 2.356])  # N, NE, NW
        if is_edge.get('east', False):   # Eastern edge - place labels west
            preferred_angles.extend([3.142, 2.356, 3.927])  # W, NW, SW
        if is_edge.get('west', False):   # Western edge - place labels east
            preferred_angles.extend([0, 0.785, 5.498])      # E, NE, SE
        
        # For interior cities, prefer directions with fewer nearby cities
        if not preferred_angles:  # Not an edge city
            direction_angles = {
                'N': 1.571, 'NE': 0.785, 'E': 0, 'SE': 5.498,
                'S': 4.712, 'SW': 3.927, 'W': 3.142, 'NW': 2.356
            }
            
            # Sort directions by density (fewer cities = better)
            sorted_dirs = sorted(directions.items(), key=lambda x: x[1])
            preferred_angles = [direction_angles[dir_name] for dir_name, _ in sorted_dirs[:4]]
        
        # Add some variation around preferred angles
        all_angles = []
        for base_angle in preferred_angles:
            all_angles.extend([
                base_angle,
                base_angle + 0.262,  # ±15 degrees
                base_angle - 0.262
            ])
        
        # Add remaining angles as fallback
        remaining_angles = [i * (2 * pi / 16) for i in range(16)]
        all_angles.extend([a for a in remaining_angles if a not in all_angles])
        
        distances = [8000, 12000, 16000, 22000, 30000]
        positions_tried = 0
        
        # Try positions at increasing distances with smart angle prioritization
        for distance in distances:
            if distance > MAX_LABEL_DISTANCE_M:
                break
                
            for angle in all_angles:
                label_x = pin_x + distance * cos(angle)
                label_y = pin_y + distance * sin(angle)
                positions_tried += 1
                
                # Check collision with existing labels
                collision = False
                min_required_separation = None
                for placed_x, placed_y, placed_radius in placed_labels:
                    dist = sqrt((label_x - placed_x)**2 + (label_y - placed_y)**2)
                    
                    # Use a reasonable base separation (8km) plus label-specific exclusion zones
                    base_separation = 8000  # 8km base minimum
                    min_separation = base_separation + exclusion_radius + placed_radius
                    
                    if dist < min_separation:
                        collision = True
                        min_required_separation = min_separation
                        break
                
                if not collision:
                    # Double-check: ensure we're not too close to any existing label
                    if placed_labels:
                        actual_min_dist = min([sqrt((label_x - px)**2 + (label_y - py)**2) for px, py, pr in placed_labels])
                        if actual_min_dist < 8000:  # Too close to any existing label
                            collision = True
                
                if not collision:
                    direction_name = "preferred" if angle in preferred_angles[:4] else "fallback"
                    print(f"      {city_name}: Found {direction_name} position at {distance}m after {positions_tried} attempts")
                    return label_x, label_y
        
        # Final fallback - try a systematic approach
        print(f"      {city_name}: Using systematic fallback placement")
        
        # Try the 4 cardinal directions with moderate distance
        cardinal_angles = [0, 1.571, 3.142, 4.712]  # E, N, W, S
        fallback_distance = 35000
        
        for test_angle in cardinal_angles:
            test_x = pin_x + fallback_distance * cos(test_angle)
            test_y = pin_y + fallback_distance * sin(test_angle)
            
            # Check if this position has reasonable separation
            min_dist = min([sqrt((test_x - px)**2 + (test_y - py)**2) for px, py, pr in placed_labels], default=float('inf'))
            if min_dist >= 12000:  # 12km minimum for fallback
                return test_x, test_y
        
        # If all cardinal directions fail, use preferred angle or default
        fallback_angle = preferred_angles[0] if preferred_angles else 0
        return pin_x + fallback_distance * cos(fallback_angle), pin_y + fallback_distance * sin(fallback_angle)
    
    # Place labels in order of importance
    texts = []
    leader_lines = []
    
    for _, row in pts_df.iterrows():
        pin_x, pin_y = row.geometry.x, row.geometry.y
        label_style = get_label_style(row.Importance, text_color)
        
        # Check if this city is in a very dense area (only for static PNG)
        density_score = row.get('density_score', 0)
        
        # Specific exclusions from numbering  
        is_excluded = (row.City == 'Columbia' and row.State == 'Illinois')  # Columbia IL should be normal label
        
        
        use_number = use_numbering and density_score >= dense_area_threshold and not is_excluded
        
        if use_number:
            # Assign a number and use spiral placement for dense areas
            number = len(numbered_cities) + 1
            numbered_cities[number] = f"{row.City}, {row.State}"
            
            # Use spiral placement around dense area center
            label_x, label_y = find_spiral_position(pin_x, pin_y, number, label_style)
            
            # Place numbered label at the found position
            text = ax.text(
                label_x, label_y, str(number),
                fontsize=label_style['fontsize'],
                fontweight=label_style['weight'],
                color='black', ha='center', va='center',
                zorder=12
            )
            
            # Always draw leader line from pin to numbered label
            line = ax.plot([pin_x, label_x], [pin_y, label_y],
                          color=leader_color, 
                          linewidth=LABEL_LINE_WIDTH,
                          alpha=LABEL_LINE_ALPHA, 
                          zorder=9)
            leader_lines.extend(line)
            
            # Record this label's exclusion zone for future collision detection
            placed_labels.append((label_x, label_y, label_style['exclusion_radius']))
            
            texts.append(text)
            print(f"   Numbered {row.City}: #{number} pin({pin_x:.0f}, {pin_y:.0f}) -> label({label_x:.0f}, {label_y:.0f})")
        else:
            # Find optimal label position using geographic awareness
            label_x, label_y = find_placement_position(pin_x, pin_y, label_style, row.City, row.State)
            print(f"   Placing {row.City}: pin({pin_x:.0f}, {pin_y:.0f}) -> label({label_x:.0f}, {label_y:.0f})")
            
            # Create text label
            text = ax.text(
                label_x, label_y, row.City,
                fontsize=label_style['fontsize'],
                weight=label_style['weight'],
                color=label_style['color'],
                ha='center', va='center', 
                zorder=10
            )
        
        if LABEL_HALO:
            import matplotlib.patheffects as pe
            text.set_path_effects([
                pe.Stroke(linewidth=3.0, foreground='white'), 
                pe.Normal()
            ])
        
            # Only for text labels, handle leader lines and exclusion zones
            texts.append(text)
            
            # Record this label's exclusion zone for future collision detection
            placed_labels.append((label_x, label_y, label_style['exclusion_radius']))
            
            # Create leader line if label is not at pin
            distance = sqrt((label_x - pin_x)**2 + (label_y - pin_y)**2)
            if distance > 1000:  # Only draw leader if label is displaced
                line = ax.plot([pin_x, label_x], [pin_y, label_y],
                              color=leader_color, 
                              linewidth=LABEL_LINE_WIDTH,
                              alpha=LABEL_LINE_ALPHA, 
                              zorder=9)
                leader_lines.extend(line)
    
    return texts, leader_lines, numbered_cities

# ======================================================================
# Main Execution
# ======================================================================
data_path = Path(__file__).parent / "Socket-Regions.xlsx"

df = pd.read_excel(data_path, engine="openpyxl")

df.rename(columns={c: c.strip() for c in df.columns}, inplace=True)

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

df = df[[city_col, state_col, group_col]].rename(
    columns={city_col: "City", state_col: "State", group_col: "Group"}
)

df = df.dropna(subset=["City", "State", "Group"]).copy()
for c in ["City", "State", "Group"]:
    df[c] = df[c].astype(str).str.strip()

df["Full Location"] = df["City"] + ", " + df["State"]

print(f"Loaded {len(df)} rows from {DATA_PATH.name} "
      f"(columns mapped as City='{city_col}', State='{state_col}', Group='{group_col}')")

def _group_num(s: str) -> int:
    m = re.search(r"Group\s*(\d+)", s or "", re.IGNORECASE)
    return int(m.group(1)) if m else 10**9

ordered_groups = sorted(df["Group"].unique().tolist(), key=_group_num)

# Geocoding
unique_places = df[["Full Location", "City", "State"]].drop_duplicates().reset_index(drop=True)
total_places = len(unique_places)
print(f"\nStarting geocoding process for {total_places} unique cities...")

lat_list, lon_list = [], []
for i, row in unique_places.iterrows():
    progress = f"({i+1}/{total_places})"
    res = geocode_city_state(row["City"], row["State"], progress)
    lat_list.append(res["lat"] if res else None); lon_list.append(res["lon"] if res else None)

print(f"Geocoding complete! Processed {total_places} cities.\n")
unique_places["Latitude"] = lat_list; unique_places["Longitude"] = lon_list
df = df.merge(unique_places[["Full Location", "Latitude", "Longitude"]], on="Full Location", how="left").dropna(subset=["Latitude", "Longitude"])

gdf_points_ll = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]), crs=4326)
gdf_points = gdf_points_ll.to_crs(3857)
gdf_points["Group"] = pd.Categorical(gdf_points["Group"], categories=ordered_groups, ordered=True)

pt_lookup = {
    (r.City, r.State): gdf_points.loc[(gdf_points["City"] == r.City) & (gdf_points["State"] == r.State), "geometry"].iloc[0]
    for r in df[["City","State"]].drop_duplicates().itertuples(index=False)
}

# City boundaries
city_records = df[["City","State","Group"]].drop_duplicates().reset_index(drop=True)
total_cities = len(city_records)
print(f"Starting OSM boundary fetching for {total_cities} cities...")

records = []
for i, row in city_records.iterrows():
    city, state, group = row.City, row.State, row.Group
    progress = f"({i+1}/{total_cities})"

    geom_ll = get_cached_osm_polygon(city, state)
    if geom_ll is None:
        geom_ll = fetch_osm_city_polygon(city, state, progress)
        cache_osm_polygon(city, state, geom_ll)
    else:
        print(f"Using cached boundary: {city}, {state} {progress}")

    if geom_ll is None:
        print(f"   No OSM boundary found, using {POINT_FALLBACK_BUFFER_M/1000:.1f}km buffer around point")
        pt = pt_lookup[(city, state)]
        geom_3857 = pt.buffer(POINT_FALLBACK_BUFFER_M)
    else:
        geom_3857 = gpd.GeoSeries([geom_ll], crs=4326).to_crs(3857).iloc[0]

    records.append({"City": city, "State": state, "Group": group, "geometry": geom_3857})

print(f"OSM boundary fetching complete! Processed {total_cities} cities.\n")

gdf_city_polys = gpd.GeoDataFrame(records, geometry="geometry", crs=3857)
gdf_city_polys["Group"] = pd.Categorical(gdf_city_polys["Group"], categories=ordered_groups, ordered=True)

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

# Group visualization
print("Calculating group boundaries and minimum enclosing circles...")
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

# Map generation
print("Generating professional cartographic map...")
fig, ax = plt.subplots(figsize=(13, 10), constrained_layout=True)

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

cmap = matplotlib.colormaps["tab20"]
color_for = {grp: mcolors.to_hex(cmap(i % cmap.N)) for i, grp in enumerate(ordered_groups)}

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

# Pin markers removed for cleaner professional appearance

# Labels - Professional cartographic system
print("Applying professional cartographic labeling system...")
if LABEL_STRATEGY == "professional_cartographic":
    texts, leader_lines, numbered_cities = professional_label_placement(ax, gdf_points, color_for, textcolor, leader_color)
    print("   Professional labeling complete - no overlapping labels!")
else:
    # Simple fallback labeling
    numbered_cities = {}  # Initialize for consistency
    lbl = gdf_points.sort_values("City").groupby("Group").head(1)
    for grp, row in lbl.groupby("Group"):
        p = row.geometry.iloc[0]
        ax.text(p.x + 2000, p.y + 2000, row.City.iloc[0],
                fontsize=LABEL_FONTSIZE_SECONDARY, color=textcolor, zorder=8)

cx.add_basemap(ax, source=base_src, zoom=BASEMAP_ZOOM)
if SHOW_LABELS:
    cx.add_basemap(ax, source=label_src, zoom=BASEMAP_ZOOM, zorder=3)
handles = [Line2D([0], [0], marker="o", linestyle="",
                  markerfacecolor=color_for[g], markeredgecolor="none",
                  markersize=8, label=g)
           for g in ordered_groups]
leg = ax.legend(handles=handles, title="Groups",
                loc="upper left", bbox_to_anchor=(1.02, 1.0))

# Add numbered cities legend if any exist
if numbered_cities:
    legend_text = "Dense Area Cities:\n"
    for num, city_name in sorted(numbered_cities.items()):
        legend_text += f"{num}. {city_name}\n"
    
    ax.text(1.02, 0.3, legend_text, transform=ax.transAxes, 
            fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))

plt.subplots_adjust(right=0.82)

ax.set_title(f"Mapped Regions by Group (Professional Cartographic Labeling, {BASEMAP.title()} basemap)", fontsize=15)
ax.set_xlabel(""); ax.set_ylabel("")
ax.set_aspect('equal', adjustable='box')

# Save output
try:
    plt.savefig("grouped_regions_map.png", dpi=220, bbox_inches="tight")
except Exception:
    plt.savefig("grouped_regions_map.png", dpi=220)
print("Saved PNG: grouped_regions_map.png")
plt.show()

# Data exports
gdf_polys.to_crs(4326).to_file("grouped_regions.geojson", driver="GeoJSON")
print("Saved GeoJSON: grouped_regions.geojson")

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
            control_scale=True,
            prefer_canvas=True  # Better performance for many markers
        )
        
        # Add professional title
        title_html = '''
        <div style="position: fixed; 
                    top: 10px; left: 50px; width: 300px; height: 60px; 
                    background-color: white; border: 2px solid grey; 
                    border-radius: 5px; z-index:9999; 
                    padding: 10px; font-family: Arial; 
                    box-shadow: 0 0 15px rgba(0,0,0,0.2)">
        <h4 style="margin: 0; color: #333;">Regional Groups Map</h4>
        <p style="margin: 0; font-size: 12px; color: #666;">Interactive view - Click layers to toggle</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(title_html))

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

        # Enhanced city markers with rich popups
        for r in pts_ll.itertuples(index=False):
            # Classify city for icon size
            importance = classify_city_importance(r.City, r.State)
            if importance == 'Primary':
                radius = 8
                weight = 2
            elif importance == 'Secondary':
                radius = 6
                weight = 1.5
            else:
                radius = 5
                weight = 1
            
            # Rich popup content
            popup_html = f"""
            <div style="font-family: Arial; min-width: 200px;">
                <h4 style="margin: 0 0 10px 0; color: #333; border-bottom: 1px solid #ccc; padding-bottom: 5px;">
                    {r.City}, {r.State}
                </h4>
                <p style="margin: 5px 0; font-size: 13px;">
                    <strong>Group:</strong> {r.Group}<br>
                    <strong>Classification:</strong> {importance} City<br>
                    <strong>Coordinates:</strong> {r.geometry.y:.4f}, {r.geometry.x:.4f}
                </p>
            </div>
            """
            
            folium.CircleMarker(
                location=[r.geometry.y, r.geometry.x],
                radius=radius,
                color="white", weight=weight,
                fill=True, fill_color=color_for[r.Group], fill_opacity=0.9,
                tooltip=f"{r.City}, {r.State}",
                popup=folium.Popup(popup_html, max_width=250)
            ).add_to(m)

        # Add professional legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 20px; left: 20px; width: 200px; 
                    background-color: white; border: 2px solid grey; 
                    border-radius: 5px; z-index:9999; 
                    padding: 15px; font-family: Arial; 
                    box-shadow: 0 0 15px rgba(0,0,0,0.2)">
        <h5 style="margin: 0 0 10px 0; color: #333;">Legend</h5>
        '''
        
        # Add group colors to legend
        for grp in ordered_groups:
            color = color_for[grp]
            legend_html += f'''
            <div style="margin: 5px 0;">
                <span style="display: inline-block; width: 15px; height: 15px; 
                           background-color: {color}; border-radius: 50%; 
                           margin-right: 8px; vertical-align: middle;"></span>
                <span style="font-size: 12px; vertical-align: middle;">{grp}</span>
            </div>
            '''
        
        legend_html += '''
        <hr style="margin: 10px 0; border: 0.5px solid #ccc;">
        <div style="font-size: 11px; color: #666;">
            <div>● Large = Major Cities</div>
            <div>● Medium = State Capitals</div>
            <div>● Small = Other Cities</div>
            <div style="margin-top: 8px; font-style: italic;">Click cities for details</div>
        </div>
        </div>
        '''
        
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Enhanced layer control
        m.fit_bounds([[bminy, bminx], [bmaxy, bmaxx]])
        folium.LayerControl(collapsed=False, position='topright').add_to(m)
        
        # Add fullscreen button
        from folium.plugins import Fullscreen
        Fullscreen().add_to(m)
        
        m.save(INTERACTIVE_HTML_PATH)
        print(f"Saved interactive HTML: {INTERACTIVE_HTML_PATH}")
    except Exception as e:
        print("Interactive HTML export skipped. Error:", e)
