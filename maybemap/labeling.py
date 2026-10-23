# Cartographic labeling system for city-point maps, plus the geometry helpers it
# depends on. Moved verbatim from the original single-file maybeMap.py.

from math import cos, hypot, isclose, pi, sin, sqrt

from maybemap.config import (
    LABEL_EXCLUSION_RADIUS_M,
    LABEL_FONTSIZE_PRIMARY,
    LABEL_FONTSIZE_SECONDARY,
    LABEL_FONTSIZE_TERTIARY,
    LABEL_HALO,
    LABEL_LINE_ALPHA,
    LABEL_LINE_WIDTH,
    MAX_LABEL_DISTANCE_M,
)


def _circle_from_two(p1, p2):
    (x1, y1), (x2, y2) = p1, p2
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    r = hypot(x1 - x2, y1 - y2) / 2.0
    return (cx, cy, r)


def _circle_from_three(p1, p2, p3):
    (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3
    d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if isclose(d, 0.0, abs_tol=1e-9):
        return None
    ux = (
        (x1**2 + y1**2) * (y2 - y3) + (x2**2 + y2**2) * (y3 - y1) + (x3**2 + y3**2) * (y1 - y2)
    ) / d
    uy = (
        (x1**2 + y1**2) * (x3 - x2) + (x2**2 + y2**2) * (x1 - x3) + (x3**2 + y3**2) * (x2 - x1)
    ) / d
    r = hypot(ux - x1, uy - y1)
    return (ux, uy, r)


def _covers_all(cx, cy, r, pts, eps=1e-6):
    rr = r + eps
    for x, y in pts:
        if hypot(x - cx, y - cy) > rr:
            return False
    return True


def minimum_enclosing_circle(points):
    n = len(points)
    if n == 0:
        return None
    if n == 1:
        x, y = points[0]
        return (x, y, 0.0)
    best = None
    for i in range(n):
        for j in range(i + 1, n):
            c = _circle_from_two(points[i], points[j])
            if _covers_all(*c, points):
                if best is None or c[2] < best[2]:
                    best = c
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                c = _circle_from_three(points[i], points[j], points[k])
                if c is None:
                    continue
                if _covers_all(*c, points):
                    if best is None or c[2] < best[2]:
                        best = c
    if best is None:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        cxm, cym = sum(xs) / n, sum(ys) / n
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
        "AL": "MONTGOMERY",
        "AK": "JUNEAU",
        "AZ": "PHOENIX",
        "AR": "LITTLE ROCK",
        "CA": "SACRAMENTO",
        "CO": "DENVER",
        "CT": "HARTFORD",
        "DE": "DOVER",
        "FL": "TALLAHASSEE",
        "GA": "ATLANTA",
        "HI": "HONOLULU",
        "ID": "BOISE",
        "IL": "SPRINGFIELD",
        "IN": "INDIANAPOLIS",
        "IA": "DES MOINES",
        "KS": "TOPEKA",
        "KY": "FRANKFORT",
        "LA": "BATON ROUGE",
        "ME": "AUGUSTA",
        "MD": "ANNAPOLIS",
        "MA": "BOSTON",
        "MI": "LANSING",
        "MN": "SAINT PAUL",
        "MS": "JACKSON",
        "MO": "JEFFERSON CITY",
        "MT": "HELENA",
        "NE": "LINCOLN",
        "NV": "CARSON CITY",
        "NH": "CONCORD",
        "NJ": "TRENTON",
        "NM": "SANTA FE",
        "NY": "ALBANY",
        "NC": "RALEIGH",
        "ND": "BISMARCK",
        "OH": "COLUMBUS",
        "OK": "OKLAHOMA CITY",
        "OR": "SALEM",
        "PA": "HARRISBURG",
        "RI": "PROVIDENCE",
        "SC": "COLUMBIA",
        "SD": "PIERRE",
        "TN": "NASHVILLE",
        "TX": "AUSTIN",
        "UT": "SALT LAKE CITY",
        "VT": "MONTPELIER",
        "VA": "RICHMOND",
        "WA": "OLYMPIA",
        "WV": "CHARLESTON",
        "WI": "MADISON",
        "WY": "CHEYENNE",
    }

    # Check if it's a state capital
    is_state_capital = state_upper in state_capitals and city_upper == state_capitals[state_upper]

    # Simple heuristics for major cities
    major_metros = {
        "NEW YORK",
        "LOS ANGELES",
        "CHICAGO",
        "HOUSTON",
        "PHOENIX",
        "PHILADELPHIA",
        "SAN ANTONIO",
        "SAN DIEGO",
        "DALLAS",
        "SAN JOSE",
        "AUSTIN",
        "JACKSONVILLE",
        "SAN FRANCISCO",
        "CHARLOTTE",
        "SEATTLE",
        "DENVER",
        "WASHINGTON",
        "BOSTON",
        "DETROIT",
        "NASHVILLE",
        "ATLANTA",
        "MIAMI",
        "TAMPA",
        "NEW ORLEANS",
        "CLEVELAND",
        "MINNEAPOLIS",
        "KANSAS CITY",
        "SAINT LOUIS",
        "PITTSBURGH",
        "CINCINNATI",
        "INDIANAPOLIS",
        "COLUMBUS",
        "MILWAUKEE",
        "BALTIMORE",
        "ORLANDO",
        "RALEIGH",
        "RICHMOND",
        "NORFOLK",
        "BIRMINGHAM",
        "MEMPHIS",
        "BUFFALO",
        "SACRAMENTO",
        "PORTLAND",
        "LAS VEGAS",
        "OKLAHOMA CITY",
        "LOUISVILLE",
        "TUCSON",
        "ALBUQUERQUE",
        "FRESNO",
        "OMAHA",
        "HONOLULU",
    }

    # Enhanced logic for better classification
    if city_upper in major_metros:
        return "Primary"
    elif is_state_capital:
        return "Secondary"
    elif any(
        keyword in city_upper for keyword in ["SAINT", "ST.", "FORT", "WEST ", "NORTH ", "SOUTH "]
    ):
        # Cities with common prefixes are often significant
        return "Secondary"
    else:
        return "Tertiary"


def get_label_style(importance: str, text_color: str) -> dict:
    """Get typography settings based on city importance."""
    if importance == "Primary":
        return {
            "fontsize": LABEL_FONTSIZE_PRIMARY,
            "weight": "bold",
            "color": text_color,
            "exclusion_radius": LABEL_EXCLUSION_RADIUS_M * 1.5,
        }
    elif importance == "Secondary":
        return {
            "fontsize": LABEL_FONTSIZE_SECONDARY,
            "weight": "normal",
            "color": text_color,
            "exclusion_radius": LABEL_EXCLUSION_RADIUS_M * 1.2,
        }
    else:  # Tertiary
        return {
            "fontsize": LABEL_FONTSIZE_TERTIARY,
            "weight": "normal",
            "color": text_color,
            "exclusion_radius": LABEL_EXCLUSION_RADIUS_M,
        }


def analyze_map_geography(pts_df):
    """Analyze map bounds and city distribution for smart label placement."""

    # Calculate map bounds
    all_x = [geom.x for geom in pts_df.geometry]
    all_y = [geom.y for geom in pts_df.geometry]

    map_bounds = {
        "min_x": min(all_x),
        "max_x": max(all_x),
        "min_y": min(all_y),
        "max_y": max(all_y),
        "center_x": (min(all_x) + max(all_x)) / 2,
        "center_y": (min(all_y) + max(all_y)) / 2,
    }

    # For each city, analyze density in 8 cardinal directions
    city_analysis = {}
    for _, row in pts_df.iterrows():
        city_x, city_y = row.geometry.x, row.geometry.y

        # Count cities in each direction (within 100km radius)
        directions = {"N": 0, "NE": 0, "E": 0, "SE": 0, "S": 0, "SW": 0, "W": 0, "NW": 0}

        for _, other_row in pts_df.iterrows():
            if row.City == other_row.City:  # Skip self
                continue

            other_x, other_y = other_row.geometry.x, other_row.geometry.y
            dx, dy = other_x - city_x, other_y - city_y
            distance = sqrt(dx * dx + dy * dy)

            if distance > 100000:  # Only consider nearby cities (100km)
                continue

            # Determine direction (8-way)
            if abs(dx) < distance * 0.383:  # ~22.5 degree tolerance
                if dy > 0:
                    directions["N"] += 1
                else:
                    directions["S"] += 1
            elif abs(dy) < distance * 0.383:
                if dx > 0:
                    directions["E"] += 1
                else:
                    directions["W"] += 1
            else:  # Diagonal
                if dx > 0 and dy > 0:
                    directions["NE"] += 1
                elif dx > 0 and dy < 0:
                    directions["SE"] += 1
                elif dx < 0 and dy > 0:
                    directions["NW"] += 1
                else:
                    directions["SW"] += 1

        # Determine position relative to map bounds
        edge_threshold = 0.15  # Within 15% of edge
        is_edge = {
            "north": (city_y - map_bounds["min_y"]) / (map_bounds["max_y"] - map_bounds["min_y"])
            > (1 - edge_threshold),
            "south": (city_y - map_bounds["min_y"]) / (map_bounds["max_y"] - map_bounds["min_y"])
            < edge_threshold,
            "east": (city_x - map_bounds["min_x"]) / (map_bounds["max_x"] - map_bounds["min_x"])
            > (1 - edge_threshold),
            "west": (city_x - map_bounds["min_x"]) / (map_bounds["max_x"] - map_bounds["min_x"])
            < edge_threshold,
        }

        city_analysis[row.City] = {
            "directions": directions,
            "is_edge": is_edge,
            "position": (city_x, city_y),
        }

    return map_bounds, city_analysis


def professional_label_placement(
    ax, gdf_pts, color_for, text_color, leader_color, use_numbering=True, halo_color="white"
):
    """
    Geographic-aware professional cartographic labeling system with:
    • Smart directional placement toward empty spaces
    • Hierarchical typography based on city importance
    • Collision-free placement using exclusion zones
    • Clean leader line routing
    """
    # Classify cities by importance
    pts_df = gdf_pts[["Group", "geometry", "City", "State"]].reset_index(drop=True)
    pts_df["Importance"] = pts_df.apply(
        lambda row: classify_city_importance(row.City, row.State), axis=1
    )

    # Sort by importance first, then by density (dense areas get priority within each tier)
    importance_order = {"Primary": 0, "Secondary": 1, "Tertiary": 2}
    pts_df["imp_order"] = pts_df["Importance"].map(importance_order)

    # Calculate local density for each city
    pts_df["density_score"] = 0
    for i, row in pts_df.iterrows():
        nearby_count = 0
        for j, other_row in pts_df.iterrows():
            if i != j:
                dist = sqrt(
                    (row.geometry.x - other_row.geometry.x) ** 2
                    + (row.geometry.y - other_row.geometry.y) ** 2
                )
                if dist < 50000:  # Within 50km
                    nearby_count += 1
        pts_df.at[i, "density_score"] = nearby_count

    # Sort by importance first, then by density (highest density first within each importance tier)
    pts_df = pts_df.sort_values(
        ["imp_order", "density_score"], ascending=[True, False]
    ).reset_index(drop=True)

    # Analyze map geography for smart placement
    map_bounds, city_analysis = analyze_map_geography(pts_df)
    print(f"   Map analysis: {len(city_analysis)} cities analyzed")

    # Track placed labels to avoid overlaps
    placed_labels = []  # List of (x, y, exclusion_radius) tuples
    numbered_cities = {}  # Dictionary for cities using numbered markers
    dense_area_threshold = 16  # Cities with 16+ neighbors get numbered (excludes Columbia IL)

    def find_spiral_position(pin_x, pin_y, number, label_style):
        """Place numbered labels in a spiral pattern around dense areas."""

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
        exclusion_radius = label_style["exclusion_radius"]

        # Manual positioning overrides for specific cities
        manual_positions = {
            "Harrisburg": (-20000, 0),  # Shift left (west)
            "Lebanon": (0, 15000),  # Shift up (north) between Collinsville and O'Fallon IL
        }

        # Specific city-state combinations
        if city_name == "Columbia" and city_state == "Illinois":
            return pin_x - 25000, pin_y - 25000  # Columbia IL: shift SW, left of Waterloo

        if city_name in manual_positions:
            offset_x, offset_y = manual_positions[city_name]
            return pin_x + offset_x, pin_y + offset_y

        # Get geographic analysis for this city
        city_data = city_analysis.get(city_name, {})
        directions = city_data.get("directions", {})
        is_edge = city_data.get("is_edge", {})

        # Determine preferred directions based on geography
        preferred_angles = []

        # For edge cities, prefer inward directions
        if is_edge.get("north", False):  # Northern edge - place labels south
            preferred_angles.extend([4.712, 5.498, 3.927])  # S, SW, SE
        if is_edge.get("south", False):  # Southern edge - place labels north
            preferred_angles.extend([1.571, 0.785, 2.356])  # N, NE, NW
        if is_edge.get("east", False):  # Eastern edge - place labels west
            preferred_angles.extend([3.142, 2.356, 3.927])  # W, NW, SW
        if is_edge.get("west", False):  # Western edge - place labels east
            preferred_angles.extend([0, 0.785, 5.498])  # E, NE, SE

        # For interior cities, prefer directions with fewer nearby cities
        if not preferred_angles:  # Not an edge city
            direction_angles = {
                "N": 1.571,
                "NE": 0.785,
                "E": 0,
                "SE": 5.498,
                "S": 4.712,
                "SW": 3.927,
                "W": 3.142,
                "NW": 2.356,
            }

            # Sort directions by density (fewer cities = better)
            sorted_dirs = sorted(directions.items(), key=lambda x: x[1])
            preferred_angles = [direction_angles[dir_name] for dir_name, _ in sorted_dirs[:4]]

        # Add some variation around preferred angles
        all_angles = []
        for base_angle in preferred_angles:
            all_angles.extend([base_angle, base_angle + 0.262, base_angle - 0.262])  # ±15 degrees

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
                for placed_x, placed_y, placed_radius in placed_labels:
                    dist = sqrt((label_x - placed_x) ** 2 + (label_y - placed_y) ** 2)

                    # Use a reasonable base separation (8km) plus label-specific exclusion zones
                    base_separation = 8000  # 8km base minimum
                    min_separation = base_separation + exclusion_radius + placed_radius

                    if dist < min_separation:
                        collision = True
                        break

                if not collision:
                    # Double-check: ensure we're not too close to any existing label
                    if placed_labels:
                        actual_min_dist = min(
                            [
                                sqrt((label_x - px) ** 2 + (label_y - py) ** 2)
                                for px, py, pr in placed_labels
                            ]
                        )
                        if actual_min_dist < 8000:  # Too close to any existing label
                            collision = True

                if not collision:
                    direction_name = "preferred" if angle in preferred_angles[:4] else "fallback"
                    print(
                        f"      {city_name}: Found {direction_name} position at "
                        f"{distance}m after {positions_tried} attempts"
                    )
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
            min_dist = min(
                [sqrt((test_x - px) ** 2 + (test_y - py) ** 2) for px, py, pr in placed_labels],
                default=float("inf"),
            )
            if min_dist >= 12000:  # 12km minimum for fallback
                return test_x, test_y

        # If all cardinal directions fail, use preferred angle or default
        fallback_angle = preferred_angles[0] if preferred_angles else 0
        return pin_x + fallback_distance * cos(fallback_angle), pin_y + fallback_distance * sin(
            fallback_angle
        )

    # Place labels in order of importance
    texts = []
    leader_lines = []

    for _, row in pts_df.iterrows():
        pin_x, pin_y = row.geometry.x, row.geometry.y
        label_style = get_label_style(row.Importance, text_color)

        # Check if this city is in a very dense area (only for static PNG)
        density_score = row.get("density_score", 0)

        # Specific exclusions from numbering
        is_excluded = (
            row.City == "Columbia" and row.State == "Illinois"
        )  # Columbia IL should be normal label

        use_number = use_numbering and density_score >= dense_area_threshold and not is_excluded

        if use_number:
            # Assign a number and use spiral placement for dense areas
            number = len(numbered_cities) + 1
            numbered_cities[number] = f"{row.City}, {row.State}"

            # Use spiral placement around dense area center
            label_x, label_y = find_spiral_position(pin_x, pin_y, number, label_style)

            # Place numbered label at the found position
            text = ax.text(
                label_x,
                label_y,
                str(number),
                fontsize=label_style["fontsize"],
                fontweight=label_style["weight"],
                color="black",
                ha="center",
                va="center",
                zorder=12,
            )

            # Always draw leader line from pin to numbered label
            line = ax.plot(
                [pin_x, label_x],
                [pin_y, label_y],
                color=leader_color,
                linewidth=LABEL_LINE_WIDTH,
                alpha=LABEL_LINE_ALPHA,
                zorder=9,
            )
            leader_lines.extend(line)

            # Record this label's exclusion zone for future collision detection
            placed_labels.append((label_x, label_y, label_style["exclusion_radius"]))

            texts.append(text)
            print(
                f"   Numbered {row.City}: #{number} pin({pin_x:.0f}, {pin_y:.0f}) "
                f"-> label({label_x:.0f}, {label_y:.0f})"
            )
        else:
            # Find optimal label position using geographic awareness
            label_x, label_y = find_placement_position(
                pin_x, pin_y, label_style, row.City, row.State
            )
            print(
                f"   Placing {row.City}: pin({pin_x:.0f}, {pin_y:.0f}) "
                f"-> label({label_x:.0f}, {label_y:.0f})"
            )

            # Create text label
            text = ax.text(
                label_x,
                label_y,
                row.City,
                fontsize=label_style["fontsize"],
                weight=label_style["weight"],
                color=label_style["color"],
                ha="center",
                va="center",
                zorder=10,
            )

        if LABEL_HALO:
            import matplotlib.patheffects as pe

            # The halo has to contrast with the text, not with the page: white text on
            # a dark basemap wearing a white halo is a smudge rather than a label.
            text.set_path_effects([pe.Stroke(linewidth=3.0, foreground=halo_color), pe.Normal()])

            # Only for text labels, handle leader lines and exclusion zones
            texts.append(text)

            # Record this label's exclusion zone for future collision detection
            placed_labels.append((label_x, label_y, label_style["exclusion_radius"]))

            # Create leader line if label is not at pin
            distance = sqrt((label_x - pin_x) ** 2 + (label_y - pin_y) ** 2)
            if distance > 1000:  # Only draw leader if label is displaced
                line = ax.plot(
                    [pin_x, label_x],
                    [pin_y, label_y],
                    color=leader_color,
                    linewidth=LABEL_LINE_WIDTH,
                    alpha=LABEL_LINE_ALPHA,
                    zorder=9,
                )
                leader_lines.extend(line)

    return texts, leader_lines, numbered_cities
