# Static PNG rendering, shared by the Excel and KMZ map sources.

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this runs on a server with no display

import contextily as cx  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import mercantile  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from maybemap.config import (  # noqa: E402
    BASEMAP,
    CIRCLE_ALPHA_EDGE,
    CIRCLE_ALPHA_FILL,
    CIRCLE_EDGE_WIDTH,
    CITY_BOUNDARY_EDGE_WIDTH,
    CITY_BOUNDARY_FILL_ALPHA,
    CITY_DOT_EDGE_COLOR,
    CITY_DOT_EDGE_WIDTH,
    CITY_DOT_SIZE_PRIMARY,
    CITY_DOT_SIZE_SECONDARY,
    CITY_DOT_SIZE_TERTIARY,
    LABEL_FONTSIZE_SECONDARY,
    LABEL_STRATEGY,
    MAX_BASEMAP_TILES,
    PNG_EXTENT_MARGIN,
    PNG_EXTENT_STATES,
    PROJECT_ROOT,
    SHOW_LABELS,
    SHOW_LEGEND,
    ZONE_CIRCLE_LABEL_FONTSIZE,
    ZONE_CIRCLE_LABEL_OFFSET,
    ZONE_LABEL_FONTSIZE,
)
from maybemap.labeling import classify_city_importance, professional_label_placement  # noqa: E402
from maybemap.styling import basemap_theme, lighten  # noqa: E402

# CartoDB serves no deeper than this, and nothing here is a street map anyway
MAX_TILE_ZOOM = 19


def _tile_count(bounds, zoom: int) -> int:
    # Tiles contextily would stitch together to cover bounds, as (west, south, east,
    # north) in lon/lat. Counted from the corner tiles rather than by walking the
    # grid: the whole point is to ask about zooms too big to enumerate.
    west, south, east, north = bounds
    left, top = mercantile.tile(west, north, zoom)[:2]
    right, bottom = mercantile.tile(east, south, zoom)[:2]
    return (right - left + 1) * (bottom - top + 1)


def _axes_bounds(ax):
    # The axes extent in lon/lat. Everything drawn by this module is EPSG:3857.
    x0, x1 = sorted(ax.get_xlim())
    y0, y1 = sorted(ax.get_ylim())
    west, south = mercantile.lnglat(x0, y0)
    east, north = mercantile.lnglat(x1, y1)
    return west, south, east, north


def _resolve_basemap_zoom(
    ax, requested, *, max_tiles: int = MAX_BASEMAP_TILES, provider=None
) -> int:
    # A tile zoom is a resolution, not a size: contextily builds one array spanning
    # the whole extent, so holding the zoom fixed while the data spreads out grows
    # the mosaic without limit. Both the requested zoom and "auto" resolve to the
    # sharpest basemap that still fits the tile budget.
    bounds = _axes_bounds(ax)
    # Providers do not all go as deep as CartoDB: Esri's Canvas layers stop at 16,
    # and asking past a provider's own limit answers 404 rather than a coarser tile.
    deepest = MAX_TILE_ZOOM
    if provider is not None:
        deepest = min(deepest, int(provider.get("max_zoom", MAX_TILE_ZOOM)))
    ceiling = next(
        (z for z in range(deepest, -1, -1) if _tile_count(bounds, z) <= max_tiles),
        0,
    )
    if requested == "auto":
        return ceiling

    zoom = int(requested)
    if zoom <= ceiling:
        return zoom
    # Silently rendering a coarser basemap than asked for reads as a bug later
    print(
        f"   Basemap zoom {zoom} needs {_tile_count(bounds, zoom)} tiles for this extent, "
        f"over the {max_tiles}-tile budget -- falling back to zoom {ceiling}"
    )
    return ceiling


def _extent_bounds(data, states):
    # Bounding box in EPSG:3857 the PNG should cover, or None to keep whatever the
    # artists drew. Towns outside `states` are left out of frame, but every region
    # and circle is folded back in -- cropping the map's own subject is never right.
    if not states or data.points is None or "State" not in data.points.columns:
        return None
    keep = data.points[data.points["State"].isin(states)]
    if keep.empty:
        # A state list matching nothing is a typo, not a request for an empty map
        return None

    frames = [keep, data.regions, data.circles, data.detail_polys]
    boxes = [f.total_bounds for f in frames if f is not None and not f.empty]
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _apply_extent(ax, data, *, states=PNG_EXTENT_STATES, margin=PNG_EXTENT_MARGIN):
    bounds = _extent_bounds(data, states)
    if bounds is None:
        return
    x0, y0, x1, y1 = bounds
    pad_x, pad_y = (x1 - x0) * margin, (y1 - y0) * margin
    ax.set_xlim(x0 - pad_x, x1 + pad_x)
    ax.set_ylim(y0 - pad_y, y1 + pad_y)
    # add_basemap and savefig both re-read the extent; nothing may widen it again
    ax.set_autoscale_on(False)


def _draw_polys(ax, polys, label_col, data, *, zorder, fill_alpha=None, edge_width=None):
    # Per-row Color when the source supplied one -- city boundaries take the color of
    # the dot they replace -- otherwise the group's entry in the shared palette.
    fill_alpha = data.fill_alpha if fill_alpha is None else fill_alpha
    edge_width = data.edge_width if edge_width is None else edge_width
    for grp, sub in polys.groupby(label_col, observed=True):
        bases = list(sub["Color"]) if "Color" in sub.columns else [data.colors[grp]] * len(sub)
        sub.plot(
            ax=ax,
            facecolor=[mcolors.to_rgba(b, fill_alpha) for b in bases],
            edgecolor=[mcolors.to_rgba(b, data.edge_alpha) for b in bases],
            linewidth=edge_width,
            zorder=zorder,
        )


def _draw_city_dots(ax, data):
    # One pin per point, sized by the same classification the popups use unless
    # the source asked for uniform pins
    sizes = {
        "Primary": CITY_DOT_SIZE_PRIMARY,
        "Secondary": CITY_DOT_SIZE_SECONDARY,
        "Tertiary": CITY_DOT_SIZE_TERTIARY,
    }
    points = data.points
    if "HasBoundary" in points.columns:
        # Only the cities with no boundary of their own still need a dot
        points = points[~points["HasBoundary"]]
    if points.empty:
        return
    if data.point_size is None:
        marker_sizes = [
            sizes[classify_city_importance(r.City, r.State)] for r in points.itertuples()
        ]
    else:
        # A task is a work item, not a place: every pin carries the same weight
        marker_sizes = [data.point_size] * len(points)
    ax.scatter(
        [g.x for g in points.geometry],
        [g.y for g in points.geometry],
        s=marker_sizes,
        c=list(points["Color"]),
        edgecolors=CITY_DOT_EDGE_COLOR,
        # Scale the halo with the dot so small markers are not swallowed by it
        linewidths=[
            CITY_DOT_EDGE_WIDTH * (size / CITY_DOT_SIZE_PRIMARY) ** 0.5 for size in marker_sizes
        ],
        zorder=8,
    )


def _data_per_point(ax, fig):
    # Data units per typographic point. Aspect is equal, so whichever axis is
    # more constrained sets the scale for both.
    w_in, h_in = fig.get_size_inches()
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    return max((x1 - x0) / (w_in * 72), (y1 - y0) / (h_in * 72))


def _text_box(x, y, text, fontsize, scale, ha, va):
    # Rough extent of a bold label: 0.6em average advance width, 1.2em line height
    half_w = len(text) * fontsize * 0.6 * scale / 2
    half_h = fontsize * 1.2 * scale / 2
    cx = x + (half_w if ha == "left" else -half_w if ha == "right" else 0)
    cy = y + (half_h if va == "bottom" else -half_h if va == "top" else 0)
    return (cx - half_w, cy - half_h, cx + half_w, cy + half_h)


def _overlaps(a, b, pad=0.0):
    return not (a[2] + pad < b[0] or a[0] > b[2] + pad or a[3] + pad < b[1] or a[1] > b[3] + pad)


def _draw_circle_labels(ax, data, theme, obstacles):
    # Region name just outside its circle. Tries top, then bottom, then the
    # sides -- 13 labels in one state collide constantly if placed blindly.
    scale = _data_per_point(ax, ax.figure)
    pad = 2 * scale  # keep a hair of whitespace between neighbouring labels
    placed = list(obstacles)

    # Smallest circles first: they sit in the crowded areas and need first pick
    rows = sorted(data.circles.itertuples(index=False), key=lambda r: r.r)
    for row in rows:
        center = row.geometry.centroid
        out = row.r + row.r * ZONE_CIRCLE_LABEL_OFFSET
        diag = out * 0.7071
        candidates = [
            (center.x, center.y + out, "center", "bottom"),
            (center.x, center.y - out, "center", "top"),
            (center.x + out, center.y, "left", "center"),
            (center.x - out, center.y, "right", "center"),
            (center.x + diag, center.y + diag, "left", "bottom"),
            (center.x - diag, center.y + diag, "right", "bottom"),
            (center.x + diag, center.y - diag, "left", "top"),
            (center.x - diag, center.y - diag, "right", "top"),
        ]

        best, best_box, best_cost = None, None, None
        for x, y, ha, va in candidates:
            box = _text_box(x, y, row.Group, ZONE_CIRCLE_LABEL_FONTSIZE, scale, ha, va)
            cost = sum(1 for other in placed if _overlaps(box, other, pad))
            if cost == 0:
                best, best_box = (x, y, ha, va), box
                break
            if best_cost is None or cost < best_cost:
                best, best_box, best_cost = (x, y, ha, va), box, cost

        x, y, ha, va = best
        placed.append(best_box)
        ax.text(
            x,
            y,
            row.Group,
            fontsize=ZONE_CIRCLE_LABEL_FONTSIZE,
            weight="bold",
            color=data.colors[row.Group],
            ha=ha,
            va=va,
            zorder=10,
            path_effects=[pe.withStroke(linewidth=2.5, foreground=theme["halo_color"])],
        )


def _draw_region_labels(ax, data, theme):
    # Only a handful of regions, so an interior point per polygon is enough --
    # none of the collision machinery the city maps need. Returns the label
    # boxes so any circle labels drawn afterwards can steer around them.
    scale = _data_per_point(ax, ax.figure)
    boxes = []
    for row in data.regions.itertuples(index=False):
        point = row.geometry.representative_point()
        boxes.append(
            _text_box(point.x, point.y, row.Label, ZONE_LABEL_FONTSIZE, scale, "center", "center")
        )
        ax.text(
            point.x,
            point.y,
            row.Label,
            fontsize=ZONE_LABEL_FONTSIZE,
            weight="bold",
            color=theme["textcolor"],
            ha="center",
            va="center",
            zorder=10,
            path_effects=[pe.withStroke(linewidth=3, foreground=theme["halo_color"])],
        )
    return boxes


def render_png(data, *, basemap: str = BASEMAP, output_dir=PROJECT_ROOT) -> str:
    print("Generating map...")
    theme = basemap_theme(basemap)

    fig, ax = plt.subplots(figsize=(13, 10), constrained_layout=True)

    if data.circles is not None and not data.circles.empty:
        for grp, sub in data.circles.groupby("Group", observed=False):
            base = data.colors[grp]
            sub.plot(
                ax=ax,
                facecolor=mcolors.to_rgba(lighten(base, 0.70), CIRCLE_ALPHA_FILL),
                edgecolor=mcolors.to_rgba(base, CIRCLE_ALPHA_EDGE),
                linewidth=CIRCLE_EDGE_WIDTH,
                zorder=4,
            )

    # Zone maps draw the regions and then the city shapes on top of them; the Excel
    # map's regions are dissolved group blobs it only exports, so it draws neither
    has_detail = data.detail_polys is not None and not data.detail_polys.empty
    if data.draw_regions:
        _draw_polys(ax, data.regions, "Label", data, zorder=5)
        if has_detail:
            _draw_polys(
                ax,
                data.detail_polys,
                "Group",
                data,
                zorder=6,
                fill_alpha=CITY_BOUNDARY_FILL_ALPHA,
                edge_width=CITY_BOUNDARY_EDGE_WIDTH,
            )
    elif has_detail:
        _draw_polys(ax, data.detail_polys, "Group", data, zorder=5)

    if data.draw_city_dots:
        _draw_city_dots(ax, data)

    # Before any labeling: the placement code sizes its collision boxes off the
    # axes extent, so it has to see the final one rather than the autoscaled one
    _apply_extent(ax, data)

    numbered_cities = {}
    # A source can mark which cities are worth naming; without the column, all of them
    to_label = data.points
    if to_label is not None and "Labeled" in to_label.columns:
        to_label = to_label[to_label["Labeled"]]

    if to_label is not None and not to_label.empty and data.label_cities:
        print("Applying professional cartographic labeling system...")
        if LABEL_STRATEGY == "professional_cartographic":
            _, _, numbered_cities = professional_label_placement(
                ax,
                to_label,
                data.colors,
                theme["textcolor"],
                theme["leader_color"],
                halo_color=theme["halo_color"],
            )
            print("   Professional labeling complete - no overlapping labels!")
        else:
            lbl = to_label.sort_values("City").groupby("Group").head(1)
            for _, row in lbl.groupby("Group"):
                p = row.geometry.iloc[0]
                ax.text(
                    p.x + 2000,
                    p.y + 2000,
                    row.City.iloc[0],
                    fontsize=LABEL_FONTSIZE_SECONDARY,
                    color=theme["textcolor"],
                    zorder=8,
                )

    zone_label_boxes = []
    if data.label_regions:
        zone_label_boxes = _draw_region_labels(ax, data, theme)

    if data.label_circles and data.circles is not None and not data.circles.empty:
        _draw_circle_labels(ax, data, theme, zone_label_boxes)

    # Resolved once the artists have set the extent, and before any tile is fetched
    zoom = _resolve_basemap_zoom(ax, data.basemap_zoom, provider=theme["base_src"])
    cx.add_basemap(ax, source=theme["base_src"], zoom=zoom)
    # Drawing the overlay over a basemap that already carries its own place names
    # just prints them twice, slightly offset
    if SHOW_LABELS and not theme["labels_in_base"]:
        cx.add_basemap(ax, source=theme["label_src"], zoom=zoom, zorder=3)

    if SHOW_LEGEND or not data.label_regions:
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=data.colors[g],
                markeredgecolor="none",
                markersize=8,
                label=g,
            )
            for g in data.order
        ]
        ax.legend(
            handles=handles,
            title=data.legend_title,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
        )

    if numbered_cities:
        legend_text = "STL Metro Cities:\n"
        for num, city_name in sorted(numbered_cities.items()):
            legend_text += f"{num}. {city_name}\n"
        ax.text(
            1.02,
            0.3,
            legend_text,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
        )

    ax.set_title(f"{data.title} ({theme['label']} basemap)", fontsize=15)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()

    out_path = str(Path(output_dir) / f"{data.output_stem}_map.png")
    try:
        fig.savefig(out_path, dpi=220, bbox_inches="tight")
    except ValueError:
        fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"Saved PNG: {out_path}")
    return out_path
