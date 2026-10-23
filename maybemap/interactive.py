# Interactive Folium HTML export, shared by the Excel and KMZ map sources.

from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd
from branca.element import MacroElement
from folium.plugins import Fullscreen
from jinja2 import Template
from shapely.geometry import box

from maybemap.config import (
    BASEMAP,
    CIRCLE_ALPHA_FILL,
    CIRCLE_EDGE_WIDTH,
    CITY_BOUNDARY_EDGE_WIDTH,
    CITY_BOUNDARY_FILL_ALPHA,
    PROJECT_ROOT,
)
from maybemap.labeling import classify_city_importance
from maybemap.styling import basemap_theme, lighten

# Customer-count buckets for the filter box, as inclusive (low, high) pairs. The
# last one is open-ended: the counts run from 1 to tens of thousands, so past a
# couple of dozen the exact number stops being a category.
CUSTOMER_BUCKETS = [(1, 5), (6, 10), (11, 20), (21, None)]

# The export's Service Mix values, shortened for the filter box
SERVICE_TYPES = {
    "Business Only": "Business",
    "Residential Only": "Residential",
    "Mixed": "Mixed",
}

TITLE_TEMPLATE = """
<div style="position: fixed;
            top: 10px; left: 50px; width: 300px;
            background-color: white; border: 2px solid grey;
            border-radius: 5px; z-index:9999;
            padding: 10px; font-family: Arial;
            box-shadow: 0 0 15px rgba(0,0,0,0.2)">
<h4 style="margin: 0; color: #333;">{title}</h4>
<p style="margin: 0; font-size: 12px; color: #666;">Interactive view - Click layers to toggle</p>
</div>
"""


def _bounds_4326(data):
    frames = [data.regions.geometry]
    for extra in (data.circles, data.detail_polys, data.points):
        if extra is not None and not extra.empty:
            frames.append(extra.geometry)
    merged = gpd.GeoSeries(pd.concat(frames, ignore_index=True), crs=3857)
    xmin, ymin, xmax, ymax = merged.total_bounds
    return gpd.GeoSeries([box(xmin, ymin, xmax, ymax)], crs=3857).to_crs(4326).total_bounds


def _add_region_polys(m, data):
    # The Excel map's regions are dissolved group blobs it only exports, never draws,
    # so it routes its per-city shapes through here instead
    polys = data.regions if data.draw_regions else data.detail_polys
    if polys is None or polys.empty:
        # The task map is pins on a basemap: it carries neither kind of polygon
        return
    label_col = "Label" if data.draw_regions else "Group"
    suffix = "zone" if data.draw_regions else "cities"
    for grp, sub in polys.to_crs(4326).groupby(label_col, observed=True):
        base = data.colors[grp]
        folium.GeoJson(
            sub.__geo_interface__,
            style_function=lambda feat, base=base: {
                "color": base,
                "weight": data.edge_width,
                "fillColor": base,
                "fillOpacity": data.fill_alpha,
            },
            tooltip=str(grp),
            name=f"{grp} – {suffix}",
        ).add_to(m)


def _add_city_polys(m, data):
    # One FeatureGroup so the layer control gets a single toggle rather than one per
    # city, but a GeoJson each so every boundary keeps its own tooltip and popup.
    # The tooltip is sticky, so the city name holds across the whole shape.
    polys = data.detail_polys.to_crs(4326)
    has_zone = "Zone" in polys.columns
    group = folium.FeatureGroup(name="City boundaries", show=True)

    for row in polys.itertuples(index=False):
        base = row.Color
        folium.GeoJson(
            gpd.GeoSeries([row.geometry], crs=4326).__geo_interface__,
            style_function=lambda feat, base=base: {
                "color": base,
                "weight": CITY_BOUNDARY_EDGE_WIDTH,
                "fillColor": base,
                "fillOpacity": CITY_BOUNDARY_FILL_ALPHA,
            },
            tooltip=f"{row.City}, {row.State}",
            popup=folium.Popup(_city_popup_html(row, has_zone), max_width=250),
        ).add_to(group)

    group.add_to(m)
    return group


def _add_region_circles(m, data):
    # Leaflet gives the hover to the last interactive layer containing the cursor, so
    # circles are added after the areas and before the city dots to get the hierarchy
    # areas -> regions -> cities. Muting them by the area's own alpha reproduces what
    # they looked like when that fill was painted over them, so nothing gets louder.
    mute = 1.0 - data.fill_alpha
    # Largest first so the smallest circle lands on top: Columbia Core sits almost
    # entirely inside Jefferson City Area and would otherwise be unreachable.
    layers = []
    circles = data.circles.to_crs(4326).sort_values("r", ascending=False)
    for grp, sub in circles.groupby("Group", observed=True, sort=False):
        base = data.colors[grp]
        pale = lighten(base, 0.70)
        layer = folium.GeoJson(
            sub.__geo_interface__,
            style_function=lambda feat, base=base, pale=pale: {
                "color": base,
                "weight": CIRCLE_EDGE_WIDTH,
                "opacity": mute,
                "fillColor": pale,
                "fillOpacity": CIRCLE_ALPHA_FILL * mute,
            },
            tooltip=str(grp),
            name=f"{grp} – circle",
        )
        layer.add_to(m)
        layers.append(layer)
    return layers


# Optional popup rows, in the order they are shown, as (row attribute, label)
POPUP_DETAIL = [
    ("Customers", "Customers"),
    ("ActiveAddresses", "Active Addresses"),
    ("Tasks6Mo", "Tasks (6 mo)"),
    ("ServiceMix", "Service"),
]


def _popup_row(row, attr: str, label: str) -> str:
    # Only the CSV export carries this detail, and only for some towns, so a value
    # the sheet never gave costs its row rather than printing an empty one. A blank
    # string is the same absence written differently -- the task export is full of them.
    value = getattr(row, attr, None)
    if value is None or pd.isna(value) or not str(value).strip():
        return ""
    return f"<strong>{label}:</strong> {value}<br>"


def _city_popup_html(row, has_zone: bool) -> str:
    importance = classify_city_importance(row.City, row.State)
    zone_row = f"<strong>Zone:</strong> {row.Zone}<br>" if has_zone else ""
    detail = "".join(_popup_row(row, attr, label) for attr, label in POPUP_DETAIL)

    # The split only says anything with both halves in hand
    residential = getattr(row, "ResidentialCustomers", None)
    business = getattr(row, "BusinessCustomers", None)
    if not any(v is None or pd.isna(v) for v in (residential, business)):
        detail += f"<strong>Residential / Business:</strong> {residential} / {business}<br>"

    point = row.geometry.representative_point()
    return f"""
        <div style="font-family: Arial; min-width: 200px;">
            <h4 style="margin: 0 0 10px 0; color: #333; border-bottom: 1px solid #ccc;
                       padding-bottom: 5px;">
                {row.City}, {row.State}
            </h4>
            <p style="margin: 5px 0; font-size: 13px;">
                {zone_row}<strong>Group:</strong> {row.Group}<br>
                {detail}<strong>Classification:</strong> {importance} City<br>
                <strong>Coordinates:</strong> {point.y:.4f}, {point.x:.4f}
            </p>
        </div>
        """


def _declared_popup_html(row, popup) -> str:
    # The source-described popup, for points that are not cities. Same shape as the
    # city one so the two read alike; a field the export left blank costs its row.
    detail = "".join(_popup_row(row, attr, label) for attr, label in popup.fields)
    subtitle = ""
    if popup.subtitle_attr:
        value = getattr(row, popup.subtitle_attr, None)
        if value is not None and not pd.isna(value):
            subtitle = f'<div style="font-size: 12px; color: #666;">{value}</div>'
    coordinates = ""
    if popup.show_coordinates:
        point = row.geometry.representative_point()
        coordinates = f"<strong>Coordinates:</strong> {point.y:.4f}, {point.x:.4f}<br>"

    return f"""
        <div style="font-family: Arial; min-width: 200px;">
            <h4 style="margin: 0 0 4px 0; color: #333;">
                {getattr(row, popup.title_attr)}
            </h4>
            {subtitle}
            <p style="margin: 8px 0 5px 0; font-size: 13px; border-top: 1px solid #ccc;
                      padding-top: 8px;">
                {detail}{coordinates}
            </p>
        </div>
        """


def _popup_renderer(data):
    # The city popup unless the source described its own
    if data.popup is None:
        return lambda row, has_zone: _city_popup_html(row, has_zone)
    return lambda row, has_zone: _declared_popup_html(row, data.popup)


def _tooltip(row, popup) -> str:
    if popup is None:
        return f"{row.City}, {row.State}"
    return str(getattr(row, popup.title_attr))


def _add_marker(group, row, has_zone: bool, *, popup_html=None, radius=None, popup=None):
    if radius is None:
        importance = classify_city_importance(row.City, row.State)
        radius, weight = {"Primary": (8, 2), "Secondary": (6, 1.5)}.get(importance, (5, 1))
    else:
        weight = 1.5
    if popup_html is None:
        popup_html = _city_popup_html

    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x],
        radius=radius,
        color="white",
        weight=weight,
        fill=True,
        fill_color=row.Color,
        fill_opacity=0.9,
        tooltip=_tooltip(row, popup),
        popup=folium.Popup(popup_html(row, has_zone), max_width=250),
    ).add_to(group)


def _customer_bucket(value):
    # Index into CUSTOMER_BUCKETS, or None for a town the export gave no count --
    # those stay on the map rather than hiding behind a checkbox they have no claim to.
    if pd.isna(value):
        return None
    value = int(value)
    for i, (low, high) in enumerate(CUSTOMER_BUCKETS):
        if value >= low and (high is None or value <= high):
            return i
    return None


def _bucket_label(bucket: int) -> str:
    low, high = CUSTOMER_BUCKETS[bucket]
    return f"{low}+" if high is None else f"{low}-{high}"


def _service_type(value):
    # None for a town with no service mix, which leaves it out of that dimension
    # rather than making a fourth category out of the gap.
    if pd.isna(value):
        return None
    return SERVICE_TYPES.get(str(value).strip())


def _add_city_markers(m, data):
    # Returns (always-on group, [(bucket, service, count, layer)]). The filter layers
    # are the towns the filter box can hide; everything else stays on unconditionally.
    points = data.points.to_crs(4326)
    if "HasBoundary" in points.columns:
        # Only the cities with no boundary of their own still need a dot
        points = points[~points["HasBoundary"]]
    if points.empty:
        return None, []
    has_zone = "Zone" in points.columns
    popup_html = _popup_renderer(data)
    marker = dict(popup_html=popup_html, radius=data.marker_radius, popup=data.popup)

    if data.group_point_layers:
        # One layer per Group, which hands the filtering to the layer control that
        # is already on the map rather than building a second box beside it. Nothing
        # is drawn under these, so they need neither the filter box nor a restack.
        for group_name in data.order:
            sub = points[points["Group"] == group_name]
            if sub.empty:
                continue
            layer = folium.FeatureGroup(name=f"{group_name} ({len(sub)})", show=True)
            for r in sub.itertuples(index=False):
                _add_marker(layer, r, has_zone, **marker)
            layer.add_to(m)
        return None, []

    blank = pd.Series(None, index=points.index, dtype="object")
    if data.city_filters and "Customers" in points.columns:
        buckets = points["Customers"].map(_customer_bucket)
    else:
        buckets = blank
    if data.city_filters and "ServiceMix" in points.columns:
        services = points["ServiceMix"].map(_service_type)
    else:
        services = blank
    # A town with no customer count has nothing to filter on, so it is never hidden
    filterable = buckets.notna()

    group = folium.FeatureGroup(name="Cities", show=True)
    for r in points[~filterable].itertuples(index=False):
        _add_marker(group, r, has_zone, **marker)
    group.add_to(m)

    # One layer per (bucket, service) pair actually present, so the box can AND the
    # two dimensions by adding and removing whole layers
    keyed = points[filterable].copy()
    keyed["_bucket"] = buckets[filterable]
    keyed["_service"] = services[filterable]

    filter_layers = []
    for (bucket, service), sub in keyed.groupby(["_bucket", "_service"], dropna=False):
        service = None if pd.isna(service) else service
        name = f"{_bucket_label(bucket)} customers"
        # control=False keeps these out of the layer list -- the filter box owns them
        layer = folium.FeatureGroup(
            name=f"{name} – {service}" if service else name, show=True, control=False
        )
        for r in sub.itertuples(index=False):
            _add_marker(layer, r, has_zone, **marker)
        layer.add_to(m)
        filter_layers.append((bucket, service, len(sub), layer))

    return group, filter_layers


def _legend_html(data) -> str:
    html = """
    <div style="position: fixed;
                bottom: 20px; left: 20px; width: 200px;
                background-color: white; border: 2px solid grey;
                border-radius: 5px; z-index:9999;
                padding: 15px; font-family: Arial;
                box-shadow: 0 0 15px rgba(0,0,0,0.2)">
    <h5 style="margin: 0 0 10px 0; color: #333;">Legend</h5>
    """
    for label in data.order:
        html += f"""
        <div style="margin: 5px 0;">
            <span style="display: inline-block; width: 15px; height: 15px;
                       background-color: {data.colors[label]}; border-radius: 50%;
                       margin-right: 8px; vertical-align: middle;"></span>
            <span style="font-size: 12px; vertical-align: middle;">{label}</span>
        </div>
        """
    if data.has_cities and data.marker_radius is None:
        html += """
        <hr style="margin: 10px 0; border: 0.5px solid #ccc;">
        <div style="font-size: 11px; color: #666;">
            <div>● Large = Major Cities</div>
            <div>● Medium = State Capitals</div>
            <div>● Small = Other Cities</div>
            <div style="margin-top: 8px; font-style: italic;">Click cities for details</div>
        </div>
        """
    elif data.has_cities:
        html += """
        <hr style="margin: 10px 0; border: 0.5px solid #ccc;">
        <div style="font-size: 11px; color: #666; font-style: italic;">
            Click a pin for details
        </div>
        """
    return html + "</div>"


class _KeepStackingOrder(MacroElement):
    # Leaflet appends a layer to the end of the draw order when it is added, so
    # re-checking a zone in the layer control lands it above the circles and takes
    # their hover with it. Re-asserting the order whenever an overlay comes back
    # keeps the map reading as areas -> regions -> cities however it is toggled.
    _template = Template("""
        {% macro script(this, kwargs) %}
            function {{ this.get_name() }}() {
                {%- for name in this.layer_names %}
                if ({{ this._parent.get_name() }}.hasLayer({{ name }})) {
                    {{ name }}.bringToFront();
                }
                {%- endfor %}
            }
            {{ this._parent.get_name() }}.on('overlayadd', {{ this.get_name() }});
            {{ this.get_name() }}();
        {% endmacro %}
        """)

    def __init__(self, layers):
        super().__init__()
        self._name = "restack"
        # Bottom to top: each call lifts that layer above everything before it
        self.layer_names = [layer.get_name() for layer in layers]


class _CityFilter(MacroElement):
    # Two checkbox sections wired straight at the feature groups: a town shows when
    # its customer bucket and its service type are both checked. The layer control
    # already lists a row per region and per circle, so these get their own box
    # rather than another dozen entries down a list nobody scrolls.
    _template = Template("""
        {% macro html(this, kwargs) %}
            <div id="{{ this.get_name() }}" style="position: fixed;
                        bottom: 20px; right: 20px; z-index: 9999;
                        background-color: white; border: 2px solid grey;
                        border-radius: 5px; padding: 10px 12px;
                        font-family: Arial; font-size: 12px;
                        box-shadow: 0 0 15px rgba(0,0,0,0.2)">
                <div style="font-weight: bold; color: #333; margin-bottom: 6px">
                    Customers
                </div>
                {%- for bucket, label, count in this.buckets %}
                <label style="display: block; color: #333; cursor: pointer;
                              white-space: nowrap; line-height: 1.7">
                    <input type="checkbox" checked
                           data-bucket="{{ bucket }}"
                           style="vertical-align: middle; margin-right: 6px">
                    {{ label }}
                    <span style="color: #888">({{ count }})</span>
                </label>
                {%- endfor %}
                {%- if this.services %}
                <div style="font-weight: bold; color: #333; margin: 8px 0 6px 0;
                            border-top: 1px solid #ddd; padding-top: 8px">
                    Service type
                </div>
                {%- for service, count in this.services %}
                <label style="display: block; color: #333; cursor: pointer;
                              white-space: nowrap; line-height: 1.7">
                    <input type="checkbox" checked
                           data-service="{{ service }}"
                           style="vertical-align: middle; margin-right: 6px">
                    {{ service }}
                    <span style="color: #888">({{ count }})</span>
                </label>
                {%- endfor %}
                {%- endif %}
                <div style="color: #888; margin-top: 6px; border-top: 1px solid #ddd;
                            padding-top: 5px; max-width: 150px; white-space: normal">
                    Filters city dots
                </div>
            </div>
        {% endmacro %}

        {% macro script(this, kwargs) %}
            var {{ this.get_name() }}_layers = [
                {%- for bucket, service, name in this.layers %}
                {bucket: "{{ bucket }}", service: "{{ service }}", layer: {{ name }}},
                {%- endfor %}
            ];
            var {{ this.get_name() }}_box =
                document.getElementById("{{ this.get_name() }}");

            function {{ this.get_name() }}_apply() {
                var checked = {};
                {{ this.get_name() }}_box
                    .querySelectorAll("input[data-bucket], input[data-service]")
                    .forEach(function (box) {
                        var key = box.dataset.bucket === undefined
                            ? box.dataset.service
                            : box.dataset.bucket;
                        checked[key] = box.checked;
                    });
                {{ this.get_name() }}_layers.forEach(function (entry) {
                    // An unlisted key -- a town with no service mix -- is never hidden
                    var show = checked[entry.bucket] !== false
                        && checked[entry.service] !== false;
                    if (show) {
                        {{ this._parent.get_name() }}.addLayer(entry.layer);
                    } else {
                        {{ this._parent.get_name() }}.removeLayer(entry.layer);
                    }
                });
            }

            {{ this.get_name() }}_box
                .querySelectorAll("input[type=checkbox]")
                .forEach(function (box) {
                    box.addEventListener("change", {{ this.get_name() }}_apply);
                });
        {% endmacro %}
        """)

    def __init__(self, filter_layers):
        super().__init__()
        self._name = "city_filter"
        counts_by_bucket, counts_by_service = {}, {}
        for bucket, service, count, _ in filter_layers:
            counts_by_bucket[bucket] = counts_by_bucket.get(bucket, 0) + count
            if service is not None:
                counts_by_service[service] = counts_by_service.get(service, 0) + count

        # Both sections read in their declared order, not whatever order the layers
        # happened to be built in, and only list what the export actually contains
        self.buckets = [
            (bucket, _bucket_label(bucket), counts_by_bucket[bucket])
            for bucket in range(len(CUSTOMER_BUCKETS))
            if bucket in counts_by_bucket
        ]
        self.services = [
            (service, counts_by_service[service])
            for service in dict.fromkeys(SERVICE_TYPES.values())
            if service in counts_by_service
        ]
        self.layers = [
            (bucket, service or "", layer.get_name()) for bucket, service, _, layer in filter_layers
        ]


def build_map(data, *, basemap: str = BASEMAP) -> folium.Map:
    # Layers are added coarsest first: whatever goes on last wins the hover.
    theme = basemap_theme(basemap)
    bminx, bminy, bmaxx, bmaxy = _bounds_4326(data)

    m = folium.Map(
        location=[(bminy + bmaxy) / 2.0, (bminx + bmaxx) / 2.0],
        # The same provider object the PNG draws, so the two can never disagree about
        # which basemap this map is on. folium reads its attribution off it.
        tiles=theme["base_src"],
        zoom_start=7,
        control_scale=True,
        prefer_canvas=True,
    )
    m.get_root().html.add_child(folium.Element(TITLE_TEMPLATE.format(title=data.title)))

    _add_region_polys(m, data)

    stack = []
    if data.circles is not None and not data.circles.empty:
        stack += _add_region_circles(m, data)

    # Cities are the finest level, so they sit above the circles and win the hover
    if data.draw_regions and data.detail_polys is not None and not data.detail_polys.empty:
        stack.append(_add_city_polys(m, data))

    filter_layers = []
    if data.has_cities:
        dots, filter_layers = _add_city_markers(m, data)
        if dots is not None:
            stack.append(dots)
        stack += [layer for *_, layer in filter_layers]

    m.get_root().html.add_child(folium.Element(_legend_html(data)))
    m.fit_bounds([[bminy, bminx], [bmaxy, bmaxx]])
    folium.LayerControl(collapsed=False, position="topright").add_to(m)
    Fullscreen().add_to(m)
    if stack:
        _KeepStackingOrder(stack).add_to(m)
    if filter_layers:
        _CityFilter(filter_layers).add_to(m)
    return m


def export_html(data, *, basemap: str = BASEMAP, output_dir=PROJECT_ROOT) -> str:
    m = build_map(data, basemap=basemap)
    out_path = str(Path(output_dir) / f"{data.output_stem}_map.html")
    m.save(out_path)
    print(f"Saved interactive HTML: {out_path}")
    return out_path


def export_geojson(data, *, output_dir=PROJECT_ROOT) -> str:
    out_path = str(Path(output_dir) / f"{data.output_stem}.geojson")
    # A source with no regions -- the task map is pins and nothing else -- exports
    # the points instead, so the file is still the map's own data
    layer = data.points if data.regions.empty and data.has_cities else data.regions
    layer.to_crs(4326).to_file(out_path, driver="GeoJSON")
    print(f"Saved GeoJSON: {out_path}")
    return out_path
