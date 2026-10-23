# Map data sources. Every loader returns a MapData for the shared renderers.

from dataclasses import dataclass, field

import geopandas as gpd


@dataclass
class PointPopup:
    # A popup described by the source rather than hard-coded to city semantics.
    # title_attr names the row attribute the heading reads; fields are the rows
    # under it, in order, and a value the export left blank costs its own row.
    title_attr: str
    fields: list[tuple[str, str]]
    subtitle_attr: str | None = None
    show_coordinates: bool = True


@dataclass
class MapData:
    # One row per region, columns: Label, geometry, Color. Always EPSG:3857.
    regions: gpd.GeoDataFrame
    colors: dict[str, str]
    order: list[str]
    title: str
    output_stem: str

    # City-level detail, only produced by the Excel source
    points: gpd.GeoDataFrame | None = None
    detail_polys: gpd.GeoDataFrame | None = None
    circles: gpd.GeoDataFrame | None = None

    # Renderer hints. A fixed zoom is a resolution held constant while the extent
    # varies, so the default lets the tile budget size the basemap to the data.
    basemap_zoom: int | str = "auto"
    # The Excel map's regions are dissolved group blobs it never draws, only exports
    draw_regions: bool = True
    label_regions: bool = False
    draw_city_dots: bool = False
    # Whether the interactive export gets the customer / service filter box
    city_filters: bool = False
    label_cities: bool = True
    label_circles: bool = False
    # What a point click opens. None keeps the city popup the spreadsheet map uses.
    popup: PointPopup | None = None
    # One marker layer per Group, so the layer control filters them. The city maps
    # put every dot in one layer and filter with their own box instead.
    group_point_layers: bool = False
    # Uniform point sizing, for sources whose points are not places and so have no
    # importance to be sized by. None keeps the per-city classification sizes.
    point_size: float | None = None
    marker_radius: float | None = None
    legend_title: str = "Groups"
    fill_alpha: float = 0.32
    edge_alpha: float = 0.75
    edge_width: float = 0.7
    extras: dict = field(default_factory=dict)

    @property
    def has_cities(self) -> bool:
        return self.points is not None and not self.points.empty
