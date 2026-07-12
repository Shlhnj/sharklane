"""
Global shipping lanes (Major / Middle / Minor), bundled from:

  newzealandpaul/Shipping-Lanes (CC BY 4.0, excluding Statista)
  https://github.com/newzealandpaul/Shipping-Lanes
  Georeferenced from the CIA's "Map of The World's Oceans" (Oct 2012).
  DOI: 10.5281/zenodo.6361763

This is a hand-digitized, coarse global dataset -- 3 features total (one
MultiLineString per Type: Major, Middle, Minor), useful for identifying
which broad shipping corridor passes near your site and for seeding a
representative "lane" geometry, but NOT a substitute for real AIS-derived
lane geometry at the resolution needed for the reroute/speed-reduction
simulations. Use it to orient yourself and to sanity-check where your
AOI sits relative to global traffic, or as a fallback lane input when you
don't have local AIS to derive one from directly.
"""
from __future__ import annotations

import os
import geopandas as gpd
from shapely.geometry import box

_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "world_shipping_lanes.geojson")


def load_world_shipping_lanes(bbox: tuple[float, float, float, float] = None,
                               lane_type: str | list[str] = None,
                               crs: str = None) -> gpd.GeoDataFrame:
    """
    Load the bundled world shipping lanes dataset.

    Parameters
    ----------
    bbox : optional (minx, miny, maxx, maxy) in WGS84 to clip to -- e.g.
        a padded version of your core habitat bounds.
    lane_type : optional 'Major', 'Middle', 'Minor', or a list of these,
        to filter to specific lane classes.
    crs : optional CRS to reproject the result into (e.g. your working
        projected CRS).

    Returns
    -------
    GeoDataFrame with columns [Type, geometry] (MultiLineString per type,
    clipped to bbox if given).
    """
    gdf = gpd.read_file(_DATA_PATH)

    if lane_type is not None:
        types = [lane_type] if isinstance(lane_type, str) else list(lane_type)
        gdf = gdf[gdf["Type"].isin(types)]

    if bbox is not None:
        clip_box = box(*bbox)
        gdf = gdf.clip(clip_box)
        gdf = gdf[~gdf.geometry.is_empty]

    if crs is not None:
        gdf = gdf.to_crs(crs)

    return gdf.reset_index(drop=True)


def explode_to_segments(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Break MultiLineString features into individual LineString segments
    (one row each), retaining the Type column -- easier to work with when
    picking a single representative lane geometry near your AOI."""
    exploded = gdf.explode(index_parts=False).reset_index(drop=True)
    return exploded


def nearest_lane_to_point(gdf: gpd.GeoDataFrame, x: float, y: float,
                           lane_type: str = None):
    """Return the single lane segment (LineString) closest to a given
    point -- e.g. your core habitat's centroid -- optionally restricted to
    one lane Type. Useful for picking a real-world-informed representative
    lane instead of hand-deriving one."""
    from shapely.geometry import Point

    segs = explode_to_segments(gdf)
    if lane_type is not None:
        segs = segs[segs["Type"] == lane_type]
    if len(segs) == 0:
        raise ValueError("No lane segments available for the given filter.")

    pt = Point(x, y)
    segs = segs.copy()
    segs["_dist"] = segs.geometry.distance(pt)
    nearest = segs.sort_values("_dist").iloc[0]
    return nearest.geometry


def trim_lane_to_polygon(lane_line, polygon, pad_fraction: float = 0.25):
    """
    Trim a (possibly very long) lane down to just the portion relevant to
    a given polygon: the segment that actually lies inside the polygon,
    extended by `pad_fraction` of that inside-length on EACH end.

    This matters because a lane pulled from the global shipping-lanes
    dataset (or any real-world lane source) can run for hundreds of km --
    using it untrimmed as a corridor line means simulate_redirection() /
    the ship animations end up modeling a mostly-irrelevant, enormously
    long approach/departure that has nothing to do with the actual habitat.

    Parameters
    ----------
    lane_line : shapely LineString, in the same (projected/metric) CRS as
        `polygon` -- reproject both into a metric CRS before calling this,
        since pad_fraction is computed from real length.
    polygon : shapely Polygon (or MultiPolygon) -- the risk/habitat area
        to trim the lane around.
    pad_fraction : fraction of the in-polygon lane length to extend on
        EACH end beyond the polygon boundary. E.g. 0.25 (default) means:
        if the lane crosses 40 km of the polygon, the trimmed lane extends
        an extra 10 km (25% of 40 km) past the polygon on both the entry
        and exit side, for a total trimmed length of 40 + 10 + 10 = 60 km.
        Use 0 for no extension at all (lane exactly matches the in-polygon
        segment, with no visible approach/departure outside it).

    Returns
    -------
    trimmed_line : shapely LineString
    info : dict with 'inside_length_m', 'pad_length_m', 'total_length_m'
    """
    from shapely.geometry import LineString

    inter = lane_line.intersection(polygon)
    if inter.is_empty:
        raise ValueError(
            "The lane does not intersect the polygon at all -- nothing to "
            "trim around. Check that the lane actually passes through the "
            "habitat, or use the full untrimmed lane instead "
            "(trim_to_polygon=False)."
        )

    if inter.geom_type == "MultiLineString":
        # multiple crossings -- use the longest piece as the representative
        # "inside" segment (the others are likely minor clips near the edge)
        inter = max(inter.geoms, key=lambda g: g.length)
    elif inter.geom_type == "Point":
        raise ValueError(
            "The lane only touches the polygon at a single point (tangent), "
            "not a real crossing -- cannot compute a meaningful inside "
            "length to trim around."
        )
    elif inter.geom_type != "LineString":
        raise ValueError(f"Unexpected lane/polygon intersection type: {inter.geom_type}")

    inside_length = inter.length
    pad_length = pad_fraction * inside_length

    x0, y0 = inter.coords[0]
    x1, y1 = inter.coords[-1]
    dx, dy = x1 - x0, y1 - y0
    seg_len = (dx ** 2 + dy ** 2) ** 0.5
    if seg_len == 0:
        raise ValueError("Degenerate lane/polygon intersection (zero length) -- "
                          "cannot determine a direction to extend along.")
    ux, uy = dx / seg_len, dy / seg_len

    new_x0, new_y0 = x0 - ux * pad_length, y0 - uy * pad_length
    new_x1, new_y1 = x1 + ux * pad_length, y1 + uy * pad_length

    trimmed = LineString([(new_x0, new_y0), (new_x1, new_y1)])
    info = {
        "inside_length_m": inside_length,
        "pad_length_m": pad_length,
        "total_length_m": trimmed.length,
    }
    return trimmed, info
