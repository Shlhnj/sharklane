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
