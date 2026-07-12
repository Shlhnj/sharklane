"""
Input helpers: load the core habitat polygon, load a coastline polygon for
masking, and either load or interactively draw the transit line that vessels
must cross to enter/exit the habitat (e.g. a bay mouth corridor).
"""
from __future__ import annotations

import geopandas as gpd
from shapely.geometry import LineString, shape
from shapely.ops import unary_union


def load_polygon(path: str, crs: str | None = None) -> gpd.GeoDataFrame:
    """Load a habitat/corridor polygon from any GeoPandas-readable file
    (.shp, .geojson, .gpkg, ...). Dissolves multi-feature files into one
    geometry unless you want them kept separate -- pass keep_separate=True
    at the Simulator level if so."""
    gdf = gpd.read_file(path)
    if crs is not None:
        gdf = gdf.to_crs(crs)
    return gdf


def load_line(path: str, crs: str | None = None) -> gpd.GeoDataFrame:
    """Load a pre-drawn transit line (e.g. digitized bay-mouth crossing)."""
    gdf = gpd.read_file(path)
    if crs is not None:
        gdf = gdf.to_crs(crs)
    return gdf


def draw_line_interactive(background_gdf: gpd.GeoDataFrame | None = None,
                           n_points: int = -1) -> LineString:
    """
    Pop up a matplotlib window, click points to draw the transit/corridor
    line (e.g. across a bay mouth), press Enter or right-click to finish.

    Parameters
    ----------
    background_gdf : optional GeoDataFrame to plot underneath for reference
        (e.g. the core habitat polygon or a coastline layer).
    n_points : number of points to collect; -1 lets the user click freely
        and finish with Enter/right-click.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    if background_gdf is not None:
        background_gdf.plot(ax=ax, facecolor="lightblue", edgecolor="k", alpha=0.5)
    ax.set_title("Click points to draw the transit line.\n"
                  "Right-click or Enter to finish.")
    plt.tight_layout()

    pts = plt.ginput(n_points, timeout=0)
    plt.close(fig)

    if len(pts) < 2:
        raise ValueError("Need at least 2 points to define a line.")
    return LineString(pts)


def load_coastline(path: str, crs: str | None = None) -> gpd.GeoDataFrame:
    """Load a land polygon layer (e.g. GSHHG, OSM coastline extract) used
    to build the water/land eligibility mask."""
    gdf = gpd.read_file(path)
    if crs is not None:
        gdf = gdf.to_crs(crs)
    return gdf


def dissolve(gdf: gpd.GeoDataFrame):
    """Convenience: merge all geometries in a GeoDataFrame into one."""
    return unary_union(gdf.geometry.values)
