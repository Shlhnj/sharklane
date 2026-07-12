"""
Build a binary water/land eligibility raster from a land polygon layer.
Water = 1 (traversable), Land = 0 (blocked). Used as the cost surface for
least-cost rerouting so simulated paths never cross land.
"""
from __future__ import annotations

import numpy as np
import geopandas as gpd
from rasterio import features
from rasterio.transform import from_bounds


class WaterMask:
    def __init__(self, mask: np.ndarray, transform, bounds, crs):
        self.mask = mask          # 2D array, 1=water, 0=land
        self.transform = transform
        self.bounds = bounds      # (minx, miny, maxx, maxy)
        self.crs = crs

    @property
    def shape(self):
        return self.mask.shape

    def rowcol(self, x: float, y: float):
        """Convert projected coords to (row, col) raster indices."""
        from rasterio.transform import rowcol
        r, c = rowcol(self.transform, x, y)
        r = int(np.clip(r, 0, self.mask.shape[0] - 1))
        c = int(np.clip(c, 0, self.mask.shape[1] - 1))
        return r, c

    def xy(self, row: int, col: int):
        """Convert raster indices back to projected coords (cell center)."""
        from rasterio.transform import xy
        x, y = xy(self.transform, row, col)
        return x, y

    def is_water(self, x: float, y: float) -> bool:
        r, c = self.rowcol(x, y)
        return bool(self.mask[r, c])

    def nearest_water_cell(self, x: float, y: float, max_search_px: int = 200):
        """If (x, y) lands on a 'land' pixel (e.g. a coastal AIS jitter
        point), search outward in a growing box for the nearest water
        pixel and return its row/col."""
        r0, c0 = self.rowcol(x, y)
        if self.mask[r0, c0]:
            return r0, c0
        for radius in range(1, max_search_px):
            rmin, rmax = max(0, r0 - radius), min(self.mask.shape[0], r0 + radius + 1)
            cmin, cmax = max(0, c0 - radius), min(self.mask.shape[1], c0 + radius + 1)
            window = self.mask[rmin:rmax, cmin:cmax]
            if window.any():
                rr, cc = np.argwhere(window == 1)[0]
                return rmin + rr, cmin + cc
        raise ValueError("No water pixel found near point within search radius.")


def build_water_mask(land_gdf: gpd.GeoDataFrame,
                      bounds: tuple[float, float, float, float],
                      resolution: float,
                      crs=None) -> WaterMask:
    """
    Rasterize land polygons into a binary water/land grid.

    Parameters
    ----------
    land_gdf : GeoDataFrame of land polygons (e.g. coastline/landmass layer).
    bounds : (minx, miny, maxx, maxy) in the working CRS -- usually pad this
        beyond your habitat/corridor extent so reroutes have room to work with.
    resolution : cell size in the same units as the CRS (e.g. metres if
        projected; degrees if geographic -- projecting first is recommended
        for anything bay-mouth-scale so distances/paths behave sensibly).
    crs : CRS to reproject land_gdf into before rasterizing, if not already
        in a projected CRS.
    """
    if crs is not None:
        land_gdf = land_gdf.to_crs(crs)

    minx, miny, maxx, maxy = bounds
    width = max(1, int((maxx - minx) / resolution))
    height = max(1, int((maxy - miny) / resolution))
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    if len(land_gdf) > 0:
        land_raster = features.rasterize(
            [(geom, 1) for geom in land_gdf.geometry if geom is not None],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype="uint8",
        )
    else:
        land_raster = np.zeros((height, width), dtype="uint8")

    water_raster = 1 - land_raster  # invert: water=1, land=0
    return WaterMask(water_raster, transform, bounds, crs or land_gdf.crs)
