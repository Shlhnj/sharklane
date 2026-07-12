"""
Vectorize an input raster into polygons, filtering pixels by a threshold
condition before extraction. Useful for deriving land/water masks directly
from bathymetry, classified landcover, or any other raster source, instead
of requiring a pre-made vector coastline.

Examples
--------
Extract water polygons from a bathymetry raster where elevation < 0:

    >>> water_gdf = vectorize_raster("bathymetry.tif", threshold=0,
    ...                               comparison="<", band=1)

Extract "land" class (value == 1) from a classified landcover raster:

    >>> land_gdf = vectorize_raster("landcover.tif", threshold=1,
    ...                              comparison="==")

Go straight from a raster to a WaterMask (skips the vector round-trip):

    >>> wm = water_mask_from_raster("bathymetry.tif", threshold=0,
    ...                               comparison="<", target_crs="EPSG:32750")
"""
from __future__ import annotations

import operator
import numpy as np
import geopandas as gpd
from shapely.geometry import shape

from .masking import WaterMask

_OPS = {
    ">=": operator.ge, ">": operator.gt,
    "<=": operator.le, "<": operator.lt,
    "==": operator.eq, "!=": operator.ne,
}


def _apply_threshold(arr: np.ndarray, threshold: float | None,
                      comparison: str, nodata=None) -> np.ndarray:
    if threshold is None:
        binary = np.ones(arr.shape, dtype="uint8")
    else:
        if comparison not in _OPS:
            raise ValueError(f"comparison must be one of {list(_OPS)}")
        binary = _OPS[comparison](arr, threshold).astype("uint8")
    if nodata is not None:
        binary[arr == nodata] = 0
    return binary


def vectorize_raster(path: str = None,
                      array: np.ndarray = None,
                      transform=None,
                      crs=None,
                      band: int = 1,
                      threshold: float = None,
                      comparison: str = ">=",
                      min_area: float = None,
                      simplify_tolerance: float = None) -> gpd.GeoDataFrame:
    """
    Extract polygons from raster pixels that satisfy `value {comparison} threshold`.

    Parameters
    ----------
    path : path to a raster file readable by rasterio (.tif, etc). Provide
        this OR (array + transform + crs).
    array : 2D numpy array of raster values, if not reading from a file.
    transform : affine transform for `array` (required if array is given).
    crs : CRS for `array` (required if array is given).
    band : band index to read, if reading from `path` (1-indexed).
    threshold : the value to compare each pixel against. If None, all
        non-nodata pixels are extracted as a single-class mask.
    comparison : one of '>=', '>', '<=', '<', '==', '!=' -- the filter
        condition applied as `pixel_value {comparison} threshold`.
    min_area : optional minimum polygon area (in CRS units^2) to keep --
        use this to drop speckle/noise polygons from raster classification.
    simplify_tolerance : optional simplification tolerance (in CRS units)
        applied to output polygons, to reduce vertex count from raster
        stair-stepping.

    Returns
    -------
    GeoDataFrame of polygons where the threshold condition was met, merged
    from contiguous qualifying pixels.
    """
    from rasterio import features

    nodata = None
    if path is not None:
        import rasterio
        with rasterio.open(path) as src:
            arr = src.read(band)
            transform = src.transform
            crs = src.crs
            nodata = src.nodata
    else:
        if array is None or transform is None:
            raise ValueError("Provide either `path`, or both `array` and `transform`.")
        arr = array

    binary = _apply_threshold(arr, threshold, comparison, nodata=nodata)

    shapes_gen = features.shapes(binary, mask=binary.astype(bool), transform=transform)
    geoms = [shape(geom) for geom, val in shapes_gen if val == 1]

    if not geoms:
        return gpd.GeoDataFrame({"value": []}, geometry=[], crs=crs)

    gdf = gpd.GeoDataFrame({"value": [1] * len(geoms)}, geometry=geoms, crs=crs)

    if min_area is not None:
        gdf = gdf[gdf.geometry.area >= min_area].reset_index(drop=True)
    if simplify_tolerance is not None:
        gdf["geometry"] = gdf.geometry.simplify(simplify_tolerance)

    return gdf


def water_mask_from_raster(path: str = None,
                            array: np.ndarray = None,
                            transform=None,
                            crs=None,
                            band: int = 1,
                            threshold: float = 0.0,
                            comparison: str = "<",
                            target_crs: str = None,
                            target_resolution: float = None) -> WaterMask:
    """
    Build a WaterMask directly from a raster (e.g. bathymetry: water where
    elevation < 0), without round-tripping through vector polygons.

    If target_crs differs from the raster's native CRS (e.g. you're working
    in a projected CRS but the raster is in WGS84), the raster is reprojected
    first via rasterio.warp -- provide target_resolution (in target_crs
    units, e.g. metres) in that case.

    Parameters mirror vectorize_raster(); comparison defaults to '<' since
    the common case is bathymetry where negative/below-threshold values are
    water.
    """
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling

    if path is not None:
        with rasterio.open(path) as src:
            arr = src.read(band)
            src_transform = src.transform
            src_crs = src.crs
            nodata = src.nodata

            if target_crs is not None and str(target_crs) != str(src_crs):
                dst_transform, width, height = calculate_default_transform(
                    src_crs, target_crs, src.width, src.height, *src.bounds,
                    resolution=target_resolution,
                )
                dst_arr = np.empty((height, width), dtype=arr.dtype)
                reproject(
                    source=arr, destination=dst_arr,
                    src_transform=src_transform, src_crs=src_crs,
                    dst_transform=dst_transform, dst_crs=target_crs,
                    resampling=Resampling.bilinear,
                )
                arr, transform, crs = dst_arr, dst_transform, target_crs
            else:
                transform, crs = src_transform, src_crs
    else:
        if array is None or transform is None:
            raise ValueError("Provide either `path`, or both `array` and `transform`.")
        arr = array
        nodata = None

    binary = _apply_threshold(arr, threshold, comparison, nodata=nodata)
    minx, maxy = transform.c, transform.f
    maxx = minx + transform.a * arr.shape[1]
    miny = maxy + transform.e * arr.shape[0]
    bounds = (min(minx, maxx), min(miny, maxy), max(minx, maxx), max(miny, maxy))

    return WaterMask(binary, transform, bounds, crs)
