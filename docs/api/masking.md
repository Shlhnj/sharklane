# `sharklane.masking` / `sharklane.raster_vectorize` — API Reference

## `masking.WaterMask`
Holds a binary water/land raster (1=water, 0=land), its affine transform,
bounds, and CRS. Methods: `.is_water(x, y)`, `.nearest_water_cell(x, y)`,
`.rowcol(x, y)`, `.xy(row, col)`.

**`bounds` must cover at least as much area as any zoom/pan you'll display
later** — `imshow` only draws within the mask's own extent; anything
outside stays blank white. See
[Troubleshooting](../guides/troubleshooting.md).

## `masking.build_water_mask(land_gdf, bounds, resolution, crs=None)`
Rasterize land polygons into a binary water/land grid. Returns a `WaterMask`.

## `raster_vectorize.vectorize_raster(path=None, array=None, transform=None, crs=None, band=1, threshold=None, comparison=">=", min_area=None, simplify_tolerance=None)`
Extract polygons from raster pixels satisfying `value {comparison}
threshold` (e.g. `<`, `>=`, `==`). `min_area`/`simplify_tolerance` clean
up speckle and stair-stepping from raster-derived polygons.

**On multi-patch results:** if your source raster produces several
disjoint polygons (e.g. scattered "high value" pixel clusters), merging
them via `union_all()` gives a `MultiPolygon`, which the reroute/lane-shift
geometry functions can't use directly. See
[Troubleshooting](../guides/troubleshooting.md) for merge strategies
(convex hull, largest patch, buffer-merge).

## `raster_vectorize.water_mask_from_raster(path=None, array=None, transform=None, crs=None, band=1, threshold=0.0, comparison="<", target_crs=None, target_resolution=None)`
Build a `WaterMask` directly from a raster (e.g. bathymetry), skipping the
vector round-trip. Reprojects via `rasterio.warp` if `target_crs` differs
from the raster's native CRS.
