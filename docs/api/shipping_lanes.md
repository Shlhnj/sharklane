# `sharklane.shipping_lanes` — API Reference

The bundled global shipping lanes dataset (Major/Middle/Minor), hand-
digitized from a CIA nautical chart via
[newzealandpaul/Shipping-Lanes](https://github.com/newzealandpaul/Shipping-Lanes)
(CC BY 4.0, DOI: 10.5281/zenodo.6361763). Only 3 features total worldwide
(one MultiLineString per Type) — useful for orientation, not a substitute
for real AIS.

## `load_world_shipping_lanes(bbox=None, lane_type=None, crs=None)`
Load the bundled dataset, optionally clipped to a WGS84 `bbox` and/or
filtered to one or more `lane_type` values.

## `explode_to_segments(gdf)`
Break MultiLineString features into individual LineString segments.

## `nearest_lane_to_point(gdf, x, y, lane_type=None)`
Return the single lane segment closest to a given point.

## `trim_lane_to_polygon(lane_line, polygon, pad_fraction=0.25)`
Trim a (possibly very long) lane down to just the portion inside a given
polygon, extended by `pad_fraction` of that inside-length on each end.
Raises `ValueError` if the lane doesn't intersect the polygon at all —
there's nothing to trim around in that case (use the untrimmed lane, or
a different lane/lane_type).

Returns `(trimmed_line, info)` where `info` has `inside_length_m`,
`pad_length_m`, `total_length_m`.

**Prefer calling this via `Simulator.load_world_shipping_lane(trim_to_polygon=True, ...)`**
rather than calling `trim_lane_to_polygon()` manually afterward — the
Simulator method already does this in one step, including auto-detecting
`lane_type`.
