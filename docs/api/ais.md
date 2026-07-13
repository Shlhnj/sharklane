# `sharklane.ais` — API Reference

AIS loading, speed computation, and track cleaning.

## `load_ais_csv(path, crs="EPSG:4326", **read_csv_kwargs)`
Load raw AIS positions from CSV. Requires columns: `vessel_id`,
`timestamp`, `lon`, `lat`.

## `compute_speeds(gdf, crs_metric)`
Estimate instantaneous speed (m/s) between consecutive positions per
vessel, using a projected CRS for accurate distances. Adds `speed_mps`,
`seg_dist_m`, `seg_dt_s` columns.

Uses `.diff().dt.total_seconds()` for elapsed time — deliberately not
`timestamp.astype("int64") / 1e9`, which assumes nanosecond-resolution
`datetime64`. Newer pandas versions can store `datetime64` at coarser
resolution depending on construction, which silently made that
conversion wrong by a fixed factor in an earlier version of this code.

## `clean_tracks(gdf, crs_metric, max_speed_mps=50.0, speed_percentile=99.0, stationary_thresh_mps=0.01, stationary_frac_limit=0.25, min_points=20)`
Filter AIS positions following Womersley et al. 2024's approach:
- drop positions faster than `max_speed_mps` or above `speed_percentile`
  (erroneous signals)
- drop entire vessels where more than `stationary_frac_limit` of positions
  are below `stationary_thresh_mps` (anchored/loitering vessels)
- drop vessels with fewer than `min_points` remaining positions

## `to_tracks(gdf)`
Split a cleaned point GeoDataFrame into per-vessel ordered tracks (a
`dict[vessel_id, GeoDataFrame]`) — this is the format `sim.tracks` expects.

## `track_to_linestring(track)`
Convert a single vessel's track GeoDataFrame into a shapely `LineString`.
