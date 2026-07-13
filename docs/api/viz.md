# `sharklane.viz` — API Reference

## `viz.static`
Static matplotlib plots: `plot_site_map()`, `plot_speed_reduction_curve()`,
`plot_reroute_paths()`, `plot_lane_shift_feasibility()`. Normally called
via the matching `Simulator.plot_*()` methods.

`plot_reroute_paths()` handles the case where zero transit vessels were
found (an empty `reroute_df`, which has no columns at all, not just no
rows) by showing a "no rerouted vessels" note instead of crashing.

## `viz.animate`
`animate_vessel_comparison()` — replays a single real/cleaned AIS track,
original vs. computed reroute. Called via `Simulator.animate_vessel()`.

## `viz.ship`
Ship icon geometry and track-generation math (no rendering).

- `ship_polygon(x, y, heading_rad, length, width)` — 5-point outline
  (rectangle stern + triangular bow).
- `get_valid_sides(corridor_line)` / `get_default_side(corridor_line)` —
  orientation-aware direction labels (`west`/`east` for an east-west lane,
  `south`/`north` for north-south). Also exposed as
  `Simulator.get_lane_side_options()`.
- `compute_reroute_options(corridor_line, polygon, side=None, n_arc_samples=300)`
  — computes both go-around paths around the polygon's **convex hull**.
  See [reroute algorithm guide](../guides/reroute_algorithm.md).
- `build_baseline_track()`, `build_speed_reduction_track()`,
  `build_reroute_track()` — per-scenario spatial tracks.
- `build_scenario_track_raw()` / `resample_common_timeline()` — puts
  multiple scenarios on a shared real time axis, so a comparison
  animation shows faster scenarios finishing while slower ones keep moving.

## `viz.ship_animate`
Rendering: `animate_transit()` (single scenario) and
`animate_transit_comparison()` (all three, optionally with a live
elapsed-time bar chart via `show_time_chart=True`). Normally called via
`Simulator.animate_transit()` / `Simulator.animate_transit_comparison()`.

Supports `background="light"/"dark"/"raster"/"satellite"`. `"satellite"`
needs `contextily` and live internet access to a tile server — wrapped in
a try/except that falls back to `"light"` with a warning if unavailable.
