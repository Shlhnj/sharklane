# `sharklane.simulate` — API Reference

The three mitigation simulation modules. Normally accessed via `Simulator`
methods rather than directly, but usable standalone.

## `simulate.speed`

### `simulate_speed_reduction(tracks, polygon, reductions=None)`
Applies a percentage speed reduction only to track segments intersecting
the risk polygon. `reductions=None` defaults to 10–75% at 1% increments.
Raises `ValueError` if `tracks` is empty.

### `summarize(results)`
Mean/median percent increase and extra hours per vessel, by reduction
level. Raises `ValueError` if `results` is empty (e.g. every vessel was
skipped for having too few positions).

## `simulate.reroute`

### `classify_transit_vs_terminal(tracks, polygon)`
Classifies each vessel as `"transit"` (passes through, reroutable),
`"terminal"` (origin/destination inside the polygon, not reroutable),
`"inside_only"`, or `"no_overlap"`.

### `reroute_perimeter(tracks, polygon)`
Open-water reroute: routes around the polygon's **convex hull** (not its
raw boundary — see [reroute algorithm guide](../guides/reroute_algorithm.md)).
No water mask needed.

### `reroute_least_cost(tracks, polygon, water_mask)`
Land-mask-constrained least-cost path (via `skimage.graph.MCP_Geometric`).
Needed for bay-mouth/coastline-constrained sites where a straight hull
detour might cross land.

### `estimate_reroute_time(reroute_df, tracks, polygon)`
Converts reroute extra distance into extra time using each vessel's
average speed just before/after the detour.

## `simulate.laneshift`

### `identify_lane_vessels(tracks, speed_threshold_mps=7.7)`
Vessels whose mean speed exceeds the threshold — treated as using a fixed
transit lane rather than a discretionary route.

### `representative_lane(tracks, lane_vessel_ids, polygon)`
Builds a single representative lane line from identified lane vessels.

### `test_lane_shifts(lane, polygon, water_mask, offsets_m=None, direction=(0,-1))`
Tests increasing lateral offsets, checking both that the shifted lane
clears the polygon AND stays entirely in water.

### `minimum_feasible_shift(shift_results)`
Smallest offset that satisfies both conditions. Returns `None` if no
tested offset works — a valid finding for a genuinely narrow strait, not
an error.
