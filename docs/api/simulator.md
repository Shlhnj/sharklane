# `Simulator` — API Reference

The main orchestrator class. Import via `from sharklane import Simulator`.

## Constructor

```python
Simulator(working_crs: str = "auto")
```

`working_crs`: a projected CRS (metres) appropriate for your site, or
`"auto"` (default) to determine it automatically from your habitat's own
centroid the first time you call `load_core_habitat()`. All internal
distance/area/speed math is done in this CRS — an unprojected CRS (plain
lat/lon) would silently make those numbers wrong, since 1° of longitude is
a different real distance depending on latitude. Pass an explicit EPSG
code (e.g. `"EPSG:32750"`) to skip auto-detection.

Calling any other method before `load_core_habitat()` (while `working_crs`
is still `"auto"`) raises a clear `RuntimeError` rather than a confusing
CRS-parsing error.

---

## Loading data

### `load_core_habitat(path, source_crs=None)`
Load the core habitat polygon (the whale shark aggregation site) from any
GeoPandas-readable file. Multiple features are merged via `union_all()`.
If `working_crs="auto"`, this call resolves it.

### `load_transit_line(path, source_crs=None)`
Load a pre-drawn transit/corridor line (e.g. a digitized shipping lane or
bay-mouth crossing).

### `draw_transit_line()`
Interactively click points to draw a transit line over the loaded habitat
(requires an interactive matplotlib backend).

### `load_world_shipping_lane(lane_type="auto", pad_deg=1.0, use_nearest=True, trim_to_polygon=True, trim_pad_fraction=0.25)`
Load a lane from the bundled global shipping lanes dataset (Major/Middle/
Minor). With `lane_type="auto"` (default), all three types are checked
within `pad_deg` of the habitat, and whichever type's nearest segment
**actually crosses the habitat polygon** is used (ties broken Major >
Middle > Minor). If none cross, falls back to whichever is closest
overall — check `sim.last_lane_crosses_habitat` afterward, since a
non-crossing lane can't be trimmed (see below) or used for rerouting.

`trim_to_polygon=True` (default) trims the found lane down to just the
portion inside the habitat, extended by `trim_pad_fraction` (25% default)
of that inside-length on each end — global-dataset lanes can run for
hundreds of km, almost all irrelevant to one specific site.

**Important:** if the lane doesn't cross the habitat, trimming raises a
`ValueError` (there's nothing to trim around). Wrap this call in a
try/except and check `sim.last_lane_crosses_habitat` — see
[Troubleshooting](../guides/troubleshooting.md).

Sets: `sim.last_lane_type_used`, `sim.last_lane_crosses_habitat`,
`sim.last_lane_trim_info`.

### `load_ais(path, **clean_kwargs)`
Load AIS vessel tracks from a CSV (`vessel_id, timestamp, lon, lat`
columns). `**clean_kwargs` are passed to `sharklane.ais.clean_tracks()`
(e.g. `min_points`, `stationary_frac_limit`, `max_speed_mps`).

**Every simulation method requires `sim.tracks` to be non-empty.** Forgetting
this step is the most common early mistake — see
[Troubleshooting](../guides/troubleshooting.md).

### `build_mask(land_path, bounds=None, resolution=100.0, source_crs=None)`
Build a water/land eligibility raster from a land polygon layer (e.g. a
coastline). Needed for `simulate_redirection(method="least_cost")`,
`simulate_lane_shift()`, and for showing land in animations.

**`bounds` must cover at least as much area as you'll ever zoom/pan to
later** — the mask only draws within its own bounds; anything outside
stays blank. See [Troubleshooting](../guides/troubleshooting.md).

### `build_mask_from_raster(raster_path, threshold=0.0, comparison="<", band=1, target_resolution=None)`
Build the mask directly from a raster (e.g. bathymetry: water where
elevation < 0) instead of a vector coastline.

### `vectorize_raster(raster_path, threshold=None, comparison=">=", band=1, min_area=None, simplify_tolerance=None, source_crs=None)`
Extract polygons from a raster by threshold filter (e.g. pull "high value"
pixels out of a density raster). Returns a GeoDataFrame in `working_crs`.

---

## Map navigation

### `zoom_bounds_latlon(min_lon, min_lat, max_lon, max_lat)`
Translate an ordinary WGS84 lon/lat box into `working_crs` bounds, ready
to pass as `bounds=` to any plot/animate method. This is the only
navigation helper — call it again with different coordinates to pan or
zoom further; there's no separate pan/zoom-by-factor helper by design
(lon/lat bounds alone cover both cases).

```python
bounds = sim.zoom_bounds_latlon(117.4, -8.1, 117.9, -7.7)
sim.animate_transit_comparison(..., bounds=bounds)
```

---

## Running simulations

### `simulate_speed_reduction(reductions=None, target_polygon=None)`
Speed reduction simulation. `reductions=None` defaults to 10–75% at 1%
increments (matching Womersley et al. 2024). Returns a summary DataFrame
via `sim.results["speed_reduction"]["summary"]`. Raises a clear
`ValueError` if `sim.tracks` is empty.

### `simulate_redirection(method="least_cost", target_polygon=None)`
Rerouting simulation. `method="least_cost"` uses a land-mask-constrained
least-cost path (needs `build_mask()` first); `method="perimeter"` uses a
simpler open-water convex-hull walk (no mask needed). Both route around
the polygon's **convex hull**, not its raw boundary — see
[How the reroute algorithm works](../guides/reroute_algorithm.md).

Classifies vessels into `transit` (reroutable) vs `terminal` (destination
is inside the habitat, can't be routed around) — check
`sim.results["redirection"]["labels"]`.

### `simulate_lane_shift(speed_threshold_mps=7.7, direction=(0,-1), offsets_m=None)`
Tests whether shifting the shipping lane sideways by increasing offsets
clears the habitat while staying in water. Can legitimately return
`min_feasible_offset_m=None` if no shift works (e.g. a narrow strait) —
that's a valid finding, not an error.

### `list_reroute_options(side=None, target_polygon=None, base_speed_knots=12.0)`
Inspect **both** possible go-around routes (not just the shorter one)
before picking one — useful when the habitat splits the approach into two
similar-length options. Returns length, estimated time, and compass side
for each, plus a `similar_length` flag.

### `get_lane_side_options()`
Check which `side` values are valid for the current lane (`west`/`east`
for an east-west lane, `south`/`north` for north-south) and which is the
default — call this before passing `side=` explicitly anywhere.

---

## Plotting

### `plot_site_map(target_polygon=None, **kwargs)`
### `plot_speed_reduction_curve(**kwargs)`
### `plot_reroute_paths(target_polygon=None, show_original_tracks=True, **kwargs)`
### `plot_lane_shift_feasibility(**kwargs)`

Static matplotlib plots. Call after the corresponding `simulate_*()` method.

---

## Animation

### `animate_vessel(vessel_id, out_path="vessel_comparison.gif", **kwargs)`
Replay a single real/cleaned AIS track, original vs. its computed reroute.

### `animate_transit(scenario="baseline", side=None, target_polygon=None, base_speed_knots=12.0, reduction=0.5, n_frames=150, out_path="transit.gif", **kwargs)`
Animate a single schematic ship (rectangle body + triangular bow) under
one scenario: `"baseline"`, `"speed_reduction"`, or `"reroute"`.

### `animate_transit_comparison(side=None, target_polygon=None, base_speed_knots=12.0, reduction=0.5, n_frames=150, out_path="transit_comparison.gif", scenarios=None, **kwargs)`
Animate all three scenarios simultaneously on a **shared real clock** — a
faster scenario visibly finishes and waits while slower ones keep moving.
`show_time_chart=True` (default) adds a live bar-chart panel of elapsed
time per scenario.

Both accept: `ship_color` (single color or `{scenario: color}` dict),
`ship_length`/`ship_width` (metres), `lane_color`/`lane_width`,
`background` (`"light"`/`"dark"`/`"raster"`/`"satellite"`),
`bounds` (from `zoom_bounds_latlon()`), `reroute_direction`
(`"auto"`/`"option_1"`/`"option_2"`/compass label).

Saves as `.gif` (no extra dependencies) or any other extension (e.g.
`.mp4`, needs `ffmpeg` installed on the system).
