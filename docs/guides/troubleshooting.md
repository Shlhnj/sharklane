# Troubleshooting

Real issues encountered (and fixed) during development, kept here since
they're likely to recur for new users hitting the same edge cases.

## "No vessel tracks to simulate against" / confusing `KeyError: 'reduction'`

**Cause:** you called `simulate_speed_reduction()` (or another
`simulate_*()` method) before `load_ais()` — `sim.tracks` is empty.

**Fix:** call `sim.load_ais(...)` (or set `sim.tracks` directly from
synthetic data — see [Quickstart](../quickstart.md)) before any
`simulate_*()` call. The package now raises a clear `ValueError` pointing
at this directly, instead of the confusing pandas `KeyError` an earlier
version produced deep inside `summarize()`.

## Land isn't showing in animations

**Cause:** `build_mask()` was never called. Without it, `sim.water_mask`
is `None`, and the animation background renderer has nothing to draw —
it's not that land rendering is broken, there's simply no coastline data
loaded.

**Fix:** fetch a coastline layer and call `sim.build_mask(...)` before
animating. See [Colab setup](colab_setup.md) for a working coastline
fetch snippet.

## Land mask shows only a small patch when zoomed out

**Cause:** `build_mask(bounds=...)` was given a *tight* box (e.g. ±15km
around the habitat), but the animation/plot is displayed zoomed out
further than that (e.g. via `zoom_bounds_latlon()` with large padding).
The mask only draws within its own bounds — everything outside stays
blank white.

**Fix:** build the mask using bounds *at least as large* as anything
you'll ever zoom/pan to:

```python
aoi_bounds_proj = sim.zoom_bounds_latlon(AOI_MIN_LON, AOI_MIN_LAT, AOI_MAX_LON, AOI_MAX_LAT)
sim.build_mask("land.geojson", bounds=aoi_bounds_proj, resolution=300, source_crs="EPSG:4326")
# ... later, animate/plot with the SAME bounds:
sim.animate_transit_comparison(..., bounds=aoi_bounds_proj)
```

Consider a coarser `resolution` when covering a larger area, to keep the
raster from becoming unnecessarily large for the same pixel budget.

## `load_world_shipping_lane()` raises "does not intersect the polygon"

**Cause:** the auto-picked (or explicitly requested) lane doesn't
actually cross your habitat polygon — `load_world_shipping_lane()` picks
the *nearest* segment to your habitat's centroid, which is not guaranteed
to intersect it. This happens often with the bundled dataset, since it's
coarse (3 features worldwide).

**Fix:** wrap the call and check `sim.last_lane_crosses_habitat`,
falling back to a manual lane if needed:

```python
try:
    sim.load_world_shipping_lane(lane_type="auto", pad_deg=2.0,
                                  trim_to_polygon=True, trim_pad_fraction=0.25)
    lane_ok = sim.last_lane_crosses_habitat
except ValueError as e:
    print(f"No usable global lane: {e}")
    lane_ok = False

if not lane_ok:
    # build a manual lane matching the habitat's actual shape
    minx, miny, maxx, maxy = sim.polygon.bounds
    width, height = maxx - minx, maxy - miny
    mid_y, mid_x = (miny + maxy) / 2, (minx + maxx) / 2
    if width >= height:
        sim.corridor_line = LineString([(minx - 30000, mid_y), (maxx + 30000, mid_y)])
    else:
        sim.corridor_line = LineString([(mid_x, miny - 30000), (mid_x, maxy + 30000)])
```

## `ValueError: side must be 'west' or 'east'` (or vice versa)

**Cause:** you passed a `side` label that doesn't match your lane's
actual orientation — `west`/`east` only apply to a roughly east-west
lane; a north-south lane uses `north`/`south` instead.

**Fix:** check first, or just pass `side=None` to auto-resolve:

```python
print(sim.get_lane_side_options())
# {'valid_sides': ['south', 'north'], 'default_side': 'south', 'orientation': 'north_south'}
```

## Multiple disjoint polygons from `vectorize_raster()`

**Cause:** thresholding a raster (e.g. "high value" density pixels) often
produces several separate patches. Merging them via `union_all()` gives a
`MultiPolygon`, which the reroute/lane-shift functions can't use directly
(they call `.exterior`, which doesn't exist on a MultiPolygon).

**Fix:** pick a merge strategy based on what fits your data:

```python
merged = habitat_gdf.geometry.union_all()
if merged.geom_type == "MultiPolygon":
    # Option A: smallest convex polygon containing everything (simple, always works)
    habitat_polygon = merged.convex_hull
    # Option B: just the single biggest patch (simple, drops area)
    # habitat_polygon = max(merged.geoms, key=lambda g: g.area)
    # Option C: buffer-merge nearby patches, tune distance to your data's gaps
    # habitat_polygon = merged.buffer(2000).buffer(-2000)
else:
    habitat_polygon = merged
```

## MP4 animation fails with "Requested MovieWriter (ffmpeg) not available"

**Cause:** saving to `.mp4` needs the `ffmpeg` *binary* installed on the
system, not just a Python package. `.gif` output only needs Pillow and
always works.

**Fix:** `!apt-get install -y ffmpeg` (Colab normally has this
preinstalled already). Note `IPython.display.Image` won't render `.mp4`
inline — use `IPython.display.Video(filename=...)` instead.

## pandas 3.x speed/time numbers look ~1000x off

**Cause (historical, now fixed):** an earlier version of
`ais.compute_speeds()` derived elapsed time via
`timestamp.astype("int64") / 1e9`, assuming nanosecond-resolution
`datetime64`. pandas 3.x can store `datetime64` at coarser resolution
depending on construction, silently breaking that conversion. Fixed by
using `.diff().dt.total_seconds()`, which is resolution-agnostic. If
you're on an old package version and see suspiciously small elapsed
times, upgrade.
