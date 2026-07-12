# sharklane

A Python package for simulating ship-collision mitigation strategies around
whale shark aggregation sites, following the method in Womersley et al.
2024 (*Science of the Total Environment* 934:172776), extended to handle
coastline-constrained sites (e.g. bay mouths) via a water/land eligibility
mask and least-cost-path rerouting.

## Install

Once published to PyPI:

```bash
pip install sharklane
```

For local development (editable install from source):

```bash
pip install -e .
```

Requires: geopandas, shapely, rasterio, scikit-image, matplotlib, pandas, numpy.

## Workflow

```python
from sharklane import Simulator

sim = Simulator(working_crs="EPSG:32750")  # pick a UTM zone matching your AOI

# 1. core habitat polygon (shapefile/geojson/gpkg)
sim.load_core_habitat("core_habitat.shp")

# 2. transit/corridor line -- e.g. across a bay mouth. Load a pre-digitized
#    line, or draw one interactively over the habitat:
sim.load_transit_line("corridor.shp")
# sim.draw_transit_line()   # alternative: click points in a popup window

# AIS tracks (CSV with vessel_id, timestamp, lon, lat columns)
sim.load_ais("ais_tracks.csv", min_points=20, stationary_frac_limit=0.25)

# 4. water/land mask, built from a coastline polygon layer
sim.build_mask("coastline.shp", resolution=100)

# 3. speed reduction simulation (10-75% reductions by default)
speed_summary = sim.simulate_speed_reduction()

# 5. redirection / rerouting (land-mask-constrained least-cost path)
reroute_time = sim.simulate_redirection(method="least_cost")

# 6. lane shift feasibility (for fast, consistent-route vessels)
lane_result = sim.simulate_lane_shift(speed_threshold_mps=7.7)

# 7. graphics
sim.plot_site_map()
sim.plot_speed_reduction_curve()
sim.plot_reroute_paths()
sim.plot_lane_shift_feasibility()

# 8. animation of a specific vessel, original vs. rerouted path
sim.animate_vessel("VESSEL_ID_123", out_path="comparison.gif")
```

## Notes specific to bay-mouth sites (e.g. Saleh Bay)

If your risk zone is a transit corridor (bay mouth) rather than the
aggregation habitat itself, use `load_transit_line()` / `draw_transit_line()`
-- the Simulator will automatically use a buffer around that line
(`corridor_polygon_or_habitat()`, default 500 m buffer, adjust in code) as
the risk zone for all three mitigation simulations, instead of the habitat
polygon.

`simulate_redirection()` classifies vessels into `transit` (pass through,
don't enter/exit through the corridor as an origin/destination) vs.
`terminal` (genuinely entering or leaving through the corridor) using
`sharklane.simulate.reroute.classify_transit_vs_terminal()`. Only
`transit` vessels are meaningfully reroutable -- rerouting a vessel whose
destination is inside the bay doesn't make sense, since it must cross the
corridor regardless of path. Check `sim.results["redirection"]["labels"]`
and treat these two groups separately in your reporting.

`simulate_lane_shift()` may legitimately return
`min_feasible_offset_m = None` at a narrow bay mouth -- meaning no lateral
shift clears the risk zone while staying in water. This is a valid finding
(reported as such), not a failure -- in that case speed reduction and/or
rerouting become your primary reportable mitigation levers.

## Raster vectorization (threshold-based extraction)

If your land/water data is a raster (bathymetry, classified landcover, a
risk-density grid, etc.) rather than a vector coastline, you can extract
polygons directly by threshold filter:

```python
from sharklane.raster_vectorize import vectorize_raster, water_mask_from_raster

# extract polygons where pixel value satisfies the filter
land_gdf = vectorize_raster("bathymetry.tif", threshold=0, comparison=">=")

# or build the water/land mask directly from a raster, skipping the
# vector round-trip (reprojects into your working CRS if needed)
sim.build_mask_from_raster("bathymetry.tif", threshold=0, comparison="<",
                            target_resolution=100)

# generic raster -> polygon extraction for other purposes
polys = sim.vectorize_raster("risk_density.tif", threshold=50, comparison=">=",
                              min_area=1e4, simplify_tolerance=50)
```

`comparison` is one of `>=`, `>`, `<=`, `<`, `==`, `!=`, applied as
`pixel_value {comparison} threshold`. `min_area` / `simplify_tolerance` help
clean up speckle and stair-stepping from raster-derived polygons.

## World shipping lanes (Major / Middle / Minor)

The package bundles a global shipping lanes dataset (hand-digitized from
the CIA's Map of the World's Oceans, via
[newzealandpaul/Shipping-Lanes](https://github.com/newzealandpaul/Shipping-Lanes),
CC BY 4.0). Use it to orient your site relative to global shipping traffic,
or as a fallback `corridor_line` when you don't yet have local AIS to
derive a representative lane from:

```python
from sharklane.shipping_lanes import load_world_shipping_lanes, explode_to_segments

lanes = load_world_shipping_lanes(bbox=(116, -9, 119, -7), lane_type="Middle")

# or, directly on a Simulator with a habitat already loaded:
sim.load_world_shipping_lane(lane_type="Middle", pad_deg=2.0)
```

**Important caveat:** this is a coarse, hand-digitized global dataset (3
features total worldwide -- one MultiLineString per Type), useful for
orientation and as a rough fallback, but nowhere near the resolution needed
to draw real conclusions about vessel behaviour at a bay-mouth scale. Once
you have local AIS, derive your working lane from that instead (as in
`examples/example_saleh_bay.py`) and treat this dataset as context, not
ground truth.



```
sharklane/
  io.py               loading shapefiles / interactively drawing lines
  masking.py          water/land raster mask (from vector) + WaterMask class
  raster_vectorize.py threshold-based raster -> polygon extraction, raster -> WaterMask
  shipping_lanes.py    bundled world shipping lanes (Major/Middle/Minor)
  data/
    world_shipping_lanes.geojson   bundled dataset
  ais.py              AIS loading, speed computation, track cleaning
  simulate/
    speed.py          speed reduction simulation
    reroute.py         perimeter reroute + land-mask-constrained least-cost reroute
    laneshift.py       lane shift feasibility testing
  viz/
    static.py          site map, speed curve, reroute path, lane shift plots
    animate.py          original-vs-rerouted vessel animation
  simulator.py          Simulator: orchestrates the full workflow
```

## Choosing which reroute path to animate

When the risk polygon splits the approach into two plausible go-around
routes (e.g. north vs. south of a roughly east-west habitat), the package
computes both and lets you inspect or pick one -- it doesn't just silently
default to the shorter one:

```python
# inspect both options first
opts = sim.list_reroute_options(side="west")
print(opts)
# {'option_1': {'length_km': 69.4, 'est_time_hours': 3.13, 'side': 'south'},
#  'option_2': {'length_km': 69.5, 'est_time_hours': 3.13, 'side': 'north'},
#  'similar_length': True,
#  'note': 'The two reroute options are within 15% of each other...'}

# then pick one explicitly when animating
sim.animate_transit(scenario="reroute", side="west", reroute_direction="option_1")
sim.animate_transit(scenario="reroute", side="west", reroute_direction="south")  # equivalent here

# or just take the shorter one automatically (default)
sim.animate_transit(scenario="reroute", side="west", reroute_direction="auto")
```

`side` labels are 'north'/'south' for a roughly east-west lane, or
'east'/'west' for a roughly north-south lane -- `list_reroute_options()`
tells you which apply to your actual geometry. `similar_length=True` (within
15%) is your signal that the choice is genuinely worth making deliberately
(e.g. based on real shark movement data, or which side has calmer water)
rather than leaving it on 'auto'.

**Vessel approach direction** (`side='west'` vs `side='east'`) has been
available on `animate_transit()` / `animate_transit_comparison()` since
they were added -- it picks which end of the corridor/lane line the ship
starts from.

## Testing

Real pytest test suite under `tests/` (not the old ad-hoc script) --
covers the core pipeline (habitat/mask loading, speed reduction, rerouting,
lane shift), raster vectorization, ship animation track math (including
the corner-cutting fix and shared-timeline scenario comparison), and the
bundled shipping lanes dataset. No network access required -- everything
runs against synthetic fixtures or the bundled dataset.

```bash
pip install -e .
pip install pytest
pytest tests/ -v
```

CI runs this across Python 3.9-3.12 via `.github/workflows/tests.yml`.
`python_requires>=3.9` in setup.py reflects actual tested compatibility
(the codebase uses `from __future__ import annotations` throughout, which
defers evaluation of PEP 604 union-type annotations like `str | None` so
they work fine pre-3.10).

**A note on the two real bugs this test suite caught during development,**
in case they're informative for anyone extending the package:
1. `ais.py`'s `compute_speeds()` originally derived elapsed time via
   `timestamp.astype("int64") / 1e9`, which assumes datetime64[ns]
   resolution. pandas 3.x can store datetime64 at coarser resolution
   depending on construction, silently making that conversion wrong by a
   fixed factor. Fixed by using `.diff().dt.total_seconds()`, which is
   resolution-agnostic.
2. The reroute animation's "go around the polygon" path could visually
   cut across sharp corners at low sample density -- not because the
   underlying boundary-following math was wrong, but because two
   *resampling* passes (one adaptive, one uniform-distance) each
   individually looked fine while their composition could skip a corner
   vertex. Fixed by forcing true polygon vertices into the sample set and
   removing the redundant second resampling pass entirely.

## Tested (worked example)

`examples/example_saleh_bay.py` runs the full pipeline (steps 1-8) against
a real bbox/lane/coastline setup (Moyo Island, Flores Sea) as an end-to-end
worked example -- run it to see the complete workflow with real geometry
before adapting it to your own site.
