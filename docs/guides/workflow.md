# Full worked workflow

This walks through the complete pipeline end to end, using a real habitat
site (Saleh Bay / Flores Sea, Indonesia) as a concrete example. The full
runnable script is at `examples/example_saleh_bay.py` in the repo.

## 1. Set up and load the habitat

```python
from sharklane import Simulator

sim = Simulator()  # working_crs auto-detected from the habitat's location
sim.load_core_habitat("core_habitat.geojson", source_crs="EPSG:4326")
print(f"Habitat area: {sim.polygon.area / 1e6:.1f} km^2")
```

## 2. Define your area of interest, once, up front

Define this in lon/lat so it can drive both coastline clipping and later
zoom/pan, consistently:

```python
import geopandas as gpd

_minx, _miny, _maxx, _maxy = gpd.GeoSeries([sim.polygon], crs=sim.working_crs).to_crs("EPSG:4326").total_bounds
_pad_deg = 1.0
AOI_MIN_LON, AOI_MIN_LAT = _minx - _pad_deg, _miny - _pad_deg
AOI_MAX_LON, AOI_MAX_LAT = _maxx + _pad_deg, _maxy + _pad_deg
```

## 3. Water/land mask

```python
# fetch coastline (see colab_setup.md), clip to the AOI above, then:
aoi_bounds_proj = sim.zoom_bounds_latlon(AOI_MIN_LON, AOI_MIN_LAT, AOI_MAX_LON, AOI_MAX_LAT)
sim.build_mask("land.geojson", bounds=aoi_bounds_proj, resolution=300, source_crs="EPSG:4326")
```

Using `aoi_bounds_proj` here — the same bounds you'll display later — is
deliberate; see [Troubleshooting](troubleshooting.md) for why a
mismatched, tighter mask box leaves blank space when zoomed out.

## 4. Shipping lane, with graceful fallback

```python
from shapely.geometry import LineString

try:
    sim.load_world_shipping_lane(lane_type="auto", pad_deg=2.0,
                                  trim_to_polygon=True, trim_pad_fraction=0.25)
    lane_ok = sim.last_lane_crosses_habitat
except ValueError as e:
    print(f"No usable global lane: {e}")
    lane_ok = False

if not lane_ok:
    minx, miny, maxx, maxy = sim.polygon.bounds
    width, height = maxx - minx, maxy - miny
    mid_y, mid_x = (miny + maxy) / 2, (minx + maxx) / 2
    if width >= height:
        sim.corridor_line = LineString([(minx - 30000, mid_y), (maxx + 30000, mid_y)])
    else:
        sim.corridor_line = LineString([(mid_x, miny - 30000), (mid_x, maxy + 30000)])

print(sim.get_lane_side_options())
```

The global lane not crossing the habitat is common (it's the *nearest*
segment, not a guaranteed intersection) — always check
`sim.last_lane_crosses_habitat` and have a fallback.

## 5. AIS

```python
sim.load_ais("ais_tracks.csv", min_points=20, stationary_frac_limit=0.25)
```

No real data yet? See the synthetic-track snippet in
[Quickstart](../quickstart.md).

## 6. Run the simulations

```python
speed_summary = sim.simulate_speed_reduction()
reroute_result = sim.simulate_redirection(method="least_cost")
lane_shift_result = sim.simulate_lane_shift(speed_threshold_mps=7.7)
```

## 7. Plot

```python
sim.plot_speed_reduction_curve().figure.savefig("speed_curve.png", dpi=120)
sim.plot_reroute_paths(target_polygon=sim.polygon).figure.savefig("reroutes.png", dpi=120)
```

## 8. Animate

```python
zoom_bounds = sim.zoom_bounds_latlon(AOI_MIN_LON, AOI_MIN_LAT, AOI_MAX_LON, AOI_MAX_LAT)
sim.animate_transit_comparison(
    side=None, reduction=0.6,
    out_path="comparison.gif", n_frames=100, fps=15,
    show_time_chart=True, bounds=zoom_bounds,
)
```

In Jupyter/Colab, display it:

```python
from IPython.display import Image, display
display(Image(filename="comparison.gif"))
```

## What this looks like on real data

Running this against the Saleh Bay habitat (1,158 km², Flores Sea, north
of Moyo Island and Sumbawa) produces:
- A 1,158 km² habitat polygon, correctly placed north of Moyo Island in
  real Natural Earth 10m coastline data
- A speed-reduction curve from 10-75% (1% increments) showing the
  transit-time cost of slowing vessels through the habitat
- Reroute paths that correctly hug the habitat's convex hull, avoiding
  its interior entirely (verified against a `buffer(-0.001)` tolerance)
- Both possible reroute directions (north/south of the habitat), with
  their relative lengths and an explicit "similar length, worth choosing
  deliberately" flag when applicable
