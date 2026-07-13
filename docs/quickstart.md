# Quickstart

A minimal end-to-end example: load a habitat, load AIS, run the three
mitigation simulations, and animate the result.

```python
from sharklane import Simulator

# 1. Set up -- working_crs auto-detects the correct UTM zone from your
#    habitat's own location, so you don't need to look up an EPSG code.
sim = Simulator()

# 2. Load your core habitat polygon (a whale shark aggregation site)
sim.load_core_habitat("core_habitat.geojson", source_crs="EPSG:4326")

# 3. Load a shipping lane -- either your own, or the bundled global dataset
sim.load_world_shipping_lane(lane_type="auto", pad_deg=2.0,
                              trim_to_polygon=True, trim_pad_fraction=0.25)

# 4. Load AIS vessel tracks (CSV with vessel_id, timestamp, lon, lat columns)
sim.load_ais("ais_tracks.csv", min_points=20, stationary_frac_limit=0.25)

# 5. (Optional but recommended) build a water/land mask -- needed for
#    rerouting and lane-shift feasibility, and for showing land in animations
sim.build_mask("coastline.geojson", resolution=200)

# 6. Run the three mitigation simulations
speed_summary = sim.simulate_speed_reduction()          # 1% increments, 10-75%
reroute_result = sim.simulate_redirection(method="least_cost")
lane_shift_result = sim.simulate_lane_shift(speed_threshold_mps=7.7)

# 7. Plot and save
sim.plot_speed_reduction_curve().figure.savefig("speed_curve.png", dpi=120)
sim.plot_reroute_paths().figure.savefig("reroutes.png", dpi=120)

# 8. Animate all three scenarios side by side
sim.animate_transit_comparison(
    side=None,           # auto-detected from the lane's orientation
    reduction=0.6,
    out_path="comparison.gif",   # or .mp4 (needs ffmpeg installed)
    n_frames=100, fps=15,
    show_time_chart=True,
)
```

## No real AIS yet?

Every simulation needs vessel tracks. If you don't have real AIS data,
build synthetic vessels that follow your lane, purely to test the
pipeline:

```python
import numpy as np, pandas as pd, geopandas as gpd
from sharklane.ais import clean_tracks, to_tracks

rng = np.random.default_rng(0)
x0, y0 = sim.corridor_line.coords[0]
x1, y1 = sim.corridor_line.coords[-1]

rows = []
for v in range(20):
    speed_mps = rng.uniform(10, 20) * 0.514   # each vessel gets its own random speed
    n_pts = rng.integers(30, 60)
    t_start = pd.Timestamp("2024-08-01") + pd.Timedelta(hours=int(rng.integers(0, 1440)))
    xs = np.linspace(x0, x1, n_pts) + rng.normal(0, 300, n_pts)
    ys = np.linspace(y0, y1, n_pts) + rng.normal(0, 300, n_pts)
    cum_h = np.concatenate([[0], np.cumsum(np.hypot(np.diff(xs), np.diff(ys)) / speed_mps / 3600)])
    for i in range(n_pts):
        rows.append({"vessel_id": f"V{v:03d}",
                     "timestamp": t_start + pd.Timedelta(hours=cum_h[i]),
                     "x": xs[i], "y": ys[i]})

df = pd.DataFrame(rows)
ais_gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["x"], df["y"]), crs=sim.working_crs)
sim.tracks = to_tracks(clean_tracks(ais_gdf, crs_metric=sim.working_crs,
                                     max_speed_mps=30, min_points=10,
                                     stationary_frac_limit=0.9))
```

See [Troubleshooting](guides/troubleshooting.md) if anything here doesn't
behave as expected -- most first-run issues have a known cause and fix.
