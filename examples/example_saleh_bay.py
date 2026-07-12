"""
Worked example: Saleh Bay / Flores Sea, using real coordinates supplied by
the user and real coastline geometry (Natural Earth 10m coastline, clipped
to Moyo Island + the Sumbawa mainland region).

  bbox (core habitat / risk polygon), WGS84:
      117.520752, -8.083704, 117.824249, -7.830731
  lane (representative shipping lane crossing the area), WGS84:
      117.362823, -7.946357  ->  117.925873, -7.943636

NOTE: AIS traffic here is SYNTHETIC (no real AIS feed plugged in yet) --
built to plausibly follow the given lane, for demonstration purposes only.
Swap sharklane.ais.load_ais_csv() in for real AIS once you have it.
"""
import sys
sys.path.insert(0, "/home/claude/sharklane")

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, LineString
import matplotlib
matplotlib.use("Agg")

from sharklane import Simulator
from sharklane.ais import compute_speeds, clean_tracks, to_tracks

CRS_WGS84 = "EPSG:4326"
CRS_UTM50S = "EPSG:32750"  # UTM zone 50S -- correct zone for 114-120E, southern hemisphere

# ---------------------------------------------------------------------------
# 1. Core habitat / risk polygon -- corrected bbox
# ---------------------------------------------------------------------------
bbox_coords = (117.420502, -8.051071, 117.798157, -7.799439)
habitat = box(*bbox_coords)
habitat_gdf = gpd.GeoDataFrame(geometry=[habitat], crs=CRS_WGS84)
habitat_gdf.to_file("/home/claude/sharklane/examples/saleh_bay_data/core_habitat.geojson", driver="GeoJSON")

# ---------------------------------------------------------------------------
# 2. Representative shipping lane -- derived per instructions:
#    lon extended 0.25 deg beyond each end of the bbox, at the bbox's
#    center latitude
# ---------------------------------------------------------------------------
minx, miny, maxx, maxy = bbox_coords
center_lat = (miny + maxy) / 2
lane_minx = minx - 0.25
lane_maxx = maxx + 0.25
lane = LineString([(lane_minx, center_lat), (lane_maxx, center_lat)])
lane_gdf = gpd.GeoDataFrame(geometry=[lane], crs=CRS_WGS84)
lane_gdf.to_file("/home/claude/sharklane/examples/saleh_bay_data/lane.geojson", driver="GeoJSON")
print(f"Derived lane: ({lane_minx:.6f}, {center_lat:.6f}) -> ({lane_maxx:.6f}, {center_lat:.6f})")

# ---------------------------------------------------------------------------
# 3. Real coastline (Moyo Island + Sumbawa mainland region), Natural Earth 10m
# ---------------------------------------------------------------------------
land_path = "/home/claude/sharklane/examples/saleh_bay_data/saleh_bay_land.geojson"

# ---------------------------------------------------------------------------
# Set up Simulator
# ---------------------------------------------------------------------------
sim = Simulator(working_crs=CRS_UTM50S)
sim.load_core_habitat(
    "/home/claude/sharklane/examples/saleh_bay_data/core_habitat.geojson",
    source_crs=CRS_WGS84,
)
sim.load_transit_line(
    "/home/claude/sharklane/examples/saleh_bay_data/lane.geojson",
    source_crs=CRS_WGS84,
)

print("Habitat polygon area (km^2):", sim.polygon.area / 1e6)
print("Lane length (km):", sim.corridor_line.length / 1e3)

# ---------------------------------------------------------------------------
# 4. Build water/land mask from real coastline (Moyo Island + mainland)
# ---------------------------------------------------------------------------
# pad bounds generously so Moyo Island / mainland show up and reroutes have
# room to work with
minx, miny, maxx, maxy = sim.polygon.bounds
lane_minx, lane_miny, lane_maxx, lane_maxy = sim.corridor_line.bounds
minx = min(minx, lane_minx) - 15000
miny = min(miny, lane_miny) - 20000
maxx = max(maxx, lane_maxx) + 15000
maxy = max(maxy, lane_maxy) + 15000

sim.build_mask(land_path, bounds=(minx, miny, maxx, maxy), resolution=200, source_crs=CRS_WGS84)
print("Mask shape:", sim.water_mask.shape, "| water fraction:", sim.water_mask.mask.mean())

# ---------------------------------------------------------------------------
# Synthetic AIS: vessels following the lane, at realistic cargo-ship speeds,
# with some jitter, crossing the habitat bbox. Replace with real AIS.
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
lane_x0, lane_y0 = sim.corridor_line.coords[0]
lane_x1, lane_y1 = sim.corridor_line.coords[1]

rows = []
t0 = pd.Timestamp("2024-08-01")
n_vessels = 25
for v in range(n_vessels):
    frac_offset = rng.uniform(-0.15, 0.15)  # lateral jitter as fraction of lane length
    speed_knots = rng.uniform(10, 20)
    speed_mps = speed_knots * 0.514
    n_pts = rng.integers(30, 60)
    t_start = t0 + pd.Timedelta(hours=int(rng.integers(0, 24 * 60)))

    xs = np.linspace(lane_x0, lane_x1, n_pts) + rng.normal(0, 300, n_pts)
    ys = np.linspace(lane_y0, lane_y1, n_pts) + frac_offset * (lane_y1 - lane_y0) + rng.normal(0, 300, n_pts)
    dt_hours = np.cumsum([0] + [abs(xs[i+1]-xs[i]) / speed_mps / 3600 for i in range(n_pts - 1)])

    for i in range(n_pts):
        rows.append({
            "vessel_id": f"V{v:03d}",
            "timestamp": t_start + pd.Timedelta(hours=dt_hours[i]),
            "x": xs[i], "y": ys[i],
        })

# a handful of vessels genuinely entering the habitat area (not just transiting the lane)
for v in range(3):
    n_pts = 25
    entry_x = rng.uniform(lane_x0, lane_x1)
    xs = np.full(n_pts, entry_x) + rng.normal(0, 200, n_pts)
    ys = np.linspace(lane_y0 - 5000, sim.polygon.centroid.y, n_pts)
    t_start = t0 + pd.Timedelta(hours=int(rng.integers(0, 24 * 60)))
    for i in range(n_pts):
        rows.append({
            "vessel_id": f"ENTER{v:02d}",
            "timestamp": t_start + pd.Timedelta(hours=i),
            "x": xs[i], "y": ys[i],
        })

ais_df = pd.DataFrame(rows).sort_values(["vessel_id", "timestamp"]).reset_index(drop=True)
gdf = gpd.GeoDataFrame(ais_df, geometry=gpd.points_from_xy(ais_df["x"], ais_df["y"]), crs=CRS_UTM50S)
cleaned = clean_tracks(gdf, crs_metric=CRS_UTM50S, max_speed_mps=30, min_points=10,
                        stationary_frac_limit=0.9)
sim.tracks = to_tracks(cleaned)
print(f"\nSynthetic vessels loaded: {len(sim.tracks)}")

# ---------------------------------------------------------------------------
# 3. Speed reduction simulation
# ---------------------------------------------------------------------------
# IMPORTANT: target the core HABITAT polygon here, not the lane's corridor
# buffer. The lane is ~97 km long -- a 500 m buffer around it is a long,
# thin strip, not a compact "risk zone" in the sense the paper's method
# assumes (e.g. Ewing Bank's core habitat was ~18 km^2). Speed reduction /
# rerouting should apply to the actual localized area you're protecting;
# the lane itself is only used below to identify "lane vessel" behaviour
# for the lane-shift analysis.
print("\n=== Speed reduction (target: core habitat polygon) ===")
speed_summary = sim.simulate_speed_reduction(reductions=[0.25, 0.5, 0.75],
                                              target_polygon=sim.polygon)
print(speed_summary)

# ---------------------------------------------------------------------------
# 5. Redirection (land-mask-constrained, accounts for Moyo Island)
# ---------------------------------------------------------------------------
print("\n=== Redirection (target: core habitat polygon) ===")
try:
    reroute_time = sim.simulate_redirection(method="least_cost", target_polygon=sim.polygon)
    print("Vessel classification:", sim.results["redirection"]["labels"])
    print(reroute_time)
except Exception as e:
    print("Redirection error:", repr(e))

# ---------------------------------------------------------------------------
# 6. Lane shift feasibility (checking against Moyo Island's position)
# ---------------------------------------------------------------------------
print("\n=== Lane shift ===")
try:
    lane_result = sim.simulate_lane_shift(
        speed_threshold_mps=5.0,
        direction=(0, 1),  # shift north, away from Moyo Island/mainland
        offsets_m=list(range(0, 20000, 1000)),
    )
    print("Min feasible northward offset (m):", lane_result.get("min_feasible_offset_m"))

    lane_result_south = sim.simulate_lane_shift(
        speed_threshold_mps=5.0,
        direction=(0, -1),  # shift south, toward Moyo Island/mainland
        offsets_m=list(range(0, 20000, 1000)),
    )
    print("Min feasible southward offset (m):", lane_result_south.get("min_feasible_offset_m"))
except Exception as e:
    print("Lane shift error:", repr(e))

# ---------------------------------------------------------------------------
# 7. Graphics
# ---------------------------------------------------------------------------
ax = sim.plot_site_map(target_polygon=sim.polygon,
                       title="Saleh Bay / Flores Sea example (real coastline)")
ax.figure.savefig("/home/claude/sharklane/examples/saleh_bay_site_map.png", dpi=130)

ax2 = sim.plot_speed_reduction_curve()
ax2.figure.savefig("/home/claude/sharklane/examples/saleh_bay_speed_curve.png", dpi=130)

try:
    ax3 = sim.plot_reroute_paths(target_polygon=sim.polygon)
    ax3.figure.savefig("/home/claude/sharklane/examples/saleh_bay_reroutes.png", dpi=130)
except Exception as e:
    print("Reroute plot error:", repr(e))

print("\nDone. Plots saved to examples/.")
