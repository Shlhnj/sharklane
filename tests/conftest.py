"""Shared pytest fixtures: a synthetic two-headland bay-mouth scenario,
matching the one used to smoke-test the package during development."""
import sys
from pathlib import Path

# Make the package importable directly from the source tree, regardless of
# whether `pip install -e .` was run first. This is a deliberate safety net:
# CI or local setups that only install requirements.txt (dependencies) but
# skip installing the package itself would otherwise fail with
# "ModuleNotFoundError: No module named 'sharklane'" even though everything
# needed to run the tests is actually present in the checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, LineString, box
import pytest

from sharklane import Simulator
from sharklane.ais import compute_speeds, clean_tracks, to_tracks

CRS = "EPSG:32750"  # UTM 50S


@pytest.fixture
def synthetic_paths(tmp_path):
    """Write synthetic core habitat, coastline, and corridor files to a
    temp directory, and return their paths."""
    habitat = Polygon([(2000, 8000), (4000, 8000), (4000, 9500), (2000, 9500)])
    habitat_path = tmp_path / "core_habitat.geojson"
    gpd.GeoDataFrame(geometry=[habitat], crs=CRS).to_file(habitat_path, driver="GeoJSON")

    west_headland = box(-2000, -2000, 1200, 6000)
    east_headland = box(2800, -2000, 6000, 6000)
    coastline_path = tmp_path / "coastline.geojson"
    gpd.GeoDataFrame(geometry=[west_headland, east_headland], crs=CRS).to_file(
        coastline_path, driver="GeoJSON")

    corridor = LineString([(1200, 6200), (2800, 6200)])
    corridor_path = tmp_path / "corridor.geojson"
    gpd.GeoDataFrame(geometry=[corridor], crs=CRS).to_file(corridor_path, driver="GeoJSON")

    return {"habitat": str(habitat_path), "coastline": str(coastline_path),
            "corridor": str(corridor_path)}


@pytest.fixture
def synthetic_ais_gdf():
    """Build a small synthetic AIS point GeoDataFrame directly (already in
    the working CRS, bypassing the WGS84-assuming CSV loader) covering a
    transiting vessel, a terminal (bay-entering) vessel, and a fast lane
    vessel."""
    rng = np.random.default_rng(0)
    rows = []
    t0 = pd.Timestamp("2024-07-01")

    for i, x in enumerate(np.linspace(-1500, 6500, 40)):
        rows.append({"vessel_id": "A", "timestamp": t0 + pd.Timedelta(hours=i),
                     "lon": x, "lat": 6200 + rng.normal(0, 20)})
    for i, y in enumerate(np.linspace(3000, 8500, 30)):
        rows.append({"vessel_id": "B", "timestamp": t0 + pd.Timedelta(hours=i),
                     "lon": 2000 + rng.normal(0, 15), "lat": y})
    for i, x in enumerate(np.linspace(-2000, 7000, 20)):
        rows.append({"vessel_id": "C", "timestamp": t0 + pd.Timedelta(minutes=i * 1),
                     "lon": x, "lat": 6300 + rng.normal(0, 10)})

    df = pd.DataFrame(rows)
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs=CRS
    ).sort_values(["vessel_id", "timestamp"]).reset_index(drop=True)
    return gdf


@pytest.fixture
def synthetic_sim(synthetic_paths, synthetic_ais_gdf):
    """A fully wired-up Simulator: habitat, corridor, water mask, and
    cleaned synthetic AIS tracks. Speed reduction / redirection / lane
    shift have NOT been run yet -- individual tests call those themselves."""
    sim = Simulator(working_crs=CRS)
    sim.load_core_habitat(synthetic_paths["habitat"], source_crs=CRS)
    sim.load_transit_line(synthetic_paths["corridor"], source_crs=CRS)
    sim.build_mask(synthetic_paths["coastline"],
                    bounds=(-3000, -3000, 8000, 10500), resolution=100, source_crs=CRS)

    cleaned = clean_tracks(synthetic_ais_gdf, crs_metric=CRS, max_speed_mps=500,
                            min_points=5, stationary_frac_limit=0.9)
    sim.tracks = to_tracks(cleaned)
    return sim
