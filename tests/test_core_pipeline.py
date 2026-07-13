"""Tests for the core Simulator pipeline: loading, masking, and the three
mitigation simulations."""
import numpy as np
import pytest


def test_load_core_habitat(synthetic_sim):
    assert synthetic_sim.polygon is not None
    assert synthetic_sim.polygon.area == pytest.approx(2000 * 1500, rel=0.01)


def test_zoom_bounds_latlon_produces_reasonable_width(synthetic_sim):
    # a ~0.1 degree lon span near the equator should be roughly 11km
    bounds = synthetic_sim.zoom_bounds_latlon(139.0, -8.0, 139.1, -7.9)
    width_km = (bounds[2] - bounds[0]) / 1000
    assert 5 < width_km < 15  # loose bound, just sanity-checking scale


def test_load_transit_line(synthetic_sim):
    assert synthetic_sim.corridor_line is not None
    assert synthetic_sim.corridor_line.length == pytest.approx(1600, rel=0.01)


def test_water_mask_built(synthetic_sim):
    wm = synthetic_sim.water_mask
    assert wm is not None
    assert wm.mask.ndim == 2
    # both water and land pixels should be present
    assert 0 < wm.mask.mean() < 1


def test_water_mask_water_land_lookup(synthetic_sim):
    wm = synthetic_sim.water_mask
    # a point well inside the habitat polygon should be water
    assert wm.is_water(3000, 8700) is True
    # a point well inside a headland should be land
    assert wm.is_water(0, 0) is False


def test_ais_tracks_loaded(synthetic_sim):
    assert len(synthetic_sim.tracks) == 3
    assert set(synthetic_sim.tracks.keys()) == {"A", "B", "C"}


def test_speed_reduction_runs(synthetic_sim):
    summary = synthetic_sim.simulate_speed_reduction(reductions=[0.25, 0.5, 0.75])
    assert len(summary) == 3
    assert (summary["mean_pct_increase"] >= 0).all()
    # higher speed reduction should never produce a lower mean time increase
    assert summary.sort_values("reduction")["mean_pct_increase"].is_monotonic_increasing


def test_speed_reduction_raises_clear_error_with_no_tracks(synthetic_paths):
    # regression test: forgetting to call load_ais() (a common mistake --
    # see the empty-tracks case) previously produced a confusing pandas
    # KeyError deep inside summarize(); it should instead raise a clear,
    # actionable ValueError right at the source.
    from sharklane import Simulator
    sim = Simulator(working_crs="EPSG:32750")
    sim.load_core_habitat(synthetic_paths["habitat"], source_crs="EPSG:32750")
    assert sim.tracks == {}
    with pytest.raises(ValueError, match="No vessel tracks"):
        sim.simulate_speed_reduction()


def test_redirection_classifies_vessels(synthetic_sim):
    synthetic_sim.simulate_redirection(method="least_cost")
    labels = synthetic_sim.results["redirection"]["labels"]
    assert set(labels.keys()) == {"A", "B", "C"}
    assert labels["A"] in {"transit", "no_overlap", "inside_only", "terminal"}


def test_redirection_reroute_avoids_polygon_interior(synthetic_sim):
    synthetic_sim.simulate_redirection(method="least_cost")
    reroute_df = synthetic_sim.results["redirection"]["reroute"]
    if len(reroute_df) == 0:
        pytest.skip("No transit vessels in this synthetic scenario to reroute.")
    polygon = synthetic_sim.polygon
    for _, row in reroute_df.iterrows():
        path_line = row["path_line"]
        # shrink polygon slightly to allow boundary-hugging without false positives
        assert not polygon.buffer(-10).intersects(path_line)


def test_plot_reroute_paths_handles_empty_reroute_df():
    # regression test: an empty reroute_df (built from an empty list, as
    # happens when zero transit vessels are found) has NO columns at all,
    # not just empty rows -- plot_reroute_paths must not crash trying to
    # access reroute_df["vessel_id"] on it.
    import pandas as pd
    from shapely.geometry import Polygon
    from sharklane.viz.static import plot_reroute_paths

    empty_reroute_df = pd.DataFrame([])
    polygon = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    ax = plot_reroute_paths(empty_reroute_df, polygon, water_mask=None, tracks=None)
    assert ax is not None


def test_lane_shift_runs(synthetic_sim):
    result = synthetic_sim.simulate_lane_shift(
        speed_threshold_mps=3.0, direction=(0, 1),
        offsets_m=list(range(0, 3000, 250)))
    if result.get("feasible") is False and "reason" in result:
        # valid outcome: no vessel in this synthetic scenario met the lane
        # speed threshold -- not an error, just nothing to analyze
        assert result["reason"] == "no lane vessels identified"
    else:
        assert "results" in result
        assert "min_feasible_offset_m" in result
        assert result["min_feasible_offset_m"] is None or result["min_feasible_offset_m"] >= 0


def test_lane_shift_finds_lane_vessel(synthetic_sim):
    # vessel C in the fixture is deliberately fast/consistent -- confirm it
    # actually gets identified as a lane vessel and the analysis runs fully
    result = synthetic_sim.simulate_lane_shift(
        speed_threshold_mps=3.0, direction=(0, 1),
        offsets_m=list(range(0, 3000, 250)))
    assert "results" in result
    assert len(result["results"]) == len(range(0, 3000, 250))
