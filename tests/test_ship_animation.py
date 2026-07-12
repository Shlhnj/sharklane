"""Tests for sharklane.viz.ship track-generation logic. Deliberately does
NOT render actual GIFs here (slow, and not meaningful to assert on) --
just the underlying geometry/timing math."""
import numpy as np
from shapely.geometry import Polygon, LineString
import pytest

from sharklane.viz.ship import (
    ship_polygon, build_baseline_track, build_speed_reduction_track,
    build_reroute_track, compute_reroute_options,
    build_scenario_track_raw, resample_common_timeline,
)

# a simple rectangular habitat with a lane straight through the middle
POLYGON = Polygon([(0, 0), (1000, 0), (1000, 500), (0, 500)])
CORRIDOR = LineString([(-500, 250), (1500, 250)])


def test_ship_polygon_shape():
    pts = ship_polygon(0, 0, 0, length=10, width=4)
    assert pts.shape == (5, 2)
    # bow tip should be the furthest-forward point along heading 0 (+x axis)
    assert pts[0, 0] == pts[:, 0].max()


def test_ship_polygon_rotates():
    pts_0 = ship_polygon(0, 0, 0, length=10, width=4)
    pts_90 = ship_polygon(0, 0, np.pi / 2, length=10, width=4)
    # bow tip x should collapse near 0 and y should be maximal after a 90-degree turn
    assert abs(pts_90[0, 0]) < 1e-6
    assert pts_90[0, 1] == pytest.approx(5, abs=1e-6)


def test_build_baseline_track_endpoints():
    xs, ys = build_baseline_track(CORRIDOR, side="west", n_frames=50)
    assert xs[0] == pytest.approx(-500)
    assert xs[-1] == pytest.approx(1500)
    assert len(xs) == 50


def test_build_baseline_track_side_reverses_direction():
    xs_w, _ = build_baseline_track(CORRIDOR, side="west", n_frames=10)
    xs_e, _ = build_baseline_track(CORRIDOR, side="east", n_frames=10)
    assert xs_w[0] == pytest.approx(xs_e[-1])
    assert xs_w[-1] == pytest.approx(xs_e[0])


def test_speed_reduction_slower_inside_polygon():
    xs, ys, total_time = build_speed_reduction_track(
        CORRIDOR, POLYGON, side="west", base_speed_knots=12.0, reduction=0.6, n_frames=100)
    baseline_xs, baseline_ys = build_baseline_track(CORRIDOR, side="west", n_frames=100)
    # with a positive reduction, total time to cross should exceed the
    # baseline (unreduced) constant-speed equivalent
    speed = 12.0 * 0.514
    baseline_time = CORRIDOR.length / speed
    assert total_time > baseline_time


def test_speed_reduction_zero_equals_baseline_time():
    _, _, total_time = build_speed_reduction_track(
        CORRIDOR, POLYGON, side="west", base_speed_knots=12.0, reduction=0.0, n_frames=100)
    speed = 12.0 * 0.514
    baseline_time = CORRIDOR.length / speed
    assert total_time == pytest.approx(baseline_time, rel=0.01)


def test_compute_reroute_options_two_distinct_paths():
    options, similar = compute_reroute_options(CORRIDOR, POLYGON, side="west")
    assert set(options.keys()) == {"option_1", "option_2"}
    assert options["option_1"]["side"] != options["option_2"]["side"]
    # for a symmetric rectangle bisected by a horizontal lane, both arcs
    # should be very close in length
    assert similar is True


def test_reroute_track_avoids_polygon_interior():
    xs, ys = build_reroute_track(CORRIDOR, POLYGON, side="west", n_frames=80)
    path = LineString(zip(xs, ys))
    assert not POLYGON.buffer(-1).intersects(path)


def test_reroute_direction_selection_matches_options():
    options, _ = compute_reroute_options(CORRIDOR, POLYGON, side="west")
    xs1, ys1 = build_reroute_track(CORRIDOR, POLYGON, side="west", n_frames=60,
                                    direction="option_1")
    xs2, ys2 = build_reroute_track(CORRIDOR, POLYGON, side="west", n_frames=60,
                                    direction="option_2")
    # the two explicit options should trace visibly different paths
    assert not np.allclose(ys1, ys2, atol=1.0)


def test_reroute_direction_invalid_raises():
    with pytest.raises(ValueError):
        build_reroute_track(CORRIDOR, POLYGON, side="west", direction="not_a_real_direction")


def test_resample_common_timeline_faster_scenario_arrives_first():
    raw_baseline = build_scenario_track_raw("baseline", CORRIDOR, POLYGON, "west", 12.0, 0.5)
    raw_reroute = build_scenario_track_raw("reroute", CORRIDOR, POLYGON, "west", 12.0, 0.5)
    frame_times, resampled = resample_common_timeline(
        {"baseline": raw_baseline, "reroute": raw_reroute}, n_frames=50)

    _, _, _, baseline_total = resampled["baseline"]
    _, _, _, reroute_total = resampled["reroute"]
    # reroute detours around the polygon, so it should take longer than
    # the straight baseline transit at the same speed
    assert reroute_total > baseline_total

    # at the final shared frame, the baseline's elapsed time should have
    # capped at its own total (it arrived and is waiting), while reroute's
    # elapsed time should still be climbing toward the shared max
    baseline_elapsed = resampled["baseline"][2]
    assert baseline_elapsed[-1] == pytest.approx(baseline_total, rel=0.01)
