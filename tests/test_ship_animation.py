"""Tests for sharklane.viz.ship track-generation logic. Deliberately does
NOT render actual GIFs here (slow, and not meaningful to assert on) --
just the underlying geometry/timing math."""
import numpy as np
from shapely.geometry import Polygon, LineString, Point
import pytest

from sharklane.viz.ship import (
    ship_polygon, build_baseline_track, build_speed_reduction_track,
    build_reroute_track, compute_reroute_options,
    build_scenario_track_raw, resample_common_timeline,
    get_valid_sides, get_default_side,
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


def test_reroute_routes_around_convex_hull_not_raw_concave_boundary():
    # A polygon with a notch carved out of it: the shortest way around
    # should cut straight across the notch's mouth (via the convex hull),
    # NOT detour into the notch by tracing the raw concave boundary. This
    # locks in the fix for a real bug where routes hugged every concave
    # dent, producing needlessly long, visually bad paths.
    notched = Polygon([(0, 0), (1000, 0), (1000, 500), (0, 500)]).difference(
        Polygon([(300, 0), (700, 0), (700, 250), (300, 250)]))
    lane = LineString([(-500, 480), (1500, 480)])  # runs near the top, notch is at the bottom

    options, _ = compute_reroute_options(lane, notched, side="west")
    south_option = max(options.values(), key=lambda o: o["length_m"])  # the longer ("south") route

    # what the OLD (raw concave boundary) method would have produced, for comparison
    raw_boundary = notched.exterior
    full_line = LineString([(-500, 480), (1500, 480)])
    inter = full_line.intersection(raw_boundary)
    pts = list(inter.geoms) if hasattr(inter, "geoms") else [inter]
    start = Point(-500, 480)
    pts = sorted(pts, key=lambda p: start.distance(p))
    entry, exit_ = pts[0], pts[-1]
    perim = raw_boundary.length
    d_entry, d_exit = raw_boundary.project(entry), raw_boundary.project(exit_)
    old_south_length = max((d_exit - d_entry) % perim, (d_entry - d_exit) % perim)

    # the new (hull-based) south route must be meaningfully shorter than
    # what tracing the raw concave boundary would have given
    assert south_option["length_m"] < old_south_length * 0.9

    # and it must still never enter the TRUE (concave) polygon's interior
    xs, ys = build_reroute_track(lane, notched, side="west", n_frames=80, direction="option_2")
    path = LineString(zip(xs, ys))
    assert not notched.buffer(-1).intersects(path)


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


def test_lane_orientation_east_west_detected():
    ew_lane = LineString([(0, 0), (1000, 5)])  # mostly horizontal
    assert get_valid_sides(ew_lane) == ["west", "east"]
    assert get_default_side(ew_lane) == "west"


def test_lane_orientation_north_south_detected():
    ns_lane = LineString([(0, 0), (5, 1000)])  # mostly vertical
    assert get_valid_sides(ns_lane) == ["south", "north"]
    assert get_default_side(ns_lane) == "south"


def test_north_south_lane_side_labels_work():
    ns_lane = LineString([(500, -500), (500, 1500)])  # vertical lane through POLYGON
    xs, ys = build_baseline_track(ns_lane, side="south", n_frames=20)
    assert ys[0] == pytest.approx(-500)
    assert ys[-1] == pytest.approx(1500)

    xs2, ys2 = build_baseline_track(ns_lane, side="north", n_frames=20)
    assert ys2[0] == pytest.approx(1500)
    assert ys2[-1] == pytest.approx(-500)


def test_north_south_lane_rejects_west_east_labels():
    ns_lane = LineString([(500, -500), (500, 1500)])
    with pytest.raises(ValueError, match="north-south"):
        build_baseline_track(ns_lane, side="west", n_frames=20)


def test_east_west_lane_rejects_north_south_labels():
    with pytest.raises(ValueError, match="east-west"):
        build_baseline_track(CORRIDOR, side="north", n_frames=20)


def test_side_none_uses_orientation_appropriate_default():
    # east-west lane -> defaults to 'west' start
    xs, ys = build_baseline_track(CORRIDOR, side=None, n_frames=10)
    xs_explicit, ys_explicit = build_baseline_track(CORRIDOR, side="west", n_frames=10)
    assert np.allclose(xs, xs_explicit)

    # north-south lane -> defaults to 'south' start
    ns_lane = LineString([(500, -500), (500, 1500)])
    xs_ns, ys_ns = build_baseline_track(ns_lane, side=None, n_frames=10)
    xs_south, ys_south = build_baseline_track(ns_lane, side="south", n_frames=10)
    assert np.allclose(ys_ns, ys_south)


def test_endpoints_correct_regardless_of_line_drawing_direction():
    # a line whose FIRST coordinate is the geographically eastern point
    # (drawn right-to-left) must still resolve 'west'/'east' by actual
    # position, not by raw coordinate order
    reversed_drawn = LineString([(1500, 250), (-500, 250)])  # first point is east
    xs, ys = build_baseline_track(reversed_drawn, side="west", n_frames=10)
    assert xs[0] == pytest.approx(-500)  # starts at the true west end
    assert xs[-1] == pytest.approx(1500)


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
