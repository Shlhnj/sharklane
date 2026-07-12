"""
Schematic ship animation: builds a simple ship icon (rectangle body +
triangular bow) and synthetic transit tracks for three illustrative
scenarios -- baseline transit, speed-reduced transit, and reroute-around
transit -- crossing the risk polygon from either side of the lane/corridor.

This is for illustration/communication (e.g. presentations, stakeholder
figures), distinct from sharklane.viz.animate, which replays real/cleaned
AIS tracks.
"""
from __future__ import annotations

import numpy as np
from shapely.geometry import Point, LineString


def ship_polygon(x: float, y: float, heading_rad: float,
                  length: float, width: float) -> np.ndarray:
    """Return (N,2) array of ship outline points (rectangle stern +
    triangular bow), centered at (x, y), pointing in heading_rad
    (radians, 0 = +x axis, counter-clockwise)."""
    L, W = length, width
    pts = np.array([
        [ L / 2,    0   ],   # bow tip
        [ L / 6,  W / 2],
        [-L / 2,  W / 2],
        [-L / 2, -W / 2],
        [ L / 6, -W / 2],
    ])
    c, s = np.cos(heading_rad), np.sin(heading_rad)
    R = np.array([[c, -s], [s, c]])
    return pts @ R.T + np.array([x, y])


def _lane_orientation(corridor_line: LineString) -> str:
    """Return 'east_west' or 'north_south' depending on which axis the
    corridor line predominantly runs along."""
    x0, y0 = corridor_line.coords[0]
    x1, y1 = corridor_line.coords[-1]
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    return "east_west" if dx >= dy else "north_south"


def get_valid_sides(corridor_line: LineString) -> list[str]:
    """Return the two valid `side` values for THIS lane's actual
    orientation: ('west', 'east') for a roughly east-west lane, or
    ('south', 'north') for a roughly north-south lane. A north-south lane
    has no 'west' or 'east' end to speak of -- check this (or just call
    get_default_side()) instead of assuming west/east always apply."""
    orientation = _lane_orientation(corridor_line)
    return ["west", "east"] if orientation == "east_west" else ["south", "north"]


def get_default_side(corridor_line: LineString) -> str:
    """The side used when none is specified explicitly -- the first
    option from get_valid_sides() for this lane's orientation ('west' for
    an east-west lane, 'south' for a north-south lane)."""
    return get_valid_sides(corridor_line)[0]


def _lane_endpoints(corridor_line: LineString, side: str = None):
    """
    Resolve which physical end of the corridor line a transit starts
    from, based on `side` and the line's ACTUAL orientation -- not just
    which raw coordinate happens to be first in the LineString. (An
    earlier version of this function assumed coords[0] was always the
    'west' end, which is silently wrong for a line drawn right-to-left,
    and had no concept of north-south lanes at all.)

    side : 'west'/'east' for a roughly east-west lane, 'south'/'north' for
        a roughly north-south lane. If None, defaults to
        get_default_side(corridor_line). Use get_valid_sides() to check
        which two labels apply to a given lane before picking one.
    """
    x0, y0 = corridor_line.coords[0]
    x1, y1 = corridor_line.coords[-1]
    dx, dy = x1 - x0, y1 - y0

    if side is None:
        side = get_default_side(corridor_line)

    if abs(dx) >= abs(dy):
        # east-west lane: label endpoints by actual x, not raw coord order
        west_pt, east_pt = ((x0, y0), (x1, y1)) if x0 <= x1 else ((x1, y1), (x0, y0))
        if side == "west":
            return (*west_pt, *east_pt)
        elif side == "east":
            return (*east_pt, *west_pt)
        else:
            raise ValueError(
                f"This corridor line runs roughly east-west; side must be "
                f"'west' or 'east', got {side!r}. Call "
                f"sharklane.viz.ship.get_valid_sides(corridor_line) to check."
            )
    else:
        # north-south lane: label endpoints by actual y
        south_pt, north_pt = ((x0, y0), (x1, y1)) if y0 <= y1 else ((x1, y1), (x0, y0))
        if side == "south":
            return (*south_pt, *north_pt)
        elif side == "north":
            return (*north_pt, *south_pt)
        else:
            raise ValueError(
                f"This corridor line runs roughly north-south; side must "
                f"be 'north' or 'south', got {side!r}. Call "
                f"sharklane.viz.ship.get_valid_sides(corridor_line) to check."
            )


def build_baseline_track(corridor_line: LineString, side: str = None,
                          n_frames: int = 150):
    """Straight, constant-speed transit from one lane endpoint to the other."""
    x0, y0, x1, y1 = _lane_endpoints(corridor_line, side)
    xs = np.linspace(x0, x1, n_frames)
    ys = np.linspace(y0, y1, n_frames)
    return xs, ys


def build_speed_reduction_track(corridor_line: LineString, polygon,
                                 side: str = None,
                                 base_speed_knots: float = 12.0,
                                 reduction: float = 0.5,
                                 n_spatial: int = 400,
                                 n_frames: int = 150):
    """
    Straight transit, but travelling at `base_speed_knots * (1-reduction)`
    while inside `polygon`. Frames are resampled at uniform TIME steps (not
    uniform spatial steps), so the ship icon visibly slows down while
    crossing the risk zone rather than just having denser points there.
    """
    x0, y0, x1, y1 = _lane_endpoints(corridor_line, side)
    xs = np.linspace(x0, x1, n_spatial)
    ys = np.linspace(y0, y1, n_spatial)

    inside = np.array([polygon.contains(Point(x, y)) for x, y in zip(xs, ys)])
    seg_dist = np.hypot(np.diff(xs), np.diff(ys))
    base_speed = base_speed_knots * 0.514  # m/s
    seg_speed = np.full(len(seg_dist), base_speed)
    seg_inside = inside[:-1] | inside[1:]
    seg_speed[seg_inside] = base_speed * max(1e-3, 1 - reduction)

    seg_time = seg_dist / seg_speed
    cum_time = np.concatenate([[0], np.cumsum(seg_time)])
    total_time = cum_time[-1]

    frame_times = np.linspace(0, total_time, n_frames)
    fx = np.interp(frame_times, cum_time, xs)
    fy = np.interp(frame_times, cum_time, ys)
    return fx, fy, total_time


def compute_reroute_options(corridor_line: LineString, polygon, side: str = None,
                             n_arc_samples: int = 300):
    """
    Compute BOTH possible go-around paths (the two arcs connecting the
    entry and exit points), not just the shorter one. Useful when the
    polygon splits the approach into two similar-length routes -- you can
    inspect both and pick.

    IMPORTANT: routes are built around the polygon's CONVEX HULL, not its
    raw boundary. For routing *around* an obstacle (staying outside it),
    the shortest path only ever needs to touch convex hull vertices --
    tracing into a concave notch and back out is never shorter than
    cutting straight across it, since the notch's mouth is, by
    construction, a chord of the hull. Walking the raw (possibly concave)
    boundary instead -- which earlier versions of this function did --
    forces the path to detour into every notch unnecessarily, producing
    visibly bad, needlessly long routes for any non-convex habitat shape.
    The hull is always a superset of the polygon, so a path that stays
    outside the hull also always stays outside the original polygon.

    Returns
    -------
    options : dict with keys 'option_1' (the entry->exit hull arc in the
        "forward" direction) and 'option_2' (the "backward" direction).
        Each value is a dict with:
          'arc_line'   : shapely LineString of the hull-boundary arc alone
          'length_m'   : arc length (m)
          'side'       : a rough compass label ('north'/'south' if the
                         lane runs mostly east-west, 'east'/'west' if it
                         runs mostly north-south) for which side of the
                         straight entry-exit line this arc bulges toward
          'boundary_entry', 'boundary_exit' : shapely Points (on the hull)
    similar_length : bool, True if the two options differ in length by
        less than 15% -- a signal that it's genuinely worth offering the
        choice rather than just defaulting to the shorter one.
    """
    hull = polygon.convex_hull
    boundary = hull.exterior

    x0, y0, x1, y1 = _lane_endpoints(corridor_line, side)
    full_line = LineString([(x0, y0), (x1, y1)])
    inter = full_line.intersection(boundary)

    pts = list(inter.geoms) if hasattr(inter, "geoms") else [inter]
    pts = [p for p in pts if p.geom_type == "Point"]
    if len(pts) < 2:
        raise ValueError(
            "The lane/corridor line does not cross the polygon's convex "
            "hull boundary twice from this side -- cannot compute reroute "
            "options. Check that the corridor line actually passes through "
            "the polygon."
        )

    start_pt = Point(x0, y0)
    pts_sorted = sorted(pts, key=lambda p: start_pt.distance(p))
    boundary_entry, boundary_exit = pts_sorted[0], pts_sorted[-1]

    perim = boundary.length
    d_entry = boundary.project(boundary_entry)
    d_exit = boundary.project(boundary_exit)
    forward_len = (d_exit - d_entry) % perim
    backward_len = (d_entry - d_exit) % perim

    def sample_arc(direction_sign, arc_len):
        n = max(int(n_arc_samples * arc_len / perim), 50)
        uniform_traveled = np.linspace(0, arc_len, n)

        # Force the hull's actual vertices into the sample set (uniform
        # arc-length sampling alone can straddle a corner without landing
        # exactly on it). Since the hull is convex, this now gives the
        # true taut/shortest path -- walking a convex boundary between two
        # points already IS the shortest way around, unlike for the
        # original concave polygon.
        vertex_coords = list(boundary.coords)[:-1]  # drop duplicate closing point
        vertex_offsets = np.array([boundary.project(Point(c)) for c in vertex_coords])
        if direction_sign > 0:
            vertex_traveled = (vertex_offsets - d_entry) % perim
        else:
            vertex_traveled = (d_entry - vertex_offsets) % perim
        vertex_traveled = vertex_traveled[vertex_traveled <= arc_len]

        all_traveled = np.unique(np.concatenate([uniform_traveled, vertex_traveled]))
        distances = (d_entry + direction_sign * all_traveled) % perim
        pts_ = [boundary.interpolate(d) for d in distances]
        return LineString([(p.x, p.y) for p in pts_])

    arc_fwd = sample_arc(+1, forward_len)
    arc_bwd = sample_arc(-1, backward_len)

    lane_dx, lane_dy = x1 - x0, y1 - y0

    def side_label(arc_line):
        mid = arc_line.interpolate(0.5, normalized=True)
        if abs(lane_dx) >= abs(lane_dy):
            return "north" if mid.y > boundary_entry.y else "south"
        else:
            return "east" if mid.x > boundary_entry.x else "west"

    options = {
        "option_1": {"arc_line": arc_fwd, "length_m": forward_len,
                     "side": side_label(arc_fwd),
                     "boundary_entry": boundary_entry, "boundary_exit": boundary_exit},
        "option_2": {"arc_line": arc_bwd, "length_m": backward_len,
                     "side": side_label(arc_bwd),
                     "boundary_entry": boundary_entry, "boundary_exit": boundary_exit},
    }
    lens = [forward_len, backward_len]
    similar_length = (min(lens) > 0) and (abs(lens[0] - lens[1]) / min(lens) < 0.15)
    return options, similar_length


def build_reroute_track(corridor_line: LineString, polygon, side: str = None,
                         n_frames: int = 150, direction: str = "auto"):
    """
    Straight approach -> arc around the polygon boundary -> straight
    departure.

    direction : which go-around path to use --
        'auto' / 'shortest' : the shorter of the two options (default)
        'option_1' / 'option_2' : explicit choice, see compute_reroute_options()
        'north' / 'south' / 'east' / 'west' : choice by compass side, see
            compute_reroute_options() for which label applies to your
            geometry (north/south for an east-west lane, east/west for a
            north-south lane)
    """
    options, _ = compute_reroute_options(corridor_line, polygon, side=side)

    if direction in ("auto", "shortest"):
        chosen_key = min(options, key=lambda k: options[k]["length_m"])
    elif direction in options:
        chosen_key = direction
    else:
        matches = [k for k, v in options.items() if v["side"] == direction]
        if not matches:
            available_sides = [v["side"] for v in options.values()]
            raise ValueError(
                f"direction={direction!r} not recognized. Use 'auto', 'shortest', "
                f"'option_1', 'option_2', or one of the available side labels "
                f"for this geometry: {available_sides}."
            )
        chosen_key = matches[0]

    opt = options[chosen_key]
    arc_line = opt["arc_line"]
    boundary_entry, boundary_exit = opt["boundary_entry"], opt["boundary_exit"]

    x0, y0, x1, y1 = _lane_endpoints(corridor_line, side)
    # Use arc_line's OWN coordinates directly rather than re-resampling it
    # at uniform arc-length distances: arc_line was already built with the
    # polygon's true corner vertices forced into the sample set (see
    # compute_reroute_options), and uniformly re-interpolating it here would
    # silently undo that -- straddling a corner between two new samples and
    # reintroducing the exact straight-chord cutting problem the vertex
    # forcing was meant to fix, just one resampling step removed.
    arc_coords = list(arc_line.coords)

    n_lead = max(int(n_frames * 0.2), 5)
    lead_xs = np.linspace(x0, boundary_entry.x, n_lead)
    lead_ys = np.linspace(y0, boundary_entry.y, n_lead)
    trail_xs = np.linspace(boundary_exit.x, x1, n_lead)
    trail_ys = np.linspace(boundary_exit.y, y1, n_lead)

    xs = np.concatenate([lead_xs, [c[0] for c in arc_coords], trail_xs])
    ys = np.concatenate([lead_ys, [c[1] for c in arc_coords], trail_ys])
    return xs, ys


def build_baseline_track_raw(corridor_line: LineString, side: str = None,
                              base_speed_knots: float = 12.0, n_spatial: int = 500):
    """Straight track at constant speed, with real cumulative time (s)."""
    x0, y0, x1, y1 = _lane_endpoints(corridor_line, side)
    xs = np.linspace(x0, x1, n_spatial)
    ys = np.linspace(y0, y1, n_spatial)
    speed = base_speed_knots * 0.514
    seg_dist = np.hypot(np.diff(xs), np.diff(ys))
    seg_time = seg_dist / speed
    cum_time = np.concatenate([[0], np.cumsum(seg_time)])
    return xs, ys, cum_time, cum_time[-1]


def build_speed_reduction_track_raw(corridor_line: LineString, polygon, side: str = None,
                                     base_speed_knots: float = 12.0, reduction: float = 0.5,
                                     n_spatial: int = 500):
    """Straight track, reduced speed inside polygon, with real cumulative time (s)."""
    x0, y0, x1, y1 = _lane_endpoints(corridor_line, side)
    xs = np.linspace(x0, x1, n_spatial)
    ys = np.linspace(y0, y1, n_spatial)

    inside = np.array([polygon.contains(Point(x, y)) for x, y in zip(xs, ys)])
    seg_dist = np.hypot(np.diff(xs), np.diff(ys))
    base_speed = base_speed_knots * 0.514
    seg_speed = np.full(len(seg_dist), base_speed)
    seg_inside = inside[:-1] | inside[1:]
    seg_speed[seg_inside] = base_speed * max(1e-3, 1 - reduction)

    seg_time = seg_dist / seg_speed
    cum_time = np.concatenate([[0], np.cumsum(seg_time)])
    return xs, ys, cum_time, cum_time[-1]


def build_reroute_track_raw(corridor_line: LineString, polygon, side: str = None,
                             base_speed_knots: float = 12.0, n_spatial: int = 500,
                             direction: str = "auto"):
    """Go-around track at constant speed, with real cumulative time (s)."""
    xs, ys = build_reroute_track(corridor_line, polygon, side=side, n_frames=n_spatial,
                                  direction=direction)
    speed = base_speed_knots * 0.514
    seg_dist = np.hypot(np.diff(xs), np.diff(ys))
    seg_time = seg_dist / speed
    cum_time = np.concatenate([[0], np.cumsum(seg_time)])
    return xs, ys, cum_time, cum_time[-1]


def build_scenario_track_raw(scenario: str, corridor_line: LineString, polygon, side: str,
                              base_speed_knots: float, reduction: float, n_spatial: int = 500,
                              reroute_direction: str = "auto"):
    if scenario == "baseline":
        return build_baseline_track_raw(corridor_line, side=side,
                                         base_speed_knots=base_speed_knots, n_spatial=n_spatial)
    elif scenario == "speed_reduction":
        return build_speed_reduction_track_raw(corridor_line, polygon, side=side,
                                                base_speed_knots=base_speed_knots,
                                                reduction=reduction, n_spatial=n_spatial)
    elif scenario == "reroute":
        return build_reroute_track_raw(corridor_line, polygon, side=side,
                                        base_speed_knots=base_speed_knots, n_spatial=n_spatial,
                                        direction=reroute_direction)
    else:
        raise ValueError("scenario must be 'baseline', 'speed_reduction', or 'reroute'")


def resample_common_timeline(tracks_raw: dict, n_frames: int):
    """
    Put multiple scenario tracks (each a (xs, ys, cum_time, total_time) tuple
    from the *_raw builders) onto a SHARED time axis, so scenarios that take
    longer keep moving while faster ones have already arrived and sit still
    at their endpoint. This is what makes a side-by-side comparison
    animation actually represent relative transit time correctly, rather
    than just stretching every path to the same frame count regardless of
    how long it really takes.

    Returns
    -------
    frame_times : (n_frames,) array of shared elapsed time (s)
    resampled : dict[name] -> (xs_frames, ys_frames, elapsed_time_frames, total_time)
    """
    max_time = max(t[3] for t in tracks_raw.values())
    frame_times = np.linspace(0, max_time, n_frames)
    resampled = {}
    for name, (xs, ys, cum_time, total_time) in tracks_raw.items():
        fx = np.interp(frame_times, cum_time, xs)
        fy = np.interp(frame_times, cum_time, ys)
        elapsed = np.minimum(frame_times, total_time)
        resampled[name] = (fx, fy, elapsed, total_time)
    return frame_times, resampled
