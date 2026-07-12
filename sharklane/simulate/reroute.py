"""
Rerouting simulation. Two modes:

  - 'perimeter' : Womersley et al. 2024's original method -- reroute along
    the nearest point on the risk polygon's boundary between original entry
    and exit points. Fine for open water where any detour is geometrically
    feasible.

  - 'least_cost' : land-mask-constrained shortest path (needed for
    bay-mouth / coastline-constrained sites, where a straight perimeter
    detour could cross land). Uses skimage's MCP_Geometric over the
    WaterMask as a cost surface.

Only vessels that both enter AND exit the polygon during their track are
eligible for rerouting -- vessels whose origin/destination is inside the
polygon (e.g. entering the bay itself) cannot be "rerouted around" it;
flag these separately (see `classify_transit_vs_terminal`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points


def classify_transit_vs_terminal(tracks: dict[str, gpd.GeoDataFrame],
                                  polygon) -> dict[str, str]:
    """
    Classify each vessel's relationship to the risk polygon:
      'transit'  -- track passes through but starts and ends outside
      'terminal' -- track starts or ends inside (i.e. genuinely entering/
                    leaving the habitat/bay; cannot be rerouted around)
      'inside_only' -- entire (filtered) track is inside the polygon
    """
    labels = {}
    for vid, track in tracks.items():
        if len(track) == 0:
            continue
        first_in = polygon.contains(track.geometry.iloc[0])
        last_in = polygon.contains(track.geometry.iloc[-1])
        any_in = track.geometry.apply(polygon.contains).any()
        if first_in and last_in:
            labels[vid] = "inside_only"
        elif first_in or last_in:
            labels[vid] = "terminal"
        elif any_in:
            labels[vid] = "transit"
        else:
            labels[vid] = "no_overlap"
    return labels


def _entry_exit_points(track: gpd.GeoDataFrame, polygon):
    """Find the last point before entering and first point after exiting
    the polygon along a transiting track."""
    inside = track.geometry.apply(polygon.contains).to_numpy()
    if not inside.any():
        return None, None
    first_idx = np.argmax(inside)
    last_idx = len(inside) - 1 - np.argmax(inside[::-1])
    entry_pt = track.geometry.iloc[max(first_idx - 1, 0)]
    exit_pt = track.geometry.iloc[min(last_idx + 1, len(track) - 1)]
    return entry_pt, exit_pt


def reroute_perimeter(tracks: dict[str, gpd.GeoDataFrame], polygon) -> pd.DataFrame:
    """Simple open-water reroute: shortest path along the polygon boundary
    between entry/exit points (Womersley et al. 2024's original method)."""
    from shapely.geometry import LineString

    boundary = polygon.exterior
    perimeter = boundary.length
    rows = []
    for vid, track in tracks.items():
        entry_pt, exit_pt = _entry_exit_points(track, polygon)
        if entry_pt is None:
            continue
        # project entry/exit onto boundary, take shorter arc between them
        d_entry = boundary.project(entry_pt)
        d_exit = boundary.project(exit_pt)

        # sample the boundary between d_entry and d_exit going both ways;
        # keep whichever direction is shorter
        forward_len = (d_exit - d_entry) % perimeter
        backward_len = (d_entry - d_exit) % perimeter
        if forward_len <= backward_len:
            arc_len = forward_len
            distances = np.linspace(d_entry, d_entry + arc_len, 50) % perimeter
        else:
            arc_len = backward_len
            distances = np.linspace(d_entry, d_entry - arc_len, 50) % perimeter

        arc_points = [boundary.interpolate(d) for d in distances]
        path_line = LineString([entry_pt] + [(p.x, p.y) for p in arc_points] + [exit_pt])

        straight_dist = entry_pt.distance(exit_pt)
        extra_dist = arc_len - straight_dist

        rows.append({
            "vessel_id": vid,
            "method": "perimeter",
            "entry": entry_pt,
            "exit": exit_pt,
            "path_line": path_line,
            "original_dist_m": straight_dist,
            "reroute_dist_m": arc_len,
            "extra_dist_m": max(extra_dist, 0),
        })
    return pd.DataFrame(rows)


def reroute_least_cost(tracks: dict[str, gpd.GeoDataFrame], polygon,
                        water_mask, avg_speed_lookup: dict[str, float] = None
                        ) -> pd.DataFrame:
    """
    Land-mask-constrained reroute using a least-cost path over the water
    mask (cost=1 for water, effectively infinite for land). Use this
    instead of reroute_perimeter() whenever the site has constraining
    coastline geometry (e.g. a bay mouth) where a straight perimeter detour
    might cross land.

    Requires scikit-image (skimage.graph.MCP_Geometric).
    """
    from skimage.graph import MCP_Geometric

    # cost surface: water cells = 1, land cells = very high (not inf, to
    # avoid numerical issues, but effectively impassable)
    costs = np.where(water_mask.mask == 1, 1.0, 1e6).astype("float64")
    mcp = MCP_Geometric(costs, fully_connected=True)

    rows = []
    for vid, track in tracks.items():
        entry_pt, exit_pt = _entry_exit_points(track, polygon)
        if entry_pt is None:
            continue

        try:
            r0, c0 = water_mask.nearest_water_cell(entry_pt.x, entry_pt.y)
            r1, c1 = water_mask.nearest_water_cell(exit_pt.x, exit_pt.y)
        except ValueError:
            continue

        # mask out the polygon interior so the path is forced around it
        from rasterio import features
        poly_mask = features.rasterize(
            [(polygon, 1)],
            out_shape=water_mask.shape,
            transform=water_mask.transform,
            fill=0,
            dtype="uint8",
        )
        costs_masked = costs.copy()
        costs_masked[poly_mask == 1] = 1e6
        mcp_local = MCP_Geometric(costs_masked, fully_connected=True)

        cumulative_costs, _ = mcp_local.find_costs([(r0, c0)], [(r1, c1)])
        path_cost = cumulative_costs[r1, c1]
        if not np.isfinite(path_cost) or path_cost >= 1e6:
            rows.append({
                "vessel_id": vid, "method": "least_cost",
                "entry": entry_pt, "exit": exit_pt,
                "original_dist_m": entry_pt.distance(exit_pt),
                "reroute_dist_m": np.nan, "extra_dist_m": np.nan,
                "note": "no feasible water path found",
            })
            continue

        path = mcp_local.traceback((r1, c1))
        px_size = abs(water_mask.transform.a)
        reroute_dist = path_cost * px_size

        path_xy = [water_mask.xy(r, c) for r, c in path]
        path_line = LineString(path_xy)

        straight_dist = entry_pt.distance(exit_pt)
        rows.append({
            "vessel_id": vid,
            "method": "least_cost",
            "entry": entry_pt,
            "exit": exit_pt,
            "path_line": path_line,
            "original_dist_m": straight_dist,
            "reroute_dist_m": reroute_dist,
            "extra_dist_m": reroute_dist - straight_dist,
            "path_rowcol": path,
        })
    return pd.DataFrame(rows)


def estimate_reroute_time(reroute_df: pd.DataFrame, tracks: dict,
                           polygon) -> pd.DataFrame:
    """
    Convert extra distance into extra time, using each vessel's average
    speed just before entry / after exit (as in the paper), and compute
    percent increase in total transit time.
    """
    rows = []
    for _, row in reroute_df.iterrows():
        vid = row["vessel_id"]
        track = tracks[vid]
        baseline_time = np.nansum(track["seg_dt_s"].to_numpy()[1:])
        baseline_dist = np.nansum(track["seg_dist_m"].to_numpy()[1:])
        if baseline_time == 0 or pd.isna(row.get("extra_dist_m", np.nan)):
            continue

        avg_speed = baseline_dist / baseline_time if baseline_time > 0 else np.nan
        extra_time = row["extra_dist_m"] / avg_speed if avg_speed else np.nan
        new_time = baseline_time + extra_time
        pct_increase = 100 * extra_time / baseline_time if baseline_time else np.nan

        rows.append({
            "vessel_id": vid,
            "baseline_time_s": baseline_time,
            "extra_dist_m": row["extra_dist_m"],
            "extra_time_s": extra_time,
            "new_time_s": new_time,
            "pct_increase": pct_increase,
        })
    return pd.DataFrame(rows)
