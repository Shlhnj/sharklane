"""
Shipping lane shift simulation: for vessels travelling at high, consistent
speed through the risk polygon (treated as using a fixed transit lane
rather than a discretionary route -- Womersley et al. 2024 used >15 knots),
test shifting that lane laterally by increasing offsets until it clears
the polygon, checking feasibility against the water mask at each offset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
from shapely.affinity import translate


def identify_lane_vessels(tracks: dict[str, gpd.GeoDataFrame],
                           speed_threshold_mps: float = 7.7) -> list[str]:
    """Vessels whose mean speed while inside/near the risk area exceeds the
    threshold (paper used 15 knots = 7.7 m/s) -- treated as fixed-lane
    traffic rather than vessels with a bay-specific destination."""
    lane_vessels = []
    for vid, track in tracks.items():
        mean_speed = np.nanmean(track["speed_mps"])
        if mean_speed is not None and mean_speed >= speed_threshold_mps:
            lane_vessels.append(vid)
    return lane_vessels


def representative_lane(tracks: dict[str, gpd.GeoDataFrame],
                         lane_vessel_ids: list[str], polygon) -> LineString:
    """Build a single representative lane line by averaging the tracks of
    identified lane vessels within a buffer of the polygon. Simple
    approach: take the track with the most points near the polygon and use
    it as the representative lane; refine as needed for your site."""
    candidates = [tracks[v] for v in lane_vessel_ids if v in tracks]
    if not candidates:
        raise ValueError("No lane vessels found -- lower speed_threshold or "
                          "check track data.")
    best = max(candidates, key=len)
    coords = list(zip(best.geometry.x, best.geometry.y))
    return LineString(coords)


def test_lane_shifts(lane: LineString, polygon, water_mask,
                      offsets_m: list[float] = None,
                      direction: tuple[float, float] = (0, -1)) -> pd.DataFrame:
    """
    Shift the representative lane by increasing offsets in a given
    direction (default: south, i.e. (0,-1) -- change per site) and, for
    each offset, check:
      (a) whether the shifted lane still intersects the risk polygon
      (b) whether the entire shifted lane remains in water

    Returns a DataFrame so you can find the minimum feasible offset that
    clears the polygon -- report this as the "lane shift" recommendation,
    or report that none is geometrically feasible (e.g. narrow bay mouth).
    """
    if offsets_m is None:
        offsets_m = list(range(0, 25000, 500))  # 0-25km in 500m steps; tune to site

    dx, dy = direction
    norm = (dx ** 2 + dy ** 2) ** 0.5
    dx, dy = dx / norm, dy / norm

    rows = []
    for offset in offsets_m:
        shifted = translate(lane, xoff=dx * offset, yoff=dy * offset)
        clears_polygon = not shifted.intersects(polygon)

        # sample points along shifted line, check all are in water
        n_samples = 50
        in_water = True
        for i in range(n_samples + 1):
            pt = shifted.interpolate(i / n_samples, normalized=True)
            if not water_mask.is_water(pt.x, pt.y):
                in_water = False
                break

        rows.append({
            "offset_m": offset,
            "clears_polygon": clears_polygon,
            "fully_in_water": in_water,
            "feasible": clears_polygon and in_water,
        })

    return pd.DataFrame(rows)


def minimum_feasible_shift(shift_results: pd.DataFrame) -> float | None:
    """Smallest offset (m) that both clears the polygon and stays in
    water. Returns None if no tested offset satisfies both -- meaning a
    lane shift is not geometrically viable at this site (a valid finding,
    e.g. for a narrow bay mouth) and speed reduction / rerouting should be
    reported as the primary mitigation levers instead."""
    feasible = shift_results[shift_results["feasible"]]
    if feasible.empty:
        return None
    return float(feasible["offset_m"].min())
