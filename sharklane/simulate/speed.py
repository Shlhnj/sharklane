"""
Speed reduction simulation: applies a percentage speed reduction only to the
portion of each vessel's track that intersects the risk polygon (core
habitat or corridor), and recomputes total transit time vs. baseline.
Mirrors Womersley et al. 2024's method (10-75% reductions, 1% increments).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


def _segment_in_polygon(track: gpd.GeoDataFrame, polygon) -> np.ndarray:
    """Boolean array: True where a track segment (between point i and i+1)
    falls at least partly inside the risk polygon, based on the segment's
    two endpoints."""
    pts_in = track.geometry.apply(lambda p: polygon.contains(p) or polygon.touches(p))
    pts_in = pts_in.to_numpy()
    # a segment counts as "inside" if either endpoint is inside
    seg_in = pts_in[:-1] | pts_in[1:]
    return seg_in


def baseline_transit(track: gpd.GeoDataFrame) -> dict:
    """Total time (s) and distance (m) for the cleaned, unmodified track."""
    total_time = np.nansum(track["seg_dt_s"].to_numpy()[1:])
    total_dist = np.nansum(track["seg_dist_m"].to_numpy()[1:])
    return {"total_time_s": total_time, "total_dist_m": total_dist}


def simulate_speed_reduction(tracks: dict[str, gpd.GeoDataFrame],
                              polygon,
                              reductions: list[float] = None) -> pd.DataFrame:
    """
    Parameters
    ----------
    tracks : dict of vessel_id -> cleaned track GeoDataFrame (must include
        seg_dt_s, seg_dist_m columns from sharklane.ais.compute_speeds).
    polygon : shapely polygon (projected CRS matching the tracks) defining
        the risk zone (core habitat or bay-mouth corridor).
    reductions : list of fractional reductions to test, e.g. [0.1, 0.11, ...,
        0.75]. Defaults to 10-75% at 1% increments, as in the source paper.

    Returns
    -------
    DataFrame with one row per (vessel_id, reduction) giving baseline time,
    new time, and percent increase.
    """
    if reductions is None:
        reductions = [r / 100 for r in range(10, 76)]

    rows = []
    for vid, track in tracks.items():
        if len(track) < 2:
            continue
        seg_in = _segment_in_polygon(track, polygon)
        dt = track["seg_dt_s"].to_numpy()[1:]
        dist = track["seg_dist_m"].to_numpy()[1:]
        baseline_time = np.nansum(dt)
        if baseline_time == 0 or np.isnan(baseline_time):
            continue

        for red in reductions:
            # time for in-polygon segments increases by 1/(1-red); other
            # segments are unaffected.
            new_dt = dt.copy()
            factor = 1.0 / (1.0 - red)
            new_dt[seg_in] = new_dt[seg_in] * factor
            new_time = np.nansum(new_dt)
            pct_increase = 100 * (new_time - baseline_time) / baseline_time

            rows.append({
                "vessel_id": vid,
                "reduction": red,
                "baseline_time_s": baseline_time,
                "new_time_s": new_time,
                "extra_time_s": new_time - baseline_time,
                "pct_increase": pct_increase,
            })

    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Mean/median percent increase and extra hours per vessel, by reduction
    level -- the headline numbers reported in the paper (e.g. '75% speed
    reduction -> ~5% transit time increase, 69.6 extra hours/vessel')."""
    g = results.groupby("reduction").agg(
        mean_pct_increase=("pct_increase", "mean"),
        median_pct_increase=("pct_increase", "median"),
        mean_extra_hours=("extra_time_s", lambda x: x.mean() / 3600),
        n_vessels=("vessel_id", "nunique"),
    ).reset_index()
    return g
