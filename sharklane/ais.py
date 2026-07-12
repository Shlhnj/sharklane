"""
AIS track loading and cleaning, following the filtering approach used in
Womersley et al. 2024 (Sci. Total Environ. 934:172776):
  - drop implausible speeds (> max_speed or above a high percentile)
  - drop vessels that are mostly stationary
  - drop vessels with too few positions
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString


REQUIRED_COLS = ("vessel_id", "timestamp", "lon", "lat")


def load_ais_csv(path: str, crs: str = "EPSG:4326", **read_csv_kwargs) -> gpd.GeoDataFrame:
    """Load raw AIS positions from CSV. Expects columns vessel_id, timestamp,
    lon, lat (rename beforehand if your source uses different names)."""
    df = pd.read_csv(path, **read_csv_kwargs)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"AIS input missing required columns: {missing}")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs=crs
    )
    return gdf.sort_values(["vessel_id", "timestamp"]).reset_index(drop=True)


def compute_speeds(gdf: gpd.GeoDataFrame, crs_metric: str) -> gpd.GeoDataFrame:
    """Estimate instantaneous speed (m/s) between consecutive positions per
    vessel, using a projected (metric) CRS for accurate distances."""
    proj = gdf.to_crs(crs_metric)
    gdf = gdf.copy()
    gdf["_x"] = proj.geometry.x
    gdf["_y"] = proj.geometry.y

    speeds = np.full(len(gdf), np.nan)
    seg_dist = np.full(len(gdf), np.nan)
    seg_dt = np.full(len(gdf), np.nan)

    for vid, sub in gdf.groupby("vessel_id"):
        idx = sub.index.to_numpy()
        if len(idx) < 2:
            continue
        dx = np.diff(sub["_x"].to_numpy())
        dy = np.diff(sub["_y"].to_numpy())
        dist = np.hypot(dx, dy)
        # NOTE: do not use sub["timestamp"].astype("int64") / 1e9 here --
        # that assumes datetime64[ns] internal resolution. pandas 2.x/3.x
        # can store datetime64 at us/ms/s resolution depending on how the
        # column was constructed, which silently changes what astype("int64")
        # returns (e.g. microseconds instead of nanoseconds), giving
        # per-segment times that are wrong by a fixed factor (seen: 1000x
        # too small under pandas 3.0.2). dt.total_seconds() is resolution-
        # agnostic and always correct.
        dt = sub["timestamp"].diff().dt.total_seconds().to_numpy()[1:].copy()
        dt[dt == 0] = np.nan
        spd = dist / dt
        speeds[idx[1:]] = spd
        seg_dist[idx[1:]] = dist
        seg_dt[idx[1:]] = dt

    gdf["speed_mps"] = speeds
    gdf["seg_dist_m"] = seg_dist
    gdf["seg_dt_s"] = seg_dt
    return gdf.drop(columns=["_x", "_y"])


def clean_tracks(gdf: gpd.GeoDataFrame,
                  crs_metric: str,
                  max_speed_mps: float = 50.0,
                  speed_percentile: float = 99.0,
                  stationary_thresh_mps: float = 0.01,
                  stationary_frac_limit: float = 0.25,
                  min_points: int = 20) -> gpd.GeoDataFrame:
    """
    Filter AIS positions following the paper's rules:
      - drop positions faster than max_speed_mps or above the given
        percentile (erroneous signals)
      - drop entire vessels where >stationary_frac_limit of positions are
        below stationary_thresh_mps (anchored/loitering vessels)
      - drop vessels with fewer than min_points remaining positions
    """
    gdf = compute_speeds(gdf, crs_metric)

    pctile_cutoff = np.nanpercentile(gdf["speed_mps"].dropna(), speed_percentile)
    speed_cutoff = min(max_speed_mps, pctile_cutoff)
    gdf = gdf[(gdf["speed_mps"].isna()) | (gdf["speed_mps"] <= speed_cutoff)].copy()

    keep_vessels = []
    for vid, sub in gdf.groupby("vessel_id"):
        n = len(sub)
        if n < min_points:
            continue
        stationary_frac = (sub["speed_mps"] < stationary_thresh_mps).mean()
        if stationary_frac > stationary_frac_limit:
            continue
        keep_vessels.append(vid)

    return gdf[gdf["vessel_id"].isin(keep_vessels)].reset_index(drop=True)


def to_tracks(gdf: gpd.GeoDataFrame) -> dict[str, gpd.GeoDataFrame]:
    """Split a cleaned point GeoDataFrame into per-vessel ordered tracks."""
    return {vid: sub.sort_values("timestamp").reset_index(drop=True)
            for vid, sub in gdf.groupby("vessel_id")}


def track_to_linestring(track: gpd.GeoDataFrame) -> LineString:
    coords = list(zip(track.geometry.x, track.geometry.y))
    return LineString(coords)
