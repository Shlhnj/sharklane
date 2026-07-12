"""Static summary graphics: site map, speed-reduction curve, reroute paths."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd


def plot_site_map(polygon, tracks: dict, water_mask=None, corridor_line=None,
                   title: str = "Site overview", ax=None):
    """Overview map: risk polygon, water mask (optional), vessel tracks,
    and the transit/corridor line (optional)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 9))

    if water_mask is not None:
        extent = [water_mask.bounds[0], water_mask.bounds[2],
                  water_mask.bounds[1], water_mask.bounds[3]]
        ax.imshow(water_mask.mask, extent=extent, origin="upper",
                  cmap="Blues_r", alpha=0.3, zorder=0)

    for vid, track in tracks.items():
        xs, ys = track.geometry.x, track.geometry.y
        ax.plot(xs, ys, linewidth=0.5, alpha=0.5, color="steelblue", zorder=1)

    gpd.GeoSeries([polygon]).plot(ax=ax, facecolor="none", edgecolor="crimson",
                                   linewidth=2, zorder=3)

    if corridor_line is not None:
        gpd.GeoSeries([corridor_line]).plot(ax=ax, color="orange",
                                             linewidth=2, linestyle="--", zorder=3)

    ax.set_title(title)
    ax.set_aspect("equal")
    return ax


def plot_speed_reduction_curve(summary_df, ax=None):
    """Percent transit-time increase vs. speed reduction (%) -- the
    headline chart from the paper's Fig. 5c."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(summary_df["reduction"] * 100, summary_df["mean_pct_increase"],
            marker="o", markersize=3, color="darkred")
    ax.set_xlabel("Speed reduction (%)")
    ax.set_ylabel("Mean increase in total transit time (%)")
    ax.set_title("Speed reduction impact on transit time")
    ax.grid(alpha=0.3)
    return ax


def plot_reroute_paths(reroute_df, polygon, water_mask=None, tracks=None, ax=None,
                        max_paths_labeled=1):
    """Plot the actual rerouted paths (around the risk polygon perimeter),
    entry/exit points, and optionally the vessels' original tracks for
    comparison. Handles the case where reroute_df is empty (no transit
    vessels found -- e.g. no vessel track actually entered/exited the risk
    polygon in this dataset) by showing the polygon/tracks with a note,
    rather than crashing on the missing columns an empty DataFrame has."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 9))

    if water_mask is not None:
        extent = [water_mask.bounds[0], water_mask.bounds[2],
                  water_mask.bounds[1], water_mask.bounds[3]]
        ax.imshow(water_mask.mask, extent=extent, origin="upper",
                  cmap="Blues_r", alpha=0.3, zorder=0)

    has_vessel_id = len(reroute_df) > 0 and "vessel_id" in reroute_df.columns

    if tracks is not None and has_vessel_id:
        for vid in reroute_df["vessel_id"]:
            if vid in tracks:
                t = tracks[vid]
                ax.plot(t.geometry.x, t.geometry.y, color="steelblue",
                         linewidth=0.7, alpha=0.6, zorder=1,
                         label="Original track" if vid == reroute_df["vessel_id"].iloc[0] else None)

    gpd.GeoSeries([polygon]).plot(ax=ax, facecolor="none", edgecolor="crimson",
                                   linewidth=2, zorder=2, label="Risk polygon")

    if not has_vessel_id:
        ax.text(0.5, 0.5, "No rerouted vessels to show\n(no transit vessels found)",
                transform=ax.transAxes, ha="center", va="center", fontsize=10,
                color="gray")
        ax.set_title("Rerouted vessel paths around the risk polygon")
        ax.set_aspect("equal")
        ax.legend(loc="best", fontsize=8)
        return ax

    has_path_line = "path_line" in reroute_df.columns
    for i, row in reroute_df.iterrows():
        entry, exit_ = row["entry"], row["exit"]
        if has_path_line and row.get("path_line") is not None:
            xs, ys = row["path_line"].xy
            ax.plot(xs, ys, color="darkorange", linewidth=1.3, zorder=3,
                     label="Rerouted path" if i == 0 else None)
        else:
            # fall back to a straight connector if no path geometry is available
            ax.plot([entry.x, exit_.x], [entry.y, exit_.y],
                    color="gray", linestyle=":", linewidth=0.8, zorder=1)
        ax.scatter([entry.x, exit_.x], [entry.y, exit_.y],
                   color="green", s=12, zorder=4)

    ax.set_title("Rerouted vessel paths around the risk polygon")
    ax.set_aspect("equal")
    ax.legend(loc="best", fontsize=8)
    return ax


def plot_lane_shift_feasibility(shift_results, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    colors = shift_results["feasible"].map({True: "green", False: "red"})
    ax.bar(shift_results["offset_m"] / 1000, shift_results["feasible"].astype(int),
           color=colors, width=0.4)
    ax.set_xlabel("Lateral shift offset (km)")
    ax.set_ylabel("Feasible (1) / Not feasible (0)")
    ax.set_title("Lane shift feasibility by offset distance")
    return ax
