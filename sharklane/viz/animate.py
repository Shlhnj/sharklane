"""Animate a vessel's original track vs. a mitigated (rerouted) path."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import geopandas as gpd


def animate_vessel_comparison(original_track, polygon, water_mask=None,
                               reroute_path_xy: list[tuple] = None,
                               out_path: str = "vessel_comparison.gif",
                               fps: int = 10):
    """
    Animate a single vessel's original transit alongside its rerouted path
    (if provided), moving through the scene over time.

    Parameters
    ----------
    original_track : GeoDataFrame of the vessel's cleaned track (ordered).
    polygon : risk polygon geometry to draw for reference.
    water_mask : optional WaterMask, drawn as background.
    reroute_path_xy : optional list of (x, y) coordinates for the
        alternative/rerouted path, animated alongside the original.
    out_path : filename to save the animation to (.gif or .mp4).
    fps : frames per second.
    """
    xs = original_track.geometry.x.to_numpy()
    ys = original_track.geometry.y.to_numpy()
    n_frames = len(xs)

    fig, ax = plt.subplots(figsize=(8, 8))

    if water_mask is not None:
        extent = [water_mask.bounds[0], water_mask.bounds[2],
                  water_mask.bounds[1], water_mask.bounds[3]]
        ax.imshow(water_mask.mask, extent=extent, origin="upper",
                  cmap="Blues_r", alpha=0.3, zorder=0)

    gpd.GeoSeries([polygon]).plot(ax=ax, facecolor="none", edgecolor="crimson",
                                   linewidth=2, zorder=1)

    ax.set_xlim(xs.min() - 0.05 * (xs.max() - xs.min()), xs.max() + 0.05 * (xs.max() - xs.min()))
    ax.set_ylim(ys.min() - 0.05 * (ys.max() - ys.min()), ys.max() + 0.05 * (ys.max() - ys.min()))
    ax.set_aspect("equal")

    orig_line, = ax.plot([], [], color="steelblue", linewidth=1.5, label="Original")
    orig_dot, = ax.plot([], [], "o", color="steelblue", markersize=6)

    if reroute_path_xy is not None:
        rxs = np.array([p[0] for p in reroute_path_xy])
        rys = np.array([p[1] for p in reroute_path_xy])
        reroute_line, = ax.plot([], [], color="darkorange", linewidth=1.5, label="Rerouted")
        reroute_dot, = ax.plot([], [], "o", color="darkorange", markersize=6)
        n_frames = max(n_frames, len(rxs))

    ax.legend(loc="upper right")
    ax.set_title("Vessel transit: original vs. mitigated path")

    def update(frame):
        i = min(frame, len(xs) - 1)
        orig_line.set_data(xs[:i + 1], ys[:i + 1])
        orig_dot.set_data([xs[i]], [ys[i]])
        artists = [orig_line, orig_dot]
        if reroute_path_xy is not None:
            j = min(frame, len(rxs) - 1)
            reroute_line.set_data(rxs[:j + 1], rys[:j + 1])
            reroute_dot.set_data([rxs[j]], [rys[j]])
            artists += [reroute_line, reroute_dot]
        return artists

    anim = animation.FuncAnimation(fig, update, frames=n_frames, blit=True)

    if out_path.endswith(".gif"):
        anim.save(out_path, writer=animation.PillowWriter(fps=fps))
    else:
        anim.save(out_path, writer=animation.FFMpegWriter(fps=fps))
    plt.close(fig)
    return out_path
