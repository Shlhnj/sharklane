"""Animate schematic ship icons through the three transit scenarios:
baseline, speed-reduced, and reroute-around. Builds on sharklane.viz.ship
for the icon geometry and shared-timeline track generation.
"""
from __future__ import annotations

import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon as MplPolygon
import geopandas as gpd

from .ship import ship_polygon, build_scenario_track_raw, resample_common_timeline

_DEFAULT_SCENARIO_COLORS = {
    "baseline": "steelblue",
    "speed_reduction": "darkorange",
    "reroute": "seagreen",
}
_SCENARIO_LABELS = {
    "baseline": "Baseline (no mitigation)",
    "speed_reduction": "Speed reduction",
    "reroute": "Reroute around",
}

_BACKGROUND_STYLES = {
    "light": dict(fig_facecolor="white", ax_facecolor="white", water_cmap="Blues_r",
                  water_alpha=0.3, polygon_color="crimson", text_color="black"),
    "dark":  dict(fig_facecolor="#0b0f1a", ax_facecolor="#0b0f1a", water_cmap="Blues",
                  water_alpha=0.55, polygon_color="gold", text_color="white"),
}


def _resolve_ship_colors(ship_color, scenarios):
    """Accept either a single color (applied to all scenarios) or a dict
    keyed by scenario name; falls back to sensible defaults."""
    if ship_color is None:
        return {s: _DEFAULT_SCENARIO_COLORS[s] for s in scenarios}
    if isinstance(ship_color, dict):
        return {s: ship_color.get(s, _DEFAULT_SCENARIO_COLORS[s]) for s in scenarios}
    return {s: ship_color for s in scenarios}


def _compute_bounds(polygon, corridor_line, tracks_xy: list, water_mask, bounds, pad_frac=0.08):
    """Extent covering the FULL protected area (polygon), the corridor
    line, and all animated tracks -- not just the tracks alone (which was
    the earlier bug: a reroute that only dips south, for example, would
    clip the polygon's northern edge out of frame)."""
    if bounds is not None:
        return bounds

    xs_all, ys_all = [], []
    minx, miny, maxx, maxy = polygon.bounds
    xs_all += [minx, maxx]
    ys_all += [miny, maxy]

    if corridor_line is not None:
        cminx, cminy, cmaxx, cmaxy = corridor_line.bounds
        xs_all += [cminx, cmaxx]
        ys_all += [cminy, cmaxy]

    if water_mask is not None:
        xs_all += [water_mask.bounds[0], water_mask.bounds[2]]
        ys_all += [water_mask.bounds[1], water_mask.bounds[3]]

    for xs, ys in tracks_xy:
        xs_all += [np.min(xs), np.max(xs)]
        ys_all += [np.min(ys), np.max(ys)]

    xs_all, ys_all = np.array(xs_all), np.array(ys_all)
    pad_x = pad_frac * (xs_all.max() - xs_all.min())
    pad_y = pad_frac * (ys_all.max() - ys_all.min())
    return (xs_all.min() - pad_x, ys_all.min() - pad_y,
            xs_all.max() + pad_x, ys_all.max() + pad_y)


def _draw_background(ax, style, water_mask, bounds, raster_path, raster_band,
                      raster_cmap, working_crs):
    """Draw the base map layer under everything else. style is one of:
    'light', 'dark', 'raster' (needs raster_path -- e.g. bathymetry or any
    other continuous raster), or 'satellite' (needs internet access to a
    tile server via contextily -- works on a normal internet-connected
    machine, not guaranteed in restricted/offline environments)."""
    if style in ("light", "dark"):
        cfg = _BACKGROUND_STYLES[style]
        ax.set_facecolor(cfg["ax_facecolor"])
        if water_mask is not None:
            extent = [water_mask.bounds[0], water_mask.bounds[2],
                      water_mask.bounds[1], water_mask.bounds[3]]
            ax.imshow(water_mask.mask, extent=extent, origin="upper",
                      cmap=cfg["water_cmap"], alpha=cfg["water_alpha"], zorder=0)
        return cfg

    elif style == "raster":
        if raster_path is None:
            raise ValueError("background='raster' requires background_raster_path.")
        import rasterio
        with rasterio.open(raster_path) as src:
            arr = src.read(raster_band)
            b = src.bounds
        ax.imshow(arr, extent=[b.left, b.right, b.bottom, b.top], origin="upper",
                  cmap=raster_cmap, zorder=0)
        return dict(fig_facecolor="white", ax_facecolor="white",
                    polygon_color="crimson", text_color="black")

    elif style == "satellite":
        try:
            import contextily as cx
            cx.add_basemap(ax, crs=working_crs, source=cx.providers.Esri.WorldImagery,
                            zorder=0)
        except Exception as e:
            warnings.warn(
                f"background='satellite' failed ({e!r}) -- this needs internet "
                "access to a tile server (contextily), which may not be available "
                "in a restricted/offline environment. Falling back to 'light'."
            )
            return _draw_background(ax, "light", water_mask, bounds, raster_path,
                                     raster_band, raster_cmap, working_crs)
        return dict(fig_facecolor="white", ax_facecolor="white",
                    polygon_color="yellow", text_color="black")

    else:
        raise ValueError("background must be 'light', 'dark', 'raster', or 'satellite'")


def _setup_figure(polygon, corridor_line, water_mask, tracks_xy, bounds,
                   background, background_raster_path, background_band,
                   background_cmap, working_crs, lane_color, lane_width,
                   title, show_time_chart, figsize):
    if show_time_chart:
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=figsize,
                                       gridspec_kw={"width_ratios": [2, 1]})
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax2 = None

    style_cfg = _draw_background(ax, background, water_mask, bounds,
                                  background_raster_path, background_band,
                                  background_cmap, working_crs)
    fig.patch.set_facecolor(style_cfg["fig_facecolor"])
    if ax2 is not None:
        ax2.set_facecolor(style_cfg["fig_facecolor"])

    gpd.GeoSeries([polygon]).plot(ax=ax, facecolor="none",
                                   edgecolor=style_cfg["polygon_color"],
                                   linewidth=2, zorder=2)
    if corridor_line is not None:
        gpd.GeoSeries([corridor_line]).plot(ax=ax, color=lane_color, linewidth=lane_width,
                                             linestyle="--", zorder=1)

    bounds = _compute_bounds(polygon, corridor_line, tracks_xy, water_mask, bounds)
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_aspect("equal")
    ax.set_title(title, color=style_cfg["text_color"])
    ax.tick_params(colors=style_cfg["text_color"])
    for spine in ax.spines.values():
        spine.set_color(style_cfg["text_color"])

    return fig, ax, ax2, style_cfg


def animate_transit(polygon, corridor_line, scenario: str = "baseline",
                     side: str = None, water_mask=None,
                     base_speed_knots: float = 12.0, reduction: float = 0.5,
                     n_frames: int = 150, out_path: str = "transit.gif",
                     fps: int = 15,
                     # appearance
                     ship_scale: float = 0.03, ship_length: float = None,
                     ship_width: float = None, ship_color=None,
                     lane_color: str = "gray", lane_width: float = 1.0,
                     background: str = "light", background_raster_path: str = None,
                     background_band: int = 1, background_cmap: str = "Blues",
                     working_crs: str = None,
                     bounds: tuple = None, show_time_chart: bool = False,
                     figsize: tuple = (9, 9), reroute_direction: str = "auto"):
    """
    Animate a single schematic ship transiting from one side of the lane
    through (or around) the risk polygon, under one scenario.

    scenario : 'baseline', 'speed_reduction', or 'reroute'
    side : which end of the corridor line to start from -- 'west'/'east'
        for a roughly east-west lane, 'south'/'north' for a roughly
        north-south lane. If None, defaults to whichever the lane's own
        orientation suggests (see sharklane.viz.ship.get_default_side()).
        Use sharklane.viz.ship.get_valid_sides(corridor_line) to check
        which two labels actually apply before picking one explicitly.
    ship_scale : ship length as a fraction of the polygon's bounding-box
        diagonal -- ignored if ship_length is given directly (in metres).
    ship_color : a single color, or a dict {scenario: color}.
    background : 'light' (default), 'dark', 'raster' (needs
        background_raster_path), or 'satellite' (needs internet access).
    bounds : optional (minx, miny, maxx, maxy) to override the automatic
        extent (which otherwise covers the full polygon + corridor line +
        track, so the whole protected area stays in frame).
    show_time_chart : if True, adds a right-hand panel with a live bar
        showing elapsed transit time as the ship moves.
    reroute_direction : only used when scenario='reroute' -- 'auto'
        (shortest, default), 'option_1'/'option_2', or a compass label
        ('north'/'south'/'east'/'west') to pick a specific go-around path.
        See sharklane.viz.ship.compute_reroute_options() to inspect both
        options (length, side) before choosing.
    """
    raw = {scenario: build_scenario_track_raw(scenario, corridor_line, polygon, side,
                                               base_speed_knots, reduction,
                                               reroute_direction=reroute_direction)}
    frame_times, resampled = resample_common_timeline(raw, n_frames)
    xs, ys, elapsed, total_time = resampled[scenario]

    fig, ax, ax2, style_cfg = _setup_figure(
        polygon, corridor_line, water_mask, [(xs, ys)], bounds,
        background, background_raster_path, background_band, background_cmap,
        working_crs, lane_color, lane_width,
        f"Vessel transit -- {_SCENARIO_LABELS[scenario]} ({side} approach)",
        show_time_chart, figsize)

    minx, miny, maxx, maxy = polygon.bounds
    diag = np.hypot(maxx - minx, maxy - miny)
    L = ship_length if ship_length is not None else diag * ship_scale
    W = ship_width if ship_width is not None else L * 0.5
    color = _resolve_ship_colors(ship_color, [scenario])[scenario]

    patch = MplPolygon(ship_polygon(xs[0], ys[0], 0, L, W), closed=True,
                        facecolor=color, edgecolor=style_cfg["text_color"],
                        linewidth=0.8, zorder=5)
    ax.add_patch(patch)
    trail_line, = ax.plot([], [], color=color, linewidth=1.0, alpha=0.5, zorder=3)

    bar = None
    if ax2 is not None:
        bar = ax2.bar([scenario], [0], color=color)
        ax2.set_ylim(0, total_time / 3600 * 1.15)
        ax2.set_ylabel("Elapsed transit time (hours)", color=style_cfg["text_color"])
        ax2.set_title("Transit time", color=style_cfg["text_color"])
        ax2.tick_params(colors=style_cfg["text_color"])

    def update(frame):
        i = min(frame, len(xs) - 1)
        j = min(i + 1, len(xs) - 1)
        dx, dy = xs[j] - xs[i], ys[j] - ys[i]
        if dx == 0 and dy == 0 and i > 0:
            dx, dy = xs[i] - xs[i - 1], ys[i] - ys[i - 1]
        heading = np.arctan2(dy, dx)
        patch.set_xy(ship_polygon(xs[i], ys[i], heading, L, W))
        trail_line.set_data(xs[:i + 1], ys[:i + 1])
        artists = [patch, trail_line]
        if bar is not None:
            bar[0].set_height(elapsed[i] / 3600)
            artists += list(bar)
        return artists

    anim = animation.FuncAnimation(fig, update, frames=n_frames, blit=False)
    if out_path.endswith(".gif"):
        anim.save(out_path, writer=animation.PillowWriter(fps=fps))
    else:
        anim.save(out_path, writer=animation.FFMpegWriter(fps=fps))
    plt.close(fig)
    return out_path


def animate_transit_comparison(polygon, corridor_line, side: str = None,
                                water_mask=None, base_speed_knots: float = 12.0,
                                reduction: float = 0.5, n_frames: int = 150,
                                out_path: str = "transit_comparison.gif",
                                fps: int = 15,
                                # appearance
                                ship_scale: float = 0.03, ship_length: float = None,
                                ship_width: float = None, ship_color: dict = None,
                                lane_color: str = "gray", lane_width: float = 1.0,
                                background: str = "light", background_raster_path: str = None,
                                background_band: int = 1, background_cmap: str = "Blues",
                                working_crs: str = None,
                                bounds: tuple = None, show_time_chart: bool = True,
                                figsize: tuple = (14, 7),
                                scenarios: list = None, reroute_direction: str = "auto"):
    """
    Animate all three scenarios simultaneously on a SHARED time axis, so
    the comparison is real: faster scenarios finish and sit still while
    slower ones keep moving. Optionally shows a live bar chart of elapsed
    transit time per scenario alongside the map.

    reroute_direction : passed to the reroute scenario's track builder --
        see animate_transit() for details. To compare BOTH reroute options
        directly, call this twice with reroute_direction='option_1' and
        'option_2' (or pass scenarios=['reroute'] with different directions
        each time), or use sim.list_reroute_options() to inspect first.
    """
    if scenarios is None:
        scenarios = ["baseline", "speed_reduction", "reroute"]

    raw = {s: build_scenario_track_raw(s, corridor_line, polygon, side,
                                        base_speed_knots, reduction,
                                        reroute_direction=reroute_direction) for s in scenarios}
    frame_times, resampled = resample_common_timeline(raw, n_frames)

    tracks_xy = [(resampled[s][0], resampled[s][1]) for s in scenarios]
    fig, ax, ax2, style_cfg = _setup_figure(
        polygon, corridor_line, water_mask, tracks_xy, bounds,
        background, background_raster_path, background_band, background_cmap,
        working_crs, lane_color, lane_width,
        f"Transit scenario comparison ({side} approach)", show_time_chart, figsize)

    minx, miny, maxx, maxy = polygon.bounds
    diag = np.hypot(maxx - minx, maxy - miny)
    L = ship_length if ship_length is not None else diag * ship_scale
    W = ship_width if ship_width is not None else L * 0.5
    colors = _resolve_ship_colors(ship_color, scenarios)

    patches, trails = {}, {}
    for s in scenarios:
        xs, ys, elapsed, total_time = resampled[s]
        patch = MplPolygon(ship_polygon(xs[0], ys[0], 0, L, W), closed=True,
                            facecolor=colors[s], edgecolor=style_cfg["text_color"],
                            linewidth=0.8, zorder=5, label=_SCENARIO_LABELS[s])
        ax.add_patch(patch)
        patches[s] = patch
        trail, = ax.plot([], [], color=colors[s], linewidth=1.0, alpha=0.5, zorder=3)
        trails[s] = trail

    legend = ax.legend(loc="upper right", fontsize=8)
    if style_cfg["text_color"] != "black":
        legend.get_frame().set_facecolor(style_cfg["fig_facecolor"])
        for text in legend.get_texts():
            text.set_color(style_cfg["text_color"])

    bars = None
    if ax2 is not None:
        max_hours = max(resampled[s][3] for s in scenarios) / 3600 * 1.15
        bars = ax2.bar(scenarios, [0] * len(scenarios),
                        color=[colors[s] for s in scenarios])
        ax2.set_ylim(0, max_hours)
        ax2.set_ylabel("Elapsed transit time (hours)", color=style_cfg["text_color"])
        ax2.set_title("Transit time comparison", color=style_cfg["text_color"])
        ax2.set_xticks(range(len(scenarios)))
        ax2.set_xticklabels([_SCENARIO_LABELS[s] for s in scenarios],
                             rotation=20, ha="right", color=style_cfg["text_color"])
        ax2.tick_params(colors=style_cfg["text_color"])

    def update(frame):
        artists = []
        for idx, s in enumerate(scenarios):
            xs, ys, elapsed, total_time = resampled[s]
            i = min(frame, len(xs) - 1)
            j = min(i + 1, len(xs) - 1)
            dx, dy = xs[j] - xs[i], ys[j] - ys[i]
            if dx == 0 and dy == 0 and i > 0:
                dx, dy = xs[i] - xs[i - 1], ys[i] - ys[i - 1]
            heading = np.arctan2(dy, dx)
            patches[s].set_xy(ship_polygon(xs[i], ys[i], heading, L, W))
            trails[s].set_data(xs[:i + 1], ys[:i + 1])
            artists += [patches[s], trails[s]]
            if bars is not None:
                bars[idx].set_height(elapsed[i] / 3600)
        if bars is not None:
            artists += list(bars)
        return artists

    anim = animation.FuncAnimation(fig, update, frames=n_frames, blit=False)
    if out_path.endswith(".gif"):
        anim.save(out_path, writer=animation.PillowWriter(fps=fps))
    else:
        anim.save(out_path, writer=animation.FFMpegWriter(fps=fps))
    plt.close(fig)
    return out_path
