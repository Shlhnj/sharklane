"""
Simulator: orchestrates the full workflow --

  1. read core habitat polygon (shapefile/geojson/gpkg)
  2. read or interactively draw the transit/corridor line
  3. simulate speed reduction
  4. mask eligible area (water vs land)
  5. simulate redirection (reroute)
  6. simulate lane shift
  7. create static graphics
  8. create animation
"""
from __future__ import annotations

import geopandas as gpd
from shapely.geometry import LineString

from . import io as sio
from .masking import build_water_mask, WaterMask
from .raster_vectorize import vectorize_raster, water_mask_from_raster
from .shipping_lanes import load_world_shipping_lanes, nearest_lane_to_point
from .ais import load_ais_csv, clean_tracks, to_tracks
from .simulate import speed as speed_sim
from .simulate import reroute as reroute_sim
from .simulate import laneshift as laneshift_sim
from .viz import static as static_viz
from .viz import animate as animate_viz
from .viz import ship_animate
from .viz.ship import compute_reroute_options


class Simulator:
    def __init__(self, working_crs: str = "EPSG:32750"):
        """
        working_crs : a projected CRS (metres) appropriate for your site.
            EPSG:32750 (UTM 50S) covers the Flores Sea / Sumbawa region;
            change this to whatever UTM zone matches your AOI.
        """
        self.working_crs = working_crs
        self.polygon_gdf = None
        self.polygon = None
        self.corridor_line = None
        self.land_gdf = None
        self.water_mask = None
        self.tracks = {}
        self.results = {}

    # ---- 1. Core habitat polygon --------------------------------------
    def load_core_habitat(self, path: str, source_crs: str = None):
        gdf = sio.load_polygon(path, crs=source_crs)
        gdf = gdf.to_crs(self.working_crs)
        self.polygon_gdf = gdf
        self.polygon = gdf.union_all()
        return self.polygon

    # ---- 2. Transit / corridor line ------------------------------------
    def load_transit_line(self, path: str, source_crs: str = None):
        gdf = sio.load_line(path, crs=source_crs)
        gdf = gdf.to_crs(self.working_crs)
        self.corridor_line = gdf.geometry.iloc[0]
        return self.corridor_line

    def draw_transit_line(self):
        """Interactively click a line (in the working CRS's plotted
        coordinates) across e.g. the bay mouth. Requires an interactive
        matplotlib backend."""
        bg = self.polygon_gdf if self.polygon_gdf is not None else None
        self.corridor_line = sio.draw_line_interactive(background_gdf=bg)
        return self.corridor_line

    # ---- AIS ------------------------------------------------------------
    def load_ais(self, path: str, **clean_kwargs):
        gdf = load_ais_csv(path)
        gdf = gdf.to_crs(self.working_crs)
        cleaned = clean_tracks(gdf, crs_metric=self.working_crs, **clean_kwargs)
        self.tracks = to_tracks(cleaned)
        return self.tracks

    # ---- 4. Water/land mask ---------------------------------------------
    def build_mask(self, land_path: str, bounds=None, resolution: float = 100.0,
                    source_crs: str = None):
        land = sio.load_coastline(land_path, crs=source_crs)
        land = land.to_crs(self.working_crs)
        self.land_gdf = land
        if bounds is None:
            minx, miny, maxx, maxy = self.polygon.bounds
            pad = max(maxx - minx, maxy - miny) * 0.5
            bounds = (minx - pad, miny - pad, maxx + pad, maxy + pad)
        self.water_mask = build_water_mask(land, bounds, resolution, crs=self.working_crs)
        return self.water_mask

    def build_mask_from_raster(self, raster_path: str, threshold: float = 0.0,
                                comparison: str = "<", band: int = 1,
                                target_resolution: float = None):
        """
        Build the water/land mask directly from a raster (e.g. bathymetry,
        classified landcover), instead of a vector coastline layer.
        Applies `pixel_value {comparison} threshold` to decide water (True)
        vs land (False) -- e.g. threshold=0, comparison='<' for a
        bathymetry raster where negative values are underwater.

        Reprojects the raster into self.working_crs if needed.
        """
        self.water_mask = water_mask_from_raster(
            path=raster_path, band=band, threshold=threshold, comparison=comparison,
            target_crs=self.working_crs, target_resolution=target_resolution,
        )
        return self.water_mask

    def vectorize_raster(self, raster_path: str, threshold: float = None,
                          comparison: str = ">=", band: int = 1,
                          min_area: float = None, simplify_tolerance: float = None,
                          source_crs: str = None) -> gpd.GeoDataFrame:
        """
        Extract polygons from a raster by threshold filter (e.g. pull land
        or water pixels out of a classified raster, or an area exceeding
        some density/risk threshold out of a continuous raster) and return
        them as a GeoDataFrame in the working CRS. Does not modify
        self.water_mask -- use build_mask_from_raster() for that, or pass
        this result into build_mask()-style vector workflows manually.
        """
        gdf = vectorize_raster(
            path=raster_path, band=band, threshold=threshold, comparison=comparison,
            min_area=min_area, simplify_tolerance=simplify_tolerance,
        )
        if source_crs is not None and gdf.crs is None:
            gdf = gdf.set_crs(source_crs)
        return gdf.to_crs(self.working_crs)

    # ---- World shipping lanes (Major/Middle/Minor) -----------------------
    def load_world_shipping_lane(self, lane_type: str = "Middle",
                                  pad_deg: float = 1.0, use_nearest: bool = True):
        """
        Load the bundled global shipping lanes dataset (Major/Middle/Minor,
        CIA-derived, see sharklane.shipping_lanes), clipped to a padded
        box around the core habitat, and set it as the transit/corridor
        line. Useful as a real-world-informed fallback lane when you don't
        have local AIS to derive one from directly.

        If use_nearest=True (default), picks the single lane segment of
        the given type closest to the habitat centroid. Otherwise keeps
        the full (possibly multi-segment) clipped result and uses its
        union as the corridor line -- only sensible if a single coherent
        segment passes through your AOI.
        """
        if self.polygon is None:
            raise RuntimeError("Call load_core_habitat() first.")

        habitat_wgs84 = gpd.GeoSeries([self.polygon], crs=self.working_crs).to_crs("EPSG:4326").iloc[0]
        minx, miny, maxx, maxy = habitat_wgs84.bounds
        bbox = (minx - pad_deg, miny - pad_deg, maxx + pad_deg, maxy + pad_deg)

        lanes = load_world_shipping_lanes(bbox=bbox, lane_type=lane_type)
        if len(lanes) == 0:
            raise ValueError(f"No '{lane_type}' shipping lane found within {pad_deg} deg "
                              "of the habitat -- try a larger pad_deg or a different lane_type.")

        # reproject into the working (metric) CRS before nearest-distance
        # calc for accuracy -- distance in degrees is not metrically meaningful
        lanes_proj = lanes.to_crs(self.working_crs)
        centroid = gpd.GeoSeries([self.polygon], crs=self.working_crs).iloc[0].centroid

        if use_nearest:
            line = nearest_lane_to_point(lanes_proj, centroid.x, centroid.y, lane_type=lane_type)
            self.corridor_line = line
        else:
            self.corridor_line = lanes_proj.geometry.union_all()
        return self.corridor_line

    # ---- 3. Speed reduction ----------------------------------------------
    def simulate_speed_reduction(self, reductions=None, target_polygon=None):
        polygon = target_polygon or self.corridor_polygon_or_habitat()
        df = speed_sim.simulate_speed_reduction(self.tracks, polygon, reductions)
        summary = speed_sim.summarize(df)
        self.results["speed_reduction"] = {"detail": df, "summary": summary}
        return summary

    # ---- 5. Redirection ---------------------------------------------------
    def simulate_redirection(self, method: str = "least_cost", target_polygon=None):
        polygon = target_polygon or self.corridor_polygon_or_habitat()
        labels = reroute_sim.classify_transit_vs_terminal(self.tracks, polygon)
        transit_tracks = {v: t for v, t in self.tracks.items()
                           if labels.get(v) == "transit"}

        if method == "least_cost":
            if self.water_mask is None:
                raise RuntimeError("Call build_mask() before least_cost rerouting.")
            reroute_df = reroute_sim.reroute_least_cost(transit_tracks, polygon, self.water_mask)
        elif method == "perimeter":
            reroute_df = reroute_sim.reroute_perimeter(transit_tracks, polygon)
        else:
            raise ValueError("method must be 'least_cost' or 'perimeter'")

        time_df = reroute_sim.estimate_reroute_time(reroute_df, transit_tracks, polygon)
        self.results["redirection"] = {
            "labels": labels, "reroute": reroute_df, "time": time_df
        }
        return time_df

    # ---- 6. Lane shift ------------------------------------------------------
    def simulate_lane_shift(self, speed_threshold_mps: float = 7.7,
                              direction=(0, -1), offsets_m=None,
                              target_polygon=None):
        polygon = target_polygon or self.corridor_polygon_or_habitat()
        if self.water_mask is None:
            raise RuntimeError("Call build_mask() before lane shift analysis.")

        lane_vessels = laneshift_sim.identify_lane_vessels(self.tracks, speed_threshold_mps)
        if not lane_vessels:
            self.results["lane_shift"] = {"feasible": False, "reason": "no lane vessels identified"}
            return self.results["lane_shift"]

        lane = laneshift_sim.representative_lane(self.tracks, lane_vessels, polygon)
        shift_results = laneshift_sim.test_lane_shifts(
            lane, polygon, self.water_mask, offsets_m=offsets_m, direction=direction
        )
        min_shift = laneshift_sim.minimum_feasible_shift(shift_results)
        self.results["lane_shift"] = {
            "lane": lane, "results": shift_results, "min_feasible_offset_m": min_shift
        }
        return self.results["lane_shift"]

    # ---- helper -------------------------------------------------------------
    def corridor_polygon_or_habitat(self):
        """Use the corridor line's buffer if defined (bay-mouth-style
        analysis), otherwise fall back to the core habitat polygon itself
        (open-water-style analysis, as in the original Ewing Bank case)."""
        if self.corridor_line is not None:
            return self.corridor_line.buffer(500)  # default 500m corridor width; tune to site
        return self.polygon

    # ---- 7. Graphics ---------------------------------------------------------
    def plot_site_map(self, target_polygon=None, **kwargs):
        return static_viz.plot_site_map(
            target_polygon or self.corridor_polygon_or_habitat(), self.tracks,
            water_mask=self.water_mask, corridor_line=self.corridor_line, **kwargs
        )

    def plot_speed_reduction_curve(self, **kwargs):
        return static_viz.plot_speed_reduction_curve(
            self.results["speed_reduction"]["summary"], **kwargs
        )

    def plot_reroute_paths(self, target_polygon=None, show_original_tracks=True, **kwargs):
        return static_viz.plot_reroute_paths(
            self.results["redirection"]["reroute"],
            target_polygon or self.corridor_polygon_or_habitat(),
            water_mask=self.water_mask,
            tracks=self.tracks if show_original_tracks else None,
            **kwargs
        )

    def plot_lane_shift_feasibility(self, **kwargs):
        return static_viz.plot_lane_shift_feasibility(
            self.results["lane_shift"]["results"], **kwargs
        )

    def list_reroute_options(self, side: str = "west", target_polygon=None,
                              base_speed_knots: float = 12.0):
        """
        Compute and summarize BOTH possible go-around paths (the two arcs
        around the risk polygon boundary connecting the entry/exit points),
        so you can inspect them before picking one for animate_transit() /
        animate_transit_comparison() via reroute_direction=.

        Returns a small summary dict per option: length (km), estimated
        transit time (hours, at base_speed_knots), which compass side it
        bulges toward, and whether the two options are similar in length
        (within 15%) -- a signal it's worth actually choosing rather than
        just taking the shorter one by default.
        """
        if self.corridor_line is None:
            raise RuntimeError("Load or set a transit/corridor line first.")
        polygon = target_polygon or self.polygon
        options, similar_length = compute_reroute_options(self.corridor_line, polygon, side=side)

        speed_mps = base_speed_knots * 0.514
        summary = {}
        for key, opt in options.items():
            summary[key] = {
                "length_km": opt["length_m"] / 1000,
                "est_time_hours": opt["length_m"] / speed_mps / 3600,
                "side": opt["side"],
            }
        summary["similar_length"] = similar_length
        if similar_length:
            summary["note"] = ("The two reroute options are within 15% of each other in "
                                "length -- worth picking explicitly (e.g. based on real "
                                "shark movement data or other constraints) rather than "
                                "defaulting to the shorter one.")
        return summary

    # ---- 8. Animation ----------------------------------------------------------
    def animate_vessel(self, vessel_id: str, out_path: str = "vessel_comparison.gif", **kwargs):
        track = self.tracks[vessel_id]
        reroute_df = self.results.get("redirection", {}).get("reroute")
        reroute_xy = None
        if reroute_df is not None:
            row = reroute_df[reroute_df["vessel_id"] == vessel_id]
            if not row.empty and "path_rowcol" in row.columns:
                path = row.iloc[0].get("path_rowcol")
                if path is not None:
                    reroute_xy = [self.water_mask.xy(r, c) for r, c in path]
        return animate_viz.animate_vessel_comparison(
            track, self.corridor_polygon_or_habitat(), water_mask=self.water_mask,
            reroute_path_xy=reroute_xy, out_path=out_path, **kwargs
        )

    def animate_transit(self, scenario: str = "baseline", side: str = "west",
                         target_polygon=None, base_speed_knots: float = 12.0,
                         reduction: float = 0.5, n_frames: int = 150,
                         out_path: str = "transit.gif", **kwargs):
        """
        Animate a single schematic ship (rectangle body + triangular bow)
        transiting from one side of the lane/corridor line, under one
        scenario:

          'baseline'        -- straight through at constant speed
          'speed_reduction' -- straight through, slowing down inside the
                                risk polygon (visibly, via time-based framing)
          'reroute'         -- goes around the risk polygon's perimeter

        Requires a corridor/lane line (load_transit_line, draw_transit_line,
        or load_world_shipping_lane) to define the approach direction.
        """
        if self.corridor_line is None:
            raise RuntimeError("Load or set a transit/corridor line first "
                                "(load_transit_line / draw_transit_line / "
                                "load_world_shipping_lane).")
        polygon = target_polygon or self.polygon
        return ship_animate.animate_transit(
            polygon, self.corridor_line, scenario=scenario, side=side,
            water_mask=self.water_mask, base_speed_knots=base_speed_knots,
            reduction=reduction, n_frames=n_frames, out_path=out_path,
            working_crs=self.working_crs, **kwargs
        )

    def animate_transit_comparison(self, side: str = "west", target_polygon=None,
                                     base_speed_knots: float = 12.0, reduction: float = 0.5,
                                     n_frames: int = 150, out_path: str = "transit_comparison.gif",
                                     scenarios: list = None, **kwargs):
        """
        Animate all three scenarios (baseline / speed reduction / reroute)
        simultaneously for direct visual comparison -- the mitigated ships
        visibly take longer to cross than the baseline ship.
        """
        if self.corridor_line is None:
            raise RuntimeError("Load or set a transit/corridor line first "
                                "(load_transit_line / draw_transit_line / "
                                "load_world_shipping_lane).")
        polygon = target_polygon or self.polygon
        return ship_animate.animate_transit_comparison(
            polygon, self.corridor_line, side=side, water_mask=self.water_mask,
            base_speed_knots=base_speed_knots, reduction=reduction, n_frames=n_frames,
            out_path=out_path, scenarios=scenarios, working_crs=self.working_crs, **kwargs
        )
