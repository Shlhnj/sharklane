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
from shapely.geometry import LineString, box

from . import io as sio
from .masking import build_water_mask, WaterMask
from .raster_vectorize import vectorize_raster, water_mask_from_raster
from .shipping_lanes import load_world_shipping_lanes, nearest_lane_to_point, trim_lane_to_polygon
from .ais import load_ais_csv, clean_tracks, to_tracks
from .simulate import speed as speed_sim
from .simulate import reroute as reroute_sim
from .simulate import laneshift as laneshift_sim
from .viz import static as static_viz
from .viz import animate as animate_viz
from .viz import ship_animate
from .viz.ship import compute_reroute_options, get_valid_sides, get_default_side


class Simulator:
    def __init__(self, working_crs: str = "auto"):
        """
        working_crs : a projected CRS (metres) appropriate for your site,
            or 'auto' (default) to automatically determine the correct
            UTM zone once you call load_core_habitat() -- based on your
            habitat polygon's own centroid longitude/latitude. This is
            almost always what you want: distances, areas, speeds, and
            transit times throughout sharklane are computed directly in
            working_crs units, which requires a PROJECTED (metric) CRS to
            be meaningful at all -- an unprojected CRS (plain lat/lon)
            would silently make all of those numbers wrong, since 1 degree
            of longitude is a different real distance depending on
            latitude. 'auto' just removes the need to look up and hardcode
            a UTM EPSG code yourself; pass an explicit one (e.g.
            'EPSG:32750') if you need a specific projection, e.g. for
            consistency across multiple Simulator instances covering the
            same region.
        """
        self.working_crs = working_crs
        self.polygon_gdf = None
        self.polygon = None
        self.corridor_line = None
        self.land_gdf = None
        self.water_mask = None
        self.tracks = {}
        self.results = {}
        self.last_lane_trim_info = None
        self.last_lane_type_used = None
        self.last_lane_crosses_habitat = None
        self.working_crs_auto_detected = False

    def _require_resolved_crs(self):
        """Guard for any method that needs a real working_crs -- raises a
        clear error if 'auto' hasn't been resolved yet (i.e.
        load_core_habitat() hasn't been called), instead of failing later
        with an opaque pyproj CRS-parsing error."""
        if self.working_crs == "auto":
            raise RuntimeError(
                "working_crs is still 'auto' -- call load_core_habitat() "
                "first so the correct UTM zone can be determined from your "
                "habitat's location. If you don't have a habitat polygon "
                "yet, pass an explicit working_crs (e.g. 'EPSG:32750') to "
                "Simulator() instead of relying on auto-detection."
            )

    # ---- Map navigation (zoom) translator ---------------------------------
    # All plot/animate methods take `bounds=(minx, miny, maxx, maxy)` in
    # working_crs units (metres). This translates an ordinary WGS84
    # lon/lat box into that, so you don't have to reason about UTM meters
    # just to zoom the view -- call it again with different lon/lat values
    # to pan or zoom further.

    def zoom_bounds_latlon(self, min_lon: float, min_lat: float,
                            max_lon: float, max_lat: float):
        """
        Translate an ordinary WGS84 (lon/lat) box into working_crs bounds,
        ready to pass as `bounds=` to any plot/animate method.

        Example: sim.zoom_bounds_latlon(117.4, -8.1, 117.9, -7.7)
        """
        self._require_resolved_crs()
        box_wgs84 = gpd.GeoSeries([box(min_lon, min_lat, max_lon, max_lat)], crs="EPSG:4326")
        box_proj = box_wgs84.to_crs(self.working_crs)
        return tuple(box_proj.total_bounds)

    # ---- 1. Core habitat polygon --------------------------------------
    def load_core_habitat(self, path: str, source_crs: str = None):
        gdf = sio.load_polygon(path, crs=source_crs)

        if self.working_crs == "auto":
            # Auto-detect the correct UTM zone from the habitat's own
            # centroid, so users don't have to look up and hardcode an
            # EPSG code themselves. Distances/areas/speeds throughout
            # sharklane still require a PROJECTED crs to be meaningful --
            # this just picks one automatically instead of defaulting to
            # an unprojected CRS (which would silently break all of that
            # math) or a hardcoded region-specific one (which would
            # silently be WRONG for anyone outside that region).
            gdf_wgs84 = gdf if (gdf.crs is not None and gdf.crs.is_geographic) else gdf.to_crs("EPSG:4326")
            centroid = gdf_wgs84.union_all().centroid
            lon, lat = centroid.x, centroid.y
            utm_zone = int((lon + 180) / 6) + 1
            hemisphere_base = 32600 if lat >= 0 else 32700
            self.working_crs = f"EPSG:{hemisphere_base + utm_zone}"
            self.working_crs_auto_detected = True
            print(f"[Simulator] auto-detected working_crs: {self.working_crs} "
                  f"(UTM zone {utm_zone}{'N' if lat >= 0 else 'S'}, from habitat "
                  f"centroid at {lon:.3f}, {lat:.3f})")

        gdf = gdf.to_crs(self.working_crs)
        self.polygon_gdf = gdf
        self.polygon = gdf.union_all()
        return self.polygon

    # ---- 2. Transit / corridor line ------------------------------------
    def load_transit_line(self, path: str, source_crs: str = None):
        self._require_resolved_crs()
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
        self._require_resolved_crs()
        gdf = load_ais_csv(path)
        gdf = gdf.to_crs(self.working_crs)
        cleaned = clean_tracks(gdf, crs_metric=self.working_crs, **clean_kwargs)
        self.tracks = to_tracks(cleaned)
        return self.tracks

    # ---- 4. Water/land mask ---------------------------------------------
    def build_mask(self, land_path: str, bounds=None, resolution: float = 100.0,
                    source_crs: str = None):
        self._require_resolved_crs()
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
    def load_world_shipping_lane(self, lane_type: str = "auto",
                                  pad_deg: float = 1.0, use_nearest: bool = True,
                                  trim_to_polygon: bool = True,
                                  trim_pad_fraction: float = 0.25):
        """
        Load the bundled global shipping lanes dataset (Major/Middle/Minor,
        CIA-derived, see sharklane.shipping_lanes), clipped to a padded
        box around the core habitat, and set it as the transit/corridor
        line. Useful as a real-world-informed fallback lane when you don't
        have local AIS to derive one from directly.

        lane_type : 'Major', 'Middle', 'Minor', or 'auto' (default). With
            'auto', all three types are checked within pad_deg of the
            habitat, and whichever type's nearest segment ACTUALLY CROSSES
            the habitat polygon is used (ties broken Major > Middle >
            Minor). If none cross, falls back to whichever is closest to
            the habitat centroid overall -- trimming will then raise its
            usual clear error, since there's nothing inside the polygon to
            trim around in that case. Check sim.last_lane_type_used
            afterward to see which type was actually picked.

        If use_nearest=True (default) and lane_type is NOT 'auto', picks
        the single lane segment of the given type closest to the habitat
        centroid. Otherwise keeps the full (possibly multi-segment)
        clipped result and uses its union as the corridor line -- only
        sensible if a single coherent segment passes through your AOI.

        By default the found lane is TRIMMED down to a sensible length
        around the habitat (trim_to_polygon=True): global-dataset lanes
        can run for hundreds of km, almost all of it irrelevant to this
        specific site. The trimmed lane keeps the portion of the lane
        actually inside the habitat polygon, extended by
        trim_pad_fraction (default 0.25, i.e. 25%) of that inside-length
        on EACH end. Set trim_to_polygon=False to keep the full untrimmed
        lane (the old default behaviour).
        """
        if self.polygon is None:
            raise RuntimeError("Call load_core_habitat() first.")

        habitat_wgs84 = gpd.GeoSeries([self.polygon], crs=self.working_crs).to_crs("EPSG:4326").iloc[0]
        minx, miny, maxx, maxy = habitat_wgs84.bounds
        bbox = (minx - pad_deg, miny - pad_deg, maxx + pad_deg, maxy + pad_deg)
        centroid = gpd.GeoSeries([self.polygon], crs=self.working_crs).iloc[0].centroid

        if lane_type == "auto":
            candidates = []
            for candidate_type in ["Major", "Middle", "Minor"]:
                lanes = load_world_shipping_lanes(bbox=bbox, lane_type=candidate_type)
                if len(lanes) == 0:
                    continue
                lanes_proj = lanes.to_crs(self.working_crs)
                candidate_line = nearest_lane_to_point(lanes_proj, centroid.x, centroid.y,
                                                        lane_type=candidate_type)
                candidates.append({
                    "type": candidate_type,
                    "line": candidate_line,
                    "crosses": candidate_line.intersects(self.polygon),
                    "dist": candidate_line.distance(centroid),
                })

            if not candidates:
                raise ValueError(f"No shipping lane of any type found within {pad_deg} deg "
                                  "of the habitat -- try a larger pad_deg.")

            crossing = [c for c in candidates if c["crosses"]]
            priority = {"Major": 0, "Middle": 1, "Minor": 2}
            chosen = (min(crossing, key=lambda c: priority[c["type"]]) if crossing
                      else min(candidates, key=lambda c: c["dist"]))

            line = chosen["line"]
            self.last_lane_type_used = chosen["type"]
            self.last_lane_crosses_habitat = chosen["crosses"]
        else:
            lanes = load_world_shipping_lanes(bbox=bbox, lane_type=lane_type)
            if len(lanes) == 0:
                raise ValueError(f"No '{lane_type}' shipping lane found within {pad_deg} deg "
                                  "of the habitat -- try a larger pad_deg or lane_type='auto'.")
            lanes_proj = lanes.to_crs(self.working_crs)
            if use_nearest:
                line = nearest_lane_to_point(lanes_proj, centroid.x, centroid.y, lane_type=lane_type)
            else:
                line = lanes_proj.geometry.union_all()
            self.last_lane_type_used = lane_type
            self.last_lane_crosses_habitat = line.intersects(self.polygon)

        if trim_to_polygon:
            try:
                line, trim_info = trim_lane_to_polygon(line, self.polygon,
                                                         pad_fraction=trim_pad_fraction)
                self.last_lane_trim_info = trim_info
            except ValueError as e:
                raise ValueError(
                    f"{e} This can happen if the auto-picked lane doesn't actually "
                    f"cross the habitat polygon -- try a different lane_type, a "
                    f"larger pad_deg, or set trim_to_polygon=False to use the full "
                    f"untrimmed lane instead."
                )

        self.corridor_line = line
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

    def get_lane_side_options(self):
        """
        Check which `side` values are actually valid for the current
        corridor/lane line, and which one is used by default.

        A lane's orientation determines this -- 'west'/'east' only make
        sense for a roughly east-west lane; a roughly north-south lane
        instead has 'south'/'north' ends. Call this before picking `side`
        explicitly on animate_transit() / animate_transit_comparison() /
        list_reroute_options(), rather than assuming 'west'/'east' always
        apply.

        Returns
        -------
        dict with 'valid_sides' (list of the two applicable labels),
        'default_side' (which one is used if you don't specify), and
        'orientation' ('east_west' or 'north_south').
        """
        if self.corridor_line is None:
            raise RuntimeError("Load or set a transit/corridor line first.")
        from .viz.ship import _lane_orientation
        return {
            "valid_sides": get_valid_sides(self.corridor_line),
            "default_side": get_default_side(self.corridor_line),
            "orientation": _lane_orientation(self.corridor_line),
        }

    def list_reroute_options(self, side: str = None, target_polygon=None,
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

    def animate_transit(self, scenario: str = "baseline", side: str = None,
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

    def animate_transit_comparison(self, side: str = None, target_polygon=None,
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
