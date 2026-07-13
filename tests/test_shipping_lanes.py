"""Tests for sharklane.shipping_lanes -- uses the bundled dataset, no
network access required."""
import pytest

from sharklane.shipping_lanes import (
    load_world_shipping_lanes, explode_to_segments, nearest_lane_to_point,
)


def test_load_all_lanes():
    gdf = load_world_shipping_lanes()
    assert len(gdf) == 3
    assert set(gdf["Type"]) == {"Major", "Middle", "Minor"}


def test_load_filtered_by_type():
    gdf = load_world_shipping_lanes(lane_type="Major")
    assert len(gdf) == 1
    assert gdf.iloc[0]["Type"] == "Major"


def test_load_clipped_to_bbox_near_saleh_bay():
    # bbox around Sumbawa / Flores Sea, from the worked example
    gdf = load_world_shipping_lanes(bbox=(115, -10, 120, -6))
    assert len(gdf) >= 1
    assert not gdf.geometry.is_empty.all()


def test_load_clipped_to_empty_region_returns_nothing():
    # the middle of a continental landmass should have no shipping lanes
    gdf = load_world_shipping_lanes(bbox=(10, 45, 11, 46))  # somewhere in the Alps
    assert len(gdf) == 0 or gdf.geometry.is_empty.all()


def test_explode_to_segments():
    gdf = load_world_shipping_lanes(lane_type="Major")
    segs = explode_to_segments(gdf)
    assert len(segs) >= 1
    assert (segs["Type"] == "Major").all()


def test_nearest_lane_to_point():
    gdf = load_world_shipping_lanes(bbox=(115, -10, 120, -6))
    # a point near the Saleh Bay habitat centroid used in the worked example
    line = nearest_lane_to_point(gdf, 117.6, -7.9)
    assert line.geom_type == "LineString"


def test_simulator_auto_lane_type_selects_valid_type():
    import geopandas as gpd
    from shapely.geometry import box
    from sharklane import Simulator

    habitat = box(117.420502, -8.051071, 117.798157, -7.799439)
    gdf = gpd.GeoDataFrame(geometry=[habitat], crs="EPSG:4326")
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "habitat.geojson")
        gdf.to_file(path, driver="GeoJSON")

        sim = Simulator(working_crs="EPSG:32750")
        sim.load_core_habitat(path, source_crs="EPSG:4326")
        line = sim.load_world_shipping_lane(lane_type="auto", pad_deg=2.0, trim_to_polygon=False)

        assert sim.last_lane_type_used in ("Major", "Middle", "Minor")
        assert sim.last_lane_crosses_habitat in (True, False)
        assert line.length > 0


def test_trim_lane_to_polygon_shorter_than_original():
    from shapely.geometry import Polygon, LineString
    from sharklane.shipping_lanes import trim_lane_to_polygon

    # a very long lane (like a real global-dataset segment) crossing a
    # small polygon in the middle
    long_lane = LineString([(-100000, 250), (100000, 250)])
    polygon = Polygon([(-1000, 0), (1000, 0), (1000, 500), (-1000, 500)])

    trimmed, info = trim_lane_to_polygon(long_lane, polygon, pad_fraction=0.25)

    assert trimmed.length < long_lane.length
    assert info["inside_length_m"] == pytest.approx(2000, rel=0.01)
    assert info["pad_length_m"] == pytest.approx(500, rel=0.01)  # 25% of 2000
    assert info["total_length_m"] == pytest.approx(3000, rel=0.01)  # 2000 + 500 + 500


def test_trim_lane_to_polygon_zero_pad_matches_inside_segment():
    from shapely.geometry import Polygon, LineString
    from sharklane.shipping_lanes import trim_lane_to_polygon

    long_lane = LineString([(-100000, 250), (100000, 250)])
    polygon = Polygon([(-1000, 0), (1000, 0), (1000, 500), (-1000, 500)])

    trimmed, info = trim_lane_to_polygon(long_lane, polygon, pad_fraction=0.0)
    assert trimmed.length == pytest.approx(info["inside_length_m"], rel=0.01)


def test_trim_lane_to_polygon_larger_pad_gives_longer_lane():
    from shapely.geometry import Polygon, LineString
    from sharklane.shipping_lanes import trim_lane_to_polygon

    long_lane = LineString([(-100000, 250), (100000, 250)])
    polygon = Polygon([(-1000, 0), (1000, 0), (1000, 500), (-1000, 500)])

    trimmed_small, _ = trim_lane_to_polygon(long_lane, polygon, pad_fraction=0.1)
    trimmed_large, _ = trim_lane_to_polygon(long_lane, polygon, pad_fraction=0.5)
    assert trimmed_large.length > trimmed_small.length


def test_trim_lane_to_polygon_raises_if_no_intersection():
    from shapely.geometry import Polygon, LineString
    from sharklane.shipping_lanes import trim_lane_to_polygon

    lane_far_away = LineString([(10000, 10000), (20000, 20000)])
    polygon = Polygon([(-1000, 0), (1000, 0), (1000, 500), (-1000, 500)])

    with pytest.raises(ValueError, match="does not intersect"):
        trim_lane_to_polygon(lane_far_away, polygon, pad_fraction=0.25)
