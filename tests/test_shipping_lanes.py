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
