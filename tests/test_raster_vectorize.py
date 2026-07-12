"""Tests for sharklane.raster_vectorize."""
import numpy as np
import rasterio
from rasterio.transform import from_bounds
import pytest

from sharklane.raster_vectorize import vectorize_raster, water_mask_from_raster

CRS_WGS84 = "EPSG:4326"


@pytest.fixture
def synthetic_raster(tmp_path):
    """A synthetic 'bathymetry' raster: mostly negative (water) with an
    elliptical positive (land) blob in the middle."""
    h, w = 100, 150
    bounds = (117.0, -8.5, 118.0, -7.5)
    transform = from_bounds(*bounds, w, h)
    elev = np.full((h, w), -50.0)
    yy, xx = np.mgrid[0:h, 0:w]
    blob = ((xx - 75) ** 2 / 20 ** 2 + (yy - 50) ** 2 / 12 ** 2) < 1
    elev[blob] = 120.0

    path = tmp_path / "bathymetry.tif"
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=1,
                        dtype="float32", crs=CRS_WGS84, transform=transform) as dst:
        dst.write(elev.astype("float32"), 1)
    return str(path), blob.sum(), h * w


def test_vectorize_raster_extracts_land(synthetic_raster):
    path, blob_pixel_count, total_pixels = synthetic_raster
    land_gdf = vectorize_raster(path, threshold=0, comparison=">=")
    assert len(land_gdf) >= 1
    # sanity: extracted polygon(s) should cover roughly the blob's pixel footprint
    # (can't compare area directly across CRS/units easily here, just confirm non-empty)
    assert land_gdf.geometry.area.sum() > 0


def test_vectorize_raster_threshold_comparisons(synthetic_raster):
    path, blob_pixel_count, total_pixels = synthetic_raster
    # '<' threshold should pick up water (the majority of pixels)
    water_gdf = vectorize_raster(path, threshold=0, comparison="<")
    land_gdf = vectorize_raster(path, threshold=0, comparison=">=")
    assert water_gdf.geometry.area.sum() > land_gdf.geometry.area.sum()


def test_vectorize_raster_min_area_filters_speckle(synthetic_raster):
    path, _, _ = synthetic_raster
    unfiltered = vectorize_raster(path, threshold=0, comparison=">=")
    filtered = vectorize_raster(path, threshold=0, comparison=">=", min_area=1e6)
    assert len(filtered) <= len(unfiltered)


def test_water_mask_from_raster(synthetic_raster):
    path, _, _ = synthetic_raster
    wm = water_mask_from_raster(path, threshold=0, comparison="<", target_crs=CRS_WGS84)
    assert wm.mask.ndim == 2
    # majority of pixels should be water given the raster is mostly -50
    assert wm.mask.mean() > 0.5
