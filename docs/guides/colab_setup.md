# Running in Google Colab

## Install

```python
# Option A -- published to PyPI:
!pip install -q sharklane

# Option B -- straight from a GitHub repo:
!pip install -q git+https://github.com/YOUR_USERNAME/sharklane.git

# Option C -- a wheel file you uploaded to /content/ yourself:
!pip install -q /content/sharklane-0.2.0-py3-none-any.whl
```

## Fetching a real coastline for the water/land mask

Colab has full internet access, so you can pull real coastline data
on the fly (Natural Earth 10m, via a GitHub mirror since
`naturalearthdata.com` itself isn't always reachable from restricted
environments):

```python
import urllib.request
import geopandas as gpd
from shapely.geometry import box, Polygon

base = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/10m_physical/ne_10m_coastline"
for ext in ["shp", "shx", "dbf", "prj"]:
    urllib.request.urlretrieve(f"{base}.{ext}", f"/tmp/coastline.{ext}")

coastline = gpd.read_file("/tmp/coastline.shp")
clip_box = box(min_lon - 1, min_lat - 1, max_lon + 1, max_lat + 1)  # your AOI, padded
clipped = coastline[coastline.intersects(clip_box)]

# coastline features are closed rings -- turn them into land polygons
land = [Polygon(g.coords) for g in clipped.geometry if g.coords[0] == g.coords[-1]]
gpd.GeoDataFrame(geometry=land, crs="EPSG:4326").to_file("/tmp/land.geojson", driver="GeoJSON")
```

Then `sim.build_mask("/tmp/land.geojson", bounds=..., resolution=..., source_crs="EPSG:4326")`.
See [Troubleshooting](troubleshooting.md) for why `bounds` here matters a lot.

## Displaying animations inline

`animate_transit()` / `animate_transit_comparison()` only **save** a file
— they don't display anything on their own:

```python
from IPython.display import Image, display

sim.animate_transit_comparison(..., out_path="comparison.gif")
display(Image(filename="comparison.gif"))
```

For `.mp4` output, use `IPython.display.Video` instead of `Image`.

## Uploading your own habitat file

```python
from google.colab import files
uploaded = files.upload()  # then reference the uploaded filename, e.g. "/content/high_value_pixels.geojson"
```
