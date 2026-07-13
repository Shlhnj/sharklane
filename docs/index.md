# sharklane

A Python package for simulating ship-collision mitigation strategies around
whale shark aggregation sites, following the method in Womersley et al.
2024 (*Science of the Total Environment* 934:172776), extended to handle
coastline-constrained sites (bay mouths, islands) via convex-hull-based
routing, a water/land eligibility mask, and schematic ship animations.

## What it does

Given a **core habitat polygon** (a whale shark aggregation site) and a
**shipping lane** (real AIS data, a bundled global dataset, or a manual
line), sharklane simulates three ship-collision mitigation strategies:

1. **Speed reduction** — vessels slow down while transiting the habitat
2. **Rerouting** — vessels go around the habitat's convex hull instead of
   through it
3. **Lane shift** — the shipping lane itself is displaced sideways to clear
   the habitat entirely, if geometrically feasible

...and quantifies the cost to shipping (extra transit time, extra
distance) for each, so mitigation decisions can be evidence-based.

## Install

```bash
pip install sharklane          # once published to PyPI
# or, for local development:
pip install -e .
```

Requires: geopandas, shapely, rasterio, scikit-image, matplotlib, pandas, numpy.

## Where to go next

- **[Quickstart](quickstart.md)** — minimal end-to-end example
- **[Guides](guides/)** — worked examples and troubleshooting
  - [Full worked workflow](guides/workflow.md)
  - [Troubleshooting / common gotchas](guides/troubleshooting.md)
  - [How the reroute algorithm works](guides/reroute_algorithm.md)
  - [Running in Google Colab](guides/colab_setup.md)
- **[API reference](api/)** — every module and method
  - [`Simulator`](api/simulator.md) — the main orchestrator class
  - [`ais`](api/ais.md) — AIS loading and cleaning
  - [`masking` / `raster_vectorize`](api/masking.md) — water/land masks
  - [`shipping_lanes`](api/shipping_lanes.md) — bundled global lanes dataset
  - [`simulate`](api/simulate.md) — speed reduction, reroute, lane shift math
  - [`viz`](api/viz.md) — plots and ship animations

## Data sources bundled with the package

- **World shipping lanes** (Major/Middle/Minor): hand-digitized from a CIA
  nautical chart, via
  [newzealandpaul/Shipping-Lanes](https://github.com/newzealandpaul/Shipping-Lanes)
  (CC BY 4.0). Coarse (3 features total worldwide) — useful for
  orientation and as a fallback, not a substitute for real AIS.

## Testing

```bash
pip install -e .
pip install pytest
pytest tests/ -v
```

Full suite runs offline against synthetic fixtures and the bundled
dataset — no network access required. CI runs this across Python
3.9–3.12.
