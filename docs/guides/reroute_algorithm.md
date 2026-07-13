# How the reroute algorithm works

## The core idea: route around the convex hull, not the raw polygon

If your habitat polygon has any concave notches (a bay, an irregular
coastline-derived shape, anything non-convex), routing "around" it should
**never** trace into those notches — the shortest path around an obstacle
only ever needs to touch its **convex hull** vertices. Any concave notch's
opening is, by construction, a straight chord of the hull; cutting
straight across that chord is always shorter than detouring in and back
out.

An earlier version of this package walked the polygon's *raw* boundary
instead, which for a concave habitat produced visibly bad, needlessly long
routes that hugged every dent. Fixed by routing around
`polygon.convex_hull` instead — verified on a notched test polygon: **20%
shorter path**, while still never entering the true (concave) polygon's
interior.

## Two possible routes, not just the shorter one

Since a polygon can be gone around in two directions, `compute_reroute_options()`
computes **both**, not just the shorter one:

```python
options, similar_length = compute_reroute_options(corridor_line, polygon, side="west")
# options = {"option_1": {...}, "option_2": {...}}
```

Each option reports `length_m`, a rough compass `side` label
(`north`/`south` for an east-west lane, `east`/`west` for a north-south
lane), and the arc geometry itself. `similar_length=True` (within 15%)
signals it's genuinely worth picking deliberately rather than defaulting
to the shorter one — e.g. based on known shark movement patterns, or
which side has calmer water.

```python
opts = sim.list_reroute_options(side="west")
sim.animate_transit(scenario="reroute", reroute_direction="option_1")  # or "north"/"south"
```

## Vertex-forcing to avoid corner-cutting

Uniform arc-length sampling alone can straddle a sharp polygon corner
without ever landing exactly on it — the straight chord connecting two
nearby samples then cuts slightly inside the polygon near that corner.
Fixed by forcing every real hull vertex into the sample set, and by
**not** re-resampling the resulting fine polyline down to a coarser one
for animation (an earlier version did this as a second pass, which
silently undid the vertex-forcing one level removed). Verified: the
reroute path never intersects `polygon.buffer(-0.001)` — i.e. it doesn't
cut inside even at sub-millimeter tolerance.

## Two implementations, same principle

- `sharklane.viz.ship.compute_reroute_options()` / `build_reroute_track()`
  — used for animation, works purely geometrically (no water mask needed).
- `sharklane.simulate.reroute.reroute_perimeter()` — the numerical
  (distance/time-accounting) equivalent, also convex-hull-based.
- `sharklane.simulate.reroute.reroute_least_cost()` — a genuinely
  different approach: a land-mask-constrained least-cost path search
  (`skimage.graph.MCP_Geometric`) over a raster grid. This naturally
  handles concave obstacles correctly without any special-casing, since
  it searches the whole grid rather than walking a boundary — but it
  needs `build_mask()` to have been called first, and its output quality
  depends on mask resolution.

## Circular / many-vertex polygons

A "circle" loaded from GeoJSON is always an approximation (dozens to
hundreds of straight edges — there's no true curve in vector geometry).
Since a circle is already convex, this is actually the easiest case: no
concave notches to worry about, and `polygon.convex_hull` is virtually
identical to the polygon itself. Verified directly: a 129-vertex circle
approximation routes correctly with no special handling needed.
