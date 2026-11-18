"""1° grid coverage diffing for the downstream raster-cache change stream.

The flood raster cache is organised in 1°×1° lon/lat cells identified by
the integer coordinates of their south-west corner: cell ``(lon, lat)``
spans ``[lon, lon+1) × [lat, lat+1)``.

A cell counts as *covered* when the water polygon overlaps it with
**positive area**:

* inner rings are holes in the polygon, so they never count as coverage;
* contact along the cell edge only (a shared boundary) has zero area and
  therefore does not count as coverage either.

The change record of one upload is the set of cells where coverage
changed by a positive area between the previously served face and the
newly served face — computed from the symmetric difference of the two
polygons, so it captures appearances, disappearances and area changes
inside a cell alike.  Geometrically identical faces (e.g. a re-upload
with reordered ways) yield no recompute cells.
"""

from __future__ import annotations

from math import floor

from shapely.geometry import box, shape
from shapely.geometry.base import BaseGeometry


def serving_geometry(revision_row: dict | None) -> dict | None:
    """GeoJSON geometry a stored revision row serves as current face.

    Mirrors the ``/current`` contract: the latest revision serves its
    polygon only when it is valid; an invalid latest revision serves
    nothing at all.
    """
    if revision_row is not None and revision_row["status"] == "valid":
        return revision_row["geometry"]
    return None


def _cells_with_positive_overlap(geom: BaseGeometry) -> list[dict]:
    """Integer 1° cells whose intersection with ``geom`` has positive area."""
    if geom.is_empty:
        return []
    minx, miny, maxx, maxy = geom.bounds
    cells: list[dict] = []
    # floor(max) is included deliberately: a polygon ending exactly on an
    # integer meridian/parallel only touches the next cell along its edge,
    # which the positive-area test below rejects.
    for lon in range(floor(minx), floor(maxx) + 1):
        for lat in range(floor(miny), floor(maxy) + 1):
            if geom.intersection(box(lon, lat, lon + 1, lat + 1)).area > 0:
                cells.append({"lon": lon, "lat": lat})
    return cells


def changed_cells(old_geometry: dict | None, new_geometry: dict | None) -> list[dict]:
    """Cells whose water coverage changed by a positive area.

    ``old_geometry`` / ``new_geometry`` are the GeoJSON polygons of the
    served face before and after one commit (``None`` when nothing is
    served, e.g. while the latest revision is invalid).  The result is
    sorted by ``(lon, lat)`` for deterministic storage and comparison.
    """
    old = shape(old_geometry) if old_geometry else None
    new = shape(new_geometry) if new_geometry else None

    if old is None and new is None:
        # e.g. consecutive invalid revisions: nothing was served before,
        # nothing is served now -> the old face is not taken down twice.
        return []
    if old is None:
        delta = new
    elif new is None:
        # invalid revision taking the old face down.
        delta = old
    else:
        if old.equals(new):
            # Same face (possibly reordered ways): no recompute cells.
            return []
        delta = old.symmetric_difference(new)

    return _cells_with_positive_overlap(delta)
