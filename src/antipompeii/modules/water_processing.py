"""
ANTIPOMPEII Water-Distance Processing Module

Takes:
- An enriched street network GeoPackage (any GPKG with a "Street Network" layer).
- A water-features GeoPackage produced by water_downloader.py.

Does:
- Projects both layers to a local UTM zone (chosen from the centroid via
  osmnx.projection.project_gdf) so distances come out in meters.
- Builds a shapely 2 STRtree of the water geometries and runs
  ``query_nearest(return_distance=True)`` against every street segment in
  one vectorised call.
- Appends the resulting per-segment minimum distance to water as
  ``water_dist_min`` (meters, float64) and writes the enriched GeoPackage.

Output column added
-------------------
  water_dist_min   float64   Minimum 2D distance from the street segment
                             to the nearest water feature (meters).

The attribute name is prefixed with ``water_`` so that
``graph_builder._filter_attributes`` keeps it when building the .gt network.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
import numpy as np
import osmnx as ox
from shapely import STRtree

from src.antipompeii.utils.logger import get_module_logger as _get_logger


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def append_water_distance_to_streets(
    streets_gpkg: Union[Path, gpd.GeoDataFrame],
    water_gpkg:   Union[Path, gpd.GeoDataFrame],
    output_path:  Optional[Path] = None,
    logger:       Optional[logging.Logger] = None,
) -> gpd.GeoDataFrame:
    """
    Compute minimum distance to water (in meters) for each street segment
    and append it as ``water_dist_min``.

    Parameters
    ----------
    streets_gpkg
        Enriched GeoPackage (or in-memory GDF) that contains a ``layer_name``
        column with ``"Street Network"`` rows.
    water_gpkg
        Water-feature GeoPackage produced by ``WaterDownloader.download()``.
        May be empty: every segment then receives NaN and a warning is logged.
    output_path
        If provided, the enriched GeoDataFrame is written here as a GPKG.
    logger
        Optional logger; module-level logger used when None.

    Returns
    -------
    gdf_enriched
        The full GeoDataFrame (all layers) with ``water_dist_min`` added on
        the street rows.  NaN on non-street rows and on streets with
        invalid geometry.
    """
    log = _get_logger(logger)
    log.info("=" * 70)
    log.info("ANTIPOMPEII Water Processing: distance-to-water per street segment")
    log.info("=" * 70)

    gdf = _load_layers(streets_gpkg, log=log)
    water = _load_water(water_gpkg, log=log)

    street_mask = (
        (gdf["layer_name"] == "Street Network")
        & gdf.geometry.notnull()
        & gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])
    )
    n_streets = int(street_mask.sum())
    if n_streets == 0:
        raise ValueError("No 'Street Network' line features found in input GPKG.")

    log.info(f"Total features: {len(gdf):,}, street segments: {n_streets:,}")

    gdf["water_dist_min"] = np.nan
    if water.empty:
        log.warning(
            "Water layer is empty — assigning NaN to every street segment. "
            "Downstream code (vulnerability simulator) mean-imputes NaNs."
        )
    else:
        distances = _compute_distances(
            streets=gdf.loc[street_mask].copy(),
            water=water,
            log=log,
        )
        gdf.loc[street_mask, "water_dist_min"] = distances

        covered = int(np.isfinite(gdf.loc[street_mask, "water_dist_min"]).sum())
        finite  = gdf.loc[street_mask, "water_dist_min"].to_numpy()
        finite  = finite[np.isfinite(finite)]
        if finite.size:
            log.info(
                f"water_dist_min : {finite.min():.1f}–{finite.max():.1f} m "
                f"(median {np.median(finite):.1f} m, {covered:,}/{n_streets:,} segments)"
            )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(output_path, driver="GPKG")
        log.info(f"Enriched GeoPackage written to: {output_path}")

    log.info("=" * 70)
    log.info("✓ Water processing complete")
    log.info("=" * 70)
    return gdf


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_layers(
    source: Union[Path, gpd.GeoDataFrame],
    log: logging.Logger,
) -> gpd.GeoDataFrame:
    """Load a street GeoPackage (all layers) or accept an in-memory GDF."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        log.info(f"Loading streets GeoPackage: {path}")
        gdf = gpd.read_file(path)
    else:
        gdf = source.copy()
        log.info("Using in-memory streets GeoDataFrame")
    if "layer_name" not in gdf.columns:
        raise ValueError("Expected 'layer_name' column in the input data.")
    return gdf


def _load_water(
    source: Union[Path, gpd.GeoDataFrame],
    log: logging.Logger,
) -> gpd.GeoDataFrame:
    """Load the water GeoPackage or accept an in-memory GDF."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        log.info(f"Loading water layer: {path}")
        gdf = gpd.read_file(path)
    else:
        gdf = source.copy()
        log.info("Using in-memory water GeoDataFrame")

    if gdf.empty:
        return gdf

    # Drop invalid / empty geometries before they reach STRtree.
    valid = gdf.geometry.notna() & ~gdf.geometry.is_empty
    if (~valid).any():
        log.info(f"Dropping {int((~valid).sum())} invalid/empty water feature(s).")
        gdf = gdf[valid].copy()

    log.info(
        f"Water features: {len(gdf):,} "
        f"(polygons={int(gdf.geom_type.isin(['Polygon','MultiPolygon']).sum()):,}, "
        f"lines={int(gdf.geom_type.isin(['LineString','MultiLineString']).sum()):,}, "
        f"points={int(gdf.geom_type.isin(['Point','MultiPoint']).sum()):,})"
    )
    if gdf.crs is None:
        log.warning("Water layer has no CRS; assuming EPSG:4326.")
        gdf = gdf.set_crs("EPSG:4326")
    return gdf


def _compute_distances(
    streets: gpd.GeoDataFrame,
    water:   gpd.GeoDataFrame,
    log:     logging.Logger,
) -> np.ndarray:
    """
    Project to a shared local UTM and return per-street nearest-water distance.

    Uses :class:`shapely.STRtree` with
    ``query_nearest(return_distance=True, all_matches=False)``.  Returns a
    float array of shape ``(len(streets),)`` in meters; entries for which
    no nearest neighbour could be found (extremely degenerate cases) are NaN.
    """
    # osmnx picks a UTM zone from the centroid; this gives meters.
    if streets.crs is None:
        log.warning("Streets layer has no CRS; assuming EPSG:4326.")
        streets = streets.set_crs("EPSG:4326")

    # Align both to the same projected CRS via osmnx's UTM helper.
    streets_proj = ox.projection.project_gdf(streets)
    water_proj   = water.to_crs(streets_proj.crs)
    log.info(
        f"Projected to {streets_proj.crs} for distance computation."
    )

    tree = STRtree(water_proj.geometry.to_numpy())
    log.info(
        f"Querying STRtree: {len(streets_proj):,} streets × "
        f"{len(water_proj):,} water features …"
    )

    # query_nearest returns
    #   idx: (2, K) array — row 0 input indices, row 1 tree indices
    #   dists: (K,) array  — Euclidean distances in CRS units (meters here)
    idx, dists = tree.query_nearest(
        streets_proj.geometry.to_numpy(),
        return_distance=True,
        all_matches=False,
    )

    out = np.full(len(streets_proj), np.nan, dtype="float64")
    # idx[0] is the street index, dists[k] the corresponding distance.
    out[idx[0]] = dists
    return out


__all__ = ["append_water_distance_to_streets"]
