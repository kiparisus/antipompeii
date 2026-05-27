"""
ANTIPOMPEII DEM-to-Street Processing Module

Takes:
- An enriched street network GeoPackage produced by previous pipeline stages
  (e.g. _demography, _disruption — any GPKG with a "Street Network" layer).
- A DEM GeoTIFF produced by dem_downloader.py (or any valid raster).

Does:
- For each street segment (LineString), densifies it into N evenly-spaced
  sample points and queries the DEM raster at each point using
  rasterio.DatasetReader.sample() — a single vectorised call across the
  entire network.
- Assigns the minimum valid (non-nodata, finite) elevation to the segment.
  Segments entirely outside the DEM extent receive NaN.
- Appends the new attribute to the full GeoDataFrame and writes the
  enriched GeoPackage to disk.

Output columns added
---------------------
  elev_min   float64   Minimum DEM elevation (same unit as raster, usually meters)

The attribute name is intentionally prefixed with "elev_" so that
graph_builder._filter_attributes keeps it when building the .gt network.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import LineString, MultiLineString


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


from src.antipompeii.utils.logger import get_module_logger as _get_logger


def _load_streets(
    source: Union[Path, gpd.GeoDataFrame],
    logger: Optional[logging.Logger] = None,
) -> gpd.GeoDataFrame:
    """
    Load the street network from a GeoPackage or in-memory GeoDataFrame.
    """
    log = _get_logger(logger)

    if isinstance(source, (str, Path)):
        path = Path(source)
        log.info(f"Loading GeoPackage: {path}")
        gdf = gpd.read_file(path)
    else:
        gdf = source.copy()
        log.info("Using in-memory GeoDataFrame")

    if "layer_name" not in gdf.columns:
        raise ValueError("Expected 'layer_name' column in the input data.")

    n_streets = (gdf["layer_name"] == "Street Network").sum()
    if n_streets == 0:
        raise ValueError("No features with layer_name == 'Street Network' found.")

    log.info(f"Total features: {len(gdf)}, street segments: {n_streets}")
    return gdf


def _sample_points_along_line(
    line: LineString,
    n_samples: int,
) -> list[tuple[float, float]]:
    """
    Return ``n_samples`` evenly-spaced (x, y) coordinates along ``line``.
    """
    length = line.length
    if length == 0.0 or n_samples <= 1:
        pt = line.coords[0]
        return [(pt[0], pt[1])] * max(1, n_samples)

    return [
        (line.interpolate(d).x, line.interpolate(d).y)
        for d in np.linspace(0.0, length, n_samples)
    ]


def _sample_dem_along_streets(
    streets: gpd.GeoDataFrame,
    dem_path: Path,
    n_samples: int = 5,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """
    Sample DEM values at regularly-spaced points along every street segment
    and return the minimum valid elevation per segment.
    """
    log = _get_logger(logger)
    n_samples = max(2, int(n_samples))
    dem_path = Path(dem_path)

    streets = streets.reset_index(drop=True)

    with rasterio.open(dem_path) as src:
        nodata = src.nodata
        raster_crs = src.crs

        # Align CRS — reproject streets into raster space for sampling
        if streets.crs is not None and streets.crs != raster_crs:
            log.info(
                f"Reprojecting streets from {streets.crs} to raster CRS {raster_crs} "
                "for elevation sampling."
            )
            streets_proj = streets.to_crs(raster_crs)
        else:
            streets_proj = streets

        # Build a flat list of (x, y) sample coordinates and a parallel
        # array of segment indices.  This lets us issue a single
        # rasterio.sample() call for the entire network.
        all_xy: list[tuple[float, float]] = []
        seg_idx: list[int] = []

        for i, geom in enumerate(streets_proj.geometry):
            if geom is None or geom.is_empty:
                continue

            parts: list[LineString] = (
                list(geom.geoms)
                if isinstance(geom, MultiLineString)
                else [geom]
            )

            for part in parts:
                pts = _sample_points_along_line(part, n_samples)
                all_xy.extend(pts)
                seg_idx.extend([i] * len(pts))

        if not all_xy:
            log.warning("No valid street geometries found; returning NaN elevations.")
            return np.full(len(streets), np.nan, dtype="float64")

        log.info(
            f"Sampling DEM at {len(all_xy):,} points across "
            f"{len(streets):,} street segments …"
        )

        # Single vectorized rasterio call
        raw = np.array(
            [v[0] for v in src.sample(all_xy)],
            dtype="float64",
        )

    # Mask nodata and non-finite values, then aggregate per segment
    if nodata is not None:
        raw[raw == nodata] = np.nan
    raw[~np.isfinite(raw)] = np.nan

    seg_idx_arr = np.asarray(seg_idx, dtype="int64")

    df = pd.DataFrame({"seg": seg_idx_arr, "elev": raw})
    valid = df.dropna(subset=["elev"])

    elev_min = np.full(len(streets), np.nan, dtype="float64")
    if not valid.empty:
        mins = valid.groupby("seg")["elev"].min()
        elev_min[mins.index.to_numpy()] = mins.to_numpy()

    covered = np.sum(~np.isnan(elev_min))
    log.info(
        f"Elevation assigned to {covered:,}/{len(streets):,} street segments. "
        f"Min={np.nanmin(elev_min):.1f} m, Max={np.nanmax(elev_min):.1f} m"
        if covered > 0
        else f"Elevation assigned to 0/{len(streets):,} segments — check DEM coverage."
    )

    if covered == 0:
        log.warning(
            "No street segments received elevation values. "
            "Check that the DEM covers the study area and CRS is correct."
        )

    return elev_min


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def append_elevation_to_streets(
    streets_gpkg: Union[Path, gpd.GeoDataFrame],
    dem_path: Path,
    output_path: Optional[Path] = None,
    n_samples: int = 5,
    logger: Optional[logging.Logger] = None,
) -> gpd.GeoDataFrame:
    """
    Append minimum DEM elevation to street segments and return enriched GDF.

    For each "Street Network" segment the DEM is sampled at ``n_samples``
    evenly-spaced points along the LineString.  The lowest valid (non-nodata,
    finite) value is assigned as ``elev_min``.  Non-street features in the
    GeoPackage pass through unchanged so the output file is a full drop-in
    replacement for the input.
    """
    log = _get_logger(logger)

    log.info("=" * 70)
    log.info("ANTIPOMPEII DEM Processing: sampling elevation onto streets")
    log.info("=" * 70)

    dem_path = Path(dem_path)
    if not dem_path.exists():
        raise FileNotFoundError(f"DEM file not found: {dem_path}")

    # Load full GDF (all layers) and isolate streets for sampling
    gdf = _load_streets(streets_gpkg, logger=log)
    street_mask = (
        (gdf["layer_name"] == "Street Network")
        & gdf.geometry.notnull()
        & gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])
    )
    streets = gdf[street_mask].copy().reset_index(drop=True)

    # Sample DEM along all street segments
    elev_min_values = _sample_dem_along_streets(
        streets=streets,
        dem_path=dem_path,
        n_samples=n_samples,
        logger=log,
    )

    # Write back into the full GDF using the original index positions
    gdf["elev_min"] = np.nan
    original_indices = gdf.index[street_mask].tolist()
    for local_i, orig_i in enumerate(original_indices):
        gdf.at[orig_i, "elev_min"] = elev_min_values[local_i]

    log.info("Column 'elev_min' added to street segments.")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(output_path, driver="GPKG")
        log.info(f"Enriched GeoPackage written to: {output_path}")

    log.info("=" * 70)
    log.info("✓ DEM processing complete")
    log.info("=" * 70)

    return gdf
