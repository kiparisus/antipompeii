"""
ANTIPOMPEII Population-to-Street Processing Module

Takes:
- Merged OSM-derived GeoPackage produced by `autoloader.py`
  containing polygons for:
  * total population (e.g. aut_pop_2026_Laxenburg_Austria.tif)
  * 0–14, 15–64, 65+ by sex (e.g. aut_f_15-64_2026_Laxenburg_Austria.tif)

Does:
- Builds a residential-building-based dasymetric mask for each raster:
  * Rasterize "Social" buildings into weights (area × levels).
  * Multiply population raster by weights and rescale to preserve totals.
- Converts non-zero population cells to polygons.
- For each cell, allocates its population to intersecting street
  segments (layer_name == "Street Network") proportionally to the
  length of intersection.
- Appends new attributes to streets for each timestamp and demographic
  band, e.g.:
  * pop_total_2026
  * pop_f_0_14_2026, pop_f_15_64_2026, pop_f_65_plus_2026
  * pop_m_0_14_2026, pop_m_15_64_2026, pop_m_65_plus_2026
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Iterable, Tuple, Union

import logging

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize, shapes
from rasterio.mask import mask
from shapely.geometry import shape as shapely_shape
from shapely.strtree import STRtree


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class DemographicRasters:
    """
    Container for all rasters for a single timestamp.
    """

    total: Path
    female_0_14: Path
    female_15_64: Path
    female_65_plus: Path
    male_0_14: Path
    male_15_64: Path
    male_65_plus: Path

    def items(self) -> Iterable[Tuple[str, Path]]:
        """
        Iterate over (band_key, raster_path) pairs with stable keys.
        """
        return (
            ("total", self.total),
            ("f_0_14", self.female_0_14),
            ("f_15_64", self.female_15_64),
            ("f_65_plus", self.female_65_plus),
            ("m_0_14", self.male_0_14),
            ("m_15_64", self.male_15_64),
            ("m_65_plus", self.male_65_plus),
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


from src.antipompeii.utils.logger import get_module_logger as _get_logger


def _split_layers(
    merged_source: Union[Path, gpd.GeoDataFrame],
    logger: Optional[logging.Logger] = None,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Load merged data and return (buildings, streets).
    """
    log = _get_logger(logger)

    if isinstance(merged_source, (str, Path)):
        merged_gpkg = Path(merged_source)
        log.info(f"Loading merged GeoPackage: {merged_gpkg}")
        gdf = gpd.read_file(merged_gpkg)
    else:
        gdf = merged_source
        log.info("Using in-memory merged GeoDataFrame")

    if "layer_name" not in gdf.columns:
        raise ValueError("Expected 'layer_name' column in merged data.")

    # Buildings: Social polygons
    buildings = gdf[gdf["layer_name"] == "Social"].copy()
    buildings = buildings[buildings.geometry.notnull()]
    buildings = buildings[buildings.geometry.geom_type.isin(
        ["Polygon", "MultiPolygon"]
    )]

    # Streets: Street Network lines
    streets = gdf[gdf["layer_name"] == "Street Network"].copy()
    streets = streets[streets.geometry.notnull()]
    streets = streets[streets.geometry.geom_type.isin(
        ["LineString", "MultiLineString"]
    )]

    log.info(
        f"Found {len(buildings)} Social building polygons and "
        f"{len(streets)} street segments."
    )

    if len(streets) == 0:
        raise ValueError("No 'Street Network' line features found in data.")

    return buildings, streets


def _rasterize_building_weights(
    raster: rasterio.io.DatasetReader,
    buildings: gpd.GeoDataFrame,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """
    Rasterize residential buildings into a weight grid aligned to `raster`.

    Weight per pixel is:
        building_area_in_projection * building_levels (if available),
        or 1.0 if levels missing.
    """
    log = _get_logger(logger)

    if buildings.empty:
        log.warning("No buildings found; falling back to uniform weights.")
        return np.ones((raster.height, raster.width), dtype="float32")

    # Reproject buildings to raster CRS
    if buildings.crs is None:
        raise ValueError("Buildings GeoDataFrame has no CRS.")
    if buildings.crs != raster.crs:
        buildings = buildings.to_crs(raster.crs)

    # Estimate per-building weight
    if "building:levels" in buildings.columns:
        def _safe_levels(val) -> float:
            try:
                if val is None:
                    return 1.0
                if isinstance(val, str) and val.strip() == "":
                    return 1.0
                return float(val)
            except Exception:
                return 1.0

        levels = buildings["building:levels"].apply(_safe_levels).astype("float32")
    else:
        levels = np.ones(len(buildings), dtype="float32")

    # Projected area
    if raster.crs.is_geographic:
        log.warning(
            "Raster CRS is geographic; building area in degrees is not meaningful. "
            "Using 'building:levels' only for weights."
        )
        weights_list = levels.values
    else:
        areas = buildings.geometry.area.astype("float32").values
        weights_list = areas * levels.values

    shapes_iter = (
        (geom, float(w))
        for geom, w in zip(buildings.geometry, weights_list)
        if geom is not None and not geom.is_empty and w > 0
    )

    weights = rasterize(
        shapes=shapes_iter,
        out_shape=(raster.height, raster.width),
        transform=raster.transform,
        fill=0.0,
        dtype="float32",
        all_touched=True,
    )

    if np.all(weights <= 0):
        log.warning(
            "All building weights rasterized to zero; falling back to uniform=1."
        )
        weights = np.ones((raster.height, raster.width), dtype="float32")

    return weights


def _dasymetric_reweight_population(
    raster_path: Path,
    buildings: gpd.GeoDataFrame,
    logger: Optional[logging.Logger] = None,
) -> Tuple[np.ndarray, rasterio.Affine, Dict]:
    """
    Read a WorldPop raster and apply a simple building-based dasymetric
    reweighting:

      - Rasterize buildings into a weight grid W (area × levels).
      - Compute P' = P * W.
      - If sum(P') > 0, rescale P' so that sum(P') = sum(P).
      - Otherwise, fall back to original P.
    """
    log = _get_logger(logger)
    log.info(f"Loading population raster: {raster_path}")

    with rasterio.open(raster_path) as src:
        data = src.read(1).astype("float64")
        transform = src.transform
        meta = src.meta.copy()
        nodata = src.nodata

        # Basic valid mask
        if nodata is not None:
            valid_mask = data != nodata
        else:
            valid_mask = np.isfinite(data)

        total_pop = data[valid_mask].sum()

        # Building weights, aligned with this raster
        weights = _rasterize_building_weights(src, buildings, logger=log)

        # Apply weights only where population is valid
        w = weights.astype("float64")
        P_weighted = np.zeros_like(data, dtype="float64")
        P_weighted[valid_mask] = data[valid_mask] * w[valid_mask]

        total_weighted = P_weighted[valid_mask].sum()

        if total_weighted > 0 and total_pop > 0:
            scale = total_pop / total_weighted
            data_adj = np.zeros_like(data, dtype="float64")
            data_adj[valid_mask] = P_weighted[valid_mask] * scale
            log.info(
                "Dasymetric reweighting applied: "
                f"total={total_pop:.1f}, weighted={total_weighted:.1f}, "
                f"scale={scale:.4f}"
            )
        else:
            log.warning(
                "Dasymetric reweighting skipped (no overlap or zero weights). "
                "Using original population raster."
            )
            data_adj = data

        # Preserve nodata where appropriate
        if nodata is not None:
            data_adj[~valid_mask] = nodata

        meta.update(dtype="float64", count=1)

    return data_adj, transform, meta


def _population_cells_from_array(
    data: np.ndarray,
    transform: rasterio.Affine,
    nodata: Optional[float] = None,
    min_value: float = 0.0,
) -> gpd.GeoDataFrame:
    """
    Convert non-nodata, above-threshold population cells to polygons.

    Returns
    -------
    cells_gdf : GeoDataFrame
        Columns: 'value', 'geometry' (polygons in raster CRS).
    """
    if nodata is not None:
        mask_arr = (data != nodata) & np.isfinite(data)
    else:
        mask_arr = np.isfinite(data)

    if min_value is not None:
        mask_arr &= data > min_value

    polygons = []
    values = []

    for geom, val in shapes(data, mask=mask_arr, transform=transform):
        v = float(val)
        if min_value is not None and v <= min_value:
            continue
        polygons.append(shapely_shape(geom))
        values.append(v)

    if not polygons:
        return gpd.GeoDataFrame({"value": []}, geometry=[], crs=None)

    return gpd.GeoDataFrame({"value": values}, geometry=polygons)


def _allocate_population_to_roads(
    pop_cells: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """
    Allocate population from cell polygons to road segments.

    For each cell:
      - Find intersecting road segments via STRtree.
      - Compute intersection length.
      - Distribute cell population proportionally to intersection length.

    Returns
    -------
    allocations : np.ndarray
        1D array of length len(roads), population allocated per road segment.
    """
    log = _get_logger(logger)

    if pop_cells.empty:
        log.warning("No populated cells; returning zeros for all roads.")
        return np.zeros(len(roads), dtype="float64")

    # Ensure common CRS
    if roads.crs is None:
        raise ValueError("Roads GeoDataFrame has no CRS.")
    if pop_cells.crs is None:
        raise ValueError("Population cells GeoDataFrame has no CRS.")
    if roads.crs != pop_cells.crs:
        pop_cells = pop_cells.to_crs(roads.crs)

    # If CRS is geographic, intersections lengths are in degrees; warn
    if roads.crs.is_geographic:
        log.warning(
            "Roads CRS is geographic; intersection lengths will be in degrees. "
            "Consider reprojecting to a projected CRS before processing."
        )

    road_geoms = list(roads.geometry.values)
    tree = STRtree(road_geoms)

    allocations = np.zeros(len(roads), dtype="float64")

    for _, cell in pop_cells.iterrows():
        cell_geom = cell.geometry
        cell_pop = float(cell["value"])

        if cell_geom is None or cell_geom.is_empty or cell_pop <= 0:
            continue

        candidates = tree.query(cell_geom)

        # Shapely 2.x: query() returns an array of integer indices
        # Shapely 1.x: query() returns an array/list of geometries
        if len(candidates) == 0:
            continue

        # Peek at the first element to decide how to interpret it
        first = candidates[0]
        if isinstance(first, (int, np.integer)):
            # candidates are indices into road_geoms
            candidate_indices = candidates
        else:
            # candidates are geometries; map them back to indices
            # Build a geometry->index dict once for this case
            geom_to_index = {geom: idx for idx, geom in enumerate(road_geoms)}
            candidate_indices = [geom_to_index[geom] for geom in candidates]

        intersections = []
        total_len = 0.0

        for idx in candidate_indices:
            road_geom = road_geoms[int(idx)]
            inter = road_geom.intersection(cell_geom)
            inter_len = inter.length
            if inter_len > 0:
                intersections.append((int(idx), inter_len))
                total_len += inter_len

        if not intersections or total_len <= 0:
            continue

        for idx, inter_len in intersections:
            share = cell_pop * (inter_len / total_len)
            allocations[idx] += share

    log.info(
        f"Allocated {allocations.sum():.1f} people across "
        f"{(allocations > 0).sum()} street segments."
    )

    return allocations




def _process_single_raster_to_roads(
    raster_path: Path,
    buildings: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    band_key: str,
    timestamp_label: str,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """
    Full pipeline for a single raster:
      1) Dasymetric reweighting by buildings.
      2) Convert to polygons.
      3) Allocate to roads.

    Returns
    -------
    allocations : np.ndarray
        Population per road segment for this raster.
    """
    log = _get_logger(logger)
    log.info(
        f"Processing raster '{band_key}' for timestamp '{timestamp_label}' "
        f"({raster_path.name})"
    )

    data_adj, transform, meta = _dasymetric_reweight_population(
        raster_path, buildings, logger=log
    )

    pop_cells = _population_cells_from_array(
        data_adj,
        transform,
        nodata=meta.get("nodata"),
        min_value=0.0,
    )
    pop_cells.set_crs(meta.get("crs"), inplace=True)

    log.info(f"Converted to {len(pop_cells)} population cell polygons.")

    allocations = _allocate_population_to_roads(pop_cells, roads, logger=log)
    return allocations


def _make_column_name(band_key: str, timestamp_label: str) -> str:
    """
    Construct a safe column name from band_key and timestamp label.
    Example:
        band_key="f_0_14", timestamp_label="2026" -> "pop_f_0_14_2026"
    """
    band_safe = band_key.replace("+", "plus").replace("-", "_")
    ts_safe = timestamp_label.replace("-", "_")
    return f"pop_{band_safe}_{ts_safe}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def append_population_to_streets(
    merged_gpkg: Union[Path, gpd.GeoDataFrame],
    demographics_by_timestamp: Dict[str, DemographicRasters],
    output_path: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> gpd.GeoDataFrame:
    """
    Append WorldPop population and age/sex attributes to street segments.
    """
    log = _get_logger(logger)

    buildings, streets = _split_layers(merged_gpkg, logger=log)

    streets_enriched = streets.copy()
    streets_enriched = streets_enriched.reset_index(drop=True)

    for ts_label, rasters in demographics_by_timestamp.items():
        log.info(f"Processing demographic rasters for timestamp: {ts_label}")

        for band_key, raster_path in rasters.items():
            col_name = _make_column_name(band_key, ts_label)

            if col_name in streets_enriched.columns:
                log.info(f"Column '{col_name}' already exists; skipping.")
                continue

            allocations = _process_single_raster_to_roads(
                raster_path=raster_path,
                buildings=buildings,
                roads=streets_enriched,
                band_key=band_key,
                timestamp_label=ts_label,
                logger=log,
            )

            streets_enriched[col_name] = allocations

    if output_path is not None:
        log.info(f"Writing enriched street network to: {output_path}")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        streets_enriched.to_file(output_path, driver="GPKG")

    return streets_enriched
