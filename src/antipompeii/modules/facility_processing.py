"""
ANTIPOMPEII Facility-to-Street Processing Module
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import logging
import geopandas as gpd
from shapely.strtree import STRtree

from src.antipompeii.utils.logger import get_module_logger as _get_logger


def _ensure_columns(gdf: gpd.GeoDataFrame, cols: Iterable[str]) -> None:
    """Ensure columns exist in GeoDataFrame, creating them with None if missing."""
    for col in cols:
        if col not in gdf.columns:
            gdf[col] = None



def append_facilities_to_streets(
    merged_gpkg: Path,
    facility_layer_names: Sequence[str] = (
        "Health",
        "Emergency",
        "Convertible Shelter",
        "Commercial",
        "Power",
    ),
    max_distance: float = 0.00009,
    max_neighbors: Optional[int] = 1,
    facilities_subset: Optional[gpd.GeoDataFrame] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Append facility attributes to nearest street segments."""
    log = _get_logger(logger)

    merged_gpkg = Path(merged_gpkg)
    if not merged_gpkg.exists():
        raise FileNotFoundError(f"Merged GeoPackage not found: {merged_gpkg}")

    log.info(f"Loading merged GeoPackage for facility processing: {merged_gpkg}")
    gdf = gpd.read_file(merged_gpkg)

    if "layer_name" not in gdf.columns:
        raise ValueError("Expected 'layer_name' column in merged GPKG.")

    streets_mask = (
        (gdf["layer_name"] == "Street Network")
        & gdf.geometry.notnull()
        & gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])
    )
    streets_idx = gdf[streets_mask].index
    streets = gdf.loc[streets_idx].copy()

    if facilities_subset is not None:
        facilities = facilities_subset.copy()
        log.info(f"Processing subset of {len(facilities)} facilities.")
    else:
        facilities_mask = (
            gdf["layer_name"].isin(list(facility_layer_names))
            & gdf.geometry.notnull()
            & gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
        )
        facilities_idx = gdf[facilities_mask].index
        facilities = gdf.loc[facilities_idx].copy()

        log.info(
            f"Found {len(streets)} street segments and "
            f"{len(facilities)} facility polygons in target layers "
            f"{list(facility_layer_names)}."
        )

    if streets.empty or facilities.empty:
        log.warning("No streets or no facility polygons found; nothing to join.")
        return gdf, facilities

    if streets.crs is None or facilities.crs is None:
        raise ValueError("Both streets and facilities must have a CRS defined.")

    if streets.crs != facilities.crs:
        log.info("Reprojecting facilities to match streets CRS.")
        facilities = facilities.to_crs(streets.crs)

    if streets.crs.is_geographic:
        log.info(
            f"CRS is geographic ({streets.crs}). max_distance={max_distance} degrees "
            f"(≈{max_distance*111000:.0f}m at this latitude)."
        )

    logical_source_cols = ["layer_name", "fid", "name", "building", "amenity", "leisure"]
    target_cols = ["fac_layer_name", "fac_fid", "fac_name", "fac_building", "fac_amenity", "fac_leisure"]

    fac_cols_lower = {c.lower(): c for c in facilities.columns}
    source_physical_cols = []
    for logical in logical_source_cols:
        physical = fac_cols_lower.get(logical.lower())
        if physical is None:
            log.warning(f"Facility attribute '{logical}' not found.")
        source_physical_cols.append(physical)

    col_mapping = dict(zip(source_physical_cols, target_cols))
    _ensure_columns(streets, target_cols)

    # Add facility count column
    _ensure_columns(streets, ["fac_count"])
    streets["fac_count"] = 0

    street_geoms = list(streets.geometry.values)
    tree = STRtree(street_geoms)

    log.info(
        f"Running STRtree search: {len(facilities)} facilities → "
        f"{len(streets)} streets (max_distance={max_distance:.6f} degrees, "
        f"max_neighbors={max_neighbors if max_neighbors else 'all'})."
    )

    unmatched_indices = []
    matched_facility_indices = set()

    # Track which facilities match which streets for aggregation
    from collections import defaultdict
    street_to_facilities = defaultdict(list)  # {street_idx: [(fac_row, distance), ...]}

    for fac_idx, fac_row in facilities.iterrows():
        fac_geom = fac_row.geometry
        if fac_geom is None or fac_geom.is_empty:
            unmatched_indices.append(fac_idx)
            continue

        fac_buffered = fac_geom.buffer(max_distance)
        candidate_indices = tree.query(fac_buffered)

        if candidate_indices is None or len(candidate_indices) == 0:
            unmatched_indices.append(fac_idx)
            continue

        dist_list = []
        for pos in candidate_indices:
            pos = int(pos)
            road_geom = street_geoms[pos]
            d = fac_geom.distance(road_geom)
            if d <= max_distance:
                street_idx = streets.index[pos]
                dist_list.append((street_idx, d))

        if not dist_list:
            unmatched_indices.append(fac_idx)
            continue

        dist_list.sort(key=lambda x: x[1])

        if max_neighbors is not None and max_neighbors > 0:
            dist_list = dist_list[:max_neighbors]

        matched_facility_indices.add(fac_idx)

        # Collect facilities per street for aggregation
        for street_idx, d in dist_list:
            street_to_facilities[street_idx].append((fac_row, d))

    # Now aggregate facilities per street segment
    for street_idx, fac_list in street_to_facilities.items():
        # Sort by distance (closest first)
        fac_list.sort(key=lambda x: x[1])

        # Update facility count
        streets.at[street_idx, "fac_count"] = len(fac_list)

        # Aggregate each attribute with semicolon delimiter
        for src_physical, dst in col_mapping.items():
            if src_physical is None:
                continue

            values = []
            for fac_row, _ in fac_list:
                value = fac_row.get(src_physical, None)
                if value is not None and str(value).strip():
                    values.append(str(value))

            if values:
                # Join with semicolon (OSM standard)
                streets.at[street_idx, dst] = ";".join(values)

    unmatched_facilities = facilities.loc[unmatched_indices].copy()

    log.info(
        f"Facilities matched: {len(matched_facility_indices)}; "
        f"unmatched (beyond max_distance): {len(unmatched_facilities)}. "
        f"Total facility-street connections: {sum(len(v) for v in street_to_facilities.values())}."
    )

    gdf_updated = gdf.copy()
    for dst in target_cols + ["fac_count"]:
        if dst not in gdf_updated.columns:
            gdf_updated[dst] = None
        gdf_updated.loc[streets.index, dst] = streets[dst]

    log.info(f"Writing facility-enriched data back to: {merged_gpkg}")
    gdf_updated.to_file(merged_gpkg, driver="GPKG")

    return gdf_updated, unmatched_facilities
