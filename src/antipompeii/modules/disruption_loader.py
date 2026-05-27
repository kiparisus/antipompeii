"""
ANTIPOMPEII Disruption-to-Street Processing Module

Takes:
- Disruption polygons (or multipolygons) stored as GeoPackage files
  in the disruption sub-folder of the input folder:

    src/data/input/disruption/disruption_{timestamp}.gpkg

  e.g. disruption_20260102.gpkg

- A demography-enriched street network GeoPackage, e.g.:

    20260102_Laxenburg_Austria_demography.gpkg

Does:
- Looks up the disruption file for a given timestamp.
- For each street segment in the demography GeoPackage, checks whether it
  touches (intersects) any disruption polygon.
- Appends a disruption attribute (default: disruption = 1) to line segments
  that touch a disruption polygon; others are set to 0 (or left as-is if the
  column already exists).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import logging
import re

import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


from src.antipompeii.utils.logger import get_module_logger as _get_logger


def _default_input_dir() -> Path:
    """
    Resolve the default disruption input directory:

        src/data/input/disruption
    """
    return Path(__file__).resolve().parents[2] / "data" / "input" / "disruption"


def _extract_timestamp_from_stem(stem: str) -> Optional[str]:
    """
    Extract an 8-digit timestamp (YYYYMMDD) from a filename stem.

    Example:
        "20260102_Laxenburg_Austria_demography" -> "20260102"
    """
    m = re.search(r"(\d{8})", stem)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def find_disruption_file(
    timestamp: str,
    input_dir: Optional[Path] = None,
) -> Optional[Path]:
    """
    Return the path to ``disruption_{timestamp}.gpkg`` under *input_dir* if it
    exists, else ``None``. Pure lookup; no side effects.
    """
    if input_dir is None:
        input_dir = _default_input_dir()
    else:
        input_dir = Path(input_dir)

    candidate = input_dir / f"disruption_{timestamp}.gpkg"
    return candidate if candidate.exists() else None


def append_disruption_to_streets(
    demography_gpkg: Path,
    timestamp: Optional[str] = None,
    input_dir: Optional[Path] = None,
    output_path: Optional[Path] = None,
    disruption_field: str = "disruption",
    disruption_value: int = 1,
    disruption_path: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> gpd.GeoDataFrame:
    """
    Append disruption attribute to demography-enriched street segments.
    """
    log = _get_logger(logger)

    demography_gpkg = Path(demography_gpkg)
    if not demography_gpkg.exists():
        raise FileNotFoundError(f"Demography GeoPackage not found: {demography_gpkg}")

    if input_dir is None:
        input_dir = _default_input_dir()
    else:
        input_dir = Path(input_dir)

    if disruption_path is None:
        if timestamp is None:
            timestamp = _extract_timestamp_from_stem(demography_gpkg.stem)
            if timestamp is None:
                raise ValueError(
                    "Timestamp not provided and could not be inferred from "
                    f"filename: {demography_gpkg.name} (expected 8-digit YYYYMMDD)."
                )

        disruption_path = input_dir / f"disruption_{timestamp}.gpkg"
        log.info(
            f"Looking for disruption polygons for timestamp {timestamp} at: "
            f"{disruption_path}"
        )

        if not disruption_path.exists():
            log.warning(
                f"No disruption file found for timestamp {timestamp} "
                f"({disruption_path}); returning streets unchanged."
            )
            streets = gpd.read_file(demography_gpkg)
            return streets
    else:
        disruption_path = Path(disruption_path)
        if not disruption_path.exists():
            raise FileNotFoundError(
                f"Disruption file does not exist: {disruption_path}"
            )
        log.info(f"Using caller-provided disruption file: {disruption_path}")

    # Load streets and disruption polygons
    log.info(f"Loading demography-enriched streets: {demography_gpkg}")
    streets = gpd.read_file(demography_gpkg)
    if streets.empty:
        log.warning("Demography GeoPackage contains no features; nothing to do.")
        return streets

    log.info(f"Loading disruption polygons: {disruption_path}")
    disruption = gpd.read_file(disruption_path)
    disruption = disruption[disruption.geometry.notnull()]

    # Keep only polygonal geometries
    disruption = disruption[
        disruption.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    ]

    if disruption.empty:
        log.warning(
            "Disruption file has no polygonal geometries; "
            "no disruption will be appended."
        )
        return streets

    # Ensure CRS compatibility
    if streets.crs is None:
        raise ValueError("Streets GeoDataFrame has no CRS.")
    if disruption.crs is None:
        raise ValueError("Disruption GeoDataFrame has no CRS.")

    if streets.crs != disruption.crs:
        log.info(
            "Reprojecting disruption polygons to match streets CRS: "
            f"{streets.crs.to_string()}"
        )
        disruption = disruption.to_crs(streets.crs)

    # Build a combined disruption geometry and check intersects
    log.info(
        "Computing intersection between street segments and disruption extent..."
    )
    disruption_union = disruption.geometry.unary_union

    if disruption_union is None or disruption_union.is_empty:
        log.warning(
            "Combined disruption geometry is empty; "
            "no disruption will be appended."
        )
        return streets

    # Compute boolean mask: True if a street touches/intersects any disruption polygon
    intersects_mask = streets.geometry.intersects(disruption_union)

    # Append / update disruption attribute
    streets_with_disruption = streets.copy()

    if disruption_field not in streets_with_disruption.columns:
        streets_with_disruption[disruption_field] = 0

    # Set disruption_value where intersection occurs
    streets_with_disruption.loc[intersects_mask, disruption_field] = disruption_value

    log.info(
        f"Marked {int(intersects_mask.sum())} out of "
        f"{len(streets_with_disruption)} street segments as disrupted "
        f"({disruption_field} = {disruption_value})."
    )

    # Optionally write to output GeoPackage
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        log.info(f"Writing streets with disruption attribute to: {output_path}")
        streets_with_disruption.to_file(output_path, driver="GPKG")

    return streets_with_disruption
