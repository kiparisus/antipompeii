"""
ANTIPOMPEII Water-feature Downloader

Pulls OSM water-related geometries for a given study extent via osmnx, writing
the result as a single GeoPackage under ``src/data/input/water/``.

    natural   : water, coastline, wetland, spring
    waterway  : river, stream, canal, drain, ditch, brook
    landuse   : reservoir, basin
    leisure   : marina
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import geopandas as gpd
import osmnx as ox
import pandas as pd

from src.antipompeii.modules.dem_downloader import BoundingBox


# ---------------------------------------------------------------------------
# Tag dictionaries
# ---------------------------------------------------------------------------

# Always-on water tags.  natural=coastline and natural=wetland are toggled
# separately because they tend to dominate the layer in coastal and alluvial
# study areas respectively.
_BASE_TAGS: Dict[str, Union[List[str], bool]] = {
    "natural":  ["water", "spring"],
    "waterway": ["river", "stream", "canal", "drain", "ditch", "brook"],
    "landuse":  ["reservoir", "basin"],
    "leisure":  ["marina"],
}


def build_water_tags(
    include_wetlands:     bool = True,
    include_coastline:    bool = True,
    include_intermittent: bool = False,    # noqa: ARG001 — filtered downstream
) -> Dict[str, Union[List[str], bool]]:
    """
    Build the osmnx ``tags`` dictionary used for the water Overpass query.

    ``include_intermittent`` does not change the query; intermittent streams
    are post-filtered after download because osmnx cannot AND-combine tag
    keys natively.  See :func:`WaterDownloader._post_filter`.
    """
    tags = {k: (list(v) if isinstance(v, list) else v) for k, v in _BASE_TAGS.items()}
    if include_wetlands:
        tags["natural"] = sorted(set(tags["natural"]) | {"wetland"})
    if include_coastline:
        tags["natural"] = sorted(set(tags["natural"]) | {"coastline"})
    return tags


# ---------------------------------------------------------------------------
# Downloader
# ---------------------------------------------------------------------------

@dataclass
class WaterDownloadResult:
    path:           Path
    n_features:     int
    n_polygons:     int
    n_lines:        int
    n_points:       int
    elapsed_s:      float
    cached:         bool


class WaterDownloader:
    """
    Download OSM water features for a bounding box (or place name) and write
    them to a GeoPackage.

    Parameters
    ----------
    output_dir
        Directory under which the GeoPackage is written.  Created if missing.
    bbox
        Geographic bounding box; pass either this or *city*.
    city
        Place-name query (e.g. ``"Laxenburg, Austria"``); ignored when *bbox*
        is given.
    include_wetlands, include_coastline, include_intermittent
        Toggle the corresponding tag families.  Defaults match the typical
        urban-vulnerability use case.
    logger
        Module-level logger fallback when None.
    """

    def __init__(
        self,
        output_dir:           Path,
        *,
        bbox:                 Optional[BoundingBox]  = None,
        city:                 Optional[str]          = None,
        include_wetlands:     bool                   = True,
        include_coastline:    bool                   = True,
        include_intermittent: bool                   = False,
        logger:               Optional[logging.Logger] = None,
    ) -> None:
        if bbox is None and not city:
            raise ValueError("Either bbox or city must be provided.")
        self.output_dir = Path(output_dir)
        self.bbox  = bbox
        self.city  = city
        self.include_wetlands     = include_wetlands
        self.include_coastline    = include_coastline
        self.include_intermittent = include_intermittent
        self.logger = logger or logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_cli_params(
        cls,
        output_dir: Path,
        city: Optional[str] = None,
        use_coordinates: bool = False,
        long_min: Optional[float] = None,
        lat_min:  Optional[float] = None,
        long_max: Optional[float] = None,
        lat_max:  Optional[float] = None,
        include_wetlands:     bool = True,
        include_coastline:    bool = True,
        include_intermittent: bool = False,
        logger: Optional[logging.Logger] = None,
    ) -> "WaterDownloader":
        """Build a downloader from the CLI session parameters."""
        bbox: Optional[BoundingBox] = None
        if use_coordinates:
            if None in (long_min, lat_min, long_max, lat_max):
                raise ValueError(
                    "use_coordinates=True requires all four corner values."
                )
            bbox = BoundingBox(
                west=float(long_min), south=float(lat_min),
                east=float(long_max), north=float(lat_max),
            )
        return cls(
            output_dir=output_dir,
            bbox=bbox,
            city=city if not use_coordinates else None,
            include_wetlands=include_wetlands,
            include_coastline=include_coastline,
            include_intermittent=include_intermittent,
            logger=logger,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download(self) -> WaterDownloadResult:
        """Run the Overpass query (or skip on cache hit) and write the GPKG."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / self._build_filename()

        if output_path.exists():
            gdf = gpd.read_file(output_path)
            self.logger.info(
                f"Water cache hit: {output_path.name} ({len(gdf):,} features)."
            )
            return WaterDownloadResult(
                path=output_path,
                n_features=len(gdf),
                n_polygons=int((gdf.geom_type.isin(["Polygon", "MultiPolygon"])).sum()),
                n_lines=int((gdf.geom_type.isin(["LineString", "MultiLineString"])).sum()),
                n_points=int((gdf.geom_type.isin(["Point", "MultiPoint"])).sum()),
                elapsed_s=0.0,
                cached=True,
            )

        tags = build_water_tags(
            include_wetlands=self.include_wetlands,
            include_coastline=self.include_coastline,
            include_intermittent=self.include_intermittent,
        )
        self.logger.info(
            f"Querying OSM water features via osmnx (wetlands={self.include_wetlands}, "
            f"coastline={self.include_coastline}, intermittent={self.include_intermittent})…"
        )
        t0 = time.perf_counter()
        try:
            if self.bbox is not None:
                # osmnx 2.x expects (left, bottom, right, top) = (W, S, E, N)
                bbox_tuple = (self.bbox.west, self.bbox.south,
                              self.bbox.east, self.bbox.north)
                gdf = ox.features_from_bbox(bbox=bbox_tuple, tags=tags)
            else:
                gdf = ox.features_from_place(query=self.city, tags=tags)
        except Exception as exc:
            raise RuntimeError(f"osmnx water query failed: {exc}") from exc
        elapsed = time.perf_counter() - t0

        gdf = self._post_filter(gdf)

        if gdf.empty:
            self.logger.warning(
                f"OSM returned no water features for the requested extent "
                f"({self.bbox or self.city!r}).  Writing an empty GeoPackage so "
                "downstream stages can treat 'no water' as a valid state."
            )
        else:
            self.logger.info(
                f"OSM returned {len(gdf):,} water features in {elapsed:.1f}s."
            )

        gdf = self._normalise(gdf)
        gdf.to_file(output_path, driver="GPKG")
        self.logger.info(f"Water layer written to {output_path}")

        return WaterDownloadResult(
            path=output_path,
            n_features=len(gdf),
            n_polygons=int((gdf.geom_type.isin(["Polygon", "MultiPolygon"])).sum()),
            n_lines=int((gdf.geom_type.isin(["LineString", "MultiLineString"])).sum()),
            n_points=int((gdf.geom_type.isin(["Point", "MultiPoint"])).sum()),
            elapsed_s=elapsed,
            cached=False,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _post_filter(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Drop intermittent streams unless the user explicitly opted in."""
        if gdf.empty:
            return gdf
        if self.include_intermittent or "intermittent" not in gdf.columns:
            return gdf
        before = len(gdf)
        keep = gdf["intermittent"].fillna("no").astype(str).str.lower().isin(
            ("no", "false", "0", "")
        )
        out = gdf[keep].copy()
        self.logger.info(
            f"Filtered {before - len(out):,} intermittent water feature(s); "
            f"{len(out):,} remain."
        )
        return out

    @staticmethod
    def _normalise(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Trim osmnx's wide tag-soup down to the columns we actually need.

        The full GeoDataFrame can contain hundreds of OSM tag columns that
        osmnx promoted from raw XML; we keep only the geometry, the tag keys
        we queried on, plus the OSM id.  This keeps the output GPKG compact
        and stable across runs.
        """
        if gdf.empty:
            cols = ["geometry"]
        else:
            keep = ["geometry", "natural", "waterway", "landuse", "leisure",
                    "water", "name", "intermittent"]
            cols = [c for c in keep if c in gdf.columns]
            if "geometry" not in cols:
                cols.append("geometry")
        out = gdf[cols].copy() if not gdf.empty else gpd.GeoDataFrame(
            {"geometry": []}, geometry="geometry", crs="EPSG:4326"
        )
        if not out.empty and out.crs is None:
            out = out.set_crs("EPSG:4326")
        return out

    def _build_filename(self) -> str:
        """
        Bounding-box-stamped filename, mirroring ``DEMDownloader._build_filename``.

        Place-name queries hash the geocoded bbox first so that distinct
        ``city`` strings still produce stable, comparable filenames.
        """
        bbox = self.bbox or self._geocode_bbox()
        return (
            f"water"
            f"_W{bbox.west:.4f}"
            f"_S{bbox.south:.4f}"
            f"_E{bbox.east:.4f}"
            f"_N{bbox.north:.4f}"
            ".gpkg"
        )

    def _geocode_bbox(self) -> BoundingBox:
        """Resolve self.city → BoundingBox via osmnx geocoding."""
        try:
            gdf = ox.geocode_to_gdf(self.city)
        except Exception as exc:
            raise RuntimeError(
                f"Could not geocode '{self.city}' via osmnx: {exc}"
            ) from exc
        west, south, east, north = gdf.total_bounds
        self.bbox = BoundingBox(west=west, south=south, east=east, north=north)
        return self.bbox


__all__ = ["WaterDownloader", "WaterDownloadResult", "build_water_tags"]
