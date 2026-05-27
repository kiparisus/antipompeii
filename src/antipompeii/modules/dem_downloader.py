"""
ANTIPOMPEII DEM Downloader Module

Downloads Digital Elevation Model (DEM) rasters from the OpenTopography
Global DEM API for the city extent established in the current session.

The downloaded GeoTIFF is written to src/data/input/elevation/ and its
path is stored in session_data["elevation_dem_path"] for downstream use.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import requests


# ---------------------------------------------------------------------------
# DEM product catalog
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DEMProduct:
    """Describes a single OpenTopography global DEM product."""

    code: str
    name: str
    resolution: str
    source: str
    note: str = ""


DEM_PRODUCTS: Dict[str, DEMProduct] = {
    "AW3D30": DEMProduct(
        code="AW3D30",
        name="ALOS World 3D",
        resolution="30 m",
        source="JAXA",
        note="DSM — includes vegetation & buildings; global coverage",
    ),
    "SRTMGL1": DEMProduct(
        code="SRTMGL1",
        name="NASA SRTM (1 arc-second)",
        resolution="30 m",
        source="NASA",
        note="Geoid heights; coverage 56°S – 60°N",
    ),
    "SRTMGL1_E": DEMProduct(
        code="SRTMGL1_E",
        name="NASA SRTM (1 arc-second, ellipsoidal)",
        resolution="30 m",
        source="NASA",
        note="Ellipsoidal heights",
    ),
    "SRTMGL3": DEMProduct(
        code="SRTMGL3",
        name="NASA SRTM (3 arc-second)",
        resolution="90 m",
        source="NASA",
        note="Coarser; geoid heights",
    ),
    "SRTMGL3_E": DEMProduct(
        code="SRTMGL3_E",
        name="NASA SRTM (3 arc-second, ellipsoidal)",
        resolution="90 m",
        source="NASA",
        note="Ellipsoidal heights",
    ),
    "COP90": DEMProduct(
        code="COP90",
        name="Copernicus GLO-90",
        resolution="90 m",
        source="ESA / Copernicus",
        note="Global DTM derived from TanDEM-X",
    ),
    "COP30": DEMProduct(
        code="COP30",
        name="Copernicus GLO-30",
        resolution="30 m",
        source="ESA / Copernicus",
        note="May require institutional access",
    ),
    "NASADEM": DEMProduct(
        code="NASADEM",
        name="NASADEM",
        resolution="30 m",
        source="NASA",
        note="Void-filled reprocessing of SRTM",
    ),
    "EU_DTM": DEMProduct(
        code="EU_DTM",
        name="EU Digital Terrain Model",
        resolution="30 m",
        source="Copernicus Land Service",
        note="Europe only",
    ),
}

# Ordered list for the numbered CLI menu
DEM_PRODUCT_KEYS: list[str] = list(DEM_PRODUCTS.keys())


# ---------------------------------------------------------------------------
# BoundingBox
# ---------------------------------------------------------------------------

@dataclass
class BoundingBox:
    """Geographic bounding box in WGS-84 decimal degrees."""

    west: float    # min longitude
    south: float   # min latitude
    east: float    # max longitude
    north: float   # max latitude

    def __str__(self) -> str:
        return (
            f"W={self.west:.5f}, S={self.south:.5f}, "
            f"E={self.east:.5f}, N={self.north:.5f}"
        )


# ---------------------------------------------------------------------------
# Downloader
# ---------------------------------------------------------------------------

class DEMDownloader:
    """
    Downloads a global DEM GeoTIFF from OpenTopography for a given extent.

    Parameters
    ----------
    bbox        : Geographic bounding box to download.
    dem_type    : One of the keys in DEM_PRODUCTS (e.g. "AW3D30").
    api_key     : OpenTopography API key.
    output_dir  : Directory where the GeoTIFF will be saved.
    logger      : Optional logger; a module-level logger is used if None.
    """

    API_URL = "https://portal.opentopography.org/API/globaldem"
    _CHUNK = 1 << 20  # 1 MiB streaming chunks

    def __init__(
        self,
        bbox: BoundingBox,
        dem_type: str,
        api_key: str,
        output_dir: Path,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if dem_type not in DEM_PRODUCTS:
            raise ValueError(
                f"Unknown DEM type '{dem_type}'. "
                f"Valid options: {DEM_PRODUCT_KEYS}"
            )
        self.bbox = bbox
        self.dem_type = dem_type
        self.api_key = api_key
        self.output_dir = Path(output_dir)
        self.logger = logger or logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_cli_params(
        cls,
        dem_type: str,
        api_key: str,
        output_dir: Path,
        city: Optional[str] = None,
        use_coordinates: bool = False,
        long_min: Optional[float] = None,
        lat_min: Optional[float] = None,
        long_max: Optional[float] = None,
        lat_max: Optional[float] = None,
        logger: Optional[logging.Logger] = None,
    ) -> "DEMDownloader":
        """
        Create a DEMDownloader from ANTIPOMPEII CLI session parameters.

        Pass either ``use_coordinates=True`` with the four coordinate values,
        or a ``city`` name that will be geocoded via osmnx.
        """
        bbox = cls._resolve_bbox(
            city=city,
            use_coordinates=use_coordinates,
            long_min=long_min,
            lat_min=lat_min,
            long_max=long_max,
            lat_max=lat_max,
            logger=logger,
        )
        return cls(
            bbox=bbox,
            dem_type=dem_type,
            api_key=api_key,
            output_dir=output_dir,
            logger=logger,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download(self, timeout: int = 300) -> Path:
        """
        Stream the DEM GeoTIFF to disk and return its path.

        Parameters
        ----------
        timeout : HTTP timeout in seconds (default 300). Increase for
                  large extents or slow connections.

        Returns
        -------
        Path of the saved GeoTIFF.

        Raises
        ------
        RuntimeError on HTTP errors, API-level error responses, or
        connection failures.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / self._build_filename()

        if output_path.exists():
            size_mb = output_path.stat().st_size / 1_048_576
            self.logger.info(
                f"DEM cache hit: {output_path.name} ({size_mb:.1f} MiB) — skipping download."
            )
            return output_path

        product = DEM_PRODUCTS[self.dem_type]
        self.logger.info(
            f"Requesting {product.name} ({self.dem_type}) from OpenTopography | "
            f"extent: {self.bbox}"
        )

        params = {
            "demtype": self.dem_type,
            "south": self.bbox.south,
            "north": self.bbox.north,
            "west": self.bbox.west,
            "east": self.bbox.east,
            "outputFormat": "GTiff",
            "API_Key": self.api_key,
        }

        t0 = time.perf_counter()
        try:
            response = requests.get(
                self.API_URL,
                params=params,
                timeout=timeout,
                stream=True,
            )
        except requests.exceptions.Timeout as exc:
            raise RuntimeError(
                f"OpenTopography request timed out after {timeout}s. "
                "Consider reducing the extent or increasing the timeout."
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                f"Could not connect to OpenTopography: {exc}"
            ) from exc

        self._raise_for_api_error(response)
        self._stream_to_file(response, output_path)

        elapsed = time.perf_counter() - t0
        size_mb = output_path.stat().st_size / 1_048_576
        self.logger.info(
            f"DEM saved: {output_path.name} ({size_mb:.1f} MiB in {elapsed:.1f}s)"
        )
        return output_path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_bbox(
        city: Optional[str],
        use_coordinates: bool,
        long_min: Optional[float],
        lat_min: Optional[float],
        long_max: Optional[float],
        lat_max: Optional[float],
        logger: Optional[logging.Logger] = None,
    ) -> BoundingBox:
        log = logger or logging.getLogger(__name__)

        if use_coordinates:
            if None in (long_min, lat_min, long_max, lat_max):
                raise ValueError(
                    "Explicit bounding box requested but one or more "
                    "coordinate values are None."
                )
            return BoundingBox(
                west=float(long_min),
                south=float(lat_min),
                east=float(long_max),
                north=float(lat_max),
            )

        if not city:
            raise ValueError(
                "Either use_coordinates=True with four coordinate values, "
                "or a city name must be provided."
            )

        import osmnx as ox

        log.info(f"Geocoding '{city}' via osmnx to obtain bounding box …")
        try:
            gdf = ox.geocode_to_gdf(city)
        except Exception as exc:
            raise RuntimeError(
                f"Could not geocode '{city}' via osmnx: {exc}"
            ) from exc

        west, south, east, north = gdf.total_bounds
        log.info(
            f"Bounding box resolved for '{city}': "
            f"W={west:.5f}, S={south:.5f}, E={east:.5f}, N={north:.5f}"
        )
        return BoundingBox(west=west, south=south, east=east, north=north)

    def _build_filename(self) -> str:
        return (
            f"dem_{self.dem_type}"
            f"_W{self.bbox.west:.4f}"
            f"_S{self.bbox.south:.4f}"
            f"_E{self.bbox.east:.4f}"
            f"_N{self.bbox.north:.4f}"
            ".tif"
        )

    @staticmethod
    def _raise_for_api_error(response: requests.Response) -> None:
        """
        Guard against two failure modes:

        1. Non-200 HTTP status  → always an error.
        2. HTTP 200 but body is text/JSON error, not a GeoTIFF — OpenTopography
           sometimes returns 200 with a plain-text error message (e.g. when the
           API key is invalid or the requested area has no coverage).
        """
        if response.status_code != 200:
            try:
                detail = response.text[:500]
            except Exception:
                detail = "(unreadable body)"
            raise RuntimeError(
                f"OpenTopography returned HTTP {response.status_code}: {detail}"
            )

        content_type = response.headers.get("Content-Type", "")
        # Reject only text/JSON responses — those are API error messages.
        # Accept image/tiff, application/octet-stream, and any binary type
        # (OpenTopography uses octet-stream for Cloud-Optimized GeoTIFFs).
        is_text_error = any(
            ct in content_type.lower()
            for ct in ("text/html", "text/plain", "application/json", "application/xml")
        )
        if is_text_error:
            peek = b""
            for chunk in response.iter_content(chunk_size=4096):
                peek = chunk
                break
            try:
                msg = peek.decode("utf-8", errors="replace")[:500]
            except Exception:
                msg = repr(peek[:200])
            raise RuntimeError(
                f"OpenTopography returned an error response "
                f"(Content-Type: {content_type!r}).\n"
                f"Server message: {msg}"
            )

    def _stream_to_file(self, response: requests.Response, path: Path) -> None:
        """Write streaming response to disk in chunks."""
        with open(path, "wb") as fh:
            for chunk in response.iter_content(chunk_size=self._CHUNK):
                if chunk:
                    fh.write(chunk)
