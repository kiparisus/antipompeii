"""
ANTIPOMPEII WorldPop Demography Loader Module

Downloads:

- 100m total population GeoTIFF rasters (2015–2030, Global_2015_2030 R2025A)
- 100m age/sex rasters (2015–2030) from the Global_2015_2030 R2025A release
- Age/sex statistics via WorldPop API (years 2000–2020 only)

Population raster URL pattern (R2025A, 100 m):

https://data.worldpop.org/GIS/Population/Global_2015_2030/R2025A/{year}/{ISO3}/v1/100m/{model}/
{iso3_lower}_pop_{year}_{model_code}_100m_R2025A_v1.tif

Age/sex raster URL pattern (R2025A, 100 m):

https://data.worldpop.org/GIS/AgeSex_structures/Global_2015_2030/R2025A/{year}/{ISO3}/v1/100m/{model}/
{iso3_lower}_{sex}_{agecode}_{year}_{model_code}_100m_R2025A_v1.tif

Where:

iso3_lower = 3‑letter country code, lower case (e.g. "dza", "aut")
ISO3       = same, upper case (e.g. "DZA", "AUT")
sex        = "f" (female) or "m" (male), lower case
agecode    = "00","01","05","10","15","20","25","30","35",
             "40","45","50","55","60","65","70","75","80"
year       = 2015–2030
model      = "constrained" or "unconstrained"
model_code = "CN" or "UN"
"""

from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging
from dataclasses import dataclass, field
from datetime import datetime
import time
import json
import tempfile
import warnings

import requests
import pandas as pd
import geopandas as gpd
import osmnx as ox
import numpy as np
from shapely.geometry import Polygon, box as shapely_box

# Raster processing imports
try:
    import rasterio
    from rasterio.mask import mask
    from rasterio.crs import CRS

    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    warnings.warn("rasterio not installed. Raster download features will be disabled.")


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class BoundingBox:
    """Represents a geographic bounding box."""
    west: float
    south: float
    east: float
    north: float

    def to_tuple(self) -> Tuple[float, float, float, float]:
        """Return as (north, south, east, west) – OSMnx style."""
        return (self.north, self.south, self.east, self.west)

    def to_bbox_tuple(self) -> Tuple[float, float, float, float]:
        """Return as (west, south, east, north)."""
        return (self.west, self.south, self.east, self.north)

    def to_worldpop_geojson(self) -> Dict:
        """Convert bbox to WorldPop API compatible GeoJSON."""
        coordinates = [
            [self.west, self.south],
            [self.east, self.south],
            [self.east, self.north],
            [self.west, self.north],
            [self.west, self.south],
        ]

        return {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coordinates],
                },
            }],
        }

    def to_shapely_polygon(self) -> Polygon:
        """Convert to Shapely polygon for clipping."""
        return shapely_box(self.west, self.south, self.east, self.north)


@dataclass
class PopulationData:
    """Container for population statistics and rasters."""
    year: int
    total_population: float
    age_sex_pyramid: Optional[pd.DataFrame] = None
    execution_time: Optional[float] = None
    raw_response: Optional[Dict] = None

    # Total population raster
    raster_path: Optional[Path] = None
    raster_clipped_path: Optional[Path] = None
    raster_metadata: Optional[Dict] = None

    # 100m age/sex rasters (keys: "female_5-9", "male_80+", "female_0-14", etc.)
    age_sex_rasters: Dict[str, Path] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class PopulationLoaderEnhanced:
    """
    WorldPop demography loader for ANTIPOMPEII.

    Downloads:
    1. 100m total population rasters (2015–2030, Global_2015_2030 R2025A)
    2. 100m age/sex rasters (2015–2030, R2025A Global_2015_2030 release)
    3. Age/sex statistics via WorldPop API for validation (years 2000–2020)
    """

    WORLDPOP_API_BASE = "https://api.worldpop.org/v1"
    WORLDPOP_REST_BASE = "https://www.worldpop.org/rest/data"
    WORLDPOP_HTTP_BASE = "https://data.worldpop.org/GIS"

    # Data availability
    TOTAL_POP_YEARS = list(range(2015, 2031))      # 2015–2030 total counts, R2025A
    AGE_SEX_100M_YEARS = list(range(2015, 2031))   # 2015–2030 age/sex counts, R2025A

    # Age groups (WorldPop API / docs)
    AGE_GROUPS = [
        "0-1", "1-4", "5-9", "10-14", "15-19", "20-24", "25-29",
        "30-34", "35-39", "40-44", "45-49", "50-54", "55-59",
        "60-64", "65-69", "70-74", "75-79", "80+",
    ]

    # Age group → 2‑digit agecode in filenames
    AGE_CODE_MAP = {
        "0-1": "00",
        "1-4": "01",
        "5-9": "05",
        "10-14": "10",
        "15-19": "15",
        "20-24": "20",
        "25-29": "25",
        "30-34": "30",
        "35-39": "35",
        "40-44": "40",
        "45-49": "45",
        "50-54": "50",
        "55-59": "55",
        "60-64": "60",
        "65-69": "65",
        "70-74": "70",
        "75-79": "75",
        "80+": "80",
    }

    # Country name → ISO3
    COUNTRY_ISO3_MAP = {
        # Europe
        "austria": "AUT", "germany": "DEU", "france": "FRA", "italy": "ITA",
        "spain": "ESP", "united kingdom": "GBR", "netherlands": "NLD",
        "belgium": "BEL", "switzerland": "CHE", "poland": "POL",
        "czech republic": "CZE", "czechia": "CZE", "slovakia": "SVK",
        "hungary": "HUN", "slovenia": "SVN", "croatia": "HRV",
        "romania": "ROU", "bulgaria": "BGR", "portugal": "PRT",
        "greece": "GRC", "sweden": "SWE", "norway": "NOR",
        "denmark": "DNK", "finland": "FIN", "ireland": "IRL",

        # North America
        "united states": "USA", "usa": "USA", "canada": "CAN",
        "mexico": "MEX",

        # Asia
        "china": "CHN", "india": "IND", "japan": "JPN",
        "south korea": "KOR", "korea": "KOR", "thailand": "THA",
        "vietnam": "VNM", "indonesia": "IDN", "philippines": "PHL",
        "malaysia": "MYS", "singapore": "SGP", "pakistan": "PAK",
        "bangladesh": "BGD", "nepal": "NPL", "sri lanka": "LKA",

        # Middle East & North Africa
        "lebanon": "LBN", "syria": "SYR", "jordan": "JOR",
        "israel": "ISR", "palestine": "PSE", "iraq": "IRQ",
        "iran": "IRN", "saudi arabia": "SAU", "kuwait": "KWT",
        "bahrain": "BHR", "qatar": "QAT", "united arab emirates": "ARE",
        "uae": "ARE", "oman": "OMN", "yemen": "YEM", "turkey": "TUR",
        "turkiye": "TUR", "algeria": "DZA", "tunisia": "TUN",
        "libya": "LBY",

        # Africa
        "south africa": "ZAF", "nigeria": "NGA", "kenya": "KEN",
        "ethiopia": "ETH", "egypt": "EGY", "morocco": "MAR",
        "ghana": "GHA", "tanzania": "TZA", "uganda": "UGA",

        # South America
        "brazil": "BRA", "argentina": "ARG", "chile": "CHL",
        "colombia": "COL", "peru": "PER", "venezuela": "VEN",

        # Oceania
        "australia": "AUS", "new zealand": "NZL",
    }

    def __init__(
        self,
        city: Optional[str] = None,
        bbox: Optional[BoundingBox] = None,
        years: Optional[List[int]] = None,
        api_key: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        download_rasters: bool = True,
        constrained: bool = True,
        output_dir: Optional[Path] = None,
        disaggregate: bool = False,
    ):
        """
        Initialize PopulationLoader.

        Args:
            city: Place name, e.g. "Laxenburg, Austria".
            bbox: BoundingBox instead of city.
            years: List of years. If None, defaults to [current_year].
            api_key: Optional WorldPop API key.
            logger: Logger instance.
            download_rasters: Download total population rasters (100 m).
            constrained: Use constrained model for rasters.
            output_dir: Output directory for rasters.
            disaggregate: Download 100 m age/sex rasters (no synthetic disaggregation).
        """
        if city is None and bbox is None:
            raise ValueError("Either 'city' or 'bbox' must be provided")

        if city is not None and bbox is not None:
            raise ValueError("Provide either 'city' OR 'bbox', not both")

        if download_rasters and not RASTERIO_AVAILABLE:
            raise ImportError("rasterio required. Install with: pip install rasterio")

        self.city = city
        self.bbox = bbox
        self.use_coordinates = bbox is not None

        # Default years: current year if not specified
        if years is None:
            years = [datetime.now().year]
        self.years = years

        self.api_key = api_key
        self.logger = logger or logging.getLogger(__name__)
        self.session = requests.Session()

        self.download_rasters = download_rasters
        self.constrained = constrained
        self.disaggregate = disaggregate

        self.output_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp())
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.population_data: Dict[int, PopulationData] = {}
        self.country_iso3: Optional[str] = None
        self._download_times: Dict[str, float] = {}

        self._place_geometry: Optional[Dict] = None
        self._place_gdf: Optional[gpd.GeoDataFrame] = None

        # Validate with logger already available
        self._validate_years(self.years, disaggregate)

    # ------------------------------------------------------------------ #
    # Constructors                                                       #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_cli_params(
        cls,
        city: str,
        use_coordinates: bool,
        years: Optional[List[int]] = None,
        timestamps: Optional[List[str]] = None,
        long_min: Optional[float] = None,
        lat_min: Optional[float] = None,
        long_max: Optional[float] = None,
        lat_max: Optional[float] = None,
        api_key: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        download_rasters: bool = True,
        constrained: bool = True,
        output_dir: Optional[Path] = None,
        disaggregate: bool = False,
    ) -> "PopulationLoaderEnhanced":
        """Create loader from CLI parameters."""
        # Derive years from timestamps if provided; otherwise use current year
        if years is None:
            if timestamps:
                years = cls._timestamps_to_years(timestamps)
            else:
                years = [datetime.now().year]

        if use_coordinates:
            if None in [long_min, lat_min, long_max, lat_max]:
                raise ValueError("All coordinates must be provided when use_coordinates=True")

            bbox = BoundingBox(
                west=long_min,
                south=lat_min,
                east=long_max,
                north=lat_max,
            )

            return cls(
                bbox=bbox,
                years=years,
                api_key=api_key,
                logger=logger,
                download_rasters=download_rasters,
                constrained=constrained,
                output_dir=output_dir,
                disaggregate=disaggregate,
            )

        return cls(
            city=city,
            years=years,
            api_key=api_key,
            logger=logger,
            download_rasters=download_rasters,
            constrained=constrained,
            output_dir=output_dir,
            disaggregate=disaggregate,
        )

    @staticmethod
    def _timestamps_to_years(timestamps: List[str]) -> List[int]:
        """Convert YYYYMMDD timestamps to unique sorted years."""
        years: List[int] = []
        for ts in timestamps:
            try:
                dt = datetime.strptime(ts, "%Y%m%d")
                years.append(dt.year)
            except ValueError:
                raise ValueError(f"Invalid timestamp: {ts}. Expected YYYYMMDD")
        return sorted(list(set(years)))

    def _validate_years(self, years: List[int], download_age_sex: bool) -> None:
        """Validate years based on WorldPop data availability."""
        if download_age_sex:
            invalid = [y for y in years if y not in self.AGE_SEX_100M_YEARS]
            if invalid:
                raise ValueError(
                    f"Years {invalid} not available for 100m age/sex rasters. "
                    f"Available: {self.AGE_SEX_100M_YEARS[0]}–{self.AGE_SEX_100M_YEARS[-1]}"
                )
        else:
            invalid = [y for y in years if y not in self.TOTAL_POP_YEARS]
            if invalid:
                self.logger.warning(
                    f"Years {invalid} outside 100m total population R2025A range "
                    f"({self.TOTAL_POP_YEARS[0]}–{self.TOTAL_POP_YEARS[-1]})"
                )

    # ------------------------------------------------------------------ #
    # Geocoding / geometry                                               #
    # ------------------------------------------------------------------ #

    def _get_country_iso3(self) -> str:
        """Determine ISO3 country code from city or bbox."""
        if self.country_iso3:
            return self.country_iso3

        # Parse from "City, Country"
        if self.city:
            parts = self.city.split(",")
            if len(parts) >= 2:
                country_name = parts[-1].strip().lower()
                if country_name in self.COUNTRY_ISO3_MAP:
                    self.country_iso3 = self.COUNTRY_ISO3_MAP[country_name]
                    self.logger.info(f"Detected country: {country_name} ({self.country_iso3})")
                    return self.country_iso3

            # Fallback geocoding
            try:
                gdf = ox.geocode_to_gdf(self.city)
                if "country" in gdf.columns:
                    country_name = gdf["country"].iloc[0].lower()
                    if country_name in self.COUNTRY_ISO3_MAP:
                        self.country_iso3 = self.COUNTRY_ISO3_MAP[country_name]
                        self.logger.info(f"Geocoded country: {country_name} ({self.country_iso3})")
                        return self.country_iso3
            except Exception as e:
                self.logger.warning(f"Geocoding failed: {e}")

        # Try bbox center
        if self.bbox:
            center_lat = (self.bbox.north + self.bbox.south) / 2
            center_lon = (self.bbox.east + self.bbox.west) / 2
            try:
                gdf = ox.geocode_to_gdf(f"{center_lat}, {center_lon}")
                if "country" in gdf.columns:
                    country_name = gdf["country"].iloc[0].lower()
                    if country_name in self.COUNTRY_ISO3_MAP:
                        self.country_iso3 = self.COUNTRY_ISO3_MAP[country_name]
                        self.logger.info(f"Detected country from bbox: {country_name} ({self.country_iso3})")
                        return self.country_iso3
            except Exception:
                pass

        raise ValueError(
            "Could not determine country ISO3 code. "
            "Please specify city with country (e.g., 'Munich, Germany')."
        )

    def _get_place_geojson(self) -> Dict:
        """Convert place name to GeoJSON using OSMnx."""
        if self._place_geometry is not None:
            return self._place_geometry

        self.logger.info(f"Geocoding: {self.city}")
        try:
            gdf = ox.geocode_to_gdf(self.city)
            if gdf.crs != "EPSG:4326":
                gdf = gdf.to_crs("EPSG:4326")
            self._place_gdf = gdf
            self._place_geometry = json.loads(gdf.to_json())
            self.logger.info(f"✓ Geocoded {self.city}")
            return self._place_geometry
        except Exception as e:
            self.logger.error(f"Geocoding failed: {e}")
            raise ValueError(f"Could not find place: {self.city}")

    def _get_query_geojson(self) -> Dict:
        """Get query geometry as GeoJSON."""
        if self.use_coordinates:
            return self.bbox.to_worldpop_geojson()
        return self._get_place_geojson()

    def _get_clip_geometry(self):
        """Get geometry for raster clipping (Shapely)."""
        if self.use_coordinates:
            return self.bbox.to_shapely_polygon()
        if self._place_gdf is None:
            self._get_place_geojson()
        return self._place_gdf.geometry.iloc[0]

    def _get_location_slug(self) -> str:
        """Create a stable location slug for filenames."""
        if self.city:
            return (
                self.city.replace(" ", "_")
                .replace(",", "")
                .replace("/", "_")
            )
        if self.use_coordinates and self.bbox:
            return (
                f"bbox_{self.bbox.west}_{self.bbox.south}_"
                f"{self.bbox.east}_{self.bbox.north}"
            )
        return "area"

    # ------------------------------------------------------------------ #
    # Download helpers                                                   #
    # ------------------------------------------------------------------ #

    def _download_raster_file(
        self,
        url: str,
        output_path: Path,
        chunk_size: int = 8192,
        max_retries: int = 3,
    ) -> Path:
        """Download raster file from URL with retry on timeout."""
        self.logger.info(f"  → Downloading: {Path(url).name}")

        for attempt in range(1, max_retries + 1):
            try:
                response = self.session.get(url, stream=True, timeout=120)
                response.raise_for_status()

                total_size = int(response.headers.get("content-length", 0))
                downloaded = 0

                with open(output_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0 and downloaded % (10 * 1024 * 1024) < chunk_size:
                            progress = downloaded / total_size * 100
                            self.logger.info(f"    Progress: {progress:.1f}%")

                size_mb = output_path.stat().st_size / (1024 * 1024)
                self.logger.info(f"  ✓ Downloaded {size_mb:.1f} MB")
                return output_path

            except requests.exceptions.Timeout as e:
                if output_path.exists():
                    output_path.unlink()
                if attempt < max_retries:
                    wait = 5 * (2 ** (attempt - 1))  # 5s, 10s, 20s
                    self.logger.warning(
                        f"  ⚠ Timeout (attempt {attempt}/{max_retries}), retrying in {wait}s..."
                    )
                    time.sleep(wait)
                else:
                    self.logger.error(f"Download failed after {max_retries} attempts: {e}")
                    raise

            except requests.exceptions.RequestException as e:
                if output_path.exists():
                    output_path.unlink()
                self.logger.error(f"Download failed: {e}")
                raise

    def _clip_raster_to_extent(
        self,
        raster_path: Path,
        geometry,
        output_path: Path,
    ) -> Path:
        """Clip raster to study area geometry."""
        self.logger.info("  → Clipping raster...")

        try:
            with rasterio.open(raster_path) as src:
                out_image, out_transform = mask(src, [geometry], crop=True, filled=True)

                out_meta = src.meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform,
                    "compress": "lzw",
                })

                with rasterio.open(output_path, "w", **out_meta) as dest:
                    dest.write(out_image)

            size_mb = output_path.stat().st_size / (1024 * 1024)
            self.logger.info(f"  ✓ Clipped raster saved ({size_mb:.1f} MB)")
            return output_path
        except Exception as e:
            self.logger.error(f"Clipping failed: {e}")
            # The raw raster is likely corrupt (truncated download); remove it
            # so the retry pass will re-download rather than reuse the bad file.
            if raster_path.exists():
                raster_path.unlink()
                self.logger.info(f"  ✗ Removed corrupt raster: {raster_path.name}")
            raise

    # ------------------------------------------------------------------ #
    # Age/sex 100m raster download (R2025A)                              #
    # ------------------------------------------------------------------ #

    def _download_age_sex_rasters_100m(self, year: int) -> Dict[str, Path]:
        """
        Download 100m age/sex rasters using the R2025A Global_2015_2030 release.

        Base:
          {WORLDPOP_HTTP_BASE}/AgeSex_structures/Global_2015_2030/R2025A/{year}/{ISO3}/v1/100m/{model}/
        Filename:
          {iso3_lower}_{sex}_{agecode}_{year}_{model_code}_100m_R2025A_v1.tif
        """
        iso3 = self._get_country_iso3()

        if year not in self.AGE_SEX_100M_YEARS:
            self.logger.warning(
                f"100m age/sex rasters only available for "
                f"{self.AGE_SEX_100M_YEARS[0]}–{self.AGE_SEX_100M_YEARS[-1]}, requested {year}"
            )
            return {}

        model = "constrained" if self.constrained else "unconstrained"
        model_code = "CN" if self.constrained else "UN"

        base_url = (
            f"{self.WORLDPOP_HTTP_BASE}/AgeSex_structures/Global_2015_2030/"
            f"R2025A/{year}/{iso3.upper()}/v1/100m/{model}"
        )

        self.logger.info(
            f"  → Downloading 100m age/sex rasters for {iso3} {year} "
            f"(model={model}, 36 files: 18 age groups × 2 sexes)"
        )

        clip_geometry = self._get_clip_geometry()
        downloaded: Dict[str, Path] = {}
        failed: List[str] = []

        location_slug = self._get_location_slug()

        # (url, raster_path, clipped_path, key) for items that need a retry pass
        pending_retry: List[tuple] = []

        for age_group in self.AGE_GROUPS:
            age_code = self.AGE_CODE_MAP[age_group]

            for sex_code, sex_name in [("f", "female"), ("m", "male")]:
                filename = (
                    f"{iso3.lower()}_{sex_code}_{age_code}_"
                    f"{year}_{model_code}_100m_R2025A_v1.tif"
                )
                url = f"{base_url}/{filename}"
                raster_path = self.output_dir / filename
                clipped_filename = (
                    f"{iso3.lower()}_{sex_code}_{age_code}_"
                    f"{year}_{location_slug}.tif"
                )
                clipped_path = self.output_dir / clipped_filename
                key = f"{sex_name}_{age_group}"

                try:
                    if not raster_path.exists():
                        self._download_raster_file(url, raster_path)

                    self._clip_raster_to_extent(raster_path, clip_geometry, clipped_path)
                    downloaded[key] = clipped_path
                except Exception as e:
                    self.logger.warning(f"  ⚠ Failed {filename}: {e}")
                    failed.append(key)
                    pending_retry.append((url, raster_path, clipped_path, key))

        # Retry pass: come back to all failed downloads after completing the others
        if pending_retry:
            self.logger.info(
                f"  ↻ Retrying {len(pending_retry)} failed download(s) "
                f"(pausing 10s before retry pass)..."
            )
            time.sleep(10)
            still_failed: List[str] = []
            for url, raster_path, clipped_path, key in pending_retry:
                try:
                    # Always re-download: the file may be corrupt (truncated stream)
                    if raster_path.exists():
                        raster_path.unlink()
                    self._download_raster_file(url, raster_path)
                    self._clip_raster_to_extent(raster_path, clip_geometry, clipped_path)
                    downloaded[key] = clipped_path
                    failed.remove(key)
                except Exception as e:
                    self.logger.warning(f"  ⚠ Retry failed {raster_path.name}: {e}")
                    still_failed.append(key)
            recovered = len(pending_retry) - len(still_failed)
            if recovered:
                self.logger.info(f"  ✓ Recovered {recovered}/{len(pending_retry)} on retry pass")
            failed = still_failed

        total = len(self.AGE_GROUPS) * 2
        ok = len(downloaded)
        self.logger.info(f"  ✓ Downloaded {ok}/{total} age/sex rasters")

        if failed:
            self.logger.warning(f"  ⚠ Failed: {len(failed)} groups")
            preview = ", ".join(failed[:5])
            if len(failed) > 5:
                preview += ", ..."
            self.logger.warning(f"     Missing: {preview}")

        if ok == 0:
            self.logger.error(
                f"  ✗ No age/sex rasters downloaded for {iso3} {year}\n"
                f"     This most likely means AgeSex_structures R2025A is not "
                f"published for this country/year.\n"
                f"     Check: https://hub.worldpop.org/geodata/listing?id=138"
            )

        return downloaded

    def _aggregate_age_groups(
        self,
        pop_data: PopulationData,
        iso3: str,
        year: int,
    ) -> None:
        """
        Aggregate age/sex rasters into 3 age bands per sex and write
        iso3_sex_ageband_YEAR_location_slug.tif files.

        Age bands (by WorldPop age groups):

          - 0–14 years: 0-1, 1-4, 5-9, 10-14
          - 15–64 years: 15-19, 20-24, 25-29, 30-34, 35-39,
                         40-44, 45-49, 50-54, 55-59, 60-64
          - 65+ years: 65-69, 70-74, 75-79, 80+
        """
        if not pop_data.age_sex_rasters:
            return

        location_slug = self._get_location_slug()

        groups = {
            "0-14": ["0-1", "1-4", "5-9", "10-14"],
            "15-64": ["15-19", "20-24", "25-29", "30-34", "35-39",
                      "40-44", "45-49", "50-54", "55-59", "60-64"],
            "65+": ["65-69", "70-74", "75-79", "80+"],
        }

        # Create 6 rasters: 3 age bands × 2 sexes
        for sex_code, sex_name in [("f", "female"), ("m", "male")]:
            for group_label, age_groups in groups.items():
                # Collect all rasters for this sex and age band
                paths: List[Path] = []
                for ag in age_groups:
                    key = f"{sex_name}_{ag}"
                    if key in pop_data.age_sex_rasters:
                        paths.append(pop_data.age_sex_rasters[key])

                if not paths:
                    self.logger.warning(
                        f"  ⚠ No rasters found for {sex_name} {group_label} ({year})"
                    )
                    continue

                sum_data = None
                all_nodata_mask = None
                profile = None
                common_nodata = None

                for idx, p in enumerate(paths):
                    with rasterio.open(p) as src:
                        data = src.read(1).astype("float64")
                        nodata = src.nodata

                        if idx == 0:
                            profile = src.profile
                            common_nodata = nodata

                        if nodata is not None:
                            layer_mask = (data == nodata)
                            data = np.where(layer_mask, 0.0, data)
                        else:
                            layer_mask = np.zeros_like(data, dtype=bool)

                        if sum_data is None:
                            sum_data = data
                            all_nodata_mask = layer_mask
                        else:
                            sum_data += data
                            all_nodata_mask &= layer_mask

                if sum_data is None or profile is None:
                    continue

                # Restore nodata where all contributing layers were nodata
                if common_nodata is not None and all_nodata_mask is not None:
                    sum_data = np.where(all_nodata_mask, common_nodata, sum_data)

                profile.update(
                    count=1,
                    dtype="float32",
                    compress="lzw",
                )

                out_filename = (
                    f"{iso3.lower()}_{sex_code}_{group_label}_{year}_{location_slug}.tif"
                )
                out_path = self.output_dir / out_filename

                with rasterio.open(out_path, "w", **profile) as dst:
                    dst.write(sum_data.astype("float32"), 1)

                self.logger.info(
                    f"  ✓ Aggregated {sex_name} {group_label} raster written: {out_filename}"
                )

                # Store in memory with a clear key
                agg_key = f"{sex_name}_{group_label}"
                pop_data.age_sex_rasters[agg_key] = out_path

    # ------------------------------------------------------------------ #
    # WorldPop API for age/sex statistics                                #
    # ------------------------------------------------------------------ #

    def _query_worldpop_api(
        self,
        geojson: Dict,
        year: int,
        poll_interval: int = 2,
        max_wait: int = 300,
    ) -> Dict:
        """Query WorldPop API for age/sex disaggregated statistics."""
        url = f"{self.WORLDPOP_API_BASE}/services/stats"

        params = {
            "dataset": "wpgpas",
            "year": year,
            "geojson": json.dumps(geojson),
            "runasync": "true",
        }

        if self.api_key:
            params["key"] = self.api_key

        self.logger.info(f"  → Querying API for age/sex data ({year})...")

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            result = response.json()

            if result.get("status") == "created":
                task_id = result["taskid"]
                result = self._poll_task(task_id, poll_interval, max_wait)

            return result
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            raise

    def _poll_task(self, task_id: str, poll_interval: int, max_wait: int) -> Dict:
        """Poll asynchronous API task until completion."""
        url = f"{self.WORLDPOP_API_BASE}/tasks/{task_id}"
        elapsed = 0

        self.logger.info(f"  → Task ID: {task_id}, polling...")

        while elapsed < max_wait:
            time.sleep(poll_interval)
            elapsed += poll_interval

            try:
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                result = response.json()

                if result["status"] == "finished":
                    self.logger.info(f"  ✓ Task completed ({elapsed}s)")
                    return result
                if result.get("error"):
                    raise RuntimeError(result.get("error_message", "Unknown error"))
            except requests.exceptions.RequestException:
                continue

        raise TimeoutError(f"Task {task_id} exceeded maximum wait time")

    def _parse_api_response(self, response: Dict, year: int) -> PopulationData:
        """Parse WorldPop API response into PopulationData."""
        try:
            data = response.get("data", {})
            total_pop = data.get("total_population", 0.0)
            pyramid_data = data.get("agesexpyramid", [])

            if pyramid_data:
                df = pd.DataFrame(pyramid_data)
                if "male" in df.columns and "female" in df.columns:
                    df["male"] = pd.to_numeric(df["male"], errors="coerce")
                    df["female"] = pd.to_numeric(df["female"], errors="coerce")
                    df["total"] = df["male"] + df["female"]
                age_sex_df = df

                if total_pop == 0 and "total" in age_sex_df.columns:
                    total_pop = age_sex_df["total"].sum()
                    self.logger.info(f"  ℹ Calculated total from pyramid: {total_pop:,.0f}")
            else:
                age_sex_df = None

            execution_time = response.get("executionTime", None)

            return PopulationData(
                year=year,
                total_population=total_pop,
                age_sex_pyramid=age_sex_df,
                execution_time=execution_time,
                raw_response=response,
            )
        except Exception as e:
            self.logger.error(f"Failed to parse API response: {e}")
            raise

    # ------------------------------------------------------------------ #
    # Main pipeline                                                      #
    # ------------------------------------------------------------------ #

    def load_all_data(self) -> Union[PopulationData, Dict[int, PopulationData]]:
        """
        Run full pipeline:

        - Query API for age/sex statistics (only for years where API supports it).
        - Download 100m total population rasters (2015–2030 R2025A, if enabled).
        - Download 100m age/sex rasters from R2025A (if disaggregate=True).
        - Aggregate age/sex rasters into 3 age bands per sex (if disaggregate=True).
        """
        self.logger.info("=" * 60)
        self.logger.info("ANTIPOMPEII DEMOGRAPHY LOADER")
        self.logger.info("=" * 60)

        if self.use_coordinates:
            self.logger.info(f"Query area: Bbox {self.bbox.to_bbox_tuple()}")
        else:
            self.logger.info(f"Query area: {self.city}")

        self.logger.info(f"Years: {self.years}")
        self.logger.info(f"Download rasters: {self.download_rasters}")
        self.logger.info(f"Download 100m age/sex rasters: {self.disaggregate}")
        self.logger.info(f"Model: {'Constrained' if self.constrained else 'Unconstrained'}")
        self.logger.info(f"Output dir: {self.output_dir}")

        start_time = time.time()

        try:
            geojson = self._get_query_geojson()
            clip_geometry = self._get_clip_geometry()

            if self.download_rasters or self.disaggregate:
                iso3 = self._get_country_iso3()
                self.logger.info(f"Country: {iso3}")
            else:
                iso3 = None  # for type checkers

            location_slug = self._get_location_slug()

            for year in self.years:
                year_start = time.time()
                self.logger.info(f"\nProcessing year {year}:")

                # 1) API statistics (only up to 2020)
                if year <= 2020:
                    # WorldPop API wpgpas only supports 2000–2020
                    try:
                        response = self._query_worldpop_api(geojson, year)
                        pop_data = self._parse_api_response(response, year)
                    except Exception as e:
                        self.logger.warning(
                            f"  ⚠ Failed to retrieve API statistics for {year}: {e}. "
                            f"Continuing with rasters only."
                        )
                        pop_data = PopulationData(
                            year=year,
                            total_population=0.0,
                            age_sex_pyramid=None,
                            execution_time=None,
                            raw_response=None,
                        )
                else:
                    # For 2021+ (e.g. 2026), stats API is not available, but rasters exist.
                    self.logger.warning(
                        f"  ⚠ WorldPop stats API dataset 'wpgpas' only supports years "
                        f"2000–2020; skipping API statistics for {year} and using rasters only."
                    )
                    pop_data = PopulationData(
                        year=year,
                        total_population=0.0,
                        age_sex_pyramid=None,
                        execution_time=None,
                        raw_response=None,
                    )

                # 2) Total population raster (2015–2030, Global_2015_2030 R2025A)
                if self.download_rasters and year in self.TOTAL_POP_YEARS and iso3 is not None:
                    try:
                        model = "constrained" if self.constrained else "unconstrained"
                        model_code = "CN" if self.constrained else "UN"

                        base_url = (
                            f"{self.WORLDPOP_HTTP_BASE}/Population/Global_2015_2030/"
                            f"R2025A/{year}/{iso3.upper()}/v1/100m/{model}"
                        )

                        raster_filename = f"{iso3.lower()}_pop_{year}_{model_code}_100m_R2025A_v1.tif"
                        raster_url = f"{base_url}/{raster_filename}"
                        raster_path = self.output_dir / raster_filename

                        if not raster_path.exists():
                            self._download_raster_file(raster_url, raster_path)
                        else:
                            self.logger.info(f"  → Using cached: {raster_path.name}")

                        pop_data.raster_path = raster_path

                        clipped_filename = f"{iso3.lower()}_pop_{year}_{location_slug}.tif"
                        clipped_path = self.output_dir / clipped_filename
                        self._clip_raster_to_extent(raster_path, clip_geometry, clipped_path)
                        pop_data.raster_clipped_path = clipped_path

                        # Metadata and total population from clipped raster
                        with rasterio.open(clipped_path) as src:
                            pop_data.raster_metadata = {
                                "bounds": src.bounds,
                                "crs": str(src.crs),
                                "shape": src.shape,
                                "resolution": src.res,
                                "nodata": src.nodata,
                            }

                            # If API does not provide total (e.g. 2026), compute from raster
                            if pop_data.total_population == 0.0:
                                arr = src.read(1).astype(float)
                                nodata = src.nodata
                                if nodata is not None:
                                    arr = np.where(arr == nodata, np.nan, arr)
                                total_from_raster = float(np.nansum(arr))
                                pop_data.total_population = total_from_raster
                                self.logger.info(
                                    f"  ℹ Calculated total from clipped raster: "
                                    f"{total_from_raster:,.0f}"
                                )
                    except Exception as e:
                        self.logger.error(f"Total raster failed: {e}")
                        self.logger.info("Continuing with API data only...")
                elif self.download_rasters and year not in self.TOTAL_POP_YEARS:
                    self.logger.warning(
                        f"  ⚠ 100m total population R2025A counts not available for {year} "
                        f"(only {self.TOTAL_POP_YEARS[0]}–{self.TOTAL_POP_YEARS[-1]})"
                    )

                # 3) Age/sex 100m rasters (no synthetic disaggregation)
                if self.disaggregate and year in self.AGE_SEX_100M_YEARS:
                    try:
                        pop_data.age_sex_rasters = self._download_age_sex_rasters_100m(year)

                        # 4) Aggregate into 3 age bands per sex
                        if iso3 is not None and pop_data.age_sex_rasters:
                            self._aggregate_age_groups(pop_data, iso3, year)

                    except Exception as e:
                        self.logger.error(f"Age/sex rasters failed: {e}")
                        self.logger.info("Continuing with API data only...")
                elif self.disaggregate and year not in self.AGE_SEX_100M_YEARS:
                    self.logger.warning(
                        f"  ⚠ 100m age/sex rasters not available for {year} "
                        f"(only {self.AGE_SEX_100M_YEARS[0]}–{self.AGE_SEX_100M_YEARS[-1]})"
                    )

                self.population_data[year] = pop_data

                year_time = time.time() - year_start
                self._download_times[f"year_{year}"] = year_time
                self.logger.info(
                    f"✓ Year {year} complete: {pop_data.total_population:,.0f} people "
                    f"({year_time:.2f}s)"
                )

            total_time = time.time() - start_time
            self._download_times["total"] = total_time

            self.logger.info("\n" + "=" * 60)
            self.logger.info("DEMOGRAPHY LOADER: Complete")
            self.logger.info(f"Total time: {total_time:.2f}s")
            self.logger.info(f"Years processed: {len(self.population_data)}")
            self.logger.info("=" * 60)

            if len(self.years) == 1:
                return self.population_data[self.years[0]]
            return self.population_data

        except Exception as e:
            self.logger.error(f"Data loading failed: {e}", exc_info=True)
            raise

    # ------------------------------------------------------------------ #
    # Public helpers                                                     #
    # ------------------------------------------------------------------ #

    def get_summary_dataframe(self) -> pd.DataFrame:
        """Summary of all downloaded population data."""
        if not self.population_data:
            return pd.DataFrame({"Message": ["No data available"]})

        summary_rows = []
        for year, pop_data in sorted(self.population_data.items()):
            row = {
                "Year": year,
                "Total Population": f"{pop_data.total_population:,.0f}",
                "Has Age/Sex Data": pop_data.age_sex_pyramid is not None,
                "Has Total Raster": pop_data.raster_clipped_path is not None,
                "Has Age/Sex Rasters": len(pop_data.age_sex_rasters) > 0,
                "Age/Sex Raster Count": len(pop_data.age_sex_rasters),
            }

            if pop_data.raster_metadata:
                row["Raster Shape"] = f"{pop_data.raster_metadata['shape']}"
                row["Resolution (deg)"] = f"{pop_data.raster_metadata['resolution']}"

            summary_rows.append(row)

        return pd.DataFrame(summary_rows)

    def get_raster_paths(self, year: int) -> Dict[str, Path]:
        """Get all raster file paths for a specific year."""
        if year not in self.population_data:
            return {}

        pop_data = self.population_data[year]
        paths: Dict[str, Path] = {}

        if pop_data.raster_path:
            paths["total_population_full"] = pop_data.raster_path
        if pop_data.raster_clipped_path:
            paths["total_population_clipped"] = pop_data.raster_clipped_path

        # Includes per-age rasters and aggregated 0-14 / 15-64 / 65+ rasters
        paths.update(pop_data.age_sex_rasters)
        return paths

    def get_performance_summary(self) -> pd.DataFrame:
        """Return timing information for major operations."""
        if not self._download_times:
            return pd.DataFrame({"Message": ["No performance data"]})

        rows = []
        for key, duration in self._download_times.items():
            rows.append({"Operation": key, "Duration (seconds)": round(duration, 2)})
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Convenience function used by CLI
# ---------------------------------------------------------------------------

def load_enhanced_from_cli(
    city: str,
    use_coordinates: bool,
    timestamps: Optional[List[str]] = None,
    coordinates: Optional[Dict] = None,
    download_rasters: bool = True,
    disaggregate: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Union[PopulationData, Dict[int, PopulationData]]:
    """Thin wrapper for CLI integration."""
    loader = PopulationLoaderEnhanced.from_cli_params(
        city=city,
        use_coordinates=use_coordinates,
        timestamps=timestamps,
        long_min=coordinates.get("long_min") if coordinates else None,
        lat_min=coordinates.get("lat_min") if coordinates else None,
        long_max=coordinates.get("long_max") if coordinates else None,
        lat_max=coordinates.get("lat_max") if coordinates else None,
        logger=logger,
        download_rasters=download_rasters,
        disaggregate=disaggregate,
    )
    return loader.load_all_data()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    log = logging.getLogger(__name__)

    print("ANTIPOMPEII Demography Loader")
    print("=" * 60)

    loader = PopulationLoaderEnhanced(
        city="Laxenburg, Austria",
        years=None,  # will default to current year
        logger=log,
        download_rasters=True,
        disaggregate=True,
    )

    data = loader.load_all_data()

    print("\nSummary:")
    print(loader.get_summary_dataframe().to_string(index=False))

    print("\nRaster Paths:")
    for name, path in loader.get_raster_paths(loader.years[0]).items():
        print(f"  {name}: {path}")
