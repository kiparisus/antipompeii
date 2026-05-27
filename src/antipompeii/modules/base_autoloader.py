"""
ANTIPOMPEII Data Autoloader Module

Downloads street networks and building footprints from OpenStreetMap
using OSMnx, organizing data into thematic layers for urban vulnerability analysis.

Supports temporal data download using OSM attic data.
"""

from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

import osmnx as ox
import geopandas as gpd
import pandas as pd
from shapely.geometry import box

# Configure OSMnx settings for better performance
ox.settings.log_console = False
ox.settings.use_cache = True
ox.settings.timeout = 300  # Increase timeout for large queries
ox.settings.max_query_area_size = 50_000 * 50_000 * 50  # Increase max query area


@dataclass
class BoundingBox:
    """Represents a geographic bounding box."""

    west: float  # min longitude
    south: float  # min latitude
    east: float  # max longitude
    north: float  # max latitude

    def to_tuple(self) -> Tuple[float, float, float, float]:
        """Return as (north, south, east, west) for OSMnx."""
        return (self.north, self.south, self.east, self.west)

    def to_bbox_tuple(self) -> Tuple[float, float, float, float]:
        """Return as (west, south, east, north) for some operations."""
        return (self.west, self.south, self.east, self.north)


@dataclass
class LayerDefinition:
    """Defines a thematic building layer with OSM tags."""

    name: str
    sphere: str  # Methodology classification: Sociosphere, Orgsphere, or Technosphere
    tags: Dict[str, Union[str, List[str], bool]]
    description: str


class DataLoader:
    """
    Main data loading class for ANTIPOMPEII.
    """

    # Layer definitions
    LAYERS: Dict[str, LayerDefinition] = {
        "layer_1_social": LayerDefinition(
            name="Social",
            sphere="Sociosphere",
            tags={
                "building": [
                    "house",
                    "apartments",
                    "barracks",
                    "bungalow",
                    "cabin",
                    "detached",
                    "dormitory",
                    "farm",
                    "ger",
                    "hotel",
                    "houseboat",
                    "residential",
                    "semidetached_house",
                    "static_caravan",
                    "stilt_house",
                    "terrace",
                    "tree_house",
                    "trullo",
                    "hut",
                    "shed",
                ]
            },
            description="Residential and social housing structures",
        ),
        "layer_2_health": LayerDefinition(
            name="Health",
            sphere="Orgsphere",
            tags={
                "amenity": ["hospital", "clinic", "pharmacy"],
            },
            description="Healthcare facilities",
        ),
        "layer_3_emergency": LayerDefinition(
            name="Emergency",
            sphere="Orgsphere",
            tags={
                "amenity": ["police", "fire_station", "shelter", "bunker"],
                "emergency": ["ambulance_station", "bunker"],
            },
            description="Emergency response facilities",
        ),
        "layer_4_shelter": LayerDefinition(
            name="Convertible Shelter",
            sphere="Orgsphere",
            tags={
                "building": [
                    "religious",
                    "cathedral",
                    "chapel",
                    "church",
                    "kingdom_hall",
                    "monastery",
                    "mosque",
                    "presbytery",
                    "shrine",
                    "synagogue",
                    "temple",
                    "public",
                    "college",
                    "government",
                    "kindergarten",
                    "school",
                    "university",
                    "grandstand",
                    "pavilion",
                    "riding_hall",
                    "sports_hall",
                    "sports_centre",
                    "stadium",
                ],
                "leisure": ["park"],
            },
            description="Buildings convertible into emergency shelters",
        ),
        "layer_5_commercial": LayerDefinition(
            name="Commercial",
            sphere="Orgsphere",
            tags={
                "amenity": ["bank"],
                "building": [
                    "commercial",
                    "industrial",
                    "office",
                    "retail",
                    "supermarket",
                    "warehouse",
                    "kiosk",
                ],
            },
            description="Commercial and industrial structures",
        ),
        "layer_6_power": LayerDefinition(
            name="Power",
            sphere="Technosphere",
            tags={
                "power": True,  # Any power-related infrastructure
            },
            description="Power infrastructure",
        ),
        "layer_7_entertainment": LayerDefinition(
            name="Entertainment",
            sphere="Orgsphere",
            tags={
                "amenity": [
                    "arts_centre",
                    "casino",
                    "cinema",
                    "exhibition_centre",
                    "community_centre",
                    "events_venue",
                    "bar",
                    "gambling",
                    "music_venue",
                    "nightclub",
                    "planetarium",
                    "social_centre",
                    "theatre",
                ]
            },
            description="Entertainment and cultural venues",
        ),
    }

    # Group layers into a small number of broad key-based Overpass queries;
    # per-layer classification then runs locally on the returned features.
    OPTIMIZED_QUERY_GROUPS = {
        "buildings": {
            "name": "Buildings (Social, Shelter, Commercial)",
            "tags": {
                "building": True,  # query all building=*; classify by value in Python
            },
            "layers_included": ["layer_1_social", "layer_4_shelter", "layer_5_commercial"],
        },
        "amenities": {
            "name": "Amenities (Health, Emergency, Entertainment)",
            "tags": {
                "amenity": True,  # all amenity=*
            },
            "layers_included": [
                "layer_2_health",
                "layer_3_emergency",
                "layer_5_commercial",
                "layer_7_entertainment",
            ],
        },
        "infrastructure": {
            "name": "Infrastructure (Emergency, Shelter, Power)",
            "tags": {
                "emergency": True,  # all emergency=*
                "leisure": True,    # all leisure=*
                "power": True,      # all power=*
            },
            "layers_included": ["layer_3_emergency", "layer_4_shelter", "layer_6_power"],
        },
    }

    def __init__(
        self,
        city: Optional[str] = None,
        bbox: Optional[BoundingBox] = None,
        timestamps: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None,
        use_optimization: bool = True,
    ):
        """
        Initialize the DataLoader.
        """
        if city is None and bbox is None:
            raise ValueError("Either 'city' or 'bbox' must be provided")
        if city is not None and bbox is not None:
            raise ValueError("Provide either 'city' OR 'bbox', not both")

        self.city = city
        self.bbox = bbox
        self.use_coordinates = bbox is not None
        self.timestamps = timestamps or []
        self.use_temporal = len(self.timestamps) > 0
        self.use_optimization = use_optimization
        self.logger = logger or logging.getLogger(__name__)

        # Storage for downloaded data
        self.street_network_graph = None
        self.street_network_gdf: Optional[gpd.GeoDataFrame] = None
        self.building_layers: Dict[str, gpd.GeoDataFrame] = {}
        self.merged_data: Optional[gpd.GeoDataFrame] = None

        # Temporal data storage
        self.temporal_data: Dict[str, gpd.GeoDataFrame] = {}

        # Store original Overpass and cache settings
        self._original_overpass_settings = ox.settings.overpass_settings
        self._original_overpass_url = ox.settings.overpass_url
        self._original_use_cache = ox.settings.use_cache

        # Performance tracking
        self._download_times: Dict[str, float] = {}

    @classmethod
    def from_cli_params(
        cls,
        city: str,
        use_coordinates: bool,
        use_temporal: bool = False,
        timestamps: Optional[List[str]] = None,
        long_min: Optional[float] = None,
        lat_min: Optional[float] = None,
        long_max: Optional[float] = None,
        lat_max: Optional[float] = None,
        logger: Optional[logging.Logger] = None,
        use_optimization: bool = True,
    ) -> "DataLoader":
        """
        Create DataLoader from CLI parameters.
        """
        if use_coordinates:
            if None in [long_min, lat_min, long_max, lat_max]:
                raise ValueError(
                    "When use_coordinates=True, all coordinates must be provided: "
                    "long_min, lat_min, long_max, lat_max"
                )
            bbox = BoundingBox(west=long_min, south=lat_min, east=long_max, north=lat_max)
            return cls(
                bbox=bbox,
                timestamps=timestamps if use_temporal else None,
                logger=logger,
                use_optimization=use_optimization,
            )
        else:
            return cls(
                city=city,
                timestamps=timestamps if use_temporal else None,
                logger=logger,
                use_optimization=use_optimization,
            )

    @staticmethod
    def parse_timestamp(timestamp: str) -> str:
        """
        Parse and validate timestamp, converting YYYYMMDD to ISO format.

        Args:
            timestamp: Timestamp string in YYYYMMDD format

        Returns:
            ISO-formatted timestamp string (YYYY-MM-DDTHH:MM:SSZ)

        Raises:
            ValueError: If timestamp format is invalid
        """
        try:
            dt = datetime.strptime(timestamp, "%Y%m%d")
            return dt.strftime("%Y-%m-%dT00:00:00Z")
        except ValueError:
            raise ValueError(
                f"Invalid timestamp format: {timestamp}. "
                "Expected YYYYMMDD format (e.g., 20181203)"
            )

    def _set_temporal_overpass(self, timestamp: str) -> None:
        """Configure OSMnx to use Overpass attic data for a specific timestamp."""
        # Ensure we hit an attic-capable instance
        ox.settings.overpass_url = "https://overpass-api.de/api"

        # Use placeholders {timeout} and {maxsize} so OSMnx can fill them in
        ox.settings.overpass_settings = (
            f'[out:json][timeout:{{timeout}}][date:"{timestamp}"]{{maxsize}}'
        )
        self.logger.info(f"Overpass configured for historical date: {timestamp}")

    def _restore_overpass_settings(self) -> None:
        """Restore original Overpass and cache settings."""
        ox.settings.overpass_settings = self._original_overpass_settings
        ox.settings.overpass_url = self._original_overpass_url
        ox.settings.use_cache = self._original_use_cache

    def load_all_data(self) -> Union[gpd.GeoDataFrame, Dict[str, gpd.GeoDataFrame]]:
        """
        Execute the complete data loading pipeline.

        Returns:
            If temporal: Dict mapping timestamps to GeoDataFrames
            If not temporal: Single merged GeoDataFrame

        Raises:
            Exception: If data download or processing fails
        """
        self.logger.info("=" * 60)
        self.logger.info("ANTIPOMPEII AUTOLOADER: Starting data acquisition")
        self.logger.info("=" * 60)

        start_time = time.time()
        try:
            if self.use_temporal:
                result = self._load_temporal_data()
            else:
                result = self._load_current_data()

            total_time = time.time() - start_time
            self.logger.info(f"Total download time: {total_time:.2f} seconds")
            return result
        except Exception as e:
            self.logger.error(f"Data loading failed: {str(e)}", exc_info=True)
            raise
        finally:
            # Ensure settings are restored if something went wrong early
            self._restore_overpass_settings()

    def _load_current_data(self) -> gpd.GeoDataFrame:
        """Load current (non-temporal) data."""
        self.logger.info("Loading current state of the network...")

        # Step 1: Download street network
        self._download_street_network()

        # Step 2: Download buildings (optimized or standard)
        if self.use_optimization:
            self._download_buildings_optimized()
        else:
            self._download_all_building_layers()

        # Step 3: Merge all data
        self._merge_layers()

        self.logger.info("=" * 60)
        self.logger.info("AUTOLOADER: Data acquisition complete")
        self.logger.info(f"Total features loaded: {len(self.merged_data)}")
        self.logger.info("=" * 60)

        return self.merged_data

    def _load_temporal_data(self) -> Dict[str, gpd.GeoDataFrame]:
        """Load temporal (historical) data for multiple timestamps."""
        self.logger.info(
            f"Loading temporal data for {len(self.timestamps)} timestamps..."
        )

        # For temporal runs, do not use cache (we want real attic responses)
        ox.settings.use_cache = False

        try:
            for timestamp in self.timestamps:
                try:
                    iso_timestamp = self.parse_timestamp(timestamp)
                    self.logger.info("\n" + "=" * 60)
                    self.logger.info(
                        f"Processing timestamp: {timestamp} ({iso_timestamp})"
                    )
                    self.logger.info("=" * 60)

                    # Configure Overpass for this date
                    self._set_temporal_overpass(iso_timestamp)

                    # Download data for this timestamp
                    self._download_street_network(temporal_key=timestamp)
                    if self.use_optimization:
                        self._download_buildings_optimized(temporal_key=timestamp)
                    else:
                        self._download_all_building_layers(temporal_key=timestamp)

                    self._merge_layers(temporal_key=timestamp)
                    self.temporal_data[timestamp] = self.merged_data

                    self.logger.info(
                        f"✓ Timestamp {timestamp}: {len(self.merged_data)} features loaded"
                    )

                    # Reset for next timestamp
                    self.street_network_gdf = None
                    self.building_layers = {}
                    self.merged_data = None

                except Exception as e:
                    self.logger.error(
                        f"Failed to load data for timestamp {timestamp}: {str(e)}"
                    )
                    continue
        finally:
            # Restore everything after the entire temporal sequence
            self._restore_overpass_settings()
            self.logger.info("\n" + "=" * 60)
            self.logger.info("AUTOLOADER: Temporal data acquisition complete")
            self.logger.info(
                f"Successfully loaded {len(self.temporal_data)} temporal snapshots"
            )
            self.logger.info("=" * 60)

        return self.temporal_data

    def _download_buildings_optimized(self, temporal_key: Optional[str] = None) -> None:
        """
        Download buildings and related objects using combined queries.

        Args:
            temporal_key: Optional timestamp key for temporal data
        """
        prefix = f"[{temporal_key}] " if temporal_key else ""
        self.logger.info(f"{prefix}Downloading building layers ...")

        start_time = time.time()
        for group_id, group_def in self.OPTIMIZED_QUERY_GROUPS.items():
            try:
                self.logger.info(f"{prefix} → {group_def['name']}...")

                # Download combined data
                if self.use_coordinates:
                    gdf = ox.features_from_bbox(
                        bbox=self.bbox.to_tuple(),
                        tags=group_def["tags"],
                    )
                else:
                    gdf = ox.features_from_place(
                        query=self.city,
                        tags=group_def["tags"],
                    )

                if gdf.empty:
                    self.logger.warning(
                        f"{prefix} No features found for {group_def['name']}"
                    )
                    continue

                # Split the combined data into individual layers
                self._classify_features_to_layers(
                    gdf, group_def["layers_included"], temporal_key
                )
                self.logger.info(f"{prefix} ✓ Downloaded {len(gdf)} features")

            except Exception as e:
                self.logger.warning(
                    f"{prefix} Failed to download {group_def['name']}: {str(e)}"
                )
                continue

        download_time = time.time() - start_time
        self._download_times["buildings_optimized"] = download_time
        self.logger.info(
            f"{prefix}Buildings download completed in {download_time:.2f} seconds"
        )

    def _classify_features_to_layers(
        self,
        gdf: gpd.GeoDataFrame,
        layer_ids: List[str],
        temporal_key: Optional[str] = None,
    ) -> None:
        """
        Classify downloaded features into appropriate layers based on their attributes.

        Args:
            gdf: GeoDataFrame with mixed features
            layer_ids: List of layer IDs to classify into
            temporal_key: Optional timestamp key
        """
        for layer_id in layer_ids:
            if layer_id not in self.LAYERS:
                continue

            layer_def = self.LAYERS[layer_id]

            # Filter features matching this layer's tags
            mask = pd.Series([False] * len(gdf), index=gdf.index)
            for tag_key, tag_values in layer_def.tags.items():
                if tag_key not in gdf.columns:
                    continue

                if isinstance(tag_values, bool):
                    # For boolean tags like power=True
                    mask |= gdf[tag_key].notna()
                elif isinstance(tag_values, list):
                    # For list of values
                    mask |= gdf[tag_key].isin(tag_values)
                else:
                    # For single value
                    mask |= gdf[tag_key] == tag_values

            if mask.any():
                layer_gdf = gdf[mask].copy()
                layer_gdf["layer"] = layer_id
                layer_gdf["layer_name"] = layer_def.name
                layer_gdf["sphere"] = layer_def.sphere
                # Everything in these groups is treated as "building" type
                layer_gdf["feature_type"] = "building"

                if temporal_key:
                    layer_gdf["timestamp"] = temporal_key

                # Merge with existing layer data if present
                if layer_id in self.building_layers:
                    self.building_layers[layer_id] = pd.concat(
                        [self.building_layers[layer_id], layer_gdf],
                        ignore_index=True,
                    )
                else:
                    self.building_layers[layer_id] = layer_gdf

    def _download_street_network(self, temporal_key: Optional[str] = None) -> None:
        """Download street network and convert to GeoDataFrame."""
        prefix = f"[{temporal_key}] " if temporal_key else ""
        self.logger.info(f"{prefix}Downloading street network...")

        start_time = time.time()
        try:
            if self.use_coordinates:
                self.logger.info(f"{prefix}Using bounding box: {self.bbox.to_tuple()}")
                graph = ox.graph_from_bbox(
                    bbox=self.bbox.to_tuple(),
                    network_type="all",
                    simplify=True,
                    retain_all=False,
                )
            else:
                self.logger.info(f"{prefix}Using place name: {self.city}")
                graph = ox.graph_from_place(
                    query=self.city,
                    network_type="all",
                    simplify=True,
                    retain_all=False,
                )

            self.street_network_graph = graph
            gdf_nodes, gdf_edges = ox.graph_to_gdfs(graph)

            self.street_network_gdf = gdf_edges.copy()
            self.street_network_gdf["layer"] = "street_network"
            self.street_network_gdf["layer_name"] = "Street Network"
            self.street_network_gdf["sphere"] = "Technosphere"  # Streets are infrastructure
            self.street_network_gdf["feature_type"] = "street"

            if temporal_key:
                self.street_network_gdf["timestamp"] = temporal_key

            download_time = time.time() - start_time
            key = f"streets_{temporal_key or 'current'}"
            self._download_times[key] = download_time

            self.logger.info(
                f"{prefix}✓ Street network downloaded: {len(gdf_edges)} edges, "
                f"{len(gdf_nodes)} nodes ({download_time:.2f}s)"
            )
        except Exception as e:
            self.logger.error(f"{prefix}Failed to download street network: {str(e)}")
            raise

    def _download_all_building_layers(
        self, temporal_key: Optional[str] = None
    ) -> None:
        """Download all building layers sequentially, one Overpass query per layer."""
        prefix = f"[{temporal_key}] " if temporal_key else ""
        self.logger.info(
            f"{prefix}Downloading building layers (one query per layer, "
            f"{len(self.LAYERS)} layers)..."
        )

        start_time = time.time()
        for layer_id, layer_def in self.LAYERS.items():
            try:
                self._download_building_layer(layer_id, layer_def, temporal_key)
            except Exception as e:
                self.logger.warning(
                    f"{prefix}Failed to download {layer_def.name} ({layer_id}): {str(e)}"
                )
                continue

        download_time = time.time() - start_time
        self._download_times["buildings_standard"] = download_time
        self.logger.info(
            f"{prefix}Buildings download completed in {download_time:.2f} seconds"
        )

    def _download_building_layer(
        self,
        layer_id: str,
        layer_def: LayerDefinition,
        temporal_key: Optional[str] = None,
    ) -> None:
        """Download a single building layer."""
        prefix = f"[{temporal_key}] " if temporal_key else ""
        self.logger.info(f"{prefix} → {layer_def.name} ({layer_id})...")

        try:
            if self.use_coordinates:
                gdf = ox.features_from_bbox(
                    bbox=self.bbox.to_tuple(),
                    tags=layer_def.tags,
                )
            else:
                gdf = ox.features_from_place(
                    query=self.city,
                    tags=layer_def.tags,
                )

            if gdf.empty:
                self.logger.warning(
                    f"{prefix} No features found for {layer_def.name}"
                )
                return

            gdf["layer"] = layer_id
            gdf["layer_name"] = layer_def.name
            gdf["sphere"] = layer_def.sphere
            gdf["feature_type"] = "building"

            if temporal_key:
                gdf["timestamp"] = temporal_key

            self.building_layers[layer_id] = gdf
            self.logger.info(f"{prefix} ✓ Downloaded {len(gdf)} features")
        except Exception as e:
            self.logger.error(
                f"{prefix} ✗ Error downloading {layer_def.name}: {str(e)}"
            )
            raise

    def _merge_layers(self, temporal_key: Optional[str] = None) -> None:
        """Merge street network and all building/amenity layers into a single GeoDataFrame."""
        prefix = f"[{temporal_key}] " if temporal_key else ""
        self.logger.info(f"{prefix}Merging all layers...")

        all_gdfs: List[gpd.GeoDataFrame] = []

        if self.street_network_gdf is not None and not self.street_network_gdf.empty:
            all_gdfs.append(self.street_network_gdf)
            self.logger.info(
                f"{prefix} → Street network: {len(self.street_network_gdf)} features"
            )

        for layer_id, gdf in self.building_layers.items():
            if gdf is not None and not gdf.empty:
                all_gdfs.append(gdf)
                self.logger.info(f"{prefix} → {layer_id}: {len(gdf)} features")

        if not all_gdfs:
            raise ValueError("No data available to merge")

        try:
            self.merged_data = gpd.GeoDataFrame(
                pd.concat(all_gdfs, ignore_index=True, sort=False)
            )

            if self.merged_data.crs is None:
                self.merged_data.set_crs(epsg=4326, inplace=True)

            self.logger.info(
                f"{prefix}✓ Successfully merged {len(self.merged_data)} total features"
            )
            self.logger.info(f"{prefix} CRS: {self.merged_data.crs}")
        except Exception as e:
            self.logger.error(f"{prefix}Failed to merge layers: {str(e)}")
            raise

    def _sanitize_columns_for_gpkg(
        self, gdf: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """Rename columns so GDAL's GPKG writer accepts them.
        """
        geom_col = gdf.geometry.name
        invalid = re.compile(r"[^\w:]+", re.UNICODE)

        renames: Dict[str, str] = {}
        used: set = {geom_col}
        for col in gdf.columns:
            if col == geom_col:
                continue
            cleaned = invalid.sub("_", str(col))
            cleaned = re.sub(r"_+", "_", cleaned).strip("_")
            if not cleaned:
                cleaned = "col"
            base = cleaned
            n = 1
            while cleaned in used:
                n += 1
                cleaned = f"{base}_{n}"
            used.add(cleaned)
            if cleaned != col:
                renames[col] = cleaned

        if not renames:
            return gdf

        self.logger.info(
            f"Sanitized {len(renames)} GPKG-incompatible column name(s); "
            f"sample: {dict(list(renames.items())[:3])}"
        )
        return gdf.rename(columns=renames)

    def save_data(self, output_path: Union[str, Path], format: str = "gpkg") -> None:
        """Save merged data to file."""
        output_path = Path(output_path)
        try:
            if self.use_temporal:
                if not self.temporal_data:
                    raise ValueError(
                        "No temporal data to save. Run load_all_data() first."
                    )

                output_dir = output_path if output_path.is_dir() else output_path.parent
                output_dir.mkdir(parents=True, exist_ok=True)

                for timestamp, gdf in self.temporal_data.items():
                    city_safe = (self.city or "area").replace(" ", "_").replace(",", "")
                    filename = f"{timestamp}_{city_safe}.{format}"
                    file_path = output_dir / filename

                    if format.lower() == "gpkg":
                        gdf = self._sanitize_columns_for_gpkg(gdf)
                        self.temporal_data[timestamp] = gdf
                        gdf.to_file(file_path, driver="GPKG")
                    elif format.lower() == "geojson":
                        gdf.to_file(file_path, driver="GeoJSON")
                    elif format.lower() == "shp":
                        gdf.to_file(file_path, driver="ESRI Shapefile")
                    else:
                        raise ValueError(f"Unsupported format: {format}")

                    self.logger.info(f"✓ Saved timestamp {timestamp} to {file_path}")
            else:
                if self.merged_data is None:
                    raise ValueError(
                        "No data to save. Run load_all_data() first."
                    )

                if format.lower() == "gpkg":
                    self.merged_data = self._sanitize_columns_for_gpkg(self.merged_data)
                    self.merged_data.to_file(output_path, driver="GPKG")
                elif format.lower() == "geojson":
                    self.merged_data.to_file(output_path, driver="GeoJSON")
                elif format.lower() == "shp":
                    self.merged_data.to_file(output_path, driver="ESRI Shapefile")
                else:
                    raise ValueError(f"Unsupported format: {format}")

                self.logger.info(f"✓ Data saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save data: {str(e)}")
            raise

    def get_layer_summary(self, timestamp: Optional[str] = None) -> pd.DataFrame:
        """Generate a summary of downloaded layers."""
        summary_data = []

        if timestamp and timestamp in self.temporal_data:
            data_source = self.temporal_data[timestamp]
        elif self.merged_data is not None:
            data_source = self.merged_data
        else:
            return pd.DataFrame({"Message": ["No data available"]})

        if "layer" in data_source.columns:
            grouped = data_source.groupby("layer")
            for layer_id, group in grouped:
                layer_name = (
                    group["layer_name"].iloc[0]
                    if "layer_name" in group.columns
                    else layer_id
                )
                sphere = (
                    group["sphere"].iloc[0]
                    if "sphere" in group.columns
                    else "N/A"
                )
                summary_data.append(
                    {
                        "Layer": layer_id,
                        "Name": layer_name,
                        "Sphere": sphere,
                        "Feature Count": len(group),
                        "Geometry Types": ", ".join(group.geometry.type.unique()),
                    }
                )

        df = pd.DataFrame(summary_data)
        if not df.empty:
            df = df.sort_values("Layer")
        return df

    def get_temporal_summary(self) -> pd.DataFrame:
        """Generate a summary of all temporal snapshots."""
        if not self.temporal_data:
            return pd.DataFrame({"Message": ["No temporal data available"]})

        summary_data = []
        for timestamp, gdf in self.temporal_data.items():
            summary_data.append(
                {
                    "Timestamp": timestamp,
                    "Total Features": len(gdf),
                    "Street Features": len(gdf[gdf["feature_type"] == "street"])
                    if "feature_type" in gdf.columns
                    else 0,
                    "Building Features": len(gdf[gdf["feature_type"] == "building"])
                    if "feature_type" in gdf.columns
                    else 0,
                    "CRS": str(gdf.crs),
                }
            )

        return pd.DataFrame(summary_data).sort_values("Timestamp")

    def get_performance_summary(self) -> pd.DataFrame:
        """Get download performance statistics."""
        if not self._download_times:
            return pd.DataFrame({"Message": ["No performance data available"]})

        perf_data = []
        for key, duration in self._download_times.items():
            perf_data.append(
                {"Operation": key, "Duration (seconds)": round(duration, 2)}
            )
        return pd.DataFrame(perf_data)


# Convenience functions
def load_from_coordinates(
    west: float,
    south: float,
    east: float,
    north: float,
    timestamps: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None,
    use_optimization: bool = True,
) -> Union[gpd.GeoDataFrame, Dict[str, gpd.GeoDataFrame]]:
    """Load data using bounding box coordinates."""
    bbox = BoundingBox(west=west, south=south, east=east, north=north)
    loader = DataLoader(
        bbox=bbox,
        timestamps=timestamps,
        logger=logger,
        use_optimization=use_optimization,
    )
    return loader.load_all_data()


def load_from_city(
    city: str,
    timestamps: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None,
    use_optimization: bool = True,
) -> Union[gpd.GeoDataFrame, Dict[str, gpd.GeoDataFrame]]:
    """Load data using city/place name."""
    loader = DataLoader(
        city=city,
        timestamps=timestamps,
        logger=logger,
        use_optimization=use_optimization,
    )
    return loader.load_all_data()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logger = logging.getLogger(__name__)

    print("Example: Loading data with optimization")
    loader = DataLoader.from_cli_params(
        city="Laxenburg, Austria",
        use_coordinates=False,
        use_temporal=False,
        logger=logger,
        use_optimization=True,
    )

    data = loader.load_all_data()
    print(loader.get_layer_summary())

    print("\nPerformance:")
    print(loader.get_performance_summary())
