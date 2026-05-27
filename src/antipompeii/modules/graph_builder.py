"""
ANTIPOMPEII Graph-Tool Network Builder Module

Takes:
- Enriched GeoPackage(s) produced by the ANTIPOMPEII pipeline containing:
  * Street network segments (LineString geometries)
  * Facility attributes (fac_*, fac_count)
  * Population attributes (pop_total_*, pop_f_*, pop_m_*)
  * Disruption attributes (disruption)
  * All original OSM attributes

Does:
- Converts LineString street network to graph-tool Graph
- Preserves selected geometry and attributes as graph property maps
- Creates vertices from line endpoints with coordinate deduplication
- Stores edge geometries as flattened coordinate arrays
- Exports graph to .gt binary format for fast loading
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List
import logging
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString

from src.antipompeii.utils.graph_tool_compat import gt, GRAPH_TOOL_AVAILABLE
if GRAPH_TOOL_AVAILABLE:
    from graph_tool import Graph, PropertyMap


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

from src.antipompeii.utils.logger import get_module_logger as _get_logger


def _infer_graph_tool_type(dtype, sample_value=None) -> str:
    """
    Map pandas/numpy dtype to graph-tool property type.

    Valid graph-tool types: bool, int16_t, int32_t (int), int64_t (long),
    double, float, long double, string, python::object
    """
    dtype_str = str(dtype)

    if dtype_str in ['int64', 'Int64']:
        return 'long'
    elif dtype_str in ['int32', 'Int32']:
        return 'int'
    elif dtype_str in ['int16', 'Int16']:
        return 'int16_t'
    elif dtype_str in ['int8', 'Int8']:
        return 'int16_t'
    elif dtype_str in ['uint64', 'UInt64', 'uint32', 'UInt32',
                       'uint16', 'UInt16', 'uint8', 'UInt8']:
        return 'long'
    elif dtype_str in ['float64', 'Float64']:
        return 'double'
    elif dtype_str in ['float32', 'Float32']:
        return 'float'
    elif dtype_str == 'bool':
        return 'bool'
    elif dtype_str == 'object':
        if sample_value is not None:
            if isinstance(sample_value, (int, np.integer)):
                return 'long'
            elif isinstance(sample_value, (float, np.floating)):
                return 'double'
            elif isinstance(sample_value, bool):
                return 'bool'
            elif isinstance(sample_value, str):
                return 'string'
        return 'string'
    else:
        return 'string'


def _sanitize_column_name(col: str) -> str:
    """Sanitize column names for graph-tool (alphanumeric + underscore only)."""
    safe = col.replace(' ', '_').replace('-', '_').replace('.', '_').replace(':', '_')
    return ''.join(c for c in safe if c.isalnum() or c == '_')


def _round_coord(coord: Tuple[float, float], tolerance: float) -> Tuple[float, float]:
    """Round coordinate to tolerance for vertex deduplication."""
    decimals = int(-np.log10(tolerance))
    return (round(coord[0], decimals), round(coord[1], decimals))


def _filter_attributes(
    available_attrs: List[str],
    include_attrs: Optional[List[str]] = None
) -> List[str]:
    """
    Filter attributes based on a whitelist.
    """
    # Default whitelist (exact names)
    default_exact = {
        'fid',
        'osmid',
        'highway',
        'length',
        'bridge',
        'service',
        'tunnel',
        'layer_name',
        'sphere',
        'fac_layer_name',
        'fac_name',
        'fac_amenity',
        'fac_leisure',
        'disruption',
    }

    if include_attrs is None:
        filtered: List[str] = []
        for attr in available_attrs:
            if (
                attr in default_exact
                or attr.startswith('fac_')
                or attr.startswith('pop_')
                or attr.startswith('elev_')
                or attr.startswith('water_')
            ):
                filtered.append(attr)
        return filtered

    # Custom list provided: support exact matches and simple prefix '*'
    filtered: List[str] = []
    for attr in available_attrs:
        keep = False
        for pattern in include_attrs:
            if pattern.endswith('*'):
                prefix = pattern[:-1]
                if attr.startswith(prefix):
                    keep = True
                    break
            else:
                if attr == pattern:
                    keep = True
                    break
        if keep:
            filtered.append(attr)
    return filtered


# ---------------------------------------------------------------------------
# Graph Builder Class
# ---------------------------------------------------------------------------

class GraphBuilder:
    """
    Converts ANTIPOMPEII enriched street network GeoPackages to graph-tool networks.

    Preserves:
    - Full LineString geometry for each edge
    - Selected street attributes (OSM tags, facilities, population, disruption)
    - Vertex coordinates
    - Edge lengths
    """

    def __init__(
        self,
        tolerance: float = 1e-8,
        directed: bool = False,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize GraphBuilder.

        Parameters
        ----------
        tolerance : float, default 1e-8
            Coordinate tolerance for vertex deduplication (degrees or meters).
        directed : bool, default False
            Create directed graph (True) or undirected (False) graph.
        logger : logging.Logger, optional
            Logger instance for output.
        """
        if not GRAPH_TOOL_AVAILABLE:
            raise ImportError(
                "graph-tool is required for graph building. "
                "Install via: conda install -c conda-forge graph-tool"
            )

        self.tolerance = tolerance
        self.directed = directed
        self.logger = _get_logger(logger)

        # Graph and property maps
        self.graph = Graph(directed=directed)
        self.v_pos = self.graph.new_vertex_property("vector<double>")     # [x, y]
        self.e_geometry = self.graph.new_edge_property("vector<double>")  # flattened coords
        self.e_length = self.graph.new_edge_property("double")

        # Vertex coordinate mapping for deduplication
        self.vertex_coords_map: Dict[Tuple[float, float], Any] = {}

        # Edge attribute property maps
        self.edge_attr_maps: Dict[str, PropertyMap] = {}

    def build_from_gpkg(
        self,
        gpkg_path: Path,
        street_layer_name: str = "Street Network",
        include_attrs: Optional[List[str]] = None,
    ) -> Graph:
        """
        Build graph-tool network from enriched GeoPackage.

        Parameters
        ----------
        gpkg_path : Path
            Path to enriched GeoPackage containing street network.
        street_layer_name : str, default "Street Network"
            Value in 'layer_name' column identifying street segments.
        include_attrs : list of str, optional
            Whitelist of attribute column names to include in graph.
            If None, uses default set described in _filter_attributes.

        Returns
        -------
        graph : graph_tool.Graph
            Constructed network with all property maps internalized.
        """
        gpkg_path = Path(gpkg_path)
        if not gpkg_path.exists():
            raise FileNotFoundError(f"GeoPackage not found: {gpkg_path}")

        self.logger.info(f"Loading enriched GeoPackage: {gpkg_path}")
        gdf = gpd.read_file(gpkg_path)

        if "layer_name" not in gdf.columns:
            raise ValueError("Expected 'layer_name' column in GeoPackage.")

        # Extract street network lines
        streets = gdf[gdf["layer_name"] == street_layer_name].copy()
        streets = streets[streets.geometry.notnull()]
        streets = streets[streets.geometry.geom_type.isin(["LineString", "MultiLineString"])]

        if streets.empty:
            raise ValueError(f"No street segments found with layer_name='{street_layer_name}'")

        self.logger.info(f"Found {len(streets)} street segments")
        self.logger.info(f"CRS: {streets.crs}")
        self.logger.info(f"Bounds: {streets.total_bounds}")

        # Filter attributes based on whitelist
        all_attrs = [col for col in streets.columns if col != 'geometry']
        filtered_attrs = _filter_attributes(all_attrs, include_attrs)

        self.logger.info(
            f"Attribute filtering: {len(all_attrs)} total → {len(filtered_attrs)} preserved"
        )

        # Initialize edge attribute property maps
        self._init_attribute_maps(streets, filtered_attrs)

        # Build graph from LineString geometries
        self.logger.info("Building graph from LineString geometries...")
        skipped = 0

        for _, row in streets.iterrows():
            geom = row['geometry']

            # Handle MultiLineString by extracting individual LineStrings
            if geom.geom_type == 'MultiLineString':
                for line in geom.geoms:
                    if not self._add_edge_from_linestring(line, row):
                        skipped += 1
            elif geom.geom_type == 'LineString':
                if not self._add_edge_from_linestring(geom, row):
                    skipped += 1
            else:
                skipped += 1

        if skipped > 0:
            self.logger.warning(f"Skipped {skipped} invalid geometries")

        # Internalize all property maps
        self._internalize_properties()

        self.logger.info("✓ Graph construction complete:")
        self.logger.info(f"  Vertices: {self.graph.num_vertices():,}")
        self.logger.info(f"  Edges: {self.graph.num_edges():,}")
        self.logger.info(f"  Vertex properties: {list(self.graph.vp.keys())}")
        self.logger.info(f"  Edge properties: {list(self.graph.ep.keys())}")

        return self.graph

    def _init_attribute_maps(
        self,
        gdf: gpd.GeoDataFrame,
        preserve_attrs: List[str]
    ) -> None:
        """Initialize edge property maps for all attributes with correct types."""
        for attr in preserve_attrs:
            if attr not in gdf.columns:
                continue

            dtype = gdf[attr].dtype
            sample_value = gdf[attr].dropna().iloc[0] if not gdf[attr].dropna().empty else None

            ptype = _infer_graph_tool_type(dtype, sample_value)
            self.edge_attr_maps[attr] = self.graph.new_edge_property(ptype)

            self.logger.info(f"  {attr}: {dtype} → {ptype}")

    def _get_or_create_vertex(self, coord: Tuple[float, float]) -> Any:
        """Get existing vertex or create new one at coordinate."""
        rounded_coord = _round_coord(coord, self.tolerance)

        if rounded_coord in self.vertex_coords_map:
            return self.vertex_coords_map[rounded_coord]

        # Create new vertex
        v = self.graph.add_vertex()
        self.v_pos[v] = list(coord)  # Store as [x, y]
        self.vertex_coords_map[rounded_coord] = v
        return v

    def _add_edge_from_linestring(
        self,
        geom: LineString,
        attributes: Dict
    ) -> bool:
        """
        Add edge to graph from LineString geometry.

        Returns
        -------
        success : bool
            True if edge added successfully, False if skipped.
        """
        try:
            coords = list(geom.coords)
            if len(coords) < 2:
                return False

            # Get or create start/end vertices
            start_coord = coords[0]
            end_coord = coords[-1]
            v_start = self._get_or_create_vertex(start_coord)
            v_end = self._get_or_create_vertex(end_coord)

            # Skip self-loops: endpoints round to the same coordinate.
            # These arise from tiny closed ways (roundabouts, micro-loops) in
            # dense OSM data and cause inf betweenness in graph-tool's Dijkstra.
            if v_start == v_end:
                self.logger.debug(
                    f"Skipping self-loop: start and end coordinates both round to "
                    f"{_round_coord(start_coord, self.tolerance)}"
                )
                return False

            # Add edge
            edge = self.graph.add_edge(v_start, v_end)

            # Store full geometry as flattened coordinate array
            flat_coords = np.array(coords, dtype=np.float64).flatten()
            self.e_geometry[edge] = flat_coords

            # Store length
            self.e_length[edge] = geom.length

            # Store all attributes
            for attr, pmap in self.edge_attr_maps.items():
                value = attributes.get(attr)

                # Handle missing/null values
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    continue

                # Type conversion for safety
                ptype = pmap.value_type()
                try:
                    if ptype == 'string':
                        pmap[edge] = str(value)
                    elif ptype in ['long', 'int', 'int16_t']:
                        pmap[edge] = int(value)
                    elif ptype in ['double', 'float', 'long double']:
                        pmap[edge] = float(value)
                    elif ptype == 'bool':
                        pmap[edge] = bool(value)
                    else:
                        pmap[edge] = value
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"Could not convert {attr}={value} to {ptype}: {e}")

            return True

        except Exception as e:
            self.logger.warning(f"Error adding edge: {e}")
            return False

    def _internalize_properties(self) -> None:
        """Make all property maps internal (saved with graph)."""
        self.graph.vp['pos'] = self.v_pos
        self.graph.ep['geometry'] = self.e_geometry
        self.graph.ep['length'] = self.e_length

        for attr, pmap in self.edge_attr_maps.items():
            safe_name = _sanitize_column_name(attr)
            self.graph.ep[safe_name] = pmap

    def save_graph(self, output_path: Path) -> None:
        """
        Save graph to binary .gt file.

        Parameters
        ----------
        output_path : Path
            Output path for .gt file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.graph.save(str(output_path))
        size_mb = output_path.stat().st_size / (1024 * 1024)
        self.logger.info(f"✓ Graph saved to: {output_path} ({size_mb:.1f} MB)")

    @staticmethod
    def load_graph(graph_path: Path, logger: Optional[logging.Logger] = None) -> Graph:
        """
        Load previously saved graph from .gt file.
        """
        if not GRAPH_TOOL_AVAILABLE:
            raise ImportError("graph-tool is required to load graphs.")

        log = _get_logger(logger)
        graph_path = Path(graph_path)

        if not graph_path.exists():
            raise FileNotFoundError(f"Graph file not found: {graph_path}")

        log.info(f"Loading graph from: {graph_path}")
        g = gt.load_graph(str(graph_path))

        log.info("✓ Graph loaded:")
        log.info(f"  Vertices: {g.num_vertices():,}")
        log.info(f"  Edges: {g.num_edges():,}")
        log.info(f"  Vertex properties: {list(g.vp.keys())}")
        log.info(f"  Edge properties: {list(g.ep.keys())}")

        return g


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_graph_from_streets(
    enriched_gpkg: Path,
    output_path: Path,
    street_layer_name: str = "Street Network",
    tolerance: float = 1e-8,
    directed: bool = False,
    include_attrs: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None,
) -> Graph:
    """
    Build graph-tool network from ANTIPOMPEII enriched street network.

    If include_attrs is None, keeps:
    - fid, osmid, highway, length, bridge, service, tunnel
    - layer_name, sphere
    - fac_layer_name, fac_name, fac_amenity, fac_leisure
    - disruption
    - all 'fac_*' and all 'pop_*' attributes.
    """
    log = _get_logger(logger)

    log.info("=" * 70)
    log.info("ANTIPOMPEII Graph Builder: Converting streets to graph-tool network")
    log.info("=" * 70)

    builder = GraphBuilder(
        tolerance=tolerance,
        directed=directed,
        logger=log,
    )

    graph = builder.build_from_gpkg(
        gpkg_path=enriched_gpkg,
        street_layer_name=street_layer_name,
        include_attrs=include_attrs,
    )

    builder.save_graph(output_path)

    log.info("=" * 70)
    log.info("✓ Graph building complete")
    log.info("=" * 70)

    return graph


def build_graphs_from_streets_temporal(
    enriched_gpkgs: Dict[str, Path],
    output_dir: Path,
    street_layer_name: str = "Street Network",
    tolerance: float = 1e-8,
    directed: bool = False,
    include_attrs: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Path]:
    """
    Build graph-tool networks for multiple temporal enriched street GeoPackages.

    Parameters
    ----------
    enriched_gpkgs : dict
        Mapping {timestamp: gpkg_path} where each GPKG is an enriched
        street network (facilities, population, disruption).
    output_dir : Path
        Directory where .gt graph files will be written.
    street_layer_name : str, default "Street Network"
        Value in 'layer_name' column identifying street segments.
    tolerance : float, default 1e-8
        Coordinate tolerance for vertex deduplication.
    directed : bool, default False
        Create directed (True) or undirected (False) graphs.
    include_attrs : list of str, optional
        Whitelist of attributes to include; if None, uses default set.
    logger : logging.Logger, optional
        Logger instance for output.

    Returns
    -------
    graph_paths : dict
        Mapping {timestamp: graph_path} for all successfully built graphs.
    """
    log = _get_logger(logger)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    graph_paths: Dict[str, Path] = {}

    log.info("=" * 70)
    log.info("ANTIPOMPEII Temporal Graph Builder")
    log.info("=" * 70)

    # Process in sorted timestamp order for reproducibility
    for ts in sorted(enriched_gpkgs.keys()):
        gpkg_path = Path(enriched_gpkgs[ts])

        if not gpkg_path.exists():
            log.warning(f"Temporal GPKG for {ts} not found: {gpkg_path}; skipping.")
            continue

        graph_filename = gpkg_path.stem + "_network.gt"
        graph_path = output_dir / graph_filename

        log.info("-" * 70)
        log.info(f"[{ts}] Building graph from: {gpkg_path}")
        log.info(f"[{ts}] Output graph: {graph_path}")

        builder = GraphBuilder(
            tolerance=tolerance,
            directed=directed,
            logger=log,
        )

        try:
            g = builder.build_from_gpkg(
                gpkg_path=gpkg_path,
                street_layer_name=street_layer_name,
                include_attrs=include_attrs,
            )
            builder.save_graph(graph_path)
            graph_paths[ts] = graph_path

            log.info(
                f"[{ts}] ✓ Graph built: {g.num_vertices():,} vertices, "
                f"{g.num_edges():,} edges"
            )
        except Exception as e:
            log.error(f"[{ts}] Graph building failed: {e}", exc_info=True)

    log.info("=" * 70)
    log.info(
        f"Temporal graph building complete: {len(graph_paths)}/{len(enriched_gpkgs)} snapshots"
    )
    log.info("=" * 70)

    return graph_paths
