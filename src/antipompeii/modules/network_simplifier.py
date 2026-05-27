"""
ANTIPOMPEII Network Simplification Module

Takes:
- graph-tool Graph (.gt file) produced by graph_builder.py
- Contains full street network with all enriched attributes

Does:
- Two-stage simplification:
  1. Consolidate parallel/duplicate edges
  2. Remove degree-2 nodes (merge linear segments)
- Preserves all edge attributes via configurable aggregation strategies
- Maintains geometry continuity by merging LineString coordinates
- Reduces network complexity while preserving topology

Output:
- Simplified graph-tool network (.gt file)
- Diagnostic reports showing reduction statistics
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Set, Callable, Any, Optional
from collections import defaultdict
import logging

import numpy as np

from src.antipompeii.utils.graph_tool_compat import gt, GRAPH_TOOL_AVAILABLE
if GRAPH_TOOL_AVAILABLE:
    from graph_tool import Graph


# ---------------------------------------------------------------------------
# Attribute Aggregation Strategies
# ---------------------------------------------------------------------------

class AttributeAggregator:
    """Aggregation strategies for merging edge attributes."""

    @staticmethod
    def first(values: List) -> Any:
        """Take first non-None value."""
        for v in values:
            if v is not None:
                return v
        return None

    @staticmethod
    def sum_numeric(values: List) -> float:
        """Sum numeric values (e.g., length, population)."""
        try:
            return sum(float(v) for v in values if v is not None)
        except:
            return 0.0

    @staticmethod
    def mean(values: List) -> float:
        """Average numeric values."""
        try:
            numeric = [float(v) for v in values if v is not None]
            return sum(numeric) / len(numeric) if numeric else 0.0
        except:
            return 0.0

    @staticmethod
    def max_value(values: List) -> Any:
        """Take maximum value."""
        try:
            numeric = [float(v) for v in values if v is not None]
            return max(numeric) if numeric else None
        except:
            return values[0] if values else None

    @staticmethod
    def concatenate(values: List, delimiter: str = ";") -> str:
        """Concatenate string values with delimiter."""
        str_vals = [str(v) for v in values if v is not None and str(v).strip()]
        return delimiter.join(str_vals) if str_vals else ""


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------

from src.antipompeii.utils.logger import get_module_logger as _get_logger


def _convert_to_property_type(graph: Graph, prop_name: str, value: Any) -> Any:
    """Convert value to match graph property type."""
    if prop_name not in graph.ep or value is None:
        return value

    prop_type = graph.ep[prop_name].value_type()

    # Unwrap if list
    if isinstance(value, list) and len(value) > 0:
        value = value[0]

    try:
        if prop_type in ['int', 'long', 'int16_t']:
            return int(value)
        elif prop_type in ['double', 'float', 'long double']:
            return float(value)
        elif prop_type == 'bool':
            return bool(value)
        elif prop_type == 'string':
            return str(value)
    except:
        pass

    return value


# ---------------------------------------------------------------------------
# Network Diagnostics
# ---------------------------------------------------------------------------

class NetworkDiagnostics:
    """Analyze and report network structure statistics."""

    @staticmethod
    def diagnose(graph: Graph, logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
        """
        Analyze network structure.

        Returns
        -------
        stats : dict
            Dictionary containing network statistics.
        """
        log = _get_logger(logger)

        log.info("=" * 70)
        log.info("NETWORK DIAGNOSTICS")
        log.info("=" * 70)

        is_directed = graph.is_directed()
        degree_counts = defaultdict(int)
        edge_pairs = defaultdict(int)
        self_loop_count = 0

        # Analyze edges
        for e in graph.edges():
            src, tgt = int(e.source()), int(e.target())
            pair = tuple(sorted([src, tgt])) if not is_directed else (src, tgt)
            edge_pairs[pair] += 1

        parallel_edge_count = sum(1 for count in edge_pairs.values() if count > 1)

        # Analyze vertex degrees
        for v in graph.vertices():
            neighbors = set(graph.get_out_neighbors(v))
            if is_directed:
                neighbors |= set(graph.get_in_neighbors(v))

            if int(v) in neighbors:
                self_loop_count += 1

            degree = len(neighbors)
            degree_counts[degree] += 1

        # Report
        log.info(f"Total vertices: {graph.num_vertices():,}")
        log.info(f"Total edges: {graph.num_edges():,}")
        log.info(f"Directed: {is_directed}")
        log.info(f"Parallel edge pairs: {parallel_edge_count:,}")
        log.info(f"Self-loops: {self_loop_count}")

        log.info("\nDegree Distribution (unique neighbors):")
        for deg in sorted(degree_counts.keys())[:10]:
            count = degree_counts[deg]
            pct = count / graph.num_vertices() * 100
            log.info(f"  Degree {deg}: {count:,} nodes ({pct:.1f}%)")

        degree_2_count = degree_counts.get(2, 0)
        degree_2_pct = degree_2_count / graph.num_vertices() * 100 if graph.num_vertices() > 0 else 0

        log.info(f"\nSimplification potential:")
        log.info(f"  Degree-2 nodes: {degree_2_count:,} ({degree_2_pct:.1f}%)")

        if degree_2_count == 0:
            log.warning("  ⚠ No degree-2 nodes - network may already be simplified")

        return {
            'vertices': graph.num_vertices(),
            'edges': graph.num_edges(),
            'directed': is_directed,
            'parallel_edges': parallel_edge_count,
            'self_loops': self_loop_count,
            'degree_2_nodes': degree_2_count,
            'degree_counts': dict(degree_counts)
        }


# ---------------------------------------------------------------------------
# Topological Simplifier
# ---------------------------------------------------------------------------

class TopologicalSimplifier:
    """Remove degree-2 nodes by merging linear path segments."""

    def __init__(
        self,
        graph: Graph,
        aggregators: Dict[str, Callable],
        logger: Optional[logging.Logger] = None
    ):
        self.graph = graph
        self.is_directed = graph.is_directed()
        self.aggregators = aggregators
        self.logger = _get_logger(logger)

    def simplify(self) -> Graph:
        """
        Remove degree-2 nodes and merge their incident edges.
        """
        self.logger.info("=" * 70)
        self.logger.info("STAGE 2: TOPOLOGICAL SIMPLIFICATION")
        self.logger.info("=" * 70)

        initial_v = self.graph.num_vertices()
        initial_e = self.graph.num_edges()

        self.logger.info(f"Initial: {initial_v:,} vertices, {initial_e:,} edges")

        # Find endpoints (non-degree-2 nodes)
        endpoints = self._get_endpoints()
        simplifiable = initial_v - len(endpoints)

        self.logger.info(f"Endpoints (degree != 2): {len(endpoints):,}")
        self.logger.info(f"Simplifiable (degree == 2): {simplifiable:,}")

        # Get linear paths through degree-2 nodes
        paths = self._get_paths(endpoints)
        self.logger.info(f"Paths to simplify: {len(paths):,}")

        if len(paths) == 0:
            self.logger.warning("⚠ No paths to simplify")
            return self.graph

        # Merge edges along paths
        self._simplify_paths(paths)

        final_v = self.graph.num_vertices()
        final_e = self.graph.num_edges()

        v_reduction = (initial_v - final_v) / initial_v * 100 if initial_v > 0 else 0
        e_reduction = (initial_e - final_e) / initial_e * 100 if initial_e > 0 else 0

        self.logger.info(f"Final: {final_v:,} vertices, {final_e:,} edges")
        self.logger.info(f"Reduction: {v_reduction:.1f}% vertices, {e_reduction:.1f}% edges")

        return self.graph

    def _get_endpoints(self) -> Set[int]:
        """Find nodes that cannot be simplified (degree != 2)."""
        endpoints = set()

        for v in self.graph.vertices():
            neighbors = set(self.graph.get_out_neighbors(v))
            if self.is_directed:
                neighbors |= set(self.graph.get_in_neighbors(v))

            num_neighbors = len(neighbors)
            has_self_loop = int(v) in neighbors

            # Endpoint if not exactly 2 neighbors or has self-loop
            if has_self_loop or num_neighbors != 2:
                endpoints.add(int(v))
            elif self.is_directed:
                # For directed, check in/out degree balance
                in_n = set(self.graph.get_in_neighbors(v))
                out_n = set(self.graph.get_out_neighbors(v))
                if len(in_n) != 1 or len(out_n) != 1:
                    endpoints.add(int(v))

        return endpoints

    def _get_paths(self, endpoints: Set[int]) -> List[List[int]]:
        """Find all linear paths through degree-2 nodes."""
        paths = []
        visited = set()

        for ep in endpoints:
            v = self.graph.vertex(ep)
            neighbors = set(self.graph.get_out_neighbors(v))

            for neighbor in neighbors:
                if neighbor not in endpoints:
                    path = self._follow_path(ep, neighbor, endpoints)
                    if len(path) > 2:
                        # Use sorted tuple to avoid duplicate paths
                        key = tuple(sorted([path[0], path[-1]]))
                        if key not in visited:
                            paths.append(path)
                            visited.add(key)

        return paths

    def _follow_path(self, start: int, current: int, endpoints: Set[int]) -> List[int]:
        """Follow a path through degree-2 nodes until hitting an endpoint."""
        path = [start, current]
        previous = start

        for _ in range(10000):  # Safety limit
            if current in endpoints:
                break

            neighbors = set(self.graph.get_out_neighbors(self.graph.vertex(current)))
            neighbors.discard(previous)

            if len(neighbors) != 1:
                break

            previous = current
            current = next(iter(neighbors))
            path.append(current)

        return path

    def _simplify_paths(self, paths: List[List[int]]) -> None:
        """Merge edges along each path into a single edge."""
        nodes_to_remove = set()
        edges_to_remove = set()

        for path in paths:
            if len(path) < 3:
                continue

            start_v = self.graph.vertex(path[0])
            end_v = self.graph.vertex(path[-1])

            # Collect edge data and geometries
            edge_data = []
            geometries = []

            for i in range(len(path) - 1):
                u = self.graph.vertex(path[i])
                for e in u.out_edges():
                    if int(e.target()) == path[i + 1]:
                        # Extract geometry
                        geom = np.array(self.graph.ep.geometry[e]).reshape(-1, 2)
                        geometries.append(geom)

                        # Extract attributes
                        attrs = {}
                        for prop in self.graph.ep.keys():
                            if prop != 'geometry':
                                attrs[prop] = self.graph.ep[prop][e]
                        edge_data.append(attrs)
                        edges_to_remove.add(e)

            if not edge_data:
                continue

            # Aggregate attributes
            agg_attrs = {}
            for prop in edge_data[0].keys():
                values = [d.get(prop) for d in edge_data if prop in d]

                if prop in self.aggregators:
                    agg_val = self.aggregators[prop](values)
                else:
                    agg_val = values[0] if values else None

                agg_attrs[prop] = _convert_to_property_type(self.graph, prop, agg_val)

            # Merge geometries
            merged = [geometries[0]]
            for i in range(1, len(geometries)):
                if len(merged[-1]) > 0 and len(geometries[i]) > 0:
                    # Check if geometries connect
                    if np.allclose(geometries[i][0], merged[-1][-1], atol=1e-8):
                        merged.append(geometries[i][1:])  # Skip duplicate point
                    else:
                        merged.append(geometries[i])
                else:
                    if len(geometries[i]) > 0:
                        merged.append(geometries[i])

            merged_coords = np.vstack([m for m in merged if len(m) > 0])

            # Create new simplified edge
            new_edge = self.graph.add_edge(start_v, end_v)
            self.graph.ep.geometry[new_edge] = merged_coords.flatten()

            # Set aggregated attributes
            for prop, val in agg_attrs.items():
                if prop in self.graph.ep and val is not None:
                    try:
                        self.graph.ep[prop][new_edge] = val
                    except:
                        pass

            # Mark intermediate nodes for removal
            nodes_to_remove.update(path[1:-1])

        # Cleanup: remove old edges and nodes
        for e in edges_to_remove:
            try:
                self.graph.remove_edge(e)
            except:
                pass

        for n in sorted(nodes_to_remove, reverse=True):
            try:
                self.graph.remove_vertex(n)
            except:
                pass


# ---------------------------------------------------------------------------
# Comprehensive Simplifier
# ---------------------------------------------------------------------------

class ComprehensiveSimplifier:
    """Two-stage network simplification: parallel edges + topology."""

    def __init__(
        self,
        graph: Graph,
        aggregators: Optional[Dict[str, Callable]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize simplifier.
        """
        if not GRAPH_TOOL_AVAILABLE:
            raise ImportError("graph-tool is required for network simplification.")

        self.graph = graph
        self.is_directed = graph.is_directed()
        self.logger = _get_logger(logger)

        # Validate required properties
        if 'pos' not in graph.vp or 'geometry' not in graph.ep:
            raise ValueError("Graph must have 'pos' vertex property and 'geometry' edge property")

        # Setup default aggregators
        self.aggregators = self._setup_aggregators(aggregators)

    def _setup_aggregators(self, custom: Optional[Dict[str, Callable]] = None) -> Dict[str, Callable]:
        """Setup aggregation strategies for each edge property."""
        agg = {}

        for prop in self.graph.ep.keys():
            if prop == 'geometry':
                continue  # Handled separately
            elif prop == 'length':
                agg[prop] = AttributeAggregator.sum_numeric
            elif prop.startswith('pop_'):
                # Population attributes: sum
                agg[prop] = AttributeAggregator.sum_numeric
            elif prop == 'fac_count':
                # Facility count: sum
                agg[prop] = AttributeAggregator.sum_numeric
            elif prop == 'disruption':
                # Disruption: max (if any segment disrupted, whole path is)
                agg[prop] = AttributeAggregator.max_value
            elif prop in ['name', 'highway', 'layer_name', 'layer']:
                # Categorical attributes: first
                agg[prop] = AttributeAggregator.first
            else:
                # Default: first value
                agg[prop] = AttributeAggregator.first

        # Override with custom aggregators
        if custom:
            agg.update(custom)

        return agg

    def consolidate_parallel_edges(self) -> Graph:
        """
        Stage 1: Merge parallel/duplicate edges between same node pairs.
        """
        self.logger.info("=" * 70)
        self.logger.info("STAGE 1: PARALLEL EDGE CONSOLIDATION")
        self.logger.info("=" * 70)

        initial_e = self.graph.num_edges()

        # Group edges by source-target pair
        edge_groups = defaultdict(list)
        for e in self.graph.edges():
            src, tgt = int(e.source()), int(e.target())
            pair = tuple(sorted([src, tgt])) if not self.is_directed else (src, tgt)
            edge_groups[pair].append(e)

        parallel_groups = sum(1 for edges in edge_groups.values() if len(edges) > 1)

        self.logger.info(f"Initial edges: {initial_e:,}")
        self.logger.info(f"Parallel groups: {parallel_groups:,}")

        # Consolidate each group
        edges_to_remove = []

        for pair, edges in edge_groups.items():
            if len(edges) <= 1:
                continue

            # Keep first edge, aggregate others into it
            base = edges[0]

            for prop in self.graph.ep.keys():
                if prop == 'geometry':
                    continue  # Keep first geometry

                values = [self.graph.ep[prop][e] for e in edges]

                if prop in self.aggregators:
                    agg_val = self.aggregators[prop](values)
                    agg_val = _convert_to_property_type(self.graph, prop, agg_val)

                    try:
                        self.graph.ep[prop][base] = agg_val
                    except:
                        pass

            edges_to_remove.extend(edges[1:])

        # Remove duplicate edges
        for e in edges_to_remove:
            try:
                self.graph.remove_edge(e)
            except:
                pass

        final_e = self.graph.num_edges()
        reduction = (initial_e - final_e) / initial_e * 100 if initial_e > 0 else 0

        self.logger.info(f"Final edges: {final_e:,}")
        self.logger.info(f"Removed: {initial_e - final_e:,} ({reduction:.1f}%)")

        return self.graph

    def simplify_topology(self) -> Graph:
        """
        Stage 2: Remove degree-2 nodes.
        """
        simplifier = TopologicalSimplifier(self.graph, self.aggregators, self.logger)
        return simplifier.simplify()

    def full_simplification(self) -> Graph:
        """
        Run both simplification stages.
        """
        self.logger.info("=" * 70)
        self.logger.info("TWO-STAGE NETWORK SIMPLIFICATION")
        self.logger.info("=" * 70)

        initial_v = self.graph.num_vertices()
        initial_e = self.graph.num_edges()

        self.logger.info(f"\nInitial: {initial_v:,} vertices, {initial_e:,} edges")

        # Stage 1
        self.consolidate_parallel_edges()
        after1_v = self.graph.num_vertices()
        after1_e = self.graph.num_edges()

        # Stage 2
        self.simplify_topology()
        final_v = self.graph.num_vertices()
        final_e = self.graph.num_edges()

        # Summary
        self.logger.info("\n" + "=" * 70)
        self.logger.info("SIMPLIFICATION SUMMARY")
        self.logger.info("=" * 70)
        self.logger.info(f"Initial:        {initial_v:,} vertices, {initial_e:,} edges")
        self.logger.info(f"After Stage 1:  {after1_v:,} vertices, {after1_e:,} edges")
        self.logger.info(f"After Stage 2:  {final_v:,} vertices, {final_e:,} edges")

        v_reduction = (initial_v - final_v) / initial_v * 100 if initial_v > 0 else 0
        e_reduction = (initial_e - final_e) / initial_e * 100 if initial_e > 0 else 0

        self.logger.info(f"\nTotal reduction:")
        self.logger.info(f"  Vertices: {initial_v - final_v:,} ({v_reduction:.1f}%)")
        self.logger.info(f"  Edges:    {initial_e - final_e:,} ({e_reduction:.1f}%)")

        return self.graph


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def simplify_network(
    graph_path: Path,
    output_path: Path,
    run_diagnostics: bool = True,
    custom_aggregators: Optional[Dict[str, Callable]] = None,
    logger: Optional[logging.Logger] = None,
) -> Graph:
    """
    Simplify ANTIPOMPEII street network graph.

    This is the main public API function for integration with the ANTIPOMPEII CLI.

    Performs two-stage simplification:
    1. Consolidate parallel/duplicate edges
    2. Remove degree-2 nodes (merge linear segments)
    """
    if not GRAPH_TOOL_AVAILABLE:
        raise ImportError(
            "graph-tool is required for network simplification. "
            "Install via: conda install -c conda-forge graph-tool"
        )

    log = _get_logger(logger)

    graph_path = Path(graph_path)
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_path}")

    log.info("=" * 70)
    log.info("ANTIPOMPEII Network Simplifier")
    log.info("=" * 70)
    log.info(f"Loading graph: {graph_path}")

    # Load graph
    g = gt.load_graph(str(graph_path))

    log.info(f"✓ Graph loaded: {g.num_vertices():,} vertices, {g.num_edges():,} edges")

    # Initial diagnostics
    if run_diagnostics:
        log.info("\n" + "=" * 70)
        log.info("INITIAL STATE")
        log.info("=" * 70)
        NetworkDiagnostics.diagnose(g, logger=log)

    # Simplify
    simplifier = ComprehensiveSimplifier(g, custom_aggregators, logger=log)
    g_simplified = simplifier.full_simplification()

    # Final diagnostics
    if run_diagnostics:
        log.info("\n" + "=" * 70)
        log.info("FINAL STATE")
        log.info("=" * 70)
        NetworkDiagnostics.diagnose(g_simplified, logger=log)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    g_simplified.save(str(output_path))
    size_mb = output_path.stat().st_size / (1024 * 1024)

    log.info("\n" + "=" * 70)
    log.info(f"✓ Simplified graph saved to: {output_path}")
    log.info(f"  File size: {size_mb:.1f} MB")
    log.info("=" * 70)

    return g_simplified
