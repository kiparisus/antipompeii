"""
ANTIPOMPEII Multi-Component Percolation Analysis

Implements the percolation framework described in the formalization:
  - At each step t, remove one edge from the graph G_t.
  - Decompose G_t into connected components C_i^(t).
  - Compute service penalty P_s^(t): the population-weighted share of
    components that have lost access to service s.

Three degradation scenarios are offered:

  1. Betweenness centrality attack
       Remove the edge with maximum b_e(G_t) at each step.
       Betweenness is recomputed every K steps (user choice).
       Compared against a Soft Random Geometric Graph (SRGG) null model.

  2. Random failure
       Edges removed in a uniformly random order.
       Run N_runs times; mean ± std plotted.

  3. Elevation-based removal
       Edges removed in batches defined by elevation quantile thresholds
       (flood scenario: lowest-elevation edges removed first; or reversed).
       Requires elev_min edge property from dem_processing.py.

For each scenario, P_s^(t) is tracked for every facility type and for
"any critical facility".  Results are exported as CSV, LaTeX, and PNG.

SRGG null model (Scenario 1 only):
  Calibrates connection probability P(i,j) = exp(-(d_ij/r)^α) to match the
  empirical mean degree and clustering coefficient; generates M random
  graph instances with spatially-interpolated population weights and
  uniformly-sampled facility edges.  Skipped if n_vertices > 1 500.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

from src.antipompeii.utils.graph_tool_compat import gt, GRAPH_TOOL_AVAILABLE
if GRAPH_TOOL_AVAILABLE:
    from graph_tool import Graph, GraphView
    from graph_tool.centrality import betweenness as gt_betweenness
    from graph_tool.topology import label_components

try:
    from scipy.spatial.distance import cdist
    from scipy.optimize import brentq
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Facility catalog (individual services only; compound rows added for output)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _FacSpec:
    key: str
    display: str
    amenity_values: Tuple[str, ...]
    layer_values: Tuple[str, ...]


_INDIVIDUAL_FACILITIES: List[_FacSpec] = [
    _FacSpec("hospital",   "Hospital",           ("hospital",),                         ()),
    _FacSpec("clinic",     "Clinic",              ("clinic",),                           ()),
    _FacSpec("pharmacy",   "Pharmacy",            ("pharmacy",),                         ()),
    _FacSpec("police",     "Police station",      ("police",),                           ()),
    _FacSpec("fire",       "Fire station",        ("fire_station", "ambulance_station"),  ()),
    _FacSpec("shelter",    "Emergency shelter",   ("shelter", "bunker"),                 ("Emergency",)),
    _FacSpec("conv_shelt", "Convertible shelter", ("school", "kindergarten",
                                                    "place_of_worship"),                ("Convertible Shelter",)),
]
_ANY_FAC = _FacSpec(
    "any_crit", "Any critical facility",
    ("hospital", "clinic", "pharmacy", "police", "fire_station",
     "ambulance_station", "shelter", "bunker", "school",
     "kindergarten", "place_of_worship"),
    ("Health", "Emergency", "Convertible Shelter"),
)
PERC_FACILITIES: List[_FacSpec] = _INDIVIDUAL_FACILITIES + [_ANY_FAC]

# Color palette (tab10 order, last = any-facility in bold black)
_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#000000",
]

# Elevation scenario constants
_N_ELEV_THRESHOLDS = 10   # decile quantiles
_SRGG_MAX_VERTICES = 1500
_SRGG_ALPHA        = 2.0  # Gaussian-like decay
_SRGG_M_DEFAULT    = 5    # null-model ensemble size


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class PercolationStep:
    step: int
    fraction_removed: float
    pop_direct_fraction: float            # pop on removed edges / total_pop
    penalties: Dict[str, float]           # facility_key → global P_s^(t) (direct + indirect)


@dataclass
class PercolationResult:
    scenario: str                          # "betweenness" | "random" | "elevation"
    label: str
    steps: List[PercolationStep]
    frac_arr: np.ndarray                   # x-axis: fraction of edges removed
    null_mean: Optional[Dict[str, np.ndarray]] = None   # SRGG mean per facility
    null_std:  Optional[Dict[str, np.ndarray]] = None   # SRGG std  per facility
    output_dir: Path = field(default_factory=Path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

from src.antipompeii.utils.logger import get_module_logger as _get_logger


def _has_fac(amenity: str, layer: str, spec: _FacSpec) -> bool:
    av = {v.strip() for v in amenity.split(";") if v.strip()}
    lv = {v.strip() for v in layer.split(";")   if v.strip()}
    return bool(
        (spec.amenity_values and av.intersection(spec.amenity_values)) or
        (spec.layer_values   and lv.intersection(spec.layer_values))
    )


def _str_arr(g: Graph, prop: str) -> np.ndarray:
    if prop not in g.ep:
        return np.full(g.num_edges(), "", dtype=object)
    return np.array([g.ep[prop][e] for e in g.edges()], dtype=object)


def _num_arr(g: Graph, prop: str, default: float = 0.0) -> np.ndarray:
    if prop not in g.ep:
        return np.full(g.num_edges(), default, dtype=float)
    try:
        return g.ep[prop].a.astype(float)
    except Exception:
        arr = np.array([g.ep[prop][e] for e in g.edges()], dtype=float)
        return np.nan_to_num(arr, nan=default)


def _pop_year(g: Graph) -> Optional[str]:
    years = [
        int(p[len("pop_total_"):])
        for p in g.ep.keys()
        if p.startswith("pop_total_") and p[len("pop_total_"):].isdigit()
    ]
    return str(max(years)) if years else None


# ---------------------------------------------------------------------------
# Core Percolator
# ---------------------------------------------------------------------------

class Percolator:
    """
    Loads an enriched graph-tool network and runs one or more percolation
    scenarios, tracking population-weighted service-accessibility penalties.
    """

    def __init__(
        self,
        graph_path: Path,
        output_dir: Path,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if not GRAPH_TOOL_AVAILABLE:
            raise ImportError(
                "graph-tool is required for percolation analysis. "
                "Install via: conda install -c conda-forge graph-tool"
            )
        self.log = _get_logger(logger)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        graph_path = Path(graph_path)
        self.log.info(f"Loading base graph: {graph_path}")
        self.g: Graph = gt.load_graph(str(graph_path))
        self.n_edges    = self.g.num_edges()
        self.n_vertices = self.g.num_vertices()
        self.log.info(
            f"Graph: {self.n_vertices:,} vertices, {self.n_edges:,} edges"
        )

        # Source vertex per edge (for component → edge mapping)
        self._src_v = self.g.get_edges()[:, 0]

        # Facility masks (bool, shape n_edges) — computed once
        fac_am  = _str_arr(self.g, "fac_amenity")
        fac_lyr = _str_arr(self.g, "fac_layer_name")
        self._fac_masks: Dict[str, np.ndarray] = {
            spec.key: np.array(
                [_has_fac(str(a), str(l), spec) for a, l in zip(fac_am, fac_lyr)],
                dtype=bool,
            )
            for spec in PERC_FACILITIES
        }
        self._present_facs: List[_FacSpec] = [
            spec for spec in PERC_FACILITIES
            if self._fac_masks[spec.key].any()
        ]
        if not self._present_facs:
            self.log.warning("No facility edges found; service penalties will be zero.")

        # Population weights (total population on edge, most recent year)
        year = _pop_year(self.g)
        if year:
            self._pop = _num_arr(self.g, f"pop_total_{year}", default=0.0)
            self.log.info(f"Population year: {year}")
        else:
            self._pop = np.zeros(self.n_edges, dtype=float)
            self.log.warning("No population data found; penalties will be edge-count based.")
        self._total_pop = self._pop.sum()
        if self._total_pop == 0.0:
            # Fallback: treat each edge as weight 1
            self._pop        = np.ones(self.n_edges, dtype=float)
            self._total_pop  = float(self.n_edges)

        # Elevation (for scenario 3)
        self._elev = _num_arr(self.g, "elev_min", default=np.nan)

        # Geometry for visualizations
        self._has_geom = "geometry" in self.g.ep

        # Vertex positions for SRGG (stored as vector<double> by graph_builder)
        self._vertex_pos: Optional[np.ndarray] = None
        if "pos" in self.g.vp:
            try:
                pos = np.array(
                    [[self.g.vp["pos"][v][0], self.g.vp["pos"][v][1]]
                     for v in self.g.vertices()],
                    dtype=float,
                )
                self._vertex_pos = pos
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Scenario 1: betweenness centrality attack
    # ------------------------------------------------------------------

    def run_betweenness(
        self,
        n_steps: int,
        recompute_every: int = 1,
        run_null: bool = False,
        null_m: int = _SRGG_M_DEFAULT,
    ) -> PercolationResult:
        """
        Progressive removal of highest-betweenness edges.
        """
        n_steps = min(n_steps, self.n_edges)
        active = np.ones(self.n_edges, dtype=bool)

        steps: List[PercolationStep] = []
        # Step 0: intact network
        d0, p0 = self._service_penalties(active)
        steps.append(PercolationStep(0, 0.0, d0, p0))

        bw_cache: Optional[np.ndarray] = None
        removal_order: List[int] = []

        for t in range(1, n_steps + 1):
            # Recompute betweenness if needed
            if bw_cache is None or (t - 1) % recompute_every == 0:
                efilt = self.g.new_edge_property("bool")
                efilt.a[:] = active
                g_view = GraphView(self.g, efilt=efilt)
                _, ep_bw = gt_betweenness(g_view)
                bw_cache = ep_bw.a.astype(float).copy()
                bw_cache[~active] = -1.0   # mask inactive

            # Remove the edge with maximum betweenness
            e_star = int(np.argmax(bw_cache))
            active[e_star] = False
            bw_cache[e_star] = -1.0
            removal_order.append(e_star)

            frac = t / self.n_edges
            dt, pt = self._service_penalties(active)
            steps.append(PercolationStep(t, frac, dt, pt))
            if t % max(1, n_steps // 10) == 0:
                self.log.info(
                    f"  Betweenness attack: step {t}/{n_steps} "
                    f"({100*frac:.0f}% removed)"
                )

        frac_arr = np.array([s.fraction_removed for s in steps])
        result = PercolationResult(
            scenario="betweenness",
            label="Betweenness centrality attack",
            steps=steps,
            frac_arr=frac_arr,
            output_dir=self.output_dir,
        )

        # SRGG null model
        if run_null and SCIPY_AVAILABLE and self._vertex_pos is not None:
            if self.n_vertices <= _SRGG_MAX_VERTICES:
                self.log.info(
                    f"Generating SRGG null model (M={null_m}) …"
                )
                null_mean, null_std = self._srgg_null_model(
                    null_m, n_steps, recompute_every, frac_arr
                )
                result.null_mean = null_mean
                result.null_std  = null_std
            else:
                self.log.warning(
                    f"SRGG null model skipped: n_vertices={self.n_vertices} "
                    f"> {_SRGG_MAX_VERTICES}."
                )

        self._export(result)
        self._visualize_curves(result)
        return result

    # ------------------------------------------------------------------
    # Scenario 2: random failure
    # ------------------------------------------------------------------

    def run_random(
        self,
        n_steps: int,
        n_runs: int = 10,
    ) -> PercolationResult:
        """
        Random edge-removal percolation (averaged over n_runs realisations).
        """
        n_steps = min(n_steps, self.n_edges)

        # Collect penalty trajectories per run
        # penalties_all[fac_key] = (n_runs, n_steps+1)
        all_runs: Dict[str, List[np.ndarray]] = {
            spec.key: [] for spec in self._present_facs
        }
        all_direct_runs: List[np.ndarray] = []

        for run in range(n_runs):
            rng   = np.random.default_rng(seed=run)
            order = rng.permutation(self.n_edges)
            active = np.ones(self.n_edges, dtype=bool)

            run_pens: Dict[str, List[float]] = {
                spec.key: [] for spec in self._present_facs
            }
            run_direct: List[float] = []
            # Step 0
            d0, p0 = self._service_penalties(active)
            run_direct.append(d0)
            for spec in self._present_facs:
                run_pens[spec.key].append(p0.get(spec.key, 0.0))

            for t in range(n_steps):
                active[order[t]] = False
                dt, pt = self._service_penalties(active)
                run_direct.append(dt)
                for spec in self._present_facs:
                    run_pens[spec.key].append(pt.get(spec.key, 0.0))

            all_direct_runs.append(np.array(run_direct))
            for spec in self._present_facs:
                all_runs[spec.key].append(np.array(run_pens[spec.key]))

            if (run + 1) % max(1, n_runs // 5) == 0:
                self.log.info(f"  Random failure: run {run+1}/{n_runs} done")

        # Build averaged steps
        frac_arr = np.linspace(0, n_steps / self.n_edges, n_steps + 1)
        mean_dict: Dict[str, np.ndarray] = {}
        std_dict:  Dict[str, np.ndarray] = {}
        for spec in self._present_facs:
            mat = np.stack(all_runs[spec.key], axis=0)  # (n_runs, n_steps+1)
            mean_dict[spec.key] = mat.mean(axis=0)
            std_dict [spec.key] = mat.std(axis=0)
        direct_mean = np.stack(all_direct_runs, axis=0).mean(axis=0)

        # Build synthetic PercolationStep list (using mean for CSV export)
        steps: List[PercolationStep] = []
        for i in range(n_steps + 1):
            steps.append(PercolationStep(
                step=i,
                fraction_removed=frac_arr[i],
                pop_direct_fraction=float(direct_mean[i]),
                penalties={k: float(v[i]) for k, v in mean_dict.items()},
            ))

        result = PercolationResult(
            scenario="random",
            label=f"Random failure (mean of {n_runs} runs)",
            steps=steps,
            frac_arr=frac_arr,
            null_mean=mean_dict,
            null_std=std_dict,
            output_dir=self.output_dir,
        )
        self._export(result)
        self._visualize_curves(result, is_random=True)
        return result

    # ------------------------------------------------------------------
    # Scenario 3: elevation-based removal
    # ------------------------------------------------------------------

    def run_elevation(
        self,
        ascending: bool = True,
    ) -> PercolationResult:
        """
        Batch removal of edges by elevation threshold (decile quantiles).
        """
        valid_mask = np.isfinite(self._elev)
        if not valid_mask.any():
            raise RuntimeError(
                "No valid elevation data (elev_min) found in graph. "
                "Run DEM processing (step 11) before percolation."
            )

        elev_valid = self._elev[valid_mask]
        # Compute N case-specific thresholds from the elevation distribution
        n_thresholds = _compute_elevation_thresholds_count(elev_valid)
        quantiles    = np.linspace(0, 100, n_thresholds + 1)[1:]  # exclude 0th
        thresholds   = np.percentile(elev_valid, quantiles)
        thresholds   = np.unique(thresholds)  # remove duplicates

        direction_label = "ascending (flood)" if ascending else "descending (closure)"
        self.log.info(
            f"Elevation percolation: {len(thresholds)} thresholds, {direction_label}, "
            f"range {elev_valid.min():.1f}–{elev_valid.max():.1f} m"
        )

        steps: List[PercolationStep] = []
        active = np.ones(self.n_edges, dtype=bool)
        # Edges with no elevation are NOT removed in this scenario
        elev_known = valid_mask

        # Step 0: intact
        d0, p0 = self._service_penalties(active)
        steps.append(PercolationStep(0, 0.0, d0, p0))

        removed_so_far = 0
        for thresh in (thresholds if ascending else thresholds[::-1]):
            if ascending:
                batch = elev_known & (self._elev <= thresh) & active
            else:
                batch = elev_known & (self._elev >= thresh) & active
            active[batch] = False
            removed_so_far += int(batch.sum())
            frac = removed_so_far / self.n_edges
            thresh_label = f"{thresh:.1f} m"
            self.log.info(
                f"  Elevation threshold {thresh_label}: "
                f"{batch.sum()} edges removed ({100*frac:.1f}% total)"
            )
            dt, pt = self._service_penalties(active)
            steps.append(
                PercolationStep(
                    step=len(steps),
                    fraction_removed=frac,
                    pop_direct_fraction=dt,
                    penalties=pt,
                )
            )

        frac_arr = np.array([s.fraction_removed for s in steps])
        result = PercolationResult(
            scenario="elevation",
            label=f"Elevation-based removal ({direction_label})",
            steps=steps,
            frac_arr=frac_arr,
            output_dir=self.output_dir,
        )
        self._export(result)
        self._visualize_curves(result)
        return result

    # ------------------------------------------------------------------
    # Core: service penalty computation  P_s^(t)
    # ------------------------------------------------------------------

    def _service_penalties(
        self, active: np.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute global P_s^(t) for every present facility type and the
        any-facility aggregate.
        """
        direct_frac = float(self._pop[~active].sum() / self._total_pop)

        if not active.any():
            return 1.0, {spec.key: 1.0 for spec in self._present_facs}

        # Connected components on the current active sub-graph
        efilt = self.g.new_edge_property("bool")
        efilt.a[:] = active
        g_view = GraphView(self.g, efilt=efilt)
        comp_map, _ = label_components(g_view)
        comp_arr = comp_map.a                      # (n_vertices,)
        edge_comp = comp_arr[self._src_v]          # component of each base edge

        active_comp = edge_comp[active]            # components of active edges
        active_pop  = self._pop[active]

        penalties: Dict[str, float] = {}
        for spec in self._present_facs:
            fac_active = self._fac_masks[spec.key][active]
            if not fac_active.any():
                # Facility fully removed → everyone loses access
                penalties[spec.key] = 1.0
                continue

            # Components that still contain at least one facility edge
            comps_with_fac = set(active_comp[fac_active].tolist())
            has_access = np.isin(active_comp, list(comps_with_fac))
            # Indirect loss: active-edge population without facility access
            indirect_pop = float(active_pop[~has_access].sum())
            # Global penalty = direct loss + indirect loss
            penalties[spec.key] = min(
                1.0, direct_frac + indirect_pop / self._total_pop
            )

        return direct_frac, penalties

    # ------------------------------------------------------------------
    # SRGG null model
    # ------------------------------------------------------------------

    def _srgg_null_model(
        self,
        m: int,
        n_steps: int,
        recompute_every: int,
        frac_arr: np.ndarray,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Generate M SRGG instances, run betweenness percolation on each,
        and return (mean_penalty, std_penalty) per facility over the trajectory.
        """
        pos = self._vertex_pos                 # (n_v, 2)
        n_v = len(pos)
        D   = cdist(pos, pos)                  # pairwise distances

        # Calibrate r to match empirical mean degree
        degrees      = np.array([self.g.vertex(v).out_degree() +
                                  self.g.vertex(v).in_degree()
                                  for v in self.g.vertices()], dtype=float)
        target_deg   = float(degrees.mean())
        target_clust = self._empirical_clustering()

        r = self._calibrate_srgg_r(D, target_deg, _SRGG_ALPHA)
        self.log.info(
            f"SRGG: r={r:.4f}, α={_SRGG_ALPHA}, target ⟨k⟩={target_deg:.2f}"
        )

        # Service fractions (proportions to maintain in null model)
        fac_fracs: Dict[str, float] = {
            spec.key: float(self._fac_masks[spec.key].sum()) / self.n_edges
            for spec in self._present_facs
        }
        # Population per vertex (sum of adjacent-edge populations) for interpolation
        vpop = self._vertex_populations()

        # Run M instances
        all_curves: Dict[str, List[np.ndarray]] = {
            spec.key: [] for spec in self._present_facs
        }

        for m_idx in range(m):
            self.log.info(f"  SRGG instance {m_idx+1}/{m} …")
            g_srgg, ep_pop, fac_masks_srgg = self._build_srgg_instance(
                pos, D, r, _SRGG_ALPHA, vpop, fac_fracs, seed=m_idx
            )
            if g_srgg is None or g_srgg.num_edges() == 0:
                continue
            curve = self._srgg_percolation(
                g_srgg, ep_pop, fac_masks_srgg, n_steps, recompute_every, frac_arr
            )
            for spec in self._present_facs:
                if spec.key in curve:
                    all_curves[spec.key].append(curve[spec.key])

        # Aggregate
        mean_d: Dict[str, np.ndarray] = {}
        std_d:  Dict[str, np.ndarray] = {}
        for spec in self._present_facs:
            runs = all_curves[spec.key]
            if runs:
                mat = np.stack(runs, axis=0)
                mean_d[spec.key] = mat.mean(axis=0)
                std_d [spec.key] = mat.std(axis=0)

        return mean_d, std_d

    def _calibrate_srgg_r(
        self, D: np.ndarray, target_deg: float, alpha: float
    ) -> float:
        """Binary search for r such that E[degree] ≈ target_deg."""
        n = D.shape[0]

        def _expected_deg(r: float) -> float:
            P = np.exp(-(D / r) ** alpha)
            np.fill_diagonal(P, 0.0)
            return float(P.sum() / n)

        r_lo, r_hi = 1e-6, float(D.max())
        # Check boundaries
        if _expected_deg(r_hi) < target_deg:
            return r_hi
        if _expected_deg(r_lo) > target_deg:
            return r_lo
        try:
            r = brentq(lambda r: _expected_deg(r) - target_deg, r_lo, r_hi, xtol=1e-4)
        except Exception:
            r = float(D.mean()) * 0.5   # fallback
        return float(r)

    def _empirical_clustering(self) -> float:
        """Average local clustering coefficient of the base graph."""
        try:
            from graph_tool.clustering import local_clustering
            lc = local_clustering(self.g, undirected=True)
            return float(lc.a.mean())
        except Exception:
            return 0.0

    def _vertex_populations(self) -> np.ndarray:
        """Population per vertex = mean of incident-edge populations."""
        vp = np.zeros(self.n_vertices, dtype=float)
        counts = np.zeros(self.n_vertices, dtype=int)
        edges  = self.g.get_edges()         # (n_edges, 2)
        for idx, (u, v) in enumerate(edges):
            vp[u] += self._pop[idx]
            vp[v] += self._pop[idx]
            counts[u] += 1
            counts[v] += 1
        with np.errstate(invalid="ignore"):
            vp = np.where(counts > 0, vp / counts, 0.0)
        return vp

    def _build_srgg_instance(
        self,
        pos: np.ndarray,
        D: np.ndarray,
        r: float,
        alpha: float,
        vpop: np.ndarray,
        fac_fracs: Dict[str, float],
        seed: int,
    ) -> Tuple[Optional[Graph], Optional[Any], Optional[Dict[str, np.ndarray]]]:
        """Build one SRGG graph with population and facility assignments."""
        rng = np.random.default_rng(seed=seed)
        n_v = len(pos)
        P   = np.exp(-(D / r) ** alpha)
        np.fill_diagonal(P, 0.0)
        # Upper-triangle edges only (undirected)
        u_idx, v_idx = np.triu_indices(n_v, k=1)
        probs = P[u_idx, v_idx]
        mask  = rng.uniform(0.0, 1.0, size=len(probs)) < probs
        u_e   = u_idx[mask]
        v_e   = v_idx[mask]
        n_e   = int(mask.sum())

        if n_e == 0:
            return None, None, None

        g_srgg = gt.Graph(directed=False)
        g_srgg.add_vertex(n_v)
        edges_arr = np.column_stack([u_e, v_e])
        g_srgg.add_edge_list(edges_arr)

        # Population weights: Gaussian kernel from empirical vertex populations
        sigma = r
        ep_pop = g_srgg.new_edge_property("double")
        emid_x = (pos[u_e, 0] + pos[v_e, 0]) / 2.0
        emid_y = (pos[u_e, 1] + pos[v_e, 1]) / 2.0
        emid   = np.column_stack([emid_x, emid_y])          # (n_e, 2)
        vmid   = pos                                          # (n_v, 2)
        K      = np.exp(-cdist(emid, vmid) ** 2 / (2 * sigma ** 2))  # (n_e, n_v)
        edge_pops = (K * vpop[np.newaxis, :]).sum(axis=1)
        ep_pop.a[:] = edge_pops

        # Assign facility edges uniformly at random per facility type
        fac_masks_srgg: Dict[str, np.ndarray] = {}
        for spec in self._present_facs:
            frac = fac_fracs.get(spec.key, 0.0)
            k    = int(round(frac * n_e))
            fm   = np.zeros(n_e, dtype=bool)
            if k > 0:
                chosen = rng.choice(n_e, size=min(k, n_e), replace=False)
                fm[chosen] = True
            fac_masks_srgg[spec.key] = fm

        return g_srgg, ep_pop, fac_masks_srgg

    def _srgg_percolation(
        self,
        g_srgg: Graph,
        ep_pop: Any,
        fac_masks: Dict[str, np.ndarray],
        n_steps: int,
        recompute_every: int,
        frac_arr: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Run betweenness-based percolation on one SRGG instance.
        Returns {fac_key: penalty_array (len = n_steps+1)}.
        """
        n_e_srgg = g_srgg.num_edges()
        n_steps  = min(n_steps, n_e_srgg)
        src_v    = g_srgg.get_edges()[:, 0]
        total_pop = float(ep_pop.a.sum())
        if total_pop == 0.0:
            total_pop = float(n_e_srgg)
            pop_arr = np.ones(n_e_srgg, dtype=float)
        else:
            pop_arr = ep_pop.a.astype(float).copy()

        active   = np.ones(n_e_srgg, dtype=bool)
        bw_cache: Optional[np.ndarray] = None

        curves: Dict[str, List[float]] = {k: [] for k in fac_masks}

        # Step 0
        _, pen0 = self._srgg_penalties(active, g_srgg, src_v, pop_arr, total_pop, fac_masks)
        for k in fac_masks:
            curves[k].append(pen0.get(k, 0.0))

        for t in range(1, n_steps + 1):
            if bw_cache is None or (t - 1) % recompute_every == 0:
                efilt = g_srgg.new_edge_property("bool")
                efilt.a[:] = active
                gv = GraphView(g_srgg, efilt=efilt)
                _, ep_bw = gt_betweenness(gv)
                bw_cache = ep_bw.a.astype(float).copy()
                bw_cache[~active] = -1.0
            e_star = int(np.argmax(bw_cache))
            active[e_star] = False
            bw_cache[e_star] = -1.0
            _, pen = self._srgg_penalties(active, g_srgg, src_v, pop_arr, total_pop, fac_masks)
            for k in fac_masks:
                curves[k].append(pen.get(k, 0.0))

        # Interpolate to match the empirical frac_arr
        n_pts    = len(frac_arr)
        srgg_arr = np.linspace(0.0, n_steps / n_e_srgg, n_steps + 1)
        result: Dict[str, np.ndarray] = {}
        for k, vals in curves.items():
            result[k] = np.interp(frac_arr, srgg_arr, vals)
        return result

    @staticmethod
    def _srgg_penalties(
        active: np.ndarray,
        g: Graph,
        src_v: np.ndarray,
        pop_arr: np.ndarray,
        total_pop: float,
        fac_masks: Dict[str, np.ndarray],
    ) -> Tuple[float, Dict[str, float]]:
        """Returns (direct_frac, global_penalties) matching _service_penalties."""
        direct_frac = float(pop_arr[~active].sum() / total_pop)
        if not active.any():
            return 1.0, {k: 1.0 for k in fac_masks}
        efilt = g.new_edge_property("bool")
        efilt.a[:] = active
        gv = GraphView(g, efilt=efilt)
        comp_map, _ = label_components(gv)
        comp_arr    = comp_map.a
        edge_comp   = comp_arr[src_v]
        active_comp = edge_comp[active]
        active_pop  = pop_arr[active]
        penalties: Dict[str, float] = {}
        for k, fm in fac_masks.items():
            fac_active = fm[active]
            if not fac_active.any():
                penalties[k] = 1.0
                continue
            comps_with_fac = set(active_comp[fac_active].tolist())
            has_access = np.isin(active_comp, list(comps_with_fac))
            indirect_frac = float(active_pop[~has_access].sum() / total_pop)
            penalties[k] = min(1.0, direct_frac + indirect_frac)
        return direct_frac, penalties

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def _visualize_curves(
        self,
        result: PercolationResult,
        is_random: bool = False,
    ) -> Path:
        """
        Plot percolation curves P_s(t) vs fraction of edges removed for all
        present facilities.  If null model data present, overlays mean ± 1σ band.
        """
        fac_list = self._present_facs
        if not fac_list:
            return self.output_dir / "percolation" / "no_data.png"

        colors   = {spec.key: _PALETTE[i % len(_PALETTE)] for i, spec in enumerate(fac_list)}
        frac_arr = result.frac_arr

        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        fig.patch.set_facecolor("white")

        # Direct-loss reference line (same for all services)
        direct_arr = np.array([s.pop_direct_fraction for s in result.steps])
        ax.fill_between(
            frac_arr, 0, direct_arr,
            color="#BBBBBB", alpha=0.35, zorder=0, label="Direct loss (all services)",
        )
        ax.plot(frac_arr, direct_arr, color="#888888", lw=1.0, ls=":", zorder=1)

        for spec in fac_list:
            pens = np.array([s.penalties.get(spec.key, 0.0) for s in result.steps])
            c    = colors[spec.key]
            lw   = 2.5 if spec.key == "any_crit" else 1.5
            ls   = "--" if spec.key == "any_crit" else "-"

            # Null model band (SRGG or random std)
            if result.null_mean and spec.key in result.null_mean:
                nm = result.null_mean[spec.key]
                ns = result.null_std [spec.key] if result.null_std else np.zeros_like(nm)
                x  = result.frac_arr[: len(nm)]
                ax.fill_between(x, np.clip(nm - ns, 0, 1), np.clip(nm + ns, 0, 1),
                                color=c, alpha=0.15)
                if is_random:
                    ax.plot(x, nm, color=c, lw=lw, ls=ls, label=spec.display)
                    continue  # penalty curves ARE the mean for random
                else:
                    ax.plot(x, nm, color=c, lw=0.8, ls=":", alpha=0.7)

            ax.plot(frac_arr, pens, color=c, lw=lw, ls=ls, label=spec.display)

        ax.set_xlabel("Fraction of edges removed", fontsize=12)
        ax.set_ylabel(r"Population penalty  $P_s^{(t)}$  (direct + indirect)", fontsize=12)
        ax.set_xlim(0.0, frac_arr[-1])
        ax.set_ylim(-0.02, 1.05)
        ax.set_title(result.label, fontsize=13, pad=10)
        ax.grid(True, alpha=0.3, lw=0.5)

        legend = ax.legend(loc="upper left", fontsize=9, frameon=True, framealpha=0.9,
                           ncol=2 if len(fac_list) > 4 else 1)
        fig.tight_layout()

        viz_dir = self.output_dir / "percolation"
        viz_dir.mkdir(parents=True, exist_ok=True)
        slug    = result.scenario
        path    = viz_dir / f"{slug}_curves.png"
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        self.log.info(f"Percolation curves saved: {path}")
        return path

    def visualize_network_state(
        self,
        active: np.ndarray,
        label: str,
        scenario_slug: str,
    ) -> Optional[Path]:
        """
        Geographic edge plot colored by active (gray) vs removed (red).
        Returns path of the saved PNG, or None if no geometry available.
        """
        if not self._has_geom:
            return None

        active_segs:   list = []
        removed_segs:  list = []

        for i, e in enumerate(self.g.edges()):
            flat = self.g.ep["geometry"][e]
            if len(flat) < 4:
                continue
            coords = np.array(flat, dtype=float).reshape(-1, 2).tolist()
            (active_segs if active[i] else removed_segs).append(coords)

        fig, ax = plt.subplots(figsize=(10, 10), dpi=120)
        fig.patch.set_facecolor("white")
        if active_segs:
            ax.add_collection(
                LineCollection(active_segs, colors="#BBBBBB", linewidths=0.5, zorder=1)
            )
        if removed_segs:
            ax.add_collection(
                LineCollection(removed_segs, colors="#D32F2F", linewidths=1.0, zorder=2)
            )
        ax.autoscale_view()
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(label, fontsize=11, pad=8)

        handles = [
            mlines.Line2D([], [], color="#BBBBBB", lw=1.5, label="Active"),
            mlines.Line2D([], [], color="#D32F2F", lw=1.5, label="Removed"),
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.9)

        viz_dir = self.output_dir / "percolation"
        viz_dir.mkdir(parents=True, exist_ok=True)
        safe = label.replace(" ", "_").replace("%", "pct").replace("/", "-")
        path = viz_dir / f"{scenario_slug}_net_{safe}.png"
        fig.savefig(path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # Export (CSV + LaTeX)
    # ------------------------------------------------------------------

    def _export(self, result: PercolationResult) -> None:
        """Write CSV and a summary LaTeX table with key threshold statistics."""
        out = self.output_dir / "percolation"
        out.mkdir(parents=True, exist_ok=True)

        slug = result.scenario

        # Full trajectory CSV
        # Columns: step, fraction_removed, pop_direct, {fac_key} (global = direct+indirect)
        rows = []
        for s in result.steps:
            row = {
                "step": s.step,
                "fraction_removed": s.fraction_removed,
                "pop_direct": s.pop_direct_fraction,
            }
            row.update(s.penalties)
            rows.append(row)
        pd.DataFrame(rows).to_csv(out / f"{slug}_trajectory.csv", index=False)

        # Summary table: T_10, T_50, T_90 per facility (fraction removed when penalty reaches threshold)
        frac_arr = result.frac_arr
        summary_rows = []
        for spec in self._present_facs:
            pens = np.array([s.penalties.get(spec.key, 0.0) for s in result.steps])
            row  = {"Facility": spec.display}
            for pct in (0.10, 0.25, 0.50, 0.75, 0.90):
                idx = np.searchsorted(pens, pct)
                row[f"T{int(100*pct)}%"] = (
                    f"{frac_arr[idx]*100:.1f}%" if idx < len(frac_arr) else ">100%"
                )
            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows).set_index("Facility")
        summary_df.to_csv(out / f"{slug}_summary.csv")

        # LaTeX
        col_fmt = "l " + " c" * len(summary_df.columns)
        header  = "    Facility & " + " & ".join(
            [rf"$T_{{{c}}}$" for c in summary_df.columns]
        ) + r" \\"
        lines = [
            r"\begin{table}[ht]",
            r"    \centering",
            rf"    \caption{{Percolation thresholds — {result.label}. "
            r"$T_k$ is the fraction of edges removed when $P_s^{(t)} \geq k$\%.}}",
            rf"    \label{{tab:perc_{slug}}}",
            r"    \footnotesize",
            r"    \begin{tabular}{" + col_fmt + "}",
            r"    \toprule",
            header,
            r"    \midrule",
        ]
        for fac, row in summary_df.iterrows():
            cells = [str(fac)] + [str(v) for v in row]
            lines.append("    " + " & ".join(cells) + r" \\")
        lines += [r"    \bottomrule", r"    \end{tabular}", r"\end{table}"]
        (out / f"{slug}_summary.tex").write_text("\n".join(lines))
        self.log.info(f"Exported: {slug}_trajectory.csv, {slug}_summary.csv/.tex")

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------

    def print_summary(self, results: List[PercolationResult]) -> None:
        """Report most vulnerable facilities briefly for each scenario."""
        for result in results:
            frac_arr = result.frac_arr
            print()
            print(f"  ── {result.label} {'─' * max(0, 56 - len(result.label))}")
            print(f"  {'Facility':<25}  {'T10%':>6}  {'T25%':>6}  {'T50%':>6}  {'T90%':>6}")
            print(f"  {'─'*25}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}")

            rows_for_sort = []
            for spec in self._present_facs:
                pens = np.array(
                    [s.penalties.get(spec.key, 0.0) for s in result.steps]
                )
                thresholds = {}
                for pct in (0.10, 0.25, 0.50, 0.90):
                    idx = np.searchsorted(pens, pct)
                    thresholds[pct] = frac_arr[idx] if idx < len(frac_arr) else float("inf")
                rows_for_sort.append((spec, thresholds))

            # Sort by T10 (most vulnerable first)
            rows_for_sort.sort(key=lambda x: x[1][0.10])

            for spec, thr in rows_for_sort:
                def _fmt(v):
                    return f"{v*100:.1f}%" if v < float("inf") else ">100%"
                marker = " ← most vulnerable" if spec == rows_for_sort[0][0] else ""
                print(
                    f"  {spec.display:<25}  "
                    f"{_fmt(thr[0.10]):>6}  {_fmt(thr[0.25]):>6}  "
                    f"{_fmt(thr[0.50]):>6}  {_fmt(thr[0.90]):>6}"
                    f"{marker}"
                )
            print()


# ---------------------------------------------------------------------------
# Elevation threshold helper
# ---------------------------------------------------------------------------

def _compute_elevation_thresholds_count(elev_values: np.ndarray) -> int:
    """
    Return a case-specific number of elevation thresholds based on the
    observed elevation range:
      range < 20 m  → 5 thresholds (very flat terrain)
      20–50 m       → 8 thresholds
      50–150 m      → 10 thresholds  (default)
      > 150 m       → 15 thresholds (mountainous)
    """
    elev_range = float(elev_values.max() - elev_values.min())
    if elev_range < 20:
        return 5
    if elev_range < 50:
        return 8
    if elev_range < 150:
        return 10
    return 15


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_percolation(
    graph_path: Path,
    output_dir: Path,
    run_betweenness: bool = True,
    run_random: bool = True,
    run_elevation: bool = True,
    n_steps: Optional[int] = None,
    recompute_every: int = 1,
    run_null_model: bool = False,
    null_m: int = _SRGG_M_DEFAULT,
    n_random_runs: int = 10,
    elevation_ascending: bool = True,
    logger: Optional[logging.Logger] = None,
) -> List[PercolationResult]:
    """
    Run percolation analysis for one or more scenarios.

    Parameters
    ----------
    graph_path          : Path to the enriched .gt file.
    output_dir          : Root output directory.
    run_betweenness     : Run betweenness centrality attack scenario.
    run_random          : Run random failure scenario.
    run_elevation       : Run elevation-based removal scenario.
    n_steps             : Number of steps for betweenness/random scenarios.
                          Defaults to min(500, 30% of edges).
    recompute_every     : Recompute betweenness every K steps.
    run_null_model      : Include SRGG null model in betweenness scenario.
    null_m              : SRGG ensemble size.
    n_random_runs       : Number of random-failure runs.
    elevation_ascending : True = flood (remove lowest first).
    logger              : Optional logger.

    Returns
    -------
    list of PercolationResult (one per requested scenario).
    """
    log = _get_logger(logger)
    perc = Percolator(graph_path=graph_path, output_dir=output_dir, logger=log)

    if n_steps is None:
        n_steps = min(500, max(50, int(perc.n_edges * 0.30)))

    log.info("=" * 70)
    log.info("ANTIPOMPEII Percolation Analysis")
    log.info("=" * 70)

    results: List[PercolationResult] = []

    if run_betweenness:
        log.info(
            f"Scenario 1: Betweenness attack "
            f"(n_steps={n_steps}, recompute_every={recompute_every})"
        )
        results.append(
            perc.run_betweenness(n_steps, recompute_every, run_null_model, null_m)
        )

    if run_random:
        log.info(f"Scenario 2: Random failure (n_steps={n_steps}, n_runs={n_random_runs})")
        results.append(perc.run_random(n_steps, n_random_runs))

    if run_elevation:
        log.info("Scenario 3: Elevation-based removal")
        results.append(perc.run_elevation(ascending=elevation_ascending))

    print("\n" + "=" * 60)
    print("PERCOLATION SUMMARY")
    print("=" * 60)
    perc.print_summary(results)
    print("=" * 60 + "\n")

    log.info(f"Percolation outputs in: {output_dir / 'percolation'}")
    return results
