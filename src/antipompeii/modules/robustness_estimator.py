"""
ANTIPOMPEII Network Robustness and Resilience Estimator

Evaluates structural graph metrics across multiple network states:

  S_0        — intact network (all edges, no disruption applied)
  S_1 … S_k  — network after each disaster event (disrupted edges removed)
  S_rec      — optional recovered / future-state graph (.gt file)

For each state the following metrics are computed:

  n_e         number of active edges
  #C          number of connected components
  κ           vertex connectivity (minimum vertex cut); always approximated
              via the NetworkX White--Newman lower bound and marked **
  λ           edge connectivity (minimum edge cut) via Stoer--Wagner
  d_max       pseudo-diameter (approximate longest shortest path)
  L           average shortest path length (reachable pairs only; marked ** when
              the graph is disconnected)
  E           global network efficiency  E = Σ_{i≠j} 1/d_{ij} / [n(n−1)]
  C           average local clustering coefficient
  R*          Kirchhoff index  R* = Σ_k 1/λ_k  (non-zero Laplacian eigenvalues;
              exact for n_vertices ≤ 1000, sparse-approximated otherwise and
              marked **)
  λ̄           natural connectivity  λ̄ = ln(Σ_i exp(λ_i)/n)  over the adjacency
              spectrum (exact dense eigendecomposition; reported only for
              n_vertices ≤ 2000 because the full spectrum is required)
  b_e^max     maximum edge betweenness centrality
  b̄_e         mean edge betweenness centrality
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.antipompeii.utils.graph_tool_compat import gt, GRAPH_TOOL_AVAILABLE
if GRAPH_TOOL_AVAILABLE:
    from graph_tool import Graph, GraphView
    from graph_tool.centrality import betweenness as gt_betweenness
    from graph_tool.clustering import local_clustering as gt_local_clustering
    from graph_tool.flow import min_cut as gt_min_cut
    from graph_tool.topology import (
        label_components,
        pseudo_diameter as gt_pseudo_diameter,
        shortest_distance as gt_shortest_distance,
    )

_SPECTRAL_AVAILABLE = False
try:
    from graph_tool.spectral import (  # type: ignore
        laplacian as _gt_laplacian,
        adjacency as _gt_adjacency,
    )
    import scipy.linalg as _sla
    import scipy.sparse.linalg as _spla
    _SPECTRAL_AVAILABLE = True
except ImportError:
    pass

_NETWORKX_AVAILABLE = False
try:
    import networkx as _nx  # noqa: F401
    _NETWORKX_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Metric specification
# (internal_attr, LaTeX column header, display format for finite values)
# ---------------------------------------------------------------------------

METRIC_SPECS: List[Tuple[str, str, str]] = [
    ("n_edges",         r"$n_e$",              "{:,.0f}"),
    ("n_components",    r"$\#C$",              "{:,.0f}"),
    ("vertex_conn",     r"$\kappa$",           "{:.0f}"),
    ("edge_conn",       r"$\lambda$",          "{:.0f}"),
    ("diameter",        r"$d_{\max}$",         "{:.0f}"),
    ("avg_path_len",    r"$L$",                "{:.3f}"),
    ("efficiency",      r"$E$",                "{:.4f}"),
    ("clustering",      r"$C$",                "{:.4f}"),
    ("eff_resistance",  r"$R^{*}$",            "{:.3f}"),
    ("natural_conn",    r"$\bar{\lambda}$",    "{:.4f}"),
    ("betweenness_max", r"$b_e^{\max}$",       "{:.4f}"),
    ("betweenness_avg", r"$\bar{b}_e$",        "{:.5f}"),
]
METRIC_KEYS: List[str] = [k for k, _, _ in METRIC_SPECS]

# Short names used in the console summary
METRIC_SHORT: Dict[str, str] = {
    "n_edges":         "n_e",
    "n_components":    "#C",
    "vertex_conn":     "κ",
    "edge_conn":       "λ",
    "diameter":        "d_max",
    "avg_path_len":    "L",
    "efficiency":      "E",
    "clustering":      "C",
    "eff_resistance":  "R*",
    "natural_conn":    "λ̄",
    "betweenness_max": "b_max",
    "betweenness_avg": "b_avg",
}


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class StateMetrics:
    """All structural metrics for one graph state."""
    label:           str
    n_vertices:      int
    n_edges:         int
    n_components:    int
    vertex_conn:     float
    edge_conn:       float
    diameter:        float
    avg_path_len:    float
    efficiency:      float
    clustering:      float
    eff_resistance:  float
    natural_conn:    float
    betweenness_max: float
    betweenness_avg: float
    approx_paths:    bool = False   # L/E from reachable pairs only (disconnected)
    approx_resist:   bool = False   # R* from partial Laplacian spectrum (large)
    approx_vconn:    bool = False   # κ from White--Newman lower-bound approx


@dataclass
class RobustnessReport:
    """Full robustness analysis outputs."""
    states:     List[StateMetrics]
    table_df:   pd.DataFrame   # formatted comparison table
    output_dir: Path


# ---------------------------------------------------------------------------
# Internal metric helpers
# ---------------------------------------------------------------------------

from src.antipompeii.utils.logger import get_module_logger as _get_logger


def _weight_ep(g: Any, prop: str) -> Optional[Any]:
    if prop and prop in g.ep:
        return g.ep[prop]
    return None


def _n_components(g: Any) -> int:
    if g.num_vertices() == 0:
        return 0
    comp, _ = label_components(g)
    return int(comp.a.max()) + 1


def _diameter(g: Any, weights: Optional[Any]) -> float:
    if g.num_vertices() < 2 or g.num_edges() == 0:
        return np.nan
    try:
        d, _ = gt_pseudo_diameter(g, weights=weights)
        return float(d)
    except Exception:
        return np.nan


# Threshold for BFS integer sentinel (graph-tool uses INT_MAX for unreachable)
_LARGE = 1e15


def _path_and_efficiency(
    g: Any,
    weights: Optional[Any],
) -> Tuple[float, float, bool]:
    """
    Compute (avg_shortest_path_length, global_efficiency, approx_flag).
    """
    n_v = g.num_vertices()
    if n_v < 2:
        return np.nan, 0.0, False

    total_d  = 0.0
    n_reach  = 0
    sum_inv  = 0.0
    any_disc = False

    for v in g.vertices():
        dist_map = gt_shortest_distance(g, source=v, weights=weights)
        d = np.asarray(dist_map.a, dtype=float)

        # Exclude self (d == 0)
        pos = d[d > 0]

        # Reachable: finite and below the integer sentinel
        reachable = pos[pos < _LARGE]
        if len(reachable) < len(pos):
            any_disc = True

        total_d += reachable.sum()
        n_reach += len(reachable)
        if len(reachable):
            sum_inv += (1.0 / reachable).sum()

    L = total_d / n_reach if n_reach > 0 else np.nan
    E = sum_inv / (n_v * (n_v - 1))
    return L, E, any_disc


def _clustering(g: Any) -> float:
    """Average local clustering coefficient (undirected interpretation)."""
    try:
        lc = gt_local_clustering(g, undirected=True)
        c = lc.a.astype(float)
        valid = c[np.isfinite(c)]
        return float(valid.mean()) if len(valid) else 0.0
    except Exception:
        return np.nan


def _kirchhoff(g: Any, n_eigs: int = 200) -> Tuple[float, bool]:
    """
    Kirchhoff index (total effective resistance):
      R* = Σ_k 1/λ_k  over all non-zero Laplacian eigenvalues.

    Returns (R_star, approx_flag).  Returns (nan, False) when unavailable.
    """
    if not _SPECTRAL_AVAILABLE:
        return np.nan, False

    n = g.num_vertices()
    if n < 2:
        return 0.0, False

    try:
        L = _gt_laplacian(g)
    except Exception:
        return np.nan, False

    approx = False
    try:
        if n <= 1000:
            eigvals = _sla.eigvalsh(L.toarray())
        else:
            k = min(n_eigs, n - 1)
            eigvals = _spla.eigsh(L, k=k, which="SM", return_eigenvectors=False)
            approx = True

        nonzero = eigvals[eigvals > 1e-8]
        if len(nonzero) == 0:
            return np.nan, approx
        return float(np.sum(1.0 / nonzero)), approx
    except Exception:
        return np.nan, approx


def _to_networkx(g: Any) -> Any:
    """
    Convert a graph-tool graph (or GraphView) to a plain undirected NetworkX Graph
    """
    import networkx as nx
    gx = nx.Graph()
    gx.add_nodes_from(range(g.num_vertices()))
    gx.add_edges_from((int(e.source()), int(e.target())) for e in g.edges())
    return gx


def _vertex_connectivity(
    g: Any,
    n_components_val: int,
    logger: Optional[logging.Logger] = None,
) -> Tuple[float, bool]:
    """
    Vertex connectivity κ(G): minimum number of vertices whose removal
    disconnects the graph.  Returns (κ, approx_flag).
    """
    if g.num_vertices() < 2 or g.num_edges() == 0:
        return 0.0, False
    if n_components_val > 1:
        return 0.0, False
    if not _NETWORKX_AVAILABLE:
        return np.nan, False
    log = logger or logging.getLogger(__name__)
    try:
        from networkx.algorithms.approximation import (
            node_connectivity as _approx_node_connectivity,
        )
        gx = _to_networkx(g)
        return float(_approx_node_connectivity(gx)), True
    except Exception as exc:
        log.warning(
            f"Vertex connectivity computation failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return np.nan, False


def _edge_connectivity(
    g: Any,
    n_components_val: int,
    logger: Optional[logging.Logger] = None,
) -> float:
    """
    Edge connectivity λ(G): minimum number of edges whose removal
    disconnects the graph.

    Returns 0 for disconnected or trivial graphs; nan if the algorithm
    raises.
    """
    if g.num_vertices() < 2 or g.num_edges() == 0:
        return 0.0
    if n_components_val > 1:
        return 0.0
    log = logger or logging.getLogger(__name__)
    try:
        ones = g.new_edge_property("double")
        ones.a = 1.0
        cut_val, _ = gt_min_cut(g, ones)
        return float(cut_val)
    except Exception as exc:
        log.warning(
            f"Edge connectivity (min_cut) failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return np.nan


def _natural_connectivity(
    g: Any,
    max_n_for_exact: int = 2000,
    logger: Optional[logging.Logger] = None,
) -> float:
    """
    Natural connectivity λ̄ = ln(Σ_i exp(λ_i) / n) over the adjacency spectrum.
    """
    if not _SPECTRAL_AVAILABLE:
        return np.nan
    n = g.num_vertices()
    if n < 2:
        return 0.0
    log = logger or logging.getLogger(__name__)
    if n > max_n_for_exact:
        log.info(
            f"  Natural connectivity skipped: n_vertices={n:,} > "
            f"{max_n_for_exact:,} (full adjacency spectrum required)."
        )
        return np.nan
    try:
        A = _gt_adjacency(g)
        eigvals = _sla.eigvalsh(A.toarray())
        m = float(eigvals.max())
        # log-sum-exp form for numerical stability
        return m + float(np.log(np.exp(eigvals - m).sum() / n))
    except Exception as exc:
        log.warning(
            f"Natural connectivity computation failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return np.nan


def _sanitize_weights(g: Any, weights: Optional[Any]) -> Optional[Any]:
    """
    Return a clean copy of *weights* with NaN / non-positive values replaced
    by the median of the finite-positive values (or 1.0 if none exist).
    Returns *weights* unchanged when it is None or already fully finite+positive.
    """
    if weights is None:
        return None
    w_arr = weights.a.astype(float)
    bad   = ~np.isfinite(w_arr) | (w_arr <= 0)
    if not bad.any():
        return weights
    finite_pos = w_arr[~bad]
    fill = float(np.median(finite_pos)) if len(finite_pos) > 0 else 1.0
    clean = g.new_edge_property("double")
    clean.a[:] = np.where(bad, fill, w_arr)
    return clean


def _betweenness(
    g: Any,
    weights: Optional[Any],
    logger: Optional[logging.Logger] = None,
) -> Tuple[float, float]:
    """Return (max, mean) edge betweenness centrality."""
    if g.num_edges() == 0:
        return 0.0, 0.0
    log = logger or logging.getLogger(__name__)

    try:
        _, ep = gt_betweenness(g, weight=weights)
        b = ep.a.astype(float)

        # np.nanmax/nanmean pass through inf, which graph-tool produces for
        # self-loop edges (zero-length cycles corrupt Dijkstra path counts).
        # Filter to strictly finite values and warn when any are dropped.
        finite = b[np.isfinite(b)]
        n_bad = len(b) - len(finite)
        if n_bad:
            log.warning(
                f"  {n_bad} non-finite betweenness value(s) discarded "
                "(likely self-loop edges in the graph; rebuild with graph_builder "
                "to eliminate them at source)."
            )
        if not len(finite):
            return np.nan, np.nan
        return float(finite.max()), float(finite.mean())
    except Exception as exc:
        log.warning(
            f"  Betweenness (weighted) failed: {type(exc).__name__}: {exc}"
        )
        if weights is not None:
            try:
                log.info("  Retrying betweenness without weights (hop-count) …")
                _, ep = gt_betweenness(g)
                b = ep.a.astype(float)
                return float(np.nanmax(b)), float(np.nanmean(b))
            except Exception as exc2:
                log.warning(
                    f"  Betweenness (unweighted) also failed: "
                    f"{type(exc2).__name__}: {exc2}"
                )
        return np.nan, np.nan


# ---------------------------------------------------------------------------
# Core estimator
# ---------------------------------------------------------------------------

class RobustnessEstimator:
    """
    Loads a graph-tool .gt network and evaluates structural robustness metrics
    across multiple states (intact → disrupted → recovered).

    Parameters
    ----------
    graph_path  : Path to the enriched .gt file from graph_builder.py.
    output_dir  : Root output directory; sub-directories are created as needed.
    weight_prop : Edge property name used as path weight (default: "length").
                  Pass "" or None for unweighted (hop-count) analysis.
    logger      : Optional logger.
    """

    def __init__(
        self,
        graph_path: Path,
        output_dir: Path,
        weight_prop: str = "length",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if not GRAPH_TOOL_AVAILABLE:
            raise ImportError(
                "graph-tool is required for robustness estimation. "
                "Install via: conda install -c conda-forge graph-tool"
            )
        self.log = _get_logger(logger)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.weight_prop = weight_prop or ""

        graph_path = Path(graph_path)
        self.log.info(f"Loading base graph: {graph_path}")
        self.g: Graph = gt.load_graph(str(graph_path))
        self.n_edges = self.g.num_edges()
        self.log.info(
            f"Graph: {self.g.num_vertices():,} vertices, {self.n_edges:,} edges"
        )

        if "disruption" in self.g.ep:
            self._disruption = self.g.ep["disruption"].a.astype(int).copy()
        else:
            self._disruption = np.zeros(self.n_edges, dtype=int)
            self.log.warning("No 'disruption' edge property found in base graph.")

    # ------------------------------------------------------------------
    # Public runner
    # ------------------------------------------------------------------

    def estimate(
        self,
        disruption_vectors: List[np.ndarray],
        state_labels: List[str],
        compound: bool = True,
        recovery_path: Optional[Path] = None,
        recovery_label: str = r"$S_2$",
    ) -> RobustnessReport:
        """
        Evaluate robustness for all states and produce comparison outputs.
        """
        states: List[StateMetrics] = []

        # S_0 — intact (no disruption filter)
        self.log.info("── State S_0 (intact) ──────────────────────────────────────")
        states.append(self._eval_state("$S_0$", GraphView(self.g)))

        # S_k — disrupted states
        cumulative = np.zeros(self.n_edges, dtype=int)
        for k, (vec, label) in enumerate(zip(disruption_vectors, state_labels), 1):
            if compound:
                cumulative = np.clip(cumulative + (vec == 1), 0, 1)
                mask = cumulative.astype(bool)
            else:
                mask = (vec == 1).astype(bool)

            self.log.info(
                f"── State S_{k} ({label}): "
                f"{mask.sum():,}/{self.n_edges:,} edges removed ────"
            )
            efilt = self.g.new_edge_property("bool")
            efilt.a = ~mask
            g_disrupted = GraphView(self.g, efilt=efilt)
            states.append(self._eval_state(label, g_disrupted))

        # S_rec — recovery (optional, may have different topology)
        if recovery_path is not None:
            recovery_path = Path(recovery_path)
            if recovery_path.exists():
                self.log.info(
                    f"── Recovery state ({recovery_path.name}) ──────────────────"
                )
                g_rec = gt.load_graph(str(recovery_path))
                states.append(self._eval_state(recovery_label, GraphView(g_rec)))
            else:
                self.log.warning(f"Recovery graph not found: {recovery_path}")

        table_df = self._build_table(states)
        self._export(table_df, states)
        return RobustnessReport(
            states=states, table_df=table_df, output_dir=self.output_dir
        )

    # ------------------------------------------------------------------
    # Metric evaluation
    # ------------------------------------------------------------------

    def _eval_state(self, label: str, g: Any) -> StateMetrics:
        """Compute all structural metrics for one graph state."""
        raw_w   = _weight_ep(g, self.weight_prop)
        weights = _sanitize_weights(g, raw_w)
        if weights is not raw_w and raw_w is not None:
            self.log.warning(
                "  Edge weights contain NaN / non-positive values; "
                "replaced with median for metric computations."
            )
        n_v = g.num_vertices()
        n_e = g.num_edges()
        self.log.info(f"  {label}: {n_v:,} vertices, {n_e:,} active edges")

        n_comp = _n_components(g)
        diam   = _diameter(g, weights)

        self.log.info("  Computing edge and vertex connectivity …")
        vconn, approx_vconn = _vertex_connectivity(g, n_comp, logger=self.log)
        econn = _edge_connectivity(g, n_comp, logger=self.log)

        self.log.info("  Computing shortest-path metrics (L, E) …")
        L, E, approx_paths = _path_and_efficiency(g, weights)

        self.log.info("  Computing clustering coefficient …")
        C = _clustering(g)

        self.log.info("  Computing Kirchhoff index (effective resistance) …")
        R_star, approx_R = _kirchhoff(g)

        self.log.info("  Computing natural connectivity …")
        nat = _natural_connectivity(g, logger=self.log)

        self.log.info("  Computing edge betweenness centrality …")
        b_max, b_avg = _betweenness(g, weights, logger=self.log)

        m = StateMetrics(
            label=label,
            n_vertices=n_v,
            n_edges=n_e,
            n_components=n_comp,
            vertex_conn=vconn,
            edge_conn=econn,
            diameter=diam,
            avg_path_len=L,
            efficiency=E,
            clustering=C,
            eff_resistance=R_star,
            natural_conn=nat,
            betweenness_max=b_max,
            betweenness_avg=b_avg,
            approx_paths=approx_paths,
            approx_resist=approx_R,
            approx_vconn=approx_vconn,
        )
        self.log.info(
            f"  → #C={n_comp:,}  κ={vconn:.0f}  λ={econn:.0f}  "
            f"d_max={diam:.0f}  L={L:.3f}  E={E:.5f}  C={C:.4f}  "
            f"R*={R_star:.3f}  λ̄={nat:.4f}  "
            f"b_max={b_max:.4f}  b_avg={b_avg:.5f}"
        )
        return m

    # ------------------------------------------------------------------
    # Table construction
    # ------------------------------------------------------------------

    def _build_table(self, states: List[StateMetrics]) -> pd.DataFrame:
        """
        Build a formatted comparison DataFrame:
          rows  = one per state + consecutive Δ rows
          cols  = one per metric (LaTeX header names)
        """
        def _fmt(key: str, val: float, fmt: str, approx: bool) -> str:
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return "—"
            s = fmt.format(val)
            return s + r"$^{**}$" if approx else s

        def _delta(v0: float, v1: float) -> str:
            if any(
                x is None or (isinstance(x, float) and np.isnan(x))
                for x in (v0, v1)
            ):
                return "—"
            if v0 == 0:
                return r"$+\infty$" if v1 != 0 else r"$0\%$"
            pct = 100.0 * (v1 - v0) / abs(v0)
            sign = "+" if pct >= 0 else ""
            return f"${sign}{pct:.1f}\\%$"

        rows: List[Tuple[str, Dict]] = []

        # State rows
        for s in states:
            row: Dict = {}
            for key, _, fmt in METRIC_SPECS:
                val = getattr(s, key)
                approx = (
                    (key in ("avg_path_len", "efficiency") and s.approx_paths)
                    or (key == "eff_resistance" and s.approx_resist)
                    or (key == "vertex_conn" and s.approx_vconn)
                )
                row[key] = _fmt(key, val, fmt, approx)
            rows.append((s.label, row))

        # Delta rows (consecutive pairs)
        for i in range(1, len(states)):
            s_prev, s_curr = states[i - 1], states[i]
            row = {}
            for key, _, _ in METRIC_SPECS:
                row[key] = _delta(getattr(s_prev, key), getattr(s_curr, key))
            rows.append((f"$\\delta_{{{i}}}$", row))

        index = [label for label, _ in rows]
        data  = [row   for _, row   in rows]
        df = pd.DataFrame(data, index=index)
        df.rename(columns={k: h for k, h, _ in METRIC_SPECS}, inplace=True)
        df.index.name = "State"
        return df

    # ------------------------------------------------------------------
    # Export (CSV + LaTeX)
    # ------------------------------------------------------------------

    def _export(self, table_df: pd.DataFrame, states: List[StateMetrics]) -> None:
        out = self.output_dir / "robustness"
        out.mkdir(parents=True, exist_ok=True)

        # Raw CSV (numeric values)
        csv_rows = []
        for s in states:
            row: Dict = {"state": s.label, "n_vertices": s.n_vertices}
            for key in METRIC_KEYS:
                row[key] = getattr(s, key)
            csv_rows.append(row)
        pd.DataFrame(csv_rows).set_index("state").to_csv(out / "robustness.csv")
        self.log.info(f"CSV saved: {out / 'robustness.csv'}")

        tex = self._to_latex(table_df, states)
        (out / "robustness.tex").write_text(tex)
        self.log.info(f"LaTeX saved: {out / 'robustness.tex'}")

    def _to_latex(
        self, df: pd.DataFrame, states: List[StateMetrics]
    ) -> str:
        n_cols = len(df.columns)
        col_fmt = "l " + " ".join(["c"] * n_cols)

        delta_prefix = r"$\delta_"
        state_idx = [i for i in df.index if not str(i).startswith(delta_prefix)]
        delta_idx  = [i for i in df.index if     str(i).startswith(delta_prefix)]

        def _row(label: str) -> str:
            cells = [label] + [str(df.loc[label, c]) for c in df.columns]
            return "    " + " & ".join(cells) + r" \\"

        note = (
            r"$^{**}$~Approximations: "
            r"$L$ averaged over reachable pairs only (disconnected graph); "
            r"$R^{*}$ from a partial Laplacian spectrum ($n > 1000$); "
            r"$\kappa$ from a White--Newman lower-bound approximation. "
            r"$\bar{\lambda}$ requires the full adjacency spectrum and is "
            r"reported only for $n \leq 2000$."
        )

        lines = [
            r"\begin{table}[htbp]",
            r"    \centering",
            r"    \caption[Road network robustness and resilience properties comparison]{%",
            r"        Road network robustness and resilience properties comparison.",
            r"        $n_e$ = edges, $\#C$ = components, $\kappa$ = vertex connectivity,",
            r"        $\lambda$ = edge connectivity, $d_{\max}$ = pseudo-diameter,",
            r"        $L$ = average shortest path length, $E$ = network efficiency,",
            r"        $C$ = clustering coefficient, $R^{*}$ = Kirchhoff index,",
            r"        $\bar{\lambda}$ = natural connectivity,",
            r"        $b_e^{\max}$ = max edge betweenness, $\bar{b}_e$ = mean edge betweenness.",
            f"        {note}",
            r"    }",
            r"    \label{tab:netrobustness}",
            r"    \footnotesize",
            r"    \begin{tabular}{" + col_fmt + "}",
            r"    \toprule",
            "    " + " & ".join([""] + list(df.columns)) + r" \\",
            r"    \midrule",
        ]
        for label in state_idx:
            lines.append(_row(label))
        if delta_idx:
            lines.append(r"    \midrule")
            for label in delta_idx:
                lines.append(_row(label))
        lines += [
            r"    \bottomrule",
            r"    \end{tabular}",
            r"\end{table}",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------

    @staticmethod
    def print_summary(states: List[StateMetrics]) -> None:
        """Print a compact aligned comparison table to stdout."""
        FMT = {
            "n_edges":         "{:>10,.0f}",
            "n_components":    "{:>8,.0f}",
            "vertex_conn":     "{:>6,.0f}",
            "edge_conn":       "{:>6,.0f}",
            "diameter":        "{:>7.0f}",
            "avg_path_len":    "{:>9.3f}",
            "efficiency":      "{:>9.5f}",
            "clustering":      "{:>9.5f}",
            "eff_resistance":  "{:>9.3f}",
            "natural_conn":    "{:>9.4f}",
            "betweenness_max": "{:>9.5f}",
            "betweenness_avg": "{:>10.6f}",
        }
        HDR = {
            "n_edges":         "      n_e",
            "n_components":    "      #C",
            "vertex_conn":     "     κ",
            "edge_conn":       "     λ",
            "diameter":        "  d_max",
            "avg_path_len":    "        L",
            "efficiency":      "        E",
            "clustering":      "        C",
            "eff_resistance":  "       R*",
            "natural_conn":    "       λ̄",
            "betweenness_max": "    b_max",
            "betweenness_avg": "     b_avg",
        }
        DLTW = 10   # width for delta columns

        label_w = max(len(s.label) for s in states) + 2

        bar  = "  " + "─" * (label_w + sum(len(v) for v in HDR.values()) + 2)
        hdr  = f"  {'State':<{label_w}}" + "".join(HDR.values())

        print()
        print(bar)
        print("  ROBUSTNESS COMPARISON")
        print(bar)
        print(hdr)
        print(bar)

        for s in states:
            suffix = " **" if s.approx_paths or s.approx_resist else ""
            vals = []
            for key in METRIC_KEYS:
                v = getattr(s, key)
                if isinstance(v, float) and np.isnan(v):
                    # Match column width
                    w = len(FMT[key].format(0))
                    vals.append(f"{'—':>{w}}")
                else:
                    vals.append(FMT[key].format(v))
            print(f"  {s.label:<{label_w}}" + "".join(vals) + suffix)

        # Delta rows
        print(bar)
        for i in range(1, len(states)):
            s0, s1 = states[i - 1], states[i]
            dlts = []
            for key in METRIC_KEYS:
                v0, v1 = getattr(s0, key), getattr(s1, key)
                if any(
                    x is None or (isinstance(x, float) and np.isnan(x))
                    for x in (v0, v1)
                ):
                    dlts.append(f"{'—':>{DLTW}}")
                elif v0 == 0:
                    dlts.append(f"{'∞':>{DLTW}}" if v1 != 0 else f"{'0%':>{DLTW}}")
                else:
                    pct = 100.0 * (v1 - v0) / abs(v0)
                    s = f"{'+' if pct >= 0 else ''}{pct:.1f}%"
                    dlts.append(f"{s:>{DLTW}}")
            print(f"  {f'δ{i}':<{label_w}}" + "".join(dlts))

        print(bar)
        print()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def estimate_robustness(
    graph_path: Path,
    output_dir: Path,
    additional_graph_paths: Optional[List[Path]] = None,
    recovery_path: Optional[Path] = None,
    recovery_label: str = r"$S_2$",
    scenario_labels: Optional[List[str]] = None,
    compound: bool = True,
    weight_prop: str = "length",
    logger: Optional[logging.Logger] = None,
) -> RobustnessReport:
    """
    Estimate network robustness across all graph states.
    """
    log = _get_logger(logger)
    graph_path = Path(graph_path)
    estimator = RobustnessEstimator(
        graph_path=graph_path,
        output_dir=output_dir,
        weight_prop=weight_prop,
        logger=log,
    )

    all_paths = [graph_path] + [Path(p) for p in (additional_graph_paths or [])]
    disruption_vectors: List[np.ndarray] = []
    valid_labels: List[str] = []

    for idx, p in enumerate(all_paths):
        default_label = f"Event {idx + 1}"
        label = (
            (scenario_labels or [])[idx]
            if scenario_labels and idx < len(scenario_labels)
            else default_label
        )
        if idx == 0:
            vec = estimator._disruption.copy()
        else:
            if not p.exists():
                log.warning(f"Additional graph not found: {p}; skipping.")
                continue
            try:
                g_extra = gt.load_graph(str(p))
            except Exception as exc:
                log.warning(f"Could not load {p}: {exc}; skipping.")
                continue
            if g_extra.num_edges() != estimator.n_edges:
                log.warning(
                    f"{p.name} has {g_extra.num_edges()} edges vs "
                    f"{estimator.n_edges} in base graph; skipping."
                )
                continue
            if "disruption" not in g_extra.ep:
                log.warning(f"No 'disruption' property in {p.name}; skipping.")
                continue
            vec = g_extra.ep["disruption"].a.astype(int).copy()

        disruption_vectors.append(vec)
        valid_labels.append(label)

    if not disruption_vectors:
        raise RuntimeError(
            "No valid disruption vector could be extracted from the provided graph(s)."
        )

    log.info("=" * 70)
    log.info("ANTIPOMPEII Robustness Estimation")
    log.info("=" * 70)

    report = estimator.estimate(
        disruption_vectors=disruption_vectors,
        state_labels=valid_labels,
        compound=compound,
        recovery_path=recovery_path,
        recovery_label=recovery_label,
    )

    RobustnessEstimator.print_summary(report.states)

    log.info("=" * 70)
    log.info(
        f"Robustness estimation complete: {len(report.states)} state(s). "
        f"Outputs in: {output_dir / 'robustness'}"
    )
    log.info("=" * 70)

    return report
