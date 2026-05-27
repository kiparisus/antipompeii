"""
ANTIPOMPEII Network Statistics and Vulnerability Analysis Module

Takes:
- One or more enriched graph-tool .gt files produced by graph_builder.py
  (each encoding one disaster scenario via its `disruption` edge property).
- Output directory for CSVs, LaTeX tables, and visualizations.

Does:
1. Applies each disaster's disruption layer as an edge filter on the base
   graph.  For multiple events the user chooses:
     - Compound   : each event additionally filters edges from the previous state.
     - Independent: each event is analyzed against the full intact network.
2. Identifies DIRECT impact:  edges with disruption == 1.
3. Identifies INDIRECT impact: non-disrupted edges whose connected component
   (in the filtered graph) has lost access to at least one facility type.
   Computed per facility category.
4. Generates four output products per scenario:
     a. Road-network table: counts/share by highway class (direct + indirect).
     b. Facility table: edge counts + unique facility counts; multi-edge
        facilities checked for complete shutdown vs. partial hindrance.
     c. Population table: population losing access (direct, indirect, total),
        disaggregated by sex × age band, per facility type and overall.
     d. Visualization PNG: geographic edge map colored by impact state.
   All tables are exported as CSV and LaTeX.
5. Prints a brief console summary.

Facility categories detected via:
  - fac_amenity edge property   (semicolon-separated OSM amenity tag values)
  - fac_layer_name edge property (semicolon-separated ANTIPOMPEII layer names)

Highway classification handles both plain strings ("residential") and
list-format strings stored by osmnx ("['residential', 'service']").
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive — safe in CLI context
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

from src.antipompeii.utils.graph_tool_compat import gt, GRAPH_TOOL_AVAILABLE
if GRAPH_TOOL_AVAILABLE:
    from graph_tool import Graph, GraphView
    from graph_tool.topology import label_components


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Highway class → list of OSM tag values to match (including _link variants)
HIGHWAY_CLASSES: Dict[str, List[str]] = {
    "Motorway":    ["motorway",  "motorway_link"],
    "Trunk":       ["trunk",     "trunk_link"],
    "Primary":     ["primary",   "primary_link"],
    "Secondary":   ["secondary", "secondary_link"],
    "Tertiary":    ["tertiary",  "tertiary_link"],
    "Residential": ["residential"],
    "Service":     ["service"],
}
# "Other" catches everything not matched above

# Facility catalog — order determines table row order
@dataclass(frozen=True)
class FacilitySpec:
    key: str
    display: str
    amenity_values: Tuple[str, ...]
    layer_values: Tuple[str, ...]


FACILITIES: List[FacilitySpec] = [
    FacilitySpec("hospital",   "Hospital",              ("hospital",),                          ()),
    FacilitySpec("clinic",     "Clinic",                ("clinic",),                            ()),
    FacilitySpec("pharmacy",   "Pharmacy",              ("pharmacy",),                          ()),
    FacilitySpec("police",     "Police station",        ("police",),                            ()),
    FacilitySpec("fire",       "Fire station",          ("fire_station", "ambulance_station"),   ()),
    FacilitySpec("shelter",    "Emergency shelter",     ("shelter", "bunker"),                  ("Emergency",)),
    FacilitySpec("conv_shelt", "Convertible shelter",   ("school", "kindergarten",
                                                          "place_of_worship"),                  ("Convertible Shelter",)),
    FacilitySpec("health_all", "Health (all)",          ("hospital", "clinic", "pharmacy"),     ("Health",)),
    FacilitySpec("emerg_all",  "Emergency (all)",       ("police", "fire_station",
                                                          "ambulance_station", "shelter",
                                                          "bunker"),                            ("Emergency",)),
    FacilitySpec("any_crit",   "Any critical facility", ("hospital", "clinic", "pharmacy",
                                                          "police", "fire_station",
                                                          "ambulance_station", "shelter",
                                                          "bunker", "school", "kindergarten",
                                                          "place_of_worship"),                  ("Health", "Emergency",
                                                                                                 "Convertible Shelter")),
]

# Population demographic bands — (column_suffix, display_label, prefix)
POP_BANDS: List[Tuple[str, str, str]] = [
    ("total",     "Total",       "pop_total"),
    ("f_0_14",    "Female 0–14", "pop_f_0_14"),
    ("f_15_64",   "Female 15–64","pop_f_15_64"),
    ("f_65_plus", "Female 65+",  "pop_f_65_plus"),
    ("m_0_14",    "Male 0–14",   "pop_m_0_14"),
    ("m_15_64",   "Male 15–64",  "pop_m_15_64"),
    ("m_65_plus", "Male 65+",    "pop_m_65_plus"),
]

COLORS = {
    "normal":   "#BBBBBB",
    "indirect": "#FF8C00",  # orange
    "direct":   "#D32F2F",  # red
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

from src.antipompeii.utils.logger import get_module_logger as _get_logger


def _str_array(g: Graph, prop: str) -> np.ndarray:
    """Extract a string edge property as a numpy object array."""
    if prop not in g.ep:
        return np.full(g.num_edges(), "", dtype=object)
    return np.array([g.ep[prop][e] for e in g.edges()], dtype=object)


def _num_array(g: Graph, prop: str, default: float = 0.0) -> np.ndarray:
    """Extract a numeric edge property as a float64 numpy array."""
    if prop not in g.ep:
        return np.full(g.num_edges(), default, dtype="float64")
    try:
        return g.ep[prop].a.astype("float64")
    except Exception:
        arr = np.array([g.ep[prop][e] for e in g.edges()], dtype="float64")
        return np.nan_to_num(arr, nan=default)


def _classify_highway(hw_str: str) -> str:
    """
    Map a highway property value to one of the HIGHWAY_CLASSES keys.

    Handles both plain strings ("residential") and list-format strings
    produced by osmnx ("['footway', 'service']").
    """
    s = hw_str.strip()
    if s.startswith("["):
        # Parse Python list literal: "['a', 'b']"
        vals = [v.strip().strip("'\"\t ") for v in s.strip("[] ").split(",")]
    else:
        vals = [s]

    for cls_name, cls_vals in HIGHWAY_CLASSES.items():
        for v in vals:
            if v in cls_vals:
                return cls_name
    return "Other"


def _has_facility_type(amenity_str: str, layer_str: str, spec: FacilitySpec) -> bool:
    """
    Return True if an edge's facility tags match the given FacilitySpec.
    Both amenity and layer fields may contain semicolon-separated values.
    """
    amenity_vals = {v.strip() for v in amenity_str.split(";") if v.strip()}
    layer_vals   = {v.strip() for v in layer_str.split(";") if v.strip()}

    if spec.amenity_values and amenity_vals.intersection(spec.amenity_values):
        return True
    if spec.layer_values and layer_vals.intersection(spec.layer_values):
        return True
    return False


def _detect_pop_year(g: Graph) -> Optional[str]:
    """Return the most recent year suffix found in pop_total_* edge props."""
    years = []
    for prop in g.ep.keys():
        if prop.startswith("pop_total_"):
            suffix = prop[len("pop_total_"):]
            if suffix.isdigit():
                years.append(int(suffix))
    if not years:
        return None
    return str(max(years))


def _pop_arrays(g: Graph, year: str) -> Dict[str, np.ndarray]:
    """Return dict of {band_suffix: numpy array} for the given year."""
    result = {}
    for suffix, _, prefix in POP_BANDS:
        col = f"{prefix}_{year}"
        result[suffix] = _num_array(g, col, default=0.0)
    return result


def _unique_facility_keys(fac_name_arr: np.ndarray,
                           fac_amenity_arr: np.ndarray,
                           fac_layer_arr: np.ndarray,
                           edge_mask: np.ndarray) -> Set[str]:
    """
    Return a set of unique facility identifiers among the masked edges.
    Key = name if non-empty, else "layer::amenity" compound.
    """
    keys: Set[str] = set()
    for i in np.where(edge_mask)[0]:
        name = str(fac_name_arr[i]).split(";")[0].strip()
        layer = str(fac_layer_arr[i]).split(";")[0].strip()
        amenity = str(fac_amenity_arr[i]).split(";")[0].strip()
        if name:
            keys.add(f"{layer}::{name}")
        elif layer or amenity:
            keys.add(f"{layer}::{amenity}")
    return keys


def _pct(n: int, total: int, decimals: int = 1) -> str:
    """Format count with percentage, e.g. '56 (4.5%)'."""
    if total == 0:
        return f"{n} (—)"
    return f"{n} ({round(100 * n / total, decimals)}\\%)"


def _to_latex(df: pd.DataFrame, caption: str = "", label: str = "") -> str:
    """Render a DataFrame as a LaTeX table with booktabs rules."""
    n_cols = len(df.columns)
    col_fmt = "l" + "r" * n_cols
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{" + col_fmt + "}",
        r"\toprule",
    ]
    # Header
    header_cells = [""] + [str(c) for c in df.columns]
    lines.append(" & ".join(header_cells) + r" \\")
    lines.append(r"\midrule")
    # Rows
    for idx, row in df.iterrows():
        cells = [str(idx)] + [str(v) for v in row]
        lines.append(" & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
    ]
    if caption:
        lines.append(rf"\caption{{{caption}}}")
    if label:
        lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analysis result container
# ---------------------------------------------------------------------------

@dataclass
class ScenarioResult:
    """All analysis outputs for one disaster scenario."""

    label: str
    n_base_edges: int
    direct_mask: np.ndarray            # bool (n_edges,) — directly disrupted
    indirect_masks: Dict[str, np.ndarray]  # fac_key → bool (n_edges,)
    noaccess_mask: np.ndarray          # bool — direct OR any indirect
    road_df: pd.DataFrame
    facility_df: pd.DataFrame
    population_df: pd.DataFrame
    viz_path: Optional[Path] = None


# ---------------------------------------------------------------------------
# NetworkAnalyst
# ---------------------------------------------------------------------------

class NetworkAnalyst:
    """
    Loads a graph-tool network and performs multi-scenario disruption analysis.
    """

    def __init__(
        self,
        graph_path: Path,
        output_dir: Path,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if not GRAPH_TOOL_AVAILABLE:
            raise ImportError(
                "graph-tool is required for network analysis. "
                "Install via: conda install -c conda-forge graph-tool"
            )
        self.log = _get_logger(logger)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load base graph
        graph_path = Path(graph_path)
        self.log.info(f"Loading base graph: {graph_path}")
        self.g: Graph = gt.load_graph(str(graph_path))
        self.n_edges = self.g.num_edges()
        self.log.info(
            f"Graph: {self.g.num_vertices():,} vertices, {self.n_edges:,} edges"
        )

        # Pre-extract arrays for efficiency (O(n) one-time cost)
        self._src_v = self.g.get_edges()[:, 0]   # source vertex per edge
        self._disruption = _num_array(self.g, "disruption", 0.0).astype(int)
        self._highway_arr = _str_array(self.g, "highway")
        self._fac_amenity = _str_array(self.g, "fac_amenity")
        self._fac_layer = _str_array(self.g, "fac_layer_name")
        self._fac_name = _str_array(self.g, "fac_name")

        # Population
        self._year = _detect_pop_year(self.g)
        if self._year is None:
            self.log.warning(
                "No pop_total_* properties found; population stats will be zero."
            )
        self._pop = _pop_arrays(self.g, self._year) if self._year else {
            s: np.zeros(self.n_edges) for s, _, _ in POP_BANDS
        }

        # Highway classification (vectorised)
        self._hw_class = np.array(
            [_classify_highway(hw) for hw in self._highway_arr], dtype=object
        )

    # ------------------------------------------------------------------
    # Public runner
    # ------------------------------------------------------------------

    def run(
        self,
        disruption_vectors: List[np.ndarray],
        labels: List[str],
        compound: bool = True,
    ) -> List[ScenarioResult]:
        """
        Run disruption analysis for one or more disaster scenarios.
        """
        if not disruption_vectors:
            raise ValueError("At least one disruption vector is required.")

        results: List[ScenarioResult] = []
        cumulative = np.zeros(self.n_edges, dtype=int)

        for vec, label in zip(disruption_vectors, labels):
            if compound:
                cumulative = np.clip(cumulative + (vec == 1).astype(int), 0, 1)
                direct_mask = cumulative.astype(bool)
            else:
                direct_mask = (vec == 1)

            self.log.info(
                f"Scenario '{label}': {direct_mask.sum():,}/{self.n_edges:,} "
                f"edges directly disrupted."
            )
            result = self._analyse_scenario(direct_mask, label)
            self._export(result)
            results.append(result)

        return results

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def _analyse_scenario(
        self, direct_mask: np.ndarray, label: str
    ) -> ScenarioResult:
        """Full analysis pipeline for one disruption state."""
        # Build filtered graph view (non-disrupted edges only)
        efilt = self.g.new_edge_property("bool")
        efilt.a = ~direct_mask
        g_filt = GraphView(self.g, efilt=efilt)

        # Connected components on the filtered graph
        comp_labels, _ = label_components(g_filt)
        comp_arr = comp_labels.a            # shape (n_vertices,)
        edge_comp = comp_arr[self._src_v]   # component of each base-graph edge

        # Indirect loss per facility
        indirect_masks = self._compute_indirect_loss(
            direct_mask, edge_comp
        )

        # Any-facility indirect mask
        any_indirect = np.zeros(self.n_edges, dtype=bool)
        for mask in indirect_masks.values():
            any_indirect |= mask

        noaccess_mask = direct_mask | any_indirect

        # Statistics tables
        road_df      = self._road_stats(direct_mask, any_indirect)
        facility_df  = self._facility_stats(direct_mask)
        pop_df       = self._population_stats(direct_mask, indirect_masks, any_indirect)

        # Visualization
        viz_path = self._visualize(direct_mask, any_indirect, label)

        return ScenarioResult(
            label=label,
            n_base_edges=self.n_edges,
            direct_mask=direct_mask,
            indirect_masks=indirect_masks,
            noaccess_mask=noaccess_mask,
            road_df=road_df,
            facility_df=facility_df,
            population_df=pop_df,
            viz_path=viz_path,
        )

    def _compute_indirect_loss(
        self,
        direct_mask: np.ndarray,
        edge_comp: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        For each facility type, find non-disrupted edges whose component has
        no accessible (non-disrupted) edge of that facility type.

        Returns
        -------
        dict: facility_key → bool numpy array (n_edges,)
        """
        non_disrupted = ~direct_mask
        indirect_masks: Dict[str, np.ndarray] = {}

        for spec in FACILITIES:
            # Mask: edges that belong to this facility type (in the base graph)
            fac_mask = np.array(
                [
                    _has_facility_type(str(a), str(l), spec)
                    for a, l in zip(self._fac_amenity, self._fac_layer)
                ],
                dtype=bool,
            )

            if not fac_mask.any():
                # Facility type absent entirely — no one has access to lose
                indirect_masks[spec.key] = np.zeros(self.n_edges, dtype=bool)
                continue

            # Accessible facility edges (non-disrupted AND of this type)
            accessible_fac = fac_mask & non_disrupted
            comps_with_fac: Set[int] = set(edge_comp[accessible_fac].tolist())

            # Indirect loss: non-disrupted edge whose component lacks the facility
            in_fac_comp = np.isin(edge_comp, list(comps_with_fac))
            indirect = non_disrupted & ~in_fac_comp
            indirect_masks[spec.key] = indirect

        return indirect_masks

    # ------------------------------------------------------------------
    # Statistics tables
    # ------------------------------------------------------------------

    def _road_stats(
        self,
        direct_mask: np.ndarray,
        indirect_mask: np.ndarray,
    ) -> pd.DataFrame:
        """
        Road network table: counts and percentages by highway class.

        Columns: Base n, Direct blocked, Indirect blocked, No access
        Rows:    Each highway class + Other + TOTAL
        """
        all_classes = list(HIGHWAY_CLASSES.keys()) + ["Other"]
        rows = []

        for cls in all_classes:
            cls_mask = self._hw_class == cls
            n_base = cls_mask.sum()
            if n_base == 0:
                continue
            n_dir  = (cls_mask & direct_mask).sum()
            n_ind  = (cls_mask & indirect_mask & ~direct_mask).sum()
            n_noa  = (cls_mask & (direct_mask | indirect_mask)).sum()
            rows.append({
                "Highway class":    cls,
                "Base [n]":         n_base,
                "Direct [n (%)]":   _pct(n_dir, n_base),
                "Indirect [n (%)]": _pct(n_ind, n_base),
                "No access [n (%)]":_pct(n_noa, n_base),
            })

        # Total row
        n_base = self.n_edges
        n_dir  = direct_mask.sum()
        n_ind  = (indirect_mask & ~direct_mask).sum()
        n_noa  = (direct_mask | indirect_mask).sum()
        rows.append({
            "Highway class":    "TOTAL",
            "Base [n]":         n_base,
            "Direct [n (%)]":   _pct(n_dir, n_base),
            "Indirect [n (%)]": _pct(n_ind, n_base),
            "No access [n (%)]":_pct(n_noa, n_base),
        })

        df = pd.DataFrame(rows).set_index("Highway class")
        return df

    def _facility_stats(self, direct_mask: np.ndarray) -> pd.DataFrame:
        """
        Facility table: edge counts + unique facility counts.
        Multi-edge facilities assessed for shutdown vs hindrance.
        """
        rows = []

        for spec in FACILITIES:
            fac_mask = np.array(
                [
                    _has_facility_type(str(a), str(l), spec)
                    for a, l in zip(self._fac_amenity, self._fac_layer)
                ],
                dtype=bool,
            )
            n_edges_total = fac_mask.sum()
            if n_edges_total == 0:
                continue

            n_edges_direct = (fac_mask & direct_mask).sum()

            # Unique facilities (by name+layer key)
            all_keys    = _unique_facility_keys(
                self._fac_name, self._fac_amenity, self._fac_layer, fac_mask
            )
            direct_keys = _unique_facility_keys(
                self._fac_name, self._fac_amenity, self._fac_layer,
                fac_mask & direct_mask
            )
            n_unique_total   = len(all_keys)
            n_unique_affected = len(direct_keys)

            # Complete shutdown vs. partial hindrance per named facility
            complete_shutdown = 0
            hindered = 0
            for key in all_keys:
                # edges of this individual facility
                def _key_mask(i: int) -> bool:
                    nm   = str(self._fac_name[i]).split(";")[0].strip()
                    lyr  = str(self._fac_layer[i]).split(";")[0].strip()
                    am   = str(self._fac_amenity[i]).split(";")[0].strip()
                    fkey = f"{lyr}::{nm}" if nm else f"{lyr}::{am}"
                    return fkey == key and fac_mask[i]

                key_indices = [i for i in range(self.n_edges) if _key_mask(i)]
                if not key_indices:
                    continue
                key_arr = np.array(key_indices)
                n_key_total   = len(key_arr)
                n_key_direct  = direct_mask[key_arr].sum()
                frac = n_key_direct / n_key_total
                if frac == 1.0:
                    complete_shutdown += 1
                elif frac > 0:
                    hindered += 1

            rows.append({
                "Facility":               spec.display,
                "Edges (base)":           n_edges_total,
                "Edges disrupted [n (%)]":_pct(n_edges_direct, n_edges_total),
                "Unique facilities":      n_unique_total,
                "Facilities affected":    n_unique_affected,
                "Complete shutdown":      complete_shutdown,
                "Partially hindered":     hindered,
            })

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows).set_index("Facility")

    def _population_stats(
        self,
        direct_mask: np.ndarray,
        indirect_masks: Dict[str, np.ndarray],
        any_indirect: np.ndarray,
    ) -> pd.DataFrame:
        """
        Population impact table disaggregated by demographic band.
        """
        def _psum(mask: np.ndarray, band: str) -> float:
            return float(self._pop[band][mask].sum())

        # Services to show as individual columns: all present facilities
        # except the "any_crit" aggregate (that is the "Any service" columns).
        service_specs = []
        for spec in FACILITIES:
            if spec.key == "any_crit":
                continue
            present = np.array(
                [
                    _has_facility_type(str(a), str(l), spec)
                    for a, l in zip(self._fac_amenity, self._fac_layer)
                ],
                dtype=bool,
            ).any()
            if present:
                service_specs.append(spec)

        rows = []
        for suffix, disp, _ in POP_BANDS:
            net_pop = float(self._pop[suffix].sum())
            direct  = _psum(direct_mask, suffix)
            any_ind = _psum(any_indirect & ~direct_mask, suffix)
            any_tot = direct + any_ind

            row: Dict = {
                "Demographic":            disp,
                "Network population":     round(net_pop, 1),
                "Direct loss":            round(direct,  1),
                "Any service — indirect": round(any_ind, 1),
                "Any service — total":    round(any_tot, 1),
            }
            for spec in service_specs:
                imask = indirect_masks.get(spec.key)
                svc_total = (
                    _psum(direct_mask | imask, suffix)
                    if imask is not None
                    else direct
                )
                row[spec.display] = round(svc_total, 1)

            rows.append(row)

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows).set_index("Demographic")

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def _visualize(
        self,
        direct_mask: np.ndarray,
        indirect_any_mask: np.ndarray,
        label: str,
    ) -> Optional[Path]:
        """
        Generate a geographic edge map with three color categories:
          - Gray   : unaffected edges
          - Orange : indirectly blocked (isolated component, no facility)
          - Red    : directly disrupted by the disaster

        Returns the path to the saved PNG.
        """
        if "geometry" not in self.g.ep:
            self.log.warning("No 'geometry' edge property — cannot create map.")
            return None

        normal_segs:   list = []
        indirect_segs: list = []
        direct_segs:   list = []

        for i, e in enumerate(self.g.edges()):
            flat = self.g.ep["geometry"][e]
            if len(flat) < 4:
                continue
            coords = np.array(flat, dtype="float64").reshape(-1, 2)
            seg = coords.tolist()
            if direct_mask[i]:
                direct_segs.append(seg)
            elif indirect_any_mask[i]:
                indirect_segs.append(seg)
            else:
                normal_segs.append(seg)

        fig, ax = plt.subplots(figsize=(14, 14), dpi=150)
        fig.patch.set_facecolor("white")

        if normal_segs:
            ax.add_collection(
                LineCollection(normal_segs,
                               colors=COLORS["normal"], linewidths=0.5,
                               zorder=1, label="Unaffected")
            )
        if indirect_segs:
            ax.add_collection(
                LineCollection(indirect_segs,
                               colors=COLORS["indirect"], linewidths=1.2,
                               zorder=2, label="Indirect access loss")
            )
        if direct_segs:
            ax.add_collection(
                LineCollection(direct_segs,
                               colors=COLORS["direct"], linewidths=1.8,
                               zorder=3, label="Directly disrupted")
            )

        ax.autoscale_view()
        ax.set_aspect("equal")
        ax.axis("off")

        legend_handles = [
            mlines.Line2D([], [], color=COLORS["direct"],   lw=2, label="Directly disrupted"),
            mlines.Line2D([], [], color=COLORS["indirect"], lw=2, label="Indirect access loss"),
            mlines.Line2D([], [], color=COLORS["normal"],   lw=1.5, label="Unaffected"),
        ]
        ax.legend(handles=legend_handles, loc="upper right", fontsize=10,
                  frameon=True, framealpha=0.9)

        n_dir = direct_mask.sum()
        n_ind = (indirect_any_mask & ~direct_mask).sum()
        title = (
            f"Disaster impact — {label}\n"
            f"Directly disrupted: {n_dir:,}  |  "
            f"Indirect access loss: {n_ind:,}  |  "
            f"Total: {n_dir + n_ind:,} / {self.n_edges:,} edges"
        )
        ax.set_title(title, fontsize=12, pad=12)

        viz_dir = self.output_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        slug = label.replace(" ", "_").replace("/", "-")
        path = viz_dir / f"impact_{slug}.png"
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        self.log.info(f"Visualization saved: {path}")
        return path

    # ------------------------------------------------------------------
    # Export (CSV + LaTeX)
    # ------------------------------------------------------------------

    def _export(self, result: ScenarioResult) -> None:
        """Write CSV and LaTeX files for all three stats tables."""
        slug = result.label.replace(" ", "_").replace("/", "-")
        stats_dir = self.output_dir / "stats"
        stats_dir.mkdir(parents=True, exist_ok=True)

        tables = [
            (result.road_df,      f"{slug}_roads",      "Road network impact by highway class"),
            (result.facility_df,  f"{slug}_facilities", "Facility disruption analysis"),
            (result.population_df,f"{slug}_population", "Population impact by demographic group"),
        ]

        for df, stem, caption in tables:
            if df is None or df.empty:
                continue
            csv_path = stats_dir / f"{stem}.csv"
            tex_path = stats_dir / f"{stem}.tex"
            df.to_csv(csv_path)
            with open(tex_path, "w") as fh:
                fh.write(_to_latex(df, caption=caption, label=f"tab:{stem}"))
            self.log.info(f"Exported: {csv_path.name}, {tex_path.name}")

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------

    @staticmethod
    def print_summary(result: ScenarioResult) -> None:
        """Print a full console summary for one scenario."""
        n     = result.n_base_edges
        n_dir = int(result.direct_mask.sum())
        n_ind = int((result.noaccess_mask & ~result.direct_mask).sum())
        n_noa = int(result.noaccess_mask.sum())

        def _pct(num: float, base: float) -> str:
            return "—" if base == 0 else f"{100 * num / base:.1f}%"

        print()
        print(f"  ─── {result.label} ───")
        print(f"  Base edges            : {n:>10,}")
        print(f"  Directly disrupted    : {n_dir:>10,}  ({_pct(n_dir, n)})")
        print(f"  Indirect access loss  : {n_ind:>10,}  ({_pct(n_ind, n)})")
        print(f"  Without any access    : {n_noa:>10,}  ({_pct(n_noa, n)})")

        pop_df = result.population_df
        if pop_df is None or pop_df.empty or "Total" not in pop_df.index:
            if result.viz_path:
                print(f"  Visualization         : {result.viz_path}")
            print()
            return

        try:
            tr      = pop_df.loc["Total"]
            if hasattr(tr, "iloc") and tr.ndim == 2:
                tr = tr.iloc[0]
            net_pop = float(tr["Network population"])
            direct  = float(tr["Direct loss"])
            any_ind = float(tr["Any service — indirect"])
            any_tot = float(tr["Any service — total"])

            # ── Population overview ───────────────────────────────────────
            print()
            print(f"  ── Population ─────────────────────────────────────────────")
            print(f"  Network population    : {net_pop:>10,.0f}")
            print(f"  Direct loss           : {direct:>10,.0f}  ({_pct(direct,  net_pop)})")
            print(f"  Indirect loss         : {any_ind:>10,.0f}  ({_pct(any_ind, net_pop)})")
            print(f"  Any service — total   : {any_tot:>10,.0f}  ({_pct(any_tot, net_pop)})")

            # ── Breakdown by demographic band ─────────────────────────────
            band_rows = [
                (disp, suffix)
                for suffix, disp, _ in POP_BANDS
                if disp != "Total" and disp in pop_df.index
            ]
            if band_rows:
                print()
                print(
                    f"  {'Demographic':<18}  {'Network pop':>11}  "
                    f"{'Direct':>9}  {'Indirect':>9}  "
                    f"{'Any svc total':>13}  {'% of total':>10}"
                )
                print(
                    f"  {'─'*18}  {'─'*11}  {'─'*9}  {'─'*9}  {'─'*13}  {'─'*10}"
                )
                for disp, _ in band_rows:
                    row = pop_df.loc[disp]
                    if hasattr(row, "iloc") and row.ndim == 2:
                        row = row.iloc[0]
                    bn   = float(row["Network population"])
                    bd   = float(row["Direct loss"])
                    bi   = float(row["Any service — indirect"])
                    bt   = float(row["Any service — total"])
                    print(
                        f"  {disp:<18}  {bn:>11,.0f}  "
                        f"{bd:>9,.0f}  {bi:>9,.0f}  "
                        f"{bt:>13,.0f}  {_pct(bt, net_pop):>10}"
                    )

            # ── Per-service losses (Total row only) ───────────────────────
            svc_cols = [
                c for c in pop_df.columns
                if c not in (
                    "Network population", "Direct loss",
                    "Any service — indirect", "Any service — total",
                )
            ]
            if svc_cols:
                print()
                print(
                    f"  {'Service':<25}  {'Total loss':>10}  "
                    f"{'Indirect only':>13}  {'% of total':>10}"
                )
                print(f"  {'─'*25}  {'─'*10}  {'─'*13}  {'─'*10}")
                for col in svc_cols:
                    try:
                        svc_tot = float(tr[col])
                        svc_ind = svc_tot - direct   # indirect component
                        if svc_tot > 0:
                            print(
                                f"  {col:<25}  {svc_tot:>10,.0f}  "
                                f"{max(0, svc_ind):>13,.0f}  {_pct(svc_tot, net_pop):>10}"
                            )
                    except (KeyError, TypeError, ValueError):
                        pass

        except (KeyError, TypeError, AttributeError):
            pass

        if result.viz_path:
            print(f"  Visualization         : {result.viz_path}")
        print()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyse_network(
    graph_path: Path,
    output_dir: Path,
    additional_graph_paths: Optional[List[Path]] = None,
    scenario_labels: Optional[List[str]] = None,
    compound: bool = True,
    logger: Optional[logging.Logger] = None,
) -> List[ScenarioResult]:
    """
    Perform network disruption analysis and generate all statistical outputs.
    """
    log = _get_logger(logger)

    graph_path = Path(graph_path)
    analyst = NetworkAnalyst(graph_path, output_dir, logger=log)

    # Collect disruption vectors: base graph first
    all_paths = [graph_path] + [Path(p) for p in (additional_graph_paths or [])]

    disruption_vectors: List[np.ndarray] = []
    valid_labels: List[str] = []

    for idx, p in enumerate(all_paths):
        default_label = f"Crisis {idx + 1}"
        label = (scenario_labels or [])[idx] if (
            scenario_labels and idx < len(scenario_labels)
        ) else default_label

        if idx == 0:
            vec = analyst._disruption.copy()
        else:
            if not p.exists():
                log.warning(f"Additional graph not found: {p}; skipping.")
                continue
            try:
                g_extra = gt.load_graph(str(p))
            except Exception as exc:
                log.warning(f"Could not load {p}: {exc}; skipping.")
                continue
            if g_extra.num_edges() != analyst.n_edges:
                log.warning(
                    f"{p.name} has {g_extra.num_edges()} edges vs "
                    f"{analyst.n_edges} in base graph; skipping (edge mismatch)."
                )
                continue
            vec = _num_array(g_extra, "disruption", 0.0).astype(int)

        disruption_vectors.append(vec)
        valid_labels.append(label)

    if not disruption_vectors:
        raise RuntimeError(
            "No valid disruption vector could be extracted from the provided graph(s)."
        )

    log.info("=" * 70)
    log.info("ANTIPOMPEII Network Analysis")
    log.info("=" * 70)

    results = analyst.run(disruption_vectors, valid_labels, compound=compound)

    # Console summary
    print("\n" + "=" * 56)
    print("NETWORK ANALYSIS SUMMARY")
    print("=" * 56)
    for r in results:
        NetworkAnalyst.print_summary(r)
    print("=" * 56 + "\n")

    log.info("=" * 70)
    log.info(
        f"Analysis complete: {len(results)} scenario(s). "
        f"Outputs in: {output_dir}"
    )
    log.info("=" * 70)

    return results
