"""
Cross-graph comparative reporting for ANTIPOMPEII Mode 5.

Builds side-by-side CSV/LaTeX tables and PNG charts comparing analytics
outputs (stats, robustness, percolation, vulnerability) across multiple
graphs.  Used exclusively by the existing-graph workflow; lives in its own
module so :mod:`cli` doesn't have to carry ~1,200 lines of plotting and
table-stacking code.
"""
from __future__ import annotations

import logging
import math
import re as _re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.antipompeii.ui.layout import typing, print_section


# ---------------------------------------------------------------------------
# Helpers used by the visualizations
# ---------------------------------------------------------------------------

def graph_palette(n: int) -> list:
    """Return *n* perceptually distinct colors for comparison plots."""
    import matplotlib.pyplot as plt
    if n <= 10:
        return list(plt.cm.tab10.colors[:n])
    return [plt.cm.tab20(i / n) for i in range(n)]


def parse_n_pct(cell) -> Tuple[float, float]:
    """
    Parse a cell that is either a plain number or ``'n (pct%)'`` text.
    Returns ``(count, fraction)``; NaN on parse failure.
    """
    if cell is None:
        return (math.nan, math.nan)
    s = str(cell).strip()
    m = _re.match(r"^([\d,]+)\s*\(([0-9.]+)%\)", s)
    if m:
        return (float(m.group(1).replace(",", "")), float(m.group(2)) / 100.0)
    try:
        return (float(s.replace(",", "")), math.nan)
    except ValueError:
        return (math.nan, math.nan)


# ---------------------------------------------------------------------------
# ComparativeReport
# ---------------------------------------------------------------------------

class ComparativeReport:
    """
    Renders comparative tables, charts, and a master dashboard for a set of
    graphs that were each analyzed in the same per-graph output layout.

    Single public entry point: :meth:`render`.  Everything else is an
    implementation detail that happens to be a method for shared logger
    access and inter-method calls.
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(__name__)

    # ── public entry point ────────────────────────────────────────────────

    def render(
        self,
        graph_results: Dict[str, Dict],
        modules: List[str],
        comparison_root: Path,
    ) -> None:
        """
        Build side-by-side comparative tables (CSV + LaTeX) for each analytics
        module across all graphs, plus visualizations and a console overview.
        """
        labels = list(graph_results.keys())
        typing("\nBuilding comparative tables and visualizations...\n")

        if "stats" in modules:
            self._compare_stats(graph_results, labels, comparison_root)
            self._compare_stats_viz(graph_results, labels, comparison_root)
        if "robustness" in modules:
            self._compare_robustness(graph_results, labels, comparison_root)
            self._compare_robustness_viz(graph_results, labels, comparison_root)
        if "percolation" in modules:
            self._compare_percolation(graph_results, labels, comparison_root)
            self._compare_percolation_viz(graph_results, labels, comparison_root)
        if "vulnerability" in modules:
            self._compare_vulnerability(graph_results, labels, comparison_root)
            self._compare_vulnerability_viz(graph_results, labels, comparison_root)

        self._dashboard(graph_results, labels, modules, comparison_root)

        print_section("Comparative Overview")
        self._overview(graph_results, labels, modules)
        typing(
            f"\n✓ Comparative outputs saved to: {comparison_root / 'comparison'}\n"
        )

    # ── table / LaTeX core ────────────────────────────────────────────────

    @staticmethod
    def _df_to_latex(
        df: "pd.DataFrame",
        caption: str = "",
        label: str = "",
        float_fmt: str = "{:.4f}",
    ) -> str:
        """Render a DataFrame as a publication-ready LaTeX booktabs table."""
        def _esc(s: str) -> str:
            return (
                str(s)
                .replace("&", r"\&")
                .replace("%", r"\%")
                .replace("_", r"\_")
                .replace("#", r"\#")
                .replace("$", r"\$")
                .replace("{", r"\{")
                .replace("}", r"\}")
                .replace("~", r"\textasciitilde{}")
                .replace("^", r"\textasciicircum{}")
            )

        ncols = len(df.columns)
        col_fmt = "l" + "r" * ncols
        lines = [
            r"\begin{table}[htbp]",
            r"  \centering",
            rf"  \caption{{{_esc(caption)}}}",
            rf"  \label{{{label}}}",
            rf"  \begin{{tabular}}{{{col_fmt}}}",
            r"    \toprule",
        ]
        header = [""] + [_esc(str(c)) for c in df.columns]
        lines.append("    " + " & ".join(header) + r" \\")
        lines.append(r"    \midrule")
        for idx, row in df.iterrows():
            cells = [_esc(str(idx))]
            for v in row:
                if v is None:
                    cells.append("—")
                elif isinstance(v, float):
                    cells.append("—" if math.isnan(v) else float_fmt.format(v))
                else:
                    cells.append(_esc(str(v)))
            lines.append("    " + " & ".join(cells) + r" \\")
        lines += [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
        ]
        return "\n".join(lines) + "\n"

    def _save(
        self,
        df: "pd.DataFrame",
        comparison_root: Path,
        stem: str,
        caption: str,
    ) -> None:
        """Save a comparative DataFrame as both CSV and LaTeX."""
        comp_dir = comparison_root / "comparison"
        comp_dir.mkdir(parents=True, exist_ok=True)
        csv_path = comp_dir / f"{stem}.csv"
        tex_path = comp_dir / f"{stem}.tex"
        df.to_csv(csv_path)
        with open(tex_path, "w", encoding="utf-8") as fh:
            fh.write(self._df_to_latex(df, caption=caption, label=f"tab:{stem}"))
        typing(f"  ✓ {stem}:  {csv_path.name}  +  {tex_path.name}\n")
        self.logger.info(f"Comparative table: {csv_path}")

    # ── per-module table builders ─────────────────────────────────────────

    def _compare_stats(
        self,
        graph_results: Dict[str, Dict],
        labels: List[str],
        comparison_root: Path,
    ) -> None:
        """Build comparative road-disruption and population tables."""
        import pandas as pd

        # ── Roads: stack all graphs ──────────────────────────────────────
        road_frames: List[pd.DataFrame] = []
        for label in labels:
            stats_dir = graph_results[label]["output_dir"] / "stats"
            for csv in sorted(stats_dir.glob("*_roads.csv")):
                try:
                    df = pd.read_csv(csv, index_col=0)
                    df.insert(0, "Graph", label)
                    road_frames.append(df)
                    break
                except Exception as e:
                    self.logger.warning(f"Could not read {csv}: {e}")

        if road_frames:
            stacked = pd.concat(road_frames, axis=0)
            self._save(
                stacked,
                comparison_root,
                "comparison_roads",
                "Road-network disruption comparison by highway class",
            )
            # Wide pivot: highway class × (graph × metric)
            num_cols = [
                c for c in road_frames[0].columns
                if c != "Graph" and pd.api.types.is_numeric_dtype(road_frames[0][c])
            ]
            pivots: Dict[str, pd.DataFrame] = {}
            for col in num_cols[:8]:
                try:
                    piv = stacked.pivot_table(
                        index=stacked.index,
                        columns="Graph",
                        values=col,
                        aggfunc="first",
                    )
                    pivots[col] = piv
                except Exception:
                    pass
            if pivots:
                wide = pd.concat(pivots, axis=1)
                self._save(
                    wide,
                    comparison_root,
                    "comparison_roads_wide",
                    "Road disruption pivot: highway class × metric × graph",
                )

        # ── Population total row ─────────────────────────────────────────
        pop_rows: Dict[str, Dict] = {}
        for label in labels:
            stats_dir = graph_results[label]["output_dir"] / "stats"
            for csv in sorted(stats_dir.glob("*_population.csv")):
                try:
                    df = pd.read_csv(csv, index_col=0)
                    if "Total" in df.index:
                        pop_rows[label] = df.loc["Total"].to_dict()
                    break
                except Exception as e:
                    self.logger.warning(f"Could not read {csv}: {e}")

        if pop_rows:
            pop_df = pd.DataFrame(pop_rows)
            pop_df.index.name = "Metric"
            self._save(
                pop_df,
                comparison_root,
                "comparison_population",
                "Population impact comparison — Total row across graphs",
            )

        # ── Full population table stacked (all demographic bands) ────────
        pop_all: List[pd.DataFrame] = []
        for label in labels:
            stats_dir = graph_results[label]["output_dir"] / "stats"
            for csv in sorted(stats_dir.glob("*_population.csv")):
                try:
                    df = pd.read_csv(csv, index_col=0)
                    df.insert(0, "Graph", label)
                    pop_all.append(df)
                    break
                except Exception as e:
                    self.logger.warning(f"Could not read {csv}: {e}")

        if pop_all:
            self._save(
                pd.concat(pop_all, axis=0),
                comparison_root,
                "comparison_population_full",
                "Population impact by demographic band — all graphs",
            )

    def _compare_robustness(
        self,
        graph_results: Dict[str, Dict],
        labels: List[str],
        comparison_root: Path,
    ) -> None:
        """Build comparative robustness metrics tables."""
        import pandas as pd

        frames: List[pd.DataFrame] = []
        for label in labels:
            csv = graph_results[label]["output_dir"] / "robustness" / "robustness.csv"
            if csv.exists():
                try:
                    df = pd.read_csv(csv, index_col=0)
                    df.insert(0, "Graph", label)
                    frames.append(df)
                except Exception as e:
                    self.logger.warning(f"Could not read {csv}: {e}")

        if not frames:
            return

        stacked = pd.concat(frames, axis=0)
        self._save(
            stacked,
            comparison_root,
            "comparison_robustness",
            "Robustness metrics — all states and graphs",
        )

        metric_cols = [c for c in frames[0].columns if c != "Graph"]
        pivots: Dict[str, pd.DataFrame] = {}
        for col in metric_cols:
            try:
                piv = stacked.pivot_table(
                    index=stacked.index,
                    columns="Graph",
                    values=col,
                    aggfunc="first",
                )
                pivots[col] = piv
            except Exception:
                pass
        if pivots:
            wide = pd.concat(pivots, axis=1)
            self._save(
                wide,
                comparison_root,
                "comparison_robustness_wide",
                "Robustness comparison: network states × metrics × graphs",
            )

        # Delta table: change from intact state per graph
        delta_rows: Dict[str, Dict] = {}
        for label in labels:
            sub = stacked[stacked["Graph"] == label].drop(columns=["Graph"])
            num_sub = sub.select_dtypes("number")
            if len(num_sub) >= 2:
                intact = num_sub.iloc[0]
                for state, row in num_sub.iloc[1:].iterrows():
                    key = f"{label} → {state}"
                    delta_rows[key] = (row - intact).to_dict()
        if delta_rows:
            delta_df = pd.DataFrame(delta_rows).T
            delta_df.index.name = "Transition"
            self._save(
                delta_df,
                comparison_root,
                "comparison_robustness_delta",
                "Robustness Δ from intact state",
            )

    def _compare_percolation(
        self,
        graph_results: Dict[str, Dict],
        labels: List[str],
        comparison_root: Path,
    ) -> None:
        """Build comparative percolation threshold tables."""
        import pandas as pd

        all_summaries: Dict[str, pd.DataFrame] = {}
        for label in labels:
            perc_dir = graph_results[label]["output_dir"] / "percolation"
            if not perc_dir.exists():
                continue
            for csv in sorted(perc_dir.glob("*_summary.csv")):
                scenario = csv.stem.replace("_summary", "")
                try:
                    df = pd.read_csv(csv, index_col=0)
                    all_summaries[f"{label} / {scenario}"] = df
                except Exception as e:
                    self.logger.warning(f"Could not read {csv}: {e}")

        if not all_summaries:
            return

        frames_list = []
        for key, df in all_summaries.items():
            d = df.copy()
            d.insert(0, "Scenario", key)
            frames_list.append(d)

        stacked = pd.concat(frames_list, axis=0)
        self._save(
            stacked,
            comparison_root,
            "comparison_percolation",
            "Percolation threshold comparison — all scenarios and graphs",
        )

        if "T_50" in stacked.columns:
            try:
                t50 = stacked.pivot_table(
                    index=stacked.index,
                    columns="Scenario",
                    values="T_50",
                    aggfunc="first",
                )
                self._save(
                    t50,
                    comparison_root,
                    "comparison_percolation_T50",
                    "Percolation T\\textsubscript{50} critical threshold: "
                    "facility × scenario/graph",
                )
            except Exception as e:
                self.logger.warning(f"Could not build T_50 pivot: {e}")

    def _compare_vulnerability(
        self,
        graph_results: Dict[str, Dict],
        labels: List[str],
        comparison_root: Path,
    ) -> None:
        """Build comparative vulnerability indices table."""
        import pandas as pd

        index_dicts: Dict[str, Dict] = {}
        for label in labels:
            csv = (
                graph_results[label]["output_dir"]
                / "vulnerability"
                / "global_indices.csv"
            )
            if csv.exists():
                try:
                    df = pd.read_csv(csv)
                    index_dicts[label] = dict(zip(df["index"], df["value"]))
                except Exception as e:
                    self.logger.warning(f"Could not read {csv}: {e}")

        if not index_dicts:
            return

        vuln_df = pd.DataFrame(index_dicts)
        vuln_df.index.name = "Index"
        self._save(
            vuln_df,
            comparison_root,
            "comparison_vulnerability",
            "Global vulnerability indices comparison",
        )

    # ── per-module visualizations ─────────────────────────────────────────

    def _compare_stats_viz(
        self,
        graph_results: Dict[str, Dict],
        labels: List[str],
        comparison_root: Path,
    ) -> None:
        """Bar charts for edge disruption, population impact, demographics."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        import numpy as np
        import pandas as pd

        comp_dir = comparison_root / "comparison"
        comp_dir.mkdir(parents=True, exist_ok=True)
        palette = graph_palette(len(labels))

        # ── 1. Edge disruption fractions ─────────────────────────────────
        direct_fracs: List[float] = []
        indirect_fracs: List[float] = []
        edge_labels: List[str] = []

        for label in labels:
            stats_dir = graph_results[label]["output_dir"] / "stats"
            for csv in sorted(stats_dir.glob("*_roads.csv")):
                try:
                    df = pd.read_csv(csv, index_col=0)
                    if "TOTAL" not in df.index:
                        break
                    row = df.loc["TOTAL"]
                    _, d_frac = parse_n_pct(row.get("Direct [n (%)]"))
                    _, n_frac = parse_n_pct(row.get("No access [n (%)]"))
                    if not math.isnan(d_frac):
                        d = d_frac
                        na = n_frac if not math.isnan(n_frac) else d_frac
                        direct_fracs.append(d)
                        indirect_fracs.append(max(0.0, na - d))
                        edge_labels.append(label)
                except Exception as e:
                    self.logger.warning(f"Stats viz [edges] {label}: {e}")
                break

        if edge_labels:
            fig, ax = plt.subplots(figsize=(max(6, len(edge_labels) * 1.6), 5))
            x = np.arange(len(edge_labels))
            w = 0.5
            ax.bar(x, direct_fracs, w, color="#c0392b",
                   alpha=0.85, label="Direct disruption")
            ax.bar(x, indirect_fracs, w, bottom=direct_fracs,
                   color="#e67e22", alpha=0.85,
                   label="Indirect (network isolation)")
            ax.set_xticks(x)
            ax.set_xticklabels(edge_labels, rotation=30, ha="right")
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
            ax.set_ylabel("Share of network edges")
            ax.set_title("Network Edge Disruption Comparison", fontweight="bold")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            fig.tight_layout()
            path = comp_dir / "comparison_disruption.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            typing(f"  ✓ Edge disruption chart  → {path.name}\n")
            self.logger.info(f"Saved: {path}")

        # ── 2. Population impact ──────────────────────────────────────────
        net_pops: List[float] = []
        dir_losses: List[float] = []
        any_totals: List[float] = []
        pop_labels: List[str] = []

        for label in labels:
            stats_dir = graph_results[label]["output_dir"] / "stats"
            for csv in sorted(stats_dir.glob("*_population.csv")):
                try:
                    df = pd.read_csv(csv, index_col=0)
                    if "Total" not in df.index:
                        break
                    row = df.loc["Total"]
                    net = float(row.get("Network population", 0) or 0)
                    dl  = float(row.get("Direct loss", 0) or 0)
                    at  = float(row.get("Any service — total", 0) or 0)
                    net_pops.append(net)
                    dir_losses.append(dl)
                    any_totals.append(at)
                    pop_labels.append(label)
                except Exception as e:
                    self.logger.warning(f"Stats viz [pop] {label}: {e}")
                break

        if pop_labels and any(p > 0 for p in net_pops):
            fig, ax = plt.subplots(figsize=(max(6, len(pop_labels) * 1.6), 5))
            x = np.arange(len(pop_labels))
            w = 0.25
            ax.bar(x - w, net_pops,   w, color="#bdc3c7",
                   alpha=0.8,  label="Network population")
            ax.bar(x,     dir_losses, w, color="#8e44ad",
                   alpha=0.85, label="Direct loss")
            ax.bar(x + w, any_totals, w, color="#2980b9",
                   alpha=0.85, label="Any-service total loss")
            ax.set_xticks(x)
            ax.set_xticklabels(pop_labels, rotation=30, ha="right")
            ax.set_ylabel("Population (persons)")
            ax.set_title("Population Impact Comparison", fontweight="bold")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            fig.tight_layout()
            path = comp_dir / "comparison_population.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            typing(f"  ✓ Population impact chart → {path.name}\n")
            self.logger.info(f"Saved: {path}")

        # ── 3. Demographic breakdown ──────────────────────────────────────
        demo_bands = ["Total", "Female 0–14", "Female 15–64", "Female 65+",
                      "Male 0–14", "Male 15–64", "Male 65+"]
        demo_data: Dict[str, Dict[str, float]] = {}
        for label in labels:
            stats_dir = graph_results[label]["output_dir"] / "stats"
            for csv in sorted(stats_dir.glob("*_population.csv")):
                try:
                    df = pd.read_csv(csv, index_col=0)
                    col = "Any service — total"
                    if col in df.columns:
                        demo_data[label] = {
                            band: float(df.loc[band, col])
                            for band in demo_bands
                            if band in df.index
                        }
                except Exception:
                    pass
                break

        if len(demo_data) >= 2:
            bands_present = [b for b in demo_bands[1:]
                             if any(b in d for d in demo_data.values())]
            if bands_present:
                fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.8), 5))
                x = np.arange(len(labels))
                w_total = 0.7
                w = w_total / len(bands_present)
                band_colors = plt.cm.Paired(
                    [i / len(bands_present) for i in range(len(bands_present))]
                )
                for bi, (band, bc) in enumerate(zip(bands_present, band_colors)):
                    offset = (bi - len(bands_present) / 2 + 0.5) * w
                    vals = [demo_data.get(lbl, {}).get(band, 0.0) for lbl in labels]
                    ax.bar(x + offset, vals, w, color=bc, alpha=0.85, label=band)
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=30, ha="right")
                ax.set_ylabel("Population with any-service loss (persons)")
                ax.set_title("Demographic Impact Comparison", fontweight="bold")
                ax.legend(fontsize="x-small", ncol=2)
                ax.grid(axis="y", alpha=0.3)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                fig.tight_layout()
                path = comp_dir / "comparison_demographics.png"
                fig.savefig(path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                typing(f"  ✓ Demographic breakdown  → {path.name}\n")
                self.logger.info(f"Saved: {path}")

    def _compare_robustness_viz(
        self,
        graph_results: Dict[str, Dict],
        labels: List[str],
        comparison_root: Path,
    ) -> None:
        """Multi-panel robustness profile: one subplot per metric."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        comp_dir = comparison_root / "comparison"
        comp_dir.mkdir(parents=True, exist_ok=True)
        palette = graph_palette(len(labels))

        data: Dict[str, pd.DataFrame] = {}
        for label in labels:
            csv = graph_results[label]["output_dir"] / "robustness" / "robustness.csv"
            if csv.exists():
                try:
                    data[label] = pd.read_csv(csv, index_col=0)
                except Exception as e:
                    self.logger.warning(f"Robustness viz {label}: {e}")

        if not data:
            return

        plot_metrics = [
            ("n_components",    "Connected components (κ)"),
            ("betweenness_max", "Max edge betweenness (b_max)"),
            ("betweenness_avg", "Avg edge betweenness (b̄)"),
            ("eff_resistance",  "Effective resistance (R*)"),
        ]
        avail = [(k, t) for k, t in plot_metrics
                 if any(k in df.columns for df in data.values())]
        if not avail:
            return

        ncols = min(2, len(avail))
        nrows = math.ceil(len(avail) / ncols)
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(6 * ncols, 4 * nrows),
            squeeze=False,
        )

        state_names: List[str] = []
        for idx, (key, title) in enumerate(avail):
            ax = axes[idx // ncols][idx % ncols]
            for (label, df), color in zip(data.items(), palette):
                if key not in df.columns:
                    continue
                vals = pd.to_numeric(df[key], errors="coerce").values
                x = np.arange(len(vals))
                ax.plot(x, vals, marker="o", color=color, label=label,
                        linewidth=2, markersize=6)
                if idx == 0:
                    state_names = list(df.index)
            if idx == 0 and state_names:
                ax.set_xticks(np.arange(len(state_names)))
                ax.set_xticklabels(state_names, rotation=20, ha="right",
                                   fontsize="small")
            else:
                ax.set_xlabel("Network state")
            ax.set_title(title)
            ax.grid(alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        for idx in range(len(avail), nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        handles, lbl_names = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, lbl_names, loc="lower center",
                   ncol=min(len(labels), 5), bbox_to_anchor=(0.5, -0.02),
                   fontsize="small")
        fig.suptitle("Network Robustness Comparison", fontsize=13,
                     fontweight="bold")
        fig.tight_layout(rect=[0, 0.06, 1, 1])
        path = comp_dir / "comparison_robustness.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        typing(f"  ✓ Robustness profile      → {path.name}\n")
        self.logger.info(f"Saved: {path}")

    def _compare_percolation_viz(
        self,
        graph_results: Dict[str, Dict],
        labels: List[str],
        comparison_root: Path,
    ) -> None:
        """Overlay percolation trajectories and render a T_50 heatmap."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        comp_dir = comparison_root / "comparison"
        comp_dir.mkdir(parents=True, exist_ok=True)
        palette = graph_palette(len(labels))

        scenarios: Dict[str, Dict[str, pd.DataFrame]] = {}
        for label in labels:
            perc_dir = graph_results[label]["output_dir"] / "percolation"
            if not perc_dir.exists():
                continue
            for csv in sorted(perc_dir.glob("*_trajectory.csv")):
                scenario = csv.stem.replace("_trajectory", "")
                try:
                    df = pd.read_csv(csv)
                    scenarios.setdefault(scenario, {})[label] = df
                except Exception as e:
                    self.logger.warning(f"Percolation viz {label}/{scenario}: {e}")

        if not scenarios:
            return

        scenario_titles = {
            "betweenness": "Betweenness Attack",
            "random":      "Random Failure",
            "elevation":   "Elevation Removal",
        }

        n_scen = len(scenarios)
        fig, axes = plt.subplots(
            1, n_scen,
            figsize=(7 * n_scen, 5),
            squeeze=False,
        )

        meta_cols = {"step", "fraction_removed", "pop_direct"}

        for s_idx, (scenario, label_dfs) in enumerate(sorted(scenarios.items())):
            ax = axes[0][s_idx]
            for (label, df), color in zip(label_dfs.items(), palette):
                if "fraction_removed" not in df.columns:
                    continue
                x = df["fraction_removed"].values
                svc_cols = [c for c in df.columns if c not in meta_cols]
                if "any_crit" in svc_cols:
                    y = df["any_crit"].values
                elif svc_cols:
                    y = df[svc_cols].mean(axis=1).values
                else:
                    continue
                ax.plot(x, y, color=color, label=label, linewidth=2, alpha=0.9)
                if "pop_direct" in df.columns:
                    d = df["pop_direct"].values
                    ax.fill_between(x, 0, d, color=color, alpha=0.10)

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Fraction of edges removed")
            ax.set_ylabel("Population penalty\n(direct + indirect)")
            ax.set_title(scenario_titles.get(scenario, scenario.title()),
                         fontweight="bold")
            ax.grid(alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        handles, lbl_names = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, lbl_names, loc="lower center",
                   ncol=min(len(labels), 5), bbox_to_anchor=(0.5, -0.04),
                   fontsize="small")
        fig.suptitle("Percolation Curves Comparison", fontsize=13,
                     fontweight="bold")
        fig.tight_layout(rect=[0, 0.06, 1, 1])
        path = comp_dir / "comparison_percolation.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        typing(f"  ✓ Percolation overlay     → {path.name}\n")
        self.logger.info(f"Saved: {path}")

        # ── Per-service T_50 heatmap ──────────────────────────────────────
        t50_data: Dict[str, Dict[str, float]] = {}
        for label in labels:
            perc_dir = graph_results[label]["output_dir"] / "percolation"
            if not perc_dir.exists():
                continue
            for csv in sorted(perc_dir.glob("*_summary.csv")):
                scenario = csv.stem.replace("_summary", "")
                key = f"{label} / {scenario}"
                try:
                    df = pd.read_csv(csv, index_col=0)
                    if "T_50" in df.columns:
                        t50_data[key] = df["T_50"].to_dict()
                except Exception:
                    pass

        if len(t50_data) >= 2:
            all_facs = sorted({f for d in t50_data.values() for f in d.keys()})
            matrix = np.full((len(t50_data), len(all_facs)), fill_value=np.nan)
            for ri, (key, vals) in enumerate(t50_data.items()):
                for ci, fac in enumerate(all_facs):
                    matrix[ri, ci] = vals.get(fac, np.nan)

            fig, ax = plt.subplots(
                figsize=(max(6, len(all_facs) * 1.0), max(4, len(t50_data) * 0.7))
            )
            im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn",
                           vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks(range(len(all_facs)))
            ax.set_xticklabels(all_facs, rotation=40, ha="right", fontsize="small")
            ax.set_yticks(range(len(t50_data)))
            ax.set_yticklabels(list(t50_data.keys()), fontsize="small")
            fig.colorbar(im, ax=ax, label="T₅₀ (fraction removed at 50% penalty)")
            ax.set_title("Percolation T₅₀ Heatmap: Scenario × Facility",
                         fontweight="bold")
            fig.tight_layout()
            path = comp_dir / "comparison_percolation_T50_heatmap.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            typing(f"  ✓ Percolation T₅₀ heatmap → {path.name}\n")
            self.logger.info(f"Saved: {path}")

    def _compare_vulnerability_viz(
        self,
        graph_results: Dict[str, Dict],
        labels: List[str],
        comparison_root: Path,
    ) -> None:
        """Grouped bar chart of global vulnerability indices across graphs."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        comp_dir = comparison_root / "comparison"
        comp_dir.mkdir(parents=True, exist_ok=True)
        palette = graph_palette(len(labels))

        data: Dict[str, Dict[str, float]] = {}
        for label in labels:
            csv = (
                graph_results[label]["output_dir"]
                / "vulnerability"
                / "global_indices.csv"
            )
            if csv.exists():
                try:
                    df = pd.read_csv(csv)
                    data[label] = dict(zip(df["index"], df["value"].astype(float)))
                except Exception as e:
                    self.logger.warning(f"Vulnerability viz {label}: {e}")

        if not data:
            return

        priority = ["V_len", "V_pop", "G_pop"]
        extras = sorted({k for d in data.values() for k in d if k not in priority})
        indices = priority + extras[:6]

        x = np.arange(len(indices))
        w = 0.8 / len(labels)
        offsets = (np.arange(len(labels)) - (len(labels) - 1) / 2) * w

        fig, ax = plt.subplots(figsize=(max(7, len(indices) * 1.4), 5))
        for (label, vals), color, offset in zip(data.items(), palette, offsets):
            ys = [vals.get(idx, math.nan) for idx in indices]
            ax.bar(x + offset, ys, w, color=color, label=label, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(indices, rotation=30, ha="right")
        ax.set_ylabel("Index value")
        ax.set_title("Vulnerability Index Comparison", fontweight="bold")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        path = comp_dir / "comparison_vulnerability.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        typing(f"  ✓ Vulnerability chart      → {path.name}\n")
        self.logger.info(f"Saved: {path}")

    # ── master dashboard + console summary ────────────────────────────────

    def _dashboard(
        self,
        graph_results: Dict[str, Dict],
        labels: List[str],
        modules: List[str],
        comparison_root: Path,
    ) -> None:
        """Multi-panel master comparison figure: A–E panels drawn as available."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import matplotlib.ticker as mticker
        import numpy as np
        import pandas as pd

        comp_dir = comparison_root / "comparison"
        comp_dir.mkdir(parents=True, exist_ok=True)
        palette = graph_palette(len(labels))
        meta_cols = {"step", "fraction_removed", "pop_direct"}

        # A: edge disruption
        edg_dir: List[float] = []
        edg_ind: List[float] = []
        edg_lbl: List[str] = []
        if "stats" in modules:
            for label in labels:
                for csv in sorted(
                    (graph_results[label]["output_dir"] / "stats").glob("*_roads.csv")
                ):
                    try:
                        df = pd.read_csv(csv, index_col=0)
                        if "TOTAL" in df.index:
                            row = df.loc["TOTAL"]
                            _, d = parse_n_pct(row.get("Direct [n (%)]"))
                            _, na = parse_n_pct(row.get("No access [n (%)]"))
                            if not math.isnan(d):
                                edg_dir.append(d)
                                edg_ind.append(max(0.0, (na if not math.isnan(na) else d) - d))
                                edg_lbl.append(label)
                    except Exception:
                        pass
                    break

        # B: population
        pop_net: List[float] = []
        pop_dir: List[float] = []
        pop_tot: List[float] = []
        pop_lbl: List[str] = []
        if "stats" in modules:
            for label in labels:
                for csv in sorted(
                    (graph_results[label]["output_dir"] / "stats").glob("*_population.csv")
                ):
                    try:
                        df = pd.read_csv(csv, index_col=0)
                        if "Total" in df.index:
                            row = df.loc["Total"]
                            pop_net.append(float(row.get("Network population", 0) or 0))
                            pop_dir.append(float(row.get("Direct loss", 0) or 0))
                            pop_tot.append(float(row.get("Any service — total", 0) or 0))
                            pop_lbl.append(label)
                    except Exception:
                        pass
                    break

        # C: robustness n_components
        rob_data: Dict[str, pd.DataFrame] = {}
        if "robustness" in modules:
            for label in labels:
                csv = graph_results[label]["output_dir"] / "robustness" / "robustness.csv"
                if csv.exists():
                    try:
                        rob_data[label] = pd.read_csv(csv, index_col=0)
                    except Exception:
                        pass

        # D: percolation any_crit trajectory
        perc_data: Dict[str, pd.DataFrame] = {}
        if "percolation" in modules:
            for label in labels:
                perc_dir_p = graph_results[label]["output_dir"] / "percolation"
                for csv in sorted(perc_dir_p.glob("*_trajectory.csv")):
                    try:
                        df = pd.read_csv(csv)
                        if "fraction_removed" in df.columns:
                            perc_data[label] = df
                            break
                    except Exception:
                        pass

        # E: vulnerability
        vuln_data: Dict[str, Dict[str, float]] = {}
        if "vulnerability" in modules:
            for label in labels:
                csv = (
                    graph_results[label]["output_dir"]
                    / "vulnerability"
                    / "global_indices.csv"
                )
                if csv.exists():
                    try:
                        df = pd.read_csv(csv)
                        vuln_data[label] = dict(zip(df["index"], df["value"].astype(float)))
                    except Exception:
                        pass

        panels = []
        if edg_lbl:
            panels.append("A")
        if pop_lbl and any(p > 0 for p in pop_net):
            panels.append("B")
        if rob_data:
            panels.append("C")
        if perc_data:
            panels.append("D")
        if vuln_data:
            panels.append("E")

        if not panels:
            self.logger.warning("Dashboard: no data for any panel; skipping.")
            return

        n = len(panels)
        ncols = min(3, n)
        nrows = math.ceil(n / ncols)
        fig = plt.figure(figsize=(7 * ncols, 5 * nrows))
        gs = gridspec.GridSpec(nrows, ncols, figure=fig,
                               hspace=0.45, wspace=0.35)

        def _ax(panel_idx: int):
            r, c = divmod(panel_idx, ncols)
            return fig.add_subplot(gs[r, c])

        for pi, panel in enumerate(panels):
            ax = _ax(pi)

            if panel == "A":
                x = np.arange(len(edg_lbl))
                ax.bar(x, edg_dir, 0.55, color="#c0392b", alpha=0.85,
                       label="Direct")
                ax.bar(x, edg_ind, 0.55, bottom=edg_dir, color="#e67e22",
                       alpha=0.85, label="Indirect")
                ax.set_xticks(x)
                ax.set_xticklabels(edg_lbl, rotation=25, ha="right",
                                   fontsize="x-small")
                ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
                ax.set_title("Edge Disruption", fontweight="bold", fontsize=10)
                ax.legend(fontsize="xx-small")
                ax.grid(axis="y", alpha=0.3)

            elif panel == "B":
                x = np.arange(len(pop_lbl))
                w = 0.25
                ax.bar(x - w, pop_net, w, color="#bdc3c7", alpha=0.8,
                       label="Network pop.")
                ax.bar(x,     pop_dir, w, color="#8e44ad", alpha=0.85,
                       label="Direct loss")
                ax.bar(x + w, pop_tot, w, color="#2980b9", alpha=0.85,
                       label="Any-svc total")
                ax.set_xticks(x)
                ax.set_xticklabels(pop_lbl, rotation=25, ha="right",
                                   fontsize="x-small")
                ax.set_ylabel("Persons", fontsize="small")
                ax.set_title("Population Impact", fontweight="bold", fontsize=10)
                ax.legend(fontsize="xx-small")
                ax.grid(axis="y", alpha=0.3)

            elif panel == "C":
                metric = "n_components"
                for (label, df), color in zip(rob_data.items(), palette):
                    if metric not in df.columns:
                        continue
                    vals = pd.to_numeric(df[metric], errors="coerce").values
                    ax.plot(np.arange(len(vals)), vals, marker="o",
                            color=color, label=label, linewidth=2, markersize=5)
                ax.set_xlabel("Network state", fontsize="small")
                ax.set_ylabel("Components", fontsize="small")
                ax.set_title("Robustness: Connected Components",
                             fontweight="bold", fontsize=10)
                ax.legend(fontsize="xx-small")
                ax.grid(alpha=0.3)

            elif panel == "D":
                for (label, df), color in zip(perc_data.items(), palette):
                    x = df["fraction_removed"].values
                    svc_cols = [c for c in df.columns if c not in meta_cols]
                    y_col = "any_crit" if "any_crit" in svc_cols else (
                        svc_cols[0] if svc_cols else None
                    )
                    if y_col is None:
                        continue
                    y = df[y_col].values
                    ax.plot(x, y, color=color, label=label, linewidth=2, alpha=0.9)
                    if "pop_direct" in df.columns:
                        ax.fill_between(x, 0, df["pop_direct"].values,
                                        color=color, alpha=0.08)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_xlabel("Fraction removed", fontsize="small")
                ax.set_ylabel("Population penalty", fontsize="small")
                ax.set_title("Percolation Curves", fontweight="bold", fontsize=10)
                ax.legend(fontsize="xx-small")
                ax.grid(alpha=0.3)

            elif panel == "E":
                v_idxs = ["V_len", "V_pop", "G_pop"]
                present = [i for i in v_idxs
                           if any(i in d for d in vuln_data.values())]
                if present:
                    x_v = np.arange(len(present))
                    w_v = 0.8 / len(labels)
                    offs = (np.arange(len(labels)) - (len(labels) - 1) / 2) * w_v
                    for (label, vals), color, off in zip(
                        vuln_data.items(), palette, offs
                    ):
                        ys = [vals.get(i, math.nan) for i in present]
                        ax.bar(x_v + off, ys, w_v, color=color,
                               label=label, alpha=0.85)
                    ax.set_xticks(x_v)
                    ax.set_xticklabels(present, fontsize="x-small")
                    ax.set_ylabel("Index value", fontsize="small")
                    ax.set_title("Vulnerability Indices",
                                 fontweight="bold", fontsize=10)
                    ax.legend(fontsize="xx-small")
                    ax.grid(axis="y", alpha=0.3)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        fig.suptitle("ANTIPOMPEII Comparative Dashboard",
                     fontsize=14, fontweight="bold", y=1.01)
        path = comp_dir / "comparison_dashboard.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        typing(f"  ✓ Master dashboard         → {path.name}\n")
        self.logger.info(f"Saved: {path}")

    def _overview(
        self,
        graph_results: Dict[str, Dict],
        labels: List[str],
        modules: List[str],
    ) -> None:
        """Print a brief tabular comparative summary to the console."""
        import pandas as pd

        col_w = max(len(lbl) for lbl in labels) + 2

        def _row(metric: str, vals: Dict[str, Any]) -> None:
            line = f"  {metric:<40s}"
            for lbl in labels:
                v = vals.get(lbl, "—")
                if isinstance(v, float):
                    line += ("—" if math.isnan(v) else f"{v:>{col_w}.4f}")
                else:
                    line += f"{str(v):>{col_w}}"
            print(line)

        header = f"  {'Metric':<40s}" + "".join(
            f"{lbl:>{col_w}}" for lbl in labels
        )
        print(header)
        print("  " + "─" * (40 + col_w * len(labels)))

        if "robustness" in modules:
            for metric_key in ["betweenness_max", "betweenness_avg", "connectivity"]:
                vals: Dict[str, Any] = {}
                for label in labels:
                    csv = (
                        graph_results[label]["output_dir"]
                        / "robustness"
                        / "robustness.csv"
                    )
                    try:
                        df = pd.read_csv(csv, index_col=0)
                        vals[label] = (
                            df[metric_key].iloc[0]
                            if metric_key in df.columns
                            else "—"
                        )
                    except Exception:
                        vals[label] = "—"
                _row(f"Robustness (intact) [{metric_key}]", vals)

        if "vulnerability" in modules:
            for idx_name in ["V_len", "V_pop", "G_pop"]:
                vals = {}
                for label in labels:
                    csv = (
                        graph_results[label]["output_dir"]
                        / "vulnerability"
                        / "global_indices.csv"
                    )
                    try:
                        df = pd.read_csv(csv)
                        row = df[df["index"] == idx_name]
                        vals[label] = (
                            row["value"].iloc[0] if not row.empty else "—"
                        )
                    except Exception:
                        vals[label] = "—"
                _row(f"Vulnerability [{idx_name}]", vals)

        print()


__all__ = ["ComparativeReport", "graph_palette", "parse_n_pct"]
