"""
Typed pipeline state for the ANTIPOMPEII CLI.

Replaces the ``session_data: Dict[str, Any]`` god-bag with three small
dataclasses:

* :class:`PipelineConfig` — settings chosen during configuration (city,
  extent, timestamps, recovery graph).
* :class:`Snapshot`       — one walk through the enrichment pipeline.  In
  non-temporal runs there is exactly one snapshot; temporal runs produce
  one per timestamp.  Each snapshot tracks the artifacts it has accumulated.
* :class:`PipelineState`  — the configuration plus the list of snapshots
  plus a few orphan fields that aren't per-snapshot (loaders, DEM, results).

The temporal/single split that used to clutter every stage —
``if use_temporal: ... else: ...`` — collapses to a loop over
``state.snapshots`` once stages move to this model.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.antipompeii.ui.preconfig import Preconfig


def _slugify(value: str) -> str:
    """Collapse a free-form string into a filesystem-safe stem."""
    cleaned = re.sub(r"[^\w\-]+", "_", value or "").strip("_")
    return cleaned or "unknown"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """User-chosen settings.  Filled in by the early prompts."""

    mode: int = 1
    city: str = ""
    use_coordinates: bool = False
    coordinates: Dict[str, float] = field(default_factory=dict)
    use_temporal: bool = False
    timestamps: List[str] = field(default_factory=list)
    recovery_graph: Optional[Path] = None
    recovery_label: str = "Recovery"


# ---------------------------------------------------------------------------
# Snapshot — one path through the enrichment pipeline
# ---------------------------------------------------------------------------

@dataclass
class Snapshot:
    """
    All artifacts produced for one timestamp.  In non-temporal mode the
    pipeline has a single ``Snapshot(timestamp=None)``; in temporal mode it
    holds one per requested date.
    """

    timestamp: Optional[str] = None      # YYYYMMDD, or None for "current"

    # Enrichment pipeline outputs, in the order they are produced.
    osm_gpkg:                 Optional[Path] = None
    streets_with_facilities:  Optional[Path] = None
    streets_with_population:  Optional[Path] = None
    streets_with_disruption:  Optional[Path] = None
    streets_with_elevation:   Optional[Path] = None
    streets_with_water:       Optional[Path] = None

    # Graph artifacts.
    graph:            Optional[Path] = None
    simplified_graph: Optional[Path] = None

    @property
    def enriched_gpkg(self) -> Optional[Path]:
        """
        Return the most-enriched GeoPackage that exists on disk so far,
        falling through from water → elevation → disruption → population →
        facilities → raw OSM.
        """
        candidates = (
            self.streets_with_water,
            self.streets_with_elevation,
            self.streets_with_disruption,
            self.streets_with_population,
            self.streets_with_facilities,
            self.osm_gpkg,
        )
        for p in candidates:
            if p is not None and p.exists():
                return p
        return None

    @property
    def best_graph(self) -> Optional[Path]:
        """
        Most-processed graph available on disk: a simplified graph if one
        exists, otherwise the raw graph.  Returned to analytics stages so
        they can run against whichever ``.gt`` artifact is present.
        """
        for p in (self.simplified_graph, self.graph):
            if p is not None and p.exists():
                return p
        return None

    @property
    def label(self) -> str:
        """Human-readable identifier (timestamp, or ``"current"``)."""
        return self.timestamp or "current"


# ---------------------------------------------------------------------------
# PipelineState — root container
# ---------------------------------------------------------------------------

@dataclass
class PipelineState:
    """
    Root container.  Lives on the CLI instance for the duration of one run.
    """

    config:    PipelineConfig = field(default_factory=PipelineConfig)
    snapshots: List[Snapshot] = field(default_factory=list)

    # Pre-configured (Mode 4) answer source.  Always populated — defaults to
    # an inert ``Preconfig()`` so the ``preconfigured`` property is the single
    # switch that stage fast-paths read.
    preconfig: Optional["Preconfig"] = None

    # Cross-snapshot inputs (shared between all snapshots).
    osm_dir:        Optional[Path] = None    # where OSM files were saved
    elevation_dem:  Optional[Path] = None    # downloaded DEM
    water_layer:    Optional[Path] = None    # downloaded OSM water-feature GPKG
    osm_loader:        Any = None
    population_loader: Any = None

    # Analysis results — kept here so Mode 5 / IA interpreter can find them.
    analysis_results:     Any = None
    robustness_report:    Any = None
    percolation_results:  Any = None
    vulnerability_result: Any = None

    # ── ergonomics ────────────────────────────────────────────────────────

    @property
    def is_temporal(self) -> bool:
        return self.config.use_temporal and len(self.snapshots) > 1

    @property
    def preconfigured(self) -> bool:
        """True when the run is Mode 4 (pre-configured / non-interactive)."""
        return self.config.mode == 4 and self.preconfig is not None

    @property
    def case_slug(self) -> str:
        """
        Filesystem-safe identifier for this study.

        Forms:

        * ``{YYYYMMDD}_{Location}`` — single timestamp (Modes 1, 2 non-temporal)
        * ``{Location}_temporal_{first}_{last}`` — temporal run with N events
        * ``{today}_{Location}`` — fallback when no timestamp is recorded

        The slug is deterministic: re-running the same study yields the same
        slug, so subsequent runs land in the same case directory.
        """
        city       = _slugify(self.config.city)
        timestamps = [s.timestamp for s in self.snapshots if s.timestamp]

        if not timestamps:
            return f"{datetime.now().strftime('%Y%m%d')}_{city}"
        if self.is_temporal and len(timestamps) > 1:
            return f"{city}_temporal_{timestamps[0]}_{timestamps[-1]}"
        return f"{timestamps[0]}_{city}"

    def get(self, timestamp: Optional[str]) -> Optional[Snapshot]:
        """Look up a snapshot by timestamp string (``None`` for non-temporal)."""
        for s in self.snapshots:
            if s.timestamp == timestamp:
                return s
        return None

    def ensure(self, timestamp: Optional[str]) -> Snapshot:
        """Return the snapshot for *timestamp*, creating it if missing."""
        snap = self.get(timestamp)
        if snap is None:
            snap = Snapshot(timestamp=timestamp)
            self.snapshots.append(snap)
        return snap


__all__ = ["PipelineConfig", "Snapshot", "PipelineState"]
