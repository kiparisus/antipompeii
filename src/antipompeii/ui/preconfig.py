"""
Typed pre-configuration model for ANTIPOMPEII Mode 4 (PRE-CONFIGURED).

A :class:`Preconfig` is the in-memory mirror of ``src/antipompeii/config.yaml``
for batch / headless runs.  When ``mode == 4`` is selected (either in the YAML
or via ``--mode 4`` on the command line), every interactive prompt that the
pipeline would otherwise ask is answered from this object instead.

Design
------
* Each nested dataclass mirrors one prompt-heavy stage.  Field names match
  the identifiers used in :mod:`src.antipompeii.ui.cli`; defaults match the
  prompt defaults so a config that omits a key falls through to established
  interactive behaviour silently.
* :func:`load_preconfig` is *strict on unknown keys* (warns) and *lenient on
  missing keys* (uses dataclass defaults).  Enum-typed fields are validated
  at load time so bad input fails fast instead of mid-pipeline.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Nested sections
# ---------------------------------------------------------------------------

@dataclass
class ExtentCfg:
    use_coordinates: bool  = False
    min_longitude:   float = 16.33848
    min_latitude:    float = 48.05052
    max_longitude:   float = 16.41860
    max_latitude:    float = 48.07905
    crs:             str   = "EPSG:4326"


@dataclass
class TemporalCfg:
    enabled:        bool          = False
    timestamps:     List[str]     = field(default_factory=list)
    recovery_graph: Optional[str] = None
    recovery_label: str           = "Recovery"


@dataclass
class FacilitiesCfg:
    max_distance:  float         = 0.0001
    strategy:      str           = "all"          # single | n_nearest | all
    max_neighbors: Optional[int] = None


@dataclass
class DemographyCfg:
    download:     bool = True
    disaggregate: bool = True


@dataclass
class DisruptionCfg:
    skip_if_missing: bool = True


@dataclass
class DemCfg:
    download:  bool          = True
    api_key:   Optional[str] = None
    product:   str           = "SRTMGL3"
    n_samples: int           = 5


@dataclass
class WaterCfg:
    download:             bool = True
    include_wetlands:     bool = True
    include_coastline:    bool = True
    include_intermittent: bool = False


@dataclass
class GraphCfg:
    build:           bool  = True
    directed:        bool  = False
    tolerance:       float = 1.0e-8
    simplify:        bool  = True
    run_diagnostics: bool  = False


@dataclass
class StatsAnalysisCfg:
    run:      bool = True
    compound: bool = True


@dataclass
class RobustnessAnalysisCfg:
    run:    bool = True
    weight: str  = "length"          # length | hops


@dataclass
class PercolationCfg:
    run:             bool          = True
    scenarios:       List[int]     = field(default_factory=lambda: [1, 2])
    n_steps:         Optional[int] = None
    recompute_every: int           = 1
    run_null:        bool          = False
    null_m:          int           = 5
    n_random_runs:   int           = 10
    elev_direction:  str           = "flood"    # flood | closure


@dataclass
class VulnerabilityCfg:
    run:            bool = True
    n_sim:          int  = 200
    use_elevation:  bool = True
    use_water:      bool = True
    run_service_mc: bool = True


@dataclass
class AnalysisCfg:
    stats:         StatsAnalysisCfg      = field(default_factory=StatsAnalysisCfg)
    robustness:    RobustnessAnalysisCfg = field(default_factory=RobustnessAnalysisCfg)
    percolation:   PercolationCfg        = field(default_factory=PercolationCfg)
    vulnerability: VulnerabilityCfg      = field(default_factory=VulnerabilityCfg)


@dataclass
class IACfg:
    run:      bool          = False
    backend:  str           = "ollama"   # ollama | claude | openai | perplexity | custom
    model:    str           = "llama3"
    api_key:  Optional[str] = None
    api_base: Optional[str] = "http://localhost:11434"
    timeout:  int           = 300
    modules:  List[str]     = field(default_factory=lambda: ["all"])


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------

@dataclass
class Preconfig:
    mode:           int           = 1
    config_version: int           = 1
    city:           str           = ""
    extent:         ExtentCfg     = field(default_factory=ExtentCfg)
    temporal:       TemporalCfg   = field(default_factory=TemporalCfg)
    facilities:     FacilitiesCfg = field(default_factory=FacilitiesCfg)
    demography:     DemographyCfg = field(default_factory=DemographyCfg)
    disruption:     DisruptionCfg = field(default_factory=DisruptionCfg)
    dem:            DemCfg        = field(default_factory=DemCfg)
    water:          WaterCfg      = field(default_factory=WaterCfg)
    graph:          GraphCfg      = field(default_factory=GraphCfg)
    analysis:       AnalysisCfg   = field(default_factory=AnalysisCfg)
    ia:             IACfg         = field(default_factory=IACfg)

    @property
    def recovery_graph_path(self) -> Optional[Path]:
        if self.temporal.recovery_graph:
            return Path(self.temporal.recovery_graph)
        return None


# ---------------------------------------------------------------------------
# Loader / validator
# ---------------------------------------------------------------------------

# Enum-valued fields that must validate at load time.  Keys are dotted paths
# matching the dataclass hierarchy.
_ENUMS: Dict[str, set] = {
    "facilities.strategy":                {"single", "n_nearest", "all"},
    "analysis.robustness.weight":         {"length", "hops"},
    "analysis.percolation.elev_direction":{"flood", "closure"},
    "ia.backend":                         {"ollama", "claude", "openai", "perplexity", "custom"},
}


def _coerce(value: Any, current: Any) -> Any:
    """Forgive YAML int/float quirks; pass everything else through."""
    if value is None or current is None:
        return value
    # bool is a subclass of int, so guard explicitly.
    if isinstance(current, bool) or isinstance(value, bool):
        return value
    if isinstance(current, int) and isinstance(value, float):
        return int(value)
    if isinstance(current, float) and isinstance(value, int):
        return float(value)
    return value


def _apply_dict(data: Dict[str, Any], target: Any, *, key_path: str = "") -> None:
    """Populate dataclass *target* in place from *data*."""
    known = {f.name for f in fields(target)}
    for raw_key, raw_val in (data or {}).items():
        if raw_key not in known:
            _LOG.warning(
                "Unknown config key: %s%s — ignoring",
                f"{key_path}." if key_path else "",
                raw_key,
            )
            continue
        current = getattr(target, raw_key)
        full_key = f"{key_path}.{raw_key}" if key_path else raw_key
        if is_dataclass(current) and isinstance(raw_val, dict):
            _apply_dict(raw_val, current, key_path=full_key)
            continue
        value = _coerce(raw_val, current)
        if full_key in _ENUMS and value not in _ENUMS[full_key]:
            raise ValueError(
                f"Invalid value for '{full_key}': {value!r}; "
                f"expected one of {sorted(_ENUMS[full_key])}"
            )
        setattr(target, raw_key, value)


def load_preconfig(path: Path) -> Preconfig:
    """
    Parse *path* into a :class:`Preconfig`.

    * Missing file → returns defaults (``mode=1``, interactive run).
    * Unknown keys → logged as warnings and ignored.
    * Bad enum values → :class:`ValueError`.
    """
    pre = Preconfig()
    if not path.exists():
        _LOG.info("No config file at %s; using defaults.", path)
        return pre
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"{path}: top-level YAML must be a mapping, got {type(raw).__name__}"
        )
    _apply_dict(raw, pre)
    return pre


__all__ = [
    "Preconfig",
    "ExtentCfg",
    "TemporalCfg",
    "FacilitiesCfg",
    "DemographyCfg",
    "DisruptionCfg",
    "DemCfg",
    "WaterCfg",
    "GraphCfg",
    "StatsAnalysisCfg",
    "RobustnessAnalysisCfg",
    "PercolationCfg",
    "VulnerabilityCfg",
    "AnalysisCfg",
    "IACfg",
    "load_preconfig",
]
