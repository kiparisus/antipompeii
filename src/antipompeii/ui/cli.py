########################################
####### ANTIPOMPEII interface ##########
########################################

from typing import Dict, Any, Tuple, Optional, List, Set
from pathlib import Path
from datetime import datetime
import json
import re as _re


def _sanitize_label(label: str) -> str:
    """Convert a graph label to a filesystem-safe path component."""
    cleaned = _re.sub(r"[^\w\-]", "_", label).strip("_")
    return cleaned or "graph"


# ---------------------------------------------------------------------------
# Local-automated mode: filesystem discovery
#
# Files in ``DATA_INPUT_DIR`` follow the convention
#
#     {YYYYMMDD}_{Location}[_demography][_disruption][_elevation][_network][_simplified].{gpkg,gt}
#
# where ``Location`` may itself contain underscores (e.g. ``Mexico_Beach_Florida_USA``).
# Parsing peels the known suffixes from the right end of the filename to
# isolate ``date`` and ``location``.
# ---------------------------------------------------------------------------


_LOCAL_FILE_PATTERN = _re.compile(
    r"^(?P<date>\d{8})_(?P<rest>.+)\.(?P<ext>gpkg|gt)$"
)
_LOCAL_SUFFIXES = ("_demography", "_disruption", "_elevation", "_water", "_network", "_simplified")


def _parse_local_filename(name: str) -> Optional[Tuple[str, str, Set[str], str]]:
    """
    Decompose a filename into ``(date, location, {suffixes}, extension)``.
    Returns ``None`` if the name doesn't follow the convention.
    """
    m = _LOCAL_FILE_PATTERN.match(name)
    if m is None:
        return None
    date = m["date"]
    rest = m["rest"]
    ext  = m["ext"]

    suffixes: Set[str] = set()
    # Peel known suffixes from the right end of ``rest`` until none match.
    while True:
        for suffix in _LOCAL_SUFFIXES:
            if rest.endswith(suffix):
                suffixes.add(suffix[1:])     # drop the leading underscore
                rest = rest[: -len(suffix)]
                break
        else:
            break

    return date, rest, suffixes, ext


from src.antipompeii.utils.configmanager import ConfigManager
from src.antipompeii.utils.logger import get_logger
from src.antipompeii.ui.layout import (
    print_banner,
    print_section,
    wait_for_enter,
    typing,
    typereal,
    paint,
    get_user_input,
)
from src.antipompeii.ui.prompts import (
    confirm,
    ask_int,
    ask_float,
    ask_choice,
    ask_text,
)
from src.antipompeii.ui.feedback import (
    info,
    step,
    success,
    warn,
    error,
    stage,
)
from src.antipompeii.ui.state import PipelineState, Snapshot
from src.antipompeii.ui.preconfig import Preconfig, load_preconfig
from src.antipompeii.ui.art import POMPEII_ASCII
from src.antipompeii.modules.base_autoloader import DataLoader
from src.antipompeii.modules.demography_loader import PopulationLoaderEnhanced
from src.antipompeii.modules.demography_processing import (
    DemographicRasters,
    append_population_to_streets,
)
from src.antipompeii.modules.disruption_loader import (
    append_disruption_to_streets,
    find_disruption_file,
)
from src.antipompeii.modules.disaster_lookup import lookup_disasters
from src.antipompeii.modules.facility_processing import append_facilities_to_streets
from src.antipompeii.modules.dem_downloader import (
    DEMDownloader,
    DEM_PRODUCTS,
    DEM_PRODUCT_KEYS,
)
from src.antipompeii.modules.dem_processing import append_elevation_to_streets
from src.antipompeii.modules.water_downloader import WaterDownloader
from src.antipompeii.modules.water_processing import append_water_distance_to_streets
from src.antipompeii.modules.graph_builder import build_graph_from_streets
from src.antipompeii.modules.network_simplifier import simplify_network
from src.antipompeii.modules.stats_analyst import analyse_network
from src.antipompeii.modules.robustness_estimator import estimate_robustness
from src.antipompeii.modules.percolator import run_percolation
from src.antipompeii.modules.vulnerability_simulator import run_vulnerability_simulation
from src.antipompeii.ui.comparison import ComparativeReport

ANTIPOMPEII_DIR = Path(__file__).parent.parent  # points to src/antipompeii/
DEFAULT_CONFIG = ANTIPOMPEII_DIR / "config.yaml"

DATA_INPUT_DIR  = ANTIPOMPEII_DIR.parent / "data" / "input"
DATA_OUTPUT_DIR = ANTIPOMPEII_DIR.parent / "data" / "output"

TITLE = "ANTIPOMPEII"
SUBTITLE = "urban vulnerability assessment and resilience improvement tool"
AUTHOR = "Pavel Kiparisov | pavel@kiparisov.space"
VERSION = "0.1.1"

MODE_OPTIONS = """
1. ONLINE [default]
   ANTIPOMPEII downloads core data from the web. Other data are looked up in the input
   folder. If missing, the user is asked.
2. LOCAL-AUTOMATED
   ANTIPOMPEII attempts to look up data layers stored in the input folder based on their
   naming conventions and timestamps.
3. LOCAL-MANUAL
   User indicates paths to each layer of data.
4. PRE-CONFIGURED
   ANTIPOMPEII runs non-interactively, taking every answer from a YAML config file
   (default: src/antipompeii/config.yaml; override with --config <path>).
5. EXISTING GRAPH
   ANTIPOMPEII initializes a graph created by the user or by itself earlier.
"""

# ---------------------------------------------------------------------------
# Reusable prompt copy
#
# Hoisted out of method bodies so the spoken-to-user wording lives in one
# place and isn't smeared across the orchestration logic.  Format-string
# placeholders (``{n}``) are filled at the call site.
# ---------------------------------------------------------------------------

COMBINE_EVENTS_STATS = """\
Found {n} disaster event(s).
 How should they be combined?
  compound    — each event adds disruption to the previous state
  independent — each event is analyzed on the intact network"""

COMBINE_EVENTS_ROBUSTNESS = """\
Found {n} disaster event(s). Combine as:
  compound    — each event cumulatively adds disrupted edges
  independent — each event analyzed on the intact network"""

WEIGHT_PROMPT = """\
Weight paths by edge length or use hop count?
  length — use 'length' edge property (meters)
  hops   — unweighted (edge count)"""

ELEV_REMOVAL_FLOOD_CLOSURE = """\
Elevation removal direction:
  flood   — remove lowest-elevation edges first (water rises)
  closure — remove highest-elevation edges first"""

LLM_BACKEND_MENU = """\

Select LLM backend:
  1. Local Ollama   (no API key required)
  2. Claude         (Anthropic)
  3. OpenAI / GPT
  4. Perplexity AI
  5. Other          (custom litellm model string)
"""


# ---------------------------------------------------------------------------
# Local-automated mode: case discovery
#
# A "disaster case" is the set of artifacts that share a ``(date, location)``
# stem under ``DATA_INPUT_DIR``.  The discovery routine groups files by
# stem and returns one :class:`Snapshot` per case, populated with whatever
# pipeline artifacts exist on disk.
# ---------------------------------------------------------------------------

def _discover_local_cases(input_dir: Path) -> List[Tuple[Snapshot, str]]:
    """
    Scan *input_dir* for ANTIPOMPEII-conventioned GPKG/GT files and group
    them into disaster cases.

    Returns a list of ``(snapshot, location_slug)`` pairs sorted by date
    descending (newest first), then by location alphabetically.
    """
    if not input_dir.exists():
        return []

    by_case: Dict[Tuple[str, str], Snapshot] = {}

    for path in sorted(input_dir.iterdir()):
        if not path.is_file():
            continue
        parsed = _parse_local_filename(path.name)
        if parsed is None:
            continue
        date, location, suffixes, ext = parsed

        key  = (date, location)
        snap = by_case.setdefault(key, Snapshot(timestamp=date))

        if ext == "gpkg":
            if "water" in suffixes:
                snap.streets_with_water = path
            elif "elevation" in suffixes:
                snap.streets_with_elevation = path
            elif "disruption" in suffixes:
                snap.streets_with_disruption = path
            elif "demography" in suffixes:
                snap.streets_with_population = path
            else:
                snap.osm_gpkg = path
        else:  # gt
            if "simplified" in suffixes:
                snap.simplified_graph = path
            elif "network" in suffixes:
                snap.graph = path

    return sorted(
        ((snap, loc) for (_, loc), snap in by_case.items()),
        key=lambda item: (item[0].timestamp or "", item[1]),
        reverse=True,
    )


def _format_case_table(cases: List[Tuple[Snapshot, str]]) -> str:
    """Render the case-selection menu as a readable text table."""

    def pretty_date(ts: Optional[str]) -> str:
        if ts and len(ts) == 8:
            return f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}"
        return ts or "?"

    def tick(p: Optional[Path]) -> str:
        return "✓" if p is not None else "·"

    header = (
        f"{'#':>3}  {'Date':<10}  {'Location':<32}  "
        f"{'OSM':>3}  {'pop':>3}  {'dis':>3}  {'elev':>4}  {'wat':>3}  {'graph':>5}  {'simp':>4}"
    )
    rule = "─" * len(header)
    lines = [header, rule]

    for i, (snap, location) in enumerate(cases, 1):
        loc_pretty = location.replace("_", " ")[:32]
        lines.append(
            f"{i:>3}  {pretty_date(snap.timestamp):<10}  {loc_pretty:<32}  "
            f"{tick(snap.osm_gpkg):>3}  "
            f"{tick(snap.streets_with_population):>3}  "
            f"{tick(snap.streets_with_disruption):>3}  "
            f"{tick(snap.streets_with_elevation):>4}  "
            f"{tick(snap.streets_with_water):>3}  "
            f"{tick(snap.graph):>5}  "
            f"{tick(snap.simplified_graph):>4}"
        )

    return "\n".join(lines)


class antipompeiiCLI:
    """ANTIPOMPEII command-line interface."""

    # Ordered list of pipeline stages.  Each is a parameterless method on
    # this class that consults ``self.state`` to decide whether and how to
    # run.  Adding or reordering a stage is a one-line change here.
    PIPELINE: List[str] = [
        "_load_osm_data",
        "_process_facilities",
        "_load_popdata",
        "_process_population_data",
        "_process_disruption_data",
        "_download_dem",
        "_process_dem_data",
        "_download_water",
        "_process_water_data",
        "_build_graph_network",
        "_simplify_network",
        "_run_network_analysis",
        "_run_robustness_estimation",
        "_run_percolation",
        "_run_vulnerability_simulation",
        "_run_ia_interpreter",
    ]

    # Mode 2 (LOCAL-AUTOMATED) skips the enrichment stages — the user has
    # already produced those artifacts offline — and goes straight to graph
    # building (idempotent when artifacts are present) and analytics.
    # Disruption is still offered as a stage because a discovered case may
    # arrive without a `_disruption.gpkg`; the stage itself short-circuits
    # idempotently when disruption is already attached.
    LOCAL_AUTOMATED_PIPELINE: List[str] = [
        "_process_disruption_data",
        "_build_graph_network",
        "_simplify_network",
        "_run_network_analysis",
        "_run_robustness_estimation",
        "_run_percolation",
        "_run_vulnerability_simulation",
        "_run_ia_interpreter",
    ]

    def __init__(
        self,
        *,
        config_path: Optional[Path] = None,
        mode_override: Optional[int] = None,
    ):
        self.logger = get_logger(__name__)
        self.config_manager = ConfigManager()
        self.state = PipelineState()

        # Load the YAML preconfig once, up-front.  The object is always
        # present (defaults when no file or empty file), but only consulted
        # when ``state.preconfigured`` is true (mode == 4).
        cfg_path = Path(config_path) if config_path is not None else DEFAULT_CONFIG
        try:
            self.state.preconfig = load_preconfig(cfg_path)
        except (ValueError, OSError) as exc:
            self.logger.error(f"Failed to load preconfig {cfg_path}: {exc}")
            self.state.preconfig = Preconfig()
        # Seed the runtime mode from the loaded YAML so an interactive run
        # can still override it at the prompt.
        self.state.config.mode = self.state.preconfig.mode
        if mode_override is not None:
            self.state.config.mode = mode_override
            self.state.preconfig.mode = mode_override
        self.config_path = cfg_path

        # Free-form scratch dict shared with Mode 5, the IA interpreter, and
        # analytics result blobs.  Typed pipeline paths live on
        # ``self.state.snapshots``; everything else still flows through here.
        self.session_data: Dict[str, Any] = {}

    def run(self):
        """
        Run the full interactive session.  Mode 2 (local-automated) and
        Mode 5 (existing-graph analytics) each fork off after mode
        selection; every other mode walks ``PIPELINE``.
        """
        print_banner(title=TITLE, subtitle=SUBTITLE, author=AUTHOR, version=VERSION)

        wait_for_enter("Press Enter to display the cover art and continue...")
        paint(POMPEII_ASCII)
        typing("The Last Day of Pompeii (1833) by \033[1mKarl Bryullov\033[0m -- ASCII fragment\n")
        wait_for_enter("Press Enter to view the welcome message and continue...")
        print_section("Welcome to ANTIPOMPEII")
        print(
            " A Nested Tool for Integrated Planning of Multi-hazard Preparedness\n"
            " and Emergency Infrastructure Intervention (ANTIPOMPEII)\n"
        )
        print(" developed by \033[1mPavel Kiparisov\033[0m\n")
        typing(' \033[3m"...Earth\'s utmost bounds shall join the glad acclaim,')
        typing(" And distant Camus bless Pompeii's name\".\n")
        print(" Pompeii (1819) by T.B. Macaulay\033[0m")

        mode = self._select_mode()
        if mode == 5:
            self._run_existing_graph_mode()
            return
        if mode == 2:
            self._run_local_automated_mode()
            return
        if mode == 4:
            self._run_preconfigured_mode()
            return

        self._select_city()
        self._enter_coordinates()
        self._set_temporal()

        for stage_method in self.PIPELINE:
            getattr(self, stage_method)()

    # ------------------------------------------------------------------
    # Per-case output directory
    #
    # All analytical artifacts for a single study live under
    # ``output/cases/{case_slug}/``.  Multi-case Mode 5 comparisons live
    # under ``output/comparisons/{timestamp}/``.  A small ``case.json``
    # manifest is dropped into each case directory the first time it is
    # touched, giving downstream tooling a stable handle on what's there.
    # ------------------------------------------------------------------

    def _case_output_dir(self) -> Path:
        """Return (and create) the per-case output directory."""
        case_dir = DATA_OUTPUT_DIR / "cases" / self.state.case_slug
        case_dir.mkdir(parents=True, exist_ok=True)
        self._write_case_manifest(case_dir)
        return case_dir

    def _write_case_manifest(self, case_dir: Path) -> None:
        """Drop a ``case.json`` describing the study; idempotent."""
        manifest_path = case_dir / "case.json"
        if manifest_path.exists():
            return
        payload = {
            "case_slug":   self.state.case_slug,
            "city":        self.state.config.city,
            "mode":        self.state.config.mode,
            "is_temporal": self.state.is_temporal,
            "timestamps":  [s.timestamp for s in self.state.snapshots if s.timestamp],
            "created_at":  datetime.now().isoformat(timespec="seconds"),
            "antipompeii_version": VERSION,
        }
        manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
        self.logger.info(f"Wrote case manifest: {manifest_path}")

    # ------------------------------------------------------------------
    # Configuration prompts
    # ------------------------------------------------------------------

    def _select_mode(self) -> int:
        """Ask for the operating mode and record it on ``state.config``."""
        print_section("ANTIPOMPEII mode selection")

        # When mode 4 (pre-configured) is set in the YAML or via --mode, skip
        # the prompt: the entire point of pre-configured is no questions.
        if self.state.config.mode == 4:
            success(
                f"\033[1mMode 4 (PRE-CONFIGURED)\033[0m loaded from "
                f"{self.config_path}"
            )
            self.logger.info(
                f"ANTIPOMPEII working mode is set to 4 (pre-configured from "
                f"{self.config_path})"
            )
            self.session_data["mode"] = 4
            return 4

        typing("See which modes are available.")
        print(MODE_OPTIONS)

        default_mode = self.state.config.mode if 1 <= self.state.config.mode <= 5 else 1
        apmode = ask_int("Select processing mode", default=default_mode, bounds=(1, 5))
        success(f"\033[1mMode {apmode}\033[0m selected")
        self.logger.info(f"ANTIPOMPEII working mode is set to {apmode}")
        self.state.config.mode = apmode
        self.session_data["mode"] = apmode
        return apmode

    def _select_city(self) -> None:
        """Ask for the city name; loops until the user confirms the choice."""
        print_section("Name of the settlement")
        while True:
            city = ask_text(
                "Which city, country would you like to analyze?",
                default="Laxenburg, Austria",
            )
            typing(f"\nSelecting \033[3m{city}\033[0m for analysis\n")
            self.logger.info(f"Settlement is set to {city}")
            if self._display_disaster_lookup(city):
                self.state.config.city = city
                self.session_data["city"] = city
                return

    def _display_disaster_lookup(self, city: str) -> bool:
        """
        Look up *city* in the offline disaster databases, show the 5 most
        recent activations, optionally the full list, and print download
        instructions.

        Returns True if the user confirms the settlement, False to re-select.
        """
        country_hint = city.split(",")[-1].strip() or city

        recent, all_matched = lookup_disasters(city, max_recent=5)

        if not all_matched:
            typing(
                f"\n  No disaster activations found for '{country_hint}' "
                "in local databases (Sentinel Asia / UN Charter).\n"
            )
        else:
            n_total  = len(all_matched)
            n_recent = len(recent)
            typing(
                f"\n  Found {n_total} disaster activation(s) for "
                f"\033[1m{country_hint}\033[0m. "
                f"Showing {n_recent} most recent:\n\n"
            )
            for ev in recent:
                print(ev.display())
                print()

            if n_total > n_recent:
                show_all = confirm(
                    f"  Show all {n_total} activations for {country_hint}?",
                    default=False,
                )
                if show_all:
                    print()
                    for ev in all_matched:
                        print(ev.display())
                        print()

        typing(
            "  Satellite imagery for these events can be downloaded from "
            "the URLs above.\n"
            "  Place the files in \033[3msrc/data/input/disruption/\033[0m "
            "to be processed by ANTIPOMPEII.\n"
        )

        return confirm(
            f"Continue with \033[1m{city}\033[0m as the study area?",
            default=True,
        )

    def _enter_coordinates(self) -> Tuple[bool, Dict[str, Optional[float]]]:
        """
        Get coordinates from user and return them.

        Returns
        -------
        (use_coordinates, coordinates_dict)
        """
        print_section("Geographic coordinates")
        use_coordinates = confirm(
            "Would you like to enter the coordinates of the bounding box manually?",
            default=False,
        )

        coordinates: Dict[str, Optional[float]] = {}
        if use_coordinates:
            coordinates["long_min"] = ask_float(
                "\nMinimum LONGITUDE (Westernmost point of the extent)",
                default=16.33848,
            )
            coordinates["lat_min"] = ask_float(
                "Minimum LATITUDE (Southernmost point of the extent)",
                default=48.05052,
            )
            coordinates["long_max"] = ask_float(
                "Maximum LONGITUDE (Easternmost point of the extent)",
                default=16.41860,
            )
            coordinates["lat_max"] = ask_float(
                "Maximum LATITUDE (Northernmost point of the extent)",
                default=48.07905,
            )

            success("Geographic coordinates received")
            typing(
                f" The extent is set to {coordinates['long_min']} (W), "
                f"{coordinates['lat_min']} (S), {coordinates['long_max']} (E), "
                f"{coordinates['lat_max']} (N)\n"
            )
            self.logger.info(
                "The extent of the bounding box is defined by these coordinates: "
                f"{coordinates['long_min']},{coordinates['lat_min']},"
                f"{coordinates['long_max']},{coordinates['lat_max']}"
            )
        else:
            success("Proceeding to the next step")
            self.logger.info("The extent will be set automatically")

        self.state.config.use_coordinates = use_coordinates
        self.state.config.coordinates = dict(coordinates)
        self.session_data["use_coordinates"] = use_coordinates
        self.session_data["coordinates"] = coordinates
        return use_coordinates, coordinates

    def _set_temporal(self) -> Tuple[bool, Optional[List[str]]]:
        """
        Configure temporal data settings.

        Returns
        -------
        (use_temporal, timestamps_list)
        """
        print_section("Temporal data")
        use_temporal = confirm("Would you like to use temporal data?", default=False)

        timestamps: Optional[List[str]] = None
        if use_temporal:
            timestamps_str = ask_text(
                "Enter all timestamps you would like to use in the YYYYMMDD format "
                "using comma as a separator.\n"
                "E.g. 20181203,20191103,20251014",
                default="20251014",
            )
            timestamps = [ts.strip() for ts in timestamps_str.split(",") if ts.strip()]
            try:
                for ts in timestamps:
                    datetime.strptime(ts, "%Y%m%d")
            except ValueError as e:
                error(f"Invalid timestamp format: {str(e)}")
                print("Please use YYYYMMDD format (e.g., 20181203)\n")
                return self._set_temporal()

            success("Timestamps received")
            typing(
                f" Will download data for {len(timestamps)} temporal snapshot(s): "
                f"{', '.join(timestamps)}\n"
            )
            self.logger.info(
                f"Temporal analysis enabled for timestamps: {timestamps}"
            )

            # Ask whether a recovery / future-state graph will be compared
            has_recovery = confirm(
                "Do you plan to compare a recovered / future-state network\n"
                " in the robustness analysis?",
                default=False,
            )
            if has_recovery:
                rec_path_str = get_user_input(
                    "Path to the recovery .gt graph file (leave blank to specify later)",
                    default="",
                ).strip()
                recovery_path = Path(rec_path_str) if rec_path_str else None
                if recovery_path and not recovery_path.exists():
                    warn(f"Recovery graph not found: {recovery_path} — will ask again later.")
                    recovery_path = None
                rec_label = get_user_input(
                    "Label for recovery state (e.g. '2024' or 'Post-recovery')",
                    default="Recovery",
                ).strip() or "Recovery"
                self.state.config.recovery_graph = recovery_path
                self.state.config.recovery_label = rec_label
                self.session_data["recovery_graph_path"] = recovery_path
                self.session_data["recovery_label"] = rec_label
                if recovery_path:
                    success(f"Recovery state registered: {recovery_path.name}")
        else:
            success("Proceeding with the current state of the network")
            self.logger.info(
                "The current state of the street network will be loaded"
            )

        self.state.config.use_temporal = use_temporal
        self.state.config.timestamps = list(timestamps or [])

        # Seed snapshots once timestamps are known.  Single-mode runs get one
        # snapshot with ``timestamp=None``; temporal runs get one per date.
        if use_temporal and timestamps:
            for ts in timestamps:
                self.state.ensure(ts)
        else:
            self.state.ensure(None)

        self.session_data["use_temporal"] = use_temporal
        self.session_data["timestamps"] = timestamps
        return use_temporal, timestamps

    # ------------------------------------------------------------------
    # Facilities → streets
    # ------------------------------------------------------------------

    def _process_facilities(self) -> None:
        """
        Append facility polygon attributes to nearest street segments in the
        original OSM GeoPackage(s).
        """
        print_section("Facilities to streets")

        if self.state.preconfigured:
            fcfg = self.state.preconfig.facilities
            max_distance = fcfg.max_distance
            if fcfg.strategy == "single":
                max_neighbors = 1
            elif fcfg.strategy == "all":
                max_neighbors = None
            else:  # n_nearest
                max_neighbors = fcfg.max_neighbors or 3
            info(
                f"  Facilities   : strategy={fcfg.strategy}, "
                f"max_distance={max_distance}, max_neighbors={max_neighbors}"
            )
        else:
            max_distance = ask_float(
                "Maximum distance from facility polygon to nearest street (degrees/meters)",
                default=0.0001,
            )

            print("\nSelect facility-to-street connection strategy:")
            print(" 1. Single nearest street per facility")
            print(" 2. Multiple (N) nearest streets per facility")
            print(" 3. All streets within distance threshold (multiedge facility)")

            strategy = ask_int("Choose strategy [1-3]", default=3, bounds=(1, 3))

            if strategy == 1:
                max_neighbors = 1
            elif strategy == 2:
                max_neighbors = ask_int(
                    "Maximum number of nearest streets per facility",
                    default=3,
                    bounds=(1, 50),
                )
            else:
                max_neighbors = None

        # Walk every snapshot that has an OSM GPKG on disk.
        targets = [
            s for s in self.state.snapshots
            if s.osm_gpkg is not None and s.osm_gpkg.exists()
        ]
        if not targets:
            warn("No saved OSM GeoPackage path found; complete the Data Acquisition step first.")
            return

        if self.state.is_temporal:
            typing(
                f"\nAppending facility attributes for {len(targets)} temporal snapshot(s) "
                f"(initial max_distance={max_distance})...\n"
            )

        for snap in targets:
            self._process_facilities_for_snapshot(snap, max_distance, max_neighbors)

        if self.state.is_temporal:
            success("Facility processing for all temporal snapshots completed.")

    def _process_facilities_for_snapshot(
        self,
        snap: Snapshot,
        max_distance: float,
        max_neighbors: Optional[int],
    ) -> None:
        """Run facility join on one snapshot, with an interactive retry loop."""
        gpkg_path = snap.osm_gpkg
        label_str = f" for timestamp {snap.timestamp}" if snap.timestamp else ""

        with stage(
            f"Facility-to-street processing{label_str} ({gpkg_path.name}, "
            f"max_distance={max_distance})",
            logger=self.logger,
        ) as st:
            updated_gdf, unmatched = append_facilities_to_streets(
                merged_gpkg=gpkg_path,
                max_distance=max_distance,
                max_neighbors=max_neighbors,
                logger=self.logger,
            )
            total_facilities = len(
                updated_gdf[
                    updated_gdf["layer_name"].isin(
                        ["Health", "Emergency", "Convertible Shelter", "Commercial", "Power"]
                    )
                ]
            )
            st.note(
                f"Matched {total_facilities - len(unmatched)} of {total_facilities} "
                f"facilities; {len(unmatched)} unmatched beyond {max_distance}"
            )
        if st.failed:
            return

        snap.streets_with_facilities = gpkg_path  # in-place enrichment

        # In pre-configured mode the retry loop is interactive by nature
        # (asks for a new distance); skip it and log the leftover count.
        if self.state.preconfigured and len(unmatched) > 0:
            warn(
                f"{len(unmatched)} facilities unmatched{label_str} "
                f"(max_distance={max_distance}); leaving as-is in Mode 4."
            )
            self.logger.info(
                f"Mode 4: skipping facility retry loop; {len(unmatched)} unmatched."
            )
            return

        # ── unmatched retry loop ─────────────────────────────────────────
        current_dist = max_distance
        while len(unmatched) > 0:
            header = f"UNMATCHED FACILITIES{f' FOR TIMESTAMP {snap.timestamp}' if snap.timestamp else ''}"
            print("\n" + "=" * 70)
            print(header)
            print("=" * 70)

            for _, row in unmatched.iterrows():
                name     = row.get("name", "Unnamed")
                layer    = row.get("layer_name", "Unknown")
                centroid = row.geometry.centroid
                print(f" • {layer}: {name}")
                print(f"   Location: ({centroid.x:.6f}, {centroid.y:.6f})")
            print("=" * 70 + "\n")

            retry = confirm(
                f"\n{len(unmatched)} facilities remain unmatched{label_str}. "
                "Increase distance and retry for unmatched facilities only?",
                default=True,
            )
            if not retry:
                success(
                    f"Facility processing{label_str} complete. "
                    f"{len(unmatched)} facilities left unmatched."
                )
                return

            new_dist = ask_float(
                "New maximum distance (degrees)",
                default=current_dist * 2,
            )

            with stage(
                f"Facility retry{label_str} at max_distance={new_dist}",
                logger=self.logger,
            ) as retry_st:
                _, unmatched = append_facilities_to_streets(
                    merged_gpkg=gpkg_path,
                    max_distance=new_dist,
                    max_neighbors=max_neighbors,
                    facilities_subset=unmatched,
                    logger=self.logger,
                )
                retry_st.note(f"Remaining unmatched: {len(unmatched)}")
            if retry_st.failed:
                return
            current_dist = new_dist

            if len(unmatched) == 0:
                success(f"All facilities{label_str} matched to street segments!")

    # ------------------------------------------------------------------
    # Demographic loading
    # ------------------------------------------------------------------

    def _load_popdata(self):
        """
        Load WorldPop demographic data with enhanced raster capabilities.
        """
        print_section("Demographic data")

        if self.state.preconfigured:
            dem_cfg = self.state.preconfig.demography
            download_worldpop = dem_cfg.download
            disaggregate = dem_cfg.disaggregate
            info(
                f"  Demography   : download={download_worldpop}, "
                f"disaggregate={disaggregate}"
            )
        else:
            download_worldpop = confirm(
                "Would you like to download corresponding demographic data?",
                default=True,
            )
            disaggregate = True

        if not download_worldpop:
            success("Skipping demographic data download")
            self.logger.info("User opted to skip demographic data download")
            return

        download_rasters = True

        print("\nInitializing WorldPop demographic data loader...")
        if download_rasters:
            print("Preparing to download 100m resolution GeoTIFF rasters...")
            typing("Querying age/sex disaggregated statistics...\n")

        try:
            city = self.session_data.get("city")
            use_coordinates = self.session_data.get("use_coordinates", False)
            coordinates = self.session_data.get("coordinates", {})
            timestamps = self.session_data.get("timestamps")

            output_dir = DATA_INPUT_DIR / "demographics"
            output_dir.mkdir(parents=True, exist_ok=True)

            loader = PopulationLoaderEnhanced.from_cli_params(
                city=city,
                use_coordinates=use_coordinates,
                timestamps=timestamps,
                long_min=coordinates.get("long_min") if use_coordinates else None,
                lat_min=coordinates.get("lat_min") if use_coordinates else None,
                long_max=coordinates.get("long_max") if use_coordinates else None,
                lat_max=coordinates.get("lat_max") if use_coordinates else None,
                logger=self.logger,
                download_rasters=download_rasters,
                disaggregate=disaggregate,
                output_dir=output_dir,
            )

            pop_data = loader.load_all_data()

            self.session_data["population_loader"] = loader
            self.session_data["population_data"] = pop_data

            print("\n" + "=" * 60)
            print("DEMOGRAPHIC DATA SUMMARY")
            print("=" * 60 + "\n")
            summary_df = loader.get_summary_dataframe()
            print(summary_df.to_string(index=False))
            print()

            print("\nDownload Performance:\n")
            perf_df = loader.get_performance_summary()
            print(perf_df.to_string(index=False))
            print()

            if download_rasters:
                print("\nRaster Files Created:\n")
                for year in loader.years:
                    paths = loader.get_raster_paths(year)
                    print(f" Year {year}: {len(paths)} files")
                    if disaggregate:
                        print("  - 1 total population (clipped)")
                        print(
                            f"  - {len(paths) - 1} demographic groups "
                            "(male/female × age)"
                        )
                    else:
                        print("  - 1 total population (clipped)")

            success("Demographic data acquisition complete")
            print(f"All files saved to: {output_dir}\n")
            self.logger.info(
                "Demographic data successfully acquired from WorldPop"
            )

        except ImportError:
            self.logger.error("rasterio package required for raster downloads")
            error("Error: rasterio package not installed")
            print("Install with: pip install rasterio\n")
            print("Falling back to API-only mode...\n")
            self._load_popdata_api_only()

        except Exception as e:
            self.logger.error(f"Failed to load demographic data: {str(e)}")
            error(f"Error: {str(e)}")
            print("Please check your internet connection and query parameters.\n")
            if self.state.preconfigured:
                warn("Continuing without demographic data (Mode 4 is non-interactive).")
                return
            if not confirm("\nContinue without demographic data?", default=True):
                raise

    def _load_popdata_api_only(self) -> None:
        """Fallback when rasterio is not available: API-only stats, no rasters."""
        print("\nRunning API-only demographic loading (no rasters)...\n")
        self.logger.info("API-only demographic loading not yet implemented.")

    # ------------------------------------------------------------------
    # Population → streets
    # ------------------------------------------------------------------

    def _process_population_data(self) -> None:
        """
        Attach WorldPop demographic rasters to street segments.
        """
        print_section("Processing population data onto street network")

        population_loader = self.session_data.get("population_loader")
        if population_loader is None:
            print(
                "\n⚠ No demographic loader found in session; "
                "skipping population processing.\n"
            )
            self.logger.warning(
                "Population loader not available; skipping processing."
            )
            return

        pop_data_by_year = getattr(population_loader, "population_data", None)
        if not pop_data_by_year:
            print(
                "\n⚠ No raster metadata in population loader; skipping processing.\n"
            )
            self.logger.warning(
                "population_loader.population_data is empty; skipping."
            )
            return

        demographics_by_year: Dict[int, DemographicRasters] = {}
        for year, pop_data in sorted(pop_data_by_year.items()):
            total_path = pop_data.raster_clipped_path or pop_data.raster_path
            if total_path is None:
                self.logger.warning(
                    f"No total population raster for year {year}; skipping year."
                )
                continue

            age_rasters = pop_data.age_sex_rasters or {}
            required_keys = [
                "female_0-14",
                "female_15-64",
                "female_65+",
                "male_0-14",
                "male_15-64",
                "male_65+",
            ]
            missing = [k for k in required_keys if k not in age_rasters]
            if missing:
                print(
                    f"\n⚠ Aggregated age/sex rasters missing for {year}: "
                    f"{', '.join(missing)}.\n"
                    " Ensure you selected 'y' for disaggregated age/sex rasters "
                    "in the Demographic data step.\n"
                )
                self.logger.warning(
                    f"Missing aggregated age/sex rasters for {year}: {missing}; "
                    "skipping this year."
                )
                continue

            demographics_by_year[year] = DemographicRasters(
                total=total_path,
                female_0_14=age_rasters["female_0-14"],
                female_15_64=age_rasters["female_15-64"],
                female_65_plus=age_rasters["female_65+"],
                male_0_14=age_rasters["male_0-14"],
                male_15_64=age_rasters["male_15-64"],
                male_65_plus=age_rasters["male_65+"],
            )

        if not demographics_by_year:
            print(
                "\n⚠ No valid demographic raster sets found; nothing to process.\n"
            )
            self.logger.warning(
                "No DemographicRasters built; aborting population processing."
            )
            return

        targets = [
            s for s in self.state.snapshots
            if s.osm_gpkg is not None and s.osm_gpkg.exists()
        ]
        if not targets:
            warn("No saved OSM GeoPackage available; complete the Data Acquisition step first.")
            return

        for snap in targets:
            self._attach_population_to_snapshot(snap, demographics_by_year)

    def _attach_population_to_snapshot(
        self,
        snap: Snapshot,
        demographics_by_year: Dict[int, DemographicRasters],
    ) -> None:
        """Pick the right WorldPop year for *snap* and append demographics."""
        if snap.timestamp is None:
            # Single-mode: every year's rasters are available; pass them all.
            demographics_by_timestamp = {
                str(y): dem for y, dem in demographics_by_year.items()
            }
            year_label = ""
        else:
            try:
                year = datetime.strptime(snap.timestamp, "%Y%m%d").year
            except ValueError:
                self.logger.warning(
                    f"Invalid timestamp '{snap.timestamp}'; expected YYYYMMDD; skipping."
                )
                return
            if year not in demographics_by_year:
                self.logger.warning(
                    f"No WorldPop rasters for year {year} (timestamp {snap.timestamp}); skipping."
                )
                return
            demographics_by_timestamp = {snap.timestamp: demographics_by_year[year]}
            year_label = f" (year {year})"

        osm_gpkg_path = snap.osm_gpkg
        enriched_path = osm_gpkg_path.with_name(
            f"{osm_gpkg_path.stem}_demography{osm_gpkg_path.suffix}"
        )

        with stage(
            f"Population-to-street processing ({snap.label}){year_label}",
            logger=self.logger,
        ) as st:
            streets_with_pop = append_population_to_streets(
                merged_gpkg=osm_gpkg_path,
                demographics_by_timestamp=demographics_by_timestamp,
                output_path=enriched_path,
                logger=self.logger,
            )
            snap.streets_with_population = enriched_path
            if snap.timestamp is None:
                self.session_data["streets_with_population"] = streets_with_pop
                self.session_data["streets_with_population_path"] = enriched_path
            else:
                self.session_data.setdefault(
                    "streets_with_population_temporal", {}
                )[snap.timestamp] = streets_with_pop
                self.session_data.setdefault(
                    "streets_with_population_temporal_paths", {}
                )[snap.timestamp] = enriched_path
            st.note(f"Output: {enriched_path}")

    # ------------------------------------------------------------------
    # OSM data loading
    # ------------------------------------------------------------------

    def _load_osm_data(self) -> None:
        """
        Load OpenStreetMap data via the autoloader.  Active in ONLINE (mode 1)
        and PRE-CONFIGURED (mode 4) runs; other modes acquire data through
        their own paths.
        """
        if self.state.config.mode not in (1, 4):
            return

        cfg             = self.state.config
        city            = cfg.city
        use_coordinates = cfg.use_coordinates
        coordinates     = cfg.coordinates
        use_temporal    = cfg.use_temporal
        timestamps      = list(cfg.timestamps)

        print_section("Data Acquisition")

        output_dir = DATA_INPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        # ── Cache check: skip download if matching file(s) already exist ──
        if not use_temporal:
            if use_coordinates:
                filename = (
                    f"{datetime.now().strftime('%Y%m%d')}_"
                    f"{coordinates['long_min']}_"
                    f"{coordinates['lat_min']}.gpkg"
                )
            else:
                filename = (
                    f"{datetime.now().strftime('%Y%m%d')}_"
                    f"{city.replace(' ', '_').replace(',', '')}.gpkg"
                )
            cached_path = output_dir / filename
            if cached_path.exists():
                success(f"Using cached GeoPackage: {cached_path.name}")
                self.logger.info(
                    f"Cache hit: skipping OSM download, using {cached_path}"
                )
                self.state.ensure(None).osm_gpkg = cached_path
                self.state.osm_dir = output_dir
                self.session_data["osm_output_path"] = cached_path
                return
        else:
            # Temporal: skip timestamps whose files are already on disk
            city_safe = (
                "area"
                if use_coordinates
                else city.replace(" ", "_").replace(",", "")
            )
            cached_ts = {
                ts
                for ts in (timestamps or [])
                if (output_dir / f"{ts}_{city_safe}.gpkg").exists()
            }
            if cached_ts:
                for ts in sorted(cached_ts):
                    cached = output_dir / f"{ts}_{city_safe}.gpkg"
                    self.logger.info(f"Cache hit for {ts}: {cached}")
                    self.state.ensure(ts).osm_gpkg = cached
                typing(
                    f"\n  {len(cached_ts)} timestamp(s) already cached"
                    + (" — all done, skipping download.\n" if len(cached_ts) == len(timestamps or []) else ".\n")
                )
                timestamps = [ts for ts in (timestamps or []) if ts not in cached_ts]
                if not timestamps:
                    self.state.osm_dir = output_dir
                    self.session_data["osm_output_dir_temporal"] = output_dir
                    return
                typing(f"  Downloading {len(timestamps)} missing timestamp(s)...\n")

        typing("Initializing ANTIPOMPEII autoloader...")

        if use_temporal:
            typing(
                f"Downloading temporal data from OpenStreetMap for "
                f"{len(timestamps)} timestamp(s)...\n"
            )
        else:
            typing("Downloading current data from OpenStreetMap...\n")

        try:
            loader = DataLoader.from_cli_params(
                city=city,
                use_coordinates=use_coordinates,
                use_temporal=use_temporal,
                timestamps=timestamps,
                long_min=coordinates.get("long_min") if use_coordinates else None,
                lat_min=coordinates.get("lat_min") if use_coordinates else None,
                long_max=coordinates.get("long_max") if use_coordinates else None,
                lat_max=coordinates.get("lat_max") if use_coordinates else None,
                logger=self.logger,
                use_optimization=True,
            )

            data = loader.load_all_data()

            self.session_data["loader"] = loader

            if use_temporal:
                self.session_data["temporal_data"] = data
                success("Temporal data acquisition complete")

                print("\nTemporal Data Summary:")
                temporal_summary = loader.get_temporal_summary()
                print(temporal_summary.to_string(index=False))
                print()

                typing("\nPerformance Statistics:\n")
                perf_summary = loader.get_performance_summary()
                print(perf_summary.to_string(index=False))
                print()

                show_details = confirm(
                    "\nWould you like to see detailed layer breakdown for each timestamp?",
                    default=False,
                )
                if show_details and timestamps:
                    for timestamp in timestamps:
                        print(f"\n{'='*60}")
                        print(f"Layer Summary for {timestamp}:")
                        print("=" * 60)
                        layer_summary = loader.get_layer_summary(timestamp)
                        print(layer_summary.to_string(index=False))
                save_data = True
            else:
                self.session_data["osm_data"] = data
                success("Data acquisition complete")

                print("\nLayer Summary:")
                summary_df = loader.get_layer_summary()
                print(summary_df.to_string(index=False))
                print()

                typing("\nPerformance Statistics:\n")
                perf_summary = loader.get_performance_summary()
                print(perf_summary.to_string(index=False))
                print()

                save_data = True

            if save_data:
                if use_temporal:
                    typing(f"\nSaving temporal data to: {output_dir}\n")
                    loader.save_data(output_dir, format="gpkg")
                    success(f"All temporal snapshots saved to: {output_dir}")
                    city_safe = (
                        "area"
                        if use_coordinates
                        else city.replace(" ", "_").replace(",", "")
                    )
                    for ts in (timestamps or []):
                        self.state.ensure(ts).osm_gpkg = output_dir / f"{ts}_{city_safe}.gpkg"
                    self.state.osm_dir = output_dir
                    self.session_data["osm_output_dir_temporal"] = output_dir
                else:
                    output_path = output_dir / filename
                    loader.save_data(output_path, format="gpkg")
                    success(f"Data saved to: {output_path}")
                    self.state.ensure(None).osm_gpkg = output_path
                    self.state.osm_dir = output_dir
                    self.session_data["osm_output_path"] = output_path

            return data

        except Exception as e:
            self.logger.error(f"Data loading failed: {str(e)}")
            error(f"Error: {str(e)}")
            typing("Please check your internet connection and try again.\n")
            typing(
                "Note: Temporal data requires Overpass API support for attic data.\n"
            )
            raise

    # ------------------------------------------------------------------
    # Elevation DEM download
    # ------------------------------------------------------------------

    def _download_dem(self) -> None:
        """
        Interactively download a DEM GeoTIFF from OpenTopography for the
        city extent established in the session.  The file path is stored
        in session_data["elevation_dem_path"].
        """
        import os

        print_section("Elevation DEM download")

        if self.state.preconfigured:
            dcfg = self.state.preconfig.dem
            if not dcfg.download:
                success("Skipping DEM download (disabled in config).")
                self.logger.info("Mode 4: DEM download disabled in config.")
                return
            api_key = dcfg.api_key or os.environ.get("OPENTOPO_API_KEY", "")
            if not api_key:
                warn(
                    "No OpenTopography API key in config (`dem.api_key`) or in "
                    "$OPENTOPO_API_KEY; skipping DEM download."
                )
                self.logger.warning("Mode 4: DEM skipped (no API key).")
                return
            if dcfg.product not in DEM_PRODUCT_KEYS:
                error(
                    f"Unknown DEM product {dcfg.product!r}; valid choices: "
                    f"{', '.join(DEM_PRODUCT_KEYS)}"
                )
                self.logger.error(f"Mode 4: invalid DEM product {dcfg.product!r}.")
                return
            dem_type = dcfg.product
            product = DEM_PRODUCTS[dem_type]
            info(
                f"  DEM          : product={dem_type} "
                f"({product.resolution}, {product.source})"
            )
            self.logger.info(f"Mode 4: DEM type {dem_type}")
        else:
            if not confirm("Download a Digital Elevation Model from OpenTopography?", default=True):
                success("Skipping DEM download")
                self.logger.info("User opted to skip DEM download.")
                return

            # -- API key ---------------------------------------------------------
            api_key = ask_text(
                "Enter your OpenTopography API key\n"
                " (register free at https://portal.opentopography.org)",
            )
            if not api_key:
                error("No API key provided; skipping DEM download.")
                self.logger.warning("DEM download skipped: no API key entered.")
                return

            # -- DEM type --------------------------------------------------------
            print("\nAvailable DEM products:\n")
            for i, key in enumerate(DEM_PRODUCT_KEYS, start=1):
                p = DEM_PRODUCTS[key]
                note = f"  [{p.note}]" if p.note else ""
                print(f" {i:2d}. {key:<12}  {p.resolution:<8}  {p.source:<26}  {p.name}{note}")
            print()

            dem_index = ask_int(
                f"Select DEM type [1–{len(DEM_PRODUCT_KEYS)}]",
                default=1,
                bounds=(1, len(DEM_PRODUCT_KEYS)),
            )
            dem_type = DEM_PRODUCT_KEYS[dem_index - 1]
            product = DEM_PRODUCTS[dem_type]
            typing(
                f"\n✓ Selected: {dem_type} — {product.name} "
                f"({product.resolution}, {product.source})\n"
            )
            self.logger.info(f"DEM type selected: {dem_type}")

        # -- Resolve output directory ----------------------------------------
        output_dir = DATA_INPUT_DIR / "elevation"

        # -- Build downloader and run ----------------------------------------
        city = self.session_data.get("city")
        use_coordinates = self.session_data.get("use_coordinates", False)
        coordinates = self.session_data.get("coordinates", {})

        try:
            downloader = DEMDownloader.from_cli_params(
                dem_type=dem_type,
                api_key=api_key,
                output_dir=output_dir,
                city=city,
                use_coordinates=use_coordinates,
                long_min=coordinates.get("long_min") if use_coordinates else None,
                lat_min=coordinates.get("lat_min") if use_coordinates else None,
                long_max=coordinates.get("long_max") if use_coordinates else None,
                lat_max=coordinates.get("lat_max") if use_coordinates else None,
                logger=self.logger,
            )
        except (ValueError, RuntimeError) as e:
            error(f"Could not initialise DEM downloader: {e}")
            self.logger.error(f"DEM downloader init failed: {e}")
            return

        typing(
            f"\nDownloading {dem_type} for extent: {downloader.bbox}\n"
            f"Output directory: {output_dir}\n"
        )

        try:
            dem_path = downloader.download()
        except RuntimeError as e:
            error(f"DEM download failed: {e}")
            self.logger.error(f"DEM download failed: {e}", exc_info=True)
            return

        size_mb = dem_path.stat().st_size / 1_048_576
        print(
            f"\n✓ DEM download complete.\n"
            f"  File  : {dem_path.name}\n"
            f"  Size  : {size_mb:.1f} MiB\n"
            f"  Path  : {dem_path}\n"
        )
        self.state.elevation_dem = dem_path
        self.session_data["elevation_dem_path"] = dem_path
        self.logger.info(f"DEM saved to session: {dem_path}")

    # ------------------------------------------------------------------
    # DEM → streets (elevation processing)
    # ------------------------------------------------------------------

    def _process_dem_data(self) -> None:
        """
        Sample the DEM along every street segment of every snapshot and append
        ``elev_min`` to the enriched GeoPackage.  Skipped silently when no DEM
        was downloaded earlier.
        """
        dem_path = self.state.elevation_dem
        if dem_path is None:
            return

        print_section("DEM elevation processing")

        # Each snapshot needs an input GPKG — anything from the enrichment chain
        # that exists on disk.
        targets = [
            (s, s.enriched_gpkg) for s in self.state.snapshots
            if s.enriched_gpkg is not None
        ]
        if not targets:
            warn("No enriched GeoPackage found; skipping DEM elevation processing.")
            return

        if self.state.preconfigured:
            n_samples = self.state.preconfig.dem.n_samples
            info(f"  DEM samples  : n_samples={n_samples}")
        else:
            n_samples = ask_int(
                "Number of DEM sample points per street segment\n"
                " (higher = more accurate, slower; recommended: 5–15)",
                default=5,
                bounds=(2, 50),
            )

        elevation_paths: Dict[str, Path] = {}
        for snap, input_gpkg in targets:
            output_path = input_gpkg.with_name(
                f"{input_gpkg.stem}_elevation{input_gpkg.suffix}"
            )

            with stage(f"DEM elevation ({snap.label})", logger=self.logger) as st:
                gdf = append_elevation_to_streets(
                    streets_gpkg=input_gpkg,
                    dem_path=Path(dem_path),
                    output_path=output_path,
                    n_samples=n_samples,
                    logger=self.logger,
                )
                streets = gdf[gdf["layer_name"] == "Street Network"]
                covered = streets["elev_min"].notna().sum()
                st.note(
                    f"{covered:,}/{len(streets):,} segments enriched, "
                    f"range {streets['elev_min'].min():.1f}–{streets['elev_min'].max():.1f} m"
                )
                st.note(f"Output: {output_path}")
                snap.streets_with_elevation = output_path
                if snap.timestamp is None:
                    self.session_data["streets_with_elevation_path"] = output_path
                else:
                    elevation_paths[snap.timestamp] = output_path

        if self.state.is_temporal and elevation_paths:
            self.session_data["streets_with_elevation_temporal_paths"] = elevation_paths
            self.logger.info(f"Temporal elevation paths stored: {list(elevation_paths.keys())}")

    # ------------------------------------------------------------------
    # Water-feature download
    # ------------------------------------------------------------------

    def _download_water(self) -> None:
        """
        Download OSM water features (rivers, lakes, wetlands, coastline …)
        for the city extent.  The result is a single GeoPackage under
        ``src/data/input/water/``; the file path is stored on the state.

        Water bodies are quasi-static, so a single download is shared across
        all temporal snapshots in a run.
        """
        print_section("Water-feature download")

        if self.state.preconfigured:
            wcfg = self.state.preconfig.water
            if not wcfg.download:
                success("Skipping water-feature download (disabled in config).")
                self.logger.info("Mode 4: water download disabled.")
                return
            include_wetlands     = wcfg.include_wetlands
            include_coastline    = wcfg.include_coastline
            include_intermittent = wcfg.include_intermittent
            info(
                f"  Water query  : wetlands={include_wetlands}, "
                f"coastline={include_coastline}, intermittent={include_intermittent}"
            )
        else:
            if not confirm(
                "Download OSM water features (rivers, lakes, coastline …)?",
                default=True,
            ):
                success("Skipping water-feature download")
                self.logger.info("User opted to skip water download.")
                return
            include_wetlands     = confirm(
                "Include wetlands (natural=wetland)? "
                "(noisy in alluvial regions; disable if results look flat)",
                default=True,
            )
            include_coastline    = confirm(
                "Include coastline (natural=coastline)?",
                default=True,
            )
            include_intermittent = confirm(
                "Include intermittent streams?",
                default=False,
            )

        city            = self.session_data.get("city")
        use_coordinates = self.session_data.get("use_coordinates", False)
        coordinates     = self.session_data.get("coordinates", {})
        output_dir      = DATA_INPUT_DIR / "water"

        try:
            downloader = WaterDownloader.from_cli_params(
                output_dir=output_dir,
                city=city,
                use_coordinates=use_coordinates,
                long_min=coordinates.get("long_min") if use_coordinates else None,
                lat_min=coordinates.get("lat_min")   if use_coordinates else None,
                long_max=coordinates.get("long_max") if use_coordinates else None,
                lat_max=coordinates.get("lat_max")   if use_coordinates else None,
                include_wetlands=include_wetlands,
                include_coastline=include_coastline,
                include_intermittent=include_intermittent,
                logger=self.logger,
            )
        except ValueError as exc:
            error(f"Could not initialise water downloader: {exc}")
            self.logger.error(f"Water downloader init failed: {exc}")
            return

        try:
            result = downloader.download()
        except RuntimeError as exc:
            error(f"Water download failed: {exc}")
            self.logger.error(f"Water download failed: {exc}", exc_info=True)
            return

        cache_note = " (cached)" if result.cached else ""
        success(
            f"Water layer{cache_note}: {result.n_features:,} features "
            f"({result.n_polygons:,} polygons, {result.n_lines:,} lines, "
            f"{result.n_points:,} points)"
        )
        self.state.water_layer = result.path
        self.session_data["water_layer_path"] = result.path

    # ------------------------------------------------------------------
    # Water → streets (distance covariate)
    # ------------------------------------------------------------------

    def _process_water_data(self) -> None:
        """
        Compute ``water_dist_min`` (meters) per street segment and append it
        to each snapshot's enriched GeoPackage.  Skipped silently when no
        water layer was downloaded earlier.
        """
        water_path = self.state.water_layer
        if water_path is None or not Path(water_path).exists():
            return

        print_section("Water distance processing")

        targets = [
            (s, s.enriched_gpkg) for s in self.state.snapshots
            if s.enriched_gpkg is not None
        ]
        if not targets:
            warn("No enriched GeoPackage found; skipping water-distance processing.")
            return

        water_paths: Dict[str, Path] = {}
        for snap, input_gpkg in targets:
            output_path = input_gpkg.with_name(
                f"{input_gpkg.stem}_water{input_gpkg.suffix}"
            )

            with stage(f"Water distance ({snap.label})", logger=self.logger) as st:
                gdf = append_water_distance_to_streets(
                    streets_gpkg=input_gpkg,
                    water_gpkg=water_path,
                    output_path=output_path,
                    logger=self.logger,
                )
                streets = gdf[gdf["layer_name"] == "Street Network"]
                finite  = streets["water_dist_min"].dropna()
                covered = int(len(finite))
                if covered:
                    st.note(
                        f"{covered:,}/{len(streets):,} segments enriched, "
                        f"range {finite.min():.1f}–{finite.max():.1f} m"
                    )
                else:
                    st.note(f"{covered:,}/{len(streets):,} segments enriched")
                st.note(f"Output: {output_path}")
                snap.streets_with_water = output_path
                if snap.timestamp is None:
                    self.session_data["streets_with_water_path"] = output_path
                else:
                    water_paths[snap.timestamp] = output_path

        if self.state.is_temporal and water_paths:
            self.session_data["streets_with_water_temporal_paths"] = water_paths
            self.logger.info(f"Temporal water paths stored: {list(water_paths.keys())}")

    # ------------------------------------------------------------------
    # Disruption → streets
    # ------------------------------------------------------------------

    def _resolve_disruption_path(
        self, dem_path: Path, timestamp: Optional[str]
    ) -> Optional[Path]:
        """
        Resolve the disruption GeoPackage to use for *dem_path*.

        Tries timestamp-based auto-match first. On miss, prompts the user
        with a numbered list of available ``disruption_*.gpkg`` files in
        ``src/data/input/disruption/`` and offers a custom path or skip.
        Returns the chosen ``Path``, or ``None`` if the user chose to skip
        disruption for this snapshot.
        """
        if timestamp is None:
            m = _re.search(r"(\d{8})", dem_path.stem)
            timestamp = m.group(1) if m else None

        if timestamp is not None:
            auto = find_disruption_file(timestamp)
            if auto is not None:
                return auto

        input_dir = DATA_INPUT_DIR / "disruption"
        available = sorted(input_dir.glob("disruption_*.gpkg")) if input_dir.exists() else []

        snap_label = timestamp or dem_path.stem

        # Mode 4: never prompt.  Either skip silently (default) or raise.
        if self.state.preconfigured:
            if self.state.preconfig.disruption.skip_if_missing:
                warn(
                    f"No disruption file for {snap_label}; skipping "
                    "(disruption.skip_if_missing=true)."
                )
                self.logger.info(
                    f"Mode 4: no disruption file for {snap_label}; skipping."
                )
                return None
            raise RuntimeError(
                f"No disruption file for {snap_label} and "
                "`disruption.skip_if_missing` is false."
            )

        typing(
            f"\n  No disruption file matches timestamp {timestamp or '?'} "
            f"(looked for disruption_{timestamp or 'YYYYMMDD'}.gpkg "
            f"in {input_dir}).\n"
        )

        # Build menu: N files + path + skip
        n = len(available)
        if n:
            typing("  Available disruption files:\n")
            for i, p in enumerate(available, start=1):
                print(f"    {i}. {p.name}\n")
            typing(f"    {n + 1}. Enter a custom path\n")
            typing(f"    {n + 2}. Skip disruption for this snapshot\n")
            choice = ask_int(
                f"\n  Choose 1–{n + 2} for {snap_label}",
                default=n + 2,
                bounds=(1, n + 1),
            )
            if 1 <= choice <= n:
                return available[choice - 1]
            if choice == n + 1:
                return self._prompt_custom_disruption_path(snap_label)
            return None

        # No files at all: only custom path or skip
        typing("  No disruption_*.gpkg files found in the input directory.\n")
        typing("    1. Enter a custom path\n")
        typing("    2. Skip disruption for this snapshot\n")
        choice = ask_int(
            f"\n  Choose 1–2 for {snap_label}",
            default=2,
            bounds=(1, 2),
        )
        if choice == 1:
            return self._prompt_custom_disruption_path(snap_label)
        return None

    def _prompt_custom_disruption_path(self, snap_label: str) -> Optional[Path]:
        """Ask for a custom disruption file path; re-prompt until valid or skipped."""
        while True:
            raw = ask_text(
                f"  Path to disruption GeoPackage for {snap_label} "
                f"(empty to skip)",
                default="",
            )
            if not raw:
                return None
            candidate = Path(raw).expanduser()
            if candidate.exists():
                return candidate
            typing(f"  ⚠ Path does not exist: {candidate}\n")

    def _process_disruption_data(self) -> None:
        """
        Append disruption attributes from disruption_{timestamp}.gpkg
        to demography-enriched street networks.
        """
        # Idempotency: skip when every snapshot already has disruption
        # attached, or when a built graph is already present (disruption is
        # baked into the .gt artifact).  Lets Mode 2 cases that arrived with
        # a `_disruption.gpkg` or `_network.gt` pass through without
        # re-prompting the user.
        if self.state.snapshots and all(
            s.streets_with_disruption is not None or s.best_graph is not None
            for s in self.state.snapshots
        ):
            info("Disruption already present for every snapshot; skipping stage.")
            return

        print_section("Disruption data")

        use_temporal = self.session_data.get("use_temporal", False)

        if not use_temporal:
            dem_path = self.session_data.get("streets_with_population_path")
            if dem_path is None:
                typing(
                    "\n⚠ No demography-enriched street GeoPackage found; "
                    "skip disruption processing.\n"
                )
                self.logger.warning(
                    "streets_with_population_path not set; cannot append disruption."
                )
                return

            dem_path = Path(dem_path)
            if not dem_path.exists():
                typing(
                    "\n⚠ Demography-enriched GeoPackage path does not exist on disk; "
                    "skip disruption processing.\n"
                )
                self.logger.warning(f"Demography GPKG not found: {dem_path}")
                return

            resolved = self._resolve_disruption_path(dem_path, timestamp=None)
            if resolved is None:
                typing("\n  Disruption stage skipped for this snapshot.\n")
                self.logger.info(
                    "Disruption stage skipped: no disruption file selected."
                )
                return

            disrupted_path = dem_path.with_name(
                f"{dem_path.stem}_disruption{dem_path.suffix}"
            )

            with stage("Disruption processing", logger=self.logger) as st:
                self.session_data["streets_with_disruption"] = (
                    append_disruption_to_streets(
                        demography_gpkg=dem_path,
                        disruption_path=resolved,
                        output_path=disrupted_path,
                        logger=self.logger,
                    )
                )
                # In non-temporal mode there's exactly one snapshot.  Target
                # it directly instead of ensure(None) so that Mode 2 (where
                # the snapshot's timestamp is the case date, not None) does
                # not spawn a phantom second snapshot.
                target_snap = (
                    self.state.snapshots[0]
                    if self.state.snapshots
                    else self.state.ensure(None)
                )
                target_snap.streets_with_disruption = disrupted_path
                self.session_data["streets_with_disruption_path"] = disrupted_path
                st.note(f"Output: {disrupted_path}")
            return

        # Temporal mode
        temporal_paths = self.session_data.get(
            "streets_with_population_temporal_paths"
        )
        timestamps = self.session_data.get("timestamps") or []

        if not temporal_paths or not timestamps:
            typing(
                "\n⚠ No temporal demography outputs recorded; "
                "skip disruption processing.\n"
            )
            self.logger.warning(
                "streets_with_population_temporal_paths or timestamps missing; "
                "cannot append disruption for temporal snapshots."
            )
            return

        typing("\nAppending disruption attributes for temporal snapshots...\n")
        disrupted_results: Dict[str, Any] = {}
        disrupted_paths: Dict[str, Path] = {}

        for ts in timestamps:
            dem_path = temporal_paths.get(ts)
            if dem_path is None:
                self.logger.warning(
                    f"No demography-enriched path recorded for timestamp {ts}; "
                    "skipping."
                )
                continue

            dem_path = Path(dem_path)
            if not dem_path.exists():
                self.logger.warning(
                    "Demography GPKG for timestamp {ts} not found on disk: "
                    f"{dem_path}; skipping."
                )
                continue

            resolved = self._resolve_disruption_path(dem_path, timestamp=ts)
            if resolved is None:
                typing(f"\n  Disruption stage skipped for snapshot {ts}.\n")
                self.logger.info(
                    f"Disruption stage skipped for {ts}: no file selected."
                )
                continue

            disrupted_path = dem_path.with_name(
                f"{dem_path.stem}_disruption{dem_path.suffix}"
            )

            with stage(f"Disruption processing for {ts}", logger=self.logger) as st:
                disrupted_results[ts] = append_disruption_to_streets(
                    demography_gpkg=dem_path,
                    disruption_path=resolved,
                    output_path=disrupted_path,
                    logger=self.logger,
                )
                self.state.ensure(ts).streets_with_disruption = disrupted_path
                disrupted_paths[ts] = disrupted_path
                st.note(f"Output: {disrupted_path}")

        if disrupted_results:
            self.session_data["streets_with_disruption_temporal"] = disrupted_results
            self.session_data["streets_with_disruption_temporal_paths"] = (
                disrupted_paths
            )

    # ------------------------------------------------------------------
    # Graph builder
    # ------------------------------------------------------------------

    def _find_enriched_gpkg(self) -> Optional[Path]:
        """Most-enriched GPKG so far (non-temporal path, used by Mode 5)."""
        for snap in self.state.snapshots:
            if snap.enriched_gpkg is not None:
                return snap.enriched_gpkg
        return None

    def _build_graph_network(self):
        """Build a graph-tool network for every snapshot that has an enriched GPKG."""
        print_section("Graph Network Builder")

        # Idempotency: any snapshot that already has a built or simplified
        # graph on disk needs no work here.
        targets = [
            s for s in self.state.snapshots
            if s.enriched_gpkg is not None and s.best_graph is None
        ]
        if not targets:
            if any(s.best_graph is not None for s in self.state.snapshots):
                info("Graph already present for every snapshot; skipping build.")
                return
            warn("No enriched GeoPackage available; complete the earlier pipeline steps first.")
            return

        if self.state.preconfigured:
            gcfg = self.state.preconfig.graph
            if not gcfg.build:
                success("Skipping graph network construction (disabled in config).")
                self.logger.info("Mode 4: graph build disabled in config.")
                return
            directed = gcfg.directed
            tolerance = gcfg.tolerance
            info(
                f"  Graph build  : directed={directed}, tolerance={tolerance}"
            )
        else:
            if not confirm("Build graph-tool network from enriched streets?", default=True):
                success("Skipping graph network construction")
                self.logger.info("User opted to skip graph building.")
                return

            directed = confirm("Create directed graph (for one-way streets)?", default=False)
            tolerance = ask_float(
                "Vertex deduplication tolerance (degrees, e.g. 1e-8 ≈ 1cm)",
                default=1e-8,
            )

        built: List[Path] = []
        for snap in targets:
            gpkg_path  = snap.enriched_gpkg
            graph_path = gpkg_path.parent / (gpkg_path.stem + "_network.gt")

            with stage(f"Graph build ({snap.label})", logger=self.logger) as st:
                graph = build_graph_from_streets(
                    enriched_gpkg=gpkg_path,
                    output_path=graph_path,
                    tolerance=tolerance,
                    directed=directed,
                    logger=self.logger,
                )
                snap.graph = graph_path
                built.append(graph_path)
                st.note(
                    f"Saved {graph_path.name} "
                    f"({graph.num_vertices():,} vertices, {graph.num_edges():,} edges)"
                )

        # Keep session_data in sync for downstream consumers (analytics, IA).
        if self.state.is_temporal:
            self.session_data["graph_paths_temporal"] = built
        elif built:
            self.session_data["graph_path"] = built[0]

    # ------------------------------------------------------------------
    # Network simplifier
    # ------------------------------------------------------------------

    def _simplify_network(self):
        """Simplify each snapshot's graph (remove parallel edges and degree-2 nodes)."""
        print_section("Network Simplification")

        # Idempotency: only snapshots with a built graph but no simplified
        # version need work.
        targets = [
            s for s in self.state.snapshots
            if s.graph is not None and s.graph.exists()
               and s.simplified_graph is None
        ]
        if not targets:
            if any(s.simplified_graph is not None for s in self.state.snapshots):
                info("Simplified graph already present for every snapshot; skipping.")
                return
            warn("No graph network found; build the graph first (step 12).")
            return

        if self.state.preconfigured:
            gcfg = self.state.preconfig.graph
            if not gcfg.simplify:
                success("Skipping network simplification (disabled in config).")
                self.logger.info("Mode 4: simplification disabled in config.")
                return
            run_diagnostics = gcfg.run_diagnostics
            info(f"  Simplify     : run_diagnostics={run_diagnostics}")
        else:
            if not confirm("Simplify graph network (reduce complexity)?", default=True):
                success("Skipping network simplification")
                self.logger.info("User opted to skip network simplification.")
                return

            run_diagnostics = confirm(
                "Run network diagnostics (before/after comparison)?",
                default=True,
            )

        simplified: List[Path] = []
        for snap in targets:
            graph_path      = snap.graph
            simplified_path = graph_path.parent / (graph_path.stem + "_simplified.gt")

            with stage(f"Simplify {graph_path.name}", logger=self.logger) as st:
                simplified_graph = simplify_network(
                    graph_path=graph_path,
                    output_path=simplified_path,
                    run_diagnostics=run_diagnostics,
                    logger=self.logger,
                )
                snap.simplified_graph = simplified_path
                simplified.append(simplified_path)
                st.note(
                    f"Saved {simplified_path.name} "
                    f"({simplified_graph.num_vertices():,} vertices, "
                    f"{simplified_graph.num_edges():,} edges)"
                )

        if self.state.is_temporal:
            self.session_data["simplified_graph_paths_temporal"] = simplified
        elif simplified:
            self.session_data["simplified_graph_path"] = simplified[0]

    # ------------------------------------------------------------------
    # Network disruption analysis
    # ------------------------------------------------------------------

    # ── helper: assemble graph paths + labels for multi-event analytics ──

    def _graphs_for_analysis(
        self,
    ) -> Tuple[Optional[Path], Optional[List[Path]], List[str]]:
        """
        Walk ``state.snapshots`` and return:

        * the first existing ``.gt`` graph (the base),
        * any further graphs (``additional`` — ``None`` if only one),
        * a parallel list of labels (timestamps or fallback names).

        Picks the simplified graph when available, raw graph otherwise.
        """
        usable = [
            (s.best_graph, s.label)
            for s in self.state.snapshots
            if s.best_graph is not None
        ]
        if not usable:
            return None, None, []

        paths  = [p for p, _ in usable]
        labels = [lbl for _, lbl in usable]
        base   = paths[0]
        extra  = paths[1:] if len(paths) > 1 else None
        return base, extra, labels

    def _run_network_analysis(self) -> None:
        """Run network disruption analysis across all built graphs."""
        print_section("Network Disruption Analysis")

        if self.state.preconfigured:
            scfg = self.state.preconfig.analysis.stats
            if not scfg.run:
                success("Skipping network analysis (disabled in config).")
                self.logger.info("Mode 4: stats analysis disabled in config.")
                return
            preconf_compound = scfg.compound
        else:
            if not confirm("Run network disruption analysis?", default=True):
                success("Skipping network analysis")
                self.logger.info("User opted to skip network analysis.")
                return
            preconf_compound = None

        base_path, additional, labels = self._graphs_for_analysis()
        if base_path is None:
            warn("No graph network found; build the graph first (step 12).")
            return

        compound = True
        if additional:
            if preconf_compound is not None:
                compound = preconf_compound
                success(
                    f"Multi-event mode: "
                    f"{'compound (cascading)' if compound else 'independent'} (from config)"
                )
            else:
                mode_choice = ask_choice(
                    COMBINE_EVENTS_STATS.format(n=len(additional) + 1),
                    options=["compound", "independent"],
                    default="compound",
                )
                compound = mode_choice == "compound"
                success(f"Multi-event mode: {'compound (cascading)' if compound else 'independent'}")

        output_dir = self._case_output_dir()

        with stage("Network disruption analysis", logger=self.logger) as st:
            self.session_data["analysis_results"] = analyse_network(
                graph_path=base_path,
                output_dir=output_dir,
                additional_graph_paths=additional,
                scenario_labels=labels if additional else None,
                compound=compound,
                logger=self.logger,
            )
            self.state.analysis_results = self.session_data["analysis_results"]
            st.note(f"Outputs saved to: {output_dir}")

    # ------------------------------------------------------------------
    # Network robustness estimation
    # ------------------------------------------------------------------

    def _run_robustness_estimation(self) -> None:
        """
        Estimate structural robustness metrics across intact, disrupted, and
        (optionally) recovered network states.
        """
        print_section("Network Robustness Estimation")

        if self.state.preconfigured:
            rcfg = self.state.preconfig.analysis.robustness
            if not rcfg.run:
                success("Skipping robustness estimation (disabled in config).")
                self.logger.info("Mode 4: robustness disabled in config.")
                return
            weight_prop = "length" if rcfg.weight == "length" else ""
            preconf_compound = self.state.preconfig.analysis.stats.compound
            info(f"  Robustness   : weight={rcfg.weight or 'hops'}")
        else:
            if not confirm("Run network robustness estimation?", default=True):
                success("Skipping robustness estimation")
                self.logger.info("User opted to skip robustness estimation.")
                return

            # Weight preference
            weight_choice = ask_choice(
                WEIGHT_PROMPT,
                options=["length", "hops"],
                default="length",
            )
            weight_prop = "length" if weight_choice == "length" else ""
            preconf_compound = None

        # Recovery state — use value from temporal selection if set,
        # otherwise ask (Mode 1 only).
        recovery_path  = self.session_data.get("recovery_graph_path")
        recovery_label = self.session_data.get("recovery_label", "Recovery")

        if recovery_path is None and not self.state.preconfigured:
            rec_input = get_user_input(
                "Path to recovery / future-state .gt graph file "
                "(press Enter to skip)",
                default="",
            ).strip()
            if rec_input:
                recovery_path = Path(rec_input)
                if not recovery_path.exists():
                    print(
                        f"\n⚠ Recovery graph not found: {recovery_path}; skipping.\n"
                    )
                    recovery_path = None
                else:
                    rec_label_raw = get_user_input(
                        "Label for recovery state (e.g. '2024')",
                        default="Recovery",
                    ).strip()
                    recovery_label = rec_label_raw or "Recovery"

        base_path, additional, labels = self._graphs_for_analysis()
        if base_path is None:
            warn("No graph network found; build the graph first (step 12).")
            return

        compound = True
        if additional:
            if preconf_compound is not None:
                compound = preconf_compound
                success(
                    f"Mode: {'compound (cascading)' if compound else 'independent'} (from config)"
                )
            else:
                mode = ask_choice(
                    COMBINE_EVENTS_ROBUSTNESS.format(n=len(additional) + 1),
                    options=["compound", "independent"],
                    default="compound",
                )
                compound = mode == "compound"
                success(f"Mode: {'compound (cascading)' if compound else 'independent'}")
        elif not labels:
            labels = ["Disrupted"]

        output_dir = self._case_output_dir()

        with stage("Robustness estimation", logger=self.logger) as st:
            self.session_data["robustness_report"] = estimate_robustness(
                graph_path=base_path,
                output_dir=output_dir,
                additional_graph_paths=additional,
                recovery_path=recovery_path,
                recovery_label=recovery_label,
                scenario_labels=labels,
                compound=compound,
                weight_prop=weight_prop,
                logger=self.logger,
            )
            self.state.robustness_report = self.session_data["robustness_report"]
            st.note(f"Outputs in: {output_dir / 'robustness'}")

    # ------------------------------------------------------------------
    # Percolation analysis
    # ------------------------------------------------------------------

    def _run_percolation(self) -> None:
        """
        Interactive percolation analysis: betweenness attack, random failure,
        and/or elevation-based removal.
        """
        print_section("Percolation Analysis")

        if self.state.preconfigured:
            pcfg = self.state.preconfig.analysis.percolation
            if not pcfg.run:
                success("Skipping percolation analysis (disabled in config).")
                self.logger.info("Mode 4: percolation disabled in config.")
                return

        elif not confirm("Run percolation analysis?", default=True):
            success("Skipping percolation analysis")
            self.logger.info("User opted to skip percolation analysis.")
            return


        # Resolve graph path (base, non-simplified)
        use_temporal   = self.session_data.get("use_temporal", False)
        graph_path_raw = (
            self.session_data.get("graph_paths_temporal", [None])[0]
            if use_temporal
            else self.session_data.get("graph_path")
        )
        if graph_path_raw is None:
            warn("No graph network found; build the graph first (step 12).")
            return
        graph_path = Path(graph_path_raw)
        if not graph_path.exists():
            warn(f"Graph file not found: {graph_path}")
            return

        output_dir = self._case_output_dir()

        if self.state.preconfigured:
            pcfg = self.state.preconfig.analysis.percolation
            chosen = {int(s) for s in pcfg.scenarios if s in (1, 2, 3)}
            if not chosen:
                warn("No valid percolation scenarios in config. Skipping.")
                return
            run_bw  = 1 in chosen
            run_rnd = 2 in chosen
            run_elv = 3 in chosen
            n_steps         = pcfg.n_steps
            recompute_every = pcfg.recompute_every
            run_null        = pcfg.run_null
            null_m          = pcfg.null_m
            n_random_runs   = pcfg.n_random_runs
            elev_ascending  = pcfg.elev_direction == "flood"
            info(
                f"  Percolation  : scenarios={sorted(chosen)}, "
                f"recompute_every={recompute_every}, n_random_runs={n_random_runs}"
            )
        else:
            # ── Scenario selection ─────────────────────────────────────────
            print(
                "\n  Available percolation scenarios:\n"
                "   1. Betweenness centrality attack\n"
                "   2. Random failure\n"
                "   3. Elevation-based removal\n"
            )
            scenarios_raw = ask_text(
                "Which scenarios to run? Enter numbers separated by commas\n"
                " (e.g. '1,2' or '1,2,3' for all)",
                default="1,2,3",
            )
            chosen = {
                int(s.strip())
                for s in scenarios_raw.split(",")
                if s.strip().isdigit() and int(s.strip()) in (1, 2, 3)
            }
            if not chosen:
                warn("No valid scenario numbers entered. Skipping.")
                return

            run_bw  = 1 in chosen
            run_rnd = 2 in chosen
            run_elv = 3 in chosen

            # ── Betweenness options ────────────────────────────────────────
            recompute_every = 1
            run_null        = False
            null_m          = 5
            n_steps         = None

            if run_bw:
                n_steps_raw = ask_text(
                    "Number of edge-removal steps for betweenness attack\n"
                    " (leave blank for auto: 30% of edges, max 500)",
                    default="auto",
                ).lower()
                if n_steps_raw not in ("", "auto"):
                    try:
                        n_steps = int(n_steps_raw)
                    except ValueError:
                        pass

                recompute_every = ask_int(
                    "Recompute betweenness centrality every how many steps?\n"
                    " (1 = fully dynamic; larger K is faster for big networks)",
                    default=1,
                    bounds=(1, 10000),
                )

                run_null = confirm(
                    "Generate SRGG null model for comparison?\n"
                    " (skipped automatically if network > 1 500 vertices)",
                    default=False,
                )
                if run_null:
                    null_m = ask_int(
                        "SRGG ensemble size M",
                        default=5,
                        bounds=(1, 50),
                    )

            # ── Random failure options ─────────────────────────────────────
            n_random_runs = 10
            if run_rnd:
                n_random_runs = ask_int(
                    "Number of random-failure runs (more = smoother curve)",
                    default=10,
                    bounds=(1, 200),
                )

            # ── Elevation options ──────────────────────────────────────────
            elev_ascending = True
            if run_elv:
                direction = ask_choice(
                    ELEV_REMOVAL_FLOOD_CLOSURE,
                    options=["flood", "closure"],
                    default="flood",
                )
                elev_ascending = direction == "flood"

        # ── Run ────────────────────────────────────────────────────────
        typing(
            f"\n→ Running percolation analysis "
            f"(scenarios: {', '.join(sorted(str(s) for s in chosen))})…\n"
        )
        self.logger.info(
            f"Percolation: scenarios={chosen}, n_steps={n_steps}, "
            f"recompute_every={recompute_every}, null={run_null}"
        )

        with stage("Percolation analysis", logger=self.logger) as st:
            results = run_percolation(
                graph_path=graph_path,
                output_dir=output_dir,
                run_betweenness=run_bw,
                run_random=run_rnd,
                run_elevation=run_elv,
                n_steps=n_steps,
                recompute_every=recompute_every,
                run_null_model=run_null,
                null_m=null_m,
                n_random_runs=n_random_runs,
                elevation_ascending=elev_ascending,
                logger=self.logger,
            )
            self.session_data["percolation_results"] = results
            st.note(f"{len(results)} scenario(s). Outputs in: {output_dir / 'percolation'}")


    # ------------------------------------------------------------------
    # Vulnerability simulation
    # ------------------------------------------------------------------

    def _run_vulnerability_simulation(self) -> None:
        """
        EB-centered joint logistic vulnerability estimation.
        Auto-configures from graph properties; exports CSV + LaTeX reports
        and a vulnerability map PNG.
        """
        print_section("Vulnerability Simulation")

        if self.state.preconfigured:
            vcfg = self.state.preconfig.analysis.vulnerability
            if not vcfg.run:
                success("Skipping vulnerability simulation (disabled in config).")
                self.logger.info("Mode 4: vulnerability disabled in config.")
                return
        elif not confirm("Run vulnerability simulation?", default=True):
            success("Skipping vulnerability simulation")
            self.logger.info("User opted to skip vulnerability simulation.")
            return

        use_temporal = self.session_data.get("use_temporal", False)

        # Resolve base graph path
        if use_temporal:
            graph_paths = self.session_data.get("graph_paths_temporal", [])
            base_graph = graph_paths[0] if graph_paths else None
            additional = graph_paths[1:] if len(graph_paths) > 1 else []
        else:
            base_graph = self.session_data.get("graph_path")
            additional = []

        if base_graph is None:
            warn("No graph network found; build the graph first (step 12).")
            return
        base_path = Path(base_graph)
        if not base_path.exists():
            warn(f"Graph file not found: {base_path}")
            return

        output_dir = self._case_output_dir()

        if self.state.preconfigured:
            vcfg = self.state.preconfig.analysis.vulnerability
            n_sim          = vcfg.n_sim
            use_elevation  = vcfg.use_elevation
            use_water      = vcfg.use_water
            run_service_mc = vcfg.run_service_mc
            info(
                f"  Vulnerability: n_sim={n_sim}, use_elevation={use_elevation}, "
                f"use_water={use_water}, run_service_mc={run_service_mc}"
            )
        else:
            # ── Monte Carlo simulation count ─────────────────────────────────
            n_sim = ask_int(
                "Monte Carlo simulations for service-accessibility estimation\n"
                " (higher = more accurate; 200 recommended)",
                default=200,
                bounds=(10, 5000),
            )

            # ── Elevation covariate ──────────────────────────────────────────
            use_elevation = confirm(
                "Include elevation (elev_min) as adjustment covariate if available?",
                default=True,
            )

            # ── Distance-to-water covariate ──────────────────────────────────
            use_water = confirm(
                "Include distance-to-water (water_dist_min) as adjustment covariate if available?",
                default=True,
            )

            # ── Service-accessibility MC ─────────────────────────────────────
            run_service_mc = confirm(
                "Run service-accessibility Monte Carlo?\n"
                " (recommended; may take a few minutes for large networks)",
                default=True,
            )

        # ── Run ─────────────────────────────────────────────────────────
        event_desc = (
            f"{len(graph_paths)} temporal event(s)" if use_temporal else "single event"
        )
        step(f"Running vulnerability simulation ({event_desc}, n_sim={n_sim})…")
        self.logger.info(
            f"Vulnerability simulation: base={base_path.name}, "
            f"additional={[Path(p).name for p in additional]}, "
            f"n_sim={n_sim}, use_elevation={use_elevation}, use_water={use_water}, "
            f"run_service_mc={run_service_mc}"
        )

        with stage("Vulnerability simulation", logger=self.logger) as st:
            self.session_data["vulnerability_result"] = run_vulnerability_simulation(
                graph_path=base_path,
                output_dir=output_dir,
                additional_graphs=[Path(p) for p in additional] if additional else None,
                n_sim=n_sim,
                use_elevation=use_elevation,
                use_water=use_water,
                run_service_mc=run_service_mc,
                logger=self.logger,
            )
            st.note(f"Outputs in: {output_dir / 'vulnerability'}")


    # ==================================================================
    # Mode 5 — Existing Graph
    # ==================================================================

    def _run_existing_graph_mode(self) -> None:
        """
        Mode 5: load one or more existing .gt graph files and run the full
        analytical suite without re-running the data-acquisition pipeline.

        Flow
        ----
        1. User selects one or more .gt graph files.
        2. Auto-labels are shown; user may rename.
        3. User chooses which analytics modules to run.
        4. Analytics parameters are gathered once (shared across all graphs).
        5. Each graph is analyzed in turn; outputs go to per-graph sub-dirs
           inside a shared comparison root.
        6. Comparative CSV + LaTeX tables are generated (multiple graphs only).
        7. LLM interpretation runs per-graph and cross-graph.
        """
        print_section("Existing Graph Mode")
        typing(
            "Load one or more existing .gt graph files and run analytics:\n"
            "network disruption analysis, robustness estimation, percolation,\n"
            "vulnerability simulation, and LLM-powered interpretation.\n"
        )

        # 1 ── Select graph files ──────────────────────────────────────────
        graph_paths = self._select_graph_files()
        if not graph_paths:
            warn("No valid graph files selected. Exiting mode 5.")
            return
        success(f"{len(graph_paths)} graph(s) selected.")

        # 2 ── Assign labels ───────────────────────────────────────────────
        graph_labels = self._assign_graph_labels(graph_paths)

        # 3 ── Select analytics modules ────────────────────────────────────
        modules_to_run = self._select_analytics_modules()
        if not modules_to_run:
            warn("No analytics modules selected. Exiting.")
            return

        # 4 ── Gather analytics parameters once for all graphs ─────────────
        analytics_config = self._get_analytics_config(modules_to_run)

        # 5 ── Resolve output root for this Mode 5 run ────────────────────
        # Single graph is logically a single case; multi-graph runs go
        # under ``output/comparisons/{timestamp}/`` so they don't pollute
        # the case directory.
        if len(graph_paths) == 1:
            safe = _sanitize_label(graph_labels[0])
            comparison_root = DATA_OUTPUT_DIR / "cases" / safe
            comparison_root.mkdir(parents=True, exist_ok=True)
        else:
            ts_now = datetime.now().strftime("%Y%m%d_%H%M%S")
            comparison_root = DATA_OUTPUT_DIR / "comparisons" / ts_now
            comparison_root.mkdir(parents=True, exist_ok=True)
            typing(
                f"\n✓ Comparison outputs will be written to:\n"
                f"  {comparison_root}\n"
            )
        self.session_data["comparison_root"] = comparison_root

        # 6 ── Run analytics on each graph ─────────────────────────────────
        print_section("Running Analytics")
        graph_results: Dict[str, Dict] = {}
        for graph_path, label in zip(graph_paths, graph_labels):
            safe = _sanitize_label(label)
            graph_output_dir = (
                comparison_root
                if len(graph_paths) == 1
                else comparison_root / safe
            )
            graph_output_dir.mkdir(parents=True, exist_ok=True)

            typing(
                f"\n{'─' * 60}\n"
                f"  Graph : {label}\n"
                f"  File  : {graph_path.name}\n"
                f"  Out   : {graph_output_dir}\n"
                f"{'─' * 60}\n"
            )
            self.logger.info(f"Mode 5: analyzing '{label}' ({graph_path})")

            results = self._run_analytics_on_graph(
                graph_path=graph_path,
                label=label,
                output_dir=graph_output_dir,
                modules=modules_to_run,
                config=analytics_config,
            )
            graph_results[label] = {
                "graph_path": graph_path,
                "output_dir": graph_output_dir,
                "results": results,
            }

        # 7 ── Comparative tables (only when multiple graphs) ──────────────
        if len(graph_paths) > 1:
            print_section("Comparative Analysis")
            ComparativeReport(self.logger).render(
                graph_results=graph_results,
                modules=modules_to_run,
                comparison_root=comparison_root,
            )

        # 8 ── IA interpretation ───────────────────────────────────────────
        self._run_ia_interpreter_multi_graph(
            graph_results=graph_results,
            modules=modules_to_run,
            comparison_root=comparison_root,
        )

        typing(
            f"\n✓ Existing graph mode complete.\n"
            f"  All outputs in: {comparison_root}\n"
        )

    # ==================================================================
    # Mode 4 — Pre-configured (non-interactive)
    # ==================================================================

    def _run_preconfigured_mode(self) -> None:
        """
        Mode 4: drive the full enrichment + analysis pipeline from
        ``self.state.preconfig`` without asking any questions.

        Hydrates ``state.config`` and ``session_data`` from the YAML, seeds
        snapshots, then walks the same ``PIPELINE`` list as Mode 1.  Every
        prompt-heavy stage method checks ``state.preconfigured`` at its top
        and short-circuits to the configured values.
        """
        print_section("Pre-configured run")
        pre = self.state.preconfig
        if pre is None:
            error("No pre-configuration loaded; aborting Mode 4.")
            self.logger.error("Mode 4 selected but state.preconfig is None.")
            return
        if not pre.city.strip():
            error(
                "Mode 4 requires a non-empty `city` in the YAML config "
                f"({self.config_path})."
            )
            self.logger.error("Mode 4 aborted: empty city in preconfig.")
            return

        # ── hydrate state.config from the preconfig ──────────────────────
        cfg = self.state.config
        cfg.city = pre.city
        cfg.use_coordinates = pre.extent.use_coordinates
        cfg.coordinates = (
            {
                "long_min": pre.extent.min_longitude,
                "lat_min":  pre.extent.min_latitude,
                "long_max": pre.extent.max_longitude,
                "lat_max":  pre.extent.max_latitude,
            }
            if pre.extent.use_coordinates
            else {}
        )
        cfg.use_temporal = pre.temporal.enabled
        cfg.timestamps = list(pre.temporal.timestamps)
        cfg.recovery_graph = pre.recovery_graph_path
        cfg.recovery_label = pre.temporal.recovery_label

        # ── mirror onto session_data (legacy consumers still read it) ────
        self.session_data["city"] = pre.city
        self.session_data["use_coordinates"] = pre.extent.use_coordinates
        self.session_data["coordinates"] = dict(cfg.coordinates)
        self.session_data["use_temporal"] = pre.temporal.enabled
        self.session_data["timestamps"] = list(pre.temporal.timestamps)
        if pre.recovery_graph_path is not None:
            self.session_data["recovery_graph_path"] = pre.recovery_graph_path
            self.session_data["recovery_label"] = pre.temporal.recovery_label

        # ── validate timestamps ──────────────────────────────────────────
        bad = []
        for ts in cfg.timestamps:
            try:
                datetime.strptime(ts, "%Y%m%d")
            except ValueError:
                bad.append(ts)
        if bad:
            error(f"Invalid timestamp(s) in config (expected YYYYMMDD): {bad}")
            self.logger.error(f"Mode 4 aborted: invalid timestamps {bad}.")
            return

        # ── seed snapshots ───────────────────────────────────────────────
        if pre.temporal.enabled and cfg.timestamps:
            for ts in cfg.timestamps:
                self.state.ensure(ts)
        else:
            self.state.ensure(None)

        # ── banner ───────────────────────────────────────────────────────
        info(f"  Config       : {self.config_path}")
        info(f"  City         : {pre.city}")
        if pre.extent.use_coordinates:
            info(
                f"  Extent       : ({pre.extent.min_longitude}, {pre.extent.min_latitude}) "
                f"→ ({pre.extent.max_longitude}, {pre.extent.max_latitude})"
            )
        else:
            info("  Extent       : auto (place-name geocode)")
        if pre.temporal.enabled:
            info(f"  Timestamps   : {', '.join(cfg.timestamps) or '(none)'}")
        else:
            info("  Temporal     : single snapshot (current)")
        modules_on = [
            name for name, flag in (
                ("stats",         pre.analysis.stats.run),
                ("robustness",    pre.analysis.robustness.run),
                ("percolation",   pre.analysis.percolation.run),
                ("vulnerability", pre.analysis.vulnerability.run),
                ("ia",            pre.ia.run),
            ) if flag
        ]
        info(f"  Modules      : {', '.join(modules_on) or '(none)'}")

        # ── walk the same pipeline as Mode 1 ─────────────────────────────
        for stage_method in self.PIPELINE:
            getattr(self, stage_method)()

    # ==================================================================
    # Mode 2 — Local-Automated
    # ==================================================================

    def _run_local_automated_mode(self) -> None:
        """
        Mode 2: discover ready-made disaster cases under ``DATA_INPUT_DIR``,
        let the user pick one, then run the analysis tail of the pipeline.
        Enrichment stages (facilities, demography, disruption, elevation)
        are skipped — the user is expected to have produced them offline.
        """
        print_section("Local-automated mode")

        cases = _discover_local_cases(DATA_INPUT_DIR)
        if not cases:
            warn(f"No disaster cases found under {DATA_INPUT_DIR}.")
            return

        info(f"Found {len(cases)} disaster case(s) on disk:")
        print()
        print(_format_case_table(cases))
        print()

        idx = ask_int(
            f"Pick a case to analyze [1–{len(cases)}]",
            default=1,
            bounds=(1, len(cases)),
        )
        snap, location = cases[idx - 1]
        self._adopt_local_case(snap, location)

        for stage_method in self.LOCAL_AUTOMATED_PIPELINE:
            getattr(self, stage_method)()

    def _adopt_local_case(self, snap: Snapshot, location: str) -> None:
        """Install the selected case as the sole snapshot for this session."""
        pretty_city = location.replace("_", " ")

        self.state.snapshots = [snap]
        self.state.config.city = pretty_city
        self.state.config.use_temporal = False
        self.state.config.timestamps = []

        # Mirror to ``session_data`` for any stages / consumers still reading it.
        self.session_data["city"] = pretty_city
        self.session_data["use_temporal"] = False
        self.session_data["timestamps"] = []
        if snap.osm_gpkg:
            self.session_data["osm_output_path"] = snap.osm_gpkg
        if snap.streets_with_population:
            self.session_data["streets_with_population_path"] = snap.streets_with_population
        if snap.streets_with_disruption:
            self.session_data["streets_with_disruption_path"] = snap.streets_with_disruption
        if snap.streets_with_elevation:
            self.session_data["streets_with_elevation_path"] = snap.streets_with_elevation
        if snap.streets_with_water:
            self.session_data["streets_with_water_path"] = snap.streets_with_water
        if snap.graph:
            self.session_data["graph_path"] = snap.graph
        if snap.simplified_graph:
            self.session_data["simplified_graph_path"] = snap.simplified_graph

        present = sum(
            1 for p in (
                snap.osm_gpkg, snap.streets_with_population,
                snap.streets_with_disruption, snap.streets_with_elevation,
                snap.streets_with_water,
                snap.graph, snap.simplified_graph,
            ) if p is not None
        )
        success(
            f"Loaded case: {pretty_city} ({snap.timestamp}) — "
            f"{present}/7 artifacts on disk"
        )
        self.logger.info(
            f"Mode 2: adopted case {snap.timestamp}/{location}; "
            f"{present}/7 artifacts present"
        )

    # ── Graph selection ────────────────────────────────────────────────────────

    def _select_graph_files(self) -> List[Path]:
        """
        Discover .gt files under src/data/input/ (recursively, newest first)
        and let the user select one or more, or enter custom paths.
        """
        search_dir = DATA_INPUT_DIR
        candidates: List[Path] = (
            sorted(
                search_dir.rglob("*.gt"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if search_dir.exists()
            else []
        )

        if candidates:
            print("\nAvailable .gt graph files (newest first):\n")
            for i, p in enumerate(candidates, 1):
                size_mb = p.stat().st_size / 1_048_576
                try:
                    rel = p.relative_to(search_dir)
                except ValueError:
                    rel = p
                print(f"  {i:3d}.  {str(rel):<55s}  ({size_mb:.1f} MiB)")
            print()

            sel_raw = get_user_input(
                "Select files by number (comma-separated, e.g. '1,3'),\n"
                " 'all' to use all listed, or enter full path(s) separated by commas:",
                default="all",
            ).strip().lower()
        else:
            print(
                "\nNo .gt files found in the default input directory "
                f"({search_dir}).\n"
            )
            sel_raw = ""

        selected: List[Path] = []

        if sel_raw == "all" and candidates:
            selected = list(candidates)
        else:
            tokens = [t.strip() for t in sel_raw.split(",") if t.strip()]
            for token in tokens:
                if token.isdigit():
                    idx = int(token) - 1
                    if 0 <= idx < len(candidates):
                        selected.append(candidates[idx])
                    else:
                        print(f"  ⚠ Index {token} out of range; skipping.")
                else:
                    p = Path(token)
                    if p.exists() and p.suffix == ".gt":
                        selected.append(p)
                    elif token:
                        print(
                            f"  ⚠ Not a valid .gt file: {token}; skipping."
                        )

            # No candidates listed and nothing selected → prompt again
            if not selected and not candidates:
                paths_raw = get_user_input(
                    "Enter .gt file path(s) (comma-separated):", default=""
                ).strip()
                for token in (t.strip() for t in paths_raw.split(",") if t.strip()):
                    p = Path(token)
                    if p.exists() and p.suffix == ".gt":
                        selected.append(p)
                    else:
                        print(f"  ⚠ Not a valid .gt file: {token}; skipping.")

        # Deduplicate preserving insertion order
        seen: set = set()
        result: List[Path] = []
        for p in selected:
            key = p.resolve()
            if key not in seen:
                seen.add(key)
                result.append(p)

        if result:
            success(f"Selected {len(result)} graph(s):")
            for p in result:
                print(f"  • {p}")
            print()

        return result

    def _assign_graph_labels(self, graph_paths: List[Path]) -> List[str]:
        """
        Auto-generate descriptive labels from file stems and offer the user a
        chance to rename any of them.
        """
        def _auto_label(p: Path) -> str:
            """Peel every known enrichment suffix so the label collapses to
            the case stem (matches the slug used by Modes 1 and 2)."""
            stem = p.stem
            while True:
                for sfx in _LOCAL_SUFFIXES:
                    if stem.endswith(sfx):
                        stem = stem[: -len(sfx)]
                        break
                else:
                    break
            return stem

        auto_labels = [_auto_label(p) for p in graph_paths]

        print("\nAuto-generated labels:\n")
        for i, (p, lbl) in enumerate(zip(graph_paths, auto_labels), 1):
            print(f"  {i:3d}.  {lbl!r:<40s}  ← {p.name}")
        print()

        rename = confirm("Rename any labels?", default=False)

        labels = list(auto_labels)
        if rename:
            for i in range(len(labels)):
                new_label = get_user_input(
                    f"  Label for graph {i + 1} [{labels[i]}]",
                    default=labels[i],
                ).strip()
                labels[i] = new_label or labels[i]

        return labels

    # ── Module / parameter selection ───────────────────────────────────────────

    def _select_analytics_modules(self) -> List[str]:
        """Let the user select which analytics modules to run."""
        print_section("Select Analytics Modules")
        print(
            "  1. Network Disruption Analysis   (stats_analyst)\n"
            "  2. Network Robustness Estimation\n"
            "  3. Percolation Analysis\n"
            "  4. Vulnerability Simulation\n"
        )
        sel_raw = ask_text(
            "Which modules to run? (comma-separated numbers, or 'all')",
            default="all",
        ).lower()

        module_map = {
            "1": "stats",
            "2": "robustness",
            "3": "percolation",
            "4": "vulnerability",
        }
        all_modules = list(module_map.values())

        if sel_raw == "all":
            chosen = list(all_modules)
        else:
            chosen = []
            for token in (t.strip() for t in sel_raw.split(",") if t.strip()):
                if token in module_map:
                    m = module_map[token]
                    if m not in chosen:
                        chosen.append(m)
                else:
                    print(f"  ⚠ Unknown module: {token!r}; skipping.")
            if not chosen:
                typing("  No valid selection; running all modules.\n")
                chosen = list(all_modules)

        success(f"Selected modules: {', '.join(chosen)}")
        return chosen

    def _get_analytics_config(self, modules: List[str]) -> Dict[str, Any]:
        """
        Gather analytics parameters for the selected modules upfront once,
        so the user is not asked the same questions for each graph.
        """
        config: Dict[str, Any] = {}

        if "robustness" in modules:
            print_section("Robustness Parameters")
            weight_choice = ask_choice(
                WEIGHT_PROMPT,
                options=["length", "hops"],
                default="length",
            )
            config["rob_weight_prop"] = "length" if weight_choice == "length" else ""

            rec_raw = get_user_input(
                "Path to a recovery/future-state .gt graph for comparison\n"
                " (applies to all graphs; press Enter to skip)",
                default="",
            ).strip()
            if rec_raw:
                rp = Path(rec_raw)
                if rp.exists() and rp.suffix == ".gt":
                    config["rob_recovery_path"] = rp
                    rec_lbl = get_user_input(
                        "Label for recovery state", default="Recovery"
                    ).strip()
                    config["rob_recovery_label"] = rec_lbl or "Recovery"
                else:
                    print(
                        f"  ⚠ Recovery graph not found or invalid: {rec_raw}; "
                        "skipping."
                    )
                    config["rob_recovery_path"] = None
                    config["rob_recovery_label"] = "Recovery"
            else:
                config["rob_recovery_path"] = None
                config["rob_recovery_label"] = "Recovery"

        if "percolation" in modules:
            print_section("Percolation Parameters")
            print(
                "  Available scenarios:\n"
                "   1. Betweenness centrality attack\n"
                "   2. Random failure\n"
                "   3. Elevation-based removal\n"
            )
            scen_raw = ask_text(
                "Scenarios to run (comma-separated, e.g. '1,2' or '1,2,3')",
                default="1,2",
            )
            chosen_scen = {
                int(s.strip())
                for s in scen_raw.split(",")
                if s.strip().isdigit() and int(s.strip()) in (1, 2, 3)
            }
            config["perc_run_bw"]  = 1 in chosen_scen
            config["perc_run_rnd"] = 2 in chosen_scen
            config["perc_run_elv"] = 3 in chosen_scen
            config["perc_n_steps"] = None
            config["perc_recompute_every"] = 1
            config["perc_run_null"] = False
            config["perc_null_m"] = 5
            config["perc_n_random_runs"] = 10
            config["perc_elev_ascending"] = True

            if config["perc_run_bw"]:
                n_raw = ask_text(
                    "Betweenness attack steps (blank = auto: 30% of edges, max 500)",
                    default="auto",
                ).lower()
                if n_raw not in ("", "auto"):
                    try:
                        config["perc_n_steps"] = int(n_raw)
                    except ValueError:
                        pass
                config["perc_recompute_every"] = ask_int(
                    "Recompute betweenness every N steps (1 = fully dynamic)",
                    default=1,
                    bounds=(1, 10000),
                )

            if config["perc_run_rnd"]:
                config["perc_n_random_runs"] = ask_int(
                    "Number of random failure runs",
                    default=10,
                    bounds=(1, 200),
                )

            if config["perc_run_elv"]:
                direction = ask_choice(
                    ELEV_REMOVAL_FLOOD_CLOSURE,
                    options=["flood", "closure"],
                    default="flood",
                )
                config["perc_elev_ascending"] = direction == "flood"

        if "vulnerability" in modules:
            print_section("Vulnerability Parameters")
            config["vuln_n_sim"] = ask_int(
                "Monte Carlo simulations (higher = more accurate; 200 recommended)",
                default=200,
                bounds=(10, 5000),
            )
            config["vuln_use_elevation"] = confirm(
                "Include elevation (elev_min) as adjustment covariate if available?",
                default=True,
            )
            config["vuln_use_water"] = confirm(
                "Include distance-to-water (water_dist_min) as adjustment covariate if available?",
                default=True,
            )
            config["vuln_run_service_mc"] = confirm(
                "Run service-accessibility Monte Carlo?",
                default=True,
            )

        return config

    # ── Per-graph analytics runner ─────────────────────────────────────────────

    def _run_analytics_on_graph(
        self,
        graph_path: Path,
        label: str,
        output_dir: Path,
        modules: List[str],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run all selected analytics modules on a single graph, writing every
        output under *output_dir*.  Returns a dict of {module_name: result}.
        """
        results: Dict[str, Any] = {}
        output_dir.mkdir(parents=True, exist_ok=True)

        if "stats" in modules:
            with stage(f"[{label}] Network disruption analysis", logger=self.logger) as st:
                results["stats"] = analyse_network(
                    graph_path=graph_path,
                    output_dir=output_dir,
                    logger=self.logger,
                )
                st.note(f"Output: {output_dir / 'stats'}")

        if "robustness" in modules:
            with stage(f"[{label}] Robustness estimation", logger=self.logger) as st:
                results["robustness"] = estimate_robustness(
                    graph_path=graph_path,
                    output_dir=output_dir,
                    recovery_path=config.get("rob_recovery_path"),
                    recovery_label=config.get("rob_recovery_label", "Recovery"),
                    weight_prop=config.get("rob_weight_prop", "length"),
                    logger=self.logger,
                )
                st.note(f"Output: {output_dir / 'robustness'}")

        if "percolation" in modules:
            with stage(f"[{label}] Percolation analysis", logger=self.logger) as st:
                results["percolation"] = run_percolation(
                    graph_path=graph_path,
                    output_dir=output_dir,
                    run_betweenness=config.get("perc_run_bw", True),
                    run_random=config.get("perc_run_rnd", True),
                    run_elevation=config.get("perc_run_elv", False),
                    n_steps=config.get("perc_n_steps"),
                    recompute_every=config.get("perc_recompute_every", 1),
                    run_null_model=config.get("perc_run_null", False),
                    null_m=config.get("perc_null_m", 5),
                    n_random_runs=config.get("perc_n_random_runs", 10),
                    elevation_ascending=config.get("perc_elev_ascending", True),
                    logger=self.logger,
                )
                st.note(f"Output: {output_dir / 'percolation'}")

        if "vulnerability" in modules:
            with stage(f"[{label}] Vulnerability simulation", logger=self.logger) as st:
                results["vulnerability"] = run_vulnerability_simulation(
                    graph_path=graph_path,
                    output_dir=output_dir,
                    n_sim=config.get("vuln_n_sim", 200),
                    use_elevation=config.get("vuln_use_elevation", True),
                    use_water=config.get("vuln_use_water", True),
                    run_service_mc=config.get("vuln_run_service_mc", True),
                    logger=self.logger,
                )
                st.note(f"Output: {output_dir / 'vulnerability'}")

        return results

    # ── IA interpretation for multiple graphs ──────────────────────────────────

    def _run_ia_interpreter_multi_graph(
        self,
        graph_results: Dict[str, Dict],
        modules: List[str],
        comparison_root: Path,
    ) -> None:
        """
        Run LLM-powered interpretation for each graph individually, generate
        per-graph executive summaries, then synthesize a cross-graph comparative
        narrative.
        """
        from src.antipompeii.modules.ia_interpreter import (
            AnalysisModule,
            LLMConfig,
            run_interpretations,
            save_reports,
            save_summary,
            synthesize_reports,
            synthesize_comparison,
            save_comparison,
        )

        print_section("IA Interpretation")
        if not confirm(
            "Generate intelligence-augmented interpretation of results?",
            default=False,
        ):
            success("Skipping IA interpretation")
            self.logger.info("User opted to skip IA interpretation (mode 5).")
            return

        # Build module mapping
        module_map: Dict[str, AnalysisModule] = {}
        if "stats" in modules:
            module_map["stats"] = AnalysisModule.STATS_ANALYST
        if "robustness" in modules:
            module_map["robustness"] = AnalysisModule.ROBUSTNESS
        if "percolation" in modules:
            module_map["percolation"] = AnalysisModule.PERCOLATOR
        if "vulnerability" in modules:
            module_map["vulnerability"] = AnalysisModule.VULNERABILITY

        if not module_map:
            warn("No IA-supported modules were run.")
            return

        # ── LLM backend selection ──────────────────────────────────────────
        typing(LLM_BACKEND_MENU)
        backend = ask_choice(
            "Backend (1–5)",
            options=["1", "2", "3", "4", "5"],
            default="1",
        )
        api_key:  Optional[str] = None
        api_base: Optional[str] = None
        model: str
        timeout: int = 300

        if backend == "1":
            model_name = (
                get_user_input("Ollama model name", default="llama3").strip()
                or "llama3"
            )
            api_base = (
                get_user_input(
                    "Ollama API base URL", default="http://localhost:11434"
                ).strip()
                or "http://localhost:11434"
            )
            model = f"ollama/{model_name}"

            import urllib.request as _urlreq
            import urllib.error as _urlerr
            try:
                _urlreq.urlopen(f"{api_base}/api/tags", timeout=5)
            except (_urlerr.URLError, OSError) as _e:
                typing(
                    f"\n⚠ Cannot reach Ollama at {api_base} ({_e}).\n"
                    "  Make sure the Ollama server is running (`ollama serve`) "
                    "and the URL is correct.\n"
                    "  Skipping IA interpretation.\n"
                )
                self.logger.error(f"Ollama preflight failed: {_e}")
                return

            timeout_raw = get_user_input(
                "Request timeout in seconds (large models may need 300–600)",
                default="300",
            ).strip()
            try:
                timeout = max(30, int(timeout_raw))
            except ValueError:
                timeout = 300

        elif backend == "2":
            api_key = get_user_input("Anthropic API key", default="").strip() or None
            model = (
                get_user_input("Claude model", default="claude-sonnet-4-6").strip()
                or "claude-sonnet-4-6"
            )
        elif backend == "3":
            api_key = get_user_input("OpenAI API key", default="").strip() or None
            model = (
                get_user_input("OpenAI model", default="gpt-4o").strip() or "gpt-4o"
            )
        elif backend == "4":
            api_key = get_user_input("Perplexity API key", default="").strip() or None
            model_name = (
                get_user_input(
                    "Perplexity model",
                    default="llama-3.1-sonar-large-128k-online",
                ).strip()
                or "llama-3.1-sonar-large-128k-online"
            )
            model = f"perplexity/{model_name}"
            api_base = "https://api.perplexity.ai"
        else:
            model = get_user_input("litellm model string", default="").strip()
            if not model:
                warn("No model specified; skipping IA interpretation.")
                return
            api_key = (
                get_user_input("API key (Enter to skip)", default="").strip() or None
            )
            api_base = (
                get_user_input("API base URL (Enter to skip)", default="").strip()
                or None
            )

        ia_config = LLMConfig(model=model, api_key=api_key, api_base=api_base, timeout=timeout)
        selected_modules = list(module_map.values())

        # ── Per-graph interpretation loop ──────────────────────────────────
        graph_summaries: Dict[str, str] = {}

        for label, gr in graph_results.items():
            output_dir = gr["output_dir"]
            event_label = _sanitize_label(label)

            print_section(f"IA — {label}")
            typing(
                f"\n→ Interpreting {len(selected_modules)} module(s) "
                f"for [{label}] with [{ia_config.model}]...\n"
            )
            self.logger.info(
                f"Mode 5 IA: graph={label}, model={ia_config.model}, "
                f"output_root={output_dir}"
            )

            interpretations = run_interpretations(
                config=ia_config,
                modules=selected_modules,
                output_root=output_dir,
                logger=self.logger,
            )

            for module, text in interpretations.items():
                print_section(f"  {module.value}")
                print(text)
                print()

            written = save_reports(
                interpretations, output_dir,
                event_label=event_label, logger=self.logger,
            )
            if written:
                success("Module reports saved:")
                for mod, path in written.items():
                    print(f"  {mod.value:30s} → {path}")
                print()

            # Per-graph executive summary
            valid_count = sum(
                1 for t in interpretations.values()
                if not t.startswith("ERROR:") and not t.startswith("(No data")
            )
            if valid_count >= 1:
                step("Generating per-graph executive summary...")
                summary = synthesize_reports(
                    config=ia_config,
                    interpretations=interpretations,
                    logger=self.logger,
                )
                if summary and not summary.startswith("ERROR:"):
                    print_section(f"IA Summary — {label}")
                    print(summary)
                    print()
                    graph_summaries[label] = summary
                    sum_path = save_summary(
                        summary, output_dir,
                        event_label=event_label, logger=self.logger,
                    )
                    if sum_path:
                        typing(f"✓ Summary saved: {sum_path}\n")
                else:
                    warn(f"Executive summary failed: {summary[:120]}")

        # ── Cross-graph comparative synthesis ──────────────────────────────
        if len(graph_summaries) >= 2:
            typing(
                "\n→ Generating cross-graph comparative analysis "
                f"({len(graph_summaries)} graphs)...\n"
            )
            comparison_text = synthesize_comparison(
                config=ia_config,
                graph_summaries=graph_summaries,
                logger=self.logger,
            )
            if comparison_text and not comparison_text.startswith("ERROR:"):
                print_section("IA — Cross-Graph Comparative Analysis")
                print(comparison_text)
                print()
                comp_path = save_comparison(
                    comparison_text,
                    comparison_root,
                    logger=self.logger,
                )
                if comp_path:
                    typing(f"✓ Comparative analysis saved: {comp_path}\n")
            else:
                typing(
                    f"\n⚠ Cross-graph comparison failed: "
                    f"{comparison_text[:120]}\n"
                )
        elif len(graph_summaries) == 1:
            typing(
                "\n  (Cross-graph comparison requires ≥ 2 graphs with valid "
                "summaries; skipping.)\n"
            )

        success("IA interpretation complete.")

    # ------------------------------------------------------------------
    # IA interpretation
    # ------------------------------------------------------------------

    def _run_ia_interpreter(self) -> None:
        """
        Intelligence-augmented report interpretation.

        Offers an LLM-powered narrative interpretation of the CSV outputs
        produced by the four analysis modules (stats_analyst, robustness,
        percolation, vulnerability).  Supports local Ollama, Claude,
        OpenAI / GPT, Perplexity, or any custom litellm model string.
        """
        print_section("IA Interpretation")

        if self.state.preconfigured:
            if not self.state.preconfig.ia.run:
                success("Skipping IA interpretation (disabled in config).")
                self.logger.info("Mode 4: IA disabled in config.")
                return
        elif not confirm(
            "Generate intelligence-augmented interpretation of ANTIPOMPEII results?",
            default=False,
        ):
            success("Skipping IA interpretation")
            self.logger.info("User opted to skip IA interpretation.")
            return

        from src.antipompeii.modules.ia_interpreter import (
            AnalysisModule,
            LLMConfig,
            run_interpretations,
            save_reports,
            save_summary,
            synthesize_reports,
        )

        # ── Locate output root first (needed for disk-based checks) ──────
        graph_path_raw = self.session_data.get("graph_path") or (
            (self.session_data.get("graph_paths_temporal") or [None])[0]
        )
        if graph_path_raw is None:
            typing(
                "\n⚠ Cannot locate analysis outputs: no graph path in session. "
                "Build the graph network first.\n"
            )
            return
        output_root = self._case_output_dir()

        # Derive event label from the graph filename (strip pipeline suffixes)
        event_label = Path(graph_path_raw).stem
        for _sfx in ("_simplified", "_network"):
            if event_label.endswith(_sfx):
                event_label = event_label[: -len(_sfx)]
                break

        # ── Discover available modules: session_data OR CSV files on disk ─
        available: Dict[str, AnalysisModule] = {}
        if (self.session_data.get("analysis_results")
                or any(output_root.glob("stats/*_roads.csv"))):
            available["1"] = AnalysisModule.STATS_ANALYST
        if (self.session_data.get("robustness_report")
                or (output_root / "robustness" / "robustness.csv").exists()):
            available["2"] = AnalysisModule.ROBUSTNESS
        if (self.session_data.get("percolation_results")
                or any(output_root.glob("percolation/*_summary.csv"))):
            available["3"] = AnalysisModule.PERCOLATOR
        if (self.session_data.get("vulnerability_result")
                or (output_root / "vulnerability" / "global_indices.csv").exists()):
            available["4"] = AnalysisModule.VULNERABILITY

        if not available:
            typing(
                "\n⚠ No analysis outputs found (neither in session nor on disk). "
                "Run at least one analysis module first.\n"
            )
            return

        if self.state.preconfigured:
            ia_cfg = self.state.preconfig.ia
            name_to_module = {
                "stats":         AnalysisModule.STATS_ANALYST,
                "robustness":    AnalysisModule.ROBUSTNESS,
                "percolation":   AnalysisModule.PERCOLATOR,
                "vulnerability": AnalysisModule.VULNERABILITY,
            }
            avail_mods = list(available.values())
            if "all" in ia_cfg.modules:
                selected_modules = avail_mods
            else:
                requested = {name_to_module[n] for n in ia_cfg.modules if n in name_to_module}
                selected_modules = [m for m in avail_mods if m in requested]
            if not selected_modules:
                warn(
                    f"No matching IA modules from config {ia_cfg.modules!r} "
                    f"(available: {[m.value for m in avail_mods]}); skipping."
                )
                return
            info(f"  IA modules   : {[m.value for m in selected_modules]}")
        else:
            typing("\nModules with results available for interpretation:\n")
            for num, mod in available.items():
                print(f"  {num}. {mod.value}")
            print()

            sel_raw = get_user_input(
                "Which modules to interpret? (comma-separated numbers, or 'all')",
                default="all",
            ).strip().lower()

            if sel_raw == "all":
                selected_modules = list(available.values())
            else:
                selected_modules = []
                for token in (t.strip() for t in sel_raw.split(",")):
                    if token in available:
                        selected_modules.append(available[token])
                if not selected_modules:
                    warn("No valid module number selected; skipping.")
                    return

        # ── Backend / provider selection ──────────────────────────────────
        if self.state.preconfigured:
            ia_cfg = self.state.preconfig.ia
            backend_map = {"ollama": "1", "claude": "2", "openai": "3",
                           "perplexity": "4", "custom": "5"}
            backend = backend_map.get(ia_cfg.backend, "5")
            info(
                f"  IA backend   : {ia_cfg.backend} (model={ia_cfg.model})"
            )
        else:
            typing(LLM_BACKEND_MENU)
            backend = ask_choice(
                "Backend (1–5)",
                options=["1", "2", "3", "4", "5"],
                default="1",
            )

        api_key:  Optional[str] = None
        api_base: Optional[str] = None
        model:    str

        timeout: int = 300

        if self.state.preconfigured:
            ia_cfg = self.state.preconfig.ia
            api_key = ia_cfg.api_key
            api_base = ia_cfg.api_base
            timeout = max(30, ia_cfg.timeout)
            if backend == "1":      # Ollama
                model = f"ollama/{ia_cfg.model}"
                api_base = api_base or "http://localhost:11434"
            elif backend == "4":    # Perplexity
                model = f"perplexity/{ia_cfg.model}"
                api_base = api_base or "https://api.perplexity.ai"
            else:                   # Claude / OpenAI / custom
                model = ia_cfg.model
                if not model:
                    warn("No `ia.model` in config; skipping IA interpretation.")
                    return

        elif backend == "1":  # ── Ollama ──────────────────────────────────
            model_name = (
                get_user_input(
                    "Ollama model name (e.g. llama3, mistral, gemma2)",
                    default="llama3",
                ).strip()
                or "llama3"
            )
            api_base = (
                get_user_input(
                    "Ollama API base URL",
                    default="http://localhost:11434",
                ).strip()
                or "http://localhost:11434"
            )
            model = f"ollama/{model_name}"

            # Preflight: verify Ollama is reachable before queuing a long call
            import urllib.request as _urlreq
            import urllib.error as _urlerr
            try:
                _urlreq.urlopen(f"{api_base}/api/tags", timeout=5)
            except (_urlerr.URLError, OSError) as _e:
                typing(
                    f"\n⚠ Cannot reach Ollama at {api_base} ({_e}).\n"
                    "  Make sure the Ollama server is running (`ollama serve`) "
                    "and the URL is correct.\n"
                    "  Skipping IA interpretation.\n"
                )
                self.logger.error(f"Ollama preflight failed: {_e}")
                return

            timeout_raw = get_user_input(
                "Request timeout in seconds (large models may need 300–600)",
                default="300",
            ).strip()
            try:
                timeout = max(30, int(timeout_raw))
            except ValueError:
                timeout = 300

        elif backend == "2":  # ── Claude ──────────────────────────────────
            api_key = get_user_input("Anthropic API key", default="").strip() or None
            model = (
                get_user_input(
                    "Claude model (e.g. claude-sonnet-4-6, claude-opus-4-6)",
                    default="claude-sonnet-4-6",
                ).strip()
                or "claude-sonnet-4-6"
            )

        elif backend == "3":  # ── OpenAI ──────────────────────────────────
            api_key = get_user_input("OpenAI API key", default="").strip() or None
            model = (
                get_user_input(
                    "OpenAI model (e.g. gpt-4o, gpt-4o-mini)",
                    default="gpt-4o",
                ).strip()
                or "gpt-4o"
            )

        elif backend == "4":  # ── Perplexity ───────────────────────────────
            api_key = get_user_input("Perplexity API key", default="").strip() or None
            model_name = (
                get_user_input(
                    "Perplexity model "
                    "(e.g. llama-3.1-sonar-large-128k-online)",
                    default="llama-3.1-sonar-large-128k-online",
                ).strip()
                or "llama-3.1-sonar-large-128k-online"
            )
            model    = f"perplexity/{model_name}"
            api_base = "https://api.perplexity.ai"

        else:  # ── Custom litellm model string ─────────────────────────────
            model = get_user_input(
                "litellm model string "
                "(e.g. together_ai/mistral-7b, groq/llama3-8b-8192)",
                default="",
            ).strip()
            if not model:
                warn("No model specified; skipping IA interpretation.")
                return
            api_key = (
                get_user_input("API key (Enter to skip)", default="").strip()
                or None
            )
            api_base = (
                get_user_input("API base URL (Enter to skip)", default="").strip()
                or None
            )

        config = LLMConfig(model=model, api_key=api_key, api_base=api_base, timeout=timeout)

        # ── Run ───────────────────────────────────────────────────────────
        typing(
            f"\n→ Interpreting {len(selected_modules)} module(s) "
            f"with [{config.model}]...\n"
        )
        self.logger.info(
            f"IA interpretation: model={config.model}, "
            f"modules={[m.value for m in selected_modules]}, "
            f"output_root={output_root}"
        )

        interpretations = run_interpretations(
            config=config,
            modules=selected_modules,
            output_root=output_root,
            logger=self.logger,
        )

        # ── Display ───────────────────────────────────────────────────────
        for module, text in interpretations.items():
            print_section(f"IA — {module.value}")
            print(text)
            print()

        # ── Save per-module reports ───────────────────────────────────────
        written = save_reports(
            interpretations, output_root,
            event_label=event_label, logger=self.logger,
        )
        if written:
            success("IA reports saved:")
            for module, path in written.items():
                print(f"  {module.value:30s} → {path}")
            print()

        # ── Executive synthesis (cross-module summary) ────────────────────
        valid_count = sum(
            1 for t in interpretations.values()
            if not t.startswith("ERROR:") and not t.startswith("(No data")
        )
        if valid_count >= 1:
            step("Generating executive summary across all module reports...")
            summary = synthesize_reports(
                config=config,
                interpretations=interpretations,
                logger=self.logger,
            )
            if summary and not summary.startswith("ERROR:"):
                print_section("IA — Executive Summary")
                print(summary)
                print()
                summary_path = save_summary(
                    summary, output_root,
                    event_label=event_label, logger=self.logger,
                )
                if summary_path:
                    typing(f"✓ Executive summary saved: {summary_path}\n")
            else:
                warn(f"Executive summary failed: {summary[:120]}")

        success("IA interpretation complete.")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    cli = antipompeiiCLI()
    cli.run()
