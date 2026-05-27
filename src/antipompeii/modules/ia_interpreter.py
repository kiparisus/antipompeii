"""
ANTIPOMPEII Intelligence-Augmented Interpretation Module

Reads the CSV outputs produced by ANTIPOMPEII's analysis modules and sends
them to an LLM for natural-language interpretation.  Supports any backend
accessible through litellm: local Ollama instances, Claude (Anthropic),
OpenAI / GPT, Perplexity AI, and any other litellm-compatible provider.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional


# ── Analysis module identifiers ────────────────────────────────────────────────

class AnalysisModule(str, Enum):
    """Identifies an ANTIPOMPEII output-producing module."""
    STATS_ANALYST = "stats_analyst"
    ROBUSTNESS    = "robustness_estimator"
    PERCOLATOR    = "percolator"
    VULNERABILITY = "vulnerability_simulator"


# ── LLM configuration ─────────────────────────────────────────────────────────

@dataclass
class LLMConfig:
    """
    Connection parameters passed to litellm.completion().
    """
    model: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    temperature: float = 0.3
    max_tokens: int = 4096
    timeout: int = 300  # seconds; increase for large local models (e.g. 70B+)


# ── Per-module prompt definitions ─────────────────────────────────────────────

@dataclass
class ModulePrompt:
    """
    System prompt and user-message template for a single analysis module.

    Both fields support free-form text.  The user template may contain the
    literal ``{tables}`` placeholder, which is replaced at call-time with the
    concatenated CSV content of the module's output files.

    If ``{tables}`` is absent from ``user``, the tables are appended after a
    blank line at the end of the user message.
    """
    system: str = ""
    user: str   = "{tables}"


@dataclass
class SummaryPrompt:
    """
    System prompt and user-message template for the cross-module synthesis.

    The ``{reports}`` placeholder is replaced with all individual module
    interpretations concatenated under ``=== module_name ===`` section headers.

    If ``{reports}`` is absent from ``user``, the reports block is appended
    after a blank line at the end of the user message.
    """
    system: str = ""
    user: str   = "{reports}"


# ── Prompts ────────────────────────────────────────────────────────────────────
# Each entry defines two strings:
#   system — the LLM's role, analytical context, and output format
#            (e.g. "You are an urban resilience analyst...").
#   user   — the request paired with the data.  The token {tables} is
#            substituted with the module's CSV outputs at runtime.

PROMPTS: Dict[AnalysisModule, ModulePrompt] = {

    AnalysisModule.STATS_ANALYST: ModulePrompt(
        system=(
            "You are an expert urban resilience analyst specializing in disaster "
            "impact assessment on critical infrastructure networks. You interpret "
            "quantitative road-network and population-exposure outputs from the "
            "ANTIPOMPEII spatial analysis platform. Your task is to produce a "
            "structured, policy-oriented narrative from three tabular outputs: "
            "(1) road impact by highway class, (2) facility disruption analysis, "
            "and (3) population at risk disaggregated by facility type and "
            "demographic band.\n\n"
            "Write in clear, precise American English. Organize your response under three "
            "section headings matching the three tables. Within each section, lead "
            "with the most critical finding, quantify it specifically using the "
            "numbers in the tables, and close with a brief implication for emergency "
            "response or infrastructure planning. Total length: 400–600 words."
        ),
        user=(
            "Interpret the following ANTIPOMPEII disruption analysis outputs for "
            "a disaster event.\n\n"
            "Three tables are provided:\n\n"
            "1. ROAD NETWORK IMPACT — rows are highway classes (Motorway, Trunk, "
            "Primary, Secondary, Tertiary, Residential, Service, Other, TOTAL); "
            "columns are:\n"
            "   · Base [n]            — total edges in that class\n"
            "   · Direct [n (%)]      — edges physically blocked by the disaster\n"
            "   · Indirect [n (%)]    — edges isolated by network fragmentation "
            "(their component lost access to ≥1 critical facility, but not "
            "physically blocked)\n"
            "   · No access [n (%)]   — direct + indirect combined\n\n"
            "2. FACILITY DISRUPTION — rows are facility types; columns are:\n"
            "   · Edges (base)                — street edges serving that facility\n"
            "   · Edges disrupted [n (%)]     — directly blocked edges\n"
            "   · Unique facilities           — distinct named facilities total\n"
            "   · Facilities affected         — those with ≥1 disrupted edge\n"
            "   · Complete shutdown           — facilities where every serving edge is blocked\n"
            "   · Partially hindered          — facilities with some (not all) edges blocked\n\n"
            "3. POPULATION IMPACT — rows are demographic bands "
            "(Total, Female 0–14, Female 15–64, Female 65+, Male 0–14, Male 15–64, "
            "Male 65+); columns are:\n"
            "   · Network population      — total persons on all street edges for this band\n"
            "   · Direct loss             — persons on physically blocked edges\n"
            "   · Any service — indirect  — persons on active edges whose network "
            "component lost access to any critical facility\n"
            "   · Any service — total     — direct + indirect (the headline risk figure)\n"
            "   · {Service name} columns  — total loss (direct + service-specific indirect) "
            "for each individual facility type present in the network\n\n"
            "Focus on: which road classes bear the greatest burden; which facilities "
            "face complete shutdown vs. partial disruption; which demographic groups "
            "show the highest absolute and relative exposure; whether indirect loss "
            "substantially exceeds direct loss (signaling network fragmentation as "
            "the dominant mechanism). Identify the single most urgent finding in "
            "each section.\n\n"
            "{tables}"
        ),
    ),

    AnalysisModule.ROBUSTNESS: ModulePrompt(
        system=(
            "You are a quantitative network scientist specializing in urban "
            "infrastructure resilience. You interpret graph-theoretic robustness "
            "metrics produced by the ANTIPOMPEII robustness estimator. Your "
            "audience is infrastructure planners and emergency managers who need "
            "to understand how a disaster has degraded the street network's "
            "structural integrity — beyond its physical blockages.\n\n"
            "Write in precise, jargon-light English. Structure your response as: "
            "(1) a brief characterization of the intact network baseline, "
            "(2) the impact of each disruption state on key metrics with specific "
            "numbers, (3) which metrics show the most severe degradation and why "
            "that matters operationally, (4) one practical implication for "
            "network repair or route planning. Total length: 300–450 words."
        ),
        user=(
            "Interpret the following ANTIPOMPEII network robustness table for "
            "a disaster event.\n\n"
            "Table structure:\n"
            "  · Rows alternate: a state row (S_0 = intact baseline, S_1 = after "
            "first disruption event, S_2 = after second, etc.) followed by a Δ row "
            "expressing the percentage change relative to the previous state.\n"
            "  · Columns and their meanings:\n"
            "      n_e   — number of active edges (roads); reduction = physical damage\n"
            "      κ     — number of connected components; higher = greater fragmentation\n"
            "      d_max — network diameter (longest shortest path in largest component); "
            "larger = worse connectivity\n"
            "      L     — average shortest path length across reachable vertex pairs; "
            "larger = slower routing\n"
            "      E     — global efficiency (harmonic mean of inverse distances, 0–1); "
            "lower = worse\n"
            "      C     — average clustering coefficient (local connectivity density, 0–1)\n"
            "      R*    — Kirchhoff index (sum of effective resistances); higher = more brittle\n"
            "      b_max — maximum edge betweenness centrality; high = critical bottleneck exists\n"
            "      b_avg — mean edge betweenness centrality\n\n"
            "Pay particular attention to fragmentation (κ), efficiency (E), and Kirchhoff "
            "index (R*) as the three most policy-relevant indicators of functional network "
            "collapse. Flag any metric that worsens by more than 20% between states.\n\n"
            "{tables}"
        ),
    ),

    AnalysisModule.PERCOLATOR: ModulePrompt(
        system=(
            "You are a network resilience analyst specializing in percolation theory "
            "applied to urban infrastructure. You interpret vulnerability threshold "
            "analyses produced by ANTIPOMPEII's percolation module, which simulates "
            "progressive edge removal under different failure strategies to reveal how "
            "robustly the road network sustains access to critical facilities.\n\n"
            "Write in clear, policy-oriented English. For each removal strategy present, "
            "explain the T_10, T_50, and T_90 thresholds in practical terms (e.g. 'after "
            "removing only 8% of roads, hospital access degrades by half'). Compare "
            "strategies if multiple are present. Close with a risk-priority ranking of "
            "the most vulnerable facilities and a note on whether the network shows "
            "concentrated vs. distributed fragility. Total length: 300–500 words."
        ),
        user=(
            "Interpret the following ANTIPOMPEII percolation analysis summary table(s).\n\n"
            "Each table corresponds to one edge-removal strategy:\n"
            "  · betweenness — targeted removal of highest-betweenness edges first; "
            "represents a deliberate attack or spatially concentrated damage\n"
            "  · random      — uniform random edge removal; represents diffuse, "
            "spatially unstructured damage\n"
            "  · elevation   — removal ordered by edge elevation (lowest first); "
            "represents flood or inundation scenarios\n\n"
            "Column meanings:\n"
            "  · Facility — the critical service type being monitored\n"
            "  · T_10     — fraction of edges removed before service degrades by 10%; "
            "low T_10 = early fragility onset\n"
            "  · T_50     — fraction removed for 50% degradation; the key resilience threshold\n"
            "  · T_90     — fraction removed for 90% degradation; approaches total failure\n\n"
            "Interpretation guidance:\n"
            "  · T_50 < 0.20 indicates a highly fragile facility–network relationship.\n"
            "  · A large gap between betweenness and random T_50 reveals structural weak "
            "points that a targeted disruption can exploit disproportionately.\n"
            "  · Facilities with low T_10 under elevation removal are at elevated flood risk.\n\n"
            "Identify the three most vulnerable facility types, the most dangerous removal "
            "strategy overall, and whether fragility is concentrated in a few bottlenecks "
            "or distributed across the network.\n\n"
            "{tables}"
        ),
    ),

    AnalysisModule.VULNERABILITY: ModulePrompt(
        system=(
            "You are an urban vulnerability analyst with expertise in Bayesian spatial "
            "statistics and infrastructure risk modelling. You interpret composite "
            "vulnerability index outputs from the ANTIPOMPEII vulnerability simulator, "
            "which combines network topology, population exposure, and empirical Bayes "
            "shrinkage to estimate where and how severely people are at risk when road "
            "infrastructure fails.\n\n"
            "Write in precise, structured English. Address each of the three output "
            "tables in turn: global indices first, class parameters second, model "
            "accuracy third. Interpret numbers in human-readable terms (e.g. 'X% of "
            "the population lives in high-vulnerability zones'). Flag any accuracy "
            "metric below acceptable thresholds. Close with a one-paragraph synthesis "
            "of whether the vulnerability predictions are trustworthy and what they "
            "imply for spatial planning. Total length: 350–500 words."
        ),
        user=(
            "Interpret the following ANTIPOMPEII vulnerability simulator outputs.\n\n"
            "Three tables are provided:\n\n"
            "1. GLOBAL VULNERABILITY INDICES (global_indices.csv):\n"
            "   · Ṽ_len    — length-weighted mean vulnerability (0–1); average "
            "vulnerability per meter of road in the network\n"
            "   · Ṽ_pop    — population-weighted mean vulnerability (0–1); average "
            "vulnerability experienced per resident\n"
            "   · G̃_pop    — global population exposure index (0–1); approximate "
            "fraction of the total population residing in vulnerable zones\n"
            "   · Expected persons — estimated number of people exposed to high vulnerability\n"
            "   Higher values indicate more severe systemic vulnerability.\n\n"
            "2. CLASS PARAMETERS (classes.csv):\n"
            "   Empirical Bayes shrinkage results per vulnerability class. Columns "
            "typically include: class label, raw rate, EB-smoothed rate, shrinkage "
            "factor, posterior mean. These reveal which vulnerability classes are most "
            "prevalent and how strongly local estimates are regularised toward the "
            "global mean (high shrinkage = sparse local data).\n\n"
            "3. MODEL ACCURACY (accuracy.csv):\n"
            "   Validation metrics for the vulnerability classification model:\n"
            "   · Sensitivity (recall)   — fraction of truly vulnerable segments correctly flagged\n"
            "   · Specificity            — fraction of non-vulnerable segments correctly excluded\n"
            "   · Balanced accuracy      — mean of sensitivity and specificity; "
            "> 0.75 is acceptable\n"
            "   · AUC-PR                 — area under precision-recall curve; "
            "> 0.60 is acceptable for imbalanced spatial data\n"
            "   · Other metrics as present (F1, PPV, NPV, etc.)\n\n"
            "Assess: (a) the overall scale and spatial concentration of vulnerability, "
            "(b) which classes drive the exposure and how reliable their EB estimates are, "
            "(c) whether the model's discriminatory power is sufficient to trust the "
            "spatial predictions for operational use.\n\n"
            "{tables}"
        ),
    ),
}


# ── Cross-module synthesis prompt ──────────────────────────────────────────────

SUMMARY_PROMPT = SummaryPrompt(
    system=(
        "You are a senior urban resilience analyst preparing a decision briefing "
        "for city officials and civil protection planners. You have reviewed "
        "technical reports from the ANTIPOMPEII platform covering up to four "
        "analytical dimensions of a single disaster event: network disruption "
        "statistics, graph-theoretic structural robustness, percolation fragility "
        "thresholds, and composite vulnerability indices.\n\n"
        "Your task is to synthesize the key cross-cutting findings into a concise, "
        "decision-ready executive summary. Structure your response under exactly "
        "these four headings:\n\n"
        "1. Overall risk assessment — How severe is the event's impact on the urban "
        "infrastructure system as a whole? What is the dominant failure mode?\n"
        "2. Critical infrastructure findings — Which roads, facilities, or network "
        "zones are most compromised? Are failures concentrated or distributed?\n"
        "3. Population at risk — How many people are affected, which demographic "
        "groups face the greatest exposure, and through which facility failures?\n"
        "4. Priority recommendations — Three concrete, specific actions for emergency "
        "managers or planners, ranked by urgency, each actionable within 72 hours "
        "or the short-term recovery window.\n\n"
        "Write in direct, non-technical American English suitable for a non-specialist audience. "
        "Do not repeat detailed statistics from individual reports unless a single "
        "number is the most important finding. Each section: 3–5 sentences. "
        "Total length: 300–450 words."
    ),
    user=(
        "The following are ANTIPOMPEII intelligence-augmented module reports for "
        "a single disaster event. Each section corresponds to one analytical module. "
        "Synthesise the cross-cutting key messages into an executive summary using "
        "the four-section structure specified in your instructions.\n\n"
        "{reports}"
    ),
)


# ── File-system layout per module ─────────────────────────────────────────────
# Glob patterns relative to the shared output root (graph_path.parent).
# The output subdirectory is where the txt report will be written.

_MODULE_OUTPUT_SUBDIRS: Dict[AnalysisModule, str] = {
    AnalysisModule.STATS_ANALYST: "stats",
    AnalysisModule.ROBUSTNESS:    "robustness",
    AnalysisModule.PERCOLATOR:    "percolation",
    AnalysisModule.VULNERABILITY: "vulnerability",
}

_MODULE_CSV_GLOBS: Dict[AnalysisModule, List[str]] = {
    AnalysisModule.STATS_ANALYST: [
        "stats/*_roads.csv",
        "stats/*_facilities.csv",
        "stats/*_population.csv",
    ],
    AnalysisModule.ROBUSTNESS: [
        "robustness/robustness.csv",
    ],
    AnalysisModule.PERCOLATOR: [
        "percolation/*_summary.csv",
    ],
    AnalysisModule.VULNERABILITY: [
        "vulnerability/global_indices.csv",
        "vulnerability/classes.csv",
        "vulnerability/accuracy.csv",
    ],
}


# ── Table collection ──────────────────────────────────────────────────────────

def collect_tables(module: AnalysisModule, output_root: Path) -> str:
    """
    Read all CSV output files for *module* and concatenate them into a
    single plain-text block for embedding in an LLM prompt.
    """
    globs = _MODULE_CSV_GLOBS.get(module, [])
    sections: List[str] = []

    for pattern in globs:
        for csv_path in sorted(output_root.glob(pattern)):
            try:
                content = csv_path.read_text(encoding="utf-8").strip()
                if content:
                    sections.append(f"### {csv_path.name}\n{content}")
            except OSError:
                pass

    return "\n\n".join(sections)


# ── LLM call helpers ──────────────────────────────────────────────────────────

def _llm_call(
    config: LLMConfig,
    messages: List[Dict],
    label: str,
    logger: logging.Logger,
) -> str:
    """Shared litellm call; returns response text or an ERROR string."""
    import litellm  # lazy import — missing dep only fails at call-time

    call_kwargs: dict = dict(
        model=config.model,
        messages=messages,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout=config.timeout,
    )
    if config.api_key:
        call_kwargs["api_key"] = config.api_key
    if config.api_base:
        call_kwargs["api_base"] = config.api_base

    logger.info(f"Calling LLM [{config.model}] for '{label}'")
    try:
        response = litellm.completion(**call_kwargs)
        return response.choices[0].message.content or ""
    except Exception as exc:
        logger.error(f"LLM call failed for '{label}': {exc}")
        return f"ERROR: LLM call failed — {exc}"


# ── Per-module interpretation ─────────────────────────────────────────────────

def interpret(
    config: LLMConfig,
    module: AnalysisModule,
    tables: str,
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    Send *tables* to the configured LLM and return the response text.

    If the module's system prompt is empty a generic fallback is used so the
    call still works before prompts have been filled in manually.
    """
    log = logger or logging.getLogger(__name__)
    prompt = PROMPTS[module]

    system_text = prompt.system.strip() or (
        f"You are an expert urban resilience analyst. "
        f"Interpret the following tabular outputs from the ANTIPOMPEII "
        f"{module.value} module. Be concise, precise, and highlight "
        f"the most policy-relevant findings."
    )

    if "{tables}" in prompt.user:
        # Use plain str.replace so other {…} in the prompt template (e.g.
        # "{Service name}") are never mis-parsed as format fields.
        user_text = prompt.user.replace("{tables}", tables)
    else:
        user_text = f"{prompt.user}\n\n{tables}" if prompt.user.strip() else tables

    messages = [
        {"role": "system", "content": system_text},
        {"role": "user",   "content": user_text},
    ]
    return _llm_call(config, messages, module.value, log)


# ── Cross-module synthesis ────────────────────────────────────────────────────

def synthesize_reports(
    config: LLMConfig,
    interpretations: Dict[AnalysisModule, str],
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    Send all per-module interpretations to the LLM in one call and return
    a cross-module executive summary.
    """
    log = logger or logging.getLogger(__name__)

    valid = {
        m: t for m, t in interpretations.items()
        if not t.startswith("ERROR:") and not t.startswith("(No data")
    }
    if not valid:
        log.warning("No valid module interpretations to synthesize.")
        return ""

    reports_block = "\n\n".join(
        f"=== {module.value} ===\n{text}" for module, text in valid.items()
    )

    system_text = SUMMARY_PROMPT.system.strip() or (
        "You are a senior urban resilience analyst. You have read individual "
        "ANTIPOMPEII module reports covering network disruption statistics, "
        "structural robustness, percolation thresholds, and vulnerability "
        "indices for a disaster event. Synthesise the most policy-relevant "
        "cross-cutting findings into a concise executive summary."
    )

    if "{reports}" in SUMMARY_PROMPT.user:
        user_text = SUMMARY_PROMPT.user.replace("{reports}", reports_block)
    else:
        user_text = (
            f"{SUMMARY_PROMPT.user}\n\n{reports_block}"
            if SUMMARY_PROMPT.user.strip()
            else reports_block
        )

    messages = [
        {"role": "system", "content": system_text},
        {"role": "user",   "content": user_text},
    ]
    log.info(
        f"Synthesising {len(valid)} module report(s) into executive summary..."
    )
    return _llm_call(config, messages, "synthesis", log)


# ── Report saving ─────────────────────────────────────────────────────────────

def save_reports(
    interpretations: Dict[AnalysisModule, str],
    output_root: Path,
    event_label: str = "",
    logger: Optional[logging.Logger] = None,
) -> Dict[AnalysisModule, Path]:
    """
    Write each interpretation to a timestamped report file inside the
    module's own output subdirectory.
    """
    log = logger or logging.getLogger(__name__)
    written: Dict[AnalysisModule, Path] = {}

    filename = f"ia_report_{event_label}.txt" if event_label else "ia_report.txt"

    for module, text in interpretations.items():
        if text.startswith("ERROR:") or text.startswith("(No data"):
            log.warning(f"Skipping save for '{module.value}': {text[:60]}")
            continue

        subdir = _MODULE_OUTPUT_SUBDIRS.get(module, module.value)
        report_dir = Path(output_root) / subdir
        report_dir.mkdir(parents=True, exist_ok=True)

        report_path = report_dir / filename
        report_path.write_text(text, encoding="utf-8")
        log.info(f"IA report saved: {report_path}")
        written[module] = report_path

    return written


def save_summary(
    summary: str,
    output_root: Path,
    event_label: str = "",
    logger: Optional[logging.Logger] = None,
) -> Optional[Path]:
    """
    Write the cross-module executive summary to
    ``{output_root}/ia_summary_{event_label}.txt``.

    Returns the written path, or ``None`` if *summary* is empty or an error.
    """
    log = logger or logging.getLogger(__name__)

    if not summary or summary.startswith("ERROR:"):
        if summary.startswith("ERROR:"):
            log.warning(f"Summary not saved due to LLM error: {summary[:80]}")
        return None

    filename = f"ia_summary_{event_label}.txt" if event_label else "ia_summary.txt"
    path = Path(output_root) / filename
    path.write_text(summary, encoding="utf-8")
    log.info(f"IA executive summary saved: {path}")
    return path


# ── Cross-graph comparative synthesis ─────────────────────────────────────────

def synthesize_comparison(
    config: LLMConfig,
    graph_summaries: Dict[str, str],
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    Send per-graph executive summaries to the LLM in one call and return a
    cross-graph comparative analysis.
    """
    log = logger or logging.getLogger(__name__)

    valid = {
        label: text
        for label, text in graph_summaries.items()
        if text and not text.startswith("ERROR:") and not text.startswith("(No data")
    }
    if len(valid) < 2:
        log.warning(
            "Fewer than 2 valid graph summaries available; cannot compare."
        )
        return next(iter(valid.values()), "")

    reports_block = "\n\n".join(
        f"=== {label} ===\n{text}" for label, text in valid.items()
    )

    system_text = (
        "You are a senior urban resilience analyst comparing multiple network "
        "scenarios or temporal snapshots from the ANTIPOMPEII urban vulnerability "
        "assessment tool. Each scenario represents a distinct disaster event, "
        "time period, or network configuration. Synthesise a structured comparative "
        "analysis from the individual scenario summaries provided. Focus on: "
        "(1) which scenario shows the greatest disruption and population impact; "
        "(2) differences in structural robustness; "
        "(3) demographic groups most differentially affected; "
        "(4) percolation behavior differences (fragility vs resilience); "
        "(5) cross-cutting policy insights that emerge only from the comparison."
    )
    user_text = (
        "Compare the following ANTIPOMPEII scenario summaries and produce a "
        "structured cross-scenario analysis. Organise your response into:\n"
        "1. Disruption severity ranking\n"
        "2. Structural robustness comparison\n"
        "3. Population vulnerability comparison\n"
        "4. Network fragility comparison (percolation thresholds)\n"
        "5. Cross-cutting policy recommendations\n\n"
        + reports_block
    )

    messages = [
        {"role": "system", "content": system_text},
        {"role": "user",   "content": user_text},
    ]
    log.info(
        f"Cross-graph comparison: {len(valid)} scenario summaries, "
        f"model={config.model}"
    )
    return _llm_call(config, messages, "Cross-Graph Comparison", log)


def save_comparison(
    comparison: str,
    output_root: Path,
    logger: Optional[logging.Logger] = None,
) -> Optional[Path]:
    """
    Write the cross-graph comparative analysis to
    ``{output_root}/comparison/ia_comparison_{timestamp}.txt``.

    Returns the written path, or ``None`` on empty input or error.
    """
    log = logger or logging.getLogger(__name__)
    if not comparison or comparison.startswith("ERROR:"):
        if comparison.startswith("ERROR:"):
            log.warning(f"Comparison not saved due to LLM error: {comparison[:80]}")
        return None

    from datetime import datetime as _dt
    comp_dir = Path(output_root) / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)
    ts = _dt.now().strftime("%Y%m%d_%H%M%S")
    path = comp_dir / f"ia_comparison_{ts}.txt"
    path.write_text(comparison, encoding="utf-8")
    log.info(f"Cross-graph comparative IA saved: {path}")
    return path


# ── Public orchestration entry point ──────────────────────────────────────────

def run_interpretations(
    config: LLMConfig,
    modules: List[AnalysisModule],
    output_root: Path,
    logger: Optional[logging.Logger] = None,
) -> Dict[AnalysisModule, str]:
    """
    Collect CSV tables and run LLM interpretation for each requested module.
    """
    log = logger or logging.getLogger(__name__)
    results: Dict[AnalysisModule, str] = {}

    for module in modules:
        tables = collect_tables(module, output_root)
        if not tables:
            log.warning(
                f"No CSV outputs found for '{module.value}' under {output_root}; "
                "skipping LLM call."
            )
            results[module] = "(No data files found for this module.)"
            continue

        log.info(f"Interpreting '{module.value}' outputs ({len(tables)} chars of data)...")
        results[module] = interpret(config, module, tables, logger=log)

    return results
