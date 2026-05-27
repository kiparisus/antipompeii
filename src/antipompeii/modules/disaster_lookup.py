"""
Offline disaster activation lookup from bundled CSV databases.

Sources
-------
* Sentinel Asia Emergency Observations  (db/sentinel.csv)
* UN International Charter Space & Major Disasters  (db/charter.csv)

Usage
-----
    from src.antipompeii.modules.disaster_lookup import lookup_disasters

    recent, all_events = lookup_disasters("Manila, Philippines")
    for ev in recent:
        print(ev.display())
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

# ── Paths ──────────────────────────────────────────────────────────────────────

_DB_DIR       = Path(__file__).parent.parent / "db"
_SENTINEL_CSV = _DB_DIR / "sentinel.csv"
_CHARTER_CSV  = _DB_DIR / "charter.csv"

_SOURCE_SENTINEL = "Sentinel Asia"
_SOURCE_CHARTER  = "UN Charter"

# ── Data model ─────────────────────────────────────────────────────────────────


@dataclass
class DisasterActivation:
    """One activation record from either database."""

    date:          str   # YYYYMMDD
    year:          int
    disaster_type: str
    country:       str
    code:          str   # ISO 3166-1 alpha-3
    source:        str   # "Sentinel Asia" or "UN Charter"
    url:           str

    @property
    def date_formatted(self) -> str:
        d = self.date
        return f"{d[:4]}-{d[4:6]}-{d[6:8]}" if len(d) == 8 else d

    def display(self) -> str:
        type_col = self.disaster_type.capitalize()
        return (
            f"  [{self.date_formatted}]  {type_col:<18}  {self.country}\n"
            f"             Source : {self.source}\n"
            f"             → {self.url}"
        )


# ── CSV loader ─────────────────────────────────────────────────────────────────


def _load_csv(path: Path, source: str) -> List[DisasterActivation]:
    if not path.exists():
        return []
    rows: List[DisasterActivation] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                rows.append(DisasterActivation(
                    date=row["date"].strip(),
                    year=int(row["year"]),
                    disaster_type=row["type"].strip(),
                    country=row["country"].strip(),
                    code=row["code"].strip(),
                    source=source,
                    url=row["link"].strip(),
                ))
            except (KeyError, ValueError):
                continue
    return rows


# ── Country matching ───────────────────────────────────────────────────────────


def _matches(area_name: str, activation: DisasterActivation) -> bool:
    """
    Return True when *activation*'s country covers *area_name*.

    """
    area_lower    = area_name.lower()
    country_lower = activation.country.lower()
    code_lower    = activation.code.lower()

    area_parts    = [p.strip() for p in area_lower.split(",")]
    country_parts = [p.strip() for p in country_lower.split(",") if p.strip()]

    # How many trailing area parts to consider (≥ 1)
    n = max(1, len(country_parts))
    area_suffix = area_parts[-n:] if n <= len(area_parts) else area_parts

    # Bidirectional substring check on positionally aligned suffix pairs
    for ap, cp in zip(area_suffix, country_parts):
        if len(ap) >= 3 and len(cp) >= 3 and (ap in cp or cp in ap):
            return True

    # ISO3 code match on the last area token (e.g. "usa" == "usa")
    last = area_parts[-1] if area_parts else ""
    if last and last == code_lower:
        return True

    # Last area part vs full country name (catches "philippines" ↔ "the philippines")
    if len(last) >= 3 and len(country_lower) >= 3:
        if last in country_lower or country_lower in last:
            return True

    return False


# ── Public API ─────────────────────────────────────────────────────────────────


def lookup_disasters(
    area_name: str,
    max_recent: int = 10,
) -> Tuple[List[DisasterActivation], List[DisasterActivation]]:
    """
    Return disaster activations matching the country implied by *area_name*.
    """
    all_events = (
        _load_csv(_SENTINEL_CSV, _SOURCE_SENTINEL)
        + _load_csv(_CHARTER_CSV, _SOURCE_CHARTER)
    )
    matched = sorted(
        (ev for ev in all_events if _matches(area_name, ev)),
        key=lambda e: e.date,
        reverse=True,
    )
    return matched[:max_recent], matched
