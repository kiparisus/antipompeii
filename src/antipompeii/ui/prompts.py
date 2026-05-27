"""
Prompt vocabulary for the ANTIPOMPEII CLI.

Each function loops until input is valid and delegates the actual I/O to
:func:`layout.get_user_input`.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple, Union

from src.antipompeii.ui.layout import get_user_input, typing

Number = Union[int, float]


# ---------------------------------------------------------------------------
# Yes / no
# ---------------------------------------------------------------------------

_YES = {"y", "yes"}
_NO  = {"n", "no"}


def confirm(message: str, *, default: bool = True) -> bool:
    """Ask a yes/no question.  Empty input falls back to *default*."""
    suffix      = " (y/n)" if not message.rstrip().endswith(")") else ""
    default_str = "y" if default else "n"

    while True:
        reply = get_user_input(message + suffix, default=default_str).strip().lower()
        if reply in _YES:
            return True
        if reply in _NO:
            return False
        typing(f"Please answer 'y' or 'n' (got: {reply!r}).")


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------

def _ask_number(
    message: str,
    default: Number,
    parser: type,
    bounds: Optional[Tuple[Number, Number]],
) -> Number:
    while True:
        raw = get_user_input(message, default=str(default)).strip()
        if not raw:
            typing("Input cannot be empty.")
            continue
        try:
            value = parser(raw)
        except ValueError:
            typing(f"Not a valid {parser.__name__}: {raw!r}.")
            continue
        if bounds is not None and not (bounds[0] <= value <= bounds[1]):
            typing(f"Value must be between {bounds[0]} and {bounds[1]}.")
            continue
        return value


def ask_int(
    message: str,
    *,
    default: int,
    bounds: Optional[Tuple[int, int]] = None,
) -> int:
    """Prompt for an integer.  *bounds* is an inclusive ``(low, high)`` pair."""
    return int(_ask_number(message, default, int, bounds))


def ask_float(
    message: str,
    *,
    default: float,
    bounds: Optional[Tuple[float, float]] = None,
) -> float:
    """Prompt for a float.  *bounds* is an inclusive ``(low, high)`` pair."""
    return float(_ask_number(message, default, float, bounds))


# ---------------------------------------------------------------------------
# Choice / free text
# ---------------------------------------------------------------------------

def ask_choice(
    message: str,
    *,
    options: Sequence[str],
    default: Optional[str] = None,
) -> str:
    """
    Prompt for one of *options* (case-insensitive).  Returns the canonical
    option string from *options*, not whatever case the user typed.
    """
    options = list(options)
    if not options:
        raise ValueError("ask_choice requires a non-empty options list.")
    if default is not None and default not in options:
        raise ValueError(
            f"default {default!r} is not one of {options!r}."
        )

    canonical = {opt.lower(): opt for opt in options}
    default_str = default if default is not None else options[0]

    # Append an inline option list — same convention as confirm()'s (y/n).
    # Single-line messages get the list inline; multi-line messages
    # (typically bullet-formatted descriptions) get it on its own indented
    # line so it doesn't tail-end onto a description.
    inline = "(" + "/".join(options) + ")"
    if inline in message:
        prompt = message
    elif "\n" in message:
        prompt = f"{message}\n  {inline}"
    else:
        prompt = f"{message} {inline}"

    while True:
        reply = get_user_input(prompt, default=default_str).strip().lower()
        if reply in canonical:
            return canonical[reply]
        typing(f"Please choose one of: {', '.join(options)}.")


def ask_text(message: str, *, default: str = "") -> str:
    """Prompt for free-form text.  Empty input is allowed when *default* is empty."""
    return get_user_input(message, default=default).strip()


__all__ = ["confirm", "ask_int", "ask_float", "ask_choice", "ask_text"]
