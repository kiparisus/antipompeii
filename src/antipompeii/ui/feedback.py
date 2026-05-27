"""
Feedback vocabulary for the ANTIPOMPEII CLI.

Five verbs — ``step``, ``success``, ``warn``, ``error``, ``info`` — and one
:func:`stage` context manager that prints start/done banners, logs to an
optional logger, and swallows exceptions so the caller can branch on
``stage.failed``.  All output flows through :func:`layout.typing`.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, List, Optional

from src.antipompeii.ui.layout import typing


# Single-character glyphs.  Kept in one place so the whole CLI agrees on
# what "success" or "warning" look like.
_GLYPH = {
    "step":    "→",
    "success": "✓",
    "warn":    "⚠",
    "error":   "✗",
}


# ---------------------------------------------------------------------------
# Atomic feedback verbs
# ---------------------------------------------------------------------------

def _emit(prefix: str, msg: str) -> None:
    """Print one feedback line with a leading blank for breathing room."""
    typing(f"\n{prefix} {msg}")


def step(msg: str) -> None:
    """Announce the start of an action."""
    _emit(_GLYPH["step"], msg)


def success(msg: str) -> None:
    """Announce successful completion."""
    _emit(_GLYPH["success"], msg)


def warn(msg: str) -> None:
    """Report a recoverable problem."""
    _emit(_GLYPH["warn"], msg)


def error(msg: str) -> None:
    """Report a fatal problem for the current stage."""
    _emit(_GLYPH["error"], msg)


def info(msg: str) -> None:
    """Plain status line, no glyph."""
    typing(f"\n  {msg}")


# ---------------------------------------------------------------------------
# Stage context manager
# ---------------------------------------------------------------------------

@dataclass
class Stage:
    """
    Handle returned by :func:`stage`.  Outlives the ``with`` block so the
    caller can inspect ``failed`` and decide whether to short-circuit.
    """
    name:      str
    failed:    bool                       = False
    exception: Optional[BaseException]    = None
    notes:     List[str]                  = field(default_factory=list)

    def note(self, msg: str) -> None:
        """Print an in-progress status line and remember it on the stage."""
        self.notes.append(msg)
        info(msg)


@contextmanager
def stage(
    name: str,
    *,
    logger: Optional[logging.Logger] = None,
) -> Iterator[Stage]:
    """
    Mark a pipeline stage.  Prints ``→ name`` on entry, ``✓ name complete``
    on clean exit, or ``✗ Error during name: …`` if the body raised.  The
    exception is captured (not re-raised) so the caller can branch on
    ``stage.failed`` without a try/except wrapper.

    Usage
    -----
    >>> with stage("Disruption processing", logger=self.logger) as st:
    ...     result = append_disruption_to_streets(...)
    ...     self.session_data["streets_with_disruption_path"] = path
    >>> if st.failed:
    ...     return
    """
    step(name)
    if logger is not None:
        logger.info(f"{name}: started")

    handle = Stage(name=name)
    try:
        yield handle
    except Exception as exc:
        handle.failed    = True
        handle.exception = exc
        error(f"Error during {name}: {exc}")
        if logger is not None:
            logger.error(f"{name} failed: {exc}", exc_info=True)
    else:
        if not handle.failed:
            success(f"{name} complete")
            if logger is not None:
                logger.info(f"{name}: complete")


__all__ = ["step", "success", "warn", "error", "info", "stage", "Stage"]
