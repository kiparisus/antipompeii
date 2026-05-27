"""Centralized graph-tool import guard.

Every module that depends on graph-tool should import from here::

    from src.antipompeii.utils.graph_tool_compat import gt, GRAPH_TOOL_AVAILABLE

Then do module-specific sub-imports behind ``if GRAPH_TOOL_AVAILABLE:``.
"""

import warnings

try:
    import graph_tool as gt
    GRAPH_TOOL_AVAILABLE = True
except ImportError:
    gt = None  # type: ignore[assignment]
    GRAPH_TOOL_AVAILABLE = False
    warnings.warn(
        "graph-tool not installed. Graph-dependent features will be disabled. "
        "Install via conda: conda install -c conda-forge graph-tool"
    )
