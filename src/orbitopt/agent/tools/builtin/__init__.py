"""The first batch of built-in scientific tools, each verified against an independent reference.

Importing this package imports no numpy or scipy; a tool imports them when
it runs. See ``docs/agent/tools.md`` for the table of tools and references.
"""

from __future__ import annotations

from ..registry import ToolRegistry
from . import heat, quadrature, special

BUILTIN_TOOLS = (*special.TOOLS, *quadrature.TOOLS, *heat.TOOLS)


def builtin_registry() -> ToolRegistry:
    """A new registry holding every built-in tool."""
    return ToolRegistry(BUILTIN_TOOLS)
