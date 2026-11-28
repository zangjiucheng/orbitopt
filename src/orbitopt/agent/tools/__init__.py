"""Scientific tool registry (E14-04): declared, allowlisted, logged and replayable tool calls.

An agent, or an IR ``ToolCall`` objective or constraint, calls a tool only
through a ``ToolView`` over a ``ToolRegistry``; every call lands in a
hash-chained ``CallLog`` that ``replay`` re-executes on a fresh registry.
Importing this package pulls in jsonschema and the optimization IR, not
numpy or scipy. See ``docs/agent/tools.md``.
"""

from __future__ import annotations

from .builtin import BUILTIN_TOOLS, builtin_registry
from .calllog import CallLog, CallRecord, ReplayEntry, ReplayReport
from .registry import CallBudget, Caller, ToolRegistry, ToolView, bind_problem, replay
from .spec import COST_CLASSES, CostModel, DomainError, Interval, Tool, ToolError, ToolErrorCode, ToolSpec

__all__ = [
    "BUILTIN_TOOLS", "COST_CLASSES", "CallBudget", "CallLog", "CallRecord", "Caller", "CostModel", "DomainError",
    "Interval", "ReplayEntry", "ReplayReport", "Tool", "ToolError", "ToolErrorCode", "ToolRegistry", "ToolSpec",
    "ToolView", "bind_problem", "builtin_registry", "replay",
]
