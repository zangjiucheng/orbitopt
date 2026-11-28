"""LLM-assisted optimization runtime (E14).

Importing this package imports nothing. The entry points of each part are
re-exported lazily (PEP 562), so ``from orbitopt.agent import compile_problem,
solve, make_provider`` loads only the part it names, and ``import
orbitopt.agent`` costs nothing and pulls in no dependency. The parts:

- ``orbitopt.agent.providers``: the LLM provider abstraction (stdlib and jsonschema only).
- ``orbitopt.agent.compiler``: natural language to a validated IR problem.
- ``orbitopt.agent.runtime``: the bounded, verified, replayable optimization loop.
- ``orbitopt.agent.tools``: the scientific tool registry.
- ``orbitopt.agent.commands``: what the ``orbitopt agent`` commands compute.

Names that two parts share (``ToolSpec``, ``ToolCall``, ``Violation``, the
tool log's ``replay``) are not re-exported; import them from their part.
``replay`` here is the run-directory replay of the runtime.
"""

from __future__ import annotations

_PROVIDERS = ("BudgetExceededError", "ChatProvider", "DeepSeekProvider", "LLMProvider", "LocalProvider", "Message",
              "MockProvider", "OpenAICompatProvider", "ProviderAccountError", "ProviderError", "ProviderUnavailableError", "RecordingProvider",
              "ReplayMismatchError", "ReplayProvider", "StructuredOutputError", "Usage", "UsageMeter",
              "make_provider")
_COMPILER = ("CompileFailed", "CompileOutcome", "Compiled", "NeedsClarification", "RejectedUnauthorizedChange",
             "Unsupported", "compile_problem")
_RUNTIME = ("CheckReport", "LLMPlanner", "Policy", "RecordedPlanner", "ReplayReport", "RuleBasedPlanner",
            "RunOutcome", "RunStatus", "StopReason", "check_run", "load_trace", "replay", "save_run", "solve")
_TOOLS = ("CallLog", "ToolRegistry", "ToolView", "bind_problem", "builtin_registry")
__all__ = list(_PROVIDERS) + list(_COMPILER) + list(_RUNTIME) + list(_TOOLS)


def __getattr__(name):
    import importlib

    for module, names in ((".providers", _PROVIDERS), (".compiler", _COMPILER), (".runtime", _RUNTIME),
                          (".tools", _TOOLS)):
        if name in names:
            return getattr(importlib.import_module(module, __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
