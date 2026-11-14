"""Flight Lab's Mission Control widgets, plus the Qt-free telemetry helpers.

A Flight Lab run renders in Mission Control's main view like a mission; the
widgets here add what only a flight has (``flight_panel``: the Flight tab and
the Telemetry charts; ``nav``: the run list; ``compare_page``: two runs side
by side). They need PySide6, so they are imported lazily: the telemetry
helpers below stay importable on a headless install.
"""
from __future__ import annotations

from orbitopt.flightlab.ui.telemetry import (
    TelemetryChannel,
    TelemetrySeries,
    build_telemetry_channels,
    channel_values_at,
    interpolated_value,
    load_run_scene,
)

__all__ = [
    "FlightLabComparePage",
    "FlightLabNav",
    "FlightRunPanel",
    "TelemetryChannel",
    "TelemetryPanel",
    "TelemetrySeries",
    "build_telemetry_channels",
    "channel_values_at",
    "interpolated_value",
    "load_run_scene",
]

_LAZY = {
    "FlightLabComparePage": "orbitopt.flightlab.ui.compare_page",
    "FlightLabNav": "orbitopt.flightlab.ui.nav",
    "FlightRunPanel": "orbitopt.flightlab.ui.flight_panel",
    "TelemetryPanel": "orbitopt.flightlab.ui.flight_panel",
}


def __getattr__(name):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module), name)
