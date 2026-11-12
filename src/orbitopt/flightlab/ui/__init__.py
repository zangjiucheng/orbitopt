"""Qt-free helpers and the lazily-imported Qt replay page for Flight Lab.

The public Flight Lab page lives in ``replay_page.py`` and needs PySide6 /
pyvista / pyvistaqt. Importing this package must not force that dependency on
contract-only installs, so the widget is resolved lazily while the pure
telemetry/cache helpers below stay importable on a headless install.
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
    "FlightLabDesignerPage",
    "FlightLabReplayPage",
    "FlightLabWorkbench",
    "TelemetryChannel",
    "TelemetrySeries",
    "build_telemetry_channels",
    "channel_values_at",
    "interpolated_value",
    "load_run_scene",
]


def __getattr__(name):
    if name == "FlightLabReplayPage":
        from orbitopt.flightlab.ui.replay_page import FlightLabReplayPage

        return FlightLabReplayPage
    if name == "FlightLabComparePage":
        from orbitopt.flightlab.ui.compare_page import FlightLabComparePage

        return FlightLabComparePage
    if name == "FlightLabDesignerPage":
        from orbitopt.flightlab.ui.designer_page import FlightLabDesignerPage

        return FlightLabDesignerPage
    if name == "FlightLabWorkbench":
        from orbitopt.flightlab.ui.workbench import FlightLabWorkbench

        return FlightLabWorkbench
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
