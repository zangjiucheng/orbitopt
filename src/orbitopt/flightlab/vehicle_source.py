"""Vehicle-file resolution for the Flight Lab CLIs (FL-202).

One resolver, adopted first by ``orbitopt flight optimize`` and later by
``flight simulate``: a mission file may declare its vehicle with a path
that is only correct relative to *where the fixture lives* (the shipped
``leo_two_stage.yaml`` says ``scenarios/vehicles/synthetic_two_stage.yaml``
while the data package keeps the file under ``vehicles/``), so resolution
walks a fixed, fully-reported order and never silently guesses:

1. the explicit CLI path (``--vehicle``), used verbatim;
2. ``target.vehicle.source``, resolved relative to the target file's
   directory;
3. ``target.vehicle.source``, resolved relative to the installed package
   data root ``orbitopt/data/flightlab/``;
4. the *basename* of ``target.vehicle.source``, looked up in
   ``orbitopt/data/flightlab/vehicles/``.

A miss raises :class:`FileNotFoundError` whose message lists every path
that was tried -- "file not found" must never be a mystery (FL-202).
"""
from __future__ import annotations

from pathlib import Path

from orbitopt.flightlab.contract import FlightTarget

__all__ = ["data_root", "resolve_vehicle_path"]


def data_root() -> Path:
    """The installed ``orbitopt/data/flightlab`` directory."""
    import orbitopt

    return Path(orbitopt.__file__).parent / "data" / "flightlab"


def resolve_vehicle_path(vehicle: str | Path | None, target: FlightTarget,
                        target_path: str | Path) -> Path:
    """Return the vehicle file to fly, applying the documented 4-step order.

    ``vehicle`` is the explicit CLI path or ``None``; ``target``/``target_path``
    come from ``contract.load_target`` (the path is used only for step 2).
    Raises ``FileNotFoundError`` listing every candidate considered."""
    tried: list[Path] = []
    if vehicle is not None:
        path = Path(vehicle)
        if path.is_file():
            return path
        tried.append(path.resolve())
        raise FileNotFoundError(f"vehicle file not found: {path}")

    declared = getattr(getattr(target, "vehicle", None), "source", None)
    if not declared:
        raise FileNotFoundError(
            "no --vehicle given and the target declares no vehicle.source; "
            "pass --vehicle <file.yaml>")
    root = data_root()
    candidates = [Path(target_path).resolve().parent / declared,
                  root / declared,
                  (root / "vehicles" / Path(declared).name)]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
        tried.append(candidate)
    raise FileNotFoundError(
        f"vehicle '{declared}' (from {target_path}) not found; tried:\n"
        + "\n".join(f"  {p}" for p in tried))
