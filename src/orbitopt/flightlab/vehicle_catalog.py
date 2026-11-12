"""Vehicle data catalog with provenance & uncertainty (FL-602).

The packaged vehicle files under ``orbitopt/data/flightlab/vehicles/`` are
*data*, not fixtures: every number in them must say where it came from and
how much to trust it. That bookkeeping lives in a ``data_provenance`` sidecar
section -- the OpenTsiolkovsky flight article (``simulation``,
``attitude_program``, ``stages``) stays plain numbers, because the OT adapter
passes it through untouched and OT's own config schema rejects anything else
there. Only the ``booster:`` section (FL-301) additionally supports inline
``{value, provenance, source, ...}`` nodes, the form ``BoosterModel.load``
already reads.

Schema: ``data/flightlab/schemas/flight-vehicle-1.0.json``, validated with
the OT cached config schema's ``$defs`` injected (same trick as
``designer.validate_vehicle_document``), so ``stages`` is checked against
OT's own Stage definition rather than a copy.

A sidecar key is a document path (``stages[0].engine.isp_vac_s``,
``attitude_program``, ``booster.engines.thrust_vacuum_n``). It covers the
numeric node it names, and -- if that node is a container -- every number
below it; the nearest (longest) match wins, so a per-field entry overrides a
subtree-wide one. An inline booster node may carry its own tag/source; a bare
number inside one inherits, and a number with neither an inline tag nor a
sidecar entry is *uncovered* -- :func:`uncovered_numeric_paths` lists them,
and the catalog test demands that list be empty for every packaged file.

Uncertainties feed the FL-304 Monte Carlo: :func:`vehicle_dispersions`
converts the file's ``uncertainty_pct`` tags on thrust / Isp / Cd into a
:class:`~orbitopt.flightlab.recovery.montecarlo.Dispersions`, so a dispersed
run inherits the vehicle file's stated confidence bands by default (the
scenario-side knobs -- density scale, separation state -- keep the class
defaults; they belong to the mission, not the vehicle).

    from orbitopt.flightlab.vehicle_catalog import (
        load_vehicle_file, provenance_of, vehicle_dispersions)
    doc = load_vehicle_file("f9_class_public_estimate.yaml")   # by name or path
    print(provenance_of(doc, "stages[0].engine.isp_vac_s").as_dict())
    disp = vehicle_dispersions("f9_class_public_estimate.yaml")

See docs/flightlab.md ("Vehicle data & provenance").
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Mapping

import jsonschema
import yaml

__all__ = [
    "SCHEMA_VERSION",
    "Provenance",
    "VehicleCatalogError",
    "catalog_paths",
    "dangling_sidecar_keys",
    "load_vehicle_file",
    "load_validated",
    "provenance_of",
    "provenance_annotations",
    "uncovered_numeric_paths",
    "resolve_path",
    "validate_vehicle_document",
    "vehicle_dispersions",
    "vehicle_schema",
]

#: The schema this module and the packaged files implement.
SCHEMA_VERSION = "flight-vehicle-1.0"

_SCHEMA_FILE = "flight-vehicle-1.0.json"

#: Path -> MC dispersion field, for :func:`vehicle_dispersions`. The
#: candidates are the vehicle-side knobs FL-304 disperses; the file decides
#: the half-widths, the class defaults cover whatever the file leaves open.
_MC_FIELDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("thrust_pct", ("stages[0].engine.thrust_vac_n",
                    "booster.engines.thrust_vacuum_n")),
    ("isp_pct", ("stages[0].engine.isp_vac_s",
                 "booster.engines.isp_vacuum_s",
                 "booster.engines.isp_sea_level_s")),
    ("cd_pct", ("stages[0].aero.ca", "booster.cd_mach")),
)


class VehicleCatalogError(ValueError):
    """A packaged vehicle file is malformed: schema violation, dangling
    sidecar key or uncovered numeric field -- with the fix in the message."""


@dataclass(frozen=True)
class Provenance:
    """The resolved provenance of one document path: which sidecar entry
    (or inline booster node) covers it, with inheritance applied."""

    path: str
    provenance: str
    source: str
    as_of: str
    accuracy: str = ""
    uncertainty_pct: float | None = None

    def as_dict(self) -> dict:
        return {"provenance": self.provenance, "source": self.source,
                "as_of": self.as_of, "accuracy": self.accuracy,
                "uncertainty_pct": self.uncertainty_pct}


# ---------------------------------------------------------------------------
# schema / loading
# ---------------------------------------------------------------------------

def vehicle_schema() -> dict:
    """The flight-vehicle-1.0 JSON Schema, with the pinned OpenTsiolkovsky
    config schema's ``$defs`` injected so ``stages`` resolves against OT's own
    ``Stage`` definition (the OT $defs never contain FL names, so the merge
    cannot shadow them)."""
    from orbitopt.flightlab.adapters.opentsiolkovsky import pinned_config_schema

    schema = json.loads(_schema_resource().read_text(encoding="utf-8"))
    schema["$defs"] = {**pinned_config_schema()["$defs"], **schema["$defs"]}
    return schema


def _schema_resource() -> Path:
    return (resources.files("orbitopt")
            .joinpath("data", "flightlab", "schemas", _SCHEMA_FILE))


def catalog_paths() -> dict[str, Path]:
    """The packaged vehicle catalog: ``{file_stem: path}`` for every YAML in
    ``orbitopt/data/flightlab/vehicles/``."""
    base = Path(str(resources.files("orbitopt").joinpath(
        "data", "flightlab", "vehicles")))
    return {p.stem: p for p in sorted(base.glob("*.yaml"))}


def load_vehicle_file(vehicle: str | Path) -> dict:
    """Read a packaged vehicle file -- a catalog name (``"f9_class_..."``) or
    any path -- and return the YAML document (not validated; the loaders that
    fly it validate what they need, this function is for data work)."""
    if not isinstance(vehicle, Path) and vehicle in catalog_paths():
        return yaml.safe_load(catalog_paths()[vehicle].read_text(encoding="utf-8"))
    path = Path(vehicle)
    if not path.is_file():
        raise FileNotFoundError(
            f"not a packaged vehicle and no such file: {vehicle!r} "
            f"(catalog: {', '.join(catalog_paths()) or 'empty'})")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def validate_vehicle_document(doc: Mapping[str, Any]) -> None:
    """Schema-validate a vehicle document against flight-vehicle-1.0 (with the
    OT stage definitions injected); raise :class:`VehicleCatalogError` naming
    every violation."""
    schema = vehicle_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(doc), key=lambda e: str(list(e.absolute_path)))
    if errors:
        lines = [f"  {json.dumps(list(e.absolute_path)) or '(root)'}: {e.message}"
                 for e in errors]
        raise VehicleCatalogError(
            f"vehicle document fails {_SCHEMA_FILE}: "
            f"{len(errors)} violation(s):\n" + "\n".join(lines))


def load_validated(vehicle: str | Path | Mapping[str, Any]) -> dict:
    """Catalog file (or in-memory document) read + schema-validated, with the
    sidecar path check from :func:`dangling_sidecar_keys` (the entry point
    tests and tooling use)."""
    doc = dict(vehicle) if isinstance(vehicle, Mapping) else load_vehicle_file(vehicle)
    validate_vehicle_document(doc)
    dangling = dangling_sidecar_keys(doc)
    if dangling:
        raise VehicleCatalogError(
            "data_provenance keys point at no document node: "
            + ", ".join(sorted(dangling)))
    return doc


# ---------------------------------------------------------------------------
# provenance resolution
# ---------------------------------------------------------------------------

def resolve_path(doc: Mapping[str, Any], dotted: str) -> Any:
    """The document node a sidecar path names (``stages[0].aero.ca``);
    ``None`` when the path dangles. Raises ``ValueError`` on malformed paths."""
    if not dotted:
        return None
    tokens = _tokenize(dotted)
    if not tokens:
        raise ValueError(f"malformed provenance path: {dotted!r}")
    node: Any = doc
    for key, index in tokens:
        if key is not None:
            if not isinstance(node, Mapping) or key not in node:
                return None
            node = node[key]
        else:
            if not isinstance(node, list) or index >= len(node):
                return None
            node = node[index]
    return node


def _tokenize(dotted: str) -> list[tuple[str | None, int | None]]:
    """`stages[0].engine.ca` -> [('stages',None),(None,0),('engine',None),...];
    empty list when the string is not a well-formed path."""
    head = re.match(r"[A-Za-z_][A-Za-z0-9_]*", dotted)
    if not head:
        return []
    tokens: list[tuple[str | None, int | None]] = [(head.group(0), None)]
    pos = head.end()
    while pos < len(dotted):
        step = re.match(r"\.([A-Za-z_][A-Za-z0-9_]*)|\[([0-9]+)\]", dotted[pos:])
        if not step:
            return []
        if step.group(1) is not None:
            tokens.append((step.group(1), None))
        else:
            tokens.append((None, int(step.group(2))))
        pos += step.end()
    return tokens


def _entry_dict(node: Any) -> dict | None:
    """A provenance node (sidecar entry or inline ``{value, ...}`` node) read
    as a plain field dict, or None when the node isn't one."""
    if not isinstance(node, Mapping):
        return None
    if not ({"provenance", "source", "as_of", "accuracy",
             "uncertainty_pct"} & set(node)):
        return None
    return {k: node[k] for k in ("provenance", "source", "as_of", "accuracy",
                                 "uncertainty_pct") if k in node}


def _sidecar(doc: Mapping[str, Any]) -> dict[str, dict]:
    return {str(k): dict(v) for k, v in (doc.get("data_provenance") or {}).items()}


def _covers(key: str, path: str) -> bool:
    return path == key or path.startswith(key + ".") or path.startswith(key + "[")


def _ancestors(path: str) -> list[str]:
    """Path plus every proper prefix that is itself a path (longest first)."""
    out = [path]
    cur = path
    while True:
        cut = max(cur.rfind("."), cur.rfind("["))
        if cut <= 0:
            break
        cur = cur[:cut]
        out.append(cur)
    return out


def _section_default(doc: Mapping[str, Any], path: str) -> str | None:
    """Inheritance for bare numbers inside ``booster`` (BoosterModel.load's
    rule: section provenance, else document provenance)."""
    if path == "booster" or path.startswith("booster.") or path.startswith("booster["):
        boot = doc.get("booster")
        if isinstance(boot, Mapping) and "provenance" in boot:
            return str(boot["provenance"])
    return None


def _inherit(doc: Mapping[str, Any], path: str, entry: dict) -> dict:
    """Tag inheritance for a sidecar/inline merge: explicit tag wins, then
    booster section, then document provenance."""
    out = dict(entry)
    if "provenance" not in out:
        out["provenance"] = _section_default(doc, path) or str(doc.get("provenance", ""))
    return out


def dangling_sidecar_keys(doc: Mapping[str, Any]) -> list[str]:
    """``data_provenance`` keys that name no node in the document (typos in a
    provenance map are data bugs -- they silently un-tag a number)."""
    return sorted(k for k in (doc.get("data_provenance") or {})
                  if resolve_path(doc, k) is None)


def provenance_of(doc: Mapping[str, Any], path: str) -> Provenance:
    """Resolved provenance of one path (see module docstring for precedence).
    Raises ``KeyError`` when the number at ``path`` carries no tag anywhere."""
    entry = _resolve_entry(doc, path)
    if entry is None or "provenance" not in entry or not entry["provenance"]:
        raise KeyError(f"{path}: no provenance tag (inline node or data_provenance entry)")
    return Provenance(
        path=path,
        provenance=str(entry["provenance"]),
        source=str(entry.get("source", "")),
        as_of=str(entry.get("as_of", "")),
        accuracy=str(entry.get("accuracy", "")),
        uncertainty_pct=(float(entry["uncertainty_pct"])
                         if entry.get("uncertainty_pct") is not None else None),
    )


def _resolve_entry(doc: Mapping[str, Any], path: str) -> dict | None:
    """Merged entry covering ``path``: nearest inline booster node wins per
    field, then nearest sidecar entry, then section/document inheritance."""
    inline: dict = {}
    for anc in _ancestors(path):                      # longest prefix first
        node = resolve_path(doc, anc)
        if isinstance(node, Mapping) and "value" in node:
            got = _entry_dict(node)
            if got:
                inline = {**got, **inline}            # nearer wins
            break
    sidecar: dict = {}
    covering = [k for k in _sidecar(doc) if _covers(k, path)]
    if covering:
        sidecar = _sidecar(doc)[max(covering, key=len)]
    if not inline and not sidecar:
        fallback = _section_default(doc, path)
        return {"provenance": fallback} if fallback else None
    merged = {**sidecar, **{k: v for k, v in inline.items() if v is not None}}
    return _inherit(doc, path, merged)


def provenance_annotations(doc: Mapping[str, Any]) -> dict[str, Provenance]:
    """``{numeric_path: Provenance}`` for every numeric leaf in the document
    (the ``data_provenance`` subtree itself excluded); uncovered leaves are
    absent -- see :func:`uncovered_numeric_paths`."""
    out: dict[str, Provenance] = {}
    for path in _numeric_paths(doc):
        entry = _resolve_entry(doc, path)
        if entry and entry.get("provenance"):
            out[path] = provenance_of(doc, path)
    return out


def uncovered_numeric_paths(doc: Mapping[str, Any]) -> list[str]:
    """Numeric document paths with no provenance tag from any source. The
    catalog test asserts this is empty for every packaged vehicle file."""
    return sorted(p for p in _numeric_paths(doc)
                  if not (_resolve_entry(doc, p) or {}).get("provenance"))


def _numeric_paths(doc: Mapping[str, Any]) -> list[str]:
    out: list[str] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, Mapping):
            if "value" in node and isinstance(node["value"], (int, float)) \
                    and not isinstance(node["value"], bool) \
                    and _entry_dict(node) is not None:
                out.append(path)                    # inline node: the number IS the leaf
                return
            for key, child in node.items():
                if path == "" and key == "data_provenance":
                    continue
                walk(child, f"{path}.{key}" if path else str(key))
        elif isinstance(node, list):
            for i, child in enumerate(node):
                walk(child, f"{path}[{i}]")
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            out.append(path)

    walk(doc, "")
    return out


# ---------------------------------------------------------------------------
# uncertainties -> FL-304 Monte Carlo dispersions
# ---------------------------------------------------------------------------

def vehicle_dispersions(vehicle: str | Path):
    """The FL-304 :class:`~orbitopt.flightlab.recovery.montecarlo.Dispersions`
    implied by a vehicle file's ``uncertainty_pct`` tags: for each of thrust /
    Isp / Cd, the widest band the file states across the candidate paths.
    Returns ``(dispersions, notes)``; ``notes`` is empty (and the class
    defaults stand) when the file tags no MC-relevant uncertainty, so callers
    can record the substitution honestly."""
    from orbitopt.flightlab.recovery.montecarlo import Dispersions

    doc = load_validated(vehicle) if isinstance(vehicle, (str, Path)) else vehicle
    overrides: dict[str, float] = {}
    notes: list[str] = []
    for field, candidates in _MC_FIELDS:
        widths = []
        for path in candidates:
            entry = _resolve_entry(doc, path) or {}
            unc = entry.get("uncertainty_pct")
            if unc is not None:
                widths.append(float(unc))
        if widths:
            overrides[field] = max(widths)
    if overrides:
        notes.append(
            "FL-304 dispersions from vehicle-file uncertainty (FL-602): "
            + ", ".join(f"{k}={v:g}%" for k, v in sorted(overrides.items())))
    return Dispersions(**overrides), notes
