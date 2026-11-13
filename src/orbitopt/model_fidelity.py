"""Per-segment physics fidelity metadata for mission runs (V-05, epic #218).

The question this answers, without the reader having to guess: *which
physical effects were not modelled, and at what fidelity was each mission
phase actually computed?*

One :class:`MissionSegment` per mission phase that ran -- the pipeline
stages are the phases of a mission run (search on a patched-conic grid,
refine under tudatpy N-body, ...). Builders already report a
``physical_model`` per stage through the ``StageRecorder``; V-05 adds the
three metadata lists (``assumptions``, ``limitations``, ``not_modelled``)
to the same record, and this module derives and aggregates them.

Segments are derived from the pipeline result, not a parallel registry:
one source of truth, so a stage can never claim a model its stage record
does not carry. :func:`model_coverage` is the summary a mission report
embeds -- per-segment model + metadata plus the union of every effect the
run left unmodelled -- and :func:`format_coverage` renders it for
human-facing output (``PipelineResult.summary()``, ``orbitopt run``).
"""
from __future__ import annotations

from dataclasses import dataclass, field

# physical-model fidelity vocabulary, lowest to highest. Kept here (stdlib
# only) so the pipeline can import it at module level without a cycle.
FIDELITIES = ("none", "analytic", "patched-conic", "two-body", "high-fidelity")

# Standard names for the effects most mission phases leave out -- so the
# mission report says the same words for the same gap everywhere.
NOT_MODELLED_IMPULSIVE_BURNS = "finite_burn_thrust_arcs (burns are impulsive)"
NOT_MODELLED_SRP = "solar_radiation_pressure"
NOT_MODELLED_DRAG = "atmospheric_drag"
NOT_MODELLED_RELATIVITY = "relativistic_correction"
NOT_MODELLED_POINT_MASS_GRAVITY = "non-spherical_gravity (bodies are point masses)"
NOT_MODELLED_THIRD_BODIES = "third_body_gravity (Sun/Moon/planets)"


def _strings(values) -> tuple:
    return tuple(str(v) for v in (values or ()))


@dataclass(frozen=True)
class MissionSegment:
    """The physics fidelity metadata of one mission phase.

    ``not_modelled`` lists physical effects the phase's model excludes
    (empty only when there is nothing to disclose); ``assumptions`` are
    the things the model takes for granted (impulsive burns, ephemeris
    source, constant thrust...); ``limitations`` are what the phase's
    answer cannot be trusted for.
    """

    name: str
    physical_model: str
    fidelity: str
    status: str
    assumptions: tuple = field(default_factory=tuple)
    limitations: tuple = field(default_factory=tuple)
    not_modelled: tuple = field(default_factory=tuple)

    def __post_init__(self):
        if self.fidelity not in FIDELITIES:
            raise ValueError(f"unknown fidelity level {self.fidelity!r}; expected one of {FIDELITIES}")

    @classmethod
    def from_stage(cls, stage: dict) -> MissionSegment:
        """Build from a ``StageResult.to_dict()`` document (or the dataclass
        duck-type). Returns a segment for every stage, physics model or not:
        the record states what the phase was, even when that is nothing."""
        model = stage.get("physical_model") or {}
        return cls(
            name=str(stage.get("stage", "?")),
            physical_model=str(model.get("name", "")),
            fidelity=str(model.get("fidelity", "none")),
            status=str(stage.get("status", "not_run")),
            assumptions=_strings(stage.get("assumptions")),
            limitations=_strings(stage.get("limitations")),
            not_modelled=_strings(stage.get("not_modelled")),
        )

    @property
    def has_physics(self) -> bool:
        return bool(self.physical_model) and self.fidelity != "none"

    @property
    def modelled(self) -> bool:
        """Whether this segment contributes to the run's fidelity ladder
        (a physics model that actually ran)."""
        return self.has_physics and self.status == "completed"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "physical_model": self.physical_model,
            "fidelity": self.fidelity,
            "status": self.status,
            "assumptions": list(self.assumptions),
            "limitations": list(self.limitations),
            "not_modelled": list(self.not_modelled),
        }


def segments_from_stages(stages) -> list:
    """Every mission segment of a run, from stage dicts (``to_dict()``) or
    ``StageResult`` objects."""
    out = []
    for st in stages:
        doc = st.to_dict() if hasattr(st, "to_dict") else dict(st)
        out.append(MissionSegment.from_stage(doc))
    return out


def model_coverage(result) -> dict:
    """Aggregate a run's mission segments into the model-fidelity section of
    the mission report. ``result`` is a ``PipelineResult``, a pipeline-result
    document or a plain list of stage records.

    ``segments`` lists each phase's model, fidelity, assumptions,
    limitations and unmodelled effects -- the phases that ran with a
    physics model first, in lifecycle order. ``unmodelled_effects`` is the
    union over those segments: the direct answer to "which physical
    effects were not modelled in this run"."""
    stages = (result.stages if hasattr(result, "stages")
              else result["stages"] if isinstance(result, dict) and "stages" in result
              else result)
    segments = segments_from_stages(stages)
    physics = [s for s in segments if s.modelled]
    unmodelled = sorted({e for s in physics for e in s.not_modelled})
    return {
        "segments": [s.to_dict() for s in physics + [s for s in segments if not s.modelled]],
        "fidelity_ladder": {s.name: s.fidelity for s in physics},
        "highest_fidelity": max((s.fidelity for s in physics), key=FIDELITIES.index, default="none"),
        "unmodelled_effects": unmodelled,
    }


def format_coverage(coverage: dict, indent: str = "  ") -> list:
    """Human-readable lines for the coverage of ``model_coverage()``: one
    per physics segment plus the unmodelled-effects union. Empty when the
    run recorded no physics model at all."""
    lines = []
    for s in coverage["segments"]:
        if not s["assumptions"] and not s["limitations"] and not s["not_modelled"]:
            continue
        lines.append(f"{indent}{s['name']}: {s['physical_model']}")
        for key, label in (("assumptions", "assumes"), ("limitations", "limitation"),
                           ("not_modelled", "not modelled")):
            for item in s[key]:
                lines.append(f"{indent}  {label}: {item}")
    if coverage["unmodelled_effects"]:
        lines.append(f"{indent}unmodelled physics (union over segments): "
                     f"{', '.join(coverage['unmodelled_effects'])}")
    return lines
