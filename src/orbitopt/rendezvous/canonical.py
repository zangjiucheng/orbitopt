"""The canonical sealed rendezvous plan (E12-08, task #410).

E12-05 re-planned with the unsafe E12-03 optimizer, the safety gate re-planned
constrained, Mission Studio replayed the unsafe plan again, and the Monte
Carlo flew a plan the benchmark had just re-derived -- four near-identical
trajectories, none of them provably the same artifact. This module ends that:
:func:`canonical_plan` runs the safety-constrained planner (E12-04) once and
seals the result into a ``SealedRendezvousPlan`` carrying the four identity
digests -- over the full mission config, over the initial pair states, over
the maneuver sequence, and a plan digest combining all three. Every
downstream consumer (safety report, TudatPy re-flight, studio replay, robust
campaign) consumes the sealed plan and records its digest, so the reports can
be checked to describe the SAME trajectory, and a mismatched digest is a hard
failure, not a shrug.

The digests are pure functions of ``(config, plan)`` (and of the config
alone for the initial states), canonical-JSON then sha256, so a consumer that
holds the config and the maneuver sequence can recompute the identity without
the sealed file; the file exists to pin one artifact on disk and to carry the
E12-04 acceptance record (verification + safety reports) alongside it.

    from orbitopt.rendezvous import canonical_plan, write_sealed_plan
    sealed = canonical_plan(load_reference_config())
    write_sealed_plan(sealed, "runs/leo-rdv-001.sealed.json")
    same = read_sealed_plan("runs/leo-rdv-001.sealed.json", cfg)  # verifies
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import RendezvousMissionConfig
from .planner import Maneuver, RendezvousPlan, initial_states, plan_rendezvous
from .verify import RendezvousVerification, verify_plan

#: Sealed-document identity (schema tag of this module's own document).
SEALED_KIND = "orbitopt.rendezvous.sealed-plan"
SEALED_SCHEMA_VERSION = "1.0"


class PlanSealError(ValueError):
    """A sealed plan does not match the config it is consumed under, or the
    document's own digests do not match its contents (tampered or stale)."""


def _canonical_sha256(doc: Any) -> str:
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode("utf-8")).hexdigest()


def _as_plain(value: Any) -> Any:
    """JSON-safe plain data, float-exact (``float(x)`` of a numpy scalar
    round-trips bit for bit through ``json.dumps``/``repr``)."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _as_plain(getattr(value, f.name))
                for f in dataclasses.fields(value)}
    if isinstance(value, tuple):
        return [_as_plain(x) for x in value]
    if isinstance(value, list):
        return [_as_plain(x) for x in value]
    if isinstance(value, dict):
        return {str(k): _as_plain(v) for k, v in value.items()}
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    return float(value)


def canonical_config_sha256(cfg: RendezvousMissionConfig) -> str:
    """Digest of the WHOLE parsed mission config -- every field, the E12-04
    ``safety`` block included (E12-06's run-identity digest predating this
    omitted it, so two configs differing only in safety constraints used to
    collide; that is exactly what this digest must distinguish)."""
    return _canonical_sha256(_as_plain(cfg))


def initial_states_sha256(cfg: RendezvousMissionConfig) -> str:
    """Digest of the pair state at the shared epoch t0 the mission starts
    from: the four GCRS vectors of :func:`orbitopt.rendezvous.planner.
    initial_states`, chief position/velocity then deputy position/velocity."""
    states = [[float(x) for x in vec] for vec in initial_states(cfg)]
    return _canonical_sha256(states)


def maneuvers_sha256(plan: RendezvousPlan) -> str:
    """Digest of the maneuver sequence -- kind, epoch and inertial delta-v of
    every burn, in order (the same list ``RendezvousPlan.to_dict`` reports;
    identical to E12-05's worker-side ``maneuvers_sha256``)."""
    return _canonical_sha256([m.to_dict() for m in plan.maneuvers])


def rendezvous_plan_digest(cfg: RendezvousMissionConfig, plan: RendezvousPlan) -> str:
    """THE plan digest (E12-08): sha256 over the canonical triple of config
    digest, initial-states digest and maneuver-sequence digest. Every
    consumer recomputes it from what it actually has, and every report
    records what it consumed."""
    return _canonical_sha256({
        "config_sha256": canonical_config_sha256(cfg),
        "initial_states_sha256": initial_states_sha256(cfg),
        "maneuvers_sha256": maneuvers_sha256(plan),
    })


def _plan_to_full_dict(plan: RendezvousPlan) -> dict:
    """Every field of the plan (``to_dict`` drops the gate/arrival chief
    states); the sealed document restores the plan without loss."""
    return {
        "mission_id": plan.mission_id,
        "frame": plan.frame,
        "drift_end_s": plan.drift_end_s,
        "transfer_tof_s": plan.tof_s,
        "arrival_epoch_s": plan.arrival_epoch_s,
        "maneuvers": [m.to_dict() for m in plan.maneuvers],
        "total_delta_v_m_s": plan.total_delta_v_m_s,
        "delta_v_budget_m_s": plan.delta_v_budget_m_s,
        "chief_at_gate_r_m": [float(x) for x in plan.chief_at_gate_r_m],
        "chief_at_gate_v_m_s": [float(x) for x in plan.chief_at_gate_v_m_s],
        "chief_at_arrival_r_m": [float(x) for x in plan.chief_at_arrival_r_m],
        "chief_at_arrival_v_m_s": [float(x) for x in plan.chief_at_arrival_v_m_s],
        "hold_point_rel_lvlh_m": [float(x) for x in plan.hold_point_rel_lvlh_m],
    }


def _plan_from_full_dict(doc: dict) -> RendezvousPlan:
    maneuvers = tuple(
        Maneuver(str(m["kind"]), float(m["epoch_s"]),
                 tuple(float(v) for v in m["delta_v_eci_m_s"]))
        for m in doc["maneuvers"])
    return RendezvousPlan(
        mission_id=str(doc["mission_id"]), frame=str(doc["frame"]),
        drift_end_s=float(doc["drift_end_s"]), tof_s=float(doc["transfer_tof_s"]),
        maneuvers=maneuvers, total_delta_v_m_s=float(doc["total_delta_v_m_s"]),
        delta_v_budget_m_s=float(doc["delta_v_budget_m_s"]),
        chief_at_gate_r_m=tuple(float(x) for x in doc["chief_at_gate_r_m"]),
        chief_at_gate_v_m_s=tuple(float(x) for x in doc["chief_at_gate_v_m_s"]),
        chief_at_arrival_r_m=tuple(float(x) for x in doc["chief_at_arrival_r_m"]),
        chief_at_arrival_v_m_s=tuple(float(x) for x in doc["chief_at_arrival_v_m_s"]),
        hold_point_rel_lvlh_m=tuple(float(x) for x in doc["hold_point_rel_lvlh_m"]))


@dataclass(frozen=True)
class SealedRendezvousPlan:
    """One canonical plan with its identity digests and the E12-04 acceptance
    record (E12-03 verification, and the whole-trajectory safety report as
    its document) that qualified it. Downstream components consume this and
    publish ``plan_digest``; ``safety`` is ``None`` only for a config with no
    ``safety`` block (the E12-03 pipeline unchanged)."""

    plan: RendezvousPlan
    verification: RendezvousVerification
    safety: dict[str, Any] | None
    config_sha256: str
    initial_states_sha256: str
    maneuvers_sha256: str
    plan_digest: str

    @property
    def passed(self) -> bool:
        return (self.verification.passed
                and (self.safety is None or bool(self.safety.get("passed"))))

    def to_document(self) -> dict:
        return {
            "kind": SEALED_KIND,
            "schema_version": SEALED_SCHEMA_VERSION,
            "canonical_planner": "orbitopt.rendezvous.canonical_plan "
                                 "(E12-04 plan_rendezvous_safe when the config has a safety "
                                 "block; E12-03 plan_rendezvous otherwise)",
            "mission_id": self.plan.mission_id,
            "config_sha256": self.config_sha256,
            "initial_states_sha256": self.initial_states_sha256,
            "maneuvers_sha256": self.maneuvers_sha256,
            "plan_digest": self.plan_digest,
            "plan": _plan_to_full_dict(self.plan),
            "verification": self.verification.to_dict(),
            "safety": self.safety,
            "passed": self.passed,
        }


def seal_plan(cfg: RendezvousMissionConfig, plan: RendezvousPlan,
              verification: RendezvousVerification,
              safety: dict | None = None) -> SealedRendezvousPlan:
    """Seal an already-planned-and-verified mission: compute the four digests
    over the inputs as given and cross-check them against the verification
    record's identity. Raises :class:`PlanSealError` when the plan is not the
    plan for this config (digest mismatch)."""
    sealed = SealedRendezvousPlan(
        plan=plan, verification=verification, safety=safety,
        config_sha256=canonical_config_sha256(cfg),
        initial_states_sha256=initial_states_sha256(cfg),
        maneuvers_sha256=maneuvers_sha256(plan),
        plan_digest=rendezvous_plan_digest(cfg, plan))
    verify_seal(cfg, sealed)
    return sealed


def canonical_plan(cfg: RendezvousMissionConfig) -> SealedRendezvousPlan:
    """Produce THE canonical plan for ``cfg``: the safety-constrained
    optimum of :func:`orbitopt.rendezvous.safety.plan_rendezvous_safe` when
    the config carries a ``safety`` block (raises what it raises when nothing
    in the fixed windows is safe -- there is then no canonical plan, and no
    consumer may show green), else the plain E12-03 optimum verified by
    :func:`orbitopt.rendezvous.verify.verify_plan`. Deterministic: the same
    config seals to the same ``plan_digest`` every run.

    This is the single planning entry point for E12-04 safety reports, the
    E12-05 TudatPy re-flight, the E12-06 Mission Studio replay and the E12-07
    Monte Carlo campaign (E12-08): each consumes this and records the digest.
    """
    if cfg.safety is not None:
        from .safety import plan_rendezvous_safe  # local: safety imports this module
        result = plan_rendezvous_safe(cfg)
        return seal_plan(cfg, result.plan, result.verification, result.safety.to_dict())
    plan = plan_rendezvous(cfg)
    verification = verify_plan(cfg, plan)
    if not verification.passed:
        failed = sorted(n for n, ok in verification.checks.items() if not ok)
        raise PlanSealError(
            f"{cfg.mission_id}: the unconstrained optimum fails {failed}; nothing to seal")
    return seal_plan(cfg, plan, verification, None)


def verify_seal(cfg: RendezvousMissionConfig, sealed: SealedRendezvousPlan) -> None:
    """The consumption guard: recompute every digest from ``cfg`` (and the
    sealed maneuver sequence) and require agreement with what the document
    claims. Raises :class:`PlanSealError` naming the first mismatch -- a
    consumer running under a changed config, or a tampered/stale document,
    must fail here, not report green."""
    recomputed = {
        "config digest": (canonical_config_sha256(cfg), sealed.config_sha256),
        "initial-states digest": (initial_states_sha256(cfg), sealed.initial_states_sha256),
        "maneuver digest": (maneuvers_sha256(sealed.plan), sealed.maneuvers_sha256),
        "plan digest": (rendezvous_plan_digest(cfg, sealed.plan), sealed.plan_digest),
    }
    for what, (expected, claimed) in recomputed.items():
        if expected != claimed:
            raise PlanSealError(
                f"{cfg.mission_id}: sealed plan {what} mismatch -- consumed "
                f"{claimed[:16]}..., recomputed {expected[:16]}... (the plan was "
                f"sealed against different inputs than these)")
    if sealed.plan.mission_id != cfg.mission_id:
        raise PlanSealError(
            f"sealed plan is for {sealed.plan.mission_id!r}, config declares "
            f"{cfg.mission_id!r}")


def consume_sealed_plan(sealed: SealedRendezvousPlan,
                        cfg: RendezvousMissionConfig) -> RendezvousPlan:
    """What every downstream component does with the sealed plan: verify the
    seal against the config it is running under, then hand out the plan."""
    verify_seal(cfg, sealed)
    return sealed.plan


def write_sealed_plan(sealed: SealedRendezvousPlan, path: str | Path) -> Path:
    """Write the sealed document (deterministic: no timestamps -- the plan
    digest is the identity, so regenerating the file reproduces it)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sealed.to_document(), indent=1, sort_keys=False) + "\n",
                   encoding="utf-8")
    return out


def read_sealed_plan(path: str | Path, cfg: RendezvousMissionConfig) -> SealedRendezvousPlan:
    """Read a sealed document and re-verify it under ``cfg`` (digest by
    digest). Any mismatch -- wrong kind, edited digests, edited maneuvers, a
    plan sealed for another config -- raises :class:`PlanSealError`."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    return sealed_from_document(doc, cfg)


def sealed_from_document(doc: dict, cfg: RendezvousMissionConfig) -> SealedRendezvousPlan:
    """Rebuild and verify a sealed plan from its document dict (what
    :func:`read_sealed_plan` does after parsing; public so a worker that got
    the document over the wire verifies it the same way)."""
    if doc.get("kind") != SEALED_KIND or doc.get("schema_version") != SEALED_SCHEMA_VERSION:
        raise PlanSealError(
            f"document is {doc.get('kind')!r}/{doc.get('schema_version')!r}, not "
            f"{SEALED_KIND!r}/{SEALED_SCHEMA_VERSION}")
    for field in ("config_sha256", "initial_states_sha256", "maneuvers_sha256", "plan_digest"):
        if not isinstance(doc.get(field), str):
            raise PlanSealError(f"sealed document is missing its {field}")
    plan = _plan_from_full_dict(doc["plan"])
    vdoc = doc["verification"]
    verification = RendezvousVerification(
        mission_id=str(vdoc["mission_id"]), frame=str(vdoc["frame"]),
        drift_end_s=float(vdoc["drift_end_s"]), arrival_epoch_s=float(vdoc["arrival_epoch_s"]),
        arrival_rel_lvlh_m=tuple(float(x) for x in vdoc["arrival_rel_lvlh_m"]),
        arrival_rel_lvlh_rate_m_s=tuple(float(x) for x in vdoc["arrival_rel_lvlh_rate_m_s"]),
        position_error_m=float(vdoc["position_error_m"]),
        relative_speed_m_s=float(vdoc["relative_speed_m_s"]),
        total_delta_v_m_s=float(vdoc["total_delta_v_m_s"]),
        delta_v_budget_m_s=float(vdoc["delta_v_budget_m_s"]),
        delta_v_margin_m_s=float(vdoc["delta_v_margin_m_s"]),
        checks={k: bool(v) for k, v in vdoc["checks"].items()},
        _maneuvers=plan.maneuvers)
    sealed = SealedRendezvousPlan(
        plan=plan, verification=verification,
        safety=doc.get("safety"),
        config_sha256=str(doc["config_sha256"]),
        initial_states_sha256=str(doc["initial_states_sha256"]),
        maneuvers_sha256=str(doc["maneuvers_sha256"]),
        plan_digest=str(doc["plan_digest"]))
    verify_seal(cfg, sealed)
    return sealed
