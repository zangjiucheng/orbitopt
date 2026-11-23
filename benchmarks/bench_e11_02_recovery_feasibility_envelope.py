"""E11-02 (task #309): the recovery feasibility envelope benchmark.

E10-03 closed with the planner spending its whole search (and an entire
evolution-strategy budget, DC8 168/168 ``planning_infeasible``, robust arm
trained to 0/11) inside a region of separation states no script of its
family flies. E11-01 measured the grid-fin authority DC8 actually has.
E11-02 (``orbitopt.flightlab.recovery.feasibility``) turns that into a
screen: from the separation state alone, one reference rollout (zero search
fevals) decides IN or OUT of the recovery envelope and predicts the flight
reason. This benchmark validates the screen five ways:

A  reference points: the envelope verdict at each rig's nominal separation
   state, with the predicted reason cross-checked against the unscreened
   planner verdict (sealed DC7 / Starbase recipes, the E11-01 DC8 grid-fin
   mission, and the E10-03-era DC7 recipe flown on the DC8 vehicle).
B  boundary maps in separation-state space: offset grids (downrange x speed
   at the hot-staging handoff) graded over the envelope; the severity field
   and its 1.0 contour (the boundary) are plotted as PNGs.
C  hold-out agreement: the E10-03 hold-out grid (4 arms x 8 seeds x 21 =
   672 dispersed samples) re-flown twice per sample -- unscreened (the truth,
   the E10-03 worker verbatim) and screened (the prediction) -- into a
   confusion matrix per arm; false rejects (screen OUT, flight landed) must
   be zero, and truth counts must reproduce the committed E10-03 numbers.
D  twin boundary reproduction: the E10-07 independent 6-DOF twin re-flies
   the boundary pair C selects (one IN, one OUT sample, closest pair across
   the boundary) plus per-rig baselines that calibrate the twin-vs-planner
   bands, on the twin's own aerothermal line and fin model.
E  search budget: on an OUT sample a full Nelder-Mead search (fevals counted
   through a search wrapper) vs the screened call, and on an IN sample proof
   the screen lets the search run.

Run (repo root):
    python3 benchmarks/bench_e11_02_recovery_feasibility_envelope.py \
        --parts A,B,C,D,E --workers 8 --cache /tmp/e1102-cache.json
Writes docs/benchmarks/e11_02_recovery_feasibility_envelope.{json,md} and
docs/benchmarks/figures/e11_02_boundary_*.png.
"""
from __future__ import annotations

import argparse
import datetime as dtm
import json
import math
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
VEH_DIR = REPO / "src/orbitopt/data/flightlab/vehicles"
SCN_DIR = REPO / "src/orbitopt/data/flightlab/scenarios"
REF_DC7 = REPO / "tests/data/mission/reference/e5-08-new-vehicle-dc7"
REF_SB = REPO / "tests/data/mission/reference/e5-08-new-vehicle-starbase"
E10_03_REPORT = REPO / "docs/benchmarks/e10_03_robust_rtls_closure.json"
E10_07_REPORT = REPO / "docs/benchmarks/e10_07_full_recovery_reflight.json"
OUT_JSON = REPO / "docs/benchmarks/e11_02_recovery_feasibility_envelope.json"
OUT_MD = REPO / "docs/benchmarks/e11_02_recovery_feasibility_envelope.md"
FIG_DIR = REPO / "docs/benchmarks/figures"

#: the E10-03 hold-out stream, reproduced exactly (same seeds, same draws)
HOLDOUT_SEEDS = [501, 502, 503, 504, 505, 506, 507, 508]
N_PER_SEED = 21
ARMS = ("dc7_pinned", "starbase_pinned", "dc8_robust", "dc8_binary")
RIG_OF_ARM = {"dc7_pinned": "dc7", "starbase_pinned": "starbase",
              "dc8_robust": "dc8", "dc8_binary": "dc8"}

#: the E11-01 winner schedule (task #308): 60 deg alpha above the 70 km gate
FIN_SPEC: dict[str, Any] = {"mode": "schedule", "alpha_deg": 60.0,
                            "schedule": [[70000.0, 0.0, 60.0]]}
DC8_FINS = dict(entry_burns=0, boostback_cutoff="iip",
                landing_need_model="closed_loop", corridor_aim_margin_m=7000.0,
                entry_q_max_pa=250_000.0, entry_g_max=12.0,
                landing_reserve_margin_kg=2500.0, entry_aero=dict(FIN_SPEC),
                horizon_s=900.0, dt_opt=0.5, dt_final=0.25)
DC8_FINS_SCRIPT = (21.2, -2.0, 1.0, 0.0, 0.0)

#: separation-state offsets for the boundary maps (km-class dispersion
#: tails, the scale E10-03's own dispersion model draws at)
GRID_DR = tuple(float(x) for x in np.linspace(-9000.0, 9000.0, 7))
GRID_V = tuple(float(x) for x in np.linspace(-60.0, 60.0, 7))

TWIN_BASE_BANDS = {"q": 20.0, "g": 2.0, "speed": 40.0,
                   "tilt": 3.0, "miss": 1500.0, "prop": 3.0}


# --------------------------------------------------------------------------
# rig contexts (built once per worker process)
# --------------------------------------------------------------------------
_CTX: dict[str, dict[str, Any]] = {}


def _rig(name: str) -> dict[str, Any]:
    """The shared parts of one rig: target, recipes, nominal separation
    state (ECEF), vehicle, and the E10-03 dispersion dictionary."""
    from orbitopt.flightlab.contract import load_target
    from orbitopt.flightlab.recovery.propagator import BoosterModel, _to_ecef_state
    from orbitopt.flightlab.recovery.pipeline import landing_reserve_booster
    from orbitopt.flightlab.recovery.uncertainty import UncertaintyModel
    from orbitopt.flightlab.vehicle_catalog import vehicle_dispersions
    from orbitopt.mission.coupled_branches import (booster_state_from_run,
                                                   separation_handoff)

    if name in _CTX:
        return _CTX[name]
    recipe_path = (REF_SB / "recovery-script.json") if name == "starbase" \
        else (REF_DC7 / "recovery-script.json")
    scenario = (SCN_DIR / "leo_starship_class_rtls_e508_starbase.yaml"
                if name == "starbase"
                else SCN_DIR / "leo_starship_class_rtls_e508_dc7.yaml")
    vehicle_path = {"dc7": "starship_class_rtls_dc7", "starbase": "starship_class_rtls_starbase",
                    "dc8": "starship_class_rtls_dc8", "dc8_fins": "starship_class_rtls_dc8"}[name]
    target = load_target(scenario)
    vehicle = BoosterModel.load(VEH_DIR / f"{vehicle_path}.yaml")
    booster = landing_reserve_booster(vehicle, target, "target")
    recipe = json.loads(recipe_path.read_text())["recipe"]
    ctx: dict[str, Any] = {"target": target, "recipe": dict(recipe),
                           "booster": booster}
    if name in ("dc7", "starbase"):
        from orbitopt.flightlab.contract import read_run
        run_dir = (REF_SB if name == "starbase" else REF_DC7) / "mission-optimized/launch/ascent"
        ctx["ascent"] = read_run(run_dir)
        sep = booster_state_from_run(ctx["ascent"], separation_handoff(ctx["ascent"]))
        if sep.frame != "ECEF":
            r_e, v_e = _to_ecef_state(sep)
            sep = replace(sep, frame="ECEF", position_m=tuple(float(c) for c in r_e),
                          velocity_m_s=tuple(float(c) for c in v_e))
        ctx["sep"] = sep
    else:  # DC8: the sealed E10-07 separation handoff (digest-verified)
        from orbitopt.flightlab.handoff import HandoffState, handoff_digest
        from orbitopt.flightlab.recovery.propagator import BoosterState
        doc = json.loads(E10_07_REPORT.read_text())
        st, digest = doc["handoff"]["state"], doc["handoff"]["digest"]
        ctx["ascent"] = read_run_ctx(REF_DC7 / "mission-optimized/launch/ascent")
        sep = BoosterState(t_s=float(st["t_s"]),
                           epoch_utc=dtm.datetime.fromisoformat(str(st["epoch_utc"])),
                           frame="ECEF",
                           position_m=tuple(float(c) for c in st["position_m"]),
                           velocity_m_s=tuple(float(c) for c in st["velocity_m_s"]),
                           mass_kg=float(st["mass_kg"]))
        hs = HandoffState.of(sep, from_phase="separation", to_phase="boostback",
                             dry_mass_kg=vehicle.dry_mass_kg)
        if handoff_digest(hs) != digest:
            raise RuntimeError("DC8 start state does not match the sealed E10-07 digest")
        ctx["sep"] = sep
    disp_path = (VEH_DIR / "starship_class_rtls_dc7.yaml" if name in ("dc8", "dc8_fins")
                 else VEH_DIR / f"{vehicle_path}.yaml")
    disp, _notes = vehicle_dispersions(disp_path)
    ctx["uncertainty"] = UncertaintyModel.from_dispersions(disp).to_dict()
    _CTX[name] = ctx
    return ctx


def read_run_ctx(p: Path):
    from orbitopt.flightlab.contract import read_run
    return read_run(p)


def _spec_for(rig: str, arm: str | None = None,
              script: Sequence[float] | None = None) -> dict[str, Any]:
    """The envelope spec (plain dict) one flight should be screened with:
    the limits the planner call itself carries, with the flown script as the
    reference mission."""
    from orbitopt.flightlab.recovery.feasibility import EnvelopeSpec
    from orbitopt.flightlab.recovery.pipeline import _landing_limits_for_target
    if rig == "dc8_fins":
        return EnvelopeSpec.from_mapping({**DC8_FINS, "name": "dc8_fins_a60",
                                          "reference_script": list(DC8_FINS_SCRIPT)}).to_dict()
    # fly_recovery injects the closed-loop forecast's landing limits from the
    # target (pipeline.py:325); the spec must carry them too, or the forecast
    # grades a heavier burn than the flight flies.
    limits = _landing_limits_for_target(_rig(rig)["target"])
    if rig == "dc8":  # the E10-03 era: the DC7 recipe's limits, graded reserve 0
        r = dict(_rig("dc8")["recipe"])
        r.update({"landing_reserve_margin_kg": 0.0, "landing_divert_budget_m": None,
                  "landing_limits": limits, "name": "dc8_e1003_era"})
        if script is not None:
            return EnvelopeSpec.from_recipe(r, script=script).to_dict()
        return EnvelopeSpec.from_recipe(r).to_dict()
    r = dict(_rig(rig)["recipe"])
    r.update({"name": rig, "landing_limits": limits})
    return EnvelopeSpec.from_recipe(r).to_dict()


def _dispersed_state(job: Mapping[str, Any]):
    """Rebuild one separation sample: the E10-03 worker pipeline verbatim
    (draw, engine/drag scaling, physics, ECEF handoff, dispersion shift), or
    just fixed offsets over the nominal state when ``draw`` is null."""
    from orbitopt.flightlab.recovery.pipeline import dispersed_separation_state
    from orbitopt.flightlab.recovery.uncertainty import UncertaintyModel

    rig = _rig(str(job["rig"]))
    sep, booster = rig["sep"], rig["booster"]
    vehicle = replace(booster)
    draw: dict[str, float] = {}
    if job.get("draw") is not None:
        draw = UncertaintyModel.from_dict(rig["uncertainty"]).draw(
            np.random.default_rng([int(job["draw"][0]), int(job["draw"][1])]))
        eng = vehicle.engine
        vehicle = replace(
            vehicle,
            engine=replace(eng,
                           thrust_vacuum_n=eng.thrust_vacuum_n * draw.get("thrust_factor", 1.0),
                           isp_sea_level_s=eng.isp_sea_level_s * draw.get("isp_factor", 1.0),
                           isp_vacuum_s=eng.isp_vacuum_s * draw.get("isp_factor", 1.0)),
            cd_mach=tuple((m, c * draw.get("cd_factor", 1.0)) for m, c in vehicle.cd_mach))
    from orbitopt.flightlab.recovery.propagator import BoosterPhysics
    physics = BoosterPhysics(density_scale=draw.get("density_scale", 1.0),
                             wind_east_m_s=draw.get("wind_east_m_s", 0.0),
                             wind_north_m_s=draw.get("wind_north_m_s", 0.0),
                             wind_up_m_s=draw.get("wind_up_m_s", 0.0))
    off = dict(job.get("offsets") or {})
    sep = dispersed_separation_state(
        sep,
        altitude_m=float(off.get("sep_altitude_m", draw.get("sep_altitude_m", 0.0))),
        downrange_m=float(off.get("sep_downrange_m", draw.get("sep_downrange_m", 0.0))),
        crossrange_m=float(off.get("sep_crossrange_m", draw.get("sep_crossrange_m", 0.0))),
        speed_m_s=float(off.get("sep_speed_m_s", draw.get("sep_speed_m_s", 0.0))),
        heading_deg=float(off.get("sep_heading_deg", draw.get("sep_heading_deg", 0.0))))
    return rig, sep, vehicle, physics


# --------------------------------------------------------------------------
# workers (top level: fork pool)
# --------------------------------------------------------------------------
def _worker(job: dict[str, Any]) -> dict[str, Any]:
    kind = job["kind"]
    t0 = time.time()
    out: dict[str, Any] = {k: job[k] for k in
                           ("part", "key", "rig", "arm", "seed", "idx")
                           if k in job}
    try:
        if kind == "verdict":
            out.update(_do_verdict(job))
        elif kind == "grid":
            out.update(_do_grid(job))
        elif kind == "holdout":
            out.update(_do_holdout(job))
        elif kind == "twin":
            out.update(_do_twin(job))
        elif kind == "budget":
            out.update(_do_budget(job))
        else:
            raise ValueError(kind)
    except Exception as exc:  # a broken sample must never erase the run
        out["error"] = f"{type(exc).__name__}: {exc}"
    out["dur_s"] = round(time.time() - t0, 1)
    return out


def _do_verdict(job: dict[str, Any]) -> dict[str, Any]:
    """Envelope verdict at the nominal state + unscreened/screened planner
    cross-checks (part A)."""
    from orbitopt.flightlab.recovery import RecoveryTarget, plan_boostback_entry
    from orbitopt.flightlab.recovery.feasibility import EnvelopeSpec, envelope_verdict

    rig, sep, vehicle, physics = _dispersed_state(job)
    spec = EnvelopeSpec.from_mapping(job["spec"])
    rt = RecoveryTarget.from_scenario(rig["target"].to_dict())
    v = envelope_verdict(sep, vehicle, rt, spec, physics=physics)
    opts = spec.planner_kwargs()
    unscreened = plan_boostback_entry(sep, vehicle, rt,
                                      initial_guess=list(spec.reference_script),
                                      search="keep", dt_opt=spec.dt_opt,
                                      dt_final=spec.dt_final, verdict_physics=physics,
                                      **opts)
    screened = plan_boostback_entry(sep, vehicle, rt,
                                    initial_guess=list(spec.reference_script),
                                    search="keep", dt_opt=spec.dt_opt,
                                    dt_final=spec.dt_final, verdict_physics=physics,
                                    feasibility_envelope=spec.to_dict(), **opts)
    return {"verdict": v.to_dict(),
            "unscreened_feasible": bool(unscreened.feasible),
            "unscreened_codes": list(unscreened.reason_codes),
            "screened_codes": list(screened.reason_codes),
            "screened_rejected": screened.screen is not None
            and not screened.screen.get("in_envelope", True),
            "prediction_matches": (list(unscreened.reason_codes)[:1]
                                   == list(screened.reason_codes)[:1])}


def _do_grid(job: dict[str, Any]) -> dict[str, Any]:
    rig, sep, vehicle, physics = _dispersed_state(job)
    from orbitopt.flightlab.recovery import RecoveryTarget
    from orbitopt.flightlab.recovery.feasibility import EnvelopeSpec, envelope_verdict
    spec = EnvelopeSpec.from_mapping(job["spec"])
    rt = RecoveryTarget.from_scenario(rig["target"].to_dict())
    v = envelope_verdict(sep, vehicle, rt, spec, physics=physics)
    d = v.to_dict()
    probe = d.pop("probe", {})
    return {"verdict": d, "probe": {k: probe.get(k) for k in
                                    ("miss_m", "tilt_deg", "speed_m_s",
                                     "entry_q_pa", "entry_g")}}


def _do_holdout(job: dict[str, Any]) -> dict[str, Any]:
    """One E10-03 hold-out sample twice: truth (unscreened fly_recovery,
    the committed E10-03 worker semantics) and prediction (same options +
    the envelope screen). The screen is the planner-internal *bool* path,
    which grades against the exact kwargs the planner call itself carries
    (limits, guidance, divert, dt) -- the prediction therefore agrees with
    the flight verdict by construction on the probe it grades."""
    from orbitopt.flightlab.recovery.pipeline import fly_recovery

    rig, sep, vehicle, physics = _dispersed_state(job)
    options = dict(job["options"])
    truth = fly_recovery(rig["ascent"], rig["target"], job["vehicle_path"],
                         sep_state=sep, booster=vehicle, physics=physics,
                         plan_options=dict(options))
    rec: dict[str, Any] = {"truth_landed": bool(truth.success),
                           "truth_reason": ("LANDED" if truth.success
                                           else str(truth.reason_codes[0]))}
    pred = fly_recovery(rig["ascent"], rig["target"], job["vehicle_path"],
                        sep_state=sep, booster=vehicle, physics=physics,
                        plan_options={**options, "feasibility_envelope": True})
    screen = getattr(pred.plan, "screen", None) or {}
    rec.update({
        "pred_landed": bool(pred.success),
        "pred_reason": ("LANDED" if pred.success else str(pred.reason_codes[0])),
        "pred_in_envelope": bool(screen.get("in_envelope", True)),
        "pred_governing": screen.get("governing_condition"),
        "pred_code": screen.get("governing_reason"),
        "pred_worst_severity": screen.get("worst_severity"),
        "pred_violated": list(screen.get("violated_codes", ())),
        "screen_rejected": bool(screen) and not screen.get("in_envelope", True)})
    return rec


def _do_twin(job: dict[str, Any]) -> dict[str, Any]:
    import orbitopt.verify.full_recovery_reflight as V
    from orbitopt.flightlab.recovery import RecoveryTarget
    from orbitopt.flightlab.recovery.feasibility import EnvelopeSpec

    rig, sep, vehicle, _physics = _dispersed_state(job)
    rt = RecoveryTarget.from_scenario(rig["target"].to_dict())
    spec = EnvelopeSpec.from_mapping(job["spec"])
    r = V.twin_envelope_reflight(sep, vehicle, rt, spec,
                                 bands=dict(job.get("bands") or TWIN_BASE_BANDS),
                                 main_dt_final=spec.dt_final)
    checks = [{k: c[k] for k in ("quantity", "main", "twin", "delta", "band",
                                 "band_unit", "main_side", "twin_side", "ok")}
              for c in r["checks"]]
    return {"verdict": r["verdict"], "checks": checks,
            "boundary_agreement": bool(r["boundary_agreement"]),
            "main_model": r["main_model"],
            "twin": {k: r["twin"].get(k) for k in
                     ("entry_max_q_pa", "entry_max_proper_g", "h0_speed_m_s",
                      "h0_tilt_deg", "h0_mass_kg", "h0_cross_t_s", "cut")}}


def _do_budget(job: dict[str, Any]) -> dict[str, Any]:
    """Full Nelder-Mead vs screened planning on one state, with the search
    fevals counted through the planner's own ``search`` hook."""
    from scipy.optimize import minimize

    from orbitopt.flightlab.recovery import RecoveryTarget, plan_boostback_entry
    from orbitopt.flightlab.recovery.feasibility import EnvelopeSpec

    rig, sep, vehicle, physics = _dispersed_state(job)
    rt = RecoveryTarget.from_scenario(rig["target"].to_dict())
    spec = EnvelopeSpec.from_mapping(job["spec"])
    calls = {"n": 0}

    def search(obj, x0):  # mirrors the planner's default Nelder-Mead call
        def counted(x):
            calls["n"] += 1
            return obj(x)
        return np.asarray(minimize(counted, x0, method="Nelder-Mead",
                                   options=dict(maxfev=int(job["max_fevals"]),
                                                xatol=0.5, fatol=1e-4)).x)
    kwargs = dict(spec.planner_kwargs())
    t_full = time.time()
    full = plan_boostback_entry(sep, vehicle, rt,
                                initial_guess=list(spec.reference_script),
                                search=search, max_fevals=int(job["max_fevals"]),
                                dt_opt=spec.dt_opt, dt_final=spec.dt_final,
                                verdict_physics=physics, **kwargs)
    full_s = time.time() - t_full
    calls["n"] = 0
    t_screen = time.time()
    screened = plan_boostback_entry(sep, vehicle, rt,
                                    initial_guess=list(spec.reference_script),
                                    search=search, max_fevals=int(job["max_fevals"]),
                                    dt_opt=spec.dt_opt, dt_final=spec.dt_final,
                                    verdict_physics=physics,
                                    feasibility_envelope=spec.to_dict(), **kwargs)
    screen_s = time.time() - t_screen
    return {"full_fevals": int(calls["n"]), "full_wall_s": round(full_s, 1),
            "full_feasible": bool(full.feasible),
            "full_codes": list(full.reason_codes),
            "screen_fevals": int(calls["n"]), "screen_wall_s": round(screen_s, 1),
            "screen_codes": list(screened.reason_codes),
            "screen_rejected": screened.screen is not None
            and not screened.screen.get("in_envelope", True)}


# --------------------------------------------------------------------------
# job builders / runner
# --------------------------------------------------------------------------
def _arm_options(rig: str, arm: str, selected: Sequence[float] | None) -> dict[str, Any]:
    if rig == "dc8":
        o = dict(_rig("dc8")["recipe"])
        o.update({"search": "keep", "landing_divert_budget_m": None,
                  "landing_reserve_margin_kg": 0.0, "joint_chance": {},
                  "joint_admission": {"min_admission_probability": 0.208}})
        if arm == "dc8_binary":
            o.pop("joint_admission", None)
        if arm == "dc8_robust" and selected is not None:
            o["initial_guess"] = [float(x) for x in selected]
        return o
    o = dict(_rig(rig)["recipe"])
    o["search"] = "keep"
    return o


class Rig:
    def __init__(self, workers: int, cache_path: Path | None) -> None:
        self.workers = int(workers)
        self.cache_path = cache_path
        self.fly: dict[str, dict[str, Any]] = {}
        if cache_path is not None and cache_path.exists():
            self.fly = json.loads(cache_path.read_text())
        self.tmp = tempfile.TemporaryDirectory(prefix="e1102-")
        self.pool: ProcessPoolExecutor | None = None

    def run(self, jobs: Sequence[dict[str, Any]]) -> None:
        todo = [j for j in jobs if j["key"] not in self.fly]
        if not todo:
            return
        if self.workers > 1:
            if self.pool is None:
                self.pool = ProcessPoolExecutor(max_workers=self.workers)
            recs = list(self.pool.map(_worker, todo, chunksize=1))
        else:
            recs = [_worker(j) for j in todo]
        for rec in recs:
            self.fly[rec["key"]] = rec
        if self.cache_path is not None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.fly))

    def close(self) -> None:
        if self.pool is not None:
            self.pool.shutdown()
        self.tmp.cleanup()


def _jobs_A() -> list[dict[str, Any]]:
    return [{"part": "A", "key": f"A|{name}", "kind": "verdict", "rig": name,
             "spec": _spec_for(name)}
            for name in ("dc7", "starbase", "dc8_fins", "dc8")]


def _jobs_B() -> list[dict[str, Any]]:
    out = []
    for rig in ("dc7", "dc8_fins"):
        spec = _spec_for(rig)
        for dr in GRID_DR:
            for v in GRID_V:
                out.append({"part": "B", "key": f"B|{rig}|{dr}|{v}", "kind": "grid",
                            "rig": rig, "spec": spec,
                            "offsets": {"sep_downrange_m": dr, "sep_speed_m_s": v}})
    return out


def _jobs_C(selected: Sequence[float] | None) -> list[dict[str, Any]]:
    out = []
    for arm in ARMS:
        rig = RIG_OF_ARM[arm]
        opts = _arm_options(rig, arm, selected)
        script = opts.get("initial_guess")
        spec = _spec_for(rig, arm=arm,
                         script=[float(x) for x in script] if script else None)
        vpath = str(VEH_DIR / f"starship_class_rtls_{'dc8' if rig == 'dc8' else rig}.yaml")
        for seed in HOLDOUT_SEEDS:
            for idx in range(N_PER_SEED):
                out.append({"part": "C", "key": f"C|{arm}|{seed}|{idx}",
                            "kind": "holdout", "arm": arm, "rig": rig,
                            "seed": int(seed), "idx": int(idx),
                            "draw": [int(seed), int(idx)], "options": opts,
                            "spec": spec, "vehicle_path": vpath})
    return out


def _select_boundary_pair(c_recs: Sequence[Mapping[str, Any]]) -> tuple[dict, dict] | None:
    """The closest IN/OUT pair across the envelope boundary among the dc7
    hold-out samples (part D's straddle): minimise the severity distance of
    both sides to 1.0, require the same governing condition."""
    ins = [r for r in c_recs if r.get("pred_in_envelope") and "error" not in r
           and r.get("pred_worst_severity") is not None]
    outs = [r for r in c_recs if not r.get("pred_in_envelope")
            and r.get("pred_code") == "CORRIDOR_MISSED" and r.get("pred_landed") is False
            and "error" not in r and r.get("pred_worst_severity") is not None]
    best = None
    for i in ins:
        for o in outs:
            d = (1.0 - float(i["pred_worst_severity"])) + (float(o["pred_worst_severity"]) - 1.0)
            if d < 0.6 and (best is None or d < best[0]):
                best = (d, i, o)
    return (best[1], best[2]) if best else None


def _jobs_D(bands: dict[str, float],
            pair: tuple[dict, dict] | None) -> list[dict[str, Any]]:
    out = [{"part": "D", "key": f"D|base|{name}", "kind": "twin", "rig": name,
            "spec": _spec_for(name), "bands": bands}
           for name in ("dc7", "dc8_fins")]
    if pair is not None:
        # the cache is keyed by job key, so the calibrated re-run needs a
        # distinct key or it silently replays the raw-band result
        suffix = "" if bands == TWIN_BASE_BANDS else "|cal"
        for tag, rec in (("in", pair[0]), ("out", pair[1])):
            out.append({"part": "D",
                        "key": f"D|pair|{tag}|{rec['arm']}|{rec['seed']}|{rec['idx']}{suffix}",
                        "kind": "twin", "rig": RIG_OF_ARM[rec["arm"]],
                        "draw": [rec["seed"], rec["idx"]],
                        "spec": _spec_for(RIG_OF_ARM[rec["arm"]]), "bands": bands})
    return out


def _jobs_E(bands: dict[str, float]) -> list[dict[str, Any]]:
    return [{"part": "E", "key": "E|dc7|out-corner", "kind": "budget", "rig": "dc7",
             "spec": _spec_for("dc7"), "max_fevals": 132,
             "offsets": {"sep_downrange_m": 9000.0, "sep_speed_m_s": -60.0}},
            {"part": "E", "key": "E|dc8_fins|out", "kind": "budget",
             "rig": "dc8_fins",
             "spec": _spec_for("dc8_fins"), "max_fevals": 132},
            {"part": "E", "key": "E|dc7|in-nominal", "kind": "budget", "rig": "dc7",
             "spec": _spec_for("dc7"), "max_fevals": 132}]


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def _fig_boundary(rig: str, cells: list[dict[str, Any]], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dr = sorted({c["key"].split("|")[2] for c in cells}, key=float)
    vv = sorted({c["key"].split("|")[3] for c in cells}, key=float)
    idx = {c["key"]: c for c in cells}
    sev = np.array([[idx[f"B|{rig}|{a}|{b}"]["verdict"]["worst_severity"]
                     for b in vv] for a in dr])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax = axes[0]
    mesh = ax.pcolormesh(np.array(vv), np.array(dr), np.log10(sev).T,
                         cmap="RdYlGn_r", shading="auto",
                         vmin=math.log10(0.4), vmax=math.log10(2.5))
    cs = ax.contour(np.array(vv), np.array(dr), np.log10(sev).T, levels=[0.0],
                    colors="k", linewidths=1.6)
    if any(len(seg) for seg in cs.allsegs):
        ax.clabel(cs, fmt=lambda v: "in | out", fontsize=8)
    fig.colorbar(mesh, ax=ax, label="log10 worst severity")
    ax.set_xlabel("speed offset [m/s]")
    ax.set_ylabel("downrange offset [m]")
    ax.set_title(f"{rig}: envelope severity (boundary = 1.0)")
    ax2 = axes[1]
    colors = {None: "#2ca02c", "corridor_reached": "#d62728",
              "corridor_miss": "#e377c2", "corridor_tilt": "#d62728",
              "arrival_speed": "#ff7f0e", "landing_forecast": "#9467bd",
              "fuel_margin": "#8c564b", "entry_q": "#1f77b4", "entry_g": "#17becf",
              "fuel_floor": "#bcbd22"}
    for i, a in enumerate(dr):
        for j, b in enumerate(vv):
            c = idx[f"B|{rig}|{a}|{b}"]["verdict"]
            g = c.get("governing_condition")
            ax2.scatter(float(b), float(a), marker="s", s=44,
                        color=colors.get(g, "#7f7f7f") if g else "#2ca02c")
    handles = [plt.Line2D([], [], marker="s", ls="", color=col, label=lab or "IN")
               for lab, col in sorted(colors.items(), key=lambda kv: str(kv[0]))]
    ax2.legend(handles=handles, fontsize=7, loc="lower left")
    ax2.set_xlabel("speed offset [m/s]")
    ax2.set_ylabel("downrange offset [m]")
    ax2.set_title(f"{rig}: governing condition")
    fig.suptitle(f"E11-02 boundary map ({rig}) -- separation-state offsets at the "
                 "hot-staging handoff", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------
# aggregation + report
# --------------------------------------------------------------------------
def _confusion(recs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    code_match = code_mismatch = 0
    code_mix: dict[str, dict[str, int]] = {}
    false_rejects = []
    for r in recs:
        if "error" in r:
            continue
        in_env, landed = bool(r["pred_in_envelope"]), bool(r["truth_landed"])
        if in_env and landed:
            tp += 1
        elif in_env and not landed:
            fn += 1
        elif not in_env and landed:
            fp += 1
            false_rejects.append(r)
        else:
            tn += 1
        if not in_env:
            pred, truth = r.get("pred_code"), r.get("truth_reason")
            if pred == truth:
                code_match += 1
            else:
                code_mismatch += 1
            m = code_mix.setdefault(str(pred), {})
            m[str(truth)] = m.get(str(truth), 0) + 1
    n = tp + fn + fp + tn
    return {"n": n, "in_and_landed": tp, "in_not_landed": fn,
            "out_not_landed": tn, "out_but_landed": fp,
            "agreement": round((tp + tn) / n, 4) if n else None,
            "false_rejects": fp,
            "reject_code_match": code_match, "reject_code_mismatch": code_mismatch,
            "reject_code_mix": code_mix,
            "false_reject_detail": [
                {k: r.get(k) for k in ("arm", "seed", "idx", "truth_reason",
                                       "pred_governing", "pred_worst_severity")}
                for r in false_rejects]}


def _committed_counts() -> dict[str, dict[str, Any]]:
    doc = json.loads(E10_03_REPORT.read_text())
    return {arm: {"landed": doc["arms"][arm]["landed"],
                  "per_seed": {str(s): v["landed"]
                               for s, v in doc["arms"][arm]["per_seed"].items()}}
            for arm in ARMS}


def _md(doc: dict[str, Any]) -> str:
    L = ["# E11-02 — recovery feasibility envelope (task #309)\n",
         doc["framing"] + "\n",
         "## 1. Reference verdicts (part A)\n",
         "| rig | spec | in envelope | governing | predicted reason | unscreened reason | match |",
         "|---|---|---|---|---|---|---|"]
    for r in doc["A"].values():
        v = r.get("verdict", {})
        L.append(f"| `{r['rig']}` | `{v.get('probe', {}).get('spec_name')}` | "
                 f"{'IN' if v.get('in_envelope') else 'OUT'} | {v.get('governing_condition')} | "
                 f"`{v.get('governing_reason')}` | "
                 f"`{(r.get('unscreened_codes') or ['-'])[0]}` | "
                 f"{'yes' if r.get('prediction_matches') else 'NO'} |")
    L.append("")
    L.append("## 2. Boundary maps (part B)\n")
    for rig, st in doc["B_stats"].items():
        L.append(f"- `{rig}`: {st['in']}/{st['n']} inside; severity field and governing "
                 f"map in `{st['figure']}`; grid = downrange ±9 km x speed ±60 m/s at the "
                 f"hot-staging handoff.")
    L.append("")
    L.append("## 3. E10-03 hold-out agreement (part C)\n")
    L.append("Each sample is flown unscreened (the truth, the E10-03 worker verbatim) "
             "and screened (the prediction). A *false reject* — screen OUT but the "
             "flight lands — is the only error the screen is not allowed to make.\n")
    L.append("| arm | n | screen-in & landed | in & not landed | out & not landed | "
             "out but landed (false rejects) | agreement | rejected-sample code match | "
             "truth vs committed |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for arm, c in doc["C_confusion"].items():
        cc = doc["C_crosscheck"][arm]
        L.append(f"| `{arm}` | {c['n']} | {c['in_and_landed']} | {c['in_not_landed']} | "
                 f"{c['out_not_landed']} | **{c['out_but_landed']}** | {c['agreement']} | "
                 f"{c['reject_code_match']}/{c['reject_code_match'] + c['reject_code_mismatch']} | "
                 f"{cc['truth_landed']}/{cc['committed_landed']} "
                 f"({'match' if cc['match'] else 'DRIFT'}) |")
    L.append("")
    L.append(doc["C_notes"] + "\n")
    L.append("## 4. Twin boundary reproduction (part D)\n")
    L.append("E10-07's independent twin re-flies the sealed script from the same "
             "separation state on its own atmosphere line, wind, and fin model; "
             "bands are calibrated on the baseline runs and then applied to the "
             "boundary pair C selects.\n")
    L.append("| run | boundary condition side | agreement | q | g | speed | tilt | miss | prop |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in doc["D"].values():
        if "error" in r:
            L.append(f"| `{r['key']}` | — | error: {r['error']} | | | | | | |")
            continue
        chk = {c["quantity"]: c for c in r.get("checks", [])}

        def cell(name: str) -> str:
            c = chk.get(name)
            if not c:
                return "—"
            return (f"{c['twin']:.3g}{c['band_unit']} "
                    f"({c['twin_side']}, ±{c['band']:g}{c['band_unit']})"
                    f"{' ✓' if c['ok'] else ' ✗'}")
        side = ("—" if r.get("pair") is None else r["pair"])
        L.append(f"| `{r['key']}` | {side} | {'yes' if r['boundary_agreement'] else 'no'} | "
                 + " | ".join(cell(n) for n in ("q", "g", "speed", "tilt", "miss", "prop")) + " |")
    L.append("")
    L.append(doc["D_notes"] + "\n")
    L.append("## 5. Search budget (part E)\n")
    L.append("| state | full search: wall / fevals / verdict | screened: wall / fevals / reason |")
    L.append("|---|---|---|")
    for r in doc["E"]:
        if "error" in r:
            L.append(f"| `{r['key']}` | error: {r['error']} | |")
            continue
        L.append(f"| `{r['key']}` | {r['full_wall_s']} s / {r['full_fevals']} fevals / "
                 f"{'feasible' if r['full_feasible'] else '`' + (r['full_codes'] or ['-'])[0] + '`'} | "
                 f"{r['screen_wall_s']} s / {r['screen_fevals']} fevals / "
                 f"{'accepted' if not r['screen_rejected'] else '`' + (r['screen_codes'] or ['-'])[0] + '`'} |")
    L.append("")
    L.append(doc["E_notes"] + "\n")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parts", default="A,B,C,D,E")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--limit", type=int, default=0, help="cap C samples (smoke)")
    ap.add_argument("--report", default=str(OUT_JSON))
    args = ap.parse_args(argv)
    parts = {p.strip().upper() for p in args.parts.split(",")}
    t_start = time.time()

    committed = _committed_counts()
    doc_1003 = json.loads(E10_03_REPORT.read_text())
    selected = doc_1003.get("robust_arm", {}).get("selected_script")

    rig = Rig(args.workers, Path(args.cache) if args.cache else None)
    try:
        if "A" in parts:
            rig.run(_jobs_A())
        if "B" in parts:
            rig.run(_jobs_B())
        c_jobs = _jobs_C(selected)
        if args.limit:
            per: dict[str, int] = {}
            kept: list[dict[str, Any]] = []
            for j in c_jobs:
                if per.get(j["arm"], 0) < int(args.limit):
                    kept.append(j)
                    per[j["arm"]] = per.get(j["arm"], 0) + 1
            c_jobs = kept
        if "C" in parts:
            rig.run(c_jobs)
        c_recs = [r for r in rig.fly.values() if str(r.get("key", "")).startswith("C|")]
        bands = dict(TWIN_BASE_BANDS)
        pair = _select_boundary_pair(sorted(
            (r for r in c_recs if r.get("arm") == "dc7_pinned"),
            key=lambda r: (r.get("seed", -1), r.get("idx", -1))))
        if "D" in parts:
            rig.run(_jobs_D(bands, pair))
            # calibrate bands from the baselines (1.5x measured delta,
            # floored): accumulate the MAXIMUM over both baselines, else the
            # second (dc8) baseline silently undoes the first (dc7) one
            for base_key in ("D|base|dc7", "D|base|dc8_fins"):
                r = rig.fly.get(base_key, {})
                for c in r.get("checks", []):
                    d = abs(float(c["delta"]))
                    q = c["quantity"]
                    if q == "q":
                        bands["q"] = max(bands["q"], 5.0 * math.ceil(1.5 * d / 5))
                    elif q == "g":
                        bands["g"] = max(bands["g"], 0.25 * math.ceil(1.5 * d / 0.25))
                    elif q == "speed":
                        bands["speed"] = max(bands["speed"], 5.0 * math.ceil(1.5 * d / 5))
                    elif q == "tilt":
                        bands["tilt"] = max(bands["tilt"], 0.5 * math.ceil(1.5 * d / 0.5))
                    elif q == "miss":
                        bands["miss"] = max(bands["miss"], 25.0 * math.ceil(1.5 * d / 25))
                    elif q == "prop":
                        bands["prop"] = max(bands["prop"], 0.5 * math.ceil(1.5 * d / 0.5))
            if pair is not None:  # re-run the pair with calibrated bands
                for j in _jobs_D(bands, pair):
                    if "|pair|" in j["key"]:
                        rig.fly.pop(j["key"], None)
                rig.run(_jobs_D(bands, pair))
        if "E" in parts:
            rig.run(_jobs_E(bands))
    finally:
        rig.close()

    doc: dict[str, Any] = {
        "benchmark": "e11_02_recovery_feasibility_envelope",
        "created_utc": dtm.datetime.now(dtm.timezone.utc).isoformat(timespec="seconds"),
        "runtime_s": round(time.time() - t_start, 1),
        "framing": (
            "E11-02 screens the separation state against a recovery feasibility "
            "envelope (orbitopt.flightlab.recovery.feasibility) BEFORE the "
            "planner searches: one reference rollout plus an analytic fuel floor "
            "decides IN/OUT and predicts the flight reason with zero search "
            "fevals. Envelope conditions: fuel floor (rocket equation on the "
            "separation velocity), corridor reach/miss/tilt/arrival-speed, "
            "closed-loop landing forecast, entry q and g, landing propellant "
            "reserve margin — every one measured on the planner's own physics. "
            "Era note: the DC8 hold-out arms are the E10-03 era (the DC7 sealed "
            "POWERED recipe flown on the DC8 vehicle, no grid fins); the "
            "envelope that applies to DC8 as authored (E11-01 grid-fin mission) "
            "is graded in parts A/B/E."),
        "holdout_stream": {"seeds": HOLDOUT_SEEDS, "n_per_seed": N_PER_SEED,
                           "arms": list(ARMS)},
        "twin_bands": bands,
        "boundary_pair": None if pair is None else
        {"in": {k: pair[0].get(k) for k in ("arm", "seed", "idx", "pred_worst_severity")},
         "out": {k: pair[1].get(k) for k in ("arm", "seed", "idx", "pred_worst_severity")}},
        "A": {k.split("|")[1]: v for k, v in rig.fly.items() if k.startswith("A|")},
    }
    b_cells: dict[str, list[dict[str, Any]]] = {"dc7": [], "dc8_fins": []}
    for k, v in rig.fly.items():
        if k.startswith("B|"):
            b_cells[k.split("|")[1]].append(v)
    doc["B_stats"] = {}
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for name, cells in b_cells.items():
        if not cells:
            continue
        fig = FIG_DIR / f"e11_02_boundary_{name}.png"
        try:
            _fig_boundary(name, cells, fig)
            fig_path = str(fig.relative_to(REPO))
        except Exception as exc:
            fig_path = f"figure failed: {exc}"
        doc["B_stats"][name] = {
            "n": len(cells), "figure": fig_path,
            "in": sum(1 for c in cells if c["verdict"]["in_envelope"]),
            "governing_mix": _mix(c["verdict"].get("governing_condition")
                                  for c in cells)}
    doc["B_cells"] = {k.split("|")[1] + "|" + k.split("|")[2] + "|" + k.split("|")[3]:
                      {**v["verdict"], "probe": {kk: vv for kk, vv in v.get("probe", {}).items()}}
                      for k, v in rig.fly.items() if k.startswith("B|")}
    c_recs = [r for k, r in sorted(rig.fly.items()) if k.startswith("C|")]
    doc["C_records"] = c_recs
    doc["C_confusion"] = {arm: _confusion([r for r in c_recs if r["arm"] == arm])
                          for arm in ARMS}
    truth_counts = {arm: sum(1 for r in c_recs if r["arm"] == arm
                             and r.get("truth_landed")) for arm in ARMS}
    doc["C_crosscheck"] = {
        arm: {"truth_landed": truth_counts[arm],
              "committed_landed": committed[arm]["landed"],
              "match": truth_counts[arm] == committed[arm]["landed"]}
        for arm in ARMS}
    total_fr = sum(c["false_rejects"] for c in doc["C_confusion"].values())
    doc["C_notes"] = (
        f"Total false rejects (screen OUT, flight landed): **{total_fr}** — the "
        "screen's only unforgivable error is zero." if total_fr == 0 else
        f"**{total_fr} false rejects found — acceptance fails**, see "
        "false_reject_detail.")
    doc["D"] = {k: v for k, v in sorted(rig.fly.items()) if k.startswith("D|")}
    for v in doc["D"].values():
        v["pair"] = ("IN" if "|in|" in str(v.get("key", ""))
                     else "OUT" if "|out|" in str(v.get("key", "")) else None)
    agreed = [v for v in doc["D"].values()
              if "pair" in v and v["pair"] in ("IN", "OUT") and "error" not in v]
    doc["D_notes"] = (
        "Baselines carry the band calibration (1.5x the measured twin-vs-planner "
        "delta, floored at the E11-01 FIN_TOL values); twin disagreement is "
        "reported, never hidden. The twin's arrival-speed channel is the known "
        "atmosphere-line quantity (E11-01 measured 24%): when that quantity "
        "alone disagrees, the boundary verdict rests on q/g/tilt/miss/prop.")
    doc["D_pair_agreement"] = bool(agreed) and all(v["boundary_agreement"] for v in agreed)
    doc["E"] = sorted((v for k, v in rig.fly.items() if k.startswith("E|")),
                     key=lambda r: str(r.get("key")))
    doc["E_notes"] = (
        "The screened calls spend zero search fevals on an OUT state and return "
        "the predicted flight reason; the full search spends its budget (counted "
        "through the planner's own search hook, same options as the default "
        "path). The DC8-era arm's real budget was the 300+-flight ES training "
        "(see the committed E10-03 report), which the screen retires for "
        "states outside the envelope.")
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(doc, indent=1) + "\n")
    OUT_MD.write_text(_md(doc))
    print(f"wrote {args.report} and {OUT_MD}")
    return 0


def _mix(values: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        key = str(v)
        out[key] = out.get(key, 0) + 1
    return out


if __name__ == "__main__":
    raise SystemExit(main())
