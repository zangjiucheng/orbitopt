"""Run-directory builders for the FL-503 compare/recovery tests.

FL-304's recovery pipeline is not on this branch, so the ascent+recovery run
is *built* here in the FL-304 output format: the ballistic ``stage1_debris``
track is swapped for an illustrative ``booster`` return arc back to the
launch site, the FL-303 touchdown checks + FL-304 ``recovery-landing`` gate
are merged into the verification, and a ``montecarlo.json`` in the FL-304
schema is written next to it. Every number is fabricated for UI tests
(provenance ``illustrative``); nothing here is physics.
"""
from __future__ import annotations

import dataclasses
import json
import math
import shutil
from pathlib import Path

import yaml

from orbitopt.flightlab.checks import CONSTRAINT_CHECK_IDS
from orbitopt.flightlab.contract import BodyTrack, Check, Event, Verification, read_run, write_run

REPO = Path(__file__).resolve().parents[1]
OT_FIX = Path(__file__).parent / "fixtures" / "flightlab" / "opentsiolkovsky"
VEHICLE = REPO / "src" / "orbitopt" / "data" / "flightlab" / "vehicles" / "synthetic_two_stage.yaml"
TARGET = REPO / "src" / "orbitopt" / "data" / "flightlab" / "scenarios" / "leo_two_stage.yaml"
EARTH_R = 6378137.0
LANDING_IDS = ("landing-position", "landing-vertical-speed", "landing-lateral-speed",
               "landing-tilt", "landing-propellant-reserve", "landing-max-g")


def fly_stub_run(out: Path, tmp: Path, monkeypatch) -> Path:
    """A real Gate-B run from ``simulate`` flown by the FL-004 OT CLI stub."""
    from orbitopt.flightlab.contract import load_target
    from orbitopt.flightlab.simulate import simulate

    cli = tmp / "otcli"
    shutil.copy(OT_FIX / "stub_cli.py", cli)
    cli.chmod(0o755)
    monkeypatch.setenv("ORBITOPT_OPENTSIOLKOVSKY", str(cli))
    monkeypatch.setenv("OT_STUB_FIXTURE", str(OT_FIX / "synthetic_out"))
    monkeypatch.setenv("OT_STUB_SRC", str(REPO / "src" / "orbitopt"))
    simulate(load_target(TARGET), VEHICLE, out, timeout_s=60)
    return out


def _write_side(out: Path, payload_kg: float, recovery: str) -> None:
    vehicle = yaml.safe_load(VEHICLE.read_text(encoding="utf-8"))
    vehicle["payload_kg"] = float(payload_kg)
    (out / "vehicle.yaml").write_text(yaml.safe_dump(vehicle, sort_keys=False), encoding="utf-8")
    (out / "designer-form.json").write_text(json.dumps({
        "mission_form": {"recovery": recovery, "apoapsis_km": 400.0, "periapsis_km": 400.0,
                         "inclination_deg": 51.6,
                         "recovery_reserve_kg": 1000.0 if recovery != "none" else None},
    }, sort_keys=True), encoding="utf-8")


def make_ascent_only(src: Path, out: Path, *, payload_kg: float = 15000.0) -> Path:
    """Copy of ``src`` as an expendable (recovery = none) run."""
    run = read_run(src)
    checks = tuple(c for c in run.verification.checks if c.id != "recovery-propellant-reserve")
    constraints = [c for c in checks if c.id in CONSTRAINT_CHECK_IDS]
    run = dataclasses.replace(run, run_id=f"{run.run_id}-expendable", verification=dataclasses.replace(
        run.verification, checks=checks,
        mission_constraints_passed=all(c.passed for c in constraints) if constraints else None))
    if out.exists():
        shutil.rmtree(out)
    write_run(out, run)
    _write_side(out, payload_kg, "none")
    return out


def _rotate_toward(u0, u1, theta):
    """Unit vector at angle ``theta`` from u0 in the (u0, u1) plane."""
    dot = sum(a * b for a, b in zip(u0, u1))
    perp = [b - dot * a for a, b in zip(u0, u1)]
    n = math.sqrt(sum(x * x for x in perp)) or 1.0
    perp = [x / n for x in perp]
    return [math.cos(theta) * a + math.sin(theta) * p for a, p in zip(u0, perp)]


def _booster_track(run, *, prop_td_kg: float, dry_kg: float = 30000.0, n: int = 120) -> BodyTrack:
    stack = next(t for t in run.tracks if t.parent_body_id is None)
    debris = next(t for t in run.tracks if t.body_id == "stage1_debris")
    r0 = stack.position_m[0]
    rs = debris.position_m[0]
    n0 = math.sqrt(sum(x * x for x in r0))
    ns = math.sqrt(sum(x * x for x in rs))
    u0 = [x / n0 for x in r0]
    us = [x / ns for x in rs]
    theta_s = math.acos(max(-1.0, min(1.0, sum(a * b for a, b in zip(u0, us)))))
    h_s = ns - EARTH_R
    t0, dur = debris.t_s[0], 420.0
    # Illustrative: the booster separates with its return budget on board
    # (dry + boostback/entry/landing propellant), then only ever depletes.
    m0 = dry_kg + prop_td_kg + 12000.0
    ts, pos, mass, alt, dr = [], [], [], [], []
    for i in range(n):
        u = i / (n - 1)
        theta = theta_s * (1.0 - u) + 0.35 * theta_s * math.sin(math.pi * u) * (1.0 - u)
        h = max(0.0, h_s * (1.0 - u) ** 1.5 + 2.2 * h_s * u * (1.0 - u) ** 2)
        d = _rotate_toward(u0, us, theta)
        ts.append(t0 + dur * u)
        pos.append(tuple((EARTH_R + h) * x for x in d))
        burn = min(1.0, u / 0.25) * 0.6 + max(0.0, (u - 0.9) / 0.1) * 0.4
        mass.append(m0 - (m0 - dry_kg - prop_td_kg) * min(1.0, burn))
        alt.append(h)
        dr.append(theta * EARTH_R)
    vel = []
    for i in range(n):
        j0, j1 = max(0, i - 1), min(n - 1, i + 1)
        dt = ts[j1] - ts[j0]
        vel.append(tuple((pos[j1][k] - pos[j0][k]) / dt for k in range(3)))
    return BodyTrack(
        body_id="booster", parent_body_id=debris.parent_body_id, frame=debris.frame,
        epoch_utc=debris.epoch_utc, provenance="illustrative", t_s=tuple(ts), mass_kg=tuple(mass),
        position_m=tuple(pos), velocity_m_s=tuple(vel),
        extra_columns={"altitude_m": tuple(alt), "downrange_m": tuple(dr)})


def montecarlo_doc(n: int = 24, seed: int = 7, fail_every: int = 6) -> dict:
    samples = []
    for i in range(n):
        ang = 2.399963 * i
        rad = 18.0 * math.sqrt((i + 0.5) / n)
        east, north = 6.0 + rad * math.cos(ang) * 1.8, -3.0 + rad * math.sin(ang) * 0.8
        failed = i % fail_every == fail_every - 1
        hard = (i // fail_every) % 2 == 0  # alternate: hard touchdown / never got back
        rec = {"idx": i, "outcome": "failed" if failed else "landed",
               "reason": ("LANDING_VERTICAL_SPEED_EXCEEDED" if hard else "INSUFFICIENT_PROPELLANT_BOOSTBACK")
               if failed else "LANDED"}
        if not failed or hard:
            rec.update(touchdown_east_m=east, touchdown_north_m=north,
                       position_error_m=math.hypot(east, north), vertical_speed_m_s=1.2, lateral_speed_m_s=0.3)
        samples.append(rec)
    landed = [s for s in samples if s["outcome"] == "landed"]
    pts = [(s["touchdown_east_m"], s["touchdown_north_m"]) for s in landed]
    me = sum(p[0] for p in pts) / len(pts)
    mn = sum(p[1] for p in pts) / len(pts)
    sxx = sum((p[0] - me) ** 2 for p in pts) / (len(pts) - 1)
    syy = sum((p[1] - mn) ** 2 for p in pts) / (len(pts) - 1)
    hist: dict[str, int] = {}
    for s in samples:
        hist[s["reason"]] = hist.get(s["reason"], 0) + 1
    return {
        "kind": "flightlab-recovery-montecarlo", "n": n, "seed": seed,
        "dispersions": {"thrust_pct": 5.0, "isp_pct": 3.0, "density_pct": 10.0, "cd_pct": 8.0},
        "success_rate": len(landed) / n,
        "reason_histogram": dict(sorted(hist.items())),
        "landing_stats": {"position_error_m": {"n": len(landed), "p50": 9.0, "p95": 21.0}},
        "ellipse": {"state": "assessed", "n": len(landed), "mean_east_m": me, "mean_north_m": mn,
                    "sigma_along_m": math.sqrt(max(sxx, syy)), "sigma_cross_m": math.sqrt(min(sxx, syy)),
                    "bearing_deg": 90.0 if sxx >= syy else 0.0},
        "samples": samples,
        "assumptions": ["illustrative FL-503 test fixture, FL-304 schema"],
    }


def make_ascent_recovery(src: Path, out: Path, *, payload_kg: float = 12600.0,
                         prop_td_kg: float = 1850.0, landed: bool = True,
                         montecarlo: bool = True) -> Path:
    """``src`` with an RTLS booster return in the FL-304 output format."""
    run = read_run(src)
    existing = next((t for t in run.tracks if t.body_id == "booster"), None)
    if existing is not None:  # the small synthetic fixture already has a return track
        booster, tracks = existing, run.tracks
    else:
        booster = _booster_track(run, prop_td_kg=prop_td_kg)
        tracks = tuple(t for t in run.tracks if t.body_id != "stage1_debris") + (booster,)
    t0, t1 = booster.t_s[0], booster.t_s[-1]
    events = [e for e in run.events if e.body_id not in ("stage1_debris", "booster")]
    events += [Event(t0 + 8.0, "boostback_start", "booster", {}),
               Event(t0 + 60.0, "boostback_end", "booster", {}),
               Event(t0 + 300.0, "entry_start", "booster", {}),
               Event(t0 + 320.0, "entry_end", "booster", {}),
               Event(t1 - 35.0, "landing_start", "booster", {}),
               Event(t1, "touchdown", "booster", {"vertical_speed_m_s": 1.4 if landed else 6.2,
                                                   "lateral_speed_m_s": 0.3})]
    events.sort(key=lambda e: (e.t_s, e.body_id, e.kind))
    vs = 1.4 if landed else 6.2
    landing = (
        Check("landing-position", True, 7.8, 30.0, "m"),
        Check("landing-vertical-speed", vs <= 3.0, vs, 3.0, "m/s",
              "" if vs <= 3.0 else "LANDING_VERTICAL_SPEED_EXCEEDED"),
        Check("landing-lateral-speed", True, 0.3, 1.0, "m/s"),
        Check("landing-tilt", True, 4.1, 15.0, "deg"),
        Check("landing-propellant-reserve", landed, prop_td_kg if landed else 120.0, 500.0, "kg",
              "" if landed else "LANDING_PROPELLANT_RESERVE_LOW"),
        Check("landing-max-g", True, 3.2, 6.0, "g"),
    )
    gate = Check("recovery-landing", landed, None, None, "",
                 "" if landed else "LANDING_VERTICAL_SPEED_EXCEEDED")
    reserve = Check("recovery-propellant-reserve", prop_td_kg >= 1000.0, prop_td_kg, 1000.0, "kg",
                    "" if prop_td_kg >= 1000.0 else "PROPELLANT_RESERVE_LOW")
    kept = [c for c in run.verification.checks if c.id != "recovery-propellant-reserve"]
    checks = tuple(kept) + (reserve, gate) + landing
    ids = set(CONSTRAINT_CHECK_IDS) | {"recovery-landing", *LANDING_IDS}
    constraints = [c for c in checks if c.id in ids]
    run = dataclasses.replace(
        run, run_id=f"{run.run_id}-rtls", tracks=tracks, events=tuple(events),
        assumptions=run.assumptions + ("FL-503 test fixture: illustrative booster return arc and "
                                       "touchdown checks in the FL-304 format (not flown)",),
        verification=Verification(run.verification.optimizer_converged, run.verification.physics_verified,
                                  all(c.passed for c in constraints), checks))
    if out.exists():
        shutil.rmtree(out)
    write_run(out, run)
    _write_side(out, payload_kg, "rtls")
    if montecarlo:
        (out / "montecarlo.json").write_text(json.dumps(montecarlo_doc(), indent=1, sort_keys=True),
                                             encoding="utf-8")
    return out
