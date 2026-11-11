"""FL-201: the pygmo ascent UDP -- fitness shape, cache, single-source residuals.

The recorded FL-004 CLI stub (same fixture as the Gate-B simulate tests)
replays OpenTsiolkovsky v1.0.0 outputs for any config, so fitness() exercises
the real run_ascent -> evaluate path without the binary. The residuals the
fitness vector carries are asserted against the VERIFICATION the spy captured
-- the single-source-of-truth acceptance criterion made executable.
"""
from __future__ import annotations

import copy
import shutil
from pathlib import Path

import pytest
import yaml

from orbitopt.flightlab import checks
from orbitopt.flightlab.adapters.opentsiolkovsky import Vehicle
from orbitopt.flightlab.contract import load_target
from orbitopt.flightlab.guidance import ascent_param
from orbitopt.flightlab.optimize import ascent_problem as mod
from orbitopt.flightlab.optimize.ascent_problem import (
    PENALTY_OBJECTIVE,
    PENALTY_VIOLATION,
    AscentProblem,
)

FIX = Path(__file__).parent / "fixtures" / "flightlab" / "opentsiolkovsky"
REPO = Path(__file__).resolve().parents[1]
VEHICLE = REPO / "src" / "orbitopt" / "data" / "flightlab" / "vehicles" / "synthetic_two_stage.yaml"
TARGET = REPO / "src" / "orbitopt" / "data" / "flightlab" / "scenarios" / "leo_two_stage.yaml"


@pytest.fixture
def target():
    return load_target(TARGET)


@pytest.fixture
def stub(monkeypatch, tmp_path):
    """OT binary override -> the recorded-output stub (needs the depth pytest's
    tmp_path provides; the fixture dir itself would also do)."""
    path = tmp_path / "otcli"
    shutil.copy(FIX / "stub_cli.py", path)
    path.chmod(0o755)
    monkeypatch.setenv("ORBITOPT_OPENTSIOLKOVSKY", str(path))
    monkeypatch.setenv("OT_STUB_FIXTURE", str(FIX / "synthetic_out"))
    monkeypatch.setenv("OT_STUB_SRC", str(REPO / "src" / "orbitopt"))
    return path


@pytest.fixture
def spy(monkeypatch):
    """Capture every Verification evaluate() returns inside fitness(). The
    adapter itself already evaluates the run once; the problem's re-grade is
    the LAST capture."""
    captured = []
    original = checks.evaluate

    def wrapped(run, target):
        verdict = original(run, target)
        captured.append(verdict)
        return verdict

    monkeypatch.setattr(checks, "evaluate", wrapped)
    return captured


@pytest.fixture
def flycount(monkeypatch):
    """Count real run_ascent subprocess flights the problem performs."""
    calls = [0]
    original = mod.run_ascent

    def wrapped(*args, **kwargs):
        calls[0] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(mod, "run_ascent", wrapped)
    return calls


def _mid(udp: AscentProblem):
    lo, hi = udp.bounds()
    return [(a + b) / 2.0 for a, b in zip(lo, hi)]


def _expected_residual(verdict, check_id: str, signed: bool) -> float:
    """Independent restatement of the fitness mapping for the comparison."""
    check = next((c for c in verdict.checks if c.id == check_id), None)
    if check is None:
        return 0.0
    if check.value is None:
        return PENALTY_VIOLATION
    delta = float(check.value) - float(check.limit)
    if signed:
        return delta
    return -delta if check_id == "propellant-reserve" else delta


# --- interface ---------------------------------------------------------------

def test_unknown_objective_rejected(target):
    with pytest.raises(ValueError):
        AscentProblem(target, VEHICLE, "cheapest")


def test_bounds_interface(target):
    pay = AscentProblem(target, VEHICLE, "payload")
    res = AscentProblem(target, VEHICLE, "reserve")
    lo, hi = pay.get_bounds()
    assert pay.bounds() == (lo, hi) and res.bounds()[0] == res.get_bounds()[0]
    assert len(lo) == 8 and len(res.bounds()[0]) == 7
    doc = yaml.safe_load(VEHICLE.read_text(encoding="utf-8"))
    assert (lo, hi) == ascent_param.bounds(doc, target, "payload")
    bands = pay.get_bounds_eq_ineq()
    assert len(bands) == 6
    for band in bands[:3]:
        assert band[0] < 0.0 < band[1] and band[2] == float("-inf") and band[3] == float("inf")
    for band in bands[3:]:
        assert band[3] == 0.0


def test_decode_interface(target):
    udp = AscentProblem(target, VEHICLE, "payload")
    prog, doc = udp.decode(_mid(udp))
    assert prog and doc["stages"][-1]["mass_dry_kg"] > \
        yaml.safe_load(VEHICLE.read_text())["stages"][-1]["mass_dry_kg"]


def test_write_vehicle_never_touches_base(target, tmp_path):
    udp = AscentProblem(target, VEHICLE, "payload")
    before = VEHICLE.read_bytes()
    path = udp.write_vehicle(_mid(udp), tmp_path / "derived")
    assert path.name == "vehicle.yaml" and path.exists()
    assert VEHICLE.read_bytes() == before           # base file byte-identical
    vehicle = Vehicle.load(path)                    # OT schema, via the real loader
    assert vehicle.stages[-1]["mass_dry_kg"] > \
        yaml.safe_load(VEHICLE.read_text())["stages"][-1]["mass_dry_kg"]
    assert vehicle.stages[0] == yaml.safe_load(VEHICLE.read_text())["stages"][0]


# --- fitness against the recorded stub ----------------------------------------

@pytest.mark.parametrize("objective", ["payload", "reserve"])
def test_fitness_shape_and_residuals_match_evaluate(target, stub, spy, objective):
    udp = AscentProblem(target, VEHICLE, objective, payload_kg=1000.0)
    x = _mid(udp)
    f = udp.fitness(x)
    assert len(f) == 7 and all(v == v and abs(v) != float("inf") for v in f)
    verdict = spy[-1]                               # the problem's own re-grade
    by_id = {c.id: c for c in verdict.checks}
    # eq: actual - target, straight off the evaluate() checks
    for i, cid in enumerate(mod._EQ_CHECKS, start=1):
        assert f[i] == pytest.approx(_expected_residual(verdict, cid, True))
    # ineq: violation convention, straight off the evaluate() checks
    for i, cid in enumerate(mod._INEQ_CHECKS, start=4):
        assert f[i] == pytest.approx(_expected_residual(verdict, cid, False))
    # objective
    if objective == "payload":
        assert f[0] == pytest.approx(-x[7])         # negated for maximization
    else:
        assert f[0] == pytest.approx(-float(by_id["propellant-reserve"].value))


def test_fitness_is_cached(target, stub, flycount):
    udp = AscentProblem(target, VEHICLE, "payload")
    x = _mid(udp)
    assert udp.fitness(x) == udp.fitness(x)
    assert flycount[0] == 1                         # second call never re-flew
    other = [a * 0.5 for a in x]
    udp.fitness(other)
    assert flycount[0] == 2                         # a different x does re-fly
    clone = copy.deepcopy(udp)                      # pygmo copies UDPs constantly
    assert clone.fitness(x) == udp.fitness(x)
    assert flycount[0] == 2                         # the copy shares the parent's cache


def test_failed_flight_is_a_penalty_not_an_exception(target, monkeypatch, stub):
    monkeypatch.setenv("ORBITOPT_OPENTSIOLKOVSKY", str(stub.parent / "no-such-binary"))
    udp = AscentProblem(target, VEHICLE, "reserve")
    f = udp.fitness(_mid(udp))
    assert f == [PENALTY_OBJECTIVE] + [PENALTY_VIOLATION] * 6
    assert udp.penalty_fitness(_mid(udp)) > 0.0     # finite, hugely infeasible


def test_wrong_shape_vector_is_a_penalty(target, stub):
    udp = AscentProblem(target, VEHICLE, "payload")
    assert udp.fitness([3.0, 5.0, 20.0]) == [PENALTY_OBJECTIVE] + [PENALTY_VIOLATION] * 6
    bad = _mid(udp)
    bad[1] = float("nan")                           # unpack rejects it
    assert udp.fitness(bad)[0] == PENALTY_OBJECTIVE


# --- pygmo wiring (conda-only: skipped on base installs) ----------------------

def test_pygmo_roundtrip(target, stub):
    pg = pytest.importorskip("pygmo")
    udp = AscentProblem(target, VEHICLE, "payload")
    prob = pg.problem(udp)
    assert prob.get_ncx() == 8 and prob.has_constraints()
    lo, hi = prob.get_bounds()
    assert len(lo) == 8 and lo[0] < hi[0]
    pop = pg.population(prob, size=4, seed=11)
    assert len(pop.champion_f) == 7
