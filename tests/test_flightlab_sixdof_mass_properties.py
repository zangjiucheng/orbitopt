"""F-03 (task #228): mass, CG and inertia tensor as the propellant burns.

The acceptance claims, split the way the code is split:

* the declared two-endpoint model in ``sixdof/mass_properties.py`` is exact at
  its endpoints, linear in the remaining propellant mass, and reproduces the
  spike_stage.xml decomposition (10 t dry at x 20 m + a 2.5:1 LOX/CH4 load in
  two 1.8 m spheres at x 14 m / 28 m) -- pure Python, no jsbsim;
* a JSBSim run's mass trace equals the integral of its own flow trace (the
  mass-conservation acceptance claim), up to the one frame the mass balance is
  known to lag, and its sampled CG/inertia agree with the declared model at
  full load, along the burn and at burnout (``slow``, opt-in ``jsbsim``);
* the packaged ``f03_propellant_depletion`` scenario flies through burnout and
  its ``sixdof-mass-*`` / ``sixdof-cg-*`` / ``sixdof-inertia-*`` checks pass
  with the config's declared limits (``slow``).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from orbitopt.flightlab.sixdof.mass_properties import (
    MassModelTolerances,
    MassProperties,
    MassPropertiesError,
    MassState,
    load_mass_model,
    parallel_axis,
)

REPO = Path(__file__).resolve().parents[1]

# spike_stage.xml, as documented in the packaged F-03 scenario: dry 10 t at
# x=20 m with I = 3.4e4 / 1.7e6 / 1.7e6, two spherical tanks of radius 1.8 m.
DRY = (10000.0, 20.0, 3.4e4, 1.7e6, 1.7e6)
TANKS = ((42857.142857, 14.0), (17142.857143, 28.0))
TANK_RADIUS_M = 1.8


def _xml_full_state() -> MassState:
    """The loaded-at-capacity state, computed from the XML figures directly
    (spheres as their own 2/5*m*r^2 plus a point mass at the tank centre) --
    the independent cross-check of the declared endpoints."""
    dry_m, dry_x, ixx, iyy, izz = DRY
    mass = dry_m + sum(m for m, _ in TANKS)
    cg = (dry_m * dry_x + sum(m * x for m, x in TANKS)) / mass
    dx = dry_x - cg
    iyy += dry_m * dx * dx
    izz += dry_m * dx * dx
    for m, x in TANKS:
        own = 0.4 * m * TANK_RADIUS_M ** 2
        d = x - cg
        ixx += own
        iyy += own + m * d * d
        izz += own + m * d * d
    return MassState(mass_kg=mass, cg_m=(cg, 0.0, 0.0),
                     inertia_kg_m2=(ixx, iyy, izz, 0.0, 0.0, 0.0))


def _spike_model() -> MassProperties:
    """The declared model the packaged scenario uses, endpoints first."""
    full = _xml_full_state()
    return MassProperties(full=full, empty=MassState(
        mass_kg=DRY[0], cg_m=(DRY[1], 0.0, 0.0),
        inertia_kg_m2=(DRY[2], DRY[3], DRY[4], 0.0, 0.0, 0.0)))


def _close(a: float, b: float, rel: float = 1e-12) -> bool:
    return abs(a - b) <= rel * max(abs(a), abs(b), 1.0)


# --------------------------------------------------------------------------
# the declared model (pure Python)
# --------------------------------------------------------------------------

def test_endpoints_are_the_declared_states():
    props = _spike_model()
    for fraction, ref in ((1.0, props.full), (0.0, props.empty)):
        got = props.at_fraction(fraction)
        assert _close(got.mass_kg, ref.mass_kg)
        assert all(_close(a, b) for a, b in zip(got.cg_m, ref.cg_m))
        assert all(_close(a, b, 1e-9) for a, b in zip(got.inertia_kg_m2,
                                                      ref.inertia_kg_m2))


def _state_at_alpha(props: MassProperties, alpha: float) -> MassState:
    """The state at burn fraction ``alpha``, rebuilt from the public pieces:
    the dry endpoint and ``alpha`` times the propellant lump, each carried to
    their common CG with the parallel-axis theorem. Independent of how
    :meth:`MassProperties.at_fraction` is organised inside."""
    dry = props.empty
    m_p = alpha * props.propellant_mass_kg
    mass = dry.mass_kg + m_p
    cg = tuple((dry.mass_kg * d + m_p * p) / mass
               for d, p in zip(dry.cg_m, props.propellant_cg_m))
    i_dry = parallel_axis(dry.inertia_kg_m2, dry.mass_kg,
                          tuple(d - c for d, c in zip(dry.cg_m, cg)))
    i_pro = parallel_axis(tuple(alpha * v for v in props.propellant_inertia_kg_m2), m_p,
                          tuple(p - c for p, c in zip(props.propellant_cg_m, cg)))
    return MassState(mass_kg=mass, cg_m=cg,
                     inertia_kg_m2=tuple(a + b for a, b in zip(i_dry, i_pro)))


def test_interpolation_is_linear_in_remaining_propellant_mass():
    props = _spike_model()
    half = props.at_fraction(0.5)
    assert _close(half.mass_kg, props.empty.mass_kg + 0.5 * props.propellant_mass_kg)
    for alpha in (0.0, 0.13, 0.5, 0.75, 1.0):
        got = props.at_fraction(alpha)
        ref = _state_at_alpha(props, alpha)
        assert _close(got.mass_kg, ref.mass_kg)
        assert all(_close(a, b) for a, b in zip(got.cg_m, ref.cg_m))
        assert all(_close(a, b, 1e-9) for a, b in zip(got.inertia_kg_m2,
                                                      ref.inertia_kg_m2))
    mid = props.at_propellant_mass_kg(0.5 * props.propellant_mass_kg)
    assert _close(mid.mass_kg, half.mass_kg)
    assert all(_close(a, b, 1e-9) for a, b in zip(mid.inertia_kg_m2, half.inertia_kg_m2))
    # mass is the affine invariant of the model: fraction_at_mass_kg inverts it
    for fraction in (0.0, 0.13, 0.5, 1.0):
        assert _close(props.fraction_at_mass_kg(props.at_fraction(fraction).mass_kg),
                      fraction, 1e-9)


def test_model_reproduces_the_spike_stage_decomposition():
    props = _spike_model()
    assert _close(props.propellant_mass_kg, 60000.0)
    assert _close(props.propellant_cg_m[0],
                  sum(m * x for m, x in TANKS) / sum(m for m, _ in TANKS))
    # the lump's own inertia = the spheres' own + their spread about x 18 m
    own = props.propellant_inertia_kg_m2
    expected_own = sum(0.4 * m * TANK_RADIUS_M ** 2 for m, _ in TANKS)
    spread = sum(m * (x - props.propellant_cg_m[0]) ** 2 for m, x in TANKS)
    assert _close(own[0], expected_own, 1e-9)
    assert _close(own[1], expected_own + spread, 1e-9)
    assert _close(own[2], own[1], 1e-9)
    # and the full-load state it implies is the XML's own arithmetic
    full = _xml_full_state()
    assert _close(props.full.cg_m[0], full.cg_m[0], 1e-12)
    assert _close(props.full.inertia_kg_m2[0], 111760.0, 1e-9)


def test_parallel_axis_hand_case():
    m = 10.0
    i = parallel_axis((0.0, 0.0, 0.0, 0.0, 0.0, 0.0), m, (0.6, 0.8, 0.0))
    assert _close(i[0], m * 0.8 ** 2) and _close(i[1], m * 0.6 ** 2)
    assert _close(i[3], -m * 0.6 * 0.8)
    assert _close(i[2], m * (0.6 ** 2 + 0.8 ** 2))
    # shifting a tensor away and back is the identity
    back = parallel_axis(i, -m, (-0.6, -0.8, 0.0))
    assert all(_close(v, 0.0, 1e-9) for v in back)


def test_declared_model_validation():
    ok = {"full": {"mass_kg": 2.0, "cg_m": [0.0, 0.0, 0.0],
                   "inertia_kg_m2": {"ixx": 1.0, "iyy": 1.0, "izz": 1.0}},
          "empty": {"mass_kg": 1.0, "cg_m": [0.0, 0.0, 0.0],
                    "inertia_kg_m2": {"ixx": 1.0, "iyy": 1.0, "izz": 1.0}}}
    cfg = load_mass_model(ok)
    assert isinstance(cfg.tolerances, MassModelTolerances)
    for mut, message in (
        (lambda d: d.pop("empty"), "missing"),
        (lambda d: d["full"].update(mass_kg=1.0), "must exceed"),
        (lambda d: d["full"].update(inertia_kg_m2={"ixx": -1.0, "iyy": 1.0, "izz": 1.0}),
         "positive semi-definite"),
        (lambda d: d.update(extra=1), "unknown key"),
        (lambda d: d.update(tolerances={"nonsense": 1.0}), "unknown key"),
    ):
        doc = {"full": dict(ok["full"], inertia_kg_m2=dict(ok["full"]["inertia_kg_m2"])),
               "empty": dict(ok["empty"], inertia_kg_m2=dict(ok["empty"]["inertia_kg_m2"]))}
        mut(doc)
        with pytest.raises(MassPropertiesError, match=message):
            load_mass_model(doc)
    props = MassProperties(**{k: MassState.from_mapping(v, k)
                              for k, v in (("full", ok["full"]), ("empty", ok["empty"]))})
    with pytest.raises(MassPropertiesError):
        props.at_propellant_mass_kg(1.5)  # capacity is 1 kg
    with pytest.raises(MassPropertiesError):
        MassProperties(full=MassState(1.0, (0, 0, 0), (1, 1, 1, 0, 0, 0)),
                       empty=MassState(2.0, (0, 0, 0), (1, 1, 1, 0, 0, 0)))


# --------------------------------------------------------------------------
# the flown trace against the declared model (jsbsim, slow)
# --------------------------------------------------------------------------

jsbsim = pytest.importorskip("jsbsim", reason="6-DOF backend is opt-in (orbitopt[sixdof])")

from orbitopt.flightlab.adapters.sixdof import (  # noqa: E402
    load_config, packaged_config, run_ascent,
)
from orbitopt.flightlab.sixdof.jsbsim_driver import (  # noqa: E402
    InitialConditions, SixDofSim, mass_state_from_sample,
)

F03_CONFIG = "f03_propellant_depletion"


@pytest.fixture(scope="module")
def depletion_samples():
    """One full-throttle spike_stage run to burnout + settle (dt 0.02 s)."""
    sim = SixDofSim(0.02)
    sim.initialize(InitialConditions(alt_m=300e3, pitch_deg=90.0))
    sim.ignite(1.0)
    sim.run_for(104.0)
    return sim.samples


def _flow_integral_kg(samples) -> tuple[list, float]:
    """Running trapezoid of the sampled flow, plus the worst mismatch against
    the mass trace (the one-frame drain lag shows up as a constant offset)."""
    integral, residual, prev = 0.0, 0.0, samples[0]
    out = [0.0]
    for s in samples[1:]:
        integral += 0.5 * (prev["flow_kg_s"] + s["flow_kg_s"]) * (s["t_s"] - prev["t_s"])
        residual = max(residual, abs(integral - (samples[0]["mass_kg"] - s["mass_kg"])))
        out.append(integral)
        prev = s
    return out, residual


@pytest.mark.slow
def test_flow_integral_matches_the_mass_trace(depletion_samples):
    s = depletion_samples
    flow_peak = max(x["flow_kg_s"] for x in s)
    dt = (s[-1]["t_s"] - s[0]["t_s"]) / (len(s) - 1)
    integral, residual = _flow_integral_kg(s)
    # the whole 60 t load, integrated from the engine's own flow reports...
    assert integral[-1] == pytest.approx(60000.0, rel=2e-2)
    # ... agrees with the mass the vehicle reports having lost, within the
    # declared one-frame drain lag (plus the half-frame ignition ramp)
    assert residual <= 1.5 * flow_peak * dt + 2.0
    # and the tank bookkeeping closes: mass - dry = tanks, every step
    for x in s[::37]:
        assert x["mass_kg"] - x["propellant_kg"] == pytest.approx(10000.0, abs=13.0)
        if x["propellant_kg"] <= 0.0:
            assert x["flow_kg_s"] == 0.0  # a dry engine reports no delivered flow


@pytest.mark.slow
def test_sampled_mass_properties_match_the_declared_model(depletion_samples):
    props = _spike_model()
    first, last = mass_state_from_sample(depletion_samples[0]), \
        mass_state_from_sample(depletion_samples[-1])
    for state, ref in ((first, props.full), (last, props.empty)):
        assert state.mass_kg == pytest.approx(ref.mass_kg, rel=1e-6)
        for got, want in zip(state.cg_m, ref.cg_m):
            assert got == pytest.approx(want, abs=2e-2)
        for got, want in zip(state.inertia_kg_m2, ref.inertia_kg_m2):
            assert got == pytest.approx(want, rel=2e-3)
    # along the burn: the model interpolated at each step's own tank load
    worst_cg = worst_i = 0.0
    for x in depletion_samples:
        ref = props.at_propellant_mass_kg(x["propellant_kg"])
        state = mass_state_from_sample(x)
        worst_cg = max(worst_cg, max(abs(a - b) for a, b in zip(state.cg_m, ref.cg_m)))
        worst_i = max(worst_i, max(abs(a - b) / max(abs(b), 1.0)
                                   for a, b in zip(state.inertia_kg_m2,
                                                   ref.inertia_kg_m2)))
    assert worst_cg <= 0.05  # m: the CG moves 1.7 m over the burn
    assert worst_i <= 2e-2   # the drain's one-frame lag, worst near burnout
    # sixdof/mass_properties.py documents the CG moving forward as the forward
    # LOX goes first: the trend is what the F-03 interpolation buys over a
    # fixed-CG model, so check that it is actually there.
    assert mass_state_from_sample(depletion_samples[-1]).cg_m[0] \
        > mass_state_from_sample(depletion_samples[0]).cg_m[0]


@pytest.mark.slow
def test_packaged_f03_scenario_passes_its_declared_checks():
    cfg = load_config(packaged_config(F03_CONFIG))
    assert cfg.mass_model is not None
    with tempfile.TemporaryDirectory() as out:
        run = run_ascent(cfg, out)
    by_id = {c.id: c for c in run.verification.checks}
    for cid in ("sixdof-mass-conservation", "sixdof-cg-full-load",
                "sixdof-inertia-full-load", "sixdof-cg-interpolation",
                "sixdof-inertia-interpolation", "sixdof-cg-burnout",
                "sixdof-inertia-burnout"):
        assert cid in by_id, cid
        assert by_id[cid].passed, (cid, by_id[cid].value, by_id[cid].limit)
        assert by_id[cid].limit is not None  # limits come from the config
    assert run.verification.physics_verified
    assert not any(c.id.endswith("burnout-not-assessed") for c in run.verification.checks)


@pytest.mark.slow
def test_burnout_endpoint_reports_not_assessed_when_the_run_stops_early(tmp_path):
    """The same config flown for 30 s: the tanks are still 70% full, so the
    burnout claim is not silently passed -- it says so."""
    cfg = load_config(packaged_config(F03_CONFIG))
    short = load_config(dict(cfg.raw, duration_s=30.0,
                             mass_model=dict(cfg.raw["mass_model"],
                                             tolerances={"burnout_fraction": 0.001})))
    run = run_ascent(short, tmp_path)
    by_id = {c.id: c for c in run.verification.checks}
    assert by_id["sixdof-mass-model-burnout-not-assessed"].passed is False
    assert by_id["sixdof-mass-model-burnout-not-assessed"].reason_code \
        == "sixdof_burnout_not_reached"
    assert "sixdof-cg-burnout" not in by_id
    assert by_id["sixdof-mass-conservation"].passed


def test_packaged_f03_endpoints_match_the_aircraft_xml():
    """The config's declared endpoints are the aircraft file's own numbers --
    'endpoint values consistent with the configuration' cuts both ways."""
    cfg = load_config(packaged_config(F03_CONFIG))
    full = _xml_full_state()
    declared = cfg.mass_model.properties
    assert declared.full.mass_kg == pytest.approx(full.mass_kg, rel=1e-6)
    assert declared.full.cg_m[0] == pytest.approx(full.cg_m[0], abs=1e-6)
    for got, want in zip(declared.full.inertia_kg_m2, full.inertia_kg_m2):
        assert got == pytest.approx(want, rel=5e-4)
    assert declared.empty.mass_kg == DRY[0]
    assert declared.empty.cg_m[0] == DRY[1]
    assert declared.empty.inertia_kg_m2[:3] == (DRY[2], DRY[3], DRY[4])


@pytest.mark.slow
def test_mass_model_absent_keeps_the_backend_checks_unchanged():
    """F-03 checks only exist for configs that declare a mass model: the
    pre-F-03 packaged scenarios carry exactly the checks they did before."""
    cfg = load_config(packaged_config("fl401_vertical_hop"))
    assert cfg.mass_model is None
    with tempfile.TemporaryDirectory() as out:
        run = run_ascent(cfg, out)
    ids = {c.id for c in run.verification.checks}
    assert not any(k in i for i in ids
                   for k in ("mass-conservation", "full-load", "burnout", "interpolation"))
    assert "sixdof-states-finite" in ids
