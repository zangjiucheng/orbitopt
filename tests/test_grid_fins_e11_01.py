"""E11-01 (task #308) unit checks for orbitopt.flightlab.recovery.grid_fins.

Covers the acceptance list for the grid-fin model:

* two-state switching   -- deploy only below deploy altitude AND above the q
  gate, stow below the stow altitude (the altitude gap is the hysteresis),
  and nothing steers while stowed;
* deflection saturation -- realised deflection never exceeds +-delta_max
  under any trim + roll demand, and alpha commands are authority-limited;
* rate limiting         -- a 1 Hz command update may not move a fin more
  than delta_rate * dt (nor bank more than bank_rate * dt);
* force / moment sign and magnitude -- deployed lift positive for positive
  alpha and negative for the sign-flipped deflection, drag increment
  positive and growing with alpha, pitch slope restoring (weathervane) and
  the trim law cancelling it, roll moment linear in the aileron differential
  with the documented sign, and the forces at a reference entry condition in
  the physical band for a ~9 m / 400 t class booster.

Everything here is model/actuator-level (no propagation): propagation with
the fins is covered by the E11-01 benchmark and the DC8 planner runs.
"""
import math

import numpy as np
import pytest

from orbitopt.flightlab.recovery.grid_fins import (
    STOWED,
    X_CONFIG_SIGN,
    DESIGN_ASSUMPTIONS,
    GridFinActuator,
    GridFinCommand,
    GridFinGuidance,
    GridFinModel,
)

Q_REF_PA = 50_000.0      # a mid-entry dynamic pressure for the moment tests
MACH_REF = 4.0


def _model(**over):
    kw = dict(reference_area_m2=63.617, count=4, fin_area_m2=4.0,
              cd_add_mach=((0.3, 1.65), (1.0, 2.35), (2.0, 2.10),
                           (5.0, 1.60), (12.0, 1.35)),
              cn_alpha_mach=((0.3, 1.55), (1.0, 1.15), (2.0, 0.85),
                             (5.0, 0.60), (12.0, 0.50)),
              body_cn_alpha_per_rad=2.0, trim_effectiveness=1.8,
              delta_max_deg=40.0, delta_rate_max_deg_s=20.0,
              bank_max_deg=60.0, bank_rate_max_deg_s=30.0,
              alpha_max_deg=60.0, stall_alpha_deg=45.0,
              deploy_altitude_m=70_000.0, stow_altitude_m=1500.0,
              deploy_q_min_pa=100.0, induced_k=0.28, length_m=71.0,
              diameter_m=9.0, static_margin=0.08, roll_lever_ratio=0.5)
    kw.update(over)
    return GridFinModel(**kw)


class TestTwoStateSwitching:
    def test_stowed_until_gate(self):
        act = GridFinActuator(_model())
        # above the deploy altitude: no deployment, no lift
        s = act.step(1.0, GridFinCommand(30.0, 30.0), height_m=80_000.0,
                     dynamic_pressure_pa=5_000.0, mach=MACH_REF)
        assert not s.deployed and s.cl == 0.0 and s.cd_increment == 0.0
        # below the deploy altitude but below the q gate: still stowed
        s = act.step(1.0, GridFinCommand(30.0, 30.0), height_m=69_000.0,
                     dynamic_pressure_pa=99.9, mach=MACH_REF)
        assert not s.deployed

    def test_deploy_then_stow_with_hysteresis(self):
        act = GridFinActuator(_model())
        s = act.step(1.0, GridFinCommand(30.0, 0.0), height_m=69_000.0,
                     dynamic_pressure_pa=1_000.0, mach=MACH_REF)
        assert s.deployed and s.cl > 0.0
        # between stow and deploy altitude: stays deployed (hysteresis gap),
        # bank demand building toward 30 deg
        for _ in range(2):
            s = act.step(1.0, GridFinCommand(30.0, 30.0), height_m=20_000.0,
                         dynamic_pressure_pa=20_000.0, mach=MACH_REF)
        assert s.deployed and s.bank_deg > 0.0
        # below the stow altitude: stow, forces vanish, bank slews out at
        # the bank rate (not instantly -- dt 0.2 s moves it by <= 6 deg)
        s = act.step(0.2, GridFinCommand(30.0, 30.0), height_m=1_499.0,
                     dynamic_pressure_pa=30_000.0, mach=0.9)
        assert not s.deployed and s.cl == 0.0
        assert 0.0 < s.bank_deg < 30.0 and abs(s.bank_deg - 24.0) < 1e-9
        s2 = act.step(0.2, GridFinCommand(30.0, 30.0), height_m=1_490.0,
                      dynamic_pressure_pa=30_000.0, mach=0.9)
        assert abs(s2.bank_deg) < abs(s.bank_deg)  # still slewing back to zero

    def test_snapshot_matches_last_step(self):
        act = GridFinActuator(_model())
        s = act.step(1.0, GridFinCommand(30.0, 10.0), height_m=60_000.0,
                     dynamic_pressure_pa=2_000.0, mach=MACH_REF)
        assert act.snapshot is s


class TestSaturation:
    def test_alpha_authority_limit(self):
        m = _model()
        assert m.alpha_authority_deg == min(60.0, 1.8 * 40.0)  # 60 vs 72
        act = GridFinActuator(m)
        s = act.step(1.0, GridFinCommand(75.0, 0.0), height_m=60_000.0,
                     dynamic_pressure_pa=2_000.0, mach=MACH_REF)
        assert s.alpha_deg == pytest.approx(m.alpha_authority_deg)

    def test_deflection_never_exceeds_delta_max(self):
        # weak trim g = 1.2: alpha 60 deg is authority-limited to 48 deg, the
        # trim deflection is then exactly delta_max -- any roll demand rides
        # on a saturated pair and the clip keeps every fin inside +-40 deg
        act = GridFinActuator(_model(trim_effectiveness=1.2), deployed=True)
        worst = 0.0
        for bank in (60.0, -60.0, 60.0):
            act.step(0.5, GridFinCommand(60.0, bank), height_m=40_000.0,
                     dynamic_pressure_pa=20_000.0, mach=MACH_REF)
            worst = max(worst, float(np.max(np.abs(act.delta_deg))))
        assert worst <= 40.0 + 1e-12
        # default model: X-pair differential keeps two fins tighter than the
        # others under full bank (they carry the differential first)
        act2 = GridFinActuator(_model(), deployed=True)
        for _ in range(60):
            act2.step(0.5, GridFinCommand(30.0, 60.0), height_m=40_000.0,
                      dynamic_pressure_pa=20_000.0, mach=MACH_REF)
        hi, lo = act2.delta_deg[0], act2.delta_deg[1]
        assert hi > lo
        assert hi <= 40.0 and lo >= -40.0

    def test_bank_saturates_at_bank_max(self):
        act = GridFinActuator(_model(), deployed=True)
        for _ in range(200):
            s = act.step(0.5, GridFinCommand(30.0, 500.0), height_m=40_000.0,
                         dynamic_pressure_pa=20_000.0, mach=MACH_REF)
        assert s.bank_deg == pytest.approx(60.0)


class TestRateLimiting:
    def test_fin_and_bank_slew_limits(self):
        act = GridFinActuator(_model(), deployed=True)
        act.step(0.001, GridFinCommand(0.0, 0.0), height_m=60_000.0,
                 dynamic_pressure_pa=1_000.0, mach=MACH_REF)
        d0 = act.delta_deg.copy()
        b0 = act.bank_deg
        dt = 0.1
        act.step(dt, GridFinCommand(30.0, 45.0), height_m=60_000.0,
                 dynamic_pressure_pa=1_000.0, mach=MACH_REF)
        assert float(np.max(np.abs(act.delta_deg - d0))) <= 20.0 * dt + 1e-9
        assert abs(act.bank_deg - b0) <= 30.0 * dt + 1e-9

    def test_guidance_schedule_mode_reaches_target(self):
        g = GridFinGuidance(model=_model(), mode="schedule",
                            schedule=((60_000.0, 0.0), (30_000.0, -45.0, 20.0)))
        assert g.command(75_000.0, [1, 0, 0], [0, 1, 0]) is None   # above window
        # above the first gate: default bank 0, default alpha (30)
        assert g.command(65_000.0, [1, 0, 0], [0, 1, 0]) == GridFinCommand(30.0, 0.0)
        # below the last gate: that gate's bank AND its own alpha
        assert g.command(25_000.0, [1, 0, 0], [0, 1, 0]) == GridFinCommand(20.0, -45.0)
        # a 2-gate schedule holds the gate bank to the stow altitude
        g2 = GridFinGuidance(model=_model(), mode="schedule",
                             schedule=((70_000.0, 0.0, 40.0), (20_000.0, 0.0, 0.0)))
        assert g2.command(50_000.0, [1, 0, 0], [0, 1, 0]) == GridFinCommand(40.0, 0.0)
        assert g2.command(10_000.0, [1, 0, 0], [0, 1, 0]) == GridFinCommand(0.0, 0.0)
        with pytest.raises(ValueError):    # gates must descend
            GridFinGuidance(model=_model(), mode="schedule",
                            schedule=((20_000.0, 0.0), (70_000.0, 0.0)))
        with pytest.raises(ValueError):    # gate alpha inside fin authority
            GridFinGuidance(model=_model(), mode="schedule",
                            schedule=((60_000.0, 0.0, 75.0),))


class TestForcesAndMoments:
    def test_signs(self):
        m = _model()
        # lift: positive for positive alpha, antisymmetric, monotone up to the
        # stall clip
        assert m.cl(30.0, MACH_REF) > 0.0
        assert abs(m.cl(-30.0, MACH_REF) + m.cl(30.0, MACH_REF)) < 1e-12
        assert m.cl(20.0, MACH_REF) < m.cl(30.0, MACH_REF)
        # drag increment: positive, increasing with alpha, zero at zero alpha
        assert m.cd_increment(0.0, MACH_REF) == pytest.approx(m.cd_add(MACH_REF))
        assert m.cd_increment(30.0, MACH_REF) > m.cd_increment(10.0, MACH_REF)
        # pitch: restoring slope; the trim deflection cancels it exactly
        assert m.cm_alpha(MACH_REF) < 0.0
        assert m.pitch_moment_nm(20.0, MACH_REF, Q_REF_PA) < 0.0
        assert m.pitch_moment_nm(20.0, MACH_REF, Q_REF_PA,
                                 delta_ctrl_deg=m.trim_delta_deg(20.0)) \
            == pytest.approx(0.0, abs=1.0)
        # a nose-up control deflection beyond trim yields a nose-up moment
        assert m.pitch_moment_nm(0.0, MACH_REF, Q_REF_PA, delta_ctrl_deg=5.0) > 0.0
        # roll: linear in the differential, positive = bank right
        r1 = m.roll_moment_nm(10.0, MACH_REF, Q_REF_PA)
        assert r1 > 0.0
        assert m.roll_moment_nm(-10.0, MACH_REF, Q_REF_PA) == pytest.approx(-r1)
        assert m.roll_moment_nm(20.0, MACH_REF, Q_REF_PA) == pytest.approx(2 * r1)

    def test_magnitudes_at_entry_reference(self):
        m = _model()
        q = 50_000.0
        v_ref = math.sqrt(2.0 * q / 0.02)  # 50 kPa at rho ~0.02 kg/m^3
        l_n = 0.5 * q * m.cl(30.0, MACH_REF) * m.reference_area_m2
        d_add_n = 0.5 * q * m.cd_increment(30.0, MACH_REF) * m.reference_area_m2
        # a 400 t booster at 50 kPa: lift between 10 % and 80 % of weight is
        # a controllable but survivable band; deployed drag increment at
        # least 1.5x stowed reference drag at the same q
        assert 0.1 * 400_000.0 * 9.81 < l_n < 0.8 * 400_000.0 * 9.81
        assert d_add_n > 1.5 * 0.5 * q * 1.2 * m.reference_area_m2
        assert v_ref > 0.0

    def test_x_config_sign_alternates(self):
        assert X_CONFIG_SIGN == (1.0, -1.0, -1.0, 1.0)
        assert sum(X_CONFIG_SIGN) == 0.0  # no net pitch coupling from roll

    def test_design_assumptions_are_documented(self):
        assert len(DESIGN_ASSUMPTIONS) >= 4
        assert all("+-" in a for a in DESIGN_ASSUMPTIONS)


class TestVehicleLoad:
    def test_dc8_yaml_carries_the_fins(self):
        from orbitopt.flightlab.recovery.propagator import BoosterModel
        veh = BoosterModel.load(
            "src/orbitopt/data/flightlab/vehicles/starship_class_rtls_dc8.yaml")
        gf = veh.grid_fins
        assert gf is not None and gf.count == 4
        assert gf.provenance == "public_estimate"
        assert 0.15 < gf.fin_area_ratio < 0.4
        assert gf.alpha_authority_deg == pytest.approx(60.0)
