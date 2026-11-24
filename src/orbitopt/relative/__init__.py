"""Relative orbital dynamics: LVLH/RTN frames, the CW planning model, and a
nonlinear two-spacecraft propagator (E12-02).

* :mod:`orbitopt.relative.frames` -- ECI <-> LVLH/RTN kinematics of a pair.
* :mod:`orbitopt.relative.cw` -- closed-form Clohessy-Wiltshire propagation
  (fast planning model; see its docstring and
  ``docs/benchmarks/e12_02_cw_validity.json`` for its measured scope).
* :mod:`orbitopt.relative.propagation` -- nonlinear pair propagation
  (two-body + optional J2): the model acceptance runs on.
"""
from orbitopt.relative.cw import cw_state_transition, mean_motion_circular, propagate_cw
from orbitopt.relative.frames import eci_to_lvlh, lvlh_dcm, lvlh_to_eci
from orbitopt.relative.propagation import EARTH_J2, EARTH_MU, EARTH_RE, PairTrajectory, propagate_pair

__all__ = [
    "cw_state_transition",
    "mean_motion_circular",
    "propagate_cw",
    "eci_to_lvlh",
    "lvlh_dcm",
    "lvlh_to_eci",
    "PairTrajectory",
    "propagate_pair",
    "EARTH_MU",
    "EARTH_J2",
    "EARTH_RE",
]
