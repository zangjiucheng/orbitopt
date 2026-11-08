"""The two-body conservation check must evaluate energy with the GM tudat
actually integrates with. tudat's default Earth field carries GM
3.986004415e14, the SPICE kernel 3.9860043547e14 (~1.5e-8 relative) -- using
the latter made every default (Earth-centred, free-return-style) report fail
its 1e-8 conservation tolerance."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tudatpy")

from orbitopt.verify.tudat_propagate import (  # noqa: E402
    body_gravitational_parameter,
    propagate_multi_arc,
    propagation_report_section,
)


def test_default_earth_report_conservation_uses_propagation_gm():
    r = 6578e3
    v = np.sqrt(3.986e14 / r)
    res = propagate_multi_arc(
        [r, 0, 0], [0, v, 0], 8.0e8,
        arcs=[{"type": "impulsive_burn", "delta_v": [0, 3150.0, 0]}, {"type": "coast", "duration": 86400.0}],
        step_size=600.0,
    )
    gm = res.model["central_gravitational_parameter"]
    assert gm != body_gravitational_parameter("Earth")  # the mismatch this guards against is real
    cons = propagation_report_section(res, "tli", convergence_check=False)["conservation"]
    tb = cons["two_body_reference"]
    assert tb["mu_m3_s2"] == gm
    assert tb["within_tolerance"], tb
