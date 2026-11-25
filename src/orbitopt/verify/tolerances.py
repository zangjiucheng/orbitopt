"""The verification layer's integration tolerances, in a module that imports
nothing heavy.

:meth:`orbitopt.verify.tudat_propagate.propagate_pair_with_events` keeps these
values as its defaults, and so does the E12-07 independent-integration spot
check (:mod:`orbitopt.rendezvous.robustness`), but a numeric tolerance must not
be reachable only through ``import tudatpy`` -- the point of the cross-check is
that environments *without* tudatpy can still bound their integration error.
So the numbers live here, and the propagator module imports them from here."""

from __future__ import annotations

__all__ = ["VERIFY_ATOL", "VERIFY_RTOL"]

VERIFY_RTOL = 1e-11
VERIFY_ATOL = 1e-6  # m and m/s (element-wise on the Cartesian state)
