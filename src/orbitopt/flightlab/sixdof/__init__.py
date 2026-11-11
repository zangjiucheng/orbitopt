"""Flight Lab E4 (Epic #196): JSBSim 6-DOF feasibility spike.

Spike code only -- time-boxed go/no-go work, not a production backend. The
package imports nothing heavy itself (``import orbitopt.flightlab.sixdof``
works without jsbsim); import the driver explicitly::

    from orbitopt.flightlab.sixdof.jsbsim_driver import SixDofSim, InitialConditions

``jsbsim==1.3.3`` is an opt-in pip dependency (``envs/sixdof-requirements.txt``
or ``pip install orbitopt[sixdof]``); the JSBSim model XML lives under
``orbitopt/data/flightlab/jsbsim/``.
"""
