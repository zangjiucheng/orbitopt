"""Flight Lab E4 (Epic #196): JSBSim 6-DOF feasibility spike.

Spike code only -- time-boxed go/no-go work, not a production backend. The
package imports nothing heavy itself (``import orbitopt.flightlab.sixdof``
works without jsbsim); import the driver explicitly::

    from orbitopt.flightlab.sixdof.jsbsim_driver import SixDofSim, InitialConditions

FL-402 (#207) adds the staging handoff spike: freeze the stack state and fly
two independent top-level instances (booster + upper stage) side by side::

    from orbitopt.flightlab.sixdof.staging import run_staging_scenario, to_flight_run

FL-403 (#208) adds the closed-loop attitude-tracking spike (PID on the TVC
gimbals following the 3-DOF ascent program to MECO) and the go/no-go ADR::

    from orbitopt.flightlab.sixdof.tracking import run_tracking_spike
    # docs/adr/flightlab-0001-jsbsim-6dof.md

``jsbsim==1.3.3`` is an opt-in pip dependency (``envs/sixdof-requirements.txt``
or ``pip install orbitopt[sixdof]``); the JSBSim model XML lives under
``orbitopt/data/flightlab/jsbsim/``.
"""
