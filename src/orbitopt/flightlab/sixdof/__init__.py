"""Flight Lab E4 (Epic #196): JSBSim 6-DOF drivers -- spike-proven, now the
engine room of the formal 6-DOF backend.

The modules below are the FL-401/402/403 spike drivers (:mod:`jsbsim_driver`
single-body simulation, :mod:`staging` two-instance handoff, :mod:`tracking`
closed-loop TVC) plus :mod:`mass_properties`, the declared variable mass / CG
/ inertia model F-03 (task #228) checks the drivers' mass bookkeeping against.
F-01 (task #226) promoted the spike path to a supported,
config-driven backend -- ``backend="6DOF"`` in
:func:`orbitopt.flightlab.simulate.simulate` / ``orbitopt flight simulate
--backend 6DOF``, scenario configs in ``orbitopt/data/flightlab/sixdof/``:

    from orbitopt.flightlab.adapters.sixdof import run_ascent, packaged_config
    run = run_ascent(packaged_config("fl401_vertical_hop"), Path("runs/jsb6-1"))

The direct driver APIs below remain the building blocks (and what the
backend drives); prefer the adapter for scenario flights. The package
imports nothing heavy itself (``import orbitopt.flightlab.sixdof`` works
without jsbsim)::

    from orbitopt.flightlab.sixdof.jsbsim_driver import SixDofSim, InitialConditions
    from orbitopt.flightlab.sixdof.staging import run_staging_scenario, to_flight_run
    from orbitopt.flightlab.sixdof.tracking import run_tracking_spike
    # docs/adr/flightlab-0001-jsbsim-6dof.md (the FL-403 go ADR)

``jsbsim==1.3.3`` is an opt-in pip dependency (``envs/sixdof-requirements.txt``
or ``pip install orbitopt[sixdof]``); the JSBSim model XML lives under
``orbitopt/data/flightlab/jsbsim/``.

F-02 (task #227) adds the two modules the JSBSim backend cannot host: an
in-house coefficient model and the native plant that flies it. The bindings
expose no per-step force/moment injection hook, so the aero authored here is
flew here -- the *reference* 6-DOF plant, not a JSBSim replacement::

    from orbitopt.flightlab.sixdof.aero import AeroModel, load_packaged_aero
    from orbitopt.flightlab.sixdof.rigidbody import propagate, RigidBodyVehicle
    # docs/flightlab.md (the F-02 section, and why the split exists)

F-04 (task #229) closes that plant's attitude loop: a first-order-lag /
slew-limited / saturating TVC actuator and a basic ZOH PID tracker on the
FL-403 body-axis attitude error, with the post-actuator deflection fed to the
plant and deflection-margin diagnostics recorded per run::

    from orbitopt.flightlab.sixdof.tvc import AttitudeTvcLaw, run_closed_loop_ascent
    # docs/flightlab.md (the F-04 section, including the two-gimbal scope limit)

F-05 (task #230) flies that closed-loop plant against a point-mass 3-DOF
reference on the identical vehicle, physics and integration core -- the
OT-class model, natively -- and reports the acceptance's altitude/speed/FPA/
mass difference curves plus an exact telescoping attribution of where the
difference comes from (aerodynamics, attitude dynamics, control lag)::

    from orbitopt.flightlab.sixdof.gravity_turn import f05_scenario, run_f05_study
    # docs/flightlab.md (the F-05 section) and
    # docs/benchmarks/fl605_gravity_turn_ascent.json (the committed study)
"""
