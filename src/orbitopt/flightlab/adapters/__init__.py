"""Flight Lab physics-backend adapters (FL-004+).

One module per external ascent/simulation engine that produces FlightResult
v1 run directories through the flightlab contract. This package imports
nothing itself -- import the adapter you need, e.g.::

    from orbitopt.flightlab.adapters.opentsiolkovsky import run_ascent
"""
