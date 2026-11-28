"""Textbook 1-D heat conduction: a steady layered wall, and a transient slab by finite differences."""

from __future__ import annotations

from ..spec import CostModel, DomainError, Interval, Tool, ToolSpec
from ._common import inputs, integer, number, numbers, outputs, tests

_TEMPERATURE = Interval(1.0, 5000.0)


def layered_wall(thicknesses: list, conductivities: list, t_hot: float, t_cold: float,
                 h_hot: float, h_cold: float) -> dict:
    """Series thermal resistances: R = 1/h_hot + sum(L_i/k_i) + 1/h_cold per unit area, q = (T_hot - T_cold)/R."""
    if len(thicknesses) != len(conductivities):
        raise DomainError(f"{len(thicknesses)} thicknesses but {len(conductivities)} conductivities")
    resistances = [1.0 / h_hot, *(L / k for L, k in zip(thicknesses, conductivities)), 1.0 / h_cold]
    total = sum(resistances)
    flux = (t_hot - t_cold) / total
    surfaces, temperature = [], t_hot
    for r in resistances[:-1]:
        temperature -= flux * r
        surfaces.append(temperature)
    return {"heat_flux": flux, "resistance": total, "surface_temperatures": surfaces}


def transient_slab(length: float, diffusivity: float, time: float, position: float, initial_temperature: float,
                   boundary_temperature: float, nx: int, nt: int, theta: float) -> dict:
    """Theta-scheme finite differences for u_t = alpha u_xx on [0, L], u(0) = u(L) = T_b, u(x, 0) = T_0."""
    import numpy as np
    from scipy.linalg import solve_banded

    if not 0.0 <= position <= length:
        raise DomainError(f"position {position} is outside the slab [0, {length}]")
    dx, dt = length / nx, time / nt
    r = diffusivity * dt / dx ** 2
    if theta < 0.5 and r > 0.5 / (1.0 - 2.0 * theta):
        raise DomainError(f"mesh ratio alpha*dt/dx^2 = {r:.4g} exceeds the stability limit "
                          f"{0.5 / (1.0 - 2.0 * theta):.4g} of theta = {theta}; raise nt or theta")
    u = np.full(nx - 1, initial_temperature - boundary_temperature)
    bands = np.empty((3, nx - 1))
    bands[0], bands[1], bands[2] = -theta * r, 1.0 + 2.0 * theta * r, -theta * r
    explicit = (1.0 - theta) * r
    for _ in range(nt):
        rhs = (1.0 - 2.0 * explicit) * u
        rhs[1:] += explicit * u[:-1]
        rhs[:-1] += explicit * u[1:]
        u = solve_banded((1, 1), bands, rhs) if theta > 0.0 else rhs
    nodes = np.linspace(0.0, length, nx + 1)
    profile = np.concatenate(([0.0], u, [0.0]))
    return {"temperature": boundary_temperature + float(np.interp(position, nodes, profile)), "mesh_ratio": r}


TOOLS = (
    Tool(ToolSpec(
        tool_id="heat.layered_wall.v1",
        description="Steady heat flux through a plane wall of layers in series, with convection on both faces, "
                    "and the temperature at each face and interface.",
        input_schema=inputs({
            "thicknesses": numbers("Layer thicknesses, hot side first."),
            "conductivities": numbers("Layer thermal conductivities, same order."),
            "t_hot": number("Hot-side fluid temperature."), "t_cold": number("Cold-side fluid temperature."),
            "h_hot": number("Hot-side film coefficient."), "h_cold": number("Cold-side film coefficient."),
        }),
        output_schema=outputs({
            "heat_flux": number("Heat flux from hot to cold side."),
            "resistance": number("Total thermal resistance per unit area."),
            "surface_temperatures": {"type": "array", "items": {"type": "number"},
                                     "description": "Hot face, each interface, cold face."},
        }),
        input_units={"thicknesses": "m", "conductivities": "W/(m*K)", "t_hot": "K", "t_cold": "K",
                     "h_hot": "W/(m^2*K)", "h_cold": "W/(m^2*K)"},
        output_units={"heat_flux": "W/m^2", "resistance": "m^2*K/W", "surface_temperatures": "K"},
        assumptions=("Steady state, one-dimensional conduction normal to the wall.",
                     "Constant conductivity in each layer, perfect contact between layers, no heat generation.",
                     "Uniform convective film coefficients on both faces; radiation neglected."),
        domain={"thicknesses": Interval(1e-6, 10.0), "conductivities": Interval(1e-3, 1e4),
                "t_hot": _TEMPERATURE, "t_cold": _TEMPERATURE,
                "h_hot": Interval(1e-2, 1e7), "h_cold": Interval(1e-2, 1e7)},
        cost=CostModel("trivial"),
        tested_by=tests("test_layered_wall_matches_hand_calculation_and_conserves_flux"),
    ), layered_wall),
    Tool(ToolSpec(
        tool_id="heat.transient_slab.v1",
        description="Temperature at one point and time in a slab whose faces are suddenly held at a boundary "
                    "temperature, by theta-scheme finite differences (0 explicit, 0.5 Crank-Nicolson, 1 implicit).",
        input_schema=inputs({
            "length": number("Slab thickness L."), "diffusivity": number("Thermal diffusivity alpha."),
            "time": number("Time since the faces changed temperature."),
            "position": number("Distance from one face, in [0, L]."),
            "initial_temperature": number("Uniform initial temperature."),
            "boundary_temperature": number("Temperature both faces are held at."),
            "nx": integer("Grid intervals across the slab."), "nt": integer("Equal time steps."),
            "theta": number("Implicitness: 0 explicit, 0.5 Crank-Nicolson, 1 implicit Euler."),
        }),
        output_schema=outputs({"temperature": number("Temperature at position and time."),
                               "mesh_ratio": number("alpha*dt/dx^2.")}),
        input_units={"length": "m", "diffusivity": "m^2/s", "time": "s", "position": "m",
                     "initial_temperature": "K", "boundary_temperature": "K", "nx": "", "nt": "", "theta": ""},
        output_units={"temperature": "K", "mesh_ratio": ""},
        assumptions=("One-dimensional conduction with constant diffusivity and no heat generation.",
                     "Uniform initial temperature; both faces held at the boundary temperature for t > 0.",
                     "Second-order central differences in space on nx uniform intervals, nt equal time steps; "
                     "the value at position is linearly interpolated between nodes.",
                     "Error is O(dx^2) in space and O(dt) in time (O(dt^2) at theta = 0.5); a scheme with theta < 0.5 "
                     "is refused when its mesh ratio is past the stability limit."),
        domain={"length": Interval(1e-4, 10.0), "diffusivity": Interval(1e-9, 1e-2), "time": Interval(0.0, 1e9),
                "position": Interval(0.0, 10.0), "initial_temperature": _TEMPERATURE,
                "boundary_temperature": _TEMPERATURE, "nx": Interval(2, 1000), "nt": Interval(1, 20_000),
                "theta": Interval(0.0, 1.0)},
        cost=CostModel("moderate", "nt"),
        replay_tolerance=1e-10,
        tested_by=tests("test_transient_slab_matches_series_solution_with_measured_orders"),
    ), transient_slab),
)
