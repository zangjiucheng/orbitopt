"""Flight Lab atmosphere models (F-02, task #227): the air state a 6-DOF
aerodynamic model needs -- density, pressure, temperature, speed of sound --
as functions of geometric height, plus the dynamic pressure they imply.

The US76 layer table and its four readers (``temperature_k``,
``pressure_pa``, ``density_us76_kg_m3``, ``speed_of_sound_m_s``) lived in
:mod:`orbitopt.flightlab.recovery.propagator` since FL-301; they moved here
so the ascent side of Flight Lab can read the same table the booster
propagator does (``BoosterPhysics(density_model="ussa1976")`` selects it
there). The propagator keeps re-exporting all four names, so nothing that
imported them from there breaks.

Two model shapes are offered, both behind the tiny :class:`Atmosphere`
protocol (``density_kg_m3`` / ``temperature_k`` / ``speed_of_sound_m_s``):

* :class:`StandardAtmosphere1976` -- the layer table, Flight Lab's default;
* :class:`ExponentialAtmosphere` -- ``rho = rho_0 exp(-h/H)`` at constant
  temperature. Not "better": it is *analytically integrable*, which is what
  makes a closed-form max-Q baseline possible (the F-02 acceptance case).

``orbitopt.entry.density_kg_m3`` (Vallado Table 8-4, US76 fitted with
piecewise exponentials) is the third profile in the code base: it agrees with
the layer table to a few percent but reads ~15 % light near sea level, so a
cross-model comparison must say which one it used. Nothing here imports
``scipy``/``orbitopt.entry`` -- this module is math only, so the 6-DOF aero
stack stays importable on a base install.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

__all__ = [
    "Atmosphere",
    "ExponentialAtmosphere",
    "GAMMA_AIR",
    "P_SEA_LEVEL_PA",
    "R_SPEC_DRY_AIR",
    "STANDARD_ATMOSPHERE",
    "StandardAtmosphere1976",
    "US76_LAYERS",
    "density_us76_kg_m3",
    "dynamic_pressure_pa",
    "pressure_pa",
    "speed_of_sound_m_s",
    "temperature_k",
]

#: Standard gravity [m/s^2] -- the same value as ``orbitopt.entry.G0_M_S2``,
#: restated here because that module pulls in scipy and this one must not.
G0_M_S2 = 9.80665
#: Specific gas constant of dry air [J/(kg K)].
R_SPEC_DRY_AIR = 287.05287
#: Ratio of specific heats of air (subsonic flight regime of these models).
GAMMA_AIR = 1.4
#: ISA/US76 sea-level standard pressure [Pa].
P_SEA_LEVEL_PA = 101325.0

_G0_BY_R = G0_M_S2 / R_SPEC_DRY_AIR

#: US Standard Atmosphere 1976 layer bases (geometric height km, base
#: pressure Pa, base temperature K, lapse rate K/km) -- consistent with the
#: density bases of ``orbitopt.entry`` (both are the US76 table below 50 km).
#: Above 84.56 km the standard holds the base temperature, where
#: p/p_sea_level < 4e-7 and nothing but a 4th-decimal Isp correction sees it.
US76_LAYERS: tuple[tuple[float, float, float, float], ...] = (
    (0.0, 101325.0, 288.150, -6.5),
    (11.0, 22632.06, 216.650, 0.0),
    (20.0, 5474.889, 216.650, 1.0),
    (32.0, 868.0187, 228.650, 2.8),
    (47.0, 110.5234, 270.650, 0.0),
    (51.0, 66.93887, 270.650, -2.8),
    (71.0, 3.956420, 214.650, 0.0),
    (84.560, 0.373388, 186.870, 0.0),
)


def temperature_k(h_m: float) -> float:
    """US76 standard temperature at geometric height ``h_m`` [m] (negative
    heights clamp to sea level)."""
    h_km = max(float(h_m), 0.0) / 1000.0
    base = US76_LAYERS[0]
    for layer in US76_LAYERS:
        if h_km >= layer[0]:
            base = layer
    return base[2] + base[3] * (h_km - base[0])


def pressure_pa(h_m: float) -> float:
    """US76 standard pressure at geometric height ``h_m`` [m] (hydrostatic
    within each layer, isothermal above the top layer)."""
    h_km = max(float(h_m), 0.0) / 1000.0
    base = US76_LAYERS[0]
    for layer in US76_LAYERS:
        if h_km >= layer[0]:
            base = layer
    p_b, t_b, lapse = base[1], base[2], base[3]
    dh = (h_km - base[0]) * 1000.0
    if lapse == 0.0:
        return p_b * math.exp(-_G0_BY_R * dh / t_b)
    t = t_b + lapse * dh / 1000.0
    return p_b * (t / t_b) ** (-_G0_BY_R / (lapse / 1000.0))


def density_us76_kg_m3(h_m: float) -> float:
    """US76 density at height ``h_m`` from the layer pressure/temperature
    (ideal gas) -- the textbook table itself. See this module's docstring for
    how it relates to ``orbitopt.entry.density_kg_m3``."""
    return pressure_pa(h_m) / (R_SPEC_DRY_AIR * temperature_k(h_m))


def speed_of_sound_m_s(h_m: float) -> float:
    """Speed of sound of the US76 standard atmosphere at height ``h_m``."""
    return math.sqrt(GAMMA_AIR * R_SPEC_DRY_AIR * temperature_k(h_m))


def dynamic_pressure_pa(density_kg_m3: float, speed_m_s: float) -> float:
    """Dynamic pressure ``q = 1/2 rho V^2`` [Pa] on an air-relative ``speed``.
    The ``V`` must be the speed through the air (co-rotating atmosphere: the
    ECEF velocity, wind aside), not the ground speed."""
    return 0.5 * float(density_kg_m3) * float(speed_m_s) * float(speed_m_s)


class Atmosphere(Protocol):
    """What an aero/propulsion model needs from an atmosphere: the air
    properties a force and an Isp are read from. Both models below satisfy it;
    so does any caller's own profile (a unit test's constant-density air, a
    Monte-Carlo's dispersed one)."""

    name: str

    def density_kg_m3(self, h_m: float) -> float: ...

    def pressure_pa(self, h_m: float) -> float: ...

    def temperature_k(self, h_m: float) -> float: ...

    def speed_of_sound_m_s(self, h_m: float) -> float: ...


@dataclass(frozen=True)
class StandardAtmosphere1976:
    """The US76 layer table of this module, as an :class:`Atmosphere`
    instance: Flight Lab's default air model."""

    name: str = "ussa1976"

    def density_kg_m3(self, h_m: float) -> float:
        return density_us76_kg_m3(h_m)

    def pressure_pa(self, h_m: float) -> float:
        return pressure_pa(h_m)

    def temperature_k(self, h_m: float) -> float:
        return temperature_k(h_m)

    def speed_of_sound_m_s(self, h_m: float) -> float:
        return speed_of_sound_m_s(h_m)


@dataclass(frozen=True)
class ExponentialAtmosphere:
    """``rho(h) = rho_0 exp(-h/H)`` at a constant temperature -- the
    textbook idealisation, kept because it is analytically integrable: for
    constant vertical acceleration the max-Q time is ``sqrt(2 H / a)``, which
    is the closed-form baseline the F-02 tests check the 6-DOF run against.

    Defaults are the sea-level US76 values with the classic 8.5 km scale
    height; ``temperature_k`` sets the (constant) speed of sound."""

    density_sea_level_kg_m3: float = 1.225
    scale_height_m: float = 8500.0
    temperature_k_value: float = 288.150
    name: str = "exponential"

    def __post_init__(self) -> None:
        if self.density_sea_level_kg_m3 <= 0.0:
            raise ValueError("density_sea_level_kg_m3 must be positive")
        if self.scale_height_m <= 0.0:
            raise ValueError("scale_height_m must be positive")
        if self.temperature_k_value <= 0.0:
            raise ValueError("temperature_k_value must be positive")

    def density_kg_m3(self, h_m: float) -> float:
        return self.density_sea_level_kg_m3 * math.exp(-max(float(h_m), 0.0)
                                                       / self.scale_height_m)

    def pressure_pa(self, h_m: float) -> float:
        """Ideal-gas pressure of the same exponential column at the constant
        temperature (hydrostatically consistent: ``p = rho R T``, so
        ``dp/dh = -rho g`` holds by construction)."""
        return self.density_kg_m3(h_m) * R_SPEC_DRY_AIR * self.temperature_k_value

    def temperature_k(self, h_m: float) -> float:
        return self.temperature_k_value

    def speed_of_sound_m_s(self, h_m: float) -> float:
        return math.sqrt(GAMMA_AIR * R_SPEC_DRY_AIR * self.temperature_k_value)


#: The default atmosphere for every Flight Lab aero read.
STANDARD_ATMOSPHERE = StandardAtmosphere1976()
