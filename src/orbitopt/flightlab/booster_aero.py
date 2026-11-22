"""E11-05: qualified power-off drag model for the Starship-class booster.

E5-08 flagged a 12.80 % Starbase stage-1 debris impact-speed gap (band 12 %)
and E10-04 evidenced it as a genuine power-off drag-model difference rather
than an implementation artifact.  E11-05 qualifies the booster power-off drag
model with a semi-empirical component build-up (no CFD is available in this
repository's runtime, so the "high-fidelity method" the task brief allows --
"CFD or empirical/semi-empirical methods or public data" -- is the classical
DATCOM-style build-up already used for the vehicle noses in
:mod:`orbitopt.flightlab.nose_aero`, extended with the components a booster
actually carries and run against the two independent public-data checks the
repo already holds: the recorded OpenTsiolkovsky debris trajectory (an
in-flight Cd inversion) and the E11-01 grid-fin tables.

The model covers the three power-off states that matter to this epic:

``cd_engines_first``
    Booster falling engines-first, grid fins STOWED -- the attitude behind
    the vehicle's ``booster.cd_mach`` block (DC8 declares an illustrative
    blunt-cylinder table there, +/-30 %, "not derived from the declared
    nose").  Build-up: aft blunt face at the pitot (Newtonian) pressure,
    flat-base suction on the forward carry-deck base, Prandtl-Glauert
    corrected skin friction, stowed-fin protuberance drag.

``ca_stacked_stowed``
    Whole stage falling velocity-aligned, nose/ship-attachment end forward,
    fins STOWED -- the attitude and polar behind the OT ``ca_power_off``
    tables that E10-04/E5-08 check against.  Build-up: the nose-generated
    power-off table (wave + friction + base from :mod:`nose_aero`) plus the
    stowed-protuberance increment.

``fin_state_delta_cd``
    Deployed-minus-stowed increment from the E11-01 :class:`GridFinModel`
    declared tables, so the acceptance criterion "include the E11-01 grid-fin
    deployed/stowed effect on Cd" is quantified from the same single source
    both models already share.

Every component carries an explicit source tag; DESIGN ASSUMPTIONS are listed
in :data:`DESIGN_ASSUMPTIONS` with uncertainty ranges, in the style of E11-01.
The module generates the tables; it never edits a vehicle YAML (sealed
provenance digests forbid it -- see the E11-05 report for how the qualified
table is applied to a vehicle revision when one is cut).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from orbitopt.flightlab import nose_aero
from orbitopt.flightlab.recovery.grid_fins import GridFinModel

GAMMA = nose_aero.GAMMA

#: Mach grid of every table generated here (same as the vehicle declarations).
QUAL_MACH = (0.0, 0.3, 0.8, 1.0, 1.1, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 12.0)

DESIGN_ASSUMPTIONS: tuple[str, ...] = (
    "Grid-fin stowed frontal blockage: stowed fin treated as a flat fairing "
    "of frontal area planform * (t/c), t/c = 0.12 (lattice truss depth over "
    "chord) -- DESIGN ASSUMPTION from E11-01 photos, +/-50 % on t/c, which "
    "propagates linearly into the stowed increment.",
    "Protuberance interference factor K = 1.9 (Hoerner fairing-body "
    "interference) -- DESIGN ASSUMPTION, range 1.5..2.3.",
    "Engines-first aft face effective area fraction 0.85 of the body "
    "cross-section (engine gimbal ring + landing-leg recesses cover the "
    "rest) -- DESIGN ASSUMPTION, +/-10 %.",
    "Subsonic/low-supersonic forward base treated as a flat carry-deck base (DESIGN ASSUMPTION: "
    "(full suction, Missile DATCOM base term) -- the real forward end is a "
    "shallow cavity, so the true base drag is smaller; the table errs high "
    "in subsonic Cd by up to ~0.05.",
)

#: Public-data anchors this qualification is cross-checked against (see the
#: E11-05 benchmark report for the numbers and how they are used).
QUALIFICATION_SOURCES: tuple[tuple[str, str], ...] = (
    ("nose_aero primitives",
     "orbitopt.flightlab.nose_aero: Taylor-Maccoll cone/ogive wave drag, "
     "Rayleigh pitot Cp, Prandtl-Glauert corrected turbulent Cf, DATCOM "
     "power-off base term -- the repo's own semi-empirical aero generator."),
    ("E11-01 grid fins",
     "vehicle grid_fins block (cd_add_mach / cn_alpha_mach), provenance "
     "public_estimate, derivation in docs/benchmarks/"
     "e11_01_grid_fin_authority.md."),
    ("recorded debris flight",
     "tests/data/mission/reference/e5-08-new-vehicle-starbase/.../"
     "stage1_debris.csv and the DC7 twin: an in-flight power-off Cd "
     "inversion is public data in the strongest sense the repo holds -- a "
     "measured trajectory of the actual model."),
    ("F9-class ballistic coefficient",
     "f9_class_public_estimate.yaml declares ballistic_coefficient_"
     "kg_m2 for the booster stage; m/(Cd*S) at the declared masses bounds "
     "the power-off Cd from published empty-mass and diameter figures."),
)


@dataclass(frozen=True)
class StowedProtuberances:
    """Fins fairings and legs a stowed power-off booster drags along."""

    fin_planform_m2: float = 4.0
    fin_count: int = 4
    fin_thickness_to_chord: float = 0.12
    #: Hoerner fairing-body interference factor (DESIGN ASSUMPTIONS).
    interference_k: float = 1.9

    @property
    def wetted_m2(self) -> float:
        return self.fin_count * 2.0 * self.fin_planform_m2

    @property
    def frontal_m2(self) -> float:
        return self.fin_count * self.fin_planform_m2 * self.fin_thickness_to_chord

    def cd_increment(self, mach: float, ref_area_m2: float) -> float:
        """Delta-CD0 from stowed fins: both-side friction on the fairing
        faces (interference-corrected) plus frontal blockage at blunt-plate
        Cd.  Mach dependence enters through the compressibility correction on
        the friction term."""
        cf = 0.455 / math.log10(nose_aero.DEFAULT_REYNOLDS) ** 2.58
        cf /= (1.0 + 0.144 * mach * mach) ** 0.65
        friction = self.interference_k * 2.0 * cf * self.wetted_m2 / ref_area_m2
        frontal = 2.0 * self.frontal_m2 / ref_area_m2
        return friction + frontal


#: The DC8 declaration read from the vehicle grid_fins block (E11-01).
DC8_STOWED = StowedProtuberances()
#: The Starbase v1 record carries no fins and no legs fairings; the only
#: protuberances on the recorded debris article are the interstage ring and
#: the forward carry-deck antennas, bounded below the grid-fin increment.
STARBASE_V1_STOWED = StowedProtuberances(fin_planform_m2=0.0, fin_count=0)


def _pitot_cp(mach: float) -> float:
    """Face stagnation pressure coefficient: 1.0 by definition in subsonic
    flow (dynamic-pressure reference), Rayleigh pitot ratio above M = 1."""
    if mach <= 1.0:
        return 1.0
    return float(nose_aero._pitot_cp(mach))


def cd_engines_first(mach: float, *, diameter_m: float = 9.0,
                     length_m: float = 71.0,
                     protuberances: StowedProtuberances = DC8_STOWED,
                     aft_face_fraction: float = 0.85,
                     reynolds: float = nose_aero.DEFAULT_REYNOLDS) -> float:
    """Engines-first, fins-stowed power-off Cd over the body cross-section."""
    ref = 0.25 * math.pi * diameter_m ** 2
    wetted = math.pi * diameter_m * length_m
    aft = aft_face_fraction * _pitot_cp(mach)
    base = aft_face_fraction * nose_aero.base_drag_ca(mach)
    cf = 0.455 / math.log10(reynolds) ** 2.58 / (1.0 + 0.144 * mach * mach) ** 0.65
    friction = cf * wetted / ref
    protuberance = protuberances.cd_increment(mach, ref)
    return aft - base + friction + protuberance


def cd_engines_first_table(
        mach: Sequence[float] = QUAL_MACH, *, diameter_m: float = 9.0,
        length_m: float = 71.0,
        protuberances: StowedProtuberances = DC8_STOWED,
        aft_face_fraction: float = 0.85, decimals: int = 3) -> list[list[float]]:
    return [[float(m), round(cd_engines_first(
        m, diameter_m=diameter_m, length_m=length_m,
        protuberances=protuberances,
        aft_face_fraction=aft_face_fraction), decimals)] for m in mach]


def ca_stacked_stowed(ca_power_off: Sequence[Sequence[float]],
                      protuberances: StowedProtuberances = STARBASE_V1_STOWED,
                      ref_area_m2: float | None = None,
                      decimals: int = 4) -> list[list[float]]:
    """Axial power-off polar: the nose-generated (or recorded-declared)
    ``ca_power_off`` curve of a vehicle plus the stowed-protuberance
    increment -- the qualified version of the polar E10-04 checks."""
    out = []
    for m, ca in ca_power_off:
        ref = float(ref_area_m2 or 0.25 * math.pi * 9.0 ** 2)
        out.append([float(m), round(float(ca) + protuberances.cd_increment(
            float(m), ref), decimals)])
    return out


def fin_state_delta_cd(gf: GridFinModel, mach: Sequence[float] = QUAL_MACH,
                       alpha_deg: float = 0.0,
                       ref_area_m2: float | None = None) -> list[list[float]]:
    """Deployed-minus-stowed Cd at zero incidence (pure profile + the
    stowed fairing increment coming back off) at each Mach, referred to the
    body cross-section the vehicle declares.  At flying alpha the deployed
    increment is larger again by the normal-force and induced terms; E11-01
    quantifies that envelope."""
    ref = float(ref_area_m2 or 0.25 * math.pi * gf.diameter_m ** 2)
    stow = StowedProtuberances(fin_planform_m2=gf.fin_area_m2,
                               fin_count=gf.count)
    return [[float(m), round(gf.cd_increment(alpha_deg, float(m))
                             - stow.cd_increment(float(m), ref), 3)]
            for m in mach]


def interp_cd(cd_mach: Sequence[Sequence[float]], mach: float) -> float:
    """The single interpolation both the propagator and the twin apply."""
    pts = [(float(m), float(c)) for m, c in cd_mach]
    return float(_interp(pts, mach))


def _interp(pts: list[tuple[float, float]], mach: float) -> float:
    xs = [p[0] for p in pts]
    if mach <= xs[0]:
        return pts[0][1]
    if mach >= xs[-1]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= mach <= x1:
            w = (mach - x0) / (x1 - x0)
            return y0 + w * (y1 - y0)
    raise AssertionError("unreachable")


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--diameter", type=float, default=9.0)
    ap.add_argument("--length", type=float, default=71.0)
    ap.add_argument("--fins", action="store_true",
                    help="include the DC8 grid-fin stowed block")
    ap.add_argument("--fin-area", type=float, default=4.0)
    ap.add_argument("--fin-count", type=int, default=4)
    args = ap.parse_args(argv)
    prot = (StowedProtuberances(fin_planform_m2=args.fin_area,
                                fin_count=args.fin_count)
            if args.fins else STARBASE_V1_STOWED)
    print("cd_mach (engines-first, stowed, generated by booster_aero E11-05):")
    print("  value:", cd_engines_first_table(
        diameter_m=args.diameter, length_m=args.length, protuberances=prot))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
