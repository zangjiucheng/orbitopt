# GOES Launch-to-GEO Mission Plan Design

This GOES-16-like reference models a transfer orbit and a spacecraft
apogee-raising campaign. It is a planning model, not a reconstruction of
unpublished burn-by-burn telemetry.

{: .warning }
> The built-in GOES scene loads **NOT CHECKED**. Run high-fidelity
> verification from the Verification tab, or enable `verification.enabled`
> in a `geo-raising` config. A smooth animation is not a verification result.

## 1. Mission Overview

GOES-R/16, GOES-S/17 and GOES-T/18 launched on Atlas V 541; GOES-U/19 used
Falcon Heavy. The reference profile below uses the GOES-16 Atlas V case, with
an approximately 5,192 kg spacecraft. The goal is a circular, equatorial GEO
orbit at about 35,786 km altitude (42,164 km radius).

Centaur's three burns place the spacecraft in an elevated-perigee transfer
orbit. The spacecraft then raises perigee and removes remaining inclination
through apogee burns before drifting to its slot.

## 2. Mission Phases

| Phase | Reference or assumption |
|---|---|
| Centaur burn 1 | Parking orbit: approximately 167 × 541 km, 28.15°; period about 91.6 min. |
| Burn 2 | Transfer orbit: approximately 187 × 32,716 km, 25.68°. |
| Burn 3 and separation | Approximately 8,108 × 35,286 km, 10.6°; separation about 3.5 h after launch, orbit period about 13 h. |
| Spacecraft raising | Approximately five apogee burns inferred for GOES-16; intermediate states and exact schedule are illustrative. |
| GEO and drift | Circularize near 35,786 km altitude and drift toward the target longitude. |

The 35,286 km injection apogee is about 500 km **below** GEO altitude, not
supersynchronous. GOES-U's exact Falcon Heavy separation elements are not
established in the reference data; do not reuse GOES-16 elements as its flight
trajectory.

## 3. Delta-v Budget

For the GOES-16-like injection, an ideal single-impulse apogee maneuver gives
a computed idealized estimate of about **0.98 km/s** for the spacecraft share.
Including multiple burns and drift gives a low-confidence planning estimate of
approximately **1.0–1.3 km/s**, not a published Atlas V flight budget. The
original reference research reports 566 m/s for GOES-U after Falcon Heavy
injection against a 987 m/s mission requirement; its injection elements remain
unknown. Propellant and maneuver assumptions should be checked against
mission-specific sources before use.

## 4. Key Parameters & Confidence Flags

- The GOES-16 reference uses **10.6°** injection inclination; a commonly cited
  9.52° value applies to the GOES-17 reference profile, not GOES-16.
- The approximate five-burn count is inferred from GOES-16's campaign, which
  included a thermal anomaly. Nominal burn count and intermediate states are
  not established here.
- The injection argument of perigee and exact GOES-U transfer elements are
  assumptions or unknowns. The model assumes an apogee near an equatorial node.
- The source research distinguishes the 15-year design life from longer
  propellant-life estimates. Neither is a validation of this model.

## 5. Modeling & Optimizing It in orbitopt

`GeoRaisingProblem` screens impulsive apogee-burn schedules. The number of
burns, perigee raises and inclination changes determine the campaign. The
scene is exported by `viz/geo_raising.py`; run the example with:

```bash
python examples/10_gto_geo_orbit_raising.py
orbitopt view goes
```

`verify/geo_insertion.py` re-flies a schedule with Earth J2/J22 and Sun/Moon
point-mass perturbations, refines the final burn, and reports short-term drift
and verification checks. Solar radiation pressure and finite-duration burns
are not propagated. The screening model's finite-burn feasibility estimates
are not a substitute for a finite-burn propagation.

## 6. Open Questions & Assumptions

Confirm the injection orientation, per-burn states and schedule against
mission-specific sources if available. GOES-U separation elements, the
Atlas V spacecraft Δv budget and a precise apogee-burn limit remain uncertain.
The optimizer's burn split is a modeled result, not measured flight telemetry.
