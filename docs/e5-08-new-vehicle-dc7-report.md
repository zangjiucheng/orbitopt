# E5-08 new-vehicle design & qualification report

Recorded commit `c847a46f82d0d9aa10e54e1f9ad76bdbb888c99b`; toolchain python 3.12.13, numpy 2.5.4, scipy 1.18.1, gfold 0.3.2.

The vehicle: `starship_class_rtls_derived` -- the E5-07 reference stack with the DC1 powered-entry RTLS recovery scheme (three ignitions: boostback, one throttled entry burn, landing burn). Constraints inherited from the E5-07 target UNCHANGED (max-q 55 kPa, 3.0 g, 230 km floor, 250 km insertion band +/- 60 km, RTLS touchdown 3/1 m/s, reserve >= 100 t). Design changes and their reasons are declared in the vehicle file; every non-design number carries its source there.

## Comparison against the E5-07 reference

```json
{
  "ascent": {
    "max_q_pa": {
      "reference": 37479.27754272647,
      "derived": 37479.27754270955,
      "delta": "-1.692e-08 Pa"
    },
    "peak_g": {
      "reference": 2.8875404152803634,
      "derived": 2.9758741727162406,
      "delta": "+0.08833 g"
    },
    "periapsis_m": {
      "reference": 232497.0131449364,
      "derived": 232026.43047139887,
      "delta": "-470.6 m"
    },
    "apoapsis_m": {
      "reference": 292126.0433095135,
      "derived": 234780.31626438256,
      "delta": "-5.735e+04 m"
    },
    "inclination_deg": {
      "reference": 51.98429660817715,
      "derived": 51.97680703539541,
      "delta": "-0.00749 deg"
    },
    "note": "DC7 (stage-1 cutoff 12 s earlier, ship burn longer, post-MECO pitch program re-tuned): every inherited limit still holds -- 3.0 g, max-q 55 kPa, periapsis >= 230 km, apoapsis 250 +/- 60 km, inclination 51.6 +/- 1 deg; the numbers above are measured against the E5-07 reference, not 'unchanged by construction'"
  },
  "payload_capability": {
    "delivered_orbit": "inside the inherited insertion band (see the ascent rows): the same flight target, with every constraint unchanged",
    "ship_propellant_margin": {
      "reference_kept_kg": 118801.3500445286,
      "derived_kept_kg": 101213.52171249603,
      "delta_kg": -17587.82833203257,
      "statement": "DC7 spends this much of the ship's spare propellant (the ship burns longer to make up the earlier booster cutoff). It is the price of the booster's reserve; nothing absorbs it silently."
    }
  },
  "propellant_margin": {
    "booster_sep_propellant_kg": null,
    "touchdown_propellant_kg": 113115.84898297,
    "graded_reserve_kg": 100000,
    "attribution": "DC1 (powered-entry RTLS) plus DC7 (earlier stage-1 cutoff): the booster separates heavier and slower, the graded reserve (>= 100 t, inherited unchanged) is met by the recorded touchdown figure, flown not predicted"
  },
  "recovery": {
    "reference": "NOT FLOWN (RECOVERY_GAP.md: restart-2 aerodynamic entry has no planner representation)",
    "derived": "FLOWN (DC1): touch-down numbers below",
    "touchdown": {
      "landed": true,
      "reason_codes": [],
      "position_error_m": 0.33908419777976007,
      "vertical_speed_m_s": -1.50003341739229,
      "lateral_speed_m_s": 0.03155062180058664,
      "max_proper_g": 4.591903271073983,
      "propellant_remaining_kg": 113115.84898297
    }
  },
  "model_uncertainty": "aero tables +-30 % (nose_aero band, inherited): recorded max-q 37.5 kPa -> 48.75 kPa at +30 %, still inside the 55 kPa ceiling; engine/Isp bands enter the recovery leg as the FL-304 MC dispersion (see mc-summary.json)"
}
```

## E8-05 independent-dynamics validation (new vehicle)

Verdict **passed** (27/27 independent; 28/28 total), independent native point-mass ascent twin + two-model powered-landing refly on the DECLARED design corridor in the 3-engine landing configuration (the vehicle file records the 13-engine set cannot throttle below weight; inherited limitation, DC1 unchanged). The mission's ACTUAL corridor state (recorded engines=13, landing burn aborted by the planner) is independently UNPLANNABLE for a soft landing from here -- the two models agree with the E3 planner, corroborating the landing gap below.

## E7-05-methods recovery Monte Carlo

Success rate **0.083** (3/36), whole-seed cluster bootstrap 95% CI [0.0, 0.16666666666666666]. Seeds [8, 9, 10] are hold-out (disjoint from the search seed 287); per-seed rates: seed 8: 0.00, seed 9: 0.08, seed 10: 0.17

## Reason classification of the unclosed fraction

Failure classes per seed are in `tests/data/mission/reference/e5-08-new-vehicle-reports/mc-summary.json`; classified as vehicle capability / guidance / constraint in the per-sample reason codes.
