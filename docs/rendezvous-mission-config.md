# Rendezvous mission configuration

A rendezvous mission is one YAML file at schema version `1.0`. The
reference is `src/orbitopt/data/leo_rdv_001.yaml`;
its values are task acceptance numbers, fixed before any optimisation
runs — not tunables discovered at runtime.

```bash
orbitopt rendezvous studio --config PATH   # any file this schema
```

## Mission and dynamics

| Key | Meaning |
|---|---|
| `mission.id` / `title` / `kind` | Identity; `kind: rendezvous-two-impulse`. |
| `dynamics.frame` | Inertial frame of the states — `GCRS`. |
| `dynamics.mu` | Earth μ. The loader demands the exact packaged value 3.9860044150×10¹⁴ m³/s²; the Lambert solver's solar-system default can never leak in. |
| `dynamics.j2` | Plan and verify against the same model — J2 on. |

## Orbits and geometry

| Key | Meaning |
|---|---|
| `target.*` | Chief: altitude, inclination, RAAN, argument of latitude. |
| `chaser.altitude_m` | Deputy altitude; below the target, so drift carries it forward along V-bar. |
| `chaser.initial_separation_m` | Separation at the shared epoch t0 (coplanar in v1). |
| `hold_point.rel_lvlh` | LVLH offset of the hold point; arrival means hovering here with zero relative velocity. |

## Budget, tolerances, windows

| Key | Meaning |
|---|---|
| `budget.delta_v_budget_m_s` | Total Δv cap, fixed before optimisation. |
| `tolerances.max_position_error_m` / `max_relative_speed_m_s` | Landing gate at the hold point (50 m / 0.2 m/s for LEO-RDV-001). |
| `windows.drift_max_s` | The departure gate must be crossed within one drift phase. |
| `windows.transfer_tof_min_s` / `max_s` | Time-of-flight window, fixed before optimisation. |
| `phasing.departure_gate_y_m` | V-bar gate where the transfer departs. |
| `optimizer.*` | Newton targeting tolerance/iteration caps for the J2 correction. |

## Safety block (optional, additive)

A config may carry a `safety` block
([ADR rendezvous-0002](adr/rendezvous-0002-safety-constraints-whole-trajectory.md));
its presence changes the canonical planner from `plan_rendezvous` to
`plan_rendezvous_safe`, which targets within the constraints:

| Key | Meaning |
|---|---|
| `keep_out_radius_m` | Keep-out sphere around the target, checked at every sample. |
| `cross_track_max_m` | Corridor bound on cross-track excursion. |
| `cone_half_angle_deg` / `cone_range_m` | Terminal cone forbidding forward excursion near the hold point. |
| `approach_speed.range_knots_m` / `cap_m_s` | Range-dependent relative-speed ceiling, piecewise-linear. |
| `passive_contingency_s` | No-thrust coast from t0, gate and hold point must stay clear. |
| `check_sample_dt_s` | Dense check grid; burn epochs are always sampled. |

## Robustness campaigns

`src/orbitopt/data/leo_rdv_001_robustness.yaml` is a separate campaign
schema (stress arms, clustering, hold-out split) consumed by
`orbitopt.rendezvous.robustness`; it declares the dispersion study behind
the robust numbers in the
[verification report](rendezvous-verification-report.md), and is not a
mission config.
