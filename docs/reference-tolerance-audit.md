# Reference tolerance audit (E8-04)

Every regression that compares against committed reference data must be
reproducible **bit-fairly**: the same inputs, the same locked package stack
and the same Earth orientation. When a band changes, this page and the
machine-readable audit inside the reference's `PROVENANCE.json` record
*why*, from what old value to what new value. CI runs
`scripts/fl601_reference.py tag-check` so a diff under the reference
directory without an `[fl601-reference-update]` commit fails, and
`tests/test_e804_reference_data.py` fails if this page's tolerance blocks
drift from the enforced `tolerances.json`.

## The unified EOP pinning mechanism

Astropy's offline IERS default changed between major versions (astropy 6
flies IERS-B offline; astropy 8 prefers IERS-A when downloads are on), and
that alone moves the frame bridge by metres. So no reference regression
may rely on whatever the installed astropy happens to default to. Gate B
and E2-04 freeze Earth orientation through `orbitopt.reference_eop`
(`pin_no_download` and `freeze_reference`), with two modes chosen by how the
reference was recorded:

- **`pinned-bundled`** (Gate B, `tests/data/flightlab/reference/`): the
  tracks were flown on the *default offline table of the recording stack*
  (astropy 6.1.7 + `astropy-iers-data 0.2026.8.31.0.57.9`, an IERS-B
  table). The snapshot records the recording-stack version, the active
  table, the effective UT1-UTC/polar motion at the epoch and the exact
  bracketing rows; verification re-derives all of it from the installed
  bundled table and refuses (`ToolchainMismatch`) if anything differs.
  No download ever happens (`IERS_AUTO_DOWNLOAD = False`).
- **`override`** (E2-04, `tests/data/mission/reference/e2-04-frozen-mission/`):
  the reference was flown against specific IERS-A rows, so the snapshot's
  four rows *become* the active table at regression time, on any stack.

E5-05 currently uses the scoped snapshot context in `scripts/e505_reference.py`:
its own recording-package IERS-A rows override the host table. This is not
the Gate B bundled-stack policy, and a stored snapshot does not imply every
row was a finalized measurement.

`scripts/fl601_reference.py update` re-flies the Gate B reference and
records fresh digests. For a comparison-policy-only change, use
`update --tolerances-only --reason "..."`: it preserves flight output,
requires the reason, and appends the old→new tolerance/hash audit to
`PROVENANCE.json`. `verify` checks all 13 files against the recorded digests.

## RAAN / Earth-orientation benchmarks

`tests/test_e804_raan_benchmark.py` grades the frame bridge against
references independent of the recorded tracks: the analytic sidereal
rotation angle for a UT1 shift (omega_E = 360.9856235279 deg/day), the
polar-motion vector norm as the tilt the matrix must show, the
pure-z-rotation signature of UT1 (dRAAN = sidereal angle, dInclination
still), and an astropy-free IAU-76 precession + ERA pipeline whose
external RAAN must band within 0.60 deg of the recorded track (the band
absorbs only the IAU-76-vs-IAU-2000 CIP/CIO model gap; structural frame
errors move RAAN by degrees). Bands were measured 2026-11-15 on the
recording stack with >100x margin against the analytic values.

## Cross-platform consistency

`tests/test_e804_cross_platform.py` re-derives the committed GCRS tracks
from the raw recorded OpenTsiolkovsky outputs on the running platform and
requires the difference inside `cross_platform` below. CI runs this on the
identical locked stack on linux-x86_64 (`test` job), macOS ARM
(`test-macos`, osx-arm64) and macOS Intel (`test-macos`, osx-64);
observations per platform are ledgered in
`tests/data/e804/cross-platform.json` (`cross-platform-record` prints the
line for a platform). The committed ledger currently contains a Linux
observation; both macOS entries are `null` (not recorded). A configured CI
matrix is not a stored cross-platform result.

## History

| date | change | reason |
|---|---|---|
| 2026-10-28 `286ebe2` | Gate B position/altitude/downrange 1e-6 -> 1e-4 m, velocity 1e-8 -> 1e-6 m/s | cross-machine floor: the reference OT binary itself differs at this level between build hosts; the band only needs to prove no regression beyond the recording platform's own reproducibility |
| 2026-10-28 `e97e147` | Gate B `t_s_abs` 1e-9 -> 1e-6 s | same cross-machine floor for the logged time column |
| 2026-11-15 `3051848` | E2-04 add `ascent_build_{position,velocity}_m_abs` 1e-6 and `landing_plan_diagnostic_abs` {fuel_planned_kg 1e-3, tof_flying_s 1e-2} | physics verdicts preserved across platforms; diagnostics rebuilt per-run need their own bands |
| 2026-11-15 (this task) | Gate B add `cross_platform` {position_m_abs 1e-4, velocity_m_s_abs 1e-6} | E8-04 cross-platform re-derivation exercises the same pinned frame bridge as the flight bands, so the band equals them; full audit entry in `PROVENANCE.json` |
| 2026-11-17 (this task) | E8-01 full-chain bands added (`FULL_CHAIN_TOLERANCES`, table below) | E8-01 grades the recorded E5-05 chain against the raw OpenTsiolkovsky recording and closed-form mechanics; bands derive from recorder grid, element-convention and environment-model budgets, not post-hoc tuning |

## Current enforced tolerances

Each block below must equal the referenced `tolerances.json` byte-for-byte
in parsed form (enforced by `tests/test_e804_reference_data.py`).

```json
<!-- tolerances: gateb-leo-two-stage (tests/data/flightlab/reference/tolerances.json) -->
{
  "t_s_abs": 1e-06,
  "mass_kg_abs": 1e-06,
  "position_m_abs": 0.0001,
  "velocity_m_s_abs": 1e-06,
  "extra_abs": {
    "altitude_m": 0.0001,
    "latitude_deg": 1e-09,
    "longitude_deg": 1e-09,
    "downrange_m": 0.0001,
    "dynamic_pressure_pa": 0.01,
    "mach": 1e-07
  },
  "extra_abs_default": 1e-06,
  "check_value_abs": 1e-09,
  "check_value_rel": 1e-09,
  "event_data_abs": 1e-09,
  "event_data_rel": 1e-09,
  "cross_platform": {
    "position_m_abs": 0.0001,
    "velocity_m_s_abs": 1e-06
  }
}
```

```json
<!-- tolerances: e2-04-frozen-mission (tests/data/mission/reference/e2-04-frozen-mission/tolerances.json) -->
{
  "ascent_build_position_m_abs": 1e-06,
  "ascent_build_velocity_m_s_abs": 1e-06,
  "t_s_abs": 1e-06,
  "mass_kg_abs": 1e-06,
  "position_m_abs": 0.01,
  "velocity_m_s_abs": 0.0001,
  "extra_abs": {
    "altitude_m": 0.01,
    "latitude_deg": 1e-08,
    "longitude_deg": 1e-08,
    "downrange_m": 0.01,
    "dynamic_pressure_pa": 0.1,
    "mach": 1e-06,
    "accel_g": 1e-06,
    "q_bar_pa": 0.1,
    "density_kg_m3": 1e-09
  },
  "extra_abs_default": 0.001,
  "check_value_abs": 0.0001,
  "check_value_rel": 1e-09,
  "event_data_abs": 1e-06,
  "event_data_rel": 1e-09,
  "landing_plan_diagnostic_abs": {
    "fuel_planned_kg": 0.001,
    "tof_flying_s": 0.01
  },
  "report_value_abs": 0.0001,
  "report_value_rel": 1e-09
}
```

## Full-chain reference bands (E8-01, task #281)

`src/orbitopt/verify/full_chain.py` grades the recorded E5-05 chain against
its committed references; the bands below are the single source of truth
(`FULL_CHAIN_TOLERANCES`, embedded verbatim in every report) and are NOT
in a `tolerances.json` file — they are graded against committed files, not
a replay, so the replay-tolerance mechanism of E8-04 does not apply. None
of the bands was tuned after seeing the residuals; each derives from the
recorder's own discretisation, a model-class difference budget, or the
physics of the comparison, and the value observed at recording is listed.

| check | unit | band | observed | basis (summary) |
|---|---|---|---|---|
| `ascent.event.meco` / `.fairing_jettison` | s | 2.0 | 0.0 / 0.0 | one OpenTsiolkovsky output-grid interval; events re-extracted from raw thrust/mass columns |
| `ascent.event.separation` / `.seco` | s | 2.0 | 1.0 / 1.83 | same; SECO sits on an appended event sample, so the nearest grid sample is one step away |
| `ascent.separation_window_position` | m | 5000 | 4.7 | gfold vs raw-OT debris over the shared unpowered window: environment-model budget |
| `ascent.separation_window_velocity` | m/s | 5.0 | 0.002 | same window, velocity |
| `insertion.semi_major_axis` | m | 250 | −67 | osculating (external) vs J2-mean (chain) elements at 300 km / ~12 min |
| `insertion.eccentricity` | − | 5e-6 | 1.2e-6 | same element-convention difference |
| `insertion.inclination` / `.raan` | deg | 0.05 / 0.05 | −0.0002 / −7.7e-5 | mean-vs-osculating drift allowance, ~100× margin |
| `ascent.seco_state_position` / `_velocity` | m / m/s | 1 / 0.01 | 5.8e-11 / 9.4e-13 | sealed-record round trip through the pinned frame bridge; float64 text serialisation |
| `analytic.seco_orbit_energy` (Δa) | m | 1.0 | 0.005 | vis-viva identity on one state; double-rounding scale |
| `analytic.seco_orbit_eccentricity` | − | 1e-8 | −6.6e-10 | same |
| `analytic.seco_orbit_period` | s | 0.5 | 8e-6 | same |
| `analytic.seco_orbit_inclination` | deg | 0.05 | 0.018 | covers the external element-convention spread seen at recording |
| `ascent.stage2_dv_ledger` | − | 0.15 | 0.122 | Tsiolkovsky ideal-Δv bound; gravity+drag loss budget for a lofted vacuum ascent |
| `recovery.coast_arc_position` | m | 1000 | 575 | exact two-body vs J2+gravity-gradient propagation over the ~207 s coast |
| `recovery.coast_arc_velocity` | m/s | 10.0 | 3.77 | same arc, velocity |
| `recovery.event.landing_burn_ignition` | s | 2.0 | 0.5 | recorder grid again: first powered booster sample after the event |
| `recovery.event.touchdown` | s | 1.0 | 0.0 | sealed records must agree exactly; 1 s of float round-trip headroom |
| `recovery.touchdown_gates` | − | 1.0 | 0.70 | design-input gates (target.yaml limits), composite utilisation ≤ 1 |
