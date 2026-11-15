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
may rely on whatever the installed astropy happens to default to. All of
them freeze Earth orientation through `orbitopt.reference_eop` (one
snapshot format, one `pin_no_download`, one `freeze_reference`), with two
modes chosen by *how the reference was recorded*:

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

Regenerating or re-recording a reference uses the same entry point:
`scripts/fl601_reference.py update [--tolerances-only] --reason "..."`
writes the old→new diff into `PROVENANCE.json` (`tolerance_audit`)
alongside the new `tolerances.json` digest; a change without `--reason`
is refused. `verify` then checks all 13 reference files against the
recorded digests.

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
line for a platform).

## History

| date | change | reason |
|---|---|---|
| 2026-10-28 `286ebe2` | Gate B position/altitude/downrange 1e-6 -> 1e-4 m, velocity 1e-8 -> 1e-6 m/s | cross-machine floor: the reference OT binary itself differs at this level between build hosts; the band only needs to prove no regression beyond the recording platform's own reproducibility |
| 2026-10-28 `e97e147` | Gate B `t_s_abs` 1e-9 -> 1e-6 s | same cross-machine floor for the logged time column |
| 2026-11-12 `3051848` | E2-04 add `ascent_build_{position,velocity}_m_abs` 1e-6 and `landing_plan_diagnostic_abs` {fuel_planned_kg 1e-3, tof_flying_s 1e-2} | physics verdicts preserved across platforms; diagnostics rebuilt per-run need their own bands |
| 2026-11-15 (this task) | Gate B add `cross_platform` {position_m_abs 1e-4, velocity_m_s_abs 1e-6} | E8-04 cross-platform re-derivation exercises the same pinned frame bridge as the flight bands, so the band equals them; full audit entry in `PROVENANCE.json` |

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
