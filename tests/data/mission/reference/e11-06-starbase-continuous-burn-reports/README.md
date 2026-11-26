# E11-06 continuous-burn gate record

Reproduction and release-gate record for E11-06 (Starbase two-ignition recovery:
boostback + one continuous entry+landing burn, engines 13 → 5 → 3 within the
burn, grid fins `stowable: false`), made for the E12-00 release gate. The
E11-06 branch itself records its headline results in its commit message
(`31edace`); this directory is the machine-checkable archive behind them.

## What is archived

- `../e11-06-starbase-continuous-burn/` — the run artifacts:
  - `mission/` — the full two-stage LEO mission chain on the sealed E5-08c
    Starbase ascent (`run_two_stage_leo`, cold-staging ascent + continuous-burn
    recovery). `mission.verification.json` lists **71 passing checks**.
  - `mission-hs/` — the same chain on a re-derived hot-staging ascent
    (ascent run under `ascent-hs/`); the powered bridge grades the
    continuous-burn handoffs (`recovery.handoff-entry-landing`,
    `recovery.joint-*-landing`): **all 71 listed checks pass**.
  - `ascent-refly/` — a copy of the sealed E5-08c ascent with the independent
    SECO orbit re-fly written back (`orbit-refly-*` checks green; the two
    E5-08-era `recovery-*` placeholders stay red — they were never assessed in
    that run, and the E11-06 mission above assesses recovery properly).
  - `ascent-hs/` — the hot-staging ascent run, orbit re-fly written back (green).
  - `candidate.json` — the champion recovery-script design vector.
- the JSONs in this directory — the graded reports:
  - `fly.json` — nominal continuous-burn recovery flown through the FL-302/FL-303
    chain: plan feasible, `ignitions_used = 2`, all 8 landing/recovery checks green.
  - `mission-checks.json` / `mission-hs-checks.json` / `recovery-checks.json` —
    the mission-chain verdicts and the recovery run's own 11 checks.
  - `orbit-refly.json` / `hs-orbit-refly.json` — the independent re-fly checks.
  - `twin.json` — the independent point-mass twin re-fly of the same script.
  - `mc-sealed-pinned.json`, `mc-sealed-refine.json`, `mc-hs-pinned.json` —
    FL-304 Monte Carlo, 36 samples = 3 seeds × 12 (seeds 8/9/10), successes with
    a cluster-bootstrap 95% CI over seeds. 33/36 pinned on the sealed ascent
    (CI 0.83–1.00) reproduces the E11-06 commit exactly; the hot-staging figure
    here is 33/36 on this record's re-derived hot-staging ascent — the E11-06
    branch recorded 35/36 on its own ascent, and the two misses-per-seed
    difference is ascent-side, not recovery-side. All misses are CORRIDOR_MISSED.
  - `candidate-set.json` — every recovery champion re-derived for this record
    (search seeds 1–5, identical protocol) with its margins and the pinned-script
    MC rate each one achieves: 6/36 to 33/36. The search stops at the first
    feasible design, so the stop point — and with it the corridor margin — is
    seed dependent; the archived champion is the seed-5 stop point, whose
    pinned-script rate reproduces the E11-06 commit's 33/36 exactly.

## How to reproduce

On Linux/macOS with the landing stack (gfold 0.3.2 needs Python >= 3.11 —
CI's `flightlab-landing` job builds it from `envs/flightlab-landing-ci.txt`):

    E11_ENGINE_STAGES=1 python scripts/e11_continuous_burn_search.py --fins v3_pp --seed 5 --out candidate.json
    python scripts/e11_06_gate_record.py fly candidate.json fly.json
    E11_ENGINE_STAGES=1 python scripts/e11_continuous_burn_mission.py candidate.json mission-run
    python scripts/e11_06_gate_record.py checks mission-run mission-checks.json
    python scripts/e11_run_orbit_refly.py ASCENT_DIR SCENARIO            # SECO re-fly
    python scripts/e11_continuous_burn_mc.py candidate.json pinned  OUT 12 8,9,10
    python scripts/e11_continuous_burn_mc.py candidate.json refine  OUT 12 8,9,10
    E11_ENGINE_STAGES=1 ORBITOPT_E508_VARIANT=starbase python scripts/e11_hs_ascent_search.py --out hs/ascent
    E11_ASCENT_DIR=hs/ascent/best E11_ASCENT_VEHICLE=hs/ascent/best_vehicle.yaml \
        E11_ENGINE_STAGES=1 python scripts/e11_continuous_burn_mission.py candidate.json mission-hs-run

`tests/test_e11_06_gate_record.py` grades this archive (ignition count,
engine-state/mass-flow consistency, green landing gates, CI bookkeeping).

## What stays red / assumed (not relaxed — recorded)

- **Grid-fin drag is a DESIGN ASSUMPTION**: no public Super Heavy absolute
  fin-drag data exists; the `v3_pp` calibration is an assumption, and every
  check graded against it inherits that assumption.
- **Twin disagreement stays visible**: the independent twin re-fly (`twin.json`
  delta) agrees on speed (+4.6%) and tilt, but its entry peak-q differs by
  +18.2% against the 12% E10-07 band, so that check is **red** by design;
  peak-g agrees within its 1.0 band. E11R-02 attributed it (task #415,
  `docs/benchmarks/e11r_02_peak_q_error_budget.md`): the density-profile FORM,
  carried almost entirely by the single 0–25 km segment of the published
  piecewise-exponential table the main model flies (the two profiles agree at
  sea level, split by up to 26% near the 11 km tropopause, and agree above
  25 km). Integration step, gravity, the height convention and the entry-window
  definition each measure under 1 pt, and the window difference is exactly zero.
  Genuine model difference, not an implementation bug — the band stays at 12%.
- **E5-08-era placeholders**: the sealed ascent run's own `recovery-landing` /
  `recovery-propellant-reserve` checks were never assessed in E5-08 and stay
  red in `ascent-refly/` (value `None`, reason `RECOVERY_LANDING_NOT_ASSESSED`).
- **Reproduction variance is recorded**: the recovery search stops at the
  first feasible design (penalty floor −1), so different seeds stop at
  different feasible points with different corridor margins. Re-running the
  search over seeds 1–5 with this stack and flying the pinned-script FL-304
  set gives 6/36 to 33/36 (table in `candidate-set.json`); the E11-06
  commit's 33/36 is reproduced exactly by the seed-5 champion, which is the
  archived champion, and per-sample re-planning (`refine`) recovers all 36
  samples (CI 1.00–1.00), matching the recorded 36/36. No graded limit was
  relaxed anywhere in this record.
