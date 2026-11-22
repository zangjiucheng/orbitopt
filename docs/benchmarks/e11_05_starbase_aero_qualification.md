# E11-05 — Starbase aero model qualification (task #312)

Epic #307, base commit `add2cae`. E5-08 recorded a 12.80 % Starbase stage-1
debris impact-speed difference against the 12 % independent-model band
(`docs/e5-08-new-vehicle-report.md`), and E10-04 confirmed it as a real drag
model difference: replaying the recorded separation state through the twin
debris integrator with the declared CA polar gives +12.7 %
(`tests/data/mission/reference/e5-08-new-vehicle-starbase-reports/
debris-model-difference.json`). This task qualifies the booster power-off
drag model against that discrepancy — high-fidelity-adjacent build-up plus
public-class data and the recorded debris flights themselves, including the
E11-01 grid-fin stowed/deployed states — and reports where the gap lives.
The machine reading is
[`e11_05_starbase_aero_qualification.json`](e11_05_starbase_aero_qualification.json);
reproduce with
`PYTHONPATH=src python benchmarks/bench_e11_05_starbase_aero_qualification.py --out …`
(≈5–10 min, CPU only) and check with
`PYTHONPATH=src pytest tests/test_booster_aero_e11_05.py`.

## The qualified model

`orbitopt.flightlab.booster_aero` builds the power-off Cd(Mach) from the
repo's own semi-empirical primitives (`nose_aero`: Taylor–Maccoll wave drag,
Rayleigh pitot Cp, PG-corrected turbulent Cf, DATCOM base term): forward
nose+shoulder, wetted skin friction, base drag, a stowed-protuberance
increment, and — engines-first — a full-Newtonian aft face over
`aft_face_fraction` of the body cross-section. Three qualification sources
anchor it:

* the E11-01 `grid_fins` block of the DC8 vehicle (deployed-state Cd,
  provenance `public_estimate`, derivation in
  [e11_01_grid_fin_authority.md](e11_01_grid_fin_authority.md));
* the two recorded debris flights themselves (Starbase and DC7): an
  in-flight power-off Cd inversion is the strongest "public data" this repo
  holds;
* the DC7 booster stage's declared ballistic coefficient
  (BC = 3600 kg/m², dry 275 t, d = 9 m ⇒ implied Cd ≈ 1.20), the F9-class
  published-figure pattern the vehicle table was built on.

DESIGN ASSUMPTIONS (echoed in the JSON): stowed fin t/c = 0.12 (±50 %,
linear into the stowed increment); protuberance interference K = 1.9
(1.5–2.3); aft-face opening fraction 0.85 (±10 %); flat carry-deck forward
base (errs high subsonic by ≤ ~0.05).

## Part A — component tables vs the declared polar

Engines-first stowed build-up: 0.87 @ M0 → 0.754 @ M1 → 1.059 @ M1.1 →
1.408 @ M2 → 1.558–1.61 @ M4–M12. Against the declared DC8 booster table
(0.8 / 0.9 / 1.3 / 1.25 / 1.2 / 1.15): within the declared ±30 %
`public_estimate` band at M0–M0.8 (×1.09, ×0.88), 18–40 % **above** it at
M1.1–M8 — the declared table reads low against a blunt aft-face upper bound,
consistent with real aft cavities being partly vented (hot bleed, open
engine bays). The DC7 BC anchor (Cd ≈ 1.2) sits inside the declared band and
below the full-Newtonian build-up, the same direction. Axial, stowed: the
protuberance increment adds 0.064 to the vehicle's own power-off table
(Starbase v1 has no fins or legs: its qualified axial polar **is** the
declared one, bit-for-bit — see `axial_stacked_stowed_starbase_v1`).

## Part B — recorded-flight inversion (the decisive evidence)

Both debris CSVs were inverted: a_drag from central-differenced recorded
ECEF velocity with the twin's own gravity (J2) + Coriolis + centrifugal
force package removed, projected on −v, divided by recorded q·S, rows with
q > 50 kPa (Mach 4.5–6 dense band). Starbase inverts to Cd_eff 0.334–0.348
against declared 0.355–0.377 (bin ratios 0.914 / 0.957 / 0.980, mean 0.960);
DC7 to 0.321–0.338 against 0.350–0.376 (mean 0.934). **The recorded flights
run 4–7 % BELOW the declared polar.** Every physically supported power-off
correction this qualification produces — protuberances (+0.064), deployed
fins (+1.3–2.3 at α = 0), blunt aft face (+) — moves the replay *further*
out of band. There is no defensible correction that closes the E10-04 gap.

## Part C — same-state replays (E10-04 protocol, unchanged)

| polar | twin impact | recorded | Δ % |
|---|---|---|---|
| declared, Starbase (control) | 1741.2 m/s | 1545.0 m/s | **+12.70** |
| qualified axial, Starbase v1 (= declared) | 1741.2 m/s | 1545.0 m/s | **+12.70** |
| declared + E11-01 fins deployed (bound) | 796.3 m/s | 1545.0 m/s | −48.5 |
| declared, DC7 (control) | 1744.1 m/s | 1587.4 m/s | +9.87 |

The control reproduces the committed E10-04 numbers to 0.03 pt (+12.70 vs
+12.68; DC7 exact) — same separation state (raw ECI columns at t_sep, rotated
with the verified `_ecef_to_eci` convention), same recorded mass, twin's own
integrator. The fin-deployed bound shows the Cd lever is large — but points
the wrong way (v1 debris flew with no fins, and the lever only adds drag).

## Part D — grid-fin stowed/deployed Cd states

Stowed increment +0.064 (two-side fairing friction + frontal blockage,
K = 1.9). Deployed-minus-stowed at zero incidence, referred to the body:
+1.59 @ M0, +2.29 @ M1.0 peaking, falling to +1.29 @ M12 (E11-01
`cd_add_mach` plateau; the α > 0 flying state adds the normal-force and
induced terms on top, E11-01 Part A). At the next vehicle revision these
become the two axial polars: `ca_stacked_stowed(declared, DC8_STOWED, S)`
and that plus `fin_state_delta_cd(gf)`.

## Part E — main-model / independent-model sync on the qualified table

One module generates the table; both models consume the same numbers.
`booster_aero.interp_cd` reproduces `BoosterModel.cd()` (main model, np.interp
clamped) bit-for-bit (`interp_agreement_abs = 0.0` over 9 probes), and the
twin's `_debris_to_impact` takes the same table directly. Coasting from the
sealed E10-07 separation handoff with the qualified DC8 engines-first table:
main 1059.0 m/s vs twin 988.2 m/s — 7.17 %, inside the same 12 %
model-agreement band E10-07 uses (its own re-flight delta is 8.7 %). The
sync claim is asserted in `tests/test_booster_aero_e11_05.py` to 0.1 pt.

## Verdict — gate 12 % unchanged, Starbase stays red

The qualified power-off model leaves the Starbase debris replay at **+12.70 %
> 12 %**, so the check stays recorded as a genuine aero model difference
(acceptance branch 2). The new evidence localises it: the recorded flights
invert *below* the declared polar, so the residual is not a booster Cd
error — it is the aero/atmosphere line between OpenTsiolkovsky and the twin
in dense high-q air, exactly where E10-04 put it. The DC7 case stays inside
the band (+9.87 %). No sealed artifact was touched: the vehicle YAMLs are
digest-sealed, and the qualified tables land here as the module-generated
successor to be applied, one command, at the next vehicle revision
(`booster_aero.cd_engines_first_table()` / `ca_stacked_stowed()` /
`fin_state_delta_cd()`, provenance tag `qualified (E11-05)`).
