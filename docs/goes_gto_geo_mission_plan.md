> **Provenance:** researched and adversarially cross-verified by a 13-agent
> workflow (5 parallel web-research tracks -> per-track skeptic verification +
> an orbitopt codebase-mapping agent -> synthesis -> physical-plausibility
> critic). Numbers carry inline confidence flags. **Reviewer corrections folded
> in:** injection-orbit period 13.0 h (a = 28,075 km); parking-orbit period
> 91.6 min. The single-impulse ~0.98 km/s spacecraft-share is a *floor* (the
> injection apogee sits ~500 km below GEO); with the 13.0 h period, ~5 LAE
> burns over ~8 days necessarily skip apogees between burns.
>
> **Implemented in this repo** (screening/optimization + visualization half of
> the two-stage pattern): `problems/geo_raising.py` (`GeoRaisingProblem`),
> `examples/10_gto_geo_orbit_raising.py`, `viz/geo_raising.py` (Mission Control
> "GOES — GTO to GEO" scene), `tests/test_geo_raising.py`. The tudatpy
> verification stage (`target_geo_insertion`, perturbed `propagate_multi_arc`)
> is the remaining next increment (see Section 5.6).

# GOES Launch-to-GEO Mission Plan Design

## GOES-R Series (GOES-16/-17/-18/-19) — Geostationary Weather Satellite Delivery

*Reference design for a ~5.2 t NOAA/NASA meteorological satellite from launch to an operational geostationary slot. Figures are drawn from the researched-and-verified source set; every value that is unofficial, computed, weakly sourced, or in dispute is flagged inline and consolidated in Section 4.*

---

## 1. Mission Overview

### 1.1 Objective

Deliver a Lockheed Martin A2100A-based GOES-R series weather satellite (launch mass ≈ 5,192 kg for GOES-R/S/T; ≈ 5,000 kg for GOES-U) from a Cape Canaveral / Kennedy launch pad into a geosynchronous slot on the equator (radius 42,164.17 km, altitude 35,786 km, orbital period one sidereal day = 86,164 s, circular velocity 3.0747 km/s), circular and equatorial (a = a_GEO, e ≈ 0, i ≈ 0°), then station it at an assigned longitude (GOES-East ≈ 75.2° W; GOES-West ≈ 137° W) and complete on-orbit checkout of the instrument suite.

### 1.2 Satellites and Launch Vehicles

| Satellite (pre-launch → on-orbit) | Launch vehicle | Pad | Launch (UTC) | GEO reached |
|---|---|---|---|---|
| GOES-R → GOES-16 | ULA Atlas V 541 (AV-069) | SLC-41, CCAFS | 2016-11-19, 23:42 | ~10 days |
| GOES-S → GOES-17 | ULA Atlas V 541 | SLC-41, CCAFS | 2018-03-01, 22:02 | 2018-03-12 (~11 d) |
| GOES-T → GOES-18 | ULA Atlas V 541 | SLC-41, CCSFS | 2022-03-01, 21:38 | 2022-03-14 (~13 d)* |
| GOES-U → GOES-19 | SpaceX Falcon Heavy | LC-39A, KSC | 2024-06-25, 21:26 | 2024-07-07 (~12 d) |

\*GOES-18 "renamed on 2022-03-14 after reaching GEO"; the exact final-burn date vs. rename date is not separately pinned in the source set, so ~13 days is approximate.

**Atlas V 541** (first three flights): 5-meter payload fairing (the "5"), four AJ-60A solid rocket boosters (the "4"), and a single-engine (the "1") Centaur upper stage powered by one RL10C-1 (LH2/LOX, ~101.8 kN / ~22,890 lbf vacuum). The SRBs (17.0 m / 55.7 ft each, ~94 s nominal burn, jettisoned ~110 s after liftoff in pairs ~1.5 s apart) augment the Common Core Booster during first-stage flight.

**Falcon Heavy** (GOES-U only): three Falcon 9 cores, 27 Merlin engines, >5 million lbf liftoff thrust. The two side boosters (B1072, B1086) performed return-to-launch-site landings at LZ-1 and LZ-2 (~T+8 min); the center core (B1087) was **expended** (jettisoned ~T+4 min 04 s), trading recovery for the extra injection energy that made this the highest-energy GOES delivery. NASA selected SpaceX (Sept 2021, ~$152.5M) because no Atlas V slots remained.

### 1.3 Two-Stage Division of Labor

The mission is deliberately split so that the launch vehicle does the expensive low-altitude work and the spacecraft does only the efficient high-altitude finishing:

- **Stage 1 — Launch vehicle → transfer orbit.** The upper stage does *not* drop the payload in a standard low-perigee GTO. Instead it delivers an elevated-perigee, reduced-inclination transfer orbit that pre-completes much of the plane change, minimizing the propellant the spacecraft must carry.
  - *Atlas V / Centaur:* a **three-burn** profile — parking-orbit insertion, transfer-orbit injection, then a third burn at apogee that simultaneously raises perigee and cuts inclination — yielding ≈ 8,108 × 35,286 km at 10.6° (GOES-16). Separation ~3.5 h after launch.
  - *Falcon Heavy:* **three** second-stage (MVac) burns over ~4.5 h to a high-energy transfer orbit; separation ~T+4.5 h.
- **Stage 2 — Spacecraft → GEO.** The A2100's bipropellant Liquid Apogee Engine (LAE) executes a series of apogee-centered burns (≈ 5 for GOES-16) that raise perigee/circularize while removing the residual inclination, reaching preliminary GEO in ~10–12 days (nominal plan ~2 weeks), followed by drift to slot and checkout.

> **Correction to a common premise:** The Atlas V GOES injection is **not** supersynchronous. Its apogee altitude 35,286 km corresponds to a radius of 41,664 km, which is ~500 km **below** the geostationary radius of 42,164 km. The energy advantage comes from the high perigee (~8,100 km vs. ~185–650 km for a standard GTO) and the low inclination (10.6° vs. ~27° for a standard Cape GTO), **not** from an apogee above GEO. The GOES-U Falcon Heavy injection is often described in press as "supersynchronous"/"almost to GEO," but SpaceX/NOAA published no numeric separation elements, so its apogee/perigee/inclination are not confidently known (see Section 4).

---

## 2. Mission Phases

Representative profile below is the well-documented **Atlas V 541 / GOES-16** case. Launch-vehicle burn elements are sourced; spacecraft-phase intermediate orbital states are *illustrative* (the real per-burn intermediate elements were not published) and are marked **[illus.]**. Elapsed times for the LAE campaign are approximate.

| Phase | Event | Perigee alt (km) | Apogee alt (km) | Incl (°) | Period | Δv | Elapsed time |
|---|---|---|---|---|---|---|---|
| Liftoff & ascent | CCB + 4 SRBs ignite; SRB burnout ~94 s, jettison ~110 s in pairs; BECO; Centaur staging | — | — | ~28.5 (launch az.) | — | LV | T0 → ~T+4.5 min |
| Parking-orbit insertion | Centaur **burn 1** (7 min 38 s) | ~167 | ~541 | 28.15 | ~91.6 min | LV | ~T+5 → ~T+13 min |
| Coast (phasing) | Ballistic coast to first descending/ascending node | ~167 | ~541 | 28.15 | ~91.6 min | 0 | ~T+13 min → ~T+45 min |
| Transfer-orbit injection | Centaur **burn 2** (5 min 36 s) | ~187 | ~32,716 | 25.68 | ~9.6 h | LV | ~T+45 → ~T+51 min |
| Long coast to apogee | Ballistic coast to apogee near equatorial node | ~187 | ~32,716 | 25.68 | — | 0 | ~51 min → ~3 h 28 min |
| Apogee raise + plane cut (SECO-equiv.) | Centaur **burn 3** (1 min 33 s); spacecraft separation | ~8,108 | ~35,286 | **10.6** | ~13.0 h | LV | ~T+3 h 28 min; sep ~T+3.5 h |
| Initial coast / commissioning | Spacecraft acquisition, sun-pointing, LAE priming | ~8,108 | ~35,286 | 10.6 | ~13.0 h | 0 | sep → first apogee |
| Apogee-raising burn 1 (LAE) | LAE at apogee: raise perigee, remove part of Δi | ~13,000 **[illus.]** | ~35,400 **[illus.]** | ~8 **[illus.]** | — | ≈ 150–250 m/s **[illus.]** | days 0–2 |
| Apogee-raising burns 2–4 (LAE) | Successive apogee burns, progressive rounding; GOES-16 limited each to <41 min after a nozzle-truss thermal anomaly | ~20,000 → ~32,000 **[illus.]** | ~35,600 → 35,786 **[illus.]** | ~6 → ~2 **[illus.]** | — | ≈ 150–250 m/s each **[illus.]** | days 2–8 |
| GEO circularization | Final LAE burn nulls residual to circular-equatorial | 35,786 | 35,786 | ~0 | 86,164 s | (folds into total) | ~day 8–10 |
| Drift-to-slot | Small a-offset drift orbit → phase to target longitude → drift-stop Δv | ~35,786 ± drift | ~35,786 ± drift | ~0 | ≈ 1 sidereal day ± | small (tens of m/s) | ~days 8–14 |
| On-orbit test & checkout | Instrument outgassing/activation (ABI, GLM, SUVI, EXIS, SEISS, MAG), PLT | 35,786 | 35,786 | ~0 | 86,164 s | station-keeping only | ~weeks–months post-GEO |

**Falcon Heavy / GOES-U variant of the first six rows:** three MVac burns over ~4.5 h to a higher-energy transfer orbit; separation ~T+4.5 h; spacecraft then needed only **~566 m/s** to reach GEO (vs. the mission-maximum requirement of ~987 m/s). Exact injection elements unpublished.

---

## 3. Delta-v Budget

### 3.1 Launch-vehicle-provided (to transfer orbit)

The Centaur/MVac Δv is not the useful decomposition here — the meaningful figure is *what the launch vehicle leaves for the spacecraft*. The GOES-U mission requirement is the one hard published anchor:

- **Spacecraft Δv required to reach GEO, GOES-U mission maximum:** **987 m/s** (published requirement).
- **Spacecraft Δv actually needed after Falcon Heavy injection, GOES-U:** **566 m/s** (the ~421 m/s margin extended expected propellant life well beyond the 15-yr spec).

### 3.2 Spacecraft-provided (transfer orbit → GEO → slot)

No official spacecraft-only orbit-raising Δv was published for the Atlas V flights. The following is a **computed estimate**, cross-checked against the published GOES-U requirement:

| Item | Δv | Basis |
|---|---|---|
| Combined apogee-raise + plane-change (injection → GEO circular-equatorial), Atlas V case | **≈ 0.98 km/s** *(computed)* | Single-impulse vector estimate: v_apogee ≈ 2.22 km/s (vis-viva at r = 41,664 km on the 14,486 × 41,664 km injection ellipse), to v_GEO = 3.0747 km/s across Δi = 10.6°: Δv = √(v₁²+v₂²−2v₁v₂cos Δi) ≈ 0.98 km/s |
| Multi-burn / finite-burn / gravity-loss penalty | +tens of m/s | Splitting into ~5 apogee passes adds a small penalty vs. the ideal single impulse |
| Drift-to-slot + drift-stop | tens of m/s | Small a-offset drift orbit + phasing coast |
| **Spacecraft orbit-raising total (Atlas V), estimate** | **≈ 1.0–1.3 km/s** *(low confidence)* | Consistent with the computed ~0.98 km/s and with the ~987 m/s GOES-U mission-max requirement |

**Context anchors (verified):** a *standard* GTO→GEO circularization is ~1.5 km/s; GOES is injected at low inclination with apogee already near GEO, so its spacecraft share is lower. The Tsiolkovsky ceiling from wet/dry mass (5,192/2,857 kg → ~2,335 kg usable propellant, LEROS-1c Isp 324 s) is v_e·ln(m0/m_dry) = 9.81·324·ln(5,192/2,857) ≈ **1.90 km/s** of *total* onboard Δv capacity — comfortably above the ~1.0–1.3 km/s orbit-raising need, with the remainder reserved for 10–15 yr of N-S/E-W station-keeping.

### 3.3 Contingency mode

If the LAE failed, GOES could reach GEO on its monopropellant station-keeping thrusters in **~4 weeks** (vs. ~2), reducing operational life from ~20 to ~18 years — still within mission need. No single-point failure per NASA.

---

## 4. Key Parameters & Confidence Flags

**High confidence (multiple independent sources or primary source):**
- Bus: Lockheed Martin A2100A, three-axis stabilized (now marketed "LM2100").
- Wet mass 5,192 kg (R/S/T) / 5,000 kg (U, exact per goes-r.gov); dry 2,857 kg / 2,925 kg.
- LAE: Nammo LEROS-1c, 458 N thrust, Isp 324 s, dual-mode MON/hydrazine.
- GOES-16 injection: 8,108 × 35,286 km; Centaur three-burn durations 7:38 / 5:36 / 1:33; third burn begins ~T+3 h 28 min; separation ~3.5 h.
- GEO target: 35,786 km altitude, radius 42,164 km, v 3.0747 km/s, period 86,164 s.
- Time to GEO 10–12 days; nominal ~2 weeks.
- GOES-U: 566 m/s used vs. 987 m/s requirement; final slot 75.2° W.
- Design life: 15 years (10 operational + up to 5 on-orbit storage/spare).

**Injection inclination — sources disagree; corrected value adopted:**
- The launch-vehicle primary sources (spaceflightnow ascent table, spaceflight101 launch profile) state **10.6°** for GOES-R/16. The GOES-16 Wikipedia infobox states **9.52°** (the 9.52° figure genuinely applies to **GOES-17/S**: 8,215 × 35,286 km at 9.52°). **10.6° is adopted as the GOES-16 injection inclination**; the ~9.5° "series-typical" value seen elsewhere conflates the two flights. Per-mission variation of ~1° is real.

**Corrected from initial claims:**
- Centaur performs **three** burns, not two (the "two-burn GTO" description understates it).
- Centaur burn-1 perigee is ~167 × 541 km (from "104 × 336 statute miles"), **not** ~193 km (a unit slip).
- SRB "110 s" is time-to-jettison; AJ-60A nominal propellant burn is ~94 s.
- Falcon Heavy center-core expenditure (jettison ~T+4:04) upgraded from medium to confirmed.

**Medium / low confidence (flag explicitly, do not treat as precise):**
- Usable propellant ~2,335 kg — **derived** (wet − dry), not an official loading figure.
- Number of LAE burns "≈ 5" — inferred (GOES-16: one initial burn + four post-anomaly burns each <41 min). Nominal count for a clean flight not separately published.
- GOES-16 "8 days raising + 4 days fine-tuning" split — **single-source (Wikipedia) and internally inconsistent** (8 + 4 ≠ the stated 10 days). Treat as illustrative only.
- Station-keeping thruster Isp ~200 s — generic monopropellant-hydrazine value, not confirmed for this bus.
- Spacecraft orbit-raising Δv ~1.0–1.3 km/s (Atlas V) — **computed estimate**, no official figure exists.
- Per-burn intermediate orbital states in Section 2 — illustrative.

**Design-life vs. fuel-life apparent tension (state, don't reconcile silently):** the **design life is 15 years** (goes-r.gov), yet contingency reporting cites "~20 → ~18 years," and the GOES-U propellant margin was described as extending "expected fuel life to 20+ years." These describe different things (contractual design life vs. propellant-limited life expectation) and the sources use both framings; both are reported here rather than choosing one.

**Not published / unknown:**
- Argument of perigee at Atlas V GOES injection (orientation so apogee sits near an equatorial node is physically standard but unverified for this mission).
- Exact GOES-U Falcon Heavy separation elements (perigee/apogee/inclination/arg. of perigee).

---

## 5. Modeling & Optimizing It in orbitopt

The mission maps onto orbitopt's existing **two-stage pattern** (fast vectorizable pygmo global search → tudatpy high-fidelity verification), the same architecture used for the Artemis II free-return work. Because every burn happens at the shared apogee (r ≈ R_GEO) rather than at a boundary-value transfer point, this is a fixed-N **impulsive apogee-burn sequence** with no Lambert solve — `lambert/*` is intentionally *not* reused.

### 5.1 Modules to reuse (no changes)
- `core/problem.py::OrbitOptProblem` — subclass it; implement `get_bounds`/`fitness`/`batch_fitness` so `has_batch_fitness()` is True and the run routes through the GPU BFE.
- `core/gpu.py::get_array_module()/to_numpy()` — write `batch_fitness` against `xp` so it runs on CuPy or numpy (same style as `dynamics/nbody_gpu.py`).
- `optimize/runner.py::run_optimization(...)` with `pg.pso_gen`/`pg.cmaes`, and `optimize/gpu_bfe.py::GpuBatchFitnessEvaluator` — drive the search unchanged.
- `bodies.py::mjd2000_from_date()`, `mjd2000_to_ephemeris_seconds()` — epoch → SPICE ephemeris-seconds for verification.
- `verify/tudat_propagate.py::propagate_multi_arc()`, `PropagationResult`, `_propagate_single_coast()`, `body_position_at_absolute_epoch()`, `find_altitude_crossing()`.
- `verify/differential_correction.py::target_lunar_flyby()` — structural template (damped 3×3 Newton, finite-difference Jacobian, backtracking).
- `viz/scene.py` (`body_entry`, `scene_document`, `COLOR`, `RADIUS_DISPLAY`); `viz/mission_timeline.py::compute_and_export_mission()` as the copy-from exporter template; the unchanged `viz/scene_renderer.py` + `viz/pv_viewer.py` render path.
- *Optional:* `dynamics/nbody_gpu.py::propagate_kepler_batch()` (+ `MU_EARTH_KM3_S2`) if a propagate-and-burn screening variant is wanted.

### 5.2 New components
- **`problems/geo_raising.py` — `GeoRaisingProblem(OrbitOptProblem)`**: the fixed-N apogee-burn UDP with a real vectorized `batch_fitness()` (vis-viva at r = R_GEO + out-of-plane velocity composition, closed-form, GPU-capable). Constructor: `gto_perigee_km`, `gto_apogee_km`, `gto_inclination_deg` (~10.6 for the modeled GOES injection, or ~27 for a generic Cape GTO), `n_burns`, mass/Isp/thrust, `target_longitude_deg`.
- **`examples/10_gto_geo_orbit_raising.py`**: build GTO from GOES-like elements (8,108 × 35,286 km, 10.6°, 5,192 kg, LEROS-1c 458 N / 324 s), run `run_optimization` for an **outer sweep of n_burns = 1…6** (pygmo continuous can't vary vector length), pick the min-total-Δv champion, then refine + verify in tudatpy and print achieved (a, e, i) and drift longitude.
- **Terminal-element targeter** — add `target_geo_insertion()` to `differential_correction.py` (or sibling): unknowns = final apogee-burn 3-vector; residuals = (a − a_GEO, e − 0, i − 0); reuse `target_lunar_flyby`'s damped-Newton / FD-Jacobian / backtracking verbatim.
- **Acceleration-model extension to `propagate_multi_arc()`**: optional perturbations config — `earth_sh_degree_order=(2,2)` (J2 + J22 triaxiality), `third_bodies=('Sun','Moon')`, `srp=dict(area_m2, Cr, mass_kg)` — keeping the point-mass default for back-compat (the README's named extension point).
- **`viz/geo_raising.py::compute_and_export_geo_mission()`**: Earth-centered SceneData (`centralBodyId='earth'`, `distance_unit='km'`), spacecraft trail over the raising sequence, static 42,164 km GEO reference ring (`orbitDashed=True`), per-apogee-burn timeline events, info rows (total Δv, achieved a/e/i, station longitude).
- **`tests/test_geo_raising.py`**: `batch_fitness`-matches-`fitness` + known-optimum-split sanity test, plus a targeter-converges test (mirroring `test_problem_interfaces.py`, `test_cislunar.py`).

### 5.3 Decision variables
- **Per-burn plane-change split** `Δi[k]` (or fractions `f[1..N]` of i₀ removed at each apogee, N−1 free, last dependent so Σ = i₀) — the primary lever; the classic result concentrates plane change at the slowest/highest apogees.
- **Per-burn intermediate perigee radii** `rp[1..N-1]` (rp[N] = R_GEO fixed) — how much apogee-raise energy each burn adds. For a supersynchronous variant, additionally expose an intermediate super-apogee radius > R_GEO with a perigee-lowering final burn.
- **Number of burns N** — constructor arg, swept in the outer loop; a per-burn magnitude lower-bounded at 0 lets an unneeded burn vanish inside a fixed-N run.
- **Drift/station longitude** `target_longitude_deg` (75.2° W East, ~137° W West) → small `delta_a_km`, phasing-coast duration, drift-stop Δv; resolved in the tudatpy stage (where J22 matters), folded as a soft target in screening.
- **Departure/first-apogee epoch** (mjd2000) → `mjd2000_to_ephemeris_seconds` for the verification stage (sets Sun/Moon geometry and plane-change RAAN).

### 5.4 Objective
Minimize total impulsive Δv = Σ|Δv_k| (m/s) in one vectorized array op: each |Δv_k| from the vis-viva speed change at r = R_GEO plus the out-of-plane component (law of cosines on the apogee velocity triangle). Single-objective (`get_nobj()==1`). Terminal orbit is circular-equatorial-GEO **by construction**, so the primary formulation has no penalty terms. The alternate (supersync / explicit-constraint) formulation adds soft `w_a|a−a_GEO| + w_e|e| + w_i|i| + w_lon|lon−target|` in the `|value−target|` style of `FreeReturnScreeningProblem`.

### 5.5 Constraints
- Terminal a = R_GEO ≈ 42,164.17 km (v = 3.0747 km/s, T = 86,164 s) — by construction or soft penalty.
- Terminal e ≈ 0, i ≈ 0 — by construction (plane-change fractions sum to i₀ ≈ 10.6°/27°) or soft penalty.
- Per-burn finite-burn feasibility: |Δv_k| ≤ Δv_max_per_pass from apogee dwell time and LEROS-1c thrust/mass (the physical reason multi-burn beats one big AKM burn; GOES-16's <41-min-per-burn limit is a concrete instance).
- Propellant budget: total Δv ≤ v_e·ln(m0/m_dry) ≈ 1.90 km/s (5,192/2,857 kg, Isp 324 s) — hard cap/penalty.
- Drift longitude within tolerance with ~0 drift rate — soft target in screening; verified against real J22 triaxiality in tudatpy (stable points 75.3° E / 104.7° W).

### 5.6 tudatpy verification
Re-fly the optimized impulsive burn+coast sequence with the extended `propagate_multi_arc` (`central_body='Earth'`, alternating `{impulsive_burn: Δv_k}` / `{coast: time-to-next-apogee}` arcs), then refine the final burn with `target_geo_insertion()` so achieved (a, e, i) hit GEO within tolerance. Perturbations (README extension point): (1) Earth `spherical_harmonic_gravity(2,2)` = J2 (residual RAAN/arg drift) + J22 tesseral (E-W libration toward 75.3° E / 104.7° W); (2) Sun + Moon `point_mass_gravity()` (the ~0.85°/yr N-S inclination drift — dominant long-term SK cost at GEO); (3) SRP via cannonball/panelled `radiation_pressure` from the Sun with Earth occultation (eccentricity oscillation of the light bus). Follow the repo's ECLIPJ2000-everywhere and SI-meters (tudat) vs. km/km·s (screening) conventions, converting at the handoff like `free_return.as_meters()`. An independent finer-step re-propagation (finer than the corrector step, per the RK4 gotcha) confirms terminal a/e/i and reports the residual one-week station-keeping Δv — the analog of examples/05's post-hoc closest-approach re-measure.

### 5.7 Mission Control scene
`compute_and_export_geo_mission()` builds the SceneData via `viz.scene` helpers (copying `mission_timeline.py`): Earth at origin (`COLOR['earth']`), spacecraft as a `body_entry` with `trail={times, positions}` decimated to `max_output_points` from the `propagate_multi_arc` history (GTO ellipse → successively rounder transfer ellipses → final circular GEO ring), plus the static 42,164 km equatorial GEO reference polyline (`orbitDashed=True`). Timeline events: one `{label:'Apogee burn k', time, note:'Δv … m/s'}` per burn, plus `GEO insertion` and `On station at <lon>`. Info rows: total Δv, achieved a/e/i, station longitude. Register in `viz/app.py::_builtin_missions()` as `('GOES GTO→GEO Raising', load_geo_raising)`; the shared `scene_renderer.py` + `pv_viewer.py` path and time-warp scrubber work unchanged, and it opens via `orbitopt view` on the exported JSON. Optionally add `examples/11_geo_raising_data.py` to cache the JSON (06/07 convention).

---

## 6. Open Questions & Assumptions

1. **Argument of perigee / RAAN at injection.** Not published for any GOES flight. The screening problem assumes apogee occurs at an equatorial node (so the apogee burn combines plane change with perigee raise); the tudatpy stage must be seeded with an assumed arg-of-perigee, and results depend on it. **Assumption, unverified.**
2. **GOES-U injection elements.** SpaceX/NOAA published no numeric perigee/apogee/inclination. Modeling the Falcon Heavy case requires either back-solving from the 566 m/s spacecraft requirement or assuming a near-GEO/supersynchronous apogee. Flag any GOES-U-specific orbital state as reconstructed.
3. **Official spacecraft orbit-raising Δv (Atlas V flights).** None exists in the source set; Section 3.2 uses a computed ~1.0–1.3 km/s. If a program value surfaces it should replace the estimate.
4. **Nominal LAE burn count.** "≈ 5" is inferred largely from GOES-16's anomaly-shaped campaign (1 + 4 burns). A clean nominal count is unconfirmed; the outer n_burns sweep (1–6) brackets it deliberately.
5. **Per-burn intermediate elements.** Section 2's intermediate states are illustrative; the optimizer produces the actual split, which is not validated against flight telemetry (none published).
6. **Station-keeping Isp and propellant loading.** ~200 s (monopropellant) and ~2,335 kg usable are generic/derived; the Tsiolkovsky ceiling and SK-reserve split inherit that uncertainty.
7. **Design-life vs. fuel-life framing.** 15-yr design life vs. 18–20+ yr propellant-life statements coexist in the sources; the model should treat the propellant budget (Section 3.2) as the governing constraint and report life as propellant-limited, noting the 15-yr contractual figure separately.
8. **Finite-burn feasibility bound.** `Δv_max_per_pass` depends on apogee dwell time and instantaneous mass; the 458 N thrust and the GOES-16 <41-min-per-burn datum give a starting point, but the exact per-pass limit used by the program is unknown and is an assumed constraint.