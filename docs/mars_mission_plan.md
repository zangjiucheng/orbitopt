# Mars — Earth to Orbit Insertion: Mission Plan

## Scope

The Mars mission models departure, heliocentric cruise and Mars orbit
insertion. It does not model launch ascent, atmospheric descent or landing.
The default trail starts in a 185 km parking orbit; an optional launch
configuration adds a site, vehicle, launch window and verified injection.
See [`mars2020_replay.yaml`](../examples/mars2020_replay.yaml).

## Why one continuous heliocentric-km scene

Earth departure and Mars approach are propagated in their natural
planet-centered frames, then translated using the planets' SPICE positions at
each epoch. The exported trail uses one heliocentric frame in kilometers.
Planetary details are small at the whole-mission scale; use Mission Control's
focus and tracking controls to inspect them.

## Real reference numbers used

The documented 2026-window search found a departure near 2026-10-30 and
arrival near 2027-09-06, with a 310.67-day flight and C3 around 9.2 km²/s².
The 185 km parking orbit and approximately 500 × 20,000 km Mars capture orbit
are modeling choices, not a reconstruction of a particular flight. The Earth
sphere-of-influence handoff uses 924,000 km.

## Phase-boundary idealizations (stated plainly, not hidden)

1. **Departure:** the Lambert solution assumes departure at Earth's
   heliocentric position. The propagated Earth-escape state differs; in the
   documented 2026 case, an uncorrected perturbed cruise misses the Lambert
   arrival by over 1.27 million km.
2. **Cruise correction:** `target_mars_approach` uses differential correction
   and a B-plane refinement to target periapsis. Targeting a point at a fixed
   epoch alone does not account for gravitational focusing. The targeter can
   fail; the report records the achieved result.
3. **Orbit insertion:** the approach is propagated with a local refinement
   around periapsis. A tangential impulsive burn targets capture, followed by a
   short orbit check.
4. **Scene continuity:** the displayed cruise uses the same numerical
   trajectory as the periapsis locator. Independently re-propagating the two
   pieces at different step sizes previously produced a visible seam.

## Optional real launch (`launch` config section)

A `launch` section supplies site, vehicle, payload and parking-orbit choices.
The search applies launch constraints, computes daily windows and plans the
site-to-injection sequence. A pinned departure skips the search. The injection
is numerically checked and gets its own report. Omitting `launch`, or setting
`enabled: false`, retains the idealized parking-orbit start.

## Modules

| Component | Purpose |
|---|---|
| `launch_injection.py` | Launch window, parking orbit and verified injection |
| `viz/mars_transfer.py` | Search, mission pipeline and scene export |
| `verify/mars_insertion.py` | Cruise correction, periapsis and insertion |
| `schemas/mission-mars-transfer-1.0.json` | Mission configuration schema |

## What this does not claim

The model does not reconstruct launch ascent or Mars landing. Convergence is
not guaranteed for every requested geometry. Candidates with unsafe modeled
Mars periapsis are rejected; if no candidate passes the verification gate, the
export is identified as **best effort**, not a verified success. See
[engineering notes](engineering-notes.md#candidate-selection-and-the-verification-gate-task-171).
