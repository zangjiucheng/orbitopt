# Launch-site and launch-vehicle data: sources, confidence and limitations

orbitopt ships two small reference databases:

| file | contents | loader |
|---|---|---|
| `src/orbitopt/data/launch_sites.yaml` | pad location, altitude, Earth-rotation gain, range-safety azimuth corridors | `orbitopt.launch_sites` |
| `src/orbitopt/data/launch_vehicles.yaml` | LEO/GTO/TLI capability, C3-vs-payload curve, upper-stage restarts/coast | `orbitopt.launch_vehicles` |

**This is planning-level data, not authoritative data.** It is good enough
to rank trajectories and reject clearly infeasible ones. It must not be
quoted as a vehicle's or range's actual capability. For a real mission,
use the provider's current user's guide and a mission-specific performance
quote.

## What every record carries

Each data item records where it came from. The JSON schemas enforce this,
and the loaders reject entries that don't:

| field | sites (`sources[*]`) | vehicles (orbit entry / `c3_curve` / `upper_stage`) |
|---|---|---|
| `source` / `ref` | required | required |
| `provenance` | required: `official` · `public_estimate` · `derived` | required: `official` · `public_estimate` |
| `accuracy` | stated or estimated accuracy | same |
| `as_of` | `YYYY[-MM[-DD]]`: publication date of the document, or access date for web pages | same |

A launch site must cite at least one source for each field group
(`location`, `altitude`, `azimuth_corridors`, `rotation_speed_gain`).
Otherwise `load_launch_sites` raises `ValueError`.

### Provenance vocabulary

- **official**: taken from a document published by the launch provider
  or agency, such as a user's manual, reference guide or vehicle web page.
  The number is official, but it is still rounded, often quoted for a
  reference orbit that isn't stated, and can change between issues.
- **public_estimate**: everything else. That covers secondary references
  (Isakowitz et al.), public compilations (Wikipedia coordinates, the GCAT
  launch log), hand-digitized plots (NASA LSP performance curves) and
  reconstructions from mission history. **These are approximations. Do not
  present them as facts.**
- **derived**: computed by orbitopt from other fields, for example the
  rotation gain computed from latitude and altitude. Its confidence is
  that of its inputs, which are listed separately.

Launch sites also keep a summary field, `confidence: published | approximate`,
for their azimuth corridors. `published` means the corridor limits appear in
a published document. `approximate` means they were rebuilt from the
inclinations that have actually been flown.

## Confidence in reports

Outputs that use this data report its sources:

- `LaunchConstraints.describe()` (printed by
  `examples/12_launch_constrained_transfer.py`) has a `data_sources` block.
  It holds the overall `confidence` and one record per item behind the C3
  limit (the vehicle's `c3_curve`) and the DLA limit (the site's `location`
  and `azimuth_corridors`).
- `orbitopt launch-window <site> ...` (table output) ends with a
  "Data sources" block for the site location and the corridors it used.
  Corridors given with `--azimuth` replace the site's corridors, so they
  are not listed. `--json` output is unchanged.
- Programmatic: `site.data_sources()`, `vehicle.data_sources()` and
  `orbitopt.data_provenance.format_data_sources(...)` /
  `data_sources_report(...)`.

The overall confidence is `public_estimate` if **any** contributing item is an
estimate. It is `official` only when every item is official or derived.

## Known limitations

Launch vehicles

- **C3 curves are all `public_estimate`.** Falcon 9, Falcon Heavy and
  Atlas V 551 were hand-digitized from NASA Launch Services Program plots
  (about ±10 %). SLS, Vulcan and Long March curves come from presentations
  or mission history (about ±15–20 %). No official C3 table is used.
- Official LEO/GTO figures are headline numbers. They are rounded, often
  quoted for an orbit that isn't stated (inclination, altitude, GTO Δv
  to GEO), and Falcon numbers assume full expendability. They are not
  comparable across providers without normalization.
- Some TLI values are estimates read off or near the C3 curve at
  C3 ≈ −2 km²/s². The Falcon 9 TLI value (4400 kg) is lower than its own
  curve (5200 kg). Both are estimates.
- Upper-stage `max_restarts` / `max_coast_s` are all `public_estimate`.
  They are inferred from flown missions, not from published qualification
  limits. Real limits depend on the mission kit, propellant boil-off and
  thermal constraints.
- Payload-adapter mass, fairing limits, performance reserves and
  reusability (drone-ship or RTLS) are not modelled.

Launch sites

- Pad coordinates mostly come from public compilations (about 100 m).
  That is plenty for the rotation gain and for launch-window timing, but
  it is not survey grade. Each site is represented by one reference pad.
- Only Kourou's corridors come from a launch-provider document. The
  Eastern and Western Range corridors come from a secondary reference
  (2004). The corridors for the Chinese sites and Baikonur were rebuilt
  from flown inclinations (about ±5°). Real range-safety limits are set
  per mission, depend on the vehicle and drop zones, and are not public.
- Inclinations are for direct ascent only. Dog-leg manoeuvres, which
  real vehicles do fly, are not part of the corridor model.

Dates

- `as_of` gives the publication date of the document, or the year a web
  page was accessed. Provider web pages change without notice. Re-check
  them before relying on a value.

## Updating the data

Copy the YAML file and edit it, keeping `source`, `provenance`, `accuracy`
and `as_of` for every item. Then load it with
`load_launch_sites(path)` / `load_launch_vehicles(path)`. Mark a value
`official` only when you took it directly from a provider or agency
document, and cite that document's issue and date.
