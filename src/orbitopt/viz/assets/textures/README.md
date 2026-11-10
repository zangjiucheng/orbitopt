# Texture assets

Equirectangular JPGs used by `orbitopt.viz.scene_renderer` to render bodies
with a `texture`+`radius` set (see `orbitopt.viz.scene.TEXTURE`) as real,
textured 3D spheres instead of flat point markers.

Resolution is tiered by how close the viewer gets to each body:

| texture | size | from |
|---|---|---|
| `earth_daymap.jpg` | 8192 x 4096 | Solar System Scope `8k_earth_daymap.jpg`, unmodified (LEO missions view Earth from a few hundred km) |
| `moon.jpg`, `mars.jpg` | 4096 x 2048 | Solar System Scope 8K maps, downscaled (smooth resampling, JPEG quality 90) |
| everything else | 2048 x 1024 | Solar System Scope 2K maps (only seen from afar, in the solar-system scene) |

All 8K would add ~60 MB to the package; this tiering adds ~8 MB.

Source: [Solar System Scope](https://www.solarsystemscope.com/textures/),
based on NASA imagery/elevation data, licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

No texture is bundled for Pluto (not available from this source) or
spacecraft (a synthetic marker, not a real body with imagery) -- both fall
back to the point-marker rendering, same as any body with no `texture` field
at all.

## Streamed Earth imagery (not bundled)

Close to Earth, Mission Control drapes tiles of
[EOX Sentinel-2 cloudless 2024](https://s2maps.eu) over the globe: up to
~10 m per pixel, picked for the current camera and downloaded on demand
(`orbitopt.viz.earth_tiles`), cached under `~/Library/Caches/orbitopt/tiles`
(macOS) or `~/.cache/orbitopt/tiles` (`ORBITOPT_TILE_CACHE` overrides).
"Sentinel-2 cloudless - https://s2maps.eu by EOX IT Services GmbH (Contains
modified Copernicus Sentinel data 2024)", licensed
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/):
non-commercial use only; the attribution is shown in the view while tiles
are drawn. View > High-res Earth imagery turns it off, and
`ORBITOPT_EARTH_TILES=off` disables the network access entirely.
