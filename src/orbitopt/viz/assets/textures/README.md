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
