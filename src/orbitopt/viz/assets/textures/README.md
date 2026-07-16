# Texture assets

2K equirectangular JPGs used by `orbitopt.viz.scene_renderer` to render bodies
with a `texture`+`radius` set (see `orbitopt.viz.scene.TEXTURE`) as real,
textured 3D spheres instead of flat point markers.

Source: [Solar System Scope](https://www.solarsystemscope.com/textures/),
based on NASA imagery/elevation data, licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

No texture is bundled for Pluto (not available from this source) or
spacecraft (a synthetic marker, not a real body with imagery) -- both fall
back to the point-marker rendering, same as any body with no `texture` field
at all.
