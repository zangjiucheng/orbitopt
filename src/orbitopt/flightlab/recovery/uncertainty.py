"""R-04 (task #236): configuration-declared uncertainty for the recovery
Monte Carlo.

FL-304 shipped one flat :class:`~orbitopt.flightlab.recovery.montecarlo.
Dispersions` setup: multiplicative uniform factors (thrust / Isp / density /
Cd) plus normal separation-state sigmas, the distribution *kind* of every
source hardcoded. R-04 promotes the perturbation model to a declaration: one
named source per physical disturbance (wind, density, thrust, Isp, drag,
initial state), each carrying its distribution kind, its parameters and an
individual on/off switch, all loaded from a config mapping or file:

    uncertainty:
      thrust:  {enabled: true, distribution: uniform_pct, half_width_pct: 5.0}
      isp:     {enabled: true, distribution: normal_pct, sigma_pct: 3.0}
      density: {enabled: true, distribution: uniform_pct, half_width_pct: 10.0}
      drag:    {enabled: false}
      wind:
        enabled: true
        distribution: normal
        sigma_east_m_s: 3.0
        sigma_north_m_s: 3.0
        sigma_up_m_s: 1.0
      initial_state:
        enabled: true
        distribution: normal
        sigma_altitude_m: 400.0
        sigma_downrange_m: 800.0
        sigma_crossrange_m: 800.0
        sigma_speed_m_s: 4.0
        sigma_heading_deg: 0.15

The declaration (kind + every parameter + the switches) is recorded verbatim
into the ``montecarlo.json`` report, so a report always says which
distribution it was flown under.

Determinism is a stream-layout contract: sources draw in the fixed order
``thrust, isp, density, drag, wind(east, north, up),
initial_state(altitude, downrange, crossrange, speed, heading)``, a disabled
source consumes no RNG and a zero-width component consumes no RNG either.
A config translated from the legacy flat :class:`Dispersions` therefore
draws *the same numbers from the same seed* as FL-304 always did (wind
inactive), so pre-R-04 fixed-seed runs stay reproducible; wind draws insert
themselves between drag and the separation state.

Physics accounting (unchanged from FL-304, extended to wind): thrust, Isp
and drag enter the FL-302 planner *and* the acceptance re-integration;
density scale and wind enter the acceptance atmosphere only (the planner's
rollout keeps the nominal air) -- a deliberate plan/reality mismatch; the
initial-state source moves the separation state itself, so it enters
everywhere. The wind itself lives in the FL-301 propagator
(:class:`~orbitopt.flightlab.recovery.propagator.BoosterPhysics`): one
steady ENU vector per sample, no shear, no turbulence.

    from orbitopt.flightlab.recovery.uncertainty import UncertaintyModel
    model = UncertaintyModel.from_file("uncertainty.yaml")
    mc = run_monte_carlo(target, vehicle, ascent, uncertainty=model, seed=7, n=24)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

__all__ = ["SOURCE_ORDER", "UncertaintyModel", "UncertaintySource"]

#: The canonical draw order (the stream layout is the determinism contract).
SOURCE_ORDER: tuple[str, ...] = (
    "thrust", "isp", "density", "drag", "wind", "initial_state")

#: source -> draw keys, in fixed intra-source order
_DRAW_KEYS: Mapping[str, tuple[str, ...]] = {
    "thrust": ("thrust_factor",),
    "isp": ("isp_factor",),
    "density": ("density_scale",),
    "drag": ("cd_factor",),
    "wind": ("wind_east_m_s", "wind_north_m_s", "wind_up_m_s"),
    "initial_state": ("sep_altitude_m", "sep_downrange_m", "sep_crossrange_m",
                      "sep_speed_m_s", "sep_heading_deg"),
}

#: source -> {distribution kind: (params, neutral draw key -> value)}
_KIND_PARAMS: Mapping[str, Mapping[str, tuple[str, ...]]] = {
    "thrust": {"uniform_pct": ("half_width_pct",),
               "normal_pct": ("sigma_pct",)},
    "isp": {"uniform_pct": ("half_width_pct",),
            "normal_pct": ("sigma_pct",)},
    "density": {"uniform_pct": ("half_width_pct",),
                "normal_pct": ("sigma_pct",)},
    "drag": {"uniform_pct": ("half_width_pct",),
             "normal_pct": ("sigma_pct",)},
    "wind": {"normal": ("sigma_east_m_s", "sigma_north_m_s", "sigma_up_m_s",
                        "mean_east_m_s", "mean_north_m_s", "mean_up_m_s")},
    "initial_state": {"normal": ("sigma_altitude_m", "sigma_downrange_m",
                                 "sigma_crossrange_m", "sigma_speed_m_s",
                                 "sigma_heading_deg")},
}

#: params that are sigmas/widths (must be >= 0); the rest (wind means) are
#: free-signed
_NONNEG: frozenset[str] = frozenset(
    p for kinds in _KIND_PARAMS.values() for params in kinds.values()
    for p in params if p.startswith("sigma") or p.startswith("half_width"))

#: optional wind means default to calm
_DEFAULTS: Mapping[str, float] = {
    "mean_east_m_s": 0.0, "mean_north_m_s": 0.0, "mean_up_m_s": 0.0}

#: legacy flat Dispersions field -> (source, param)
_LEGACY_MAP: tuple[tuple[str, str, str, str], ...] = (
    ("thrust_pct", "thrust", "uniform_pct", "half_width_pct"),
    ("isp_pct", "isp", "uniform_pct", "half_width_pct"),
    ("density_pct", "density", "uniform_pct", "half_width_pct"),
    ("cd_pct", "drag", "uniform_pct", "half_width_pct"),
    ("sep_altitude_m", "initial_state", "normal", "sigma_altitude_m"),
    ("sep_downrange_m", "initial_state", "normal", "sigma_downrange_m"),
    ("sep_crossrange_m", "initial_state", "normal", "sigma_crossrange_m"),
    ("sep_speed_m_s", "initial_state", "normal", "sigma_speed_m_s"),
    ("sep_heading_deg", "initial_state", "normal", "sigma_heading_deg"),
)


@dataclass(frozen=True)
class UncertaintySource:
    """One declared disturbance: its distribution kind, its parameters and
    its switch. ``distribution="none"`` is the shape of a disabled source
    that declared nothing else; :meth:`is_active` separates "off" from "on
    with zero width"."""

    distribution: str = "none"
    params: Mapping[str, float] = field(default_factory=dict)
    enabled: bool = True

    def is_active(self) -> bool:
        """Enabled and at least one component can actually move the draw:
        a factor with width/sigma > 0, a wind component with sigma > 0 or a
        nonzero bias, an initial-state component with sigma > 0."""
        if not self.enabled or self.distribution == "none":
            return False
        if self.distribution in ("uniform_pct", "normal_pct"):
            return float(self.params.get("half_width_pct")
                         or self.params.get("sigma_pct") or 0.0) > 0.0
        if self.distribution == "normal":
            return any(
                (float(self.params.get(p, 0.0)) > 0.0 if p.startswith("sigma")
                 else float(self.params.get(p, 0.0)) != 0.0)
                for p in self.params)
        return False

    def to_dict(self) -> dict[str, Any]:
        if self.distribution == "none":
            return {"enabled": bool(self.enabled)}
        return {"enabled": bool(self.enabled), "distribution": self.distribution,
                **{k: float(v) for k, v in sorted(self.params.items())}}


@dataclass(frozen=True)
class UncertaintyModel:
    """The full declaration: one :class:`UncertaintySource` per named source,
    always all six, always in :data:`SOURCE_ORDER`. Validate through
    :meth:`from_config` / :meth:`from_file` / :meth:`from_dispersions`."""

    sources: Mapping[str, UncertaintySource]

    def __post_init__(self) -> None:
        if set(self.sources) != set(SOURCE_ORDER):
            raise ValueError(
                "an uncertainty model needs exactly these sources: "
                f"{', '.join(SOURCE_ORDER)}")
        ordered = {name: self.sources[name] for name in SOURCE_ORDER}
        object.__setattr__(self, "sources", ordered)
        active = [n for n in SOURCE_ORDER if self.sources[n].is_active()]
        if not active:
            raise ValueError(
                "an uncertainty setup needs at least one active disturbance "
                "source (enabled with a nonzero width/sigma); every source is "
                "off -- that is not a Monte Carlo, it is a nominal re-fly")

    # -- construction -------------------------------------------------------

    @staticmethod
    def from_config(config: Mapping[str, Any]) -> "UncertaintyModel":
        """Parse a declaration mapping -- either the ``uncertainty:`` document
        or its inner source map directly. Unknown sources, kinds, parameters
        or malformed values raise ``ValueError``: a silently-ignored typo in
        a disturbance declaration is how a Monte Carlo starts meaning
        something other than what its report says."""
        if not isinstance(config, Mapping):
            raise ValueError(f"uncertainty config must be a mapping, got {type(config).__name__}")
        if isinstance(config.get("uncertainty"), Mapping):
            config = config["uncertainty"]
        unknown = sorted(set(config) - set(SOURCE_ORDER))
        if unknown:
            raise ValueError(
                f"unknown uncertainty source(s) {unknown}; declared sources: "
                + ", ".join(SOURCE_ORDER))
        sources: dict[str, UncertaintySource] = {}
        for name in SOURCE_ORDER:
            node = config.get(name, {"enabled": False})
            if not isinstance(node, Mapping):
                raise ValueError(
                    f"uncertainty.{name} must be a mapping "
                    "({enabled, distribution, ...params})")
            enabled = bool(node.get("enabled", True))
            dist = str(node.get("distribution", "none"))
            if dist == "none":
                if enabled:
                    raise ValueError(
                        f"uncertainty.{name}: distribution 'none' means off; "
                        "set enabled: false or declare a distribution")
                sources[name] = UncertaintySource(enabled=False)
                continue
            kinds = _KIND_PARAMS[name]
            if dist not in kinds:
                raise ValueError(
                    f"uncertainty.{name}: unknown distribution {dist!r}; "
                    f"{name} supports {sorted(kinds)}")
            allowed = kinds[dist]
            bad = sorted(set(node) - {"enabled", "distribution"} - set(allowed))
            if bad:
                raise ValueError(
                    f"uncertainty.{name}: unknown parameter(s) {bad} for "
                    f"distribution {dist!r}; allowed: {sorted(allowed)}")
            params: dict[str, float] = {}
            for p in allowed:
                raw = node.get(p, _DEFAULTS.get(p))
                if raw is None:
                    raise ValueError(
                        f"uncertainty.{name}: {p} is required for distribution {dist!r}")
                try:
                    val = float(raw)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"uncertainty.{name}: {p} is not a number: {raw!r}") from exc
                if p in _NONNEG and val < 0.0:
                    raise ValueError(f"uncertainty.{name}: {p} must be >= 0, got {val}")
                params[p] = val
            sources[name] = UncertaintySource(dist, params, enabled)
        return UncertaintyModel(sources)

    @staticmethod
    def from_dict(decl: Mapping[str, Any]) -> "UncertaintyModel":
        """Inverse of :meth:`to_dict` (same flat shape; also accepts the
        ``uncertainty:`` document root)."""
        return UncertaintyModel.from_config(decl)

    @staticmethod
    def from_file(path: str | Path) -> "UncertaintyModel":
        """Read a YAML/JSON declaration file (YAML reads JSON)."""
        import yaml

        doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if doc is None:
            raise ValueError(f"{path}: empty uncertainty config")
        return UncertaintyModel.from_config(doc)

    @staticmethod
    def from_dispersions(disp: Any) -> "UncertaintyModel":
        """Translate the legacy flat FL-304 :class:`Dispersions` (or its
        ``to_dict()``) into the declarative form: uniform factor half-widths,
        normal separation-state sigmas, wind off. The draw stream is
        identical to the legacy one, so legacy fixed-seed runs stay
        reproducible through the new code path."""
        get = disp.get if isinstance(disp, Mapping) else (
            lambda k, d=None: getattr(disp, k, d))
        by_source: dict[str, dict[str, float]] = {}
        for legacy, name, _dist, param in _LEGACY_MAP:
            by_source.setdefault(name, {})[param] = float(get(legacy, 0.0) or 0.0)
        config: dict[str, Any] = {}
        for name in ("thrust", "isp", "density", "drag"):
            width = by_source.get(name, {}).get("half_width_pct", 0.0)
            config[name] = ({"enabled": True, "distribution": "uniform_pct",
                             "half_width_pct": width} if width > 0.0
                            else {"enabled": False})
        sig = by_source.get("initial_state", {})
        sigmas = [sig.get(f"sigma_{k}", 0.0) for k in
                  ("altitude_m", "downrange_m", "crossrange_m", "speed_m_s",
                   "heading_deg")]
        config["wind"] = {"enabled": False}
        config["initial_state"] = (
            {"enabled": True, "distribution": "normal",
             **{f"sigma_{k}": sig.get(f"sigma_{k}", 0.0) for k in
                ("altitude_m", "downrange_m", "crossrange_m", "speed_m_s",
                 "heading_deg")}}
            if any(s > 0.0 for s in sigmas) else {"enabled": False})
        return UncertaintyModel.from_config(config)

    # -- inspection ---------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """The declaration in config shape (exactly what :meth:`from_config`
        takes; byte-identical for equal models) -- this is what the
        ``montecarlo.json`` report records under ``uncertainty``."""
        return {name: self.sources[name].to_dict() for name in SOURCE_ORDER}

    def active_sources(self) -> tuple[str, ...]:
        return tuple(n for n in SOURCE_ORDER if self.sources[n].is_active())

    def dispersions_view(self) -> dict[str, float]:
        """The flat legacy :class:`Dispersions` view of this declaration
        (the report's ``dispersions`` section): factor widths and
        initial-state sigmas in their FL-304 field names; a ``normal_pct``
        source reports its sigma in the legacy ``*_pct`` slot (the kind is
        declared under ``uncertainty``), wind has no legacy field and does
        not appear. Exact for every FL-304-era setup."""
        view = {legacy: 0.0 for legacy, _n, _d, _p in _LEGACY_MAP}
        for legacy, name, _dist, param in _LEGACY_MAP:
            src = self.sources[name]
            if src.distribution == "none":
                continue
            view[legacy] = float(src.params.get(param)
                                 or src.params.get("sigma_pct") or 0.0)
        return view

    def neutral_draw(self) -> dict[str, float]:
        """Every draw key at its neutral (no-perturbation) value."""
        out: dict[str, float] = {}
        for name in SOURCE_ORDER:
            for key in _DRAW_KEYS[name]:
                out[key] = _NEUTRAL[key]
        return out

    # -- sampling ------------------------------------------------------------

    def draw(self, rng: np.random.Generator) -> dict[str, float]:
        """One sample's perturbation as plain data. Fixed order and fixed
        RNG-call sequence per active component (module docstring: the stream
        layout IS the determinism contract); disabled sources and zero-width
        components consume no RNG."""
        out: dict[str, float] = {}
        for name in SOURCE_ORDER:
            src, keys = self.sources[name], _DRAW_KEYS[name]
            if not src.is_active():
                for key in keys:
                    out[key] = _NEUTRAL[key]
                continue
            if src.distribution in ("uniform_pct", "normal_pct"):
                if src.distribution == "uniform_pct":
                    hw = float(src.params["half_width_pct"])
                    factor = 1.0 + float(rng.uniform(-hw, hw)) / 100.0
                else:
                    sig = float(src.params["sigma_pct"])
                    factor = 1.0 + float(rng.normal(0.0, sig)) / 100.0
                out[keys[0]] = factor
            elif name == "wind":
                for comp in ("east", "north", "up"):
                    sigma = float(src.params[f"sigma_{comp}_m_s"])
                    mean = float(src.params[f"mean_{comp}_m_s"])
                    out[f"wind_{comp}_m_s"] = (
                        mean + float(rng.normal(0.0, sigma)) if sigma > 0.0 else mean)
            else:  # initial_state
                for key in keys:
                    sigma = float(src.params[f"sigma_{_STATE_PARAM[key]}"])
                    out[key] = float(rng.normal(0.0, sigma)) if sigma > 0.0 else 0.0
        return out


#: draw key -> neutral value (factor 1.0, offset 0.0)
_NEUTRAL: Mapping[str, float] = {
    "thrust_factor": 1.0, "isp_factor": 1.0, "density_scale": 1.0,
    "cd_factor": 1.0, "wind_east_m_s": 0.0, "wind_north_m_s": 0.0,
    "wind_up_m_s": 0.0, "sep_altitude_m": 0.0, "sep_downrange_m": 0.0,
    "sep_crossrange_m": 0.0, "sep_speed_m_s": 0.0, "sep_heading_deg": 0.0}

#: initial-state draw key -> its sigma parameter suffix under ``sigma_``
_STATE_PARAM: Mapping[str, str] = {
    "sep_altitude_m": "altitude_m", "sep_downrange_m": "downrange_m",
    "sep_crossrange_m": "crossrange_m", "sep_speed_m_s": "speed_m_s",
    "sep_heading_deg": "heading_deg"}
