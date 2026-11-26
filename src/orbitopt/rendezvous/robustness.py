"""Robustness of a planned rendezvous under dispersion (E12-07, task #397).

E12-03 proves LEO-RDV-001 arrives; E12-04 proves the plan is safe. Neither
answers the operational question: how often does the mission still work when the
vehicle is not exactly where the config says, the burns are not exactly
commanded, the gravity field is not exactly J2 = 1.08262668e-3, and the
thermosphere is denser than the plan assumed? This module flies the reference
mission many times under those four disturbance families and reports what a
campaign decision-maker needs:

* a **hold-out success rate** -- a sample succeeds only when the chaser reaches
  the Hold Point inside both configured tolerances *and* every E12-04 safety
  constraint holds on every sample of the trajectory actually flown *and* the
  executed delta-v stays inside the fixed budget. Planning, safety, budget and
  accuracy are the mission's own contract
  (:func:`orbitopt.rendezvous.safety.evaluate_trajectory_safety` and the
  config's ``budget`` / ``tolerances`` / ``windows``), never a
  robustness-specific copy of them;
* **confidence intervals that match the sampling design** -- a Wilson score
  interval on the pooled rate, plus a whole-cluster bootstrap interval, because
  samples inside one cluster share one gravity-field / space-weather
  realization (*Clustering*) and are therefore not iid draws;
* a **failure taxonomy** with a sensitivity reading -- every failed sample
  carries exactly one governing class (``planning_infeasible`` /
  ``safety_violation`` / ``fuel_over_budget`` / ``arrival_accuracy`` /
  ``other``) and every source reports how strongly its own draw associates with
  failure;
* a **train / hold-out split** -- the two arms draw from disjoint RNG streams,
  so the headline rate is read off samples that informed nothing;
* an **independent spot check** of the borderline hold-out samples, in the
  E12-05 sense (:func:`spot_check`).

The four families, each a declared distribution in the campaign file and never
a hard-coded factor:

``initial_state``   Gaussian navigation error of the chaser state at the shared
                    epoch t0, in the chief's LVLH axes (radial, along-track,
                    cross-track) and rates, rotated into the config frame. The
                    *truth* state is the config's; the *planner* plans from
                    truth + error -- the direction that makes a nav error cost
                    arrival accuracy rather than something the error
                    conveniently cancels.
``execution``       per-burn magnitude factor ``1 + N(0, sigma)``, then an
                    isotropic pointing error: the commanded vector is rotated
                    by a Gaussian rotation vector whose Cartesian components are
                    each ``N(0, sigma_angle / sqrt(3))``, so the total
                    misalignment has the declared RMS.
``j2``              multiplicative error on the zonal coefficient the *truth*
                    feels; the planner keeps the declared nominal, so this is a
                    model-mismatch source, not a parameter sweep.
``drag``            thermospheric density factor, log-normal with median 1
                    because the solar-activity range at 500 km is multiplicative
                    and spans an order of magnitude. The atmosphere is present
                    whenever the source is enabled: the plan ignores it, and
                    that omission is the mismatch being priced. What bends a
                    *relative* trajectory is differential drag, so the two
                    vehicles carry separately declared ballistic coefficients.

Clustering (why the CI is a cluster bootstrap): ``j2`` and ``drag`` are drawn
once per **cluster** -- one gravity-field / space-weather realization inside
which every sample of that cluster flies -- while ``initial_state`` and
``execution`` are drawn per sample. A cluster is the honest resampling unit, so
:func:`orbitopt.flightlab.recovery.reliability.cluster_bootstrap_rate_ci`
resamples whole clusters. The pooled Wilson interval is reported beside it and
labelled the optimistic reading when the two differ: the same discipline R-06
(task #238) applies to per-seed clustering in the recovery Monte Carlo.

Sample size (the task's "sample size set by the target CI width"): the campaign declares a
target Wilson width, a confidence level and a cluster size; each arm's size is
derived with :func:`orbitopt.flightlab.recovery.stats.required_sample_size` at
the worst-case planning rate and rounded up to whole clusters, so the quoted
interval is one the run was designed to support rather than one discovered
afterwards.

Campaign shape: one ground plan (by default
:func:`orbitopt.rendezvous.plan_rendezvous_safe`'s constrained optimum) is
fixed for the campaign. Each sample re-opens the transfer window on its
*estimated* state (:func:`orbitopt.rendezvous.planner._gate_crossing_at`),
re-targets burn 1 on the configured nonlinear model at the campaign
time-of-flight, re-screens the candidate against the fixed constraints, and
flies the result on the perturbed truth. The time-of-flight stays the ground
plan's: this prices a locked mission design, and per-sample re-optimisation
costs ~9 s for a delta-v change E12-05 measured as flat to within 0.01 m/s
across the window (docs/adr/rendezvous-0003-robustness-dispersion-campaign.md).
The re-plan step is what makes ``planning_infeasible`` a live outcome under
dispersion: a sample fails that way when its estimate never reaches the gate
inside the drift window, when burn-1 targeting does not close, or when the
re-targeted candidate violates a fixed constraint the screen can see.

Determinism: each cluster and sample owns its own RNG stream
(``numpy.random.default_rng([seed_base, cluster])`` for the cluster's
environment, ``numpy.random.default_rng([seed_base, cluster, index])`` for the
sample's errors), so a report is a pure function of (config, campaign, ground
plan, ``jobs``): the same seeds give the same report on 1 worker or 16.

    from orbitopt.rendezvous import load_reference_config, plan_rendezvous_safe
    from orbitopt.rendezvous.robustness import load_reference_campaign, run_campaign
    cfg = load_reference_config()
    plan = plan_rendezvous_safe(cfg).plan
    result = run_campaign(cfg, load_reference_campaign(), plan, jobs=8)
    result.holdout.summary["success_rate"]
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.integrate import solve_ivp
from scipy.stats import spearmanr

from orbitopt.flightlab.atmosphere import ExponentialAtmosphere
from orbitopt.flightlab.frames import EARTH_ROTATION_RATE
from orbitopt.flightlab.recovery.reliability import cluster_bootstrap_rate_ci
from orbitopt.flightlab.recovery.stats import (
    required_sample_size,
    success_rate_interval,
)
from orbitopt.relative.frames import eci_to_lvlh, lvlh_to_eci
from orbitopt.relative.propagation import EARTH_J2, EARTH_RE, _accel
from orbitopt.verify.rendezvous_refly import (
    CW_VS_NONLINEAR_TOLERANCES,
    cw_mission_replay,
    main_model_key_quantities,
    resolve_refly_backend,
)

from .canonical import (
    SealedRendezvousPlan,
    canonical_config_sha256,
    consume_sealed_plan,
    maneuvers_sha256,
    rendezvous_plan_digest,
)
from .config import RendezvousMissionConfig
from .planner import (
    RendezvousPlan,
    RendezvousPlanError,
    _gate_crossing_at,
    _solve_transfer,
    initial_states,
)
from .safety import (
    TRAJECTORY_CHECKS,
    PassiveRelease,
    candidate_safety_reason,
    evaluate_trajectory_safety,
)

__all__ = [
    "DISTURBANCE_ORDER",
    "FAILURE_CLASSES",
    "FAILURE_PRECEDENCE",
    "INTEGRATION_VS_RK4_TOLERANCES",
    "PASSIVE_CHECK_PREFIX",
    "SAMPLE_REFLY_MAX_PASSIVE_CAUSED",
    "SAMPLE_REFLY_MAX_SUCCESSES",
    "SAMPLE_REFLY_MAX_TOTAL",
    "SAMPLE_REFLY_SPEC_KIND",
    "ArmResult",
    "CampaignResult",
    "DisturbanceModel",
    "DragError",
    "ExecutionError",
    "FlightOutcome",
    "InitialStateError",
    "J2Error",
    "RobustnessCampaign",
    "RobustnessConfigError",
    "SampleDraws",
    "SampleRecord",
    "build_reflight_spec",
    "draw_sample",
    "failure_summary",
    "fly_truth",
    "fly_truth_sample",
    "grade_flight",
    "grade_trajectory_checks",
    "grid_convergence",
    "load_campaign",
    "load_reference_campaign",
    "parse_campaign",
    "plan_sample",
    "run_campaign",
    "select_reflight_samples",
    "run_campaign_arm",
    "select_spot_checks",
    "sensitivity_reading",
    "solo_campaigns",
    "spot_check",
]

#: Canonical source order -- the determinism contract, same discipline as
#: ``orbitopt.flightlab.recovery.uncertainty.SOURCE_ORDER``.
DISTURBANCE_ORDER: tuple[str, ...] = ("initial_state", "execution", "j2", "drag")

#: Governing failure families, in report order. Every failed sample carries
#: exactly one, chosen by :data:`FAILURE_PRECEDENCE`. Counts are reported for
#: all of them, so a clean campaign shows explicit zeros.
FAILURE_CLASSES: tuple[str, ...] = (
    "planning_infeasible",
    "safety_violation",
    "fuel_over_budget",
    "arrival_accuracy",
    "other",
)

#: Precedence when one sample fails several gates, highest first: planning (a
#: sample that could not be planned is not graded), then safety (what a
#: range-safety reviewer acts on), then the fixed budget, then arrival
#: accuracy. ``other`` catches everything else and is never inferred.
FAILURE_PRECEDENCE: tuple[str, ...] = ("planning_infeasible", "safety_violation",
                                       "fuel_over_budget", "arrival_accuracy")

#: Tail threshold in sigma-equivalents for the sensitivity reading: the same
#: 1-sigma split R-06 reports, so the two campaigns' tail rates read side by side.
TAIL_SIGMA = 1.0

_OMEGA_Z = np.array([0.0, 0.0, EARTH_ROTATION_RATE])  # rad/s, equatorial inertial
_REFERENCE_CAMPAIGN = "leo_rdv_001_robustness.yaml"

#: Bands for the independent-integration line of the spot check: same force law,
#: different integrator, so what is bounded is integration error alone. Declared
#: here rather than borrowed from E12-05 because these trajectories carry
#: dispersion, and the reference-mission bands of
#: ``orbitopt.verify.rendezvous_refly`` were tuned (with >= 2x margin) on a
#: flight that does not.
INTEGRATION_VS_RK4_TOLERANCES = {
    "position_error_m": 0.10,
    "relative_speed_m_s": 1.0e-3,
    "min_approach_distance_m": 0.50,
    "realized_delta_v_m_s": 0.0,   # burns are commanded, not re-solved: must be exact
}
#: The independent re-fly runs at E12-05's production adaptive-integration
#: settings, so the line measures the model difference and not a loose
#: integrator (``orbitopt.verify.tudat_propagate``).
INTEGRATION_METHOD = "DOP853"


class RobustnessConfigError(ValueError):
    """A robustness campaign declaration is malformed or self-contradictory."""


# --------------------------------------------------------------------------- #
# the disturbance model
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class InitialStateError:
    """Navigation error of the chaser state at t0 in chief-LVLH components
    ``(radial, along-track, cross-track)`` [m] and their rates [m/s]. A zero
    component consumes no randomness."""

    sigma_pos_lvlh_m: tuple[float, float, float]
    sigma_vel_lvlh_m_s: tuple[float, float, float]
    enabled: bool = True

    def __post_init__(self) -> None:
        for name in ("sigma_pos_lvlh_m", "sigma_vel_lvlh_m_s"):
            tri = getattr(self, name)
            if len(tri) != 3 or any(not math.isfinite(v) or v < 0.0 for v in tri):
                raise RobustnessConfigError(
                    f"initial_state.{name} must be three finite non-negative numbers, got {tri}")

    @property
    def sigmas(self) -> tuple[float, ...]:
        return tuple(self.sigma_pos_lvlh_m) + tuple(self.sigma_vel_lvlh_m_s)

    @property
    def n_components(self) -> int:
        return sum(1 for s in self.sigmas if s > 0.0)


@dataclass(frozen=True)
class ExecutionError:
    """Per-burn delta-v magnitude factor ``1 + N(0, sigma_magnitude_rel)`` and
    isotropic pointing error (total misalignment RMS ``sigma_pointing_deg``),
    drawn independently for each of the two burns."""

    sigma_magnitude_rel: float
    sigma_pointing_deg: float
    enabled: bool = True

    def __post_init__(self) -> None:
        for name in ("sigma_magnitude_rel", "sigma_pointing_deg"):
            v = getattr(self, name)
            if not math.isfinite(v) or v < 0.0:
                raise RobustnessConfigError(f"execution.{name} must be finite and >= 0, got {v}")

    @property
    def n_components(self) -> int:
        per_burn = (1 if self.sigma_magnitude_rel > 0.0 else 0) \
            + (3 if self.sigma_pointing_deg > 0.0 else 0)
        return 2 * per_burn


@dataclass(frozen=True)
class J2Error:
    """Multiplicative uncertainty on the zonal coefficient the truth feels; the
    planner keeps the declared nominal, so this prices gravity-model error."""

    sigma_rel: float
    enabled: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.sigma_rel) or self.sigma_rel < 0.0:
            raise RobustnessConfigError(f"j2.sigma_rel must be finite and >= 0, got {self.sigma_rel}")

    @property
    def n_components(self) -> int:
        return 1 if self.sigma_rel > 0.0 else 0


@dataclass(frozen=True)
class DragError:
    """Thermospheric drag the plan ignores: an exponential density profile
    anchored at ``reference_altitude_m``, a log-normal solar-activity factor
    (median 1), and the two vehicles' ballistic coefficients ``Cd*A/m``
    [m^2/kg]. Only the *difference* between the two coefficients bends the
    relative trajectory, so the chief's is not optional bookkeeping: a target
    with an open payload bay presents a very different drag area from a stowed
    chaser, and ``chief_ballistic_coefficient_m2_per_kg = 0`` states a
    drag-free target rather than omitting one."""

    sigma_ln: float
    reference_density_kg_m3: float
    reference_altitude_m: float
    scale_height_m: float
    ballistic_coefficient_m2_per_kg: float
    chief_ballistic_coefficient_m2_per_kg: float = 0.0
    enabled: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.sigma_ln) or self.sigma_ln < 0.0:
            raise RobustnessConfigError(f"drag.sigma_ln must be finite and >= 0, got {self.sigma_ln}")
        if self.reference_density_kg_m3 <= 0.0 or self.scale_height_m <= 0.0:
            raise RobustnessConfigError("drag.reference_density_kg_m3 and scale_height_m must be positive")
        for name in ("ballistic_coefficient", "chief_ballistic_coefficient"):
            v = getattr(self, f"{name}_m2_per_kg")
            if not math.isfinite(v) or v < 0.0:
                raise RobustnessConfigError(f"drag.{name}_m2_per_kg must be finite and >= 0, got {v}")
        if self.enabled and self.ballistic_coefficient_m2_per_kg <= 0.0:
            raise RobustnessConfigError(
                "drag.ballistic_coefficient_m2_per_kg must be positive when drag is enabled; "
                "a chaser with no drag area makes the source inert, so declare it disabled")

    @property
    def atmosphere(self) -> ExponentialAtmosphere:
        """The declared profile on the repo's exponential-atmosphere primitive:
        anchoring at ``h_ref`` is the same column as a sea-level density of
        ``rho_ref * exp(h_ref / H)``."""
        return ExponentialAtmosphere(
            density_sea_level_kg_m3=self.reference_density_kg_m3
            * math.exp(self.reference_altitude_m / self.scale_height_m),
            scale_height_m=self.scale_height_m)

    @property
    def n_components(self) -> int:
        return 1 if self.sigma_ln > 0.0 else 0


@dataclass(frozen=True)
class DisturbanceModel:
    """The four declared families, and the vocabulary the RNG stream and the
    sensitivity reading share."""

    initial_state: InitialStateError
    execution: ExecutionError
    j2: J2Error
    drag: DragError

    def is_active(self, name: str) -> bool:
        src = getattr(self, name)
        return bool(src.enabled) and src.n_components > 0

    def active(self) -> tuple[str, ...]:
        return tuple(n for n in DISTURBANCE_ORDER if self.is_active(n))

    def disabled_except(self, name: str) -> DisturbanceModel:
        """The same declaration with every other family off: the one-at-a-time
        lineup for the marginal sensitivity reading."""
        off = {n: replace(getattr(self, n), enabled=False)
               for n in DISTURBANCE_ORDER if n != name}
        return replace(self, **off)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name in DISTURBANCE_ORDER:
            src = getattr(self, name)
            entry = {"enabled": bool(src.enabled)}
            for k, v in vars(src).items():
                if k != "enabled":
                    entry[k] = list(v) if isinstance(v, tuple) else v
            entry["n_components"] = src.n_components
            out[name] = entry
        return out


@dataclass(frozen=True)
class ArmPlan:
    """One arm's seed base and cluster layout. ``n_clusters`` is derived from the
    campaign's target CI width and never declared beside it, so an arm cannot be
    quietly shrunk below the size the quoted interval needs."""

    seed_base: int
    n_clusters: int
    samples_per_cluster: int

    @property
    def n(self) -> int:
        return self.n_clusters * self.samples_per_cluster

    def to_dict(self) -> dict[str, Any]:
        return {"seed_base": self.seed_base, "n_clusters": self.n_clusters,
                "samples_per_cluster": self.samples_per_cluster, "n": self.n}


@dataclass(frozen=True)
class RobustnessCampaign:
    """Parsed, cross-checked campaign declaration: the disturbance model, the
    two arms, the statistics the run must quote, the truth grid, and the
    spot-check plan."""

    mission_id: str
    disturbances: DisturbanceModel
    train: ArmPlan
    holdout: ArmPlan
    ci_level: float
    target_ci_width: float
    planning_rate: float
    sample_dt_s: float
    grid_convergence_dt_s: float
    grid_convergence_samples: int
    spot_check_samples: int
    cluster_bootstrap_trials: int = 4000
    #: E12-10 (task #412): grade success with the E12-04 passive contingency
    #: releases included -- a sample only succeeds when the uncontrolled coasts
    #: (no burn at t0, gate abort, Hold Point coast) also stay outside the
    #: keep-out sphere on the truth physics. False keeps the E12-07 semantics
    #: (trajectory checks only) so packaged campaigns and old artifacts stand.
    passive_contingency_in_success: bool = False

    @property
    def derived_n(self) -> int:
        """The arm size the declared CI width requires at the worst-case
        planning rate: what each arm's cluster count was rounded up from."""
        return required_sample_size(self.target_ci_width, self.ci_level, self.planning_rate)

    def arm(self, name: str) -> ArmPlan:
        if name not in ("train", "holdout"):
            raise RobustnessConfigError(f"unknown arm {name!r}; the campaign arms are 'train', 'holdout'")
        return getattr(self, name)

    def with_model(self, model: DisturbanceModel) -> RobustnessCampaign:
        return replace(self, disturbances=model)

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": "1.0", "mission_id": self.mission_id,
                "disturbances": self.disturbances.to_dict(),
                "arms": {"train": self.train.to_dict(), "holdout": self.holdout.to_dict()},
                "statistics": {"ci_level": self.ci_level, "target_ci_width": self.target_ci_width,
                               "planning_rate": self.planning_rate,
                               "derived_n_per_arm": self.derived_n,
                                "cluster_bootstrap_trials": self.cluster_bootstrap_trials,
                               "passive_contingency_in_success":
                               bool(self.passive_contingency_in_success)},
                 "grading": {"passive_contingency_in_success":
                             bool(self.passive_contingency_in_success)},
                 "truth_grid": {"sample_dt_s": self.sample_dt_s,
                               "grid_convergence_dt_s": self.grid_convergence_dt_s,
                               "grid_convergence_samples": self.grid_convergence_samples},
                "spot_check_samples": self.spot_check_samples}


def _tri(doc: Mapping, key: str, where: str) -> tuple[float, float, float]:
    raw = doc.get(key)
    if not isinstance(raw, Sequence) or isinstance(raw, str) or len(raw) != 3:
        raise RobustnessConfigError(f"{where}: needs three numbers under {key}, got {raw!r}")
    return tuple(float(v) for v in raw)  # type: ignore[return-value]


def parse_campaign(doc: Mapping[str, Any], *,
                   where: str = "robustness campaign") -> RobustnessCampaign:
    """Validate a campaign document and build a :class:`RobustnessCampaign`.
    Checks what a schema validator cannot express: the two arms' seed bases must
    put them on disjoint RNG streams (that is what "hold-out" means here), the
    cluster layout must deliver at least the sample count the declared CI width
    requires, and the truth grid must be interrogable against a finer one."""
    if not isinstance(doc, Mapping):
        raise RobustnessConfigError(f"{where}: expected a mapping at the top level")
    if str(doc.get("schemaVersion")) != "1.0":
        raise RobustnessConfigError(
            f"{where}: schemaVersion must be '1.0', got {doc.get('schemaVersion')!r}")
    dist_doc, camp_doc = doc.get("disturbances"), doc.get("campaign")
    if not isinstance(dist_doc, Mapping) or not isinstance(camp_doc, Mapping):
        raise RobustnessConfigError(f"{where}: needs 'disturbances' and 'campaign' mappings")

    def sub(name: str) -> Mapping[str, Any]:
        s = dist_doc.get(name, {})
        if not isinstance(s, Mapping):
            raise RobustnessConfigError(f"{where}: disturbances.{name} must be a mapping")
        return s

    def num(s: Mapping[str, Any], key: str, dflt: float) -> float:
        return float(s[key]) if key in s else float(dflt)

    nav, exe, j2d, drg = (sub(n) for n in DISTURBANCE_ORDER)
    disturbances = DisturbanceModel(
        initial_state=InitialStateError(
            sigma_pos_lvlh_m=_tri(nav, "sigma_pos_lvlh_m", where),
            sigma_vel_lvlh_m_s=_tri(nav, "sigma_vel_lvlh_m_s", where),
            enabled=bool(nav.get("enabled", True))),
        execution=ExecutionError(
            sigma_magnitude_rel=num(exe, "sigma_magnitude_rel", 0.0),
            sigma_pointing_deg=num(exe, "sigma_pointing_deg", 0.0),
            enabled=bool(exe.get("enabled", True))),
        j2=J2Error(sigma_rel=num(j2d, "sigma_rel", 0.0), enabled=bool(j2d.get("enabled", True))),
        drag=DragError(
            sigma_ln=num(drg, "sigma_ln", 0.0),
            reference_density_kg_m3=num(drg, "reference_density_kg_m3", 0.0),
            reference_altitude_m=num(drg, "reference_altitude_m", 0.0),
            scale_height_m=num(drg, "scale_height_m", 0.0),
            ballistic_coefficient_m2_per_kg=num(drg, "ballistic_coefficient_m2_per_kg", 0.0),
            chief_ballistic_coefficient_m2_per_kg=num(
                drg, "chief_ballistic_coefficient_m2_per_kg", 0.0),
            enabled=bool(drg.get("enabled", True))))

    ci_level = float(camp_doc.get("ci_level", 0.95))
    width = float(camp_doc.get("target_ci_width", 0.12))
    rate = float(camp_doc.get("planning_rate", 0.5))
    if not 0.0 < ci_level < 1.0 or not 0.0 < rate < 1.0:
        raise RobustnessConfigError(f"{where}: campaign.ci_level and planning_rate must be inside (0, 1)")
    if not 0.0 < width < 2.0:
        raise RobustnessConfigError(f"{where}: campaign.target_ci_width must be inside (0, 2), got {width}")
    dt = float(camp_doc.get("sample_dt_s", 5.0))
    fine_dt = float(camp_doc.get("grid_convergence_dt_s", 1.0))
    if not (0.0 < fine_dt <= dt):
        raise RobustnessConfigError(
            f"{where}: need 0 < grid_convergence_dt_s <= sample_dt_s, got {fine_dt} / {dt}")

    def arm(key: str) -> ArmPlan:
        a = camp_doc.get(key)
        if not isinstance(a, Mapping):
            raise RobustnessConfigError(f"{where}: campaign.{key} must be a mapping")
        if "n_clusters" in a:
            raise RobustnessConfigError(
                f"{where}: campaign.{key}.n_clusters is derived from campaign.target_ci_width, "
                f"not declared; declare samples_per_cluster only")
        seed_base, spc = int(a["seed_base"]), int(a["samples_per_cluster"])
        if seed_base < 0 or spc < 1:
            raise RobustnessConfigError(
                f"{where}: campaign.{key} needs seed_base >= 0 and samples_per_cluster >= 1")
        return ArmPlan(seed_base=seed_base,
                       n_clusters=math.ceil(required_sample_size(width, ci_level, rate) / spc),
                       samples_per_cluster=spc)

    train, holdout = arm("train"), arm("holdout")
    if abs(train.seed_base - holdout.seed_base) < max(train.n, holdout.n) + 1:
        raise RobustnessConfigError(
            f"{where}: train seed_base {train.seed_base} and holdout seed_base {holdout.seed_base} "
            f"are less than max(n)+1 = {max(train.n, holdout.n) + 1} apart, so the arms would draw "
            f"overlapping [seed_base, cluster, index] streams; a hold-out means disjoint seeds")
    n_grid = int(camp_doc.get("grid_convergence_samples", 4))
    if n_grid < 1:
        raise RobustnessConfigError(f"{where}: campaign.grid_convergence_samples must be >= 1")
    spot_doc = camp_doc.get("spot_check", {})
    spot = int(spot_doc.get("n_samples", 5) if isinstance(spot_doc, Mapping) else spot_doc)
    if spot < 1:
        raise RobustnessConfigError(f"{where}: campaign.spot_check.n_samples must be >= 1")
    return RobustnessCampaign(
        mission_id=str(doc.get("mission", {}).get("id", "")),
        disturbances=disturbances, train=train, holdout=holdout,
        ci_level=ci_level, target_ci_width=width, planning_rate=rate,
        sample_dt_s=dt, grid_convergence_dt_s=fine_dt, grid_convergence_samples=n_grid,
        spot_check_samples=spot,
        cluster_bootstrap_trials=int(camp_doc.get("cluster_bootstrap_trials", 4000)),
        passive_contingency_in_success=bool(camp_doc.get("passive_contingency_in_success", False)))


def load_campaign(path: str | Path) -> RobustnessCampaign:
    """Load and validate a campaign declaration from a YAML or JSON file."""
    return parse_campaign(yaml.safe_load(Path(path).read_text(encoding="utf-8")), where=str(path))


def load_reference_campaign() -> RobustnessCampaign:
    """The packaged LEO-RDV-001 campaign, ``orbitopt/data/leo_rdv_001_robustness.yaml``."""
    text = (resources.files("orbitopt") / "data" / _REFERENCE_CAMPAIGN).read_text(encoding="utf-8")
    return parse_campaign(yaml.safe_load(text), where=_REFERENCE_CAMPAIGN)


# --------------------------------------------------------------------------- #
# drawing one sample
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SampleDraws:
    """One sample's draws: the cluster's environment (gravity factor, density
    factor) and the sample's own navigation and execution errors, plus each
    family's draw in sigma-equivalents -- the unit the sensitivity reading
    divides by, computed as R-06 computes it (RMS of the source's per-component
    z-scores)."""

    cluster: int
    index: int
    seed: int
    j2_factor: float
    drag_factor: float
    nav_pos_lvlh_m: tuple[float, float, float]
    nav_vel_lvlh_m_s: tuple[float, float, float]
    burn_factors: tuple[float, float]
    burn_angle_vectors_rad: tuple[tuple[float, float, float], tuple[float, float, float]]
    z: Mapping[str, float | None]

    def to_dict(self) -> dict[str, Any]:
        return {"cluster": self.cluster, "index": self.index, "seed": self.seed,
                "j2_factor": self.j2_factor, "drag_density_factor": self.drag_factor,
                "nav_pos_lvlh_m": list(self.nav_pos_lvlh_m),
                "nav_vel_lvlh_m_s": list(self.nav_vel_lvlh_m_s),
                "burn_magnitude_factors": list(self.burn_factors),
                "burn_angle_vectors_rad": [list(a) for a in self.burn_angle_vectors_rad],
                "sigma_equivalents": {k: v for k, v in self.z.items()}}


def _rms_z(components: Sequence[float]) -> float | None:
    if not components:
        return None
    return float(math.sqrt(sum(c * c for c in components) / len(components)))


def draw_sample(model: DisturbanceModel, seed_base: int, cluster: int, index: int) -> SampleDraws:
    """Draw one sample. The cluster's environment comes from
    ``default_rng([seed_base, cluster])`` and the sample's own errors from
    ``default_rng([seed_base, cluster, index])``, in the fixed order
    ``initial_state`` (position then velocity), then ``execution`` per burn
    (magnitude, then the three rotation-vector components). A disabled or
    zero-width component consumes no randomness, so switching a family off
    leaves every other family's draws untouched (the R-04 stream-layout rule)."""
    env = np.random.default_rng([int(seed_base), int(cluster)]).standard_normal(2)
    j2_factor = float(max(0.0, 1.0 + model.j2.sigma_rel * env[0])) if model.is_active("j2") else 1.0
    drag_factor = float(math.exp(model.drag.sigma_ln * env[1])) if model.is_active("drag") else 1.0

    rng = np.random.default_rng([int(seed_base), int(cluster), int(index)])
    nav = model.initial_state
    pos = tuple(float(s * rng.standard_normal()) if (nav.enabled and s > 0.0) else 0.0
                for s in nav.sigma_pos_lvlh_m)
    vel = tuple(float(s * rng.standard_normal()) if (nav.enabled and s > 0.0) else 0.0
                for s in nav.sigma_vel_lvlh_m_s)

    exe = model.execution
    factors: list[float] = []
    angles: list[tuple[float, float, float]] = []
    z_exe: list[float] = []
    for _burn in range(2):
        if exe.enabled and exe.sigma_magnitude_rel > 0.0:
            mag = float(1.0 + exe.sigma_magnitude_rel * rng.standard_normal())
            z_exe.append((mag - 1.0) / exe.sigma_magnitude_rel)
        else:
            mag = 1.0
        if exe.enabled and exe.sigma_pointing_deg > 0.0:
            sig = math.radians(exe.sigma_pointing_deg) / math.sqrt(3.0)
            ang = tuple(float(sig * rng.standard_normal()) for _ in range(3))
            z_exe.extend(a / sig for a in ang)
        else:
            ang = (0.0, 0.0, 0.0)
        factors.append(mag)
        angles.append(ang)

    z_nav = _rms_z([d / s for d, s in zip(pos, nav.sigma_pos_lvlh_m) if s > 0.0]
                   + [d / s for d, s in zip(vel, nav.sigma_vel_lvlh_m_s) if s > 0.0])
    return SampleDraws(
        cluster=int(cluster), index=int(index),
        seed=int(seed_base) + int(cluster) * 1_000_003 + int(index),
        j2_factor=j2_factor, drag_factor=drag_factor,
        nav_pos_lvlh_m=pos, nav_vel_lvlh_m_s=vel,
        burn_factors=(factors[0], factors[1]), burn_angle_vectors_rad=(angles[0], angles[1]),
        z={"initial_state": z_nav, "execution": _rms_z(z_exe),
           "j2": (j2_factor - 1.0) / model.j2.sigma_rel if model.is_active("j2") else None,
           "drag": math.log(drag_factor) / model.drag.sigma_ln if model.is_active("drag") else None})


# --------------------------------------------------------------------------- #
# the truth model: the same pair physics, with the disturbances in it
# --------------------------------------------------------------------------- #

def _drag_accel(r: np.ndarray, v: np.ndarray, bc_m2_kg: float, atm: ExponentialAtmosphere,
                density_factor: float) -> np.ndarray:
    """Co-rotating-atmosphere drag on one body, ``-0.5 rho |v_rel| v_rel CdA/m``
    with ``v_rel = v - omega x r``: at 500 km the thermosphere co-rotates, so the
    ram speed is not the inertial speed. No drag area or a non-positive density
    factor means no force, exactly as declared."""
    if bc_m2_kg <= 0.0 or density_factor <= 0.0:
        return np.zeros_like(r)
    v_rel = v - np.cross(_OMEGA_Z, r)
    speed = float(np.linalg.norm(v_rel))
    if speed <= 0.0:
        return np.zeros_like(r)
    rho = atm.density_kg_m3(float(np.linalg.norm(r)) - EARTH_RE) * density_factor
    return -0.5 * rho * speed * bc_m2_kg * v_rel


def _truth_deriv(y: np.ndarray, mu: float, j2: float, drag: DragError | None,
                 density_factor: float) -> np.ndarray:
    """Pair derivative in :func:`orbitopt.relative.propagation.propagate_pair`'s
    own state order ``[r_c, r_d, v_c, v_d]``. The two-body and J2 terms are that
    module's ``_accel``, imported not restated: the truth model is meant to
    differ from the planning model by the disturbances, not by a private
    re-implementation of the shared physics."""
    acc_c = _accel(y[0:3], mu, j2)
    acc_d = _accel(y[3:6], mu, j2)
    if drag is not None:
        atm = drag.atmosphere
        acc_c = acc_c + _drag_accel(y[0:3], y[6:9], drag.chief_ballistic_coefficient_m2_per_kg,
                                    atm, density_factor)
        acc_d = acc_d + _drag_accel(y[3:6], y[9:12], drag.ballistic_coefficient_m2_per_kg,
                                    atm, density_factor)
    return np.concatenate((y[6:9], y[9:12], acc_c, acc_d))


def _grid(t_start: float, t_end: float, dt: float) -> np.ndarray:
    """Inclusive sample grid from ``t_start`` to ``t_end``, the discipline
    ``safety._flight_grid`` uses so burn epochs are always sampled."""
    ts = np.arange(t_start, t_end, dt, dtype=float)
    if ts.size == 0:
        return np.array([t_end], dtype=float)
    if ts[-1] != t_end:
        ts = np.append(ts, t_end)
    return ts


def _integrate_pair(y0: np.ndarray, ts: np.ndarray, deriv, *, sub_dt: float,
                    integrator: str) -> np.ndarray:
    """States ``(len(ts), 12)`` on ``ts``, from ``y0`` at ``ts[0]``.

    ``"rk4"`` is the production discipline: classical fixed-step RK4 with steps
    no coarser than ``sub_dt`` -- the orbit/500 default of ``propagate_pair``,
    which is what the plan and its acceptance checks were flown on.
    ``"dop853"`` is the independent-integration line of :func:`spot_check`: the
    same force law under an adaptive integrator at E12-05's production
    tolerances."""
    if str(integrator).lower() == "dop853":
        from orbitopt.verify.tolerances import VERIFY_ATOL, VERIFY_RTOL
        sol = solve_ivp(lambda t, y: deriv(y), (float(ts[0]), float(ts[-1])),
                        np.asarray(y0, dtype=float), method=INTEGRATION_METHOD,
                        t_eval=ts, rtol=VERIFY_RTOL, atol=VERIFY_ATOL)
        if not sol.success or sol.y.shape[1] != ts.size:
            raise RuntimeError(f"robustness: independent re-fly did not integrate: {sol.message}")
        return sol.y.T
    if str(integrator).lower() != "rk4":
        raise ValueError(f"robustness: unknown integrator {integrator!r}; use 'rk4' or 'dop853'")
    out = np.empty((int(ts.size), 12), dtype=float)
    y = np.array(y0, dtype=float)
    out[0] = y
    t_now = float(ts[0])
    for m in range(1, ts.size):
        span = float(ts[m]) - t_now
        n_sub = max(1, math.ceil(span / sub_dt))
        h = span / n_sub
        for _ in range(n_sub):
            k1 = deriv(y)
            k2 = deriv(y + 0.5 * h * k1)
            k3 = deriv(y + 0.5 * h * k2)
            k4 = deriv(y + h * k3)
            y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        out[m] = y
        t_now = float(ts[m])
    return out


def _rotate(vec: np.ndarray, angle_rad: Sequence[float]) -> np.ndarray:
    """Rotate ``vec`` by the rotation vector ``angle_rad`` with the exact
    Rodrigues formula, so a large declared pointing sigma cannot shrink the
    delta-v the way a first-order truncation would."""
    th = np.asarray(angle_rad, dtype=float)
    norm = float(np.linalg.norm(th))
    if norm == 0.0:
        return vec
    axis = th / norm
    c, s = math.cos(norm), math.sin(norm)
    return vec * c + np.cross(axis, vec) * s + axis * float(np.dot(axis, vec)) * (1.0 - c)


def _check_value(check, key: str):
    return check[key] if isinstance(check, Mapping) else getattr(check, key)


def grade_trajectory_checks(cfg: RendezvousMissionConfig, checks: Mapping[str, Any], *,
                            realized_delta_v_m_s: float, planned_delta_v_m_s: float,
                            position_error_m: float, relative_speed_m_s: float,
                            passive: Sequence[PassiveRelease] = (),
                            finite: bool = True) -> dict[str, Any]:
    """The mission's verdict on an already-evaluated flight, as one shared
    function of its parts -- the single place the failure precedence
    (:data:`FAILURE_PRECEDENCE`) is applied, so the Monte Carlo
    (:meth:`FlightOutcome.grade`) and the E12-10 independent re-flight grade the
    same contract on the two models. ``checks`` carries the four
    :data:`TRAJECTORY_CHECKS` (``SafetyCheck``s or plain dicts with ``passed`` /
    ``min_margin`` / ``unit``); ``passive`` the :class:`PassiveRelease` coasts
    (E12-10: a failed passive release is a safety violation -- keep-out broken
    with no thrust to fix it). ``finite=False`` is the ``other`` outcome: a
    trajectory that went non-finite cannot be graded against margins at all."""
    margins = {name: float(_check_value(checks[name], "min_margin")) for name in TRAJECTORY_CHECKS}
    failed = tuple(name for name in TRAJECTORY_CHECKS if not _check_value(checks[name], "passed"))
    bad_passive = tuple(p for p in passive if not p.passed)
    over_budget = realized_delta_v_m_s > cfg.delta_v_budget_m_s
    off_target = (position_error_m > cfg.max_position_error_m
                  or relative_speed_m_s > cfg.max_relative_speed_m_s)
    worst = min(margins.values()) if margins else float("nan")
    if not finite:
        cls, reason = "other", "truth propagation produced a non-finite state"
    elif failed or bad_passive:
        cls = "safety_violation"
        reason = "; ".join(
            [*(f"{n} margin {margins[n]:+.3f} {_check_value(checks[n], 'unit')}" for n in failed),
             *(f"passive release {p.kind}: keep-out margin {p.keep_out_margin_m:+.3f} m"
               for p in bad_passive)])
    elif over_budget:
        cls = "fuel_over_budget"
        reason = (f"executed {realized_delta_v_m_s:.2f} m/s over the fixed budget "
                  f"{cfg.delta_v_budget_m_s:.2f} m/s (planned {planned_delta_v_m_s:.2f})")
    elif off_target:
        cls = "arrival_accuracy"
        reason = (f"arrival {position_error_m:.1f} m / {relative_speed_m_s:.3f} m/s "
                  f"against {cfg.max_position_error_m:.0f} m / "
                  f"{cfg.max_relative_speed_m_s:.1f} m/s")
    else:
        cls, reason = None, (f"arrival {position_error_m:.1f} m / "
                             f"{relative_speed_m_s:.3f} m/s; worst safety margin {worst:+.1f}")
    return {"failure_class": cls, "reason": reason, "margins": margins,
            "failed_checks": [*(failed), *(f"passive_{p.kind}" for p in bad_passive)]}


def grade_flight(cfg: RendezvousMissionConfig, times_s: np.ndarray, rel_lvlh: np.ndarray, *,
                 realized_delta_v_m_s: float, planned_delta_v_m_s: float,
                 passive: Sequence[PassiveRelease] = ()) -> dict[str, Any]:
    """Grade an LVLH trajectory pair against the mission's contract: evaluate
    the four trajectory checks on the samples, then
    :func:`grade_trajectory_checks`. The E12-10 independent model calls this on
    its own re-flown samples -- the same contract the Monte Carlo grades, not a
    re-statement of it."""
    return grade_trajectory_checks(
        cfg, evaluate_trajectory_safety(cfg, times_s, rel_lvlh[:, :3], rel_lvlh[:, 3:]),
        realized_delta_v_m_s=realized_delta_v_m_s, planned_delta_v_m_s=planned_delta_v_m_s,
        position_error_m=float(np.linalg.norm(
            rel_lvlh[-1, :3] - np.asarray(cfg.hold_point_rel_lvlh, dtype=float))),
        relative_speed_m_s=float(np.linalg.norm(rel_lvlh[-1, 3:])),
        passive=passive, finite=bool(np.all(np.isfinite(rel_lvlh))))


@dataclass(frozen=True)
class FlightOutcome:
    """What the truth model says one commanded plan did: the acceptance
    quantities, the flown samples, and what the planner believed, so a report can
    show command-versus-trust without a second run. ``ok=False`` is the
    ``planning_infeasible`` outcome: no flyable, screened candidate for this
    sample's estimate. ``passive`` (E12-10) holds the uncontrolled-contingency
    coasts graded on the same truth physics -- empty unless the campaign grades
    success with them (``passive_contingency_in_success``)."""

    ok: bool
    failure_class: str | None
    reason: str
    position_error_m: float
    relative_speed_m_s: float
    realized_delta_v_m_s: float
    planned_delta_v_m_s: float
    min_range_m: float
    min_range_epoch_s: float
    gate_epoch_s: float
    tof_s: float
    times_s: np.ndarray
    rel_lvlh: np.ndarray
    passive: tuple[PassiveRelease, ...] = ()

    def grade(self, cfg: RendezvousMissionConfig) -> dict[str, Any]:
        """The mission's own verdict on the flown trajectory: the four E12-04
        trajectory checks on the flown samples, plus the config's budget and
        arrival tolerances and -- under E12-10 grading -- the passive
        contingency releases. Returns the governing failure class (``None`` on
        success) by :data:`FAILURE_PRECEDENCE`, a reason string, the per-check
        margins and the list of checks that failed (passive releases appear as
        ``passive_<kind>``)."""
        return grade_trajectory_checks(
            cfg, evaluate_trajectory_safety(cfg, self.times_s, self.rel_lvlh[:, :3],
                                            self.rel_lvlh[:, 3:]),
            realized_delta_v_m_s=self.realized_delta_v_m_s,
            planned_delta_v_m_s=self.planned_delta_v_m_s,
            position_error_m=self.position_error_m,
            relative_speed_m_s=self.relative_speed_m_s,
            passive=self.passive,
            finite=bool(np.all(np.isfinite(self.rel_lvlh))) if self.ok else True)


def _executed_burns(sol: Mapping[str, Any], draws: SampleDraws) -> tuple[np.ndarray, np.ndarray]:
    """The commanded burns after execution error: magnitude factor first, then
    the pointing rotation -- a throttle error scales the vector the gimbal
    mis-points, not the other way round."""
    return (_rotate(np.asarray(sol["dv1"], dtype=float) * draws.burn_factors[0],
                    draws.burn_angle_vectors_rad[0]),
            _rotate(np.asarray(sol["dv2"], dtype=float) * draws.burn_factors[1],
                    draws.burn_angle_vectors_rad[1]))


def _planning_view(cfg: RendezvousMissionConfig, draws: SampleDraws) -> tuple[np.ndarray, ...]:
    """The four t0 states the onboard navigation *believes* it has: the truth
    states of :func:`orbitopt.rendezvous.planner.initial_states`, the chaser's
    moved by the declared navigation error expressed in the chief's LVLH axes at
    t0."""
    r0c, v0c, r0d, v0d = initial_states(cfg)
    dr, ddv = lvlh_to_eci(r0c, v0c, np.asarray(draws.nav_pos_lvlh_m, dtype=float),
                          np.asarray(draws.nav_vel_lvlh_m_s, dtype=float), cfg.dynamics.frame,
                          where="robustness: navigation error")
    return r0c, v0c, r0d + dr, v0d + ddv


def plan_sample(cfg: RendezvousMissionConfig, tof_s: float, draws: SampleDraws) -> dict:
    """One sample's re-plan: re-open the gate on the estimated state, re-target
    burn 1 on the configured nonlinear model at the campaign time-of-flight, and
    re-screen the candidate against the fixed constraints. Raises
    :class:`RendezvousPlanError` with the governing reason when no step yields a
    flyable, screened candidate -- the ``planning_infeasible`` outcome."""
    t1, rt1, vt1, rc1, vc1 = _gate_crossing_at(cfg, *_planning_view(cfg, draws))
    sol = _solve_transfer(cfg, t1, rt1, vt1, rc1, vc1, float(tof_s))
    reason = candidate_safety_reason(cfg, sol)
    if reason:
        raise RendezvousPlanError(
            f"{cfg.mission_id}: the re-planned candidate is not admissible ({reason})")
    return sol


def _truth_passive_releases(cfg: RendezvousMissionConfig, deriv, *, sub_dt: float,
                            integrator: str, dt_s: float, t1: float, tof: float,
                            initial_pair: np.ndarray, gate_pair: np.ndarray,
                            arrival_pair: np.ndarray) -> tuple[PassiveRelease, ...]:
    """The three E12-04 passive releases flown on THIS sample's truth physics
    (E12-10): no burn from t0, gate abort (release at the gate with burn 1
    never fired), and a coast from the Hold Point after insertion. Same
    constraint as ``safety._passive_release`` -- keep-out clear for
    ``passive_contingency_s`` with no thrust -- but integrated with the
    sample's perturbed J2 and drag instead of the nominal plan's forces.
    ``worst_epoch_s`` is seconds since the release, as in the nominal report."""
    s = cfg.safety
    dur = float(s.passive_contingency_s)
    ts = _grid(0.0, dur, float(dt_s))
    frame = cfg.dynamics.frame
    out = []
    for kind, epoch, y0 in (("no_burn_at_t0", 0.0, initial_pair),
                           ("gate_abort_no_burn", t1, gate_pair),
                           ("hold_point_coast", t1 + tof, arrival_pair)):
        st = _integrate_pair(np.asarray(y0, dtype=float), ts, deriv,
                             sub_dt=sub_dt, integrator=integrator)
        rho, _ = eci_to_lvlh(st[:, 0:3], st[:, 6:9], st[:, 3:6] - st[:, 0:3],
                             st[:, 9:12] - st[:, 6:9], frame,
                             where=f"robustness: passive release {kind}")
        rng = np.linalg.norm(rho, axis=1)
        i = int(np.argmin(rng))
        margin = float(rng[i] - s.keep_out_radius_m)
        out.append(PassiveRelease(
            kind=kind, release_epoch_s=float(epoch), duration_s=dur,
            min_range_m=float(rng[i]), worst_epoch_s=float(ts[i]),
            worst_rel_lvlh_m=tuple(float(x) for x in rho[i]),
            keep_out_margin_m=margin, passed=margin >= 0.0))
    return tuple(out)


def fly_truth(cfg: RendezvousMissionConfig, sol: Mapping[str, Any], dv1_exec: np.ndarray,
              dv2_exec: np.ndarray, *, j2_truth: float, drag: DragError | None,
              density_factor: float, dt_s: float, integrator: str = "rk4",
              passive_contingency: bool = False) -> FlightOutcome:
    """Fly one commanded plan on the perturbed truth. Both spacecraft start at
    the config's declared states (the errors that matter are in what the planner
    knew, not in where the vehicles really were); the drift leg coasts to the
    commanded gate epoch, burn 1 fires there, the transfer leg coasts to
    arrival, burn 2 fires, and the post-insertion state is the arrival sample.

    Grading is left to the mission's own checker -- :func:`FlightOutcome.grade`
    calls :func:`evaluate_trajectory_safety` on the flown samples -- so a
    robustness "success" is an E12-04 pass on the trajectory actually flown and
    not a robustness-specific idea of safety.

    ``integrator="dop853"`` re-flies the same force law adaptively: the
    independent-integration line of :func:`spot_check`, not a different physics
    model."""
    drag_on = drag if (drag is not None and drag.enabled) else None
    r0c, v0c, r0d, v0d = initial_states(cfg)
    mu, frame = cfg.dynamics.mu, cfg.dynamics.frame
    period = 2.0 * math.pi * math.sqrt(float(np.linalg.norm(r0c)) ** 3 / mu)
    sub_dt = min(float(dt_s), period / 500.0)   # never coarser than the acceptance model

    def deriv(y: np.ndarray) -> np.ndarray:
        return _truth_deriv(y, mu, j2_truth, drag_on, density_factor)

    t1, tof = float(sol["t1"]), float(sol["tof"])
    grid_a = _grid(0.0, t1, dt_s)
    leg_a = _integrate_pair(np.concatenate((r0c, r0d, v0c, v0d)), grid_a, deriv,
                            sub_dt=sub_dt, integrator=integrator)
    after1 = leg_a[-1].copy()
    after1[9:12] += np.asarray(dv1_exec, dtype=float)          # impulsive burn: a velocity pulse
    grid_b = _grid(0.0, tof, dt_s)
    leg_b = _integrate_pair(after1, grid_b, deriv, sub_dt=sub_dt, integrator=integrator)
    after2 = leg_b[-1].copy()
    after2[9:12] += np.asarray(dv2_exec, dtype=float)          # hover insertion

    passive: tuple[PassiveRelease, ...] = ()
    if passive_contingency and cfg.safety is not None:
        passive = _truth_passive_releases(
            cfg, deriv, sub_dt=sub_dt, integrator=integrator, dt_s=float(dt_s),
            t1=t1, tof=tof, initial_pair=np.concatenate((r0c, r0d, v0c, v0d)),
            gate_pair=leg_a[-1], arrival_pair=after2)

    times = np.concatenate((grid_a, grid_b[1:] + t1, [t1 + tof]))
    states = np.vstack((leg_a, leg_b[1:], after2[None, :]))
    rho, rho_dot = eci_to_lvlh(states[:, 0:3], states[:, 6:9],
                              states[:, 3:6] - states[:, 0:3],
                              states[:, 9:12] - states[:, 6:9], frame,
                              where="robustness: flown rel_lvlh")
    rng = np.linalg.norm(rho, axis=1)
    i = int(np.argmin(rng))
    hold = np.asarray(cfg.hold_point_rel_lvlh, dtype=float)
    return FlightOutcome(
        ok=True, failure_class=None, reason="",
        position_error_m=float(np.linalg.norm(rho[-1] - hold)),
        relative_speed_m_s=float(np.linalg.norm(rho_dot[-1])),
        realized_delta_v_m_s=float(np.linalg.norm(dv1_exec) + np.linalg.norm(dv2_exec)),
        planned_delta_v_m_s=float(sol["total_m_s"]),
        min_range_m=float(rng[i]), min_range_epoch_s=float(times[i]),
        gate_epoch_s=t1, tof_s=tof, times_s=times, rel_lvlh=np.hstack((rho, rho_dot)),
        passive=passive)


def fly_truth_sample(cfg: RendezvousMissionConfig, campaign: RobustnessCampaign,
                     tof_s: float, draws: SampleDraws, *,
                     integrator: str = "rk4") -> FlightOutcome:
    """Plan one sample, fly it on the perturbed truth, hand back the outcome --
    with a ``planning_infeasible`` outcome for a sample that never got a screened
    plan, so the caller grades one homogeneous shape."""
    model = campaign.disturbances
    nan = float("nan")
    try:
        sol = plan_sample(cfg, tof_s, draws)
    except (RendezvousPlanError, ValueError, np.linalg.LinAlgError) as exc:
        return FlightOutcome(
            ok=False, failure_class="planning_infeasible", reason=str(exc)[:400],
            position_error_m=nan, relative_speed_m_s=nan, realized_delta_v_m_s=nan,
            planned_delta_v_m_s=nan, min_range_m=nan, min_range_epoch_s=nan,
            gate_epoch_s=nan, tof_s=float(tof_s), times_s=np.empty(0), rel_lvlh=np.empty((0, 6)))
    dv1, dv2 = _executed_burns(sol, draws)
    return fly_truth(cfg, sol, dv1, dv2,
                     j2_truth=(EARTH_J2 * draws.j2_factor) if model.is_active("j2")
                     else (EARTH_J2 if cfg.dynamics.j2 else 0.0),
                     drag=model.drag if model.is_active("drag") else None,
                     density_factor=draws.drag_factor,
                     dt_s=campaign.sample_dt_s, integrator=integrator,
                     passive_contingency=bool(campaign.passive_contingency_in_success))


@dataclass(frozen=True)
class SampleRecord:
    """One flown sample: its draws, its verdict, and the quantities a reader
    needs to audit that verdict."""

    cluster: int
    index: int
    seed: int
    success: bool
    failure_class: str | None
    reason: str
    position_error_m: float
    relative_speed_m_s: float
    realized_delta_v_m_s: float
    planned_delta_v_m_s: float
    min_range_m: float
    min_range_epoch_s: float
    gate_epoch_s: float
    tof_s: float
    margins: Mapping[str, float]
    failed_checks: tuple[str, ...]
    draws: SampleDraws

    def to_dict(self) -> dict[str, Any]:
        return {"cluster": self.cluster, "index": self.index, "seed": self.seed,
                "success": bool(self.success), "failure_class": self.failure_class,
                "reason": self.reason, "position_error_m": self.position_error_m,
                "relative_speed_m_s": self.relative_speed_m_s,
                "realized_delta_v_m_s": self.realized_delta_v_m_s,
                "planned_delta_v_m_s": self.planned_delta_v_m_s,
                "min_range_m": self.min_range_m, "min_range_epoch_s": self.min_range_epoch_s,
                "gate_epoch_s": self.gate_epoch_s, "tof_s": self.tof_s,
                "margins": dict(self.margins), "failed_checks": list(self.failed_checks),
                "draws": self.draws.to_dict()}


def _fly_one(args: tuple) -> SampleRecord:
    """Worker body: draw, plan, fly, grade. Top-level and pickle-friendly so the
    campaign parallelises over processes with no shared state."""
    cfg, campaign, tof, seed_base, cluster, index = args
    draws = draw_sample(campaign.disturbances, seed_base, cluster, index)
    outcome = fly_truth_sample(cfg, campaign, tof, draws)
    if not outcome.ok:
        return SampleRecord(cluster=cluster, index=index, seed=draws.seed, success=False,
                            failure_class="planning_infeasible", reason=outcome.reason,
                            position_error_m=float("nan"), relative_speed_m_s=float("nan"),
                            realized_delta_v_m_s=float("nan"), planned_delta_v_m_s=float("nan"),
                            min_range_m=float("nan"), min_range_epoch_s=float("nan"),
                            gate_epoch_s=float("nan"), tof_s=outcome.tof_s,
                            margins={}, failed_checks=(), draws=draws)
    grade = outcome.grade(cfg)
    return SampleRecord(cluster=cluster, index=index, seed=draws.seed,
                        success=grade["failure_class"] is None,
                        failure_class=grade["failure_class"], reason=grade["reason"],
                        position_error_m=outcome.position_error_m,
                        relative_speed_m_s=outcome.relative_speed_m_s,
                        realized_delta_v_m_s=outcome.realized_delta_v_m_s,
                        planned_delta_v_m_s=outcome.planned_delta_v_m_s,
                        min_range_m=outcome.min_range_m, min_range_epoch_s=outcome.min_range_epoch_s,
                        gate_epoch_s=outcome.gate_epoch_s, tof_s=outcome.tof_s,
                        margins=grade["margins"], failed_checks=tuple(grade["failed_checks"]),
                        draws=draws)


# --------------------------------------------------------------------------- #
# arm: the samples, the rate, the intervals, the taxonomy, the sensitivity
# --------------------------------------------------------------------------- #

def failure_summary(records: Sequence[SampleRecord]) -> dict[str, Any]:
    """Total-by-construction failure taxonomy: a count for every class in
    :data:`FAILURE_CLASSES` (so a clean arm reports four explicit zeros), plus
    the governing reasons behind each class. Each failed sample is counted
    exactly once, so ``n_failures`` is the sum of the class counts."""
    counts = {cls: 0 for cls in FAILURE_CLASSES}
    reasons: dict[str, list[str]] = {cls: [] for cls in FAILURE_CLASSES}
    n_failed = 0
    for rec in records:
        if rec.success:
            continue
        cls = rec.failure_class if rec.failure_class in counts else "other"
        counts[cls] += 1
        n_failed += 1
        if rec.reason and rec.reason not in reasons[cls]:
            reasons[cls].append(rec.reason)
    for cls, seen in reasons.items():
        seen.sort()
    return {"n_samples": len(records), "n_failures": n_failed,
            "classes": counts, "reasons_by_class": {k: v[:12] for k, v in reasons.items()}}


def sensitivity_reading(records: Sequence[SampleRecord], model: DisturbanceModel, *,
                        threshold_sigma: float = TAIL_SIGMA) -> dict[str, Any]:
    """How much each source's own draw explains failure across the flown
    samples, under joint dispersion -- the R-06 reading
    (``flightlab.recovery.stats.source_sensitivity``: mean |z| overall / among
    failures, the failure rate above and below one sigma-equivalent, Spearman
    rank correlation of |z| against the failure indicator) restated here because
    the source vocabulary is a rendezvous one (``execution``, ``j2``) and not the
    recovery one (``thrust``, ``wind``). It is association, not causation: the
    marginal reading a one-source-at-a-time campaign gives is
    :func:`solo_campaigns`.

    ``state`` is ``inactive_or_bias_only`` for a source that was not drawn,
    ``no_sample_draws`` if its draws were somehow absent, ``uninformative`` when
    the arm did not produce a spread to correlate (all passed or all failed), and
    ``assessed`` otherwise -- a null is reported as a null, never as a zero
    correlation that would read as "measured, found nothing"."""
    z_of = {name: np.array([rec.draws.z.get(name) for rec in records], dtype=float)
            for name in DISTURBANCE_ORDER}
    failed = np.array([not rec.success for rec in records], dtype=float)
    out: dict[str, Any] = {
        "method": "Spearman rank correlation of each sample's |z| (sigma-equivalents "
                  "of its own declared distribution) against failure, under joint "
                  "dispersion; tail split at the declared threshold",
        "tail_threshold_sigma": float(threshold_sigma), "sources": {}, "ranking": []}
    scored: list[tuple[float, str]] = []
    for name in DISTURBANCE_ORDER:
        if not model.is_active(name):
            out["sources"][name] = {"state": "inactive_or_bias_only", "n": 0}
            continue
        z = z_of[name]
        keep = np.isfinite(z)
        entry: dict[str, Any] = {"n": int(keep.sum()),
                                 "components": getattr(model, name).n_components}
        if entry["n"] == 0 or np.all(failed) or not np.any(failed):
            entry["state"] = "no_sample_draws" if entry["n"] == 0 else "uninformative"
            out["sources"][name] = entry
            continue
        zz, yy = z[keep], failed[keep]
        if float(np.ptp(np.abs(zz))) == 0.0:
            entry["state"] = "uninformative"   # constant |z|: nothing to rank, no warning
            out["sources"][name] = entry
            continue
        hit = np.abs(zz) >= float(threshold_sigma)
        entry.update({
            "failure_rate": float(yy.mean()),
            "mean_abs_z": float(np.abs(zz).mean()),
            "mean_abs_z_failed": float(np.abs(zz[yy > 0]).mean()) if np.any(yy > 0) else None,
            "mean_abs_z_passed": float(np.abs(zz[yy == 0]).mean()) if np.any(yy == 0) else None,
            "tail": {"high": {"n": int(hit.sum()),
                              "failure_rate": float(yy[hit].mean()) if np.any(hit) else None},
                     "low": {"n": int((~hit).sum()),
                             "failure_rate": float(yy[~hit].mean()) if np.any(~hit) else None}}})
        rho = float(spearmanr(np.abs(zz), yy).statistic)
        entry["rank_correlation"] = None if not math.isfinite(rho) else rho
        entry["state"] = "assessed" if math.isfinite(rho) else "uninformative"
        out["sources"][name] = entry
        if math.isfinite(rho):
            scored.append((rho, name))
    scored.sort(key=lambda t: (-t[0], t[1]))
    out["ranking"] = [name for _rho, name in scored]
    out["state"] = "assessed" if scored else (
        "all_passed" if not np.any(failed) else "all_failed")
    return out


@dataclass(frozen=True)
class ArmResult:
    """One arm of a campaign: its layout, its samples, the rate with both
    intervals, the taxonomy, and the sensitivity reading. ``summary`` is what a
    report quotes; ``records`` is what an auditor checks it against."""

    name: str
    seed_base: int
    n_clusters: int
    samples_per_cluster: int
    ci_level: float
    summary: Mapping[str, Any]
    failure: Mapping[str, Any]
    sensitivity: Mapping[str, Any]
    cluster_counts: tuple[tuple[int, int], ...]
    records: tuple[SampleRecord, ...]

    @property
    def n(self) -> int:
        return len(self.records)

    @property
    def successes(self) -> int:
        return int(sum(1 for r in self.records if r.success))

    def to_dict(self) -> dict[str, Any]:
        return {"arm": self.name, "seed_base": self.seed_base,
                "n_clusters": self.n_clusters, "samples_per_cluster": self.samples_per_cluster,
                "n": self.n, "summary": dict(self.summary),
                "failure": dict(self.failure), "sensitivity": dict(self.sensitivity),
                "cluster_counts": [list(c) for c in self.cluster_counts]}


def run_campaign_arm(cfg: RendezvousMissionConfig, campaign: RobustnessCampaign,
                     plan: RendezvousPlan, arm: str = "holdout", *,
                     jobs: int = 1, model: DisturbanceModel | None = None) -> ArmResult:
    """Fly one arm: every cluster and sample of it, on the plan's
    time-of-flight, graded against the mission's own contract.

    ``jobs`` parallelises over processes; the samples are re-sorted by
    (cluster, index) afterwards, so the report -- including the bootstrap, which
    consumes the cluster counts in cluster order -- is identical at any worker
    count. ``model`` overrides the campaign's declaration (what
    :func:`solo_campaigns` runs); the arm layout and seeds do not change, so a
    solo arm is comparable to the joint one sample for sample."""
    layout = campaign.arm(arm)
    camp = campaign if model is None else campaign.with_model(model)
    tof = float(plan.tof_s)
    work = [(cfg, camp, tof, layout.seed_base, c, i)
            for c in range(layout.n_clusters) for i in range(layout.samples_per_cluster)]
    if jobs and jobs > 1:
        with ProcessPoolExecutor(max_workers=int(jobs)) as pool:
            flown = list(pool.map(_fly_one, work, chunksize=max(1, len(work) // (4 * int(jobs)))))
    else:
        flown = [_fly_one(a) for a in work]
    flown.sort(key=lambda r: (r.cluster, r.index))

    counts = []
    for c in range(layout.n_clusters):
        rs = [r for r in flown if r.cluster == c]
        counts.append((int(sum(1 for r in rs if r.success)), len(rs)))
    successes = sum(c[0] for c in counts)
    n = len(flown)
    wilson = success_rate_interval(successes, n, campaign.ci_level)
    boot = cluster_bootstrap_rate_ci(counts, trials=campaign.cluster_bootstrap_trials,
                                     seed=int(layout.seed_base) % (2 ** 31),
                                     level=campaign.ci_level)
    rates = sorted(c[0] / c[1] for c in counts if c[1])
    model_used = camp.disturbances
    return ArmResult(
        name=arm, seed_base=layout.seed_base, n_clusters=layout.n_clusters,
        samples_per_cluster=layout.samples_per_cluster, ci_level=campaign.ci_level,
        summary={
            "n": n, "successes": successes, "success_rate": successes / n,
            "ci_level": campaign.ci_level,
            "wilson": wilson, "wilson_width": wilson["upper"] - wilson["lower"],
            "cluster_bootstrap": (None if boot is None else {
                "method": "whole-cluster percentile bootstrap", "lower": boot[0], "upper": boot[1],
                "width": boot[1] - boot[0], "trials": campaign.cluster_bootstrap_trials,
                "resample_unit": "cluster (one gravity-field / space-weather realization)"}),
            "per_cluster_rate": {"min": rates[0], "median": rates[len(rates) // 2], "max": rates[-1]},
            "target_ci_width": campaign.target_ci_width,
            "meets_target_ci_width": bool((wilson["upper"] - wilson["lower"]) <= campaign.target_ci_width
                                          or (boot is not None and (boot[1] - boot[0])
                                              <= campaign.target_ci_width)),
        },
        failure=failure_summary(flown),
        sensitivity=sensitivity_reading(flown, model_used),
        cluster_counts=tuple(counts), records=tuple(flown))


def solo_campaigns(campaign: RobustnessCampaign) -> dict[str, RobustnessCampaign]:
    """One campaign per active source, carrying that source's declaration alone
    (every other family off) on the same arm layout and seeds -- the lineup that
    turns the joint-dispersion association of :func:`sensitivity_reading` into a
    marginal effect. Declaration order preserved."""
    return {name: campaign.with_model(campaign.disturbances.disabled_except(name))
            for name in campaign.disturbances.active()}


# --------------------------------------------------------------------------- #
# grid convergence
# --------------------------------------------------------------------------- #

def grid_convergence(cfg: RendezvousMissionConfig, campaign: RobustnessCampaign,
                     arm: ArmResult, *, model: DisturbanceModel | None = None) -> dict[str, Any]:
    """Does the truth grid's sampling interval decide any verdict? The campaign
    grades on ``sample_dt_s`` (coarser than the config's own
    ``check_sample_dt_s``, because a Monte Carlo cannot afford the plan-check
    grid on every sample), so the risk is a constraint clipped between two
    samples. Re-fly the marginal samples on both grids and report the largest
    change in any graded quantity; a verdict that moves between the two is
    reported as ``grid_dependent``, which is a finding, not a pass."""
    used = model if model is not None else campaign.disturbances
    picked = select_spot_checks(list(arm.records), campaign.grid_convergence_samples)
    rows = []
    for rec in picked:
        camp = campaign.with_model(used)
        fine = replace(camp, sample_dt_s=campaign.grid_convergence_dt_s)
        coarse = fly_truth_sample(cfg, camp, rec.tof_s, rec.draws)
        fine_out = fly_truth_sample(cfg, fine, rec.tof_s, rec.draws)
        row = {"cluster": rec.cluster, "index": rec.index, "seed": rec.seed,
               "dt_coarse_s": campaign.sample_dt_s, "dt_fine_s": campaign.grid_convergence_dt_s}
        if not coarse.ok or not fine_out.ok:
            row["state"] = "planning_infeasible_both_grids" if not (coarse.ok or fine_out.ok) \
                else "plan_exists_on_one_grid_only"
            rows.append(row)
            continue
        g_coarse, g_fine = coarse.grade(cfg), fine_out.grade(cfg)
        row.update({
            "verdict_coarse": g_coarse["failure_class"] or "success",
            "verdict_fine": g_fine["failure_class"] or "success",
            "verdict_agrees": g_coarse["failure_class"] == g_fine["failure_class"],
            "delta_position_error_m": abs(fine_out.position_error_m - coarse.position_error_m),
            "delta_min_range_m": abs(fine_out.min_range_m - coarse.min_range_m),
            "delta_margins": {k: abs(g_fine["margins"][k] - g_coarse["margins"][k])
                              for k in TRAJECTORY_CHECKS}})
        rows.append(row)
    if not rows:
        return {"state": "no_samples", "rows": []}
    agree = all(r.get("verdict_agrees", True) for r in rows)
    worst_pos = max((r.get("delta_position_error_m", 0.0) for r in rows), default=0.0)
    worst_rng = max((r.get("delta_min_range_m", 0.0) for r in rows), default=0.0)
    worst_margin = max((max(r["delta_margins"].values()) for r in rows if "delta_margins" in r),
                       default=0.0)
    return {"state": "grid_independent" if agree else "grid_dependent",
            "n_samples": len(rows), "verdicts_agree": bool(agree),
            "max_abs_delta_position_error_m": float(worst_pos),
            "max_abs_delta_min_range_m": float(worst_rng),
            "max_abs_delta_margin": float(worst_margin),
            "note": "coarser grid than the config's check_sample_dt_s by design; this "
                    "measures whether that interval can change a verdict",
            "rows": rows}


# --------------------------------------------------------------------------- #
# independent spot check of the results that matter
# --------------------------------------------------------------------------- #

def select_spot_checks(records: Sequence[SampleRecord], n: int) -> list[SampleRecord]:
    """Which samples an independent model should re-fly: the ones whose verdict a
    small model difference could move, not a random cut. One per graded gate --
    the tightest sample of each of the four trajectory constraints (each ranked
    in its own unit, because a metre of keep-out margin and a degree of cone
    margin do not add), the success that arrived furthest from the Hold Point and
    the fastest, and one representative of every failure class present (the one
    with the most violated samples, ties by seed) -- deduplicated, in that order,
    truncated to ``n``. A sample with no plan is not re-flyable, so it is
    represented by its class only."""
    if n < 1:
        return []
    flyable = [r for r in records if r.failure_class != "planning_infeasible"]
    picks: list[SampleRecord] = []

    def take(cand: SampleRecord | None) -> None:
        if cand is not None and not any(c is cand for c in picks):
            picks.append(cand)

    for name in TRAJECTORY_CHECKS:
        with_margin = [r for r in flyable if name in r.margins]
        if with_margin:
            take(min(with_margin, key=lambda r: (r.margins[name], r.seed)))
    passed = [r for r in records if r.success]
    if passed:
        take(max(passed, key=lambda r: (r.position_error_m, -r.seed)))
        take(max(passed, key=lambda r: (r.relative_speed_m_s, -r.seed)))
    for cls in FAILURE_CLASSES:
        failed = [r for r in records if r.failure_class == cls]
        if failed:
            take(max(failed, key=lambda r: (len(r.failed_checks), -r.seed)))
    return picks[:int(n)]


def spot_check(cfg: RendezvousMissionConfig, campaign: RobustnessCampaign,
               plan: RendezvousPlan, arm: ArmResult, *, n: int | None = None,
               config_path: str | Path | None = None,
               out_dir: str | Path = "runs/e12-07-spot-check",
               run_backend: bool = True) -> dict[str, Any]:
    """Sample-recheck of an arm's headline results in the E12-05 sense, split
    into the three lines that answer different questions:

    ``independent_integration`` -- each selected sample re-flown with an adaptive
    integrator at E12-05's production tolerances against the production fixed-step
    RK4, on the same force law. This bounds *integration* error on the exact
    trajectories that carry the headline rate, and is measurable in any
    environment. The RK4 re-fly is compared to the stored record first, so a
    reproducibility failure is reported as itself rather than absorbed into the
    model comparison.

    ``independent_formulation`` -- the ground plan replayed on the linear CW model
    (:func:`orbitopt.verify.rendezvous_refly.cw_mission_replay`, which shares only
    the pulses with the planning model), graded against E12-05's *declared
    validity* bands for that approximation. CW is not a candidate truth model for
    a mission with J2 and drag, so a pass here bounds what an approximate model
    could contribute; it is not independent-model confirmation.

    ``independent_model`` -- the TudatPy re-flight, the only line here that is an
    independent gravity/propagation implementation. With no backend in this
    environment that line is ``not_checked`` with
    ``REFLY_BACKEND_MISSING``, exactly as E12-05 records it: recorded, never
    passed. Set ``ORBITOPT_TUDAT_PYTHON`` (or install ``tudatpy``) and pass
    ``config_path`` to run the E12-05 stage itself.

    A band miss says the two models disagree by more than the declared margin on a
    borderline sample, which is a finding about the campaign, not a robustness
    failure of the mission."""
    tol = INTEGRATION_VS_RK4_TOLERANCES
    picked = select_spot_checks(list(arm.records), campaign.spot_check_samples if n is None else n)
    rows: list[dict[str, Any]] = []
    for rec in picked:
        row: dict[str, Any] = {"cluster": rec.cluster, "index": rec.index, "seed": rec.seed,
                              "stored_verdict": "success" if rec.success else rec.failure_class}
        rk4 = fly_truth_sample(cfg, campaign, rec.tof_s, rec.draws)
        if not rk4.ok:
            row["state"] = "planning_infeasible_on_recheck"
            rows.append(row)
            continue
        grade = rk4.grade(cfg)
        row["reproduces_record"] = bool(
            grade["failure_class"] == (None if rec.success else rec.failure_class)
            and abs(rk4.position_error_m - rec.position_error_m) <= 1.0e-9)
        refly = fly_truth_sample(cfg, campaign, rec.tof_s, rec.draws, integrator=INTEGRATION_METHOD)
        if not refly.ok:
            row["state"] = "independent_reflight_did_not_plan"
            rows.append(row)
            continue
        g_ref = refly.grade(cfg)
        obs = {"position_error_m": abs(refly.position_error_m - rk4.position_error_m),
               "relative_speed_m_s": abs(refly.relative_speed_m_s - rk4.relative_speed_m_s),
               "min_approach_distance_m": abs(refly.min_range_m - rk4.min_range_m),
               "realized_delta_v_m_s": abs(refly.realized_delta_v_m_s - rk4.realized_delta_v_m_s)}
        row["integration_residuals"] = {k: {"observed": v, "tolerance": tol[k],
                                            "within": bool(v <= tol[k])} for k, v in obs.items()}
        row["verdict_agrees"] = bool(g_ref["failure_class"] == grade["failure_class"])
        row["state"] = "within_bands" if (
            all(row["integration_residuals"][k]["within"] for k in obs)
            and row["verdict_agrees"] and row["reproduces_record"]) else "outside_bands"
        rows.append(row)

    main_q = main_model_key_quantities(cfg, plan)
    cw_q = cw_mission_replay(cfg, plan)
    # E12-05 names the tolerance after the gate it prices, while the two replay
    # dicts name the quantity; this is the mapping build_refly_report uses.
    _qkey = {"hold_point_position_error_m": "position_error_m"}
    cw_line = {}
    for k, band in CW_VS_NONLINEAR_TOLERANCES.items():
        q = _qkey.get(k, k)
        delta = abs(cw_q[q] - main_q[q])
        cw_line[k] = {"observed": float(delta), "tolerance": float(band),
                      "within": bool(delta <= band)}
    cw_pass = all(v["within"] for v in cw_line.values())

    backend, interpreter = resolve_refly_backend()
    model_line: dict[str, Any] = {"backend": backend,
                                  "interpreter": interpreter,
                                  "stage": "orbitopt.verify.rendezvous_refly"}
    if backend is None:
        model_line.update({"result": "not_checked",
                           "reason": "REFLY_BACKEND_MISSING: no ORBITOPT_TUDAT_PYTHON and no "
                                     "importable tudatpy in this environment -- the "
                                     "independent-model line is recorded as NOT_CHECKED, never "
                                     "passed (the same rule E12-05 applies)"})
    elif not run_backend or config_path is None:
        model_line.update({"result": "not_run",
                           "reason": "a re-flight backend is present but the campaign was asked "
                                     "not to run the E12-05 stage, or without config_path"})
    else:
        from orbitopt.verify.rendezvous_refly import verify_rendezvous_refly
        stage = verify_rendezvous_refly(config_path=Path(config_path), out_dir=Path(out_dir))
        headline = stage["report"]["headline"] if isinstance(stage["report"], Mapping) else None
        model_line.update({"result": headline, "report_path": stage["report_path"]})

    integration_pass = all(r.get("state") == "within_bands" for r in rows if "state" in r)
    return {
        "n_samples": len(rows),
        "selected_by": "tightest sample of each trajectory constraint, worst-arrival "
                       "successes, one representative per failure class",
        "independent_integration": {
            "method": f"{INTEGRATION_METHOD} at the production VERIFY tolerances vs the "
                      f"campaign's fixed-step RK4, same force law",
            "tolerances": dict(tol), "rows": rows,
            "state": "within_bands" if rows and integration_pass else
                     ("no_samples" if not rows else "outside_bands")},
        "independent_formulation": {
            "method": "linear CW replay of the ground plan (shares only the pulse sequence)",
            "residuals": cw_line,
            "state": "within_declared_validity" if cw_pass else "outside_declared_validity",
            "note": "CW is an approximation by design; these are E12-05's declared validity "
                    "bands, so this line bounds what an approximate model could explain"},
        "independent_model": model_line,
    }


# --------------------------------------------------------------------------- #
# campaign
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# E12-10: re-flight sample selection and per-sample flight specifications
# --------------------------------------------------------------------------- #

#: Evidence-document kind of the per-sample re-flight specification that
#: :func:`build_reflight_spec` hands to the independent model, and of the
#: ``campaign`` block it carries for provenance.
SAMPLE_REFLY_SPEC_KIND = "orbitopt.verify.rendezvous-sample-reflight-spec"
SAMPLE_REFLY_SPEC_SCHEMA_VERSION = "1.0"

#: ``failed_checks`` entries for the E12-10 passive contingency releases: the
#: prefix marks them as releases that failed their keep-out coast, distinct from
#: the four trajectory checks (``passive_no_burn_at_t0`` and kin). The prefix is
#: also how a sample that fails *only* under E12-10 grading is recognised.
PASSIVE_CHECK_PREFIX = "passive_"

#: Selection caps (the acceptance criterion: "selection rule and counts written
#: into the benchmark artifact"). Every re-flight is a TudatPy pulse-sequence
#: run, so the caps keep the independent campaign bounded while every *kind* of
#: outcome the hold-out arm produced is represented. The success cap sits above
#: the hold-out's success count (13 on the reference campaign): a success the
#: independent model did not re-fly is a success the qualification wording
#: cannot rest on, and the benchmark reports whether the cap bit.
SAMPLE_REFLY_MAX_TOTAL = 32
SAMPLE_REFLY_MAX_SUCCESSES = 16
SAMPLE_REFLY_MAX_PASSIVE_CAUSED = 8


def select_reflight_samples(cfg: RendezvousMissionConfig, records: Sequence[SampleRecord], *,
                            max_total: int = SAMPLE_REFLY_MAX_TOTAL,
                            max_successes: int = SAMPLE_REFLY_MAX_SUCCESSES,
                            max_passive: int = SAMPLE_REFLY_MAX_PASSIVE_CAUSED,
                            ) -> list[tuple[SampleRecord, str]]:
    """Deterministic per-sample re-flight selection over one arm's records, by
    coverage rule (each pick carries its reason):

    1. every **success** under the campaign's grading (capped, lowest
       ``(cluster, index)`` first) -- the independent model must confirm the
       successes it is asked to qualify;
    2. **passive-contingency failures** -- samples whose ``failed_checks`` carry
       a ``passive_*`` entry, capped, those failing *only* a passive check first:
       these are the verdicts the E12-10 criteria newly create, and they are
       exactly what the independent model has to check;
    3. one **boundary sample per failure class** (``safety_violation`` /
       ``arrival_accuracy`` / ``fuel_over_budget``): the flyable sample of that
       class closest to flipping (smallest distance to its governing threshold)
       -- where a model difference could plausibly move a verdict;
    4. one **planning_infeasible** representative, recorded as *skipped* rather
       than flown: a sample the planner could not fly has no trajectory for the
       independent model to re-fly, and saying so is part of the coverage claim.

    Ties break on ``(cluster, index)``; the function is a pure function of the
    records, so the same arm always yields the same selection."""
    pool = sorted(records, key=lambda r: (r.cluster, r.index))
    chosen: list[tuple[SampleRecord, str]] = []
    seen: set[tuple[int, int]] = set()

    def add(rec: SampleRecord, why: str) -> None:
        key = (rec.cluster, rec.index)
        if key not in seen and len(chosen) < int(max_total):
            seen.add(key)
            chosen.append((rec, why))

    for rec in (r for r in pool if r.success):
        if sum(1 for r, why in chosen if why == "success") >= int(max_successes):
            break
        add(rec, "success")
    def passive_kind(rec: SampleRecord) -> int | None:
        """0 = fails only passive checks, 1 = passive among others, None = none."""
        flags = [c.startswith(PASSIVE_CHECK_PREFIX) for c in rec.failed_checks]
        if rec.success or not any(flags):
            return None
        return 0 if all(flags) else 1

    passive = sorted((r for r in pool if passive_kind(r) is not None),
                     key=lambda r: (passive_kind(r), r.cluster, r.index))
    for rec in passive[:int(max_passive)]:
        add(rec, "passive_contingency_failure")

    def distance(rec: SampleRecord, cls: str) -> float:
        if cls == "safety_violation":
            vals = [abs(float(v)) for v in rec.margins.values() if np.isfinite(v)]
            return min(vals) if vals else float("inf")
        if cls == "arrival_accuracy":
            return min(abs(rec.position_error_m - cfg.max_position_error_m),
                       abs(rec.relative_speed_m_s - cfg.max_relative_speed_m_s))
        return abs(rec.realized_delta_v_m_s - cfg.delta_v_budget_m_s)

    for cls in ("safety_violation", "arrival_accuracy", "fuel_over_budget"):
        cands = [r for r in pool if r.failure_class == cls]
        if cands:
            add(min(cands, key=lambda r: (distance(r, cls), r.cluster, r.index)),
                f"boundary_{cls}")
    infeasible = [r for r in pool if r.failure_class == "planning_infeasible"]
    if infeasible:
        add(infeasible[0], "planning_infeasible")
    return chosen


def reflight_spec_digest(spec: Mapping[str, Any]) -> str:
    """Digest over exactly the flyable payload of a re-flight spec (flights,
    skipped rows, mission): :func:`build_reflight_spec` stamps it, the worker
    re-computes it over what it was handed, and the orchestrator compares --
    so the independent model provably flew the samples this benchmark selected."""
    return hashlib.sha256(json.dumps(
        {"flights": spec.get("flights", []), "skipped": spec.get("skipped", []),
         "mission_id": spec.get("mission_id")},
        sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def build_reflight_spec(cfg: RendezvousMissionConfig, campaign: RobustnessCampaign,
                        records: Sequence[SampleRecord], plan: RendezvousPlan, *,
                        selection: Sequence[tuple[SampleRecord, str]] | None = None,
                        ) -> dict[str, Any]:
    """The per-sample specification the E12-10 independent model flies: each
    selected sample's **own** re-plan (gate epoch and executed burns re-derived
    from its recorded draws, so the burns carry its execution error), its truth
    J2, and its own drag factor on the declared atmosphere -- plus the main
    model's verdict and acceptance quantities for the same sample, so the two
    verdicts can be compared side by side.

    Flyable samples become ``flights``; a ``planning_infeasible`` sample becomes
    a ``skipped`` row (nothing to re-fly). The document self-identifies: mission
    and canonical config digest, the ground plan's digest, the campaign
    declaration it was selected under, and a ``spec_digest`` over the
    flights/skipped rows, so the worker can prove it flew this exact
    specification."""
    tof = float(plan.tof_s)
    sel = select_reflight_samples(cfg, records) if selection is None else list(selection)
    model = campaign.disturbances
    drag = model.drag if model.is_active("drag") else None
    flights: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for rec, why in sel:
        row = {"cluster": int(rec.cluster), "index": int(rec.index), "seed": int(rec.seed),
               "selection": why,
               "main_failure_class": rec.failure_class, "main_reason": str(rec.reason)[:400],
               "main_position_error_m": float(rec.position_error_m),
               "main_relative_speed_m_s": float(rec.relative_speed_m_s),
               "main_min_range_m": float(rec.min_range_m),
               "main_realized_delta_v_m_s": float(rec.realized_delta_v_m_s),
               "main_planned_delta_v_m_s": float(rec.planned_delta_v_m_s),
               "main_margins": {k: float(v) for k, v in rec.margins.items()},
               "main_failed_checks": list(rec.failed_checks)}
        if rec.failure_class == "planning_infeasible":
            skipped.append({**row, "why_skipped":
                            "no screened candidate for this sample's estimate: the independent "
                            "model cannot re-fly an unflyable sample, so its main-model verdict "
                            "stands unconfirmed"})
            continue
        sol = plan_sample(cfg, tof, rec.draws)
        dv1, dv2 = _executed_burns(sol, rec.draws)
        flight = {**row, "t1_s": float(sol["t1"]), "tof_s": tof,
                  "dv1_exec_eci_m_s": [float(x) for x in dv1],
                  "dv2_exec_eci_m_s": [float(x) for x in dv2],
                  "j2_truth": (EARTH_J2 * float(rec.draws.j2_factor)) if model.is_active("j2")
                              else (EARTH_J2 if cfg.dynamics.j2 else 0.0),
                  "drag": None}
        if drag is not None:
            atm = drag.atmosphere
            flight["drag"] = {
                "density_factor": float(rec.draws.drag_factor),
                "chief_ballistic_coefficient_m2_per_kg":
                    float(drag.chief_ballistic_coefficient_m2_per_kg),
                "ballistic_coefficient_m2_per_kg": float(drag.ballistic_coefficient_m2_per_kg),
                "atmosphere": {"surface_density_kg_m3": float(atm.density_sea_level_kg_m3),
                               "scale_height_m": float(atm.scale_height_m),
                               # the sphere _drag_accel measures altitude from
                               "body_radius_m": float(EARTH_RE)}}
        flights.append(flight)
    counts: dict[str, int] = {}
    for _, why in sel:
        counts[why] = counts.get(why, 0) + 1
    doc: dict[str, Any] = {
        "kind": SAMPLE_REFLY_SPEC_KIND,
        "schema_version": SAMPLE_REFLY_SPEC_SCHEMA_VERSION,
        "mission_id": cfg.mission_id,
        "config_sha256": canonical_config_sha256(cfg),
        "plan_digest": rendezvous_plan_digest(cfg, plan),
        "campaign": campaign.to_dict(),
        "selection_rule": {
            "description": ("every success (cap 16); passive-contingency failures, passive-only "
                            "first (cap 8); one "
                            "boundary sample per failure class (smallest distance to its "
                            "governing threshold); one planning_infeasible representative "
                            "recorded as skipped; cap 32; deterministic by (cluster, index)"),
            "max_total": int(SAMPLE_REFLY_MAX_TOTAL),
            "max_successes": int(SAMPLE_REFLY_MAX_SUCCESSES),
            "max_passive_caused": int(SAMPLE_REFLY_MAX_PASSIVE_CAUSED),
            "counts_by_reason": counts,
            "n_selected": len(sel), "n_flyable": len(flights), "n_skipped": len(skipped)},
        "flights": flights,
        "skipped": skipped,
        # The worker evaluates the independent verdict, the safety scan and the
        # min-approach minimum on the campaign's own truth grid: the quantities
        # being compared are grid-located minima, and comparing a finer grid to
        # a coarser one measures the grid, not the models.
        "scan_step_s": float(campaign.sample_dt_s),
    }
    doc["spec_digest"] = reflight_spec_digest(doc)
    return doc


# --------------------------------------------------------------------------- #
# campaign result
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CampaignResult:
    """A whole robustness campaign: the declaration it ran, the ground plan it
    priced, both arms, and the evidence lines. ``holdout`` is the headline arm --
    the train arm exists to show the campaign was not tuned into its own result."""

    campaign: dict[str, Any]
    ground_plan: dict[str, Any]
    arms: Mapping[str, ArmResult]
    cross_check: Mapping[str, Any]
    grid_convergence: Mapping[str, Any]
    solo: Mapping[str, ArmResult]
    provenance: Mapping[str, Any]
    notes: tuple[str, ...]

    @property
    def holdout(self) -> ArmResult:
        return self.arms["holdout"]

    @property
    def train(self) -> ArmResult:
        return self.arms["train"]

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "orbitopt.bench.e12-07-rendezvous-robustness", "schema_version": "1.0",
                "campaign": self.campaign, "ground_plan": self.ground_plan,
                "arms": {k: v.to_dict() for k, v in self.arms.items()},
                "cross_check": dict(self.cross_check),
                "grid_convergence": dict(self.grid_convergence),
                "solo": {k: v.to_dict() for k, v in self.solo.items()},
                "provenance": dict(self.provenance), "notes": list(self.notes)}


def run_campaign(cfg: RendezvousMissionConfig, campaign: RobustnessCampaign,
                 plan: RendezvousPlan | SealedRendezvousPlan,
                 *, arms: Sequence[str] = ("train", "holdout"),
                 jobs: int = 1, model: DisturbanceModel | None = None,
                 solo: bool = False, cross_check: bool = True,
                 grid_check: bool = True, config_path: str | Path | None = None,
                 out_dir: str | Path = "runs/e12-07-spot-check") -> CampaignResult:
    """Fly the campaign. Every arm shares the ground plan's time-of-flight; each
    arm draws on its own seed base. ``jobs > 1`` parallelises inside each arm.

    ``plan`` may be a plain :class:`RendezvousPlan` or a
    :class:`~orbitopt.rendezvous.canonical.SealedRendezvousPlan` (E12-08): a
    sealed plan is digest-verified against ``cfg`` here -- a campaign flying
    a plan sealed for different inputs fails before the first sample -- and
    the ground-plan block of the result records the consumed ``plan_digest``.

    ``solo=True`` also runs the one-source-at-a-time lineup (same layout and
    seeds, one family enabled at a time): the marginal counterpart of the joint
    sensitivity reading, and the reading that separates a source which hurts on
    its own from one that only co-occurs with failure. It multiplies the runtime
    by the number of active families, which is why it is opt-in.
    """
    import platform
    import sys

    import scipy

    consumed_sealed = isinstance(plan, SealedRendezvousPlan)
    if consumed_sealed:
        plan = consume_sealed_plan(plan, cfg)

    flown: dict[str, ArmResult] = {}
    for name in arms:
        flown[name] = run_campaign_arm(cfg, campaign, plan, name, jobs=jobs, model=model)
    headline_arm = flown.get("holdout") or next(iter(flown.values()))

    solo_arms: dict[str, ArmResult] = {}
    if solo:
        for sname, solo_campaign in solo_campaigns(
                campaign if model is None else campaign.with_model(model)).items():
            solo_arms[sname] = run_campaign_arm(cfg, solo_campaign, plan, "holdout", jobs=jobs)

    return CampaignResult(
        campaign=campaign.to_dict(),
        ground_plan={"mission_id": cfg.mission_id, "drift_end_s": plan.drift_end_s,
                     "transfer_tof_s": plan.tof_s,
                     "total_delta_v_m_s": plan.total_delta_v_m_s,
                     "delta_v_budget_m_s": plan.delta_v_budget_m_s,
                     "budget_margin_m_s": plan.budget_margin_m_s,
                     "plan_digest": rendezvous_plan_digest(cfg, plan),
                     "config_sha256": canonical_config_sha256(cfg),
                     "maneuvers_sha256": maneuvers_sha256(plan),
                     "sealed_consumed": bool(consumed_sealed),
                     "planner": "orbitopt.rendezvous.canonical_plan (E12-04 "
                                "plan_rendezvous_safe, constrained optimum)"
                                if consumed_sealed or cfg.safety is not None
                                else "orbitopt.rendezvous.plan_rendezvous"},
        arms=flown,
        cross_check=spot_check(cfg, campaign, plan, headline_arm, config_path=config_path,
                               out_dir=out_dir) if cross_check else {"state": "not_run"},
        grid_convergence=grid_convergence(cfg, campaign, headline_arm, model=model)
        if grid_check else {"state": "not_run"},
        solo=solo_arms,
        provenance={"python": sys.version.split()[0], "numpy": np.__version__,
                    "scipy": scipy.__version__, "platform": platform.platform(),
                    "jobs": int(jobs),
                    "truth_model": ("nonlinear pair, two-body+J2 (perturbed) + co-rotating "
                                    "exponential drag (lognormal density)")},
        notes=(("time-of-flight held at the ground plan's: this prices a locked mission design, "
                "not a per-sample re-optimisation (the latter costs ~9 s/sample for a delta-v "
                "change E12-05 measured as flat across the window)"),
               ("the truth grid is campaign.sample_dt_s, coarser than the config's "
                "check_sample_dt_s by design; grid_convergence measures whether it can move a "
                "verdict"),
               ("drag is present in the truth model whenever the source is enabled: the plan "
                "ignores the atmosphere, and the lognormal factor prices what is unknown about "
                "it"),
               ("the independent-model line is the E12-05 TudatPy re-flight and is recorded "
                "not_checked wherever no backend is available (REFLY_BACKEND_MISSING)")))

