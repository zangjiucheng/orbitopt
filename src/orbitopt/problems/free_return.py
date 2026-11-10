"""pygmo UDP for coarse, GPU-batched global search over the translunar-
injection burn vector, screened with orbitopt.dynamics.nbody_gpu's simplified
Earth-Moon(-Sun) propagator to find a burn that brings a spacecraft close to
a specific target distance from the Moon.

This is the "cheap, wide search" half of the two-stage pipeline promised by
the framework's design: a population of candidate burns is batch-evaluated
in one GPU call per generation (same wiring as
orbitopt.problems.transfer_2body, via batch_fitness/optimize.runner), and
the best candidate found here is meant to seed
orbitopt.verify.differential_correction.target_lunar_flyby for precise,
high-fidelity refinement -- NOT to be trusted as a final answer on its own,
since it inherits nbody_gpu's two-body-osculating-ephemeris approximation
(known to be off by ~thousands of km over a several-day coast, see
orbitopt.dynamics.nbody_gpu's module docs).

Units note: orbitopt.dynamics.nbody_gpu works in km/km-s (matching its own
validation/docstrings), while orbitopt.verify.tudat_propagate and
orbitopt.verify.differential_correction work in SI meters/m-s (matching
tudatpy's convention) -- this module's decision vector and objective are in
km/km-s to match nbody_gpu, and the ``as_meters`` helper below is the
explicit conversion point when handing a result to the refinement stage.
"""
from __future__ import annotations

import warnings

import numpy as np

from orbitopt.core import gpu as gpu_mod
from orbitopt.core.problem import BackendUsage, OrbitOptProblem
from orbitopt.dynamics import nbody_fused as nf
from orbitopt.dynamics import nbody_gpu as nb

#: Values accepted by ``FreeReturnScreeningProblem(backend=...)``.
BACKENDS = ("auto", "numpy", "numba-cpu", "cupy")

#: ``backend="auto"`` switches to the fused numba-cpu kernel (when numba is
#: installed) for a batch of at least this many candidates, or for any batch
#: once the kernel is already loaded in this process. Once compiled, the fused
#: kernel is faster even for one candidate (~6 ms vs ~70 ms at 300 steps), but
#: loading it costs ~1 s from numba's on-disk cache and ~5-10 s on a cold
#: cache -- not worth paying for a lone ``fitness()`` /
#: ``time_to_closest_approach()`` call.
AUTO_NUMBA_MIN_BATCH = 64


class FreeReturnScreeningProblem(OrbitOptProblem):
    """Decision vector: [dvx, dvy, dvz] (km/s), the TLI burn added to
    ``v0_pre_burn`` at a fixed departure state/epoch. Objective: absolute
    difference (km) between ``target_distance_km`` and the closest approach
    to the Moon reached during ``coast_days`` of coarse GPU propagation --
    zero means "this burn's coarse trajectory grazes the target distance
    from the Moon," which is what a global optimizer drives toward.

    ``propagation_output`` is passed to ``nbody_gpu.propagate_spacecraft_batch``
    as ``output``: the default ``"closest_approach"`` keeps only O(N) state
    (running Moon closest approach) instead of the O(N x n_steps) history;
    ``"history"`` keeps the original full-history path (same fitness, kept
    for A/B checks).

    ``backend`` picks the propagation kernel:

    - ``"numpy"``     -- vectorized ``nbody_gpu`` path on the CPU.
    - ``"cupy"``      -- the same vectorized path on the GPU (needs CuPy/CUDA).
    - ``"numba-cpu"`` -- fused ``nbody_fused`` kernel, ``numba.njit(parallel=True)``
      (needs numba; ``propagation_output="closest_approach"`` only).
    - ``"auto"`` (default) -- ``"cupy"`` if ``use_gpu=True``; otherwise
      ``"numba-cpu"`` when numba is installed, the output mode is
      ``"closest_approach"`` and the batch has at least
      ``AUTO_NUMBA_MIN_BATCH`` candidates (or the kernel is already loaded);
      else ``"numpy"``. If the numba kernel fails at run time, ``auto``
      warns and falls back to ``"numpy"`` for the rest of the process. ``auto`` never
      picks CuPy on its own: the vectorized path measured ~5x slower on GPU
      than CPU (docs/benchmarks/fused_kernel_evaluation.md).

    All backends agree to floating-point round-off. ``last_backend`` is the
    backend the most recent ``batch_fitness()`` call actually used;
    ``backend_usage`` tallies candidates per backend (shared across pygmo's
    copies of the problem, read by run_optimization()).
    """

    def __init__(
        self,
        r0_km,
        v0_pre_burn_km_s,
        epoch0_seconds,
        target_distance_km,
        coast_days,
        dv_bounds_km_s=(-4.5, 4.5),
        n_steps=400,
        use_gpu=None,
        propagation_output="closest_approach",
        backend="auto",
    ):
        super().__init__(use_gpu=use_gpu)
        self.r0_km = np.asarray(r0_km, dtype=float)
        self.v0_pre_burn_km_s = np.asarray(v0_pre_burn_km_s, dtype=float)
        self.epoch0_seconds = float(epoch0_seconds)
        self.target_distance_km = float(target_distance_km)
        self.coast_seconds = float(coast_days) * 86400.0
        self.dv_bounds_km_s = dv_bounds_km_s
        self.n_steps = n_steps
        if propagation_output not in ("closest_approach", "history"):
            raise ValueError(
                f"propagation_output must be 'closest_approach' or 'history', got {propagation_output!r}"
            )
        self.propagation_output = propagation_output

        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
        if backend == "numba-cpu":
            if propagation_output != "closest_approach":
                raise ValueError(
                    "backend='numba-cpu' only supports propagation_output='closest_approach'"
                )
            if not nf.numba_available():
                raise RuntimeError("backend='numba-cpu' requested but numba is not installed")
        if backend == "cupy" and not gpu_mod.GPU_AVAILABLE:
            raise RuntimeError(
                "backend='cupy' requested but CuPy/CUDA is unavailable in this environment"
            )
        if backend in ("numpy", "numba-cpu") and use_gpu is True:
            raise ValueError(f"backend={backend!r} runs on the CPU; it conflicts with use_gpu=True")
        if backend == "cupy" and use_gpu is False:
            raise ValueError("backend='cupy' conflicts with use_gpu=False")
        self.backend = backend
        self.backend_usage = BackendUsage()

    @property
    def last_backend(self):
        return self.backend_usage.last

    def resolve_backend(self, n):
        """The concrete backend (``"numpy"``, ``"numba-cpu"`` or ``"cupy"``)
        a ``batch_fitness()`` call with ``n`` candidates runs on."""
        if self.backend != "auto":
            return self.backend
        if self.use_gpu is True:
            return "cupy"
        if (
            self.propagation_output == "closest_approach"
            and nf.numba_available()
            and (n >= AUTO_NUMBA_MIN_BATCH or nf.numba_cpu_warm())
        ):
            return "numba-cpu"
        return "numpy"

    def array_module(self):
        """``cupy`` only when this problem's batches really run on CuPy
        (numba-cpu is a CPU backend); used by run_optimization() to report
        the device."""
        if self.resolve_backend(AUTO_NUMBA_MIN_BATCH) == "cupy":
            return gpu_mod.get_array_module(True)
        return np

    def get_bounds(self):
        lo, hi = self.dv_bounds_km_s
        lo = np.broadcast_to(np.asarray(lo, dtype=float), (3,))
        hi = np.broadcast_to(np.asarray(hi, dtype=float), (3,))
        return lo.tolist(), hi.tolist()

    def fitness(self, dv):
        return self.batch_fitness(np.asarray(dv, dtype=float))

    def batch_fitness(self, dvs):
        dv_matrix = np.asarray(dvs, dtype=float).reshape(-1, 3)
        n = len(dv_matrix)

        r0_batch = np.broadcast_to(self.r0_km, (n, 3))
        v0_batch = self.v0_pre_burn_km_s[None, :] + dv_matrix

        backend = self.resolve_backend(n)
        result = None
        if backend == "numba-cpu":
            inputs = nf.prepare_inputs(
                r0_batch, v0_batch, self.coast_seconds, self.n_steps,
                mu_earth=nb.MU_EARTH_KM3_S2, mu_moon=nb.MU_MOON_KM3_S2,
                epoch0=self.epoch0_seconds,
            )
            try:  # only the numba kernel itself may fall back
                result = nf.run_numba_cpu(inputs)
            except Exception as exc:
                if self.backend != "auto":
                    raise
                warnings.warn(
                    f"numba-cpu backend failed ({exc!r}); falling back to numpy",
                    RuntimeWarning, stacklevel=2,
                )
                nf.disable_numba()
                backend = "numpy"
        self.backend_usage.record(backend, n)
        use_gpu = backend == "cupy"

        if result is None:
            result = nb.propagate_spacecraft_batch(
                r0_batch, v0_batch, self.coast_seconds, self.n_steps,
                mu_earth=nb.MU_EARTH_KM3_S2, mu_moon=nb.MU_MOON_KM3_S2,
                epoch0=self.epoch0_seconds, use_gpu=use_gpu,
                output=self.propagation_output,
            )

        if self.propagation_output == "closest_approach":
            self._last_min_distance_epoch_s = result.min_distance_epoch_s
            return np.abs(result.min_distance_km - self.target_distance_km)

        moon_r, _ = nb.moon_state_batch(
            result.t_abs.ravel() - self.epoch0_seconds, use_gpu=use_gpu,
            seed_epoch_ephemeris_seconds=self.epoch0_seconds,
        )
        moon_r = moon_r.reshape(result.t_abs.shape[0], n, 3)

        distance_km = np.linalg.norm(result.r - moon_r, axis=-1)
        min_distance_km = distance_km.min(axis=0)
        min_distance_epoch_s = result.t_abs[np.argmin(distance_km, axis=0), np.arange(n)]

        self._last_min_distance_epoch_s = min_distance_epoch_s
        return np.abs(min_distance_km - self.target_distance_km)

    def time_to_closest_approach(self, dv):
        """Convenience for callers (e.g. the free_return example) that want
        a coast-duration guess for the winning candidate: runs one more
        batch_fitness of size 1 and returns the elapsed seconds (from
        epoch0) to the coarse model's closest Moon approach, since
        batch_fitness doesn't return it directly (pygmo's UDP fitness
        contract is a plain float vector, not a richer struct).
        """
        self.batch_fitness(np.asarray(dv, dtype=float))
        return float(self._last_min_distance_epoch_s[0] - self.epoch0_seconds)


def as_meters(r_or_v_km):
    """Explicit km -> m unit conversion at the handoff from this module's
    (nbody_gpu-matching) km convention to tudat_propagate/
    differential_correction's SI-meters convention.
    """
    return np.asarray(r_or_v_km, dtype=float) * 1000.0
