"""CMA-ES through the ``cma`` package, imported only when it runs; unavailable when ``cma`` is not installed.

No CMA-ES is vendored. The ``cma`` package treats seed 0 as "seed from the
clock", so the adapter passes ``seed + 1`` to keep every seed deterministic.
"""

from __future__ import annotations

from ..classify import Form
from ..result import Termination
from .base import AdapterSolver, OptionSpec, Outcome, Run, SolverCapability

_CONVERGED = frozenset({"tolfun", "tolfunhist", "tolx"})
_LIMITS = frozenset({"maxiter", "maxfevals"})


class CmaEsSolver(AdapterSolver):
    """Covariance matrix adaptation evolution strategy (``cma.CMAEvolutionStrategy``), bounds only.

    Converged when ``cma`` stops on ``tolfun``, ``tolfunhist`` or ``tolx``;
    its own ``maxiter``/``maxfevals`` are ``MAX_ITERATIONS``; any other stop
    criterion (stagnation, conditioning) is ``STALLED``, with its name in the message.
    ``sigma0`` defaults to 0.3 of the narrowest bound range.
    """

    CAPABILITY = SolverCapability(
        solver_id="cma.cma_es", backend="cma", max_form=Form.BLACK_BOX, convergence="heuristic",
        needs_finite_bounds=True, is_global=True,
        iteration_cap=True,
        options={"sigma0": OptionSpec(float, 1e-12, 1e12), "popsize": OptionSpec(int, 4, 100),
                 "tolfun": OptionSpec(float, 1e-16, 1e-2, tolerance=True, default=1e-11),
                 "tolx": OptionSpec(float, 1e-16, 1e-2, tolerance=True, default=1e-11)})

    def _run(self, run: Run) -> Outcome:
        import cma

        space, ev = run.space, run.evaluator
        sigma0 = run.options.get("sigma0", 0.3 * min(hi - lo for lo, hi in zip(space.lower, space.upper)))
        opts = {"bounds": [list(space.lower), list(space.upper)], "seed": run.seed + 1, "verbose": -9,
                **{k: run.options[k] for k in ("popsize", "tolfun", "tolx") if k in run.options}}
        if run.max_backend_iterations is not None:
            opts["maxiter"] = run.max_backend_iterations
        es = cma.CMAEvolutionStrategy(run.start, sigma0, opts)
        while not es.stop():
            candidates = es.ask()
            es.tell(candidates, [ev.objective(c) for c in candidates])
        reasons = set(es.stop())
        converged = bool(reasons & _CONVERGED)
        stop = None if converged else Termination.MAX_ITERATIONS if reasons & _LIMITS else Termination.STALLED
        return Outcome(converged=converged, stop=stop, message="cma stopped on " + ", ".join(sorted(reasons)),
                       iterations=int(es.countiter))
