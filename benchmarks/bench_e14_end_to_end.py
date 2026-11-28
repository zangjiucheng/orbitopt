"""E14-10 qualification campaign: plan, freeze, run shards, collect, analyze, report.

The campaign is sharded so that every shard fits a 30-minute job and can be re-run on its own. A shard is one arm x
one instance family x every seed of the split (``cells.*``), one family's fault matrix (``faults.*``), or the
formulation suite (``formulation.*``). Each shard writes one sealed JSON file; ``run`` skips a shard whose file is
already there, sealed, and under the same digests, so a rerun only does what is missing.

    python benchmarks/bench_e14_end_to_end.py plan                       # shards and the digests
    python benchmarks/bench_e14_end_to_end.py freeze                     # once, before any scored shard
    python benchmarks/bench_e14_end_to_end.py run --shard cells.scoring.B.knapsack --out DIR
    python benchmarks/bench_e14_end_to_end.py run --all --split tuning --out DIR
    python benchmarks/bench_e14_end_to_end.py collect --out DIR          # verifies seals, writes the manifest
    python benchmarks/bench_e14_end_to_end.py verify --out DIR           # re-hash every file the manifest names
    python benchmarks/bench_e14_end_to_end.py analyze --out DIR
    python benchmarks/bench_e14_end_to_end.py report --out DIR

The provider arms (C, D, E-LLM and the formulation shard) run when the LLM_* variables configure a provider
(``--env-file`` reads them from a file) and are ``not_run: no provider`` otherwise. Every model call is metered
against the arm's per-cell call cap and the token budget, recorded in the cell's cassette under ``--runs``, and
checked first against the campaign-wide hard stop (``--max-llm-calls``, ``--max-llm-usd``; ``e14_llm``). A rerun
serves every call a cassette already holds, so it never pays for a call twice. ``--instances`` and ``--seeds`` may
narrow a tuning run:

    python benchmarks/bench_e14_end_to_end.py run --all --split tuning --arm D --seeds 1000 --out DIR \
        --env-file .env --max-llm-calls 40 --max-llm-usd 0.50

Scored shards (split ``scoring``) refuse to run without a freeze whose spec, protocol and code digests match.
``analyze`` refuses rows from shards with different digests. ``--mode pipeline-smoke`` labels every row and every
statistic as a pipeline test: mock providers stand in for a model there, and nothing it computes is evidence.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e14_common as C  # noqa: E402
import e14_llm as L  # noqa: E402

P, A, M, F = C.P, C.A, C.M, C.F
SMOKE = "pipeline-smoke"
ARM_CALLS = C.LLM_PROTOCOL["per_cell"]["calls"]
"""The spec's per-cell call cap of each provider arm; every arm also has the per-cell token budget."""
L_ARMS = ("C", "D", "E-LLM")
"""Arms whose shards call a model (the formulation shard is arm D's)."""


def AN():
    """The analysis module, imported only by the commands that need it (it is outside the code digest)."""
    import e14_analysis
    return e14_analysis


# ---------------------------------------------------------------------------------------------------------- plan
def plan(instance_ids: list[str] | None = None, seeds: list[int] | None = None,
         arms: tuple[str, ...] = C.ALL_ARMS) -> list[dict]:
    keep = (lambda i: i.id in instance_ids) if instance_ids else (lambda i: True)
    shards = []
    for split, split_seeds in (("tuning", list(M.TUNING_SEEDS)), ("scoring", list(M.SCORING_SEEDS))):
        for arm in arms:
            for fam in C.families():
                insts = [i.id for i in C.instances(split, fam) if keep(i)]
                if insts:
                    shards.append({"id": f"cells.{split}.{arm}.{fam}", "kind": "cells", "split": split, "arm": arm,
                                   "family": fam, "instances": insts, "seeds": seeds or split_seeds})
    for fam in C.families():
        insts = [i.id for i in C.instances("scoring", fam) if keep(i)]
        if insts:
            shards.append({"id": f"faults.scoring.{fam}", "kind": "faults", "split": "scoring", "arm": None,
                           "family": fam, "instances": insts, "seeds": [0]})
    shards.append({"id": "formulation.scoring", "kind": "formulation", "split": "scoring", "arm": "D",
                   "family": None, "instances": [], "seeds": seeds or list(M.SCORING_SEEDS)})
    return shards


def shard_by_id(shards: list[dict], shard_id: str) -> dict:
    for s in shards:
        if s["id"] == shard_id:
            return s
    raise KeyError(f"no shard {shard_id}; `plan` lists them")


# --------------------------------------------------------------------------------------------- provenance helpers
def provenance(command: list[str]) -> dict:
    import scipy
    ship = C.REPO / ".e14_ship.json"
    shipped = json.loads(ship.read_text()) if ship.exists() else {}
    return {"host": platform.node(), "platform": platform.platform(), "python": platform.python_version(),
            "numpy": np.__version__, "scipy": scipy.__version__, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                                   time.gmtime()),
            "command": command, "git_commit": shipped.get("commit"), "git_dirty": shipped.get("dirty"),
            "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
            "provider": provider_record()}


def provider_record() -> dict | None:
    """The provider configuration of this shard, from the LLM_* variables: never the key, only its variable name."""
    from orbitopt.llm_env import provider_spec
    spec = provider_spec()
    return None if spec is None else {**spec, "temperature": 0.0}


def warm_quadrature() -> None:
    """Build every Gauss-Legendre rule the integration family can ask for, so no cell pays for the cache."""
    for n in range(1, P.N_MAX + 1):
        P._gl(n)


# -------------------------------------------------------------------------------------------------------- arm C
PROPOSER_SCHEMA = {"type": "object", "required": ["points"],
                   "properties": {"points": {"type": "array", "items": {"type": "array", "items": {"type": "number"}},
                                             "minItems": 1, "maxItems": P.MO_MAX_POINTS},
                                  "stop": {"type": "boolean"}}}
PROPOSER_SYSTEM = ("You solve an optimisation problem by proposing candidate points; no solver is available. Reply "
                   "with one JSON object {\"points\": [[x_0, ..., x_{n-1}], ...], \"stop\": false}: at most "
                   f"{P.MO_MAX_POINTS} points per reply, each a list of every variable's value in the order the "
                   "problem states them. Each new point costs one evaluation; after each reply you are told every "
                   "point's objective values and worst constraint violation, and what budget is left. A point that "
                   "violates a bound or a constraint scores as a failure. Set \"stop\": true when you believe your "
                   "best point is optimal. For two objectives, propose points spread along the Pareto front: the "
                   f"last {P.MO_MAX_POINTS} points you propose are scored as the front.")
PROPOSER_MAX_TOKENS = 8192


def run_cell_c(inst, seed: int, provider, mode: str = "scored") -> dict:
    """Arm C (spec section 1): the model proposes points from the statement, with no solver, and reads back each
    point's objective and violation. ``provider`` carries the cell's meter (``e14_llm.cell_meter``), which refuses a
    call past the arm's call cap or the token budget; every call counts, repairs included.

    Scoring uses the catalogue callables on the best evaluated point (feasible first). A cell whose calls passed
    the token budget is a failure, whatever its points: the answer was bought outside the budget."""
    from orbitopt.agent.providers import BudgetExceededError, Message, ProviderError, StructuredOutputError
    budget = A.BUDGET.evaluations(inst.dimension)
    cap = ARM_CALLS["C"]
    counted, counter = C.counted_instance(inst)
    ev = C.Evaluator(counted, budget, A.BUDGET.wall_clock_s)
    messages = [Message("system", PROPOSER_SYSTEM),
                Message("user", f"{inst.statement}\n\nBudget: {cap} replies and {budget} evaluations.")]
    stop, error, front = "completed", None, []
    started = time.perf_counter()
    try:
        while True:
            reply = provider.generate_structured(messages, PROPOSER_SCHEMA, max_tokens=PROPOSER_MAX_TOKENS,
                                                 seed=seed)
            lines = []
            for point in reply.parsed["points"]:
                if len(point) != inst.dimension:
                    lines.append(f"{point}: wrong length, expected {inst.dimension}")
                    continue
                fs, _ = ev.evaluate(point)
                front.append([float(v) for v in point])
                lines.append(f"{point}: objectives {list(fs)}, worst violation {C.violation(inst, point):.3g}")
            if reply.parsed.get("stop"):
                break
            left = cap - provider.meter.calls
            lines.append(f"Budget left: {left} replies, {budget - ev.count} evaluations.")
            messages += [Message("assistant", json.dumps(reply.parsed)), Message("user", "\n".join(lines))]
    except BudgetExceededError as exc:
        stop = f"budget_exhausted:llm_{exc.kind}"
    except C.BudgetExhausted as exc:
        stop = f"budget_exhausted:{exc.kind}"
    except (ProviderError, StructuredOutputError) as exc:
        stop, error = "error", f"{type(exc).__name__}: {exc}"
    wall = time.perf_counter() - started
    llm = L.llm_fields(provider.meter)
    x = ev.best_x()
    mo = len(inst.objectives) > 1
    sc = C.score(inst, x=None if mo else x, front=front[-P.MO_MAX_POINTS:] if mo else None)
    if llm["llm_over_budget"]:
        sc = {**sc, "success": False}
    row = C.base_row("C", inst, seed, mode)
    row.update({"compile_outcome": None, "stop": stop, "error": error, "solvers": [], "messages": [],
                "status": C.outcome_status(sc, stop, x is not None or bool(front)), "x": x,
                "front": front[-P.MO_MAX_POINTS:] if mo else None,
                **{k: sc[k] for k in ("objective", "hypervolume", "max_violation", "gap", "gap_norm", "feasible",
                                      "success", "alarm")},
                "n_evaluations": ev.count, "n_evaluations_independent": counter.calls,
                "within_budget": counter.calls <= budget and llm["llm_calls"] <= cap and not llm["llm_over_budget"],
                "budget_typed": True, "wall_clock_s": wall, **llm, "verification": None})
    row["row_digest"] = C.row_digest(row)
    return row


# ---------------------------------------------------------------------------------------------- smoke providers
def smoke_provider(arm: str, inst):
    """A MockProvider for pipeline-smoke runs of the provider arms: scripted, labelled, never evidence."""
    from orbitopt.agent.providers import MockProvider
    if arm == "C":
        centre = [0.5 * (lo + hi) for lo, hi in inst.bounds]
        return MockProvider([json.dumps({"points": [centre]})] * A.BUDGET.llm_calls_max)
    if arm == "E-LLM":
        return MockProvider([json.dumps({"action": "stop", "rationale": "pipeline smoke"})] * 40)
    import bench_e14_formulation as FM
    replies = next((c["replies"] for c in FM.load_cases() if c.get("reference_problem") == inst.id
                    and c["expected_outcome"] == "compile"), None)
    return MockProvider(list(replies or ["not json"]) * 12)


# ----------------------------------------------------------------------------------------------------- cells
def run_one(arm: str, inst, seed: int, runs_dir: Path, mode: str, llm: L.LLMContext | None = None) -> dict:
    import bench_e14_agent_iteration as E
    import bench_e14_formulation as FM
    import bench_e14_solver_selection as S
    if arm in ("A", "B"):
        row = S.run_cell(arm, inst, seed)
        again = S.run_cell(arm, inst, seed)
        row["mode"] = mode
        row["row_digest"] = C.row_digest(row)
        again["mode"] = mode
        row["replay"] = {"kind": "re-execution from the recorded seed", "equal":
                         C.row_digest(again) == row["row_digest"], "differences": [], "integrity": []}
        row["replay_wall_clock_s"] = again["wall_clock_s"]
        return row
    run_dir = runs_dir / arm / inst.id / f"seed-{seed}"
    if run_dir.exists():
        shutil.rmtree(run_dir)  # a half-written run of an interrupted shard; this shard owns the directory
    if arm == "E-rule":
        return E.run_cell("E-rule", inst, seed, run_dir, mode=mode)
    if mode == SMOKE:
        llm = L.LLMContext(smoke_provider(arm, inst), None)
    if llm is None:
        return FM.not_run_row(arm, inst, seed, mode=mode)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    cassette = run_dir.parent / f"seed-{seed}.cassette.jsonl"
    if mode == SMOKE:
        cassette.unlink(missing_ok=True)  # a mock's replies are not worth resuming
    caps = (ARM_CALLS[arm], A.BUDGET.llm_tokens_max)
    row = provider_cell(arm, inst, seed, llm.cell(cassette, *caps), run_dir, mode)
    return with_replay(row, cassette, lambda provider: provider_cell(
        arm, inst, seed, provider, run_dir.with_name(run_dir.name + "-replay"), mode), caps,
        cleanup=lambda: shutil.rmtree(run_dir, ignore_errors=True))


REEXECUTION_VOLATILE = ("artifact_digest",)
"""Row fields a re-execution may change: an E-LLM run directory's manifest digests files that hold wall-clock
times. The run's ``trace_digest``, which leaves those out, is compared instead."""


def reexecution_digest(row: dict) -> str:
    """The row digest without ``REEXECUTION_VOLATILE``: what a rerun of the cell from its cassette reproduces."""
    return C.row_digest({k: v for k, v in row.items() if k not in REEXECUTION_VOLATILE})


def with_replay(row: dict, cassette: Path, again, caps: tuple[int, int], cleanup=lambda: None) -> dict:
    """Seal ``row`` with its cassette's digest, then rerun the cell with every model call served from that
    cassette (``ReplayProvider``, a fresh meter) and record whether the rerun produced the same row."""
    row["transcript_digest"] = C.file_sha256(cassette) if cassette.exists() else None
    row["cassette"] = cassette.name if cassette.exists() else None
    row["row_digest"] = C.row_digest(row)
    if cassette.exists():
        cleanup()
        started = time.perf_counter()
        other = again(replay_provider(cassette, caps))
        other["transcript_digest"], other["cassette"] = row["transcript_digest"], row["cassette"]
        skip = C.TIMING_FIELDS + REEXECUTION_VOLATILE + ("replay", "row_digest")
        differences = sorted(k for k in set(row) | set(other)
                             if k not in skip and C.jsonable(row.get(k)) != C.jsonable(other.get(k)))
        row["replay"] = {"kind": "re-execution from the recorded cassette", "equal": not differences,
                         "differences": differences, "integrity": []}
        row["replay_wall_clock_s"] = time.perf_counter() - started
    return row


def replay_provider(cassette: Path, caps: tuple[int, int]):
    """The cell's cassette served back with no model call, under a fresh meter with the cell's caps."""
    from orbitopt.agent.providers import ReplayProvider
    return ReplayProvider(cassette, meter=L.cell_meter(*caps))


def provider_cell(arm: str, inst, seed: int, provider, run_dir: Path, mode: str) -> dict:
    import bench_e14_formulation as FM
    try:
        if arm == "C":
            return run_cell_c(inst, seed, provider, mode)
        if arm == "D":
            return FM.run_cell_d(inst, seed, provider, mode)
        return run_cell_e_llm(inst, seed, provider, run_dir, mode)
    except Exception as exc:  # an outage or a provider fault is recorded as the cell's result, never dropped
        row = FM.not_run_row(arm, inst, seed, reason=None, mode=mode)
        row.update(status="error", stop="error", error=f"{type(exc).__name__}: {exc}", success=False,
                   feasible=False, gap="inf", gap_norm="inf", **L.llm_fields(provider.meter))
        row["row_digest"] = C.row_digest(row)
        return row


def run_cell_e_llm(inst, seed: int, provider, run_dir: Path, mode: str) -> dict:
    """Arm E as pre-registered: compile the statement, then the runtime with the LLM planner on the compiled problem.

    The compiled problem replaces the reference for the run only; x is mapped back by variable order and scored
    with the reference callables, and hard-constraint preservation is the probe-point equivalence check. The
    compiler and the planner share the cell's meter: the spec's 20 calls and 200,000 tokens cover both."""
    import bench_e14_agent_iteration as E
    import bench_e14_formulation as FM
    from orbitopt.agent.providers import BudgetExceededError
    try:
        outcome, view = FM.compile_statement(inst.statement, provider, seed, inst.id)
    except BudgetExceededError:
        outcome, view = None, None
    if outcome is None or outcome.outcome != "compile":
        compiled = "budget_exhausted" if outcome is None else outcome.outcome
        row = FM.not_run_row("E-LLM", inst, seed, reason=f"compile outcome {compiled}", mode=mode)
        stop = f"budget_exhausted:llm_{provider.meter.cut}" if outcome is None else "rejected"
        row.update(status="budget_exhausted" if outcome is None else "rejected", stop=stop, compile_outcome=compiled,
                   not_run=None, success=False, within_budget=True, budget_typed=True,
                   **L.llm_fields(provider.meter))
        row["row_digest"] = C.row_digest(row)
        return row
    left = A.BUDGET.llm_tokens_max - provider.meter.usage.total
    row = E.run_cell("E-LLM", inst, seed, run_dir, provider=provider, mode=mode, problem=outcome.problem,
                     tools=view, max_tokens=left)
    llm = L.llm_fields(provider.meter)
    row.update(llm, compile_outcome="compile", hard_constraints=FM.hard_constraints(outcome.problem, inst))
    if llm["llm_cut"] is not None:
        row["stop"] = f"budget_exhausted:llm_{llm['llm_cut']}"
    if llm["llm_over_budget"]:
        row.update(success=False, within_budget=False)
    row["row_digest"] = C.row_digest(row)
    return row


# --------------------------------------------------------------------------------------------- fault matrix
class FaultGrader:
    """The spec's artifact dict mapped onto the guard (as tests/test_e14_06 maps it), graded in memory and sealed."""

    def __init__(self):
        self.ir = {i.id: C.instance_ir(i) for i in P.INSTANCES}
        self.bench_to_ir = {F.problem_digest(i): self.ir[i.id].digest() for i in P.INSTANCES}

    def resolve(self, bench_digest: str) -> str:
        import hashlib
        return self.bench_to_ir.get(bench_digest) or hashlib.sha256(bench_digest.encode()).hexdigest()

    @staticmethod
    def artifact(inst, x, solver: str) -> dict:
        x = [float(v) for v in x]
        f = float(inst.objectives[0](x))
        form = P.neutral_form(inst)
        pd = C.digest(form)
        return F.seal({
            "problem_digest": pd, "x": x, "objective": f,
            "constraint_residuals": [float(c.fn(x)) for c in inst.constraints],
            "statuses": {"solver": "converged", "verification": "verified"},
            "evidence": {"solver": solver, "solved_problem": form, "cache": {"hit": False, "problem_digest": pd},
                         "independent_check": {"verifier": F.INDEPENDENT_VERIFIERS[0], "objective": f,
                                               "max_violation": F.max_violation(inst, x), "verdict": "pass"}}})

    def inputs(self, artifact: dict, inst):
        from orbitopt.optimization import OptimizationResult, Provenance
        from orbitopt.optimization.verification import ReferenceSolution, result_digest
        problem = self.ir[inst.id]
        ev = artifact["evidence"]
        names = [v.name for v in problem.variables]
        converged = artifact["statuses"]["solver"] == "converged"
        verified = artifact["statuses"]["verification"] == "verified"
        objectives = {"f0": artifact["objective"]}
        for k, fn in enumerate(inst.objectives[1:], start=1):
            objectives[f"f{k}"] = float(fn(artifact["x"]))
        result = OptimizationResult(
            problem_digest=self.resolve(artifact["problem_digest"]), solver_id=ev["solver"],
            termination="converged" if converged else "error", solver_converged=converged,
            constraints_satisfied=verified, independently_verified=verified,
            x=dict(zip(names, artifact["x"])), objective_values=objectives,
            constraint_residuals={c.name: r for c, r in zip(problem.constraints, artifact["constraint_residuals"])},
            provenance=Provenance(cache_key=self.resolve(ev["cache"]["problem_digest"]) if ev["cache"]["hit"]
                                  else None))
        evidence = []
        check = ev.get("independent_check")
        if check is not None and inst.reference.indicator == "objective":
            evidence.append(ReferenceSolution(
                id=check["verifier"], producer=check["verifier"], problem_digest=problem.digest(),
                result_digest=result_digest(result), value=inst.reference.value,
                tolerance=max(1e-9, inst.tolerance.objective_abs), source=inst.reference.derivation))
        solved = C.to_ir(dict(ev["solved_problem"], constraint_abs=inst.tolerance.constraint_abs))
        return problem, result, evidence, solved

    def grade(self, artifact: dict, inst, resealed: bool, work: Path) -> dict:
        from orbitopt.optimization.verification import result_digest, seal_run, verify_result, verify_run
        problem, result, evidence, solved = self.inputs(artifact, inst)
        tools = C.tools_for(inst)
        seal = result_digest(result)
        flip = F.tampered_artifact({"evidence": {"artifact_digest": seal}})["evidence"]["artifact_digest"]
        mem = verify_result(problem, result, evidence=evidence, solved_problem=solved,
                            artifact_digest=seal if resealed else flip, tools=tools)
        run_dir = Path(tempfile.mkdtemp(dir=work))
        seal_run(solved, result, evidence, run_dir)
        if not resealed:
            manifest = json.loads((run_dir / "manifest.json").read_text())
            manifest["files"]["result.json"] = F.tampered_artifact(
                {"evidence": {"artifact_digest": manifest["files"]["result.json"]}})["evidence"]["artifact_digest"]
            (run_dir / "manifest.json").write_text(json.dumps(manifest))
        sealed = verify_run(run_dir, problem=problem, tools=tools)
        shutil.rmtree(run_dir)
        return {path: {"verdict": rep.verdict.value, "codes": sorted(c.value for c in rep.codes)}
                for path, rep in (("verify_result", mem), ("verify_run", sealed))}


def fault_params(fault, inst) -> dict:
    other = next(i for i in P.INSTANCES if i.id != inst.id)
    stale = dict(P.neutral_form(inst), objective_exprs=["0"])
    outside = [hi + 1.0 for _, hi in inst.bounds]
    return {"F02": {"x": outside, "objective": inst.objectives[0](outside)},
            "F05": {"other_problem_digest": F.problem_digest(other)},
            "F07": {"expr": "x[0]"},
            "F08": {"previous_problem_digest": F.digest(stale)}}.get(fault.id, {})


def fault_rows(inst, runs_dir: Path, grader: FaultGrader, work: Path) -> list[dict]:
    """Every fault on every host of one instance: each rule arm's seed-0 point, and the stored optimum."""
    hosts = []
    for arm in C.RULE_ARMS:
        row = run_one(arm, inst, 0, runs_dir / "fault-hosts", "scored")
        point = row["x"] if row["x"] is not None else (row["front"] or [None])[0]
        if point is not None:
            hosts.append((arm, point, row["row_digest"]))
    if inst.reference.x is not None:
        hosts.append(("reference", list(inst.reference.x), None))
    out = []
    for host, point, host_digest in hosts:
        honest = grader.artifact(inst, point, f"{host}:solver")
        honest_grade = grader.grade(honest, inst, True, work)
        for fault in F.FAULTS:
            faulty = fault.inject(honest, **fault_params(fault, inst))
            graded = grader.grade(faulty, inst, fault.resealed, work)
            for path, g in graded.items():
                out.append({"instance_id": inst.id, "family": inst.family, "problem_class": inst.problem_class,
                            "host": host, "host_row_digest": host_digest, "fault": fault.id, "path": path,
                            "verdict": g["verdict"], "reason": fault.reason, "reason_named": fault.reason in g["codes"],
                            "codes": g["codes"], "honest_verdict": honest_grade[path]["verdict"]})
    return out


# ------------------------------------------------------------------------------------------------------- run
def shard_path(out: Path, shard_id: str) -> Path:
    return out / "shards" / f"{shard_id}.json"


def current_digests(freeze: dict | None) -> dict:
    return {"spec_digest": C.SPEC.spec_digest(), "protocol_digest": C.protocol_digest(),
            "code_digest": C.code_digest(), "freeze_digest": None if freeze is None else freeze["content_digest"]}


def done(path: Path, digests: dict) -> bool:
    if not path.exists():
        return False
    try:
        doc = json.loads(path.read_text())
    except json.JSONDecodeError:
        return False
    return C.seal_ok(doc) and all(doc.get(k) == v for k, v in digests.items())


def run_shard(shard: dict, out: Path, runs_dir: Path, mode: str, freeze_path: Path, command: list[str],
              llm: L.LLMContext | None = None) -> str:
    """Run one shard and write its sealed file; ``llm`` is the live provider context (None: provider arms are
    ``not_run``). A ``CampaignHardStop`` leaves the shard unwritten; its cassettes let a rerun resume it."""
    freeze = C.check_freeze(freeze_path) if shard["split"] == "scoring" else C.load_freeze(freeze_path)
    if shard["split"] == "scoring" and llm is not None and mode != SMOKE and shard["arm"] in L_ARMS:
        C.check_provider(llm.spec)
    digests = current_digests(freeze)
    path = shard_path(out, shard["id"])
    if done(path, digests):
        return "skipped (already done under these digests)"
    warm_quadrature()
    started = time.perf_counter()
    doc = {"schema": C.SCHEMA, "mode": mode, "shard": shard, **digests, "provenance": provenance(command),
           "rows": [], "faults": [], "formulation": None}
    if shard["kind"] == "cells":
        for inst_id in shard["instances"]:
            inst = P.by_id(inst_id)
            for seed in shard["seeds"]:
                doc["rows"].append(C.jsonable(run_one(shard["arm"], inst, seed, runs_dir / shard["id"], mode,
                                                      llm)))
    elif shard["kind"] == "faults":
        grader = FaultGrader()
        work = Path(tempfile.mkdtemp(prefix="e14-faults-"))
        for inst_id in shard["instances"]:
            doc["faults"] += fault_rows(P.by_id(inst_id), runs_dir / shard["id"], grader, work)
        shutil.rmtree(work, ignore_errors=True)
    else:
        import bench_e14_formulation as FM
        doc["formulation"] = C.jsonable(FM.run_suite(None if mode == SMOKE else llm,
                                                     shard["seeds"][:1] if mode == SMOKE else shard["seeds"], mode,
                                                     runs_dir / shard["id"]))
    doc["wall_s"] = time.perf_counter() - started
    C.write_json_atomic(path, C.sealed(C.jsonable(doc)))
    return f"done in {doc['wall_s']:.1f} s"


# ------------------------------------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "freeze", "run", "collect", "verify", "analyze", "report"):
        p = sub.add_parser(name)
        p.add_argument("--out", type=Path, default=Path("e14-campaign"))
        p.add_argument("--freeze-path", type=Path, default=C.FREEZE_PATH)
        p.add_argument("--mode", choices=("scored", SMOKE), default="scored")
        p.add_argument("--instances", default=None, help="comma-separated instance ids (smoke runs only)")
        p.add_argument("--seeds", default=None, help="comma-separated seeds (smoke runs only)")
        p.add_argument("--json", action="store_true")
        p.add_argument("--env-file", type=Path, default=None,
                       help="read KEY=VALUE lines (the LLM_* variables) into the environment first; a variable "
                            "that is already set wins, and no value is ever printed")
        if name == "run":
            p.add_argument("--shard", action="append", default=[])
            p.add_argument("--all", action="store_true")
            p.add_argument("--split", choices=("tuning", "scoring"), default=None)
            p.add_argument("--kind", choices=("cells", "faults", "formulation"), default=None)
            p.add_argument("--family", default=None, help="only the shards of this instance family")
            p.add_argument("--arm", default=None, help="only the shards of this arm")
            p.add_argument("--runs", type=Path, default=None)
            p.add_argument("--max-llm-calls", type=int, default=None,
                           help=f"campaign-wide hard stop on model calls, over every process writing --out "
                                f"(default ${L.ENV_MAX_CALLS}, else {L.DEFAULT_MAX_CALLS})")
            p.add_argument("--max-llm-usd", type=float, default=None,
                           help=f"campaign-wide hard stop on estimated spend at list price (default "
                                f"${L.ENV_MAX_USD}, else {L.DEFAULT_MAX_USD})")
        if name == "analyze":
            p.add_argument("--allow-smoke", action="store_true")
        if name == "report":
            p.add_argument("--tables", type=Path, default=None)
    args = ap.parse_args(argv)
    if args.env_file is not None:
        from orbitopt.llm_env import load_env_file
        print(f"{args.env_file}: {len(load_env_file(args.env_file))} variables set", file=sys.stderr)
    tuning_run = args.cmd == "run" and args.split == "tuning" and not args.shard
    if (args.instances or args.seeds) and args.mode != SMOKE and not tuning_run:
        ap.error("--instances and --seeds change the pre-registered plan; they are allowed in pipeline-smoke only, "
                 "or on the tuning split (run --all --split tuning)")
    shards = plan(args.instances.split(",") if args.instances else None,
                  [int(s) for s in args.seeds.split(",")] if args.seeds else None)
    if args.cmd == "plan":
        info = {**current_digests(C.load_freeze(args.freeze_path)), "shards": len(shards),
                "cells": sum(len(s["instances"]) * len(s["seeds"]) for s in shards if s["kind"] == "cells")}
        if args.json:
            print(json.dumps({**info, "plan": shards}, indent=1))
        else:
            print(json.dumps(info, indent=1))
            for s in shards:
                print(f"{s['id']:48s} {len(s['instances']):3d} instances x {len(s['seeds']):2d} seeds")
        return 0
    if args.cmd == "freeze":
        existing = C.load_freeze(args.freeze_path)
        record = C.sealed(C.freeze_record())
        if existing is not None:
            same = all(existing[k] == record[k] for k in ("spec_digest", "protocol_digest", "code_digest"))
            print("freeze already present and current" if same else
                  "a different freeze exists; a new freeze means a full re-run: delete it by hand to proceed")
            return 0 if same else 2
        C.write_json_atomic(args.freeze_path, record)
        print(json.dumps({k: record[k] for k in ("spec_digest", "protocol_digest", "code_digest", "content_digest")},
                         indent=1))
        return 0
    if args.cmd == "run":
        chosen = [shard_by_id(shards, s) for s in args.shard] if args.shard else [
            s for s in shards
            if args.all and all(getattr(args, k) in (None, s[k]) for k in ("split", "kind", "family", "arm"))]
        runs = args.runs or args.out / "runs"
        llm = None
        if args.mode != SMOKE and any(s["arm"] in L_ARMS for s in chosen):
            llm = L.context_from_env(args.out, args.max_llm_calls, args.max_llm_usd)
        if llm is not None:
            print(f"provider {llm.spec}; hard stop {llm.ledger.max_calls} calls, ${llm.ledger.max_usd:.2f}; "
                  f"ledger {llm.ledger.path}", flush=True)
        for s in chosen:
            t = time.perf_counter()
            try:
                status = run_shard(s, args.out, runs, args.mode, args.freeze_path, sys.argv, llm)
            except L.CampaignHardStop as exc:
                print(f"{s['id']}: STOPPED, shard not written: {exc}", flush=True)
                print(f"ledger totals {llm.ledger.totals() if llm else None}", flush=True)
                return 3
            print(f"{s['id']}: {status} ({time.perf_counter() - t:.1f} s wall)", flush=True)
        if llm is not None:
            print(f"ledger totals {llm.ledger.totals()}", flush=True)
        return 0
    if args.cmd == "collect":
        print(json.dumps(AN().collect(args.out, shards), indent=1))
        return 0
    if args.cmd == "verify":
        problems = AN().verify(args.out)
        print("\n".join(problems) if problems else "every file matches the manifest")
        return 1 if problems else 0
    if args.cmd == "analyze":
        an = AN().analyze(args.out, allow_smoke=args.allow_smoke)
        print(json.dumps({k: an[k] for k in ("mode", "spec_digest", "arms_run", "arms_not_run", "cells")}, indent=1))
        return 0
    if args.cmd == "report":
        an = json.loads((args.out / "analysis.json").read_text())
        text = AN().render_tables(an)
        target = args.tables or args.out / "tables.md"
        target.write_text(text)
        print(f"wrote {target}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
