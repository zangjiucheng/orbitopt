# Experiment memory

`orbitopt.agent.experiments` keeps a database of saved
[runtime](runtime.md) runs. It records each run's configuration, its planner
and provider, its seeds and solver steps, the best objective, the largest
constraint violation, the four verification statuses, and what the run cost
in evaluations, wall clock and tokens. The database can resume, query, cache
and compare runs. It is an SQLite index beside a content-addressed JSON
store, and it uses the standard library, the IR and the runtime. It imports
no LLM SDK.

```python
from orbitopt.agent.experiments import ExperimentDB, compare, lookup, query, record_run, resume_plan
from orbitopt.agent.runtime import RuleBasedPlanner, save_run, solve

digest = save_run(solve(problem, planner=RuleBasedPlanner(), seed=0), "runs/a")
with ExperimentDB("experiments") as db:
    record = record_run(db, "runs/a", expected_manifest_digest=digest)
    record.status, record.objective, record.statuses
    query(db, problem_digest=problem.digest(), verified=True).to_text()
    step = record.best_step
    found = lookup(db, problem, solver_id=step.solver_id, options=step.options, seed=step.seed, x0=step.x0,
                   evaluations=step.evaluations)
    found.hit, found.evaluations          # True, 0
```

## The database

`ExperimentDB(root)` opens or creates the directory `root`.

| Path | Content |
|---|---|
| `index.sqlite` | One row per record, its solver steps and its cache keys. |
| `artifacts/sha256/<2 hex>/<64 hex>.json` | Every run file, stored under the SHA-256 of its exact bytes. |

The index runs in WAL mode with a 30 s busy timeout. Every write is one
`BEGIN IMMEDIATE` transaction, so two processes that write at once wait for
each other. Every statement takes its values as parameters. A sort column
comes from a fixed list.

`ArtifactStore.put` writes a blob to a temporary file in its directory and
renames it into place. A reader never sees half a blob, and two writers of
the same bytes end with one file. `get` hashes what it reads again and
raises `ArtifactError` when the bytes changed. The store accepts UTF-8 JSON
without NaN or infinity, and it parses with `json.loads` only. It never
unpickles, evaluates or imports stored data.

Use one `ExperimentDB` per process or thread.

## Records

`ExperimentRecord.to_dict()` validates against `agent-experiment-1.0.json`.
`from_dict` checks the schema first and raises `MalformedRecordError`.

| Field | Meaning |
|---|---|
| `run_id` | The `sha256:` digest of the run's `manifest.json`, the value `save_run` returns. |
| `problem_digest`, `fingerprint`, `trace_digest`, `result_digest` | The problem's IR digest and hard-constraint fingerprint, the trace digest, and `result_digest` of the best result. |
| `sealed_digest` | The manifest digest of the sealed run (`orbitopt.optimization.run/2`). |
| `problem_id`, `family` | The family is the `family` argument, else `problem.metadata["family"]`, else the problem id. |
| `planner`, `provider`, `model` | The planner name from the trace. The trace does not hold the provider or the model, so `record_run` takes them as arguments. |
| `seed`, `steps` | The run seed and every solver step: each planner decision (`iteration-0003`) and each final-verification alternate (`verify-1-scipy.slsqp`), with its solver, options, seed, start point and evaluation allowance. |
| `best`, `objective`, `sense`, `max_violation` | The best result's label, its first objective in the declared sense, and its largest hard-constraint violation (0 when the problem has none). |
| `statuses` | The final report's four statuses, or the best candidate's screen when no final verification ran. |
| `status`, `stop_reason`, `limits`, `usage` | As in the trace. `usage` holds iterations, evaluations, tokens and wall seconds. |
| `created_at` | When the run was recorded, UTC, ISO 8601 with microseconds. |
| `orbitopt_version` | The version that replayed the run at ingest. |
| `integrity` | `checked` (the default ingest), `replayed`, or `iterations_replayed` for a cancelled run that was replayed. See below. |
| `files` | Each run file by its path in the run directory, to its digest in the store. |
| `parent_id`, `resumed_from` | Links to recorded runs, or `None`. |

## Recording a run

`record_run(db, run_dir, *, expected_manifest_digest=None, verify="check", tools=None, registry=None, provider=None, model=None, family=None, parent_id=None, resumed_from=None)`
trusts nothing in `run_dir`:

1. It reads `manifest.json`. Its digest is the run id. With
   `expected_manifest_digest`, a different digest is refused.
2. A run already recorded returns its first record, and nothing is written.
3. Every file the manifest names, and every file of the sealed run, must
   have a safe name and hash to its recorded digest. A name with `..`, an
   absolute path, a backslash or any character outside `[A-Za-z0-9_.-]`
   raises `UnsafePathError` before any of them is read. The check is the
   runtime's `safe_relative`, the one the runtime's `replay` uses.
4. With `verify="check"`, the default, the runtime's
   [`check_run`](runtime.md#run-directories-and-replay) runs, pinned to the
   manifest bytes read in step 1. It checks the trace against its digest
   and against the saved results, checks that the sealed result is the
   trace's best, and has the guard re-grade the sealed run with the
   alternates' recorded results. No solver runs. Any finding in
   `CheckReport.integrity` refuses the run with `RunIntegrityError`. The
   record's `integrity` is `checked`.
5. With `verify="replay"`, the runtime's `replay` runs instead. It does the
   same check with the alternates run again, then re-executes the recorded
   proposals. Any finding and any difference refuse the run. A forger who
   rewrites any result and every digest fails here, because the solvers
   recompute the result. The record's `integrity` is `replayed`.
6. The record's numbers come from the checked trace, and the store gets the
   exact bytes the check read. The files go into the store, and then the
   record goes into the index in one transaction. A crash between the two
   leaves unreferenced blobs, which the next ingest reuses.

Use the default to record runs cheaply. Replay a run whose steps you want
to serve from the cache, since only a `replayed` record holds cache
entries (see below).

With `verify="replay"`, a cancelled run cannot be re-executed, because replay has no cancel signal:
it asks the recorded planner for one more proposal, and the recording has
none. Such a run is recorded with `integrity` `iterations_replayed` when
every iteration matched and the differences are confined to the outcome,
the usage and the final verification, which the cancelled run skipped. Its
outcome rests on the manifest, so keep the digest that `save_run` returns
and pass it. A run that the wall clock ended is refused unless replay
reproduces it exactly.

Wall-clock fields are not part of the trace digest and replay does not
compare them. Without `expected_manifest_digest`, whoever can rewrite the
directory can also rewrite `usage.wall_s`.

`export_run(db, run_id, dest)` writes a recorded run back out of the store,
each file through its digest check. `replay` then runs on `dest`.

## The cache

A cache entry is the step that produced the best result of a verified,
fully replayed run. A `checked` record holds no entry: its key would say
that a solver with these options and this seed produces this result, and
only re-execution shows that. Its key is
`cache_key(problem_digest, solver_id, options, seed, x0, evaluations, *, version, policy)`:

- Options and start points are normalised. Keys are sorted, `None` values
  are dropped, every number is a float, and an empty mapping equals `None`.
  Numbers count at the IR digest's 12 significant digits.
- `evaluations` is the step's allowance.
- `VersionPolicy` reads the orbitopt version: `exact` (the default), `minor`
  (`major.minor`), or `any`. A record is entered under all three, so a
  lookup chooses the policy.

`lookup(db, problem, *, solver_id, options=None, seed=0, x0=None, evaluations, version_policy=VersionPolicy.EXACT, version=None, tools=None)`
walks the entries under the key, newest first, and returns a `CacheLookup`.
An entry is served only when all of these hold:

- Its record still parses and is verified and `replayed`.
- Every file of its sealed run still hashes to its digest.
- It answers `problem`, by its record and by its stored problem.
- The guard re-grades the stored sealed run against `problem` with
  `verify_run`, pinned to the recorded manifest digest. The verdict passes
  with `constraints_satisfied` required, so an objective or residual that no
  longer re-evaluates, a domain violation or a stale cache fails it.

Every other entry is named in `CacheLookup.ignored`, as `corrupt` (a missing
or changed artifact, or an unreadable record) or `stale` (intact, but it no
longer verifies for this problem, for example because a tool now answers
differently). It is never returned.

A hit is marked twice. `CacheLookup.hit` is true, and the result's
`Provenance.cache_key` is the problem digest, which the guard checks against
`stale_cache`. A hit costs no new solver evaluations, and
`CacheLookup.evaluations` is always 0. That follows the runtime's meter, for
which a documented cache hit is free. `recorded_evaluations` is what the
original run spent. The guard's re-grade evaluates the stored point once
more, and `check_evaluations` counts it. On the tests' problems that is one
evaluation per entry checked, confirmed by an independent counter that every
evaluation calls.

## Resuming

`resume_plan(db, run_id)` returns a `ResumePlan` with the problem (read from
the store), the run seed, its policy, the recorded proposals in order, the
limits, what was used, and `remaining`, which is `limits` minus `used` and
never below zero. `resumable` is true when a limit, a cancellation or a
failed planner ended the run. The runtime is not changed. To pick a run up,
serve `plan.proposals` with a `RecordedPlanner` under `plan.limits`,
`plan.policy` and `plan.seed`, which reproduces the completed iterations.
Then hand over to a live planner with the remaining limits.

## Query, compare, aggregate

`query(db, *, problem_digest=None, solver_id=None, status=None, verified=None, family=None, seed=None, since=None, until=None, order_by="created_at", descending=True, limit=50, offset=0)`
returns a `Page` with the records and `total`, the number of matches.
`solver_id` matches a run that ran that solver at any step. `seed` is the
run seed. `since` and `until` bound `created_at` inclusively, and a naive
datetime is UTC. `order_by` is one of `ORDER_BY` (`created_at`, `objective`,
`max_violation`, `evaluations`, `wall_s`, `tokens`). Missing values sort
last, and the run id breaks ties. `limit` is 1 to 1000.

`compare(db, run_ids)` puts runs side by side: objective, the best objective,
`gap` (how much worse than the best, in the objective's sense),
`relative_gap` (`gap / max(1, |best|)`, the guard's objective scale),
evaluations, wall seconds, tokens and the four statuses. The best is taken
per problem digest, among the runs whose `constraints_satisfied` passed. An
infeasible run with a better objective therefore gets a negative gap and
never becomes the best.

`aggregate(db)` gives, per family, the number of runs, of `verified` and of
`unverified_best` runs, the rest, and the median evaluations, wall seconds
and tokens.

`Page`, `Comparison`, `Aggregate` and `ResumePlan` each have `to_dict()`
(JSON-ready) and `to_text()` (a plain table).

## Migrations

`schema_version` holds one integer. `MIGRATIONS[i]` takes the index from
version `i` to `i + 1`. A new file starts at version 0, so creating the
tables is migration 0. Opening a database runs the missing steps in one
transaction, so a step that raises leaves the version where it was.
`ExperimentDB(root, migrations=...)` is the hook for a later step. A
database newer than the steps the code knows, or an SQLite file that is not
an experiment database, raises `SchemaVersionError`.

## Errors

Every error is an `ExperimentError`, except `UnsafePathError`, which the
runtime defines and raises for both packages. Branch on the class.

| Error | When |
|---|---|
| `RunIntegrityError` | A run directory was changed, forged, does not replay, or is not a run. `problems` lists the findings. |
| `UnsafePathError` | A file name would leave its directory. |
| `InvalidIdError` | An id is not `sha256:` and 64 lowercase hex digits. |
| `UnknownRunError` | No record has the id, or a link names no recorded run. |
| `ArtifactError` | A stored blob is missing, changed, or not JSON. |
| `MalformedRecordError` | A record does not match its schema. |
| `SchemaVersionError` | The database is newer than the code, or is not an experiment database. |

## Measured

The rule-based planner ran each instance with seeds 0 to 4, and each run
was recorded, then looked up with its best step (a hit) and with another
seed (a miss). The table gives medians over the five seeds. It was measured
on one machine (Apple M1 Pro, macOS, Python 3.12, the `orbitopt-py312`
environment). Every run was verified, and every hit was served.

| Instance | Run evaluations | `solve` | `replay` alone | Hit | Miss | Evaluations per hit |
|---|---|---|---|---|---|---|
| cqp-worked-d2 | 807 | 53 ms | 107 ms | 5.2 ms | 0.14 ms | 1 |
| lp-wyndor-d2 | 419 | 37 ms | 73 ms | 5.7 ms | 0.17 ms | 1 |
| rastrigin-d5 | 7848 | 343 ms | 444 ms | 7.4 ms | 0.18 ms | 1 |

`record_run` was timed again in each mode: five seeds, each run ingested
into a fresh database three times, 15 timings per cell, on the same
machine. The table gives the median with the range.

| Instance | Run evaluations | `solve` | `verify="check"` | `verify="replay"` | Evaluations at a checked ingest |
|---|---|---|---|---|---|
| cqp-worked-d2 | 807 | 52 ms | 8.6 ms (7.8 to 11.6) | 111 ms (95 to 115) | 3 |
| lp-wyndor-d2 | 419 | 35 ms | 10.0 ms (9.1 to 10.4) | 75 ms (71 to 80) | 3 |
| rastrigin-d5 | 7848 | 305 ms | 13.9 ms (13.0 to 16.5) | 401 ms (382 to 455) | 3 |

What limits these numbers:

- A replayed ingest runs every solve again and then the final
  verification's alternates a second time, when it re-grades the sealed
  run. It costs more than the run itself and grows with it. It took 112,
  77 and 397 ms before the default changed, the same within the spread.
- A checked ingest does not depend on the run's evaluations. It evaluates
  the problem three times: the sealed point and the two alternates'
  points. A profile of the Rastrigin ingest puts half of its time in
  `jsonschema` (the trace, the problem twice, the result and the record),
  an eighth in writing the 13 files to the store with an `fsync` each, and
  a tenth in the guard. Rastrigin's trace has five iterations and the others
  two, which is the difference between the rows.
- A hit does not depend on the run's size. It writes the four sealed files
  to a temporary directory and runs the guard once. A profile puts most of
  that time in `jsonschema`, which checks the stored problem twice and the
  record once. The guard evaluates the stored point once.
- A miss is the problem digest (about 0.09 ms) and one indexed SQLite read.
  Before the installed version was cached, reading it from the package
  metadata took 1.7 ms of the 1.9 ms a miss cost.
- A query costs about 0.25 ms for each record it returns, almost all of it
  the schema check of each record read from the index.

## Tests

`tests/test_e14_07_experiment_memory.py` drives the package through its
public API on real runs: the rule-based, recorded and mock-LLM planners on
`cqp-worked-d2`, `lp-wyndor-d2` and `rastrigin-d5` from the E14-10
catalogue, and a tool-evaluated problem whose tool counts every evaluation.
It covers the round trip, idempotent ingest, six tampered files, a forger
who rewrites the manifest, a manifest that names a file outside the run,
cache hits, misses, version policies, corrupt and stale entries and the
zero-evaluation accounting, resume plans after a cancellation and after the
evaluation and token limits, gaps checked against values worked out from
the runs, the migration hook on a version 0 database, processes that create
and write one database at once, hostile ids, filter values and file names,
and the record schema.

```bash
PYTHONPATH=src python -m pytest -q tests/test_e14_07_experiment_memory.py
```
