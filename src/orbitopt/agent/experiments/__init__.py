"""Experiment memory and provenance (E14-07): an SQLite index and a content-addressed JSON store of agent runs.

``record_run(db, run_dir)`` verifies a saved E14-05 run directory by
replaying it, stores every file by digest and indexes one
``ExperimentRecord``. ``lookup`` serves a prior verified result for an
identical solve while it still verifies, ``resume_plan`` describes what an
interrupted run completed and may still spend, and ``query``, ``compare``
and ``aggregate`` read the records back. Standard library plus the IR and
the runtime; no LLM SDK. See ``docs/agent/experiments.md``.
"""

from orbitopt.agent.runtime import UnsafePathError

from .analysis import (ORDER_BY, RESUMABLE, Aggregate, Comparison, ComparisonRow, FamilyStats, Page, ResumePlan,
                       aggregate, compare, query, resume_plan)
from .cache import CacheLookup, Ignored, IgnoredReason, lookup
from .index import MIGRATIONS, SCHEMA_VERSION, ExperimentDB
from .ingest import export_run, record_run
from .model import (EXPERIMENT_SCHEMA, ArtifactError, ExperimentError, ExperimentRecord, Integrity, InvalidIdError,
                    MalformedRecordError, RunIntegrityError, SchemaVersionError, SolverStep, UnknownRunError,
                    VersionPolicy, cache_key)
from .store import ArtifactStore

__all__ = [
    "Aggregate", "ArtifactError", "ArtifactStore", "CacheLookup", "Comparison", "ComparisonRow",
    "EXPERIMENT_SCHEMA", "ExperimentDB", "ExperimentError", "ExperimentRecord", "FamilyStats", "Ignored",
    "IgnoredReason", "Integrity", "InvalidIdError", "MIGRATIONS", "MalformedRecordError", "ORDER_BY", "Page",
    "RESUMABLE", "ResumePlan", "RunIntegrityError", "SCHEMA_VERSION", "SchemaVersionError", "SolverStep",
    "UnknownRunError", "UnsafePathError", "VersionPolicy", "aggregate", "cache_key", "compare", "export_run",
    "lookup", "query", "record_run", "resume_plan",
]
