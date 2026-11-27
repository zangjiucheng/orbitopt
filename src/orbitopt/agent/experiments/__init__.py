"""Experiment memory and provenance (E14-07): an SQLite index and a content-addressed JSON store of agent runs.

See ``docs/agent/experiments.md``.
"""

from .cache import CacheLookup, Ignored, IgnoredReason, lookup
from .index import MIGRATIONS, SCHEMA_VERSION, ExperimentDB
from .ingest import export_run, orbitopt_version, record_run
from .model import (EXPERIMENT_SCHEMA, ArtifactError, ExperimentError, ExperimentRecord, Integrity, InvalidIdError,
                    MalformedRecordError, RunIntegrityError, SchemaVersionError, SolverStep, UnknownRunError,
                    UnsafePathError, VersionPolicy, cache_key)
from .store import ArtifactStore, safe_relative

__all__ = [
    "ArtifactError",
    "ArtifactStore",
    "cache_key",
    "CacheLookup",
    "EXPERIMENT_SCHEMA",
    "ExperimentDB",
    "ExperimentError",
    "ExperimentRecord",
    "export_run",
    "Ignored",
    "IgnoredReason",
    "Integrity",
    "InvalidIdError",
    "lookup",
    "MalformedRecordError",
    "MIGRATIONS",
    "orbitopt_version",
    "record_run",
    "RunIntegrityError",
    "safe_relative",
    "SCHEMA_VERSION",
    "SchemaVersionError",
    "SolverStep",
    "UnknownRunError",
    "UnsafePathError",
    "VersionPolicy",
]
