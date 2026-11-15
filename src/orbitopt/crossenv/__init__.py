"""Cross-environment mission orchestration (task #270, epic E5).

TudatPy lives in the conda-locked Python 3.10 environment (tudatpy ships only
from conda-forge/tudat-team) while the landing stack (gfold) needs Python
3.11+ (gfold 0.3.2 publishes no 3.10 wheel) -- so one interpreter cannot fly
both legs of the continuous mission. This package runs each leg in its own
environment and hands the results across the boundary through a *versioned
exchange envelope*: the stage writes its ``FlightRun`` directories and
``HandoffState`` documents into an exchange directory, seals them with a
``crossenv-exchange-1.0.json`` manifest that records a schema version, a
SHA-256 digest of every artifact and the producing environment's provenance
(interpreter, platform, ``toolchain_record()``), and the consuming stage
re-validates the schema and re-computes every digest before touching a byte.

The orchestrator (:func:`orbitopt.crossenv.orchestrator.run_plan`, driven by
``orbitopt crossenv run --plan <plan.yaml>``) launches each plan stage with
the interpreter that environment declares -- ``<env-python> -m
orbitopt.crossenv stage`` -- verifies each exchange as it lands, and writes
one mission report joining the legs.
"""

from __future__ import annotations

from orbitopt.crossenv.envelope import (
    ENVELOPE_KIND,
    ENVELOPE_SCHEMA_VERSION,
    EXCHANGE_FILE,
    CrossEnvError,
    file_digest,
    read_envelope,
    read_payloads,
    run_dir_digest,
    write_envelope,
)
from orbitopt.crossenv.orchestrator import run_plan
from orbitopt.crossenv.plan import CrossEnvPlan, CrossEnvStage, PlanError, load_plan

__all__ = [
    "CrossEnvError",
    "CrossEnvPlan",
    "CrossEnvStage",
    "ENVELOPE_KIND",
    "ENVELOPE_SCHEMA_VERSION",
    "EXCHANGE_FILE",
    "PlanError",
    "file_digest",
    "load_plan",
    "read_envelope",
    "read_payloads",
    "run_dir_digest",
    "run_plan",
    "write_envelope",
]
