"""The SQLite index of the experiment database (stdlib ``sqlite3``), beside its artifact store.

A database is a directory: ``index.sqlite`` holds one row per record, its
solver steps and its cache keys; ``artifacts/`` holds every run file by
digest. The index runs in WAL mode with a busy timeout, and every write is
one ``BEGIN IMMEDIATE`` transaction, so two processes that record at once
queue instead of corrupting it. Every statement takes its values as
parameters; no value is ever formatted into SQL.

``schema_version`` holds one integer. ``MIGRATIONS[i]`` takes the index
from version ``i`` to ``i + 1``; opening a database runs the missing steps
in one transaction. A new file is version 0, so creating the tables is
migration 0.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, Sequence

from .model import ExperimentRecord, SchemaVersionError, UnknownRunError, check_run_id
from .store import ArtifactStore

INDEX = "index.sqlite"
ARTIFACTS = "artifacts"
BUSY_TIMEOUT_S = 30.0

Migration = Callable[[sqlite3.Connection], None]


def _create_v1(conn: sqlite3.Connection) -> None:
    for statement in (
        """CREATE TABLE experiments (
            run_id TEXT PRIMARY KEY, record TEXT NOT NULL, problem_digest TEXT NOT NULL, problem_id TEXT NOT NULL,
            family TEXT NOT NULL, status TEXT NOT NULL, verified INTEGER NOT NULL, objective REAL,
            max_violation REAL, evaluations INTEGER NOT NULL, wall_s REAL NOT NULL, tokens INTEGER NOT NULL,
            seed INTEGER NOT NULL, planner TEXT NOT NULL, provider TEXT, model TEXT, created_at TEXT NOT NULL,
            orbitopt_version TEXT NOT NULL, parent_id TEXT REFERENCES experiments(run_id),
            resumed_from TEXT REFERENCES experiments(run_id))""",
        """CREATE TABLE steps (
            run_id TEXT NOT NULL REFERENCES experiments(run_id), position INTEGER NOT NULL,
            solver_id TEXT NOT NULL, seed INTEGER NOT NULL, PRIMARY KEY (run_id, position))""",
        """CREATE TABLE cache (
            cache_key TEXT NOT NULL, run_id TEXT NOT NULL REFERENCES experiments(run_id),
            PRIMARY KEY (cache_key, run_id))""",
        "CREATE INDEX experiments_problem ON experiments (problem_digest)",
        "CREATE INDEX experiments_created ON experiments (created_at)",
        "CREATE INDEX steps_solver ON steps (solver_id)",
    ):
        conn.execute(statement)


MIGRATIONS: tuple[Migration, ...] = (_create_v1,)
SCHEMA_VERSION = len(MIGRATIONS)


def _enable_wal(conn: sqlite3.Connection, timeout_s: float) -> None:
    """Switch the file to WAL. SQLite answers a racing switch with SQLITE_BUSY without waiting on the busy
    timeout, so the switch is retried until the timeout instead."""
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            return
        except sqlite3.OperationalError as exc:
            if exc.sqlite_errorcode != sqlite3.SQLITE_BUSY or time.monotonic() > deadline:
                raise
            time.sleep(0.01)


class ExperimentDB:
    """An experiment database at ``root``. Use one instance per process or thread.

    ``migrations`` is the migration hook: the steps that take the index from
    each version to the next. Opening a database newer than the last step,
    or an SQLite file that is not an experiment database, raises
    ``SchemaVersionError``.
    """

    def __init__(self, root: str | Path, *, migrations: Sequence[Migration] = MIGRATIONS,
                 timeout_s: float = BUSY_TIMEOUT_S):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.store = ArtifactStore(self.root / ARTIFACTS)
        self._conn = sqlite3.connect(self.root / INDEX, timeout=timeout_s, isolation_level=None)
        try:
            _enable_wal(self._conn, timeout_s)
            self._conn.execute("PRAGMA foreign_keys=ON")
            self.version = self._migrate(tuple(migrations))
        except BaseException:
            self._conn.close()
            raise

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "ExperimentDB":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """One write transaction; it takes the write lock up front and rolls back on any error."""
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield self._conn
        except BaseException:
            if self._conn.in_transaction:
                self._conn.execute("ROLLBACK")
            raise
        self._conn.execute("COMMIT")

    def _migrate(self, migrations: tuple[Migration, ...]) -> int:
        with self.transaction() as conn:
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
            if "schema_version" not in tables:
                if tables:
                    raise SchemaVersionError(f"{self.root / INDEX} is an SQLite file but not an experiment database")
                conn.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
                conn.execute("INSERT INTO schema_version (version) VALUES (0)")
            rows = conn.execute("SELECT version FROM schema_version").fetchall()
            if len(rows) != 1 or not isinstance(rows[0][0], int):
                raise SchemaVersionError("schema_version must hold exactly one integer")
            version = rows[0][0]
            if version > len(migrations):
                raise SchemaVersionError(f"the database is at version {version}; this code knows {len(migrations)}")
            for step in range(version, len(migrations)):
                migrations[step](conn)
                conn.execute("UPDATE schema_version SET version = ?", (step + 1,))
            return len(migrations)

    # ------------------------------------------------------------------------------------------- records
    def rows(self, sql: str, params: Sequence = ()) -> list[tuple]:
        """Run a read statement with its values as parameters."""
        return self._conn.execute(sql, tuple(params)).fetchall()

    def find(self, run_id: str) -> ExperimentRecord | None:
        found = self.rows("SELECT record FROM experiments WHERE run_id = ?", (check_run_id(run_id),))
        return ExperimentRecord.from_dict(json.loads(found[0][0])) if found else None

    def get(self, run_id: str) -> ExperimentRecord:
        """The record of ``run_id``; ``UnknownRunError`` when there is none."""
        record = self.find(run_id)
        if record is None:
            raise UnknownRunError(f"no record has run id {run_id}")
        return record

    def add(self, record: ExperimentRecord, cache_keys: Sequence[str] = ()) -> ExperimentRecord:
        """Insert ``record`` with its steps and cache keys in one transaction, and return the stored record.

        A run already recorded keeps its first record, so adding it again
        changes nothing. A parent or resume link must name a recorded run.
        """
        data = record.to_dict()
        with self.transaction() as conn:
            found = conn.execute("SELECT record FROM experiments WHERE run_id = ?", (record.run_id,)).fetchone()
            if found:
                return ExperimentRecord.from_dict(json.loads(found[0]))
            for link in (record.parent_id, record.resumed_from):
                if link is not None and not conn.execute("SELECT 1 FROM experiments WHERE run_id = ?",
                                                         (link,)).fetchone():
                    raise UnknownRunError(f"the link {link} names no recorded run")
            conn.execute(
                "INSERT INTO experiments (run_id, record, problem_digest, problem_id, family, status, verified, "
                "objective, max_violation, evaluations, wall_s, tokens, seed, planner, provider, model, created_at, "
                "orbitopt_version, parent_id, resumed_from) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                "?, ?, ?, ?)",
                (record.run_id, json.dumps(data, sort_keys=True), record.problem_digest, record.problem_id,
                 record.family, record.status.value, int(record.verified), record.objective, record.max_violation,
                 record.usage.evaluations, record.usage.wall_s, record.usage.tokens, record.seed, record.planner,
                 record.provider, record.model, record.created_at, record.orbitopt_version, record.parent_id,
                 record.resumed_from))
            conn.executemany("INSERT INTO steps (run_id, position, solver_id, seed) VALUES (?, ?, ?, ?)",
                             [(record.run_id, i, s.solver_id, s.seed) for i, s in enumerate(record.steps)])
            conn.executemany("INSERT OR IGNORE INTO cache (cache_key, run_id) VALUES (?, ?)",
                             [(key, record.run_id) for key in cache_keys])
        return record

    def cached_runs(self, key: str) -> list[str]:
        """The run ids stored under a cache key, newest first."""
        return [row[0] for row in self.rows(
            "SELECT c.run_id FROM cache c JOIN experiments e ON e.run_id = c.run_id WHERE c.cache_key = ? "
            "ORDER BY e.created_at DESC, c.run_id", (key,))]
