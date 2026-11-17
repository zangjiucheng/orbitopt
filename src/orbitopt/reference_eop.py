"""Unified Earth-orientation pinning for the reference regressions (E8-04).

Every reference regression in this repository turns raw flight state vectors
into GCRS states through astropy's frame bridge, and astropy's frame bridge
reads Earth orientation parameters (EOP) from a *mutable* source: unless told
otherwise it downloads the latest IERS-A predictions, which change daily. Two
reference pipelines do this today -- the Gate B flightlab regression
(``scripts/fl601_reference.py``) and the E2-04 frozen mission
(``scripts/e204_reference.py``) -- and both froze the EOP by hand. This module
is the single mechanism they (and any future reference) share.

Two freeze modes, one snapshot format
------------------------------------
A snapshot is a JSON file next to the reference it pins, recording the IERS
rows (and, for ``pinned-bundled`` snapshots, the effective values at the
reference epoch) that the recording stack saw::

    {"mode": "override", ...}          # build a table from the recorded rows
    {"mode": "pinned-bundled", ...}    # use the bundled table, verify first

``override`` (E2-04): the snapshot's rows become the active table. Exact for
pipelines whose reference frames were themselves produced through the table
mechanism (E2-04 propagates with tudatpy and only cross-checks EOP scope).

``pinned-bundled`` (Gate B): the recorded GCRS reference was produced by the
*default* table path of astropy 6.1 with downloads off -- the active table is
the bundled IERS-B (astropy 6 offline defaults to IERS-B, later versions to
IERS-A). An override table cannot reproduce that frame: the finals2000A
(IERS-A) rows imply a UT1-UTC that differs from IERS-B's by 0.071 s at the
Gate B epoch (1.07 arcsec, ~28 m), so pinning the *data* is not pinning the
*orientation*. The Gate B reference is therefore defined by the bundled table
itself: :func:`pinned_bundled_table` verifies the installed
``astropy-iers-data`` version, the active table and the effective EOP at the
epoch against the snapshot and *refuses* to proceed on any mismatch, so the
frame bridge provably sees the same bytes the recording saw.

Both modes always disable ``iers.conf.auto_download`` first; a regression
that cannot prove it is looking at the recorded Earth orientation fails
closed with :class:`ToolchainMismatch` instead of silently drifting.
"""
from __future__ import annotations

import json
import math
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

#: Snapshot schema version, bumped when the layout below changes.
SNAPSHOT_FORMAT = 1

SNAPSHOT_UNITS = {"MJD": "day", "UT1_UTC": "s", "PM_x": "arcsec", "PM_y": "arcsec"}


class ToolchainMismatch(RuntimeError):
    """The installed stack is not the stack the reference was recorded on."""


def _parse_utc(text: str) -> datetime:
    """ISO-8601 UTC datetime, tolerant of the trailing Z (py3.10-safe)."""
    return datetime.fromisoformat(text.removesuffix("Z"))


def pin_no_download() -> str:
    """Disable IERS downloads and return the installed data version string.

    This is the shared half of every ``_pin_earth_orientation``: astropy
    otherwise fetches fresh IERS-A predictions (daily churn -> frame drift
    between runs), and degraded-accuracy warnings from out-of-range queries
    would turn green CI runs orange without changing any number.
    """
    from astropy.utils import iers

    iers.conf.auto_download = False
    iers.conf.iers_degraded_accuracy = "ignore"
    try:
        import astropy_iers_data

        return str(astropy_iers_data.__version__)
    except ImportError:
        return "astropy bundled"


def utc_to_mjd(dt: datetime) -> float:
    """MJD(TAI-agnostic UTC) of a naive-or-aware UTC datetime, pure math.

    Kept astropy-free so snapshot coverage can be checked on any machine;
    matches ``astropy.time.Time(dt, scale='utc').mjd`` to far below the
    millisecond tolerances used here.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    epoch = datetime(1858, 11, 17, tzinfo=timezone.utc)
    return (dt - epoch).total_seconds() / 86400.0


def load_snapshot(path: Path) -> dict:
    """Load and schema-check an EOP snapshot written in either mode."""
    snap = json.loads(Path(path).read_text(encoding="utf-8"))
    mode = snap.get("mode", "override")  # pre-E8-04 files are override tables
    if mode not in ("override", "pinned-bundled"):
        raise ToolchainMismatch(f"{path}: unknown snapshot mode {mode!r}")
    rows = snap.get("rows")
    if not rows:
        raise ToolchainMismatch(f"{path}: snapshot has no EOP rows")
    for row in rows:
        if set(row) != set(SNAPSHOT_UNITS):
            raise ToolchainMismatch(f"{path}: row {row!r} must carry {sorted(SNAPSHOT_UNITS)}")
    if snap.get("units", SNAPSHOT_UNITS) != SNAPSHOT_UNITS:
        raise ToolchainMismatch(f"{path}: snapshot units must be {SNAPSHOT_UNITS}")
    if not snap.get("source_version"):
        raise ToolchainMismatch(f"{path}: snapshot records no source_version")
    if mode == "pinned-bundled":
        for key in ("epoch_utc", "astropy_series", "effective"):
            if key not in snap:
                raise ToolchainMismatch(f"{path}: pinned-bundled snapshot lacks {key!r}")
    snap["mode"] = mode
    return snap


def _override_table(snapshot: dict):
    import astropy.units as u
    import numpy as np
    from astropy.utils import iers

    unit_of = {"MJD": u.day, "UT1_UTC": u.s, "PM_x": u.arcsec, "PM_y": u.arcsec}
    return iers.IERS({name: np.array([row[name] for row in snapshot["rows"]]) * unit_of[name]
                      for name in SNAPSHOT_UNITS})


@contextmanager
def frozen_earth_orientation(snapshot: dict):
    """Override mode: the snapshot's rows ARE the active EOP table."""
    from astropy.utils import iers

    pin_no_download()
    with iers.earth_orientation_table.set(_override_table(snapshot)):
        yield


def _verify_bundled(snapshot: dict) -> None:
    """Check the installed stack reproduces the snapshot's effective EOP."""
    import astropy.units as u
    import numpy as np
    from astropy.time import Time
    from astropy.utils import iers

    installed = pin_no_download()
    if installed != snapshot["source_version"]:
        raise ToolchainMismatch(
            f"reference recorded with astropy-iers-data {snapshot['source_version']}, "
            f"installed {installed!r} -- reinstall the locked stack or re-record")
    import astropy

    if not str(astropy.__version__).startswith(snapshot["astropy_series"] + "."):
        raise ToolchainMismatch(
            f"reference recorded on astropy {snapshot['astropy_series']}.x, "
            f"installed {astropy.__version__}")
    table = iers.earth_orientation_table.get()
    table_name = type(table).__name__
    expected = snapshot.get("active_table", "IERS_A")
    if table_name != expected:
        raise ToolchainMismatch(
            f"reference frame bridge ran with the default {expected} table; the "
            f"installed stack defaults to {table_name} (astropy picks IERS-B "
            "offline but IERS-A when downloads are on) -- matching the lockfile "
            "or re-recording is the way out, not pressing on")
    epoch = _parse_utc(snapshot["epoch_utc"])
    eff = snapshot["effective"]
    t = Time(epoch, scale="utc")
    dut1 = float(np.atleast_1d(table.ut1_utc(t).to(u.s).value)[0])
    pm_x, pm_y = (float(np.atleast_1d(v.to(u.arcsec).value)[0]) for v in table.pm_xy(t))
    dev = float(eff.get("max_deviation_s", 1e-3))
    if abs(dut1 - float(eff["ut1_utc_s"])) > dev:
        raise ToolchainMismatch(
            f"UT1-UTC at epoch {epoch} is {dut1!r}s, snapshot recorded "
            f"{eff['ut1_utc_s']!r}s (deviation > {dev}s): the bundled table is "
            "not the recorded one, refusing a silent frame change")
    pm_dev = float(eff.get("max_deviation_arcsec", 1e-2))
    for got, key in ((pm_x, "pm_x_arcsec"), (pm_y, "pm_y_arcsec")):
        if abs(got - float(eff[key])) > pm_dev:
            raise ToolchainMismatch(
                f"polar motion {key} at epoch is {got!r} arcsec, snapshot "
                f"recorded {eff[key]!r} (deviation > {pm_dev} arcsec)")
    rowsd = {float(row["MJD"]): row for row in snapshot["rows"]}
    file_rows = {float(r["MJD"].value): r
                 for r in iers.IERS_A.open(iers.IERS_A_FILE)}
    for mjd, row in rowsd.items():
        if mjd not in file_rows:
            raise ToolchainMismatch(f"snapshot row MJD {mjd} missing from the bundled table")
        live = file_rows[mjd]
        for col in ("UT1_UTC", "PM_x", "PM_y"):
            if abs(float(live[col].value) - float(row[col])) > 1e-9:
                raise ToolchainMismatch(
                    f"bundled table row {mjd} {col}={live[col].value!r} != snapshot "
                    f"{row[col]!r}: recorded data file was patched under us")


@contextmanager
def pinned_bundled_table(snapshot: dict):
    """Pinned-bundled mode: verify, then keep the bundled default table active."""
    pin_no_download()
    _verify_bundled(snapshot)
    yield


@contextmanager
def freeze_reference(snapshot: dict):
    """Freeze the frame bridge for one reference regression, per its mode."""
    if snapshot["mode"] == "override":
        with frozen_earth_orientation(snapshot):
            yield
    else:
        with pinned_bundled_table(snapshot):
            yield


def bracketing_rows(epoch_utc: str, source_version_required: str) -> list[dict]:
    """Rows around ``epoch_utc`` from the installed IERS-A table, refusing a
    different data version than the recording's -- snapshots must bootstrap
    from the original recording data, never a newer prediction."""
    import astropy_iers_data
    from astropy.time import Time
    from astropy.utils import iers

    if astropy_iers_data.__version__ != source_version_required:
        raise ToolchainMismatch(
            f"snapshot creation needs astropy-iers-data {source_version_required}, "
            f"installed {astropy_iers_data.__version__}")
    table = iers.IERS_A.open(iers.IERS_A_FILE)
    day = math.floor(Time(_parse_utc(epoch_utc), scale="utc").mjd)
    rows = [{name: float(row[name].value) for name in ("MJD", "UT1_UTC", "PM_x", "PM_y")}
            for row in table if day - 1 <= float(row["MJD"].value) <= day + 2]
    if len(rows) != 4:
        raise ToolchainMismatch("recording table does not bracket the reference epoch")
    return rows


def effective_eop(epoch_utc: str) -> dict:
    """Effective UT1-UTC / polar motion at the epoch through the default
    table path (the exact path a pinned-bundled reference will use)."""
    import astropy.units as u
    import numpy as np
    from astropy.time import Time
    from astropy.utils import iers

    t = Time(datetime.fromisoformat(epoch_utc), scale="utc")
    table = iers.earth_orientation_table.get()
    dut1 = float(np.atleast_1d(table.ut1_utc(t).to(u.s).value)[0])
    pm_x, pm_y = (float(np.atleast_1d(v.to(u.arcsec).value)[0]) for v in table.pm_xy(t))
    return {"ut1_utc_s": dut1, "pm_x_arcsec": pm_x, "pm_y_arcsec": pm_y,
            "max_deviation_s": 1e-3, "max_deviation_arcsec": 1e-2}


def write_override_snapshot(path: Path, epoch_utc: str, source_version: str) -> None:
    """Create an override snapshot (idempotent: never overwrites)."""
    path = Path(path)
    if path.exists():
        return
    pin_no_download()
    snap = {"mode": "override", "format": SNAPSHOT_FORMAT,
            "source": "IERS-A finals2000A.all", "source_version": source_version,
            "epoch_utc": epoch_utc, "epoch_mjd_utc": utc_to_mjd(_parse_utc(epoch_utc)),
            "units": SNAPSHOT_UNITS, "rows": bracketing_rows(epoch_utc, source_version)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snap, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_pinned_bundled_snapshot(path: Path, epoch_utc: str, source_version: str,
                                 astropy_series: str, note: str) -> None:
    """Create a pinned-bundled snapshot (idempotent: never overwrites)."""
    path = Path(path)
    if path.exists():
        return
    import astropy

    if not str(astropy.__version__).startswith(astropy_series + "."):
        raise ToolchainMismatch(
            f"snapshot creation needs astropy {astropy_series}.x, "
            f"installed {astropy.__version__}")
    pin_no_download()
    from astropy.utils import iers

    snap = {"mode": "pinned-bundled", "format": SNAPSHOT_FORMAT,
            "source": "IERS-A finals2000A.all", "source_version": source_version,
            "epoch_utc": epoch_utc, "epoch_mjd_utc": utc_to_mjd(_parse_utc(epoch_utc)),
            "astropy_series": astropy_series,
            "active_table": type(iers.earth_orientation_table.get()).__name__,
            "units": SNAPSHOT_UNITS,
            "rows": bracketing_rows(epoch_utc, source_version),
            "effective": effective_eop(epoch_utc), "note": note}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snap, indent=2, sort_keys=True) + "\n", encoding="utf-8")
