"""Record the FL-403 3-DOF reference run (task #208).

Flies the OpenTsiolkovsky ascent defined by the FL-403 spike fixture files
and records the *raw* OT outputs (``trajectory.csv``, ``summary.json``,
``bodies/``, plus the exact ``config.json`` sent to the binary) into the
fixture tree, so the offline test suite can replay this exact reference
without the OT binary (see
``tests/test_flightlab_sixdof_tracking_jsbsim.py``).

Usage::

    PYTHONPATH=src python scripts/fl403_record_reference.py

Requires the openTsiolkovsky-cli binary (see ``scripts/fetch_opentsiolkovsky.py``
or env ``ORBITOPT_OPENTSIOLKOVSKY``).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orbitopt.flightlab.adapters.opentsiolkovsky import (  # noqa: E402
    Vehicle, build_config, find_binary, probe_version,
)
from orbitopt.flightlab.contract import load_target  # noqa: E402

FIXTURE = Path("tests/fixtures/flightlab/opentsiolkovsky/fl403_spike")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(FIXTURE / "ot_out"), type=Path)
    args = ap.parse_args()

    binary = find_binary()
    print(f"binary: {binary} ({probe_version(binary)})")
    target = load_target(FIXTURE / "ascent_target.yaml")
    vehicle = Vehicle.load(FIXTURE / "ascent_vehicle.yaml")
    config = build_config(target, vehicle)

    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "config.json"
        cfg_path.write_text(json.dumps(config))
        raw = Path(tmp) / "opentsiolkovsky"
        proc = subprocess.run([str(binary), "run", str(cfg_path), "-o", str(raw), "--json"],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stderr[:2000])
            return 1
        args.out.mkdir(parents=True, exist_ok=True)
        shutil.copytree(raw, args.out, dirs_exist_ok=True)

    (args.out / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(f"recorded -> {args.out}")

    traj = json.loads((args.out / "summary.json").read_text())
    meco = float(vehicle.stages[0]["engine"]["burn_end_s"])
    import csv as _csv
    row = [r for r in _csv.DictReader(open(args.out / "trajectory.csv"))
           if float(r["time_s"]) == meco][-1]
    print(f"MECO @ {meco}s: alt {float(row['altitude_m'])/1000:.2f} km, "
          f"v_earth {float(row['speed_earth_mps']):.1f} m/s, "
          f"mach {float(row['mach']):.2f}, max_q {traj['maxima']['dynamic_pressure_pa']['value']:.0f} Pa")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
