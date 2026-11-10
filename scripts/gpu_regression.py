"""Run the CUDA regression suite on a local GPU host and write a test record.

Default CI has no CUDA device, so the ``gpu``-marked tests only run here (or
on the self-hosted runner, ``.github/workflows/gpu.yml``). This script runs
them and writes one Markdown record with everything a reviewer needs to
attribute a failure:

* GPU model, driver version, CUDA driver/runtime versions, CuPy (and numba)
  versions, Python/numpy versions, host name
* the commit hash tested (and whether the work tree was dirty)
* every test case with its result and duration
* the full failure log of every failed/errored case

Two pytest passes run over the same scope (Lambert, N-body, closest
approach, plus the device-reporting tests):

1. ``cuda``: ``-m gpu`` with ``ORBITOPT_REQUIRE_GPU=1``, so a missing device
   errors out instead of skipping.
2. ``cpu-reference``: ``-m "not gpu"``, the numpy reference for the same
   code. If a CUDA case fails while its CPU reference passes, the cause is in
   the GPU backend or the environment. If both fail, the regression is not
   GPU-specific.

Usage (from the repo root, inside the ``orbitopt-gpu`` env)::

    python scripts/gpu_regression.py                     # -> docs/gpu-regression/runs/
    python scripts/gpu_regression.py --out-dir /tmp/rec  # elsewhere
    python scripts/gpu_regression.py -- -x               # extra pytest args

Exit code: 0 when every case passed, 1 when any failed or errored, 2 when
the CUDA pass could not start (for example no device).
Standard library only, apart from whatever the tests import.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import shlex
import shutil
import socket
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Task scope: Lambert, N-body, closest-approach (+ device reporting).
SCOPE = [
    "tests/test_gpu_lambert.py",
    "tests/test_nbody_gpu_backend.py",
    "tests/test_nbody_closest_approach.py",
    "tests/test_nbody_fused.py",
    "tests/test_nbody_ephemeris_precompute.py",
    "tests/test_runner_device.py",
]

PASSES = [
    ("cuda", "gpu", {"ORBITOPT_REQUIRE_GPU": "1"}),
    ("cpu-reference", "not gpu", {}),
]


def _run(cmd, **kw):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=60, **kw)
        return p.stdout.strip() if p.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def collect_environment() -> dict:
    env: dict = {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    env["commit"] = _run(["git", "rev-parse", "HEAD"], cwd=REPO) or "unknown"
    env["commit_subject"] = _run(["git", "log", "-1", "--format=%s"], cwd=REPO) or ""
    env["branch"] = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO) or ""
    status = _run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO)
    env["dirty"] = bool(status)

    if shutil.which("nvidia-smi"):
        q = _run(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total",
                  "--format=csv,noheader"])
        env["nvidia_smi_gpus"] = [line.strip() for line in (q or "").splitlines() if line.strip()]
        header = _run(["nvidia-smi"]) or ""
        for line in header.splitlines():
            if "CUDA Version" in line:
                env["nvidia_smi_cuda_version"] = line.split("CUDA Version:")[-1].strip(" |")
                break
    else:
        env["nvidia_smi_gpus"] = []

    # Ask a fresh interpreter, so this script never imports CuPy itself and a
    # broken CUDA install cannot crash it before anything is recorded.
    probe = r"""
import json, sys
out = {}
try:
    import numpy; out["numpy"] = numpy.__version__
except Exception as e: out["numpy"] = f"unavailable ({e})"
try:
    import numba; out["numba"] = numba.__version__
except Exception: out["numba"] = "not installed"
try:
    import cupy as cp
    out["cupy"] = cp.__version__
    rt = cp.cuda.runtime
    def ver(v): return f"{v // 1000}.{(v % 1000) // 10}"
    out["cuda_runtime"] = ver(rt.runtimeGetVersion())
    out["cuda_driver_api"] = ver(rt.driverGetVersion())
    out["device_count"] = rt.getDeviceCount()
    props = rt.getDeviceProperties(0)
    name = props["name"]
    out["device0"] = name.decode() if isinstance(name, bytes) else name
    out["device0_cc"] = f"{props['major']}.{props['minor']}"
    out["device0_mem_mb"] = round(props["totalGlobalMem"] / 1024**2)
except Exception as e:
    out.setdefault("cupy", "unavailable")
    out["cupy_error"] = f"{type(e).__name__}: {e}"
try:
    from orbitopt.core.gpu import GPU_AVAILABLE
    out["orbitopt_gpu_available"] = GPU_AVAILABLE
except Exception as e:
    out["orbitopt_gpu_available"] = f"import failed ({e})"
print(json.dumps(out))
"""
    raw = _run([sys.executable, "-c", probe], cwd=REPO)
    try:
        env.update(json.loads(raw) if raw else {"probe_error": "python probe failed"})
    except json.JSONDecodeError:
        env["probe_error"] = raw
    env["ORBITOPT_FORCE_CPU"] = os.environ.get("ORBITOPT_FORCE_CPU", "")
    return env


def run_pass(name, marker, extra_env, junit_path, log_path, pytest_args):
    cmd = [sys.executable, "-m", "pytest", *SCOPE, "-m", marker, "-v", "-rA",
           "-p", "no:cacheprovider", f"--junitxml={junit_path}", *pytest_args]
    env = dict(os.environ, **extra_env)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    log_path.write_text(f"$ {shlex.join(cmd)}\n\n{proc.stdout}\n{proc.stderr}", encoding="utf-8")
    return {"name": name, "marker": marker, "cmd": shlex.join(cmd), "returncode": proc.returncode,
            "wall_s": wall, "cases": parse_junit(junit_path), "log": log_path,
            "tail": (proc.stdout + proc.stderr)[-4000:]}


def parse_junit(path: Path) -> list[dict]:
    if not path.exists():
        return []
    cases = []
    for tc in ET.parse(path).getroot().iter("testcase"):
        outcome, detail = "passed", ""
        for tag in ("failure", "error", "skipped"):
            el = tc.find(tag)
            if el is not None:
                outcome = {"failure": "FAILED", "error": "ERROR", "skipped": "skipped"}[tag]
                detail = (el.get("message") or "") + ("\n" + el.text if el.text else "")
                break
        cls = tc.get("classname", "")
        cases.append({"id": f"{cls.replace('.', '/')}.py::{tc.get('name')}" if cls else tc.get("name"),
                      "outcome": outcome, "time_s": float(tc.get("time") or 0.0),
                      "detail": detail.strip()})
    return cases


def _fence(text):
    return "```text\n" + text.replace("```", "'''") + "\n```"


def render(env, passes) -> str:
    sha = env.get("commit", "unknown")
    lines = [f"# GPU regression record: `{sha[:7]}` on {env.get('host')}", ""]
    lines += ["## Environment", "", "| Field | Value |", "|---|---|"]
    gpus = "<br>".join(env.get("nvidia_smi_gpus") or []) or "(nvidia-smi unavailable)"
    rows = [
        ("Date", env.get("timestamp_utc")),
        ("Commit", f"`{sha}`" + (" (**dirty work tree**)" if env.get("dirty") else "")),
        ("Commit subject", env.get("commit_subject")),
        ("Branch", env.get("branch")),
        ("GPU (nvidia-smi: idx, name, driver, memory)", gpus),
        ("Device 0 (CuPy)", f"{env.get('device0', 'n/a')} (cc {env.get('device0_cc', '?')}, "
                            f"{env.get('device0_mem_mb', '?')} MiB)"),
        ("NVIDIA driver max CUDA (nvidia-smi)", env.get("nvidia_smi_cuda_version", "n/a")),
        ("CUDA driver API (CuPy)", env.get("cuda_driver_api", "n/a")),
        ("CUDA runtime (CuPy)", env.get("cuda_runtime", "n/a")),
        ("CuPy", env.get("cupy")),
        ("numba", env.get("numba")),
        ("numpy", env.get("numpy")),
        ("Python", env.get("python")),
        ("Platform", env.get("platform")),
        ("orbitopt GPU_AVAILABLE", env.get("orbitopt_gpu_available")),
    ]
    if env.get("cupy_error"):
        rows.append(("CuPy error", env["cupy_error"]))
    if env.get("ORBITOPT_FORCE_CPU"):
        rows.append(("ORBITOPT_FORCE_CPU", env["ORBITOPT_FORCE_CPU"]))
    lines += [f"| {k} | {v} |" for k, v in rows]

    lines += ["", "## Summary", "", "| Pass | Marker | passed | failed | errors | skipped | wall (s) | exit |",
              "|---|---|---|---|---|---|---|---|"]
    for p in passes:
        c = [x["outcome"] for x in p["cases"]]
        lines.append(f"| {p['name']} | `{p['marker']}` | {c.count('passed')} | {c.count('FAILED')} | "
                     f"{c.count('ERROR')} | {c.count('skipped')} | {p['wall_s']:.1f} | {p['returncode']} |")

    cpu_outcome = {x["id"]: x["outcome"] for p in passes if p["name"] == "cpu-reference"
                   for x in p["cases"]}
    for p in passes:
        lines += ["", f"## Cases: {p['name']}", "", f"`{p['cmd']}`", "",
                  "| Test | Result | Time (s) |", "|---|---|---|"]
        lines += [f"| `{x['id']}` | {x['outcome']} | {x['time_s']:.3f} |" for x in p["cases"]]
        if not p["cases"]:
            lines += ["", "No test cases were reported; pytest did not get past collection or start-up:",
                      "", _fence(p["tail"])]

    failed = [(p, x) for p in passes for x in p["cases"] if x["outcome"] in ("FAILED", "ERROR")]
    lines += ["", "## Failures and attribution", ""]
    if not failed and all(p["returncode"] == 0 for p in passes):
        lines.append("None. Every case passed.")
    else:
        lines += ["Fill in **Attribution** for each item below. Use one of: GPU backend "
                  "divergence, environment/driver, test tolerance, or a regression that is "
                  "not GPU-specific. Give the reason.", ""]
        for p, x in failed:
            hint = ""
            if p["name"] == "cuda":
                base = x["id"].split("[")[0]
                ref = [o for i, o in cpu_outcome.items() if i.split("[")[0] == base]
                if ref:
                    hint = (" CPU reference for this test: " + ", ".join(sorted(set(ref))) + ".")
            lines += [f"### {p['name']}: `{x['id']}` ({x['outcome']})", "",
                      f"**Attribution:** _TODO_.{hint}", "", _fence(x["detail"][-6000:]), ""]
    lines += ["", "Raw pytest logs: " + ", ".join(f"`{p['log'].name}`" for p in passes), ""]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", type=Path, default=REPO / "docs" / "gpu-regression" / "runs")
    ap.add_argument("pytest_args", nargs="*", help="extra pytest args (after --)")
    args = ap.parse_args(argv)

    env = collect_environment()
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    stem = f"{stamp}-{env.get('commit', 'unknown')[:7]}"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    passes = []
    for name, marker, extra in PASSES:
        print(f"[gpu_regression] pass {name} (-m '{marker}') ...", flush=True)
        passes.append(run_pass(name, marker, extra, args.out_dir / f"{stem}-{name}.junit.xml",
                               args.out_dir / f"{stem}-{name}.log", args.pytest_args))
        print(f"[gpu_regression]   exit {passes[-1]['returncode']}, {passes[-1]['wall_s']:.1f}s", flush=True)

    record = args.out_dir / f"{stem}.md"
    record.write_text(render(env, passes), encoding="utf-8")
    (args.out_dir / f"{stem}.env.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    print(f"[gpu_regression] record: {record}")

    cuda = passes[0]
    if not cuda["cases"]:
        return 2
    # pytest exit 5 = "no tests collected", which is fine for an empty pass.
    return 0 if all(p["returncode"] in (0, 5) for p in passes) else 1


if __name__ == "__main__":
    sys.exit(main())
