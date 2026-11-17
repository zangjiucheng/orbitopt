"""``python -m orbitopt.crossenv`` -- the cross-environment orchestration CLI.

``run`` is the one command that completes the whole mission across both
environments; ``stage`` is the internal entry the orchestrator invokes inside
each declared interpreter; ``verify`` re-validates a sealed exchange. The
argument parser is the main CLI's (the ``crossenv`` group is declared in
``orbitopt/cli.py`` so the docs generator can extract it); standalone use
just prepends the group name.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main(argv=None) -> int:
    from orbitopt.cli import build_parser

    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(["crossenv", *argv])
    return dispatch(args)


def dispatch(args) -> int:
    """Execute a parsed ``crossenv`` command (from the main CLI or from this
    module's own parser); returns the exit code."""
    command = getattr(args, "crossenv_command", None)
    if command == "stage":
        from orbitopt.crossenv.plan import load_plan
        from orbitopt.crossenv.stages import execute_stage

        plan = load_plan(args.plan)
        return execute_stage(plan, plan.stage(args.stage), Path(args.out))

    if command == "verify":
        import json

        from orbitopt.crossenv.orchestrator import verify_exchange

        result = verify_exchange(args.exchange_dir)
        if getattr(args, "json", False):
            print(json.dumps(result, indent=1))
        else:
            env = result["environment"]
            print(f"valid: {result['exchange']} stage={result['stage']} env={env['env']} "
                  f"python={env['python']} toolchain={env['toolchain']}")
        return 0

    if command != "run":
        return 2

    from orbitopt.crossenv.orchestrator import run_plan

    overrides = {}
    for item in getattr(args, "env", []):
        if "=" not in item:
            print(f"crossenv: --env wants NAME=PYTHON, got {item!r}", file=sys.stderr)
            return 2
        name, value = item.split("=", 1)
        overrides[name] = value
    report = run_plan(args.plan, out=getattr(args, "out", None), env_overrides=overrides,
                      verbose=not getattr(args, "quiet", False))
    if getattr(args, "json", False):
        import json

        print(json.dumps(report, indent=1))
    return 0 if report["overall"]["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
