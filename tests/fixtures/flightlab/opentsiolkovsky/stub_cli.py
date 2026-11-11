#!/usr/bin/env python3
"""OpenTsiolkovsky CLI stand-in for the FL-004 adapter tests (no binary in CI).

Speaks just enough of ``openTsiolkovsky-cli`` for the adapter's whole
pipeline: ``--version``, ``schema`` and ``run <cfg> -o <dir> --json`` (which
copies the recorded v1.0.0 outputs into ``-o`` and prints the success
document). Tests point ORBITOPT_OPENTSIOLKOVSKY at a chmod'ed copy of this
file, so run_ascent / find_binary / probe_version exercise their real code
paths against recorded data.
"""
import json
import os
import pathlib
import shutil
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[3]  # tests/fixtures/flightlab/opentsiolkovsky -> repo root
SRC = pathlib.Path(os.environ.get("OT_STUB_SRC",
                                  str(REPO / "src" / "orbitopt")))


def main(argv):
    if "--version" in argv:
        print("openTsiolkovsky-cli 1.0.0")
        return 0
    if len(argv) > 1 and argv[1] == "schema":
        print((SRC / "data" / "flightlab" / "schemas" /
               "opentsiolkovsky-config-v1.0.0.json").read_text(encoding="utf-8"))
        return 0
    if len(argv) > 1 and argv[1] == "run":
        out = pathlib.Path(argv[argv.index("-o") + 1])
        out.mkdir(parents=True, exist_ok=True)  # the real CLI creates its output dir
        src = pathlib.Path(os.environ["OT_STUB_FIXTURE"])
        for item in src.iterdir():
            if item.is_dir():
                shutil.copytree(item, out / item.name, dirs_exist_ok=True)
            else:
                shutil.copy(item, out / item.name)
        config = json.loads(pathlib.Path(argv[2]).read_text(encoding="utf-8"))
        print(json.dumps({"status": "ok", "config_name": config["name"],
                          "output_dir": str(out)}))
        return 0
    print(json.dumps({"status": "error", "error": {"kind": "cli", "message": "stub: unsupported args"}}))
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
