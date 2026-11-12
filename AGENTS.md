# Agent notes for orbitopt

## Running tests

The full suite is slow (real tudatpy propagations, optimizers and
benchmarks: well over an hour on a shared machine). Do not run all of
`tests/` while working on a task. Run:

1. **The tests for what you changed**: the `tests/test_<module>.py` files
   that cover the modules you touched, plus any test you added.
2. **The fast suite once, at the end**: `pytest tests/ -m "not slow and not gpu" -q -x`.

Run only the `slow` tests that exercise code you changed, by file or `-k`.
Give long commands an explicit `timeout` (e.g. `timeout 900 pytest ...`) and
read the result. Don't start a run in the background and poll a log file.

### Environment

- If `pip install -e .` refuses the interpreter (`requires-python` is
  `<3.13`), don't fight it: run from the source tree with `PYTHONPATH=src`.
- Qt/PyVista tests (`test_app*.py`, `test_viewport.py`, `test_viz_*.py`,
  `test_compute_pool.py`, `test_timeline_bar.py`, `test_flight_labels.py`, the
  `flightlab/ui` page tests) need a display. Run them under a virtual X
  server, as CI does: `xvfb-run -a pytest tests/test_viz_state.py`.
  Plain-Qt tests without VTK also work with `QT_QPA_PLATFORM=offscreen`.
- Don't try to install system packages (no root). If a library is missing,
  say so in the result instead of unpacking .debs by hand.

### Writing tests

Mark anything that takes more than a few seconds `@pytest.mark.slow`. Tests
that need CUDA get `@pytest.mark.gpu` (see `tests/conftest.py`).
