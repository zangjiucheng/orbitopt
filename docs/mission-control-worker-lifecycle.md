# Mission Control — compute worker lifecycle

Mission Control runs mission loaders in one persistent worker **process**
(`orbitopt.viz.compute_pool.ComputePool`, started with `spawn` on every OS)
so a GIL-holding compute cannot freeze the UI. This page lists what must hold
for that process and how it is tested.

## Guarantees

| Scenario | Expected behaviour |
|---|---|
| Worker crashes mid-compute (native segfault, OOM kill) | That one load fails with *"the compute process exited unexpectedly"*; the badge shows FAILED, the mission list is re-enabled, the dead worker is reaped. The next load starts a fresh worker. |
| Worker dies while idle (killed between loads) | The next load is **not** affected: a health ping before each submit replaces the dead worker transparently. |
| Quit while a compute is running | Window closes in well under 10 s; the worker is terminated *and joined* (no zombie); no stale result is applied after close. |
| Quit while idle | Worker terminated and joined. |
| `QApplication.quit()` without closing the window | `aboutToQuit` terminates the worker, so `concurrent.futures`' exit hook does not wait out the compute. |
| Load racing the quit | A closed pool refuses new work, so no worker is spawned after quit. |
| App crashes / is force-killed | The worker notices via its parent sentinel (pipe EOF on POSIX, process handle on Windows) and exits within ~1 s. Windows never re-parents orphans, so the old `getppid()` check alone never fired there. |
| Repeated mission switching | One worker process for the whole session; re-activating while loading is ignored. |
| UI responsiveness during compute | A 16 ms main-thread timer keeps ticking; no stall > 0.5 s. |

## Automated tests

```bash
# Headless, Qt-free except QtCore -- runs anywhere, including Windows/macOS:
pytest tests/test_compute_pool.py -v

# Real window (needs a display; CI runs it under xvfb):
pytest tests/test_app_worker_lifecycle.py -v
```

`test_compute_pool.py` covers every row above at the pool level, including a
subprocess "fake app" that is SIGKILLed / TerminateProcess'd to check that its
worker exits. `test_app_worker_lifecycle.py` checks the window wiring:
quit mid-compute, quit idle, crash → FAILED → next load works, and repeated
switching.

CI runs on Linux only. Run `pytest tests/test_compute_pool.py` on macOS and
Windows before a release (it is written to be OS-independent, but CI only
exercises it on Linux).

## Manual checklist (macOS and Windows)

Start the app with `orbitopt-app` (or `python -m orbitopt.viz.app`). Delete
the scene cache first so loads really compute. Find the worker with
Activity Monitor / `ps -ef | grep spawn_main` (macOS) or Task Manager →
Details / `Get-CimInstance Win32_Process -Filter "Name='python.exe'" | ft ProcessId,ParentProcessId,CommandLine`
(Windows). It is the child `python` whose command line contains
`multiprocessing.spawn`.

1. **One worker per session.** Switch Solar System → Artemis II → GOES →
   Solar System. Exactly one worker process exists, with the same PID throughout.
2. **UI stays live during compute.** While Artemis II computes, the spinner
   animates, the window moves and resizes smoothly, and Windows never marks it
   "Not Responding".
3. **Worker killed mid-compute.** Start Artemis II, then kill the worker
   (`kill -9 <pid>` / End task). A *"Failed to load"* dialog appears, the
   badge shows FAILED, the mission list is usable. Pick GOES: it loads on a
   new worker PID.
4. **Worker killed while idle.** After a load finishes, kill the worker, then
   pick another uncached mission. It loads with no error dialog.
5. **Quit mid-compute.** Start Artemis II and quit immediately (⌘Q /
   window close / Alt+F4). The app exits in a few seconds at most and no
   worker process remains. On macOS, `ps -o pid,stat,command` shows no
   `Z` (zombie) entry.
6. **App force-killed.** Start Artemis II, then kill the *app* process
   (`kill -9` / End task on the parent). The worker disappears within ~1–2 s.
   This is the Windows-specific case.
7. **Rapid clicks.** Click several missions quickly while one is computing.
   The list is disabled during the load; nothing queues up and no extra
   worker appears.
8. **In-process fallback.** `ORBITOPT_INPROCESS_COMPUTE=1 orbitopt-app`: no
   worker process is started; loads still work (the UI may stutter — expected).
