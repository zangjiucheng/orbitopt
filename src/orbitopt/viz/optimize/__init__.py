"""Mission Control's Optimize workspace (E14-09): natural language in, a verified and replayable run out.

Three layers, so the logic is testable without a window:

* ``views``    -- Qt-free read models: the trace of a run as iteration rows,
  a convergence curve and the four verification statuses; the problem as
  math; report export.
* ``backend``  -- the work: provider choice from the environment, the bundled
  offline examples, compile / run / replay jobs over the public
  ``orbitopt.agent`` API, and the experiment directory they write.
* ``session``  -- the presenter: one state machine (idle, compiling,
  clarifying, draft, editing, confirmed, running, finished, cancelled,
  failed, replaying, replay) and the structured IR editor.
* ``workspace`` -- the PySide6 page that renders a session and runs its jobs
  on a background thread.

Importing this package imports nothing; ``orbitopt.agent`` is imported only
when a job runs, so Mission Control starts without the ``agent`` extras.
"""
