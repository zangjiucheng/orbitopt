"""Launch Mission Control -- the persistent, general-purpose desktop app
for browsing every scene this framework can produce, with a mission list
sidebar, a time-warp control strip, and live per-body orbit-info cards.

Prefer the installed console command: ``orbitopt`` (or ``orbitopt-app``).
This script is a thin reference kept for the examples set; it just calls
``orbitopt.viz.app.main``.
"""
from __future__ import annotations

from orbitopt.viz.app import main

if __name__ == "__main__":
    main()
