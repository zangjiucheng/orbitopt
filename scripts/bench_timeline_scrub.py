"""Per-frame CPU cost of SceneRenderer.set_time(): before vs after task #142.

"old" re-implements the pre-#142 hot path verbatim (Python-list trail lookup,
then a fresh pv.MultipleLines + add_mesh() replacing the "past" actor every
call); "new" is the current in-place _PastTrail update. Runs off-screen with
render=False, so it measures the CPU-side mesh construction/VTK upload prep
the task targets, not GPU draw time.

    PYTHONPATH=src python scripts/bench_timeline_scrub.py [trail_points]
"""
import sys, time
import numpy as np
import pyvista as pv
from orbitopt.viz import scene_renderer as sr

def scene(n_bodies=3, n=5000):
    t = np.linspace(0, 10, n)
    bodies = [{"id": "earth", "name": "Earth", "color": "#3399ff", "position": [0, 0, 0]}]
    for b in range(n_bodies):
        pos = np.c_[np.cos(t + b) * (1 + b), np.sin(t + b) * (1 + b), 0.01 * t]
        bodies.append({"id": f"sc{b}", "name": f"SC{b}", "color": "#ffaa00",
                       "trail": {"times": t.tolist(), "positions": pos.tolist()}})
    return {"bodies": bodies, "timeline": {"min": 0.0, "max": 10.0}}

def orig_position_at_time(body, t):
    times, positions = body["trail"]["times"], body["trail"]["positions"]
    if t <= times[0]:
        return np.asarray(positions[0], dtype=float)
    if t >= times[-1]:
        return np.asarray(positions[-1], dtype=float)
    idx = int(np.searchsorted(times, t))
    idx = max(1, min(idx, len(times) - 1))
    t0, t1 = times[idx - 1], times[idx]
    frac = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
    p0 = np.asarray(positions[idx - 1], dtype=float)
    p1 = np.asarray(positions[idx], dtype=float)
    return p0 + frac * (p1 - p0)

def old_set_time(self, t):
    for body_id, entry in self._moving.items():
        body = entry["body"]
        pos = orig_position_at_time(body, t)
        entry["point_poly"].points = pos.reshape(1, 3)
        idx = int(np.searchsorted(body["trail"]["times"], t))
        past_pts = np.vstack([entry["full_positions"][: max(idx, 1)], pos.reshape(1, 3)])
        poly = pv.MultipleLines(past_pts)
        self.plotter.add_mesh(poly, color=entry["color"], line_width=2.6, opacity=0.95,
                              name=f"past-{body_id}", pickable=False)

def run(fn, ts):
    t0 = time.perf_counter()
    for t in ts: fn(t)
    return (time.perf_counter() - t0) / len(ts) * 1000

n = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
p = pv.Plotter(off_screen=True)
r = sr.SceneRenderer(p); r.load(scene(n=n))
rng = np.random.default_rng(0)
ts = np.concatenate([np.linspace(0, 10, 150), rng.uniform(0, 10, 150)])  # sweep + random jumps
new = run(lambda t: r.set_time(t, render=False), ts)
old = run(lambda t: old_set_time(r, t), ts)
print(f"trail={n} pts x3 bodies: old {old:.3f} ms/frame, new {new:.3f} ms/frame, speedup {old/new:.1f}x")
