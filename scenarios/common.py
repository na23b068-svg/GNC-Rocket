"""Shared helpers for the phase-2 demo scenarios (results JSON per ARCHITECTURE.md "Results JSON schema")."""
import json
import math
import os
from multiprocessing import Pool

import numpy as np

from gnc.frames import quat_to_dcm
from gnc.rocket import ROOT, Rocket
from scenarios.passive_flight import LABELS, _r, _resample  # noqa: F401  (re-exported)

RES = ROOT / "results"
INDEX = ["passive.json", "analysis.json", "attitude_control.json", "los_path.json", "observers.json",
         "robustness.json"]
LIM_DEG = 5.0
DEG = 180.0 / math.pi

# Aggressive zig-zag (East jog -> North jog -> back West -> NNW climb-out), all below 180 m AGL / t < 2.1 s,
# placed at the edge of the TVC authority envelope (see scenarios/los_path.py, ENVELOPE note).
AGGRESSIVE_WP = [[0, 0, 0], [6, 0, 40], [6, 10, 100], [0, 10, 180], [-25, 60, 780]]
# Same, plus a 20 m East jog at 300-450 m AGL (t ~ 2.6-3.2 s, qbar > 30 kPa): beyond the authority envelope.
BEYOND_WP = [[0, 0, 0], [6, 0, 40], [6, 10, 100], [0, 10, 180], [-6, 22, 300], [14, 27, 450], [14, 42, 900]]
DOC_WP = [[0, 0, 0], [0, 30, 300], [0, 90, 900]]   # gentle baseline path from docs/gnc_design.md
# Disturbance set used across demos: 10 m/s wind from the West at 10 m (1/7 shear; the adverse direction for
# the aggressive path, whose last leg turns back West), 3 m/s Gauss-Markov
# turbulence, a 4 m/s 1-cos side gust from the South at t = 1.5 s, and 0.25 deg thrust misalignment per axis.
DIST = dict(wind=(10.0, 0.0, 0.0), wind_shear_exponent=1 / 7,
            gust={"std_mps": 3.0, "tau_s": 1.5, "one_cos": [{"t0": 1.5, "duration": 1.0,
                                                             "amplitude": [0.0, 4.0, 0.0]}]},
            thrust_misalignment_rad=(math.radians(0.25), math.radians(0.25)))
DIST_TEXT = ("10 m/s wind from the West (1/7 shear), 3 m/s turbulence, 4 m/s 1-cos side gust at 1.5 s, "
             "0.25 deg thrust misalignment per axis")

_ROCKET = None


def rocket():
    global _ROCKET
    if _ROCKET is None:
        _ROCKET = Rocket.load()
    return _ROCKET


def pmap(fn, items, procs=None):
    """Parallel map over processes (falls back to serial for tiny jobs)."""
    items = list(items)
    if len(items) <= 2:
        return [fn(i) for i in items]
    with Pool(min(procs or os.cpu_count() or 4, len(items))) as pool:
        return pool.map(fn, items)


def rnd(v, n=3):
    if v is None:
        return None
    v = float(v)
    return round(v, n) if math.isfinite(v) else None


def series(name, x, y, n=4, mode="lines"):
    return {"name": name, "x": _r(x, 3), "y": _r(y, n), "mode": mode}


def plot(pid, title, ylabel, ser, xlabel="t [s]"):
    return {"id": pid, "title": title, "xlabel": xlabel, "ylabel": ylabel, "series": ser}


def hlines(x0, x1, levels, name="limit"):
    """Constant reference lines (e.g. gimbal limits) as 2-point series."""
    return [series(f"{name} {'+' if v > 0 else ''}{v:g}", [x0, x1], [v, v]) for v in levels]


def events(res, extra=None):
    ev = dict(res["events"])
    ev.pop("drogue", None)
    ev.update(extra or {})
    lab = dict(LABELS, tvc_end="TVC off (tail-off)")
    return [{"t": round(float(v), 3), "label": lab.get(k, k)} for k, v in sorted(ev.items(), key=lambda kv: kv[1])]


def nose(q):
    """Body x-axis (nose) in ENU for an (N,4) quaternion array."""
    return np.array([quat_to_dcm(qq)[:, 0] for qq in np.atleast_2d(q)])


def angle_between(a, b):
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.arccos(np.clip(np.sum(a * b, axis=-1), -1.0, 1.0))


def flog(fc, key):
    return np.array(fc.log[key], float)


def truth_at(res, t):
    """Sim truth state interpolated at times t (N x 13), quaternion renormalised."""
    X = np.column_stack([np.interp(t, res["t"], res["x"][:, i]) for i in range(13)])
    X[:, 6:10] /= np.linalg.norm(X[:, 6:10], axis=1, keepdims=True)
    return X


def rocket_block(res, fc=None):
    """25 Hz time base + truth pos/quat (+ nav pos_est when a flight computer ran)."""
    t, pos, q = _resample(res)
    out = {"pos": [[round(float(v), 2) for v in p] for p in pos], "quat": [[round(float(v), 5) for v in qq] for qq in q]}
    if fc is not None and fc.log.get("pos_est"):
        lt, pe = flog(fc, "t"), flog(fc, "pos_est")
        out["pos_est"] = [[round(float(v), 2) for v in p] for p in
                          np.column_stack([np.interp(t, lt, pe[:, i]) for i in range(3)])]
    return _r(t, 3), out


def sat_frac(u_act_deg, lim=LIM_DEG, tol=1e-3):
    """Fraction of samples with either gimbal axis at the limit."""
    return float(np.mean(np.any(np.abs(u_act_deg) >= lim - tol, axis=1))) if len(u_act_deg) else 0.0


def guided_mask(res, fc):
    """Sim-time mask of the guided (phase 1) window of the flight computer."""
    lt, ph = flog(fc, "t"), flog(fc, "phase")
    g = lt[ph == 1]
    return (res["t"] >= g[0]) & (res["t"] <= g[-1]) if g.size else np.zeros(len(res["t"]), bool)


def write_demo(demo):
    """Write results/<id>.json and refresh results/index.json (fixed display order, existing files only)."""
    RES.mkdir(exist_ok=True)
    path = RES / f"{demo['id']}.json"
    path.write_text(json.dumps(demo, separators=(",", ":"), allow_nan=False))
    (RES / "index.json").write_text(json.dumps({"demos": [f for f in INDEX if (RES / f).exists()]}, indent=1))
    print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1e6:.2f} MB)")
    return path
