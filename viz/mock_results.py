"""Write a mock results set (ARCHITECTURE.md "Results JSON schema") for dashboard development.

    python -m viz.mock_results [--out viz/mock]

Five demos built from simple closed-form curves (no simulator): passive flight with parachute descent, sounding-rocket
waypoint path following, an orbital launch ascent (scene LAUNCH), a spacecraft docking approach (scene LVLH, keep-out
sphere) and an analysis-only demo (rocket null). Values only need to look plausible.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DT = 0.04  # 25 Hz output


def quat_from_nose(nose, roll):
    """Scalar-first quaternions (body -> ENU) with body x along `nose` (N,3) and roll angle `roll` (N,)."""
    out = []
    for d, r in zip(nose, roll):
        x = d / np.linalg.norm(d)
        ref = np.array([0.0, 0.0, 1.0]) if abs(x[2]) < 0.95 else np.array([1.0, 0.0, 0.0])
        a = np.cross(ref, x); a /= np.linalg.norm(a)
        y = np.cos(r) * a + np.sin(r) * np.cross(x, a)
        R = np.column_stack([x, y, np.cross(x, y)])
        out.append(dcm_to_quat(R))
    return np.array(out)


def dcm_to_quat(R):
    """Rotation matrix -> scalar-first unit quaternion (Shepperd's method, robust near 180 deg)."""
    tr = np.trace(R)
    i = int(np.argmax([tr, R[0, 0], R[1, 1], R[2, 2]]))
    if i == 0:
        s = 2 * np.sqrt(1 + tr)
        q = [s / 4, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        a, b, c = i - 1, i % 3, (i + 1) % 3
        s = 2 * np.sqrt(1 + R[a, a] - R[b, b] - R[c, c])
        q = np.zeros(4)
        q[0] = (R[c, b] - R[b, c]) / s
        q[1 + a] = s / 4
        q[1 + b] = (R[b, a] + R[a, b]) / s
        q[1 + c] = (R[c, a] + R[a, c]) / s
    q = np.asarray(q, float)
    return q / np.linalg.norm(q) * (1 if q[0] >= 0 else -1)


def _r(a, nd=3):
    return np.round(np.asarray(a, dtype=float), nd).tolist()


def _series(name, x, y, mode="lines"):
    return {"name": name, "x": _r(x, 3), "y": _r(y, 4), "mode": mode}


def _events(ev):
    return [{"t": round(float(t), 3), "label": k} for k, t in sorted(ev.items(), key=lambda kv: kv[1])]


def ascent(t, tb=3.6, a_boost=70.0, g_eff=12.05):
    """Closed-form vertical profile: constant accel boost, then constant effective deceleration (gravity + drag)."""
    vb, zb = a_boost * tb, 0.5 * a_boost * tb ** 2
    tau = np.clip(t - tb, 0, None)
    z = np.where(t < tb, 0.5 * a_boost * t ** 2, zb + vb * tau - 0.5 * g_eff * tau ** 2)
    vz = np.where(t < tb, a_boost * t, vb - g_eff * tau)
    return z, vz, tb + vb / g_eff


# --------------------------------------------------------------------------------------------- passive
def passive_variant(vid, label, wind):
    _, _, t_apo = ascent(np.zeros(1))
    ta = np.arange(0, t_apo + 1e-9, DT)
    z, vz, _ = ascent(ta)
    tilt = np.tan(np.radians(6.0))
    # rail tilt carries it east; in wind the rocket weathercocks upwind during boost
    x = tilt * z - wind[0] * 0.6 * ta * np.clip(ta / 3.6, 0, 1)
    y = -wind[1] * 0.6 * ta * np.clip(ta / 3.6, 0, 1)
    pos_up = np.column_stack([x, y, z])
    z_apo, z_main = z[-1], 300.0
    t_main = t_apo + (z_apo - z_main) / 24.0
    t_land = t_main + z_main / 5.5
    td = np.arange(ta[-1] + DT, t_land + 1e-9, DT)
    zd = np.where(td < t_main, z_apo - 24.0 * (td - t_apo), z_main - 5.5 * (td - t_main)).clip(0)
    pos_dn = np.column_stack([x[-1] + wind[0] * (td - t_apo), y[-1] + wind[1] * (td - t_apo), zd])
    t = np.concatenate([ta, td])
    pos = np.vstack([pos_up, pos_dn])
    vel = np.gradient(pos, t, axis=0)
    nose = vel.copy()
    nose[:3] = [np.sin(np.radians(6)), 0, np.cos(np.radians(6))]
    nose[len(ta):] = [1.0, 0.2, -0.6]          # hanging under the drogue/main
    q = quat_from_nose(nose, 0.3 * t)
    speed = np.linalg.norm(vel, axis=1)
    mach = speed / (340.3 - 0.004 * pos[:, 2])
    ev = {"Liftoff": 0.0, "Rail exit": 0.40, "Burnout": 3.6, "Apogee": t_apo, "Drogue deploy": t_apo + 0.5,
          "Main deploy": t_main, "Landing": t[-1]}
    return {
        "id": vid, "label": label, "t": _r(t, 3),
        "rocket": {"pos": _r(pos), "quat": _r(q, 5)},
        "events": _events(ev),
        "plots": [
            {"id": "alt", "title": "Altitude", "xlabel": "t [s]", "ylabel": "m AGL", "series": [_series("altitude", t, pos[:, 2])]},
            {"id": "vel", "title": "Velocity", "xlabel": "t [s]", "ylabel": "m/s",
             "series": [_series("speed", t, speed), _series("vertical", t, vel[:, 2])]},
            {"id": "mach", "title": "Mach number", "xlabel": "t [s]", "ylabel": "-", "series": [_series("Mach", t, mach)]},
        ],
        "metrics": {"Apogee AGL [m]": round(float(z_apo), 1), "Time to apogee [s]": round(float(t_apo), 2),
                    "Max speed [m/s]": round(float(speed.max()), 1), "Max Mach [-]": round(float(mach.max()), 3),
                    "Landing distance [m]": round(float(np.hypot(*pos[-1, :2])), 1)},
        "tables": [{"title": "Event timeline", "columns": ["Event", "t [s]", "Altitude [m]"],
                    "rows": [[k, round(float(tv), 2), round(float(np.interp(tv, t, pos[:, 2])), 1)]
                             for k, tv in sorted(ev.items(), key=lambda kv: kv[1])]}],
        "notes": "Recovery: drogue at apogee (24 m/s), main at 300 m AGL (5.5 m/s), drifting with the wind.",
    }


def passive_demo():
    return {"id": "passive", "group": "Uncontrolled flight", "title": "Passive ballistic flight",
            "description": "Uncontrolled flight with the gimbal locked at zero, from the 6 deg launch rail to "
                           "landing under parachutes.\n\nIn wind the rocket weathercocks upwind during boost and then "
                           "drifts downwind under the drogue and main chutes.",
            "variants": [passive_variant("nominal", "No wind", np.zeros(3)),
                         passive_variant("wind", "Wind 6 m/s from W", np.array([6.0, 1.0, 0.0]))]}


# --------------------------------------------------------------------------------------------- waypoints
def cross_track(p, path):
    """Distance from each point of p (N,3) to the polyline `path` (M,3)."""
    best = np.full(len(p), np.inf)
    for a, b in zip(path[:-1], path[1:]):
        ab = b - a
        s = np.clip(((p - a) @ ab) / (ab @ ab), 0, 1)
        best = np.minimum(best, np.linalg.norm(p - (a + s[:, None] * ab), axis=1))
    return best


def waypoint_variant(vid, label, gust):
    wps = np.array([[0, 0, 400], [120, 60, 1100], [260, -40, 1800], [360, 20, 2500]], float)
    path = np.vstack([[0, 0, 0], wps])
    cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    _, _, t_apo = ascent(np.zeros(1))
    t = np.arange(0, t_apo + 1e-9, DT)
    z, vz, _ = ascent(t)
    s = np.minimum(z * (cum[-1] / z[-1]), cum[-1])               # arclength along the reference path
    ref = np.column_stack([np.interp(s, cum, path[:, j]) for j in range(3)])
    k = 40                                                       # round the corners like a real turn would
    pos = np.column_stack([np.convolve(np.pad(ref[:, j], (k // 2, k - k // 2 - 1), mode="edge"), np.ones(k) / k, "valid")
                           for j in range(3)])
    amp = 2.0 if gust else 1.0
    dev = np.zeros_like(pos)
    gim = np.zeros(len(t))
    for c in cum[1:-1]:                                          # decaying overshoot after each corner
        tc = np.interp(c, s, t)
        tau = np.clip(t - tc, 0, None)
        env = np.where(t > tc, np.exp(-tau / 2.5), 0)
        dev[:, 0] += 12 * amp * env * np.sin(1.6 * tau)
        dev[:, 1] += -8 * amp * env * np.sin(1.9 * tau)
        gim += 3.5 * env * np.cos(1.6 * tau)
    if gust:
        dev[:, 0] += 6 * np.sin(0.7 * t) * np.sin(0.13 * t)
        dev[:, 1] += 5 * np.sin(1.1 * t + 1) * np.sin(0.09 * t)
        gim += 1.2 * np.sin(0.7 * t + 0.4)
    pos = pos + dev
    pos[0] = 0
    vel = np.gradient(pos, t, axis=0)
    q = quat_from_nose(vel + [0, 0, 1e-3], 0.2 * t)
    est = pos + np.column_stack([2.0 * np.sin(0.31 * t), 1.5 * np.cos(0.23 * t), 1.2 * np.sin(0.47 * t + 1)])
    xt = cross_track(pos, path)
    gim = np.clip(gim, -5, 5)
    miss = [float(np.min(np.linalg.norm(pos - w, axis=1))) for w in wps]
    ev = {"Liftoff": 0.0, "Rail exit": 0.4, "Guidance on": 0.6, "Burnout": 3.6, "Apogee": float(t[-1])}
    ev.update({f"WP{j + 1}": float(t[int(np.argmin(np.linalg.norm(pos - w, axis=1)))]) for j, w in enumerate(wps)})
    return {
        "id": vid, "label": label, "t": _r(t, 3),
        "rocket": {"pos": _r(pos), "quat": _r(q, 5), "pos_est": _r(est)},
        "objects": [{"name": "Waypoints", "kind": "waypoints", "pos": _r(wps)},
                    {"name": "Reference path", "kind": "path", "pos": _r(path)}],
        "events": _events(ev),
        "plots": [
            {"id": "xte", "title": "Cross-track error", "xlabel": "t [s]", "ylabel": "m", "series": [_series("cross-track", t, xt)]},
            {"id": "gimbal", "title": "Gimbal angle", "xlabel": "t [s]", "ylabel": "deg",
             "series": [_series("delta_p", t, gim), _series("delta_y", t, 0.6 * np.roll(gim, 15))]},
            {"id": "naverr", "title": "Navigation position error", "xlabel": "t [s]", "ylabel": "m",
             "series": [_series("|p_est - p|", t, np.linalg.norm(est - pos, axis=1))]},
        ],
        "metrics": {"RMS cross-track error [m]": round(float(np.sqrt(np.mean(xt ** 2))), 2),
                    "Max cross-track error [m]": round(float(xt.max()), 2),
                    **{f"WP{j + 1} miss [m]": round(m, 2) for j, m in enumerate(miss)},
                    "Waypoints within 25 m": f"{sum(m < 25 for m in miss)}/{len(miss)}"},
    }


def waypoint_demo():
    return {"id": "los_path", "group": "Path following", "title": "LOS waypoint path following",
            "description": "Line-of-sight guidance steers the ascent through four waypoints. Cross-track error is "
                           "measured to the reference polyline; the dashed line is the navigation estimate.",
            "variants": [waypoint_variant("nominal", "No disturbance", False),
                         waypoint_variant("gusts", "With wind gusts", True)]}


# --------------------------------------------------------------------------------------------- orbital ascent
def ascent_orbital_variant(vid, label, wind):
    T, dt = 540.0, 0.5
    t = np.arange(0, T + 1e-9, dt)
    u = t / T
    gamma = np.radians(90 - 88 * u ** 0.6)                       # flight-path angle above the horizon
    v = 7400 * u ** 1.5 + 30
    x = np.cumsum(v * np.cos(gamma)) * dt
    z = np.cumsum(v * np.sin(gamma)) * dt
    x *= 1.5e6 / x[-1] * (1 - 0.01 * wind)                       # scale to ~1,500 km downrange / 150 km altitude
    z *= 1.5e5 / z[-1]
    y = wind * 15000 * u ** 2
    pos = np.column_stack([x, y, z])
    vel = np.gradient(pos, t, axis=0)
    nose = vel / np.linalg.norm(vel, axis=1, keepdims=True)
    nose[0] = [0.0, 0.0, 1.0]
    q = quat_from_nose(nose + [0, 0, 1e-4], 0.05 * t)
    speed = np.linalg.norm(vel, axis=1)
    qdyn = 0.5 * 1.225 * np.exp(-z / 7200) * speed ** 2
    i_q = int(np.argmax(qdyn))
    ev = {"Liftoff": 0.0, "Max-Q": t[i_q], "MECO": 162.0, "Stage separation": 166.0, "SES": 172.0, "Fairing sep": 215.0, "SECO": T}
    return {
        "id": vid, "label": label, "t": _r(t, 2),
        "rocket": {"pos": _r(pos, 1), "quat": _r(q, 5)},
        "scene": {"frame": "LAUNCH", "axis_labels": ["Downrange [m]", "Crossrange [m]", "Altitude [m]"],
                  "views2d": [{"title": "Altitude vs downrange", "x": "downrange", "y": 2}], "ground": True,
                  "glyph": "rocket", "glyph_length_m": 70},
        "objects": [{"name": "Insertion target", "kind": "waypoints", "pos": [[float(x[-1]), float(y[-1]), 150000.0]]}],
        "events": _events(ev),
        "plots": [
            {"id": "alt", "title": "Altitude", "xlabel": "t [s]", "ylabel": "km", "series": [_series("altitude", t, z / 1e3)]},
            {"id": "speed", "title": "Speed", "xlabel": "t [s]", "ylabel": "m/s", "series": [_series("speed", t, speed)]},
            {"id": "q", "title": "Dynamic pressure", "xlabel": "t [s]", "ylabel": "kPa", "series": [_series("q", t, qdyn / 1e3)]},
        ],
        "metrics": {"Insertion altitude [m]": round(float(z[-1]), 0), "Downrange [m]": round(float(x[-1]), 0),
                    "Max-Q [kPa]": round(float(qdyn.max() / 1e3), 1), "Max-Q time [s]": round(float(t[i_q]), 1)},
        "notes": "Mock ascent: flight-path angle falls from 90 to 2 deg; positions are launch-frame ENU (downrange, crossrange, up).",
    }


def ascent_orbital_demo():
    return {"id": "ascent_orbital", "group": "Orbital ascent", "title": "Orbital launch ascent",
            "description": "Two-stage ascent to a ~150 km insertion point about 1,500 km downrange, with max-Q, MECO, "
                           "staging and SECO events.\n\nThe scene is very wide and low, so the 3D view defaults to a stretched scale.",
            "variants": [ascent_orbital_variant("nominal", "Nominal", 0.0), ascent_orbital_variant("crosswind", "Crosswind", 1.0)]}


# --------------------------------------------------------------------------------------------- docking (LVLH)
def docking_variant(vid, label, noise):
    holds = [1000.0, 250.0, 60.0]
    k, v0, r0, dt, dwell = 6e-4, 0.05, 3000.0, 2.0, 120.0
    r, t, i, w = r0, 0.0, 0, 0.0
    ts, rs, hold_t = [], [], {}
    while r > 11.0:
        ts.append(t); rs.append(r)
        if i < len(holds) and r <= holds[i]:
            w += dt                                          # dwell at the hold point
            hold_t.setdefault(i, t)
            if w >= dwell:
                i += 1; w = 0.0
        else:
            r -= (k * r + v0) * dt
        t += dt
    ts.append(t); rs.append(r)
    t, r = np.array(ts), np.array(rs)
    rr = np.append(np.diff(r) / np.diff(t), 0.0)
    a = 1 + 2 * noise
    x = 14 * a * np.sin(t / 310.0) * (r / r0) ** 0.8             # R-bar offset inside the corridor
    z = 9 * a * np.sin(t / 240.0 + 1) * (r / r0) ** 0.8          # H-bar offset
    pos = np.column_stack([x, r, z])
    nose = np.column_stack([np.zeros_like(r), -np.ones_like(r), np.zeros_like(r)]) + [0.0, 0.0, 1e-3]
    q = quat_from_nose(nose, 0.004 * t)
    lat = np.hypot(x, z)
    ev = {"Phasing hand-over": 0.0, "Hold 1": hold_t[0], "Enter keep-out sphere": float(t[np.argmax(r < 200)]),
          "Hold 2": hold_t[1], "Hold 3": hold_t[2], "Contact": float(t[-1])}
    return {
        "id": vid, "label": label, "t": _r(t, 2),
        "rocket": {"pos": _r(pos, 2), "quat": _r(q, 5)},
        "scene": {"frame": "LVLH", "axis_labels": ["R-bar [m]", "V-bar [m]", "H-bar [m]"],
                  "views2d": [{"title": "V-bar / R-bar", "x": 1, "y": 0}, {"title": "V-bar / H-bar", "x": 1, "y": 2}],
                  "ground": False, "glyph": "spacecraft", "glyph_length_m": 10},
        "objects": [{"name": "ISS", "kind": "path", "pos": [[0, 0, 0]]},
                    {"name": "Keep-out sphere (200 m)", "kind": "sphere", "center": [0, 0, 0], "radius": 200},
                    {"name": "Hold points", "kind": "waypoints", "pos": [[0, h, 0] for h in holds]},
                    {"name": "Approach corridor", "kind": "path", "pos": [[0, r0, 0], [0, 10, 0]]}],
        "events": _events(ev),
        "plots": [
            {"id": "range", "title": "Range to station", "xlabel": "t [s]", "ylabel": "m", "series": [_series("range", t, r)]},
            {"id": "rate", "title": "Range rate", "xlabel": "t [s]", "ylabel": "m/s", "series": [_series("range rate", t, rr)]},
            {"id": "lat", "title": "Lateral offset from corridor", "xlabel": "t [s]", "ylabel": "m",
             "series": [_series("R-bar", t, x), _series("H-bar", t, z), _series("total", t, lat)]},
        ],
        "metrics": {"Duration [s]": round(float(t[-1]), 0), "Contact speed [m/s]": round(float(abs(rr[-2])), 3),
                    "Contact lateral offset [m]": round(float(lat[-1]), 2), "Max lateral offset [m]": round(float(lat.max()), 1)},
        "notes": "Chaser decelerates from 3 km along +V-bar with dwells at three hold points; closing speed stays below 2 m/s.",
    }


def docking_demo():
    return {"id": "docking_lvlh", "group": "Rendezvous", "title": "Docking approach (LVLH)",
            "description": "Spacecraft approach along the +V-bar corridor of the station through three hold points to "
                           "soft contact, with the 200 m keep-out sphere.\n\nPositions are station-relative LVLH metres.",
            "variants": [docking_variant("nominal", "Nominal", 0.0), docking_variant("noisy", "Noisy navigation", 1.0)]}


# --------------------------------------------------------------------------------------------- analysis
def analysis_variant(vid, label, sigma_scale):
    ol = np.array([1.2 * sigma_scale, -1.6 * sigma_scale, 0.0, 0.0, -25.0])
    cl = np.array([-4 + 3j, -4 - 3j, -6, -8, -25])
    noise = np.array([0.25, 0.5, 1, 2, 4, 8])
    luen = 0.08 * noise ** 0.9 * sigma_scale
    kal = 0.05 * noise ** 0.75 * sigma_scale
    kal[-1] = np.nan                                   # e.g. a diverged run: must show as a gap
    return {
        "id": vid, "label": label, "t": [], "rocket": None,
        "plots": [
            {"id": "poles", "title": "Pole map (pitch plane)", "xlabel": "Re [1/s]", "ylabel": "Im [rad/s]",
             "series": [_series("open loop", ol.real, np.zeros_like(ol.real), "markers"),
                        _series("closed loop", cl.real, cl.imag, "markers")]},
            {"id": "noise", "title": "Attitude RMSE vs sensor noise", "xlabel": "noise std multiplier [-]", "ylabel": "RMSE [deg]",
             "series": [_series("Luenberger", noise, luen, "lines+markers"), _series("Kalman", noise, kal, "lines+markers")]},
        ],
        "metrics": {"Open-loop unstable poles": 1, "Closed-loop damping [-]": 0.8, "Kalman RMSE @1x [deg]": round(float(kal[2]), 4),
                    "Phase margin [deg]": float("nan")},
        "tables": [{"title": "Controllability / observability", "columns": ["Model", "States", "rank ctrb", "rank obsv", "Verdict"],
                    "rows": [["pitch", 4, 4, 4, "controllable, observable"],
                             ["yaw", 4, 4, 4, "controllable, observable"],
                             ["roll", 2, 0, 2, "uncontrollable (single gimbal)"]]}],
        "notes": f"Linearised about the {label.lower()} operating point (finite-difference Jacobians of gnc.dynamics).",
    }


def analysis_demo():
    return {"id": "linear_analysis", "group": "Analysis", "title": "Linear analysis",
            "description": "Pole locations, controllability/observability ranks and observer error versus sensor noise.",
            "variants": [analysis_variant("maxq", "Max-Q", 1.0), analysis_variant("burnout", "Burnout", 0.6)]}


def all_demos():
    return [passive_demo(), waypoint_demo(), ascent_orbital_demo(), docking_demo(), analysis_demo()]


def write_mock(out="viz/mock"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    names = []
    for demo in all_demos():
        names.append(demo["id"] + ".json")
        (out / names[-1]).write_text(json.dumps(demo))   # NaN written as NaN on purpose (builder must cope)
    (out / "index.json").write_text(json.dumps({"demos": names}, indent=1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="viz/mock")
    print("wrote", write_mock(ap.parse_args().out))
