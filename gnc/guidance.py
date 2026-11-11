"""Guidance (3D, ENU): lookahead-based LOS path following over a waypoint list for the powered ascent.

LOSPathGuidance.command(p, v) returns a lateral-acceleration command a_cmd (m/s^2, ENU, perpendicular to the
vehicle velocity) plus a dict of logged quantities. Gravity compensation and the conversion
acceleration -> attitude command happen in gnc.flight_computer.
"""
import numpy as np


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.zeros_like(v)


def los_angles(r):
    """LOS azimuth (from East, CCW, about Up) and elevation (above horizon) of vector r, rad."""
    return float(np.arctan2(r[1], r[0])), float(np.arctan2(r[2], np.hypot(r[0], r[1])))


def steer_toward(d, v, k):
    """Lateral accel (perpendicular to v) rotating the velocity toward direction d at rate k * angle."""
    vh, dh = _unit(v), _unit(d)
    perp = dh - (dh @ vh) * vh
    eps = float(np.arctan2(np.linalg.norm(perp), dh @ vh))   # heading error angle
    return k * np.linalg.norm(v) * eps * _unit(perp), eps


class LOSPathGuidance:
    """Lookahead LOS on a 3D waypoint polyline. On segment W_i -> W_{i+1} (unit tangent t):
    along-track s = (p - W_i).t, cross-track e = (p - W_i) - s t, lookahead point P = W_i + (s + Delta) t,
    Delta = max(lookahead_m, lookahead_time * |v|); the velocity is steered toward P (LOS angle) with
    a = k |v| eps e_perp, i.e. a commanded flight-path turn rate k * eps.
    Switch to the next segment when |p - W_{i+1}| < accept_radius or s >= segment length."""

    def __init__(self, waypoints, lookahead_m=100.0, lookahead_time=0.0, accept_radius=30.0, k=1.5,
                 a_max=np.inf):
        self.W = np.asarray(waypoints, float)
        assert self.W.ndim == 2 and self.W.shape[1] == 3 and len(self.W) >= 2, "need >= 2 waypoints (n,3)"
        self.lookahead_m, self.lookahead_time = lookahead_m, lookahead_time
        self.accept_radius, self.k, self.a_max = accept_radius, k, a_max
        self.i = 0

    def reset(self):
        self.i = 0

    def command(self, p, v):
        W = self.W
        while self.i < len(W) - 2:
            seg = W[self.i + 1] - W[self.i]
            if np.linalg.norm(p - W[self.i + 1]) < self.accept_radius or (p - W[self.i]) @ seg >= seg @ seg:
                self.i += 1
            else:
                break
        a0, a1 = W[self.i], W[self.i + 1]
        t = _unit(a1 - a0)
        d = p - a0
        s = float(d @ t)
        e = d - s * t
        Delta = max(self.lookahead_m, self.lookahead_time * np.linalg.norm(v))
        los = a0 + (s + Delta) * t - p
        a, eps = steer_toward(los, v, self.k)
        a = a * min(1.0, self.a_max / max(np.linalg.norm(a), 1e-12))
        az, el = los_angles(los)
        # cross-track split: horizontal (ground plane, perpendicular to t) and the in-vertical-plane remainder
        h = _unit(np.cross([0.0, 0.0, 1.0], t)) if np.linalg.norm(t[:2]) > 1e-6 else np.array([1.0, 0, 0])
        return a, {"wp_index": self.i, "cross_track": float(np.linalg.norm(e)),
                   "cross_track_h": float(e @ h), "cross_track_v": float(e @ _unit(np.cross(t, h))),
                   "along_track": s, "los_az": az, "los_el": el, "heading_error": eps,
                   "dist_to_wp": float(np.linalg.norm(p - a1))}


def kinematic_path_check(guidance, p0, v0, dt=0.01, t_max=10.0, a_max=np.inf, accel=None):
    """Point-mass kinematic check of path following: v' = a_cmd (+ accel(t, v), e.g. thrust along v).
    Returns time histories, logged info and control effort (integral of |a|^2)."""
    p, v = np.array(p0, float), np.array(v0, float)
    out = {"t": [], "p": [], "a": [], "info": []}
    for k in range(int(round(t_max / dt))):
        t = k * dt
        a, info = guidance.command(p, v)
        a = a * min(1.0, a_max / max(np.linalg.norm(a), 1e-12))
        for key, val in (("t", t), ("p", p.copy()), ("a", a), ("info", info)):
            out[key].append(val)
        acc = a + (accel(t, v) if accel is not None else 0.0)
        p = p + v * dt + 0.5 * acc * dt**2
        v = v + acc * dt
    out = {k: (np.array(val) if k != "info" else val) for k, val in out.items()}
    out["control_effort"] = float(np.sum(out["a"] ** 2) * dt)
    return out
