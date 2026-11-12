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


class InterceptGuidance:
    """3D intercept guidance against a target (p_t, v_t): Pure PN or LOS pursuit.

    Modes:
      - "pn": Pure Proportional Navigation, a = k * (omega_los x v), perpendicular to v.
      - "los": Line-of-Sight pursuit via steer_toward(r, v, k), perpendicular to v.
    """

    def __init__(self, mode="pn", k=3.0, a_max=np.inf):
        assert mode in ("pn", "los"), f"unknown mode: {mode}"
        self.mode, self.k, self.a_max = mode, k, a_max

    def command(self, p, v, p_t, v_t=None):
        p, v = np.asarray(p, float), np.asarray(v, float)
        r = np.asarray(p_t, float) - p
        v_rel = (np.zeros(3) if v_t is None else np.asarray(v_t, float)) - v
        r_norm = float(np.linalg.norm(r))
        if r_norm < 1e-9:
            return np.zeros(3), {"dist": 0.0, "closing_vel": 0.0, "los_rate": 0.0, "heading_error": 0.0}
        r_hat = r / r_norm
        omega_los = np.cross(r, v_rel) / (r_norm ** 2)
        if self.mode == "pn":
            a = self.k * np.cross(omega_los, v)
            _, eps = steer_toward(r, v, 1.0)
        else:
            a, eps = steer_toward(r, v, self.k)
        a = a * min(1.0, self.a_max / max(float(np.linalg.norm(a)), 1e-12))
        return a, {
            "dist": r_norm,
            "closing_vel": float(-r_hat @ v_rel),
            "los_rate": float(np.linalg.norm(omega_los)),
            "heading_error": eps,
        }


def kinematic_intercept_check(guidance, p0, v0, target, dt=0.01, t_max=30.0, hit_radius=1.0, accel=None):
    """Point-mass kinematic intercept check. target(t) -> (p_t, v_t)."""
    p, v = np.array(p0, float), np.array(v0, float)
    out = {"t": [], "p": [], "v": [], "p_t": [], "a": [], "info": []}
    for k in range(int(round(t_max / dt))):
        t = k * dt
        p_t, v_t = target(t)
        a, info = guidance.command(p, v, p_t, v_t)
        for key, val in (("t", t), ("p", p.copy()), ("v", v.copy()), ("p_t", np.asarray(p_t, float)),
                         ("a", a), ("info", info)):
            out[key].append(val)
        if info["dist"] <= hit_radius:
            break
        acc = a + (accel(t, v) if accel is not None else 0.0)
        p = p + v * dt + 0.5 * acc * dt**2
        v = v + acc * dt
    out = {k: (np.array(val) if k != "info" else val) for k, val in out.items()}
    out["control_effort"] = float(np.sum(out["a"] ** 2) * dt)
    out["miss_dist"] = float(out["info"][-1]["dist"])
    return out


def demo_intercept_guidance(save_path=None, show=True):
    """Compare 3D PN vs LOS intercept guidance against a circularly moving target."""
    import matplotlib.pyplot as plt

    def target(t):
        w = 20.0 / 200.0
        return (np.array([200.0 * np.cos(w * t), 200.0 * np.sin(w * t), 500.0]),
                np.array([-20.0 * np.sin(w * t), 20.0 * np.cos(w * t), 0.0]))

    p0, v0 = [0.0, 0.0, 0.0], [0.0, 0.0, 30.0]
    pn = kinematic_intercept_check(InterceptGuidance("pn", k=3.0, a_max=50.0), p0, v0, target)
    los = kinematic_intercept_check(InterceptGuidance("los", k=2.0, a_max=50.0), p0, v0, target)

    fig = plt.figure(figsize=(12, 4.5))
    ax3 = fig.add_subplot(131, projection="3d")
    t_ref = np.linspace(0.0, max(pn["t"][-1], los["t"][-1]), 300)
    pt_ref = np.array([target(t)[0] for t in t_ref])
    ax3.plot(*pt_ref.T, "k--", lw=1.2, label="Target")
    ax3.plot(*pn["p"].T, lw=2, label=f"PN ({pn['t'][-1]:.1f}s)")
    ax3.plot(*los["p"].T, lw=2, label=f"LOS ({los['t'][-1]:.1f}s)")
    ax3.set(xlabel="East [m]", ylabel="North [m]", zlabel="Up [m]", title="3D Intercept Trajectory")
    ax3.legend(fontsize=8)

    ax1 = fig.add_subplot(132)
    ax1.plot(pn["t"], [i["dist"] for i in pn["info"]], lw=2, label="PN")
    ax1.plot(los["t"], [i["dist"] for i in los["info"]], lw=2, ls="--", label="LOS")
    ax1.set(xlabel="Time [s]", ylabel="Range to Target [m]", title="Miss Distance")
    ax1.grid(alpha=0.3); ax1.legend()

    ax2 = fig.add_subplot(133)
    ax2.plot(pn["t"], np.linalg.norm(pn["a"], axis=1), lw=2, label=f"PN (effort={pn['control_effort']:.0f})")
    ax2.plot(los["t"], np.linalg.norm(los["a"], axis=1), lw=2, ls="--", label=f"LOS (effort={los['control_effort']:.0f})")
    ax2.set(xlabel="Time [s]", ylabel="|a_cmd| [m/s²]", title="Lateral Accel Command")
    ax2.grid(alpha=0.3); ax2.legend(fontsize=8)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    print(f"PN : miss={pn['miss_dist']:.2f} m, t_hit={pn['t'][-1]:.2f} s, effort={pn['control_effort']:.1f}")
    print(f"LOS: miss={los['miss_dist']:.2f} m, t_hit={los['t'][-1]:.2f} s, effort={los['control_effort']:.1f}")
    return pn, los


if __name__ == "__main__":
    demo_intercept_guidance()
