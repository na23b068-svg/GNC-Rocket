"""Live-generated guidance demos for the dashboard, built directly from gnc.guidance."""
import numpy as np

from gnc.frames import quat_from_axis_angle
from gnc.guidance import (
    InterceptGuidance,
    LOSPathGuidance,
    kinematic_intercept_check,
    kinematic_path_check,
    los_angles,
    steer_toward,
)

from scenarios.common import AGGRESSIVE_WP


def _vel_to_quat(v):
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return [1.0, 0.0, 0.0, 0.0]
    vh = v / n
    c = np.cross([1.0, 0.0, 0.0], vh)
    cn, d = float(np.linalg.norm(c)), float(np.dot([1.0, 0.0, 0.0], vh))
    if cn > 1e-9:
        q = quat_from_axis_angle(c, float(np.arctan2(cn, d)))
    elif d < 0.0:
        q = quat_from_axis_angle([0.0, 0.0, 1.0], np.pi)
    else:
        q = np.array([1.0, 0.0, 0.0, 0.0])
    return q.tolist()


def _primitives_demo():
    s = np.linspace(0.0, 1.0, 180)
    az_in = np.linspace(-np.pi, np.pi, len(s))
    el_in = np.radians(60.0) * np.sin(2.0 * np.pi * s)
    pts = np.column_stack([np.cos(el_in) * np.cos(az_in), np.cos(el_in) * np.sin(az_in), np.sin(el_in)])
    az_out, el_out = zip(*(los_angles(r) for r in pts))
    az111, el111 = los_angles(np.array([1.0, 1.0, 1.0]))

    v_ref = np.array([100.0, 0.0, 0.0])
    angles_deg = np.linspace(0.0, 175.0, 90)
    dirs = [np.array([np.cos(np.radians(a)), np.sin(np.radians(a)), 0.0]) for a in angles_deg]
    mag_series, ortho_series = [], []
    for k in (0.5, 1.0, 2.0, 3.0):
        cmds = [steer_toward(d, v_ref, k)[0] for d in dirs]
        mag_series.append({"name": f"k={k}", "x": angles_deg.tolist(),
                           "y": [float(np.linalg.norm(c)) for c in cmds], "mode": "lines"})
        if k == 1.0:
            ortho_series.append({"name": "|a_cmd . v| (k=1.0)", "x": angles_deg.tolist(),
                                 "y": [float(abs(c @ v_ref)) for c in cmds], "mode": "lines"})

    return {
        "id": "guidance_primitives",
        "group": "Guidance (guidance.py)",
        "title": "Guidance primitives: los_angles & steer_toward",
        "description": "Direct functional checks of los_angles(r) and steer_toward(d, v, k) from gnc/guidance.py.",
        "variants": [
            {
                "id": "los_angles",
                "label": "los_angles(r)",
                "t": None, "rocket": None, "objects": [], "events": [],
                "plots": [
                    {"id": "los_az_sweep", "title": "los_angles(r): azimuth over 3D spiral",
                     "xlabel": "sample", "ylabel": "deg",
                     "series": [{"name": "azimuth [deg]", "x": s.tolist(), "y": np.degrees(az_out).tolist(), "mode": "lines"}]},
                    {"id": "los_el_sweep", "title": "los_angles(r): elevation over 3D spiral",
                     "xlabel": "sample", "ylabel": "deg",
                     "series": [{"name": "elevation [deg]", "x": s.tolist(), "y": np.degrees(el_out).tolist(), "mode": "lines"}]},
                ],
                "metrics": {
                    "Max |az| [deg]": round(float(np.max(np.abs(np.degrees(az_out)))), 2),
                    "Max |el| [deg]": round(float(np.max(np.abs(np.degrees(el_out)))), 2),
                    "los_angles([1,1,1]) az [deg]": round(float(np.degrees(az111)), 2),
                    "los_angles([1,1,1]) el [deg]": round(float(np.degrees(el111)), 2),
                },
            },
            {
                "id": "steer_toward",
                "label": "steer_toward(d, v, k)",
                "t": None, "rocket": None, "objects": [], "events": [],
                "plots": [
                    {"id": "steer_mag", "title": "Lateral accel |a_cmd| vs heading error (|v|=100 m/s)",
                     "xlabel": "heading error [deg]", "ylabel": "m/s²", "series": mag_series},
                    {"id": "steer_ortho", "title": "Orthogonality |a_cmd . v|",
                     "xlabel": "heading error [deg]", "ylabel": "m²/s³", "series": ortho_series},
                ],
                "metrics": {
                    "|a_cmd| at 45 deg, k=2, |v|=100 [m/s^2]": round(float(np.linalg.norm(steer_toward([1, 0, 1], [0, 0, 100], 2.0)[0])), 2),
                    "Max |a_cmd . v|": round(float(max(ortho_series[0]["y"])), 12),
                },
            },
        ],
    }


def _los_kinematic_demo():
    variants = []
    configs = [
        ("unconstrained", "LOSPathGuidance (unbounded a_max)", dict(lookahead_m=100.0, lookahead_time=0.0, accept_radius=30.0, k=1.5), np.inf),
        ("bounded_time_lookahead", "LOSPathGuidance (a_max=25, lookahead_time=1s)", dict(lookahead_m=100.0, lookahead_time=1.0, accept_radius=30.0, k=1.5), 25.0),
    ]
    for vid, label, kw, a_max in configs:
        g = LOSPathGuidance(AGGRESSIVE_WP, **kw)
        out = kinematic_path_check(g, [0, 0, 0], [0, 0, 150.0], dt=0.01, t_max=7.5, a_max=a_max)
        t = out["t"].tolist()
        vel = np.gradient(out["p"], out["t"], axis=0)
        xte = [float(i["cross_track"]) for i in out["info"]]
        xte_h = [float(i["cross_track_h"]) for i in out["info"]]
        xte_v = [float(i["cross_track_v"]) for i in out["info"]]
        amag = np.linalg.norm(out["a"], axis=1).tolist()
        az = [float(np.degrees(i["los_az"])) for i in out["info"]]
        el = [float(np.degrees(i["los_el"])) for i in out["info"]]
        eps = [float(np.degrees(i["heading_error"])) for i in out["info"]]
        wp = [int(i["wp_index"]) for i in out["info"]]
        dwp = [float(i["dist_to_wp"]) for i in out["info"]]
        variants.append({
            "id": vid, "label": label, "t": t,
            "rocket": {"pos": out["p"].tolist(), "quat": [_vel_to_quat(v) for v in vel]},
            "objects": [{"name": "Waypoints", "kind": "waypoints", "pos": AGGRESSIVE_WP},
                        {"name": "Reference path", "kind": "path", "pos": AGGRESSIVE_WP}],
            "events": [{"t": 0.0, "label": "Start"}],
            "plots": [
                {"id": "xte", "title": "Cross-track error (LOSPathGuidance)", "xlabel": "t [s]", "ylabel": "m",
                 "series": [{"name": "total |e|", "x": t, "y": xte, "mode": "lines"},
                            {"name": "horizontal", "x": t, "y": xte_h, "mode": "lines"},
                            {"name": "vertical-plane", "x": t, "y": xte_v, "mode": "lines"}]},
                {"id": "acmd", "title": "Commanded lateral acceleration |a_cmd|", "xlabel": "t [s]", "ylabel": "m/s²",
                 "series": [{"name": "|a_cmd|", "x": t, "y": amag, "mode": "lines"}]},
                {"id": "los_ang", "title": "LOS angles (az & el)", "xlabel": "t [s]", "ylabel": "deg",
                 "series": [{"name": "los_az", "x": t, "y": az, "mode": "lines"},
                            {"name": "los_el", "x": t, "y": el, "mode": "lines"}]},
                {"id": "head_err", "title": "Heading error & active waypoint index", "xlabel": "t [s]", "ylabel": "deg | idx",
                 "series": [{"name": "heading_error [deg]", "x": t, "y": eps, "mode": "lines"},
                            {"name": "wp_index", "x": t, "y": wp, "mode": "lines"}]},
                {"id": "dist_wp", "title": "Distance to active segment end waypoint", "xlabel": "t [s]", "ylabel": "m",
                 "series": [{"name": "dist_to_wp", "x": t, "y": dwp, "mode": "lines"}]},
            ],
            "metrics": {
                "RMS cross-track [m]": round(float(np.sqrt(np.mean(np.square(xte)))), 2),
                "Max cross-track [m]": round(float(max(xte)), 2),
                "Control effort [m^2/s^3]": round(out["control_effort"], 1),
                "Max |a_cmd| [m/s^2]": round(float(max(amag)), 2),
            },
        })
    return {
        "id": "guidance_los_kinematic",
        "group": "Guidance (guidance.py)",
        "title": "LOSPathGuidance & kinematic_path_check",
        "description": "3D point-mass kinematic path-following executing LOSPathGuidance and kinematic_path_check().",
        "variants": variants,
    }


def _circular_target(t):
    w = 20.0 / 200.0
    return (np.array([200.0 * np.cos(w * t), 200.0 * np.sin(w * t), 500.0]),
            np.array([-20.0 * np.sin(w * t), 20.0 * np.cos(w * t), 0.0]))


def _crossing_target(t):
    vt = np.array([-15.0, 20.0, 0.0])
    return np.array([300.0, -150.0, 450.0]) + vt * t, vt


def _intercept_variant(vid, label, mode, k, target_fn, target_name, notes, a_max=50.0):
    out = kinematic_intercept_check(InterceptGuidance(mode, k=k, a_max=a_max), [0.0, 0.0, 0.0], [0.0, 0.0, 30.0], target_fn)
    t = out["t"].tolist()
    dist = [float(i["dist"]) for i in out["info"]]
    vc = [float(i["closing_vel"]) for i in out["info"]]
    omega = [float(np.degrees(i["los_rate"])) for i in out["info"]]
    eps = [float(np.degrees(i["heading_error"])) for i in out["info"]]
    amag = np.linalg.norm(out["a"], axis=1).tolist()
    t_hit = float(out["t"][-1])
    return {
        "id": vid, "label": label, "t": t,
        "rocket": {"pos": out["p"].tolist(), "quat": [_vel_to_quat(v) for v in out["v"]]},
        "objects": [{"name": target_name, "kind": "moving", "pos": out["p_t"].tolist()}],
        "events": [{"t": 0.0, "label": "Launch (v0 = 30 m/s Up)"},
                   {"t": round(t_hit, 2), "label": f"Intercept (miss {out['miss_dist']:.2f} m)"}],
        "plots": [
            {"id": "range", "title": f"Range to target ({mode.upper()})", "xlabel": "t [s]", "ylabel": "m",
             "series": [{"name": "range |r|", "x": t, "y": dist, "mode": "lines"}]},
            {"id": "accel", "title": f"Commanded lateral acceleration |a_cmd| (limit {a_max:.0f} m/s²)",
             "xlabel": "t [s]", "ylabel": "m/s²",
             "series": [{"name": "|a_cmd|", "x": t, "y": amag, "mode": "lines"},
                        {"name": f"a_max ({a_max:.0f})", "x": [t[0], t[-1]], "y": [a_max, a_max], "mode": "lines"}]},
            {"id": "kinematics", "title": "Closing velocity & LOS angular rate", "xlabel": "t [s]", "ylabel": "m/s | deg/s",
             "series": [{"name": "closing vel Vc [m/s]", "x": t, "y": vc, "mode": "lines"},
                        {"name": "LOS rate |omega| [deg/s]", "x": t, "y": omega, "mode": "lines"}]},
            {"id": "heading", "title": "Heading error to LOS", "xlabel": "t [s]", "ylabel": "deg",
             "series": [{"name": "heading error [deg]", "x": t, "y": eps, "mode": "lines"}]},
        ],
        "metrics": {
            "Miss distance [m]": round(out["miss_dist"], 3),
            "Intercept time [s]": round(t_hit, 2),
            "Control effort [m^2/s^3]": round(out["control_effort"], 1),
            "Peak |a_cmd| [m/s^2]": round(float(max(amag)), 2),
            "Max LOS rate [deg/s]": round(float(max(omega)), 2),
        },
        "notes": notes,
    }, out


def _intercept_demo():
    v_pn, out_pn = _intercept_variant(
        "pn_circular", "PN: Circular target (k=3.0)", "pn", 3.0, _circular_target, "Target (circle 20 m/s)",
        "3D Pure Proportional Navigation: a_cmd = k * (omega_los x v), strictly perpendicular to v. "
        "Leads onto a near-collision triangle and intercepts cleanly.")
    v_los, out_los = _intercept_variant(
        "los_circular", "LOS: Circular target (k=2.0)", "los", 2.0, _circular_target, "Target (circle 20 m/s)",
        "3D Line-of-Sight pursuit: steer_toward(r, v, k). Turns velocity directly onto the instantaneous LOS, "
        "producing a longer tail-chase and higher control effort than PN.")
    v_cross, _ = _intercept_variant(
        "pn_crossing", "PN: Crossing linear target (k=4.0)", "pn", 4.0, _crossing_target, "Target (linear 25 m/s)",
        "3D Pure PN against a constant-velocity crossing target. Lateral accel decays toward zero on the collision triangle.")

    t_pn, t_los = out_pn["t"].tolist(), out_los["t"].tolist()
    a_pn = np.linalg.norm(out_pn["a"], axis=1).tolist()
    a_los = np.linalg.norm(out_los["a"], axis=1).tolist()
    v_cmp = {
        "id": "pn_vs_los_comparison", "label": "PN vs LOS comparison (circular target)",
        "t": None, "rocket": None, "objects": [], "events": [],
        "plots": [
            {"id": "cmp_range", "title": "Range to target: PN vs LOS", "xlabel": "t [s]", "ylabel": "m",
             "series": [{"name": "PN (k=3.0)", "x": t_pn, "y": [float(i["dist"]) for i in out_pn["info"]], "mode": "lines"},
                        {"name": "LOS (k=2.0)", "x": t_los, "y": [float(i["dist"]) for i in out_los["info"]], "mode": "lines"}]},
            {"id": "cmp_accel", "title": "Lateral acceleration |a_cmd|: PN vs LOS", "xlabel": "t [s]", "ylabel": "m/s²",
             "series": [{"name": f"PN (effort={out_pn['control_effort']:.1f})", "x": t_pn, "y": a_pn, "mode": "lines"},
                        {"name": f"LOS (effort={out_los['control_effort']:.1f})", "x": t_los, "y": a_los, "mode": "lines"}]},
            {"id": "cmp_omega", "title": "LOS angular rate |omega_los|: PN vs LOS", "xlabel": "t [s]", "ylabel": "deg/s",
             "series": [{"name": "PN", "x": t_pn, "y": [float(np.degrees(i["los_rate"])) for i in out_pn["info"]], "mode": "lines"},
                        {"name": "LOS", "x": t_los, "y": [float(np.degrees(i["los_rate"])) for i in out_los["info"]], "mode": "lines"}]},
        ],
        "tables": [{
            "title": "Circular target intercept summary (p0=[0,0,0], v0=[0,0,30], a_max=50 m/s²)",
            "columns": ["Mode", "Gain k", "Miss [m]", "Intercept time [s]", "Control effort [m²/s³]", "Peak |a_cmd| [m/s²]"],
            "rows": [
                ["Pure PN", "3.0", f"{out_pn['miss_dist']:.2f}", f"{t_pn[-1]:.2f}", f"{out_pn['control_effort']:.1f}", f"{max(a_pn):.2f}"],
                ["LOS Pursuit", "2.0", f"{out_los['miss_dist']:.2f}", f"{t_los[-1]:.2f}", f"{out_los['control_effort']:.1f}", f"{max(a_los):.2f}"],
            ],
        }],
        "metrics": {
            "PN miss [m]": round(out_pn["miss_dist"], 3), "LOS miss [m]": round(out_los["miss_dist"], 3),
            "PN intercept time [s]": round(float(t_pn[-1]), 2), "LOS intercept time [s]": round(float(t_los[-1]), 2),
            "PN control effort": round(out_pn["control_effort"], 1), "LOS control effort": round(out_los["control_effort"], 1),
        },
        "notes": "Direct comparison of InterceptGuidance('pn') vs InterceptGuidance('los') on the circular target demo.",
    }
    return {
        "id": "guidance_intercept",
        "group": "Guidance (guidance.py)",
        "title": "InterceptGuidance: 3D PN vs LOS",
        "description": "3D point-mass intercept guidance from gnc/guidance.py comparing Pure PN against LOS pursuit.",
        "variants": [v_pn, v_los, v_cmp, v_cross],
    }


def guidance_demos():
    """Return validated guidance demo dicts built live from gnc.guidance."""
    return [_primitives_demo(), _los_kinematic_demo(), _intercept_demo()]
