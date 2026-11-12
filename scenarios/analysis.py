"""System analysis demo: linearisation, controllability/observability, PID gain schedule, delay margins.

    .venv/bin/python -m scenarios.analysis        -> results/analysis.json  (analysis-only, rocket = null)
"""
import math

import control as ct
import numpy as np

from gnc import analysis as an
from gnc.flight_computer import design_schedule, tvc_end_time
from gnc.linearize import STATE_NAMES, operating_points
from gnc.observers import attitude_plane_model, axis_nav_model
from gnc.pid import closed_loop_poles, specs_of
from scenarios.common import plot, rnd, rocket, series, write_demo

TAU_ACT = 0.02          # servo lag used in all loop analyses (matches SimConfig.actuator_tau_s)
STEP_IDX = (0, 4, 8)    # operating points for the step responses: rail exit, mid-boost, TVC end


def cstr(z, n=2):
    z = complex(z)
    if abs(z.imag) < 1e-9:
        return f"{z.real:.{n}f}"
    return f"{z.real:.{n}f} {'+' if z.imag >= 0 else '-'} {abs(z.imag):.{n}f}j"


def pole_list(p, n=2):
    p = sorted(np.asarray(p, complex), key=lambda c: (round(c.real, 6), c.imag))
    return ", ".join(cstr(c, n) for c in p if c.imag >= -1e-9)   # one of each conjugate pair


def gains_at(tab, i):
    return {"Kp": tab["Kp"][i], "Ki": tab["Ki"][i], "Kd": tab["Kd"][i]}


def _empty(vid, label):
    return {"id": vid, "label": label, "t": [], "rocket": None, "objects": [], "events": []}


# ------------------------------------------------------------------------------------------------ linearisation
def v_linearisation(pts, tab):
    v = _empty("linearisation", "Linearisation")
    rows_op, rows_ab, rows_ev = [], [], []
    ol_re, ol_im, cl_re, cl_im = [], [], [], []
    max_diff, max_cpl = 0.0, 0.0
    for i, p in enumerate(pts):
        f, A, B = p["flight"], p["pitch"]["A"], p["pitch"]["B"]
        Aa, Ba = p["analytic"]["A"], p["analytic"]["B"]
        diff = max(np.abs(A - Aa).max() / np.abs(A).max(), np.abs(B - Ba).max() / np.abs(B).max())
        max_diff, max_cpl = max(max_diff, diff), max(max_cpl, p["pitch"]["coupling"])
        rows_op.append([rnd(p["t"], 2), rnd(f["V"], 1), rnd(f["mach"], 3), rnd(f["qbar"] / 1e3, 2), rnd(f["T"], 0),
                        rnd(f["m"], 2), rnd(f["static_margin_m"], 3), rnd(f["l_gimbal_m"], 3)])
        rows_ab.append([rnd(p["t"], 2), rnd(A[1, 1], 3), rnd(A[1, 2], 1), rnd(A[3, 1], 4), rnd(A[3, 2], 1),
                        rnd(A[3, 3], 3), rnd(B[1, 0], 2), rnd(B[3, 0], 2), f"{diff:.1e}",
                        f"{p['pitch']['coupling']:.1e}"])
        ev = np.linalg.eigvals(A)
        cx = ev[np.abs(ev.imag) > 1e-6]
        re = ev[np.abs(ev.imag) <= 1e-6].real
        rows_ev.append([rnd(p["t"], 2), cstr(cx[np.argmax(cx.imag)]) if cx.size else "-",
                        f"{re.max():+.3f}", f"{re[np.argmin(np.abs(re))]:+.1e}",
                        rnd(np.abs(cx[0]) if cx.size else float("nan"), 2)])
        ol_re += list(ev.real); ol_im += list(ev.imag)
        cl = np.linalg.eigvals(an.plane_pid_closed_loop(A, B, *(tab[k][i] for k in ("Kp", "Ki", "Kd"))))
        cl_re += list(cl.real); cl_im += list(cl.imag)
    i_mid = 4
    A, B = pts[i_mid]["pitch"]["A"], pts[i_mid]["pitch"]["B"]
    rows_mat = [[f"d{n}/dt"] + [rnd(a, 4) for a in A[r]] + [rnd(B[r, 0], 3)]
                for r, n in enumerate(STATE_NAMES["pitch"])]
    t = np.array(tab["t"])
    v["plots"] = [
        plot("polemap", "Pole map, pitch plane: open loop vs PID closed loop (all 9 operating points)",
             "Im [rad/s]", [series("open loop (airframe)", ol_re, ol_im, 3, "markers"),
                            series("closed loop (PID, full 4-state plane + integrator)", cl_re, cl_im, 3,
                                   "markers")], xlabel="Re [1/s]"),
        plot("a0", "Weathercock stiffness a0 = -dq'/dtheta and damping a1 = -dq'/dq", "1/s^2, 1/s",
             [series("a0 [1/s^2]", t, tab["a0"], 2, "lines+markers"),
              series("a1 [1/s]", t, tab["a1"], 3, "lines+markers")], xlabel="time since launch [s]"),
        plot("b", "TVC effectiveness b = dq'/d(delta)", "1/s^2",
             [series("b", t, tab["b"], 2, "lines+markers")], xlabel="time since launch [s]"),
        plot("modes", "Open-loop modes vs time: short-period |s| and drift divergence rate",
             "rad/s, 1/s", [series("short-period natural freq. [rad/s]", t, [r[4] for r in rows_ev], 2,
                                   "lines+markers"),
                            series("drift mode (unstable real root) [1/s]", t, [float(r[2]) for r in rows_ev], 3,
                                   "lines+markers")], xlabel="time since launch [s]"),
    ]
    v["tables"] = [
        {"title": "Operating points on the nominal passive boost (calm air, noiseless)",
         "columns": ["t [s]", "V [m/s]", "Mach", "qbar [kPa]", "T [N]", "m [kg]", "static margin [m]",
                     "l_gimbal [m]"], "rows": rows_op},
        {"title": "Pitch-plane A, B (numeric Jacobian through gnc.dynamics), state [z_b, w, theta, q]",
         "columns": ["t [s]", "A_ww", "A_w,theta", "A_q,w", "A_q,theta", "A_qq", "B_w", "B_q",
                     "max rel. diff vs analytic", "coupling to other states"], "rows": rows_ab},
        {"title": f"Full pitch-plane model [A | B] at t = {tab['t'][i_mid]:.2f} s (qbar "
                  f"{tab['qbar'][i_mid] / 1e3:.1f} kPa)",
         "columns": ["row", "z_b", "w", "theta", "q", "delta_p"], "rows": rows_mat},
        {"title": "Open-loop eigenvalues, pitch plane (yaw plane identical by symmetry)",
         "columns": ["t [s]", "short period", "drift (real, max)", "lateral position", "|s_sp| [rad/s]"],
         "rows": rows_ev},
    ]
    v["metrics"] = {"Operating points": len(pts), "Max numeric vs analytic rel. diff": float(f"{max_diff:.2e}"),
                    "Max pitch/yaw cross-coupling (rel.)": float(f"{max_cpl:.2e}"),
                    "Max open-loop unstable eigenvalue [1/s]": rnd(max(float(r[2]) for r in rows_ev), 3),
                    "Max closed-loop Re(pole) excl. lateral position [1/s]":
                        rnd(sorted(cl_re)[-len(pts) - 1], 3)}
    v["notes"] = (
        "The 13-state nonlinear model is linearised by central differences through gnc.dynamics.derivatives at 9 "
        "points between rail exit (0.37 s) and the start of the thrust tail-off (TVC end, "
        f"{tvc_end_time(rocket()):.2f} s). The attitude error is a body-frame rotation vector, so the quaternion "
        "norm constraint never enters; expressing dp, dv in body axes splits the 12-state model into two "
        "identical 4-state planes (pitch [z_b, w, theta, q], yaw mirror image), with cross-coupling < 2e-4. "
        "The numeric Jacobian matches the analytic plane model to ~1e-10.\n"
        "Open loop: a lightly damped short-period pair (weathercock oscillation) whose frequency rises with "
        "dynamic pressure, a weakly unstable real 'drift' mode (thrusting weathercock divergence, fastest at low "
        "speed) and a zero root (lateral position). Closed loop (pole-placement PID on theta): every pole moves "
        "into the left half-plane except lateral position, which the guidance loop closes.")
    return v


# ------------------------------------------------------------------------------------------------ ctrb / obsv
SENSORS = [("gyro q", [[0, 0, 0, 1]]), ("attitude theta", [[0, 0, 1, 0]]), ("GPS velocity w", [[0, 1, 0, 0]]),
           ("GPS position z_b", [[1, 0, 0, 0]]), ("gyro + GPS position", [[0, 0, 0, 1], [1, 0, 0, 0]])]


def v_ctrb_obsv(pts, tab):
    v = _empty("ctrb_obsv", "Controllability & observability")
    t = np.array(tab["t"])
    rows, cond = [], {"ctrb (delta_p)": []}
    for name, _ in SENSORS:
        cond[f"obsv: {name}"] = []
    gram = []
    for p in pts:
        A, B = p["pitch"]["A"], p["pitch"]["B"]
        rc = an.rank_test(A, B, "ctrb")
        row = [rnd(p["t"], 2), f"{rc['rank']}/4", f"{rc['cond']:.1e}"]
        cond["ctrb (delta_p)"].append(math.log10(rc["cond"]))
        for name, C in SENSORS:
            ro = an.rank_test(A, np.array(C, float), "obsv")
            row.append(f"{ro['rank']}/4 ({ro['cond']:.0e})")
            cond[f"obsv: {name}"].append(math.log10(ro["cond"]) if math.isfinite(ro["cond"]) else None)
        rows.append(row)
        gram.append(an.gramian_summary(an.gramian(A, B, "c", T=1.0, n_steps=400))["eig_min"])
    # full 12-state model at mid-boost: what is structurally lost
    p = pts[4]
    A12, B12 = p["A12"], p["B12"]
    r12 = an.rank_test(A12, B12, "ctrb")
    names12 = ["x_b", "y_b", "z_b", "u", "v", "w", "phi", "theta", "psi", "p", "q", "r"]
    lost = [names12[int(np.argmax(np.abs(d)))] for d in r12["deficient_directions"]]
    C_nav = np.zeros((9, 12)); C_nav[0:3, 0:3] = C_nav[3:6, 3:6] = C_nav[6:9, 9:12] = np.eye(3)
    o12 = an.rank_test(A12, C_nav, "obsv")
    o12g = an.rank_test(A12, C_nav[6:9], "obsv")
    rows12 = [["12-state, ctrb [delta_p, delta_y]", f"{r12['rank']}/12", f"{r12['cond']:.1e}",
               "uncontrollable: " + ", ".join(lost) + " (roll: single gimbal; along-track: thrust not throttleable)"],
              ["12-state, obsv GPS pos+vel + gyro", f"{o12['rank']}/12", f"{o12['cond']:.1e}", "fully observable"],
              ["12-state, obsv gyro only", f"{o12g['rank']}/12", f"{o12g['cond']:.1e}",
               "position, velocity, roll angle unobservable"]]
    # navigation and attitude-estimator models (what the observers actually run on)
    An, _ = axis_nav_model()
    Aatt = np.array([[0, -1.0], [0, 0]])
    rows_nav = []
    for model, Am, sets in (("nav axis [p, v, b_a]", An, [("GPS position", [[1, 0, 0]]), ("baro (vertical)", [[1, 0, 0]]),
                                                         ("GPS velocity only", [[0, 1, 0]]),
                                                         ("GPS pos + vel", [[1, 0, 0], [0, 1, 0]]),
                                                         ("accelerometer only (no aiding)", [[0, 0, 0]])]),
                            ("attitude plane [theta, b_gyro]", Aatt, [("attitude reference (mag)", [[1, 0]]),
                                                                    ("gyro only (no reference)", [[0, 0]])])):
        for name, C in sets:
            ro = an.rank_test(Am, np.array(C, float), "obsv")
            verdict = "observable" if ro["full"] else f"{Am.shape[0] - ro['rank']} unobservable direction(s)"
            rows_nav.append([model, name, f"{ro['rank']}/{Am.shape[0]}",
                             f"{ro['cond']:.1e}" if math.isfinite(ro["cond"]) else "inf", verdict])
    assert attitude_plane_model(0.01)[0].shape == (2, 2)
    v["plots"] = [
        plot("cond", "Condition number of the block-scaled ctrb / obsv matrices (log10), pitch plane", "log10(cond)",
             [series(k, t, c, 3, "lines+markers") for k, c in cond.items() if not k.startswith("obsv: attitude")],
             xlabel="time since launch [s]"),
        plot("gram", "Controllability Gramian (1 s horizon), smallest eigenvalue (log10)", "log10(lambda_min)",
             [series("lambda_min(W_c)", t, np.log10(np.maximum(gram, 1e-300)), 3, "lines+markers")],
             xlabel="time since launch [s]"),
    ]
    v["tables"] = [
        {"title": "Pitch plane [z_b, w, theta, q]: Kalman rank (cond) per operating point and sensor set",
         "columns": ["t [s]", "ctrb rank", "ctrb cond (block-scaled)"] + [f"obsv: {n}" for n, _ in SENSORS], "rows": rows},
        {"title": f"Full 12-state model at t = {p['t']:.2f} s", "columns": ["test", "rank", "cond", "meaning"],
         "rows": rows12},
        {"title": "Observer models: observability per sensor set (incl. unobservable cases)",
         "columns": ["model", "measurement", "rank", "cond", "verdict"], "rows": rows_nav},
    ]
    v["metrics"] = {"Pitch-plane ctrb rank (all points)": min(int(r[1][0]) for r in rows),
                    "12-state ctrb rank": r12["rank"], "12-state obsv rank (GPS + gyro)": o12["rank"],
                    "Max ctrb cond": float(f"{10 ** max(cond['ctrb (delta_p)']):.2e}")}
    v["notes"] = (
        "Kalman rank tests (singular-value tolerance 1e-8 relative) with condition numbers of the block-scaled "
        "Kalman matrices (each A^k B block normalised, so |A|^k growth does not mask rank). Pitch plane: "
        "controllable from delta_p at every point, but the condition number grows by more than an order of magnitude over the boost: "
        "the TVC has to beat a stiffening airframe. Gyro, attitude or GPS-velocity outputs alone leave lateral "
        "position z_b unobservable (rank 3); GPS position makes the plane fully observable.\n"
        "Structural losses in the 12-state model: roll angle and roll rate cannot be controlled by a single "
        "gimbal, and along-track position is uncontrollable because the motor cannot be throttled. "
        "For the observers: GPS position or baro make [p, v, b_a] observable, GPS velocity alone leaves "
        "position unobservable, and the accelerometer alone gives pure dead reckoning (rank 0). The attitude "
        "estimator [theta, b_gyro] needs an attitude reference (magnetometer). With the gyro alone the bias "
        "is unobservable, so the rail-averaged gyro bias and the magnetometer correction are both needed.")
    return v


# ------------------------------------------------------------------------------------------------ PID schedule
def step_sim(a1, a0, b, g, d_mag, t_end=3.0, t_d=1.5, r_deg=1.0):
    """theta and gimbal responses (deg) of the linear closed loop (servo lag) to a 1 deg reference step,
    optionally with a step disturbance moment d_mag (deg/s^2 of angular accel) at t_d."""
    sys = an.closed_loop(a1, a0, b, g, tau_act=TAU_ACT)["ss"]
    t = np.linspace(0, t_end, 601)
    U = np.vstack([np.full_like(t, r_deg), np.where(t >= t_d, d_mag, 0.0)])
    r = ct.forced_response(sys, t, U)
    return t, r.outputs[0], r.outputs[1]


def v_pid(pts, tab):
    v = _empty("pid_schedule", "PID gain schedule")
    t = np.array(tab["t"])
    tf = np.linspace(t[0], t[-1], 120)
    rows_p = []
    for i, p in enumerate(pts):
        want = [complex(*c) for c in tab["poles"][i]]
        got = closed_loop_poles(tab["a1"][i], tab["a0"][i], tab["b"][i], tab["Kp"][i], tab["Ki"][i], tab["Kd"][i])
        full = np.linalg.eigvals(an.plane_pid_closed_loop(p["pitch"]["A"], p["pitch"]["B"], tab["Kp"][i],
                                                          tab["Ki"][i], tab["Kd"][i]))
        full = full[np.abs(full) > 1e-6]
        err = max(min(abs(w - g) for g in got) for w in want)
        rows_p.append([rnd(t[i], 2), rnd(tab["wn"][i], 2), pole_list(want), pole_list(got), f"{err:.1e}",
                       pole_list(full)])
    plots = [
        plot("kp_kd", "Scheduled gains Kp, Kd (lines: in-flight interpolation; markers: design points)",
             "Kp [rad/rad], Kd [rad/(rad/s)]",
             [series("Kp", tf, np.interp(tf, t, tab["Kp"]), 4), series("Kd", tf, np.interp(tf, t, tab["Kd"]), 4),
              series("Kp design", t, tab["Kp"], 4, "markers"), series("Kd design", t, tab["Kd"], 4, "markers")],
             xlabel="time since launch [s]"),
        plot("ki", "Scheduled gain Ki", "Ki [rad/(rad s)]",
             [series("Ki", tf, np.interp(tf, t, tab["Ki"]), 4), series("Ki design", t, tab["Ki"], 4, "markers")],
             xlabel="time since launch [s]"),
        plot("wn", "Desired dominant frequency wn = max(8, 0.8 sqrt(a0)) vs airframe frequency sqrt(a0)", "rad/s",
             [series("wn (design)", t, tab["wn"], 3, "lines+markers"),
              series("sqrt(a0) (weathercock)", t, np.sqrt(np.maximum(tab["a0"], 0)), 3, "lines+markers")],
             xlabel="time since launch [s]"),
    ]
    rows_s = []
    for i in STEP_IDX:
        a1, a0, b = tab["a1"][i], tab["a0"][i], tab["b"][i]
        g = gains_at(tab, i)
        d = b * 0.5     # step moment = 0.5 deg of gimbal-equivalent misalignment (deg/s^2 per deg units)
        tt, y0, u0 = step_sim(a1, a0, b, g, 0.0)
        _, y1, u1 = step_sim(a1, a0, b, g, d)
        _, y2, _ = step_sim(a1, a0, b, dict(g, Ki=0.0), d)
        _, y3, _ = step_sim(a1, a0, b, dict(g, Ki=0.0), 0.0)
        m0 = an.step_metrics(tt, y0, 1.0)
        dev = np.abs(y1 - y0)[tt >= 1.5].max()
        sp = specs_of(tab["zeta"], tab["wn"][i])
        rows_s.append([rnd(t[i], 2), rnd(m0["overshoot_pct"], 1), rnd(m0["settling_2pct_s"], 2),
                       rnd(sp["overshoot_pct"], 1), rnd(sp["settling_2pct_s"], 2), rnd(dev, 3),
                       rnd(y1[-1] - 1.0, 4), rnd(y2[-1] - y3[-1], 4), rnd(np.abs(u1).max(), 2)])
        plots.append(plot(f"step_{i}", f"Step response at t = {t[i]:.2f} s (qbar {tab['qbar'][i] / 1e3:.1f} kPa): "
                          "1 deg attitude step, disturbance moment step at 1.5 s", "theta [deg]",
                          [series("PID, no disturbance", tt, y0), series("PID, with disturbance", tt, y1),
                           series("PD only (Ki = 0), with disturbance", tt, y2)]))
        plots.append(plot(f"step_u_{i}", f"Gimbal deflection for the step at t = {t[i]:.2f} s", "delta [deg]",
                          [series("PID, no disturbance", tt, u0), series("PID, with disturbance", tt, u1)]))
    v["plots"] = plots
    v["tables"] = [
        {"title": "Desired vs achieved closed-loop poles (zeta = 0.7, integral pole -4)",
         "columns": ["t [s]", "wn [rad/s]", "desired", "achieved (2nd-order plant + PID)", "max error",
                     "full 4-state plane + PID (excl. lateral position)"], "rows": rows_p},
        {"title": "Step responses of the linear closed loop (20 ms servo lag): reference 1 deg, disturbance "
                  "= 0.5 deg gimbal-equivalent moment",
         "columns": ["t [s]", "overshoot [%]", "settling 2% [s]", "2nd-order spec overshoot [%]",
                     "2nd-order spec settling [s]", "peak disturbance deviation [deg]",
                     "steady-state error PID [deg]", "steady-state error PD [deg]", "peak gimbal [deg]"],
         "rows": rows_s},
        {"title": "Gain schedule (pitch; yaw identical)", "columns": ["t [s]", "qbar [kPa]", "a1", "a0", "b", "wn",
                                                                      "Kp", "Ki", "Kd"],
         "rows": [[rnd(t[i], 2), rnd(tab["qbar"][i] / 1e3, 1), rnd(tab["a1"][i], 2), rnd(tab["a0"][i], 1),
                   rnd(tab["b"][i], 1), rnd(tab["wn"][i], 1), rnd(tab["Kp"][i], 4), rnd(tab["Ki"][i], 3),
                   rnd(tab["Kd"][i], 4)] for i in range(len(t))]},
    ]
    v["metrics"] = {"Max pole-placement error": float(f"{max(float(r[4]) for r in rows_p):.1e}"),
                    "Damping ratio (design)": tab["zeta"], "Integral pole [1/s]": -tab["p_int"],
                    "wn range [rad/s]": f"{min(tab['wn']):.1f} - {max(tab['wn']):.1f}",
                    "Steady-state error with integral action [deg]": rows_s[1][6],
                    "Steady-state error without integral action [deg]": rows_s[1][7]}
    v["notes"] = (
        "Per operating point the short-period reduction theta'' + a1 theta' + a0 theta = b delta is closed with "
        "delta = Kp e + Ki int(e) - Kd q_gyro. Matching s^3 + (a1 + b Kd) s^2 + (a0 + b Kp) s + b Ki to "
        "(s + p)(s^2 + 2 zeta wn s + wn^2) gives the gains in closed form (gnc.pid.place_pid). The 'achieved' "
        "column is the root set of the resulting characteristic polynomial. The last column closes the same gains "
        "around the full 4-state plane, including the drift mode. The dominant pair stays close to the design, and the "
        "integrator merges with the stabilised drift mode into slow, well-damped poles.\n"
        "Step responses: derivative-on-measurement and the PI zero shape the response. The servo lag adds "
        "a little overshoot over the 2nd-order spec. A step disturbance moment (equivalent to a 0.5 deg thrust "
        "misalignment) is fully trimmed out by the integral action. Without it (PD only, same Kp, Kd) a "
        "steady-state error d/(a0 + b Kp) remains. At the end of the boost the same 1 deg step would need a "
        "peak gimbal well above the 5 deg limit (linear model). There the TVC can only trim, not manoeuvre.")
    return v


# ------------------------------------------------------------------------------------------------ delays
def v_delays(pts, tab):
    v = _empty("delays", "Delays & margins")
    t = np.array(tab["t"])
    pm, gm, dm, dc = [], [], [], []
    for i in range(len(t)):
        a1, a0, b, g = tab["a1"][i], tab["a0"][i], tab["b"][i], gains_at(tab, i)
        m = an.margins(an.loop_tf(a1, a0, b, g, TAU_ACT))
        pm.append(m["phase_margin_deg"]); gm.append(m["gain_margin_dB"]); dm.append(1e3 * m["delay_margin_s"])
        dc.append(1e3 * an.critical_delay(a1, a0, b, g, TAU_ACT, pade_order=5, d_max=0.3))
    delays = np.arange(0.0, 0.0801, 0.005)
    plots = [
        plot("pm", "Phase margin vs time (20 ms servo lag, no extra delay)", "deg",
             [series("phase margin", t, pm, 2, "lines+markers")], xlabel="time since launch [s]"),
        plot("dm", "Delay margin vs time: PM / w_c and Pade-5 bisection", "ms",
             [series("PM / w_c", t, dm, 2, "lines+markers"), series("Pade bisection", t, dc, 2, "lines+markers"),
              series("15 + 15 ms demo", [t[0], t[-1]], [30, 30]), series("30 + 30 ms demo", [t[0], t[-1]], [60, 60])],
             xlabel="time since launch [s]"),
    ]
    rows_sw, crossing, maxre, pm_series = [], {}, {}, []
    for i in (4, 8):
        a1, a0, b, g = tab["a1"][i], tab["a0"][i], tab["b"][i], gains_at(tab, i)
        sw = an.delay_sweep(a1, a0, b, g, delays, TAU_ACT, pade_order=3)
        maxre[i] = [r["max_real"] for r in sw]
        crossing[i] = 1e3 * an.critical_delay(a1, a0, b, g, TAU_ACT, pade_order=5, d_max=0.3)
        if i == 4:
            rows_sw = [[rnd(1e3 * r["delay_s"], 0), rnd(r["max_real"], 2), rnd(r["dominant_zeta"], 3),
                        rnd(r["phase_margin_deg"], 1), "stable" if r["stable"] else "UNSTABLE"] for r in sw]
            for k in (0, 4, 8, 12, 16):
                p = np.array([complex(*c) for c in sw[k]["poles"]])
                p = p[p.real > -40]
                pm_series.append(series(f"{1e3 * delays[k]:.0f} ms", p.real, p.imag, 3, "markers"))
    plots.append(plot("polemig", f"Closed-loop poles vs loop delay at t = {t[4]:.2f} s (Pade-3, poles with Re > -40)",
                      "Im [rad/s]", pm_series, xlabel="Re [1/s]"))
    plots.append(plot("maxre", "Most unstable closed-loop pole vs loop delay", "max Re(pole) [1/s]",
                      [series(f"t = {t[4]:.2f} s", 1e3 * delays, maxre[4], 3, "lines+markers"),
                       series(f"t = {t[8]:.2f} s (TVC end)", 1e3 * delays, maxre[8], 3, "lines+markers"),
                       series("stability boundary", [0, 80], [0, 0])], xlabel="loop delay [ms]"))
    v["plots"] = plots
    v["tables"] = [
        {"title": "Margins along the boost (servo lag 20 ms)",
         "columns": ["t [s]", "qbar [kPa]", "PM [deg]", "GM [dB]", "DM = PM/w_c [ms]", "DM Pade bisection [ms]"],
         "rows": [[rnd(t[i], 2), rnd(tab["qbar"][i] / 1e3, 1), rnd(pm[i], 1),
                   rnd(gm[i], 1) if math.isfinite(gm[i]) else "inf", rnd(dm[i], 1), rnd(dc[i], 1)]
                  for i in range(len(t))]},
        {"title": f"Pole migration vs loop delay at t = {t[4]:.2f} s", "columns": ["loop delay [ms]",
                                                                                   "max Re(pole) [1/s]",
                                                                                   "dominant zeta", "PM [deg]",
                                                                                   "verdict"],
         "rows": rows_sw},
    ]
    v["metrics"] = {"Min phase margin [deg]": rnd(min(pm), 1), "Min delay margin (Pade) [ms]": rnd(min(dc), 1),
                    f"Critical loop delay at t = {t[4]:.2f} s [ms]": rnd(crossing[4], 1),
                    "Critical loop delay at TVC end [ms]": rnd(crossing[8], 1),
                    "Max PM/w_c vs Pade discrepancy [%]": rnd(max(abs(a - b) / b * 100 for a, b in zip(dm, dc)), 1)}
    v["notes"] = (
        "Actuation delay (command transport) and estimation delay (sensor samples arriving late) multiply the "
        "same single loop by e^{-s tau}, so they are analysed together as the loop delay tau = tau_act + tau_est "
        "(Pade approximant). The delay margin is tightest at the end of the boost, where the design bandwidth "
        "wn is highest. The pole-migration plot shows the dominant pair travelling right as the delay grows and "
        "crossing the imaginary axis at the critical delay.\n"
        "In the 6-DOF sim the loop also contains ~6 ms of intrinsic delay (10 ms controller sample-and-hold "
        "~5 ms, plus the 1 ms integration step), on top of the configured delays. The 15 + 15 ms case "
        "(~36 ms in total) sits just under the mid-boost margin and only touches the end-of-boost margin in the last "
        "few tenths of a second. 30 + 30 ms (~66 ms) exceeds the margin everywhere (see the 'Control' demo).")
    return v


def main():
    r = rocket()
    _, tab = design_schedule(r)
    pts = operating_points(r, tab["t"])
    demo = {"id": "analysis", "group": "System analysis", "title": "Linearisation, ctrb/obsv, PID design, delays",
            "description": (
                "Linear analysis of the TVC rocket about 9 operating points on the nominal boost, from rail exit to "
                "the start of the thrust tail-off, where TVC authority ends. Linearisation is numeric, by "
                "finite differences straight through the 6-DOF dynamics, and checked against an analytic "
                "plane model. The variants cover the brief's control-design items: linearisation and pole maps, "
                "controllability/observability rank tests (including the unobservable cases), PID tuning by "
                "pole placement with a time-scheduled gain table and step responses with/without a disturbance "
                "moment, and the effect of estimation + actuation delay on the closed loop.\n\n"
                "All plots here are analysis-only (no trajectory). The nonlinear flights that use these gains are in "
                "the Control, Path-following and Robustness demos."),
            "variants": [v_linearisation(pts, tab), v_ctrb_obsv(pts, tab), v_pid(pts, tab), v_delays(pts, tab)]}
    write_demo(demo)
    return demo


if __name__ == "__main__":
    main()
