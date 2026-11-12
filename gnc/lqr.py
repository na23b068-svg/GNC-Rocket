"""LQR attitude control with integral action, gain-scheduled like the pole-placement PID, plus a comparison
of the two designs.

Per plane, on the same short-period model as the PID (gnc.pid / gnc.linearize.short_period), augmented with
the integral of the attitude error:
    x = [theta, q, xi],  theta' = q,  q' = -a0 theta - a1 q + b delta,  xi' = theta   (xi = int(theta - theta_c))
    delta = -K x,  K = R^-1 B^T P,  P from the continuous ARE.
With e = theta_c - theta this is exactly a PID with derivative on the measured rate:
    Kp = K_theta, Kd = K_q, Ki = K_xi
so it reuses gnc.pid.PID / GainSchedule and the loop analysis in gnc.analysis (the loop broken at the plant
input is the same C(s) G(s)). LQR guarantees >= 60 deg phase margin and infinite upward gain margin at the
plant input for the delay- and lag-free model.

Weights (Bryson's rule, final values after tuning against delay margin with the 20 ms servo lag):
    theta_max = 4 deg, q_max = 40 deg/s, xi_max = theta_max * 0.3 s, delta_max = 5 deg (gimbal limit).
"""
import control as ct
import numpy as np

from .analysis import closed_loop, critical_delay, loop_tf, margins, step_metrics
from .pid import GainSchedule

BRYSON = {"theta_max": np.radians(4.0), "q_max": np.radians(40.0), "Ti": 0.3, "delta_max": np.radians(5.0)}


def bryson(theta_max, q_max, Ti, delta_max):
    """Q = diag(1/x_max^2), R = 1/u_max^2 (integral state allowed theta_max * Ti)."""
    Q = np.diag([1 / theta_max**2, 1 / q_max**2, 1 / (theta_max * Ti) ** 2])
    return Q, np.array([[1 / delta_max**2]])


def augmented_model(a1, a0, b):
    A = np.array([[0.0, 1.0, 0.0], [-a0, -a1, 0.0], [1.0, 0.0, 0.0]])
    B = np.array([[0.0], [b], [0.0]])
    return A, B


def attitude_lqr(a1, a0, b, Q=None, R=None):
    """LQR gains for one operating point. Returns dict(Kp, Ki, Kd, K, P, poles)."""
    if Q is None:
        Q, R = bryson(**BRYSON)
    A, B = augmented_model(a1, a0, b)
    K, P, E = ct.lqr(A, B, Q, R)
    return {"Kp": float(K[0, 0]), "Kd": float(K[0, 1]), "Ki": float(K[0, 2]), "K": K, "P": P, "poles": E}


def design_lqr_schedule(table, Q=None, R=None):
    """LQR gains at the PID schedule's operating points (table from flight_computer.design_schedule).
    Returns (GainSchedule, list of per-point gain dicts)."""
    g = [attitude_lqr(a1, a0, b, Q, R) for a1, a0, b in zip(table["a1"], table["a0"], table["b"])]
    sched = GainSchedule(np.asarray(table["t"], float), *(np.array([gi[k] for gi in g]) for k in ("Kp", "Ki", "Kd")))
    return sched, g


def _metrics(a1, a0, b, gains, tau_act):
    cl = closed_loop(a1, a0, b, gains, tau_act)
    t, y = ct.step_response(cl["T_ref"], 10.0)
    m = margins(loop_tf(a1, a0, b, gains, tau_act))
    m0 = margins(loop_tf(a1, a0, b, gains, 0.0))
    p = np.sort_complex(ct.poles(cl["ss"]))
    return {"poles": [[float(c.real), float(c.imag)] for c in p], **step_metrics(t, y),
            "gain_margin_dB": m["gain_margin_dB"], "phase_margin_deg": m["phase_margin_deg"],
            "phase_margin_no_lag_deg": m0["phase_margin_deg"],
            "delay_margin_s": critical_delay(a1, a0, b, gains, tau_act),
            "Kp": gains["Kp"], "Ki": gains["Ki"], "Kd": gains["Kd"]}


def compare(table, tau_act=0.02, Q=None, R=None):
    """Pole-placement PID vs LQR at every schedule point, both with the same servo lag tau_act.
    Returns a list of rows {t, qbar, pid: {...}, lqr: {...}} with closed-loop poles, step overshoot /
    2 % settling, gain / phase / delay margins and gains."""
    _, lq = design_lqr_schedule(table, Q, R)
    rows = []
    for i, t in enumerate(table["t"]):
        a1, a0, b = table["a1"][i], table["a0"][i], table["b"][i]
        pid = {k: table[k][i] for k in ("Kp", "Ki", "Kd")}
        rows.append({"t": float(t), "qbar": float(table["qbar"][i]),
                     "pid": _metrics(a1, a0, b, pid, tau_act), "lqr": _metrics(a1, a0, b, lq[i], tau_act)})
    return rows
