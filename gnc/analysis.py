"""System analysis helpers returning plain data (lists/floats) ready for results JSON:
controllability/observability, Gramians, poles/zeros, stability margins, delay (Pade) analysis,
disturbance rejection of the attitude loop."""
import functools
import warnings

import control as ct
import numpy as np
import scipy.linalg as sla


def _quiet(f):
    """python-control 0.10.2 sets ndarray.shape on static (0-state) systems, deprecated in numpy 2.5."""
    @functools.wraps(f)
    def g(*a, **k):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Setting the shape on a NumPy array", DeprecationWarning)
            return f(*a, **k)
    return g


def _cond(M):
    s = np.linalg.svd(M, compute_uv=False)
    return float(s[0] / s[-1]) if s[-1] > 1e-300 else float("inf")


def _kalman_matrix(A, M_in, kind):
    """ctrb [B, AB, ...] / obsv [C; CA; ...] with every block A^k B (C A^k) scaled to unit Frobenius norm.
    Block scaling leaves rank, column space (ctrb) and null space (obsv) unchanged but removes the |A|^k
    growth (|A| ~ 500 at max qbar -> raw blocks span ~8 orders of magnitude and fool an SVD rank test)."""
    blocks, X = [], M_in
    tol = A.shape[0] * np.finfo(float).eps * np.linalg.norm(A)
    prev = None                                   # norm of the previous raw block
    for _ in range(A.shape[0]):
        nrm = np.linalg.norm(X)
        if nrm == 0 or (prev is not None and nrm < tol * prev):
            X = np.zeros_like(X)                  # roundoff-only block (true A^k B = 0): keep it zero
            blocks.append(X)
        else:
            blocks.append(X / nrm)
            prev = nrm
        X = A @ X if kind == "ctrb" else X @ A
    return np.hstack(blocks) if kind == "ctrb" else np.vstack(blocks)


def rank_test(A, M_in, kind="ctrb", rtol=1e-8):
    """Kalman rank test. kind='ctrb': M_in = B; kind='obsv': M_in = C.
    Numerical rank = number of singular values above rtol * sigma_max of the block-normalised Kalman matrix.
    rtol = 1e-8 treats directions whose influence is 8 orders of magnitude weaker (e.g. lateral position acting
    only through the air-density gradient) as practically uncontrollable/unobservable; `cond` reports how close
    to deficient it is. Returns dict(n, rank, full, cond, singular_values, deficient_directions)."""
    A = np.atleast_2d(np.asarray(A, float))
    M = _kalman_matrix(A, np.atleast_2d(np.asarray(M_in, float)), kind)
    U, s, Vt = np.linalg.svd(M)
    r = int(np.sum(s > rtol * s[0])) if s.size and s[0] > 0 else 0
    # uncontrollable directions: left singular vectors of the ctrb matrix beyond the rank;
    # unobservable directions: right singular vectors of the obsv matrix beyond the rank
    ns = U[:, r:].T if kind == "ctrb" else Vt[r:]
    return {"n": A.shape[0], "rank": r, "full": r == A.shape[0], "cond": _cond(M),
            "singular_values": s.tolist(), "deficient_directions": ns.round(6).tolist()}


def gramian(A, M_in, kind="c", T=None, n_steps=2000):
    """Controllability ('c', M_in=B) or observability ('o', M_in=C) Gramian. Infinite horizon (Lyapunov)
    if A is Hurwitz and T is None, otherwise finite horizon over [0, T] by quadrature."""
    A = np.atleast_2d(A)
    Q = M_in @ M_in.T if kind == "c" else M_in.T @ M_in
    if T is None and np.all(np.linalg.eigvals(A).real < 0):
        At = A if kind == "c" else A.T
        return sla.solve_continuous_lyapunov(At, -Q)
    T = 1.0 if T is None else T
    ts = np.linspace(0, T, n_steps + 1)
    W = np.zeros_like(A, dtype=float)
    for i, t in enumerate(ts):
        E = sla.expm(A * t) if kind == "c" else sla.expm(A.T * t)
        w = 0.5 if i in (0, n_steps) else 1.0
        W += w * (E @ Q @ E.T)
    return W * (T / n_steps)


def gramian_summary(W):
    ev = np.linalg.eigvalsh(0.5 * (W + W.T))
    return {"eig_min": float(ev[0]), "eig_max": float(ev[-1]),
            "cond": float(ev[-1] / ev[0]) if ev[0] > 0 else float("inf")}


@_quiet
def poles_zeros(sys):
    sys = ct.ss(sys) if not isinstance(sys, ct.StateSpace) else sys
    p = np.linalg.eigvals(sys.A)
    z = ct.zeros(sys) if sys.ninputs == 1 and sys.noutputs == 1 else np.array([])
    fmt = lambda v: [[float(c.real), float(c.imag)] for c in sorted(v, key=lambda c: (c.real, c.imag))]
    return {"poles": fmt(p), "zeros": fmt(z)}


# ---------------------------------------------------------------- attitude loop (plant + PID + actuator + delay)

def attitude_plant(a1, a0, b):
    """theta/delta = b / (s^2 + a1 s + a0)."""
    return ct.tf([b], [1.0, a1, a0])


def pid_tf(Kp, Ki, Kd, tau_f=0.0, **_):
    return ct.tf([Kd + Kp * tau_f, Kp + Ki * tau_f, Ki], [tau_f, 1.0, 0.0]) if tau_f else ct.tf([Kd, Kp, Ki], [1.0, 0.0])


@_quiet
def loop_tf(a1, a0, b, gains, tau_act=0.0, delay=0.0, pade_order=3):
    """Open loop broken at the actuator input: L = C(s) G(s) A(s) e^{-s*delay} (delay via Pade).
    Actuation delay and estimation delay add up in this single loop."""
    L = pid_tf(**gains) * attitude_plant(a1, a0, b)
    if tau_act > 0:
        L = L * ct.tf([1.0], [tau_act, 1.0])
    if delay > 0:
        L = L * ct.tf(*ct.pade(delay, pade_order))
    return L


@_quiet
def closed_loop(a1, a0, b, gains, tau_act=0.0, delay=0.0, pade_order=3):
    """Closed-loop transfer functions (derivative on measurement, as implemented in gnc.pid.PID):
    T_ref: theta_cmd -> theta, T_dist: disturbance moment (as angular accel, rad/s^2) -> theta,
    T_u: theta_cmd -> gimbal angle. Built as a state-space interconnection."""
    # (b*delta + d) -> [theta, rate]; the gyro gives the rate directly, so derivative action is proper
    G = ct.ss([[0, 1], [-a0, -a1]], [[0], [1]], np.eye(2), 0, inputs="w", outputs=["y", "q"], name="G")
    one = ct.tf([1.0], [1.0])
    act = ct.tf([1.0], [tau_act, 1.0]) if tau_act > 0 else one
    dly = ct.tf(*ct.pade(delay, pade_order)) if delay > 0 else one
    Kp, Ki, Kd, tf_ = gains["Kp"], gains["Ki"], gains["Kd"], gains.get("tau_f", 0.0)
    filt = ct.tf([1.0], [tf_, 1.0]) if tf_ > 0 else one
    PI = ct.tf([Kp, Ki], [1.0, 0.0]) if Ki else ct.tf([Kp], [1.0])
    # u = (Kp + Ki/s)(r - y) - Kd * filt * q
    sys_list = [G, ct.ss(act * dly, inputs="uc", outputs="ua", name="A"),
                ct.summing_junction(["r", "-y"], "e", name="sume"),
                ct.ss(PI, inputs="e", outputs="ue", name="C"),
                ct.ss(-Kd * filt, inputs="q", outputs="uq", name="Cq")]
    sums = [ct.summing_junction(["ue", "uq"], "uc", name="sumu"),
            ct.summing_junction(inputs=["bu", "d"], output="w", name="sumw"),
            ct.ss(ct.tf([b], [1.0]), inputs="ua", outputs="bu", name="gain")]
    sys = ct.interconnect(sys_list + sums, inputs=["r", "d"], outputs=["y", "ua"])
    return {"T_ref": sys[0, 0], "T_dist": sys[0, 1], "T_u": sys[1, 0], "ss": sys}


@_quiet
def margins(L):
    """Gain/phase/delay margins of open loop L (python-control). Delay margin = PM / w_crossover."""
    gm, pm, _, wpc, wgc, _ = ct.stability_margins(L)
    dm = np.deg2rad(pm) / wgc if np.isfinite(pm) and wgc and wgc > 0 else float("inf")
    return {"gain_margin_dB": float(20 * np.log10(gm)) if np.isfinite(gm) and gm > 0 else float("inf"),
            "phase_margin_deg": float(pm), "w_gain_crossover": float(wgc),
            "w_phase_crossover": float(wpc), "delay_margin_s": float(dm)}


@_quiet
def delay_sweep(a1, a0, b, gains, delays, tau_act=0.0, pade_order=3):
    """Closed-loop pole migration vs loop delay. Returns rows: delay, max Re(pole), damping of the
    dominant pole, stable flag, phase margin (deg)."""
    rows = []
    for d in delays:
        L = loop_tf(a1, a0, b, gains, tau_act, d, pade_order)
        p = ct.poles(ct.feedback(L, 1))
        dom = p[np.argmax(p.real)]
        m = margins(L)
        rows.append({"delay_s": float(d), "max_real": float(dom.real),
                     "dominant_zeta": float(-dom.real / abs(dom)) if abs(dom) > 0 else 1.0,
                     "stable": bool(np.all(p.real < 0)), "phase_margin_deg": m["phase_margin_deg"],
                     "poles": [[float(c.real), float(c.imag)] for c in p]})
    return rows


@_quiet
def critical_delay(a1, a0, b, gains, tau_act=0.0, **_):
    """Exact delay margin of the attitude loop (no Pade): min over ALL gain crossovers w_i of
    ((angle L(j w_i) + 180 deg) mod 360 deg) / w_i, with L = C(s) G(s) / (tau_act s + 1).
    Every crossover is checked because the loop can have several (the PM of the last one is not enough), and
    bisection on delay is invalid since stability is not monotonic in delay. Returns inf without crossover."""
    from scipy.optimize import brentq
    Kp, Ki, Kd, tf_ = gains["Kp"], gains["Ki"], gains["Kd"], gains.get("tau_f", 0.0)

    def L(w):
        s = 1j * w
        return (Kp + Ki / s + Kd * s / (tf_ * s + 1)) * b / (s * s + a1 * s + a0) / (tau_act * s + 1)

    w = np.logspace(-4, 5, 40001)
    lm = np.log(np.abs(L(w)))
    dms = []
    for i in np.nonzero(np.diff(np.sign(lm)))[0]:
        wc = brentq(lambda x: np.log(abs(L(x))), w[i], w[i + 1], xtol=1e-12)
        pm = (np.degrees(np.angle(L(wc))) + 180.0) % 360.0
        dms.append(np.radians(pm) / wc)
    return float(min(dms)) if dms else float("inf")


def step_metrics(t, y, final=None):
    final = y[-1] if final is None else final
    os = (np.max(y) - final) / abs(final) * 100 if final else float("nan")
    out = np.abs(y - final) > 0.02 * abs(final)
    ts = float(t[np.where(out)[0][-1] + 1]) if out.any() and np.where(out)[0][-1] + 1 < len(t) else float(t[0])
    return {"overshoot_pct": float(os), "settling_2pct_s": ts, "final": float(final)}


def plane_pid_closed_loop(A, B, Kp, Ki, Kd, i_ang=2, i_rate=3, **_):
    """Closed-loop matrix of a full plane model [pos, vel, angle, rate] (gnc.linearize.plane) under the
    attitude PID delta = Kp*(-angle) + Ki*xi - Kd*rate, xi' = -angle (regulation about the operating point).
    Verifies the 2nd-order design on the full model, incl. the slow lateral-drift mode."""
    n = A.shape[0]
    K = np.zeros(n)
    K[i_ang], K[i_rate] = -Kp, -Kd
    Acl = np.zeros((n + 1, n + 1))
    Acl[:n, :n] = A + B @ K[None, :]
    Acl[:n, n] = B[:, 0] * Ki
    Acl[n, i_ang] = -1.0
    return Acl
