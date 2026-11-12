"""Independent checker tests for the gnc-designer's work (linearisation, ctrb/obsv, PID, delays, observers,
closed-loop flights). Derivations here deliberately avoid calling the code under test where possible."""
import numpy as np
import pytest
import scipy.linalg as sla
from scipy.integrate import solve_ivp

from gnc.dynamics import derivatives, make_env
from gnc.flight_computer import FlightComputer, design_schedule
from gnc.frames import quat_conj, quat_from_axis_angle, quat_mul, quat_to_dcm
from gnc.linearize import nominal_trajectory, operating_points
from gnc.rocket import Rocket
from gnc.sim import SimConfig, simulate

W_PATH = np.array([[0, 0, 0], [0, 30, 300], [0, 90, 900]], float)


@pytest.fixture(scope="module")
def rocket():
    return Rocket.load()


@pytest.fixture(scope="module")
def pts(rocket):
    return operating_points(rocket, [0.374, 2.17, 3.0], nominal_trajectory(rocket))


@pytest.fixture(scope="module")
def tab(rocket):
    return design_schedule(rocket)[1]


def _perturbed(x0, zb, w, th, q):
    """Pitch-plane perturbation in nominal body axes (body z position/velocity, rotation about body y, rate)."""
    ez = quat_to_dcm(x0[6:10])[:, 2]
    x = x0.copy()
    x[0:3] += zb * ez
    x[3:6] += w * ez
    x[6:10] = quat_mul(x0[6:10], quat_from_axis_angle([0, 1, 0], th)) if th else x0[6:10]
    x[11] += q
    return x


def test_pitch_plane_matches_independent_finite_difference(rocket, pts):
    env = make_env(rocket)
    for p in pts:
        t, x0 = p["t"], p["x0"]
        ez = quat_to_dcm(x0[6:10])[:, 2]
        f0 = derivatives(t, x0, np.zeros(2), rocket, env)

        def rows(e, dp=0.0):
            xd = derivatives(t, _perturbed(x0, *e), np.array([dp, 0.0]), rocket, env)
            return np.array([ez @ (xd[0:3] - f0[0:3]), ez @ (xd[3:6] - f0[3:6]), xd[11] - f0[11]])

        A = np.zeros((3, 4))
        for i, h in enumerate((1e-3, 1e-3, 1e-6, 1e-6)):
            d = np.zeros(4); d[i] = h
            A[:, i] = (rows(d) - rows(-d)) / (2 * h)
        b = (rows(np.zeros(4), 1e-6) - rows(np.zeros(4), -1e-6)) / 2e-6
        Ap, Bp = p["pitch"]["A"], p["pitch"]["B"]
        sel = [0, 1, 3]                                    # z_b', w', q' rows (theta' = q is kinematic)
        assert np.allclose(A, Ap[sel], rtol=1e-5, atol=1e-6 * np.abs(Ap).max())
        assert np.allclose(b, Bp[sel, 0], rtol=1e-6)


def test_drift_mode_is_a_frozen_coefficient_artefact(rocket, pts):
    """DEFECT evidence: the +1.56/s 'drift' eigenvalue (doc sec. 3) does not appear in the nonlinear
    time-varying dynamics. Seeded along its eigenvector, the nonlinear attitude perturbation stays bounded
    (~1.9x in 1 s, flat after 0.5 s) while the frozen LTI model predicts e^{1.56} = 4.8x and growing."""
    env = make_env(rocket)
    p = pts[0]
    x0, A = p["x0"], p["pitch"]["A"]
    ev, V = np.linalg.eig(A)
    i = np.argmax(ev.real)
    assert ev[i].real > 1.4
    v = V[:, i].real / V[2, i].real * 1e-3
    f = lambda t, x: derivatives(t, x, np.zeros(2), rocket, env)  # noqa: E731
    T = 1.0
    xn = solve_ivp(f, (p["t"], p["t"] + T), x0, rtol=1e-11, atol=1e-12).y[:, -1]
    xp = solve_ivp(f, (p["t"], p["t"] + T), _perturbed(x0, *v), rtol=1e-11, atol=1e-12).y[:, -1]
    dth_nl = 2 * quat_mul(quat_conj(xn[6:10]), xp[6:10])[2]
    dth_lti = (sla.expm(A * T) @ v)[2]
    assert dth_lti / v[2] > 4.0
    assert dth_nl / v[2] < 0.5 * dth_lti / v[2]


def test_doc_claim_gyro_only_rank3_late_boost(rocket):
    from gnc.analysis import rank_test
    for p in operating_points(rocket, [3.0, 3.9]):
        assert rank_test(p["pitch"]["A"], np.array([[0, 0, 0, 1.0]]), "obsv")["rank"] == 3


@pytest.mark.parametrize("i", [0, 4, 8])
def test_schedule_gains_by_hand(tab, i):
    a1, a0, b = tab["a1"][i], tab["a0"][i], tab["b"][i]
    zeta, p = 0.7, 4.0
    wn = max(8.0, 0.8 * np.sqrt(a0))
    Kd = (2 * zeta * wn + p - a1) / b
    Kp = (wn**2 + 2 * zeta * wn * p - a0) / b
    Ki = p * wn**2 / b
    assert np.allclose([tab["Kp"][i], tab["Ki"][i], tab["Kd"][i]], [Kp, Ki, Kd], rtol=1e-9)
    want = np.sort_complex(np.array([-zeta * wn + 1j * wn * np.sqrt(1 - zeta**2),
                                     -zeta * wn - 1j * wn * np.sqrt(1 - zeta**2), -p]))
    got = np.sort_complex(np.roots([1, a1 + b * Kd, a0 + b * Kp, b * Ki]))
    assert np.allclose(got, want, atol=1e-8)


def _exact_margins(a1, a0, b, Kp, Ki, Kd, tau=0.02):
    """PM and delay margin from the exact frequency response (no Pade), min over all gain crossovers."""
    w = np.logspace(-2, 3, 200001)
    s = 1j * w
    L = (Kp + Ki / s + Kd * s) * b / (s * s + a1 * s + a0) / (tau * s + 1)
    idx = np.where(np.diff(np.sign(np.abs(L) - 1)))[0]
    pm = [(np.degrees(np.angle(L[j])) + 180) % 360 for j in idx]
    return pm[-1], min(np.radians(a) / w[j] for a, j in zip(pm, idx))


def test_phase_and_delay_margins_frequency_domain(tab):
    from gnc.analysis import critical_delay
    dms = []
    for i in range(len(tab["t"])):
        g = dict(Kp=tab["Kp"][i], Ki=tab["Ki"][i], Kd=tab["Kd"][i], tau_f=0.0)
        pm, dm = _exact_margins(tab["a1"][i], tab["a0"][i], tab["b"][i], g["Kp"], g["Ki"], g["Kd"])
        assert pm >= 47.0
        assert dm == pytest.approx(critical_delay(tab["a1"][i], tab["a0"][i], tab["b"][i], g, tau_act=0.02),
                                   rel=0.01)
        dms.append(dm)
    assert 0.032 < min(dms) and max(dms) < 0.074


class _Poison:
    """Wraps a controller and replaces meas['truth'] (NaN or removed) before it reaches the controller."""

    def __init__(self, inner, mode):
        self.inner, self.mode, self.dt = inner, mode, inner.dt

    def reset(self, rocket, cfg):
        self.inner.reset(rocket, cfg)

    def update(self, t, meas):
        meas = dict(meas)
        if self.mode == "nan":
            meas["truth"] = np.full(13, np.nan)
        else:
            meas.pop("truth", None)
        return self.inner.update(t, meas)

    @property
    def log(self):
        return self.inner.log


def _los():
    return FlightComputer(guidance="los_path", waypoints=W_PATH)


def test_controller_never_reads_truth(rocket):
    cfg = SimConfig(t_max=2.5, seed=5)
    base = simulate(rocket, cfg, _los())["u_cmd"]
    for mode in ("nan", "remove"):
        u = simulate(rocket, cfg, _Poison(_los(), mode))["u_cmd"]
        assert np.all(np.isfinite(u)) and np.array_equal(u, base)


def test_deterministic_with_seed(rocket):
    a = simulate(rocket, SimConfig(t_max=2.0, seed=11), _los())
    b = simulate(rocket, SimConfig(t_max=2.0, seed=11), _los())
    c = simulate(rocket, SimConfig(t_max=2.0, seed=12), _los())
    assert np.array_equal(a["x"], b["x"]) and np.array_equal(a["u_cmd"], b["u_cmd"])
    assert not np.array_equal(a["u_cmd"], c["u_cmd"])


def test_los_in_wind_gimbal_limited_and_tracks(rocket):
    cfg = SimConfig(t_max=4.5, seed=7, wind=(0.0, -10.0, 0.0), wind_shear_exponent=1 / 7,
                    gust={"std_mps": 3.0, "tau_s": 1.5})
    fc = _los()
    res = simulate(rocket, cfg, fc)
    assert np.abs(res["u_act"]).max() <= cfg.gimbal_limit_rad + 1e-12
    ph = np.array(fc.log["phase"])
    assert np.sqrt(np.mean(np.array(fc.log["cross_track"])[ph == 1] ** 2)) < 15.0


def _late_pp(rocket, d):
    fc = FlightComputer()
    simulate(rocket, SimConfig(t_max=4.2, seed=7, actuator_delay_s=d, estimation_delay_s=d), fc)
    t, u = np.array(fc.log["t"]), np.degrees(np.array(fc.log["u_cmd"])[:, 0])
    m = (t >= 3.5) & (np.array(fc.log["phase"]) == 1)
    return np.ptp(u[m])


def test_delay_15_15_benign_30_30_diverges(rocket):
    nominal = _late_pp(rocket, 0.0)
    assert _late_pp(rocket, 0.015) < 1.5 * nominal
    assert _late_pp(rocket, 0.030) > 5 * nominal      # loop delay 60 ms > 36 ms margin late in the burn


def test_kf_beats_luenberger_position_and_consistent_nis(rocket):
    for k in (1.0, 4.0):
        for seed in (1, 2):
            out = {}
            for nav in ("kf", "luenberger"):
                fc = FlightComputer(nav=nav)
                res = simulate(rocket, SimConfig(t_max=10.0, seed=seed, sensor_noise_scale=k), fc)
                lt = np.array(fc.log["t"])
                X = res["x"][np.searchsorted(res["t"], lt - 1e-9)]
                out[nav] = np.sqrt(np.mean((np.array(fc.log["pos_est"]) - X[:, 0:3]) ** 2, 0))
                if nav == "kf":
                    nis = np.mean([v for v in fc.log["nis_gps"] if v is not None][5:])
            ratio = out["luenberger"] / out["kf"]
            assert ratio.min() > 1.0 and ratio[2] > 3.0
            assert 3.5 < nis < 7.5
