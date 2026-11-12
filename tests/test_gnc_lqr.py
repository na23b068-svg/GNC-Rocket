import numpy as np
import pytest
import scipy.linalg as sla

from gnc.analysis import loop_tf, margins, plane_pid_closed_loop
from gnc.flight_computer import FlightComputer, design_schedule
from gnc.linearize import operating_points
from gnc.lqr import BRYSON, attitude_lqr, augmented_model, bryson, compare, design_lqr_schedule
from gnc.rocket import Rocket
from gnc.sim import SimConfig, simulate


@pytest.fixture(scope="module")
def rocket():
    return Rocket.load()


@pytest.fixture(scope="module")
def tab(rocket):
    return design_schedule(rocket)[1]


def test_gain_solves_care(tab):
    Q, R = bryson(**BRYSON)
    for i in (0, 4, 8):
        a1, a0, b = tab["a1"][i], tab["a0"][i], tab["b"][i]
        A, B = augmented_model(a1, a0, b)
        P = sla.solve_continuous_are(A, B, Q, R)
        K = np.linalg.solve(R, B.T @ P)
        g = attitude_lqr(a1, a0, b)
        assert np.allclose([g["Kp"], g["Kd"], g["Ki"]], K[0], rtol=1e-6)
        assert np.allclose(np.sort_complex(np.linalg.eigvals(A - B @ K)), np.sort_complex(g["poles"]))


def test_closed_loop_stable_every_point(rocket, tab):
    _, lq = design_lqr_schedule(tab)
    pts = operating_points(rocket, tab["t"])
    for g, p, row in zip(lq, pts, compare(tab)):
        assert np.all(np.real(g["poles"]) < 0)                              # design model
        assert max(r for r, _ in row["lqr"]["poles"]) < 0                   # + 20 ms servo lag
        ev = np.linalg.eigvals(plane_pid_closed_loop(p["pitch"]["A"], p["pitch"]["B"], **g))
        assert np.sort(ev.real)[-2] < -0.05 and ev.real.max() < 1e-4        # full frozen plane model


def test_lqr_classic_margins_without_lag(tab):
    _, lq = design_lqr_schedule(tab)
    for i, g in enumerate(lq):
        m = margins(loop_tf(tab["a1"][i], tab["a0"][i], tab["b"][i], g, tau_act=0.0))
        assert m["phase_margin_deg"] >= 60.0
        assert not np.isfinite(m["gain_margin_dB"]) or m["gain_margin_dB"] < 0   # no finite upward limit


def test_lqr_flight_holds_attitude(rocket):
    cfg = SimConfig(t_max=4.5, seed=3)
    fc = FlightComputer(attitude="lqr")
    res = simulate(rocket, cfg, fc)
    ph = np.array(fc.log["phase"])
    assert np.abs(res["u_act"]).max() <= cfg.gimbal_limit_rad + 1e-12
    assert np.degrees(np.abs(np.array(fc.log["att_err"])[ph == 1])).max() < 1.0
    lqr_sched, _ = design_lqr_schedule(design_schedule(rocket)[1])
    tl = fc.log["t"][-1] - fc.t_ld
    assert fc.log["gains"][-1] == lqr_sched(tl) != design_schedule(rocket)[0](tl)    # really used LQR gains


def test_default_pid_unchanged(rocket):
    cfg = dict(t_max=3.0, seed=2)
    a = simulate(rocket, SimConfig(**cfg), FlightComputer())
    b = simulate(rocket, SimConfig(**cfg), FlightComputer(attitude="pid"))
    assert np.array_equal(a["x"], b["x"]) and np.array_equal(a["u_cmd"], b["u_cmd"])
    sched, _ = design_schedule(rocket)
    fc = FlightComputer()
    simulate(rocket, SimConfig(**cfg), fc)
    assert fc.log["gains"][-1] == sched(fc.log["t"][-1] - fc.t_ld)
