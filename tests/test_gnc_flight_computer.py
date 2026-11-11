"""Flight computer: Controller protocol + closed-loop smoke tests against the real simulator."""
import numpy as np
import pytest

from gnc.flight_computer import FlightComputer, design_schedule
from gnc.frames import tilt_from_vertical
from gnc.pid import closed_loop_poles
from gnc.rocket import Rocket
from gnc.sim import SimConfig, simulate


@pytest.fixture(scope="module")
def rocket():
    return Rocket.load()


def test_schedule_places_requested_poles(rocket):
    _, tab = design_schedule(rocket)
    for i in range(len(tab["t"])):
        got = np.sort_complex(closed_loop_poles(tab["a1"][i], tab["a0"][i], tab["b"][i],
                                                tab["Kp"][i], tab["Ki"][i], tab["Kd"][i]))
        want = np.sort_complex(np.array([complex(*p) for p in tab["poles"][i]]))
        assert np.allclose(got, want, atol=1e-6)


def test_protocol(rocket):
    fc = FlightComputer()
    assert fc.dt > 0
    fc.reset(rocket, SimConfig())
    meas = {"imu_accel": np.array([9.75, 0, 0]), "imu_gyro": np.zeros(3), "baro_alt": 0.0,
            "gps_pos": np.zeros(3), "gps_vel": np.zeros(3), "mag": np.array([0.0, 0.5, -0.8]),
            "t_launch_detect": None, "truth": None}
    u = fc.update(0.0, meas)
    assert np.shape(u) == (2,) and np.all(u == 0)         # gimbal locked on the rail
    assert all(len(v) == 1 for k, v in fc.log.items() if k in ("t", "u_cmd", "pos_est", "phase"))


@pytest.mark.parametrize("nav", ["kf", "luenberger"])
def test_attitude_hold_flight(rocket, nav):
    cfg = SimConfig(t_max=6.0, seed=3)
    fc = FlightComputer(nav=nav)
    res = simulate(rocket, cfg, fc)
    L = fc.log
    ph = np.array(L["phase"])
    u = np.array(res["u_act"])
    assert np.all(np.abs(u) <= cfg.gimbal_limit_rad + 1e-12)
    assert set(np.unique(ph)) == {0, 1, 2}
    err = np.degrees(np.abs(np.array(L["att_err"])[ph == 1]))
    assert err.max() < 1.0
    i_bo = np.searchsorted(res["t"], fc.t_tvc_end)
    tilt = np.degrees(tilt_from_vertical(res["x"][i_bo, 6:10]))
    assert abs(tilt - 6.0) < 1.0                           # held the rail attitude through the boost
    lt = np.array(L["t"])
    p_true = np.array([np.interp(lt, res["t"], res["x"][:, j]) for j in range(3)]).T
    assert np.sqrt(np.mean((np.array(L["pos_est"]) - p_true) ** 2)) < 2.0


def test_los_path_flight(rocket):
    W = np.array([[0, 0, 0], [0, 30, 300], [0, 90, 900]], float)
    fc = FlightComputer(guidance="los_path", waypoints=W)
    simulate(rocket, SimConfig(t_max=4.5, seed=1), fc)
    ph = np.array(fc.log["phase"])
    ct = np.array(fc.log["cross_track"])[ph == 1]
    assert np.sqrt(np.mean(ct**2)) < 15.0
    assert fc.log["wp_index"][-1] == 1


def test_schedule_stabilises_full_plane_model(rocket):
    """2nd-order PID design closes the loop on the full 4-state plane model (incl. the unstable drift mode)."""
    from gnc.analysis import plane_pid_closed_loop
    from gnc.linearize import operating_points
    _, tab = design_schedule(rocket)
    pts = operating_points(rocket, tab["t"][::2])
    for i, p in zip(range(0, len(tab["t"]), 2), pts):
        assert np.linalg.eigvals(p["pitch"]["A"]).real.max() > 0          # open loop: drift divergence
        ev = np.linalg.eigvals(plane_pid_closed_loop(p["pitch"]["A"], p["pitch"]["B"],
                                                     tab["Kp"][i], tab["Ki"][i], tab["Kd"][i]))
        assert np.sort(ev.real)[-2] < -0.05 and ev.real.max() < 1e-4      # only lateral position (~0, density gradient)
