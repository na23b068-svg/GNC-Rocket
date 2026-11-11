import numpy as np
import pytest

from gnc.rocket import DATA, Rocket
from gnc.sim import SimConfig, simulate


@pytest.fixture(scope="module")
def rocket():
    return Rocket.load(DATA / "rocket_params.placeholder.json")


@pytest.fixture(scope="module")
def nominal(rocket):
    return simulate(rocket, SimConfig())


class Recorder:
    def __init__(self, dt=0.01, cmd=None):
        self.dt, self.cmd, self.log = dt, cmd or (lambda t: np.zeros(2)), {"t": [], "meas": []}

    def reset(self, rocket, cfg):
        self.log = {"t": [], "meas": []}

    def update(self, t, meas):
        self.log["t"].append(t)
        self.log["meas"].append(meas)
        return self.cmd(t)


def test_passive_apogee_and_events(nominal, rocket):
    ev, s = nominal["events"], nominal["summary"]
    assert 2000 < s["apogee_agl_m"] < 4000
    assert ev["liftoff"] < ev["rail_exit"] < ev["burnout"] < ev["apogee"]
    assert 20 < s["off_rail_velocity_mps"] < 40 and s["max_mach"] < 1.0
    N = len(nominal["t"])
    for k in ["u_cmd", "u_act"]:
        assert nominal[k].shape == (N, 2)
    assert nominal["x"].shape == (N, 13)
    for k in ["thrust", "mass", "cg", "mach", "alpha", "q_dyn"]:
        assert nominal[k].shape == (N,)
    assert np.allclose(np.linalg.norm(nominal["x"][:, 6:10], axis=1), 1)
    # passive, fin-stabilised: small AoA with no wind (until the low-speed arc-over just before apogee)
    t = nominal["t"]
    assert np.degrees(nominal["alpha"][(t > ev["rail_exit"] + 0.5) & (t < ev["apogee"] - 5)]).max() < 2.0


def test_deterministic(rocket):
    cfg = SimConfig(wind=(-5, 0, 0), gust={"std_mps": 2.0}, seed=7, t_max=6.0)
    a, b = simulate(rocket, cfg), simulate(rocket, cfg)
    assert np.array_equal(a["x"], b["x"]) and np.array_equal(a["meas_log"]["imu_accel"]["y"],
                                                             b["meas_log"]["imu_accel"]["y"])
    c = simulate(rocket, SimConfig(wind=(-5, 0, 0), gust={"std_mps": 2.0}, seed=8, t_max=6.0))
    assert not np.array_equal(a["x"], c["x"])


def test_controller_hook_and_meas(rocket):
    ctl = Recorder(dt=0.01)
    res = simulate(rocket, SimConfig(t_max=2.0), ctl)
    t = np.array(ctl.log["t"])
    assert np.allclose(np.diff(t), 0.01) and t[0] == 0.0 and len(t) == 200
    m = ctl.log["meas"]
    keys = {"imu_accel", "imu_gyro", "baro_alt", "gps_pos", "gps_vel", "mag", "t_launch_detect", "truth"}
    assert keys <= set(m[0])
    assert all(mm["imu_accel"] is not None for mm in m)                         # 200 Hz IMU > 100 Hz ctl
    assert sum(mm["gps_pos"] is not None for mm in m) == 20                     # 10 Hz
    assert m[-1]["t_launch_detect"] is not None and m[-1]["t_launch_detect"] < 0.1
    assert res["controller_log"] is ctl.log


def test_actuator_limits_rate_and_delay(rocket):
    cfg = SimConfig(t_max=3.0, actuator_delay_s=0.05, actuator_tau_s=0.02, actuator_rate_limit_rad_s=1.0)
    ctl = Recorder(cmd=lambda t: np.array([1.0, -1.0]) if t >= 1.0 else np.zeros(2))
    res = simulate(rocket, cfg, ctl)
    ua, t = res["u_act"], res["t"]
    assert np.abs(ua).max() <= cfg.gimbal_limit_rad + 1e-12
    assert np.isclose(ua[-1, 0], cfg.gimbal_limit_rad, atol=1e-6) and np.isclose(ua[-1, 1], -cfg.gimbal_limit_rad,
                                                                                    atol=1e-6)
    assert np.abs(np.diff(ua, axis=0)).max() <= cfg.actuator_rate_limit_rad_s * cfg.dt + 1e-12
    assert np.all(ua[t < 1.05 - 1e-9] == 0) and ua[np.searchsorted(t, 1.06), 0] > 0


def test_estimation_delay(rocket):
    cfg = SimConfig(t_max=8.0, sensor_noise_scale=0.0, estimation_delay_s=0.1)
    ctl = Recorder()
    res = simulate(rocket, cfg, ctl)
    tc = np.array(ctl.log["t"])
    alt = np.array([m["baro_alt"] if m["baro_alt"] is not None else np.nan for m in ctl.log["meas"]])
    ok = ~np.isnan(alt) & (tc > 2)
    truth = np.interp(tc[ok] - 0.1, res["t"], res["x"][:, 2])
    assert np.allclose(alt[ok], truth, atol=1e-3)


def test_recovery_full_flight(rocket):
    res = simulate(rocket, SimConfig(stop_at="ground", recovery=True, wind=(3, 0, 0)))
    ev = res["events"]
    assert ev["apogee"] == ev["drogue"] < ev["main"] < ev["landing"]
    assert abs(res["x"][-1, 2]) < 1e-6 and res["summary"]["landing_distance_m"] > 0
    zmain = np.interp(ev["main"], res["t"], res["x"][:, 2])
    assert abs(zmain - 300) < 5
    vz_end = res["x"][-2, 5]
    assert -8 < vz_end < -3                                                         # main chute descent rate


def test_ballistic_to_ground(rocket):
    res = simulate(rocket, SimConfig(stop_at="ground"))
    assert "landing" in res["events"] and abs(res["x"][-1, 2]) < 1e-6

