"""Independent checker tests for sim-core (contract conformance, physics, sensors, actuators, robustness)."""
import copy
import json
import math

import numpy as np
import pytest

from gnc.atmosphere import isa, pressure_altitude
from gnc.dynamics import Env, aero_state, derivatives
from gnc.frames import nose_direction, quat_from_axis_angle, quat_to_dcm
from gnc.rocket import ROOT, Rocket, read_eng
from gnc.sensors import Sensors
from gnc.sim import SimConfig, _rk4, simulate

G = 9.80665


@pytest.fixture(scope="module")
def rocket():
    return Rocket.load()


@pytest.fixture(scope="module")
def flight(rocket):
    return simulate(rocket, SimConfig(sensor_noise_scale=0.0, stop_at="ground", recovery=True, t_max=600.0))


class Ctrl:
    """Records meas; command = cmd(t)."""

    def __init__(self, cmd=None, dt=0.01):
        self.dt, self.cmd = dt, cmd or (lambda t: np.zeros(2))

    def reset(self, rocket, cfg):
        self.log = {"t": [], "meas": []}

    def update(self, t, meas):
        self.log["t"].append(t)
        self.log["meas"].append(meas)
        return self.cmd(t)


# ---------------------------------------------------------------- contract
def test_contract_simconfig_and_result_keys(rocket):
    cfg = SimConfig()
    for f in ("dt", "t_max", "stop_at", "wind", "gust", "thrust_misalignment_rad", "sensor_noise_scale",
              "actuator_tau_s", "actuator_rate_limit_rad_s", "actuator_delay_s", "estimation_delay_s",
              "gimbal_limit_rad", "seed", "recovery", "perfect_nav"):
        assert hasattr(cfg, f), f
    assert math.isclose(cfg.gimbal_limit_rad, math.radians(5))
    c = Ctrl()
    res = simulate(rocket, SimConfig(t_max=0.5), c)
    for k in ("t", "x", "u_cmd", "u_act", "thrust", "mass", "cg", "mach", "alpha", "q_dyn", "events",
              "meas_log", "controller_log", "cfg"):
        assert k in res, k
    n = len(res["t"])
    assert res["x"].shape == (n, 13) and res["u_cmd"].shape == (n, 2) and res["u_act"].shape == (n, 2)
    for k in ("thrust", "mass", "cg", "mach", "alpha", "q_dyn"):
        assert res[k].shape == (n,), k
    keys = {"imu_accel", "imu_gyro", "baro_alt", "gps_pos", "gps_vel", "mag", "t_launch_detect", "truth"}
    assert all(set(m) == keys for m in c.log["meas"])
    # at controller 100 Hz: IMU every call, baro/mag every 2nd, GPS every 10th call
    none = {k: sum(m[k] is None for m in c.log["meas"]) / len(c.log["meas"]) for k in keys - {"truth"}}
    assert none["imu_accel"] == 0 and abs(none["baro_alt"] - 0.5) < 0.05 and abs(none["gps_pos"] - 0.9) < 0.05


@pytest.mark.parametrize("t", [0.5, 2.0, 4.0])
def test_tvc_force_moment_matches_contract(rocket, t):
    """Zero airspeed (no aero): acceleration and pitch/yaw rate derivative must equal the contract formula."""
    q = quat_from_axis_angle([0.3, -0.5, 0.8], 0.7)
    R = quat_to_dcm(q)
    T, m, xcg, Ixx, Iyy, _ = rocket._thrust_mass(t)
    assert T > 100
    for dp, dy in [(0.05, 0.0), (0.0, -0.08), (0.087, 0.087), (-0.03, 0.06)]:
        x = np.zeros(13)
        x[6:10] = q
        xd = derivatives(t, x, [dp, dy], rocket, Env())
        F = T * np.array([math.cos(dy) * math.cos(dp), math.sin(dy) * math.cos(dp), -math.sin(dp)])
        M = np.cross([-(rocket.x_gimbal - xcg), 0, 0], F)
        assert np.allclose(xd[3:6], R @ F / m - [0, 0, G], atol=1e-9)
        assert np.allclose(xd[10:13] * [Ixx, Iyy, Iyy], M, atol=1e-6)


# ---------------------------------------------------------------- physics
def test_isa_table_values():
    for h, T, p, rho in [(0, 288.15, 101325, 1.2250), (1000, 281.651, 89874.6, 1.11164),
                         (11000, 216.774, 22699.9, 0.36480), (20000, 216.65, 5529.3, 0.08891)]:
        Ti, pi, rhoi, _ = isa(h)
        assert abs(Ti - T) < 0.01 and abs(pi / p - 1) < 1e-4 and abs(rhoi / rho - 1) < 1e-4
    assert abs(isa(0)[3] - 340.294) < 0.01
    for h in (0, 850, 4000, 12000):
        assert abs(pressure_altitude(isa(h)[1]) - h) < 1e-6


def test_motor_impulse_and_mass_flow(rocket):
    hdr, t, F = read_eng(ROOT / "data" / "motor.eng")
    assert math.isclose(rocket.total_impulse, np.trapezoid(F, t), rel_tol=1e-12)
    assert math.isclose(rocket.m_prop, hdr["propellant_mass_kg"], rel_tol=1e-3)
    tt = np.linspace(0, rocket.burn_time + 0.5, 200001)
    assert math.isclose(np.trapezoid(rocket.mdot(tt), tt), rocket.m_prop, rel_tol=1e-4)
    for tc in (0.5, 2.0, 4.2):
        h = 1e-5
        assert math.isclose(-(rocket.mass(tc + h) - rocket.mass(tc - h)) / (2 * h), rocket.mdot(tc), rel_tol=1e-5)


def _vacuum(rocket):
    p = copy.deepcopy(rocket.params)
    a = p["aero"]
    n = len(a["mach"])
    a.update(cd_power_off=[0.0] * n, cd_power_on=[0.0] * n, cn_alpha=[0.0] * n, cmq=0.0, clp=0.0)
    return Rocket(p)


def test_vacuum_energy_and_momentum(rocket):
    rv, env, t, h = _vacuum(rocket), Env(), 10.0, 1e-3
    x = np.zeros(13)
    x[2], x[3:6], x[6:10], x[10:13] = 1000, [30, -20, 150], quat_from_axis_angle([1, 2, 3], 0.4), [3.0, 0.4, -0.2]
    m, J = rv.mass(t), np.diag([rv.Ixx(t), rv.Iyy(t), rv.Iyy(t)])

    def inv(x):
        w = x[10:13]
        return (0.5 * m * x[3:6] @ x[3:6] + m * G * x[2], 0.5 * w @ J @ w, quat_to_dcm(x[6:10]) @ J @ w, x[3:5])
    e0 = inv(x)
    f = lambda tt, s: derivatives(tt, s, [0, 0], rv, env)  # noqa: E731
    for _ in range(5000):
        x = _rk4(f, t, x, h, f(t, x))
        t += h
    e1 = inv(x)
    assert math.isclose(e0[0], e1[0], rel_tol=1e-9) and math.isclose(e0[1], e1[1], rel_tol=1e-9)
    assert np.allclose(e0[2], e1[2], rtol=1e-8) and np.allclose(e0[3], e1[3])


def test_free_fall_analytic(rocket):
    rv, env = _vacuum(rocket), Env()
    x = np.zeros(13)
    x[2], x[5], x[6] = 2000.0, 12.0, 1.0
    f = lambda tt, s: derivatives(tt, s, [0, 0], rv, env)  # noqa: E731
    t = 10.0
    for _ in range(3000):
        x = _rk4(f, t, x, 1e-3, f(t, x))
        t += 1e-3
    assert math.isclose(x[2], 2000 + 12 * 3 - 0.5 * G * 9, rel_tol=1e-10) and math.isclose(x[5], 12 - G * 3)


def test_weathercock_into_relative_wind(rocket):
    """Vertical coasting rocket in a crosswind blowing towards +E: the nose must turn upwind (towards -E)."""
    env = Env(wind=lambda t, z: (8.0, 0.0, 0.0))
    x = np.zeros(13)
    x[2], x[5], x[6] = 500.0, 150.0, 1.0
    q = quat_from_axis_angle([0, 1, 0], -math.pi / 2)    # body x -> Up
    x[6:10] = q
    assert np.allclose(nose_direction(q), [0, 0, 1])
    f = lambda tt, s: derivatives(tt, s, [0, 0], rocket, env)  # noqa: E731
    t = 6.0
    for _ in range(100):
        x = _rk4(f, t, x, 1e-3, f(t, x))
        t += 1e-3
    assert nose_direction(x[6:10])[0] < -1e-3


@pytest.mark.parametrize("t0", [1.0, 3.0, 6.0, 10.0, 15.0])
def test_short_period_frequency_and_damping(rocket, flight, t0):
    """Eigenvalues of the numeric Jacobian on the passive trajectory vs sqrt(q S CNa (x_cp-x_cg) / Iyy)."""
    env = Env()
    k = np.searchsorted(flight["t"], t0)
    x0 = flight["x"][k]
    J = np.zeros((13, 13))
    for j in range(13):
        e = np.zeros(13)
        e[j] = 1e-6 * max(1.0, abs(x0[j]))
        J[:, j] = (derivatives(t0, x0 + e, [0, 0], rocket, env) - derivatives(t0, x0 - e, [0, 0], rocket, env)) \
            / (2 * e[j])
    lam = np.linalg.eigvals(J)
    osc = lam[np.abs(lam.imag) > 0.5]
    a = aero_state(t0, x0, rocket, env)
    wn = math.sqrt(a["q_dyn"] * rocket.S * a["cn_alpha"] * (a["x_cp"] - a["x_cg"]) / rocket.Iyy(t0))
    assert len(osc) == 4                                   # pitch + yaw pair
    assert np.allclose(np.abs(osc), wn, rtol=0.08)
    assert np.all(osc.real < 0)                            # damped


def test_pitch_damping_damps_rate_in_still_air(rocket):
    env = Env()
    x = np.zeros(13)
    x[2], x[5], x[6:10], x[11] = 500.0, 200.0, quat_from_axis_angle([0, 1, 0], -math.pi / 2), 0.5
    xd = derivatives(8.0, x, [0, 0], rocket, env)
    assert xd[11] < 0 and xd[12] == 0


def test_rail_phase(rocket):
    res = simulate(rocket, SimConfig(sensor_noise_scale=0.0, wind=(-10.0, 3.0, 0.0), t_max=1.0))
    t, X, ev = res["t"], res["x"], res["events"]
    on = t < ev["rail_exit"]
    assert np.all(X[on, 10:13] == 0) and np.all(X[on, 6:10] == X[0, 6:10])
    k = np.searchsorted(t, ev["rail_exit"])
    s = X[:, 0:3] @ res["rail_dir"]
    assert s[k - 1] < 5.4864 <= s[k] and s[k] - 5.4864 < np.linalg.norm(X[k, 3:6]) * 1e-3 + 1e-9
    assert np.allclose(np.cross(X[on, 0:3], res["rail_dir"]), 0, atol=1e-9)
    # liftoff when thrust first exceeds the weight component along the rail
    T_lift = rocket.mass(ev["liftoff"]) * G * math.cos(math.radians(6))
    assert rocket.thrust(ev["liftoff"]) >= T_lift > rocket.thrust(ev["liftoff"] - 1e-3)


def test_apogee_and_recovery_descent_rates(rocket, flight):
    t, X, ev = flight["t"], flight["x"], flight["events"]
    i = np.argmax(X[:, 2])
    assert abs(ev["apogee"] - t[i]) < 0.06
    assert abs(np.interp(ev["apogee"], t, X[:, 2]) - X[:, 2].max()) < 0.1
    assert list(ev) and ev["liftoff"] < ev["rail_exit"] < ev["burnout"] < ev["apogee"] <= ev["main"] < ev["landing"]
    m = rocket.m_dry
    for cda, z in [(0.55, 1500.0), (10.0, 100.0)]:
        k = np.argmin(np.abs(X[:, 2] - z) + 1e6 * (t < ev["apogee"]))
        v_term = math.sqrt(2 * m * G / (isa(850 + z)[2] * cda))
        assert abs(-X[k, 5] / v_term - 1) < 0.03, (cda, -X[k, 5], v_term)
    assert X[-1, 2] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------- sensors
def test_accelerometer_on_pad_reads_plus_g():
    """Contract/doc: accel truth = R^T(vdot + g z) INCLUDING the rail reaction -> at rest on the pad the IMU reads
    +g (g cos6 along the nose).  Also checks the logged kinematic acceleration is zero at rest."""
    r = Rocket.load()
    res = simulate(r, SimConfig(sensor_noise_scale=0.0, t_max=0.002))
    f0 = res["meas_log"]["imu_accel"]["y"][0]
    assert np.linalg.norm(f0) == pytest.approx(G, rel=1e-3), f0
    assert f0[0] == pytest.approx(G * math.cos(math.radians(6)), rel=1e-3)
    assert np.allclose(res["accel"][0], 0.0, atol=1e-9)


def test_accelerometer_on_rail_includes_normal_reaction(rocket):
    res = simulate(rocket, SimConfig(sensor_noise_scale=0.0, t_max=0.3))
    t, X = res["t"], res["x"]
    k = np.searchsorted(t, 0.2)
    a_kin = (X[k + 1, 3:6] - X[k - 1, 3:6]) / (t[k + 1] - t[k - 1])
    f_true = quat_to_dcm(X[k, 6:10]).T @ (a_kin + [0, 0, G])
    j = np.searchsorted(res["meas_log"]["imu_accel"]["t"], t[k])
    assert np.allclose(res["meas_log"]["imu_accel"]["y"][j], f_true, atol=0.05)


def test_accelerometer_near_zero_near_apogee_and_baro_gps(rocket, flight):
    ml, t, X, ta = flight["meas_log"], flight["t"], flight["x"], flight["events"]["apogee"]
    near = np.abs(ml["imu_accel"]["t"] - ta) < 1.0
    assert np.all(np.linalg.norm(ml["imu_accel"]["y"][near], axis=1) < 0.3)
    asc = t < ta
    for key, sl in (("baro_alt", 2), ("gps_pos", slice(0, 3))):
        mt, my = ml[key]["t"], ml[key]["y"]
        sel = mt < ta
        truth = np.column_stack([np.interp(mt[sel], t[asc], X[asc, i]) for i in np.atleast_1d(np.arange(13)[sl])])
        assert np.allclose(np.reshape(my[sel], truth.shape), truth, atol=1e-3), key


@pytest.mark.parametrize("scale", [0.0, 1.0, 3.0])
def test_sensor_noise_scale_and_bias(scale):
    cfg = SimConfig(sensor_noise_scale=scale, accel_bias=(0.1, -0.2, 0.3), gps_pos_bias=(1.0, 2.0, -3.0),
                    baro_bias_pa=0.0)
    s = Sensors(cfg, 850.0, seed=5)
    x = np.concatenate([[0, 0, 500.0], [0, 0, 0], [1.0, 0, 0, 0], [0, 0, 0]])
    acc = np.array([s.imu(x, [0, 0, -G])[0] for _ in range(20000)])
    gps = np.array([s.gps(x)[0] for _ in range(20000)])
    baro = np.array([s.baro(x) for _ in range(20000)])
    assert np.allclose(acc.mean(0), cfg.accel_bias, atol=0.01 + 0.003 * scale)  # free fall: f = bias only
    assert np.allclose(acc.std(0), cfg.accel_std * scale, rtol=0.03, atol=1e-12)
    assert np.allclose(gps.mean(0), np.array(cfg.gps_pos_bias) + [0, 0, 500], atol=0.1 * scale + 1e-9)
    assert np.allclose(gps.std(0), np.array(cfg.gps_pos_std) * scale, rtol=0.03, atol=1e-12)
    # 6 Pa near 1350 m ASL -> 6 / (rho g) metres
    sig_h = cfg.baro_std_pa / (isa(1350.0)[2] * G) * scale
    assert baro.std() == pytest.approx(sig_h, rel=0.03, abs=1e-9)


def test_sensor_rates_in_sim_and_seed(rocket):
    a = simulate(rocket, SimConfig(t_max=2.0, seed=4))
    b = simulate(rocket, SimConfig(t_max=2.0, seed=4))
    c = simulate(rocket, SimConfig(t_max=2.0, seed=5))
    for k, hz in (("imu_accel", 200), ("baro_alt", 50), ("mag", 50), ("gps_pos", 10)):
        tt = a["meas_log"][k]["t"]
        assert np.allclose(np.diff(tt), 1 / hz, atol=1.1e-3), k
        assert np.array_equal(a["meas_log"][k]["y"], b["meas_log"][k]["y"])
        assert not np.array_equal(a["meas_log"][k]["y"], c["meas_log"][k]["y"])
    w1 = simulate(rocket, SimConfig(t_max=3.0, wind=(3, 0, 0), gust={"std_mps": 2.0}, seed=1))["x"]
    w2 = simulate(rocket, SimConfig(t_max=3.0, wind=(3, 0, 0), gust={"std_mps": 2.0}, seed=2))["x"]
    assert not np.allclose(w1, w2)


# ---------------------------------------------------------------- actuator / delays
def _first(t, mask):
    return t[np.argmax(mask)]


@pytest.mark.parametrize("delay", [0.05, 0.123])
def test_actuator_delay_measured(rocket, delay):
    val = math.radians(3)
    cfg = SimConfig(sensor_noise_scale=0, t_max=0.6, actuator_tau_s=0, actuator_rate_limit_rad_s=1e6,
                    actuator_delay_s=delay)
    res = simulate(rocket, cfg, Ctrl(lambda t: np.array([val if t >= 0.1 - 1e-9 else 0.0, 0.0])))
    t_act = _first(res["t"], res["u_act"][:, 0] > 0)
    assert abs(t_act - (0.1 + delay)) <= cfg.dt + 1e-9


def test_actuator_lag_rate_and_limit(rocket):
    lag = simulate(rocket, SimConfig(sensor_noise_scale=0, t_max=0.6, actuator_tau_s=0.05,
                                     actuator_rate_limit_rad_s=1e6),
                   Ctrl(lambda t: np.array([math.radians(3) if t >= 0.1 - 1e-9 else 0.0, 0.0])))
    t, ua = lag["t"], lag["u_act"][:, 0] / math.radians(3)
    assert abs(np.interp(0.1 + 0.05, t, ua) - (1 - math.exp(-1))) < 0.03
    rl = simulate(rocket, SimConfig(sensor_noise_scale=0, t_max=0.6, actuator_tau_s=0,
                                    actuator_rate_limit_rad_s=math.radians(20)),
                  Ctrl(lambda t: np.array([5.0, -5.0])))
    du = np.abs(np.diff(rl["u_act"], axis=0)) / 1e-3
    assert du.max() <= math.radians(20) * (1 + 1e-9)
    assert np.isclose(np.abs(rl["u_act"]).max(), math.radians(5))


def test_estimation_delay_measured(rocket):
    for delay in (0.03, 0.1):
        c = Ctrl()
        res = simulate(rocket, SimConfig(sensor_noise_scale=0, t_max=1.0, estimation_delay_s=delay), c)
        ml = res["meas_log"]["baro_alt"]
        ages = [tc - ml["t"][np.argmin(np.abs(ml["y"] - m["baro_alt"]))]
                for tc, m in zip(c.log["t"], c.log["meas"]) if m["baro_alt"] is not None and tc > 0.5]
        assert np.allclose(ages, delay, atol=1e-9 + 1 / 50)
        assert min(ages) >= delay - 1e-9


# ---------------------------------------------------------------- robustness
def test_no_nan_under_stress(rocket):
    gust = {"std_mps": 3.0, "tau_s": 1.0, "one_cos": [{"t0": 2.0, "duration": 1.0, "amplitude": [0, 10, 0]}]}
    cases = [
        (SimConfig(wind=(15.0, 0.0, 0.0), gust=gust, sensor_noise_scale=10.0, stop_at="ground", recovery=True,
                   wind_shear_exponent=1 / 7, t_max=400.0), None),
        (SimConfig(wind=(0.0, -15.0, 0.0), gust=gust, sensor_noise_scale=10.0, t_max=60.0, gimbal_limit_rad=0.3),
         Ctrl(lambda t: np.array([1e3, -1e3]) * (1 if int(t * 7) % 2 else -1))),
        (SimConfig(t_max=5.0, sensor_noise_scale=10.0), None),     # t_max hit before apogee
    ]
    for cfg, ctrl in cases:
        res = simulate(rocket, cfg, ctrl)
        for k in ("x", "u_act", "thrust", "mass", "mach", "q_dyn"):
            assert np.all(np.isfinite(res[k])), k
        for k, v in res["meas_log"].items():
            assert np.all(np.isfinite(v["y"])), k
        assert np.allclose(np.linalg.norm(res["x"][:, 6:10], axis=1), 1, atol=1e-9)
    assert "apogee" not in res["events"] and res["t"][-1] == pytest.approx(5.0)


# ---------------------------------------------------------------- results/passive.json
def test_passive_json_schema():
    path = ROOT / "results" / "passive.json"
    if not path.exists():
        pytest.skip("results/passive.json not generated")
    d = json.loads(path.read_text(), parse_constant=lambda c: pytest.fail(f"non-JSON constant {c}"))
    for k in ("id", "group", "title", "description", "variants"):
        assert k in d
    for v in d["variants"]:
        n = len(v["t"])
        assert n > 10 and np.all(np.diff(v["t"]) > 0)
        pos, quat = np.array(v["rocket"]["pos"], float), np.array(v["rocket"]["quat"], float)
        assert pos.shape == (n, 3) and quat.shape == (n, 4)
        assert np.all(np.isfinite(pos)) and np.allclose(np.linalg.norm(quat, axis=1), 1, atol=1e-4)
        ts = [e["t"] for e in v["events"]]
        assert ts == sorted(ts) and all(isinstance(e["label"], str) for e in v["events"])
        for p in v["plots"]:
            for s in p["series"]:
                assert len(s["x"]) == len(s["y"]) and None not in s["x"]
        assert all(m is None or np.isfinite(m) for m in v["metrics"].values())
