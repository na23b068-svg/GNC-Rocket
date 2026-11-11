import numpy as np

from gnc.atmosphere import G0
from gnc.frames import launch_attitude
from gnc.sensors import Sensors, mag_field_enu
from gnc.sim import SimConfig

X = np.concatenate([[10, 20, 1000], [5, 0, 100], launch_attitude(np.radians(6), np.radians(90)), [0.1, -0.2, 0.3]])


def test_noise_statistics_and_scale():
    for scale in [1.0, 2.5]:
        cfg = SimConfig(sensor_noise_scale=scale)
        s = Sensors(cfg, 850.0, seed=3)
        a = np.array([s.imu(X, [0, 0, 0])[0] for _ in range(20000)])
        g = np.array([s.imu(X, [0, 0, 0])[1] for _ in range(20000)])
        b = np.array([s.baro(X) for _ in range(5000)])
        p = np.array([s.gps(X)[0] for _ in range(20000)])
        assert np.allclose(a.std(0), cfg.accel_std * scale, rtol=0.03)
        assert np.allclose(g.std(0), cfg.gyro_std * scale, rtol=0.03)
        assert np.allclose(p.std(0), np.array(cfg.gps_pos_std) * scale, rtol=0.03)
        assert np.allclose(g.mean(0), X[10:13], atol=1e-4 * scale)
        assert np.isclose(b.mean(), 1000.0, atol=0.1 * scale) and 0.3 * scale < b.std() < 1.0 * scale


def test_noiseless_truth_and_bias():
    cfg = SimConfig(sensor_noise_scale=0.0, gyro_bias=(0.01, 0, 0))
    s = Sensors(cfg, 850.0)
    acc, gyro = s.imu(np.concatenate([[0, 0, 0], [0, 0, 0], launch_attitude(0, 0), [0, 0, 0]]), [0, 0, 0])
    assert np.allclose(acc, [G0, 0, 0]) and np.allclose(gyro, [0.01, 0, 0])
    assert np.isclose(s.baro(X), 1000.0, atol=1e-6)
    pos, vel = s.gps(X)
    assert np.allclose(pos, X[0:3]) and np.allclose(vel, X[3:6])
    m = s.mag(np.concatenate([np.zeros(6), [1, 0, 0, 0], np.zeros(3)]))
    assert np.allclose(m, mag_field_enu()) and np.isclose(np.linalg.norm(m), 1)
    assert np.isclose(np.degrees(np.arcsin(-mag_field_enu()[2])), 58.0)


def test_rates():
    s = Sensors(SimConfig(), 850.0)
    counts = {}
    for t in np.arange(0, 1.0 - 1e-9, 0.001):
        for k in s.step(t, X, [0, 0, 0]):
            counts[k] = counts.get(k, 0) + 1
    assert counts == {"imu_accel": 200, "imu_gyro": 200, "baro_alt": 50, "mag": 50, "gps_pos": 10, "gps_vel": 10}
