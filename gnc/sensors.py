"""Sensor models: IMU (accelerometer + gyro), barometer, GPS, magnetometer.

Each output = truth + constant bias + N(0, (std * sensor_noise_scale)^2), sampled at its own rate.
Default std-devs (SimConfig) are representative of student-rocketry hardware:
  accelerometer 0.08 m/s^2 per axis   (MEMS IMU, e.g. ICM-42688/BMI088, ~150 ug/rtHz at ~100 Hz BW + margin)
  gyro          0.002 rad/s per axis  (~0.1 deg/s; ~0.008 deg/s/rtHz noise density)
  barometer     6 Pa                  (MS5611/BMP388 in flight, ~0.5 m altitude-equivalent near the ground)
  GPS position  1.5 / 1.5 / 3.0 m E/N/U, velocity 0.15 m/s per axis (u-blox M8/M9 class, white noise only)
  magnetometer  0.005 per axis on the unit field vector (~0.3 deg), re-normalised
Local field (Midland TX): declination 9.5 deg E, inclination 58 deg (down).
"""
import math

import numpy as np

from .atmosphere import isa, pressure_altitude
from .frames import quat_to_dcm


def mag_field_enu(declination_deg=9.5, inclination_deg=58.0):
    D, I = math.radians(declination_deg), math.radians(inclination_deg)
    return np.array([math.cos(I) * math.sin(D), math.cos(I) * math.cos(D), -math.sin(I)])


class Sensors:
    """Sensor suite.  `step(t, x, a_enu)` returns the dict of sensors that have a new sample at time t
    (subset of imu_accel, imu_gyro, baro_alt, gps_pos, gps_vel, mag); scheduling is time-based from t = 0."""

    def __init__(self, cfg, elevation_m, seed=0):
        self.cfg = cfg
        self.elev = float(elevation_m)
        self.k = float(cfg.sensor_noise_scale)
        rngs = [np.random.default_rng(s) for s in np.random.SeedSequence([seed, 2]).spawn(4)]
        self.rng_imu, self.rng_baro, self.rng_gps, self.rng_mag = rngs
        self.period = {"imu": 1.0 / cfg.imu_rate_hz, "baro": 1.0 / cfg.baro_rate_hz,
                       "mag": 1.0 / cfg.mag_rate_hz, "gps": 1.0 / cfg.gps_rate_hz}
        self.next = {k: 0.0 for k in self.period}
        self.b_mag = mag_field_enu(cfg.mag_declination_deg, cfg.mag_inclination_deg)

    def _n(self, rng, std):
        std = np.broadcast_to(np.asarray(std, float), (3,))
        return rng.standard_normal(3) * std * self.k

    def imu(self, x, a_enu):
        R = quat_to_dcm(x[6:10])
        f = R.T @ (np.asarray(a_enu, float) + np.array([0.0, 0.0, 9.80665]))
        acc = f + np.asarray(self.cfg.accel_bias, float) + self._n(self.rng_imu, self.cfg.accel_std)
        gyro = x[10:13] + np.asarray(self.cfg.gyro_bias, float) + self._n(self.rng_imu, self.cfg.gyro_std)
        return acc, gyro

    def baro(self, x):
        p = isa(self.elev + x[2])[1] + self.cfg.baro_bias_pa
        p += self.rng_baro.standard_normal() * self.cfg.baro_std_pa * self.k
        return pressure_altitude(p) - self.elev

    def gps(self, x):
        pos = x[0:3] + np.asarray(self.cfg.gps_pos_bias, float) + self._n(self.rng_gps, self.cfg.gps_pos_std)
        vel = x[3:6] + self._n(self.rng_gps, self.cfg.gps_vel_std)
        return pos, vel

    def mag(self, x):
        m = quat_to_dcm(x[6:10]).T @ self.b_mag + np.asarray(self.cfg.mag_bias, float) \
            + self._n(self.rng_mag, self.cfg.mag_std)
        return m / np.linalg.norm(m)

    def step(self, t, x, a_enu):
        out = {}
        eps = 1e-9
        if t >= self.next["imu"] - eps:
            out["imu_accel"], out["imu_gyro"] = self.imu(x, a_enu)
            self.next["imu"] += self.period["imu"]
        if t >= self.next["baro"] - eps:
            out["baro_alt"] = self.baro(x)
            self.next["baro"] += self.period["baro"]
        if t >= self.next["mag"] - eps:
            out["mag"] = self.mag(x)
            self.next["mag"] += self.period["mag"]
        if t >= self.next["gps"] - eps:
            out["gps_pos"], out["gps_vel"] = self.gps(x)
            self.next["gps"] += self.period["gps"]
        return out
