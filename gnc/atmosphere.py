"""ISA (US Standard Atmosphere 1976, up to 84.8 km geopotential) and a seeded wind model.

Altitudes passed to `isa` are *geometric* altitudes above mean sea level; the launch-site elevation
(rocket_params launch.elevation_m) is added by the caller (dynamics / sensors).
"""
import math

import numpy as np

G0 = 9.80665
R_AIR = 287.05287          # J/(kg K)
GAMMA = 1.4
R_EARTH = 6356766.0        # m, US1976 effective radius for geopotential conversion

# (base geopotential altitude m, base temperature K, lapse rate K/m)
_LAYERS = [(0.0, 288.15, -0.0065), (11000.0, 216.65, 0.0), (20000.0, 216.65, 0.001), (32000.0, 228.65, 0.0028),
           (47000.0, 270.65, 0.0), (51000.0, 270.65, -0.0028), (71000.0, 214.65, -0.002), (84852.0, 186.946, 0.0)]


def _layer_p(Hb, Tb, L, pb, H):
    if L == 0.0:
        return pb * math.exp(-G0 * (H - Hb) / (R_AIR * Tb))
    return pb * ((Tb + L * (H - Hb)) / Tb) ** (-G0 / (L * R_AIR))


_PB = [101325.0]
for _i in range(1, len(_LAYERS)):
    _Hb, _Tb, _L = _LAYERS[_i - 1]
    _PB.append(_layer_p(_Hb, _Tb, _L, _PB[-1], _LAYERS[_i][0]))
_EXP0 = -G0 / (-0.0065 * R_AIR)   # 5.2559 troposphere exponent


def geopotential(h):
    return R_EARTH * h / (R_EARTH + h)


def isa(h):
    """(T [K], p [Pa], rho [kg/m^3], a [m/s]) at geometric altitude h [m ASL] (scalar float, fast)."""
    H = R_EARTH * h / (R_EARTH + h)
    if H < 11000.0:
        T = 288.15 - 0.0065 * H
        p = 101325.0 * (T / 288.15) ** _EXP0
    else:
        i = len(_LAYERS) - 1
        while H < _LAYERS[i][0]:
            i -= 1
        Hb, Tb, L = _LAYERS[i]
        T = Tb + L * (H - Hb)
        p = _layer_p(Hb, Tb, L, _PB[i], H)
    return T, p, p / (R_AIR * T), math.sqrt(GAMMA * R_AIR * T)


def pressure_altitude(p):
    """Geometric altitude ASL [m] whose ISA pressure is p (troposphere + lower stratosphere, p > 5.5 kPa)."""
    if p >= _PB[1]:
        H = 288.15 / 0.0065 * (1.0 - (p / 101325.0) ** (1.0 / _EXP0))
    else:  # shortcut: isothermal 11-20 km layer only; ample for a 3-4 km apogee
        H = 11000.0 - R_AIR * 216.65 / G0 * math.log(p / _PB[1])
    return R_EARTH * H / (R_EARTH - H)


class WindModel:
    """ENU wind w(t, z_agl) = steady * shear(z) + gust(t).

    steady : ENU vector [m/s] (wind *velocity*, i.e. the direction the air moves; [-5,0,0] = wind from the East).
    shear_exponent : power law shear, factor (max(z, z_ref)/z_ref)**n; steady wind is the value at z_ref (0 = uniform).
    gust : None or dict with optional keys
        std_mps  : turbulence intensity (first-order Gauss-Markov per axis; vertical std = std/2) [m/s]
        tau_s    : correlation time [s] (default 1.5)
        one_cos  : list of discrete gusts {"t0": s, "duration": s, "amplitude": [E, N, U] m/s}, 1-cos shape
    Turbulence is pre-generated on a uniform 0.02 s grid from `seed` (deterministic) and linearly interpolated.
    """

    def __init__(self, steady=(0.0, 0.0, 0.0), gust=None, shear_exponent=0.0, ref_alt_m=10.0, t_max=600.0, seed=0):
        self.steady = tuple(float(v) for v in steady)
        self.n = float(shear_exponent)
        self.z_ref = float(ref_alt_m)
        gust = gust or {}
        self.one_cos = [(float(g["t0"]), float(g["duration"]), np.asarray(g["amplitude"], float))
                        for g in gust.get("one_cos", [])]
        std = float(gust.get("std_mps", 0.0))
        self.dtg = 0.02
        self.turb = None
        if std > 0:
            tau = float(gust.get("tau_s", 1.5))
            n = int(t_max / self.dtg) + 2
            sig = std * np.array([1.0, 1.0, 0.5])
            phi = math.exp(-self.dtg / tau)
            w = np.random.default_rng(seed).standard_normal((n, 3)) * sig
            g = np.empty((n, 3))
            g[0] = w[0]
            for k in range(1, n):
                g[k] = phi * g[k - 1] + math.sqrt(1 - phi * phi) * w[k]
            self.turb = g.tolist()
        self.calm = not any(self.steady) and self.turb is None and not self.one_cos

    def __call__(self, t, z_agl):
        if self.calm:
            return 0.0, 0.0, 0.0
        f = (max(z_agl, self.z_ref) / self.z_ref) ** self.n if self.n else 1.0
        wx, wy, wz = self.steady[0] * f, self.steady[1] * f, self.steady[2] * f
        if self.turb is not None:
            k = t / self.dtg
            i = min(max(int(k), 0), len(self.turb) - 2)
            a = min(max(k - i, 0.0), 1.0)
            g0, g1 = self.turb[i], self.turb[i + 1]
            wx += g0[0] + a * (g1[0] - g0[0])
            wy += g0[1] + a * (g1[1] - g0[1])
            wz += g0[2] + a * (g1[2] - g0[2])
        for t0, dur, amp in self.one_cos:
            if t0 <= t <= t0 + dur:
                s = 0.5 * (1 - math.cos(2 * math.pi * (t - t0) / dur))
                wx += s * amp[0]
                wy += s * amp[1]
                wz += s * amp[2]
        return wx, wy, wz
