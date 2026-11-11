"""Rocket model: parameters (data/rocket_params.json schema), RASP thrust curve, mass properties, aero tables.

Mass properties interpolate linearly dry <-> wet in burned fraction = cumulative impulse / total impulse
(total impulse integrated from the .eng curve).  Scalar methods are fast (bisect, pure Python) because the
dynamics call them ~10^5 times per flight; they also accept numpy arrays (vectorised).
"""
import json
import math
from bisect import bisect_right
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def read_eng(path):
    """Parse a RASP .eng file -> (header dict, t [s], thrust [N]); a (0, 0) point is prepended if missing."""
    header, t, F = None, [], []
    for line in Path(path).read_text().splitlines():
        s = line.split(";")[0].strip()
        if not s:
            continue
        parts = s.split()
        if header is None:
            header = {"name": parts[0], "diameter_mm": float(parts[1]), "length_mm": float(parts[2]),
                      "delays": parts[3], "propellant_mass_kg": float(parts[4]), "total_mass_kg": float(parts[5]),
                      "manufacturer": " ".join(parts[6:])}
            continue
        try:
            ti, Fi = float(parts[0]), float(parts[1])
        except (ValueError, IndexError):
            break  # next motor in a multi-motor file
        if t and ti <= t[-1]:
            break
        t.append(ti)
        F.append(Fi)
    if not t:
        raise ValueError(f"no thrust data in {path}")
    if t[0] > 0:
        t.insert(0, 0.0)
        F.insert(0, 0.0)
    return header, t, F


def _vec(f):
    """Make a scalar method also accept arrays."""
    def wrapper(self, x, *a):
        if np.ndim(x) == 0:
            return f(self, float(x), *a)
        return np.array([f(self, float(v), *a) for v in np.ravel(x)]).reshape(np.shape(x))
    wrapper.__doc__ = f.__doc__
    return wrapper


class Rocket:
    def __init__(self, params, eng_path=None, source=None):
        self.params = params
        self.source = str(source) if source else None
        self.name = params["name"]
        self.length = float(params["length_m"])
        self.d = float(params["diameter_m"])
        self.S = float(params.get("reference_area_m2") or math.pi * self.d ** 2 / 4)
        self.x_gimbal = float(params["gimbal_station_m"])
        m = params["mass"]
        self.m_dry, self.m_wet = float(m["dry_kg"]), float(m["wet_kg"])
        self.cg_dry, self.cg_wet = float(m["cg_dry_m"]), float(m["cg_wet_m"])
        self.Ixx_dry, self.Ixx_wet = float(m["Ixx_dry"]), float(m["Ixx_wet"])
        self.Iyy_dry, self.Iyy_wet = float(m["Iyy_dry"]), float(m["Iyy_wet"])
        a = params["aero"]
        self._mach = [float(v) for v in a["mach"]]
        self._cd_off = [float(v) for v in a["cd_power_off"]]
        self._cd_on = [float(v) for v in a.get("cd_power_on") or a["cd_power_off"]]
        self._cna = [float(v) for v in a["cn_alpha"]]
        self._cp = [float(v) for v in a["cp_m"]]
        self.cmq, self.clp = float(a["cmq"]), float(a["clp"])
        self.launch = dict(params["launch"])

        eng_path = Path(eng_path or params["motor"]["eng_file"])
        if not eng_path.is_absolute():
            eng_path = ROOT / eng_path
        self.motor_header, self._tc, self._Fc = read_eng(eng_path)
        self.eng_path = str(eng_path)
        self._Ic = [0.0]
        for i in range(1, len(self._tc)):
            self._Ic.append(self._Ic[-1] + 0.5 * (self._Fc[i] + self._Fc[i - 1]) * (self._tc[i] - self._tc[i - 1]))
        self.total_impulse = self._Ic[-1]
        self.burn_time = self._tc[-1]  # thrust is zero for t >= burn_time
        self.m_prop = self.m_wet - self.m_dry

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, path=None, eng_path=None):
        """Load data/rocket_params.json if it exists, else data/rocket_params.placeholder.json."""
        if path is None:
            path = DATA / "rocket_params.json"
            if not path.exists():
                path = DATA / "rocket_params.placeholder.json"
        path = Path(path)
        return cls(json.loads(path.read_text()), eng_path, source=path)

    @property
    def is_placeholder(self):
        return self.source is not None and "placeholder" in Path(self.source).name

    # ------------------------------------------------------------------ fast scalar core
    def _thrust_mass(self, t):
        """(T, m, x_cg, Ixx, Iyy, mdot) at time t since ignition (scalar)."""
        tc = self._tc
        if t <= 0.0:
            return 0.0, self.m_wet, self.cg_wet, self.Ixx_wet, self.Iyy_wet, 0.0
        if t >= tc[-1]:
            return 0.0, self.m_dry, self.cg_dry, self.Ixx_dry, self.Iyy_dry, 0.0
        i = bisect_right(tc, t) - 1
        F0 = self._Fc[i]
        T = F0 + (self._Fc[i + 1] - F0) * (t - tc[i]) / (tc[i + 1] - tc[i])
        f = (self._Ic[i] + 0.5 * (F0 + T) * (t - tc[i])) / self.total_impulse
        return (T, self.m_wet + f * (self.m_dry - self.m_wet), self.cg_wet + f * (self.cg_dry - self.cg_wet),
                self.Ixx_wet + f * (self.Ixx_dry - self.Ixx_wet), self.Iyy_wet + f * (self.Iyy_dry - self.Iyy_wet),
                T / self.total_impulse * self.m_prop)

    def _aero(self, mach, powered):
        """(Cd, CN_alpha [1/rad], x_cp [m from nose]) at Mach (scalar, clamped to the table)."""
        ms = self._mach
        if mach <= ms[0]:
            i, a = 0, 0.0
        elif mach >= ms[-1]:
            i, a = len(ms) - 2, 1.0
        else:
            i = bisect_right(ms, mach) - 1
            a = (mach - ms[i]) / (ms[i + 1] - ms[i])
        cd = self._cd_on if powered else self._cd_off
        return (cd[i] + a * (cd[i + 1] - cd[i]), self._cna[i] + a * (self._cna[i + 1] - self._cna[i]),
                self._cp[i] + a * (self._cp[i + 1] - self._cp[i]))

    # ------------------------------------------------------------------ public (scalar or array)
    @_vec
    def thrust(self, t):
        return self._thrust_mass(t)[0]

    @_vec
    def cumulative_impulse(self, t):
        if t <= 0:
            return 0.0
        if t >= self.burn_time:
            return self.total_impulse
        i = bisect_right(self._tc, t) - 1
        return self._Ic[i] + 0.5 * (self._Fc[i] + self.thrust(t)) * (t - self._tc[i])

    def burned_fraction(self, t):
        return self.cumulative_impulse(t) / self.total_impulse

    @_vec
    def mass(self, t):
        return self._thrust_mass(t)[1]

    @_vec
    def cg(self, t):
        return self._thrust_mass(t)[2]

    @_vec
    def Ixx(self, t):
        return self._thrust_mass(t)[3]

    @_vec
    def Iyy(self, t):
        return self._thrust_mass(t)[4]

    @_vec
    def mdot(self, t):
        """Propellant mass flow (positive, kg/s)."""
        return self._thrust_mass(t)[5]

    def mass_props(self, t):
        """(m, x_cg, Ixx, Iyy) at scalar t."""
        return self._thrust_mass(float(t))[1:5]

    @_vec
    def cd(self, mach, power_on=False):
        return self._aero(mach, power_on)[0]

    @_vec
    def cn_alpha(self, mach):
        return self._aero(mach, False)[1]

    @_vec
    def cp(self, mach):
        return self._aero(mach, False)[2]

    def stability_cal(self, t, mach):
        """Static margin (x_cp - x_cg)/d in calibres."""
        return (self.cp(mach) - self.cg(t)) / self.d
