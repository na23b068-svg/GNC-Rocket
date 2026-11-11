"""PID control: analytic pole-placement tuning, gain scheduling, discrete PID with anti-windup.

Attitude plant per plane (pitch or yaw), from the linearised short-period model (see gnc.linearize):

    theta'' + a1 * theta' + a0 * theta = b * delta

a1 = aero/jet damping (-Mq), a0 = aero static stiffness (-M_alpha, >0 for a fin-stabilised rocket),
b = TVC control effectiveness dq/d(delta) (negative with the ARCHITECTURE.md gimbal convention).

Control law (derivative on the measured rate, which a rate gyro provides directly):

    delta = Kp*e + Ki*int(e) - Kd*rate_f,   e = theta_cmd - theta
    rate_f = rate / (tau_f*s + 1)  (optional derivative filter, tau_f = 0 -> none)

Closed-loop characteristic polynomials (derived in docs/gnc_design.md):
  no filter : s^3 + (a1 + b Kd) s^2 + (a0 + b Kp) s + b Ki
  filter    : tau s^4 + (1 + tau a1) s^3 + (a1 + tau a0 + b tau Kp + b Kd) s^2 + (a0 + b Kp + b tau Ki) s + b Ki
"""
from dataclasses import dataclass

import numpy as np


def poles_from_specs(zeta, wn, p_int=None, p_filt=None):
    """Desired closed-loop poles: dominant pair (zeta, wn), integral pole -p_int (default 0.5*wn) and an
    optional fast derivative-filter pole -p_filt."""
    wd = wn * np.sqrt(max(1.0 - zeta**2, 0.0))
    poles = [complex(-zeta * wn, wd), complex(-zeta * wn, -wd), -(0.5 * wn if p_int is None else p_int)]
    if p_filt is not None:
        poles.append(-p_filt)
    return poles


def specs_of(zeta, wn):
    """Transient specs implied by a dominant 2nd-order pair."""
    os = 100 * np.exp(-np.pi * zeta / np.sqrt(1 - zeta**2)) if zeta < 1 else 0.0
    return {"zeta": zeta, "wn": wn, "overshoot_pct": os, "settling_2pct_s": 4.0 / (zeta * wn),
            "rise_time_s": 1.8 / wn}


def place_pid(a1, a0, b, poles):
    """Analytic PID gains placing the closed-loop poles of (plant + PID) exactly at `poles`.

    3 poles -> ideal PID (tau_f = 0); 4 poles -> PID with derivative filter (tau_f solved too).
    Returns dict(Kp, Ki, Kd, tau_f)."""
    c = np.real(np.poly(poles))  # monic, c[0] = 1
    if len(poles) == 3:
        _, c2, c1, c0 = c
        return {"Kp": (c1 - a0) / b, "Ki": c0 / b, "Kd": (c2 - a1) / b, "tau_f": 0.0}
    if len(poles) == 4:
        _, c3, c2, c1, c0 = c
        tau = 1.0 / (c3 - a1)
        if tau <= 0:
            raise ValueError("filter pole too slow: need sum(-poles) > a1")
        Ki = c0 * tau / b
        Kp = (c1 * tau - a0 - b * tau * Ki) / b
        Kd = (c2 * tau - a1 - tau * a0 - b * tau * Kp) / b
        return {"Kp": Kp, "Ki": Ki, "Kd": Kd, "tau_f": tau}
    raise ValueError("need 3 or 4 poles")


def closed_loop_poles(a1, a0, b, Kp, Ki, Kd, tau_f=0.0, **_):
    """Roots of the closed-loop characteristic polynomial (inverse of place_pid, for verification)."""
    if tau_f == 0:
        return np.roots([1.0, a1 + b * Kd, a0 + b * Kp, b * Ki])
    return np.roots([tau_f, 1 + tau_f * a1, a1 + tau_f * a0 + b * tau_f * Kp + b * Kd,
                     a0 + b * Kp + b * tau_f * Ki, b * Ki])


@dataclass
class GainSchedule:
    """Gains tabulated against a scheduling variable (time since launch here), linearly interpolated
    and clamped at the ends."""
    s: np.ndarray       # scheduling variable, increasing
    Kp: np.ndarray
    Ki: np.ndarray
    Kd: np.ndarray

    def __call__(self, s):
        return tuple(float(np.interp(s, self.s, k)) for k in (self.Kp, self.Ki, self.Kd))

    @classmethod
    def design(cls, s, a1, a0, b, poles):
        """poles: one pole list used at every operating point, or one list per point (a1, a0, b arrays)."""
        per_point = np.ndim(poles[0]) > 0
        g = [place_pid(*abc, poles[i] if per_point else poles) for i, abc in enumerate(zip(a1, a0, b))]
        return cls(np.asarray(s, float), *(np.array([gi[k] for gi in g]) for k in ("Kp", "Ki", "Kd")))


class PID:
    """Discrete PID, derivative on measured rate, first-order derivative filter, output saturation and
    conditional-integration (clamping) anti-windup: the integrator freezes while the output is saturated
    and the error would drive it deeper into saturation."""

    def __init__(self, Kp, Ki, Kd, dt, tau_f=0.0, u_max=np.inf):
        self.Kp, self.Ki, self.Kd, self.dt, self.tau_f, self.u_max = Kp, Ki, Kd, dt, tau_f, u_max
        self.reset()

    def reset(self):
        self.i = 0.0
        self.rate_f = None

    def set_gains(self, Kp, Ki, Kd):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd

    def update(self, e, rate):
        """e = command - measurement, rate = measured derivative of the controlled variable."""
        if self.rate_f is None or self.tau_f == 0:
            self.rate_f = rate
        else:
            a = self.dt / (self.tau_f + self.dt)
            self.rate_f += a * (rate - self.rate_f)
        u_unsat = self.Kp * e + self.Ki * self.i - self.Kd * self.rate_f
        u = float(np.clip(u_unsat, -self.u_max, self.u_max))
        if u == u_unsat or np.sign(e * self.Ki) != np.sign(u_unsat):
            self.i += e * self.dt
        return u
