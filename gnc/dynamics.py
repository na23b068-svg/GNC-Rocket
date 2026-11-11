"""Nonlinear 6-DOF variable-mass rigid-body dynamics (ARCHITECTURE.md conventions).

x = [p_enu(3), v_enu(3), q(4) scalar-first body->ENU, omega_b(3)],  u = [delta_p, delta_y] (gimbal, rad).

Forces/moments (body frame, about the instantaneous CG):
  thrust   F = T d_b,  d_b = [cos dy cos dp, sin dy cos dp, -sin dp] (dp, dy include thrust misalignment),
           applied at the gimbal pivot: M = r_g x F, r_g = [-(x_gimbal - x_cg), 0, 0]
  axial    F_x = -qbar S Cd(M) u_r/V           (u_r/V = cos(alpha_total), sign-correct for any attitude)
  normal   F_yz = -qbar S CNa(M) [v_r, w_r]/V   (|F_N| = qbar S CNa sin(alpha_total)), applied at the CP
  damping  M_y,z = qbar S d cmq (q,r) d/(2V),  L = qbar S d clp p d/(2V)
  jet damping  M_y,z = -mdot (x_gimbal - x_cg)^2 (q, r)
  gravity  -m g z_enu
`derivatives` is pure (no hidden state) and cheap, so it can be finite-differenced for linearisation.
"""
import math
from dataclasses import dataclass

import numpy as np

from .atmosphere import G0, WindModel, isa
from .frames import quat_to_dcm


@dataclass
class Env:
    """Environment seen by `derivatives`.  wind: callable (t, z_agl) -> (E, N, U) m/s, or None for calm air."""
    elevation_m: float = 850.0
    wind: object = None
    thrust_misalignment: tuple = (0.0, 0.0)
    g: float = G0


def make_env(rocket, cfg=None):
    """Env from a Rocket and an optional SimConfig (wind/gust/shear/misalignment)."""
    elev = float(rocket.launch.get("elevation_m", 0.0))
    if cfg is None:
        return Env(elevation_m=elev)
    wind = WindModel(cfg.wind, cfg.gust, cfg.wind_shear_exponent, cfg.wind_ref_alt_m,
                     t_max=cfg.t_max, seed=np.random.SeedSequence([cfg.seed, 1]).generate_state(1)[0])
    return Env(elevation_m=elev, wind=None if wind.calm else wind,
               thrust_misalignment=tuple(float(v) for v in cfg.thrust_misalignment_rad))


def _core(t, x, u, rocket, env):
    """Returns (xdot list[13], aux tuple). aux = (T, m, x_cg, mach, alpha_total, qbar, x_cp, a_enu(3))."""
    px, py, pz, vx, vy, vz, qw, qx, qy, qz, P, Q, Rr = x.tolist() if hasattr(x, "tolist") else x
    T, m, xcg, Ixx, Iyy, mdot = rocket._thrust_mass(t)
    _, _, rho, a_snd = isa(env.elevation_m + pz)
    if env.wind is not None:
        wx, wy, wz = env.wind(t, pz)
        vrx, vry, vrz = vx - wx, vy - wy, vz - wz
    else:
        vrx, vry, vrz = vx, vy, vz

    r00 = 1 - 2 * (qy * qy + qz * qz); r01 = 2 * (qx * qy - qw * qz); r02 = 2 * (qx * qz + qw * qy)
    r10 = 2 * (qx * qy + qw * qz); r11 = 1 - 2 * (qx * qx + qz * qz); r12 = 2 * (qy * qz - qw * qx)
    r20 = 2 * (qx * qz - qw * qy); r21 = 2 * (qy * qz + qw * qx); r22 = 1 - 2 * (qx * qx + qy * qy)
    ub = r00 * vrx + r10 * vry + r20 * vrz          # air-relative velocity in body = R^T v_rel
    vb = r01 * vrx + r11 * vry + r21 * vrz
    wb = r02 * vrx + r12 * vry + r22 * vrz
    V = math.sqrt(ub * ub + vb * vb + wb * wb)
    mach = V / a_snd
    qbar = 0.5 * rho * V * V
    cd, cna, xcp = rocket._aero(mach, T > 0.0)
    S, d = rocket.S, rocket.d

    Fx = Fy = Fz = L = My = Mz = 0.0
    if V > 1e-9:
        Fx = -qbar * S * cd * ub / V
        k = -qbar * S * cna / V
        Fy, Fz = k * vb, k * wb
        lcp = xcp - xcg                                 # r_cp = [-lcp, 0, 0]  ->  M = [0, lcp Fz, -lcp Fy]
        My, Mz = lcp * Fz, -lcp * Fy
        kd = 0.25 * rho * V * S * d * d                 # qbar S d (d / 2V)
        L = kd * rocket.clp * P
        My += kd * rocket.cmq * Q
        Mz += kd * rocket.cmq * Rr
    if T > 0.0:
        dp = u[0] + env.thrust_misalignment[0]
        dy = u[1] + env.thrust_misalignment[1]
        cdp = math.cos(dp)
        FTy, FTz = T * math.sin(dy) * cdp, -T * math.sin(dp)
        Fx += T * math.cos(dy) * cdp
        Fy += FTy
        Fz += FTz
        lg = rocket.x_gimbal - xcg
        jd = mdot * lg * lg
        My += lg * FTz - jd * Q
        Mz += -lg * FTy - jd * Rr

    ax = (r00 * Fx + r01 * Fy + r02 * Fz) / m
    ay = (r10 * Fx + r11 * Fy + r12 * Fz) / m
    az = (r20 * Fx + r21 * Fy + r22 * Fz) / m - env.g
    # Euler's equations with Iyy = Izz
    dP = L / Ixx
    dQ = (My - (Ixx - Iyy) * Rr * P) / Iyy
    dR = (Mz - (Iyy - Ixx) * P * Q) / Iyy
    xdot = [vx, vy, vz, ax, ay, az,
            0.5 * (-qx * P - qy * Q - qz * Rr), 0.5 * (qw * P + qy * Rr - qz * Q),
            0.5 * (qw * Q + qz * P - qx * Rr), 0.5 * (qw * Rr + qx * Q - qy * P),
            dP, dQ, dR]
    alpha_t = math.atan2(math.sqrt(vb * vb + wb * wb), ub) if V > 1e-9 else 0.0
    return xdot, (T, m, xcg, mach, alpha_t, qbar, xcp, ax, ay, az)


def derivatives(t, x, u, rocket, env):
    """Free-flight state derivative xdot (np.ndarray, 13)."""
    return np.array(_core(t, x, u, rocket, env)[0])


def rail_derivatives(t, x, u, rocket, env, rail_dir):
    """Launch-rail phase: translation constrained to the unit vector rail_dir, no rotation, no friction.
    The rocket cannot slide back down the rail before liftoff."""
    xd, aux = _core(t, x, u, rocket, env)
    rx, ry, rz = rail_dir
    a = xd[3] * rx + xd[4] * ry + xd[5] * rz
    s = x[3] * rx + x[4] * ry + x[5] * rz
    if s <= 0.0 and a < 0.0:
        a = 0.0
    out = np.zeros(13)
    out[0:3] = (s * rx, s * ry, s * rz)
    out[3:6] = (a * rx, a * ry, a * rz)
    return out, aux


def aero_state(t, x, rocket, env):
    """Aerodynamic quantities for a state: dict with v_rel_b (air-relative velocity, body), airspeed, mach,
    q_dyn, rho, alpha (pitch plane, atan2(w, u)), beta (yaw plane, atan2(v, u)), alpha_total,
    cd, cn_alpha, x_cp, x_cg, stability_cal."""
    x = np.asarray(x, float)
    T, m, xcg, *_ = rocket._thrust_mass(t)
    _, _, rho, a_snd = isa(env.elevation_m + x[2])
    w = np.array(env.wind(t, x[2]) if env.wind is not None else (0.0, 0.0, 0.0))
    vb = quat_to_dcm(x[6:10]).T @ (x[3:6] - w)
    V = float(np.linalg.norm(vb))
    mach = V / a_snd
    cd, cna, xcp = rocket._aero(mach, T > 0)
    return {"v_rel_b": vb, "airspeed": V, "mach": mach, "q_dyn": 0.5 * rho * V * V, "rho": rho,
            "alpha": math.atan2(vb[2], vb[0]), "beta": math.atan2(vb[1], vb[0]),
            "alpha_total": math.atan2(math.hypot(vb[1], vb[2]), vb[0]),
            "cd": cd, "cn_alpha": cna, "x_cp": xcp, "x_cg": xcg, "stability_cal": (xcp - xcg) / rocket.d}


def specific_force_body(x, a_enu, g=G0):
    """Accelerometer truth: f_b = R^T (a - g_vec), g_vec = [0, 0, -g]."""
    return quat_to_dcm(np.asarray(x)[6:10]).T @ (np.asarray(a_enu, float) + np.array([0.0, 0.0, g]))
