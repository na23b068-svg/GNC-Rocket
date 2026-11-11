"""Fixed-step RK4 flight simulator: launch rail -> 6-DOF free flight -> (optional) parachute descent.

    res = simulate(rocket, SimConfig(...), controller=None)

Per step k (time t): k1 = f(t, x, u_act) is evaluated and logged; sensors due at t are sampled (using the
k1 acceleration); if the controller is due it receives `meas` and its command enters the actuator chain
(delay -> limit -> first-order lag -> rate limit); a command issued at t reaches the actuator input at
t + actuator_delay_s + dt (the extra dt is the zero-order-hold step);
then x is advanced with RK4 holding u_act constant over the step (zero-order hold).
"""
import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from .atmosphere import G0, isa
from .dynamics import _core, make_env, rail_derivatives
from .frames import launch_attitude, nose_direction
from .sensors import Sensors

SENSOR_KEYS = ("imu_accel", "imu_gyro", "baro_alt", "gps_pos", "gps_vel", "mag")


@dataclass
class SimConfig:
    # integration / stopping
    dt: float = 0.001                      # s, RK4 step during powered/coast flight
    t_max: float = 400.0                   # s
    stop_at: str = "apogee"                # "apogee" | "ground"
    # environment
    wind: tuple = (0.0, 0.0, 0.0)          # steady ENU wind velocity at wind_ref_alt_m [m/s]
    gust: dict = None                      # {"std_mps", "tau_s", "one_cos": [{"t0","duration","amplitude"}]}
    wind_shear_exponent: float = 0.0       # power-law shear exponent (1/7 typical), 0 = uniform
    wind_ref_alt_m: float = 10.0
    rail_angle_deg: float = None           # None -> rocket_params launch block
    rail_azimuth_deg: float = None
    rail_length_m: float = None
    # vehicle / actuator
    thrust_misalignment_rad: tuple = (0.0, 0.0)   # constant bias added to [delta_p, delta_y]
    gimbal_limit_rad: float = math.radians(5.0)
    actuator_tau_s: float = 0.02           # first-order servo lag (0 = instantaneous)
    actuator_rate_limit_rad_s: float = math.radians(150.0)
    actuator_delay_s: float = 0.0          # pure transport delay of commands
    estimation_delay_s: float = 0.0        # sensor samples reach the controller this much later
    # sensors
    sensor_noise_scale: float = 1.0        # multiplies every std-dev (0 -> noiseless; biases still applied)
    imu_rate_hz: float = 200.0
    baro_rate_hz: float = 50.0
    mag_rate_hz: float = 50.0
    gps_rate_hz: float = 10.0
    accel_std: float = 0.08                # m/s^2 per axis (scalar or 3-vector)
    gyro_std: float = 0.002                # rad/s per axis
    baro_std_pa: float = 6.0               # Pa
    gps_pos_std: tuple = (1.5, 1.5, 3.0)   # m E/N/U
    gps_vel_std: float = 0.15              # m/s per axis
    mag_std: float = 0.005                 # per axis on the unit field vector
    accel_bias: tuple = (0.0, 0.0, 0.0)
    gyro_bias: tuple = (0.0, 0.0, 0.0)
    baro_bias_pa: float = 0.0
    gps_pos_bias: tuple = (0.0, 0.0, 0.0)
    mag_bias: tuple = (0.0, 0.0, 0.0)
    mag_declination_deg: float = 9.5
    mag_inclination_deg: float = 58.0
    launch_detect_accel_g: float = 2.0     # meas["t_launch_detect"]: first accel_x sample above this
    perfect_nav: bool = False              # controllers may use meas["truth"] for feedback only if True
    seed: int = 0
    # recovery (point-mass descent after apogee, attitude frozen)
    recovery: bool = False
    drogue_cda_m2: float = 0.55            # ~25 m/s descent
    main_cda_m2: float = 10.0              # ~5.5 m/s descent
    main_deploy_agl_m: float = 300.0
    descent_dt: float = 0.05


def _rk4(f, t, x, h, k1):
    k2 = f(t + 0.5 * h, x + 0.5 * h * k1)
    k3 = f(t + 0.5 * h, x + 0.5 * h * k2)
    k4 = f(t + h, x + h * k3)
    return x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def simulate(rocket, cfg=None, controller=None):
    cfg = cfg or SimConfig()
    env = make_env(rocket, cfg)
    L = rocket.launch
    th = math.radians(L["rail_angle_deg"] if cfg.rail_angle_deg is None else cfg.rail_angle_deg)
    az = math.radians(L["rail_azimuth_deg"] if cfg.rail_azimuth_deg is None else cfg.rail_azimuth_deg)
    rail_len = L["rail_length_m"] if cfg.rail_length_m is None else cfg.rail_length_m
    q0 = launch_attitude(th, az)
    rail_dir = tuple(nose_direction(q0))
    x = np.zeros(13)
    x[6:10] = q0
    dt, eps = cfg.dt, 1e-9
    sensors = Sensors(cfg, env.elevation_m, cfg.seed)
    lim = cfg.gimbal_limit_rad

    free = lambda tt, xx: np.array(_core(tt, xx, u_act, rocket, env)[0])  # noqa: E731
    rail = lambda tt, xx: rail_derivatives(tt, xx, u_act, rocket, env, rail_dir)[0]  # noqa: E731

    if controller is not None:
        controller.reset(rocket, cfg)
        t_next_ctrl = 0.0
    buffers = {k: deque() for k in SENSOR_KEYS}
    meas_log = {k: ([], []) for k in SENSOR_KEYS}
    u_cmd = np.zeros(2)
    u_act = np.zeros(2)
    cmd_queue = deque([(-math.inf, np.zeros(2))])
    t_ld = None
    events = {}
    T_, X_, UC_, UA_, AUX_ = [], [], [], [], []

    t, phase = 0.0, "rail"
    while True:
        if phase == "rail":
            k1, aux = rail_derivatives(t, x, u_act, rocket, env, rail_dir)
            if "liftoff" not in events and k1[3] * rail_dir[0] + k1[4] * rail_dir[1] + k1[5] * rail_dir[2] > 0:
                events["liftoff"] = t
        else:
            xd, aux = _core(t, x, u_act, rocket, env)
            k1 = np.array(xd)
        T_.append(t); X_.append(x); UA_.append(u_act); AUX_.append(aux)

        new = sensors.step(t, x, aux[7:10])
        for k, v in new.items():
            meas_log[k][0].append(t)
            meas_log[k][1].append(v)
            if controller is not None:
                buffers[k].append((t, v))

        if controller is not None and t >= t_next_ctrl - eps:
            meas = {}
            for k in SENSOR_KEYS:
                buf, val, ts = buffers[k], None, None
                while buf and buf[0][0] <= t - cfg.estimation_delay_s + eps:
                    ts, val = buf.popleft()
                    if k == "imu_accel" and t_ld is None and val[0] > cfg.launch_detect_accel_g * G0:
                        t_ld = ts
                meas[k] = val
            meas["t_launch_detect"] = t_ld
            meas["truth"] = x.copy()
            u_cmd = np.asarray(controller.update(t, meas), float).reshape(2)
            cmd_queue.append((t, u_cmd))
            t_next_ctrl += controller.dt
        UC_.append(u_cmd)  # logged after the update so u_cmd[k] is the command issued at t[k]

        x_old = x
        x = _rk4(rail if phase == "rail" else free, t, x, dt, k1)
        x[6:10] /= math.sqrt(x[6] ** 2 + x[7] ** 2 + x[8] ** 2 + x[9] ** 2)
        t_new = t + dt

        if controller is not None:  # actuator: delay -> clip -> lag -> rate limit -> clip (for next step)
            # a command issued at t_c reaches the actuator input at t_c + actuator_delay_s + dt
            while len(cmd_queue) > 1 and cmd_queue[1][0] + cfg.actuator_delay_s + dt <= t_new + eps:
                cmd_queue.popleft()
            target = np.clip(cmd_queue[0][1], -lim, lim)
            du = target - u_act
            if cfg.actuator_tau_s > 0:
                du = du * (1.0 - math.exp(-dt / cfg.actuator_tau_s))
            rl = cfg.actuator_rate_limit_rad_s * dt
            u_act = np.clip(u_act + np.clip(du, -rl, rl), -lim, lim)

        # ---- events
        stop = False
        if phase == "rail":
            if (x[0] * rail_dir[0] + x[1] * rail_dir[1] + x[2] * rail_dir[2]) >= rail_len:
                events["rail_exit"] = t_new
                phase = "flight"
        if t < rocket.burn_time <= t_new:
            events["burnout"] = rocket.burn_time
        if phase == "flight" and x_old[5] > 0.0 >= x[5]:
            events["apogee"] = float(t + dt * x_old[5] / (x_old[5] - x[5]))
            if cfg.stop_at == "apogee":
                stop = True
            elif cfg.recovery:
                phase = "descent"
        if phase == "flight" and x[2] < 0.0 and "rail_exit" in events:
            f = x_old[2] / (x_old[2] - x[2])
            x, t_new = x_old + f * (x - x_old), float(t + f * dt)
            events["landing"] = t_new
            stop = True
        t = t_new
        if t >= cfg.t_max - eps:
            stop = True
        if stop or phase == "descent":
            break

    if phase == "descent":
        events["drogue"] = events["apogee"]
        t, x = _descent(rocket, env, cfg, t, x, events, T_, X_, UC_, UA_, AUX_)
    else:  # log the final state
        if phase == "rail":
            aux = rail_derivatives(t, x, u_act, rocket, env, rail_dir)[1]
        else:
            aux = _core(t, x, u_act, rocket, env)[1]
        T_.append(t); X_.append(x); UC_.append(u_cmd); UA_.append(u_act); AUX_.append(aux)

    aux = np.array(AUX_)
    X = np.array(X_)
    res = {
        "t": np.array(T_), "x": X, "u_cmd": np.array(UC_), "u_act": np.array(UA_),
        "thrust": aux[:, 0], "mass": aux[:, 1], "cg": aux[:, 2], "mach": aux[:, 3], "alpha": aux[:, 4],
        "q_dyn": aux[:, 5], "cp": aux[:, 6], "accel": aux[:, 7:10],
        "stability_cal": (aux[:, 6] - aux[:, 2]) / rocket.d,
        "events": events,
        "meas_log": {k: {"t": np.array(v[0]), "y": np.array(v[1])} for k, v in meas_log.items()},
        "controller_log": getattr(controller, "log", {}) if controller is not None else {},
        "cfg": cfg,
        "rail_dir": np.array(rail_dir),
    }
    i_exit = np.searchsorted(res["t"], events.get("rail_exit", np.inf))
    speed = np.linalg.norm(X[:, 3:6], axis=1)
    res["summary"] = {
        "apogee_agl_m": float(X[:, 2].max()),
        "t_apogee_s": events.get("apogee"),
        "max_speed_mps": float(speed.max()),
        "max_mach": float(np.nanmax(res["mach"])),
        "max_accel_mps2": float(np.linalg.norm(res["accel"][res["t"] <= events.get("apogee", np.inf)],
                                               axis=1).max()),
        "off_rail_velocity_mps": float(speed[i_exit]) if i_exit < len(speed) else None,
        "stability_off_rail_cal": float(res["stability_cal"][i_exit]) if i_exit < len(speed) else None,
        "burnout_s": events.get("burnout"),
        "landing_distance_m": float(np.hypot(*X[-1, 0:2])) if "landing" in events else None,
    }
    return res


def _descent(rocket, env, cfg, t, x, events, T_, X_, UC_, UA_, AUX_):
    """Point-mass parachute descent (drag area only, drifting with the wind); attitude frozen, omega = 0."""
    m = rocket.mass_props(t)[0]
    xcg = rocket.mass_props(t)[1]
    cda = cfg.drogue_cda_m2
    h = cfg.descent_dt
    q_frozen = x[6:10].copy()

    def f(tt, s):
        _, _, rho, _ = isa(env.elevation_m + s[2])
        w = env.wind(tt, s[2]) if env.wind is not None else (0.0, 0.0, 0.0)
        vr = s[3:6] - w
        a = -0.5 * rho * np.linalg.norm(vr) * cda / m * vr
        a[2] -= env.g
        return np.concatenate([s[3:6], a])

    def log(tt, s, a):
        _, _, rho, a_snd = isa(env.elevation_m + s[2])
        w = np.array(env.wind(tt, s[2]) if env.wind is not None else (0.0, 0.0, 0.0))
        V = float(np.linalg.norm(s[3:6] - w))
        xs = np.concatenate([s, q_frozen, np.zeros(3)])
        T_.append(tt); X_.append(xs); UC_.append(np.zeros(2)); UA_.append(np.zeros(2))
        AUX_.append((0.0, m, xcg, V / a_snd, math.nan, 0.5 * rho * V * V, math.nan, a[0], a[1], a[2]))

    s = x[0:6].copy()
    while True:
        k1 = f(t, s)
        log(t, s, k1[3:6])
        s_old, s = s, _rk4(f, t, s, h, k1)
        if "main" not in events and s[2] <= cfg.main_deploy_agl_m:
            events["main"] = t + h
            cda = cfg.main_cda_m2
        if s[2] <= 0.0:
            fr = s_old[2] / (s_old[2] - s[2])
            s, t = s_old + fr * (s - s_old), float(t + fr * h)
            events["landing"] = t
            log(t, s, f(t, s)[3:6])
            break
        t += h
        if t >= cfg.t_max:
            log(t, s, f(t, s)[3:6])
            break
    return t, np.concatenate([s, q_frozen, np.zeros(3)])
