"""Flight computer: navigation -> guidance -> attitude control, implementing the simulator Controller protocol
(ARCHITECTURE.md): attributes `dt`, `log`; methods `reset(rocket, cfg)`, `update(t, meas) -> [delta_p, delta_y]`.

Phases (logged as `phase`): 0 = on rail (gimbal 0, gyro bias averaged since the rocket cannot rotate),
1 = powered & guided (rail exit -> thrust tail-off, see tvc_end_time), 2 = coast (no TVC authority, gimbal 0).

Navigation
  attitude : strapdown quaternion from the known rail attitude, gyro minus rail-averaged bias, plus a
             magnetometer correction w_c = k_mag (m_meas x m_pred) (Mahony; a Luenberger observer on SO(3),
             k_mag = its pole in rad/s).
  position : linear 9-state [p, v, b_a] filter in ENU, input = R(q) f_imu + g, GPS (pos+vel) and baro
             updates; nav="kf" (Kalman, Joseph form) or nav="luenberger" (fixed gains by pole placement).
Guidance   : "none" (attitude hold) or "los_path" (LOS waypoint following) -> lateral accel a_cmd (ENU).
             a_cmd + gravity compensation is turned into a commanded nose direction tilted from the velocity
             vector by alpha_c = |a| / (f_x + qbar S CNa / m) (thrust + airframe lift per radian of AoA).
Control    : body-frame attitude error (pitch about y_b, yaw about z_b), one PID per plane, gains scheduled
             vs time since launch from the numeric linearisation (gnc.linearize) by pole placement (gnc.pid).
"""
import math

import numpy as np

from .atmosphere import isa
from .frames import launch_attitude, nose_direction, quat_from_axis_angle, quat_mul, quat_normalize, quat_to_dcm
from .guidance import LOSPathGuidance
from .observers import KalmanFilter, Luenberger, nav_luenberger_gains, nav_model
from .pid import PID, GainSchedule, poles_from_specs

G = 9.80665
_SCHEDULE_CACHE = {}


def tvc_end_time(rocket, frac=0.5):
    """End of useful TVC authority: last time thrust >= frac * mean thrust (the tail-off is excluded,
    control effectiveness b ~ T collapses there and scheduled gains would blow up)."""
    tt = np.linspace(0.0, rocket.burn_time, 2001)
    T = rocket.thrust(tt)
    return float(tt[np.nonzero(T >= frac * rocket.total_impulse / rocket.burn_time)[0][-1]])


def design_schedule(rocket, zeta=0.7, wn=8.0, p_int=4.0, kappa=0.8, n_points=9):
    """Operating points along the nominal boost + pole-placement PID gains at each. Returns (GainSchedule,
    table) where table holds t, qbar, a1, a0, b, poles, gains per point (pitch plane; yaw identical by symmetry).
    Desired poles per point: dominant pair (zeta, wn_i), wn_i = max(wn, kappa*sqrt(a0_i)), integral pole -p_int.
    The floor kappa*sqrt(a0) (airframe weathercock frequency) keeps the PID from cancelling the aerodynamic
    stiffness at high dynamic pressure, which would leave only ~2 dB gain margin (docs/gnc_design.md)."""
    from .linearize import nominal_trajectory, operating_points
    traj = nominal_trajectory(rocket)
    t_rail = traj["events"].get("rail_exit", 0.3)
    times = np.linspace(t_rail, tvc_end_time(rocket), n_points)
    pts = operating_points(rocket, times, traj)
    a1, a0, b = (np.array(v) for v in zip(*[p["pitch"]["short_period"] for p in pts]))
    wns = np.maximum(wn, kappa * np.sqrt(np.maximum(a0, 0.0)))
    poles = [poles_from_specs(zeta, w, p_int) for w in wns]
    sched = GainSchedule.design(times, a1, a0, b, poles)
    table = {"t": times.tolist(), "qbar": [p["flight"]["qbar"] for p in pts],
             "V": [p["flight"]["V"] for p in pts], "a1": a1.tolist(), "a0": a0.tolist(), "b": b.tolist(),
             "Kp": sched.Kp.tolist(), "Ki": sched.Ki.tolist(), "Kd": sched.Kd.tolist(),
             "wn": wns.tolist(), "zeta": zeta, "p_int": p_int,
             "poles": [[[c.real, c.imag] for c in pl] for pl in poles]}
    # authority envelope (5 deg gimbal): AoA the TVC can hold against the aero stiffness, and the lateral
    # acceleration that AoA produces (thrust tilt + airframe normal force)
    d5 = np.radians(5.0)
    table["alpha_hold_deg"] = [float(np.degrees(min(abs(bi) * d5 / max(a0i, 1e-9), 0.5))) for a0i, bi in zip(a0, b)]
    table["a_lat_max"] = [float(np.radians(ah) * (p["flight"]["T"] + p["flight"]["qbar"] * rocket.S * p["flight"]["cna"])
                                / p["flight"]["m"]) for ah, p in zip(table["alpha_hold_deg"], pts)]
    return sched, table


class FlightComputer:
    def __init__(self, guidance="none", nav="kf", dt=0.01, waypoints=None, los_k=1.5, lookahead_m=150.0,
                 lookahead_time=0.0,
                 accept_radius=40.0, gravity_comp=True, a_max=60.0, alpha_max_deg=10.0, hold="rail",
                 zeta=0.7, wn=8.0, p_int=4.0, kappa=0.8, schedule=None, k_mag=0.2,
                 nav_sigma_a=0.3, nav_sigma_b=0.02, gps_poles=(0.8, 0.85, 0.97), baro_poles=(0.99, 0.992, 0.999)):
        """guidance: "none" | "los_path";  nav: "kf" | "luenberger";  waypoints: (n,3) ENU for "los_path".
        zeta, wn, p_int, kappa: desired attitude-loop poles (see design_schedule); schedule: optional
        GainSchedule overriding the designed gains (designs are cached per rocket file).
        hold: "rail" (hold the launch-rail direction) or "vertical" (attitude-hold mode only).
        k_mag: magnetometer correction gain (rad/s); nav_sigma_a/b: KF process noise; gps/baro_poles:
        Luenberger error-map poles per sensor update; a_max, alpha_max_deg: guidance command limits."""
        assert guidance in ("none", "los_path") and nav in ("kf", "luenberger")
        self.dt = dt
        self.mode, self.nav_kind = guidance, nav
        self.waypoints = waypoints
        self.gravity_comp, self.a_max, self.alpha_max = gravity_comp, a_max, math.radians(alpha_max_deg)
        self.hold = hold
        self.zeta, self.wn, self.p_int, self.kappa, self.schedule = zeta, wn, p_int, kappa, schedule
        self.schedule_table = None
        self.k_mag = k_mag
        self.nav_sigma_a, self.nav_sigma_b = nav_sigma_a, nav_sigma_b
        self.gps_poles, self.baro_poles = gps_poles, baro_poles
        self.guid = (LOSPathGuidance(waypoints, lookahead_m, lookahead_time, accept_radius, los_k)
                     if guidance == "los_path" else None)
        self.log = {}

    # ------------------------------------------------------------------ protocol
    def reset(self, rocket, cfg):
        self.rocket, self.cfg = rocket, cfg
        L = rocket.launch
        th = math.radians(L["rail_angle_deg"] if cfg.rail_angle_deg is None else cfg.rail_angle_deg)
        az = math.radians(L["rail_azimuth_deg"] if cfg.rail_azimuth_deg is None else cfg.rail_azimuth_deg)
        self.rail_len = L["rail_length_m"] if cfg.rail_length_m is None else cfg.rail_length_m
        self.q_rail = launch_attitude(th, az)
        self.rail_cos = math.cos(th)
        self.elev = float(L.get("elevation_m", 0.0))
        key = (rocket.source, rocket.total_impulse, self.zeta, self.wn, self.p_int, self.kappa)
        if key not in _SCHEDULE_CACHE:
            _SCHEDULE_CACHE[key] = design_schedule(rocket, self.zeta, self.wn, self.p_int, self.kappa)
        sched, self.schedule_table = _SCHEDULE_CACHE[key]
        self.gains = self.schedule or sched     # a user-supplied GainSchedule overrides the designed gains
        self.t_tvc_end = tvc_end_time(rocket)
        lim = cfg.gimbal_limit_rad
        self.pid = [PID(0, 0, 0, self.dt, u_max=lim), PID(0, 0, 0, self.dt, u_max=lim)]
        self.lim = lim
        if self.guid is not None:
            self.guid.reset()

        # navigation filter (sensor stds from the config = sensor datasheet values)
        from .sensors import mag_field_enu
        self.m_ref = mag_field_enu(cfg.mag_declination_deg, cfg.mag_inclination_deg)
        A, B, Q, C = nav_model(self.dt, self.nav_sigma_a, self.nav_sigma_b)
        k = max(cfg.sensor_noise_scale, 1e-3)
        sp = np.broadcast_to(np.asarray(cfg.gps_pos_std, float), (3,)) * k
        sv = np.broadcast_to(np.asarray(cfg.gps_vel_std, float), (3,)) * k
        self.baro_std_pa = cfg.baro_std_pa * k
        R = {"gps": np.diag(np.concatenate([sp, sv]) ** 2), "baro": np.eye(1)}
        x0 = np.zeros(9)
        if self.nav_kind == "kf":
            P0 = np.diag(np.concatenate([sp**2 + 1.0, np.full(3, 0.1), np.full(3, 0.3**2)]))
            self.nav = KalmanFilter(A, B, C, Q, R, x0, P0)
        else:
            n_gps = max(1, round(1 / (cfg.gps_rate_hz * self.dt)))
            n_baro = max(1, round(1 / (cfg.baro_rate_hz * self.dt)))
            Lg = nav_luenberger_gains(self.dt, n_gps, n_baro, self.gps_poles, self.baro_poles)
            self.nav = Luenberger(A, B, C, Lg, x0)

        self.q_est = self.q_rail.copy()
        self.b_gyro, self.n_gyro = np.zeros(3), 0
        self.phase, self.t_ld = 0, None
        self.s_rail = self.v_rail = 0.0
        self.t_exit = None
        self.f_b, self.w_b = np.array([G * self.rail_cos, 0, 0]), np.zeros(3)
        self.n_hold = nose_direction(self.q_rail) if self.hold == "rail" else np.array([0, 0, 1.0])
        self.log = {k: [] for k in ("t", "phase", "pos_est", "vel_est", "quat_est", "accel_bias_est",
                                    "gyro_bias_est", "nav_std", "u_cmd", "att_err", "rate", "a_cmd",
                                    "alpha_cmd", "nose_cmd", "gains", "nis_gps")}

    def update(self, t, meas):
        dt = self.dt
        if meas.get("imu_accel") is not None:
            self.f_b = np.asarray(meas["imu_accel"], float)
        if meas.get("imu_gyro") is not None:
            self.w_b = np.asarray(meas["imu_gyro"], float)
        if self.t_ld is None and meas.get("t_launch_detect") is not None:
            self.t_ld = meas["t_launch_detect"]
        perfect = getattr(self.cfg, "perfect_nav", False) and meas.get("truth") is not None

        # ---- phase logic: rail distance from the axial accelerometer (rail is frictionless & straight)
        if self.phase == 0:
            self.b_gyro = (self.b_gyro * self.n_gyro + self.w_b) / (self.n_gyro + 1)
            self.n_gyro += 1
            if self.t_ld is not None:
                self.v_rail = max(0.0, self.v_rail + (self.f_b[0] - G * self.rail_cos) * dt)
                self.s_rail += self.v_rail * dt
                if self.s_rail >= self.rail_len:
                    self.phase, self.t_exit = 1, t
        tl = t - self.t_ld if self.t_ld is not None else 0.0
        if self.phase == 1 and (tl > self.t_tvc_end or self.f_b[0] < 2.0):
            self.phase = 2

        # ---- attitude: strapdown + magnetometer correction (locked to the rail attitude on the rail)
        w = self.w_b - self.b_gyro
        if self.phase == 0:
            self.q_est = self.q_rail.copy()
        else:
            if self.k_mag > 0 and meas.get("mag") is not None:
                m_pred = quat_to_dcm(self.q_est).T @ self.m_ref
                w = w + self.k_mag * np.cross(np.asarray(meas["mag"], float), m_pred)
            ang = np.linalg.norm(w) * dt
            if ang > 0:
                self.q_est = quat_normalize(quat_mul(self.q_est, quat_from_axis_angle(w, ang)))

        # ---- translational navigation
        Rq = quat_to_dcm(self.q_est)
        self.nav.predict(Rq @ self.f_b + np.array([0, 0, -G]))
        nis = None
        if meas.get("gps_pos") is not None and meas.get("gps_vel") is not None:
            self.nav.update("gps", np.concatenate([meas["gps_pos"], meas["gps_vel"]]))
            nis = getattr(self.nav, "nis", {}).get("gps")
        if meas.get("baro_alt") is not None:
            if self.nav_kind == "kf":   # baro altitude std from the pressure std at the current altitude
                rho = isa(self.elev + self.nav.x[2])[2]
                self.nav.R["baro"] = np.array([[(self.baro_std_pa / (rho * G)) ** 2 + 1e-4]])
            self.nav.update("baro", [meas["baro_alt"]])
        p, v = self.nav.x[0:3].copy(), self.nav.x[3:6].copy()
        q_ctrl, w_ctrl = self.q_est, self.w_b - self.b_gyro
        if perfect:
            x = meas["truth"]
            p, v, q_ctrl, w_ctrl = x[0:3], x[3:6], x[6:10], x[10:13]
            Rq = quat_to_dcm(q_ctrl)

        # ---- guidance (evaluated every step for logging; acts only in phase 1)
        a_cmd, info = np.zeros(3), {}
        if self.guid is not None:
            a_cmd, info = self.guid.command(p, v)
        a_cmd = a_cmd * min(1.0, self.a_max / max(np.linalg.norm(a_cmd), 1e-12))

        u = np.zeros(2)
        e_att = np.zeros(2)
        alpha_c = 0.0
        n_cmd = self.n_hold
        Kp, Ki, Kd = self.gains(tl)
        if self.phase == 1:
            V = np.linalg.norm(v)
            if self.guid is not None and V > 10.0:
                vh = v / V
                a_lat = a_cmd.copy()
                if self.gravity_comp:
                    g = np.array([0, 0, -G])
                    a_lat -= g - (g @ vh) * vh
                # lateral accel per radian of AoA: thrust (minus drag) + airframe normal force
                rho, a_snd = isa(self.elev + p[2])[2:4]
                cna = self.rocket._aero(V / a_snd, True)[1]
                k_lat = max(self.f_b[0], 1.0) + 0.5 * rho * V * V * self.rocket.S * cna / self.rocket.mass(tl)
                alpha_lim = self.alpha_max
                # AoA the gimbal can hold against the aero stiffness, with 40 % reserve for regulation
                a0 = np.interp(tl, self.schedule_table["t"], self.schedule_table["a0"])
                b = np.interp(tl, self.schedule_table["t"], self.schedule_table["b"])
                if a0 > 1e-6:
                    alpha_lim = min(alpha_lim, 0.6 * abs(b) * self.lim / a0)
                alpha_c = min(np.linalg.norm(a_lat) / k_lat, alpha_lim)
                al = a_lat - (a_lat @ vh) * vh
                n_cmd = math.cos(alpha_c) * vh + math.sin(alpha_c) * al / max(np.linalg.norm(al), 1e-12)
            # attitude error as a body-frame rotation vector taking the nose onto n_cmd
            nb = Rq.T @ n_cmd
            c = np.cross([1.0, 0, 0], nb)
            sn = np.linalg.norm(c)
            ang = math.atan2(sn, nb[0])
            rot = c * (ang / sn) if sn > 1e-12 else np.zeros(3)
            e_att = np.array([rot[1], rot[2]])
            for i, ax in ((0, 1), (1, 2)):
                self.pid[i].set_gains(Kp, Ki, Kd)
                u[i] = self.pid[i].update(e_att[i], w_ctrl[ax])

        lg = self.log
        lg["t"].append(t); lg["phase"].append(self.phase)
        lg["pos_est"].append(p.copy()); lg["vel_est"].append(v.copy()); lg["quat_est"].append(np.array(q_ctrl))
        lg["accel_bias_est"].append(self.nav.x[6:9].copy()); lg["gyro_bias_est"].append(self.b_gyro.copy())
        lg["nav_std"].append(np.sqrt(np.diag(self.nav.P)) if self.nav_kind == "kf" else np.full(9, np.nan))
        lg["u_cmd"].append(u.copy()); lg["att_err"].append(e_att); lg["rate"].append(np.array(w_ctrl[1:3]))
        lg["a_cmd"].append(a_cmd); lg["alpha_cmd"].append(alpha_c); lg["nose_cmd"].append(np.array(n_cmd))
        lg["gains"].append((Kp, Ki, Kd)); lg["nis_gps"].append(nis)
        for k, val in info.items():
            lg.setdefault(k, []).append(val)
        return u
