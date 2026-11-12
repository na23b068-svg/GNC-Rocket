"""Navigation & observers: Kalman filter vs Luenberger observer under Gaussian sensor noise.

    .venv/bin/python -m scenarios.observers      -> results/observers.json

(a) in-loop flights (each filter feeds LOS guidance on the aggressive path), (b) a noise-variance sweep with
several seeds (attitude hold, so both filters see identical flights), (c) an offline single-plane attitude
estimator [theta, b_gyro] comparison using gnc.observers.attitude_plane_model.
"""
import itertools

import numpy as np

from gnc.flight_computer import FlightComputer, tvc_end_time
from gnc.observers import (KalmanFilter, Luenberger, attitude_plane_model, discretize, kalman_gain_dare,
                           luenberger_gain, nav_luenberger_gains, nav_model)
from gnc.sim import SimConfig, simulate
from scenarios.attitude_control import quat_angle
from scenarios.common import (AGGRESSIVE_WP, DEG, events, flog, plot, pmap, rnd, rocket, rocket_block, series,
                              write_demo)
from scenarios.los_path import path_metrics

SEED = 1
NOISE = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
SEEDS = (1, 2, 3, 4, 5)
T_SWEEP = 15.0
NAMES = {"kf": "Kalman filter", "luenberger": "Luenberger observer"}


class LoggingFC(FlightComputer):
    """Flight computer that also logs the KF position/velocity covariance (for NEES)."""

    def update(self, t, meas):
        u = super().update(t, meas)
        if self.nav_kind == "kf":
            self.log.setdefault("P6", []).append(self.nav.P[:6, :6].copy())
        return u


def nav_errors(res, fc):
    lt = flog(fc, "t")
    X = np.column_stack([np.interp(lt, res["t"], res["x"][:, i]) for i in range(13)])
    X[:, 6:10] /= np.linalg.norm(X[:, 6:10], axis=1, keepdims=True)
    ep, ev = flog(fc, "pos_est") - X[:, 0:3], flog(fc, "vel_est") - X[:, 3:6]
    ea = quat_angle(X[:, 6:10], flog(fc, "quat_est")) * DEG
    nees = None
    if fc.nav_kind == "kf":
        e6 = np.hstack([ep, ev])
        nees = np.array([e @ np.linalg.solve(P, e) for e, P in zip(e6, fc.log["P6"])])
    nis = np.array([n for n in fc.log["nis_gps"] if n is not None])
    win = lt >= res["events"]["rail_exit"]
    return lt, ep, ev, ea, nees, nis, win


def rms(a):
    return float(np.sqrt(np.mean(np.sum(np.atleast_2d(a) ** 2, axis=1))))


# ------------------------------------------------------------------------------------------------ (a) in-loop
def fly_inloop(nav):
    fc = LoggingFC(guidance="los_path", waypoints=np.array(AGGRESSIVE_WP, float), nav=nav)
    res = simulate(rocket(), SimConfig(seed=SEED), fc)
    return nav, res, fc


def inloop_variants():
    runs = {nav: (res, fc) for nav, res, fc in pmap(fly_inloop, ["kf", "luenberger"])}
    err = {nav: nav_errors(*runs[nav]) for nav in runs}
    out = []
    for nav, other in (("kf", "luenberger"), ("luenberger", "kf")):
        res, fc = runs[nav]
        lt, ep, ev, ea, nees, nis, win = err[nav]
        lo, epo, evo, *_ = err[other]
        k = slice(None, None, 4)     # 25 Hz plots over the full ascent
        plots = [
            plot("pos_err", "Position-estimate error norm |p_hat - p|", "m",
                 [series(NAMES[nav] + " (this flight)", lt[k], np.linalg.norm(ep, axis=1)[k], 3),
                  series(NAMES[other] + " (other variant, same seed)", lo[k], np.linalg.norm(epo, axis=1)[k], 3)]),
            plot("pos_axes", "Position-estimate error per axis" + (" with KF 1-sigma" if nav == "kf" else ""), "m",
                 [series(f"{a} error", lt[k], ep[k, i], 3) for i, a in enumerate("ENU")]
                 + ([series(f"+1 sigma {a}", lt[k], flog(fc, "nav_std")[k, i], 3) for i, a in enumerate("ENU")]
                    if nav == "kf" else [])),
            plot("vel_err", "Velocity-estimate error norm |v_hat - v|", "m/s",
                 [series(NAMES[nav], lt[k], np.linalg.norm(ev, axis=1)[k], 4),
                  series(NAMES[other] + " (other variant)", lo[k], np.linalg.norm(evo, axis=1)[k], 4)]),
            plot("att_err", "Attitude-estimate error (gyro strapdown + magnetometer observer)", "deg",
                 [series("|q_hat vs q|", lt[k], ea[k], 4)]),
            plot("bias", "Accelerometer-bias state estimate (ENU)", "m/s^2",
                 [series(f"b_{a}", lt[k], flog(fc, "accel_bias_est")[k, i], 4) for i, a in enumerate("ENU")]),
        ]
        m = {"Position RMSE (rail exit -> apogee) [m]": rnd(rms(ep[win]), 3),
             "Velocity RMSE [m/s]": rnd(rms(ev[win]), 4),
             "Position RMSE E/N/U [m]": " / ".join(f"{np.sqrt(np.mean(ep[win, i] ** 2)):.2f}" for i in range(3)),
             "Max position error [m]": rnd(np.linalg.norm(ep[win], axis=1).max(), 2),
             f"Other filter ({NAMES[other]}) position RMSE [m]": rnd(rms(epo[err[other][6]]), 3),
             f"Other filter ({NAMES[other]}) velocity RMSE [m/s]": rnd(rms(evo[err[other][6]]), 4),
             "Max attitude-estimate error [deg]": rnd(ea[win].max(), 3),
             "Guided cross-track RMS with this filter [m]": rnd(path_metrics(res, fc, AGGRESSIVE_WP)["rms"], 2),
             "Apogee AGL [m]": rnd(res["summary"]["apogee_agl_m"], 1)}
        if nav == "kf":
            ts = lt[win]
            plots.append(plot("consistency", "KF consistency: NEES of [p, v] (6 dof, 1 s moving mean) and GPS NIS",
                              "-", [series("NEES (1 s mean)", ts[::4], np.convolve(nees[win], np.ones(100) / 100,
                                                                                    "same")[::4], 3),
                                    series("expected (6)", [ts[0], ts[-1]], [6, 6])]))
            m["Mean NEES [p,v] (expected 6)"] = rnd(np.mean(nees[win]), 2)
            m["Mean GPS NIS (expected 6)"] = rnd(np.mean(nis), 2)
        tt, ab = rocket_block(res, fc)
        out.append({"id": f"inloop_{nav}", "label": f"{'KF' if nav == 'kf' else 'Luenberger'} in the loop (nominal noise)",
                    "t": tt, "rocket": ab, "objects": [{"name": "Waypoints", "kind": "waypoints", "pos": AGGRESSIVE_WP}],
                    "events": events(res, {"tvc_end": tvc_end_time(rocket())}), "plots": plots, "metrics": m,
                    "notes": (f"The {NAMES[nav]} supplies position and velocity to the LOS guidance on the aggressive "
                              "zig-zag path (the 'Nav estimate' trace in the 3D/2D view). Nominal sensor noise, "
                              "seed 1. The other filter's errors on its own flight with the same seed (identical "
                              "sensor-noise draws) are overlaid. Both filters run "
                              "on the same 9-state model [p, v, b_a] driven by the IMU (accel rotated by the attitude "
                              "estimate) and corrected by GPS (10 Hz) and baro (50 Hz). The KF weights each update "
                              "by its covariance; the Luenberger observer uses fixed pole-placement gains.")})
    return out, runs["kf"]


# ------------------------------------------------------------------------------------------------ (b) sweep
def sweep_run(args):
    k, seed, nav = args
    fc = LoggingFC(nav=nav)
    res = simulate(rocket(), SimConfig(seed=seed, t_max=T_SWEEP, sensor_noise_scale=k), fc)
    lt, ep, ev, ea, nees, nis, win = nav_errors(res, fc)
    return {"k": k, "seed": seed, "nav": nav, "pos": rms(ep[win]), "vel": rms(ev[win]),
            "pos_max": float(np.linalg.norm(ep[win], axis=1).max()), "pos_end": float(np.linalg.norm(ep[-1])),
            "nees": float(np.mean(nees[win])) if nees is not None else None,
            "nis": float(np.mean(nis)) if nis.size else None, "att": float(ea[win].max())}


def stability_rows():
    """Error-dynamics spectral radius per 0.1 s GPS cycle (per axis): Luenberger fixed gains vs steady-state KF."""
    dt, ng, nb = 0.01, 10, 2
    Ad, _, Qd, _ = nav_model(dt, 0.3, 0.02)
    A1 = Ad[np.ix_([0, 3, 6], [0, 3, 6])]       # per-axis block (nav_model = kron(axis, I3))
    L = nav_luenberger_gains(dt, ng, nb, (0.8, 0.85, 0.97), (0.99, 0.992, 0.999))
    Lg, Lb = L["gps"][np.ix_([0, 3, 6], [0, 3])], L["baro"][[2, 5, 8], :]
    Cg, Cb = np.array([[1.0, 0, 0], [0, 1.0, 0]]), np.array([[1.0, 0, 0]])
    Mh = (np.eye(3) - Lg @ Cg) @ np.linalg.matrix_power(A1, ng)
    Mv = np.eye(3)
    for s in range(1, ng + 1):
        Mv = A1 @ Mv
        if s == ng:
            Mv = (np.eye(3) - Lg @ Cg) @ Mv
        if s % nb == 0:
            Mv = (np.eye(3) - Lb @ Cb) @ Mv
    Ad10, _, Qd10, _ = nav_model(dt * ng, 0.3, 0.02)
    A10, Q10 = Ad10[np.ix_([0, 3, 6], [0, 3, 6])], Qd10[np.ix_([0, 3, 6], [0, 3, 6])]
    rows = []
    for k in NOISE:
        Lk, _ = kalman_gain_dare(A10, Cg, Q10, np.diag([1.5 ** 2, 0.15 ** 2]) * k ** 2)
        rows.append(["KF steady state, horizontal axis", k, rnd(max(abs(np.linalg.eigvals((np.eye(3) - Lk @ Cg) @ A10))), 4)])
    rows += [["Luenberger, horizontal axis (GPS)", "any", rnd(max(abs(np.linalg.eigvals(Mh))), 4)],
             ["Luenberger, vertical axis (GPS + baro, multirate)", "any", rnd(max(abs(np.linalg.eigvals(Mv))), 4)]]
    return rows


def sweep_variant():
    runs = pmap(sweep_run, itertools.product(NOISE, SEEDS, ("kf", "luenberger")))
    stat = {}
    for nav in ("kf", "luenberger"):
        for key in ("pos", "vel", "nees", "nis", "pos_max", "att"):
            vals = [[r[key] for r in runs if r["nav"] == nav and r["k"] == k] for k in NOISE]
            if vals[0][0] is not None:
                stat[nav, key] = (np.array([np.mean(v) for v in vals]), np.array([np.std(v) for v in vals]))
    x = list(NOISE)

    def band(nav, key, lab, n=4):
        mu, sd = stat[nav, key]
        return [series(f"{lab} mean", x, mu, n, "lines+markers"), series(f"{lab} mean + 1 std", x, mu + sd, n),
                series(f"{lab} mean - 1 std", x, mu - sd, n)]
    plots = [
        plot("pos_rmse", "Position RMSE vs measurement-noise scale (5 seeds)", "m",
             band("kf", "pos", "KF") + band("luenberger", "pos", "Luenberger"), xlabel="noise std scale k [-]"),
        plot("vel_rmse", "Velocity RMSE vs measurement-noise scale (5 seeds)", "m/s",
             band("kf", "vel", "KF") + band("luenberger", "vel", "Luenberger"), xlabel="noise std scale k [-]"),
        plot("ratio", "Luenberger / KF RMSE ratio", "-",
             [series("position", x, stat["luenberger", "pos"][0] / stat["kf", "pos"][0], 3, "lines+markers"),
              series("velocity", x, stat["luenberger", "vel"][0] / stat["kf", "vel"][0], 3, "lines+markers"),
              series("equal", [x[0], x[-1]], [1, 1])], xlabel="noise std scale k [-]"),
        plot("nees", "KF consistency vs noise: mean NEES [p, v] and GPS NIS (expected 6)", "-",
             [series("NEES mean", x, stat["kf", "nees"][0], 3, "lines+markers"),
              series("NIS mean", x, stat["kf", "nis"][0], 3, "lines+markers"),
              series("expected (6)", [x[0], x[-1]], [6, 6])], xlabel="noise std scale k [-]"),
    ]
    rows = []
    for i, k in enumerate(NOISE):
        f = lambda nav, key, n=3: f"{stat[nav, key][0][i]:.{n}f} +- {stat[nav, key][1][i]:.{n}f}"  # noqa: E731
        rows.append([k, f("kf", "pos"), f("luenberger", "pos"), f("kf", "vel"), f("luenberger", "vel"),
                     rnd(stat["luenberger", "pos"][0][i] / stat["kf", "pos"][0][i], 2), f("kf", "nees", 2),
                     f("kf", "nis", 2), rnd(stat["kf", "pos_max"][0][i], 2), rnd(stat["luenberger", "pos_max"][0][i], 2)])
    i1 = NOISE.index(1.0)
    return {"id": "noise_sweep", "label": "Noise-variance sweep (analysis)", "t": [], "rocket": None, "objects": [],
            "events": [], "plots": plots,
            "tables": [{"title": f"Navigation error vs noise scale ({len(SEEDS)} seeds each, mean +- std, rail exit "
                                 f"-> {T_SWEEP:.0f} s, attitude hold)",
                        "columns": ["noise k", "KF pos RMSE [m]", "Luen. pos RMSE [m]", "KF vel RMSE [m/s]",
                                    "Luen. vel RMSE [m/s]", "Luen./KF pos", "KF NEES (6)", "KF NIS (6)",
                                    "KF max pos err [m]", "Luen. max pos err [m]"], "rows": rows},
                       {"title": "Stability of the error dynamics: spectral radius per 0.1 s GPS cycle (< 1 = stable)",
                        "columns": ["observer", "noise k", "spectral radius"], "rows": stability_rows()}],
            "metrics": {"KF pos RMSE @ k=1 [m]": rnd(stat["kf", "pos"][0][i1], 3),
                        "Luenberger pos RMSE @ k=1 [m]": rnd(stat["luenberger", "pos"][0][i1], 3),
                        "KF vel RMSE @ k=1 [m/s]": rnd(stat["kf", "vel"][0][i1], 4),
                        "Luenberger vel RMSE @ k=1 [m/s]": rnd(stat["luenberger", "vel"][0][i1], 4),
                        "KF NEES range": f"{stat['kf', 'nees'][0].min():.2f} - {stat['kf', 'nees'][0].max():.2f}",
                        "Runs": len(runs), "Diverged runs": sum(r["pos_max"] > 100 for r in runs)},
            "notes": ("sensor_noise_scale k multiplies every sensor std (IMU, baro, GPS, magnetometer). The KF's R "
                      "is scaled with k (it knows the datasheet noise); the Luenberger gains are fixed (designed "
                      "for k = 1). Each point is the mean over 5 seeds of a 15 s attitude-hold flight. Navigation "
                      "is not in the control loop here, so both filters see identical flights and noise. "
                      "Both observers stay stable at every level, with errors bounded and growing roughly linearly in "
                      "k. The Luenberger error map is time-invariant with spectral radius < 1 (table), and the KF "
                      "error map is stable for every R. The KF has the smaller position error at every noise level "
                      "because it weights GPS, baro and IMU by their variances. The fixed-gain observer is tuned to "
                      "track GPS quickly, so its position error is ~2.8x the KF's at every k. On velocity the fixed "
                      "gain is slightly better at k <= 1 (the KF is conservative there), but about 2x worse at k = 8, "
                      "where the KF's R adapts and the fixed gain does not. The KF steady-state spectral radius is set by "
                      "the slow accel-bias mode. NEES/NIS below 6 means "
                      "the KF is somewhat conservative (Q covers attitude-error leakage of the ~100 m/s^2 thrust).")}


# ------------------------------------------------------------------------------------------------ (c) attitude plane
def attitude_plane_run(theta, q, k, seed, dt=0.01, b0=0.01, sg=0.002, sr=np.radians(0.5), sb=1e-4):
    """Offline [theta, b_gyro] estimators: gyro (rate + bias + noise) as input, attitude reference at 50 Hz."""
    rng = np.random.default_rng(seed)
    n = len(theta)
    gyro = q + b0 + rng.normal(0, k * sg, n)
    ref = theta + rng.normal(0, k * sr, n)
    Ad, Bd, C = attitude_plane_model(dt)
    Qc = np.diag([(k * sg) ** 2 * dt, sb ** 2])      # rate-noise density (sampled white noise) + bias walk
    _, _, Qd = discretize(np.array([[0, -1.0], [0, 0]]), np.array([[1.0], [0]]), dt, Qc)
    R = np.array([[(k * sr) ** 2]])
    kf = KalmanFilter(Ad, Bd, {"ref": C}, Qd, {"ref": R}, [ref[0], 0.0], np.diag([(k * sr) ** 2, 0.02 ** 2]))
    A2 = attitude_plane_model(2 * dt)[0]
    Lg = luenberger_gain(A2, C, [np.exp(-2 * dt / 0.3), np.exp(-2 * dt / 3.0)])
    lu = Luenberger(Ad, Bd, {"ref": C}, {"ref": Lg}, [ref[0], 0.0])
    out = np.zeros((n, 5))
    dr = ref[0]
    for i in range(n):
        if i:
            kf.predict([gyro[i - 1]]); lu.predict([gyro[i - 1]])
            dr += gyro[i - 1] * dt
        if i % 2 == 0:
            kf.update("ref", [ref[i]]); lu.update("ref", [ref[i]])
        out[i] = [kf.x[0], lu.x[0], dr, kf.x[1], lu.x[1]]
    return out, ref


def attitude_variant(res, fc):
    t = res["t"][::10]
    t = t[t >= res["events"]["rail_exit"]]
    q = np.interp(t, res["t"], res["x"][:, 11])
    theta = np.concatenate([[0.0], np.cumsum(0.5 * (q[1:] + q[:-1]) * np.diff(t))])
    out, ref = attitude_plane_run(theta, q, 1.0, 0)
    rows, rm = [], {}
    for k in NOISE:
        e = np.array([[np.sqrt(np.mean((o[:, j] - theta) ** 2)) for j in (0, 1)]
                      for o, _ in (attitude_plane_run(theta, q, k, s) for s in range(5))]) * DEG
        rm[k] = e
        rows.append([k, f"{e[:, 0].mean():.3f} +- {e[:, 0].std():.3f}", f"{e[:, 1].mean():.3f} +- {e[:, 1].std():.3f}",
                     rnd(e[:, 1].mean() / e[:, 0].mean(), 2), rnd(0.5 * k, 2)])
    w = t <= 15.0
    plots = [
        plot("theta_err", "Pitch-angle estimate error (noise k = 1)", "deg",
             [series("Kalman", t[w], (out[w, 0] - theta[w]) * DEG), series("Luenberger", t[w], (out[w, 1] - theta[w]) * DEG),
              series("gyro only (dead reckoning, bias unobservable)", t[w], (out[w, 2] - theta[w]) * DEG),
              series("raw attitude reference error", t[w][::2], (ref[w][::2] - theta[w][::2]) * DEG)]),
        plot("bias", "Gyro-bias estimate (true 0.573 deg/s)", "deg/s",
             [series("Kalman", t[w], out[w, 3] * DEG), series("Luenberger", t[w], out[w, 4] * DEG),
              series("true bias", [t[w][0], t[w][-1]], [0.01 * DEG] * 2)]),
        plot("rmse_k", "Pitch-angle RMSE vs noise scale (5 seeds)", "deg",
             [series("Kalman", NOISE, [rm[k][:, 0].mean() for k in NOISE], 4, "lines+markers"),
              series("Luenberger", NOISE, [rm[k][:, 1].mean() for k in NOISE], 4, "lines+markers"),
              series("raw reference std", NOISE, [0.5 * k for k in NOISE], 3, "lines+markers")],
             xlabel="noise std scale k [-]"),
    ]
    return {"id": "attitude_plane", "label": "Attitude estimator [theta, b_gyro] (offline)", "t": [], "rocket": None,
            "objects": [], "events": [], "plots": plots,
            "tables": [{"title": "Pitch-angle RMSE vs noise scale (gyro 0.002k rad/s, reference 0.5k deg at 50 Hz)",
                        "columns": ["noise k", "Kalman [deg]", "Luenberger [deg]", "Luen./KF", "raw reference std [deg]"],
                        "rows": rows}],
            "metrics": {"KF theta RMSE @ k=1 [deg]": rnd(rm[1.0][:, 0].mean(), 4),
                        "Luenberger theta RMSE @ k=1 [deg]": rnd(rm[1.0][:, 1].mean(), 4),
                        "Gyro-only drift after 15 s [deg]": rnd(abs(out[w, 2][-1] - theta[w][-1]) * DEG, 2),
                        "Observability rank, reference / gyro only": "2 / 0"},
            "notes": ("Single-plane attitude estimator from gnc.observers.attitude_plane_model: x = [theta, b_gyro], "
                      "the gyro is the input and an attitude reference (e.g. magnetometer tilt, 0.5 deg noise, 50 Hz) "
                      "is the measurement. True pitch rate from the KF in-loop flight, with a 0.01 rad/s gyro bias. "
                      "Without the reference the system has observability rank 0: dead reckoning drifts at the bias "
                      "rate. With it (rank 2) both observers recover the bias. The Luenberger poles are placed for 0.3 s "
                      "(angle) and 3 s (bias) time constants; the KF gain follows from Q (gyro noise, bias walk) and "
                      "R, so it filters the noisy reference much harder. On the rocket the same structure is used in 3D "
                      "(rail-averaged gyro bias + magnetometer correction).")}


def main():
    flights, (res_kf, fc_kf) = inloop_variants()
    demo = {"id": "observers", "group": "Navigation & observers", "title": "Kalman filter vs Luenberger observer",
            "description": (
                "State estimation under Gaussian sensor noise (IMU 200 Hz, baro 50 Hz, magnetometer 50 Hz, GPS 10 Hz). "
                "The navigation filter estimates [position, velocity, accelerometer bias] in ENU, driven by the "
                "accelerometer and corrected by GPS and baro. It is either a Kalman filter or a Luenberger observer "
                "with fixed gains from pole placement of the discrete error map.\n\n"
                "Variants: each filter in the guidance loop on the aggressive LOS path, a measurement-noise "
                "variance sweep (0.25x to 8x datasheet std, 5 seeds each) with RMSE, consistency (NEES/NIS) and "
                "stability, and an offline single-plane attitude estimator [theta, b_gyro] that shows why the "
                "attitude reference is needed (gyro-only is unobservable)."),
            "variants": flights + [sweep_variant(), attitude_variant(res_kf, fc_kf)]}
    for v in demo["variants"]:
        print(f"[{v['id']}] " + ", ".join(f"{k}: {x}" for k, x in list(v["metrics"].items())[:5]))
    write_demo(demo)


if __name__ == "__main__":
    main()
