"""Robustness: Monte-Carlo sweeps over process disturbances, sensor biases and measurement noise, for LOS path
following on the aggressive (envelope-edge) zig-zag and for attitude hold.

    .venv/bin/python -m scenarios.robustness      -> results/robustness.json  (analysis-only, rocket = null)
"""
import itertools
import math
import time

import numpy as np

from gnc.flight_computer import FlightComputer
from gnc.sim import SimConfig, simulate
from scenarios.common import (AGGRESSIVE_WP, DEG, angle_between, guided_mask, nose, plot, pmap, rnd, rocket,
                              sat_frac, series, write_demo)
from scenarios.los_path import path_metrics

SEEDS = (1, 2, 3, 4, 5)
T_MAX = 4.3                      # covers the whole guided window (TVC end 3.97 s)
LOS_RMS_MAX, LOS_PASS_MAX, HOLD_ERR_MAX = 10.0, 15.0, 2.0   # pass/fail criteria
SWEEPS = [  # id, label, x label, levels, SimConfig kwargs per level
    ("wind", "Wind speed (from the West, 1/7 shear)", "wind at 10 m [m/s]", [0.0, 5.0, 10.0, 15.0],
     lambda v: dict(wind=(v, 0.0, 0.0), wind_shear_exponent=1 / 7)),
    ("gust", "Turbulence intensity (Gauss-Markov, no mean wind)", "turbulence std [m/s]", [0.0, 1.5, 3.0, 4.5],
     lambda v: dict(gust={"std_mps": v, "tau_s": 1.5}) if v else {}),
    ("misalign", "Thrust-misalignment bias", "misalignment per axis [deg]", [0.0, 0.25, 0.5, 1.0],
     lambda v: dict(thrust_misalignment_rad=(math.radians(v), math.radians(v)))),
    ("gyro_bias", "Gyro bias", "gyro bias per axis [rad/s]", [0.0, 0.005, 0.01, 0.02],
     lambda v: dict(gyro_bias=(v, v, v))),
    ("accel_bias", "Accelerometer bias", "accel bias per axis [m/s^2]", [0.0, 0.1, 0.2, 0.5],
     lambda v: dict(accel_bias=(v, v, v))),
    ("noise", "Measurement-noise scale", "noise std scale k [-]", [0.5, 1.0, 2.0, 4.0, 8.0],
     lambda v: dict(sensor_noise_scale=v)),
]


def run(args):
    mode, kw, seed = args
    cfg = SimConfig(seed=seed, t_max=T_MAX, **kw)
    if mode == "los":
        fc = FlightComputer(guidance="los_path", waypoints=np.array(AGGRESSIVE_WP, float))
        res = simulate(rocket(), cfg, fc)
        m = path_metrics(res, fc, AGGRESSIVE_WP)
        return {"rms": m["rms"], "max": m["max"], "pass": max(m["pass"][:-1]), "sat": m["sat"], "clip": m["clip"]}
    fc = FlightComputer()
    res = simulate(rocket(), cfg, fc)
    g = guided_mask(res, fc)
    err = angle_between(nose(res["x"][g, 6:10]), res["rail_dir"]) * DEG
    return {"err": float(err.max()), "sat": sat_frac(res["u_act"][g] * DEG),
            "gmax": float(np.abs(res["u_act"][g]).max() * DEG)}


def main():
    t0 = time.perf_counter()
    jobs = [(sid, i, mode, seed) for sid, _, _, lv, _ in SWEEPS for i in range(len(lv))
            for mode in ("los", "hold") for seed in SEEDS]
    kw = {sid: [f(v) for v in lv] for sid, _, _, lv, f in SWEEPS}
    out = pmap(run, [(m, kw[s][i], seed) for s, i, m, seed in jobs])
    R = {}
    for (sid, i, mode, _), r in zip(jobs, out):
        R.setdefault((sid, i, mode), []).append(r)
    wall = time.perf_counter() - t0
    variants, summary = [], []
    for sid, label, xl, lv, _ in SWEEPS:
        rows, s = [], {k: [] for k in ("rms", "rms_lo", "rms_hi", "err", "err_lo", "err_hi", "sat_los", "sat_hold",
                                       "clip")}
        for i, v in enumerate(lv):
            L, H = R[sid, i, "los"], R[sid, i, "hold"]
            rms_, err_ = np.array([r["rms"] for r in L]), np.array([r["err"] for r in H])
            los_ok = sum(r["rms"] <= LOS_RMS_MAX and r["pass"] <= LOS_PASS_MAX for r in L)
            hold_ok = sum(r["err"] <= HOLD_ERR_MAX for r in H)
            for key, val in (("rms", rms_.mean()), ("rms_lo", rms_.min()), ("rms_hi", rms_.max()),
                             ("err", err_.mean()), ("err_lo", err_.min()), ("err_hi", err_.max()),
                             ("sat_los", 100 * np.mean([r["sat"] for r in L])),
                             ("sat_hold", 100 * np.mean([r["sat"] for r in H])),
                             ("clip", 100 * np.mean([r["clip"] for r in L]))):
                s[key].append(val)
            row = [v, f"{rms_.mean():.2f} +- {rms_.std():.2f}", rnd(np.mean([r["max"] for r in L]), 1),
                   rnd(np.mean([r["pass"] for r in L]), 1), rnd(s["sat_los"][-1], 1), rnd(s["clip"][-1], 0),
                   f"{err_.mean():.2f} +- {err_.std():.2f}", rnd(np.mean([r["gmax"] for r in H]), 2),
                   rnd(s["sat_hold"][-1], 1), f"{los_ok}/{len(L)}", f"{hold_ok}/{len(H)}"]
            rows.append(row)
            summary.append([label, v, row[1], f"{los_ok}/{len(L)}", "PASS" if los_ok == len(L) else "FAIL",
                            row[6], f"{hold_ok}/{len(H)}", "PASS" if hold_ok == len(H) else "FAIL"])
        x = list(lv)
        variants.append({
            "id": sid, "label": label, "t": [], "rocket": None, "objects": [], "events": [],
            "plots": [
                plot("rms", f"LOS cross-track RMS (aggressive path) vs {xl.split(' [')[0]}", "m",
                     [series("mean", x, s["rms"], 3, "lines+markers"), series("min (seeds)", x, s["rms_lo"], 3),
                      series("max (seeds)", x, s["rms_hi"], 3),
                      series(f"pass limit {LOS_RMS_MAX:g} m", [x[0], x[-1]], [LOS_RMS_MAX] * 2)], xlabel=xl),
                plot("hold", "Attitude hold: max true pointing error", "deg",
                     [series("mean", x, s["err"], 3, "lines+markers"), series("min (seeds)", x, s["err_lo"], 3),
                      series("max (seeds)", x, s["err_hi"], 3),
                      series(f"pass limit {HOLD_ERR_MAX:g} deg", [x[0], x[-1]], [HOLD_ERR_MAX] * 2)], xlabel=xl),
                plot("sat", "Gimbal saturation and AoA-clip fraction of the guided window", "%",
                     [series("LOS: gimbal at 5 deg", x, s["sat_los"], 2, "lines+markers"),
                      series("hold: gimbal at 5 deg", x, s["sat_hold"], 2, "lines+markers"),
                      series("LOS: AoA command at authority clip", x, s["clip"], 2, "lines+markers")], xlabel=xl)],
            "tables": [{"title": f"{label}: {len(SEEDS)} seeds per level",
                        "columns": [xl, "LOS RMS xte [m]", "LOS max xte [m]", "LOS worst WP pass [m]",
                                    "LOS gimbal sat [%]", "LOS AoA clip [%]", "hold max err [deg]",
                                    "hold max gimbal [deg]", "hold gimbal sat [%]", "LOS pass", "hold pass"],
                        "rows": rows}],
            "metrics": {"Worst mean LOS RMS [m]": rnd(max(s["rms"]), 2), "Worst mean hold error [deg]": rnd(max(s["err"]), 2),
                        "Max LOS gimbal saturation [%]": rnd(max(s["sat_los"]), 1)},
            "notes": NOTES[sid]})
    n_pass = sum(r[4] == "PASS" for r in summary), sum(r[7] == "PASS" for r in summary)
    variants.insert(0, {
        "id": "summary", "label": "Pass/fail summary", "t": [], "rocket": None, "objects": [], "events": [],
        "plots": [], "tables": [{"title": "Pass/fail across all sweeps (PASS = every seed meets the criteria)",
                                 "columns": ["sweep", "level", "LOS RMS xte [m]", "LOS seeds ok", "LOS",
                                             "hold max err [deg]", "hold seeds ok", "hold"], "rows": summary}],
        "metrics": {"Flights": len(out), "Wall time [s]": rnd(wall, 1),
                    "LOS cases passed": f"{n_pass[0]}/{len(summary)}", "Hold cases passed": f"{n_pass[1]}/{len(summary)}",
                    "LOS criteria": f"RMS cross-track <= {LOS_RMS_MAX:g} m and every jog waypoint passed within "
                                    f"{LOS_PASS_MAX:g} m", "Hold criterion": f"max pointing error <= {HOLD_ERR_MAX:g} deg"},
        "notes": ("Each case is flown with 5 seeds (sensor noise and turbulence draws) through the guided window "
                  f"(0-{T_MAX} s), once with LOS guidance on the aggressive zig-zag (the envelope-edge path from the "
                  "Path-following demo) and once in attitude hold. Navigation (KF + magnetometer attitude observer) "
                  "is in the loop throughout. Because the path sits at the edge of the TVC authority envelope, "
                  "these sweeps show where the margin runs out, not only that the nominal design works.")})
    print(f"{len(out)} flights in {wall:.1f} s")
    for r in summary:
        print("  ", r)
    write_demo({"id": "robustness", "group": "Robustness", "title": "Robustness to disturbances, biases and noise",
                "description": (
                    "Monte-Carlo-style sweeps of the closed-loop GNC system over the brief's robustness axes: process "
                    "disturbances (mean wind, turbulence, thrust-misalignment bias), sensor biases (gyro, "
                    "accelerometer) and measurement-noise level. Each level is flown with 5 seeds for path "
                    "following on the aggressive waypoint zig-zag and for attitude hold. Plots show the mean and "
                    "seed spread of cross-track RMS and pointing error, plus the gimbal saturation fraction. The "
                    "summary variant is the pass/fail table."), "variants": variants})


NOTES = {
    "wind": "Mean crosswind from the West with 1/7 power-law shear: the dominant threat. The fins weathercock the "
            "rocket into the relative wind, and holding a path or attitude against it uses the gimbal authority "
            "that the aggressive path needs for its jogs. Attitude hold passes up to 15 m/s (gimbal saturated "
            "for ~half the window at 15 m/s). Path following fails the 10 m RMS criterion from 5 m/s: the jog "
            "waypoints are still passed within ~10 m, but on the climb-out the AoA clip has collapsed and the "
            "wind drift can no longer be rejected.",
    "gust": "Zero-mean Gauss-Markov turbulence (tau 1.5 s). It disturbs the early, low-speed boost, where AoA = "
            "gust/V is large. Path following barely notices (the guidance re-aims continuously; RMS 7.2 -> 7.8 m). "
            "Attitude hold is more sensitive: the peak pointing error grows ~linearly with the intensity, "
            "and one seed in five exceeds 2 deg from 3 m/s.",
    "misalign": "Constant thrust-vector misalignment (both axes, same sign). In attitude hold it acts as a torque "
                "step at ignition. The integral action trims it out within ~1 s (Analysis demo), so the peak "
                "pointing error grows linearly (0.2 -> 1.8 deg at 1 deg) but stays inside the criterion. In path "
                "following this sign of bias happens to push the vehicle in a helpful direction for the zig-zag "
                "(RMS falls slightly). Like wind, the effect is direction-dependent.",
    "gyro_bias": "Constant gyro bias on all axes. It is removed exactly by the rail-averaged bias estimate (the "
                 "rocket cannot rotate on the rail), so results are identical to the unbiased case up to "
                 "0.02 rad/s (1.1 deg/s). Without that step, 0.01 rad/s would integrate to ~0.6 deg/s of attitude "
                 "drift (see the Observers demo, attitude-plane variant).",
    "accel_bias": "Constant accelerometer bias on all axes: negligible effect. The navigation filter's b_a state "
                  "absorbs it (observable through GPS/baro), and over a 4 s boost a 0.5 m/s^2 bias is small next to "
                  "the ~100 m/s^2 thrust acceleration.",
    "noise": "All sensor std-devs scaled by k. Navigation errors grow linearly with k (Observers demo), but they stay "
             "well below the path-following error. RMS cross-track is unchanged on average up to 8x, with a "
             "larger seed spread (one 8x seed exceeds 10 m). The attitude loop's peak error triples at 8x "
             "(gyro noise enters through the derivative term and the attitude estimate) but stays < 1 deg.",
}


if __name__ == "__main__":
    main()
