"""LOS waypoint path following at the edge of the TVC authority envelope.

    .venv/bin/python -m scenarios.los_path      -> results/los_path.json

Variants: aggressive zig-zag (no disturbance / wind + gusts + misalignment), a path with one jog placed beyond
the authority envelope, and the envelope sweep itself (single lateral jogs vs start altitude and size).
"""
import itertools
import math

import numpy as np

from gnc.flight_computer import FlightComputer, tvc_end_time
from gnc.sim import SimConfig, simulate
from scenarios.common import (AGGRESSIVE_WP, BEYOND_WP, DEG, DIST, DIST_TEXT, DOC_WP, LIM_DEG, events, flog,
                              guided_mask, hlines, plot, pmap, rnd, rocket, rocket_block, sat_frac, series,
                              write_demo)

T_PLOT = 6.0
SEED = 1
TAN6 = math.tan(math.radians(6.0))
ENV_H = (30, 100, 200, 300, 400)     # jog start altitude [m AGL]
ENV_J = (15, 30, 60)                 # lateral (North) jog size [m]
ENV_DH = 200.0                       # vertical span of the jog [m]


def alpha_limit(fc, tl):
    """AoA clip the flight computer applies: min(alpha_max, 0.6 * alpha_hold(t)) (gnc/flight_computer.py)."""
    tab = fc.schedule_table
    a0, b = np.interp(tl, tab["t"], tab["a0"]), np.interp(tl, tab["t"], tab["b"])
    return np.minimum(fc.alpha_max, 0.6 * np.abs(b) * fc.lim / np.maximum(a0, 1e-6))


def path_metrics(res, fc, W):
    """Guided-window path-following metrics shared with the robustness sweeps."""
    ph = flog(fc, "phase")
    g1 = ph == 1
    ct = flog(fc, "cross_track")
    g = guided_mask(res, fc)
    uact = res["u_act"] * DEG
    p = res["x"][:, :3]
    W = np.asarray(W, float)
    d_pass = [float(np.min(np.linalg.norm(p - w, axis=1))) for w in W[1:]]
    lt = flog(fc, "t")
    tl = lt - (fc.t_ld or 0.0)
    clip = flog(fc, "alpha_cmd") >= 0.99 * alpha_limit(fc, tl)
    return {"rms": float(np.sqrt(np.mean(ct[g1] ** 2))), "max": float(ct[g1].max()),
            "rms_h": float(np.sqrt(np.mean(flog(fc, "cross_track_h")[g1] ** 2))),
            "rms_v": float(np.sqrt(np.mean(flog(fc, "cross_track_v")[g1] ** 2))),
            "ct_end": float(ct[g1][-1]), "pass": d_pass, "sat": sat_frac(uact[g]),
            "near": float(np.mean(np.any(np.abs(uact[g]) >= 0.9 * LIM_DEG, axis=1))),
            "clip": float(np.mean(clip[g1])), "gmax": float(np.abs(uact[g]).max()),
            "effort": float(np.sum(flog(fc, "u_cmd") ** 2) * fc.dt * DEG ** 2)}


def fly_path(W, kw, seed=SEED, t_max=400.0):
    fc = FlightComputer(guidance="los_path", waypoints=np.array(W, float))
    res = simulate(rocket(), SimConfig(seed=seed, t_max=t_max, **kw), fc)
    return res, fc


def flight_variant(case):
    vid, label, W, kw, note = case
    res, fc = fly_path(W, kw)
    m = path_metrics(res, fc, W)
    lt = flog(fc, "t")
    wl = lt <= T_PLOT
    k = np.nonzero(res["t"] <= T_PLOT)[0][::10]
    ts, uact = res["t"][k], res["u_act"][k] * DEG
    tl = lt - (fc.t_ld or 0.0)
    ph = flog(fc, "phase")
    alim = np.where(ph == 1, alpha_limit(fc, tl), np.nan) * DEG
    eff = np.cumsum(np.sum(flog(fc, "u_cmd") ** 2, axis=1)) * fc.dt * DEG ** 2
    xte = [series("total |e|", lt[wl], flog(fc, "cross_track")[wl], 3),
           series("horizontal", lt[wl], flog(fc, "cross_track_h")[wl], 3),
           series("vertical-plane", lt[wl], flog(fc, "cross_track_v")[wl], 3)]
    metrics = {"RMS cross-track (guided) [m]": rnd(m["rms"], 2), "Max cross-track (guided) [m]": rnd(m["max"], 2),
               "RMS horizontal cross-track [m]": rnd(m["rms_h"], 2),
               "RMS vertical-plane cross-track [m]": rnd(m["rms_v"], 2),
               "Cross-track at TVC end [m]": rnd(m["ct_end"], 2)}
    for i, d in enumerate(m["pass"]):
        tag = " (direction point, reached in coast)" if i == len(m["pass"]) - 1 else ""
        metrics[f"WP{i + 1} {tuple(int(c) for c in W[i + 1])} pass distance{tag} [m]"] = rnd(d, 2)
    metrics.update({"Gimbal saturation (|delta| = 5 deg) [% guided]": rnd(100 * m["sat"], 1),
                    "Gimbal >= 4.5 deg [% guided]": rnd(100 * m["near"], 1),
                    "AoA command at authority clip [% guided]": rnd(100 * m["clip"], 1),
                    "Max gimbal [deg]": rnd(m["gmax"], 2), "Control effort [deg^2 s]": rnd(m["effort"], 3),
                    "Apogee AGL [m]": rnd(res["summary"]["apogee_agl_m"], 1)})
    if vid == "aggressive_dist":   # same disturbance with the wind reversed (from the East)
        rf, ff = fly_path(W, dict(kw, wind=tuple(-w for w in kw["wind"])))
        mf = path_metrics(rf, ff, W)
        metrics["Same, wind from the East instead: RMS cross-track [m]"] = rnd(mf["rms"], 2)
        metrics["Same, wind from the East instead: gimbal saturation [% guided]"] = rnd(100 * mf["sat"], 1)
    if vid == "aggressive":   # gentle doc path as a baseline comparison
        rb, fb = fly_path(DOC_WP, {})
        mb = path_metrics(rb, fb, DOC_WP)
        lb = flog(fb, "t")
        xte.append(series("baseline: gentle doc path, total |e|", lb[lb <= T_PLOT],
                          flog(fb, "cross_track")[lb <= T_PLOT], 3))
        metrics["Baseline gentle path (0,30,300)-(0,90,900): RMS cross-track [m]"] = rnd(mb["rms"], 2)
        metrics["Baseline gentle path: gimbal >= 4.5 deg [% guided]"] = rnd(100 * mb["near"], 1)
    plots = [
        plot("xte", "Cross-track error to the active segment", "m", xte),
        plot("gimbal", "Gimbal angles (actual) vs the +-5 deg limit", "deg",
             [series("pitch delta_p", ts, uact[:, 0], 3), series("yaw delta_y", ts, uact[:, 1], 3)]
             + hlines(0, T_PLOT, [-LIM_DEG, LIM_DEG])),
        plot("aoa", "AoA command vs the authority clip min(10 deg, 0.6 alpha_hold) and actual AoA", "deg",
             [series("commanded AoA", lt[wl], flog(fc, "alpha_cmd")[wl] * DEG, 3),
              series("authority clip", lt[wl], alim[wl], 3),
              series("actual total AoA", ts, res["alpha"][k] * DEG, 3)]),
        plot("los_az", "LOS azimuth (from East, CCW)", "deg",
             [series("LOS azimuth", lt[wl], flog(fc, "los_az")[wl] * DEG, 2)]),
        plot("los_el", "LOS elevation above the horizon", "deg",
             [series("LOS elevation", lt[wl], flog(fc, "los_el")[wl] * DEG, 3)]),
        plot("heading", "Heading error: angle between velocity and LOS", "deg",
             [series("heading error", lt[wl], flog(fc, "heading_error")[wl] * DEG, 3)]),
        plot("wp", "Active waypoint segment index", "-", [series("segment i (W_i -> W_i+1)", lt[wl],
                                                                 flog(fc, "wp_index")[wl], 0)]),
        plot("effort", "Control effort, cumulative integral of |delta|^2 dt", "deg^2 s",
             [series("effort", lt[wl], eff[wl], 3)]),
    ]
    tt, ab = rocket_block(res, fc)
    print(f"[{vid}] RMS {m['rms']:.2f} m, max {m['max']:.1f} m, pass {np.round(m['pass'], 1)}, sat {m['sat']:.2f}, "
          f">=4.5deg {m['near']:.2f}, clip {m['clip']:.2f}")
    return {"id": vid, "label": label, "t": tt, "rocket": ab,
            "objects": [{"name": "Waypoints", "kind": "waypoints", "pos": W},
                        {"name": "Reference path", "kind": "path", "pos": W}],
            "events": events(res, {"tvc_end": tvc_end_time(rocket())}), "plots": plots, "metrics": metrics,
            "notes": note}


def envelope_case(args):
    h, J = args
    W = [[0, 0, 0], [h * TAN6, 0, h], [(h + ENV_DH) * TAN6, J, h + ENV_DH],
         [(h + ENV_DH + 400) * TAN6, J, h + ENV_DH + 400]]
    res, fc = fly_path(W, {}, t_max=4.2)
    m = path_metrics(res, fc, W)
    t_h = float(np.interp(h, res["x"][:, 2], res["t"]))
    return h, J, t_h, 1 - m["pass"][1] / J, m["pass"][1], m["max"], m["near"], m["clip"]


def envelope_variant():
    rows = pmap(envelope_case, itertools.product(ENV_H, ENV_J))
    ser = []
    for J in ENV_J:
        rr = [r for r in rows if r[1] == J]
        ser.append(series(f"jog {J} m", [r[0] for r in rr], [100 * r[3] for r in rr], 1, "lines+markers"))
    ser_t = series("time at jog start", [r[0] for r in rows if r[1] == ENV_J[0]],
                   [r[2] for r in rows if r[1] == ENV_J[0]], 2, "lines+markers")
    table = [[r[0], round(r[2], 2), r[1], rnd(100 * r[3], 0), rnd(r[4], 1), rnd(r[5], 1), rnd(100 * r[6], 0),
              rnd(100 * r[7], 0), "flyable" if r[3] >= 0.8 else ("marginal" if r[3] >= 0.5 else "NOT flyable")]
             for r in rows]
    return {"id": "envelope", "label": "Authority envelope (sweep)", "t": [], "rocket": None, "objects": [],
            "events": [],
            "plots": [plot("frac", f"Fraction of a North jog achieved vs jog start altitude (jog over "
                                   f"{ENV_DH:.0f} m of climb)", "% of jog achieved", ser,
                           xlabel="jog start altitude [m AGL]"),
                      plot("tjog", "Time at which the rocket reaches the jog start altitude", "t [s] (since launch)",
                           [ser_t], xlabel="jog start altitude [m AGL]")],
            "tables": [{"title": "Envelope sweep: single lateral jog (0,0,0) -> rail line at h -> +J North over "
                                 f"{ENV_DH:.0f} m of climb -> straight",
                        "columns": ["start alt [m]", "t at start [s]", "jog J [m]", "achieved [%]",
                                    "pass distance [m]", "max cross-track [m]", "gimbal >= 4.5 deg [%]",
                                    "AoA at clip [%]", "verdict"], "rows": table}],
            "metrics": {"Jog start altitude for >= 80 % achieved [m AGL]": "<= ~30-50 m (t < ~1 s)",
                        "Achieved at 100 m start, 15 m jog [%]":
                            rnd(100 * [r for r in rows if r[:2] == (100, 15)][0][3], 0),
                        "Achieved at 30 m start, 60 m jog [%]": rnd(100 * [r for r in rows if r[:2] == (30, 60)][0][3], 0),
                        "Achieved at 300 m start, 15 m jog [%]":
                            rnd(100 * [r for r in rows if r[:2] == (300, 15)][0][3], 0)},
            "notes": "How big a lateral jog can the TVC fly, and how late? Each run is a single jog: follow the "
                     "rail line to altitude h, then move J m North over the next 200 m of climb, then continue "
                     "straight. Starting at 30 m AGL (t ~0.9 s) the rocket achieves ~80 % of even a 60 m jog; "
                     "starting at 100 m (t ~1.5 s) only ~50 % of a 15 m jog; by 200 m (t ~2.1 s) <= 21 %; and above "
                     "300 m (t > 2.6 s, qbar > 30 kPa) < 10 %: the AoA the gimbal can hold against the fins (alpha_hold) has collapsed to 1-3 deg, "
                     "and the climb leaves less than 1.5 s of TVC time. Lateral velocity built early also cannot "
                     "be removed later, so every jog in a zig-zag must be finished, and the climb-out direction "
                     "matched, by ~2 s. This sweep set the waypoints of the aggressive path."}


def main():
    agg_note = ("Aggressive zig-zag placed at the edge of the envelope (see the 'Authority envelope' variant): "
                "East jog (8.5 deg path turn) by 40 m AGL, North jog (9.5 deg) by 100 m, back West (4.3 deg) by "
                "180 m, then a NNW climb-out, all within the first ~2.1 s of the 3.97 s guided window. LOS "
                "guidance (lookahead 150 m, k = 1.5, switching radius 40 m) turns the velocity vector toward the "
                "lookahead point. Gravity-compensated lateral-acceleration commands become an AoA command, "
                "clipped to 60 % of what the gimbal can hold. From ~1.4 s on, the AoA command rides that clip "
                "and the gimbal works close to 5 deg to hold the trim AoA against the fins' restoring moment. "
                "The gentle doc path (one inclined line) is overlaid as a baseline.")
    cases = [
        ("aggressive", "Aggressive path, no disturbance", AGGRESSIVE_WP, {}, agg_note),
        ("aggressive_dist", "Aggressive path, wind + gusts + misalignment", AGGRESSIVE_WP, DIST,
         f"Same zig-zag with {DIST_TEXT}. The three jog waypoints are still passed within ~10 m, but the gimbal "
         "now spends over a third of the guided window pinned at 5 deg fighting the weathercock moment. On the "
         "climb-out (after ~2 s, when the AoA clip is down to 1-3 deg) nothing is left to reject the wind drift, so "
         "the cross-track error grows to ~40 m by TVC end (RMS ~18 m vs ~8 m in calm air). The same "
         "disturbance with the wind from the East instead happens to push the vehicle along the West leg (see "
         "metrics: RMS ~6 m). Near the authority limit, wind direction matters more than tuning."),
        ("beyond", "Beyond the envelope: late 20 m East jog at 300-450 m", BEYOND_WP, {},
         "Deliberately infeasible: the same zig-zag plus a 20 m East jog between 300 and 450 m AGL "
         "(t ~ 2.6-3.2 s, qbar > 30 kPa). The guidance asks for it, but the AoA command is pinned at the "
         "authority clip (~1-1.5 deg) and the rocket barely moves: the jog waypoint is missed by ~16 m and "
         "the cross-track error grows to ~26 m by the time TVC ends. A jog of the same size flown below 100 m is easy "
         "(see the first variant). This is where TVC authority runs out, not a tuning problem."),
    ]
    variants = pmap(flight_variant, cases) + [envelope_variant()]
    demo = {"id": "los_path", "group": "Path following", "title": "LOS waypoint path following at the TVC limit",
            "description": (
                "3D line-of-sight (lookahead) guidance through a staggered waypoint zig-zag (East, then North, then "
                "back West) during the powered ascent. Guidance steers the velocity vector toward a point 150 m "
                "ahead on the active segment; the flight computer turns that into an angle-of-attack command for "
                "the gain-scheduled attitude PID, using the KF navigation estimate (GPS + baro + IMU, Gaussian noise).\n\n"
                "The vehicle constraint drives the design: TVC only has authority while the motor burns (until "
                "3.97 s), and the AoA it can hold against the fins shrinks from ~29 deg at rail exit to <1 deg at "
                "the end of the boost. The 'Authority envelope' variant maps how big a lateral jog is flyable "
                "at each altitude. The aggressive path is placed at that edge, and the 'Beyond' variant "
                "shows a jog the vehicle cannot fly. Cross-track metrics are over the guided window; after TVC "
                "ends the rocket coasts ballistically to apogee."),
            "variants": variants}
    write_demo(demo)


if __name__ == "__main__":
    main()
