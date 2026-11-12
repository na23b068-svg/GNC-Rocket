"""TVC attitude hold with the pole-placement PID (and LQR when gnc provides it), with disturbance and delays.

    .venv/bin/python -m scenarios.attitude_control      -> results/attitude_control.json
"""
import inspect
import math

import numpy as np

from gnc import analysis as an
from gnc.flight_computer import FlightComputer, tvc_end_time
from gnc.sim import SimConfig, simulate
from scenarios.common import (DEG, DIST, DIST_TEXT, LIM_DEG, angle_between, events, flog, guided_mask, hlines,
                              nose, plot, pmap, rnd, rocket, rocket_block, sat_frac, series, write_demo)

T_PLOT = 6.0     # control plots cover the boost + 2 s of coast
SEED = 7


def lqr_available():
    try:
        import gnc.lqr  # noqa: F401
    except ImportError:
        return False
    return "attitude" in inspect.signature(FlightComputer.__init__).parameters


DELAY15 = dict(actuator_delay_s=0.015, estimation_delay_s=0.015)
DELAY30 = dict(actuator_delay_s=0.030, estimation_delay_s=0.030)
CASES = [  # id, label, sim kwargs, flight-computer kwargs, passive comparison kwargs (None = same as sim)
    ("nominal", "PID, nominal", {}, {}),
    ("disturbance", "PID, wind + gusts + 0.25 deg misalignment", DIST, {}),
    ("delay_15_15", "PID, delay 15 ms act. + 15 ms est.", DELAY15, {}),
    ("delay_30_30", "PID, delay 30 ms act. + 30 ms est. (beyond margin)", DELAY30, {}),
]
LQR_CASES = [
    ("lqr_nominal", "LQR, nominal", {}, {"attitude": "lqr"}),
    ("lqr_disturbance", "LQR, wind + gusts + 0.25 deg misalignment", DIST, {"attitude": "lqr"}),
    ("lqr_delay_15_15", "LQR, delay 15 ms act. + 15 ms est.", DELAY15, {"attitude": "lqr"}),
]


def quat_angle(q1, q2):
    return 2 * np.arccos(np.clip(np.abs(np.sum(q1 * q2, axis=1)), 0, 1))


def fly(case):
    vid, label, kw, fck = case
    r = rocket()
    cfg = SimConfig(seed=SEED, **kw)
    fc = FlightComputer(**fck)
    res = simulate(r, cfg, fc)
    env_kw = {k: v for k, v in kw.items() if k not in DELAY30}
    pas = simulate(r, SimConfig(seed=SEED, **env_kw))
    return vid, label, kw, fck, build(vid, label, kw, fck, res, fc, pas)


def build(vid, label, kw, fck, res, fc, pas):
    r = rocket()
    t_end = tvc_end_time(r)
    tt, ab = rocket_block(res, fc)
    n_hold = res["rail_dir"]
    # 100 Hz views of the 1 kHz sim log over the plot window
    k = np.nonzero(res["t"] <= T_PLOT)[0][::10]
    ts = res["t"][k]
    point_err = angle_between(nose(res["x"][k, 6:10]), n_hold) * DEG
    kp = np.nonzero(pas["t"] <= T_PLOT)[0][::10]
    tp = pas["t"][kp]
    pas_dev = angle_between(nose(pas["x"][kp, 6:10]), n_hold) * DEG
    lt, ph = flog(fc, "t"), flog(fc, "phase")
    wl = lt <= T_PLOT
    att = flog(fc, "att_err") * DEG
    ucmd = flog(fc, "u_cmd") * DEG
    uact = res["u_act"] * DEG
    q_true = np.column_stack([np.interp(lt, res["t"], res["x"][:, i]) for i in range(6, 10)])
    q_true /= np.linalg.norm(q_true, axis=1, keepdims=True)
    est_err = quat_angle(q_true, flog(fc, "quat_est")) * DEG
    effort = np.cumsum(np.sum(flog(fc, "u_cmd") ** 2, axis=1)) * fc.dt * DEG ** 2
    g = guided_mask(res, fc)
    g1 = ph == 1
    lim = [-LIM_DEG, LIM_DEG]
    plots = [
        plot("att_err", "Attitude error (controller's estimate) and true pointing error vs the rail line", "deg",
             [series("pitch error (est.)", lt[wl], att[wl, 0]), series("yaw error (est.)", lt[wl], att[wl, 1]),
              series("true pointing error |angle|", ts, point_err),
              series("passive (no TVC): deviation from rail line", tp, pas_dev)]),
        plot("gimbal_p", "Pitch gimbal: command vs actual (servo lag, rate limit) and +-5 deg limit", "deg",
             [series("command", lt[wl], ucmd[wl, 0]), series("actual", ts, uact[k, 0])] + hlines(0, T_PLOT, lim)),
        plot("gimbal_y", "Yaw gimbal: command vs actual and +-5 deg limit", "deg",
             [series("command", lt[wl], ucmd[wl, 1]), series("actual", ts, uact[k, 1])] + hlines(0, T_PLOT, lim)),
        plot("aoa", "Total angle of attack", "deg",
             [series("closed loop", ts, res["alpha"][k] * DEG), series("passive (no TVC)", tp, pas["alpha"][kp] * DEG)]),
        plot("rates", "Body rates (truth)", "deg/s",
             [series("q (pitch)", ts, res["x"][k, 11] * DEG), series("r (yaw)", ts, res["x"][k, 12] * DEG),
              series("passive q", tp, pas["x"][kp, 11] * DEG), series("passive r", tp, pas["x"][kp, 12] * DEG)]),
        plot("effort", "Control effort, cumulative integral of |delta|^2 dt", "deg^2 s",
             [series("effort", lt[wl], effort[wl])]),
        plot("est_err", "Attitude-estimate error (strapdown gyro + magnetometer observer vs truth)", "deg",
             [series("|q_est vs q_true|", lt[wl], est_err[wl])]),
    ]
    m = {"Max attitude error, controller [deg]": rnd(np.abs(att[g1]).max(), 3),
         "Max true pointing error (guided) [deg]": rnd(point_err[(ts >= lt[g1][0]) & (ts <= lt[g1][-1])].max(), 3),
         "Max gimbal [deg]": rnd(np.abs(uact[g]).max(), 2),
         "Gimbal saturation [% of guided time]": rnd(100 * sat_frac(uact[g]), 1),
         "Control effort [deg^2 s]": rnd(effort[-1], 3),
         "Max AoA (guided) [deg]": rnd(res["alpha"][g].max() * DEG, 2),
         "Max attitude-estimate error (guided) [deg]": rnd(est_err[g1].max(), 3),
         "Apogee AGL [m]": rnd(res["summary"]["apogee_agl_m"], 1),
         "Passive: max deviation from rail line (same window) [deg]":
             rnd(pas_dev[(tp >= lt[g1][0]) & (tp <= lt[g1][-1])].max(), 3),
         "Passive: apogee AGL [m]": rnd(pas["summary"]["apogee_agl_m"], 1)}
    d = kw.get("actuator_delay_s", 0) + kw.get("estimation_delay_s", 0)
    notes = {"nominal": "Calm air, nominal sensor noise. The PID holds the rail attitude through the boost with "
                        "small, smooth gimbal motion. The passive rocket weathercocks slightly as the thrust "
                        "tails off and gravity turns the flight path.",
             "disturbance": f"Disturbance: {DIST_TEXT}. Holding a fixed attitude in a crosswind means "
                            "fighting the weathercock moment: the AoA equals wind/V, which is large at low speed (over 30 deg "
                            "just off the rail, with the rail tilted downwind). The gimbal sits at its 5 deg "
                            "limit for about half of the guided window, while the integral action trims the "
                            "misalignment out. The passive rocket instead turns into the wind (large pointing "
                            "deviation, small AoA). Holding the attitude costs gimbal authority and effort.",
             "delay_15_15": "15 ms actuation + 15 ms estimation delay (30 ms loop delay plus ~6 ms intrinsic) is "
                            "inside the Pade delay margin for most of the boost (56 ms mid-boost, 33 ms at TVC end). "
                            "The response is almost identical to nominal.",
             "delay_30_30": "30 + 30 ms (66 ms loop delay incl. intrinsic) exceeds the delay margin at every "
                            "operating point. The closed loop is unstable: a gimbal oscillation grows steadily "
                            "(0.1 deg at 1 s, ~1 deg at 2.5 s, >4 deg at 3.8 s, close to the 5 deg limit) and the "
                            "attitude error and effort grow several-fold. It stops only because the TVC is "
                            "switched off at tail-off (3.97 s); a longer burn would end in saturation."}
    nt = notes.get(vid, "LQR attitude controller (gnc.lqr: Bryson weights theta 4 deg, q 40 deg/s, integral "
                        "0.3 s, delta 5 deg on [theta, q, int e], gain-scheduled at the same 9 points; it is a PID "
                        "with LQR-chosen gains)" + (f" with {DIST_TEXT}." if kw else ", nominal conditions.") +
                   " The first two plots overlay the pole-placement PID in the same conditions. Without delay the LQR "
                   "holds the attitude tighter, with less gimbal effort (and, in the disturbance case, no "
                   "saturation where the PID saturates half the time). The table shows the price: its delay margin "
                   "is only ~28 ms during the first ~2 s (PID 55-73 ms), and its integral trim is slower (settling "
                   "up to ~6 s at TVC end). The 15 + 15 ms delay variant (~36 ms with the sim's intrinsic delay) "
                   "therefore oscillates under LQR while the PID is unaffected. The PID's wn floor tracks the "
                   "airframe frequency, which is what keeps its early-boost delay margin large.")
    if d:
        nt += f" Configured loop delay {1e3 * d:.0f} ms."
    return {"id": vid, "label": label, "t": tt, "rocket": ab, "objects": [],
            "events": events(res, {"tvc_end": t_end}), "plots": plots, "metrics": m, "notes": nt,
            "_cmp": {"lt": lt[wl], "att": np.hypot(att[wl, 0], att[wl, 1]), "ts": ts, "u": np.hypot(*uact[k].T)}}


def lqr_extras(variants):
    """Overlay LQR vs PID (same disturbance) and a margins table from gnc.lqr's helpers."""
    from gnc import lqr
    by = {v["id"]: v for v in variants}
    for lid, pid in (("lqr_nominal", "nominal"), ("lqr_disturbance", "disturbance"),
                     ("lqr_delay_15_15", "delay_15_15")):
        L, P = by[lid]["_cmp"], by[pid]["_cmp"]
        by[lid]["plots"].insert(0, plot("cmp_att", "LQR vs pole-placement PID: attitude error magnitude (est.)",
                                        "deg", [series("LQR", L["lt"], L["att"]), series("PID", P["lt"], P["att"])]))
        by[lid]["plots"].insert(1, plot("cmp_u", "LQR vs PID: gimbal deflection magnitude", "deg",
                                        [series("LQR", L["ts"], L["u"]), series("PID", P["ts"], P["u"])]))
        for key in ("Max attitude error, controller [deg]", "Max true pointing error (guided) [deg]",
                    "Control effort [deg^2 s]",
                    "Gimbal saturation [% of guided time]"):
            by[lid]["metrics"][f"PID: {key}"] = by[pid]["metrics"][key]
    from gnc.flight_computer import design_schedule
    rows = []
    fmt = lambda x, n=1: rnd(x, n) if math.isfinite(x) else "inf"  # noqa: E731
    tab = design_schedule(rocket())[1]
    for i, r in enumerate(lqr.compare(tab)):
        P, L = r["pid"], r["lqr"]
        # workaround: gnc.analysis.critical_delay(d_max=1.0) (used by gnc.lqr) returns inf when the Pade loop is
        # stable again at 1 s although it is unstable at ~40 ms; bisect on a 0-0.3 s bracket instead
        for g in (P, L):
            g["delay_margin_s"] = an.critical_delay(tab["a1"][i], tab["a0"][i], tab["b"][i],
                                                    {k: g[k] for k in ("Kp", "Ki", "Kd")}, 0.02, d_max=0.3)
        rows.append([rnd(r["t"], 2), rnd(r["qbar"] / 1e3, 1), fmt(P["gain_margin_dB"]), fmt(L["gain_margin_dB"]),
                     fmt(P["phase_margin_deg"]), fmt(L["phase_margin_deg"]), fmt(1e3 * P["delay_margin_s"]),
                     fmt(1e3 * L["delay_margin_s"]), fmt(P["overshoot_pct"]), fmt(L["overshoot_pct"]),
                     fmt(P["settling_2pct_s"], 2), fmt(L["settling_2pct_s"], 2)])
    table = {"title": "LQR vs pole-placement PID along the boost (linear loop, 20 ms servo lag; gnc.lqr.compare)",
             "columns": ["t [s]", "qbar [kPa]", "GM PID [dB]", "GM LQR [dB]", "PM PID [deg]", "PM LQR [deg]",
                         "DM PID [ms]", "DM LQR [ms]", "overshoot PID [%]", "overshoot LQR [%]",
                         "settling PID [s]", "settling LQR [s]"], "rows": rows}
    for lid in ("lqr_nominal", "lqr_disturbance", "lqr_delay_15_15"):
        by[lid]["tables"] = [table]
        by[lid]["metrics"]["Min delay margin LQR / PID [ms]"] = \
            f"{min(r[7] for r in rows if r[7] != 'inf'):.1f} / {min(r[6] for r in rows if r[6] != 'inf'):.1f}"
    return True


def main():
    has_lqr = lqr_available()
    out = pmap(fly, CASES + (LQR_CASES if has_lqr else []))
    variants = [v for *_, v in out]
    lqr_table = lqr_extras(variants) if has_lqr else False
    for v in variants:
        v.pop("_cmp")
        print(f"[{v['id']}] " + ", ".join(f"{k}: {x}" for k, x in list(v["metrics"].items())[:6]))
    desc = ("Thrust-vector attitude hold during the boost: the flight computer keeps the nose on the launch-rail "
            "direction using the strapdown/magnetometer attitude estimate and one gain-scheduled PID per plane, "
            "tuned by pole placement (zeta 0.7, wn 8-18.5 rad/s, integral pole -4) on the linearised short-period "
            "model. Control is active from rail exit to the start of the thrust tail-off (3.97 s); after that "
            "the gimbal is centred and the rocket coasts to apogee.\n\n"
            "Variants show the transient response with and without disturbance (wind, turbulence, a side gust "
            "and thrust misalignment), and the closed loop with estimation + actuation delay inside (15 + 15 ms) and "
            "beyond (30 + 30 ms) the delay margin. Each plot overlays the passive (gimbal locked) rocket in the "
            "same conditions.")
    if has_lqr:
        desc += "\n\nLQR variants compare an LQR attitude controller against the pole-placement PID in the same " \
                "conditions" + (" (margins table from gnc.lqr)." if lqr_table else ".")
    write_demo({"id": "attitude_control", "group": "Control", "title": "TVC attitude control: pole-placement PID" + (" vs LQR" if has_lqr else ""),
                "description": desc, "variants": variants})
    return has_lqr


if __name__ == "__main__":
    main()
