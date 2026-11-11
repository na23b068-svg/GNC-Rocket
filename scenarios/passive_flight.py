"""Uncontrolled, fin-stabilised flight (gimbal fixed at 0) from Midland TX, full flight incl. recovery.

    .venv/bin/python -m scenarios.passive_flight

Writes figures/passive_2d.png, figures/passive_3d.png, results/passive.json.  When data/openrocket_flight.csv
exists, the OpenRocket altitude/speed are overlaid (figure + JSON plots) and apogee / max-speed differences are
added to the nominal metrics: a cross-validation of the 6-DOF sim against OpenRocket.
"""
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from gnc.rocket import ROOT, Rocket  # noqa: E402
from gnc.sim import SimConfig, simulate  # noqa: E402

FIG, RES = ROOT / "figures", ROOT / "results"
OR_CSV = ROOT / "data" / "openrocket_flight.csv"
FT = 3.28084
COLORS = {"nominal": "#2a78d6", "wind": "#eb6834", "openrocket": "#1baf7a"}
LABELS = {"liftoff": "Liftoff", "rail_exit": "Rail exit", "burnout": "Burnout", "apogee": "Apogee",
          "drogue": "Drogue", "main": "Main", "landing": "Landing"}

VARIANTS = [
    ("nominal", "No wind", SimConfig(stop_at="ground", recovery=True, t_max=600.0)),
    ("wind", "Wind 6 m/s from E (1/7 shear) + turbulence + 1-cos gust",
     SimConfig(stop_at="ground", recovery=True, t_max=600.0, wind=(-6.0, 0.0, 0.0), wind_shear_exponent=1 / 7,
               gust={"std_mps": 1.5, "tau_s": 1.5,
                     "one_cos": [{"t0": 3.0, "duration": 1.5, "amplitude": [0.0, 5.0, 0.0]}]}, seed=1)),
]


def load_openrocket_csv(path=OR_CSV):
    """Tolerant reader for an OpenRocket CSV export: '#'-comment header, columns found by name.
    Returns dict t, alt, vel (total), mach, stab (each None if the column is missing) or None."""
    if not Path(path).exists():
        return None
    header, rows = None, []
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if not s:
            continue
        parts = [p.strip() for p in s.lstrip("#").split(",")]
        try:
            rows.append([float(p) if p not in ("", "NaN") else np.nan for p in parts])
        except ValueError:
            if any("time" in p.lower() for p in parts):
                header = [p.lower() for p in parts]
    if header is None or not rows:
        return None
    n = len(header)
    data = np.array([r[:n] + [np.nan] * (n - len(r)) for r in rows])

    def col(*keys):
        for k in keys:
            for i, h in enumerate(header):
                if k in h:
                    return data[:, i]
        return None
    t, alt = col("time"), col("altitude")
    if t is None or alt is None:
        return None
    return {"t": t, "alt": alt, "vel": col("vel_total", "total velocity", "velocity"), "mach": col("mach"),
            "stab": col("stability")}


def _resample(res, dt=0.04):
    t = np.arange(0.0, res["t"][-1] + 1e-9, dt)
    pos = np.column_stack([np.interp(t, res["t"], res["x"][:, i]) for i in range(3)])
    q = np.column_stack([np.interp(t, res["t"], res["x"][:, i]) for i in range(6, 10)])
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return t, pos, q


def _r(a, n=3):
    """Rounded JSON list; NaN -> null (JSON has no NaN)."""
    return [round(float(v), n) if np.isfinite(v) else None for v in np.ravel(a)]


def _series(name, x, y, n=3):
    return {"name": name, "x": _r(x, 3), "y": _r(y, n), "mode": "lines"}


def metrics(res):
    s, ev = res["summary"], res["events"]
    m = {"Apogee AGL [m]": s["apogee_agl_m"], "Apogee AGL [ft]": s["apogee_agl_m"] * FT,
         "Time to apogee [s]": ev.get("apogee"), "Max speed [m/s]": s["max_speed_mps"],
         "Max Mach [-]": s["max_mach"], "Max acceleration [g]": s["max_accel_mps2"] / 9.80665,
         "Off-rail velocity [m/s]": s["off_rail_velocity_mps"],
         "Stability off rail [cal]": s["stability_off_rail_cal"], "Burnout time [s]": s["burnout_s"],
         "Landing distance [m]": s["landing_distance_m"], "Landing time [s]": ev.get("landing")}
    return {k: (round(float(v), 3) if v is not None else None) for k, v in m.items()}


def openrocket_comparison(rocket, orc):
    """Re-fly the ascent with OpenRocket's wind (openrocket_reference.wind_enu_mps) and compare.
    Returns (sim result, metrics dict, plots list)."""
    ref = rocket.params.get("openrocket_reference", {})
    wind = tuple(ref.get("wind_enu_mps", (0.0, 0.0, 0.0)))
    res = simulate(rocket, SimConfig(wind=wind, stop_at="apogee", sensor_noise_scale=0.0))
    s = res["summary"]
    i_ap = int(np.nanargmax(orc["alt"]))
    pairs = [("Apogee AGL [m]", s["apogee_agl_m"], orc["alt"][i_ap]),
             ("Time to apogee [s]", res["events"]["apogee"], orc["t"][i_ap])]
    if orc["vel"] is not None:
        pairs.append(("Max speed [m/s]", s["max_speed_mps"], np.nanmax(orc["vel"][:i_ap + 1])))
    if orc["mach"] is not None:
        pairs.append(("Max Mach [-]", s["max_mach"], np.nanmax(orc["mach"][:i_ap + 1])))
    if ref.get("off_rail_velocity_mps"):
        pairs.append(("Off-rail velocity [m/s]", s["off_rail_velocity_mps"], ref["off_rail_velocity_mps"]))
    m = {}
    for k, a, b in pairs:
        m[f"OR check: {k} sim"] = round(float(a), 3)
        m[f"OR check: {k} OpenRocket"] = round(float(b), 3)
        m[f"OR check: {k} diff [%]"] = round(100 * (float(a) - float(b)) / float(b), 2)
    t_or = orc["t"][:i_ap + 1]
    step = max(1, int(round(0.04 / res["cfg"].dt)))
    tt = res["t"][::step]
    plots = [{"id": "or_altitude", "title": f"OpenRocket cross-check: altitude (wind {wind[0]:+.1f} m/s E)",
              "xlabel": "t [s]", "ylabel": "m AGL",
              "series": [_series("6-DOF sim", tt, res["x"][::step, 2], 2),
                         _series("OpenRocket", t_or, orc["alt"][:i_ap + 1], 2)]}]
    if orc["vel"] is not None:
        plots.append({"id": "or_speed", "title": "OpenRocket cross-check: speed", "xlabel": "t [s]", "ylabel": "m/s",
                      "series": [_series("6-DOF sim", tt, np.linalg.norm(res["x"][::step, 3:6], axis=1), 2),
                                 _series("OpenRocket", t_or, orc["vel"][:i_ap + 1], 2)]})
    return res, m, plots


def figure_openrocket(res, orc):
    i_ap = int(np.nanargmax(orc["alt"]))
    sl = slice(0, i_ap + 1)
    t = res["t"]
    panels = [("Altitude", "altitude AGL [m]", res["x"][:, 2], orc["alt"]),
              ("Speed", "|v| [m/s]", np.linalg.norm(res["x"][:, 3:6], axis=1), orc["vel"]),
              ("Mach number", "Mach", res["mach"], orc["mach"]),
              ("Static stability margin", "cal", res["stability_cal"], orc["stab"])]
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    for ax, (ti, yl, ys, yo) in zip(axs.flat, panels):
        ax.plot(t, ys, c=COLORS["nominal"], lw=1.5, label="6-DOF sim (this work)")
        if yo is not None:
            ax.plot(orc["t"][sl], yo[sl], c=COLORS["openrocket"], lw=1.5, ls="--", label="OpenRocket")
        ax.set_title(ti, fontsize=11)
        ax.set_xlabel("t [s]")
        ax.set_ylabel(yl)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axs[0, 0].legend(fontsize=8)
    axs[1, 1].set_ylim(0, 5)
    fig.suptitle("Cross-validation vs OpenRocket (ascent, same wind as the OpenRocket run)", fontsize=12)
    fig.savefig(FIG / "passive_openrocket.png", dpi=150)
    plt.close(fig)


def variant_json(vid, label, res, extra_plots=(), extra_metrics=None):
    t, pos, q = _resample(res)
    tt, ev = res["t"], res["events"]
    asc = tt <= ev.get("apogee", tt[-1])
    step = max(1, int(round(0.04 / res["cfg"].dt)))   # ~25 Hz from the 1 ms ascent log
    ia = np.nonzero(asc)[0][::step]
    speed = np.linalg.norm(res["x"][:, 3:6], axis=1)
    plots = [
        {"id": "altitude", "title": "Altitude", "xlabel": "t [s]", "ylabel": "m AGL",
         "series": [_series("altitude", t, pos[:, 2], 2)]},
        {"id": "speed", "title": "Speed", "xlabel": "t [s]", "ylabel": "m/s",
         "series": [_series("speed", t, np.interp(t, tt, speed), 2)]},
        {"id": "mach", "title": "Mach number", "xlabel": "t [s]", "ylabel": "-",
         "series": [_series("Mach", t, np.interp(t, tt, res["mach"]), 4)]},
        {"id": "aoa", "title": "Total angle of attack (ascent)", "xlabel": "t [s]", "ylabel": "deg",
         "series": [_series("AoA", tt[ia], np.degrees(res["alpha"][ia]), 3)]},
        {"id": "stability", "title": "Static stability margin (ascent)", "xlabel": "t [s]", "ylabel": "cal",
         "series": [_series("(x_cp - x_cg)/d", tt[ia], res["stability_cal"][ia], 3)]},
        {"id": "profile", "title": "Altitude vs East (downrange)", "xlabel": "East [m]", "ylabel": "m AGL",
         "series": [{"name": "trajectory", "x": _r(pos[:, 0], 2), "y": _r(pos[:, 2], 2), "mode": "lines"}]},
    ]
    plots += list(extra_plots)
    return {"id": vid, "label": label, "t": _r(t, 3),
            "rocket": {"pos": [[round(float(v), 2) for v in p] for p in pos],
                       "quat": [[round(float(v), 5) for v in qq] for qq in q]},
            "objects": [],
            "events": [{"t": round(float(v), 3), "label": LABELS.get(k, k)} for k, v in sorted(ev.items(),
                                                                                            key=lambda kv: kv[1])],
            "plots": plots, "metrics": {**metrics(res), **(extra_metrics or {})}}


def _mark_events(ax, ev, tmax=None):
    for k, v in ev.items():
        if k in ("liftoff", "drogue") or (tmax is not None and v > tmax):
            continue
        ax.axvline(v, color="0.6", lw=0.8, ls=":")
        ax.text(v, 1.0, " " + LABELS[k], transform=ax.get_xaxis_transform(), rotation=90, va="top",
                ha="left", fontsize=7, color="0.35")


def figure_2d(runs):
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    ev0 = runs[0][2]["events"]
    t_asc = ev0["apogee"] + 3
    for vid, label, res in runs:
        c, t, X = COLORS[vid], res["t"], res["x"]
        asc = t <= res["events"]["apogee"]
        sp = np.linalg.norm(X[:, 3:6], axis=1)
        axs[0, 0].plot(t, X[:, 2], c=c, lw=1.5, label=label)
        axs[0, 1].plot(t[t <= t_asc], sp[t <= t_asc], c=c, lw=1.5, label=label)
        axs[0, 2].plot(t[t <= t_asc], np.linalg.norm(res["accel"], axis=1)[t <= t_asc] / 9.80665, c=c, lw=1.5,
                       label=label)
        axs[1, 0].plot(X[:, 0], X[:, 2], c=c, lw=1.5, label=label)
        for k in ("rail_exit", "burnout", "apogee", "main", "landing"):
            if k in res["events"]:
                i = min(np.searchsorted(t, res["events"][k]), len(t) - 1)
                axs[1, 0].plot(X[i, 0], X[i, 2], "o", ms=6, mfc=c, mec="white", mew=1.5)
                if vid == "nominal":
                    axs[1, 0].annotate(LABELS[k], (X[i, 0], X[i, 2]), xytext=(5, 3), textcoords="offset points",
                                       fontsize=7, color="0.35")
        axs[1, 1].plot(t[asc], np.degrees(res["alpha"][asc]), c=c, lw=1.2, label=label)
        axs[1, 2].plot(t[asc], res["mach"][asc], c=c, lw=1.5, label=label)
    titles = [("Altitude", "t [s]", "altitude AGL [m]"), ("Speed (ascent)", "t [s]", "|v| [m/s]"),
              ("Acceleration magnitude (ascent)", "t [s]", "|a| [g]"),
              ("Flight profile", "East (downrange) [m]", "altitude AGL [m]"),
              ("Total angle of attack (ascent)", "t [s]", "AoA [deg]"), ("Mach number (ascent)", "t [s]", "Mach")]
    for ax, (ti, xl, yl) in zip(axs.flat, titles):
        ax.set_title(ti, fontsize=11)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in (axs[0, 1], axs[0, 2], axs[1, 1], axs[1, 2]):
        _mark_events(ax, ev0, t_asc)
    _mark_events(axs[0, 0], ev0)
    axs[0, 0].legend(fontsize=8, loc="upper right")
    axs[1, 1].set_ylim(0, min(10, axs[1, 1].get_ylim()[1]))
    fig.suptitle("Passive (uncontrolled, fin-stabilised) flight, Midland TX: events from the no-wind run",
                 fontsize=12)
    fig.savefig(FIG / "passive_2d.png", dpi=150)
    plt.close(fig)


def figure_3d(runs):
    fig = plt.figure(figsize=(9, 8), constrained_layout=True)
    ax = fig.add_subplot(projection="3d")
    for vid, label, res in runs:
        c, t, X = COLORS[vid], res["t"], res["x"]
        ax.plot(X[:, 0], X[:, 1], X[:, 2], c=c, lw=1.5, label=label)
        ax.plot(X[:, 0], X[:, 1], 0 * X[:, 2], c=c, lw=0.8, alpha=0.35)
        for k in ("burnout", "apogee", "main", "landing"):
            if k in res["events"]:
                i = min(np.searchsorted(t, res["events"][k]), len(t) - 1)
                ax.scatter(*X[i, 0:3], color=c, s=30, edgecolor="white")
                if vid == "nominal":
                    ax.text(*X[i, 0:3], " " + LABELS[k], fontsize=8, color="0.3")
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_zlabel("Up [m]")
    ax.set_title("Passive flight: 3D trajectory (ground track shaded)")
    ax.legend(fontsize=8, loc="upper left")
    fig.savefig(FIG / "passive_3d.png", dpi=150)
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    rocket = Rocket.load()
    orc = load_openrocket_csv()
    runs = []
    for vid, label, cfg in VARIANTS:
        t0 = time.perf_counter()
        res = simulate(rocket, cfg)
        print(f"[{vid}] {time.perf_counter() - t0:.2f} s wall, apogee {res['summary']['apogee_agl_m']:.1f} m AGL, "
              f"events { {k: round(v, 2) for k, v in res['events'].items()} }")
        runs.append((vid, label, res))
    figure_2d(runs)
    figure_3d(runs)
    src = "PLACEHOLDER parameters (data/rocket_params.placeholder.json)" if rocket.is_placeholder \
        else "data/rocket_params.json (OpenRocket-derived)"
    notes = f"Rocket: {rocket.name}. Parameters: {src}. Motor: {rocket.motor_header['name']} " \
            f"({rocket.total_impulse:.0f} Ns)."
    or_plots, or_metrics = [], {}
    if orc is not None:
        res_or, or_metrics, or_plots = openrocket_comparison(rocket, orc)
        figure_openrocket(res_or, orc)
        notes += " 'OR check' metrics/plots (nominal variant) re-fly the ascent with the OpenRocket run's wind and " \
                 "compare against data/openrocket_flight.csv."
        if rocket.is_placeholder:
            notes += " WARNING: OpenRocket data is for the real design, the sim used placeholder parameters."
    out = {
        "id": "passive", "group": "Uncontrolled flight", "title": "Passive fin-stabilised flight",
        "description": "Uncontrolled flight of the TVC rocket with the gimbal locked at 0 deg, launched from an "
                       "18 ft rail at 6 deg towards East from Midland, TX, simulated with the 6-DOF nonlinear "
                       "model (ISA atmosphere, Mach-dependent aerodynamics, variable mass, rail phase) through "
                       "apogee, drogue descent and main-chute descent from 300 m AGL.\n\n"
                       "The 'wind' variant adds a 6 m/s wind from the East at 10 m with 1/7 power-law shear, "
                       "Gauss-Markov turbulence (1.5 m/s) and a 5 m/s 1-cos side gust at t = 3 s: the "
                       "fin-stabilised rocket weathercocks into the wind and then drifts downwind under canopy.",
        "variants": [variant_json(vid, label, res, *((or_plots, or_metrics) if vid == "nominal" else ()))
                     for vid, label, res in runs],
        "notes": notes,
    }
    (RES / "passive.json").write_text(json.dumps(out, separators=(",", ":")))
    if orc is not None:
        print("OpenRocket comparison:", or_metrics)
    else:
        print("data/openrocket_flight.csv not found: OpenRocket comparison skipped (runs automatically once present)")
    print("wrote figures/passive_2d.png, figures/passive_3d.png, results/passive.json"
          + (", figures/passive_openrocket.png" if orc is not None else ""))


if __name__ == "__main__":
    main()
