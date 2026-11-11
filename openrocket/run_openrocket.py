"""Build the TVC sounding rocket in OpenRocket 23.09 (headless), run the Midland reference simulation and export
everything the Python 6-DOF needs.

Outputs: openrocket/tvc_sounding_rocket.ork, data/motor.eng, data/rocket_params.json, data/openrocket_flight.csv,
figures/openrocket_{profile,trajectory,stability}.png

Usage: .venv/bin/python openrocket/run_openrocket.py      (run openrocket/setup.sh first to fetch the jar)
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import orkit  # noqa: E402

ROOT = orkit.HERE.parent
MOTOR = ("AeroTech", "M1845NT")
NOZZLE_EXIT_D = 0.065   # m, estimated RMS-98 nozzle exit diameter (for the power-on base-drag correction)
THROAT_FROM_AFT = 0.05  # m, nozzle throat ahead of the nozzle exit plane (motor aft end = airframe aft end)
MACH_GRID = np.round(np.arange(0.0, 1.501, 0.05), 3)


def aero_tables(rocket, fcid, cg_ref):
    from net.sf.openrocket.aerodynamics import BarrowmanCalculator, FlightConditions
    from net.sf.openrocket.logging import WarningSet
    from net.sf.openrocket.models.atmosphere import ExtendedISAModel
    from net.sf.openrocket.rocketcomponent import ComponentAssembly, FinSet
    cfg = rocket.getFlightConfiguration(fcid)
    calc = BarrowmanCalculator()
    # conditions at mid-ascent altitude (launch site + ~1500 m) for the Reynolds number
    atm = ExtendedISAModel().getConditions(orkit.LAUNCH["elevation_m"] + 1500.0)
    d = float(cfg.getReferenceLength())
    A_base_frac = (NOZZLE_EXIT_D / d) ** 2
    out = {k: [] for k in ("cd_power_off", "cd_power_on", "cn_alpha", "cp_m", "cd_base", "cmq", "clp", "clp_or")}
    # chord-weighted mean-square fin radius  y_eff^2 = int c(y) y^2 dy / int c(y) dy  (y from the rocket axis)
    g = orkit.DESIGN
    y = np.linspace(orkit.R, orkit.R + g["fin_span"], 2001)
    ch = g["fin_root"] + (g["fin_tip"] - g["fin_root"]) * (y - orkit.R) / g["fin_span"]
    y_eff2 = np.trapezoid(ch * y ** 2, y) / np.trapezoid(ch, y)

    def cond(mach, aoa=0.0):
        fc = FlightConditions(cfg)
        fc.setAtmosphericConditions(atm)
        fc.setAOA(aoa)
        fc.setMach(max(mach, 0.02))  # M=0 row evaluated at M=0.02 (friction model needs V>0)
        return fc

    for M in MACH_GRID:
        fc = cond(M)
        f = calc.getAerodynamicForces(cfg, fc, WarningSet())
        cd, cdb = float(f.getCD()), float(f.getBaseCD())
        out["cd_power_off"].append(cd)
        out["cd_base"].append(cdb)
        # OpenRocket 23.09 applies base drag even while thrusting; while the motor burns the jet fills the nozzle
        # exit area, so remove the base drag acting on that fraction of the base.
        out["cd_power_on"].append(cd - cdb * A_base_frac)
        out["cn_alpha"].append(float(f.getCNa()))
        out["cp_m"].append(float(f.getCP().x))
        # Pitch damping, standard strip formula: each component's normal force sees dalpha = q (x_i - x_cg)/V
        #   Cm_q = -2 * sum_i CNa_i ((x_i - x_cg)/d)^2      (per unit q*d/(2V))
        fa = calc.getForceAnalysis(cfg, fc, WarningSet())
        cmq = 0.0
        cna_fins = 0.0
        for comp, fo in fa.items():
            if isinstance(comp, FinSet):
                cna_fins += float(fo.getCNa())
            if comp is None or isinstance(comp, ComponentAssembly):
                continue  # rocket/stage entries hold totals; sum the individual components only
            cna = float(fo.getCNa())
            if cna != 0.0 and not math.isnan(cna):
                cmq -= 2.0 * cna * ((float(fo.getCP().x) - cg_ref) / d) ** 2
        out["cmq"].append(cmq)
        # Roll damping, same strip theory: a fin strip at radius y sees dalpha = p y / V. With the per-fin normal-force
        # slope from OpenRocket (4 fins: CNa_1fin = CNa_finset / 2, incl. body interference and Mach effects):
        #   Cl_p = -2 N CNa_1fin y_eff^2 / d^2 = -4 CNa_finset y_eff^2 / d^2      (per unit p*d/(2V))
        n_fin = g["fin_count"]
        out["clp"].append(-2.0 * n_fin * (cna_fins / (n_fin / 2.0)) * y_eff2 / d ** 2)
        # OpenRocket's own roll-damping model, for comparison only (it uses the 2-D slope 2*pi/beta -> ~3x larger)
        fc.setRollRate(1.0)
        fr = calc.getAerodynamicForces(cfg, fc, WarningSet())
        out["clp_or"].append(-float(fr.getCrollDamp()) / (1.0 * d / (2.0 * float(fc.getVelocity()))))
    return {k: np.array(v) for k, v in out.items()}


def static_margin(rocket, fcid, mach, cg):
    from net.sf.openrocket.aerodynamics import BarrowmanCalculator, FlightConditions
    from net.sf.openrocket.logging import WarningSet
    cfg = rocket.getFlightConfiguration(fcid)
    fc = FlightConditions(cfg)
    fc.setAOA(0.0)
    fc.setMach(mach)
    cp = float(BarrowmanCalculator().getCP(cfg, fc, WarningSet()).x)
    return (cp - cg) / float(cfg.getReferenceLength())


def write_eng(motor, path):
    t = np.array(list(motor.getTimePoints()))
    F = np.array(list(motor.getThrustPoints()))
    m_tot = float(motor.getLaunchMass())
    m_prop = m_tot - float(motor.getBurnoutMass())
    keep = t > 0  # RASP files have no t=0 point
    lines = [f"; {motor.getManufacturer().getDisplayName()} {motor.getDesignation()} ({motor.getCaseInfo()}, "
             f"{motor.getPropellantInfo()}) - thrust curve from the OpenRocket 23.09 motor database "
             f"(thrustcurve.org manufacturer data, digest {motor.getDigest()})",
             f"{motor.getDesignation()} {motor.getDiameter() * 1000:.0f} {motor.getLength() * 1000:.0f} P "
             f"{m_prop:.4f} {m_tot:.4f} AT"]
    lines += [f"   {ti:.3f} {Fi:.3f}" for ti, Fi in zip(t[keep], F[keep])]
    path.write_text("\n".join(lines) + "\n")
    return float(np.trapezoid(F, t)), float(t[-1]), m_prop, m_tot


CSV_COLS = [  # (OpenRocket type, column name)
    ("TYPE_TIME", "time_s"), ("TYPE_ALTITUDE", "altitude_m_agl"),
    ("TYPE_POSITION_X", "pos_east_m"), ("TYPE_POSITION_Y", "pos_north_m"),
    ("TYPE_VELOCITY_Z", "vel_vertical_mps"), ("TYPE_VELOCITY_XY", "vel_lateral_mps"),
    ("TYPE_VELOCITY_TOTAL", "vel_total_mps"), ("TYPE_ACCELERATION_Z", "acc_vertical_mps2"),
    ("TYPE_ACCELERATION_TOTAL", "acc_total_mps2"), ("TYPE_MACH_NUMBER", "mach"), ("TYPE_AOA", "aoa_rad"),
    ("TYPE_ORIENTATION_THETA", "zenith_rad"), ("TYPE_ORIENTATION_PHI", "azimuth_rad"),
    ("TYPE_PITCH_RATE", "pitch_rate_radps"), ("TYPE_ROLL_RATE", "roll_rate_radps"),
    ("TYPE_STABILITY", "stability_cal"), ("TYPE_CG_LOCATION", "cg_m"), ("TYPE_CP_LOCATION", "cp_m"),
    ("TYPE_THRUST_FORCE", "thrust_N"), ("TYPE_DRAG_FORCE", "drag_N"), ("TYPE_DRAG_COEFF", "cd"),
    ("TYPE_BASE_DRAG_COEFF", "cd_base"), ("TYPE_MASS", "mass_kg"),
    ("TYPE_LONGITUDINAL_INERTIA", "I_long_kgm2"), ("TYPE_ROTATIONAL_INERTIA", "I_roll_kgm2"),
    ("TYPE_AIR_PRESSURE", "pressure_Pa"), ("TYPE_AIR_TEMPERATURE", "temperature_K"),
    ("TYPE_WIND_VELOCITY", "wind_mps"),
]


def branch_array(b):
    from net.sf.openrocket.simulation import FlightDataType as T
    cols = []
    for tname, _ in CSV_COLS:
        cols.append([float("nan") if v is None else float(v) for v in b.get(getattr(T, tname))])
    return np.array(cols).T


def plots(data, events, path_prefix, calm=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = {n: i for i, (_, n) in enumerate(CSV_COLS)}
    t = data[:, c["time_s"]]
    ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
    s1, s2, s3 = "#2a78d6", "#eb6834", "#1baf7a"
    plt.rcParams.update({"axes.edgecolor": ink2, "axes.labelcolor": ink, "xtick.color": ink2, "ytick.color": ink2,
                         "axes.grid": True, "grid.color": grid, "grid.linewidth": 0.8, "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": 10, "lines.linewidth": 2})
    ev_style = [("rail_exit", "Rail exit"), ("burnout", "Burnout"), ("apogee", "Apogee / drogue"),
                ("main", "Main deploy")]

    def mark(ax, label=True, tmax=np.inf, only=None):
        for k, lab in ev_style:
            if events[k] > tmax:
                continue
            ax.axvline(events[k], color=ink2, lw=0.9, ls="--")
            if label and (only is None or k in only):
                ax.annotate(lab, (events[k], 1.0), xycoords=("data", "axes fraction"), rotation=90, va="top",
                            ha="right", fontsize=8, color=ink2)

    # profile: ascent (left, linear time to apogee + 5 s) and full flight (right)
    fig, axs = plt.subplots(3, 2, figsize=(12, 9), sharex="col", gridspec_kw={"width_ratios": [1.3, 1]})
    series = [("altitude_m_agl", "Altitude AGL [m]", s1), ("vel_total_mps", "Velocity [m/s]", s1),
              ("acc_total_mps2", "Acceleration [m/s²]", s1)]
    for col, tmax in ((0, events["apogee"] + 5), (1, t[-1])):
        m = t <= tmax
        for r, (k, lab, colr) in enumerate(series):
            ax = axs[r, col]
            ax.plot(t[m], data[m, c[k]], color=colr, label="total" if r else None)
            if r == 1:
                ax.plot(t[m], data[m, c["vel_vertical_mps"]], color=s2, label="vertical")
                ax.legend(frameon=False, loc="center right" if col else "upper right")
            if r == 2:
                ax.plot(t[m], data[m, c["acc_vertical_mps2"]], color=s2, label="vertical")
                ax.legend(frameon=False, loc="upper right")
            ax.set_ylabel(lab)
            mark(ax, label=(r == 0), tmax=tmax, only=None if col == 0 else ("main",))
        axs[2, col].set_xlabel("Time [s]")
    axs[0, 0].set_title("Ascent", color=ink, loc="left")
    axs[0, 1].set_title("Full flight incl. recovery", color=ink, loc="left")
    fig.suptitle("OpenRocket reference flight - passive, fin-stabilised (AeroTech M1845NT, Midland TX, 4 m/s wind)",
                 color=ink)
    fig.tight_layout()
    fig.savefig(f"{path_prefix}_profile.png", dpi=150)
    plt.close(fig)

    # trajectory: altitude vs downrange (east)
    fig, ax = plt.subplots(figsize=(8, 6))
    x, z = data[:, c["pos_east_m"]], data[:, c["altitude_m_agl"]]
    asc = t <= events["apogee"]
    ax.plot(x[asc], z[asc], color=s1, label="ascent")
    ax.plot(x[~asc], z[~asc], color=s2, label="descent (drogue, then main)")
    for k, lab in ev_style:
        i = int(np.argmin(abs(t - events[k])))
        ax.plot(x[i], z[i], "o", ms=8, color=ink, mec="white", mew=2)
        ax.annotate(lab, (x[i], z[i]), xytext=(6, 4), textcoords="offset points", fontsize=8, color=ink2)
    ax.set_xlabel("Downrange, east of pad [m]")
    ax.set_ylabel("Altitude AGL [m]")
    ax.set_title("Trajectory (rail 6° toward east, wind 4 m/s from east)", color=ink, loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"{path_prefix}_trajectory.png", dpi=150)
    plt.close(fig)

    # stability: margin and CG/CP, rail exit to apogee
    m = (t >= events["rail_exit"]) & (t <= events["apogee"] - 1.0)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    a1.plot(t[m], data[m, c["stability_cal"]], color=s1, label="4 m/s wind (reference)")
    if calm is not None:
        tc = calm[:, c["time_s"]]
        mc = (tc >= events["rail_exit"]) & (tc <= events["apogee"] - 1.0)
        a1.plot(tc[mc], calm[mc, c["stability_cal"]], color=s2, label="zero wind (AoA ~ 0)")
    a1.legend(frameon=False, loc="lower right")
    a1.axhspan(1.5, 4.0, color=s3, alpha=0.10, lw=0)
    a1.set_ylabel("Stability margin [cal]")
    a1.set_ylim(0, 5)
    a1.annotate("design band 1.5-4 cal", (t[m][-1], 4.0), ha="right", va="bottom", fontsize=8, color=ink2)
    mark(a1, tmax=events["apogee"])
    a2.plot(t[m], data[m, c["cg_m"]], color=s1, label="CG")
    a2.plot(t[m], data[m, c["cp_m"]], color=s2, label="CP")
    a2.set_ylabel("Station from nose tip [m]")
    a2.set_xlabel("Time [s]")
    a2.invert_yaxis()
    a2.legend(frameon=False)
    mark(a2, label=False, tmax=events["apogee"])
    a1.set_title("Passive stability margin (OpenRocket); CG/CP below for the reference run", color=ink, loc="left")
    fig.tight_layout()
    fig.savefig(f"{path_prefix}_stability.png", dpi=150)
    plt.close(fig)


def main():
    orkit.start()
    from net.sf.openrocket.masscalc import MassCalculator
    from net.sf.openrocket.simulation import FlightDataType as T
    from net.sf.openrocket.file import GeneralRocketSaver
    from net.sf.openrocket.document import StorageOptions
    from java.io import File

    motor = orkit.find_motor(*MOTOR)
    doc, rocket, mount, fcid = orkit.build(motor)
    cfg = rocket.getFlightConfiguration(fcid)

    # ---- simulations: reference (4 m/s) and calm; both stored in the .ork
    sim = orkit.make_sim(doc, rocket, fcid)
    sim0 = orkit.make_sim(doc, rocket, fcid, wind=0.0, name="Midland TX - zero wind")
    doc.addSimulation(sim)
    doc.addSimulation(sim0)
    b, ref = orkit.run(sim)
    b0, calm = orkit.run(sim0)
    ork = orkit.HERE / "tvc_sounding_rocket.ork"
    opts = StorageOptions()
    opts.setSaveSimulationData(True)  # so the GUI opens with the results already plotted
    GeneralRocketSaver().save(File(str(ork)), doc, opts)

    # ---- motor
    I_tot, t_end, m_prop, m_motor = write_eng(motor, ROOT / "data" / "motor.eng")
    burn = float(motor.getBurnTimeEstimate())

    # ---- mass properties (OpenRocket mass calculator; inertias about the CG)
    L, B = MassCalculator.calculateLaunch(cfg), MassCalculator.calculateBurnout(cfg)
    cg_wet, cg_dry = float(L.getCM().x), float(B.getCM().x)

    # ---- flight data
    data = branch_array(b)
    c = {n: i for i, (_, n) in enumerate(CSV_COLS)}
    hdr = ",".join(n for _, n in CSV_COLS)
    np.savetxt(ROOT / "data" / "openrocket_flight.csv", data, delimiter=",", header=hdr, comments="", fmt="%.6g")
    evs = {}
    for e in b.getEvents():
        evs.setdefault(str(e.getType().name()), []).append(float(e.getTime()))
    events = dict(rail_exit=evs["LAUNCHROD"][0], burnout=evs["BURNOUT"][0], apogee=evs["APOGEE"][0],
                  main=max(evs["RECOVERY_DEVICE_DEPLOYMENT"]), ground=evs["GROUND_HIT"][0])
    t = data[:, c["time_s"]]
    i_rail = int(np.argmin(abs(t - events["rail_exit"])))
    sm_rail = static_margin(rocket, fcid, data[i_rail, c["mach"]], data[i_rail, c["cg_m"]])
    asc = (t >= events["rail_exit"]) & (t <= events["apogee"] - 1.0)
    stab_asc = data[asc, c["stability_cal"]]
    stab_asc = stab_asc[~np.isnan(stab_asc)]
    max_acc = float(np.nanmax(data[t <= events["burnout"], c["acc_total_mps2"]]))
    plots(data, events, str(ROOT / "figures" / "openrocket"), calm=branch_array(b0))

    # ---- aero
    aero = aero_tables(rocket, fcid, 0.5 * (cg_wet + cg_dry))
    m_mask = (MACH_GRID >= 0.1) & (MACH_GRID <= 0.9)  # flight regime for the scalar damping derivatives
    cmq = float(np.mean(aero["cmq"][m_mask]))
    clp = float(np.mean(aero["clp"][m_mask]))

    payload = next(x for x in rocket.iterator(False) if str(x.getName()).startswith("Payload ("))
    payload_station = float(payload.getComponentLocations()[0].x + payload.getComponentCG().x)
    length = float(rocket.getLength())
    d = float(cfg.getReferenceLength())

    params = {
        "name": "TVC sounding rocket (6 in, AeroTech M1845NT) - OpenRocket 23.09 model",
        "length_m": length, "diameter_m": d, "reference_area_m2": float(cfg.getReferenceArea()),
        "motor": {"designation": f"AeroTech {MOTOR[1]}", "manufacturer": "AeroTech", "eng_file": "data/motor.eng",
                  "total_impulse_Ns": I_tot, "burn_time_s": burn, "propellant_mass_kg": m_prop,
                  "motor_total_mass_kg": m_motor},
        "mass": {"dry_kg": float(B.getMass()), "wet_kg": float(L.getMass()), "cg_dry_m": cg_dry, "cg_wet_m": cg_wet,
                 "Ixx_dry": float(B.getIxx()), "Ixx_wet": float(L.getIxx()),
                 "Iyy_dry": float(B.getIyy()), "Iyy_wet": float(L.getIyy())},
        "gimbal_station_m": length - THROAT_FROM_AFT,
        "aero": {"mach": MACH_GRID.tolist(), "cd_power_off": aero["cd_power_off"].tolist(),
                 "cd_power_on": aero["cd_power_on"].tolist(), "cn_alpha": aero["cn_alpha"].tolist(),
                 "cp_m": aero["cp_m"].tolist(), "cmq": cmq, "clp": clp,
                 "damping_convention": "M_pitch = qbar*S*d*cmq*(q*d/(2V)); L_roll = qbar*S*d*clp*(p*d/(2V))"},
        "payload": {"mass_kg": float(payload.getComponentMass()), "station_m": payload_station},
        "launch": {k: orkit.LAUNCH[k] for k in ("lat_deg", "lon_deg", "elevation_m", "rail_length_m",
                                                "rail_angle_deg", "rail_azimuth_deg")},
        "openrocket_reference": {
            "apogee_m_agl": ref["apogee_m_agl"], "max_velocity_mps": ref["max_velocity_mps"],
            "max_mach": ref["max_mach"], "off_rail_velocity_mps": ref["off_rail_velocity_mps"],
            "stability_cal_off_rail": sm_rail, "time_to_apogee_s": ref["time_to_apogee_s"],
            # extra (non-schema) context for validating the 6-DOF against this run
            "wind_enu_mps": [round(-orkit.WIND_MPS * math.sin(math.radians(orkit.WIND_FROM_DEG)), 6),
                             round(-orkit.WIND_MPS * math.cos(math.radians(orkit.WIND_FROM_DEG)), 6), 0.0],
            "apogee_m_agl_zero_wind": calm["apogee_m_agl"],
            "burnout_time_s": events["burnout"], "max_accel_mps2": max_acc,
            "stability_cal_min_ascent": float(stab_asc.min()), "stability_cal_max_ascent": float(stab_asc.max()),
        },
    }
    (ROOT / "data" / "rocket_params.json").write_text(json.dumps(params, indent=2) + "\n")

    # ---- summary printout (copied into docs/rocket_design.md)
    print(f"ork: {ork}")
    print(f"length {length:.3f} m, d {d * 1000:.1f} mm, wet {L.getMass():.2f} kg, dry {B.getMass():.2f} kg, "
          f"CG wet/dry {cg_wet:.3f}/{cg_dry:.3f} m, Iyy wet/dry {L.getIyy():.2f}/{B.getIyy():.2f}, "
          f"Ixx wet/dry {L.getIxx():.4f}/{B.getIxx():.4f}")
    print(f"motor: I={I_tot:.0f} Ns, burn {burn:.2f} s (curve end {t_end:.3f} s), prop {m_prop:.3f} kg")
    print(f"reference: apogee {ref['apogee_m_agl']:.1f} m ({ref['apogee_m_agl'] / 0.3048:.0f} ft), "
          f"calm apogee {calm['apogee_m_agl']:.1f} m ({calm['apogee_m_agl'] / 0.3048:.0f} ft)")
    print(f"off-rail v {ref['off_rail_velocity_mps']:.2f} m/s at t={events['rail_exit']:.3f} s; static margin "
          f"{sm_rail:.2f} cal; OR stability with wind AoA min/max {stab_asc.min():.2f}/{stab_asc.max():.2f}; calm "
          f"min/max {calm['stability_min_cal']:.2f}/{calm['stability_max_cal']:.2f}")
    print(f"max v {ref['max_velocity_mps']:.1f} m/s, max Mach {ref['max_mach']:.3f}, max accel {max_acc:.1f} m/s2, "
          f"t_apogee {ref['time_to_apogee_s']:.2f} s, events {events}")
    print(f"apogee east {data[np.argmin(abs(t - events['apogee'])), c['pos_east_m']]:.0f} m, landing east "
          f"{data[-1, c['pos_east_m']]:.0f} m north {data[-1, c['pos_north_m']]:.0f} m")
    print(f"cmq {cmq:.1f} (range {aero['cmq'][m_mask].min():.1f}..{aero['cmq'][m_mask].max():.1f}), "
          f"clp {clp:.2f} (range {aero['clp'][m_mask].min():.2f}..{aero['clp'][m_mask].max():.2f}); OpenRocket's "
          f"roll damping model {np.mean(aero['clp_or'][m_mask]):.1f}")
    for M, cd, cdon, cna, cp in zip(MACH_GRID, aero["cd_power_off"], aero["cd_power_on"], aero["cn_alpha"],
                                    aero["cp_m"]):
        if round(M * 20) % 2 == 0:
            print(f"  M {M:.2f} CD {cd:.3f}/{cdon:.3f} CNa {cna:.2f} CP {cp:.3f}")


if __name__ == "__main__":
    main()
