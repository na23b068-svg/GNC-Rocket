"""Motor trade study: fly every AeroTech / Cesaroni M motor (5-10.3 kNs) in the final airframe (motor mount sized
to the motor) under the Midland reference conditions. Writes openrocket/trade_study.csv.

Usage: .venv/bin/python openrocket/trade_study.py
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import orkit  # noqa: E402

# The OpenRocket 23.09 database files this one under Cesaroni, but thrustcurve.org lists it as AMW (Tamed Tiger).
NOT_AT_CTI = {"8212M1630-P"}


def main():
    orkit.start()
    rows = []
    for m in orkit.m_motors():
        I = float(m.getTotalImpulseEstimate())
        if not 5000 < I < 10300 or str(m.getDesignation()) in NOT_AT_CTI:
            continue
        doc, rocket, mount, fcid = orkit.build(m)
        _, s = orkit.run(orkit.make_sim(doc, rocket, fcid))
        burn = float(m.getBurnTimeEstimate())
        ok_apo = 2950 <= s["apogee_m_agl"] <= 3140
        ok_rail = s["off_rail_velocity_mps"] >= 30.5
        rows.append(dict(manufacturer=str(m.getManufacturer().getSimpleName()), designation=str(m.getDesignation()),
                         diameter_mm=round(float(m.getDiameter()) * 1000), total_impulse_Ns=round(I),
                         burn_time_s=round(burn, 2), avg_thrust_N=round(I / burn), apogee_m_agl=round(s["apogee_m_agl"]),
                         off_rail_velocity_mps=round(s["off_rail_velocity_mps"], 1), max_mach=round(s["max_mach"], 2),
                         meets_apogee=ok_apo, meets_rail=ok_rail, burn_ge_4s=burn >= 4.0))
    rows.sort(key=lambda r: (not (r["meets_apogee"] and r["meets_rail"]), -r["burn_time_s"]))
    out = orkit.HERE / "trade_study.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(*r.values())


if __name__ == "__main__":
    main()
