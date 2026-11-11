"""Checks on the rocket-designer outputs (OpenRocket model, motor, extracted parameters, reference flight)."""
import csv
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
P = json.loads((ROOT / "data" / "rocket_params.json").read_text())
ORK = ROOT / "openrocket" / "tvc_sounding_rocket.ork"
JAR = ROOT / "openrocket" / "bin" / "OpenRocket-23.09.jar"

SCHEMA = {
    "name": None, "length_m": None, "diameter_m": None, "reference_area_m2": None,
    "motor": ["designation", "manufacturer", "eng_file", "total_impulse_Ns", "burn_time_s", "propellant_mass_kg",
              "motor_total_mass_kg"],
    "mass": ["dry_kg", "wet_kg", "cg_dry_m", "cg_wet_m", "Ixx_dry", "Ixx_wet", "Iyy_dry", "Iyy_wet"],
    "gimbal_station_m": None,
    "aero": ["mach", "cd_power_off", "cd_power_on", "cn_alpha", "cp_m", "cmq", "clp"],
    "payload": ["mass_kg", "station_m"],
    "launch": ["lat_deg", "lon_deg", "elevation_m", "rail_length_m", "rail_angle_deg", "rail_azimuth_deg"],
    "openrocket_reference": ["apogee_m_agl", "max_velocity_mps", "max_mach", "off_rail_velocity_mps",
                             "stability_cal_off_rail", "time_to_apogee_s"],
}


def read_eng(path):
    lines = [ln.split() for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith(";")]
    header, pts = lines[0], np.array([[float(a), float(b)] for a, b in lines[1:]])
    return header, np.vstack([[0.0, 0.0], pts])


def test_schema_keys():
    for k, sub in SCHEMA.items():
        assert k in P, k
        for s in sub or []:
            assert s in P[k], f"{k}.{s}"
    placeholder = ROOT / "data" / "rocket_params.placeholder.json"
    if placeholder.exists():  # same schema as sim-core's stand-in
        ph = json.loads(placeholder.read_text())
        for k, v in ph.items():
            assert k in P, k
            if isinstance(v, dict):
                assert set(v) <= set(P[k]), (k, set(v) - set(P[k]))


def test_geometry_and_mass_sane():
    m, d = P["mass"], P["diameter_m"]
    assert 2.0 < P["length_m"] < 4.0 and 0.1 < d < 0.2
    assert P["reference_area_m2"] == pytest.approx(np.pi * d ** 2 / 4, rel=1e-6)
    assert m["wet_kg"] > m["dry_kg"] > 10
    assert m["wet_kg"] - m["dry_kg"] == pytest.approx(P["motor"]["propellant_mass_kg"], rel=1e-3)
    assert 0 < m["cg_dry_m"] < m["cg_wet_m"] < P["length_m"]  # propellant is aft of the dry CG
    for s in ("dry", "wet"):
        assert 0 < m[f"Ixx_{s}"] < m[f"Iyy_{s}"]
    assert m["cg_wet_m"] < P["gimbal_station_m"] < P["length_m"]
    assert P["payload"]["mass_kg"] == pytest.approx(4.0)
    assert P["payload"]["station_m"] < 0.4 * P["length_m"]  # just below the nose cone
    la = P["launch"]
    assert la["rail_length_m"] == pytest.approx(5.4864) and la["rail_angle_deg"] == pytest.approx(6.0)


def test_aero_tables():
    a = P["aero"]
    mach = np.array(a["mach"])
    assert mach[0] == 0.0 and mach[-1] >= max(1.5, P["openrocket_reference"]["max_mach"])
    assert np.all(np.diff(mach) > 0)
    for k in ("cd_power_off", "cd_power_on", "cn_alpha", "cp_m"):
        assert len(a[k]) == len(mach), k
        assert np.all(np.isfinite(a[k])), k
    off, on = np.array(a["cd_power_off"]), np.array(a["cd_power_on"])
    assert np.all((off > 0.2) & (off < 1.5)) and np.all((on > 0.1) & (on <= off))
    assert np.all(np.array(a["cn_alpha"]) > 2.0)
    cp = np.array(a["cp_m"])
    assert np.all((cp > 0) & (cp < P["length_m"]))
    # CP aft of CG (wet and dry) at every Mach the rocket flies
    flown = mach <= P["openrocket_reference"]["max_mach"] + 0.05
    assert np.all(cp[flown] > P["mass"]["cg_wet_m"] + P["diameter_m"])
    assert a["cmq"] < 0 and a["clp"] < 0


def test_requirements_met():
    r = P["openrocket_reference"]
    assert 2950 <= r["apogee_m_agl"] <= 3140  # 10,000 ft AGL +-3 %
    assert r["off_rail_velocity_mps"] >= 30.5  # 100 ft/s off an 18 ft rail
    assert 2.0 <= r["stability_cal_off_rail"] <= 3.0
    assert r.get("stability_cal_min_ascent", 1.5) >= 1.5
    assert r.get("stability_cal_max_ascent", 4.0) <= 4.0
    assert P["motor"]["burn_time_s"] >= 4.0
    assert P["motor"]["designation"].split()[-1].startswith("M")
    assert 5120 < P["motor"]["total_impulse_Ns"] <= 10240  # M class
    assert 0.5 < r["max_mach"] < 1.2 and r["time_to_apogee_s"] > 15


def test_eng_integrates_to_total_impulse():
    path = ROOT / P["motor"]["eng_file"]
    header, pts = read_eng(path)
    assert len(header) == 7 and header[0] in P["motor"]["designation"]
    assert float(header[4]) == pytest.approx(P["motor"]["propellant_mass_kg"], rel=1e-3)
    assert float(header[5]) == pytest.approx(P["motor"]["motor_total_mass_kg"], rel=1e-3)
    assert np.all(np.diff(pts[:, 0]) > 0) and np.all(pts[:, 1] >= 0) and pts[-1, 1] == 0.0
    impulse = np.trapezoid(pts[:, 1], pts[:, 0])
    assert impulse == pytest.approx(P["motor"]["total_impulse_Ns"], rel=0.05)


def test_flight_csv_consistent():
    with open(ROOT / "data" / "openrocket_flight.csv") as f:
        rows = list(csv.DictReader(f))
    t = np.array([float(r["time_s"]) for r in rows])
    h = np.array([float(r["altitude_m_agl"]) for r in rows])
    assert np.all(np.diff(t) > 0)
    assert h.max() == pytest.approx(P["openrocket_reference"]["apogee_m_agl"], rel=1e-3)
    for col in ("pos_east_m", "pos_north_m", "vel_vertical_mps", "vel_total_mps", "acc_total_mps2", "mach",
                "aoa_rad", "stability_cal", "cg_m", "cp_m", "thrust_N", "mass_kg"):
        assert col in rows[0], col
    m = np.array([float(r["mass_kg"]) for r in rows])
    assert m[0] == pytest.approx(P["mass"]["wet_kg"], rel=1e-3) and m[-1] == pytest.approx(P["mass"]["dry_kg"], rel=1e-3)


def test_ork_file_and_figures():
    with zipfile.ZipFile(ORK) as z:
        xml = z.read(z.namelist()[0]).decode()
    assert "M1845NT" in xml and "<simulation" in xml and "<launchrodlength>5.4864" in xml
    for n in ("profile", "trajectory", "stability"):
        assert (ROOT / "figures" / f"openrocket_{n}.png").stat().st_size > 10_000


@pytest.mark.skipif(not JAR.exists(), reason="OpenRocket jar not downloaded (openrocket/setup.sh)")
def test_ork_reloads_and_reproduces_apogee():
    code = ("import sys, json; sys.path.insert(0, 'openrocket'); import orkit; "
            f"s = orkit.load_and_run({str(ORK)!r}); print(json.dumps(s['apogee_m_agl']))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert float(out.stdout.strip().splitlines()[-1]) == pytest.approx(P["openrocket_reference"]["apogee_m_agl"],
                                                                      rel=1e-3)
