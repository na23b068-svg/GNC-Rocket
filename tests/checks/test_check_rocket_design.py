"""Independent checker tests for the rocket-designer outputs: hand calculations that do not use OpenRocket."""
import csv
import json
import math
import re
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
P = json.loads((ROOT / "data" / "rocket_params.json").read_text())
D = P["diameter_m"]
R = D / 2
# documented geometry (docs/rocket_design.md)
NOSE_L = 0.65
FIN = dict(n=4, root=0.30, tip=0.10, span=0.155, sweep=0.20, thick=0.00635, te_fwd_of_aft=0.01)
G0 = 9.80665


def ork_xml():
    with zipfile.ZipFile(ROOT / "openrocket" / "tvc_sounding_rocket.ork") as z:
        return z.read(z.namelist()[0]).decode()


def eng_curve():
    lines = [ln for ln in (ROOT / "data" / "motor.eng").read_text().splitlines() if ln.strip() and ln[0] != ";"]
    pts = [[float(v) for v in ln.split()] for ln in lines[1:]]  # line 0 is the RASP header
    return np.vstack([[0.0, 0.0], pts]).T


def flight():
    with open(ROOT / "data" / "openrocket_flight.csv") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def barrowman():
    """Low-Mach Barrowman: (CNa_nose, x_nose, CNa_fins incl. interference, x_fins)."""
    x = np.linspace(0, NOSE_L, 20001)
    th = np.arccos(1 - 2 * x / NOSE_L)
    r = R / np.sqrt(np.pi) * np.sqrt(th - np.sin(2 * th) / 2)  # LD-Haack (Von Karman)
    xn = NOSE_L - np.trapezoid(np.pi * r ** 2, x) / (np.pi * R ** 2)
    s, cr, ct, xs = FIN["span"], FIN["root"], FIN["tip"], FIN["sweep"]
    lm = math.hypot(s, xs + ct / 2 - cr / 2)
    cnf = (1 + R / (s + R)) * 4 * FIN["n"] * (s / D) ** 2 / (1 + math.sqrt(1 + (2 * lm / (cr + ct)) ** 2))
    xle = P["length_m"] - FIN["te_fwd_of_aft"] - cr
    xf = xle + xs * (cr + 2 * ct) / (3 * (cr + ct)) + (cr + ct - cr * ct / (cr + ct)) / 6
    return 2.0, xn, cnf, xf


def isa(h):
    T = 288.15 - 0.0065 * h
    return 101325 * (T / 288.15) ** 5.25588 / (287.053 * T), math.sqrt(1.4 * 287.053 * T)


def point_mass(rail_travel=None, dt=1e-3):
    """2-D point mass, gravity turn after the rail. Returns (t, v) at rail_travel, else (t_apogee, apogee)."""
    tc, Fc = eng_curve()
    I = np.r_[0, np.cumsum(np.diff(tc) * (Fc[1:] + Fc[:-1]) / 2)]
    mw, md, S = P["mass"]["wet_kg"], P["mass"]["dry_kg"], P["reference_area_m2"]
    a = P["aero"]
    rail = rail_travel or P["launch"]["rail_length_m"]
    t = s = v = z = 0.0
    gam = math.pi / 2 - math.radians(P["launch"]["rail_angle_deg"])
    on = True
    while True:
        F = np.interp(t, tc, Fc, right=0.0)
        m = mw - (mw - md) * np.interp(t, tc, I) / I[-1]
        rho, c = isa(P["launch"]["elevation_m"] + z)
        Dr = 0.5 * rho * v * v * S * np.interp(v / c, a["mach"], a["cd_power_off"])
        acc = (F - Dr) / m - G0 * math.sin(gam)
        if on:
            v += max(acc, 0.0) * dt
            s += v * dt
            if s >= rail:
                if rail_travel is not None:
                    return t, v
                on = False
        else:
            gam += -G0 * math.cos(gam) / v * dt
            v += acc * dt
            if v * math.sin(gam) < 0:
                return t, z
        z += v * math.sin(gam) * dt
        t += dt


# ---------------------------------------------------------------- .ork, opened independently
def test_ork_motor_masses_fins_launch():
    xml = ork_xml()
    assert "<manufacturer>AeroTech</manufacturer>" in xml and "<designation>M1845NT</designation>" in xml
    # 4 kg payload block is the first component of the first body tube after the nose cone
    after_nose = xml.split("</nosecone>", 1)[1]
    first_tube = after_nose.split("</bodytube>", 1)[0]
    m = re.search(r"<masscomponent>.*?<mass>([\d.]+)</mass>\s*<masscomponenttype>(\w+)</masscomponenttype>",
                  first_tube, re.S)
    assert m and float(m.group(1)) == pytest.approx(4.0) and m.group(2) == "payload"
    types = re.findall(r"<masscomponenttype>(\w+)</masscomponenttype>", xml)
    assert "flightcomputer" in types  # avionics
    assert xml.count("<parachute>") == 2 and xml.count("<shockcord>") == 2  # recovery
    assert "<motormount>" in xml
    fins = xml.split("<trapezoidfinset>", 1)[1].split("</trapezoidfinset>", 1)[0]
    g = {k: float(re.search(f"<{k}>([\\d.]+)</{k}>", fins).group(1))
         for k in ("rootchord", "tipchord", "sweeplength", "height", "thickness", "fincount")}
    assert g == pytest.approx(dict(rootchord=FIN["root"], tipchord=FIN["tip"], sweeplength=FIN["sweep"],
                                   height=FIN["span"], thickness=FIN["thick"], fincount=FIN["n"]))
    conds = re.findall(r"<conditions>(.*?)</conditions>", xml, re.S)
    assert len(conds) >= 1
    for c in conds:
        val = lambda k: float(re.search(f"<{k}>([-\\d.]+)</{k}>", c).group(1))  # noqa: E731
        assert val("launchrodlength") == pytest.approx(5.4864)
        assert val("launchrodangle") == pytest.approx(6.0)
        assert val("launchlatitude") == pytest.approx(31.99) and val("launchlongitude") == pytest.approx(-102.08)
        assert 840 <= val("launchaltitude") <= 880


def test_fin_flutter_margin():
    """NACA TN-4197 flutter speed (G10, G = 2.6 GPa, conservative) vs the OpenRocket velocity history."""
    f = flight()
    cr, ct, s, tk = FIN["root"], FIN["tip"], FIN["span"], FIN["thick"]
    AR, lam = s ** 2 / ((cr + ct) / 2 * s), ct / cr
    a = np.sqrt(1.4 * 287.05 * f["temperature_K"])
    vf = a * np.sqrt(2.6e9 / (1.337 * AR ** 3 * f["pressure_Pa"] * (lam + 1) / (2 * (AR + 2) * (tk / cr) ** 3)))
    assert np.min(vf / np.maximum(f["vel_total_mps"], 1.0)) > 1.25


# ---------------------------------------------------------------- motor
def test_motor_curve_impulse_class_isp():
    t, F = eng_curve()
    I = np.trapezoid(F, t)
    assert I == pytest.approx(P["motor"]["total_impulse_Ns"], rel=0.01)
    assert 5120.01 <= I <= 10240
    isp = I / (P["motor"]["propellant_mass_kg"] * G0)
    assert 180 < isp < 250  # APCP
    assert 0.85 * t[-1] < P["motor"]["burn_time_s"] <= t[-1]


def test_motor_matches_thrustcurve_org():
    ua = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/130.0 Safari/537.36",
          "Content-Type": "application/json"}

    def post(ep, body):
        req = urllib.request.Request(f"https://www.thrustcurve.org/api/v1/{ep}.json", json.dumps(body).encode(), ua)
        return json.loads(urllib.request.urlopen(req, timeout=30).read())
    try:
        res = post("search", {"designation": "M1845NT", "manufacturer": "AeroTech"})["results"]
        dl = post("download", {"motorIds": [res[0]["motorId"]], "format": "RASP", "data": "samples"})["results"]
    except Exception as e:  # offline
        pytest.skip(f"thrustcurve.org unreachable: {e}")
    assert res[0]["impulseClass"] == "M" and res[0]["propWeightG"] == pytest.approx(3772)
    s = dl[0]["samples"]
    t, F = eng_curve()
    assert np.allclose(t[1:], [p["time"] for p in s], atol=1e-3)
    assert np.allclose(F[1:], [p["thrust"] for p in s], atol=1e-2)


# ---------------------------------------------------------------- aero hand calcs
def test_barrowman_cp_cna_low_mach():
    cnn, xn, cnf, xf = barrowman()
    cna, cp = cnn + cnf, (cnn * xn + cnf * xf) / (cnn + cnf)
    a = P["aero"]
    assert np.interp(0.1, a["mach"], a["cn_alpha"]) == pytest.approx(cna, rel=0.10)
    assert np.interp(0.1, a["mach"], a["cp_m"]) == pytest.approx(cp, abs=0.05)  # 5 cm ~ 0.3 cal


def test_static_margin_rail_exit_and_ascent():
    f = flight()
    t_rail = 0.374
    cg_rail = np.interp(t_rail, f["time_s"], f["cg_m"])
    cp_rail = np.interp(0.09, P["aero"]["mach"], P["aero"]["cp_m"])
    sm = (cp_rail - cg_rail) / D
    assert sm == pytest.approx(P["openrocket_reference"]["stability_cal_off_rail"], abs=0.05)
    assert 2.0 <= sm <= 3.0
    flown = np.array(P["aero"]["mach"]) <= P["openrocket_reference"]["max_mach"]
    cp = np.array(P["aero"]["cp_m"])[flown]
    assert (cp.min() - P["mass"]["cg_wet_m"]) / D >= 1.5  # worst case: wet CG, most forward CP


def test_damping_signs_and_strip_theory():
    cnn, xn, cnf, xf = barrowman()
    xcg = 0.5 * (P["mass"]["cg_wet_m"] + P["mass"]["cg_dry_m"])
    cmq = -2 * (cnn * ((xn - xcg) / D) ** 2 + cnf * ((xf - xcg) / D) ** 2)
    y = np.linspace(R, R + FIN["span"], 2001)
    c = FIN["root"] + (FIN["tip"] - FIN["root"]) * (y - R) / FIN["span"]
    clp = -2 * FIN["n"] * (cnf / 2) * (np.trapezoid(c * y ** 2, y) / np.trapezoid(c, y)) / D ** 2
    assert P["aero"]["cmq"] < 0 and P["aero"]["clp"] < 0
    assert P["aero"]["cmq"] == pytest.approx(cmq, rel=0.15)
    assert P["aero"]["clp"] == pytest.approx(clp, rel=0.15)


# ---------------------------------------------------------------- mass
def test_cg_from_mass_table_and_inertia():
    # (kg, CG station) from the docs mass table; nose cone CG from shell geometry, motor at its mid-length
    x = np.linspace(0, NOSE_L, 5001)
    th = np.arccos(1 - 2 * x / NOSE_L)
    r = R / np.sqrt(np.pi) * np.sqrt(th - np.sin(2 * th) / 2)
    msh = 2 * np.pi * 0.0762 * 0.15 * 0.0025 * 1850
    xnose = ((1.35 - msh) * np.trapezoid(r * x, x) / np.trapezoid(r, x) + msh * 0.725) / 1.35
    tab = [(1.35, xnose), (0.68, 0.825), (4.0, 0.78), (0.11, 0.995), (0.87, 1.225), (0.85, 1.155), (0.40, 1.36),
           (0.61, 1.55), (1.40, 1.575), (0.58, 1.80), (0.55, 1.795), (1.55, 2.3485), (0.55, 2.4405), (0.11, 2.44),
           (0.15, 2.144), (1.20, 2.667), (1.29, 2.62), (P["motor"]["motor_total_mass_kg"], 2.4485)]
    m, xs = np.array(tab).T
    assert m.sum() == pytest.approx(P["mass"]["wet_kg"], abs=0.1)
    cg = (m * xs).sum() / m.sum()
    assert cg == pytest.approx(P["mass"]["cg_wet_m"], abs=0.02)
    i_pts = (m * (xs - cg) ** 2).sum()  # point-mass lower bound
    assert i_pts < P["mass"]["Iyy_wet"] < 1.3 * m.sum() * P["length_m"] ** 2 / 12
    assert P["mass"]["Iyy_dry"] < P["mass"]["Iyy_wet"]
    for s in ("wet", "dry"):
        assert 0.3 * P["mass"][f"{s}_kg"] * R ** 2 < P["mass"][f"Ixx_{s}"] < P["mass"][f"{s}_kg"] * R ** 2


# ---------------------------------------------------------------- trajectory
def test_off_rail_velocity_1d():
    _, v = point_mass(rail_travel=P["launch"]["rail_length_m"])
    assert v == pytest.approx(P["openrocket_reference"]["off_rail_velocity_mps"], rel=0.02)


def test_apogee_point_mass():
    _, h = point_mass()
    assert h == pytest.approx(P["openrocket_reference"]["apogee_m_agl_zero_wind"], rel=0.05)
    assert 0.9 * 3048 < h < 1.1 * 3048


# ---------------------------------------------------------------- schema
def test_schema_finite_units_monotonic():
    def walk(o, path=""):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{path}.{k}")
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, f"{path}[{i}]")
        elif isinstance(o, float):
            assert math.isfinite(o), path
    walk(P)
    a = P["aero"]
    assert np.all(np.diff(a["mach"]) > 0)
    assert P["reference_area_m2"] == pytest.approx(math.pi * D ** 2 / 4)
    assert 0.1 < D < 0.2 and 2 < P["length_m"] < 4  # metres, not mm/inches
    assert P["launch"]["lon_deg"] < 0 and P["launch"]["lat_deg"] == pytest.approx(31.99)
    assert P["motor"]["eng_file"] == "data/motor.eng"
    assert P["mass"]["cg_wet_m"] < P["gimbal_station_m"] < P["length_m"]
